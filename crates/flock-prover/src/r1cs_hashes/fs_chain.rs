//! The Fiat–Shamir chain: a sequential BLAKE3-compression duplex transcript.
//!
//! This is the witness generator for the FS chain's rows — the compression
//! sequence a recursion circuit has to reproduce. Each row is one
//! [`Compression`] of the shipped [`super::blake3`] table, so the FS chain adds
//! no table type; what it adds is *which* compressions, in what order, and how
//! their chaining values connect ([`FsChainTrace::links`]).
//!
//! An ordinary squeeze consumes the pending partial block as its first output
//! compression and advances the chaining value. A fused PoW+squeeze appends a
//! 64-bit nonce word, reserves output word 1 for the leading-zero predicate,
//! emits challenge words 0, 2 and 3, and continues from the output high half.
//! Thus a scalar grind adds no standalone BLAKE3 row to recursion.
//!
//! ## Correctness
//!
//! Native, recorded-trace and recursive-row outputs are checked
//! differentially. Getting output allocation or the high-half link wrong
//! could otherwise produce a self-consistent circuit for a transcript nobody
//! computes.

use flock_field::F128;
use flock_hash::blake3_compress;
use flock_transcript::challenger::pow_squeeze_counter;
use flock_transcript::transcript_record::{Stream, TranscriptOp};

use super::blake3::Compression;

/// One forked child chain: its own trace plus the four cross-chain links
/// that connect it to its parent. The child is an INDEPENDENT chain (own
/// IV lineage, own domain), so its rows carry no parent dependency — the
/// only coupling is the seed it absorbs and the digest it hands back.
pub struct ChildChain {
    pub trace: FsChainTrace,
    /// The child's domain (the fork label).
    pub label: Vec<u8>,
    /// PARENT squeeze index whose two output halves seed the child.
    pub seed_squeeze: usize,
    /// Word index in the CHILD stream of the first seed word.
    pub child_seed_word: usize,
    /// CHILD squeeze index whose two output halves the parent absorbs.
    pub digest_squeeze: usize,
    /// Word index in the PARENT stream of the first merge-digest word.
    pub parent_digest_word: usize,
}

impl ChildChain {
    /// The child chain's absorbed bytes, for cross-link inspection.
    pub fn trace_stream_bytes(
        &self,
        parent_stream: &Stream,
        values: &[F128],
        payloads: &[Vec<u8>],
    ) -> Vec<u8> {
        parent_stream
            .forks
            .iter()
            .find(|f| f.label == self.label)
            .expect("child belongs to this parent")
            .stream
            .to_bytes(values, payloads)
    }
}

/// A parent chain and every chain forked from it.
pub struct ForkedChains {
    pub parent: FsChainTrace,
    pub children: Vec<ChildChain>,
}

/// Drive ONE duplex chain from its stream: absorb up to each finalize
/// point, squeeze that op's output width, repeat, then absorb the tail.
///
/// This is the canonical stream→trace driver (it was open-coded at every
/// tower emission site). `ops` are THIS chain's ops: a `Forked` op does not
/// finalize and its child's squeezes belong to the child's chain, so the
/// top-level filter aligns with the parent stream's `finalize_after` by
/// construction.
pub fn trace_duplex(stream: &Stream, bytes: &[u8], ops: &[TranscriptOp]) -> FsChainTrace {
    let mut chain = FsChainSponge::new();
    let mut at = 0usize;
    let mut pending_pow = None;
    let mut fin_ops = Vec::new();
    for op in ops {
        match op {
            TranscriptOp::Pow { bits } => {
                assert!(
                    pending_pow.replace(*bits).is_none(),
                    "nested fused PoW markers"
                );
            }
            op if op.finalizes() => fin_ops.push((op, pending_pow.take())),
            TranscriptOp::Forked { .. } => {}
            _ => assert!(
                pending_pow.is_none(),
                "fused PoW must be followed by a squeeze"
            ),
        }
    }
    assert!(pending_pow.is_none(), "fused PoW marker without a squeeze");
    assert_eq!(
        stream.finalize_after.len(),
        fin_ops.len(),
        "finalize alignment: {} stream points vs {} finalizing ops",
        stream.finalize_after.len(),
        fin_ops.len()
    );
    for (k, &upto) in stream.finalize_after.iter().enumerate() {
        let (_, pow_bits) = fin_ops[k];
        if pow_bits.is_some() {
            chain.absorb_hold_last_block(&bytes[at * 16..upto * 16]);
        } else {
            chain.absorb(&bytes[at * 16..upto * 16]);
        }
        at = upto;
        if let Some(bits) = pow_bits {
            chain.finalize_pow(fin_ops[k].0.squeezed_bytes(), bits);
        } else {
            chain.finalize(fin_ops[k].0.squeezed_bytes());
        }
    }
    chain.absorb(&bytes[at * 16..]);
    chain.finish()
}

/// [`trace_duplex`] over a FORKED transcript: the parent chain plus one
/// independent chain per fork, each with its cross-links.
///
/// Values and payloads are shared (the recorder numbers them globally
/// across both chains), so one table feeds every stream.
pub fn trace_duplex_forked(
    ops: &[TranscriptOp],
    stream: &Stream,
    values: &[F128],
    payloads: &[Vec<u8>],
) -> ForkedChains {
    let parent_bytes = stream.to_bytes(values, payloads);
    let parent = trace_duplex(stream, &parent_bytes, ops);
    let child_ops: Vec<&Vec<TranscriptOp>> = ops
        .iter()
        .filter_map(|o| match o {
            TranscriptOp::Forked { ops, .. } => Some(ops),
            _ => None,
        })
        .collect();
    assert_eq!(
        child_ops.len(),
        stream.forks.len(),
        "fork count: {} ops vs {} streams",
        child_ops.len(),
        stream.forks.len()
    );
    let children = stream
        .forks
        .iter()
        .zip(child_ops)
        .map(|(f, cops)| {
            let bytes = f.stream.to_bytes(values, payloads);
            ChildChain {
                trace: trace_duplex(&f.stream, &bytes, cops),
                label: f.label.clone(),
                seed_squeeze: f.seed_squeeze,
                child_seed_word: f.child_seed_word,
                digest_squeeze: f.digest_squeeze,
                parent_digest_word: f.parent_digest_word,
            }
        })
        .collect();
    ForkedChains { parent, children }
}

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const BLOCK_BYTES: usize = 64;
const BLOCKS_PER_CHUNK: usize = 16;

pub const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// Where a row's `cv` input comes from — the wiring the circuit must emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvSource {
    /// The BLAKE3 IV: a public constant, no wire.
    Iv,
    /// `out_lo` of an earlier row (chunk chaining, or a parent's left input).
    Row(usize),
    /// `out_hi` of a fused PoW+squeeze row.
    RowHi(usize),
}

/// One row plus where its chaining input came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub cv: CvSource,
    /// For `PARENT` rows, the row supplying the RIGHT half of the message.
    /// (`cv` supplies the left half; a parent's own `cv` input is the IV.)
    pub right: Option<usize>,
    /// For an XOF output block, the ROOT row whose `cv` and message it repeats
    /// — only the counter differs. A circuit wires those inputs to the same
    /// places and varies just the params word.
    pub repeats: Option<usize>,
}

/// The compression sequence for one transcript.
pub struct FsChainTrace {
    /// Every compression, in emission order — the slot's rows.
    pub rows: Vec<Compression>,
    /// Per row, where its inputs come from.
    pub links: Vec<Link>,
    /// Per squeeze, the rows whose outputs carry the challenge bytes. The
    /// first is the `ROOT` compression; any others are counter-mode XOF blocks
    /// and are mutually independent.
    pub squeezes: Vec<Vec<usize>>,
    /// Exact `(row, output-word)` source for every challenge word in a
    /// squeeze. Ordinary squeezes enumerate all four row outputs; fused PoW
    /// squeezes omit the word reserved for the zero-prefix predicate.
    pub squeeze_words: Vec<Vec<(usize, usize)>>,
    /// For a row that compresses transcript bytes, the byte offset of its
    /// block; `None` for `PARENT` and XOF rows, whose message is chaining
    /// values rather than stream bytes.
    ///
    /// This is what lets a circuit wire a row's `m` back to the stream — and in
    /// particular wire a **re-absorbed challenge** from the row that produced
    /// it, instead of taking it on trust as a public constant. Without it the
    /// circuit would assert the challenges rather than derive them, which is
    /// the entire content of Fiat–Shamir.
    pub block_offsets: Vec<Option<usize>>,
    /// Number of real 16-byte stream words wired into each message block.
    /// Usually derived from `block_len`; fused PoW rows deliberately use a
    /// full-block length for SIMD while zero-padding after the nonce word.
    pub block_word_counts: Vec<usize>,
}

/// Incremental BLAKE3 with forkable finalization.
pub struct FsChain {
    rows: Vec<Compression>,
    links: Vec<Link>,
    squeezes: Vec<Vec<usize>>,
    squeeze_words: Vec<Vec<(usize, usize)>>,
    /// The current chunk's running chaining value, and the row that produced
    /// it (`None` at a chunk boundary, where it is the IV).
    chunk_cv: [u32; 8],
    chunk_cv_row: Option<usize>,
    chunk_counter: u64,
    blocks_in_chunk: usize,
    /// Completed subtree CVs, with the row that produced each.
    stack: Vec<([u32; 8], usize)>,
    buf: Vec<u8>,
    block_offsets: Vec<Option<usize>>,
    /// Byte offset of the pending block's first byte.
    buf_offset: usize,
    absorbed: usize,
}

fn words(block: &[u8]) -> [u32; 16] {
    let mut m = [0u32; 16];
    for (i, c) in block.chunks(4).enumerate() {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        m[i] = u32::from_le_bytes(w);
    }
    m
}

/// A parent node's message is `left ‖ right`.
fn parent_block(left: &[u32; 8], right: &[u32; 8]) -> [u32; 16] {
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(left);
    m[8..].copy_from_slice(right);
    m
}

impl Default for FsChain {
    fn default() -> Self {
        Self::new()
    }
}

impl FsChain {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            links: Vec::new(),
            squeezes: Vec::new(),
            squeeze_words: Vec::new(),
            chunk_cv: IV,
            chunk_cv_row: None,
            chunk_counter: 0,
            blocks_in_chunk: 0,
            stack: Vec::new(),
            buf: Vec::with_capacity(BLOCK_BYTES),
            block_offsets: Vec::new(),
            buf_offset: 0,
            absorbed: 0,
        }
    }

    fn emit(&mut self, c: Compression, link: Link, offset: Option<usize>) -> usize {
        self.rows.push(c);
        self.links.push(link);
        self.block_offsets.push(offset);
        self.rows.len() - 1
    }

    /// Absorb transcript bytes. A block is compressed once it is full *and*
    /// more input arrives — the pending block always waits, because it may yet
    /// become a chunk's last (or the root's).
    pub fn absorb(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if self.buf.len() == BLOCK_BYTES {
                self.compress_pending(false);
            }
            self.buf.push(b);
            self.absorbed += 1;
        }
    }

    /// Compress the buffered block into the live state.
    fn compress_pending(&mut self, chunk_end: bool) {
        let mut flags = 0;
        if self.blocks_in_chunk == 0 {
            flags |= CHUNK_START;
        }
        if chunk_end || self.blocks_in_chunk == BLOCKS_PER_CHUNK - 1 {
            flags |= CHUNK_END;
        }
        let m = words(&self.buf);
        let cv = self.chunk_cv;
        let out = blake3_compress(&cv, &m, self.chunk_counter, self.buf.len() as u32, flags);
        let link = Link {
            cv: self.chunk_cv_row.map_or(CvSource::Iv, CvSource::Row),
            right: None,
            repeats: None,
        };
        let row = self.emit(
            (cv, m, self.chunk_counter, self.buf.len() as u32, flags),
            link,
            Some(self.buf_offset),
        );
        self.buf.clear();
        self.buf_offset = self.absorbed;

        let next_cv: [u32; 8] = out[..8].try_into().unwrap();
        if flags & CHUNK_END != 0 {
            self.chunk_counter += 1;
            self.blocks_in_chunk = 0;
            self.chunk_cv = IV;
            self.chunk_cv_row = None;
            self.push_subtree(next_cv, row);
        } else {
            self.blocks_in_chunk += 1;
            self.chunk_cv = next_cv;
            self.chunk_cv_row = Some(row);
        }
    }

    /// Add a completed chunk's CV, merging while the chunk count is even —
    /// BLAKE3's chunk-stack rule.
    fn push_subtree(&mut self, mut cv: [u32; 8], mut row: usize) {
        let mut total = self.chunk_counter;
        while total & 1 == 0 {
            let (left, left_row) = self.stack.pop().expect("stack underflow");
            let m = parent_block(&left, &cv);
            let out = blake3_compress(&IV, &m, 0, BLOCK_BYTES as u32, PARENT);
            let link = Link {
                cv: CvSource::Row(left_row),
                right: Some(row),
                repeats: None,
            };
            row = self.emit((IV, m, 0, BLOCK_BYTES as u32, PARENT), link, None);
            cv = out[..8].try_into().unwrap();
            total >>= 1;
        }
        self.stack.push((cv, row));
    }

    /// Fork a root finalization off the current state and take `out_bytes` of
    /// output, without disturbing the live chain.
    ///
    /// The rows this emits are extra — the live state keeps its pending block,
    /// because more transcript follows.
    pub fn finalize(&mut self, out_bytes: usize) -> Vec<u8> {
        let mut ids = Vec::new();

        // The pending block, as this fork's last chunk block. It is the ROOT
        // itself when no completed subtree remains to merge with.
        let mut flags = CHUNK_END;
        if self.blocks_in_chunk == 0 {
            flags |= CHUNK_START;
        }
        let root_here = self.stack.is_empty();
        if root_here {
            flags |= ROOT;
        }
        let m = words(&self.buf);
        let cv = self.chunk_cv;
        let blen = self.buf.len() as u32;
        let mut out = blake3_compress(&cv, &m, self.chunk_counter, blen, flags);
        let mut row = self.emit(
            (cv, m, self.chunk_counter, blen, flags),
            Link {
                cv: self.chunk_cv_row.map_or(CvSource::Iv, CvSource::Row),
                right: None,
                repeats: None,
            },
            Some(self.buf_offset),
        );
        let mut node: [u32; 8] = out[..8].try_into().unwrap();
        let (mut root_m, mut root_cv, mut root_counter, mut root_blen, mut root_flags) =
            (m, cv, self.chunk_counter, blen, flags);

        // Collapse the stack, top-down; the last merge is the root.
        for i in (0..self.stack.len()).rev() {
            let (left, left_row) = self.stack[i];
            let pm = parent_block(&left, &node);
            let mut pf = PARENT;
            if i == 0 {
                pf |= ROOT;
            }
            out = blake3_compress(&IV, &pm, 0, BLOCK_BYTES as u32, pf);
            row = self.emit(
                (IV, pm, 0, BLOCK_BYTES as u32, pf),
                Link {
                    cv: CvSource::Row(left_row),
                    right: Some(row),
                    repeats: None,
                },
                None,
            );
            node = out[..8].try_into().unwrap();
            (root_m, root_cv, root_counter, root_blen, root_flags) =
                (pm, IV, 0, BLOCK_BYTES as u32, pf);
        }
        ids.push(row);
        let root_row = row;

        // The root compression yields the first 64 output bytes; further blocks
        // re-run it at counter 1, 2, … — counter-mode, hence independent.
        let mut bytes = Vec::with_capacity(out_bytes.div_ceil(BLOCK_BYTES) * BLOCK_BYTES);
        for w in out.iter() {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let mut ctr = 1u64;
        while bytes.len() < out_bytes {
            let o = blake3_compress(&root_cv, &root_m, root_counter + ctr, root_blen, root_flags);
            let r = self.emit(
                (root_cv, root_m, root_counter + ctr, root_blen, root_flags),
                Link {
                    cv: CvSource::Iv,
                    right: None,
                    repeats: Some(root_row),
                },
                None,
            );
            ids.push(r);
            for w in o.iter() {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            ctr += 1;
        }
        bytes.truncate(out_bytes);
        self.squeeze_words.push(
            ids.iter()
                .flat_map(|&row| (0..4).map(move |word| (row, word)))
                .take(out_bytes.div_ceil(16))
                .collect(),
        );
        self.squeezes.push(ids);
        bytes
    }

    pub fn finish(self) -> FsChainTrace {
        let block_word_counts = self
            .rows
            .iter()
            .zip(&self.block_offsets)
            .map(|((_, _, _, blen, _), offset)| {
                usize::from(offset.is_some()) * (*blen as usize).div_ceil(16)
            })
            .collect();
        FsChainTrace {
            rows: self.rows,
            links: self.links,
            squeezes: self.squeezes,
            squeeze_words: self.squeeze_words,
            block_offsets: self.block_offsets,
            block_word_counts,
        }
    }
}

/// Domain flag for sponge-chain absorb compressions (transcript-v2; sits
/// above BLAKE3's chunk bits). MUST equal the challenger's constant.
pub const CHAIN_ABSORB: u32 = 1 << 6;
/// Domain flag for sponge-chain squeeze/output compressions.
pub const CHAIN_SQUEEZE: u32 = 1 << 7;

/// The SPONGE-CHAINED transcript trace builder (transcript-v3 DUPLEX):
/// mirrors [`flock_transcript::challenger::FsChallenger::with_chained_blake3`]
/// row for row — a sequential compression chain, no chunk tree, and
/// squeezes that MUTATE the state. A squeeze's first row consumes the
/// pending partial block as its message (`block_offsets = Some(..)` with a
/// partial `blen` when bytes are pending, `None` with a zero message when
/// not); further output rows are zero-message (`None`); EVERY row advances
/// the cv, so the whole trace is one uniform chain (`right`/`repeats`
/// never set, no fork rows, no per-squeeze flush).
///
/// Drop-in for [`FsChain`] in the tape constructors: same `absorb` /
/// `finalize` / `finish` surface. All rows run at counter 0 — block order
/// is bound by the cv chain, and every distinct counter value would cost a
/// circuit public; squeeze rows carry `CHAIN_SQUEEZE` and their pending
/// byte count in `blen`.
pub struct FsChainSponge {
    rows: Vec<Compression>,
    links: Vec<Link>,
    squeezes: Vec<Vec<usize>>,
    squeeze_words: Vec<Vec<(usize, usize)>>,
    block_offsets: Vec<Option<usize>>,
    block_word_counts: Vec<usize>,
    cv: [u32; 8],
    cv_source: CvSource,
    buf: Vec<u8>,
    buf_offset: usize,
    absorbed: usize,
}

impl Default for FsChainSponge {
    fn default() -> Self {
        Self::new()
    }
}

impl FsChainSponge {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            links: Vec::new(),
            squeezes: Vec::new(),
            squeeze_words: Vec::new(),
            block_offsets: Vec::new(),
            block_word_counts: Vec::new(),
            cv: IV,
            cv_source: CvSource::Iv,
            buf: Vec::with_capacity(BLOCK_BYTES),
            buf_offset: 0,
            absorbed: 0,
        }
    }

    fn emit(
        &mut self,
        c: Compression,
        link: Link,
        offset: Option<usize>,
        word_count: usize,
    ) -> usize {
        self.rows.push(c);
        self.links.push(link);
        self.block_offsets.push(offset);
        self.block_word_counts.push(word_count);
        self.rows.len() - 1
    }

    fn cv_link(&self) -> Link {
        Link {
            cv: self.cv_source,
            right: None,
            repeats: None,
        }
    }

    pub fn absorb(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        self.absorbed += bytes.len();
        while self.buf.len() >= BLOCK_BYTES {
            let m = words(&self.buf[..BLOCK_BYTES]);
            let out = blake3_compress(&self.cv, &m, 0, BLOCK_BYTES as u32, CHAIN_ABSORB);
            let link = self.cv_link();
            let row = self.emit(
                (self.cv, m, 0, BLOCK_BYTES as u32, CHAIN_ABSORB),
                link,
                Some(self.buf_offset),
                4,
            );
            self.cv = out[..8].try_into().expect("8 words");
            self.cv_source = CvSource::Row(row);
            self.buf.drain(..BLOCK_BYTES);
            self.buf_offset += BLOCK_BYTES;
        }
    }

    /// Absorb while retaining the final full block. A fused PoW row must
    /// contain the nonce in the same compression that produces the challenge,
    /// so an exactly-full pending block cannot become an ordinary absorb row.
    pub fn absorb_hold_last_block(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        self.absorbed += bytes.len();
        while self.buf.len() > BLOCK_BYTES {
            self.compress_absorb_block();
        }
    }

    fn compress_absorb_block(&mut self) {
        let m = words(&self.buf[..BLOCK_BYTES]);
        let out = blake3_compress(&self.cv, &m, 0, BLOCK_BYTES as u32, CHAIN_ABSORB);
        let link = self.cv_link();
        let row = self.emit(
            (self.cv, m, 0, BLOCK_BYTES as u32, CHAIN_ABSORB),
            link,
            Some(self.buf_offset),
            4,
        );
        self.cv = out[..8].try_into().expect("8 words");
        self.cv_source = CvSource::Row(row);
        self.buf.drain(..BLOCK_BYTES);
        self.buf_offset += BLOCK_BYTES;
    }

    /// Duplex squeeze (transcript-v3): the FIRST output row consumes the
    /// pending partial block as its message and every output row advances
    /// the cv — the squeeze IS part of the chain, not a fork. Mirrors
    /// `B3Chain::squeeze_into` exactly.
    pub fn finalize(&mut self, out_bytes: usize) -> Vec<u8> {
        let mut ids = Vec::new();
        let mut bytes = Vec::with_capacity(out_bytes.div_ceil(BLOCK_BYTES) * BLOCK_BYTES);
        let mut first = true;
        while bytes.len() < out_bytes {
            let (m, blen, offset) = if first {
                (
                    words(&self.buf),
                    self.buf.len() as u32,
                    // A pending partial block is stream content — the
                    // emitter wires it exactly like a data block (the
                    // partial-width zero-fill path already exists); an
                    // empty pending block is the shared zero constant.
                    (!self.buf.is_empty()).then_some(self.buf_offset),
                )
            } else {
                ([0u32; 16], 0u32, None)
            };
            let o = blake3_compress(&self.cv, &m, 0, blen, CHAIN_SQUEEZE);
            let link = self.cv_link();
            let row = self.emit(
                (self.cv, m, 0, blen, CHAIN_SQUEEZE),
                link,
                offset,
                (blen as usize).div_ceil(16),
            );
            self.cv = o[..8].try_into().expect("8 words");
            self.cv_source = CvSource::Row(row);
            ids.push(row);
            for w in o.iter() {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            first = false;
        }
        self.buf.clear();
        self.buf_offset = self.absorbed;
        bytes.truncate(out_bytes);
        self.squeeze_words.push(
            ids.iter()
                .flat_map(|&row| (0..4).map(move |word| (row, word)))
                .take(out_bytes.div_ceil(16))
                .collect(),
        );
        self.squeezes.push(ids);
        bytes
    }

    /// Fuse verification of the recorded PoW nonce with the first squeeze
    /// compression. Output word 1 is reserved for the zero-prefix predicate;
    /// challenge words are 0, 2, 3, followed by ordinary continuation rows.
    pub fn finalize_pow(&mut self, out_bytes: usize, bits: u32) -> Vec<u8> {
        assert!(bits <= 128, "fused PoW predicate occupies one F128 word");
        assert!(!self.buf.is_empty() && self.buf.len() <= BLOCK_BYTES);
        assert_eq!(
            self.buf.len() % 16,
            0,
            "transcript words are 16-byte aligned"
        );

        let word_count = self.buf.len() / 16;
        let m = words(&self.buf);
        let counter = pow_squeeze_counter(bits, self.buf.len());
        let out = blake3_compress(&self.cv, &m, counter, BLOCK_BYTES as u32, CHAIN_SQUEEZE);
        let link = self.cv_link();
        let row = self.emit(
            (self.cv, m, counter, BLOCK_BYTES as u32, CHAIN_SQUEEZE),
            link,
            Some(self.buf_offset),
            word_count,
        );
        self.cv = out[8..16].try_into().expect("8 words");
        self.cv_source = CvSource::RowHi(row);

        let wanted_words = out_bytes.div_ceil(16);
        let mut sources = Vec::with_capacity(wanted_words);
        let mut bytes = Vec::with_capacity(wanted_words * 16);
        for word in [0usize, 2, 3].into_iter().take(wanted_words) {
            sources.push((row, word));
            for limb in &out[word * 4..word * 4 + 4] {
                bytes.extend_from_slice(&limb.to_le_bytes());
            }
        }
        let mut ids = vec![row];
        while sources.len() < wanted_words {
            let zero = [0u32; 16];
            let o = blake3_compress(&self.cv, &zero, 0, 0, CHAIN_SQUEEZE);
            let link = self.cv_link();
            let continuation = self.emit((self.cv, zero, 0, 0, CHAIN_SQUEEZE), link, None, 0);
            self.cv = o[..8].try_into().expect("8 words");
            self.cv_source = CvSource::Row(continuation);
            ids.push(continuation);
            for word in 0..4 {
                if sources.len() == wanted_words {
                    break;
                }
                sources.push((continuation, word));
                for limb in &o[word * 4..word * 4 + 4] {
                    bytes.extend_from_slice(&limb.to_le_bytes());
                }
            }
        }
        self.buf.clear();
        self.buf_offset = self.absorbed;
        bytes.truncate(out_bytes);
        self.squeeze_words.push(sources);
        self.squeezes.push(ids);
        bytes
    }

    pub fn finish(self) -> FsChainTrace {
        FsChainTrace {
            rows: self.rows,
            links: self.links,
            squeezes: self.squeezes,
            squeeze_words: self.squeeze_words,
            block_offsets: self.block_offsets,
            block_word_counts: self.block_word_counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blake3::Hasher;
    use flock_field::F128;
    use flock_transcript::challenger::{Challenger, FsChallenger};
    use flock_transcript::transcript_record::{RecordingChallenger, StreamWord};

    /// The SPONGE trace builder must equal the chained challenger byte for
    /// byte: same absorb schedule, same squeeze outputs. The challenger is
    /// the protocol; a divergent trace builder proves a transcript nobody
    /// hashes.
    #[test]
    fn sponge_finalize_matches_the_chained_challenger() {
        // Drive both through the SAME op schedule via the recording layer:
        // absorb framed values exactly as the challenger frames them.
        let mut ch = FsChallenger::with_chained_blake3(b"sponge-diff");
        let rec_bytes: Vec<u8> = Vec::new();

        // Reproduce framing through a recorded transcript.
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(b"sponge-diff"));
        let mut squeezed_ch: Vec<Vec<u8>> = Vec::new();
        for i in 0..40u64 {
            let v = F128 {
                lo: i,
                hi: i.wrapping_mul(77),
            };
            ch.observe_f128(v);
            rec.observe_f128(v);
            if i % 3 == 0 {
                let a = ch.sample_f128();
                let b = rec.sample_f128();
                assert_eq!(a, b);
                let mut bs = Vec::new();
                bs.extend_from_slice(&a.lo.to_le_bytes());
                bs.extend_from_slice(&a.hi.to_le_bytes());
                squeezed_ch.push(bs);
            }
            if i % 7 == 0 {
                let vs_a = ch.sample_f128_vec(3);
                let vs_b = rec.sample_f128_vec(3);
                assert_eq!(vs_a, vs_b);
                let mut bs = Vec::new();
                for v2 in &vs_a {
                    bs.extend_from_slice(&v2.lo.to_le_bytes());
                    bs.extend_from_slice(&v2.hi.to_le_bytes());
                }
                squeezed_ch.push(bs);
            }
        }
        // Replay the recorded byte stream through the sponge trace builder;
        // every finalize must reproduce the challenger's squeezed bytes.
        let shape = rec.shape();
        let stream = shape.stream_words_duplex(b"sponge-diff");
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let fin_ops: Vec<_> = shape.ops().iter().filter(|o| o.finalizes()).collect();
        let mut chain = FsChainSponge::new();
        let mut at = 0usize;
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            let got = chain.finalize(fin_ops[k].squeezed_bytes());
            assert_eq!(got, squeezed_ch[k], "squeeze {k}");
        }
        assert_eq!(
            stream.finalize_after.len(),
            squeezed_ch.len(),
            "every squeeze checked"
        );
        let _ = rec_bytes;
    }

    /// Every finalize must equal reference BLAKE3's XOF of the same prefix.
    ///
    /// A subtly wrong chunk stack still yields a self-consistent circuit — one
    /// proving a hash nobody computes — so this is checked against the real
    /// implementation, at lengths that straddle every boundary the tree has:
    /// inside the first block, at block ends, at the 1 KiB chunk boundary, and
    /// across several chunks so the stack actually has depth.
    #[test]
    fn every_finalize_matches_reference_blake3() {
        let data: Vec<u8> = (0..70_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();

        let lengths = [
            0usize, 1, 63, 64, 65, 127, 128, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 5000,
            16_384, 16_385, 40_000, 65_536, 65_537,
        ];
        for &len in &lengths {
            for &out_bytes in &[16usize, 32, 64, 65, 3888] {
                let mut c = FsChain::new();
                c.absorb(&data[..len]);
                let got = c.finalize(out_bytes);

                let mut want = vec![0u8; out_bytes];
                let mut h = Hasher::new();
                h.update(&data[..len]);
                h.finalize_xof().fill(&mut want);

                assert_eq!(got, want, "len={len}, out_bytes={out_bytes}");
            }
        }
    }

    /// Forking a finalize must not disturb the live chain: absorbing more after
    /// a squeeze still hashes the whole prefix correctly.
    #[test]
    fn finalizing_does_not_disturb_the_live_chain() {
        let data: Vec<u8> = (0..5000u32).map(|i| i as u8).collect();
        let mut c = FsChain::new();
        let stops = [100usize, 1024, 1100, 3000, 5000];
        let mut at = 0usize;
        for &s in &stops {
            c.absorb(&data[at..s]);
            at = s;
            let got = c.finalize(32);
            let mut want = [0u8; 32];
            Hasher::new()
                .update(&data[..s])
                .finalize_xof()
                .fill(&mut want);
            assert_eq!(got, want, "after absorbing {s} bytes");
        }
        let trace = c.finish();
        assert_eq!(trace.squeezes.len(), stops.len());
        assert_eq!(trace.rows.len(), trace.links.len());
    }

    /// The row count matches what `TranscriptShape::blake3_inventory` predicts
    /// from the schedule alone — the two derivations are independent.
    #[test]
    fn row_count_matches_the_derived_inventory() {
        // One squeeze of 16 bytes after 17,008 bytes, the element transcript's
        // shape: 265 absorb + 15 chunk parents, then the finalize.
        let data = vec![7u8; 17_008];
        let mut c = FsChain::new();
        c.absorb(&data);
        c.finalize(16);
        let t = c.finish();

        let complete_chunks = (17_008usize - 1) / 1024;
        let absorb = (17_008usize - 1) / 64;
        let chunk_parents = complete_chunks - complete_chunks.count_ones() as usize;
        let finalize = 1 + complete_chunks.count_ones() as usize;
        assert_eq!(t.rows.len(), absorb + chunk_parents + finalize);
    }

    /// THE DIFFERENTIAL for the forked transcript: a recorded fork/merge
    /// protocol, replayed as two independent chains, must reproduce the
    /// NATIVE challenger's challenges — parent and child alike — and the
    /// four cross-links must carry the exact seed/digest bytes.
    ///
    /// This is what makes the fork safe to emit as circuit rows: getting
    /// the child's lineage or the cross-wiring subtly wrong would still
    /// produce a self-consistent circuit, one proving a transcript nobody
    /// computes.
    #[test]
    fn forked_chains_reproduce_the_native_transcript() {
        // Parent absorbs, forks, both sides run, merge, parent continues —
        // the union prover's one-sided wiring branch in miniature.
        fn drive<C: Challenger + Sized>(ch: &mut C) -> Vec<F128> {
            let mut out = Vec::new();
            ch.observe_label(b"parent");
            for i in 0..40u64 {
                ch.observe_f128(F128::new(i, i * 7));
            }
            out.push(ch.sample_f128());
            let mut child = ch.fork(b"wiring-branch");
            child.observe_label(b"child");
            for i in 0..90u64 {
                child.observe_f128(F128::new(i * 3, i));
                if i % 16 == 15 {
                    out.push(child.sample_f128());
                }
            }
            out.push(child.sample_f128());
            for i in 0..30u64 {
                ch.observe_f128(F128::new(i, 0));
            }
            out.push(ch.sample_f128());
            ch.merge_child(child);
            out.push(ch.sample_f128());
            out.push(ch.sample_f128());
            out
        }

        const DOMAIN: &[u8] = b"fork-differential";
        let native = drive(&mut FsChallenger::with_chained_blake3(DOMAIN));

        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(DOMAIN));
        let recorded = drive(&mut rec);
        assert_eq!(recorded, native, "recording must be transparent");

        let shape = rec.shape();
        let stream = shape.stream_words_duplex(DOMAIN);
        let chains = trace_duplex_forked(shape.ops(), &stream, rec.values(), rec.payloads());
        assert_eq!(chains.children.len(), 1);
        let child = &chains.children[0];
        assert_eq!(child.label, b"wiring-branch".to_vec());

        // Squeeze outputs, chain by chain, as 16-byte challenges. A squeeze
        // row's output words carry the XOF bytes; `squeezes[k]` lists the
        // rows for finalize `k` (first is the ROOT).
        let chal = |t: &FsChainTrace, k: usize| -> F128 {
            let row = t.squeezes[k][0];
            let (cv, m, counter, blen, flags) = t.rows[row];
            let out = blake3_compress(&cv, &m, counter, blen, flags);
            let mut b = [0u8; 16];
            for (i, w) in out[..4].iter().enumerate() {
                b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            F128::new(
                u64::from_le_bytes(b[..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..].try_into().unwrap()),
            )
        };

        // Parent challenge order: [0] pre-fork, then the two SEED squeezes,
        // then the post-fork sample, then the two after the merge.
        assert_eq!(chal(&chains.parent, 0), native[0], "pre-fork challenge");
        // native[1..=6] are the child's (5 mid + 1 closing) — check the
        // child chain reproduces them in its own order.
        for (i, k) in (0..6).enumerate() {
            assert_eq!(
                chal(&child.trace, k),
                native[1 + i],
                "child challenge {k} diverged"
            );
        }
        // The parent's own post-fork samples: after [0] came 2 seed squeezes,
        // so the next parent finalize is index 3.
        assert_eq!(chal(&chains.parent, 3), native[7], "post-fork parent");
        assert_eq!(chal(&chains.parent, 4), native[8], "post-merge parent");
        assert_eq!(chal(&chains.parent, 5), native[9], "post-merge parent 2");

        // The CROSS-LINKS carry real bytes: the child's seed words equal the
        // parent's seed-squeeze halves, and the parent's merge words equal
        // the child's closing-squeeze halves. Checked through the recorded
        // value table the streams index.
        let parent_bytes = stream.to_bytes(rec.values(), rec.payloads());
        let child_bytes = child.trace_stream_bytes(&stream, rec.values(), rec.payloads());
        let word = |b: &[u8], i: usize| -> F128 {
            F128::new(
                u64::from_le_bytes(b[i * 16..i * 16 + 8].try_into().unwrap()),
                u64::from_le_bytes(b[i * 16 + 8..i * 16 + 16].try_into().unwrap()),
            )
        };
        assert_eq!(
            word(&child_bytes, child.child_seed_word),
            chal(&chains.parent, child.seed_squeeze),
            "the child's first seed word is the parent's seed-squeeze output"
        );
        assert_eq!(
            word(&parent_bytes, child.parent_digest_word),
            chal(&child.trace, child.digest_squeeze),
            "the parent's merge word is the child's closing-squeeze output"
        );
    }

    #[test]
    fn fused_pow_squeeze_trace_matches_recorded_challenger() {
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(b"pow-trace"));
        rec.observe_f128_slice(&[
            F128::new(1, 2),
            F128::new(3, 4),
            F128::new(5, 6),
            F128::new(7, 8),
            F128::new(9, 10),
        ]);
        let (nonce, protected) = rec.grind_pow_and_sample_f128_vec(5, 7);
        rec.observe_f128(F128::new(11, 12));
        let after = rec.sample_f128();

        let shape = rec.shape();
        let stream = shape.stream_words_duplex(b"pow-trace");
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let trace = trace_duplex(&stream, &bytes, shape.ops());

        assert_eq!(trace.squeezes.len(), 2);
        assert_eq!(trace.squeeze_words[0].len(), 7);
        assert_eq!(
            trace.squeeze_words[0][..3],
            [
                (trace.squeezes[0][0], 0),
                (trace.squeezes[0][0], 2),
                (trace.squeezes[0][0], 3)
            ]
        );
        assert!(
            matches!(trace.links[trace.squeezes[0][0] + 1].cv, CvSource::RowHi(r) if r == trace.squeezes[0][0])
        );

        let read = |fin: usize, offset: usize| {
            let (row, word) = trace.squeeze_words[fin][offset];
            let (cv, m, counter, blen, flags) = trace.rows[row];
            let out = blake3_compress(&cv, &m, counter, blen, flags);
            let mut b = [0u8; 16];
            for (i, limb) in out[word * 4..word * 4 + 4].iter().enumerate() {
                b[4 * i..4 * i + 4].copy_from_slice(&limb.to_le_bytes());
            }
            F128::new(
                u64::from_le_bytes(b[..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..].try_into().unwrap()),
            )
        };
        assert_eq!((0..7).map(|i| read(0, i)).collect::<Vec<_>>(), protected);
        assert_eq!(read(1, 0), after);

        let pow_row = trace.squeezes[0][0];
        let out = {
            let (cv, m, counter, blen, flags) = trace.rows[pow_row];
            blake3_compress(&cv, &m, counter, blen, flags)
        };
        let mut predicate = [0u8; 16];
        for (i, limb) in out[4..8].iter().enumerate() {
            predicate[4 * i..4 * i + 4].copy_from_slice(&limb.to_le_bytes());
        }
        assert_eq!(predicate[0] & 0b1111_1000, 0);
        let nonce_word = F128::new(nonce, 0);
        let nonce_at = stream
            .words
            .iter()
            .position(|w| matches!(w, StreamWord::Bytes { payload, .. } if rec.payloads()[*payload] == nonce.to_le_bytes()))
            .expect("nonce word");
        assert_eq!(
            F128::new(
                u64::from_le_bytes(bytes[16 * nonce_at..16 * nonce_at + 8].try_into().unwrap()),
                u64::from_le_bytes(
                    bytes[16 * nonce_at + 8..16 * nonce_at + 16]
                        .try_into()
                        .unwrap()
                ),
            ),
            nonce_word,
        );
        assert_eq!(trace.rows[pow_row].3, 64);
        assert!((1..=4).contains(&trace.block_word_counts[pow_row]));
    }
}
