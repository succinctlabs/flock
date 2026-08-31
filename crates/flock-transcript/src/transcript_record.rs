//! Recording the Fiat–Shamir transcript's **shape**.
//!
//! The recursive verifier has to replay this protocol's Fiat–Shamir transcript
//! inside a circuit, which means every BLAKE3 compression of it becomes
//! committed rows. Laying those rows out needs an exact, ordered account of
//! what gets absorbed and squeezed — and writing that account by hand would
//! create a second description of the transcript that can silently drift from
//! the first.
//!
//! So it is not written by hand. [`RecordingChallenger`] decorates a real
//! [`Challenger`], delegates every call unchanged, and records the sequence.
//! Running the actual verifier under it yields the schedule *as a consequence
//! of the verifier's behaviour*, so there is nothing to keep in sync.
//!
//! ## Shape, not content
//!
//! A [`TranscriptOp`] records op kind and **lengths only** — never values. Two
//! runs over different witnesses, different counts, or different proofs must
//! produce the identical op sequence; that is what makes a fixed-topology
//! circuit possible at all. It is a property of the code, not a law, so it is
//! checked rather than assumed (see the shape-diff tests). Labels *are* part of
//! the shape: they are compile-time constants that partition the transcript
//! into protocol phases.
//!
//! Prover and verifier must produce the same transcript, so recording a prove
//! and recording a verify must yield equal shapes — a free differential.
//!
//! ## The delegation trap
//!
//! [`Challenger`] gives default bodies for `observe_f128_slice` and
//! `sample_f128_vec` that decompose into per-scalar calls, and [`FsChallenger`]
//! overrides both with genuinely different absorption (a `KIND_SLICE` tag and
//! one length prefix, not `n` scalar ops). A decorator that inherits those
//! defaults would therefore **change the transcript it is trying to observe**.
//! Every method here overrides and delegates for that reason, and
//! `recording_is_transparent` pins it.
//!
//! [`FsChallenger`]: crate::challenger::FsChallenger

use sha2::{Digest, Sha256};

use crate::challenger::Challenger;
use crate::challenger::{
    KIND_NONE, KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE,
};
use flock_field::F128;
use flock_hash::HashKind;

/// One protocol-level transcript action, with values stripped.
///
/// Deliberately at *protocol* granularity rather than byte granularity: a
/// `Pow` is one op even though it absorbs a nonce and squeezes a state digest
/// internally, because that is the unit a circuit gadget will implement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptOp {
    /// `observe_label` — a domain-separation constant. Its bytes are shape.
    Label(Vec<u8>),
    /// `observe_f128`.
    ObserveScalar,
    /// `observe_f128_slice` of `n` elements.
    ObserveSlice(usize),
    /// `observe_bytes` of `len` bytes.
    ObserveBytes(usize),
    /// `sample_f128`.
    SqueezeScalar,
    /// `sample_f128_vec` of `n` elements.
    SqueezeSlice(usize),
    /// A forked child transcript (the parallel-composition branch): its
    /// complete op sequence, run on its own chain under `label` as domain.
    /// SITS AT THE FORK POSITION in the parent list; by convention the two
    /// `SqueezeScalar` ops immediately preceding it are the child's seed,
    /// the child's first two `ObserveScalar`s absorb that seed, and the
    /// child's last two `SqueezeScalar`s are its closing digest — which the
    /// two parent `ObserveScalar`s after the matching [`Self::Merge`] marker
    /// absorb. Value/payload indices are GLOBAL: the walk descends into the
    /// child inline at the fork position.
    Forked {
        label: Vec<u8>,
        ops: Vec<TranscriptOp>,
    },
    /// Marks the merge of fork number `fork` (index among `Forked` ops, in
    /// order): the two parent `ObserveScalar`s that follow absorb that
    /// child's closing digest. Contributes no words itself.
    Merge { fork: usize },
    /// Fused PoW+squeeze marker at `bits`.  This op contributes the private
    /// 16-byte nonce word; the immediately following `SqueezeScalar` or
    /// `SqueezeSlice` performs the single domain-separated compression.
    Pow { bits: u32 },
    /// Legacy standalone `grind_pow` / `verify_pow`.  Kept for callers that do
    /// not use the fused APIs; active recursive protocol paths use `Pow`.
    LegacyPow { bits: u32 },
}

impl TranscriptOp {
    /// Bytes this op absorbs: a fixed 16-byte header
    /// `[op][kind][0;6][len u64]`, then the payload zero-padded to a multiple
    /// of 16. Squeeze ops absorb too — their 16-byte header (the squeezed
    /// output itself is never fed back into the transcript).
    ///
    /// **Why everything is 16-aligned.** Every observed value is an `F128` —
    /// 16 bytes, and exactly one 128-bit committed word. A recursion circuit
    /// replaying this transcript places those bytes into BLAKE3's `m` words,
    /// and its wires carry 128-bit words, so the placement is a *pure copy*
    /// iff each value starts at a multiple of 16. The former 1–2 byte tags and
    /// 8-byte length prefixes broke that (scalars landed at `2 + 18k`, so seven
    /// in eight straddled two `m` words), which would have cost a byte-shift
    /// packing gate and a boolean glue table. Alignment removes the problem at
    /// its source for ~15% more FS compressions.
    ///
    /// This is the byte layout the circuit reproduces, which is why it is
    /// cross-checked against the live challenger's own counter rather than
    /// trusted.
    ///
    /// **v2 accounting.** The DUPLEX chain discipline (transcript-v3,
    /// `with_chained_blake3`) absorbs no squeeze headers, so its per-op
    /// byte count differs: squeezes absorb 0 there and a `Pow` absorbs only
    /// its nonce word (16 bytes: the 8-byte LE nonce plus the zero-constrained
    /// pad). Chain consumers derive the layout from
    /// [`TranscriptShape::stream_words_duplex`], never from this.
    pub fn absorbed_bytes(&self) -> usize {
        let pad16 = |n: usize| n.div_ceil(16) * 16;
        // Fork bookkeeping absorbs nothing on THIS chain: the child runs on
        // its own (its bytes are the child's), and `Merge` only marks that
        // the following `ObserveScalar`s carry the digest.
        if matches!(
            self,
            TranscriptOp::Forked { .. } | TranscriptOp::Merge { .. }
        ) {
            return 0;
        }
        16 + match self {
            TranscriptOp::Label(l) => pad16(l.len()),
            TranscriptOp::ObserveScalar => 16,
            TranscriptOp::ObserveSlice(n) => 16 * n,
            TranscriptOp::ObserveBytes(len) => pad16(*len),
            // A squeeze absorbs only its header — the output is not fed back.
            TranscriptOp::SqueezeScalar | TranscriptOp::SqueezeSlice(_) => 0,
            // The PoW nonce rides `observe_bytes(8)`.
            TranscriptOp::Pow { .. } | TranscriptOp::LegacyPow { .. } => 16,
            TranscriptOp::Forked { .. } | TranscriptOp::Merge { .. } => {
                unreachable!("early return")
            }
        }
    }

    /// Bytes of squeezed OUTPUT. Drives the XOF-output block count: each
    /// 64 bytes is one counter-mode compression, and those are mutually
    /// independent (unlike the finalizations, which serialize).
    pub fn squeezed_bytes(&self) -> usize {
        match self {
            TranscriptOp::SqueezeScalar => 16,
            TranscriptOp::SqueezeSlice(n) => 16 * n,
            TranscriptOp::Pow { .. } => 0, // fused with the following squeeze
            TranscriptOp::LegacyPow { .. } => 32,
            _ => 0,
        }
    }

    /// Whether this op owns a byte-payload ordinal: `ObserveBytes` (public
    /// bytes) and both PoW forms (the private 16-byte nonce) share the
    /// payload counter — the tower's tape walkers, fold/query regions and
    /// `bytes_payload_mask` all index payloads this way.
    pub fn carries_payload(&self) -> bool {
        matches!(
            self,
            TranscriptOp::ObserveBytes(_)
                | TranscriptOp::Pow { .. }
                | TranscriptOp::LegacyPow { .. }
        )
    }

    /// Whether this op finalizes the pending state. Finalizations are the
    /// transcript's serial depth: a squeeze finalizes the running state, and
    /// everything after it depends on that state, so nothing downstream can
    /// be computed before it.
    pub fn finalizes(&self) -> bool {
        matches!(
            self,
            TranscriptOp::SqueezeScalar
                | TranscriptOp::SqueezeSlice(_)
                | TranscriptOp::LegacyPow { .. }
        )
    }
}

/// An ordered account of one run's transcript, values stripped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptShape {
    ops: Vec<TranscriptOp>,
}

impl TranscriptShape {
    pub fn ops(&self) -> &[TranscriptOp] {
        &self.ops
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Index of the first op where two shapes differ, or `None` if identical.
    /// Reported rather than a bare bool so a staticness failure names the op
    /// that broke it instead of just asserting that something did.
    pub fn first_difference(&self, other: &Self) -> Option<usize> {
        let n = self.ops.len().min(other.ops.len());
        (0..n)
            .find(|&i| self.ops[i] != other.ops[i])
            .or(if self.ops.len() == other.ops.len() {
                None
            } else {
                Some(n)
            })
    }

    /// Total bytes absorbed, excluding the domain separator absorbed at
    /// challenger construction (the recorder wraps an already-built
    /// challenger, so it never sees that).
    pub fn absorbed_bytes(&self) -> usize {
        self.ops.iter().map(TranscriptOp::absorbed_bytes).sum()
    }

    pub fn squeezed_bytes(&self) -> usize {
        self.ops.iter().map(TranscriptOp::squeezed_bytes).sum()
    }

    /// Serial depth in finalizations — the number that actually sizes the FS
    /// chain's critical path.
    pub fn finalizations(&self) -> usize {
        self.ops.iter().filter(|o| o.finalizes()).count()
    }

    /// Each squeeze addressed as `(enclosing label, ordinal within that
    /// label)` instead of by absolute index.
    ///
    /// Absolute indices renumber whenever anything upstream changes, which
    /// would make every challenge-to-consumer wire in the circuit shift for an
    /// unrelated edit. Phase-relative addressing is stable under insertions
    /// elsewhere in the transcript.
    pub fn squeeze_roles(&self) -> Vec<(Vec<u8>, usize)> {
        let mut out = Vec::new();
        let mut phase: Vec<u8> = Vec::new();
        let mut ordinal = 0usize;
        for op in &self.ops {
            match op {
                TranscriptOp::Label(l) => {
                    phase = l.clone();
                    ordinal = 0;
                }
                TranscriptOp::SqueezeScalar | TranscriptOp::SqueezeSlice(_) => {
                    out.push((phase.clone(), ordinal));
                    ordinal += 1;
                }
                _ => {}
            }
        }
        out
    }

    /// Digest of the shape, for pinning. A protocol change that moves the FS
    /// shape moves this, so it fails loudly and gets a deliberate re-pin —
    /// the same discipline as the proof-byte fixtures.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"flock-transcript-shape-v0");
        h.update((self.ops.len() as u64).to_le_bytes());
        for op in &self.ops {
            match op {
                TranscriptOp::Label(l) => {
                    h.update([0u8]);
                    h.update((l.len() as u64).to_le_bytes());
                    h.update(l);
                }
                TranscriptOp::ObserveScalar => h.update([1u8]),
                TranscriptOp::ObserveSlice(n) => {
                    h.update([2u8]);
                    h.update((*n as u64).to_le_bytes());
                }
                TranscriptOp::ObserveBytes(len) => {
                    h.update([3u8]);
                    h.update((*len as u64).to_le_bytes());
                }
                TranscriptOp::SqueezeScalar => h.update([4u8]),
                TranscriptOp::SqueezeSlice(n) => {
                    h.update([5u8]);
                    h.update((*n as u64).to_le_bytes());
                }
                TranscriptOp::Pow { bits } => {
                    h.update([6u8]);
                    h.update(bits.to_le_bytes());
                }
                TranscriptOp::LegacyPow { bits } => {
                    h.update([9u8]);
                    h.update(bits.to_le_bytes());
                }
                // The child's shape is part of the parent's: a fork whose
                // branch changed must move this digest.
                TranscriptOp::Forked { label, ops } => {
                    h.update([7u8]);
                    h.update((label.len() as u64).to_le_bytes());
                    h.update(label);
                    h.update(TranscriptShape { ops: ops.clone() }.digest());
                }
                TranscriptOp::Merge { fork } => {
                    h.update([8u8]);
                    h.update((*fork as u64).to_le_bytes());
                }
            }
        }
        h.finalize().into()
    }

    /// Hex digest, for fixture constants.
    pub fn digest_hex(&self) -> String {
        self.digest().iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// The BLAKE3 compression inventory of one transcript — the FS chain's actual
/// row count, broken out by flavour because they differ in flags, in counter,
/// and in whether they serialize.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake3Inventory {
    /// Blocks compressed as the stream is absorbed. Sequential within a 1 KiB
    /// chunk, independent across chunks.
    pub absorb_blocks: usize,
    /// `PARENT` compressions that build the chunk tree during absorption:
    /// `C − popcount(C)` for `C` complete chunks.
    pub chunk_parents: usize,
    /// The pending block each finalization must compress — one per squeeze.
    pub finalize_blocks: usize,
    /// **The term a flat "one compression per squeeze" model misses.** A
    /// finalize is not local: it collapses the current chunk stack, so it costs
    /// `popcount(complete chunks)` `PARENT` compressions, the last of them
    /// `ROOT`. That grows as the transcript does, so late squeezes cost more
    /// than early ones.
    pub finalize_parents: usize,
    /// XOF output blocks past the first: the root compression already yields
    /// 64 bytes, so only longer squeezes need more. Counter-mode, hence
    /// mutually independent.
    pub xof_blocks: usize,
}

impl Blake3Inventory {
    pub fn total(&self) -> usize {
        self.absorb_blocks
            + self.chunk_parents
            + self.finalize_blocks
            + self.finalize_parents
            + self.xof_blocks
    }
}

impl TranscriptShape {
    /// Count the BLAKE3 compressions this transcript actually costs, by
    /// walking the recorded schedule and tracking the byte offset — which is
    /// all that is needed, since the chunk stack's depth at any point is
    /// `popcount(offset / 1024)`.
    ///
    /// Derived rather than estimated: the FS chain's row inventory is exactly
    /// this, and the flat `one per squeeze` approximation the hash-count bench
    /// uses undercounts it (see [`Blake3Inventory::finalize_parents`]).
    pub fn blake3_inventory(&self, domain_len: usize) -> Blake3Inventory {
        let mut inv = Blake3Inventory::default();
        // The domain header + padded domain is absorbed at construction.
        let mut offset = 16 + domain_len.div_ceil(16) * 16;

        // Complete chunks at a byte offset: a chunk stays "current" until more
        // data follows it, so an exact multiple of 1024 has not closed yet.
        let complete_chunks = |o: usize| o.saturating_sub(1) / 1024;

        let finalize_at = |o: usize, out_bytes: usize, inv: &mut Blake3Inventory| {
            let c = complete_chunks(o);
            inv.finalize_blocks += 1;
            inv.finalize_parents += c.count_ones() as usize;
            inv.xof_blocks += out_bytes.div_ceil(64).saturating_sub(1);
        };

        for op in &self.ops {
            match op {
                TranscriptOp::SqueezeScalar | TranscriptOp::SqueezeSlice(_) => {
                    // The header is absorbed, THEN the state is finalized;
                    // the squeezed output is emitted, never re-absorbed.
                    offset += 16;
                    finalize_at(offset, op.squeezed_bytes(), &mut inv);
                    offset += op.absorbed_bytes() - 16;
                }
                TranscriptOp::Pow { .. } | TranscriptOp::LegacyPow { .. } => {
                    // `grind_pow` digests the state first, then absorbs the nonce.
                    finalize_at(offset, 32, &mut inv);
                    offset += op.absorbed_bytes();
                }
                _ => offset += op.absorbed_bytes(),
            }
        }

        // The live hasher compresses a block once it is full AND more input
        // arrives, so the final block waits for a finalize.
        inv.absorb_blocks = offset.saturating_sub(1) / 64;
        let c = complete_chunks(offset);
        inv.chunk_parents = c - (c.count_ones() as usize);
        inv
    }
}

impl Stream {
    /// The absorbed bytes, with every word resolved.
    pub fn to_bytes(&self, values: &[F128], payloads: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.words.len() * 16);
        for w in &self.words {
            let v = match *w {
                StreamWord::Const(c) => c,
                StreamWord::Value(i) => values[i],
                StreamWord::Bytes { payload, word } => {
                    let p = &payloads[payload];
                    let mut b = [0u8; 16];
                    let lo = word * 16;
                    let hi = (lo + 16).min(p.len());
                    if lo < p.len() {
                        b[..hi - lo].copy_from_slice(&p[lo..hi]);
                    }
                    F128::new(
                        u64::from_le_bytes(b[..8].try_into().unwrap()),
                        u64::from_le_bytes(b[8..].try_into().unwrap()),
                    )
                }
            };
            out.extend_from_slice(&v.lo.to_le_bytes());
            out.extend_from_slice(&v.hi.to_le_bytes());
        }
        out
    }
}

/// One 128-bit word of the absorbed byte stream.
///
/// The 16-byte-aligned framing makes the stream a sequence of whole 128-bit
/// words, and BLAKE3 consumes it 64 bytes — i.e. exactly four of these — at a
/// time as one block's `m`. So each word maps to one `m` word of one row, and
/// the circuit's job per word is to decide *where it comes from*:
///
/// - [`Const`](StreamWord::Const) → a public cell (op headers, label bytes,
///   padding),
/// - [`Value`](StreamWord::Value) → the wire already holding that proof value,
///   by **pure copy** — the whole point of aligning the framing,
/// Squeezed output never appears: it is not absorbed, so a challenge is only
/// ever an *output* of the chain, never an input to it. That is what leaves the
/// FS chain acyclic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamWord {
    Const(F128),
    /// The `i`-th observed value, counting `F128`s in observation order.
    Value(usize),
    /// Word `word` of the `payload`-th `observe_bytes` payload (or PoW nonce),
    /// zero-padded to a whole 16-byte word.
    Bytes {
        payload: usize,
        word: usize,
    },
}

impl TranscriptShape {
    /// The absorbed byte stream as 128-bit words, given the domain the
    /// challenger was built with.
    ///
    /// This is the circuit's placement map: word `k` of the result is `m` word
    /// `k % 4` of block `k / 4`. It is derived from the recorded shape and the
    /// framing constants in [`crate::challenger`], so there is one definition
    /// of the encoding, not two.
    pub fn stream_words(&self, domain: &[u8]) -> Stream {
        let (mut values, mut payloads) = (0usize, 0usize);
        Self::walk(&self.ops, domain, false, &mut values, &mut payloads)
    }

    /// [`Self::stream_words`] for the DUPLEX chain discipline
    /// (transcript-v3, [`crate::challenger::FsChallenger::with_chained_blake3`]):
    /// squeeze ops absorb NOTHING — the squeeze compression itself advances
    /// the state, so there is no `OP_SQUEEZE` header word. `finalize_after`
    /// still marks where each squeeze falls. A `Pow` contributes only its
    /// nonce (header + payload word); its state digest is a squeeze and
    /// absorbs nothing.
    ///
    /// The v2 layout stays the truth for the SHA-256 and tree-BLAKE3
    /// transcripts, whose immutable squeezes DO absorb a separating header.
    pub fn stream_words_duplex(&self, domain: &[u8]) -> Stream {
        let (mut values, mut payloads) = (0usize, 0usize);
        Self::walk(&self.ops, domain, true, &mut values, &mut payloads)
    }

    /// The shared stream walk. `duplex` selects the v3 squeeze framing (no
    /// `OP_SQUEEZE` header word). A [`TranscriptOp::Forked`] descends into
    /// its own chain — contributing no words here — while `values` and
    /// `payloads` keep counting GLOBALLY, matching the recorder's inline
    /// splice.
    fn walk(
        ops: &[TranscriptOp],
        domain: &[u8],
        duplex: bool,
        values: &mut usize,
        payloads: &mut usize,
    ) -> Stream {
        let header = |op: u8, kind: u8, len: u64| {
            StreamWord::Const(F128::new(op as u64 | ((kind as u64) << 8), len))
        };
        let padded = |b: &[u8], out: &mut Vec<StreamWord>| {
            for c in b.chunks(16) {
                let mut w = [0u8; 16];
                w[..c.len()].copy_from_slice(c);
                out.push(StreamWord::Const(F128::new(
                    u64::from_le_bytes(w[..8].try_into().unwrap()),
                    u64::from_le_bytes(w[8..].try_into().unwrap()),
                )));
            }
        };

        let mut out = Vec::new();
        let mut finalize_after: Vec<usize> = Vec::new();
        let mut forks: Vec<ForkStream> = Vec::new();
        // The domain is absorbed at construction, before recording starts.
        out.push(header(OP_DOMAIN, KIND_NONE, domain.len() as u64));
        padded(domain, &mut out);

        for op in ops {
            match op {
                TranscriptOp::Label(l) => {
                    out.push(header(OP_LABEL, KIND_NONE, l.len() as u64));
                    padded(l, &mut out);
                }
                TranscriptOp::ObserveScalar => {
                    out.push(header(OP_OBSERVE, KIND_SCALAR, 1));
                    out.push(StreamWord::Value(*values));
                    *values += 1;
                }
                TranscriptOp::ObserveSlice(n) => {
                    out.push(header(OP_OBSERVE, KIND_SLICE, *n as u64));
                    for _ in 0..*n {
                        out.push(StreamWord::Value(*values));
                        *values += 1;
                    }
                }
                TranscriptOp::ObserveBytes(len) => {
                    out.push(header(OP_BYTES, KIND_NONE, *len as u64));
                    for w in 0..len.div_ceil(16) {
                        out.push(StreamWord::Bytes {
                            payload: *payloads,
                            word: w,
                        });
                    }
                    *payloads += 1;
                }
                TranscriptOp::SqueezeScalar => {
                    if !duplex {
                        out.push(header(OP_SQUEEZE, KIND_SCALAR, 1));
                    }
                    finalize_after.push(out.len());
                }
                TranscriptOp::SqueezeSlice(n) => {
                    if !duplex {
                        out.push(header(OP_SQUEEZE, KIND_SLICE, *n as u64));
                    }
                    finalize_after.push(out.len());
                }
                // A fork adds NO words to this chain: the child is its own
                // stream. The two squeezes immediately before are its seed;
                // the child's first two absorbed words are those halves.
                TranscriptOp::Forked { label, ops: child } => {
                    assert!(
                        finalize_after.len() >= 2,
                        "a fork must follow its two seed squeezes"
                    );
                    let seed_squeeze = finalize_after.len() - 2;
                    let stream = Self::walk(child, label, duplex, values, payloads);
                    // The child's first ObserveScalar value word: after its
                    // domain header + padded label, plus the observe header.
                    let child_seed_word = 1 + label.len().div_ceil(16) + 1;
                    assert!(
                        matches!(
                            stream.words.get(child_seed_word),
                            Some(StreamWord::Value(_))
                        ),
                        "child chain must open by absorbing its seed"
                    );
                    assert!(
                        stream.finalize_after.len() >= 2,
                        "child chain must close with its digest squeezes"
                    );
                    let digest_squeeze = stream.finalize_after.len() - 2;
                    forks.push(ForkStream {
                        label: label.clone(),
                        stream,
                        seed_squeeze,
                        child_seed_word,
                        digest_squeeze,
                        // Filled by the matching `Merge` below.
                        parent_digest_word: usize::MAX,
                    });
                }
                // No words of its own: it marks that the next two
                // ObserveScalars carry fork `fork`'s closing digest.
                TranscriptOp::Merge { fork } => {
                    let f = forks
                        .get_mut(*fork)
                        .expect("Merge names a fork that was not recorded");
                    // The next op is an ObserveScalar: header at `out.len()`,
                    // its value word immediately after.
                    f.parent_digest_word = out.len() + 1;
                }
                TranscriptOp::Pow { .. } => {
                    if duplex {
                        // The fused nonce is one aligned private word
                        // immediately before the following squeeze. Its high
                        // half is zero padding constrained by recursion.
                        out.push(StreamWord::Bytes {
                            payload: *payloads,
                            word: 0,
                        });
                        *payloads += 1;
                    } else {
                        // Non-duplex challengers use the trait's compatibility
                        // implementation (legacy PoW, then an ordinary sample).
                        finalize_after.push(out.len());
                        out.push(header(OP_BYTES, KIND_NONE, 8));
                        out.push(StreamWord::Bytes {
                            payload: *payloads,
                            word: 0,
                        });
                        *payloads += 1;
                    }
                }
                TranscriptOp::LegacyPow { .. } => {
                    finalize_after.push(out.len());
                    out.push(header(OP_BYTES, KIND_NONE, 8));
                    out.push(StreamWord::Bytes {
                        payload: *payloads,
                        word: 0,
                    });
                    *payloads += 1;
                }
            }
        }
        debug_assert!(
            forks.iter().all(|f| f.parent_digest_word != usize::MAX),
            "every fork must be merged"
        );
        Stream {
            words: out,
            finalize_after,
            forks,
        }
    }
}

/// The absorbed stream, plus where its finalizes fall.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    /// The absorbed bytes as 128-bit words. Word `k` is `m` word `k % 4` of
    /// block `k / 4`.
    pub words: Vec<StreamWord>,
    /// For each finalization in order, how many words precede it.
    ///
    /// Needed because squeezed output is no longer fed back, so nothing in the
    /// stream itself marks a squeeze's position any more.
    pub finalize_after: Vec<usize>,
    /// Child chains forked from this one, in fork order. A fork contributes
    /// NO words to this stream — the child is an independent chain — so a
    /// consumer emits one row set per stream and connects them through the
    /// four cross-links each [`ForkStream`] names.
    pub forks: Vec<ForkStream>,
}

/// A forked child chain and its four cross-chain word links.
///
/// The child is a complete independent stream (own domain = the fork label,
/// own IV lineage). Its `Value`/`Bytes` indices share the parent's GLOBAL
/// numbering — the recorder splices child values in at the fork position, so
/// a single value table serves both chains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForkStream {
    pub label: Vec<u8>,
    pub stream: Stream,
    /// Index into the PARENT's `finalize_after` of the seed squeeze: its two
    /// output halves are the child's first two absorbed words.
    pub seed_squeeze: usize,
    /// Word index in the CHILD stream of the first seed word (the second is
    /// the next `Value` word after it).
    pub child_seed_word: usize,
    /// Index into the CHILD's `finalize_after` of the closing digest squeeze:
    /// its two output halves are what the parent absorbs at merge.
    pub digest_squeeze: usize,
    /// Word index in the PARENT stream of the first merge-digest word.
    pub parent_digest_word: usize,
}

/// A [`Challenger`] decorator that records the transcript's shape while
/// delegating every call to `inner` unchanged.
///
/// Transparent by construction: it computes no challenge itself, so a proof
/// verifies under `RecordingChallenger<Ch>` exactly when it verifies under
/// `Ch`. See the module docs for why every defaulted trait method is
/// nonetheless overridden.
pub struct RecordingChallenger<Ch: Challenger> {
    inner: Ch,
    ops: Vec<TranscriptOp>,
    values: Vec<F128>,
    payloads: Vec<Vec<u8>>,
    challenges: Vec<F128>,
    /// Set on a FORKED child recorder: the placeholder slot in the parent's
    /// `ops` this child's recording splices into at merge, its label, and
    /// the parent's value/payload/challenge counts at fork time (the splice
    /// offsets that keep GLOBAL indices consistent with the inline walk).
    fork: Option<ForkedAt>,
}

#[derive(Clone, Copy, Debug)]
struct ForkedAt {
    slot: usize,
    value_base: usize,
    payload_base: usize,
    challenge_base: usize,
}

impl<Ch: Challenger> RecordingChallenger<Ch> {
    pub fn new(inner: Ch) -> Self {
        Self {
            inner,
            ops: Vec::new(),
            values: Vec::new(),
            payloads: Vec::new(),
            challenges: Vec::new(),
            fork: None,
        }
    }

    /// Every observed `F128`, in observation order — the `Value(i)` words.
    pub fn values(&self) -> &[F128] {
        &self.values
    }

    /// Every `observe_bytes` payload and every PoW nonce, in order — the
    /// `Bytes { payload, .. }` words. PoW nonces are captured here because
    /// `grind_pow` absorbs them on the INNER challenger, so the decorator never
    /// sees the `observe_bytes` call.
    pub fn payloads(&self) -> &[Vec<u8>] {
        &self.payloads
    }

    /// Every squeezed `F128`, in squeeze order. Not part of the stream — it is
    /// not absorbed — but it is what the FS chain's circuit outputs.
    pub fn challenges(&self) -> &[F128] {
        &self.challenges
    }

    /// The shape recorded so far.
    ///
    /// Callers recording a *verify* should confirm the verify actually ran to
    /// completion before using this: the verifier early-returns on rejection,
    /// so a rejected proof yields a silently TRUNCATED shape, which would
    /// generate a circuit constraining only a prefix of the transcript.
    /// Record against an honest proof.
    pub fn shape(&self) -> TranscriptShape {
        TranscriptShape {
            ops: self.ops.clone(),
        }
    }

    pub fn into_parts(self) -> (Ch, TranscriptShape) {
        let shape = TranscriptShape {
            ops: self.ops.clone(),
        };
        (self.inner, shape)
    }
}

impl<Ch: Challenger> Challenger for RecordingChallenger<Ch> {
    fn supports_fused_pow_squeeze(&self) -> bool {
        self.inner.supports_fused_pow_squeeze()
    }

    fn hash_kind(&self) -> HashKind {
        // Forward — the trait default (SHA-256) would silently diverge any
        // out-of-sponge derivation (the AG-skip nonce decode) from the
        // inner transcript's hash during recording.
        self.inner.hash_kind()
    }

    fn fork_from_seed(&self, _seed: [F128; 2], _label: &'static [u8]) -> Self {
        // `fork` (below) is the recorded entry: it samples the seed THROUGH
        // the recorder so the two seed squeezes land on the tape, then
        // splices via the placeholder. A bare `fork_from_seed` would leave
        // the seed extraction unrecorded and the replay would diverge.
        unimplemented!("use fork() on a RecordingChallenger — the seed must be recorded")
    }

    fn fork(&mut self, label: &'static [u8]) -> Self {
        // Recorded fork: the two seed squeezes go on the parent tape as
        // ordinary ops (the convention consumers rely on: the two
        // SqueezeScalar immediately before the Forked slot are the seed),
        // then a placeholder Forked op reserves the fork position; the
        // child recorder splices into it at merge.
        let seed = [self.sample_f128(), self.sample_f128()];
        let slot = self.ops.len();
        self.ops.push(TranscriptOp::Forked {
            label: label.to_vec(),
            ops: Vec::new(),
        });
        RecordingChallenger {
            inner: self.inner.fork_from_seed(seed, label),
            ops: vec![TranscriptOp::ObserveScalar, TranscriptOp::ObserveScalar],
            values: vec![seed[0], seed[1]],
            payloads: Vec::new(),
            challenges: Vec::new(),
            fork: Some(ForkedAt {
                slot,
                value_base: self.values.len(),
                payload_base: self.payloads.len(),
                challenge_base: self.challenges.len(),
            }),
        }
    }

    fn merge_child(&mut self, mut child: Self) {
        // The child's closing digest, squeezed THROUGH the child recorder
        // (landing on its tape as its final two ops).
        let d0 = child.sample_f128();
        let d1 = child.sample_f128();
        let at = child
            .fork
            .expect("merge_child on a recorder that was not forked from this one");
        // Splice the child's recording into the fork position: its ops fill
        // the placeholder; its values/payloads/challenges insert at the
        // fork-time offsets, so GLOBAL indices match the inline walk
        // (pre-fork, child, post-fork).
        let fork_count = self.ops[..at.slot]
            .iter()
            .filter(|op| matches!(op, TranscriptOp::Forked { .. }))
            .count();
        match &mut self.ops[at.slot] {
            TranscriptOp::Forked { ops, .. } => *ops = child.ops,
            _ => panic!("fork slot does not hold a Forked placeholder"),
        }
        self.values
            .splice(at.value_base..at.value_base, child.values);
        self.payloads
            .splice(at.payload_base..at.payload_base, child.payloads);
        self.challenges
            .splice(at.challenge_base..at.challenge_base, child.challenges);
        // The merge marker, then the digest absorbs on the parent.
        self.ops.push(TranscriptOp::Merge { fork: fork_count });
        self.observe_f128(d0);
        self.observe_f128(d1);
    }

    fn observe_label(&mut self, label: &[u8]) {
        self.ops.push(TranscriptOp::Label(label.to_vec()));
        self.inner.observe_label(label);
    }

    fn observe_f128(&mut self, value: F128) {
        self.ops.push(TranscriptOp::ObserveScalar);
        self.values.push(value);
        self.inner.observe_f128(value);
    }

    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.ops.push(TranscriptOp::ObserveSlice(values.len()));
        self.values.extend_from_slice(values);
        // Delegate the SLICE call — not `n` scalar calls (see module docs).
        self.inner.observe_f128_slice(values);
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.ops.push(TranscriptOp::ObserveBytes(bytes.len()));
        self.payloads.push(bytes.to_vec());
        self.inner.observe_bytes(bytes);
    }

    fn sample_f128(&mut self) -> F128 {
        self.ops.push(TranscriptOp::SqueezeScalar);
        let c = self.inner.sample_f128();
        self.challenges.push(c);
        c
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.ops.push(TranscriptOp::SqueezeSlice(n));
        // Delegate the SLICE call — one squeeze, not `n` (see module docs).
        let c = self.inner.sample_f128_vec(n);
        self.challenges.extend_from_slice(&c);
        c
    }

    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.ops.push(TranscriptOp::LegacyPow { bits });
        let nonce = self.inner.grind_pow(bits);
        self.payloads.push(nonce.to_le_bytes().to_vec());
        nonce
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.ops.push(TranscriptOp::LegacyPow { bits });
        self.payloads.push(nonce.to_le_bytes().to_vec());
        self.inner.verify_pow(nonce, bits)
    }

    fn grind_pow_and_sample_f128(&mut self, bits: u32) -> (u64, F128) {
        self.ops.push(if self.inner.supports_fused_pow_squeeze() {
            TranscriptOp::Pow { bits }
        } else {
            TranscriptOp::LegacyPow { bits }
        });
        self.ops.push(TranscriptOp::SqueezeScalar);
        let (nonce, challenge) = self.inner.grind_pow_and_sample_f128(bits);
        self.payloads.push(nonce.to_le_bytes().to_vec());
        self.challenges.push(challenge);
        (nonce, challenge)
    }

    fn verify_pow_and_sample_f128(&mut self, nonce: u64, bits: u32) -> Option<F128> {
        self.ops.push(if self.inner.supports_fused_pow_squeeze() {
            TranscriptOp::Pow { bits }
        } else {
            TranscriptOp::LegacyPow { bits }
        });
        self.ops.push(TranscriptOp::SqueezeScalar);
        self.payloads.push(nonce.to_le_bytes().to_vec());
        let challenge = self.inner.verify_pow_and_sample_f128(nonce, bits)?;
        self.challenges.push(challenge);
        Some(challenge)
    }

    fn grind_pow_and_sample_f128_vec(&mut self, bits: u32, n: usize) -> (u64, Vec<F128>) {
        self.ops.push(if self.inner.supports_fused_pow_squeeze() {
            TranscriptOp::Pow { bits }
        } else {
            TranscriptOp::LegacyPow { bits }
        });
        self.ops.push(TranscriptOp::SqueezeSlice(n));
        let (nonce, challenges) = self.inner.grind_pow_and_sample_f128_vec(bits, n);
        self.payloads.push(nonce.to_le_bytes().to_vec());
        self.challenges.extend_from_slice(&challenges);
        (nonce, challenges)
    }

    fn verify_pow_and_sample_f128_vec(
        &mut self,
        nonce: u64,
        bits: u32,
        n: usize,
    ) -> Option<Vec<F128>> {
        self.ops.push(if self.inner.supports_fused_pow_squeeze() {
            TranscriptOp::Pow { bits }
        } else {
            TranscriptOp::LegacyPow { bits }
        });
        self.ops.push(TranscriptOp::SqueezeSlice(n));
        self.payloads.push(nonce.to_le_bytes().to_vec());
        let challenges = self.inner.verify_pow_and_sample_f128_vec(nonce, bits, n)?;
        self.challenges.extend_from_slice(&challenges);
        Some(challenges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use blake3::Hasher;
    use flock_hash::{BLAKE3_IV, blake3_compress};

    /// Drive a challenger through one op of every kind, returning the
    /// challenges it produced. Shared so the bare and decorated runs are
    /// driven by literally the same code.
    fn drive<Ch: Challenger>(ch: &mut Ch) -> Vec<F128> {
        let mut out = Vec::new();
        ch.observe_label(b"flock-record-test-v0");
        ch.observe_f128(F128::new(0xABCD, 0x1234));
        out.push(ch.sample_f128());
        ch.observe_f128_slice(&[F128::new(1, 2), F128::new(3, 4), F128::new(5, 6)]);
        out.extend(ch.sample_f128_vec(5));
        ch.observe_bytes(&[0xAA; 37]);
        let nonce = ch.grind_pow(0);
        assert!(ch.verify_pow(nonce, 0));
        out.push(ch.sample_f128());
        out
    }

    /// The decorator must not perturb the transcript it observes. This is the
    /// load-bearing test: `observe_f128_slice` and `sample_f128_vec` have
    /// default bodies that decompose into scalar calls, and inheriting either
    /// would silently change every challenge from that point on.
    #[test]
    fn recording_is_transparent() {
        for kind in [HashKind::Sha256, HashKind::Blake3] {
            let mut bare = FsChallenger::with_hash(b"transparency", kind);
            let expected = drive(&mut bare);

            let mut rec = RecordingChallenger::new(FsChallenger::with_hash(b"transparency", kind));
            let got = drive(&mut rec);

            assert_eq!(
                got, expected,
                "recording changed the challenge stream under {kind:?}"
            );
        }
    }

    /// A FORK recorded through the decorator is transparent (identical
    /// challenges to the bare challenger driving the same fork/merge), and
    /// the recorded shape yields a well-formed two-chain stream: the parent
    /// contributes no words for the fork, the child is its own stream under
    /// the fork label, and the four cross-links land on `Value` words.
    #[test]
    fn recording_a_fork_is_transparent_and_yields_two_chains() {
        // The protocol under test: absorb, fork, drive the child, merge,
        // then sample on the parent — the one-sided shape of the union
        // prover's wiring branch.
        fn drive_forked<C: Challenger + Sized>(ch: &mut C) -> Vec<F128> {
            let mut out = Vec::new();
            ch.observe_label(b"parent-phase");
            ch.observe_f128(F128::new(11, 0));
            out.push(ch.sample_f128());
            let mut child = ch.fork(b"child-domain");
            child.observe_label(b"child-phase");
            child.observe_f128(F128::new(22, 0));
            out.push(child.sample_f128());
            child.observe_f128(F128::new(33, 0));
            out.push(child.sample_f128());
            // Parent work concurrent with the child, in transcript terms.
            ch.observe_f128(F128::new(44, 0));
            out.push(ch.sample_f128());
            ch.merge_child(child);
            out.push(ch.sample_f128());
            out
        }

        for kind in [HashKind::Sha256, HashKind::Blake3] {
            let mut bare = FsChallenger::with_hash(b"forked", kind);
            let expected = drive_forked(&mut bare);
            let mut rec = RecordingChallenger::new(FsChallenger::with_hash(b"forked", kind));
            let got = drive_forked(&mut rec);
            assert_eq!(
                got, expected,
                "recording changed the challenge stream across a fork under {kind:?}"
            );
        }

        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(b"forked"));
        let _ = drive_forked(&mut rec);
        let shape = rec.shape();

        // The fork sits inline, preceded by its two seed squeezes and
        // followed (later) by its merge marker.
        let fork_at = shape
            .ops()
            .iter()
            .position(|o| matches!(o, TranscriptOp::Forked { .. }))
            .expect("a fork was recorded");
        assert!(
            matches!(shape.ops()[fork_at - 1], TranscriptOp::SqueezeScalar)
                && matches!(shape.ops()[fork_at - 2], TranscriptOp::SqueezeScalar),
            "the two seed squeezes must immediately precede the fork"
        );
        let merge_at = shape
            .ops()
            .iter()
            .position(|o| matches!(o, TranscriptOp::Merge { fork: 0 }))
            .expect("the fork was merged");
        assert!(merge_at > fork_at, "merge follows its fork");
        assert!(
            matches!(shape.ops()[merge_at + 1], TranscriptOp::ObserveScalar)
                && matches!(shape.ops()[merge_at + 2], TranscriptOp::ObserveScalar),
            "the merge digest absorbs two scalars on the parent"
        );

        let stream = shape.stream_words_duplex(b"forked");
        assert_eq!(stream.forks.len(), 1, "one child chain");
        let f = &stream.forks[0];
        assert_eq!(f.label, b"child-domain".to_vec());
        // The child is its own chain: its stream opens with its own domain
        // header, and none of its words leaked into the parent's.
        assert!(matches!(f.stream.words[0], StreamWord::Const(_)));
        // The four cross-links point at real value words.
        assert!(matches!(
            f.stream.words[f.child_seed_word],
            StreamWord::Value(_)
        ));
        assert!(matches!(
            stream.words[f.parent_digest_word],
            StreamWord::Value(_)
        ));
        assert!(f.seed_squeeze + 2 <= stream.finalize_after.len());
        assert!(f.digest_squeeze + 2 <= f.stream.finalize_after.len());

        // GLOBAL value numbering: the child's values are spliced in at the
        // fork position, so every `Value(i)` across BOTH chains indexes the
        // one table, each exactly once.
        let mut seen: Vec<usize> = stream
            .words
            .iter()
            .chain(f.stream.words.iter())
            .filter_map(|w| match w {
                StreamWord::Value(i) => Some(*i),
                _ => None,
            })
            .collect();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..rec.values().len()).collect::<Vec<_>>(),
            "every recorded value is placed exactly once across the two chains"
        );
    }

    #[test]
    fn shape_records_kinds_and_lengths_in_order() {
        let mut rec = RecordingChallenger::new(FsChallenger::new(b"shape"));
        let _ = drive(&mut rec);
        let shape = rec.shape();
        assert_eq!(
            shape.ops(),
            &[
                TranscriptOp::Label(b"flock-record-test-v0".to_vec()),
                TranscriptOp::ObserveScalar,
                TranscriptOp::SqueezeScalar,
                TranscriptOp::ObserveSlice(3),
                TranscriptOp::SqueezeSlice(5),
                TranscriptOp::ObserveBytes(37),
                TranscriptOp::LegacyPow { bits: 0 },
                TranscriptOp::LegacyPow { bits: 0 },
                TranscriptOp::SqueezeScalar,
            ]
        );
        // Five finalizing ops: three squeezes plus the two PoW state digests.
        assert_eq!(shape.finalizations(), 5);
        // Squeezes address by phase, not absolute index.
        assert_eq!(
            shape.squeeze_roles(),
            vec![
                (b"flock-record-test-v0".to_vec(), 0),
                (b"flock-record-test-v0".to_vec(), 1),
                (b"flock-record-test-v0".to_vec(), 2),
            ]
        );
    }

    /// The byte model in [`TranscriptOp::absorbed_bytes`] is what the circuit's
    /// packing gadgets will reproduce, so it is checked against the live
    /// challenger's own counter rather than trusted.
    #[cfg(feature = "hash-count")]
    #[test]
    fn absorbed_byte_model_matches_the_live_challenger() {
        let domain: &[u8] = b"bytes";
        let mut rec = RecordingChallenger::new(FsChallenger::new(domain));
        let _ = drive(&mut rec);
        let (inner, shape) = rec.into_parts();
        // The recorder wraps an already-constructed challenger, so the domain
        // separator (OP_DOMAIN ‖ len ‖ domain) is absorbed before recording
        // starts and is not part of the shape.
        let domain_bytes = (16 + domain.len().div_ceil(16) * 16) as u64;
        assert_eq!(
            shape.absorbed_bytes() as u64,
            inner.absorbed_bytes() - domain_bytes,
            "TranscriptOp::absorbed_bytes disagrees with FsChallenger"
        );
    }

    /// **The stream model is right**: reconstructing the absorbed bytes from a
    /// recorded shape and hashing them with plain BLAKE3 reproduces the
    /// challenge `FsChallenger` actually produced.
    ///
    /// This is the assumption the whole FS-chain circuit rests on — the
    /// circuit hashes the stream `stream_words` describes, so if that
    /// description is off by a byte the circuit proves the wrong transcript.
    /// Checked against the live challenger rather than derived.
    #[test]
    fn stream_words_reconstruct_what_the_challenger_absorbs() {
        let domain: &[u8] = b"flock-stream-model";
        let mut rec = RecordingChallenger::new(FsChallenger::with_hash(domain, HashKind::Blake3));

        // Absorb a spread of op kinds, then take one challenge. Everything
        // before the squeeze must be in the stream, byte for byte.
        let vals = [
            F128::new(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            F128::new(1, 2),
            F128::new(3, 4),
            F128::new(5, 6),
        ];
        rec.observe_label(b"phase-one");
        rec.observe_f128(vals[0]);
        rec.observe_f128_slice(&vals[1..4]);
        let got = rec.sample_f128();
        let shape = rec.shape();

        // Rebuild the stream, substituting the observed values.
        let stream = shape.stream_words(domain);
        assert_eq!(
            stream.finalize_after,
            vec![stream.words.len()],
            "one squeeze, at the end"
        );
        // Resolve through the recorder's own captures, so the reconstruction
        // uses exactly what the challenger saw.
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        assert_eq!(rec.values(), vals, "recorder captured the observed values");
        assert_eq!(rec.challenges(), &[got], "recorder captured the challenge");

        // `sample_f128` absorbs its header, then finalizes and takes 16 bytes.
        let mut h = Hasher::new();
        h.update(&bytes);
        let mut buf = [0u8; 16];
        h.finalize_xof().fill(&mut buf);
        let want = F128::new(
            u64::from_le_bytes(buf[..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..].try_into().unwrap()),
        );

        assert_eq!(
            got, want,
            "the reconstructed stream is not what FsChallenger absorbed — the \
             FS-chain circuit would hash the wrong bytes"
        );
        // Every word is whole: the stream is a multiple of 16 bytes, so each
        // BLAKE3 block is exactly four stream words and no value straddles one.
        assert_eq!(bytes.len() % 16, 0);
    }

    /// Duplex twin of the test above, against the CHAINED challenger
    /// (transcript-v3): replay `stream_words_duplex`'s byte stream through
    /// a local duplex sponge over the same compression primitive and check
    /// every squeeze — a mid-stream scalar (partial-block message + state
    /// advance) and a multi-block slice (extra zero-message output rows).
    /// A drifted layout, a leftover OP_SQUEEZE header, or a non-mutating
    /// squeeze all fail here.
    #[test]
    fn duplex_stream_words_reconstruct_the_chained_challenger() {
        const CHAIN_ABSORB: u32 = 1 << 6;
        const CHAIN_SQUEEZE: u32 = 1 << 7;
        let domain: &[u8] = b"flock-duplex-model";
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));

        let vals = [
            F128::new(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            F128::new(1, 2),
            F128::new(3, 4),
            F128::new(5, 6),
        ];
        let mut expected: Vec<Vec<u8>> = Vec::new();
        let push16 = |vs: &[F128], out: &mut Vec<Vec<u8>>| {
            let mut b = Vec::new();
            for v in vs {
                b.extend_from_slice(&v.lo.to_le_bytes());
                b.extend_from_slice(&v.hi.to_le_bytes());
            }
            out.push(b);
        };
        rec.observe_label(b"phase-one");
        rec.observe_f128(vals[0]);
        let c1 = rec.sample_f128(); // mid-stream: partial-block message
        push16(&[c1], &mut expected);
        rec.observe_f128_slice(&vals[1..4]);
        let c2 = rec.sample_f128_vec(5); // 80 B: one extra zero-message row
        push16(&c2, &mut expected);
        let c3 = rec.sample_f128(); // back-to-back: empty-pending squeeze
        push16(&[c3], &mut expected);

        let shape = rec.shape();
        let stream = shape.stream_words_duplex(domain);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        assert_eq!(bytes.len() % 16, 0);
        let fin_ops: Vec<_> = shape.ops().iter().filter(|o| o.finalizes()).collect();
        assert_eq!(stream.finalize_after.len(), fin_ops.len());

        // Local duplex replay over the same primitive.
        let mut cv = BLAKE3_IV;
        let mut pend: Vec<u8> = Vec::new();
        let drain = |cv: &mut [u32; 8], pend: &mut Vec<u8>| {
            while pend.len() >= 64 {
                let mut m = [0u32; 16];
                for (i, c) in pend[..64].chunks(4).enumerate() {
                    m[i] = u32::from_le_bytes(c.try_into().unwrap());
                }
                let out = blake3_compress(cv, &m, 0, 64, CHAIN_ABSORB);
                cv.copy_from_slice(&out[..8]);
                pend.drain(..64);
            }
        };
        let mut at = 0usize;
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            pend.extend_from_slice(&bytes[at * 16..upto * 16]);
            at = upto;
            drain(&mut cv, &mut pend);
            let want_bytes = fin_ops[k].squeezed_bytes();
            let mut got: Vec<u8> = Vec::new();
            let mut first = true;
            while got.len() < want_bytes {
                let (mut m, blen) = ([0u32; 16], if first { pend.len() as u32 } else { 0 });
                if first {
                    for (i, c) in pend.chunks(4).enumerate() {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        m[i] = u32::from_le_bytes(w);
                    }
                }
                let out = blake3_compress(&cv, &m, 0, blen, CHAIN_SQUEEZE);
                cv.copy_from_slice(&out[..8]);
                for w in out.iter() {
                    got.extend_from_slice(&w.to_le_bytes());
                }
                pend.clear();
                first = false;
            }
            got.truncate(want_bytes);
            // Pow squeezes (state digests) are not surfaced as challenges;
            // every op in this schedule is a plain squeeze.
            assert_eq!(got, expected[k], "squeeze {k}");
        }
        assert_eq!(at * 16, bytes.len(), "every stream word consumed");
    }

    #[test]
    fn first_difference_names_the_op() {
        let a = TranscriptShape {
            ops: vec![TranscriptOp::ObserveScalar, TranscriptOp::SqueezeScalar],
        };
        let b = TranscriptShape {
            ops: vec![TranscriptOp::ObserveScalar, TranscriptOp::SqueezeSlice(2)],
        };
        assert_eq!(a.first_difference(&a), None);
        assert_eq!(a.first_difference(&b), Some(1));
        // A shorter prefix differs at the point it runs out — the truncation
        // case an early-returning verifier would produce.
        let short = TranscriptShape {
            ops: vec![TranscriptOp::ObserveScalar],
        };
        assert_eq!(a.first_difference(&short), Some(1));
    }
}
