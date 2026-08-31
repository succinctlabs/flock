//! Merkle-path layout and node-hash spec: the per-path block geometry
//! ([`MerkleTreeLayout`]), the node compression spec ([`HashSpec`],
//! [`blake3_spec`]) and the chunk-leaf path row ([`ChunkPathInput`]) that the
//! recursion tower's Merkle gates and [`super::merkle_glue`] build on.
//!
//! A whole path lives in ONE witness block, so a Merkle path is a legal
//! **table type** for the multi-table union (`flock_core::schedule::TableType`):
//! the union model forbids constraints between rows, so one row is one
//! self-contained path. The monolithic per-path walker circuit that used to
//! live here (`MerkleWalkerCircuit`, the `PathInput` row and the
//! `merkle26+blake3` registry tiers) was retired 2026-08-27 with the
//! Merkle-path product (bloat ledger §C 2.2); the layout keeps the row
//! geometry so the tower's differential Merkle gate still models a
//! full-depth path.
//!
//! ## Statement
//!
//! For a leaf digest `L`, an index `i ∈ [0, 2^depth)`, sibling digests
//! `S_0 … S_{depth−1}` and a root `R`, each level `l` computes
//!
//! ```text
//!   b_l    = bit l of i                       (the indicator bit)
//!   left   = b_l·prev ⊕ (1 ⊕ b_l)·S_l
//!   right  = left ⊕ prev ⊕ S_l
//!   prev'  = H(left ‖ right)
//! ```
//!
//! with `prev = L` at level 0 and `R = prev` after the last level. Over
//! GF(2) the pair `(left, right)` is exactly the conditional swap: `b_l = 1`
//! puts the running digest on the left, `b_l = 0` on the right.
//!
//! ## The chunk-leaf variant (PCS L0 openings)
//!
//! [`MerkleTreeLayout::with_blake3_chunk_leaf`] prepends a **chunk-leaf
//! segment** to the node levels: `leaf_bytes/64` base blocks that hash the
//! raw leaf bytes as one BLAKE3 chunk (`CHUNK_START` on the first block,
//! `CHUNK_END` on the last, chaining through `h_in`, counter 0), whose final
//! chaining value seeds `prev` in place of the leaf-digest global. One row
//! then verifies one PCS L0 opening — leaf hash AND path — under exactly
//! `flock_merkle`'s BLAKE3 tree semantics (leaf = non-root chunk CV of
//! the leaf bytes, node = non-root PARENT compression). Chunk blocks need no
//! gadget columns at all: the base encoder's free message region IS the leaf
//! data, and its chaining-value rows are witness-identical to the pin (block
//! 0: IV) and copy (block `i`: block `i−1`'s output) overrides. The walker
//! is oblivious — chunk blocks embed the same stripped base at their subcube
//! offset, and everything flavor-specific rides in the extras.
//!
//! ## The swap gadget under `C = I`
//!
//! The R1CS is the circuit shape `(A·z) ⊙ (B·z) = z`: every witness column
//! is the output of exactly one row, so a row's right-hand side is a single
//! wire and linear relations need the constant-one wire on the `B` side.
//! `left` is quadratic in the witness, so it needs one AND per bit:
//!
//! ```text
//!   t_j      = b_l · (prev_j ⊕ S_{l,j})     A = [b_l],        B = [prev_j, S_{l,j}]
//!   left_j   = S_{l,j} ⊕ t_j                A = [S_{l,j}, t_j], B = [const]
//!   right_j  = prev_j  ⊕ t_j                A = [prev_j, t_j],  B = [const]
//! ```
//!
//! (`right = left ⊕ prev ⊕ S = t ⊕ prev`, so both halves cost one linear
//! row.) Crucially `left_j`/`right_j` are **not** new columns: they ARE the
//! hash block's 512-bit message region, whose rows the composite *replaces*
//! — the base encoder makes them free inputs, we make them gadget outputs.
//! So the gadget costs only the `2^8` AND columns `t_j` per level.
//!
//! ## Composite layout (`depth = D`, base block `2^κ` wide, `useful_bits = U`)
//!
//! Level `l` occupies the **aligned subcube** `[l·2^κ, (l+1)·2^κ)` — a
//! level's slot IS the base block. The per-level gadget columns and the
//! globals live in the base block's own padding region `[U, 2^κ)`:
//!
//! ```text
//!   per level l, at l·2^κ:
//!     z[l·2^κ         .. l·2^κ + U)       = the hash block, verbatim
//!     z[l·2^κ + U     .. +256)            = sibling S_l         (free input)
//!     z[l·2^κ + U+256 .. +512)            = t_l (the ANDs)
//!   level 0 additionally:
//!     z[U+512]                            = 1     (the table's const_pin)
//!     z[U+513 .. U+769)                   = leaf digest         (free input)
//!     z[U+769 .. U+769+D)                 = index bits b_0..b_D (free input)
//!   every level's tail, and slots ≥ D     = padding (forced 0 by empty rows)
//! ```
//!
//! so `k_log = κ + log2(next_pow2(D))`.
//!
//! **The alignment is load-bearing**, not cosmetic. It makes the level index
//! a set of high address bits, which is what lets the lincheck's `eq` table
//! factor across levels (the retired walker circuit's lincheck relied on
//! that; the tower's differential gate keeps the geometry). Padding each level up
//! to `2^κ` costs ~2.7% more columns than tight packing and does not move
//! `k_log`.
//!
//! Each level embeds the base encoder's block by a **pure column shift**;
//! the composite then overrides exactly three row groups per level: the
//! block's own constant wire (re-derived from the global one), the 512-bit
//! message region (the swap gadget above), and every other free input —
//! the input chaining value, counter, block length and flags — which is
//! pinned to the Merkle node constants. Everything else, every row of the
//! hash relation, is the base matrix with its indices shifted.
//!
//! ## What this does NOT enforce
//!
//! Public-input binding, exactly as in the per-hash encoders: the leaf, the
//! index bits and the root are free witness columns at fixed offsets
//! ([`MerkleTreeLayout::leaf_bit`], [`MerkleTreeLayout::index_bit`],
//! [`MerkleTreeLayout::root_bit`]). The claim-level glue binds them to
//! public values.

use blake3::BLAKE3_IV;
use blake3::BLEN_BASE;
use blake3::CV_BASE;
use blake3::FLAGS_BASE;
use blake3::K_LOG;
use blake3::M_BASE;
use blake3::OUT_LO_BASE;
use blake3::T_HI_BASE;
use blake3::T_LO_BASE;
use blake3::USEFUL_BITS;
use blake3::WORD_BITS;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};
use flock_core::zerocheck::K_SKIP;
use flock_hash::blake3_compress;
use std::array::from_fn;
use std::sync::OnceLock;

use super::blake3;
use super::common::{empty_matrix, identity};
use flock_core::schedule::IoWord;

/// Bits in one digest / chaining value. Both supported encoders lay their
/// input and output chaining values out as aligned `2^8`-bit slots.
pub const SLOT_BITS: usize = 256;
/// 32-bit words per digest.
pub const SLOT_WORDS: usize = SLOT_BITS / 32;

// ---------------------------------------------------------------------------
// Merkle node constants
// ---------------------------------------------------------------------------

/// Counter input to every node compression. Merkle parent nodes are not
/// chunks, so the chunk counter is 0.
pub const NODE_COUNTER: u64 = 0;
/// Block length: a parent node compresses exactly two 32-byte digests.
pub const NODE_BLOCK_LEN: u32 = 64;
/// BLAKE3 `PARENT` domain flag. Note this is applied at EVERY level,
/// including the top one — real BLAKE3 tree hashing also sets `ROOT` on the
/// final parent. Keeping all levels uniform is what lets the composite be
/// `depth` copies of one base block; set [`HashSpec::flags`] if you need the
/// bit-exact BLAKE3 tree.
///
/// This IS the semantics of `flock_merkle`'s BLAKE3 mode: its
/// internal nodes are non-root PARENT-flagged chaining values
/// (`hazmat::merge_subtrees_non_root`), so the node levels here match the
/// PCS commitment bit-for-bit with no `flags` override.
pub const BLAKE3_FLAG_PARENT: u32 = 4;
/// BLAKE3 `CHUNK_START` flag: first block of a chunk.
pub const BLAKE3_FLAG_CHUNK_START: u32 = 1;
/// BLAKE3 `CHUNK_END` flag: last block of a chunk.
pub const BLAKE3_FLAG_CHUNK_END: u32 = 2;

// ---------------------------------------------------------------------------
// Hash backend description
// ---------------------------------------------------------------------------

/// Geometry and witness hooks of the per-level hash's R1CS block.
///
/// The composite needs to know only where the base encoder keeps its input
/// chaining value, its output chaining value, its message region and its
/// constant wire — plus how to build one block's witness. Both shipped
/// encoders use the same "I/O-aligned" shape (input CV in aligned slot 0,
/// output CV in aligned slot 1, message right after), so a second backend is
/// one constructor.
#[derive(Clone)]
pub struct HashSpec {
    /// log2 of the base block width.
    pub k_log: usize,
    /// Useful columns of the base block.
    pub useful_bits: usize,
    /// Base offset of the 256-bit input chaining value.
    pub in_cv_base: usize,
    /// Base offset of the 256-bit output chaining value (the node digest).
    pub out_cv_base: usize,
    /// Base offset of the 512-bit message region: `left ‖ right`.
    pub msg_base: usize,
    /// Domain flags passed to every node compression.
    pub flags: u32,
    /// One compression's output chaining value, and NOTHING else. A circuit
    /// gate computing a root does `leaf_blocks + depth` compressions per
    /// opening and needs no witness at all, so it uses this. (The witness
    /// builders that materialized a `2^k_log`-bool block per node went with
    /// the walker circuit, 2026-08-27.)
    pub compress: fn(&[u32; SLOT_WORDS], &[u32; 16], u64, u32, u32) -> [u32; SLOT_WORDS],
    /// Base-block columns the composite pins to constants, as
    /// `(column, value)`: every free input of the base encoder EXCEPT the
    /// message region, which the swap gadget drives.
    pub fixed_bits: fn() -> Vec<(usize, bool)>,
}

/// BLAKE3 backend (the default). One level = one
/// `compress(IV, left‖right, 0, 64, PARENT)`.
pub fn blake3_spec() -> HashSpec {
    HashSpec {
        k_log: K_LOG,
        useful_bits: USEFUL_BITS,
        in_cv_base: CV_BASE,
        out_cv_base: OUT_LO_BASE,
        msg_base: M_BASE,
        flags: BLAKE3_FLAG_PARENT,
        compress: blake3_compress_cv,
        fixed_bits: blake3_fixed_bits,
    }
}

/// `left ‖ right` as BLAKE3's 16-word message block.
fn node_msg(left: &[u32; SLOT_WORDS], right: &[u32; SLOT_WORDS]) -> [u32; 16] {
    let mut m = [0u32; 16];
    m[..SLOT_WORDS].copy_from_slice(left);
    m[SLOT_WORDS..].copy_from_slice(right);
    m
}

/// The output chaining value of one BLAKE3 compression — the first 8 words of
/// the 16-word output, which is what every non-XOF use takes.
fn blake3_compress_cv(
    cv: &[u32; SLOT_WORDS],
    m: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; SLOT_WORDS] {
    let out = blake3_compress(cv, m, counter, block_len, flags);
    out[..SLOT_WORDS].try_into().expect("SLOT_WORDS ≤ 16")
}

/// BLAKE3's free inputs other than the message: `cv = IV`, `counter = 0`,
/// `block_len = 64`, `flags = PARENT`.
fn blake3_fixed_bits() -> Vec<(usize, bool)> {
    let w = WORD_BITS;
    let mut out = Vec::with_capacity(SLOT_BITS + 4 * w);
    for (word, iv) in BLAKE3_IV.iter().enumerate() {
        for b in 0..w {
            out.push((CV_BASE + word * w + b, (iv >> b) & 1 == 1));
        }
    }
    for (base, val) in [
        (T_LO_BASE, NODE_COUNTER as u32),
        (T_HI_BASE, (NODE_COUNTER >> 32) as u32),
        (BLEN_BASE, NODE_BLOCK_LEN),
        (FLAGS_BASE, BLAKE3_FLAG_PARENT),
    ] {
        for b in 0..w {
            out.push((base + b, (val >> b) & 1 == 1));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Composite layout
// ---------------------------------------------------------------------------

/// Column layout of the composite Merkle-path block. All offsets are bit
/// indices into one table row (one path).
///
/// Level `l` occupies the **aligned subcube** `[l·2^κ, (l+1)·2^κ)` where
/// `κ = spec.k_log` — i.e. a level's slot IS the base block. That alignment
/// made the level index a set of high address bits, so the (since-retired)
/// walker circuit's lincheck `eq` table factored across levels; the layout
/// keeps it so the tower's differential Merkle gate models the same rows.
#[derive(Clone)]
pub struct MerkleTreeLayout {
    pub spec: HashSpec,
    /// Number of levels = tree depth.
    pub depth: usize,
    /// Chunk-leaf blocks preceding the node levels, or 0 for the plain
    /// digest-leaf path. See [`Self::with_blake3_chunk_leaf`].
    pub leaf_blocks: usize,
    /// log2 of the composite block width: `spec.k_log + log2(blocks rounded
    /// up to a power of two)` where `blocks = leaf_blocks + depth`.
    pub k_log: usize,
    /// Useful columns of the composite block. Note the useful region has
    /// interior holes — each level's slot has a tail of genuine padding
    /// (`2^κ − spec.useful_bits − 512`, and less in level 0 which also holds
    /// the globals). Those columns are forced to zero by empty rows.
    pub useful_bits: usize,
}

impl MerkleTreeLayout {
    /// Lay out a **chunk-leaf** path: `leaf_bytes` of leaf data hashed as a
    /// single BLAKE3 chunk (a chain of `leaf_bytes/64` compressions,
    /// `CHUNK_START` on the first block and `CHUNK_END` on the last, chaining
    /// through `h_in`), followed by `depth` PARENT-node levels with the swap
    /// gadget. This is exactly `flock_merkle`'s BLAKE3 mode — leaf =
    /// non-root chaining value of the leaf bytes, node = non-root
    /// PARENT-flagged compression — so one row verifies one PCS L0 opening
    /// bit-for-bit.
    ///
    /// The leaf digest global of the digest-leaf layout disappears: node
    /// level 0's `prev` IS the last chunk block's output chaining value, and
    /// the leaf enters as data through the chunk blocks' 512-bit message
    /// regions (free inputs at [`Self::leaf_data_bit`]).
    ///
    /// `leaf_bytes` must be a positive multiple of 64 and at most 1024 (one
    /// chunk): whole 64-byte blocks keep `block_len` uniform at 64, and a
    /// single chunk keeps the counter at 0 with no chunk-tree merge. The 1
    /// KiB PCS leaf is 16 blocks.
    pub fn with_blake3_chunk_leaf(depth: usize, leaf_bytes: usize, spec: HashSpec) -> Self {
        assert!(depth >= 1, "depth must be ≥ 1");
        assert!(
            (64..=1024).contains(&leaf_bytes) && leaf_bytes.is_multiple_of(64),
            "leaf_bytes {leaf_bytes} must be a positive multiple of 64 ≤ 1024 (one chunk)"
        );
        let leaf_blocks = leaf_bytes / 64;
        let blocks = leaf_blocks + depth;
        let k_log = spec.k_log + blocks.next_power_of_two().trailing_zeros() as usize;
        assert!(
            spec.k_log >= 7,
            "the union's BatchMajor chunking requires k_log ≥ 7"
        );
        // Node-level gadget columns fit the base padding (as in `new`), and
        // the globals (const + index bits) fit chunk block 0's padding.
        assert!(
            spec.useful_bits + 2 * SLOT_BITS <= 1usize << spec.k_log,
            "the swap gadget does not fit the base block's padding"
        );
        // The globals are the constant-one column plus a word-aligned 128-bit
        // index WORD (see `index_word_base`), not a tight run of `depth` bits.
        let index_end = (spec.useful_bits + 1).div_ceil(128) * 128 + 128;
        assert!(
            index_end <= 1usize << spec.k_log,
            "the index word does not fit chunk block 0's padding: {index_end} > 2^{}",
            spec.k_log
        );
        // The last block is a node level (depth ≥ 1), so the last nonzero
        // column is its `t` region's end.
        let useful_bits = ((blocks - 1) << spec.k_log) + spec.useful_bits + 2 * SLOT_BITS;
        debug_assert!(useful_bits <= 1usize << k_log);
        Self {
            spec,
            depth,
            leaf_blocks,
            k_log,
            useful_bits,
        }
    }

    /// Total base-block subcubes: the chunk-leaf segment plus the node
    /// levels. This — not `depth` — is what tiles the composite.
    pub fn total_blocks(&self) -> usize {
        self.leaf_blocks + self.depth
    }

    /// First column of subcube `t` (chunk blocks first, then node levels).
    fn block_base(&self, t: usize) -> usize {
        debug_assert!(t < self.total_blocks());
        t << self.spec.k_log
    }

    /// Composite width `2^k_log`.
    pub fn k(&self) -> usize {
        1usize << self.k_log
    }

    /// The global constant-one column, and the table's `const_pin`. Lives in
    /// the first subcube's padding region: after level 0's gadget columns on
    /// the digest-leaf path, right at the padding start on the chunk-leaf
    /// path (chunk blocks have no gadget columns).
    pub fn const_pos(&self) -> usize {
        if self.leaf_blocks == 0 {
            self.spec.useful_bits + 2 * SLOT_BITS
        } else {
            self.spec.useful_bits
        }
    }

    /// Bit `j` of chunk block `i`'s 512-bit slice of the leaf data — the
    /// block's message region.
    pub fn leaf_data_bit(&self, block: usize, j: usize) -> usize {
        debug_assert!(block < self.leaf_blocks);
        debug_assert!(j < 2 * SLOT_BITS);
        self.block_base(block) + self.spec.msg_base + j
    }

    /// First column of the index **word** — chunk-leaf layouts only.
    ///
    /// The index is a full 128-bit word at a word-aligned position, not a
    /// tight run of `depth` bits, so that a circuit can WIRE it. The Merkle
    /// index is its low `depth` bits; the rest are free and unread.
    ///
    /// That is what makes the Fiat–Shamir query binding free of any gadget.
    /// `sample_queries` computes `(v.lo as usize) & (block_len − 1)` with
    /// `block_len = 2^depth`, so the query index IS the low `depth` bits of the
    /// challenge word. Wire the challenge straight into this word and the
    /// masking is not a computation at all — it is expressed by which bits the
    /// relation reads. The high bits ride along, pinned by the copy constraint
    /// and ignored by the relation.
    pub fn index_word_base(&self) -> usize {
        debug_assert!(self.leaf_blocks > 0, "digest-leaf layouts pack the index");
        // Clear the constant-one column, then round up to a word boundary.
        (self.const_pos() + 1).div_ceil(128) * 128
    }

    /// First column of level `l`'s aligned subcube (node levels sit after
    /// the chunk-leaf segment).
    fn level_base(&self, level: usize) -> usize {
        debug_assert!(level < self.depth);
        self.block_base(self.leaf_blocks + level)
    }

    /// Domain flags of chunk block `i`: `CHUNK_START` on the first block,
    /// `CHUNK_END` on the last (both on a single-block leaf), non-root.
    fn chunk_flags(&self, block: usize) -> u32 {
        debug_assert!(block < self.leaf_blocks);
        let mut f = 0;
        if block == 0 {
            f |= BLAKE3_FLAG_CHUNK_START;
        }
        if block + 1 == self.leaf_blocks {
            f |= BLAKE3_FLAG_CHUNK_END;
        }
        f
    }

    // -----------------------------------------------------------------------
    // Wiring IO schema
    // -----------------------------------------------------------------------

    /// The wireable words of one opening — chunk-leaf layouts only.
    ///
    /// Inputs: the leaf data (`4` words per chunk block, in block order) then
    /// the index word. Outputs: the two halves of the root.
    ///
    /// **What is deliberately absent is the sibling path.** Each level's
    /// sibling digest sits inside that level's base-block padding, at
    /// `useful_bits`, which is not word-aligned
    /// and shares its word with the level's own hash columns. It is therefore
    /// not expressible as a schema word at all. That costs nothing here: a
    /// sibling is free witness read by no other gate, and the relation already
    /// binds it, because the level's compression consumes it and the chain
    /// terminates in the root this schema exports. A prover who changes a
    /// sibling changes the root. Circuits supply the path as a
    /// [`GateType::Hint`](flock_core::circuit::builder::GateType::Hint).
    ///
    /// Everything that IS here has an outside claimant: the leaf data is read
    /// by whatever proves the opened values, the index binds to the
    /// Fiat–Shamir query, and the root binds to the committed root.
    pub fn io_schema(&self) -> Vec<IoWord> {
        assert!(
            self.leaf_blocks > 0,
            "io_schema is chunk-leaf only: the digest-leaf layout packs its \
             index tightly, so no word-aligned index word exists to wire"
        );
        let w = |bit: usize| {
            debug_assert_eq!(bit % 128, 0, "schema word {bit} is not 128-aligned");
            bit / 128
        };
        let mut schema = Vec::with_capacity(4 * self.leaf_blocks + 3);
        for block in 0..self.leaf_blocks {
            for j in 0..4 {
                schema.push(IoWord::input(w(self.leaf_data_bit(block, 128 * j))));
            }
        }
        schema.push(IoWord::input(w(self.index_word_base())));
        schema.push(IoWord::output(w(self.root_bit(0))));
        schema.push(IoWord::output(w(self.root_bit(128))));
        schema
    }

    /// Base-block column `c` of level `l`'s embedded hash block. The
    /// embedding is now a pure shift by `l·2^κ`.
    pub fn hash_bit(&self, level: usize, c: usize) -> usize {
        debug_assert!(c < 1usize << self.spec.k_log);
        self.level_base(level) + c
    }

    /// Bit `j` of the root — the last level's output chaining value.
    pub fn root_bit(&self, j: usize) -> usize {
        self.hash_bit(self.depth - 1, self.spec.out_cv_base + j)
    }

    // -----------------------------------------------------------------------
    // Matrices
    // -----------------------------------------------------------------------

    /// [`BlockR1cs`] with **empty `(A_0, B_0)` stubs**: the constraint
    /// definition is supplied out of band (the tower's differential Merkle
    /// gate stubs its table this way; the retired walker circuit did too).
    ///
    /// Consequence, inherited by `Registry::digest`: the statement digest
    /// binds `k_log`, `useful_bits` and `const_pin` — hence the depth and,
    /// in practice, the hash backend — but **not** the constraint system.
    /// A verifier's guarantee rests on it constructing the matching walker
    /// out of band.
    pub fn build_block_r1cs_stub(&self, n_paths_log: usize) -> BlockR1cs {
        let k = self.k();
        self.block_r1cs_with(n_paths_log, empty_matrix(k), empty_matrix(k))
    }

    fn block_r1cs_with(
        &self,
        n_paths_log: usize,
        a_0: SparseBinaryMatrix,
        b_0: SparseBinaryMatrix,
    ) -> BlockR1cs {
        BlockR1cs {
            m: n_paths_log + self.k_log,
            k_log: self.k_log,
            k_skip: K_SKIP,
            useful_bits: self.useful_bits,
            a_0,
            b_0,
            c_0: identity(self.k()),
            layout: WitnessLayout::BatchMajor,
            const_pin: Some(self.const_pos()),
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Witness
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Chunk-leaf witness (the `_chunk` builders)
    // -----------------------------------------------------------------------

    fn assert_chunk_input(&self, input: &ChunkPathInput) {
        assert!(
            self.leaf_blocks > 0,
            "digest-leaf layout: use the PathInput builders"
        );
        assert_eq!(
            input.leaf_data.len(),
            64 * self.leaf_blocks,
            "leaf data must be {} bytes",
            64 * self.leaf_blocks
        );
        assert_eq!(
            input.siblings.len(),
            self.depth,
            "need one sibling digest per level"
        );
        // No bound on `index`: the word's high bits are free by construction
        // (see `ChunkPathInput::index`). Only the low `depth` are read.
    }

    /// The input chaining value the fixed set pins — chunk block 0 starts
    /// from the same IV the node compressions use. Decoded from
    /// [`HashSpec::fixed_bits`] so the witness and the pin rows cannot
    /// disagree.
    fn pinned_in_cv(&self) -> [u32; SLOT_WORDS] {
        let base = self.spec.in_cv_base;
        let mut cv = [0u32; SLOT_WORDS];
        for &(c, v) in &(self.spec.fixed_bits)() {
            if v && c >= base && c < base + SLOT_BITS {
                let j = c - base;
                cv[j / 32] |= 1u32 << (j % 32);
            }
        }
        cv
    }

    /// Root of a chunk-leaf opening, computing **only** the root.
    ///
    /// Folds the path through [`HashSpec::compress`] instead of the (retired)
    /// witness builders — so it does
    /// `leaf_blocks + depth` compressions and allocates nothing, rather than
    /// materializing a `2^k_log`-bool block per compression to read 256 bits
    /// out of it. At the L0 shape (16 + 13 per opening) that is the difference
    /// between ~400 µs and ~20 µs per opening.
    ///
    /// This is what a circuit gate wants: it needs the root to wire, and the
    /// witness comes later in bulk from the batch-major drivers.
    pub fn root_chunk(&self, input: &ChunkPathInput) -> [u32; SLOT_WORDS] {
        self.assert_chunk_input(input);
        let compress = self.spec.compress;
        let mut prev = self.pinned_in_cv();
        for i in 0..self.leaf_blocks {
            let m = leaf_msg_words(&input.leaf_data, i);
            prev = compress(&prev, &m, NODE_COUNTER, NODE_BLOCK_LEN, self.chunk_flags(i));
        }
        for (l, sib) in input.siblings.iter().enumerate() {
            let bit = (input.index >> l) & 1 == 1;
            let (left, right) = if bit { (*sib, prev) } else { (prev, *sib) };
            prev = compress(
                &self.pinned_in_cv(),
                &node_msg(&left, &right),
                NODE_COUNTER,
                NODE_BLOCK_LEN,
                self.spec.flags,
            );
        }
        prev
    }
}

/// One chunk-leaf opening: the raw leaf bytes (`64 · leaf_blocks` of them),
/// the leaf's index word, and one sibling chaining value per level (level 0 =
/// closest to the leaf).
#[derive(Clone, Debug)]
pub struct ChunkPathInput {
    pub leaf_data: Vec<u8>,
    /// The **whole 128-bit index word**, of which the Merkle position is the
    /// low `depth` bits — see [`MerkleTreeLayout::index_word_base`].
    ///
    /// It is a full word because it is wired: a circuit binds this opening to
    /// a Fiat–Shamir query by connecting the challenge word here, and a copy
    /// constraint relates whole words. The bits at or above `depth` are
    /// committed and pinned by that constraint but read by no relation row,
    /// so they carry the challenge's remaining bits at no cost and the
    /// masking `sample_queries` does natively — `& (block_len − 1)` — needs no
    /// gadget. A caller that only has a position passes `pos as u128`.
    pub index: u128,
    pub siblings: Vec<[u32; SLOT_WORDS]>,
}

/// Chunk block `i`'s 16-word message: bytes `[64i, 64(i+1))` of the leaf
/// data as little-endian words, per the BLAKE3 spec.
fn leaf_msg_words(data: &[u8], block: usize) -> [u32; 16] {
    from_fn(|w| {
        let o = block * 64 + 4 * w;
        u32::from_le_bytes(data[o..o + 4].try_into().unwrap())
    })
}

// ---------------------------------------------------------------------------
// Digest bit helpers — word `j/32`, bit `j%32`, matching the encoders'
// `write_word` (LSB-first within a 32-bit word).
// ---------------------------------------------------------------------------
