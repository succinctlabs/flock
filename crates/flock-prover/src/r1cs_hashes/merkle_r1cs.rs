//! Merkle-path verification as a **single monolithic R1CS block** — one
//! table row per path.
//!
//! Distinct from [`super::merkle_path_common`], which proves Merkle paths
//! with a bespoke shift sumcheck layered on top of a batch of independent
//! per-level compressions. Here the whole path lives in ONE witness block,
//! so the level-to-level dataflow is expressed as ordinary R1CS rows. That
//! is what makes a Merkle path a legal **table type** for the multi-table
//! union (`flock_core::schedule::TableType`): the union model forbids
//! constraints between rows (design doc §3, "no constraints connecting
//! different rows, neither within a table nor across tables"), so one row
//! must be one self-contained path.
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
//! `flock_core::merkle`'s BLAKE3 tree semantics (leaf = non-root chunk CV of
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
//! factor across levels — see [`MerkleWalkerCircuit`]. Padding each level up
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
//! [`MerkleTreeLayout::root_bit`]). Binding them to public values is the job
//! of the claim-level glue (or the planned lookup/bus layer — design doc
//! §8, §11).

use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};
use flock_core::schedule::IoWord;
use flock_core::union::SlotWitnessDest;
use flock_core::zerocheck::K_SKIP;
use std::array::from_fn;
use std::env::var;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;
use std::sync::OnceLock;

use super::blake3;
use super::common::{empty_matrix, identity};

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
/// This IS the semantics of `flock_core::merkle`'s BLAKE3 mode: its
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
    pub name: &'static str,
    /// log2 of the base block width.
    pub k_log: usize,
    /// Useful columns of the base block.
    pub useful_bits: usize,
    /// The base block's own constant-one column.
    pub z_const_pos: usize,
    /// Base offset of the 256-bit input chaining value.
    pub in_cv_base: usize,
    /// Base offset of the 256-bit output chaining value (the node digest).
    pub out_cv_base: usize,
    /// Base offset of the 512-bit message region: `left ‖ right`.
    pub msg_base: usize,
    /// Base offset of the 32-bit flags word — the one input whose PIN value
    /// varies across the chunk-leaf segment (see
    /// [`MerkleTreeLayout::with_blake3_chunk_leaf`]).
    pub flags_base: usize,
    /// Domain flags passed to every node compression.
    pub flags: u32,
    /// The base encoder's sparse `(A_0, B_0)`.
    pub build_matrices: fn() -> (SparseBinaryMatrix, SparseBinaryMatrix),
    /// One compression's output chaining value, and NOTHING else.
    ///
    /// The witness builders below all reach the digest by materializing a
    /// `2^k_log`-bool block and reading 256 bits back out — right for witness
    /// generation, absurd when only the digest is wanted. A circuit gate
    /// computing a root does `leaf_blocks + depth` compressions per opening
    /// and needs no witness at all, so it uses this. Pinned equal to the
    /// witness path by `fast_root_matches_the_witness_fold`.
    pub compress: fn(&[u32; SLOT_WORDS], &[u32; 16], u64, u32, u32) -> [u32; SLOT_WORDS],
    /// One node compression's boolean witness block (length `2^k_log`),
    /// given the two child digests.
    pub node_witness: fn(&[u32; SLOT_WORDS], &[u32; SLOT_WORDS]) -> Vec<bool>,
    /// One node compression's ROW-witness `(z, a, b)` into three
    /// `2^k_log / 64`-word buffers, zero on entry.
    ///
    /// The composite copies these verbatim — the base encoder's row kinds
    /// already match every row the composite overrides, *provided* pin-to-zero
    /// uses `A = [], B = [const]` rather than an empty `B` (see
    /// `MerkleTreeLayout::extras`, override 3). That is why `a`/`b` never need
    /// a matrix application anywhere on this path.
    pub node_witness_ab:
        fn(&[u32; SLOT_WORDS], &[u32; SLOT_WORDS], &mut [u64], &mut [u64], &mut [u64]),
    /// **The batch-major fast path**: `BM_V = 8` node compressions at once,
    /// written lane-interleaved into one level's row window.
    ///
    /// `rows[w][j]` is `u64` word `w` of lane `j`'s block, matching
    /// `common::BmRow` (spelled out here because `BmRow`/`BM_V` are
    /// crate-internal and this struct is public). Rows are zero on entry and
    /// the writers OR into them, exactly as the per-hash batch-major group
    /// builders do — so this is the same primitive BLAKE3's own driver uses,
    /// which is the point: the composite gets the base encoder's lane-parallel
    /// witness generation for free instead of unpacking to `bool` and back.
    ///
    /// The pairs are `(left, right)` per lane, already swapped.
    #[allow(clippy::type_complexity)]
    pub node_group_ab: fn(
        &[([u32; SLOT_WORDS], [u32; SLOT_WORDS]); 8],
        &mut [[u64; 8]],
        &mut [[u64; 8]],
        &mut [[u64; 8]],
    ),
    /// Base-block columns the composite pins to constants, as
    /// `(column, value)`: every free input of the base encoder EXCEPT the
    /// message region, which the swap gadget drives.
    pub fixed_bits: fn() -> Vec<(usize, bool)>,
    /// **Raw** base-block builders, for compressions that are not tree
    /// nodes — the chunk-leaf segment feeds these arbitrary
    /// `(h_in, msg, counter, block_len, flags)`. Same output contract as the
    /// node builders above.
    pub raw_witness: fn(&[u32; SLOT_WORDS], &[u32; 16], u64, u32, u32) -> Vec<bool>,
    #[allow(clippy::type_complexity)]
    pub raw_witness_ab:
        fn(&[u32; SLOT_WORDS], &[u32; 16], u64, u32, u32, &mut [u64], &mut [u64], &mut [u64]),
    /// Raw counterpart of [`Self::node_group_ab`]: 8 arbitrary compressions
    /// at once into one block window.
    #[allow(clippy::type_complexity)]
    pub raw_group_ab: fn(
        &[([u32; SLOT_WORDS], [u32; 16], u64, u32, u32); 8],
        &mut [[u64; 8]],
        &mut [[u64; 8]],
        &mut [[u64; 8]],
    ),
}

/// BLAKE3 backend (the default). One level = one
/// `compress(IV, left‖right, 0, 64, PARENT)`.
pub fn blake3_spec() -> HashSpec {
    HashSpec {
        name: "blake3",
        k_log: blake3::K_LOG,
        useful_bits: blake3::USEFUL_BITS,
        z_const_pos: blake3::Z_CONST_POS,
        in_cv_base: blake3::CV_BASE,
        out_cv_base: blake3::OUT_LO_BASE,
        msg_base: blake3::M_BASE,
        flags_base: blake3::FLAGS_BASE,
        flags: BLAKE3_FLAG_PARENT,
        build_matrices: blake3::build_matrices,
        compress: blake3_compress_cv,
        node_witness: blake3_node_witness,
        node_witness_ab: blake3_node_witness_ab,
        node_group_ab: blake3_node_group_ab,
        fixed_bits: blake3_fixed_bits,
        raw_witness: blake3::build_block_witness,
        raw_witness_ab: blake3::build_block_witness_ab_packed_into,
        raw_group_ab: blake3_raw_group_ab,
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
    let out = blake3::blake3_compress(cv, m, counter, block_len, flags);
    out[..SLOT_WORDS].try_into().expect("SLOT_WORDS ≤ 16")
}

fn blake3_node_witness(left: &[u32; SLOT_WORDS], right: &[u32; SLOT_WORDS]) -> Vec<bool> {
    blake3::build_block_witness(
        &blake3::BLAKE3_IV,
        &node_msg(left, right),
        NODE_COUNTER,
        NODE_BLOCK_LEN,
        BLAKE3_FLAG_PARENT,
    )
}

fn blake3_node_witness_ab(
    left: &[u32; SLOT_WORDS],
    right: &[u32; SLOT_WORDS],
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    blake3::build_block_witness_ab_packed_into(
        &blake3::BLAKE3_IV,
        &node_msg(left, right),
        NODE_COUNTER,
        NODE_BLOCK_LEN,
        BLAKE3_FLAG_PARENT,
        z,
        a,
        b,
    );
}

/// Eight node compressions at once, straight into a level's row window —
/// BLAKE3's own batch-major group builder, fed the Merkle node constants.
fn blake3_node_group_ab(
    pairs: &[([u32; SLOT_WORDS], [u32; SLOT_WORDS]); 8],
    rz: &mut [[u64; 8]],
    ra: &mut [[u64; 8]],
    rb: &mut [[u64; 8]],
) {
    let blocks: [blake3::Compression; 8] = from_fn(|j| {
        (
            blake3::BLAKE3_IV,
            node_msg(&pairs[j].0, &pairs[j].1),
            NODE_COUNTER,
            NODE_BLOCK_LEN,
            BLAKE3_FLAG_PARENT,
        )
    });
    let refs: [&blake3::Compression; 8] = from_fn(|j| &blocks[j]);
    blake3::build_group_batch_major(refs, rz, ra, rb);
}

/// Eight arbitrary compressions at once — the chunk-leaf segment's group
/// builder ([`HashSpec::raw_group_ab`]).
fn blake3_raw_group_ab(
    blocks: &[blake3::Compression; 8],
    rz: &mut [[u64; 8]],
    ra: &mut [[u64; 8]],
    rb: &mut [[u64; 8]],
) {
    let refs: [&blake3::Compression; 8] = from_fn(|j| &blocks[j]);
    blake3::build_group_batch_major(refs, rz, ra, rb);
}

/// BLAKE3's free inputs other than the message: `cv = IV`, `counter = 0`,
/// `block_len = 64`, `flags = PARENT`.
fn blake3_fixed_bits() -> Vec<(usize, bool)> {
    let w = blake3::WORD_BITS;
    let mut out = Vec::with_capacity(SLOT_BITS + 4 * w);
    for (word, iv) in blake3::BLAKE3_IV.iter().enumerate() {
        for b in 0..w {
            out.push((blake3::CV_BASE + word * w + b, (iv >> b) & 1 == 1));
        }
    }
    for (base, val) in [
        (blake3::T_LO_BASE, NODE_COUNTER as u32),
        (blake3::T_HI_BASE, (NODE_COUNTER >> 32) as u32),
        (blake3::BLEN_BASE, NODE_BLOCK_LEN),
        (blake3::FLAGS_BASE, BLAKE3_FLAG_PARENT),
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
/// is load-bearing for [`MerkleWalkerCircuit`]: it makes the level index a
/// set of high address bits, so the lincheck's `eq` table factors across
/// levels. See that type's docs.
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
    /// Lay out a `depth`-level Merkle path over `spec`.
    pub fn new(depth: usize, spec: HashSpec) -> Self {
        assert!(depth >= 1, "depth must be ≥ 1");
        // Levels tile aligned 2^κ subcubes, so the composite needs one
        // subcube per level rounded up to a power of two.
        let k_log = spec.k_log + depth.next_power_of_two().trailing_zeros() as usize;
        assert!(
            spec.k_log >= 7,
            "the union's BatchMajor chunking requires k_log ≥ 7"
        );
        // Per-level gadget columns and (in level 0) the globals live in the
        // base block's own padding region.
        let globals_end = spec.useful_bits + 2 * SLOT_BITS + 1 + SLOT_BITS + depth;
        assert!(
            globals_end <= 1usize << spec.k_log,
            "depth {depth} does not fit: the gadget ({} bits) plus the globals \
             ({} bits) exceed the base block's {} padding columns",
            2 * SLOT_BITS,
            1 + SLOT_BITS + depth,
            (1usize << spec.k_log) - spec.useful_bits,
        );
        // Last nonzero column: level 0 carries the globals; every later
        // level ends after its `t` region.
        let last_level_end = ((depth - 1) << spec.k_log) + spec.useful_bits + 2 * SLOT_BITS;
        let useful_bits = globals_end.max(last_level_end);
        debug_assert!(useful_bits <= 1usize << k_log);
        Self {
            spec,
            depth,
            leaf_blocks: 0,
            k_log,
            useful_bits,
        }
    }

    /// Lay out a **chunk-leaf** path: `leaf_bytes` of leaf data hashed as a
    /// single BLAKE3 chunk (a chain of `leaf_bytes/64` compressions,
    /// `CHUNK_START` on the first block and `CHUNK_END` on the last, chaining
    /// through `h_in`), followed by `depth` PARENT-node levels with the swap
    /// gadget. This is exactly `flock_core::merkle`'s BLAKE3 mode — leaf =
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

    /// Bit `j` of the leaf digest. Digest-leaf layouts only — the chunk-leaf
    /// path has no leaf digest global (the leaf enters as data, see
    /// [`Self::leaf_data_bit`]).
    pub fn leaf_bit(&self, j: usize) -> usize {
        debug_assert!(j < SLOT_BITS);
        debug_assert!(self.leaf_blocks == 0, "chunk-leaf has no leaf digest");
        self.const_pos() + 1 + j
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

    /// Indicator bit of level `l` (bit `l` of the index).
    pub fn index_bit(&self, level: usize) -> usize {
        debug_assert!(level < self.depth);
        if self.leaf_blocks == 0 {
            self.const_pos() + 1 + SLOT_BITS + level
        } else {
            self.index_word_base() + level
        }
    }

    /// First column of level `l`'s aligned subcube (node levels sit after
    /// the chunk-leaf segment).
    fn level_base(&self, level: usize) -> usize {
        debug_assert!(level < self.depth);
        self.block_base(self.leaf_blocks + level)
    }

    /// Bit `j` of chunk block `i`'s output chaining value.
    fn chunk_out_cv_bit(&self, block: usize, j: usize) -> usize {
        debug_assert!(block < self.leaf_blocks);
        debug_assert!(j < SLOT_BITS);
        self.block_base(block) + self.spec.out_cv_base + j
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

    /// Schema position of word `w` (`< 4`) of chunk block `block`'s leaf data.
    pub fn io_leaf(&self, block: usize, w: usize) -> usize {
        debug_assert!(block < self.leaf_blocks && w < 4);
        4 * block + w
    }

    /// Schema position of the index word.
    pub fn io_index(&self) -> usize {
        4 * self.leaf_blocks
    }

    /// Schema position of root half `h` (`0` = bits `[0,128)`).
    pub fn io_root(&self, h: usize) -> usize {
        debug_assert!(h < 2);
        4 * self.leaf_blocks + 1 + h
    }

    /// The wireable words of one opening — chunk-leaf layouts only.
    ///
    /// Inputs: the leaf data (`4` words per chunk block, in block order) then
    /// the index word. Outputs: the two halves of the root.
    ///
    /// **What is deliberately absent is the sibling path.** Each level's
    /// sibling digest sits at [`sibling_bit`](Self::sibling_bit) — inside that
    /// level's base-block padding, at `useful_bits`, which is not word-aligned
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
        use flock_core::schedule::IoWord;
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

    /// Bit `j` of level `l`'s sibling digest — in the base block's padding.
    pub fn sibling_bit(&self, level: usize, j: usize) -> usize {
        debug_assert!(j < SLOT_BITS);
        self.level_base(level) + self.spec.useful_bits + j
    }

    /// Bit `j` of level `l`'s AND column `t = b_l · (prev ⊕ sibling)`.
    fn t_bit(&self, level: usize, j: usize) -> usize {
        debug_assert!(j < SLOT_BITS);
        self.level_base(level) + self.spec.useful_bits + SLOT_BITS + j
    }

    /// Base-block column `c` of level `l`'s embedded hash block. The
    /// embedding is now a pure shift by `l·2^κ`.
    pub fn hash_bit(&self, level: usize, c: usize) -> usize {
        debug_assert!(c < 1usize << self.spec.k_log);
        self.level_base(level) + c
    }

    /// Bit `j` of the digest entering level `l`: at level 0 the leaf —
    /// the digest global, or on the chunk-leaf path the last chunk block's
    /// output chaining value — else the previous level's output.
    pub fn prev_bit(&self, level: usize, j: usize) -> usize {
        if level == 0 {
            if self.leaf_blocks == 0 {
                self.leaf_bit(j)
            } else {
                self.chunk_out_cv_bit(self.leaf_blocks - 1, j)
            }
        } else {
            self.hash_bit(level - 1, self.spec.out_cv_base + j)
        }
    }

    /// Bit `j` of the root — the last level's output chaining value.
    pub fn root_bit(&self, j: usize) -> usize {
        self.hash_bit(self.depth - 1, self.spec.out_cv_base + j)
    }

    fn left_bit(&self, level: usize, j: usize) -> usize {
        self.hash_bit(level, self.spec.msg_base + j)
    }

    fn right_bit(&self, level: usize, j: usize) -> usize {
        self.hash_bit(level, self.spec.msg_base + SLOT_BITS + j)
    }

    // -----------------------------------------------------------------------
    // Matrices
    // -----------------------------------------------------------------------

    /// Which base-block ROWS the composite replaces, as a `spec.useful_bits`
    /// mask. Everything else is embedded verbatim at a column offset.
    ///
    /// Shared by [`Self::build_matrices`] and [`Self::build_walker`] so the
    /// materialized and walked forms cannot drift apart.
    fn base_overridden(&self) -> Vec<bool> {
        let mut ovr = vec![false; self.spec.useful_bits];
        ovr[self.spec.z_const_pos] = true;
        for j in 0..2 * SLOT_BITS {
            ovr[self.spec.msg_base + j] = true;
        }
        for &(c, _) in &(self.spec.fixed_bits)() {
            ovr[c] = true;
        }
        ovr
    }

    /// Every composite row that is NOT a shifted copy of a base-block row:
    /// the globals (constant, leaf, index), the per-level gadget columns
    /// (sibling, `t`), and the three override groups inside each hash block.
    ///
    /// These are the ONLY rows that reference columns outside their own
    /// level — which is exactly why the base embedding can be walked with a
    /// pure offset.
    fn extras(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let k = self.k();
        let gc = self.const_pos();
        let mut a: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut b: Vec<Vec<usize>> = vec![Vec::new(); k];

        // A free witness column: `z_c · 1 = z_c`, satisfied by any bit.
        let free = |a: &mut Vec<Vec<usize>>, b: &mut Vec<Vec<usize>>, r: usize| {
            a[r] = vec![r];
            b[r] = vec![gc];
        };

        // The global constant: `z_0 · z_0 = z_0`. Boolean idempotence leaves
        // 0 admissible here; `const_pin` is what forces it to 1 (see
        // `docs/const-wire-pin.md`).
        a[gc] = vec![gc];
        b[gc] = vec![gc];

        if self.leaf_blocks == 0 {
            for j in 0..SLOT_BITS {
                free(&mut a, &mut b, self.leaf_bit(j));
            }
        }
        if self.leaf_blocks == 0 {
            for l in 0..self.depth {
                free(&mut a, &mut b, self.index_bit(l));
            }
        } else {
            // The whole index WORD is free, not just its `depth` index bits.
            // A zero-pinned remainder would make the committed word equal the
            // bare index, and wiring that to a Fiat–Shamir challenge would
            // demand the challenge's high bits be zero — unsatisfiable. The
            // relation still reads only the low `depth`.
            for j in 0..128 {
                free(&mut a, &mut b, self.index_word_base() + j);
            }
        }

        let fixed = (self.spec.fixed_bits)();

        // The chunk-leaf segment. Same three override groups as a node level,
        // with the message left FREE (it is the leaf data) in place of the
        // swap gadget, and the pins adjusted per block: the flags word takes
        // the block's chunk flags, and past block 0 the input chaining value
        // becomes a copy of the previous block's output — the chunk chain.
        for i in 0..self.leaf_blocks {
            let base = self.block_base(i);

            // Override 1: the block's constant wire, re-derived.
            let r = base + self.spec.z_const_pos;
            a[r] = vec![gc];
            b[r] = vec![gc];

            // Override 2: the message region is the leaf data — free inputs.
            for j in 0..2 * SLOT_BITS {
                free(&mut a, &mut b, base + self.spec.msg_base + j);
            }

            // Override 3: the pins, from the node fixed set with the two
            // chunk-specific substitutions.
            let flags = self.chunk_flags(i);
            let fb = self.spec.flags_base;
            let cvb = self.spec.in_cv_base;
            for &(c, v) in &fixed {
                let r = base + c;
                if i > 0 && c >= cvb && c < cvb + SLOT_BITS {
                    a[r] = vec![self.chunk_out_cv_bit(i - 1, c - cvb)];
                    b[r] = vec![gc];
                    continue;
                }
                let v = if c >= fb && c < fb + 32 {
                    (flags >> (c - fb)) & 1 == 1
                } else {
                    v
                };
                a[r] = if v { vec![gc] } else { Vec::new() };
                b[r] = vec![gc];
            }
        }
        for l in 0..self.depth {
            for j in 0..SLOT_BITS {
                free(&mut a, &mut b, self.sibling_bit(l, j));
            }

            // t_j = (1 + b_l) · (prev_j ⊕ sibling_j) — the only AND per bit.
            //
            // The complement, not `b_l` itself, so that `b_l` means what the
            // TREE means by it: `flock_core::merkle` puts the running digest
            // left at an EVEN node index, i.e. when bit `l` of the position is
            // 0. With `left = sibling ⊕ t` and `right = prev ⊕ t` that needs
            // `t = (1+b)·(prev ⊕ sib)`, and then the table's index IS the tree
            // position rather than its complement.
            //
            // That equality is what lets a Fiat–Shamir challenge word be wired
            // straight into the index (see `index_word_base`): `sample_queries`
            // masks the challenge to get a position, so the circuit must open
            // THAT position, not its complement.
            //
            // The `1` costs nothing — it is the global constant column, which
            // every row here already references.
            for j in 0..SLOT_BITS {
                let r = self.t_bit(l, j);
                a[r] = vec![gc, self.index_bit(l)];
                b[r] = vec![self.prev_bit(l, j), self.sibling_bit(l, j)];
            }

            // Override 1: the block's own constant wire, re-derived from the
            // global one so a single column carries the table's const_pin.
            let r = self.hash_bit(l, self.spec.z_const_pos);
            a[r] = vec![gc];
            b[r] = vec![gc];

            // Override 2: the message region IS the swap gadget's output.
            for j in 0..SLOT_BITS {
                let rl = self.left_bit(l, j);
                a[rl] = vec![self.sibling_bit(l, j), self.t_bit(l, j)];
                b[rl] = vec![gc];
                let rr = self.right_bit(l, j);
                a[rr] = vec![self.prev_bit(l, j), self.t_bit(l, j)];
                b[rr] = vec![gc];
            }

            // Override 3: pin the remaining free inputs to the node
            // constants. Value 1 → `1·1 = 1`; value 0 → `0·1 = 0` (empty A,
            // const on B — NOT an empty B, so that the row-witness `b` bit
            // stays 1 and matches what the base encoder emits for the
            // free-input row this replaces).
            for &(c, v) in &fixed {
                let r = self.hash_bit(l, c);
                a[r] = if v { vec![gc] } else { Vec::new() };
                b[r] = vec![gc];
            }
        }
        (a, b)
    }

    /// Build the composite `(A_0, B_0)` in full.
    ///
    /// **Memory warning.** This materializes `depth` copies of the base
    /// encoder's matrices. BLAKE3's block carries ~21M nonzeros (its
    /// encoding trades density for a small `k` — see the `blake3` module
    /// docs), so depth 26 is ~547M nonzeros ≈ 4.4 GB. Use this at small
    /// depth as the reference oracle; at real depth use
    /// [`Self::build_walker`], which stores one base copy.
    pub fn build_matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = self.k();
        let (mut a, mut b) = self.extras();
        let (base_a, base_b) = (self.spec.build_matrices)();
        let ovr = self.base_overridden();
        debug_assert_eq!(base_a.num_rows, 1usize << self.spec.k_log);

        for t in 0..self.total_blocks() {
            // The hash block, embedded by a pure column offset — chunk
            // blocks and node levels alike. Overridden rows are already
            // written by `extras`.
            let base = self.block_base(t);
            for c in 0..self.spec.useful_bits {
                if ovr[c] {
                    continue;
                }
                let r = base + c;
                a[r] = base_a.rows[c].iter().map(|&x| base + x).collect();
                b[r] = base_b.rows[c].iter().map(|&x| base + x).collect();
            }
        }

        // Rows [useful_bits, k) stay empty: `0 · 0 = z_r` forces the padding
        // columns to zero, which is what the union's run-list gating and the
        // dense-stack compaction rely on.
        debug_assert!(
            a.iter()
                .chain(b.iter())
                .flatten()
                .all(|&c| c < self.useful_bits),
            "a matrix row references a padding column"
        );
        (
            SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: a,
            },
            SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: b,
            },
        )
    }

    /// The composite [`BlockR1cs`] over `2^n_paths_log` rows (paths), with
    /// materialized matrices.
    ///
    /// Carries the same memory warning as [`Self::build_matrices`]. For the
    /// walker path use [`Self::build_block_r1cs_stub`].
    pub fn build_block_r1cs(&self, n_paths_log: usize) -> BlockR1cs {
        let (a_0, b_0) = self.build_matrices();
        self.block_r1cs_with(n_paths_log, a_0, b_0)
    }

    /// [`BlockR1cs`] with **empty `(A_0, B_0)` stubs** — the walker path.
    /// The constraints live in [`Self::build_walker`]'s
    /// [`LincheckCircuit`], following the same convention as the Keccak
    /// encoders (`r1cs_hashes::common`).
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

    /// Build the walker [`LincheckCircuit`]: ONE copy of the base block's
    /// CSC transpose, walked once per level at a column offset, plus the
    /// composite's small extras set. See [`MerkleWalkerCircuit`].
    pub fn build_walker(&self) -> MerkleWalkerCircuit {
        let (base_a, base_b) = (self.spec.build_matrices)();
        let ovr = self.base_overridden();

        // Empty the overridden rows before transposing: their base entries
        // do not apply to the composite, and their replacements are in
        // `extras`.
        let strip = |m: &SparseBinaryMatrix| -> SparseBinaryMatrix {
            let mut rows = m.rows.clone();
            for (c, o) in ovr.iter().enumerate() {
                if *o {
                    rows[c].clear();
                }
            }
            SparseBinaryMatrix {
                num_rows: m.num_rows,
                num_cols: m.num_cols,
                rows,
            }
        };
        let base = Csc::pair(&strip(&base_a), &strip(&base_b));
        drop((base_a, base_b));

        let k = self.k();
        let (xa, xb) = self.extras();
        let extras = Csc::pair(
            &SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: xa,
            },
            &SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: xb,
            },
        );

        MerkleWalkerCircuit {
            n_cols: k,
            // The walker walks SUBCUBES; the chunk-leaf blocks embed the
            // same stripped base as the node levels, so it need not tell
            // them apart.
            depth: self.total_blocks(),
            base_k_log: self.spec.k_log,
            base,
            extras,
            const_pin: Some(self.const_pos()),
        }
    }

    // -----------------------------------------------------------------------
    // Witness
    // -----------------------------------------------------------------------

    /// Boolean witness for ONE path (length `2^k_log`).
    ///
    /// Panics if `input.siblings.len() != depth` or the index does not fit
    /// in `depth` bits.
    pub fn build_witness(&self, input: &PathInput) -> Vec<bool> {
        assert_eq!(
            self.leaf_blocks, 0,
            "chunk-leaf layout: use the _chunk builders"
        );
        assert_eq!(
            input.siblings.len(),
            self.depth,
            "need one sibling digest per level"
        );
        assert!(
            self.depth >= 64 || input.index < (1u64 << self.depth),
            "index {} does not fit in {} bits",
            input.index,
            self.depth
        );
        let mut z = vec![false; self.k()];
        z[self.const_pos()] = true;
        write_digest(&mut z, self.leaf_bit(0), &input.leaf);
        for l in 0..self.depth {
            z[self.index_bit(l)] = (input.index >> l) & 1 == 1;
        }

        let mut prev = input.leaf;
        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            let sib = input.siblings[l];
            write_digest(&mut z, self.sibling_bit(l, 0), &sib);
            for j in 0..SLOT_BITS {
                z[self.t_bit(l, j)] = !bit && (digest_bit(&prev, j) ^ digest_bit(&sib, j));
            }
            // b_l = 0 puts the running digest on the left — the TREE's convention.
            let (left, right) = if bit { (sib, prev) } else { (prev, sib) };
            let block = (self.spec.node_witness)(&left, &right);
            let dst = self.hash_bit(l, 0);
            z[dst..dst + self.spec.useful_bits].copy_from_slice(&block[..self.spec.useful_bits]);
            prev = read_digest(&block, self.spec.out_cv_base);
        }
        z
    }

    /// Read the root digest back out of a witness built by
    /// [`Self::build_witness`].
    pub fn read_root(&self, z: &[bool]) -> [u32; SLOT_WORDS] {
        read_digest(z, self.root_bit(0))
    }

    /// Full ROW-witness `(z, a, b)` for ONE path, each of length `2^k_log`.
    ///
    /// `a = A_0·z` and `b = B_0·z` are emitted directly, never by matrix
    /// application: the base encoder's row-witness is copied in at the level
    /// offset (its row kinds already match the composite's overrides) and the
    /// gadget rows are written from the values at hand.
    pub fn build_witness_zab(&self, input: &PathInput) -> [Vec<bool>; 3] {
        assert_eq!(
            self.leaf_blocks, 0,
            "chunk-leaf layout: use the _chunk builders"
        );
        assert_eq!(
            input.siblings.len(),
            self.depth,
            "need one sibling digest per level"
        );
        let k = self.k();
        let mut z = vec![false; k];
        let mut a = vec![false; k];
        let mut b = vec![false; k];

        // A free column: `z_c · 1 = z_c`.
        let free = |z: &mut [bool], a: &mut [bool], b: &mut [bool], c: usize, v: bool| {
            z[c] = v;
            a[c] = v;
            b[c] = true;
        };

        // The global constant: 1·1 = 1.
        let gc = self.const_pos();
        z[gc] = true;
        a[gc] = true;
        b[gc] = true;

        for j in 0..SLOT_BITS {
            free(
                &mut z,
                &mut a,
                &mut b,
                self.leaf_bit(j),
                digest_bit(&input.leaf, j),
            );
        }
        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            free(&mut z, &mut a, &mut b, self.index_bit(l), bit);
        }

        let base_words = (1usize << self.spec.k_log) / 64;
        let mut wz = vec![0u64; base_words];
        let mut wa = vec![0u64; base_words];
        let mut wb = vec![0u64; base_words];

        let mut prev = input.leaf;
        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            let sib = input.siblings[l];
            for j in 0..SLOT_BITS {
                free(
                    &mut z,
                    &mut a,
                    &mut b,
                    self.sibling_bit(l, j),
                    digest_bit(&sib, j),
                );
            }
            // t_j = (1 + b_l) · (prev_j ⊕ sibling_j): the AND row's operands
            // are the COMPLEMENTED index bit and the XOR of the two
            // candidate children. See `extras` for why the complement.
            for j in 0..SLOT_BITS {
                let xor = digest_bit(&prev, j) ^ digest_bit(&sib, j);
                let c = self.t_bit(l, j);
                z[c] = !bit && xor;
                a[c] = !bit;
                b[c] = xor;
            }

            // b_l = 0 puts the running digest on the left — the TREE's convention.
            let (left, right) = if bit { (sib, prev) } else { (prev, sib) };
            wz.fill(0);
            wa.fill(0);
            wb.fill(0);
            (self.spec.node_witness_ab)(&left, &right, &mut wz, &mut wa, &mut wb);
            let dst = self.hash_bit(l, 0);
            for c in 0..self.spec.useful_bits {
                z[dst + c] = word_bit(&wz, c);
                a[dst + c] = word_bit(&wa, c);
                b[dst + c] = word_bit(&wb, c);
            }
            prev = read_digest_words(&wz, self.spec.out_cv_base);
        }
        [z, a, b]
    }

    /// One union slot's packed witness: `2^nu` rows (paths) in **BatchMajor**
    /// address order, plus the lincheck stripe.
    ///
    /// `paths.len()` may be less than `2^nu`; the trailing dummy rows are
    /// identically zero in all four outputs — **including the const-pin
    /// bit** — as the union's count-derived lincheck target requires
    /// (`flock_core::lincheck::union`). Do NOT substitute a "path over zero
    /// inputs": that would set the pin and be rejected.
    pub fn generate_witness_batch_major_partial(
        &self,
        paths: &[PathInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let k = self.k();
        let total = (1usize << nu) * (k / 128);
        let mut z = vec![F128::ZERO; total];
        let mut a = vec![F128::ZERO; total];
        let mut b = vec![F128::ZERO; total];
        let stripe = self.generate_witness_batch_major_partial_into(
            paths,
            nu,
            flock_core::union::SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
            },
        );
        (z, a, b, stripe)
    }

    /// [`Self::generate_witness_batch_major_partial`] writing into a union
    /// slot's destination buffers — the copy-free assembly path, and the
    /// implementation both variants share.
    ///
    /// Runs on the same `common::drive_witness_batch_major_partial_into`
    /// driver as the per-hash encoders, with the same `BM_V = 8` lane-parallel
    /// group builder underneath (see [`HashSpec::node_group_ab`]). The
    /// composite is expressible this way *because levels are `2^κ`-aligned*:
    /// level `l`'s subcube is exactly the base block at u64-row offset
    /// `l · 2^κ/64`, so the base encoder's packed output drops in with no bit
    /// shifting, and the only columns this function writes itself are the swap
    /// gadget (sibling, `t`) and level 0's globals.
    pub fn generate_witness_batch_major_partial_into(
        &self,
        paths: &[PathInput],
        nu: usize,
        dst: SlotWitnessDest<'_>,
    ) -> Vec<u8> {
        use super::common::{BM_V, BmRow, or_bit_row, or_u32_row};

        const _: () = assert!(BM_V == 8, "HashSpec::node_group_ab hardcodes 8 lanes");
        assert_eq!(
            self.leaf_blocks, 0,
            "chunk-leaf layout: use the _chunk builders"
        );
        assert!(
            paths.len() <= 1usize << nu,
            "{} paths > 2^{nu} rows",
            paths.len()
        );
        for p in paths {
            assert_eq!(
                p.siblings.len(),
                self.depth,
                "need one sibling digest per level"
            );
        }
        let spec = &self.spec;
        // A level window is a whole number of u64 rows — the alignment that
        // makes this driver usable at all.
        assert!(
            spec.k_log >= 6,
            "a level stride of 2^{} bits is not a u64 multiple",
            spec.k_log
        );
        let words_per_level = 1usize << (spec.k_log - 6);
        // The output chaining value must be u64-aligned so the next level's
        // input can be read straight back out of the packed rows.
        assert_eq!(
            spec.out_cv_base % 64,
            0,
            "out_cv_base must be u64-aligned to chain levels in packed form"
        );
        let out_word = spec.out_cv_base / 64;
        let depth = self.depth;

        // Window-relative offsets. Level-independent, since every level's
        // subcube is the base block (see `sibling_bit` / `t_bit`).
        let sib_off = spec.useful_bits;
        let t_off = spec.useful_bits + SLOT_BITS;
        let const_off = self.const_pos();
        let leaf_off = const_off + 1;
        let index_off = const_off + 1 + SLOT_BITS;

        super::common::drive_witness_batch_major_partial_into(
            paths,
            nu,
            self.k_log,
            self.useful_bits,
            dst,
            move |group, rz, ra, rb| {
                // Per-lane running digest: the leaf, then each level's output.
                let mut prev: [[u32; SLOT_WORDS]; BM_V] = from_fn(|j| group[j].leaf);

                for l in 0..depth {
                    let w0 = l * words_per_level;
                    let win = w0..w0 + words_per_level;
                    let (wz, wa, wb): (&mut [BmRow], &mut [BmRow], &mut [BmRow]) =
                        (&mut rz[win.clone()], &mut ra[win.clone()], &mut rb[win]);

                    // The conditional swap, per lane. `mask` is the index bit
                    // broadcast over a word — the AND row's `a` operand.
                    let mut pairs = [([0u32; SLOT_WORDS], [0u32; SLOT_WORDS]); BM_V];
                    let mut mask = [0u32; BM_V];
                    for j in 0..BM_V {
                        let bit = (group[j].index >> l) & 1 == 1;
                        mask[j] = if bit { 0 } else { !0u32 };
                        let sib = group[j].siblings[l];
                        pairs[j] = if bit { (sib, prev[j]) } else { (prev[j], sib) };
                    }

                    // The hash block itself, verbatim from the base encoder —
                    // including the rows the composite overrides, whose row
                    // kinds already agree (see `HashSpec::node_witness_ab`).
                    (spec.node_group_ab)(&pairs, wz, wa, wb);

                    for w in 0..SLOT_WORDS {
                        // Sibling: a free column (z = a = S, b = 1).
                        let sib: [u32; BM_V] = from_fn(|j| group[j].siblings[l][w]);
                        or_u32_row(wz, sib_off + 32 * w, &sib);
                        or_u32_row(wa, sib_off + 32 * w, &sib);
                        or_u32_row(wb, sib_off + 32 * w, &[!0u32; BM_V]);

                        // t = b_l · (prev ⊕ S): A = [b_l], B = [prev, S].
                        let xor: [u32; BM_V] = from_fn(|j| prev[j][w] ^ sib[j]);
                        let t: [u32; BM_V] = from_fn(|j| mask[j] & xor[j]);
                        or_u32_row(wz, t_off + 32 * w, &t);
                        or_u32_row(wa, t_off + 32 * w, &mask);
                        or_u32_row(wb, t_off + 32 * w, &xor);
                    }

                    if l == 0 {
                        // The globals live in level 0's padding region.
                        or_bit_row(wz, const_off);
                        or_bit_row(wa, const_off);
                        or_bit_row(wb, const_off);
                        for w in 0..SLOT_WORDS {
                            let leaf: [u32; BM_V] = from_fn(|j| group[j].leaf[w]);
                            or_u32_row(wz, leaf_off + 32 * w, &leaf);
                            or_u32_row(wa, leaf_off + 32 * w, &leaf);
                            or_u32_row(wb, leaf_off + 32 * w, &[!0u32; BM_V]);
                        }
                        // One index bit per level. Written one at a time
                        // rather than 32 at a stride: bits at or above `depth`
                        // are interior padding that empty rows force to zero,
                        // so a wider store would break the R1CS.
                        for i in 0..depth {
                            let v: [u32; BM_V] = from_fn(|j| ((group[j].index >> i) & 1) as u32);
                            or_u32_row(wz, index_off + i, &v);
                            or_u32_row(wa, index_off + i, &v);
                            or_bit_row(wb, index_off + i);
                        }
                    }

                    // Chain: read this level's output CV back out of the
                    // packed rows (u64-aligned, asserted above).
                    for j in 0..BM_V {
                        for w in 0..SLOT_WORDS / 2 {
                            let word = wz[out_word + w][j];
                            prev[j][2 * w] = word as u32;
                            prev[j][2 * w + 1] = (word >> 32) as u32;
                        }
                    }
                }
            },
        )
    }

    /// The original per-path `Vec<bool>` builder, retained as the reference
    /// oracle for [`Self::generate_witness_batch_major_partial_into`] — the
    /// same role `build_matrices` plays for the walker. Straightforward and
    /// slow: it unpacks the base encoder's packed rows bit-by-bit and then
    /// repacks them, allocating 3 × `2^k_log` bytes of `Vec<bool>` per path.
    /// Not `#[cfg(test)]` because the equality test is an integration test.
    pub fn generate_witness_batch_major_partial_bool(
        &self,
        paths: &[PathInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;

        let n_total = 1usize << nu;
        assert!(
            paths.len() <= n_total,
            "{} paths > 2^{nu} = {n_total} rows",
            paths.len()
        );

        // Per-path row-witnesses first (embarrassingly parallel), then the
        // BatchMajor scatter + stripe transpose.
        let per_path: Vec<[Vec<bool>; 3]> = paths
            .par_iter()
            .map(|p| self.build_witness_zab(p))
            .collect();
        self.scatter_zab_batch_major(&per_path, nu)
    }

    /// The BatchMajor scatter + stripe transpose behind the `_bool` oracles,
    /// shared by both leaf flavors.
    fn scatter_zab_batch_major(
        &self,
        per_path: &[[Vec<bool>; 3]],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;

        let n_total = 1usize << nu;
        assert!(
            n_total.is_multiple_of(8),
            "the lincheck stripe needs 2^nu ≥ 8 (nu ≥ 3)"
        );
        let k = self.k();
        let words_per_block = k / 128;
        let total = n_total * words_per_block;

        let mut z = vec![F128::ZERO; total];
        let mut a = vec![F128::ZERO; total];
        let mut b = vec![F128::ZERO; total];
        for (i, [pz, pa, pb]) in per_path.iter().enumerate() {
            for w in 0..words_per_block {
                // BatchMajor: word `w` of row `i` lives at `(w << nu) + i`.
                let addr = (w << nu) + i;
                z[addr] = pack_word(pz, w * 128);
                a[addr] = pack_word(pa, w * 128);
                b[addr] = pack_word(pb, w * 128);
            }
        }

        // Stripe: `stripe[g*k + c]` bit `r` = z of row `8g + r` at column c.
        // Rows past `paths.len()` contribute 0, so dummy rows stay zero.
        let mut stripe = vec![0u8; (n_total / 8) * k];
        stripe.par_chunks_mut(k).enumerate().for_each(|(g, chunk)| {
            for r in 0..8 {
                let row = 8 * g + r;
                if row >= per_path.len() {
                    continue;
                }
                let pz = &per_path[row][0];
                for c in 0..self.useful_bits {
                    if pz[c] {
                        chunk[c] |= 1u8 << r;
                    }
                }
            }
        });

        (z, a, b, stripe)
    }

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

    /// Boolean witness for ONE chunk-leaf opening (length `2^k_log`).
    pub fn build_witness_chunk(&self, input: &ChunkPathInput) -> Vec<bool> {
        self.assert_chunk_input(input);
        let mut z = vec![false; self.k()];
        z[self.const_pos()] = true;
        // The whole index WORD, not just the `depth` bits the fold reads — the
        // columns above are free and carry the wired challenge's high bits.
        for j in 0..128 {
            z[self.index_word_base() + j] = (input.index >> j) & 1 == 1;
        }

        // The chunk chain: h_in of block 0 is the IV, then each block's
        // output chaining value feeds the next block's input.
        let mut prev = self.pinned_in_cv();
        for i in 0..self.leaf_blocks {
            let m = leaf_msg_words(&input.leaf_data, i);
            let block = (self.spec.raw_witness)(
                &prev,
                &m,
                NODE_COUNTER,
                NODE_BLOCK_LEN,
                self.chunk_flags(i),
            );
            let dst = self.block_base(i);
            z[dst..dst + self.spec.useful_bits].copy_from_slice(&block[..self.spec.useful_bits]);
            prev = read_digest(&block, self.spec.out_cv_base);
        }

        // The node levels — identical to the digest-leaf path, with `prev`
        // seeded by the chunk chain instead of a leaf digest global.
        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            let sib = input.siblings[l];
            write_digest(&mut z, self.sibling_bit(l, 0), &sib);
            for j in 0..SLOT_BITS {
                z[self.t_bit(l, j)] = !bit && (digest_bit(&prev, j) ^ digest_bit(&sib, j));
            }
            let (left, right) = if bit { (sib, prev) } else { (prev, sib) };
            let block = (self.spec.node_witness)(&left, &right);
            let dst = self.hash_bit(l, 0);
            z[dst..dst + self.spec.useful_bits].copy_from_slice(&block[..self.spec.useful_bits]);
            prev = read_digest(&block, self.spec.out_cv_base);
        }
        z
    }

    /// Full ROW-witness `(z, a, b)` for ONE chunk-leaf opening. Chunk blocks
    /// copy the base encoder's row-witness verbatim: its free-input rows for
    /// the message ARE the composite's leaf-data rows, and its free-input
    /// rows for the chaining value are witness-identical to the pin (block
    /// 0) and copy (later blocks) overrides, because the chained values
    /// agree by construction.
    pub fn build_witness_zab_chunk(&self, input: &ChunkPathInput) -> [Vec<bool>; 3] {
        self.assert_chunk_input(input);
        let k = self.k();
        let mut z = vec![false; k];
        let mut a = vec![false; k];
        let mut b = vec![false; k];

        let free = |z: &mut [bool], a: &mut [bool], b: &mut [bool], c: usize, v: bool| {
            z[c] = v;
            a[c] = v;
            b[c] = true;
        };

        let gc = self.const_pos();
        z[gc] = true;
        a[gc] = true;
        b[gc] = true;
        // The whole index word is free, so every one of its 128 columns needs
        // its `z·1 = z` witness — not just the `depth` index bits. The bits at
        // or above `depth` are read by no row; they carry whatever the wired
        // Fiat-Shamir challenge put there.
        for j in 0..128 {
            let bit = (input.index >> j) & 1 == 1;
            free(&mut z, &mut a, &mut b, self.index_word_base() + j, bit);
        }

        let base_words = (1usize << self.spec.k_log) / 64;
        let mut wz = vec![0u64; base_words];
        let mut wa = vec![0u64; base_words];
        let mut wb = vec![0u64; base_words];

        let mut prev = self.pinned_in_cv();
        for i in 0..self.leaf_blocks {
            let m = leaf_msg_words(&input.leaf_data, i);
            wz.fill(0);
            wa.fill(0);
            wb.fill(0);
            (self.spec.raw_witness_ab)(
                &prev,
                &m,
                NODE_COUNTER,
                NODE_BLOCK_LEN,
                self.chunk_flags(i),
                &mut wz,
                &mut wa,
                &mut wb,
            );
            let dst = self.block_base(i);
            for c in 0..self.spec.useful_bits {
                z[dst + c] = word_bit(&wz, c);
                a[dst + c] = word_bit(&wa, c);
                b[dst + c] = word_bit(&wb, c);
            }
            prev = read_digest_words(&wz, self.spec.out_cv_base);
        }

        for l in 0..self.depth {
            let bit = (input.index >> l) & 1 == 1;
            let sib = input.siblings[l];
            for j in 0..SLOT_BITS {
                free(
                    &mut z,
                    &mut a,
                    &mut b,
                    self.sibling_bit(l, j),
                    digest_bit(&sib, j),
                );
            }
            for j in 0..SLOT_BITS {
                let xor = digest_bit(&prev, j) ^ digest_bit(&sib, j);
                let c = self.t_bit(l, j);
                z[c] = !bit && xor;
                a[c] = !bit;
                b[c] = xor;
            }
            let (left, right) = if bit { (sib, prev) } else { (prev, sib) };
            wz.fill(0);
            wa.fill(0);
            wb.fill(0);
            (self.spec.node_witness_ab)(&left, &right, &mut wz, &mut wa, &mut wb);
            let dst = self.hash_bit(l, 0);
            for c in 0..self.spec.useful_bits {
                z[dst + c] = word_bit(&wz, c);
                a[dst + c] = word_bit(&wa, c);
                b[dst + c] = word_bit(&wb, c);
            }
            prev = read_digest_words(&wz, self.spec.out_cv_base);
        }
        [z, a, b]
    }

    /// Chunk-leaf counterpart of
    /// [`Self::generate_witness_batch_major_partial`].
    pub fn generate_witness_batch_major_partial_chunk(
        &self,
        paths: &[ChunkPathInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let k = self.k();
        let total = (1usize << nu) * (k / 128);
        let mut z = vec![F128::ZERO; total];
        let mut a = vec![F128::ZERO; total];
        let mut b = vec![F128::ZERO; total];
        let stripe = self.generate_witness_batch_major_partial_into_chunk(
            paths,
            nu,
            flock_core::union::SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
            },
        );
        (z, a, b, stripe)
    }

    /// Chunk-leaf counterpart of
    /// [`Self::generate_witness_batch_major_partial_into`], on the same
    /// driver: the chunk segment runs the base encoder's raw group builder
    /// per block window (no gadget columns at all — the message rows ARE the
    /// leaf data, and the chaining rows match the pin/copy overrides
    /// witness-for-witness), then the node levels run exactly as the
    /// digest-leaf path, shifted by `leaf_blocks` windows.
    pub fn generate_witness_batch_major_partial_into_chunk(
        &self,
        paths: &[ChunkPathInput],
        nu: usize,
        dst: SlotWitnessDest<'_>,
    ) -> Vec<u8> {
        use super::common::{BM_V, BmRow, or_bit_row, or_u32_row};

        const _: () = assert!(BM_V == 8, "HashSpec group builders hardcode 8 lanes");
        assert!(
            paths.len() <= 1usize << nu,
            "{} paths > 2^{nu} rows",
            paths.len()
        );
        for p in paths {
            self.assert_chunk_input(p);
        }
        let spec = &self.spec;
        assert!(
            spec.k_log >= 6,
            "a block stride of 2^{} bits is not a u64 multiple",
            spec.k_log
        );
        let words_per_level = 1usize << (spec.k_log - 6);
        assert_eq!(
            spec.out_cv_base % 64,
            0,
            "out_cv_base must be u64-aligned to chain blocks in packed form"
        );
        let out_word = spec.out_cv_base / 64;
        let depth = self.depth;
        let leaf_blocks = self.leaf_blocks;
        let iv = self.pinned_in_cv();

        let sib_off = spec.useful_bits;
        let t_off = spec.useful_bits + SLOT_BITS;
        let const_off = self.const_pos();
        // The index is a word-aligned 128-bit WORD, not a tight run after the
        // constant — see `index_word_base`.
        let index_off = self.index_word_base();

        super::common::drive_witness_batch_major_partial_into(
            paths,
            nu,
            self.k_log,
            self.useful_bits,
            dst,
            move |group, rz, ra, rb| {
                let mut prev: [[u32; SLOT_WORDS]; BM_V] = [iv; BM_V];

                // The chunk chain, one block window at a time.
                for i in 0..leaf_blocks {
                    let w0 = i * words_per_level;
                    let win = w0..w0 + words_per_level;
                    let (wz, wa, wb): (&mut [BmRow], &mut [BmRow], &mut [BmRow]) =
                        (&mut rz[win.clone()], &mut ra[win.clone()], &mut rb[win]);

                    let flags = {
                        let mut f = 0;
                        if i == 0 {
                            f |= BLAKE3_FLAG_CHUNK_START;
                        }
                        if i + 1 == leaf_blocks {
                            f |= BLAKE3_FLAG_CHUNK_END;
                        }
                        f
                    };
                    let blocks: [([u32; SLOT_WORDS], [u32; 16], u64, u32, u32); BM_V] =
                        from_fn(|j| {
                            (
                                prev[j],
                                leaf_msg_words(&group[j].leaf_data, i),
                                NODE_COUNTER,
                                NODE_BLOCK_LEN,
                                flags,
                            )
                        });
                    (spec.raw_group_ab)(&blocks, wz, wa, wb);

                    if i == 0 {
                        // The globals live in chunk block 0's padding region.
                        or_bit_row(wz, const_off);
                        or_bit_row(wa, const_off);
                        or_bit_row(wb, const_off);
                        // The whole index WORD is free, so all 128 columns
                        // need their `z·1 = z` witness. Columns at or above
                        // `depth` are read by no row; they carry whatever the
                        // wired Fiat-Shamir challenge put there.
                        for l in 0..128 {
                            let v: [u32; BM_V] = from_fn(|j| ((group[j].index >> l) & 1) as u32);
                            or_u32_row(wz, index_off + l, &v);
                            or_u32_row(wa, index_off + l, &v);
                            or_bit_row(wb, index_off + l);
                        }
                    }

                    for j in 0..BM_V {
                        for w in 0..SLOT_WORDS / 2 {
                            let word = wz[out_word + w][j];
                            prev[j][2 * w] = word as u32;
                            prev[j][2 * w + 1] = (word >> 32) as u32;
                        }
                    }
                }

                // The node levels, shifted by the chunk segment.
                for l in 0..depth {
                    let w0 = (leaf_blocks + l) * words_per_level;
                    let win = w0..w0 + words_per_level;
                    let (wz, wa, wb): (&mut [BmRow], &mut [BmRow], &mut [BmRow]) =
                        (&mut rz[win.clone()], &mut ra[win.clone()], &mut rb[win]);

                    let mut pairs = [([0u32; SLOT_WORDS], [0u32; SLOT_WORDS]); BM_V];
                    let mut mask = [0u32; BM_V];
                    for j in 0..BM_V {
                        let bit = (group[j].index >> l) & 1 == 1;
                        mask[j] = if bit { 0 } else { !0u32 };
                        let sib = group[j].siblings[l];
                        pairs[j] = if bit { (sib, prev[j]) } else { (prev[j], sib) };
                    }

                    (spec.node_group_ab)(&pairs, wz, wa, wb);

                    for w in 0..SLOT_WORDS {
                        let sib: [u32; BM_V] = from_fn(|j| group[j].siblings[l][w]);
                        or_u32_row(wz, sib_off + 32 * w, &sib);
                        or_u32_row(wa, sib_off + 32 * w, &sib);
                        or_u32_row(wb, sib_off + 32 * w, &[!0u32; BM_V]);

                        let xor: [u32; BM_V] = from_fn(|j| prev[j][w] ^ sib[j]);
                        let t: [u32; BM_V] = from_fn(|j| mask[j] & xor[j]);
                        or_u32_row(wz, t_off + 32 * w, &t);
                        or_u32_row(wa, t_off + 32 * w, &mask);
                        or_u32_row(wb, t_off + 32 * w, &xor);
                    }

                    for j in 0..BM_V {
                        for w in 0..SLOT_WORDS / 2 {
                            let word = wz[out_word + w][j];
                            prev[j][2 * w] = word as u32;
                            prev[j][2 * w + 1] = (word >> 32) as u32;
                        }
                    }
                }
            },
        )
    }

    /// `Vec<bool>` reference oracle for
    /// [`Self::generate_witness_batch_major_partial_into_chunk`].
    pub fn generate_witness_batch_major_partial_bool_chunk(
        &self,
        paths: &[ChunkPathInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;

        assert!(
            paths.len() <= 1usize << nu,
            "{} paths > 2^{nu} rows",
            paths.len()
        );
        let per_path: Vec<[Vec<bool>; 3]> = paths
            .par_iter()
            .map(|p| self.build_witness_zab_chunk(p))
            .collect();
        self.scatter_zab_batch_major(&per_path, nu)
    }

    /// Root of a chunk-leaf opening, computing **only** the root.
    ///
    /// Same fold as [`Self::reference_root_chunk`], but through
    /// [`HashSpec::compress`] instead of the witness builders — so it does
    /// `leaf_blocks + depth` compressions and allocates nothing, rather than
    /// materializing a `2^k_log`-bool block per compression to read 256 bits
    /// out of it. At the L0 shape (16 + 13 per opening) that is the difference
    /// between ~400 µs and ~20 µs per opening.
    ///
    /// This is what a circuit gate wants: it needs the root to wire, and the
    /// witness comes later in bulk from the batch-major drivers.
    /// `fast_root_matches_the_witness_fold` pins the two equal.
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

    /// Reference root for a chunk-leaf opening: the chunk chain then the
    /// node fold, natively, with the same compressions the R1CS encodes.
    ///
    /// Kept as the oracle. Prefer [`Self::root_chunk`] when only the root is
    /// wanted — this one pays a full witness block per compression.
    pub fn reference_root_chunk(&self, input: &ChunkPathInput) -> [u32; SLOT_WORDS] {
        self.assert_chunk_input(input);
        let mut prev = self.pinned_in_cv();
        for i in 0..self.leaf_blocks {
            let m = leaf_msg_words(&input.leaf_data, i);
            let block = (self.spec.raw_witness)(
                &prev,
                &m,
                NODE_COUNTER,
                NODE_BLOCK_LEN,
                self.chunk_flags(i),
            );
            prev = read_digest(&block, self.spec.out_cv_base);
        }
        for (l, sib) in input.siblings.iter().enumerate() {
            let bit = (input.index >> l) & 1 == 1;
            let (left, right) = if bit { (*sib, prev) } else { (prev, *sib) };
            let block = (self.spec.node_witness)(&left, &right);
            prev = read_digest(&block, self.spec.out_cv_base);
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

/// Bit `i` of a u64-word buffer (LSB-first within each word).
#[inline]
fn word_bit(w: &[u64], i: usize) -> bool {
    (w[i / 64] >> (i % 64)) & 1 == 1
}

/// The 128 bools at `[base, base+128)` as one `F128` (bit `t` → `lo`/`hi`).
#[inline]
fn pack_word(bits: &[bool], base: usize) -> F128 {
    let mut lo = 0u64;
    let mut hi = 0u64;
    for t in 0..64 {
        if bits[base + t] {
            lo |= 1u64 << t;
        }
        if bits[base + 64 + t] {
            hi |= 1u64 << t;
        }
    }
    F128 { lo, hi }
}

fn read_digest_words(w: &[u64], base: usize) -> [u32; SLOT_WORDS] {
    let mut d = [0u32; SLOT_WORDS];
    for j in 0..SLOT_BITS {
        if word_bit(w, base + j) {
            d[j / 32] |= 1u32 << (j % 32);
        }
    }
    d
}

// ---------------------------------------------------------------------------
// The walker LincheckCircuit
// ---------------------------------------------------------------------------

/// A CSC-transposed matrix pair: column `c`'s nonzero ROWS are
/// `rows[col_ptr[c] as usize .. col_ptr[c+1] as usize]`. Same shape as
/// `flock_core::lincheck::CscCircuit`'s internals, kept local because the
/// walker needs to gather from two of them at different row offsets.
struct Csc {
    a_col_ptr: Vec<u32>,
    a_rows: Vec<u32>,
    b_col_ptr: Vec<u32>,
    b_rows: Vec<u32>,
}

impl Csc {
    fn pair(a: &SparseBinaryMatrix, b: &SparseBinaryMatrix) -> Self {
        let (a_col_ptr, a_rows) = csc_from_rows(a);
        let (b_col_ptr, b_rows) = csc_from_rows(b);
        Self {
            a_col_ptr,
            a_rows,
            b_col_ptr,
            b_rows,
        }
    }

    /// `(Σ_{r ∈ colA(c)} eq[row_off + r], Σ_{r ∈ colB(c)} eq[row_off + r])`.
    #[inline]
    fn gather(&self, c: usize, eq: &[F128], row_off: usize) -> (F128, F128) {
        let mut sa = F128::ZERO;
        for &r in &self.a_rows[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize] {
            sa += eq[row_off + r as usize];
        }
        let mut sb = F128::ZERO;
        for &r in &self.b_rows[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize] {
            sb += eq[row_off + r as usize];
        }
        (sa, sb)
    }

    fn nnz(&self) -> (usize, usize) {
        (self.a_rows.len(), self.b_rows.len())
    }
}

fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    assert!(m.num_rows <= u32::MAX as usize);
    assert!(m.num_cols <= u32::MAX as usize);
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

/// [`LincheckCircuit`] for the composite Merkle block that never
/// materializes the `depth` per-level matrix copies.
///
/// The composite's rows split cleanly in two:
///
/// * **Base rows** — level `l`'s hash relation is the base block's matrix
///   with every row AND column index shifted by the same per-level offset.
///   A column marginal over these is therefore the base matrix's own column
///   marginal evaluated against an `eq` slice starting at that offset, so
///   one CSC transpose of the base block serves all `depth` levels.
/// * **Extras** — the globals, the swap-gadget columns, and the overridden
///   rows inside each block. These are the only rows referencing columns
///   outside their own level, and there are ~3.6K nonzeros per level, so
///   they are transposed once over the full composite.
///
/// At depth 26 over BLAKE3 this is ~88 MB resident instead of ~4.4 GB (plus
/// a second 4.4 GB for the transpose).
///
/// ## The `eq` factorization
///
/// Storage was the walker's first purpose; arithmetic is the second. Because
/// levels tile **aligned** `2^κ` subcubes, the level index is a set of high
/// address bits, and lincheck's `eq_inner` table is a per-bit product above
/// the univariate-skip region — so the table is rank-1 across those subcubes:
///
/// ```text
///   eq_inner[l·2^κ + r] = ρ_l · eq_base[r]
///   ⇒ ξ_M(l·2^κ + c_b)  = ρ_l · ξ_base(c_b)
/// ```
///
/// [`Self::fold_factored`] therefore folds the base block ONCE and pays one
/// multiply per composite column, instead of `depth` independent gathers:
/// ~21.6M operations instead of ~547M at depth 26, on prover and verifier
/// alike. [`Self::factor_eq`] *verifies* the factorization rather than
/// assuming it, and [`Self::fold_per_level`] is the general fallback, so an
/// unstructured table gets a slower answer and never a wrong one.
pub struct MerkleWalkerCircuit {
    n_cols: usize,
    depth: usize,
    base_k_log: usize,
    base: Csc,
    extras: Csc,
    const_pin: Option<usize>,
}

impl Debug for MerkleWalkerCircuit {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let (ba, bb) = self.base.nnz();
        let (xa, xb) = self.extras.nnz();
        f.debug_struct("MerkleWalkerCircuit")
            .field("n_cols", &self.n_cols)
            .field("depth", &self.depth)
            .field("base_nnz", &(ba + bb))
            .field("extras_nnz", &(xa + xb))
            .field("effective_nnz", &(self.depth * (ba + bb) + xa + xb))
            .finish()
    }
}

impl MerkleWalkerCircuit {
    /// Resident bytes of the two CSC transposes — the whole point of the
    /// walker. Compare `depth · base_nnz · 8` for the materialized form.
    pub fn resident_bytes(&self) -> usize {
        let sz =
            |c: &Csc| 4 * (c.a_col_ptr.len() + c.a_rows.len() + c.b_col_ptr.len() + c.b_rows.len());
        sz(&self.base) + sz(&self.extras)
    }

    /// Nonzeros of the system the walker represents — equal to the
    /// materialized matrices' nonzero count, and the cost of
    /// [`Self::fold_per_level`]. The factored fold touches only
    /// `base_nnz + extras_nnz` of them (~1/depth).
    pub fn effective_nnz(&self) -> usize {
        let (ba, bb) = self.base.nnz();
        let (xa, xb) = self.extras.nnz();
        self.depth * (ba + bb) + xa + xb
    }

    /// Decode a composite column into `(level, base column)`. Levels tile
    /// aligned `2^κ` subcubes, so this is a shift and a mask. Columns past
    /// the last level return `None`; gadget and global columns DO decode,
    /// but land in the base block's padding where the base CSC is empty, so
    /// they contribute nothing — which is exactly right.
    #[inline]
    fn decode(&self, c: usize) -> Option<(usize, usize)> {
        let level = c >> self.base_k_log;
        if level >= self.depth {
            return None;
        }
        Some((level, c & ((1usize << self.base_k_log) - 1)))
    }

    /// First composite column of level `l`'s subcube — also the row offset
    /// the base gather uses, since the embedding shifts rows and columns
    /// alike.
    #[inline]
    fn hash_base(&self, level: usize) -> usize {
        level << self.base_k_log
    }

    /// Factor `eq_inner` across the level subcubes: find `(eq_base, ρ)` with
    ///
    /// ```text
    ///   eq_inner[l·2^κ + r] == ρ_l · eq_base[r]   for all l < depth, r < 2^κ
    /// ```
    ///
    /// or `None` when no such pair exists.
    ///
    /// Lincheck's tables always factor. `build_quirky_eq_table` produces
    /// `L_{i_skip}(z_skip) · eq(x_rest, i_rest)` with the skip dimension in the
    /// low `K_SKIP = 6` bits and `eq` a per-bit product above them, so it is
    /// rank-1 at every bit boundary ≥ `K_SKIP` — and levels tile aligned `2^κ`
    /// subcubes with `κ = 14 > K_SKIP`. (The union's per-slot `w_t` prefix
    /// weight scales the *comb*, not the table, so it does not disturb this.)
    ///
    /// The pair is **verified, not assumed**: ρ is read off one reference
    /// column and then every entry is checked. That costs `depth · 2^κ`
    /// multiplies — 0.43M at depth 26, versus the 547M the factorization
    /// removes — so the guarantee is nearly free, and a caller with an
    /// unstructured table falls back to [`Self::fold_per_level`] instead of
    /// silently getting a wrong comb.
    fn factor_eq(&self, eq_inner: &[F128]) -> Option<(Vec<F128>, Vec<F128>)> {
        use rayon::prelude::*;
        let sub = 1usize << self.base_k_log;
        let slice = |l: usize| &eq_inner[self.hash_base(l)..][..sub];

        // A reference position with a nonzero entry, to divide by. Generic
        // tables hit this at (0, 0).
        let Some((l_ref, r0)) = (0..self.depth).find_map(|l| {
            slice(l)
                .iter()
                .position(|v| *v != F128::ZERO)
                .map(|r| (l, r))
        }) else {
            // Every level slice is zero, so every base gather is zero however
            // it is computed. Report the trivial factorization and let the
            // fast path run — the extras read `eq_inner` directly and are
            // unaffected.
            return Some((vec![F128::ZERO; sub], vec![F128::ZERO; self.depth]));
        };

        // Absorb ρ_{l_ref} into eq_base, making ρ_{l_ref} = 1.
        let eq_base = slice(l_ref).to_vec();
        let inv = eq_base[r0].inv();
        let rho: Vec<F128> = (0..self.depth)
            .map(|l| eq_inner[self.hash_base(l) + r0] * inv)
            .collect();

        let exact = (0..self.depth).into_par_iter().all(|l| {
            let rho_l = rho[l];
            slice(l)
                .iter()
                .zip(&eq_base)
                .all(|(&e, &base)| e == rho_l * base)
        });
        exact.then_some((eq_base, rho))
    }

    /// Whether `eq_inner` admits the rank-1 level factorization that
    /// [`Self::fold_alpha_batched`] exploits. Lincheck's tables do; a random
    /// vector does not (unless `depth == 1`, where the claim is vacuous).
    pub fn eq_factors_over_levels(&self, eq_inner: &[F128]) -> bool {
        self.factor_eq(eq_inner).is_some()
    }

    /// The factored fold: one gather pass over the base block against
    /// `eq_base`, then one multiply per composite column.
    ///
    /// ```text
    ///   comb[c] = extras_comb(c) + ρ_{level(c)} · base_comb(c_base)
    ///   base_comb(c_b) = α · ξ^A_base(c_b) + ξ^B_base(c_b)
    /// ```
    ///
    /// α distributes over the level scale, so the base comb can be α-batched
    /// once up front and reused by every level.
    fn fold_factored(
        &self,
        alpha: F128,
        eq_inner: &[F128],
        eq_base: &[F128],
        rho: &[F128],
    ) -> Vec<F128> {
        use rayon::prelude::*;
        let sub = 1usize << self.base_k_log;

        // The base block's own comb, ONCE — this is the depth-fold saving.
        let mut base_comb = vec![F128::ZERO; sub];
        base_comb
            .par_iter_mut()
            .enumerate()
            .for_each(|(c_base, slot)| {
                let (ba, bb) = self.base.gather(c_base, eq_base, 0);
                *slot = alpha * ba + bb;
            });

        let mut out = vec![F128::ZERO; self.n_cols];
        out.par_iter_mut().enumerate().for_each(|(c, slot)| {
            let (xa, xb) = self.extras.gather(c, eq_inner, 0);
            let mut v = alpha * xa + xb;
            if let Some((level, c_base)) = self.decode(c) {
                v += rho[level] * base_comb[c_base];
            }
            *slot = v;
        });
        out
    }

    /// The general fold: `depth` independent per-level gathers, correct for
    /// **any** `eq_inner`. This is the fallback when [`Self::factor_eq`]
    /// finds no factorization, and the oracle the factored path is tested
    /// against.
    ///
    /// Cost is the full effective nonzero count ([`Self::effective_nnz`]),
    /// ~547M at depth 26 over BLAKE3.
    pub fn fold_per_level(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(eq_inner.len(), self.n_cols);
        let mut out = vec![F128::ZERO; self.n_cols];
        out.par_iter_mut().enumerate().for_each(|(c, slot)| {
            let (mut sa, mut sb) = self.extras.gather(c, eq_inner, 0);
            if let Some((level, c_base)) = self.decode(c) {
                let (ba, bb) = self.base.gather(c_base, eq_inner, self.hash_base(level));
                sa += ba;
                sb += bb;
            }
            *slot = alpha * sa + sb;
        });
        out
    }
}

impl LincheckCircuit for MerkleWalkerCircuit {
    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), self.n_cols);
        if force_per_level() {
            return self.fold_per_level(alpha, eq_inner);
        }
        match self.factor_eq(eq_inner) {
            Some((eq_base, rho)) => self.fold_factored(alpha, eq_inner, &eq_base, &rho),
            None => self.fold_per_level(alpha, eq_inner),
        }
    }
}

/// A/B knob (`FLOCK_MERKLE_FOLD_PER_LEVEL=1`): skip the factorization and
/// always take the general per-level walk. Value-identical either way — the
/// factorization is verified before use — so one process can time both and
/// cancel thermal drift. Same role as `lincheck::FOLD_IBLOCK`.
fn force_per_level() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| {
        var("FLOCK_MERKLE_FOLD_PER_LEVEL")
            .ok()
            .is_some_and(|v| v != "0" && !v.is_empty())
    })
}

/// One Merkle path: the leaf digest, its index, and one sibling per level
/// (level 0 = closest to the leaf).
#[derive(Clone, Debug)]
pub struct PathInput {
    pub leaf: [u32; SLOT_WORDS],
    pub index: u64,
    pub siblings: Vec<[u32; SLOT_WORDS]>,
}

// ---------------------------------------------------------------------------
// Digest bit helpers — word `j/32`, bit `j%32`, matching the encoders'
// `write_word` (LSB-first within a 32-bit word).
// ---------------------------------------------------------------------------

#[inline]
fn digest_bit(d: &[u32; SLOT_WORDS], j: usize) -> bool {
    (d[j / 32] >> (j % 32)) & 1 == 1
}

fn write_digest(z: &mut [bool], base: usize, d: &[u32; SLOT_WORDS]) {
    for j in 0..SLOT_BITS {
        z[base + j] = digest_bit(d, j);
    }
}

fn read_digest(z: &[bool], base: usize) -> [u32; SLOT_WORDS] {
    let mut d = [0u32; SLOT_WORDS];
    for j in 0..SLOT_BITS {
        if z[base + j] {
            d[j / 32] |= 1u32 << (j % 32);
        }
    }
    d
}

/// Reference Merkle root: fold the path natively with the same node
/// compression the R1CS encodes. Mirrors [`MerkleTreeLayout::build_witness`]
/// without touching the witness, so a test can cross-check the two.
pub fn reference_root(spec: &HashSpec, input: &PathInput) -> [u32; SLOT_WORDS] {
    let mut prev = input.leaf;
    for (l, sib) in input.siblings.iter().enumerate() {
        let bit = (input.index >> l) & 1 == 1;
        let (left, right) = if bit { (*sib, prev) } else { (prev, *sib) };
        let block = (spec.node_witness)(&left, &right);
        prev = read_digest(&block, spec.out_cv_base);
    }
    prev
}
