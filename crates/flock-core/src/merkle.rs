//! Binary Merkle tree over a selectable hash — SHA-256 (using four-way
//! hardware SHA interleaving on supported ARM and x86-64 targets) or BLAKE3.
//!
//! The choice is a *runtime* one, carried by [`HashKind`] and threaded
//! from the PCS parameters / Ligerito config down to the two primitives
//! ([`hash_leaf`] and [`hash_pair`]). Everything else in this module — tree
//! layout, path extraction, multi-proof assembly and verification — is
//! hash-agnostic: both hashes produce 32-byte digests, so the proof format is
//! identical either way and only the digest *values* differ.
//!
//! Dispatch granularity is one branch per *batch* of hashes — a whole level
//! run for BLAKE3, four nodes for SHA-256 — not per byte, so carrying the
//! choice at runtime costs nothing measurable.
//!
//! Layout for `num_leaves = 2^k` leaves:
//!   tree[0..num_leaves]                              = leaf hashes (level k)
//!   tree[num_leaves..3·num_leaves/2]                 = level k−1
//!   ...
//!   tree[2·num_leaves − 2..2·num_leaves − 1]         = root (level 0)
//!
//! Total nodes: `2·num_leaves − 1`. The flat layout keeps the tree contiguous
//! in memory for cheap Merkle-path extraction later.
//!
//! SHA-256 uses the [`sha2`] crate. On aarch64 with the `sha2` target feature
//! (set implicitly by `target-cpu=native` on M-series), the crate uses
//! `sha256h`/`sha256h2`/`sha256su0`/`sha256su1` ARM crypto extension
//! instructions; this is detected at runtime by [`cpufeatures`].
//!
//! BLAKE3 uses the [`blake3`] crate and BLAKE3's own tree semantics: a leaf is
//! a non-root chaining value, an internal node is a PARENT-flagged compression
//! of its two children. Both go through the crate's SIMD compression entry
//! point in [`BLAKE3_BATCH`]-wide calls, so the batch always fills the widest
//! vector the machine has (4 under NEON, 8 under AVX2, 16 under AVX-512) —
//! see the note above [`blake3_hash_many`] for why that API and how it is
//! kept honest.
//!
//! Which hash is faster is size- and target-dependent. On a target with the
//! ARM SHA-2 or x86 SHA-NI extensions, SHA-256 wins: its two compressions per
//! internal node run on a dedicated hardware unit, against BLAKE3's one in the
//! vector units. `benches/merkle.rs` measures both;
//! `benches/blake3_node_probe.rs` breaks the BLAKE3 side down per node shape.
//!
//! Domain separation differs between the two. BLAKE3's PARENT flag separates
//! leaf and internal pre-images inherently. The SHA-256 construction does
//! **not** — it has no leaf/internal tag, so it is open to second-preimage
//! attacks via interpretation collision, and a production PCS commit should
//! prepend `0x00`/`0x01` (or equivalent) before relying on it.

use rayon::prelude::*;
use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

pub use crate::hash::HashKind;

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
))]
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
))]
const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// 4-way interleaved SHA-256 using ARM crypto-extension intrinsics.
///
/// The M-series SHA unit is pipelined: a single dependent compress
/// chain runs at ~21 ns/compress, while interleaved independent
/// streams sustain ~16 ns/compress on real (distinct) data — a ~1.35×
/// throughput win, measured on M4 Max at m=30. The `sha2` crate hashes
/// one stream at a time, so bulk Merkle hashing (independent leaves /
/// independent nodes within a level) leaves that on the table.
///
/// Digests are byte-identical to `Sha256::digest`.
#[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
#[path = "merkle/aarch64.rs"]
mod sha256x4;

/// Four SHA-256 streams interleaved across the x86 SHA-NI pipeline.
///
/// SHA-NI accelerates one stream but retains a dependent state chain. Running
/// four independent states round-for-round exposes enough instruction-level
/// parallelism for bulk Merkle leaves and same-level parent nodes.
#[cfg(all(target_arch = "x86_64", target_feature = "sha"))]
#[path = "merkle/x86_64.rs"]
mod sha256x4;

/// Global Merkle hash call/compression counters, enabled with
/// `--features hash-count` (e.g. by `benches/verifier_hash_count.rs`).
/// Relaxed atomics — exact totals, no ordering guarantees across threads.
#[cfg(feature = "hash-count")]
pub mod hash_count {
    use super::HashKind;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static LEAF_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static LEAF_COMPRESSIONS: AtomicU64 = AtomicU64::new(0);
    pub static PAIR_CALLS: AtomicU64 = AtomicU64::new(0);

    /// SHA-256 compression count for a one-shot hash of `len` bytes:
    /// ceil((len + 9) / 64) — payload + 0x80 pad + 8-byte length.
    #[inline]
    pub fn sha256_blocks(len: usize) -> u64 {
        ((len + 9).div_ceil(64)) as u64
    }

    /// BLAKE3 compression count for a one-shot hash of `len` bytes: one
    /// compression per 64-byte block (a final partial block still costs one,
    /// and the empty input costs one), plus one parent compression per
    /// internal node of the chunk tree — `c − 1` for `c` 1 KiB chunks.
    #[inline]
    pub fn blake3_blocks(len: usize) -> u64 {
        let blocks = (len.div_ceil(64)).max(1) as u64;
        let chunks = (len.div_ceil(1024)).max(1) as u64;
        blocks + (chunks - 1)
    }

    /// Compression count for a one-shot hash of `len` bytes under `kind`.
    #[inline]
    pub fn blocks(kind: HashKind, len: usize) -> u64 {
        match kind {
            HashKind::Sha256 => sha256_blocks(len),
            HashKind::Blake3 => blake3_blocks(len),
        }
    }

    pub fn reset() {
        LEAF_CALLS.store(0, Relaxed);
        LEAF_COMPRESSIONS.store(0, Relaxed);
        PAIR_CALLS.store(0, Relaxed);
    }

    /// (leaf_calls, leaf_compressions, pair_calls). A pair hash is
    /// 2 compressions under SHA-256 (64 B payload + padding block) and
    /// 1 under BLAKE3 (a single 64-byte block, no length padding).
    pub fn snapshot() -> (u64, u64, u64) {
        (
            LEAF_CALLS.load(Relaxed),
            LEAF_COMPRESSIONS.load(Relaxed),
            PAIR_CALLS.load(Relaxed),
        )
    }
}

// ---------------------------------------------------------------------------
// BLAKE3 tree primitives.
//
// The BLAKE3 Merkle tree uses BLAKE3's *own* tree semantics rather than
// `blake3::hash` over concatenated bytes:
//
//   leaf   = Hasher::new().update(leaf_bytes).finalize_non_root()
//   parent = merge_subtrees_non_root(left_cv, right_cv, Mode::Hash)
//
// Two reasons. First, correctness: these are non-root chaining values, which is
// what interior tree nodes are supposed to be, and BLAKE3's PARENT flag gives
// leaf/parent domain separation for free — the property this module's header
// notes the SHA-256 construction lacks. Second, speed: both map onto BLAKE3's
// batched compression entry point, which is ~2× the scalar API (measured by
// `benches/blake3_node_probe.rs`).
//
// The two functions below are the *specification* — stable, public `blake3`
// API. The `blake3_hash_many_*` paths are optimizations that must agree with
// them bit-for-bit; `blake3_batched_matches_scalar_spec` in this module's
// tests is what holds them to it.
//
// NOTE — this deliberately differs from the sibling implementation on
// TomWambsgans/flock `blake3-pcs`, which defines a leaf as `blake3::hash(x)`
// and a parent as `blake3::hash(l ‖ r)` (root-flagged one-shot hashes). That
// contract is simpler and reproducible with plain `blake3::hash`; this one
// buys leaf/parent domain separation instead, which the SHA-256 construction
// lacks. The two produce *different digests* — a tree built under one does not
// verify under the other, so the choice has to be made once, project-wide.
// Both batch equally well, so it is not a performance trade.
// ---------------------------------------------------------------------------

/// Non-root chaining value of one BLAKE3 leaf, of any length.
#[inline]
fn blake3_leaf_cv(data: &[u8]) -> Hash {
    use blake3::hazmat::HasherExt;
    blake3::Hasher::new().update(data).finalize_non_root()
}

/// BLAKE3 parent-node chaining value of two children.
#[inline]
fn blake3_parent_cv(left: &Hash, right: &Hash) -> Hash {
    blake3::hazmat::merge_subtrees_non_root(left, right, blake3::hazmat::Mode::Hash)
}

/// Hash one leaf of arbitrary byte length.
#[inline]
pub fn hash_leaf(data: &[u8], kind: HashKind) -> Hash {
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::LEAF_CALLS.fetch_add(1, Relaxed);
        hash_count::LEAF_COMPRESSIONS.fetch_add(hash_count::blocks(kind, data.len()), Relaxed);
    }
    match kind {
        HashKind::Sha256 => Sha256::digest(data).into(),
        HashKind::Blake3 => blake3_leaf_cv(data),
    }
}

/// Hash a pair of children into a parent node (64 B → 32 B).
#[inline]
pub fn hash_pair(left: &Hash, right: &Hash, kind: HashKind) -> Hash {
    #[cfg(feature = "hash-count")]
    hash_count::PAIR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match kind {
        HashKind::Sha256 => {
            let mut h = Sha256::new();
            h.update(left);
            h.update(right);
            h.finalize().into()
        }
        HashKind::Blake3 => blake3_parent_cv(left, right),
    }
}

/// SHA-256 of four equal-length inputs, four-way interleaved across the
/// hardware SHA unit where the target supports it.
///
/// Digests are byte-identical to `Sha256::digest` either way; the fallback
/// exists so callers need not repeat the `cfg` test.
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
))]
#[inline]
fn sha256_hash4(inputs: [&[u8]; 4], outs: &mut [Hash]) {
    sha256x4::hash4_equal_len(inputs, outs);
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
)))]
#[inline]
fn sha256_hash4(inputs: [&[u8]; 4], outs: &mut [Hash]) {
    for (out, input) in outs.iter_mut().zip(inputs) {
        *out = Sha256::digest(input).into();
    }
}

// --- BLAKE3 batched compression -------------------------------------------
//
// `blake3::platform` is `#[doc(hidden)]` and labelled "undocumented and
// unstable". We depend on it deliberately: it is the only way to reach the
// crate's SIMD-batched compression (4-way under NEON, 8/16-way under
// AVX2/AVX-512), worth ~2× over the scalar API on our node shapes, and the
// alternative — hand-writing a 4-way NEON BLAKE3 alongside `merkle/aarch64.rs`
// — is a great deal more code to own and audit.
//
// The exposure is bounded: every batched result is checked against the stable
// `hazmat` spec above in this module's tests, so a semantic change in a
// `blake3` update fails the suite rather than silently altering commitments,
// and an API removal fails the build. Nothing here is reachable if the
// equality does not hold.

/// BLAKE3's IV — the key words for unkeyed hashing. Fixed by the spec.
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// BLAKE3 domain flags, fixed by the spec.
const BLAKE3_CHUNK_START: u8 = 1;
const BLAKE3_CHUNK_END: u8 = 2;
const BLAKE3_PARENT: u8 = 4;

/// Cached SIMD platform. `Platform::detect()` is cheap but not free, and the
/// tree build reaches the batched path once per [`BLAKE3_BATCH`] nodes.
fn blake3_platform() -> blake3::platform::Platform {
    use std::sync::OnceLock;
    static PLATFORM: OnceLock<blake3::platform::Platform> = OnceLock::new();
    *PLATFORM.get_or_init(blake3::platform::Platform::detect)
}

/// Inputs handed to `hash_many` per call.
///
/// Sized to the widest `simd_degree` that exists — 4 under NEON, 8 under AVX2,
/// 16 under AVX-512 — so the batch fills the machine's vector rather than
/// leaving lanes idle. This is portability insurance, not a local win: swept
/// over 4/8/16/64/256 on an M4 Max (NEON, degree 4) the spread was ~1-5%, i.e.
/// inside run-to-run noise, with 16 marginally best. It should matter on an
/// AVX-512 host, where a 4-input call can only ever fill a quarter of the
/// vector; that has not been measured here.
const BLAKE3_BATCH: usize = 16;

/// Drive `hash_many` over `data`, a run of `out.len()` contiguous `N`-byte
/// messages, in [`BLAKE3_BATCH`]-wide calls. `flags`/`start`/`end` select the
/// node type (chunk vs parent).
///
/// Allocation-free: the pointer array lives on the stack, so unlike a
/// `Vec`-per-call formulation this costs nothing per batch.
#[inline]
fn blake3_hash_many<const N: usize>(
    data: &[u8],
    out: &mut [Hash],
    flags: u8,
    flags_start: u8,
    flags_end: u8,
) {
    debug_assert_eq!(data.len(), out.len() * N);
    let plat = blake3_platform();
    for (outs, msgs) in out
        .chunks_mut(BLAKE3_BATCH)
        .zip(data.chunks(BLAKE3_BATCH * N))
    {
        let n = outs.len();
        // Fill a stack array of input pointers. Slot 0 seeds the array so the
        // unused tail (never passed to `hash_many`, which sees `&inputs[..n]`)
        // holds a valid reference rather than uninitialized memory.
        let first: &[u8; N] = msgs[..N].try_into().unwrap();
        let mut inputs: [&[u8; N]; BLAKE3_BATCH] = [first; BLAKE3_BATCH];
        for (i, slot) in inputs[..n].iter_mut().enumerate() {
            *slot = msgs[i * N..(i + 1) * N].try_into().unwrap();
        }
        // SAFETY: `Hash` is `[u8; 32]`, so `outs` is exactly `n * 32` bytes of
        // initialized, contiguous, unpadded storage — the amount `hash_many`
        // writes for `n` inputs.
        let out_bytes: &mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(outs.as_mut_ptr() as *mut u8, n * 32) };
        plat.hash_many(
            &inputs[..n],
            &BLAKE3_IV,
            0,
            blake3::IncrementCounter::No,
            flags,
            flags_start,
            flags_end,
            out_bytes,
        );
    }
}

/// Batched BLAKE3 leaves: `out.len()` messages of `leaf_size` bytes, laid out
/// contiguously in `data`. Equivalent to [`blake3_leaf_cv`] per leaf.
///
/// The batched entry point compresses whole 64-byte blocks of a single chunk,
/// so it applies only when `leaf_size` is a multiple of 64 and at most
/// `CHUNK_LEN` (1024) — which covers the real commit geometry, where a leaf is
/// `16 << log_batch_size` bytes. Returns `false` for any other size, leaving
/// the caller to hash leaves one at a time.
fn blake3_hash_many_leaves(data: &[u8], leaf_size: usize, out: &mut [Hash]) -> bool {
    macro_rules! dispatch {
        ($($n:literal),+ $(,)?) => {
            match leaf_size {
                $($n => {
                    blake3_hash_many::<$n>(
                        data, out, 0, BLAKE3_CHUNK_START, BLAKE3_CHUNK_END,
                    );
                    true
                })+
                _ => false,
            }
        };
    }
    // Leaf sizes are `16 << log_batch_size`, so only powers of two arise.
    dispatch!(64, 128, 256, 512, 1024)
}

/// Whether [`blake3_hash_many_leaves`] can batch this leaf size. Must list
/// exactly the sizes that function dispatches on — `blake3_batch_dispatch_agrees`
/// holds the two together.
#[inline]
fn blake3_leaf_size_is_batchable(leaf_size: usize) -> bool {
    matches!(leaf_size, 64 | 128 | 256 | 512 | 1024)
}

/// Batched BLAKE3 parent nodes: `data` is `out.len()` contiguous 64-byte
/// (left ‖ right) child pairs. Equivalent to [`blake3_parent_cv`] per node.
#[inline]
fn blake3_hash_many_parents(data: &[u8], out: &mut [Hash]) {
    blake3_hash_many::<64>(data, out, BLAKE3_PARENT, 0, 0);
}

/// Hash a run of `out.len()` equal-size leaves from `data` under `kind`.
///
/// The two hashes batch differently, so they take different shapes here:
/// SHA-256's kernel is inherently four-wide, while BLAKE3 wants the widest
/// batch the machine offers. Both are rayon-parallel and both are
/// byte-identical to calling [`hash_leaf`] on each leaf.
fn hash_leaves(data: &[u8], leaf_size: usize, out: &mut [Hash], kind: HashKind) {
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::LEAF_CALLS.fetch_add(out.len() as u64, Relaxed);
        hash_count::LEAF_COMPRESSIONS.fetch_add(
            out.len() as u64 * hash_count::blocks(kind, leaf_size),
            Relaxed,
        );
    }
    match kind {
        HashKind::Blake3 if blake3_leaf_size_is_batchable(leaf_size) => {
            out.par_chunks_mut(BLAKE3_GROUP)
                .zip(data.par_chunks(BLAKE3_GROUP * leaf_size))
                .for_each(|(outs, leaves)| {
                    blake3_hash_many_leaves(leaves, leaf_size, outs);
                });
        }
        HashKind::Blake3 => out
            .par_iter_mut()
            .zip(data.par_chunks(leaf_size))
            .for_each(|(o, leaf)| *o = blake3_leaf_cv(leaf)),
        HashKind::Sha256 => {
            out.par_chunks_mut(4)
                .zip(data.par_chunks(4 * leaf_size))
                .for_each(|(outs, leaves)| {
                    if outs.len() == 4 {
                        sha256_hash4(
                            [
                                &leaves[..leaf_size],
                                &leaves[leaf_size..2 * leaf_size],
                                &leaves[2 * leaf_size..3 * leaf_size],
                                &leaves[3 * leaf_size..],
                            ],
                            outs,
                        );
                    } else {
                        for (out, leaf) in outs.iter_mut().zip(leaves.chunks(leaf_size)) {
                            *out = Sha256::digest(leaf).into();
                        }
                    }
                });
        }
    }
}

/// Nodes per rayon task in the batched BLAKE3 paths: enough to amortize task
/// dispatch over many `hash_many` calls, small enough to stay cache-resident.
const BLAKE3_GROUP: usize = 1024;

/// Hash one internal level: `write[i] = hash_pair(read[2i], read[2i+1])`.
///
/// Children are contiguous 64-byte spans of the level below, so both hashes
/// read them zero-copy. Small upper levels can't fill the cores, so a rayon
/// dispatch per level costs more than the hashing itself (~3× at the top of a
/// 2^18 tree); those are hashed serially — still SIMD-batched — and only the
/// wide lower levels fan out.
fn hash_pairs_level(read: &[Hash], write: &mut [Hash], kind: HashKind) {
    #[cfg(feature = "hash-count")]
    hash_count::PAIR_CALLS.fetch_add(write.len() as u64, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: `Hash` is `[u8; 32]`, so a slice of `n` hashes is exactly `32n`
    // initialized bytes with no padding.
    let read_bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
    const SERIAL_LEVEL_NODES: usize = 1024;
    let serial = write.len() <= SERIAL_LEVEL_NODES;

    match kind {
        HashKind::Blake3 => {
            if serial {
                blake3_hash_many_parents(read_bytes, write);
            } else {
                write
                    .par_chunks_mut(BLAKE3_GROUP)
                    .zip(read_bytes.par_chunks(BLAKE3_GROUP * 64))
                    .for_each(|(outs, children)| blake3_hash_many_parents(children, outs));
            }
        }
        HashKind::Sha256 => {
            let hash_quad = |outs: &mut [Hash], children: &[u8]| {
                if outs.len() == 4 {
                    sha256_hash4(
                        [
                            &children[..64],
                            &children[64..128],
                            &children[128..192],
                            &children[192..256],
                        ],
                        outs,
                    );
                } else {
                    for (i, out) in outs.iter_mut().enumerate() {
                        let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
                        let r: &Hash = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                        let mut h = Sha256::new();
                        h.update(l);
                        h.update(r);
                        *out = h.finalize().into();
                    }
                }
            };
            if serial {
                for (outs, children) in write.chunks_mut(4).zip(read_bytes.chunks(256)) {
                    hash_quad(outs, children);
                }
            } else {
                write
                    .par_chunks_mut(4)
                    .zip(read_bytes.par_chunks(256))
                    .for_each(|(outs, children)| hash_quad(outs, children));
            }
        }
    }
}

/// Compute the Merkle root of `data` split into `num_leaves` equal-sized leaves.
///
/// Multi-threaded via rayon. `num_leaves` must be a power of two and divide
/// `data.len()`. Returns the 32-byte root. The intermediate tree is allocated
/// and dropped; if you need it for path opening, use [`merkle_tree`] instead.
pub fn merkle_root(data: &[u8], num_leaves: usize, kind: HashKind) -> Hash {
    let tree = merkle_tree(data, num_leaves, kind);
    tree[tree.len() - 1]
}

/// Compute the full Merkle tree (flat layout, see module docs) for `data`
/// split into `num_leaves` equal-sized leaves, hashed under `kind`.
pub fn merkle_tree(data: &[u8], num_leaves: usize, kind: HashKind) -> Vec<Hash> {
    assert!(
        num_leaves.is_power_of_two() && num_leaves > 0,
        "num_leaves must be power of 2"
    );
    assert_eq!(
        data.len() % num_leaves,
        0,
        "data length must be a multiple of num_leaves"
    );

    let leaf_size = data.len() / num_leaves;
    let total_nodes = 2 * num_leaves - 1;
    // Uninit alloc — every node is written exactly once before being read:
    // leaves at step 1, then each internal level reads the level below (which
    // was just written) and writes itself.
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes);

    // 1. Leaves — fully parallel, SIMD-batched across leaves where possible.
    hash_leaves(data, leaf_size, &mut tree[..num_leaves], kind);

    // 2. Internal levels — parallel within a level, sequential across levels.
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        // Split the buffer at the end of the current level so we get two
        // non-overlapping mutable slices: `read` (input) and `write` (output).
        let (read, rest) = tree[read_start..].split_at_mut(read_len);
        let write = &mut rest[..next_len];

        hash_pairs_level(read, write, kind);

        read_start += read_len;
        read_len = next_len;
    }

    tree
}

/// Sequential (single-threaded) version of [`merkle_tree`]. Used for
/// benchmark comparison and as the test oracle.
pub fn merkle_tree_sequential(data: &[u8], num_leaves: usize, kind: HashKind) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(data.len() % num_leaves, 0);

    let leaf_size = data.len() / num_leaves;
    let total_nodes = 2 * num_leaves - 1;
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes);

    for (i, leaf) in data.chunks(leaf_size).enumerate() {
        tree[i] = hash_leaf(leaf, kind);
    }
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        for i in 0..next_len {
            let left = tree[read_start + 2 * i];
            let right = tree[read_start + 2 * i + 1];
            tree[read_start + read_len + i] = hash_pair(&left, &right, kind);
        }
        read_start += read_len;
        read_len = next_len;
    }
    tree
}

// ---------------------------------------------------------------------------
// Merkle path opening and verification.
// ---------------------------------------------------------------------------

/// Build an opening proof for leaf `index`: the sibling hashes from the leaf
/// level up to (but not including) the root.
///
/// `tree` must be the flat tree produced by [`merkle_tree`] or
/// [`merkle_tree_sequential`] for `num_leaves` leaves. The returned vector has
/// length `log2(num_leaves)`.
///
/// Verify with [`verify_merkle_proof`].
pub fn merkle_proof(tree: &[Hash], num_leaves: usize, index: usize) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert!(index < num_leaves);
    assert_eq!(tree.len(), 2 * num_leaves - 1);

    let log_n = num_leaves.trailing_zeros() as usize;
    let mut proof = Vec::with_capacity(log_n);

    let mut level_start = 0usize;
    let mut level_len = num_leaves;
    let mut idx = index;
    while level_len > 1 {
        let sibling_idx = idx ^ 1;
        proof.push(tree[level_start + sibling_idx]);
        level_start += level_len;
        level_len >>= 1;
        idx >>= 1;
    }
    proof
}

/// Verify a Merkle opening: recomputes the root from `leaf_hash`, the path,
/// and the leaf index. Returns true iff the recomputed root matches `root`.
pub fn verify_merkle_proof(
    root: &Hash,
    leaf_hash: &Hash,
    index: usize,
    proof: &[Hash],
    kind: HashKind,
) -> bool {
    let mut acc = *leaf_hash;
    let mut idx = index;
    for sibling in proof {
        // If idx is even, our node is the LEFT child; sibling is on the RIGHT.
        let (left, right) = if idx & 1 == 0 {
            (acc, *sibling)
        } else {
            (*sibling, acc)
        };
        acc = hash_pair(&left, &right, kind);
        idx >>= 1;
    }
    &acc == root
}

// ---------------------------------------------------------------------------
// Multi-proof (Octopus / batched opening): one shared proof for multiple leaf
// positions, deduplicating siblings that lie on multiple paths.
// ---------------------------------------------------------------------------

/// Build a Merkle multi-proof for `positions`. Returns the sibling hashes
/// needed to verify ALL positions against the root, in the canonical
/// bottom-up sorted-by-position traversal order.
///
/// `positions` need not be sorted or unique; the function sorts + dedupes
/// internally. For `q` queries in a tree of depth `d`, the output is at
/// most `q · d` hashes (matching `q` independent paths) and typically much
/// smaller (siblings shared across multiple paths are emitted once).
///
/// Verify with [`verify_merkle_multi_proof`].
pub fn merkle_multi_proof(tree: &[Hash], num_leaves: usize, positions: &[usize]) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(tree.len(), 2 * num_leaves - 1);

    if positions.is_empty() || num_leaves == 1 {
        return Vec::new();
    }

    let mut active: Vec<usize> = positions.to_vec();
    active.sort_unstable();
    active.dedup();
    debug_assert!(active.iter().all(|&p| p < num_leaves));

    let mut proof = Vec::new();
    let mut level_start = 0usize;
    let mut level_len = num_leaves;

    while level_len > 1 {
        let mut next = Vec::with_capacity(active.len());
        let mut i = 0;
        while i < active.len() {
            let p = active[i];
            let sib_active = i + 1 < active.len() && active[i + 1] == (p ^ 1);
            if sib_active {
                // Both children active — no sibling hash needed; both fold into
                // the same parent.
                i += 2;
            } else {
                // Sibling not in active set; emit it.
                proof.push(tree[level_start + (p ^ 1)]);
                i += 1;
            }
            next.push(p >> 1);
        }
        // `next` is sorted-unique by construction: the input was sorted-unique;
        // consecutive sibling pairs (handled above) collapse to one; otherwise
        // p >> 1 preserves strict ordering.
        active = next;
        level_start += level_len;
        level_len >>= 1;
    }

    proof
}

/// Verify a Merkle multi-proof produced by [`merkle_multi_proof`].
///
/// `sorted_unique_positions` and `leaf_hashes` must be aligned and sorted:
/// `leaf_hashes[i]` is the hash of the leaf at `sorted_unique_positions[i]`,
/// and the position list is strictly ascending. Returns true iff the
/// reconstructed root equals `root` and the proof is consumed exactly.
pub fn verify_merkle_multi_proof(
    root: &Hash,
    num_leaves: usize,
    sorted_unique_positions: &[usize],
    leaf_hashes: &[Hash],
    proof: &[Hash],
    kind: HashKind,
) -> bool {
    if !num_leaves.is_power_of_two() || num_leaves == 0 {
        return false;
    }
    if sorted_unique_positions.len() != leaf_hashes.len() {
        return false;
    }
    if sorted_unique_positions.is_empty() {
        // Vacuous; nothing to verify. Treat as "ok" iff the proof is empty.
        return proof.is_empty();
    }
    // Verify the position list is sorted strictly ascending + in range.
    for (i, &p) in sorted_unique_positions.iter().enumerate() {
        if p >= num_leaves {
            return false;
        }
        if i > 0 && sorted_unique_positions[i - 1] >= p {
            return false;
        }
    }
    // Edge case: 1-leaf tree, no proof needed.
    if num_leaves == 1 {
        return proof.is_empty() && leaf_hashes[0] == *root;
    }

    let mut active: Vec<(usize, Hash)> = sorted_unique_positions
        .iter()
        .copied()
        .zip(leaf_hashes.iter().copied())
        .collect();
    let mut proof_iter = proof.iter().copied();
    let mut level_len = num_leaves;

    while level_len > 1 {
        let mut next = Vec::with_capacity(active.len());
        let mut i = 0;
        while i < active.len() {
            let (p, h) = active[i];
            let sib_active = i + 1 < active.len() && active[i + 1].0 == (p ^ 1);
            let (left, right) = if sib_active {
                let (_, h_sib) = active[i + 1];
                // Sorted strictly ascending → active[i+1].0 = p + 1 (= p ^ 1
                // since p is even when p ^ 1 = p + 1). So p is LEFT child.
                debug_assert_eq!(p & 1, 0);
                i += 2;
                (h, h_sib)
            } else {
                let sib = match proof_iter.next() {
                    Some(s) => s,
                    None => return false,
                };
                i += 1;
                if p & 1 == 0 { (h, sib) } else { (sib, h) }
            };
            next.push((p >> 1, hash_pair(&left, &right, kind)));
        }
        active = next;
        level_len >>= 1;
    }

    // After the loop, `active` has exactly one element (the root). Reject
    // any leftover proof bytes.
    if proof_iter.next().is_some() {
        return false;
    }
    active.len() == 1 && active[0].1 == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every structural test runs against both hashes: the tree, path and
    /// multi-proof logic is hash-agnostic, so anything true of one must hold
    /// for the other.
    const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

    #[test]
    fn two_leaves_matches_hand_computation() {
        // Two 8-byte leaves: [0,1,2,3,4,5,6,7] and [8,9,10,11,12,13,14,15].
        let data: Vec<u8> = (0..16).collect();
        for kind in KINDS {
            let tree = merkle_tree(&data, 2, kind);
            assert_eq!(tree.len(), 3); // 2 leaves + 1 root

            let h0 = hash_leaf(&data[0..8], kind);
            let h1 = hash_leaf(&data[8..16], kind);
            let root = hash_pair(&h0, &h1, kind);

            assert_eq!(tree[0], h0, "{kind}");
            assert_eq!(tree[1], h1, "{kind}");
            assert_eq!(tree[2], root, "{kind}");
        }
    }

    /// The primitives must agree with the reference APIs of the underlying
    /// crates — this is what pins the digests to the real hash functions
    /// rather than merely to themselves.
    #[test]
    fn primitives_match_reference_implementations() {
        use blake3::hazmat::HasherExt;
        let data: Vec<u8> = (0..=255u8).cycle().take(3000).collect();

        // SHA-256: a plain one-shot digest.
        assert_eq!(
            hash_leaf(&data, HashKind::Sha256),
            <[u8; 32]>::from(Sha256::digest(&data))
        );
        let (l, r) = ([7u8; 32], [9u8; 32]);
        let cat: Vec<u8> = l.iter().chain(r.iter()).copied().collect();
        assert_eq!(
            hash_pair(&l, &r, HashKind::Sha256),
            hash_leaf(&cat, HashKind::Sha256),
            "sha256 pair hash is the digest of the concatenation"
        );

        // BLAKE3: non-root chaining values, per BLAKE3's own tree semantics.
        assert_eq!(
            hash_leaf(&data, HashKind::Blake3),
            blake3::Hasher::new().update(&data).finalize_non_root()
        );
        assert_eq!(
            hash_pair(&l, &r, HashKind::Blake3),
            blake3::hazmat::merge_subtrees_non_root(&l, &r, blake3::hazmat::Mode::Hash)
        );
        // Deliberately NOT `blake3::hash` — that is the root finalization, and
        // interior tree nodes must not be root hashes.
        assert_ne!(
            hash_leaf(&data, HashKind::Blake3),
            *blake3::hash(&data).as_bytes(),
            "leaf CVs must be non-root"
        );
    }

    /// BLAKE3's PARENT flag domain-separates internal nodes from leaves, so a
    /// parent hash is not reproducible as a leaf hash of the concatenation.
    /// This is the second-preimage-via-reinterpretation gap that the SHA-256
    /// construction (see module header) still has.
    #[test]
    fn blake3_separates_leaf_and_parent_domains() {
        let (l, r) = ([7u8; 32], [9u8; 32]);
        let cat: Vec<u8> = l.iter().chain(r.iter()).copied().collect();
        assert_ne!(
            hash_pair(&l, &r, HashKind::Blake3),
            hash_leaf(&cat, HashKind::Blake3),
            "PARENT flag must separate the two domains"
        );
        // The SHA-256 construction does not have this property. Asserted so the
        // difference is recorded rather than assumed either way.
        assert_eq!(
            hash_pair(&l, &r, HashKind::Sha256),
            hash_leaf(&cat, HashKind::Sha256),
        );
    }

    /// The batched BLAKE3 path (`blake3::platform`, an unstable API) must agree
    /// bit-for-bit with the stable `hazmat` spec. This is what makes depending
    /// on that API safe: if a `blake3` update changes its semantics, this fails
    /// rather than silently changing every commitment we produce.
    #[test]
    fn blake3_batched_matches_scalar_spec() {
        // Node counts chosen around `BLAKE3_BATCH` (64): a single node, a
        // partial batch, exactly one batch, one past it, and several batches
        // with a partial tail. A width bug in the batch loop shows up here.
        let counts = [1usize, 5, 63, 64, 65, 200];

        // Parents.
        for n in counts {
            let children: Vec<u8> = (0..=255u8).cycle().take(n * 64).collect();
            let mut batched = vec![[0u8; 32]; n];
            blake3_hash_many_parents(&children, &mut batched);
            for i in 0..n {
                let l: &Hash = children[i * 64..i * 64 + 32].try_into().unwrap();
                let r: &Hash = children[i * 64 + 32..i * 64 + 64].try_into().unwrap();
                assert_eq!(batched[i], blake3_parent_cv(l, r), "parent {i} of {n}");
            }
        }

        // Leaves, at every size the batched path claims to handle.
        for leaf_size in [64usize, 128, 256, 512, 1024] {
            for n in counts {
                let data: Vec<u8> = (0..=255u8).cycle().take(n * leaf_size).collect();
                let mut batched = vec![[0u8; 32]; n];
                assert!(
                    blake3_hash_many_leaves(&data, leaf_size, &mut batched),
                    "size {leaf_size} should take the batched path"
                );
                for i in 0..n {
                    assert_eq!(
                        batched[i],
                        blake3_leaf_cv(&data[i * leaf_size..(i + 1) * leaf_size]),
                        "leaf {i} of {n} at size {leaf_size}"
                    );
                }
            }
        }
    }

    /// The cheap `blake3_leaf_size_is_batchable` predicate — which decides
    /// which code path `hash_leaves` takes — must agree exactly with what
    /// `blake3_hash_many_leaves` actually dispatches on. If they drift, leaves
    /// either silently take the slow path or hit an unreachable arm.
    #[test]
    fn blake3_batch_dispatch_agrees() {
        for leaf_size in [
            1usize, 16, 32, 48, 63, 64, 65, 100, 128, 192, 256, 512, 1000, 1024, 1088, 2048,
        ] {
            let data = vec![0u8; leaf_size];
            let mut out = [[0u8; 32]; 1];
            let dispatched = blake3_hash_many_leaves(&data, leaf_size, &mut out);
            assert_eq!(
                dispatched,
                blake3_leaf_size_is_batchable(leaf_size),
                "predicate and dispatch disagree at leaf_size={leaf_size}"
            );
        }
    }

    /// The whole point of the option: the two hashes must actually produce
    /// different commitments.
    #[test]
    fn the_two_kinds_produce_different_roots() {
        let data = random_data(64, 32, 11);
        assert_ne!(
            merkle_root(&data, 64, HashKind::Sha256),
            merkle_root(&data, 64, HashKind::Blake3)
        );
    }

    #[test]
    fn one_leaf_root_is_the_leaf_hash() {
        let data: Vec<u8> = (0..32).collect();
        for kind in KINDS {
            assert_eq!(
                merkle_root(&data, 1, kind),
                hash_leaf(&data, kind),
                "{kind}"
            );
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        // Use a non-trivial size: 1024 leaves × 64 B = 64 KB.
        let n_leaves = 1024;
        let leaf_size = 64;
        let mut data = vec![0u8; n_leaves * leaf_size];
        // Fill with a deterministic pattern.
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u8;
        }
        for kind in KINDS {
            let par = merkle_tree(&data, n_leaves, kind);
            let seq = merkle_tree_sequential(&data, n_leaves, kind);
            assert_eq!(par, seq, "{kind}");
        }
    }

    /// Leaf sizes chosen to hit every SHA-256 tail shape in the 4-way
    /// interleaved path: rem = 0 (block-aligned), rem < 56 (one tail block),
    /// and rem ≥ 56 (two tail blocks). Also a non-multiple-of-4 leaf count
    /// for the remainder fallback, and — for BLAKE3 — leaf sizes either side
    /// of its 1 KiB chunk boundary, where its internal chunk tree kicks in.
    #[test]
    fn parallel_matches_sequential_tail_shapes() {
        for (n_leaves, leaf_size) in [
            (64, 1024),
            (64, 1025),
            (64, 2048),
            (64, 100),
            (64, 60),
            (64, 56),
            (2, 48),
            (16, 1),
        ] {
            let mut data = vec![0u8; n_leaves * leaf_size];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(0x6C8E944D)) & 0xff) as u8;
            }
            for kind in KINDS {
                let par = merkle_tree(&data, n_leaves, kind);
                let seq = merkle_tree_sequential(&data, n_leaves, kind);
                assert_eq!(par, seq, "{kind} n_leaves={n_leaves} leaf_size={leaf_size}");
            }
        }
    }

    #[test]
    fn root_changes_when_any_leaf_changes() {
        let n_leaves = 64;
        let leaf_size = 32;
        let mut data = vec![0u8; n_leaves * leaf_size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31);
        }
        for kind in KINDS {
            let r0 = merkle_root(&data, n_leaves, kind);
            // Flip one bit deep in the buffer.
            data[n_leaves * leaf_size - 1] ^= 0x01;
            let r1 = merkle_root(&data, n_leaves, kind);
            assert_ne!(r0, r1, "{kind}: single-bit change should change the root");
            data[n_leaves * leaf_size - 1] ^= 0x01;
        }
    }

    #[test]
    fn power_of_two_assertion() {
        let data = vec![0u8; 64];
        // Should not panic for power-of-two leaf counts.
        for kind in KINDS {
            let _ = merkle_root(&data, 1, kind);
            let _ = merkle_root(&data, 2, kind);
            let _ = merkle_root(&data, 4, kind);
            let _ = merkle_root(&data, 8, kind);
        }
    }

    #[test]
    #[should_panic(expected = "num_leaves must be power of 2")]
    fn rejects_non_power_of_two() {
        let data = vec![0u8; 30];
        let _ = merkle_root(&data, 3, HashKind::Sha256);
    }

    #[test]
    fn merkle_proof_roundtrips_at_every_leaf() {
        let n_leaves = 16;
        let leaf_size = 8;
        let mut data = vec![0u8; n_leaves * leaf_size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u8;
        }
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            for i in 0..n_leaves {
                let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                let proof = merkle_proof(&tree, n_leaves, i);
                assert_eq!(proof.len(), 4); // log2(16) = 4
                assert!(
                    verify_merkle_proof(&root, &leaf_hash, i, &proof, kind),
                    "{kind}: verify failed at i={i}"
                );
            }
        }
    }

    /// A proof built under one hash must not verify under the other — the
    /// hash choice is part of what the root commits to.
    #[test]
    fn merkle_proof_rejects_the_other_hash() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 77);
        for kind in KINDS {
            let other = match kind {
                HashKind::Sha256 => HashKind::Blake3,
                HashKind::Blake3 => HashKind::Sha256,
            };
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();
            let leaf_hash = hash_leaf(&data[0..leaf_size], kind);
            let proof = merkle_proof(&tree, n_leaves, 0);

            assert!(verify_merkle_proof(&root, &leaf_hash, 0, &proof, kind));
            assert!(
                !verify_merkle_proof(&root, &leaf_hash, 0, &proof, other),
                "{kind} proof must not verify as {other}"
            );
        }
    }

    #[test]
    fn merkle_proof_rejects_wrong_index() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let leaf_hash = hash_leaf(&data[0..leaf_size], kind);
            let proof = merkle_proof(&tree, n_leaves, 0);

            // Same proof, but claim it's for index 1 → should fail (different
            // sibling structure).
            assert!(
                !verify_merkle_proof(&root, &leaf_hash, 1, &proof, kind),
                "{kind}"
            );
        }
    }

    #[test]
    fn merkle_proof_rejects_tampered_path() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let leaf_hash = hash_leaf(&data[0..leaf_size], kind);
            let mut proof = merkle_proof(&tree, n_leaves, 0);
            // Flip a byte in the first sibling.
            proof[0][0] ^= 1;
            assert!(
                !verify_merkle_proof(&root, &leaf_hash, 0, &proof, kind),
                "{kind}"
            );
        }
    }

    fn random_data(n_leaves: usize, leaf_size: usize, seed: u64) -> Vec<u8> {
        let mut data = vec![0u8; n_leaves * leaf_size];
        let mut z = seed;
        for b in data.iter_mut() {
            z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *b = ((z >> 33) & 0xff) as u8;
        }
        data
    }

    #[test]
    fn multi_proof_single_position_matches_single_proof() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 42);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            for i in 0..n_leaves {
                let multi = merkle_multi_proof(&tree, n_leaves, &[i]);
                let single = merkle_proof(&tree, n_leaves, i);
                assert_eq!(
                    multi, single,
                    "{kind}: multi-proof of [{i}] must equal single proof"
                );

                let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                assert!(verify_merkle_multi_proof(
                    &root,
                    n_leaves,
                    &[i],
                    &[leaf_hash],
                    &multi,
                    kind
                ));
            }
        }
    }

    #[test]
    fn multi_proof_sibling_pair_emits_no_hashes_at_leaf_level() {
        // Sibling pair (0,1) at the leaf level shares its parent → no leaf-level
        // sibling is needed; one sibling per remaining level.
        let n_leaves = 8;
        let leaf_size = 4;
        let data = random_data(n_leaves, leaf_size, 7);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let multi = merkle_multi_proof(&tree, n_leaves, &[0, 1]);
            assert_eq!(
                multi.len(),
                2,
                "sibling pair at leaves saves the leaf-level hash"
            );

            let leaves: Vec<Hash> = [0usize, 1]
                .iter()
                .map(|&i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind))
                .collect();
            assert!(verify_merkle_multi_proof(
                &root,
                n_leaves,
                &[0, 1],
                &leaves,
                &multi,
                kind
            ));
        }
    }

    #[test]
    fn multi_proof_full_query_set_is_root_only() {
        // Every leaf queried → the verifier already knows everything, so the
        // multi-proof should be empty.
        let n_leaves = 16;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 99);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions: Vec<usize> = (0..n_leaves).collect();
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            assert!(
                multi.is_empty(),
                "full-set multi-proof should have zero hashes"
            );

            let leaves: Vec<Hash> = (0..n_leaves)
                .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind))
                .collect();
            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
        }
    }

    #[test]
    fn multi_proof_random_subsets_roundtrip() {
        let n_leaves = 64;
        let leaf_size = 16;
        let data = random_data(n_leaves, leaf_size, 2024);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let all_leaves: Vec<Hash> = (0..n_leaves)
                .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind))
                .collect();

            let subsets: &[&[usize]] = &[
                &[0],
                &[63],
                &[0, 63],
                &[3, 17, 41],
                &[10, 11, 12, 13],
                &[0, 1, 2, 3, 60, 61, 62, 63],
                &[5, 5, 5, 17, 17],
                &[0, 8, 16, 24, 32, 40, 48, 56],
            ];
            for positions in subsets {
                let multi = merkle_multi_proof(&tree, n_leaves, positions);

                let mut sorted: Vec<usize> = positions.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                let leaves: Vec<Hash> = sorted.iter().map(|&p| all_leaves[p]).collect();

                assert!(
                    verify_merkle_multi_proof(&root, n_leaves, &sorted, &leaves, &multi, kind),
                    "{kind}: roundtrip failed for positions={positions:?}"
                );

                let log_n = n_leaves.trailing_zeros() as usize;
                assert!(
                    multi.len() <= sorted.len() * log_n,
                    "multi-proof can't exceed sum of independent paths"
                );
            }
        }
    }

    #[test]
    fn multi_proof_rejects_wrong_leaf() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 1);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![3usize, 7, 19, 28];
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let mut leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
            leaves[1][0] ^= 1;
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
        }
    }

    #[test]
    fn multi_proof_rejects_tampered_proof_hash() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 2);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![1usize, 14, 27];
            let mut multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
            multi[0][0] ^= 1;
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
        }
    }

    #[test]
    fn multi_proof_rejects_extra_or_missing_hashes() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 3);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![2usize, 11];
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            let mut extra = multi.clone();
            extra.push([0xaa; 32]);
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &extra, kind
            ));

            let mut short = multi.clone();
            short.pop();
            assert!(!verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &short, kind
            ));
        }
    }

    #[test]
    fn multi_proof_rejects_unsorted_positions() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 5);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions = vec![2usize, 11];
            let multi = merkle_multi_proof(&tree, n_leaves, &positions);
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            let unsorted = vec![11usize, 2];
            let unsorted_leaves = vec![leaves[1], leaves[0]];
            assert!(!verify_merkle_multi_proof(
                &root,
                n_leaves,
                &unsorted,
                &unsorted_leaves,
                &multi,
                kind
            ));
        }
    }

    #[test]
    fn multi_proof_beats_independent_paths_at_scale() {
        let n_leaves = 1024;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 4096);
        let log_n = n_leaves.trailing_zeros() as usize;
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();

            let positions_raw: Vec<usize> = (0..100)
                .map(|i| {
                    let mut z = (i as u64).wrapping_mul(0xDEAD_BEEF_F0F0_F0F0);
                    z ^= z >> 27;
                    (z as usize) & (n_leaves - 1)
                })
                .collect();
            let multi = merkle_multi_proof(&tree, n_leaves, &positions_raw);

            let mut positions = positions_raw.clone();
            positions.sort_unstable();
            positions.dedup();
            let leaves: Vec<Hash> = positions
                .iter()
                .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size], kind))
                .collect();

            assert!(verify_merkle_multi_proof(
                &root, n_leaves, &positions, &leaves, &multi, kind
            ));
            assert!(
                multi.len() < positions.len() * log_n,
                "multi-proof should beat independent paths: got {} vs {} × {}",
                multi.len(),
                positions.len(),
                log_n
            );
        }
    }
}
