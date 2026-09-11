//! Generic binary Merkle trees with flat storage.
//!
//! [`MerkleHash`] selects the hash implementation. [`HashKind`] supports runtime selection.
//! BLAKE3 separates leaf and parent inputs. The current SHA-256 format does not.

use core::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::OnceLock;

use blake3::{
    Hasher, IncrementCounter,
    hazmat::{HasherExt, Mode, merge_subtrees_non_root},
    platform::Platform,
};
use flock_hash::Digest as HashDigest;
pub use flock_hash::{BLAKE3_IV, HashKind};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSlice,
    ParallelSliceMut,
};
use sha2::{Digest, Sha256};
#[cfg(feature = "hash-count")]
use {
    crate::hashing::hash_count::{LEAF_CALLS, LEAF_COMPRESSIONS, PAIR_CALLS, blocks},
    std::sync::atomic::Ordering::Relaxed,
};

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(target_arch = "x86_64", target_feature = "sha")
))]
use crate::hashing::sha256x4::hash4_equal_len;
pub type Hash = HashDigest;

pub trait MerkleHash: Send + Sync + 'static {
    fn hash_leaf(data: &[u8]) -> Hash;

    fn hash_pair(left: &Hash, right: &Hash) -> Hash;

    fn hash_leaves(data: &[u8], leaf_size: usize, output: &mut [Hash]) {
        for (digest, leaf) in output.iter_mut().zip(data.chunks(leaf_size)) {
            *digest = Self::hash_leaf(leaf);
        }
    }

    fn hash_pairs(children: &[Hash], parents: &mut [Hash]) {
        for (parent, pair) in parents.iter_mut().zip(children.as_chunks::<2>().0) {
            *parent = Self::hash_pair(&pair[0], &pair[1]);
        }
    }
}

pub struct Sha256MerkleHash;

pub struct Blake3MerkleHash;

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

/// Eight-message interleaved NEON BLAKE3 compression (two transposed 4-wide
/// states in flight) — fills the pipes the crate's latency-bound 4-wide
/// backend leaves idle. See the module docs for the derivation.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[path = "merkle/blake3_neon.rs"]
mod blake3_neon;

/// Global Merkle hash call/compression counters, enabled with
/// `--features hash-count` (e.g. by `benches/verifier_hash_count.rs`).
/// Relaxed atomics — exact totals, no ordering guarantees across threads.
#[cfg(feature = "hash-count")]
pub mod hash_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    use crate::hashing::HashKind;

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
pub(crate) fn blake3_leaf_chaining_value(data: &[u8]) -> Hash {
    Hasher::new().update(data).finalize_non_root()
}

/// BLAKE3 parent-node chaining value of two children.
#[inline]
pub(crate) fn blake3_parent_chaining_value(left: &Hash, right: &Hash) -> Hash {
    merge_subtrees_non_root(left, right, Mode::Hash)
}

/// Hash one leaf of arbitrary byte length.
#[inline]
pub fn hash_leaf(data: &[u8], kind: HashKind) -> Hash {
    #[cfg(feature = "hash-count")]
    {
        LEAF_CALLS.fetch_add(1, Relaxed);
        LEAF_COMPRESSIONS.fetch_add(blocks(kind, data.len()), Relaxed);
    }
    match kind {
        HashKind::Sha256 => Sha256::digest(data).into(),
        HashKind::Blake3 => blake3_leaf_chaining_value(data),
    }
}

/// Hash a pair of children into a parent node (64 B → 32 B).
#[inline]
pub fn hash_pair(left: &Hash, right: &Hash, kind: HashKind) -> Hash {
    #[cfg(feature = "hash-count")]
    PAIR_CALLS.fetch_add(1, Relaxed);
    match kind {
        HashKind::Sha256 => {
            let mut h = Sha256::new();
            h.update(left);
            h.update(right);
            h.finalize().into()
        }
        HashKind::Blake3 => blake3_parent_chaining_value(left, right),
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
    hash4_equal_len(inputs, outs);
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

/// BLAKE3 domain flags, fixed by the spec. Off aarch64 the leaf batcher has
/// no kernel to hand them to (the crate path sets its own), so they are
/// test-only there.
#[cfg_attr(
    not(all(target_arch = "aarch64", target_feature = "neon")),
    allow(dead_code)
)]
const BLAKE3_CHUNK_START: u8 = 1;
#[cfg_attr(
    not(all(target_arch = "aarch64", target_feature = "neon")),
    allow(dead_code)
)]
const BLAKE3_CHUNK_END: u8 = 2;
const BLAKE3_PARENT: u8 = 4;

/// Cached SIMD platform. `Platform::detect()` is cheap but not free, and the
/// tree build reaches the batched path once per [`BLAKE3_BATCH`] nodes.
fn blake3_platform() -> Platform {
    static PLATFORM: OnceLock<Platform> = OnceLock::new();
    *PLATFORM.get_or_init(Platform::detect)
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
    // Full groups of eight go through the interleaved two×4-wide NEON kernel;
    // the tail (< 8 messages) falls through to the crate's `hash_many`.
    // Byte-identical either way (`blake3_neon8_matches_crate`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let (data, out) = {
        let full = (out.len() / 8) * 8;
        for i in (0..full).step_by(8) {
            // SAFETY: messages i..i+8 are in-bounds `N`-byte rows of `data`
            // (asserted above); `out[i..i+8]` is 256 writable bytes; `Hash`
            // is `[u8; 32]` with no padding.
            unsafe {
                blake3_neon::hash8(
                    data.as_ptr().add(i * N),
                    N,
                    N / 64,
                    64,
                    flags,
                    flags_start,
                    flags_end,
                    out.as_mut_ptr().add(i) as *mut u8,
                );
            }
        }
        (&data[full * N..], &mut out[full..])
    };
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
            unsafe { from_raw_parts_mut(outs.as_mut_ptr() as *mut u8, n * 32) };
        plat.hash_many(
            &inputs[..n],
            &BLAKE3_IV,
            0,
            IncrementCounter::No,
            flags,
            flags_start,
            flags_end,
            out_bytes,
        );
    }
}

/// Batched BLAKE3 leaves: `out.len()` messages of `leaf_size` bytes, laid out
/// contiguously in `data`. Equivalent to [`blake3_leaf_chaining_value`] per leaf, for
/// ANY single-chunk leaf size (`1..=1024`): eight leaves per NEON call, the
/// last block carrying its true length. The integer-lane commit's leaves
/// are `lanes × 16` bytes — 736 at the BLAKE3 union's m=32 (46 lanes) — and
/// the old power-of-two-only dispatch sent every one of them to the generic
/// per-leaf hasher at half the hash roofline. Tail leaves (fewer than
/// eight) take the generic path, which computes the same chunk CV.
pub(crate) fn blake3_hash_many_leaves(data: &[u8], leaf_size: usize, out: &mut [Hash]) -> bool {
    if !blake3_leaf_size_is_batchable(leaf_size) {
        return false;
    }
    debug_assert_eq!(data.len(), out.len() * leaf_size);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let (data, out) = {
        let n_blocks = leaf_size.div_ceil(64);
        let last = leaf_size - 64 * (n_blocks - 1);
        let full = (out.len() / 8) * 8;
        for i in (0..full).step_by(8) {
            // SAFETY: eight whole leaves of `leaf_size` bytes start at
            // `i * leaf_size`; the kernel reads within each leaf only.
            unsafe {
                blake3_neon::hash8(
                    data.as_ptr().add(i * leaf_size),
                    leaf_size,
                    n_blocks,
                    last,
                    0,
                    BLAKE3_CHUNK_START,
                    BLAKE3_CHUNK_END,
                    out.as_mut_ptr().add(i) as *mut u8,
                );
            }
        }
        (&data[full * leaf_size..], &mut out[full..])
    };
    for (o, leaf) in out.iter_mut().zip(data.chunks(leaf_size)) {
        *o = blake3_leaf_chaining_value(leaf);
    }
    true
}

/// Whether [`blake3_hash_many_leaves`] batches this leaf size — any single
/// BLAKE3 chunk. `blake3_batch_dispatch_agrees` holds the two together.
#[inline]
pub(crate) fn blake3_leaf_size_is_batchable(leaf_size: usize) -> bool {
    (1..=1024).contains(&leaf_size)
}

/// Batched BLAKE3 parent nodes: `data` is `out.len()` contiguous 64-byte
/// (left ‖ right) child pairs. Equivalent to [`blake3_parent_chaining_value`] per node.
#[inline]
pub(crate) fn blake3_hash_many_parents(data: &[u8], out: &mut [Hash]) {
    blake3_hash_many::<64>(data, out, BLAKE3_PARENT, 0, 0);
}

/// Hash a run of `out.len()` equal-size leaves from `data` under `kind`.
///
/// The two hashes batch differently, so they take different shapes here:
/// SHA-256's kernel is inherently four-wide, while BLAKE3 wants the widest
/// batch the machine offers. Both are rayon-parallel and both are
/// byte-identical to calling [`hash_leaf`] on each leaf.
/// Serial (no-rayon) leaf hashing over a contiguous run — the commit's
/// leaf-fused NTT calls this INSIDE a deep-pass task (the task set is the
/// parallelism; nested par-iters would just thrash). Batched kernels still
/// engage per 8-leaf group.
pub fn hash_leaves_serial(data: &[u8], leaf_size: usize, out: &mut [Hash], kind: HashKind) {
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
            for (outs, leaves) in out
                .chunks_mut(BLAKE3_GROUP)
                .zip(data.chunks(BLAKE3_GROUP * leaf_size))
            {
                blake3_hash_many_leaves(leaves, leaf_size, outs);
            }
        }
        HashKind::Blake3 => {
            for (o, leaf) in out.iter_mut().zip(data.chunks(leaf_size)) {
                *o = blake3_leaf_chaining_value(leaf);
            }
        }
        HashKind::Sha256 => {
            for (o, leaf) in out.iter_mut().zip(data.chunks(leaf_size)) {
                *o = Sha256::digest(leaf).into();
            }
        }
    }
}

/// Serial BLAKE3 parent-chunk hashing: `read` holds `2·out.len()` child
/// hashes (contiguous left‖right pairs); each pair hashes to one parent.
/// No-rayon sibling of [`hash_pairs_level`]'s BLAKE3 arm, for the commit's
/// leaf pipeline (the helper thread is the parallelism).
pub fn hash_parents_serial(read: &[Hash], out: &mut [Hash]) {
    debug_assert_eq!(read.len(), 2 * out.len());
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::PAIR_CALLS.fetch_add(out.len() as u64, Relaxed);
    }
    let bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
    for (outs, pairs) in out
        .chunks_mut(BLAKE3_GROUP)
        .zip(bytes.chunks(BLAKE3_GROUP * 64))
    {
        blake3_hash_many_parents(pairs, outs);
    }
}

/// Finish a Merkle tree whose leaf level `tree[..num_leaves]` — and, when
/// `prehashed_levels > 0`, the first that many parent levels — are already
/// hashed: runs the remaining internal levels exactly as [`merkle_tree`]
/// does.
pub fn merkle_tree_from_prehashed_level(
    mut tree: Vec<Hash>,
    num_leaves: usize,
    kind: HashKind,
    prehashed_levels: usize,
) -> Vec<Hash> {
    debug_assert_eq!(tree.len(), 2 * num_leaves - 1);
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    for _ in 0..prehashed_levels {
        debug_assert!(read_len > 1);
        read_start += read_len;
        read_len >>= 1;
    }
    while read_len > 1 {
        let next_len = read_len >> 1;
        let (read, rest) = tree[read_start..].split_at_mut(read_len);
        let write = &mut rest[..next_len];
        hash_pairs_level(read, write, kind);
        read_start += read_len;
        read_len = next_len;
    }
    tree
}

fn hash_leaves(data: &[u8], leaf_size: usize, out: &mut [Hash], kind: HashKind) {
    #[cfg(feature = "hash-count")]
    {
        LEAF_CALLS.fetch_add(out.len() as u64, Relaxed);
        LEAF_COMPRESSIONS.fetch_add(out.len() as u64 * blocks(kind, leaf_size), Relaxed);
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
            .for_each(|(o, leaf)| *o = blake3_leaf_chaining_value(leaf)),
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
    PAIR_CALLS.fetch_add(write.len() as u64, Relaxed);
    // SAFETY: `Hash` is `[u8; 32]`, so a slice of `n` hashes is exactly `32n`
    // initialized bytes with no padding.
    let read_bytes: &[u8] = unsafe { from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
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

impl MerkleHash for Sha256MerkleHash {
    #[inline]
    fn hash_leaf(data: &[u8]) -> Hash {
        hash_leaf(data, HashKind::Sha256)
    }

    #[inline]
    fn hash_pair(left: &Hash, right: &Hash) -> Hash {
        hash_pair(left, right, HashKind::Sha256)
    }

    fn hash_leaves(data: &[u8], leaf_size: usize, output: &mut [Hash]) {
        hash_leaves(data, leaf_size, output, HashKind::Sha256);
    }

    fn hash_pairs(children: &[Hash], parents: &mut [Hash]) {
        hash_pairs_level(children, parents, HashKind::Sha256);
    }
}

impl MerkleHash for Blake3MerkleHash {
    #[inline]
    fn hash_leaf(data: &[u8]) -> Hash {
        hash_leaf(data, HashKind::Blake3)
    }

    #[inline]
    fn hash_pair(left: &Hash, right: &Hash) -> Hash {
        hash_pair(left, right, HashKind::Blake3)
    }

    fn hash_leaves(data: &[u8], leaf_size: usize, output: &mut [Hash]) {
        hash_leaves(data, leaf_size, output, HashKind::Blake3);
    }

    fn hash_pairs(children: &[Hash], parents: &mut [Hash]) {
        hash_pairs_level(children, parents, HashKind::Blake3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 8-wide NEON kernel (and its crate-path tail) must be
    /// byte-identical to the blake3 crate's `hash_many` for every message
    /// size the batched paths dispatch on, at counts covering full groups,
    /// tails, and both at once — for leaves, parents, and flag combinations.
    #[test]
    fn blake3_neon8_matches_crate() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as u8
        };
        fn run<const N: usize>(data: &[u8], count: usize, f: u8, fs: u8, fe: u8) -> Vec<Hash> {
            let mut out = vec![[0u8; 32]; count];
            blake3_hash_many::<N>(&data[..count * N], &mut out, f, fs, fe);
            out
        }
        fn reference<const N: usize>(
            data: &[u8],
            count: usize,
            f: u8,
            fs: u8,
            fe: u8,
        ) -> Vec<Hash> {
            let mut out = vec![[0u8; 32]; count];
            for (o, msg) in out.iter_mut().zip(data.chunks(N)) {
                let input: &[u8; N] = msg.try_into().unwrap();
                let mut bytes = [0u8; 32];
                blake3_platform().hash_many(
                    &[input],
                    &BLAKE3_IV,
                    0,
                    blake3::IncrementCounter::No,
                    f,
                    fs,
                    fe,
                    &mut bytes,
                );
                *o = bytes;
            }
            out
        }
        macro_rules! check {
            ($n:literal, $data:expr, $count:expr) => {
                for (f, fs, fe) in [
                    (0, BLAKE3_CHUNK_START, BLAKE3_CHUNK_END), // leaves
                    (BLAKE3_PARENT, 0, 0),                     // parents
                ] {
                    assert_eq!(
                        run::<$n>($data, $count, f, fs, fe),
                        reference::<$n>($data, $count, f, fs, fe),
                        "N={} count={} flags=({},{},{})",
                        $n,
                        $count,
                        f,
                        fs,
                        fe,
                    );
                }
            };
        }
        let data: Vec<u8> = (0..24 * 1024).map(|_| next()).collect();
        for count in [1usize, 3, 7, 8, 9, 15, 16, 17, 24] {
            check!(64, &data, count);
            check!(128, &data, count);
            check!(512, &data, count);
        }
        for count in [1usize, 7, 8, 9, 16] {
            check!(1024, &data, count);
        }
    }
}
