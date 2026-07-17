//! Binary Merkle tree with SHA-256, using four-way hardware SHA interleaving
//! on supported ARM and x86-64 targets.
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
//! Hash uses the [`sha2`] crate. On aarch64 with the `sha2` target feature
//! (set implicitly by `target-cpu=native` on M-series), the crate uses
//! `sha256h`/`sha256h2`/`sha256su0`/`sha256su1` ARM crypto extension
//! instructions; this is detected at runtime by [`cpufeatures`].
//!
//! No domain separation between leaf and internal hashes — this is a
//! micro-benchmark module, not production code. A production PCS commit
//! should prepend `0x00`/`0x01` (or equivalent) to distinguish the two
//! pre-images and avoid second-preimage attacks via interpretation collision.

use rayon::prelude::*;
use sha2::{Digest, Sha256};

pub type Hash = [u8; 32];

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

/// Global SHA-256 call/compression counters, enabled with
/// `--features hash-count` (e.g. by `benches/verifier_hash_count.rs`).
/// Relaxed atomics — exact totals, no ordering guarantees across threads.
#[cfg(feature = "hash-count")]
pub mod hash_count {
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

    pub fn reset() {
        LEAF_CALLS.store(0, Relaxed);
        LEAF_COMPRESSIONS.store(0, Relaxed);
        PAIR_CALLS.store(0, Relaxed);
    }

    /// (leaf_calls, leaf_compressions, pair_calls). Each pair hash is
    /// 2 compressions (64 B payload + padding block).
    pub fn snapshot() -> (u64, u64, u64) {
        (
            LEAF_CALLS.load(Relaxed),
            LEAF_COMPRESSIONS.load(Relaxed),
            PAIR_CALLS.load(Relaxed),
        )
    }
}

/// Hash one leaf of arbitrary byte length.
#[inline]
pub fn hash_leaf(data: &[u8]) -> Hash {
    #[cfg(feature = "hash-count")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        hash_count::LEAF_CALLS.fetch_add(1, Relaxed);
        hash_count::LEAF_COMPRESSIONS.fetch_add(hash_count::sha256_blocks(data.len()), Relaxed);
    }
    Sha256::digest(data).into()
}

/// Hash a pair of children into a parent node (64 B → 32 B).
#[inline]
pub fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    #[cfg(feature = "hash-count")]
    hash_count::PAIR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut h = Sha256::new();
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// Compute the Merkle root of `data` split into `num_leaves` equal-sized leaves.
///
/// Multi-threaded via rayon. `num_leaves` must be a power of two and divide
/// `data.len()`. Returns the 32-byte root. The intermediate tree is allocated
/// and dropped; if you need it for path opening, use [`merkle_tree`] instead.
pub fn merkle_root(data: &[u8], num_leaves: usize) -> Hash {
    let tree = merkle_tree(data, num_leaves);
    tree[tree.len() - 1]
}

/// Compute the full Merkle tree (flat layout, see module docs) for `data`
/// split into `num_leaves` equal-sized leaves.
pub fn merkle_tree(data: &[u8], num_leaves: usize) -> Vec<Hash> {
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

    // 1. Leaves — fully parallel; 4-way interleaved SHA where available.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "sha2"),
        all(target_arch = "x86_64", target_feature = "sha")
    ))]
    {
        tree[..num_leaves]
            .par_chunks_mut(4)
            .zip(data.par_chunks(4 * leaf_size))
            .for_each(|(outs, leaves)| {
                if outs.len() == 4 {
                    #[cfg(feature = "hash-count")]
                    {
                        use std::sync::atomic::Ordering::Relaxed;
                        hash_count::LEAF_CALLS.fetch_add(4, Relaxed);
                        hash_count::LEAF_COMPRESSIONS
                            .fetch_add(4 * hash_count::sha256_blocks(leaf_size), Relaxed);
                    }
                    sha256x4::hash4_equal_len(
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
                        *out = hash_leaf(leaf);
                    }
                }
            });
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "sha2"),
        all(target_arch = "x86_64", target_feature = "sha")
    )))]
    {
        tree[..num_leaves]
            .par_iter_mut()
            .zip(data.par_chunks(leaf_size))
            .for_each(|(out, leaf)| *out = hash_leaf(leaf));
    }

    // 2. Internal levels — parallel within a level, sequential across levels.
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        // Split the buffer at the end of the current level so we get two
        // non-overlapping mutable slices: `read` (input) and `write` (output).
        let (read, rest) = tree[read_start..].split_at_mut(read_len);
        let write = &mut rest[..next_len];

        // 4 parents at a time = 8 contiguous children = 256 contiguous bytes;
        // each parent hashes its 64-byte child pair, interleaved 4-way.
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "sha2"),
            all(target_arch = "x86_64", target_feature = "sha")
        ))]
        {
            let read_bytes: &[u8] =
                unsafe { core::slice::from_raw_parts(read.as_ptr() as *const u8, read.len() * 32) };
            let hash_quad = |outs: &mut [Hash], children: &[u8]| {
                if outs.len() == 4 {
                    #[cfg(feature = "hash-count")]
                    hash_count::PAIR_CALLS.fetch_add(4, std::sync::atomic::Ordering::Relaxed);
                    sha256x4::hash4_equal_len(
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
                        *out = hash_pair(l, r);
                    }
                }
            };
            // Small upper levels can't fill the cores (≤ SERIAL_LEVEL_NODES / 4
            // SHA-x4 tasks), so a rayon dispatch per level costs more than the
            // hashing itself (~3× at the top of a 2^18 tree). Hash them serially
            // — still 4-way SIMD — and only fan out the wide lower levels.
            const SERIAL_LEVEL_NODES: usize = 1024;
            if write.len() <= SERIAL_LEVEL_NODES {
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
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "sha2"),
            all(target_arch = "x86_64", target_feature = "sha")
        )))]
        {
            write
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out)| *out = hash_pair(&read[2 * i], &read[2 * i + 1]));
        }

        read_start += read_len;
        read_len = next_len;
    }

    tree
}

/// Sequential (single-threaded) version of [`merkle_tree`]. Used for
/// benchmark comparison and as the test oracle.
pub fn merkle_tree_sequential(data: &[u8], num_leaves: usize) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(data.len() % num_leaves, 0);

    let leaf_size = data.len() / num_leaves;
    let total_nodes = 2 * num_leaves - 1;
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes);

    for (i, leaf) in data.chunks(leaf_size).enumerate() {
        tree[i] = hash_leaf(leaf);
    }
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        for i in 0..next_len {
            let left = tree[read_start + 2 * i];
            let right = tree[read_start + 2 * i + 1];
            tree[read_start + read_len + i] = hash_pair(&left, &right);
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
pub fn verify_merkle_proof(root: &Hash, leaf_hash: &Hash, index: usize, proof: &[Hash]) -> bool {
    let mut acc = *leaf_hash;
    let mut idx = index;
    for sibling in proof {
        // If idx is even, our node is the LEFT child; sibling is on the RIGHT.
        let (left, right) = if idx & 1 == 0 {
            (acc, *sibling)
        } else {
            (*sibling, acc)
        };
        acc = hash_pair(&left, &right);
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
            next.push((p >> 1, hash_pair(&left, &right)));
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

    #[test]
    fn two_leaves_matches_hand_computation() {
        // Two 8-byte leaves: [0,1,2,3,4,5,6,7] and [8,9,10,11,12,13,14,15].
        let data: Vec<u8> = (0..16).collect();
        let tree = merkle_tree(&data, 2);
        assert_eq!(tree.len(), 3); // 2 leaves + 1 root

        let h0 = hash_leaf(&data[0..8]);
        let h1 = hash_leaf(&data[8..16]);
        let root = hash_pair(&h0, &h1);

        assert_eq!(tree[0], h0);
        assert_eq!(tree[1], h1);
        assert_eq!(tree[2], root);
    }

    #[test]
    fn one_leaf_root_is_the_leaf_hash() {
        let data: Vec<u8> = (0..32).collect();
        let root = merkle_root(&data, 1);
        assert_eq!(root, hash_leaf(&data));
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
        let par = merkle_tree(&data, n_leaves);
        let seq = merkle_tree_sequential(&data, n_leaves);
        assert_eq!(par, seq);
    }

    /// Leaf sizes chosen to hit every SHA-256 tail shape in the 4-way
    /// interleaved path: rem = 0 (block-aligned), rem < 56 (one tail block),
    /// and rem ≥ 56 (two tail blocks). Also a non-multiple-of-4 leaf count
    /// for the remainder fallback.
    #[test]
    fn parallel_matches_sequential_tail_shapes() {
        for (n_leaves, leaf_size) in [(64, 1024), (64, 100), (64, 60), (64, 56), (2, 48), (16, 1)] {
            let mut data = vec![0u8; n_leaves * leaf_size];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(0x6C8E944D)) & 0xff) as u8;
            }
            let par = merkle_tree(&data, n_leaves);
            let seq = merkle_tree_sequential(&data, n_leaves);
            assert_eq!(par, seq, "n_leaves={n_leaves} leaf_size={leaf_size}");
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
        let r0 = merkle_root(&data, n_leaves);
        // Flip one bit deep in the buffer.
        data[n_leaves * leaf_size - 1] ^= 0x01;
        let r1 = merkle_root(&data, n_leaves);
        assert_ne!(r0, r1, "single-bit change should change the root");
    }

    #[test]
    fn power_of_two_assertion() {
        let data = vec![0u8; 64];
        // Should not panic for power-of-two leaf counts.
        let _ = merkle_root(&data, 1);
        let _ = merkle_root(&data, 2);
        let _ = merkle_root(&data, 4);
        let _ = merkle_root(&data, 8);
    }

    #[test]
    #[should_panic(expected = "num_leaves must be power of 2")]
    fn rejects_non_power_of_two() {
        let data = vec![0u8; 30];
        let _ = merkle_root(&data, 3);
    }

    #[test]
    fn merkle_proof_roundtrips_at_every_leaf() {
        let n_leaves = 16;
        let leaf_size = 8;
        let mut data = vec![0u8; n_leaves * leaf_size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u8;
        }
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        for i in 0..n_leaves {
            let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]);
            let proof = merkle_proof(&tree, n_leaves, i);
            assert_eq!(proof.len(), 4); // log2(16) = 4
            assert!(
                verify_merkle_proof(&root, &leaf_hash, i, &proof),
                "verify failed at i={i}"
            );
        }
    }

    #[test]
    fn merkle_proof_rejects_wrong_index() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let leaf_hash = hash_leaf(&data[0..leaf_size]);
        let proof = merkle_proof(&tree, n_leaves, 0);

        // Same proof, but claim it's for index 1 → should fail (different sibling structure).
        assert!(!verify_merkle_proof(&root, &leaf_hash, 1, &proof));
    }

    #[test]
    fn merkle_proof_rejects_tampered_path() {
        let n_leaves = 8;
        let leaf_size = 16;
        let data: Vec<u8> = (0..(n_leaves * leaf_size) as u8).collect();
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let leaf_hash = hash_leaf(&data[0..leaf_size]);
        let mut proof = merkle_proof(&tree, n_leaves, 0);
        // Flip a byte in the first sibling.
        proof[0][0] ^= 1;
        assert!(!verify_merkle_proof(&root, &leaf_hash, 0, &proof));
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
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        for i in 0..n_leaves {
            let multi = merkle_multi_proof(&tree, n_leaves, &[i]);
            let single = merkle_proof(&tree, n_leaves, i);
            assert_eq!(
                multi, single,
                "multi-proof of [{i}] must equal single proof"
            );

            let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]);
            assert!(verify_merkle_multi_proof(
                &root,
                n_leaves,
                &[i],
                &[leaf_hash],
                &multi
            ));
        }
    }

    #[test]
    fn multi_proof_sibling_pair_emits_no_hashes_at_leaf_level() {
        // Sibling pair (0,1) at the leaf level shares its parent → no leaf-level
        // sibling is needed; one sibling per remaining level.
        let n_leaves = 8;
        let leaf_size = 4;
        let data = random_data(n_leaves, leaf_size, 7);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let multi = merkle_multi_proof(&tree, n_leaves, &[0, 1]);
        assert_eq!(
            multi.len(),
            2,
            "sibling pair at leaves saves the leaf-level hash"
        );

        let leaves: Vec<Hash> = [0usize, 1]
            .iter()
            .map(|&i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]))
            .collect();
        assert!(verify_merkle_multi_proof(
            &root,
            n_leaves,
            &[0, 1],
            &leaves,
            &multi
        ));
    }

    #[test]
    fn multi_proof_full_query_set_is_root_only() {
        // Every leaf queried → the verifier already knows everything, so the
        // multi-proof should be empty.
        let n_leaves = 16;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 99);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions: Vec<usize> = (0..n_leaves).collect();
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        assert!(
            multi.is_empty(),
            "full-set multi-proof should have zero hashes"
        );

        let leaves: Vec<Hash> = (0..n_leaves)
            .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]))
            .collect();
        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
    }

    #[test]
    fn multi_proof_random_subsets_roundtrip() {
        let n_leaves = 64;
        let leaf_size = 16;
        let data = random_data(n_leaves, leaf_size, 2024);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let all_leaves: Vec<Hash> = (0..n_leaves)
            .map(|i| hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size]))
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
                verify_merkle_multi_proof(&root, n_leaves, &sorted, &leaves, &multi),
                "roundtrip failed for positions={positions:?}"
            );

            let log_n = n_leaves.trailing_zeros() as usize;
            assert!(
                multi.len() <= sorted.len() * log_n,
                "multi-proof can't exceed sum of independent paths"
            );
        }
    }

    #[test]
    fn multi_proof_rejects_wrong_leaf() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 1);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![3usize, 7, 19, 28];
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let mut leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
        leaves[1][0] ^= 1;
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
    }

    #[test]
    fn multi_proof_rejects_tampered_proof_hash() {
        let (n_leaves, leaf_size) = (32, 8);
        let data = random_data(n_leaves, leaf_size, 2);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![1usize, 14, 27];
        let mut multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
        multi[0][0] ^= 1;
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
        ));
    }

    #[test]
    fn multi_proof_rejects_extra_or_missing_hashes() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 3);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![2usize, 11];
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        let mut extra = multi.clone();
        extra.push([0xaa; 32]);
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &extra
        ));

        let mut short = multi.clone();
        short.pop();
        assert!(!verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &short
        ));
    }

    #[test]
    fn multi_proof_rejects_unsorted_positions() {
        let (n_leaves, leaf_size) = (16, 8);
        let data = random_data(n_leaves, leaf_size, 5);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();

        let positions = vec![2usize, 11];
        let multi = merkle_multi_proof(&tree, n_leaves, &positions);
        let leaves: Vec<Hash> = positions
            .iter()
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        let unsorted = vec![11usize, 2];
        let unsorted_leaves = vec![leaves[1], leaves[0]];
        assert!(!verify_merkle_multi_proof(
            &root,
            n_leaves,
            &unsorted,
            &unsorted_leaves,
            &multi
        ));
    }

    #[test]
    fn multi_proof_beats_independent_paths_at_scale() {
        let n_leaves = 1024;
        let leaf_size = 8;
        let data = random_data(n_leaves, leaf_size, 4096);
        let tree = merkle_tree(&data, n_leaves);
        let root = *tree.last().unwrap();
        let log_n = n_leaves.trailing_zeros() as usize;

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
            .map(|&p| hash_leaf(&data[p * leaf_size..(p + 1) * leaf_size]))
            .collect();

        assert!(verify_merkle_multi_proof(
            &root, n_leaves, &positions, &leaves, &multi
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
