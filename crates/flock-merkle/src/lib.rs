//! Generic binary Merkle trees with flat storage.
//!
//! [`MerkleHash`] selects the hash implementation. [`HashKind`] supports runtime selection.

use std::{env::var, slice::from_ref};

use flock_parallel::all_core_pool;
use rayon::current_num_threads;

#[cfg(feature = "hash-count")]
pub use crate::hashing::hash_count;
pub use crate::hashing::{
    Blake3MerkleHash, Hash, HashKind, MerkleHash, Sha256MerkleHash, hash_leaf, hash_pair,
};
#[cfg(test)]
use crate::hashing::{
    blake3_hash_many_leaves, blake3_hash_many_parents, blake3_leaf_cv,
    blake3_leaf_size_is_batchable, blake3_parent_cv,
};
mod hashing;

/// Compute the Merkle root of `data` split into `num_leaves` equal-sized leaves.
///
/// Multi-threaded via rayon. `num_leaves` must be a power of two and divide
/// `data.len()`. Returns the 32-byte root. The intermediate tree is allocated
/// and dropped; if you need it for path opening, use [`merkle_tree`] instead.
pub fn merkle_root(data: &[u8], num_leaves: usize, kind: HashKind) -> Hash {
    match kind {
        HashKind::Sha256 => merkle_root_with::<Sha256MerkleHash>(data, num_leaves),
        HashKind::Blake3 => merkle_root_with::<Blake3MerkleHash>(data, num_leaves),
    }
}

pub fn merkle_root_with<H: MerkleHash>(data: &[u8], num_leaves: usize) -> Hash {
    let tree = merkle_tree_with::<H>(data, num_leaves);
    tree[tree.len() - 1]
}

/// Data-size threshold for the all-core hop. Below this the tree builds in
/// well under a millisecond and the pool switch + E-core straggle risk at the
/// per-level barriers isn't worth it; above it the leaf level dominates
/// (~90% of SHA compressions at 1 KB leaves) and is a flat parallel-for that
/// drains cleanly around slow cores — and the E-cores have the SHA-256
/// crypto extensions too.
const MERKLE_ALLCORE_MIN_BYTES: usize = 8 << 20;

// `MERKLE_PCORES_ONLY=1` in the environment keeps [`merkle_tree`] on the
// caller's (P-core) pool even for large trees (production kill-switch). Pool
// choice cannot change output bits — every node is written deterministically.
fn merkle_use_all_cores(data_len: usize) -> bool {
    data_len >= MERKLE_ALLCORE_MIN_BYTES
        && var("MERKLE_PCORES_ONLY").is_err()
        && all_core_pool().current_num_threads() > current_num_threads()
}

/// Compute the full Merkle tree (flat layout, see module docs) for `data`
/// split into `num_leaves` equal-sized leaves, hashed under `kind`. Large
/// trees run on the all-core (P+E) pool (see [`merkle_use_all_cores`]);
/// output is identical either way.
pub fn merkle_tree(data: &[u8], num_leaves: usize, kind: HashKind) -> Vec<Hash> {
    match kind {
        HashKind::Sha256 => merkle_tree_with::<Sha256MerkleHash>(data, num_leaves),
        HashKind::Blake3 => merkle_tree_with::<Blake3MerkleHash>(data, num_leaves),
    }
}

pub fn merkle_tree_with<H: MerkleHash>(data: &[u8], num_leaves: usize) -> Vec<Hash> {
    if merkle_use_all_cores(data.len()) {
        all_core_pool().install(|| merkle_tree_impl::<H>(data, num_leaves))
    } else {
        merkle_tree_impl::<H>(data, num_leaves)
    }
}

#[allow(clippy::uninit_vec)]
fn alloc_hash_vec(len: usize) -> Vec<Hash> {
    let mut hashes = Vec::with_capacity(len);
    // SAFETY: Tree construction writes each hash before it reads that hash.
    unsafe {
        hashes.set_len(len);
    }
    hashes
}

fn merkle_tree_impl<H: MerkleHash>(data: &[u8], num_leaves: usize) -> Vec<Hash> {
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
    let mut tree = alloc_hash_vec(total_nodes);

    // 1. Leaves — fully parallel, SIMD-batched across leaves where possible.
    H::hash_leaves(data, leaf_size, &mut tree[..num_leaves]);

    // 2. Internal levels — parallel within a level, sequential across levels.
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        // Split the buffer at the end of the current level so we get two
        // non-overlapping mutable slices: `read` (input) and `write` (output).
        let (read, rest) = tree[read_start..].split_at_mut(read_len);
        let write = &mut rest[..next_len];

        H::hash_pairs(read, write);

        read_start += read_len;
        read_len = next_len;
    }

    tree
}

/// Sequential (single-threaded) version of [`merkle_tree`]. Used for
/// benchmark comparison and as the test oracle.
pub fn merkle_tree_sequential(data: &[u8], num_leaves: usize, kind: HashKind) -> Vec<Hash> {
    match kind {
        HashKind::Sha256 => merkle_tree_sequential_with::<Sha256MerkleHash>(data, num_leaves),
        HashKind::Blake3 => merkle_tree_sequential_with::<Blake3MerkleHash>(data, num_leaves),
    }
}

pub fn merkle_tree_sequential_with<H: MerkleHash>(data: &[u8], num_leaves: usize) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(data.len() % num_leaves, 0);

    let leaf_size = data.len() / num_leaves;
    let total_nodes = 2 * num_leaves - 1;
    let mut tree = alloc_hash_vec(total_nodes);

    for (i, leaf) in data.chunks(leaf_size).enumerate() {
        tree[i] = H::hash_leaf(leaf);
    }
    let mut read_start = 0usize;
    let mut read_len = num_leaves;
    while read_len > 1 {
        let next_len = read_len >> 1;
        for i in 0..next_len {
            let left = tree[read_start + 2 * i];
            let right = tree[read_start + 2 * i + 1];
            tree[read_start + read_len + i] = H::hash_pair(&left, &right);
        }
        read_start += read_len;
        read_len = next_len;
    }
    tree
}

// ---------------------------------------------------------------------------
// Merkle path opening and verification.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Merkle CAPPING: a tree's commitment is its cap — the 2^c nodes at depth c
// below the root — and an opening authenticates leaf → cap node only
// (d − c siblings). At c = ⌈log2 q⌉ the cap replaces exactly the region
// where q query paths funnel and share, so capped independent paths cost
// about what the shared multi-proof did, with NONE of its data-dependent
// shape: no sorting, no dedup, a duplicate query is just a repeated path.
// c = 0 degenerates to the classic single root; c = d to "the cap IS the
// leaf-hash layer" (empty paths — real for the shallow trees of some
// shipped configs).
// ---------------------------------------------------------------------------

/// Cap depth for `q` queries into a depth-`d` tree: `min(⌈log2 q⌉, d)`.
/// Raising c by one saves `q` path siblings but doubles the cap, and at
/// `c = log2 q` those exactly cancel — the sweet spot. `q ≤ 1` → 0.
pub fn cap_depth(q: usize, d: usize) -> usize {
    if q <= 1 {
        return 0;
    }
    (usize::BITS as usize - (q - 1).leading_zeros() as usize).min(d)
}

/// The cap layer: the `2^c` nodes at depth `c` below the root, as a slice of
/// the flat tree. The flat layout is levels concatenated bottom-up (leaves at
/// `[0, N)`, then `N/2` parents, …), so the level with `L` nodes starts at
/// `2N − 2L`. `c = 0` → `[root]`; `c = d` → the leaf-hash layer.
pub fn cap_layer(tree: &[Hash], num_leaves: usize, c: usize) -> &[Hash] {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert_eq!(tree.len(), 2 * num_leaves - 1);
    let d = num_leaves.trailing_zeros() as usize;
    assert!(c <= d, "cap depth {c} exceeds tree depth {d}");
    let l = 1usize << c;
    &tree[2 * num_leaves - 2 * l..][..l]
}

/// Build a CAPPED opening proof for leaf `index`: the sibling hashes from the
/// leaf level up to (but not including) the cap layer at depth `c` — exactly
/// `log2(num_leaves) − c` hashes. `c = 0` is the classic root-anchored path.
///
/// Verify with [`verify_merkle_proof_capped`].
pub fn merkle_proof_capped(tree: &[Hash], num_leaves: usize, index: usize, c: usize) -> Vec<Hash> {
    assert!(num_leaves.is_power_of_two() && num_leaves > 0);
    assert!(index < num_leaves);
    assert_eq!(tree.len(), 2 * num_leaves - 1);
    let d = num_leaves.trailing_zeros() as usize;
    assert!(c <= d, "cap depth {c} exceeds tree depth {d}");

    let mut proof = Vec::with_capacity(d - c);
    let mut level_start = 0usize;
    let mut level_len = num_leaves;
    let mut idx = index;
    while level_len > (1 << c) {
        let sibling_idx = idx ^ 1;
        proof.push(tree[level_start + sibling_idx]);
        level_start += level_len;
        level_len >>= 1;
        idx >>= 1;
    }
    proof
}

/// Build an opening proof for leaf `index`: the sibling hashes from the leaf
/// level up to (but not including) the root — [`merkle_proof_capped`] at
/// `c = 0`. The returned vector has length `log2(num_leaves)`.
///
/// Verify with [`verify_merkle_proof`].
pub fn merkle_proof(tree: &[Hash], num_leaves: usize, index: usize) -> Vec<Hash> {
    merkle_proof_capped(tree, num_leaves, index, 0)
}

/// Verify a CAPPED Merkle opening: recompute leaf `index`'s cap node from
/// `leaf_hash` and the path, and compare it to `cap[index >> path.len()]`.
/// Self-checking on shape: `cap.len()` a power of two ≤ `num_leaves`, and
/// `path.len()` exactly `log2(num_leaves) − log2(cap.len())` — a wrong-length
/// path can never verify.
pub fn verify_merkle_proof_capped(
    cap: &[Hash],
    num_leaves: usize,
    leaf_hash: &Hash,
    index: usize,
    path: &[Hash],
    kind: HashKind,
) -> bool {
    match kind {
        HashKind::Sha256 => verify_merkle_proof_capped_with::<Sha256MerkleHash>(
            cap, num_leaves, leaf_hash, index, path,
        ),
        HashKind::Blake3 => verify_merkle_proof_capped_with::<Blake3MerkleHash>(
            cap, num_leaves, leaf_hash, index, path,
        ),
    }
}

pub fn verify_merkle_proof_capped_with<H: MerkleHash>(
    cap: &[Hash],
    num_leaves: usize,
    leaf_hash: &Hash,
    index: usize,
    path: &[Hash],
) -> bool {
    if !num_leaves.is_power_of_two()
        || num_leaves == 0
        || !cap.len().is_power_of_two()
        || cap.len() > num_leaves
        || index >= num_leaves
    {
        return false;
    }
    let d = num_leaves.trailing_zeros() as usize;
    let c = cap.len().trailing_zeros() as usize;
    if path.len() != d - c {
        return false;
    }
    let mut acc = *leaf_hash;
    let mut idx = index;
    for sibling in path {
        // If idx is even, our node is the LEFT child; sibling is on the RIGHT.
        let (left, right) = if idx & 1 == 0 {
            (acc, *sibling)
        } else {
            (*sibling, acc)
        };
        acc = H::hash_pair(&left, &right);
        idx >>= 1;
    }
    acc == cap[idx]
}

/// Verify a Merkle opening: recomputes the root from `leaf_hash`, the path,
/// and the leaf index — [`verify_merkle_proof_capped`] against the one-node
/// cap `[root]`. Returns true iff the recomputed root matches `root`.
pub fn verify_merkle_proof(
    root: &Hash,
    leaf_hash: &Hash,
    index: usize,
    proof: &[Hash],
    kind: HashKind,
) -> bool {
    match kind {
        HashKind::Sha256 => {
            verify_merkle_proof_with::<Sha256MerkleHash>(root, leaf_hash, index, proof)
        }
        HashKind::Blake3 => {
            verify_merkle_proof_with::<Blake3MerkleHash>(root, leaf_hash, index, proof)
        }
    }
}

pub fn verify_merkle_proof_with<H: MerkleHash>(
    root: &Hash,
    leaf_hash: &Hash,
    index: usize,
    proof: &[Hash],
) -> bool {
    verify_merkle_proof_capped_with::<H>(
        from_ref(root),
        1usize << proof.len(),
        leaf_hash,
        index,
        proof,
    )
}

#[cfg(test)]
mod tests {
    use blake3::{
        Hasher, hash,
        hazmat::{HasherExt, Mode, merge_subtrees_non_root},
    };
    use sha2::{Digest, Sha256};

    use crate::{
        Blake3MerkleHash, Hash, HashKind, Sha256MerkleHash, blake3_hash_many_leaves,
        blake3_hash_many_parents, blake3_leaf_cv, blake3_leaf_size_is_batchable, blake3_parent_cv,
        cap_depth, cap_layer, hash_leaf, hash_pair, merkle_proof, merkle_proof_capped, merkle_root,
        merkle_tree, merkle_tree_sequential, merkle_tree_with, verify_merkle_proof,
        verify_merkle_proof_capped,
    };

    /// Every structural test runs against both hashes: the tree and path
    /// logic is hash-agnostic, so anything true of one must hold for the
    /// other.
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

    #[test]
    fn generic_entry_points_match_runtime_selection() {
        let data = random_data(64, 64, 17);
        assert_eq!(
            merkle_tree_with::<Sha256MerkleHash>(&data, 64),
            merkle_tree(&data, 64, HashKind::Sha256)
        );
        assert_eq!(
            merkle_tree_with::<Blake3MerkleHash>(&data, 64),
            merkle_tree(&data, 64, HashKind::Blake3)
        );
    }

    /// The primitives must agree with the reference APIs of the underlying
    /// crates — this is what pins the digests to the real hash functions
    /// rather than merely to themselves.
    #[test]
    fn primitives_match_reference_implementations() {
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
            Hasher::new().update(&data).finalize_non_root()
        );
        assert_eq!(
            hash_pair(&l, &r, HashKind::Blake3),
            merge_subtrees_non_root(&l, &r, Mode::Hash)
        );
        // Deliberately NOT `blake3::hash` — that is the root finalization, and
        // interior tree nodes must not be root hashes.
        assert_ne!(
            hash_leaf(&data, HashKind::Blake3),
            *hash(&data).as_bytes(),
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

    /// Capped roundtrip at every leaf, at EVERY cap depth 0..=d, both hashes:
    /// path length is exactly d − c and each leaf verifies against its own
    /// cap node.
    #[test]
    fn capped_proof_roundtrips_at_every_leaf_and_depth() {
        let (n_leaves, leaf_size) = (16usize, 8usize);
        let d = 4usize;
        let data = random_data(n_leaves, leaf_size, 4242);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            for c in 0..=d {
                let cap = cap_layer(&tree, n_leaves, c);
                assert_eq!(cap.len(), 1 << c);
                for i in 0..n_leaves {
                    let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                    let path = merkle_proof_capped(&tree, n_leaves, i, c);
                    assert_eq!(path.len(), d - c, "{kind}: c={c}");
                    assert!(
                        verify_merkle_proof_capped(cap, n_leaves, &leaf_hash, i, &path, kind),
                        "{kind}: verify failed at i={i}, c={c}"
                    );
                }
            }
        }
    }

    /// c = 0 IS the classic single-root opening: same path bytes, and the
    /// capped verifier against `[root]` agrees with `verify_merkle_proof`.
    #[test]
    fn cap_zero_is_the_classic_opening() {
        let (n_leaves, leaf_size) = (16usize, 8usize);
        let data = random_data(n_leaves, leaf_size, 99);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let root = *tree.last().unwrap();
            assert_eq!(cap_layer(&tree, n_leaves, 0), &[root]);
            for i in 0..n_leaves {
                let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                let capped = merkle_proof_capped(&tree, n_leaves, i, 0);
                assert_eq!(capped, merkle_proof(&tree, n_leaves, i));
                assert!(verify_merkle_proof(&root, &leaf_hash, i, &capped, kind));
            }
        }
    }

    /// Degenerate c = d: the cap IS the leaf-hash layer, paths are empty, and
    /// verification is a straight leaf-hash comparison. A wrong leaf rejects.
    #[test]
    fn cap_at_leaf_depth_is_the_leaf_layer() {
        let (n_leaves, leaf_size) = (16usize, 8usize);
        let d = 4usize;
        let data = random_data(n_leaves, leaf_size, 7);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let cap = cap_layer(&tree, n_leaves, d);
            assert_eq!(cap, &tree[..n_leaves]);
            for i in 0..n_leaves {
                let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                let path = merkle_proof_capped(&tree, n_leaves, i, d);
                assert!(path.is_empty());
                assert!(verify_merkle_proof_capped(
                    cap, n_leaves, &leaf_hash, i, &path, kind
                ));
                let mut wrong = leaf_hash;
                wrong[0] ^= 1;
                assert!(!verify_merkle_proof_capped(
                    cap, n_leaves, &wrong, i, &path, kind
                ));
            }
        }
    }

    /// Tampering ONE cap node breaks exactly the leaves under it and no
    /// others — this pins the `index >> (d − c)` cap-node indexing.
    #[test]
    fn cap_node_tamper_is_local() {
        let (n_leaves, leaf_size) = (16usize, 8usize);
        let (d, c) = (4usize, 2usize);
        let data = random_data(n_leaves, leaf_size, 1234);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let mut cap = cap_layer(&tree, n_leaves, c).to_vec();
            let bad_node = 1usize; // covers leaves 4..8 at d − c = 2
            cap[bad_node][0] ^= 1;
            for i in 0..n_leaves {
                let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
                let path = merkle_proof_capped(&tree, n_leaves, i, c);
                let ok = verify_merkle_proof_capped(&cap, n_leaves, &leaf_hash, i, &path, kind);
                let under_bad = (i >> (d - c)) == bad_node;
                assert_eq!(ok, !under_bad, "{kind}: i={i}");
            }
        }
    }

    /// Wrong index, tampered sibling, and the wrong hash kind all reject on
    /// the capped path — mirrors of the classic-opening tamper tests.
    #[test]
    fn capped_proof_rejects_tampering() {
        let (n_leaves, leaf_size) = (16usize, 8usize);
        let c = 2usize;
        let data = random_data(n_leaves, leaf_size, 555);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let cap = cap_layer(&tree, n_leaves, c);
            let i = 5usize;
            let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
            let path = merkle_proof_capped(&tree, n_leaves, i, c);
            assert!(verify_merkle_proof_capped(
                cap, n_leaves, &leaf_hash, i, &path, kind
            ));
            // Wrong index (same cap node, sibling half).
            assert!(!verify_merkle_proof_capped(
                cap,
                n_leaves,
                &leaf_hash,
                i ^ 1,
                &path,
                kind
            ));
            // Tampered sibling.
            let mut bad = path.clone();
            bad[0][0] ^= 1;
            assert!(!verify_merkle_proof_capped(
                cap, n_leaves, &leaf_hash, i, &bad, kind
            ));
            // The other hash kind.
            let other = match kind {
                HashKind::Sha256 => HashKind::Blake3,
                HashKind::Blake3 => HashKind::Sha256,
                #[allow(unreachable_patterns)]
                _ => continue,
            };
            assert!(!verify_merkle_proof_capped(
                cap, n_leaves, &leaf_hash, i, &path, other
            ));
        }
    }

    /// A path of the wrong LENGTH (±1 sibling) can never verify: the capped
    /// verifier's shape check ties `path.len()` to `log2(num_leaves) −
    /// log2(cap.len())`.
    #[test]
    fn capped_proof_rejects_wrong_length() {
        let (n_leaves, leaf_size) = (16usize, 8usize);
        let c = 2usize;
        let data = random_data(n_leaves, leaf_size, 808);
        for kind in KINDS {
            let tree = merkle_tree(&data, n_leaves, kind);
            let cap = cap_layer(&tree, n_leaves, c);
            let i = 3usize;
            let leaf_hash = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], kind);
            let path = merkle_proof_capped(&tree, n_leaves, i, c);
            let mut short = path.clone();
            short.pop();
            assert!(!verify_merkle_proof_capped(
                cap, n_leaves, &leaf_hash, i, &short, kind
            ));
            let mut long = path.clone();
            long.push([0u8; 32]);
            assert!(!verify_merkle_proof_capped(
                cap, n_leaves, &leaf_hash, i, &long, kind
            ));
        }
    }

    /// cap_depth: ⌈log2 q⌉ clamped to the tree depth; q ≤ 1 → 0.
    #[test]
    fn cap_depth_formula() {
        assert_eq!(cap_depth(0, 10), 0);
        assert_eq!(cap_depth(1, 10), 0);
        assert_eq!(cap_depth(2, 10), 1);
        assert_eq!(cap_depth(3, 10), 2);
        assert_eq!(cap_depth(53, 10), 6);
        assert_eq!(cap_depth(71, 10), 7);
        assert_eq!(cap_depth(106, 10), 7);
        assert_eq!(cap_depth(218, 10), 8);
        assert_eq!(cap_depth(256, 10), 8);
        assert_eq!(cap_depth(257, 10), 9);
        // Clamped: shallow trees cap at the leaf layer.
        assert_eq!(cap_depth(218, 4), 4);
        assert_eq!(cap_depth(131, 8), 8);
    }
}
