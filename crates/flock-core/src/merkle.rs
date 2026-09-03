//! Merkle commitments.

#[cfg(feature = "hash-count")]
pub use flock_merkle::hash_count;
pub use flock_merkle::{
    Blake3MerkleHash, Hash, HashKind, MerkleHash, Sha256MerkleHash, cap_depth, cap_layer,
    hash_leaf, hash_pair, merkle_proof, merkle_proof_capped, merkle_root, merkle_root_with,
    merkle_tree, merkle_tree_sequential, merkle_tree_sequential_with, merkle_tree_with,
    verify_merkle_proof, verify_merkle_proof_capped, verify_merkle_proof_capped_with,
    verify_merkle_proof_with,
};
