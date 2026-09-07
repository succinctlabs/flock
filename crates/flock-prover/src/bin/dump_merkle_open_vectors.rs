//! Dump a Merkle multi-proof oracle from the **real** flock
//! `merkle::merkle_multi_proof` (`src/merkle.rs`), so the host C++ port
//! (`cuda-ghash/merkle_open.hpp` / `test_merkle_open.cpp`) is validated
//! byte-for-byte against the query-opening logic the recursive prover uses.
//!
//! At each Ligerito level, the
//! prover opens the challenger-sampled query rows of the committed codeword and
//! emits a deduplicated Merkle multi-proof. This dumps a real tree + query
//! positions + the resulting multi-proof; the C++ replays the sibling-collecting
//! walk and must reproduce it.
//!
//! Output (LE) to argv[1] (default merkle_open_vectors.bin), magic "MKOP":
//!   magic u32, num_leaves u32, tree_len u32 (=2*num_leaves-1)
//!   tree[tree_len] : 32 bytes each
//!   n_positions u32, positions[n_positions] : u64 each (unsorted, distinct)
//!   proof_len u32, proof[proof_len] : 32 bytes each
//!
//! Run:  cargo run --release --bin dump_merkle_open_vectors -- cuda-ghash/merkle_open_vectors.bin 14 50

use std::{
    collections::HashSet,
    env,
    fs::File,
    io::{BufWriter, Result, Write},
};

use env::args;
use flock_core::test_rng::Rng;
use flock_hash::HashKind;
use flock_prover::merkle::merkle_tree;

use crate::merkle_octopus::merkle_multi_proof;
// The multi-proof left the live protocol (cap layers replaced it); the CUDA
// oracle pair keeps a frozen copy.
#[path = "dump_common/merkle_octopus.rs"]
mod merkle_octopus;

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "merkle_open_vectors.bin".to_string());
    let log_leaves: usize = args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(14);
    let n_queries: usize = args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let num_leaves = 1usize << log_leaves;
    let leaf_size = 64usize; // bytes per leaf (e.g. num_interleaved * 16); value-irrelevant

    let mut rng = Rng::new(0xC0FFEE);
    let mut data = vec![0u8; num_leaves * leaf_size];
    for chunk in data.chunks_mut(8) {
        let v = rng.next_u64().to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&v[..n]);
    }
    let tree = merkle_tree(&data, num_leaves, HashKind::Sha256);
    assert_eq!(tree.len(), 2 * num_leaves - 1);

    // Distinct query positions, deliberately unsorted (tests the C++ sort/dedup).
    let mut positions: Vec<usize> = Vec::with_capacity(n_queries);
    {
        let mut seen = HashSet::new();
        while positions.len() < n_queries {
            let q = (rng.next_u64() as usize) % num_leaves;
            if seen.insert(q) {
                positions.push(q);
            }
        }
    }
    let proof = merkle_multi_proof(&tree, num_leaves, &positions);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x4D4B_4F50u32.to_le_bytes())?; // "MKOP"
    w.write_all(&(num_leaves as u32).to_le_bytes())?;
    w.write_all(&(tree.len() as u32).to_le_bytes())?;
    for h in &tree {
        w.write_all(h)?;
    }
    w.write_all(&(positions.len() as u32).to_le_bytes())?;
    for &p in &positions {
        w.write_all(&(p as u64).to_le_bytes())?;
    }
    w.write_all(&(proof.len() as u32).to_le_bytes())?;
    for h in &proof {
        w.write_all(h)?;
    }
    w.flush()?;
    eprintln!(
        "wrote merkle-open oracle to {path}: num_leaves={num_leaves} n_queries={n_queries} \
         proof_len={} (vs {} independent paths)",
        proof.len(),
        n_queries * log_leaves
    );
    Ok(())
}
