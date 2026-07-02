//! Keep the probe's copied per-block builders in lockstep with production:
//! for each hash, `build_row_major` must reproduce byte-for-byte the z/a/b of
//! the public `generate_witness_with_ab_packed_and_lincheck`, and the fused
//! L1′ stripe must equal the driver's byte-stripe.

use flock_autoresearch_probes::producer::{
    PerBlock, build_l1_staged_opts_nt, build_row_major, build_row_major_with_stripe,
};
use flock_autoresearch_probes::{blake3_witness, keccak_witness, sha2_witness};
use flock_core::field::F128;

fn check_hash<S: Sync>(
    hash: &str,
    k_log: usize,
    n_log: usize,
    inputs: &[S],
    production: impl Fn(&[S], usize) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
    per_block: &impl PerBlock<S>,
) {
    let n = 1usize << n_log;
    assert_eq!(inputs.len(), n, "provide exactly 2^n_log inputs (no padding)");
    let u64_per_block = (1usize << k_log) / 64;
    let chunks_per_block = u64_per_block / 2;
    let total = n * u64_per_block;

    let (z_ref, a_ref, b_ref, stripe_ref) = production(inputs, n_log);

    // Row-major builder equality.
    let (mut z, mut a, mut b) = (vec![0u64; total], vec![0u64; total], vec![0u64; total]);
    build_row_major(inputs, k_log, n_log, 8, per_block, &mut z, &mut a, &mut b);
    for (name, ours, theirs) in [("z", &z, &z_ref), ("a", &a, &a_ref), ("b", &b, &b_ref)] {
        let theirs_u64: Vec<u64> = theirs.iter().flat_map(|w| [w.lo, w.hi]).collect();
        assert_eq!(ours, &theirs_u64, "{hash}: {name} diverged from production");
    }

    // Row-major fair baseline's fused stripe must equal production's.
    let mut stripe = vec![0u8; (n / 8) * u64_per_block * 64];
    build_row_major_with_stripe(
        inputs,
        k_log,
        n_log,
        8,
        per_block,
        Some(&mut stripe),
        &mut z,
        &mut a,
        &mut b,
    );
    assert_eq!(stripe, stripe_ref, "{hash}: row-major fused stripe diverged");

    // Fused L1' stripe equality (both nt modes).
    for nt in [false, true] {
        build_l1_staged_opts_nt(
            inputs,
            k_log,
            n_log,
            8,
            chunks_per_block,
            Some(&mut stripe),
            nt,
            per_block,
            &mut z,
            &mut a,
            &mut b,
        );
        assert_eq!(stripe, stripe_ref, "{hash}: stripe diverged (nt={nt})");
    }
}

#[test]
fn keccak_matches_public_driver() {
    let n_log = 3;
    let inputs: Vec<_> = (0..1u64 << n_log).map(keccak_witness::random_state).collect();
    check_hash(
        "keccak",
        flock_prover::r1cs_hashes::keccak::K_LOG,
        n_log,
        &inputs,
        flock_prover::r1cs_hashes::keccak::generate_witness_with_ab_packed_and_lincheck,
        &keccak_witness::build_block_witness,
    );
}

#[test]
fn sha2_matches_public_driver() {
    let n_log = 4;
    let inputs: Vec<_> = (0..1u64 << n_log)
        .map(|s| sha2_witness::random_input(0x5A5A + s))
        .collect();
    check_hash(
        "sha2",
        flock_prover::r1cs_hashes::sha2::K_LOG,
        n_log,
        &inputs,
        flock_prover::r1cs_hashes::sha2::generate_witness_with_ab_packed_and_lincheck,
        &sha2_witness::build_block_witness,
    );
}

#[test]
fn blake3_matches_public_driver() {
    let n_log = 4;
    let inputs: Vec<_> = (0..1u64 << n_log)
        .map(|s| blake3_witness::random_input(0xB3B3 + s))
        .collect();
    check_hash(
        "blake3",
        flock_prover::r1cs_hashes::blake3::K_LOG,
        n_log,
        &inputs,
        flock_prover::r1cs_hashes::blake3::generate_witness_with_ab_packed_and_lincheck,
        &blake3_witness::build_block_witness,
    );
}
