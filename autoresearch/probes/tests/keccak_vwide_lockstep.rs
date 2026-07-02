//! The direct-write V-wide producers must reproduce, at L1′ addresses,
//! exactly the word-transpose of the row-major production witness — and
//! their inline stripes must equal the production drivers' byte-stripes.

use flock_autoresearch_probes::keccak_vwide::build_l1_direct;
use flock_autoresearch_probes::keccak_witness::{U64_PER_BLOCK, random_state};
use flock_autoresearch_probes::layout::Layout;
use flock_autoresearch_probes::{blake3_witness, sha2_witness};
use flock_core::field::F128;
use flock_prover::r1cs_hashes::keccak::{
    K_LOG, State, generate_witness_with_ab_packed_and_lincheck,
};

/// Generic direct-vs-production check for the bit-packed hashes.
fn check_direct<S: Sync>(
    hash: &str,
    k_log: usize,
    inputs_of: impl Fn(usize, u64) -> Vec<S>,
    production: impl Fn(&[S], usize) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
    direct: impl Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64]),
) {
    let u64_per_block = (1usize << k_log) / 64;
    for n_log in [3usize, 5] {
        let n = 1usize << n_log;
        let inputs = inputs_of(n, 0x1234 + n_log as u64);
        let (z_ref, a_ref, b_ref, stripe_ref) = production(&inputs, n_log);

        let total = n * u64_per_block;
        let (mut z, mut a, mut b) = (vec![0u64; total], vec![0u64; total], vec![0u64; total]);
        let mut stripe = vec![0u8; (n / 8) * u64_per_block * 64];
        // Twice: second run checks buffer-reuse correctness.
        for round in 0..2 {
            direct(&inputs, n_log, Some(&mut stripe), &mut z, &mut a, &mut b);
            let l = Layout::new(k_log, n_log);
            for (name, ours, theirs) in [("z", &z, &z_ref), ("a", &a, &a_ref), ("b", &b, &b_ref)] {
                let theirs_u64: Vec<u64> = theirs.iter().flat_map(|w| [w.lo, w.hi]).collect();
                assert_eq!(
                    ours,
                    &l.permute_words_u64_row_to_l1(&theirs_u64),
                    "{hash}: direct {name} != permuted production (n_log={n_log}, round={round})"
                );
            }
            assert_eq!(stripe, stripe_ref, "{hash}: stripe (n_log={n_log})");
        }
    }
}

#[test]
fn sha2_direct_matches_production() {
    check_direct(
        "sha2",
        flock_prover::r1cs_hashes::sha2::K_LOG,
        |n, seed| (0..n as u64).map(|s| sha2_witness::random_input(seed + s)).collect(),
        flock_prover::r1cs_hashes::sha2::generate_witness_with_ab_packed_and_lincheck,
        flock_autoresearch_probes::sha2_vwide::build_l1_direct,
    );
}

#[test]
fn blake3_direct_matches_production() {
    check_direct(
        "blake3",
        flock_prover::r1cs_hashes::blake3::K_LOG,
        |n, seed| (0..n as u64).map(|s| blake3_witness::random_input(seed + s)).collect(),
        flock_prover::r1cs_hashes::blake3::generate_witness_with_ab_packed_and_lincheck,
        flock_autoresearch_probes::blake3_vwide::build_l1_direct,
    );
}

#[test]
fn direct_matches_production_permuted() {
    for n_log in [3usize, 5, 7] {
        let n = 1usize << n_log;
        let states: Vec<State> = (0..n as u64).map(|s| random_state(0xD1EC7 + s)).collect();

        let (z_ref, a_ref, b_ref, stripe_ref) =
            generate_witness_with_ab_packed_and_lincheck(&states, n_log);

        let total = n * U64_PER_BLOCK;
        let (mut z, mut a, mut b) = (vec![0u64; total], vec![0u64; total], vec![0u64; total]);
        let mut stripe = vec![0u8; (n / 8) * U64_PER_BLOCK * 64];
        build_l1_direct(&states, n_log, Some(&mut stripe), &mut z, &mut a, &mut b);

        let l = Layout::new(K_LOG, n_log);
        for (name, ours, theirs) in [("z", &z, &z_ref), ("a", &a, &a_ref), ("b", &b, &b_ref)] {
            let theirs_u64: Vec<u64> = theirs.iter().flat_map(|w| [w.lo, w.hi]).collect();
            assert_eq!(
                ours,
                &l.permute_words_u64_row_to_l1(&theirs_u64),
                "direct {name} != permuted production (n_log={n_log})"
            );
        }
        assert_eq!(stripe, stripe_ref, "direct stripe != production (n_log={n_log})");
    }
}

/// Reuse across calls with different inputs stays correct (useful words are
/// rewritten by assignment; padding words stay zero).
#[test]
fn direct_reuse_across_inputs() {
    let n_log = 4;
    let n = 1usize << n_log;
    let total = n * U64_PER_BLOCK;
    let (mut z, mut a, mut b) = (vec![0u64; total], vec![0u64; total], vec![0u64; total]);
    let mut stripe = vec![0u8; (n / 8) * U64_PER_BLOCK * 64];
    let l = Layout::new(K_LOG, n_log);

    for round in 0..3u64 {
        let states: Vec<State> = (0..n as u64)
            .map(|s| random_state(0xAB * round + s))
            .collect();
        build_l1_direct(&states, n_log, Some(&mut stripe), &mut z, &mut a, &mut b);
        let (z_ref, _, _, stripe_ref) =
            generate_witness_with_ab_packed_and_lincheck(&states, n_log);
        let z_ref_u64: Vec<u64> = z_ref.iter().flat_map(|w| [w.lo, w.hi]).collect();
        assert_eq!(z, l.permute_words_u64_row_to_l1(&z_ref_u64), "round {round}");
        assert_eq!(stripe, stripe_ref, "stripe round {round}");
    }
}
