//! E3 oracle — the fused L1′ partial fold must be byte-identical to
//! production's stripe-based fold (both the portable fast kernel and the
//! naive scalar reference) on real witness data.

use flock_autoresearch_probes::lincheck_fold::partial_fold_l1;
use flock_autoresearch_probes::{keccak_vwide, keccak_witness, sha2_vwide, sha2_witness};
use flock_core::field::F128;
use flock_core::lincheck::{
    build_eq_table, partial_fold_packed_z, partial_fold_packed_z_fast_padded,
};

struct Rng(u64);
impl Rng {
    fn f128(&mut self) -> F128 {
        let mut next = || {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        F128 { lo: next(), hi: next() }
    }
}

fn check<S>(
    hash: &str,
    k_log: usize,
    useful_bits: usize,
    m: usize,
    inputs: &[S],
    direct: impl Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64]),
) {
    let n_log = m - k_log;
    let n = 1usize << n_log;
    let total_u64 = (1usize << m) / 64;
    let u64_per_block = (1usize << k_log) / 64;

    let (mut z, mut a, mut b) =
        (vec![0u64; total_u64], vec![0u64; total_u64], vec![0u64; total_u64]);
    let mut stripe = vec![0u8; (n / 8) * u64_per_block * 64];
    direct(inputs, n_log, Some(&mut stripe), &mut z, &mut a, &mut b);

    let mut rng = Rng(0xE3 + m as u64);
    let point: Vec<F128> = (0..n_log).map(|_| rng.f128()).collect();
    let eq_outer = build_eq_table(&point);

    let fused = partial_fold_l1(&z, m, k_log, useful_bits, &eq_outer);
    let reference = partial_fold_packed_z_fast_padded(&stripe, m, k_log, useful_bits, &eq_outer);
    assert_eq!(fused, reference, "{hash}: fused L1' fold != stripe fast fold");

    let naive = partial_fold_packed_z(&stripe, m, k_log, &eq_outer);
    assert_eq!(fused, naive, "{hash}: fused L1' fold != naive stripe fold");
}

#[test]
fn keccak_fused_fold_matches_stripe() {
    for m in [19usize, 20, 22] {
        let n = 1usize << (m - flock_prover::r1cs_hashes::keccak::K_LOG);
        let inputs: Vec<_> = (0..n as u64).map(keccak_witness::random_state).collect();
        check(
            "keccak",
            flock_prover::r1cs_hashes::keccak::K_LOG,
            flock_prover::r1cs_hashes::keccak::USEFUL_BITS,
            m,
            &inputs,
            keccak_vwide::build_l1_direct,
        );
    }
}

#[test]
fn sha2_fused_fold_matches_stripe() {
    for m in [18usize, 21] {
        let n = 1usize << (m - flock_prover::r1cs_hashes::sha2::K_LOG);
        let inputs: Vec<_> = (0..n as u64)
            .map(|s| sha2_witness::random_input(0xE3E3 + s))
            .collect();
        check(
            "sha2",
            flock_prover::r1cs_hashes::sha2::K_LOG,
            flock_prover::r1cs_hashes::sha2::USEFUL_BITS,
            m,
            &inputs,
            sha2_vwide::build_l1_direct,
        );
    }
}
