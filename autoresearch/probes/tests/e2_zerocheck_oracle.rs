//! E2 oracle — zerocheck on L1′-permuted buffers must produce TRUTHFUL
//! claims: the final a/b/c evals must equal a from-scratch evaluation of the
//! (quirky) MLE of the L1′ buffers at the claim points.
//!
//! Point structure (zerocheck.rs): the low 6 dims are bound by the
//! univariate skip via the φ₈-Lagrange basis (`lagrange_weights_naive`), the
//! rest are standard multilinear:
//!   â(z, x) = Σ_addr a[addr] · ν_{addr&63}(z) · eq(x, addr>>6)
//! a_eval/b_eval at x = mlv_challenges; c_eval at x = r_rest.

use flock_autoresearch_probes::{keccak_vwide, keccak_witness, sha2_vwide, sha2_witness};
use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::zerocheck::{self, PaddingSpec};
use flock_core::zerocheck::multilinear::lagrange_weights_naive;

fn as_bytes(v: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) }
}

/// Quirky MLE eval: low 6 dims via Lagrange weights at `z_skip`, rest via eq.
/// Each u64 word of the buffer covers exactly the 64 skip positions of one
/// rest-index.
fn quirky_eval(buf: &[u64], z_skip: F128, x_rest: &[F128]) -> F128 {
    assert_eq!(buf.len(), 1 << x_rest.len());
    let w = lagrange_weights_naive(6, z_skip);
    let eq = build_eq_table(x_rest);
    let mut acc = F128::ZERO;
    for (rest, &word) in buf.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let mut wsum = F128::ZERO;
        let mut bits = word;
        while bits != 0 {
            let s = bits.trailing_zeros() as usize;
            wsum += w[s];
            bits &= bits - 1;
        }
        acc += eq[rest] * wsum;
    }
    acc
}

fn check_hash<S>(
    hash: &str,
    k_log: usize,
    useful_bits: usize,
    m: usize,
    inputs: &[S],
    direct: impl Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64]),
) {
    let n_log = m - k_log;
    let total_u64 = (1usize << m) / 64;
    let (mut z, mut a, mut b) =
        (vec![0u64; total_u64], vec![0u64; total_u64], vec![0u64; total_u64]);
    direct(inputs, n_log, None, &mut z, &mut a, &mut b);

    let useful_chunks = useful_bits.div_ceil(128);
    for pad in [
        PaddingSpec::dense(m),
        PaddingSpec {
            k_log: m,
            useful_bits_per_block: useful_chunks << (7 + n_log),
        },
    ] {
        let mut ch = FsChallenger::new(b"flock-e2-oracle");
        let (proof, claim) = zerocheck::prove_packed_padded(
            as_bytes(&a),
            as_bytes(&b),
            as_bytes(&z),
            m,
            &pad,
            &mut ch,
        );
        let mut chv = FsChallenger::new(b"flock-e2-oracle");
        let vclaim = zerocheck::verify(m, &proof, &mut chv)
            .unwrap_or_else(|e| panic!("{hash}: verify rejected: {e:?}"));
        assert_eq!(claim, vclaim);

        // Truthfulness: claims equal from-scratch MLE evals of the L1' data.
        assert_eq!(
            quirky_eval(&a, claim.z, &claim.mlv_challenges),
            claim.a_eval,
            "{hash}: a_eval untruthful on L1' buffers"
        );
        assert_eq!(
            quirky_eval(&b, claim.z, &claim.mlv_challenges),
            claim.b_eval,
            "{hash}: b_eval untruthful on L1' buffers"
        );
        assert_eq!(
            quirky_eval(&z, claim.z, &claim.r_rest),
            claim.c_eval,
            "{hash}: c_eval untruthful on L1' buffers"
        );
    }
}

#[test]
fn keccak_l1_zerocheck_truthful() {
    let m = 20; // n_log = 4
    let n = 1usize << (m - flock_prover::r1cs_hashes::keccak::K_LOG);
    let inputs: Vec<_> = (0..n as u64).map(keccak_witness::random_state).collect();
    check_hash(
        "keccak",
        flock_prover::r1cs_hashes::keccak::K_LOG,
        flock_prover::r1cs_hashes::keccak::USEFUL_BITS,
        m,
        &inputs,
        keccak_vwide::build_l1_direct,
    );
}

#[test]
fn sha2_l1_zerocheck_truthful() {
    let m = 20; // n_log = 5
    let n = 1usize << (m - flock_prover::r1cs_hashes::sha2::K_LOG);
    let inputs: Vec<_> = (0..n as u64).map(|s| sha2_witness::random_input(77 + s)).collect();
    check_hash(
        "sha2",
        flock_prover::r1cs_hashes::sha2::K_LOG,
        flock_prover::r1cs_hashes::sha2::USEFUL_BITS,
        m,
        &inputs,
        sha2_vwide::build_l1_direct,
    );
}
