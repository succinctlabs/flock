//! End-to-end prove → verify roundtrips and tamper-rejection tests.
//!
//! These live in `flock-prover` (not `flock-core`) because they exercise the
//! prove path; the verifier they call lives in `flock-core`. Moved here from
//! `flock_core::verifier`'s in-crate test module when the crates were split.

use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::{self, PcsParams};
use flock_prover::prover::prove_ligerito;
use flock_prover::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};
use flock_prover::verifier::{self, VerifyError};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn bits(&mut self, n: usize) -> Vec<bool> {
        (0..n).map(|_| self.next_u64() & 1 == 1).collect()
    }
}

fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k).map(|i| vec![i]).collect(),
    }
}

/// Build an identity-`C` R1CS with identity `A_0`/`B_0` at the given shape.
fn identity_r1cs(m: usize, k_log: usize, k_skip: usize, useful_bits: usize) -> BlockR1cs {
    BlockR1cs {
        m,
        k_log,
        k_skip,
        useful_bits,
        a_0: identity(1 << k_log),
        b_0: identity(1 << k_log),
        c_0: identity(1 << k_log),
        layout: WitnessLayout::RowMajor,
        const_pin: None,
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    }
}

/// End-to-end R1CS roundtrip using the Ligerito PCS backend, plus
/// mutation-rejection checks on the lincheck and PCS-open transcript pieces.
/// Ligerito's per-level query counts demand block_len ≥ ~243 at L0, so
/// m ≥ 19 or so.
#[test]
#[ignore] // Heavier — run with `cargo test r1cs_prove_verify_roundtrip_ligerito -- --ignored --nocapture`
fn r1cs_prove_verify_roundtrip_ligerito() {
    let m = 22;
    let k_log = 6;
    let k_skip = 6;
    let r1cs = identity_r1cs(m, k_log, k_skip, 1 << k_log);
    let mut rng = Rng::new(20_240_609);
    let z = rng.bits(r1cs.n());
    assert!(r1cs.satisfies(&z));

    // log_batch_size = 6 so Ligerito's initial_k = 6 reuses the L0 commit.
    let pcs_params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    let mut ch_p = FsChallenger::new(b"flock-lig-r1cs-v0");
    let z_packed = pcs::pack_witness(&z, r1cs.m);
    let (proof, commitment, claim_p) = prove_ligerito(&r1cs, z_packed, &pcs_params, &mut ch_p);

    let mut ch_v = FsChallenger::new(b"flock-lig-r1cs-v0");
    let lc_circuit = r1cs.sparse_lincheck_circuit();
    let claim_v = verifier::verify_ligerito(
        &r1cs,
        &commitment,
        &proof,
        &lc_circuit,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("ligerito verify rejected honest proof: {e:?}"));
    assert_eq!(claim_p, claim_v);

    // Tamper 1: corrupt the lincheck z-vector → lincheck replay rejects.
    {
        let mut bad = proof.clone();
        bad.lincheck.z_partial[0].lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-lig-r1cs-v0");
        let res =
            verifier::verify_ligerito(&r1cs, &commitment, &bad, &lc_circuit, &pcs_params, &mut ch);
        assert!(matches!(res, Err(VerifyError::Lincheck(_))));
    }

    // Tamper 2: corrupt a ring-switch s_hat_v → the PCS open rejects.
    {
        let mut bad = proof.clone();
        bad.pcs_open.ring_switches[0].s_hat_v[0].lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-lig-r1cs-v0");
        let res =
            verifier::verify_ligerito(&r1cs, &commitment, &bad, &lc_circuit, &pcs_params, &mut ch);
        assert!(matches!(res, Err(VerifyError::PcsAb(_))));
    }
}
