//! End-to-end prove → verify roundtrips and tamper-rejection tests.
//!
//! These live in `flock-prover` (not `flock-core`) because they exercise the
//! prove path; the verifier they call lives in `flock-core`. Moved here from
//! `flock_core::verifier`'s in-crate test module when the crates were split.

use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::ligerito::LigeritoProfile;
use flock_prover::pcs::{self, PcsParams};
use flock_prover::prover::prove_ligerito;
use flock_prover::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};
use flock_prover::verifier::{self, VerifyError};

use flock_core::test_rng::Rng;

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
        num_lanes: None,
        merkle_hash: Default::default(),
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

/// End-to-end check for strict Fast-profile Boolean zerocheck and lincheck
/// grinding. The default strict PCS profile selects both schedules;
/// this test deliberately remains ignored because its Ligerito opening is the
/// same heavyweight m22 workload as the ordinary end-to-end roundtrip above.
#[test]
#[ignore] // Run with `cargo test -p flock-prover --test verifier_roundtrip strict_fast_profile_grinds_boolean_piops -- --ignored`.
fn strict_fast_profile_grinds_boolean_piops() {
    let m = 22;
    let r1cs = identity_r1cs(m, 6, 6, 1 << 6);
    let mut rng = Rng::new(0x1280_0001);
    let z = rng.bits(r1cs.n());
    assert!(r1cs.satisfies(&z));
    let pcs_params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: None,
        merkle_hash: Default::default(),
    };

    let mut ch_p = FsChallenger::new(b"flock-strict-zc-grinding-v0");
    let (proof, commitment, claim_p) =
        prove_ligerito(&r1cs, pcs::pack_witness(&z, r1cs.m), &pcs_params, &mut ch_p);
    assert_eq!(
        proof.zerocheck.grinding_nonces.len(),
        2 + m - flock_prover::zerocheck::K_SKIP,
        "initial + skip + every tail-round nonce"
    );
    assert_eq!(
        proof.lincheck.grinding_nonces.len(),
        2,
        "alpha plus the final k_skip=6 evaluation (there are no inner rounds)"
    );

    let lc_circuit = r1cs.sparse_lincheck_circuit();
    let mut ch_v = FsChallenger::new(b"flock-strict-zc-grinding-v0");
    let claim_v = verifier::verify_ligerito(
        &r1cs,
        &commitment,
        &proof,
        &lc_circuit,
        &pcs_params,
        &mut ch_v,
    )
    .expect("the grinded proof verifies end to end");
    assert_eq!(claim_p, claim_v);

    let mut missing_nonce = proof.clone();
    missing_nonce.zerocheck.grinding_nonces.pop();
    let mut ch_bad = FsChallenger::new(b"flock-strict-zc-grinding-v0");
    assert!(matches!(
        verifier::verify_ligerito(
            &r1cs,
            &commitment,
            &missing_nonce,
            &lc_circuit,
            &pcs_params,
            &mut ch_bad,
        ),
        Err(VerifyError::Zerocheck(
            flock_prover::zerocheck::VerifyError::BadGrindingNonceCount { .. }
        ))
    ));

    let mut missing_lincheck_nonce = proof.clone();
    missing_lincheck_nonce.lincheck.grinding_nonces.pop();
    let mut ch_bad = FsChallenger::new(b"flock-strict-zc-grinding-v0");
    assert!(matches!(
        verifier::verify_ligerito(
            &r1cs,
            &commitment,
            &missing_lincheck_nonce,
            &lc_circuit,
            &pcs_params,
            &mut ch_bad,
        ),
        Err(VerifyError::Lincheck(
            flock_prover::lincheck::VerifyError::BadGrindingNonceCount { .. }
        ))
    ));
}

/// AG-skip mirror of the Ligerito roundtrip: prove_ligerito_ag →
/// verify_ligerito_ag, plus tamper-rejection on the AG round messages, the
/// c-eval, and the ring-switch s_hat_v (which the AG path claim-checks with
/// AG base-code skip weights).
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore] // Heavier — run with `cargo test r1cs_prove_verify_roundtrip_ligerito_ag -- --ignored --nocapture`
fn r1cs_prove_verify_roundtrip_ligerito_ag() {
    use flock_prover::field::F128;
    use flock_prover::prover::prove_ligerito_ag;

    let m = 22;
    let k_log = 16;
    let k_skip = 6;
    let r1cs = identity_r1cs(m, k_log, k_skip, 1 << k_log);
    let mut rng = Rng::new(20_260_625);
    let z = rng.bits(r1cs.n());
    assert!(
        r1cs.satisfies(&z),
        "identity R1CS: z·z = z holds for boolean z"
    );

    let pcs_params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    };
    let z_packed = pcs::pack_witness(&z, r1cs.m);
    let lc_circuit = r1cs.sparse_lincheck_circuit();

    // Honest: prover and verifier with matching transcripts.
    let mut ch_p = FsChallenger::new(b"flock-ag-r1cs-v0");
    let (proof, commitment, claim_p) = prove_ligerito_ag(&r1cs, z_packed, &pcs_params, &mut ch_p);

    let mut ch_v = FsChallenger::new(b"flock-ag-r1cs-v0");
    let claim_v = verifier::verify_ligerito_ag(
        &r1cs,
        &commitment,
        &proof,
        &lc_circuit,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("AG ligerito verify rejected honest proof: {e:?}"));
    assert_eq!(claim_p, claim_v, "verifier claim != prover claim");

    // Tamper 1: corrupt an AG multilinear round message → AG replay rejects.
    {
        let mut bad = proof.clone();
        bad.ag.multilinear_rounds[0].0 += F128::ONE;
        let mut ch = FsChallenger::new(b"flock-ag-r1cs-v0");
        assert!(
            verifier::verify_ligerito_ag(
                &r1cs,
                &commitment,
                &bad,
                &lc_circuit,
                &pcs_params,
                &mut ch
            )
            .is_err(),
            "must reject a tampered AG round message"
        );
    }

    // Tamper 2: corrupt the AG c-eval → CEvalMismatch.
    {
        let mut bad = proof.clone();
        bad.ag.final_c_eval += F128::ONE;
        let mut ch = FsChallenger::new(b"flock-ag-r1cs-v0");
        assert!(
            matches!(
                verifier::verify_ligerito_ag(
                    &r1cs,
                    &commitment,
                    &bad,
                    &lc_circuit,
                    &pcs_params,
                    &mut ch
                ),
                Err(VerifyError::Ag(_))
            ),
            "must reject a tampered c-eval"
        );
    }

    // Tamper 3: corrupt the r1 grinding nonce → BadR1Nonce / downstream reject.
    {
        let mut bad = proof.clone();
        bad.ag.r1_nonce += 1;
        let mut ch = FsChallenger::new(b"flock-ag-r1cs-v0");
        assert!(
            verifier::verify_ligerito_ag(
                &r1cs,
                &commitment,
                &bad,
                &lc_circuit,
                &pcs_params,
                &mut ch
            )
            .is_err(),
            "must reject a tampered r1 nonce"
        );
    }

    // Tamper 4: corrupt a ring-switch s_hat_v → the claim_check with AG base
    // weights (build_claim_weights_from_skip) fails at the PCS open.
    {
        let mut bad = proof.clone();
        bad.pcs_open.ring_switches[0].s_hat_v[0].lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-ag-r1cs-v0");
        assert!(
            verifier::verify_ligerito_ag(
                &r1cs,
                &commitment,
                &bad,
                &lc_circuit,
                &pcs_params,
                &mut ch
            )
            .is_err(),
            "must reject a tampered ring-switch s_hat_v"
        );
    }
}
