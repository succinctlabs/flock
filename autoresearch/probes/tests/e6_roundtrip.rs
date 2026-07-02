//! E6 oracle — the full L1′ prove → verify roundtrip must pass, claims must
//! agree between prover and verifier, and tampered proofs must be rejected.

use flock_autoresearch_probes::e6::{keccak_setup, prove_l1_keccak, verify_l1_keccak};
use flock_autoresearch_probes::keccak_witness::random_state;
use flock_core::challenger::FsChallenger;

#[test]
fn l1_prove_verify_roundtrip() {
    for n_log in [3usize, 4] {
        let (r1cs, pcs_params) = keccak_setup(n_log);
        let states: Vec<_> = (0..1u64 << n_log).map(|s| random_state(0xE6 + s)).collect();

        let mut chp = FsChallenger::new(b"flock-e6-v0");
        let res = prove_l1_keccak(&r1cs, &pcs_params, &states, &mut chp);

        let mut chv = FsChallenger::new(b"flock-e6-v0");
        let (ab, c) = verify_l1_keccak(&r1cs, &res.commitment, &res.proof, &mut chv)
            .unwrap_or_else(|e| panic!("L1' verify rejected honest proof (n_log={n_log}): {e:?}"));
        assert_eq!(ab, res.ab, "AB claim mismatch prover/verifier");
        assert_eq!(c, res.c, "C claim mismatch prover/verifier");
    }
}

#[test]
fn l1_verify_rejects_tampering() {
    let n_log = 3;
    let (r1cs, pcs_params) = keccak_setup(n_log);
    let states: Vec<_> = (0..1u64 << n_log).map(|s| random_state(0xBAD + s)).collect();

    let mut chp = FsChallenger::new(b"flock-e6-v0");
    let res = prove_l1_keccak(&r1cs, &pcs_params, &states, &mut chp);

    // Tamper each sub-proof; all must be rejected.
    {
        let mut bad = res.proof.clone();
        bad.zerocheck.round1_ab[0].lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-e6-v0");
        assert!(
            verify_l1_keccak(&r1cs, &res.commitment, &bad, &mut ch).is_err(),
            "zerocheck tamper accepted"
        );
    }
    {
        let mut bad = res.proof.clone();
        bad.zerocheck.final_a_eval.lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-e6-v0");
        assert!(
            verify_l1_keccak(&r1cs, &res.commitment, &bad, &mut ch).is_err(),
            "a_eval tamper accepted"
        );
    }
    {
        // A false statement: witness with one wrong AND output. Rebuild by
        // proving different states but verifying against the first proof's
        // commitment is covered by FS binding; here flip a lincheck message.
        let mut bad = res.proof.clone();
        bad.lincheck.z_partial[0].hi ^= 1;
        let mut ch = FsChallenger::new(b"flock-e6-v0");
        assert!(
            verify_l1_keccak(&r1cs, &res.commitment, &bad, &mut ch).is_err(),
            "lincheck tamper accepted"
        );
    }
}
