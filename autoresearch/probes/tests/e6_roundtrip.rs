//! E6 oracle — L1′ prove → verify roundtrips for all three hashes on both
//! PCS backends; precomputed-s_hat_v proofs byte-identical to the plain
//! path; tampered proofs rejected.

use flock_autoresearch_probes::e6::{
    L1HashSpec, prove_l1_basefold, prove_l1_ligerito, setup, verify_l1_basefold,
    verify_l1_ligerito,
};
use flock_autoresearch_probes::{blake3_vwide, blake3_witness, keccak_vwide, keccak_witness,
    sha2_vwide, sha2_witness};
use flock_core::challenger::FsChallenger;
use flock_core::lincheck::LincheckCircuit;

fn check_hash<S: Sync>(
    hash: &str,
    r1cs: flock_core::r1cs::BlockR1cs,
    circuit: &dyn LincheckCircuit,
    inputs: &[S],
    direct: &(dyn Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64]) + Sync),
    ligerito: bool,
) {
    let (r1cs, pcs_params) = setup(r1cs);
    let spec = L1HashSpec {
        r1cs: &r1cs,
        circuit,
        direct,
    };

    // BaseFold: roundtrip with precomputes, and byte-equality vs plain path.
    let mut ch = FsChallenger::new(b"flock-e6-v1");
    let pre = prove_l1_basefold(&spec, &pcs_params, inputs, true, &mut ch);
    let mut ch = FsChallenger::new(b"flock-e6-v1");
    let plain = prove_l1_basefold(&spec, &pcs_params, inputs, false, &mut ch);
    assert_eq!(
        bincode::serialize(&pre.proof).unwrap(),
        bincode::serialize(&plain.proof).unwrap(),
        "{hash}: precomputed s_hat_v changed the proof bytes"
    );
    let mut chv = FsChallenger::new(b"flock-e6-v1");
    let (ab, c) = verify_l1_basefold(&r1cs, circuit, &pre.commitment, &pre.proof, &mut chv)
        .unwrap_or_else(|e| panic!("{hash}: basefold verify rejected: {e:?}"));
    assert_eq!((ab, c), (pre.ab.clone(), pre.c.clone()), "{hash}: claim mismatch");

    // Ligerito (production backend) — config availability permitting.
    if ligerito {
        let mut ch = FsChallenger::new(b"flock-e6-v1");
        let res = prove_l1_ligerito(&spec, &pcs_params, inputs, true, &mut ch);
        let mut chv = FsChallenger::new(b"flock-e6-v1");
        verify_l1_ligerito(
            &r1cs,
            circuit,
            &res.commitment,
            &res.proof,
            &pcs_params,
            &mut chv,
        )
        .unwrap_or_else(|e| panic!("{hash}: ligerito verify rejected: {e:?}"));
    }
}

#[test]
fn keccak_l1_roundtrips() {
    use flock_prover::r1cs_hashes::keccak::{KeccakLincheckCircuit, build_block_r1cs};
    // BaseFold at small m; Ligerito needs an embedded config (m >= 22).
    for (n_log, ligerito) in [(3usize, false), (6, true)] {
        let inputs: Vec<_> = (0..1u64 << n_log)
            .map(|s| keccak_witness::random_state(0xE6 + s))
            .collect();
        check_hash(
            "keccak",
            build_block_r1cs(n_log),
            &KeccakLincheckCircuit,
            &inputs,
            &keccak_vwide::build_l1_direct,
            ligerito,
        );
    }
}

#[test]
fn sha2_l1_roundtrips() {
    use flock_prover::r1cs_hashes::sha2::build_block_r1cs;
    for (n_log, ligerito) in [(5usize, false), (7, true)] {
        let r1cs = build_block_r1cs(n_log);
        let inputs: Vec<_> = (0..1u64 << n_log)
            .map(|s| sha2_witness::random_input(0xE6E6 + s))
            .collect();
        let circuit = r1cs.csc_lincheck_circuit().clone();
        check_hash(
            "sha2",
            r1cs,
            &circuit,
            &inputs,
            &sha2_vwide::build_l1_direct,
            ligerito,
        );
    }
}

#[test]
fn blake3_l1_roundtrips() {
    use flock_prover::r1cs_hashes::blake3::build_block_r1cs;
    for (n_log, ligerito) in [(6usize, false), (8, true)] {
        let r1cs = build_block_r1cs(n_log);
        let inputs: Vec<_> = (0..1u64 << n_log)
            .map(|s| blake3_witness::random_input(0xB3E6 + s))
            .collect();
        let circuit = r1cs.csc_lincheck_circuit().clone();
        check_hash(
            "blake3",
            r1cs,
            &circuit,
            &inputs,
            &blake3_vwide::build_l1_direct,
            ligerito,
        );
    }
}

#[test]
fn l1_verify_rejects_tampering() {
    use flock_prover::r1cs_hashes::keccak::{KeccakLincheckCircuit, build_block_r1cs};
    let n_log = 3;
    let (r1cs, pcs_params) = setup(build_block_r1cs(n_log));
    let inputs: Vec<_> = (0..1u64 << n_log)
        .map(|s| keccak_witness::random_state(0xBAD + s))
        .collect();
    let spec = L1HashSpec {
        r1cs: &r1cs,
        circuit: &KeccakLincheckCircuit,
        direct: &keccak_vwide::build_l1_direct,
    };
    let mut ch = FsChallenger::new(b"flock-e6-v1");
    let res = prove_l1_basefold(&spec, &pcs_params, &inputs, true, &mut ch);

    for (label, mutate) in [
        ("zerocheck round1", 0usize),
        ("zerocheck final_a", 1),
        ("lincheck z_partial", 2),
    ] {
        let mut bad = res.proof.clone();
        match mutate {
            0 => bad.zerocheck.round1_ab[0].lo ^= 1,
            1 => bad.zerocheck.final_a_eval.lo ^= 1,
            _ => bad.lincheck.z_partial[0].hi ^= 1,
        }
        let mut ch = FsChallenger::new(b"flock-e6-v1");
        assert!(
            verify_l1_basefold(&r1cs, &KeccakLincheckCircuit, &res.commitment, &bad, &mut ch)
                .is_err(),
            "{label} tamper accepted"
        );
    }
}

/// Recycled-buffer contract: a maximally dirty buffer plus the suffix-only
/// zeroing done by the prover core must still produce valid proofs (covered
/// implicitly: prove_l1_* pulls dirty scratch buffers and only zeroes the
/// suffix — run twice to exercise reuse).
#[test]
fn l1_recycled_buffer_reuse() {
    use flock_prover::r1cs_hashes::keccak::{KeccakLincheckCircuit, build_block_r1cs};
    let n_log = 3;
    let (r1cs, pcs_params) = setup(build_block_r1cs(n_log));
    let spec = L1HashSpec {
        r1cs: &r1cs,
        circuit: &KeccakLincheckCircuit,
        direct: &keccak_vwide::build_l1_direct,
    };
    for round in 0..3u64 {
        let inputs: Vec<_> = (0..1u64 << n_log)
            .map(|s| keccak_witness::random_state(round * 100 + s))
            .collect();
        let mut ch = FsChallenger::new(b"flock-e6-v1");
        let res = prove_l1_basefold(&spec, &pcs_params, &inputs, true, &mut ch);
        let mut chv = FsChallenger::new(b"flock-e6-v1");
        verify_l1_basefold(&r1cs, &KeccakLincheckCircuit, &res.commitment, &res.proof, &mut chv)
            .unwrap_or_else(|e| panic!("reuse round {round} rejected: {e:?}"));
    }
}
