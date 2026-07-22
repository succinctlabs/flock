//! End-to-end tests of the **jagged opening path** (Phase 1 of the
//! multi-table design): prove → verify roundtrips on batch-major BLAKE3 and
//! SHA-256 instances, a differential check that the jagged and direct paths
//! prove the same accepted statement (same commitment root, byte-identical
//! PIOP transcript prefix), tamper rejection on every jagged-specific proof
//! component, and a proof-size comparison between the two paths.

use flock_prover::challenger::FsChallenger;
use flock_prover::prover;
use flock_prover::r1cs_hashes::{blake3, sha2};
use flock_prover::verifier;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
}

const DOMAIN: &[u8] = b"flock-jagged-e2e-v0";

fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<blake3::Compression> {
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

/// BLAKE3 batch-major: jagged prove → verify roundtrip, differential
/// statement equality against the direct path, proof-size delta, and tamper
/// rejection of every jagged-specific proof component.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test jagged_roundtrip -- --ignored`
fn blake3_jagged_roundtrip_differential_and_tamper() {
    let n_blocks = 256usize;
    let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
    let mut rng = Rng::new(0x1A66_ED_B3);
    let inputs = random_blake3_inputs(&mut rng, n_blocks);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();

    // ---- Direct (legacy) path.
    let (z, a, b, stripe) = blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log());
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_direct, comm_direct, claim_direct) = prover::prove_fast_ligerito_from_witness(
        &setup.r1cs,
        &setup.pcs_params,
        z,
        a,
        b,
        stripe,
        lc_circuit,
        None,
        &mut ch_p,
    );

    // ---- Jagged path, same statement + witness, same FS domain.
    let (z, a, b, stripe) = blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log());
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_jagged, comm_jagged, claim_jagged) = prover::prove_fast_ligerito_jagged_from_witness(
        &setup.r1cs,
        &setup.pcs_params,
        z,
        a,
        b,
        stripe,
        lc_circuit,
        None,
        &mut ch_p,
    );

    // ---- (d) Same accepted statement: same commitment root, byte-identical
    // PIOP transcript prefix (zerocheck + lincheck sub-proofs and the shared
    // ring-switch claim-assembly messages), same claims.
    assert_eq!(
        comm_direct.root, comm_jagged.root,
        "commitment root diverged"
    );
    assert_eq!(
        proof_direct.zerocheck, proof_jagged.zerocheck,
        "zerocheck transcript diverged between the two paths"
    );
    assert_eq!(
        proof_direct.lincheck, proof_jagged.lincheck,
        "lincheck transcript diverged between the two paths"
    );
    assert_eq!(
        proof_direct.pcs_open.ring_switches, proof_jagged.pcs_open.ring_switches,
        "ring-switch messages diverged (shared claim assembly must be transcript-identical)"
    );
    assert_eq!(claim_direct, claim_jagged, "accepted claims diverged");

    // ---- Both paths verify.
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_vd = verifier::verify_ligerito(
        &setup.r1cs,
        &comm_direct,
        &proof_direct,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("direct verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_vd, claim_direct);

    let verify_jagged = |proof: &flock_core::proof::R1csProofJaggedLigerito,
                         r1cs: &flock_core::r1cs::BlockR1cs| {
        let mut ch_v = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_jagged(
            r1cs,
            &comm_jagged,
            proof,
            lc_circuit,
            &setup.pcs_params,
            &mut ch_v,
        )
    };
    let claim_vj = verify_jagged(&proof_jagged, &setup.r1cs)
        .unwrap_or_else(|e| panic!("jagged verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_vj, claim_jagged);

    // ---- Proof-size delta between the two paths.
    let direct_bytes = bincode::serialize(&proof_direct).unwrap().len();
    let jagged_bytes = bincode::serialize(&proof_jagged).unwrap().len();
    println!(
        "BLAKE3 {} blocks (m={}): direct proof {} B, jagged proof {} B, delta {:+} B",
        n_blocks,
        setup.m(),
        direct_bytes,
        jagged_bytes,
        jagged_bytes as i64 - direct_bytes as i64,
    );

    // ---- (c) Tamper rejection on the jagged-specific components.
    use flock_core::pcs::VerifyErrorJagged;
    use flock_core::verifier::VerifyError;
    let expect_pcs = |proof: &flock_core::proof::R1csProofJaggedLigerito,
                      want: VerifyErrorJagged,
                      what: &str| {
        match verify_jagged(proof, &setup.r1cs) {
            Err(VerifyError::PcsJagged(e)) if e == want => {}
            other => panic!("tampered {what}: expected PcsJagged({want:?}), got {other:?}"),
        }
    };
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.f_eval.lo ^= 1;
        expect_pcs(&bad, VerifyErrorJagged::VirtualOpen, "f_eval");
    }
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.virtual_open_rounds[2].1.lo ^= 1;
        expect_pcs(&bad, VerifyErrorJagged::VirtualOpen, "virtual-open round");
    }
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.jagged_sumcheck.rounds[1].0.lo ^= 1;
        expect_pcs(&bad, VerifyErrorJagged::Jagged, "jagged sumcheck round");
    }
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.jagged_sumcheck.q_eval.lo ^= 1;
        expect_pcs(&bad, VerifyErrorJagged::Jagged, "dense claim α");
    }
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.jagged_assist.beta.lo ^= 1;
        expect_pcs(&bad, VerifyErrorJagged::Jagged, "assist β");
    }
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.jagged_assist.rounds[7].1.lo ^= 1;
        expect_pcs(&bad, VerifyErrorJagged::Jagged, "assist round");
    }
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.ring_switches[0].s_hat_v[0].lo ^= 1;
        match verify_jagged(&bad, &setup.r1cs) {
            Err(VerifyError::PcsJagged(VerifyErrorJagged::RingSwitch(_))) => {}
            other => panic!("tampered s_hat_v: expected RingSwitch error, got {other:?}"),
        }
    }
    {
        let mut bad = proof_jagged.clone();
        bad.pcs_open.ligerito.final_proof.yr[0].lo ^= 1;
        expect_pcs(&bad, VerifyErrorJagged::Ligerito, "ligerito final message");
    }
    // Wrong heights vector: a verifier whose statement declares one fewer
    // useful chunk-column derives different heights and must reject.
    {
        let mut r1cs_bad = setup.r1cs.clone();
        r1cs_bad.useful_bits -= 128;
        match verify_jagged(&proof_jagged, &r1cs_bad) {
            Err(VerifyError::PcsJagged(VerifyErrorJagged::Jagged)) => {}
            other => panic!("wrong heights: expected Jagged error, got {other:?}"),
        }
    }
    // PIOP tampering still rejects through the shared verify_core.
    {
        let mut bad = proof_jagged.clone();
        bad.zerocheck.final_a_eval.lo ^= 1;
        assert!(
            verify_jagged(&bad, &setup.r1cs).is_err(),
            "tampered zerocheck accepted on the jagged path"
        );
    }
}

/// SHA-256 batch-major: jagged prove → verify roundtrip + differential
/// statement equality against the direct path.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test jagged_roundtrip -- --ignored`
fn sha256_jagged_roundtrip_differential() {
    let n_compressions = 128usize;
    let setup = sha2::Sha256HybridSetup::new_batch_major(n_compressions);
    let mut rng = Rng::new(0x1A66_ED_52);
    let inputs: Vec<sha2::Compression> = (0..n_compressions)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.next_u32()),
                std::array::from_fn(|_| rng.next_u32()),
            )
        })
        .collect();
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();

    let (z, a, b, stripe) = sha2::generate_witness_batch_major(&inputs, setup.n_blocks_log());
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_direct, comm_direct, claim_direct) = prover::prove_fast_ligerito_from_witness(
        &setup.r1cs,
        &setup.pcs_params,
        z,
        a,
        b,
        stripe,
        lc_circuit,
        None,
        &mut ch_p,
    );

    let (z, a, b, stripe) = sha2::generate_witness_batch_major(&inputs, setup.n_blocks_log());
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_jagged, comm_jagged, claim_jagged) = prover::prove_fast_ligerito_jagged_from_witness(
        &setup.r1cs,
        &setup.pcs_params,
        z,
        a,
        b,
        stripe,
        lc_circuit,
        None,
        &mut ch_p,
    );

    assert_eq!(
        comm_direct.root, comm_jagged.root,
        "commitment root diverged"
    );
    assert_eq!(proof_direct.zerocheck, proof_jagged.zerocheck);
    assert_eq!(proof_direct.lincheck, proof_jagged.lincheck);
    assert_eq!(
        proof_direct.pcs_open.ring_switches,
        proof_jagged.pcs_open.ring_switches
    );
    assert_eq!(claim_direct, claim_jagged);

    let mut ch_v = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito(
        &setup.r1cs,
        &comm_direct,
        &proof_direct,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("direct verifier rejected honest proof: {e:?}"));

    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_vj = verifier::verify_ligerito_jagged(
        &setup.r1cs,
        &comm_jagged,
        &proof_jagged,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("jagged verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_vj, claim_jagged);

    let direct_bytes = bincode::serialize(&proof_direct).unwrap().len();
    let jagged_bytes = bincode::serialize(&proof_jagged).unwrap().len();
    println!(
        "SHA-256 {} compressions (m={}): direct proof {} B, jagged proof {} B, delta {:+} B",
        n_compressions,
        setup.m(),
        direct_bytes,
        jagged_bytes,
        jagged_bytes as i64 - direct_bytes as i64,
    );
}
