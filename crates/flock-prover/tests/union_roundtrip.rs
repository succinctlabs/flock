//! M1/M2 differential oracle for the union-instance plumbing: a SINGLE-TYPE
//! registry instance proved through
//! `prove_fast_ligerito_jagged_union_harness` must produce a proof
//! **byte-identical** (bincode equality of the whole bundle, with the
//! lincheck sub-proof asserted separately — since M2 the union entry runs
//! the union-column lincheck, whose one-slot degeneration must BE today's
//! lincheck) to the existing `prove_fast_ligerito_jagged_from_witness` on
//! the same statement + witness at full utilization — the union of one slot
//! at offset 0 is today's instance verbatim, and the harness binding
//! changes no transcript. Plus a prove → verify roundtrip through the
//! harness entry pair alone, including count-binding rejection (a verifier
//! declaring a different count computes a different lincheck const-pin
//! target — and different jagged heights — and must reject).
//!
//! Since M3 the PROTOCOL union entries bind the statement as
//! `flock-mixed-v1` (registry digest + counts + root) and are exercised in
//! `tests/union_mixed.rs`; these tests pin the M1/M2 harness binding
//! (`bind_statement_single_type`) as the regression anchor for the
//! plumbing. Proofs under the protocol binding are (correctly) NOT
//! byte-identical to the direct path — the bindings are domain-separated.

use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::{blake3, sha2};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
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

const DOMAIN: &[u8] = b"flock-union-e2e-v0";

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

/// One-type registry reproducing `setup_r1cs`'s geometry, at capacity
/// `2^{n_log}` — the M1 differential harness shape.
fn single_type_registry(setup_r1cs: &flock_core::r1cs::BlockR1cs) -> Registry {
    Registry::new(
        vec![TableType::from_block_r1cs(setup_r1cs)],
        setup_r1cs.n_log(),
    )
}

/// BLAKE3 at full utilization: the union entry's proof is byte-identical to
/// the existing jagged path's, and both entries' verifiers accept.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_roundtrip -- --ignored`
fn blake3_union_matches_jagged_byte_identical() {
    let n_blocks = 256usize;
    let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
    let mut rng = Rng::new(0x0110_0B_B3);
    let inputs = random_blake3_inputs(&mut rng, n_blocks);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();

    // ---- Existing jagged path.
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

    // ---- Union entry: one-type registry at full utilization, same
    // statement + witness, same FS domain.
    let registry = single_type_registry(&setup.r1cs);
    let union = UnionInstance::new(&registry, vec![setup.n_block_slots()]);
    let slot = UnionSlotProverInput::new(
        blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
        lc_circuit,
    );
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_union, comm_union, claim_union) = prover::prove_fast_ligerito_jagged_union_harness(
        &union,
        &setup.r1cs,
        &setup.pcs_params,
        vec![slot],
        &mut ch_p,
    );

    // ---- THE differential: the ENTIRE proofs are byte-identical. The
    // lincheck sub-proof is asserted first on its own — it is the piece M2
    // replaced with the union-column lincheck, whose single-slot
    // degeneration must be byte-for-byte today's lincheck.
    assert_eq!(
        comm_jagged.root, comm_union.root,
        "commitment root diverged"
    );
    assert_eq!(claim_jagged, claim_union, "accepted claims diverged");
    assert_eq!(
        bincode::serialize(&proof_jagged.lincheck).unwrap(),
        bincode::serialize(&proof_union.lincheck).unwrap(),
        "union-column lincheck must be byte-identical to today's lincheck on one slot"
    );
    assert_eq!(
        bincode::serialize(&proof_jagged).unwrap(),
        bincode::serialize(&proof_union).unwrap(),
        "union proof must be byte-identical to the jagged path at full utilization"
    );

    // ---- Both verifiers accept their proof.
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

    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_vu = verifier::verify_ligerito_jagged_union_harness(
        &union,
        &setup.r1cs,
        &comm_union,
        &proof_union,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("union verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_vu, claim_union);
}

/// SHA-256 at full utilization: same differential.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_roundtrip -- --ignored`
fn sha256_union_matches_jagged_byte_identical() {
    let n_compressions = 128usize;
    let setup = sha2::Sha256HybridSetup::new_batch_major(n_compressions);
    let mut rng = Rng::new(0x0110_0B_52);
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

    let registry = single_type_registry(&setup.r1cs);
    let union = UnionInstance::new(&registry, vec![1usize << setup.n_blocks_log()]);
    let slot = UnionSlotProverInput::new(
        sha2::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
        lc_circuit,
    );
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_union, comm_union, claim_union) = prover::prove_fast_ligerito_jagged_union_harness(
        &union,
        &setup.r1cs,
        &setup.pcs_params,
        vec![slot],
        &mut ch_p,
    );

    assert_eq!(
        comm_jagged.root, comm_union.root,
        "commitment root diverged"
    );
    assert_eq!(claim_jagged, claim_union, "accepted claims diverged");
    assert_eq!(
        bincode::serialize(&proof_jagged.lincheck).unwrap(),
        bincode::serialize(&proof_union.lincheck).unwrap(),
        "union-column lincheck must be byte-identical to today's lincheck on one slot"
    );
    assert_eq!(
        bincode::serialize(&proof_jagged).unwrap(),
        bincode::serialize(&proof_union).unwrap(),
        "union proof must be byte-identical to the jagged path at full utilization"
    );

    let mut ch_v = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged(
        &setup.r1cs,
        &comm_jagged,
        &proof_jagged,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("jagged verifier rejected honest proof: {e:?}"));

    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_vu = verifier::verify_ligerito_jagged_union_harness(
        &union,
        &setup.r1cs,
        &comm_union,
        &proof_union,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("union verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_vu, claim_union);
}

/// BLAKE3 prove → verify roundtrip through the union entry pair alone, plus
/// count binding: a verifier whose instance declares one invocation fewer
/// must reject. Since M2 the count binds already inside the union-column
/// lincheck — BLAKE3 pins a constant wire, and the verifier's pin target
/// carries the count-derived factor Σ_{row<n} eq(x_outer, row) — so the
/// rejection surfaces at the lincheck's final consistency check, before the
/// jagged-heights mismatch would have rejected the opening.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_roundtrip -- --ignored`
fn blake3_union_roundtrip_and_count_rejection() {
    let n_blocks = 256usize;
    let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
    let mut rng = Rng::new(0x0110_4272);
    let inputs = random_blake3_inputs(&mut rng, n_blocks);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();

    let registry = single_type_registry(&setup.r1cs);
    let union = UnionInstance::new(&registry, vec![setup.n_block_slots()]);
    let slot = UnionSlotProverInput::new(
        blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
        lc_circuit,
    );
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) = prover::prove_fast_ligerito_jagged_union_harness(
        &union,
        &setup.r1cs,
        &setup.pcs_params,
        vec![slot],
        &mut ch_p,
    );

    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_v = verifier::verify_ligerito_jagged_union_harness(
        &union,
        &setup.r1cs,
        &commitment,
        &proof,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("union verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_v, claim);

    // Wrong declared count: the lincheck's const-pin target no longer
    // matches the committed pin column's fold (and the jagged heights would
    // mismatch downstream too) — the lincheck consistency check rejects
    // first.
    use flock_core::verifier::VerifyError;
    let union_bad = UnionInstance::new(&registry, vec![setup.n_block_slots() - 1]);
    let mut ch_v = FsChallenger::new(DOMAIN);
    match verifier::verify_ligerito_jagged_union_harness(
        &union_bad,
        &setup.r1cs,
        &commitment,
        &proof,
        lc_circuit,
        &setup.pcs_params,
        &mut ch_v,
    ) {
        Err(VerifyError::Lincheck(flock_core::lincheck::VerifyError::ConsistencyFailed {
            ..
        })) => {}
        other => panic!("wrong count: expected Lincheck(ConsistencyFailed), got {other:?}"),
    }
}
