use super::*;
use crate::prover::prove_ligerito;
use flock_core::challenger::FsChallenger;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{self, PcsParams};
use flock_core::verifier;
use std::time::Instant;

#[test]
#[ignore = "m=22 Ligerito proof using the smallest registered PCS shape"]
fn converted_load_byte_proves_and_verifies_with_walk_adapter() {
    for capacity in [2, 7, 13] {
        prove_capacity(capacity);
    }
}

fn prove_capacity(capacity: usize) {
    let chip = LoadByteCircuit::build(capacity);
    let lowered = chip.lower(LoweringMode::RequireIdentityC).unwrap();
    let plan = lowered.walk_plan().unwrap();
    let k_log = plan.useful_bits().next_power_of_two().trailing_zeros() as usize;
    let n_log = 22 - k_log;
    let k_skip = flock_core::zerocheck::K_SKIP;
    let r1cs = lowered.to_block_r1cs(k_log, k_skip, n_log).unwrap();
    let inputs = chip.honest_inputs(&[event(
        LoadByteOpcode::Lb,
        0x1_0000,
        7,
        0x8000_0000_0000_0000,
    )]);
    let mut block = plan.forward(&inputs, k_log).unwrap().z;
    assert_eq!(
        block,
        physical(&lowered, &lowered.evaluate(&inputs).unwrap(), k_log)
    );
    // Prove a legal noncanonical t assignment too; zero is only an honest convention.
    block[lowered
        .layout()
        .value_position(lowered.auxiliaries()[0].cancellation)
        .unwrap()] = true;
    let witness = block.repeat(1 << n_log);
    assert!(r1cs.satisfies(&witness));
    let adapter = plan.lincheck_circuit(k_log).unwrap();
    assert_eq!(adapter.const_pin_col(), r1cs.const_pin);
    assert_eq!(
        r1cs.const_pin,
        lowered.layout().value_position(lowered.one())
    );
    assert!(block[r1cs.const_pin.unwrap()]);
    let params = PcsParams {
        m: r1cs.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    };
    let challenger = || FsChallenger::new(b"load-byte-two-variable-proof-v1");
    let packed = pcs::pack_witness(&witness, r1cs.m);
    let start = Instant::now();
    let (proof, commitment, expected) = prove_ligerito(&r1cs, packed, &params, &mut challenger());
    let proving_time = start.elapsed();
    let verify = |relation: &BlockR1cs, walk: &dyn LincheckCircuit, proof| {
        verifier::verify_ligerito(
            relation,
            &commitment,
            proof,
            walk,
            &params,
            &mut challenger(),
        )
    };
    let start = Instant::now();
    assert_eq!(verify(&r1cs, &adapter, &proof).unwrap(), expected);
    println!(
        "capacity={capacity} k_log={k_log} m={} batches={} prove_ms={:.2} verify_ms={:.2}",
        r1cs.m,
        1usize << n_log,
        proving_time.as_secs_f64() * 1e3,
        start.elapsed().as_secs_f64() * 1e3
    );

    let mut tampered = proof.clone();
    tampered.lincheck.z_partial[0].lo ^= 1;
    assert!(verify(&r1cs, &adapter, &tampered).is_err());

    // Equal proof dimensions do not make a source walk valid for the converted relation.
    let source_plan = chip.circuit().walk_plan().unwrap();
    let source_adapter = source_plan.lincheck_circuit(k_log).unwrap();
    assert!(verify(&r1cs, &source_adapter, &proof).is_err());

    // Fresh matrices avoid stale digest caches while changing the bound statement.
    let mut unpinned = lowered.to_block_r1cs(k_log, k_skip, n_log).unwrap();
    unpinned.const_pin = None;
    assert_ne!(r1cs.statement_digest(), unpinned.statement_digest());
    assert!(verify(&unpinned, &unpinned.sparse_lincheck_circuit(), &proof).is_err());

    let other = LoadByteCircuit::build(1)
        .lower(LoweringMode::RequireIdentityC)
        .unwrap();
    let other_matrix = other.to_block_r1cs(k_log, k_skip, n_log).unwrap();
    let other_plan = other.walk_plan().unwrap();
    assert_ne!(r1cs.statement_digest(), other_matrix.statement_digest());
    assert!(
        verify(
            &other_matrix,
            &other_plan.lincheck_circuit(k_log).unwrap(),
            &proof
        )
        .is_err()
    );
}
