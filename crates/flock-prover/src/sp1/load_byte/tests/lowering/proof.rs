use super::*;
use crate::prover::prove_ligerito;
use flock_core::challenger::FsChallenger;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{self, PcsParams};
use flock_core::verifier;

#[test]
#[ignore = "m=22 Ligerito proof using the smallest registered PCS shape"]
fn converted_load_byte_proves_and_verifies_with_walk_adapter() {
    let capacity = 13; // Includes an active row, padding, and alignment holes.
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
    let (proof, commitment, expected) = prove_ligerito(&r1cs, packed, &params, &mut challenger());
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
    assert_eq!(verify(&r1cs, &adapter, &proof).unwrap(), expected);

    let mut tampered = proof.clone();
    tampered.lincheck.z_partial[0].lo ^= 1;
    assert!(verify(&r1cs, &adapter, &tampered).is_err());
}
