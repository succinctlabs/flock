use super::*;
use flock_core::challenger::FsChallenger;
use flock_core::circuit::boolean::LoweringMode;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{self, PcsParams};
use flock_core::verifier;
use flock_prover::prover::prove_ligerito;

#[test]
#[ignore = "m=22 Ligerito proof; run explicitly in release mode"]
fn double_add_proves_and_verifies() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let source = compiled.circuit();
    let lowered = source.lower(LoweringMode::RequireIdentityC).unwrap();
    assert!(lowered.c_is_identity());

    // Use 128-bit blocks and repeat them to the smallest registered PCS size.
    let k_log = 7;
    let n_log = 22 - k_log;
    let r1cs = lowered
        .to_block_r1cs(k_log, flock_core::zerocheck::K_SKIP, n_log)
        .unwrap();
    let plan = lowered.walk_plan().unwrap();
    let adapter = plan.lincheck_circuit(k_log).unwrap();
    assert_eq!(adapter.const_pin_col(), r1cs.const_pin);

    let trace = generate_trace(&compiled, &[Some(Event { a: 7, b: 9, c: 3 }), None]).unwrap();
    let mut pair = Vec::new();
    for (cols, logical) in &trace {
        assert_eq!(
            read_word(cols.output),
            if cols.inputs.active { 3 } else { 0 }
        );
        let extended = lowered.extend(logical).unwrap();
        assert!(lowered.accepts(&extended));
        assert_eq!(lowered.project(&extended).as_ref(), Some(logical));
        let inputs: Vec<_> = source.inputs().iter().map(|v| logical[v.index()]).collect();
        let block = plan.forward(&inputs, k_log).unwrap().z;
        for (value, &position) in lowered.layout().value_positions().iter().enumerate() {
            assert_eq!(block[position], extended[value]);
        }
        assert!(block[r1cs.const_pin.unwrap()]);
        pair.extend(block);
    }
    let witness = pair.repeat((1 << n_log) / trace.len());
    assert_eq!(witness.len(), 1 << r1cs.m);
    assert!(r1cs.satisfies(&witness));

    let params = PcsParams {
        m: r1cs.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    };
    let challenger = || FsChallenger::new(b"boolean-columns-proof-v1");
    let packed = pcs::pack_witness(&witness, r1cs.m);
    let (proof, commitment, expected) = prove_ligerito(&r1cs, packed, &params, &mut challenger());
    let verify = |relation, walk: &dyn LincheckCircuit, proof| {
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

    // Removing the ONE pin changes the statement, even at identical dimensions.
    let mut unpinned = lowered
        .to_block_r1cs(k_log, flock_core::zerocheck::K_SKIP, n_log)
        .unwrap();
    unpinned.const_pin = None;
    assert_ne!(unpinned.statement_digest(), r1cs.statement_digest());
    assert!(verify(&unpinned, &unpinned.sparse_lincheck_circuit(), &proof).is_err());
}
