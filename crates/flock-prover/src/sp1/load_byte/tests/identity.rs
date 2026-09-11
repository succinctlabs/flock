use super::*;

#[test]
fn direct_and_bound_identity_c_reject_the_same_inputs() {
    let chip = LoadByteCircuit::build(2);
    let source = chip.circuit();
    let checker = source.identity_checker();
    let sparse = source.to_block_r1cs(k_log(source), 0, 0).unwrap();
    let raw = checker.unbound_circuit();
    let raw_sparse = raw.to_block_r1cs(k_log(raw), 0, 0).unwrap();
    let plan = raw.walk_plan().unwrap();
    assert!(raw_sparse.c0_is_identity());
    for (inputs, expected) in input_cases(&chip) {
        let logical = candidate(source, &inputs);
        let extended = checker.extend(&logical).unwrap();
        let mut padded = logical.clone();
        padded.resize(1 << k_log(source), false);
        assert_eq!(source.evaluate(&inputs).is_ok(), expected);
        assert_eq!(sparse.satisfies(&padded), expected);
        assert_eq!(checker.accepts(&extended), expected);
        assert_eq!(checker.project(&extended), Some(logical));
        let mut raw_witness = extended.clone();
        raw_witness.resize(1 << k_log(raw), false);
        assert!(
            raw_sparse.satisfies(&raw_witness),
            "local rows alone do not reject"
        );
        assert!(raw_witness[raw_sparse.const_pin.unwrap()]);
        assert_eq!(raw_witness[checker.accept().index()], expected);
        let raw_inputs: Vec<_> = raw
            .inputs()
            .iter()
            .map(|value| extended[value.index()])
            .collect();
        let walked = plan.forward(&raw_inputs, k_log(raw)).unwrap();
        assert_eq!(walked.z, raw_witness);
        assert_eq!(walked.a_z, raw_sparse.apply_a(&walked.z));
        assert_eq!(walked.b_z, raw_sparse.apply_b(&walked.z));
        assert_eq!(walked.c_z, walked.z);
        if !expected {
            let mut forged = extended;
            forged[checker.accept().index()] = true;
            assert!(!checker.accepts(&forged));
        }
    }
}

#[test]
fn inactive_advice_and_multiple_failures_survive_conversion() {
    let chip = LoadByteCircuit::build(1);
    let checker = chip.circuit().identity_checker();
    for active in [false, true] {
        for advice in 0..256 {
            let inputs = fixtures::advice_inputs(&chip, active, advice);
            let witness = checker.extend(&candidate(chip.circuit(), &inputs)).unwrap();
            assert_eq!(checker.accepts(&witness), !active || advice == 0);
        }
    }
}
