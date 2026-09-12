use super::*;
use flock_core::circuit::boolean::WalkError;

mod proof;

#[test]
fn required_identity_c_matches_direct_for_valid_and_invalid_candidates() {
    let chip = LoadByteCircuit::build(2);
    let source = chip.circuit();
    let direct = chip.lower(LoweringMode::Direct).unwrap();
    let lowered = chip.lower(LoweringMode::RequireIdentityC).unwrap();
    let original = matrix(&direct);
    let converted = matrix(&lowered);
    let plan = lowered.walk_plan().unwrap();
    assert!(!original.c0_is_identity());
    assert!(converted.c0_is_identity());

    for (name, inputs, output) in input_cases(&chip) {
        let expected = output.is_some();
        let candidate = candidate(source, &inputs);
        assert_eq!(source.evaluate(&inputs).is_ok(), expected, "{name}");
        if let Some(output) = output {
            assert_eq!(chip.output(&candidate, 0), output, "{name}");
        }
        assert_eq!(lowered.evaluate(&inputs).is_ok(), expected, "{name}");
        assert_eq!(direct.accepts(&candidate), expected, "{name}");
        assert_eq!(
            original.satisfies(&physical(&direct, &candidate, original.k_log)),
            expected
        );
        let mut extension = lowered.extend(&candidate).unwrap();
        assert_eq!(lowered.project(&extension), Some(candidate.clone()));
        assert_eq!(lowered.accepts(&extension), expected);
        assert_eq!(
            converted.satisfies(&physical(&lowered, &extension, converted.k_log)),
            expected
        );
        match plan.forward(&inputs, converted.k_log) {
            Ok(walked) => {
                assert!(expected);
                assert_eq!(walked.z, physical(&lowered, &extension, converted.k_log));
                assert_eq!(walked.a_z, converted.apply_a(&walked.z));
                assert_eq!(walked.b_z, converted.apply_b(&walked.z));
                assert_eq!(walked.c_z, walked.z);
                assert_eq!(lowered.evaluate(&inputs).unwrap(), extension);
            }
            Err(WalkError::UnsatisfiedRow(target)) => {
                assert!(!expected);
                assert_eq!(
                    source.evaluate(&inputs),
                    Err(EvaluationError::UnsatisfiedRow(
                        lowered.source_row(target).unwrap()
                    ))
                );
            }
            other => panic!("{name}: unexpected walk result: {other:?}"),
        }
        // Nonzero t is legal, but cannot rescue any failing source assertion.
        for aux in lowered.auxiliaries() {
            extension[aux.cancellation.index()] = true;
        }
        assert_eq!(lowered.accepts(&extension), expected);
        assert_eq!(
            converted.satisfies(&physical(&lowered, &extension, converted.k_log)),
            expected
        );
    }
    assert!(!lowered.accepts(&vec![false; lowered.value_count()]));
    let honest = source
        .evaluate(&chip.honest_inputs(&[event(LoadByteOpcode::Lbu, 0x1_0000, 0, 0x42)]))
        .unwrap();
    // Mutate after evaluation: neither backend may trust generated values.
    for value in [
        chip.columns()[0].selected_byte[0],
        chip.columns()[0].result[0],
    ] {
        let mut forged = honest.clone();
        forged[value.index()] ^= true;
        assert!(!original.satisfies(&physical(&direct, &forged, original.k_log)));
        let extended = lowered.extend(&forged).unwrap();
        assert_eq!(lowered.project(&extended), Some(forged));
        assert!(!converted.satisfies(&physical(&lowered, &extended, converted.k_log)));
    }
    let mut padding = source.evaluate(&chip.honest_inputs(&[])).unwrap();
    padding[chip.columns()[0].result[0].index()] ^= true;
    assert!(!converted.satisfies(&physical(
        &lowered,
        &lowered.extend(&padding).unwrap(),
        converted.k_log
    )));
}

#[test]
fn advice_is_guarded_in_both_lowering_modes() {
    let chip = LoadByteCircuit::build(1);
    for mode in [LoweringMode::Direct, LoweringMode::RequireIdentityC] {
        let lowered = chip.lower(mode).unwrap();
        let converted = matrix(&lowered);
        for active in [false, true] {
            for advice in 0..256u64 {
                let inputs = fixtures::advice_inputs(&chip, active, advice);
                let expected = !active || advice == 0;
                assert_eq!(chip.circuit().evaluate(&inputs).is_ok(), expected);
                let extended = lowered.extend(&candidate(chip.circuit(), &inputs)).unwrap();
                assert_eq!(
                    converted.satisfies(&physical(&lowered, &extended, converted.k_log)),
                    expected,
                    "{mode:?}, active={active}, advice={advice}"
                );
            }
        }
    }
}
