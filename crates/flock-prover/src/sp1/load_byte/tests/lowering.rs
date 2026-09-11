use super::*;
use flock_core::circuit::boolean::WalkError;
use flock_core::r1cs::BlockR1cs;

mod costs;
mod interfaces;
mod proof;

fn physical(lowered: &LoweredCircuit, logical: &[bool], k_log: usize) -> Vec<bool> {
    assert_eq!(logical.len(), lowered.value_count());
    let mut values = vec![false; 1 << k_log];
    for (&position, &value) in lowered.layout().value_positions().iter().zip(logical) {
        values[position] = value;
    }
    values
}

fn matrix(lowered: &LoweredCircuit) -> BlockR1cs {
    let k_log = lowered
        .layout()
        .useful_bits()
        .next_power_of_two()
        .trailing_zeros() as usize;
    lowered.to_block_r1cs(k_log, 0, 0).unwrap()
}

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
    let assertions = source
        .rows()
        .iter()
        .filter(|row| row.kind() == RowKind::Constraint)
        .count();
    assert_eq!(lowered.auxiliaries().len(), assertions);
    assert_eq!(lowered.value_count(), source.value_count() + 2 * assertions);
    assert_eq!(lowered.rows().len(), lowered.value_count());
    assert_eq!(
        converted.const_pin,
        lowered.layout().value_position(lowered.one())
    );

    for (inputs, expected) in input_cases(&chip) {
        let candidate = candidate(source, &inputs);
        assert_eq!(lowered.evaluate(&inputs).is_ok(), expected);
        assert_eq!(direct.accepts(&candidate), expected);
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
            other => panic!("unexpected walk result: {other:?}"),
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
    let mut forged = candidate(source, &chip.honest_inputs(&[]));
    forged[chip.columns()[0].result[0].index()] ^= true;
    let extended = lowered.extend(&forged).unwrap();
    assert_eq!(lowered.project(&extended), Some(forged));
    assert!(!converted.satisfies(&physical(&lowered, &extended, converted.k_log)));
}

#[test]
fn converted_capacity_preserves_every_active_prefix_pattern() {
    let chip = LoadByteCircuit::build(4);
    let lowered = chip.lower(LoweringMode::RequireIdentityC).unwrap();
    let converted = matrix(&lowered);
    let real = event(LoadByteOpcode::Lbu, 0x1_0000, 0, 0x42);
    for mask in 0..16usize {
        let rows: Vec<_> = (0..4)
            .map(|i| (mask >> i & 1 != 0).then_some(real))
            .collect();
        let inputs = chip.encode_rows(&rows);
        let candidate = candidate(chip.circuit(), &inputs);
        let extended = lowered.extend(&candidate).unwrap();
        let prefix = mask & (mask + 1) == 0;
        assert_eq!(lowered.accepts(&extended), prefix);
        assert_eq!(
            converted.satisfies(&physical(&lowered, &extended, converted.k_log)),
            prefix
        );
    }
}

#[test]
fn converted_advice_remains_guarded_and_all_failures_are_enforced() {
    let chip = LoadByteCircuit::build(1);
    let lowered = chip.lower(LoweringMode::RequireIdentityC).unwrap();
    let converted = matrix(&lowered);
    for active in [false, true] {
        for advice in 0..256u64 {
            let inputs = fixtures::advice_inputs(&chip, active, advice);
            let extended = lowered.extend(&candidate(chip.circuit(), &inputs)).unwrap();
            assert_eq!(
                converted.satisfies(&physical(&lowered, &extended, converted.k_log)),
                !active || advice == 0
            );
        }
    }
}
