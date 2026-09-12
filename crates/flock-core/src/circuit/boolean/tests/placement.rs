use super::*;
use support::Fields;

#[test]
fn automatic_layout_groups_interleaved_words_and_preserves_identity_c() {
    let compiled = CircuitBuilder::compile(
        Fields(vec![
            ("input", ColumnRole::Input, 2, 4),
            ("output", ColumnRole::Output, 2, 8),
            ("intermediate", ColumnRole::Witness, 2, 1),
        ]),
        |b, cols| {
            // Output bits are declared forwards, but defined backwards with work between them.
            b.define_linear(cols[1][1], cols[0][1]);
            b.define_and(cols[2][1], cols[0][0], cols[0][1]);
            b.define_linear(cols[1][0], cols[2][1]);
            b.define_linear(cols[2][0], cols[2][1]);
        },
    );
    let cols = compiled.columns();
    let circuit = compiled.circuit();
    let layout = circuit.layout().unwrap();
    assert_eq!(layout, circuit.layout().unwrap());
    assert_eq!(layout.value_position(circuit.one()), Some(0));
    assert_eq!(layout.value_position(cols[0][0]), Some(4));
    assert_eq!(layout.value_position(cols[0][1]), Some(5));
    assert_eq!(layout.value_position(cols[1][0]), Some(8));
    assert_eq!(layout.value_position(cols[1][1]), Some(9));
    // Internal witness fields follow evaluation order, not field bit order.
    assert_eq!(layout.value_position(cols[2][1]), Some(10));
    assert_eq!(layout.value_position(cols[2][0]), Some(11));
    assert_eq!(layout.useful_bits(), 12);
    let matrix = circuit.to_block_r1cs(4, 0, 0).unwrap();
    let plan = circuit.walk_plan().unwrap();
    assert!(matrix.c0_is_identity());
    for input in [[false, false], [false, true], [true, false], [true, true]] {
        let walked = plan.forward(&input, 4).unwrap();
        assert_eq!(walked.z, circuit.evaluate_r1cs(&input, 4).unwrap());
        assert_eq!(walked.a_z, matrix.apply_a(&walked.z));
        assert_eq!(walked.b_z, matrix.apply_b(&walked.z));
        assert_eq!(walked.c_z, matrix.apply_c(&walked.z));
        assert!(matrix.satisfies(&walked.z));
        assert_eq!(walked.z[8], input[0] & input[1]);
        assert_eq!(walked.z[9], input[1]);
        for gap in [1, 2, 3, 6, 7, 12, 15] {
            let mut invalid = walked.z.clone();
            invalid[gap] = true;
            assert!(!matrix.satisfies(&invalid));
        }
    }
    let weights: Vec<_> = (0..16).map(|i| crate::field::F128::new(i, i + 1)).collect();
    use crate::lincheck::LincheckCircuit;
    let alpha = crate::field::F128::new(13, 7);
    assert_eq!(
        plan.lincheck_circuit(4)
            .unwrap()
            .fold_alpha_batched(alpha, &weights),
        matrix
            .sparse_lincheck_circuit()
            .fold_alpha_batched(alpha, &weights)
    );

    let other = support::circuit(0, 0, |_, _| {});
    assert!(matches!(
        other.to_block_r1cs_with_layout(4, 0, 0, &layout),
        Err(R1csBuildError::InvalidLayout(LayoutError::WrongCircuit))
    ));
}

#[test]
fn unaligned_fields_keep_evaluation_order_and_alignment_overflow_is_reported() {
    let circuit = support::circuit(2, 1, |b, cols| {
        b.define_and(cols.witness[0], cols.input[0], cols.input[1]);
    });
    assert_eq!(circuit.layout().unwrap().value_positions(), &[0, 1, 2, 3]);
    let huge = 1usize << (usize::BITS - 1);
    let compiled = CircuitBuilder::compile(
        Fields(vec![
            ("a", ColumnRole::Input, 1, huge),
            ("b", ColumnRole::Input, 1, huge),
        ]),
        |_, _| {},
    );
    assert_eq!(
        compiled.circuit().layout(),
        Err(LayoutError::PositionOverflow)
    );
}

#[test]
fn rejects_unrepresentable_total_dimension() {
    let circuit = support::circuit(0, 0, |_, _| {});
    assert!(matches!(
        circuit.to_block_r1cs(0, 0, usize::BITS as usize),
        Err(R1csBuildError::DimensionOverflow)
    ));
}

#[test]
fn capacity_boundaries_are_exact_across_backends() {
    use crate::field::F128;

    let circuit = support::circuit(7, 0, |_, _| {});
    let inputs = [false; 7];
    let plan = circuit.walk_plan().unwrap();

    assert!(circuit.evaluate_r1cs(&inputs, 3).is_ok());
    assert!(circuit.to_block_r1cs(3, 0, 0).is_ok());
    assert!(plan.forward(&inputs, 3).is_ok());
    let weights = vec![F128::ZERO; 8];
    assert!(plan.transpose(&weights, &weights, &weights).is_ok());

    assert!(matches!(
        circuit.evaluate_r1cs(&inputs, 2),
        Err(EvaluationError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
    assert!(matches!(
        circuit.to_block_r1cs(2, 0, 0),
        Err(R1csBuildError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
    assert!(matches!(
        plan.forward(&inputs, 2),
        Err(WalkError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
    let too_small = vec![F128::ZERO; 4];
    assert!(matches!(
        plan.transpose(&too_small, &too_small, &too_small),
        Err(WalkError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
}
