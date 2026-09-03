use super::*;

#[test]
fn materialization_is_explicit_and_exactly_one_row() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let b = builder.input();
    let initial_values = builder.value_count();
    let initial_rows = builder.row_count();

    let sum = builder.xor([a.expr(), b.expr()]);
    assert_eq!(builder.value_count(), initial_values);
    assert_eq!(builder.row_count(), initial_rows);

    // Copies and ordinary Rust container operations are shape-neutral.
    let copied = sum;
    let expressions = [sum, copied];
    assert_eq!(builder.value_count(), initial_values);
    assert_eq!(builder.row_count(), initial_rows);
    assert_eq!(expressions[0], expressions[1]);

    let sum_bit = builder.materialize(sum);
    assert_eq!(builder.value_count(), initial_values + 1);
    assert_eq!(builder.row_count(), initial_rows + 1);
    let _product = builder.and(sum_bit.expr(), a.expr());
    assert_eq!(builder.value_count(), initial_values + 2);
    assert_eq!(builder.row_count(), initial_rows + 2);

    let circuit = builder.finish();
    let defining_row = circuit
        .rows()
        .iter()
        .find(|row| row.defined_value() == Some(sum_bit.value_id()))
        .unwrap();
    assert_eq!(defining_row.kind(), RowKind::Materialize);
}

#[test]
fn structural_dag_survives_normalization_and_cancellation() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let b = builder.input();
    let ab = builder.xor([a.expr(), b.expr()]);
    let nested = builder.xor([ab, a.expr()]);
    let nested_id = nested.id();
    let ab_id = ab.id();
    let circuit = builder.finish();

    assert_eq!(
        circuit.expression(nested_id),
        Some(&ExpressionNode::Xor(vec![ab_id, a.expr().id()]))
    );
    assert_eq!(circuit.support(nested_id), Some(vec![b.value_id()]));
}

#[test]
fn evaluator_and_sparse_r1cs_agree() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let b = builder.input();
    let xor = builder.xor([a.expr(), b.expr()]);
    let xor_bit = builder.materialize(xor);
    let product = builder.and(xor, a.expr());
    let circuit = builder.finish();
    let r1cs = circuit.to_block_r1cs(3, 0, 0).unwrap();

    for inputs in [[false, false], [false, true], [true, false], [true, true]] {
        let witness = circuit.evaluate_r1cs(&inputs, 3).unwrap();
        assert_eq!(witness[xor_bit.value_id().index()], inputs[0] ^ inputs[1]);
        assert_eq!(
            witness[product.value_id().index()],
            (inputs[0] ^ inputs[1]) & inputs[0]
        );
        assert!(r1cs.satisfies(&witness));
        assert_eq!(
            r1cs.apply_a(&witness)[product.value_id().index()],
            inputs[0] ^ inputs[1]
        );
        assert_eq!(
            r1cs.apply_b(&witness)[product.value_id().index()],
            inputs[0]
        );
    }
}

#[test]
fn changing_materialization_changes_only_the_requested_boundary() {
    fn build(materialize: bool) -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        let a = builder.input();
        let b = builder.input();
        let sum = builder.xor([a.expr(), b.expr()]);
        let lhs = if materialize {
            builder.materialize(sum).expr()
        } else {
            sum
        };
        builder.and(lhs, a.expr());
        builder.finish()
    }

    let virtual_sum = build(false);
    let materialized_sum = build(true);
    assert_eq!(
        materialized_sum.value_count(),
        virtual_sum.value_count() + 1
    );
    assert_eq!(materialized_sum.row_count(), virtual_sum.row_count() + 1);
    assert_eq!(
        materialized_sum
            .rows()
            .iter()
            .filter(|row| row.kind() == RowKind::Materialize)
            .count(),
        1
    );
    let virtual_product = &virtual_sum.rows()[3];
    assert_eq!(virtual_product.kind(), RowKind::And);
    assert_eq!(
        virtual_sum.support(virtual_product.lhs()),
        Some(vec![virtual_sum.inputs()[0], virtual_sum.inputs()[1]])
    );
    let copy = &materialized_sum.rows()[3];
    let materialized_product = &materialized_sum.rows()[4];
    assert_eq!(copy.kind(), RowKind::Materialize);
    assert_eq!(materialized_product.kind(), RowKind::And);
    assert_eq!(
        materialized_sum.support(copy.lhs()),
        Some(vec![
            materialized_sum.inputs()[0],
            materialized_sum.inputs()[1]
        ])
    );
    assert_eq!(
        materialized_sum.support(materialized_product.lhs()),
        Some(vec![copy.defined_value().unwrap()])
    );
    assert_ne!(
        virtual_sum.structure_digest(),
        materialized_sum.structure_digest()
    );
    assert_ne!(
        virtual_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest(),
        materialized_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest()
    );
}

#[test]
fn constant_is_pinned_and_padding_is_forced_to_zero() {
    use crate::challenger::FsChallenger;
    use crate::field::F128;
    use crate::lincheck::{self, LincheckCircuit, QuirkyPoint, SkipPoint};

    let mut builder = CircuitBuilder::new();
    let input = builder.input();
    builder.materialize(input.expr());
    let circuit = builder.finish();
    let r1cs = circuit.to_block_r1cs(3, 0, 0).unwrap();

    assert_eq!(r1cs.const_pin, Some(circuit.one().index()));
    assert!(r1cs.c0_is_identity());
    assert!(r1cs.a_0.rows[3].is_empty());
    assert!(r1cs.b_0.rows[3].is_empty());

    let mut witness = circuit.evaluate_r1cs(&[true], 3).unwrap();
    assert!(r1cs.satisfies(&witness));
    witness[3] = true;
    assert!(!r1cs.satisfies(&witness));

    let lincheck_r1cs = circuit.to_block_r1cs(3, 0, 3).unwrap();
    let lincheck_circuit = lincheck_r1cs.sparse_lincheck_circuit();
    assert_eq!(lincheck_circuit.const_pin_col(), lincheck_r1cs.const_pin);
    let point = QuirkyPoint {
        z_skip: SkipPoint::Phi8(F128::ZERO),
        x_inner_rest: vec![F128::ZERO; lincheck_r1cs.k_log],
        x_outer: vec![F128::ZERO; lincheck_r1cs.n_log()],
    };
    let all_zero_witness = vec![0u8; 1 << (lincheck_r1cs.m - 3)];

    let unpinned = lincheck::SparseMatrixCircuit::new(&lincheck_r1cs.a_0, &lincheck_r1cs.b_0);
    let mut unpinned_prover = FsChallenger::new(b"dsl-without-const-pin");
    let (unpinned_proof, _) = lincheck::prove(
        &all_zero_witness,
        lincheck_r1cs.m,
        lincheck_r1cs.k_log,
        0,
        &unpinned,
        &point,
        &mut unpinned_prover,
    );
    let mut unpinned_verifier = FsChallenger::new(b"dsl-without-const-pin");
    lincheck::verify(
        lincheck_r1cs.m,
        lincheck_r1cs.k_log,
        0,
        &unpinned,
        &point,
        F128::ZERO,
        F128::ZERO,
        &unpinned_proof,
        &mut unpinned_verifier,
    )
    .expect("the all-zero relation is valid without the statement-level pin");

    let mut prover_challenger = FsChallenger::new(b"dsl-const-pin");
    let (proof, _) = lincheck::prove(
        &all_zero_witness,
        lincheck_r1cs.m,
        lincheck_r1cs.k_log,
        0,
        &lincheck_circuit,
        &point,
        &mut prover_challenger,
    );
    let mut verifier_challenger = FsChallenger::new(b"dsl-const-pin");
    assert!(
        lincheck::verify(
            lincheck_r1cs.m,
            lincheck_r1cs.k_log,
            0,
            &lincheck_circuit,
            &point,
            F128::ZERO,
            F128::ZERO,
            &proof,
            &mut verifier_challenger,
        )
        .is_err(),
        "the ONE pin must reject an all-zero lincheck witness"
    );
}

#[test]
fn general_constraint_emits_non_identity_c_without_materializing() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let b = builder.input();
    let values_before = builder.value_count();
    let assertion = builder.assert_zero_product(a.expr(), b.expr());
    assert_eq!(builder.value_count(), values_before);

    let circuit = builder.finish();
    assert_eq!(
        circuit.rows()[assertion.index()].kind(),
        RowKind::Constraint
    );
    let r1cs = circuit.to_block_r1cs(2, 0, 0).unwrap();
    assert!(!r1cs.c0_is_identity());

    let witness = circuit.evaluate_r1cs(&[true, false], 2).unwrap();
    assert!(r1cs.satisfies(&witness));
    assert_eq!(
        circuit.evaluate(&[true, true]),
        Err(EvaluationError::UnsatisfiedRow(assertion))
    );
    // The same rejected assignment also fails the emitted relation.
    let mut rejected = vec![false; 4];
    rejected[circuit.one().index()] = true;
    rejected[a.value_id().index()] = true;
    rejected[b.value_id().index()] = true;
    assert!(!r1cs.satisfies(&rejected));
}

#[test]
fn ripple_carry_example_keeps_linear_carries_virtual() {
    fn full_adder(
        builder: &mut CircuitBuilder,
        a: LinearExpr,
        b: LinearExpr,
        carry_in: LinearExpr,
    ) -> (LinearExpr, LinearExpr) {
        let a_xor_b = builder.xor([a, b]);
        let sum = builder.xor([a_xor_b, carry_in]);
        let a_and_b = builder.and(a, b);
        let carry_and_xor = builder.and(carry_in, a_xor_b);
        let carry_out = builder.xor([a_and_b.expr(), carry_and_xor.expr()]);
        (sum, carry_out)
    }

    let mut builder = CircuitBuilder::new();
    let a = [builder.input(), builder.input()];
    let b = [builder.input(), builder.input()];
    let carry_in = builder.input();

    let (sum_0, carry_0) = full_adder(&mut builder, a[0].expr(), b[0].expr(), carry_in.expr());
    let values_after_low_bit = builder.value_count();
    let (sum_1, carry_1) = full_adder(&mut builder, a[1].expr(), b[1].expr(), carry_0);
    // Passing the virtual carry into the next bit does not materialize it:
    // only the two nonlinear products allocate values in each full adder.
    assert_eq!(builder.value_count(), values_after_low_bit + 2);

    let sum_0 = builder.materialize(sum_0);
    let sum_1 = builder.materialize(sum_1);
    let carry_1 = builder.materialize(carry_1);
    let circuit = builder.finish();
    let r1cs = circuit.to_block_r1cs(4, 0, 0).unwrap();
    assert!(r1cs.c0_is_identity());

    for input in 0u8..32 {
        let bits: [bool; 5] = std::array::from_fn(|i| (input >> i) & 1 == 1);
        let witness = circuit.evaluate_r1cs(&bits, 4).unwrap();
        assert!(r1cs.satisfies(&witness));
        let output = u8::from(witness[sum_0.value_id().index()])
            | (u8::from(witness[sum_1.value_id().index()]) << 1)
            | (u8::from(witness[carry_1.value_id().index()]) << 2);
        let a_value = (input & 1) | (((input >> 1) & 1) << 1);
        let b_value = ((input >> 2) & 1) | (((input >> 3) & 1) << 1);
        let carry_value = (input >> 4) & 1;
        assert_eq!(output, a_value + b_value + carry_value);
    }
}

#[test]
fn named_ports_and_layout_stay_out_of_the_arithmetic_definition() {
    // This is the intended authoring style: named typed boundaries and
    // direct bit/expression operands, with no logical or physical IDs.
    let mut builder = CircuitBuilder::new();
    let a = builder.input_bits::<2>("a");
    let b = builder.input_bits::<2>("b");
    let sum_0_expr = builder.xor2(a[0], b[0]);
    let sum_1_expr = builder.xor2(a[1], b[1]);
    let sum = [
        builder.materialize(sum_0_expr),
        builder.materialize(sum_1_expr),
    ];
    builder.output("sum", sum);
    let circuit = builder.finish();

    assert_eq!(circuit.port("a").unwrap().direction(), PortDirection::Input);
    assert_eq!(
        circuit.port("sum").unwrap().direction(),
        PortDirection::Output
    );

    // Compatibility placement is a separate concern. Reserving the first
    // four positions and moving ports does not change the circuit code.
    let mut layout = circuit.layout();
    layout.reserve(0..4).unwrap();
    layout.place_port("a", 8).unwrap();
    layout.place_port("sum", 12).unwrap();
    let layout = layout.finish().unwrap();
    assert_eq!(layout.value_position(a[0].value_id()), Some(8));
    assert_eq!(layout.value_position(a[1].value_id()), Some(9));
    assert_eq!(layout.value_position(sum[0].value_id()), Some(12));
    assert_eq!(layout.value_position(sum[1].value_id()), Some(13));
    assert_eq!(layout.useful_bits(), 14);

    let r1cs = circuit.to_block_r1cs_with_layout(4, 0, 0, &layout).unwrap();
    let witness = circuit
        .evaluate_r1cs_with_layout(&[true, false, false, true], 4, &layout)
        .unwrap();
    assert!(r1cs.c0_is_identity());
    assert!(r1cs.satisfies(&witness));
    assert!(witness[12]);
    assert!(witness[13]);
    assert_eq!(r1cs.const_pin, layout.value_position(circuit.one()));
    assert_eq!(r1cs.c_0.rows[14], vec![14]);
    assert_eq!(r1cs.c_0.rows[15], vec![15]);
}

#[test]
fn word_helpers_are_virtual_until_materialize_word() {
    let mut builder = CircuitBuilder::new();
    let input = builder.input_word::<8>("input");
    let values_before = builder.value_count();
    let rows_before = builder.row_count();
    let rotated = builder.rotate_right(input, 2);
    let shifted = builder.shift_right(input, 3);
    let mixed = builder.xor2_words(rotated, shifted);
    assert_eq!(builder.value_count(), values_before);
    assert_eq!(builder.row_count(), rows_before);

    let output = builder.materialize_word(mixed);
    assert_eq!(builder.value_count(), values_before + 8);
    assert_eq!(builder.row_count(), rows_before + 8);
    builder.output_word("output", output);
    let circuit = builder.finish();
    assert_eq!(
        circuit.port("input").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 1 }
    );
    assert_eq!(
        circuit.port("output").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 1 }
    );
    let r1cs = circuit.to_block_r1cs(5, 0, 0).unwrap();

    for value in 0u8..=u8::MAX {
        let input_bits: [bool; 8] = std::array::from_fn(|i| value >> i & 1 == 1);
        let witness = circuit.evaluate_r1cs(&input_bits, 5).unwrap();
        assert!(r1cs.satisfies(&witness));
        let output_value = output.iter().enumerate().fold(0u8, |acc, (i, bit)| {
            acc | (u8::from(witness[bit.value_id().index()]) << i)
        });
        assert_eq!(output_value, value.rotate_right(2) ^ (value >> 3));
    }
}

#[test]
fn fixed_words_are_named_and_constrained() {
    let mut builder = CircuitBuilder::new();
    let fixed = builder.fixed_word::<8>("iv", 0xa5);
    let circuit = builder.finish();
    assert_eq!(
        circuit.port("iv").unwrap().direction(),
        PortDirection::Fixed
    );
    assert_eq!(
        circuit.port("iv").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 1 }
    );
    let r1cs = circuit.to_block_r1cs(4, 0, 0).unwrap();
    let witness = circuit.evaluate_r1cs(&[], 4).unwrap();
    assert!(r1cs.satisfies(&witness));
    for (i, bit) in fixed.iter().enumerate() {
        assert_eq!(witness[bit.value_id().index()], 0xa5 >> i & 1 == 1);
    }
    let mut mutated = witness;
    mutated[fixed[3].value_id().index()] ^= true;
    assert!(!r1cs.satisfies(&mutated));
}

#[test]
fn word_ports_enforce_contiguous_aligned_layout() {
    let mut builder = CircuitBuilder::new();
    let word = builder.input_word_aligned::<8>("word", 8);
    let bits = builder.input_bits::<8>("bits");
    let circuit = builder.finish();

    assert_eq!(
        circuit.port("word").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 8 }
    );
    assert_eq!(circuit.port("bits").unwrap().encoding(), PortEncoding::Bits);
    assert!(matches!(
        circuit.to_block_r1cs(5, 0, 0),
        Err(R1csBuildError::InvalidLayout(
            LayoutError::MisalignedPort {
                ref name,
                start: 1,
                alignment_bits: 8,
            }
        )) if name == "word"
    ));
    assert!(matches!(
        circuit.walk_plan(),
        Err(LayoutError::MisalignedPort {
            ref name,
            start: 1,
            alignment_bits: 8,
        }) if name == "word"
    ));

    let mut layout = circuit.layout();
    assert!(matches!(
        layout.place_port("word", 4),
        Err(LayoutError::MisalignedPort {
            ref name,
            start: 4,
            alignment_bits: 8,
        }) if name == "word"
    ));
    layout.place_port("word", 8).unwrap();
    layout.place_port("bits", 24).unwrap();
    let layout = layout.finish().unwrap();
    for (offset, bit) in word.iter().enumerate() {
        assert_eq!(layout.value_position(bit.value_id()), Some(8 + offset));
    }
    assert_eq!(layout.value_position(bits[0].value_id()), Some(24));

    let mut scattered = circuit.layout();
    for (offset, bit) in word.iter().enumerate() {
        scattered
            .place_definition(bit.value_id(), 8 + 2 * offset)
            .unwrap();
    }
    assert!(matches!(
        scattered.finish(),
        Err(LayoutError::NonContiguousPort {
            ref name,
            offset: 1,
            expected: 9,
            actual: 10,
        }) if name == "word"
    ));
}

#[test]
fn port_encoding_is_structural_but_not_arithmetic() {
    fn bits() -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        builder.input_bits::<8>("value");
        builder.finish()
    }

    fn word(alignment_bits: usize) -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        builder.input_word_aligned::<8>("value", alignment_bits);
        builder.finish()
    }

    let bits = bits();
    let unaligned_word = word(1);
    let aligned_word = word(8);
    assert_ne!(bits.structure_digest(), unaligned_word.structure_digest());
    assert_ne!(
        unaligned_word.structure_digest(),
        aligned_word.structure_digest()
    );
    assert_eq!(
        bits.to_block_r1cs(4, 0, 0).unwrap().statement_digest(),
        unaligned_word
            .to_block_r1cs(4, 0, 0)
            .unwrap()
            .statement_digest()
    );
}

#[test]
fn layout_rejects_overlaps_and_cross_circuit_reuse() {
    let mut first = CircuitBuilder::new();
    let bits = first.input_bits::<2>("bits");
    let first = first.finish();
    let mut layout = first.layout();
    layout.place_definition(bits[0].value_id(), 7).unwrap();
    assert!(matches!(
        layout.place_definition(bits[1].value_id(), 7),
        Err(LayoutError::PositionOccupied {
            kind: PositionKind::Column,
            position: 7,
            ..
        })
    ));

    let first_layout = layout.finish().unwrap();
    let mut second_builder = CircuitBuilder::new();
    let second_bit = second_builder.input();
    let second = second_builder.finish();
    let mut second_layout = second.layout();
    assert!(matches!(
        second_layout.place_value(bits[0].value_id(), 0),
        Err(LayoutError::WrongCircuit)
    ));
    let foreign_row = first.definition_row(bits[0].value_id()).unwrap();
    assert!(matches!(
        second_layout.place_row(foreign_row, 0),
        Err(LayoutError::WrongCircuit)
    ));
    assert_eq!(
        PhysicalLayout::source_order(&second).value_position(bits[0].value_id()),
        None
    );
    assert_eq!(second.expression(bits[0].expr().id()), None);
    second_layout
        .place_definition(second_bit.value_id(), 0)
        .unwrap();
    assert!(matches!(
        second.to_block_r1cs_with_layout(3, 0, 0, &first_layout),
        Err(R1csBuildError::InvalidLayout(LayoutError::WrongCircuit))
    ));
}

#[test]
fn sha_sized_layouts_are_deterministic() {
    const N: usize = 25_500;
    fn build() -> (BooleanCircuit, Vec<Bit>) {
        let mut builder = CircuitBuilder::new();
        let bits = (0..N).map(|_| builder.input()).collect();
        (builder.finish(), bits)
    }

    let (first, first_bits) = build();
    let automatic = first.layout().finish().unwrap();
    assert_eq!(automatic.useful_bits(), N + 1);
    let mut first_layout = first.layout();
    for (i, bit) in first_bits.into_iter().enumerate() {
        first_layout
            .place_definition(bit.value_id(), N - 1 - i)
            .unwrap();
    }
    let first_layout = first_layout.finish().unwrap();

    let (second, second_bits) = build();
    let mut second_layout = second.layout();
    for (i, bit) in second_bits.into_iter().enumerate() {
        second_layout
            .place_definition(bit.value_id(), N - 1 - i)
            .unwrap();
    }
    let second_layout = second_layout.finish().unwrap();
    assert_eq!(first_layout, second_layout);
    assert_eq!(first_layout.useful_bits(), N + 1);
}

#[test]
fn rejects_unrepresentable_total_dimension() {
    let circuit = CircuitBuilder::new().finish();
    assert!(matches!(
        circuit.to_block_r1cs(0, 0, usize::BITS as usize),
        Err(R1csBuildError::DimensionOverflow)
    ));
}

#[test]
fn capacity_boundaries_are_exact_across_backends() {
    use crate::field::F128;

    let mut builder = CircuitBuilder::new();
    for _ in 0..7 {
        builder.input();
    }
    let circuit = builder.finish();
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

#[test]
fn deterministic_random_circuits_agree_with_bool_and_packed_matrices() {
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    for circuit_seed in 1u64..=8 {
        let mut state = circuit_seed;
        let mut builder = CircuitBuilder::new();
        let inputs = builder.input_bits::<8>("input");
        let one = builder.one();
        let mut expressions: Vec<LinearExpr> = inputs.iter().map(|bit| bit.expr()).collect();
        for _ in 0..40 {
            let lhs = expressions[next(&mut state) as usize % expressions.len()];
            let rhs = expressions[next(&mut state) as usize % expressions.len()];
            match next(&mut state) % 4 {
                0 => {
                    let expression = builder.xor2(lhs, rhs);
                    expressions.push(expression);
                }
                1 => {
                    let bit = builder.and(lhs, rhs);
                    expressions.push(bit.expr());
                }
                2 => {
                    let bit = builder.materialize(lhs);
                    expressions.push(bit.expr());
                }
                _ => {
                    // A general-C tautology exercises non-definitional rows
                    // without restricting the random input assignment.
                    builder.constrain(lhs, one, lhs);
                }
            }
        }
        let output = builder.materialize(*expressions.last().unwrap());
        builder.output("output", [output]);
        let circuit = builder.finish();
        let r1cs = circuit.to_block_r1cs(7, 0, 0).unwrap();
        let mut layout = circuit.layout();
        for &row_id in &circuit.definition_rows {
            let value = circuit.rows[row_id.index].defined_value.unwrap();
            layout
                .place_definition(value, circuit.value_count() - 1 - value.index)
                .unwrap();
        }
        let mut constraint_position = circuit.value_count();
        for row in circuit
            .rows()
            .iter()
            .filter(|row| row.kind() == RowKind::Constraint)
        {
            layout.place_row(row.id(), constraint_position).unwrap();
            constraint_position += 1;
        }
        let layout = layout.finish().unwrap();
        let permuted_r1cs = circuit.to_block_r1cs_with_layout(7, 0, 0, &layout).unwrap();

        for _ in 0..16 {
            let input: [bool; 8] = std::array::from_fn(|_| next(&mut state) & 1 == 1);
            let logical = circuit.evaluate(&input).unwrap();
            let witness = circuit.evaluate_r1cs(&input, 7).unwrap();
            assert!(r1cs.satisfies(&witness));
            let packed = crate::pcs::pack::pack_witness(&witness, 7);
            assert!(r1cs.satisfies_packed(&packed));

            let a = r1cs.apply_a(&witness);
            let b = r1cs.apply_b(&witness);
            let c = r1cs.apply_c(&witness);
            for row in circuit.rows() {
                let index = row.id().index();
                assert_eq!(a[index], circuit.eval_expression(row.lhs(), &logical));
                assert_eq!(b[index], circuit.eval_expression(row.rhs(), &logical));
                assert_eq!(c[index], circuit.eval_expression(row.result(), &logical));
            }

            let permuted_witness = circuit
                .evaluate_r1cs_with_layout(&input, 7, &layout)
                .unwrap();
            assert!(permuted_r1cs.satisfies(&permuted_witness));
            let a = permuted_r1cs.apply_a(&permuted_witness);
            let b = permuted_r1cs.apply_b(&permuted_witness);
            let c = permuted_r1cs.apply_c(&permuted_witness);
            for row in circuit.rows() {
                let position = layout.row_position(row.id()).unwrap();
                assert_eq!(a[position], circuit.eval_expression(row.lhs(), &logical));
                assert_eq!(b[position], circuit.eval_expression(row.rhs(), &logical));
                assert_eq!(c[position], circuit.eval_expression(row.result(), &logical));
            }
        }

        let mut witness = circuit.evaluate_r1cs(&[false; 8], 7).unwrap();
        let computed = circuit
            .rows()
            .iter()
            .find(|row| matches!(row.kind(), RowKind::And | RowKind::Materialize))
            .and_then(Row::defined_value)
            .unwrap();
        witness[computed.index()] ^= true;
        assert!(!r1cs.satisfies(&witness));
    }
}

#[test]
fn structure_and_relation_digests_are_deterministic_and_golden() {
    fn build() -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        let input = builder.input_bits::<2>("input");
        let sum_expr = builder.xor2(input[0], input[1]);
        let sum = builder.materialize(sum_expr);
        builder.assert_zero_product(input[0], input[1]);
        builder.output("sum", [sum]);
        builder.finish()
    }

    let first = build();
    let second = build();
    assert_eq!(first.structure_digest(), second.structure_digest());
    let first_layout = PhysicalLayout::source_order(&first);
    let second_layout = PhysicalLayout::source_order(&second);
    assert_eq!(first_layout, second_layout);
    assert_eq!(
        first.structure_layout_digest(&first_layout).unwrap(),
        second.structure_layout_digest(&second_layout).unwrap()
    );
    assert_eq!(
        first.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
        second.to_block_r1cs(3, 0, 0).unwrap().statement_digest()
    );
    assert_eq!(
        first.structure_digest(),
        [
            132, 17, 99, 150, 177, 225, 101, 6, 129, 196, 80, 54, 185, 56, 22, 18, 174, 130, 22,
            164, 248, 233, 108, 114, 130, 93, 11, 188, 243, 65, 167, 185,
        ]
    );
    assert_eq!(
        first.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
        [
            106, 16, 156, 236, 111, 130, 214, 34, 173, 41, 61, 220, 52, 226, 61, 162, 41, 193, 169,
            235, 128, 224, 120, 125, 227, 239, 50, 114, 147, 168, 88, 148,
        ]
    );
}

#[test]
fn structure_digest_is_process_and_thread_deterministic() {
    const CHILD_ENV: &str = "FLOCK_BOOLEAN_DIGEST_TEST_CHILD";
    const MARKER: &str = "FLOCK_STRUCTURE_DIGEST=";

    fn digest() -> [u8; 32] {
        let mut builder = CircuitBuilder::new();
        let input = builder.input_word_aligned::<8>("input", 8);
        let mixed = builder.xor3(input[0], input[3], input[7]);
        let output = builder.materialize(mixed);
        builder.output("output", [output]);
        builder.finish().structure_digest()
    }

    if std::env::var_os(CHILD_ENV).is_some() {
        println!("{MARKER}{:?}", digest());
        return;
    }

    let run = |rayon_threads: &str| {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "circuit::boolean::tests::structure_digest_is_process_and_thread_deterministic",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .env("RAYON_NUM_THREADS", rayon_threads)
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        stdout
            .split_once(MARKER)
            .map(|(_, digest)| digest.lines().next().unwrap().to_owned())
            .expect("child process did not print its structure digest")
    };

    assert_eq!(run("1"), run("4"));
    assert_eq!(run("2"), format!("{:?}", digest()));
}

#[test]
#[should_panic(expected = "expression belongs to another circuit builder")]
fn cross_builder_handles_fail_closed() {
    let mut first = CircuitBuilder::new();
    let bit = first.input();
    let mut second = CircuitBuilder::new();
    second.materialize(bit.expr());
}
