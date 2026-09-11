use super::*;

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
