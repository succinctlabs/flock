use crate::circuit::boolean::{BooleanCircuit, CircuitBuilder, LinearExpr, WalkError};
use crate::field::F128;
use crate::lincheck::LincheckCircuit;
use crate::r1cs::SparseBinaryMatrix;

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random_circuit(seed: u64) -> BooleanCircuit {
    let mut state = seed;
    let mut builder = CircuitBuilder::new();
    let inputs = builder.input_bits::<8>("input");
    let mut expressions: Vec<LinearExpr> = inputs.iter().map(|bit| bit.expr()).collect();

    for step in 0..48 {
        let lhs = expressions[next(&mut state) as usize % expressions.len()];
        let rhs = expressions[next(&mut state) as usize % expressions.len()];
        match next(&mut state) % 4 {
            0 | 1 => expressions.push(builder.xor2(lhs, rhs)),
            2 => expressions.push(builder.and(lhs, rhs).expr()),
            _ => expressions.push(builder.materialize(lhs).expr()),
        }
        if step % 11 == 0 {
            let expression = expressions[next(&mut state) as usize % expressions.len()];
            let one = builder.one();
            builder.constrain(expression, one, expression);
        }
    }
    let output = builder.materialize(*expressions.last().unwrap());
    builder.output("output", [output]);
    builder.finish()
}

fn capacity_log(required: usize) -> usize {
    required.next_power_of_two().trailing_zeros() as usize
}

fn weights(len: usize, seed: u64) -> Vec<F128> {
    let mut state = seed;
    (0..len)
        .map(|_| F128::new(next(&mut state), next(&mut state)))
        .collect()
}

fn sparse_transpose(matrix: &SparseBinaryMatrix, row_weights: &[F128]) -> Vec<F128> {
    assert_eq!(matrix.num_rows, row_weights.len());
    let mut result = vec![F128::ZERO; matrix.num_cols];
    for (row, &weight) in matrix.rows.iter().zip(row_weights) {
        for &column in row {
            result[column] += weight;
        }
    }
    result
}

fn add_three(mut a: Vec<F128>, b: &[F128], c: &[F128]) -> Vec<F128> {
    for ((value, &b), &c) in a.iter_mut().zip(b).zip(c) {
        *value += b;
        *value += c;
    }
    a
}

fn dot_bits(bits: &[bool], weights: &[F128]) -> F128 {
    bits.iter()
        .zip(weights)
        .filter(|(bit, _)| **bit)
        .fold(F128::ZERO, |sum, (_, &weight)| sum + weight)
}

#[test]
fn forward_walk_matches_sparse_matrices_on_random_circuits() {
    for seed in 0..16 {
        let circuit = random_circuit(0x51a7_0000 + seed);
        let plan = circuit.walk_plan().unwrap();
        let k_log = capacity_log(plan.useful_bits());
        let capacity = 1 << k_log;
        let inputs: Vec<bool> = (0..circuit.inputs().len())
            .map(|index| (seed as usize + index * 3) & 1 == 1)
            .collect();

        let walked = plan.forward(&inputs, k_log).unwrap();
        let r1cs = circuit.to_block_r1cs(k_log, 0, 0).unwrap();
        assert_eq!(walked.z, circuit.evaluate_r1cs(&inputs, k_log).unwrap());
        assert_eq!(walked.a_z, r1cs.apply_a(&walked.z));
        assert_eq!(walked.b_z, r1cs.apply_b(&walked.z));
        assert_eq!(walked.c_z, r1cs.apply_c(&walked.z));
        assert!(r1cs.satisfies(&walked.z));
        assert_eq!(walked.z.len(), capacity);
    }
}

#[test]
fn edge_case_xors_and_nested_general_c_match_sparse_walks() {
    let mut builder = CircuitBuilder::new();
    let inputs = builder.input_bits::<2>("input");
    let empty = builder.xor(std::iter::empty::<LinearExpr>());
    let shared = builder.xor2(inputs[0], inputs[1]);
    let duplicate = builder.xor([shared, shared, inputs[1].expr()]);
    let single = builder.xor([duplicate]);
    let nested_c = builder.xor([shared, inputs[0].expr(), inputs[0].expr()]);
    let one = builder.one();
    builder.constrain(shared, one, nested_c);
    let outputs = [builder.materialize(empty), builder.materialize(single)];
    builder.output("output", outputs);
    let circuit = builder.finish();

    assert_eq!(circuit.support(empty.id()), Some(vec![]));
    assert_eq!(
        circuit.support(duplicate.id()),
        Some(vec![inputs[1].value_id()])
    );
    assert_eq!(
        circuit.support(single.id()),
        circuit.support(duplicate.id())
    );
    assert_eq!(circuit.support(nested_c.id()), circuit.support(shared.id()));

    let plan = circuit.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let r1cs = circuit.to_block_r1cs(k_log, 0, 0).unwrap();
    for input in [[false, false], [false, true], [true, false], [true, true]] {
        let walked = plan.forward(&input, k_log).unwrap();
        assert_eq!(walked.a_z, r1cs.apply_a(&walked.z));
        assert_eq!(walked.b_z, r1cs.apply_b(&walked.z));
        assert_eq!(walked.c_z, r1cs.apply_c(&walked.z));
    }

    let e_a = weights(capacity, 0xed6e_a001);
    let e_b = weights(capacity, 0xed6e_b001);
    let e_c = weights(capacity, 0xed6e_c001);
    let expected = add_three(
        sparse_transpose(&r1cs.a_0, &e_a),
        &sparse_transpose(&r1cs.b_0, &e_b),
        &sparse_transpose(&r1cs.c_0, &e_c),
    );
    assert_eq!(plan.transpose(&e_a, &e_b, &e_c).unwrap(), expected);
}

#[test]
fn walks_match_sparse_oracles_under_permuted_layouts() {
    for seed in 0..8usize {
        let circuit = random_circuit(0xc01a_0000 + seed as u64);
        let values: Vec<_> = circuit
            .rows()
            .iter()
            .filter_map(|row| row.defined_value())
            .collect();
        let mut layout = circuit.layout();
        for (index, &value) in values.iter().enumerate() {
            layout
                .place_value(value, (values.len() - 1 - index + seed) % values.len())
                .unwrap();
        }
        for (index, row) in circuit.rows().iter().enumerate() {
            layout
                .place_row(row.id(), (index + seed) % circuit.row_count())
                .unwrap();
        }
        let layout = layout.finish().unwrap();
        let plan = circuit.walk_plan_with_layout(&layout).unwrap();
        let k_log = capacity_log(plan.useful_bits());
        let capacity = 1 << k_log;
        let inputs: Vec<bool> = (0..circuit.inputs().len())
            .map(|index| (seed + index) & 1 == 1)
            .collect();
        let walked = plan.forward(&inputs, k_log).unwrap();
        let r1cs = circuit
            .to_block_r1cs_with_layout(k_log, 0, 0, &layout)
            .unwrap();
        assert_eq!(walked.a_z, r1cs.apply_a(&walked.z));
        assert_eq!(walked.b_z, r1cs.apply_b(&walked.z));
        assert_eq!(walked.c_z, r1cs.apply_c(&walked.z));

        let e_a = weights(capacity, 0xa000 + seed as u64);
        let e_b = weights(capacity, 0xb000 + seed as u64);
        let e_c = weights(capacity, 0xc000 + seed as u64);
        let expected = add_three(
            sparse_transpose(&r1cs.a_0, &e_a),
            &sparse_transpose(&r1cs.b_0, &e_b),
            &sparse_transpose(&r1cs.c_0, &e_c),
        );
        assert_eq!(plan.transpose(&e_a, &e_b, &e_c).unwrap(), expected);
    }
}

#[test]
fn transpose_walk_handles_reserved_holes_and_rejects_bad_requests() {
    let circuit = random_circuit(0x0dd5_0000);
    let mut layout = circuit.layout();
    layout.reserve(3..11).unwrap();
    let layout = layout.finish().unwrap();
    let plan = circuit.walk_plan_with_layout(&layout).unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let r1cs = circuit
        .to_block_r1cs_with_layout(k_log, 0, 0, &layout)
        .unwrap();
    let e_a = weights(capacity, 0xa11e);
    let e_b = weights(capacity, 0xb11e);
    let e_c = weights(capacity, 0xc11e);
    let expected = add_three(
        sparse_transpose(&r1cs.a_0, &e_a),
        &sparse_transpose(&r1cs.b_0, &e_b),
        &sparse_transpose(&r1cs.c_0, &e_c),
    );
    assert_eq!(plan.transpose(&e_a, &e_b, &e_c).unwrap(), expected);

    assert!(matches!(
        plan.forward(&[], k_log),
        Err(WalkError::InputCount { .. })
    ));
    assert!(matches!(
        plan.forward(&[false; 8], 0),
        Err(WalkError::Capacity { .. })
    ));
    assert!(matches!(
        plan.transpose(&e_a[..capacity - 1], &e_b, &e_c),
        Err(WalkError::WeightCount { .. })
    ));
    assert_eq!(
        plan.transpose(
            &e_a[..capacity - 1],
            &e_b[..capacity - 1],
            &e_c[..capacity - 1]
        ),
        Err(WalkError::InvalidTransposeSize(capacity - 1))
    );
}

#[test]
fn transpose_walk_handles_columns_without_rows() {
    let circuit = random_circuit(0xc010_0000);
    let values: Vec<_> = circuit
        .rows()
        .iter()
        .filter_map(|row| row.defined_value())
        .collect();
    let mut layout = circuit.layout();
    for (position, row) in circuit.rows().iter().enumerate() {
        layout.place_row(row.id(), position).unwrap();
    }
    let occupied = circuit.row_count() + circuit.value_count();
    let gap = if (occupied + 1).is_power_of_two() {
        2
    } else {
        1
    };
    for (index, &value) in values.iter().enumerate() {
        layout
            .place_value(value, circuit.row_count() + gap + index)
            .unwrap();
    }
    let layout = layout.finish().unwrap();
    let plan = circuit.walk_plan_with_layout(&layout).unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let r1cs = circuit
        .to_block_r1cs_with_layout(k_log, 0, 0, &layout)
        .unwrap();
    let inputs = [true, false, true, false, true, false, true, false];
    let walked = plan.forward(&inputs, k_log).unwrap();
    assert_eq!(walked.a_z, r1cs.apply_a(&walked.z));
    assert_eq!(walked.b_z, r1cs.apply_b(&walked.z));
    assert_eq!(walked.c_z, r1cs.apply_c(&walked.z));

    let mut mutated_hole = walked.z.clone();
    mutated_hole[circuit.row_count()] = true;
    assert!(!r1cs.satisfies(&mutated_hole));
    let mut mutated_suffix = walked.z;
    mutated_suffix[layout.useful_bits()] = true;
    assert!(!r1cs.satisfies(&mutated_suffix));

    let e_a = weights(capacity, 0xa010);
    let e_b = weights(capacity, 0xb010);
    let e_c = weights(capacity, 0xc010);
    let expected = add_three(
        sparse_transpose(&r1cs.a_0, &e_a),
        &sparse_transpose(&r1cs.b_0, &e_b),
        &sparse_transpose(&r1cs.c_0, &e_c),
    );
    assert_eq!(plan.transpose(&e_a, &e_b, &e_c).unwrap(), expected);
}

#[test]
fn forward_walk_enforces_general_constraints() {
    let mut builder = CircuitBuilder::new();
    let input = builder.input();
    let row = builder.assert_zero(input);
    let plan = builder.finish().walk_plan().unwrap();
    let error = plan.forward(&[true], capacity_log(plan.useful_bits()));
    assert_eq!(error, Err(WalkError::UnsatisfiedRow(row)));
}

#[test]
fn reverse_walk_satisfies_each_transpose_identity() {
    let circuit = random_circuit(0x7a4e_5e00);
    let plan = circuit.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let inputs = [true, false, true, true, false, false, true, false];
    let walked = plan.forward(&inputs, k_log).unwrap();

    for (side, row_values, seed) in [
        (0, walked.a_z.as_slice(), 11),
        (1, walked.b_z.as_slice(), 22),
        (2, walked.c_z.as_slice(), 33),
    ] {
        let e = weights(capacity, seed);
        let zero = vec![F128::ZERO; capacity];
        let transposed = match side {
            0 => plan.transpose(&e, &zero, &zero).unwrap(),
            1 => plan.transpose(&zero, &e, &zero).unwrap(),
            _ => plan.transpose(&zero, &zero, &e).unwrap(),
        };
        assert_eq!(dot_bits(row_values, &e), dot_bits(&walked.z, &transposed));
    }
}

#[test]
fn lincheck_adapter_matches_sparse_fold_and_carries_the_pin() {
    let circuit = random_circuit(0x11c0_0001);
    let plan = circuit.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let r1cs = circuit.to_block_r1cs(k_log, 0, 0).unwrap();
    let adapter = plan.lincheck_circuit(k_log).unwrap();
    let sparse = r1cs.sparse_lincheck_circuit();
    let eq = weights(1 << k_log, 0x11c0_e001);
    let alpha = F128::new(0x1234, 0x5678);

    assert_eq!(adapter.n_cols(), sparse.n_cols());
    assert_eq!(adapter.const_pin_col(), r1cs.const_pin);
    assert_eq!(
        adapter.fold_alpha_batched(alpha, &eq),
        sparse.fold_alpha_batched(alpha, &eq)
    );
    assert!(matches!(
        plan.lincheck_circuit(k_log - 1),
        Err(WalkError::Capacity { .. })
    ));
}

#[test]
fn identity_c_specialization_is_explicit_and_equivalent() {
    let circuit = {
        let mut builder = CircuitBuilder::new();
        let input = builder.input_bits::<4>("input");
        let x = builder.xor2(input[0], input[1]);
        let y = builder.and(x, input[2]);
        let output_expression = builder.xor2(y, input[3]);
        let output = builder.materialize(output_expression);
        builder.output("output", [output]);
        builder.finish()
    };
    let plan = circuit.walk_plan().unwrap();
    assert!(plan.c_is_identity());
    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let e_a = weights(capacity, 41);
    let e_b = weights(capacity, 42);
    let e_c = weights(capacity, 43);
    assert_eq!(
        plan.transpose(&e_a, &e_b, &e_c).unwrap(),
        plan.transpose_identity_c(&e_a, &e_b, &e_c).unwrap()
    );

    let definitions: Vec<_> = circuit
        .rows()
        .iter()
        .filter_map(|row| row.defined_value())
        .collect();
    let mut permuted = circuit.layout();
    for (index, &value) in definitions.iter().enumerate() {
        permuted
            .place_definition(value, definitions.len() - 1 - index)
            .unwrap();
    }
    let permuted = permuted.finish().unwrap();
    let permuted_plan = circuit.walk_plan_with_layout(&permuted).unwrap();
    let permuted_r1cs = circuit
        .to_block_r1cs_with_layout(k_log, 0, 0, &permuted)
        .unwrap();
    assert!(permuted_plan.c_is_identity());
    assert!(permuted_r1cs.c0_is_identity());

    let mut near_identity = circuit.layout();
    for (index, &value) in definitions.iter().enumerate() {
        near_identity
            .place_value(value, definitions.len() - 1 - index)
            .unwrap();
    }
    for (index, row) in circuit.rows().iter().enumerate() {
        let position = if index + 1 == circuit.row_count() {
            circuit.row_count()
        } else {
            circuit.row_count() - 1 - index
        };
        near_identity.place_row(row.id(), position).unwrap();
    }
    let near_identity = near_identity.finish().unwrap();
    assert!(
        !circuit
            .walk_plan_with_layout(&near_identity)
            .unwrap()
            .c_is_identity()
    );
    assert!(
        !circuit
            .to_block_r1cs_with_layout(k_log, 0, 0, &near_identity)
            .unwrap()
            .c0_is_identity()
    );

    let non_identity = random_circuit(0xdead_beef).walk_plan().unwrap();
    assert!(!non_identity.c_is_identity());
    let k_log = capacity_log(non_identity.useful_bits());
    let weights = weights(1 << k_log, 44);
    assert_eq!(
        non_identity.transpose_identity_c(&weights, &weights, &weights),
        Err(WalkError::IdentityCRequired)
    );
}

#[test]
fn identity_c_specialization_includes_reserved_holes() {
    let mut builder = CircuitBuilder::new();
    let input = builder.input_bits::<4>("input");
    let xor = builder.xor2(input[0], input[1]);
    let output = builder.and(xor, input[2]);
    builder.output("output", [output]);
    let circuit = builder.finish();
    let mut layout = circuit.layout();
    layout.reserve(2..6).unwrap();
    let layout = layout.finish().unwrap();
    let plan = circuit.walk_plan_with_layout(&layout).unwrap();
    assert!(plan.c_is_identity());

    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let e_a = weights(capacity, 51);
    let e_b = weights(capacity, 52);
    let e_c = weights(capacity, 53);
    assert_eq!(
        plan.transpose(&e_a, &e_b, &e_c).unwrap(),
        plan.transpose_identity_c(&e_a, &e_b, &e_c).unwrap()
    );
}

#[test]
fn deep_and_reused_xors_follow_structural_edges_and_liveness() {
    let mut builder = CircuitBuilder::new();
    let inputs = builder.input_bits::<8>("input");
    let mut expression = inputs[0].expr();
    for index in 0..20_000 {
        expression = builder.xor2(expression, inputs[index % inputs.len()]);
    }
    let output = builder.materialize(expression);
    builder.output("output", [output]);
    let deep = builder.finish();
    let deep_plan = deep.walk_plan().unwrap();
    assert_eq!(deep_plan.stats().xor_nodes, 20_000);
    assert_eq!(deep_plan.stats().max_live_temporaries, 1);
    let deep_k_log = capacity_log(deep_plan.useful_bits());
    let deep_trace = deep_plan
        .forward(
            &[true, false, true, false, true, false, true, false],
            deep_k_log,
        )
        .unwrap();
    let deep_r1cs = deep.to_block_r1cs(deep_k_log, 0, 0).unwrap();
    assert_eq!(deep_trace.a_z, deep_r1cs.apply_a(&deep_trace.z));
    assert_eq!(deep_trace.b_z, deep_r1cs.apply_b(&deep_trace.z));
    assert_eq!(deep_trace.c_z, deep_r1cs.apply_c(&deep_trace.z));
    let deep_capacity = 1 << deep_k_log;
    let deep_a = weights(deep_capacity, 61);
    let deep_b = weights(deep_capacity, 62);
    let deep_c = weights(deep_capacity, 63);
    let deep_expected = add_three(
        sparse_transpose(&deep_r1cs.a_0, &deep_a),
        &sparse_transpose(&deep_r1cs.b_0, &deep_b),
        &sparse_transpose(&deep_r1cs.c_0, &deep_c),
    );
    assert_eq!(
        deep_plan.transpose(&deep_a, &deep_b, &deep_c).unwrap(),
        deep_expected
    );

    let mut builder = CircuitBuilder::new();
    let inputs = builder.input_bits::<128>("input");
    let shared = builder.xor(inputs);
    let one = builder.one();
    for _ in 0..128 {
        builder.constrain(shared, one, shared);
    }
    let output = builder.materialize(shared);
    builder.output("output", [output]);
    let reused = builder.finish();
    let plan = reused.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let r1cs = reused.to_block_r1cs(k_log, 0, 0).unwrap();
    let sparse_nonzeros: usize = r1cs
        .a_0
        .rows
        .iter()
        .chain(&r1cs.b_0.rows)
        .chain(&r1cs.c_0.rows)
        .map(Vec::len)
        .sum();
    assert!(plan.stats().structural_edges * 20 < sparse_nonzeros);
    assert_eq!(plan.stats().max_live_temporaries, 1);
    assert_eq!(plan.expression_fanout(shared.id()), Some(128 * 2 + 1));
}
