use super::*;

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
