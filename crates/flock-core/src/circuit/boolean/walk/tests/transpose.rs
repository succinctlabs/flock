use super::*;

#[test]
fn transpose_walk_rejects_bad_requests() {
    let circuit = random_circuit(0x0dd5_0000);
    let layout = circuit.layout().unwrap();
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
    let circuit = support::circuit(4, 2, |builder, cols| {
        let input = &cols.input;
        let x = builder.xor2(input[0], input[1]);
        let y = cols.witness[0];
        builder.define_and(y, x, input[2]);
        let output_expression = builder.xor2(y, input[3]);
        builder.define_linear(cols.witness[1], output_expression);
    });
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

    let non_identity = random_circuit(0xdead_beef).walk_plan().unwrap();
    assert!(!non_identity.c_is_identity());
    let k_log = capacity_log(non_identity.useful_bits());
    let weights = weights(1 << k_log, 44);
    assert_eq!(
        non_identity.transpose_identity_c(&weights, &weights, &weights),
        Err(WalkError::IdentityCRequired)
    );
}
