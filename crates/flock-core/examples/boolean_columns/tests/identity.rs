use super::*;
use flock_core::circuit::boolean::LoweredCircuit;
use flock_core::field::F128;

fn physical(lowered: &LoweredCircuit, logical: &[bool]) -> Vec<bool> {
    let mut z = vec![false; 128];
    for (&position, &value) in lowered.layout().value_positions().iter().zip(logical) {
        z[position] = value;
    }
    z
}

#[test]
fn composed_chip_has_equivalent_identity_c_witnesses() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let source = compiled.circuit();
    let lowered = source.lower_identity_c().unwrap();
    let direct = source.to_block_r1cs(6, 0, 0).unwrap();
    let sparse = lowered.to_block_r1cs(7, 0, 0).unwrap();
    assert!(sparse.c0_is_identity());
    let mut events = vec![None];
    for a in 0..16 {
        for b in 0..16 {
            for c in 0..16 {
                events.push(Some(Event { a, b, c }));
            }
        }
    }
    for (_, source) in generate_trace(&compiled, &events).unwrap() {
        let extended = lowered.extend(&source).unwrap();
        assert!(lowered.accepts(&extended));
        assert!(sparse.satisfies(&physical(&lowered, &extended)));
        assert_eq!(lowered.project(&extended), Some(source));
    }
    let trace = generate_trace(&compiled, &[Some(Event { a: 7, b: 9, c: 3 }), None]).unwrap();
    for (index, (_, logical)) in trace.iter().enumerate() {
        // These choices include several assertions failing at once.
        for advice in 0..16 {
            let mut logical = logical.clone();
            for (i, value) in compiled.columns().claimed_sum.into_iter().enumerate() {
                logical[value.index()] = advice >> i & 1 != 0;
            }
            let extended = lowered.extend(&logical).unwrap();
            logical.resize(64, false);
            let expected = index == 1 || advice == 3;
            assert_eq!(direct.satisfies(&logical), expected);
            assert_eq!(lowered.accepts(&extended), expected);
            assert_eq!(sparse.satisfies(&physical(&lowered, &extended)), expected);
        }
    }
    let source = &trace[0].1;
    let extended = lowered.extend(source).unwrap();
    for field in compiled.schema() {
        for &value in &field.values {
            assert_eq!(
                source[value.index()],
                extended[lowered.mapped_value(value).unwrap().index()]
            );
        }
    }
    for aux in lowered.auxiliaries() {
        let mut corrupted = extended.clone();
        corrupted[aux.product.index()] ^= true;
        assert!(!lowered.accepts(&corrupted));
        assert!(!sparse.satisfies(&physical(&lowered, &corrupted)));
        // t cancels from its equation; either choice must work.
        let mut alternate = extended.clone();
        alternate[aux.cancellation.index()] ^= true;
        assert!(lowered.accepts(&alternate));
        assert!(sparse.satisfies(&physical(&lowered, &alternate)));
    }
    assert_eq!(compiled.circuit().value_count(), 29);
    assert_eq!(compiled.circuit().row_count(), 33);
    assert_eq!(lowered.value_count(), 37);
    assert_eq!(lowered.rows().len(), 37);
}

#[test]
fn identity_c_uses_sparse_and_walk_consumers() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let source = compiled.circuit();
    let trace = generate_trace(
        &compiled,
        &[
            Some(Event {
                a: 15,
                b: 15,
                c: 15,
            }),
            None,
        ],
    )
    .unwrap();
    let lowered = source.lower_identity_c().unwrap();
    let plan = lowered.walk_plan().unwrap();
    let sparse = lowered.to_block_r1cs(7, 0, 0).unwrap();
    assert!(plan.c_is_identity());
    assert!(sparse.c0_is_identity());
    for (_, source) in &trace {
        let logical = lowered.extend(source).unwrap();
        let inputs: Vec<_> = lowered
            .inputs()
            .iter()
            .map(|v| logical[v.index()])
            .collect();
        let forward = plan.forward(&inputs, 7).unwrap();
        assert_eq!(forward.z, physical(&lowered, &logical));
        assert_eq!(forward.a_z, sparse.apply_a(&forward.z));
        assert_eq!(forward.b_z, sparse.apply_b(&forward.z));
        assert_eq!(forward.c_z, forward.z);
        assert!(sparse.satisfies(&forward.z));
        assert!(forward.z[sparse.const_pin.unwrap()]);
    }
    let weights: [Vec<_>; 3] = std::array::from_fn(|m| {
        (0..128)
            .map(|i| F128::new((i + 1 + m * 128) as u64, 17))
            .collect()
    });
    let mut expected = vec![F128::ZERO; 128];
    for (matrix, weights) in [&sparse.a_0, &sparse.b_0, &sparse.c_0]
        .into_iter()
        .zip(&weights)
    {
        for (row, columns) in matrix.rows.iter().enumerate() {
            for &column in columns {
                expected[column] += weights[row];
            }
        }
    }
    assert_eq!(
        plan.transpose(&weights[0], &weights[1], &weights[2])
            .unwrap(),
        expected
    );
}
