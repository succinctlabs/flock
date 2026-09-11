use super::*;
use flock_core::circuit::boolean::PhysicalLayout;

#[test]
fn composed_chip_has_equivalent_bound_identity_c_witnesses() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let checker = compiled.circuit().identity_checker();
    let direct = compiled.circuit().to_block_r1cs(6, 0, 0).unwrap();
    let unbound = checker.unbound_circuit().to_block_r1cs(7, 0, 0).unwrap();
    let mut events = vec![None];
    for a in 0..16 {
        for b in 0..16 {
            for c in 0..16 {
                events.push(Some(Event { a, b, c }));
            }
        }
    }
    for (_, source) in generate_trace(&compiled, &events).unwrap() {
        let extended = checker.extend(&source).unwrap();
        assert!(checker.accepts(&extended));
        assert_eq!(checker.project(&extended), Some(source));
    }
    let trace = generate_trace(&compiled, &[Some(Event { a: 7, b: 9, c: 3 }), None]).unwrap();
    for (index, (_, logical)) in trace.iter().enumerate() {
        // All advice choices cover zero, one, two, three, and four simultaneous failures.
        for advice in 0..16 {
            let mut logical = logical.clone();
            for (i, value) in compiled.columns().claimed_sum.into_iter().enumerate() {
                logical[value.index()] = advice >> i & 1 != 0;
            }
            let extended = checker.extend(&logical).unwrap();
            logical.resize(64, false);
            let expected = index == 1 || advice == 3;
            assert_eq!(direct.satisfies(&logical), expected);
            assert_eq!(checker.accepts(&extended), expected);
            let mut padded = extended.clone();
            padded.resize(128, false);
            assert!(unbound.satisfies(&padded));
            if !expected {
                let mut forged = extended;
                forged[checker.accept().index()] = true;
                assert!(!checker.accepts(&forged));
            }
        }
    }
    let source = &trace[0].1;
    let extended = checker.extend(source).unwrap();
    for field in compiled.schema() {
        for &value in &field.values {
            assert_eq!(
                source[value.index()],
                extended[checker.mapped_value(value).unwrap().index()]
            );
        }
    }
    for aux in checker.auxiliaries() {
        let mut corrupted = extended.clone();
        corrupted[aux.value.index()] ^= true;
        assert!(!checker.accepts(&corrupted));
    }
}

#[test]
fn identity_checker_uses_existing_sparse_and_walk_consumers() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let checker = compiled.circuit().identity_checker();
    let circuit = checker.unbound_circuit();
    let mut permuted = circuit.layout();
    for row in circuit.rows() {
        permuted
            .place_definition(
                row.defined_value().unwrap(),
                circuit.value_count() + 2 - row.id().index(),
            )
            .unwrap();
    }
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
    for layout in [
        PhysicalLayout::source_order(circuit),
        permuted.finish().unwrap(),
    ] {
        let plan = circuit.walk_plan_with_layout(&layout).unwrap();
        let sparse = circuit.to_block_r1cs_with_layout(7, 0, 0, &layout).unwrap();
        assert!(plan.c_is_identity());
        assert!(sparse.c0_is_identity());
        let [a, b, c] = downstream::inspected_matrices(circuit, &layout, 128);
        assert_eq!(a, sparse.a_0.rows);
        assert_eq!(b, sparse.b_0.rows);
        assert_eq!(c, sparse.c_0.rows);
        downstream::check_reverse(circuit, &layout);
        for (_, source) in &trace {
            let logical = checker.extend(source).unwrap();
            let inputs: Vec<_> = circuit
                .inputs()
                .iter()
                .map(|v| logical[v.index()])
                .collect();
            let mut expected = vec![false; 128];
            for (index, &position) in layout.value_positions().iter().enumerate() {
                expected[position] = logical[index];
            }
            let forward = plan.forward(&inputs, 7).unwrap();
            assert_eq!(forward.z, expected);
            assert_eq!(forward.a_z, sparse.apply_a(&forward.z));
            assert_eq!(forward.b_z, sparse.apply_b(&forward.z));
            assert_eq!(forward.c_z, forward.z);
            assert!(sparse.satisfies(&forward.z));
            assert!(forward.z[sparse.const_pin.unwrap()]);
            assert!(forward.z[layout.value_position(checker.accept()).unwrap()]);
            let projected: Vec<_> = layout
                .value_positions()
                .iter()
                .map(|&position| forward.z[position])
                .collect();
            assert_eq!(checker.project(&projected).as_ref(), Some(source));
        }
    }
}

#[test]
fn identity_experiment_costs_are_explicit() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let checker = compiled.circuit().identity_checker();
    for (name, circuit) in [
        ("direct", compiled.circuit()),
        ("identity", checker.unbound_circuit()),
    ] {
        let plan = circuit.walk_plan().unwrap();
        let stats = plan.stats();
        println!(
            "{name}: values={}, rows={}, capacity={}, actions={}, edges={}, action_bytes={}, support_bytes={}, temporary_slots={}",
            circuit.value_count(),
            circuit.row_count(),
            plan.useful_bits().next_power_of_two(),
            stats.actions,
            stats.structural_edges,
            stats.action_bytes,
            circuit.normalized_support_bytes(),
            stats.max_live_temporaries
        );
    }
    assert_eq!(compiled.circuit().value_count(), 29);
    assert_eq!(compiled.circuit().row_count(), 33);
    assert_eq!(checker.unbound_circuit().value_count(), 94);
    assert_eq!(checker.unbound_circuit().row_count(), 94);
}
