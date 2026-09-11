//! Integration checks using only the public inspection and consumer APIs.

use std::collections::BTreeSet;

use super::*;
use flock_core::circuit::boolean::{BooleanCircuit, ExpressionNode, PhysicalLayout, WalkError};
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;

// Reconstruct supports from the structural DAG, not the lowering's support cache.
pub(super) fn inspected_matrices(
    circuit: &BooleanCircuit,
    layout: &PhysicalLayout,
    capacity: usize,
) -> [Vec<Vec<usize>>; 3] {
    let mut supports: Vec<BTreeSet<usize>> = Vec::new();
    for node in circuit.expressions() {
        let mut support = BTreeSet::new();
        match node {
            ExpressionNode::Zero => {}
            ExpressionNode::Value(value) => {
                support.insert(layout.value_position(*value).unwrap());
            }
            ExpressionNode::Xor(terms) => {
                for term in terms {
                    for &value in &supports[term.index()] {
                        if !support.insert(value) {
                            support.remove(&value);
                        }
                    }
                }
            }
        }
        supports.push(support);
    }
    let mut matrices = std::array::from_fn(|_| vec![Vec::new(); capacity]);
    let mut occupied: BTreeSet<_> = layout.value_positions().iter().copied().collect();
    for row in circuit.rows() {
        let position = layout.row_position(row.id()).unwrap();
        occupied.insert(position);
        for (matrix, expression) in matrices
            .iter_mut()
            .zip([row.lhs(), row.rhs(), row.result()])
        {
            matrix[position] = supports[expression.index()].iter().copied().collect();
        }
    }
    for (position, row) in matrices[2].iter_mut().enumerate() {
        if !occupied.contains(&position) {
            row.push(position);
        }
    }
    matrices
}

fn layouts(compiled: &CompiledColumns<DoubleAdd>) -> [PhysicalLayout; 2] {
    let circuit = compiled.circuit();
    let mut permuted = circuit.layout();
    // Reverse field groups, retain each word's bit order, and leave interior holes.
    let mut position = 3;
    for field in compiled.schema().iter().rev() {
        for &value in &field.values {
            permuted.place_value(value, position).unwrap();
            position += 1;
        }
        position += 1;
    }
    permuted.place_value(circuit.one(), position).unwrap();
    // Row order is independent of column placement.
    for row in circuit.rows() {
        permuted
            .place_row(row.id(), circuit.row_count() - 1 - row.id().index())
            .unwrap();
    }
    [
        PhysicalLayout::source_order(circuit),
        permuted.finish().unwrap(),
    ]
}

fn weights(capacity: usize, seed: u64) -> Vec<F128> {
    (0..capacity)
        .map(|i| F128::new(seed + i as u64, seed.rotate_left(i as u32)))
        .collect()
}

fn dot(bits: &[bool], weights: &[F128]) -> F128 {
    assert_eq!(bits.len(), weights.len());
    bits.iter()
        .zip(weights)
        .filter(|(bit, _)| **bit)
        .fold(F128::ZERO, |sum, (_, value)| sum + *value)
}

#[test]
fn public_inspection_reconstructs_the_complete_sparse_relation() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    for layout in layouts(&compiled) {
        let circuit = compiled.circuit();
        let capacity = layout.useful_bits().next_power_of_two();
        let r1cs = circuit
            .to_block_r1cs_with_layout(capacity.trailing_zeros() as usize, 0, 0, &layout)
            .unwrap();
        let [a, b, c] = inspected_matrices(circuit, &layout, capacity);
        assert_eq!(a, r1cs.a_0.rows);
        assert_eq!(b, r1cs.b_0.rows);
        assert_eq!(c, r1cs.c_0.rows);
        assert_eq!(r1cs.const_pin, layout.value_position(circuit.one()));
    }
}

#[test]
fn schema_witnesses_match_sparse_and_forward_consumers() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let events: Vec<_> = std::iter::once(None)
        .chain((0u8..64).map(|i| {
            Some(Event {
                a: i & 15,
                b: i.wrapping_mul(3) & 15,
                c: i.wrapping_mul(7) & 15,
            })
        }))
        .collect();
    let trace = generate_trace(&compiled, &events).unwrap();
    let circuit = compiled.circuit();
    for layout in layouts(&compiled) {
        let plan = circuit.walk_plan_with_layout(&layout).unwrap();
        let k_log = layout.useful_bits().next_power_of_two().trailing_zeros() as usize;
        let r1cs = circuit
            .to_block_r1cs_with_layout(k_log, 0, 0, &layout)
            .unwrap();
        for (_, logical) in &trace {
            let inputs: Vec<_> = circuit
                .inputs()
                .iter()
                .map(|v| logical[v.index()])
                .collect();
            let forward = plan.forward(&inputs, k_log).unwrap();
            assert_eq!(
                forward.z,
                circuit
                    .evaluate_r1cs_with_layout(&inputs, k_log, &layout)
                    .unwrap()
            );
            for field in compiled.schema() {
                for &value in &field.values {
                    assert_eq!(
                        forward.z[layout.value_position(value).unwrap()],
                        logical[value.index()]
                    );
                }
            }
            assert_eq!(forward.a_z, r1cs.apply_a(&forward.z));
            assert_eq!(forward.b_z, r1cs.apply_b(&forward.z));
            assert_eq!(forward.c_z, r1cs.apply_c(&forward.z));
            assert!(r1cs.satisfies(&forward.z));
        }
        let mut bad = trace[1].1.clone();
        bad[compiled.columns().claimed_sum[0].index()] ^= true;
        let inputs: Vec<_> = circuit.inputs().iter().map(|v| bad[v.index()]).collect();
        let EvaluationError::UnsatisfiedRow(row) = circuit.evaluate(&inputs).unwrap_err() else {
            panic!("expected a failed advice check")
        };
        assert_eq!(
            plan.forward(&inputs, k_log),
            Err(WalkError::UnsatisfiedRow(row))
        );
    }
}

#[test]
fn reverse_walk_matches_all_matrix_columns_and_lincheck() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    for layout in layouts(&compiled) {
        assert!(
            !compiled
                .circuit()
                .walk_plan_with_layout(&layout)
                .unwrap()
                .c_is_identity()
        );
        check_reverse(compiled.circuit(), &layout);
    }
}

pub(super) fn check_reverse(circuit: &BooleanCircuit, layout: &PhysicalLayout) {
    let capacity = layout.useful_bits().next_power_of_two();
    let k_log = capacity.trailing_zeros() as usize;
    let plan = circuit.walk_plan_with_layout(layout).unwrap();
    let r1cs = circuit
        .to_block_r1cs_with_layout(k_log, 0, 0, layout)
        .unwrap();
    let e = std::array::from_fn::<_, 3, _>(|side| weights(capacity, 17 * (side as u64 + 1)));
    let zero = vec![F128::ZERO; capacity];
    if plan.c_is_identity() {
        assert_eq!(
            plan.transpose(&e[0], &e[1], &e[2]).unwrap(),
            plan.transpose_identity_c(&e[0], &e[1], &e[2]).unwrap()
        );
    }
    for side in 0..3 {
        let args = std::array::from_fn::<_, 3, _>(|i| if i == side { &e[i] } else { &zero });
        let transposed = plan.transpose(args[0], args[1], args[2]).unwrap();
        // Basis vectors need not satisfy the relation: check every column, including padding.
        for column in 0..capacity {
            let mut basis = vec![false; capacity];
            basis[column] = true;
            let rows = match side {
                0 => r1cs.apply_a(&basis),
                1 => r1cs.apply_b(&basis),
                _ => r1cs.apply_c(&basis),
            };
            assert_eq!(dot(&rows, &e[side]), transposed[column]);
        }
    }
    let folded = plan.lincheck_circuit(k_log).unwrap();
    assert_eq!(folded.const_pin_col(), r1cs.const_pin);
    assert_eq!(
        folded.fold_alpha_batched(F128::new(23, 47), &e[0]),
        r1cs.sparse_lincheck_circuit()
            .fold_alpha_batched(F128::new(23, 47), &e[0])
    );
}

#[test]
fn schema_and_operation_inspection_are_complete_and_stable() {
    let first = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let second = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    assert_eq!(first.schema().len(), second.schema().len());
    assert_eq!(first.operations().len(), second.operations().len());
    let indices = |values: &[flock_core::circuit::boolean::ValueId]| {
        values.iter().map(|v| v.index()).collect::<Vec<_>>()
    };
    let mut covered = BTreeSet::new();
    for (a, b) in first.schema().iter().zip(second.schema()) {
        assert_eq!(
            (&a.name, &a.role, indices(&a.values)),
            (&b.name, &b.role, indices(&b.values))
        );
        for &value in &a.values {
            assert!(covered.insert(value.index()));
        }
    }
    assert_eq!(covered, (1..first.circuit().value_count()).collect());
    for (a, b) in first.operations().iter().zip(second.operations()) {
        assert_eq!(a.inputs.len(), b.inputs.len());
        assert_eq!(
            (&a.name, &a.kind, &a.rows, &a.expressions, &a.interactions),
            (&b.name, &b.kind, &b.rows, &b.expressions, &b.interactions)
        );
        assert_eq!(indices(&a.columns), indices(&b.columns));
        let defined: Vec<_> = first.circuit().rows()[a.rows.clone()]
            .iter()
            .filter_map(|r| r.defined_value())
            .collect();
        assert_eq!(defined, a.columns);
        for (x, y) in a
            .inputs
            .iter()
            .chain([&a.output])
            .zip(b.inputs.iter().chain([&b.output]))
        {
            assert_eq!(x.name, y.name);
            assert_eq!(
                x.expressions.iter().map(|e| e.index()).collect::<Vec<_>>(),
                y.expressions.iter().map(|e| e.index()).collect::<Vec<_>>()
            );
            for &expression in &x.expressions {
                assert!(first.circuit().expression(expression).is_some());
            }
        }
    }
}
