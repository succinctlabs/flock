//! Integration checks using only the public inspection and consumer APIs.

use std::collections::BTreeSet;

use super::*;
use flock_core::circuit::boolean::{BooleanCircuit, ExpressionNode, PhysicalLayout};

// Reconstruct supports from the structural DAG, not the lowering's support cache.
fn inspected_matrices(
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

#[test]
fn public_inspection_reconstructs_the_complete_sparse_relation() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let layout = compiled.circuit().layout().unwrap();
    let circuit = compiled.circuit();
    let capacity = layout.useful_bits().next_power_of_two();
    let r1cs = circuit
        .to_block_r1cs_with_layout(capacity.trailing_zeros() as usize, 0, 0, &layout)
        .unwrap();
    let [a, b, c] = inspected_matrices(circuit, &layout, capacity);
    assert_eq!(a, *r1cs.a_0.rows);
    assert_eq!(b, *r1cs.b_0.rows);
    assert_eq!(c, *r1cs.c_0.rows);
    assert_eq!(r1cs.const_pin, layout.value_position(circuit.one()));
}
