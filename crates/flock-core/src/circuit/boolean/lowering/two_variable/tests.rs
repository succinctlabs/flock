use super::*;
use crate::circuit::boolean::{CircuitBuilder, RowKind};

mod equivalence;
mod layout;
mod mappings;
mod selection;
mod validation;
mod walks;

fn bits(number: usize, len: usize) -> Vec<bool> {
    (0..len).map(|i| number >> i & 1 != 0).collect()
}

fn physical(lowered: &LoweredCircuit, logical: &[bool], capacity: usize) -> Vec<bool> {
    let mut result = vec![false; capacity];
    for (&position, &bit) in lowered.layout().value_positions().iter().zip(logical) {
        result[position] = bit;
    }
    result
}
