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

#[test]
fn cancellation_equations_enforce_exactly_the_source_equation() {
    for lhs in [false, true] {
        for rhs in [false, true] {
            for result in [false, true] {
                let source_holds = (lhs & rhs) == result;
                for t in [false, true] {
                    let mut extension_exists = false;
                    for y in [false, true] {
                        // ONE is pinned: ONE * (t XOR y XOR result) = t.
                        let lowered_holds = (lhs & rhs) == y && (t ^ y ^ result) == t;
                        assert_eq!(lowered_holds, source_holds && y == (lhs & rhs));
                        extension_exists |= lowered_holds;
                    }
                    assert_eq!(extension_exists, source_holds);
                }
                // The honest extension fixes t = 0 and computes only y.
                let y = lhs & rhs;
                assert_eq!(!(y ^ result), source_holds);
            }
        }
    }
}

#[test]
fn cancellation_requires_the_existing_one_pin() {
    let (one, lhs, rhs, result, y, t) = (false, true, true, false, true, false);
    assert_ne!(lhs & rhs, result);
    assert_eq!(lhs & rhs, y);
    assert_eq!(one & (t ^ y ^ result), t);
}
