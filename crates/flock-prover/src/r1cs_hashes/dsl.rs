//! Shared authoring operations for the two hash circuits.

use flock_core::circuit::boolean::{CircuitBuilder, ColumnRole, ColumnVisitor, LinearExpr, Var};

mod addition;
#[cfg(test)]
pub(super) mod tests;
pub use addition::{Add3, Add4, Add32, AddConst32};

pub(super) type Word = [LinearExpr; 32];

pub(super) fn physical_witness(
    logical: &[bool],
    layout: &flock_core::circuit::boolean::PhysicalLayout,
    capacity: usize,
) -> Vec<bool> {
    assert_eq!(logical.len(), layout.value_positions().len());
    let mut physical = vec![false; capacity];
    for (&bit, &position) in logical.iter().zip(layout.value_positions()) {
        physical[position] = bit;
    }
    physical
}

pub(super) fn words<V: ColumnVisitor, const N: usize>(
    v: &mut V,
    name: &str,
    role: ColumnRole,
    alignment: usize,
) -> [[V::Value; 32]; N] {
    let mut bits = v.bits(name, role, N * 32, alignment).into_iter();
    std::array::from_fn(|_| std::array::from_fn(|_| bits.next().unwrap()))
}

pub(super) fn materialize(b: &mut CircuitBuilder, cols: &[Var; 32], word: Word) -> Word {
    for (&col, expr) in cols.iter().zip(word) {
        b.define_linear(col, expr);
    }
    cols.map(Into::into)
}

pub(super) fn constant(b: &CircuitBuilder, word: u32) -> Word {
    std::array::from_fn(|bit| {
        if word >> bit & 1 != 0 {
            b.one().into()
        } else {
            b.zero()
        }
    })
}

pub(super) fn populate_words<const N: usize>(cols: [[&mut bool; 32]; N], values: &[u32; N]) {
    for (cols, value) in cols.into_iter().zip(values) {
        for (bit, col) in cols.into_iter().enumerate() {
            *col = value >> bit & 1 != 0;
        }
    }
}
