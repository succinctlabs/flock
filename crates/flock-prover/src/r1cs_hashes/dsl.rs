//! Shared authoring operations for the two hash circuits.

use std::sync::Arc;

use flock_core::circuit::boolean::{
    BooleanCircuit, BooleanGate, BooleanGateError, CircuitBuilder, ColumnRole, ColumnVisitor,
    LinearExpr, SchemaOperation, Var,
};

mod addition;
#[cfg(test)]
pub(super) mod tests;
pub use addition::{Add3, Add4, Add32, AddConst32};

/// A DSL circuit as a gate ([`BooleanGate::new`]) with every hand-written
/// generator attached: each operation that has one (a BLAKE3 compression
/// built with [`super::blake3_dsl::compress_op`]) is packed natively rather
/// than walked. `operations` are the compiled circuit's
/// ([`flock_core::circuit::boolean::CompiledColumns::operations`]).
pub fn boolean_gate(
    name: &str,
    circuit: &BooleanCircuit,
    operations: &[SchemaOperation],
    inputs: &[&str],
    outputs: &[&str],
    nu: usize,
) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(name, circuit, inputs, outputs, nu)?.with_kernel(
        circuit,
        operations,
        super::blake3_dsl::OPERATION,
        Arc::new(super::blake3::Blake3Kernel),
    )
}

/// A 32-bit word of expressions, little-endian.
pub type Word = [LinearExpr; 32];

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
