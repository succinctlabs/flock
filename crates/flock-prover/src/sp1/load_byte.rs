//! SP1-style authoring for the Boolean load-byte pilot.

use flock_core::circuit::boolean::{
    BooleanCircuit, CircuitBuilder, CompiledColumns, LayoutError, LoweredCircuit, LoweringMode,
    ValueId,
};

mod columns;
mod eval;
pub mod operations;
mod trace;

pub use columns::{LoadByteCols, LoadByteSchema};
pub use trace::LoadByteTrace;

const ADVICE_TYPE: &str = "sp1.load-byte.selected-byte/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadByteOpcode {
    Lb,
    Lbu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadByteEvent {
    pub opcode: LoadByteOpcode,
    pub b: u64,
    pub c: u64,
    pub memory_value: u64,
}

impl LoadByteEvent {
    pub const fn result(self) -> u64 {
        let address = self.b.wrapping_add(self.c);
        let byte = (self.memory_value >> (8 * (address & 7))) as u8;
        match self.opcode {
            LoadByteOpcode::Lb => (byte as i8 as i64) as u64,
            LoadByteOpcode::Lbu => byte as u64,
        }
    }
}

pub struct LoadByteCircuit {
    compiled: CompiledColumns<LoadByteSchema>,
    capacity: usize,
}

impl LoadByteCircuit {
    pub fn build(capacity: usize) -> Self {
        assert!(capacity > 0, "load-byte capacity must be nonzero");
        Self {
            compiled: CircuitBuilder::compile(LoadByteSchema { capacity }, |b, cols| {
                eval::eval(b, cols)
            }),
            capacity,
        }
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn circuit(&self) -> &BooleanCircuit {
        self.compiled.circuit()
    }

    /// Lower the entire capacity, including constraints between invocations.
    /// Source columns keep their positions; identity C may expand and reorder rows.
    pub fn lower(&self, mode: LoweringMode) -> Result<LoweredCircuit, LayoutError> {
        self.circuit().lower(mode)
    }

    pub fn compiled(&self) -> &CompiledColumns<LoadByteSchema> {
        &self.compiled
    }

    pub fn columns(&self) -> Vec<LoadByteCols<ValueId>> {
        self.compiled.columns()
    }

    /// Decode source logical values; project a lowered witness before calling this.
    pub fn output(&self, values: &[bool], row: usize) -> u64 {
        assert!(row < self.capacity, "load-byte row is out of range");
        self.circuit()
            .column(&format!("load-byte.row-{row}.result"))
            .unwrap()
            .values
            .iter()
            .enumerate()
            .fold(0, |word, (bit, value)| {
                word | (u64::from(values[value.index()]) << bit)
            })
    }
}

#[cfg(test)]
mod tests;
