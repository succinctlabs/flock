use flock_core::circuit::boolean::{ColumnRole, ColumnSchema, ColumnVisitor};

use super::ADVICE_TYPE;
use super::operations::{Add64, Or32, SelectByte};

/// Named columns for one invocation; the schema below marks supplied and derived fields.
pub struct LoadByteCols<T> {
    // Inputs.
    pub is_lb: T,
    pub is_lbu: T,
    pub b: [T; 64],
    pub c: [T; 64],
    pub memory_value: [T; 64],
    // Advice, checked only on active rows.
    pub selected_byte: [T; 8],
    // Deterministic witnesses and output.
    pub real: T,
    pub address: Add64<T>,
    pub address_guard: Or32<T>,
    pub select: SelectByte<T>,
    pub sign: T,
    pub result: [T; 64],
    pub aligned_low: [T; 3],
}

/// A fixed row count for this compiled circuit, independent of witness values.
pub struct LoadByteSchema {
    pub capacity: usize,
}

impl ColumnSchema for LoadByteSchema {
    type Cols<T> = Vec<LoadByteCols<T>>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Self::Cols<V::Value> {
        (0..self.capacity)
            .map(|row| {
                let p = format!("load-byte.row-{row}");
                LoadByteCols {
                    is_lb: v.bit(&format!("{p}.is-lb"), ColumnRole::Input),
                    is_lbu: v.bit(&format!("{p}.is-lbu"), ColumnRole::Input),
                    b: v.word(&format!("{p}.b"), ColumnRole::Input),
                    c: v.word(&format!("{p}.c"), ColumnRole::Input),
                    memory_value: v.word(&format!("{p}.memory-value"), ColumnRole::Input),
                    selected_byte: v.word(
                        &format!("{p}.selected-byte"),
                        ColumnRole::Advice(ADVICE_TYPE),
                    ),
                    real: v.bit(&format!("{p}.real"), ColumnRole::Witness),
                    address: Add64::columns(v, &format!("{p}.address")),
                    address_guard: Or32::columns(v, &format!("{p}.address-guard")),
                    select: SelectByte::columns(v, &format!("{p}.select")),
                    sign: v.bit(&format!("{p}.sign"), ColumnRole::Witness),
                    result: v.word(&format!("{p}.result"), ColumnRole::Output),
                    aligned_low: v.word(&format!("{p}.aligned-low"), ColumnRole::Witness),
                }
            })
            .collect()
    }
}
