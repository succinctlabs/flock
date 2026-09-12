use flock_core::circuit::boolean::EvaluationError;

use super::{LoadByteCircuit, LoadByteCols, LoadByteEvent, LoadByteOpcode};

pub struct LoadByteTrace {
    pub rows: Vec<LoadByteCols<bool>>,
    pub values: Vec<bool>,
}

impl LoadByteCircuit {
    /// Supply named inputs/advice; canonical evaluation fills every derived column.
    pub fn generate_trace(
        &self,
        events: &[LoadByteEvent],
    ) -> Result<LoadByteTrace, EvaluationError> {
        let (rows, values) = self.compiled.evaluate(|cols| supply_events(cols, events))?;
        Ok(LoadByteTrace { rows, values })
    }

    /// Encode real events followed by zero-supplied padding, without checking constraints.
    pub fn honest_inputs(&self, events: &[LoadByteEvent]) -> Vec<bool> {
        self.compiled.inputs(|cols| supply_events(cols, events))
    }

    /// Explicit rows allow malformed non-prefix traces to be tested.
    pub fn encode_rows(&self, rows: &[Option<LoadByteEvent>]) -> Vec<bool> {
        assert_eq!(rows.len(), self.capacity, "wrong load-byte row count");
        self.compiled.inputs(|cols| {
            for (cols, event) in cols.into_iter().zip(rows) {
                populate(cols, *event);
            }
        })
    }
}

fn supply_events(cols: Vec<LoadByteCols<&mut bool>>, events: &[LoadByteEvent]) {
    assert!(events.len() <= cols.len(), "too many load-byte events");
    for (row, cols) in cols.into_iter().enumerate() {
        populate(cols, events.get(row).copied());
    }
}

fn populate(cols: LoadByteCols<&mut bool>, event: Option<LoadByteEvent>) {
    let Some(event) = event else {
        return;
    };
    *cols.is_lb = event.opcode == LoadByteOpcode::Lb;
    *cols.is_lbu = event.opcode == LoadByteOpcode::Lbu;
    write_word(cols.b, event.b);
    write_word(cols.c, event.c);
    write_word(cols.memory_value, event.memory_value);
    let address = event.b.wrapping_add(event.c);
    write_word(
        cols.selected_byte,
        event.memory_value >> (8 * (address & 7)),
    );
}

fn write_word<const N: usize>(cols: [&mut bool; N], value: u64) {
    for (bit, col) in cols.into_iter().enumerate() {
        *col = value >> bit & 1 != 0;
    }
}
