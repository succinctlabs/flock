//! Automatic placement in evaluation order, with aligned input/advice/output/fixed words.

use std::fmt;

use super::lowering::relation::RelationRef;
use super::{BooleanCircuit, CircuitId, ColumnRole, RowId, ValueId};

/// Witness and matrix positions. Input/advice/output/fixed words stay contiguous and aligned;
/// everything else follows evaluation order. There is no manual placement API.
#[derive(Clone, Debug)]
pub struct PhysicalLayout {
    pub(super) circuit: CircuitId,
    pub(super) value_positions: Vec<usize>,
    pub(super) row_positions: Vec<usize>,
    pub(super) useful_bits: usize,
}

impl PartialEq for PhysicalLayout {
    fn eq(&self, other: &Self) -> bool {
        // Circuit IDs are runtime guards, not layout data.
        self.value_positions == other.value_positions
            && self.row_positions == other.row_positions
            && self.useful_bits == other.useful_bits
    }
}

impl Eq for PhysicalLayout {}

impl PhysicalLayout {
    pub(super) fn new(circuit: &BooleanCircuit) -> Result<Self, LayoutError> {
        let mut word_for_value = vec![None; circuit.value_count()];
        for (index, column) in circuit.columns.iter().enumerate() {
            if column.role != ColumnRole::Witness {
                for value in &column.values {
                    // Declared fields reserve distinct values.
                    assert!(word_for_value[value.index].replace(index).is_none());
                }
            }
        }
        let mut value_positions = vec![usize::MAX; circuit.value_count()];
        let mut next = 0usize;
        for value in 0..circuit.value_count() {
            if value_positions[value] != usize::MAX {
                continue;
            }
            if let Some(index) = word_for_value[value] {
                let column = &circuit.columns[index];
                let alignment_bits = column.alignment_bits;
                let padding = (alignment_bits - next % alignment_bits) % alignment_bits;
                next = next
                    .checked_add(padding)
                    .ok_or(LayoutError::PositionOverflow)?;
                let end = next
                    .checked_add(column.values.len())
                    .ok_or(LayoutError::PositionOverflow)?;
                for (offset, value) in column.values.iter().enumerate() {
                    value_positions[value.index] = next + offset;
                }
                next = end;
            } else {
                value_positions[value] = next;
                next = next.checked_add(1).ok_or(LayoutError::PositionOverflow)?;
            }
        }
        // Definition-only circuits retain identity C. General equations keep their order.
        let row_positions = if circuit.row_count() == circuit.value_count() {
            circuit
                .rows
                .iter()
                .map(|row| value_positions[row.defined_value.unwrap().index])
                .collect()
        } else {
            (0..circuit.row_count()).collect()
        };
        Ok(Self {
            circuit: circuit.id,
            value_positions,
            row_positions,
            useful_bits: next.max(circuit.row_count()),
        })
    }

    pub fn value_position(&self, value: ValueId) -> Option<usize> {
        if value.circuit != self.circuit {
            return None;
        }
        self.value_positions.get(value.index).copied()
    }

    pub fn row_position(&self, row: RowId) -> Option<usize> {
        if row.circuit != self.circuit {
            return None;
        }
        self.row_positions.get(row.index).copied()
    }

    /// Witness positions indexed by compiled value ID.
    pub fn value_positions(&self) -> &[usize] {
        &self.value_positions
    }

    /// Matrix positions indexed by evaluation-order row ID.
    pub fn row_positions(&self) -> &[usize] {
        &self.row_positions
    }

    /// Prefix containing all values and equations, including alignment gaps.
    pub const fn useful_bits(&self) -> usize {
        self.useful_bits
    }

    pub(super) fn validate_for(&self, circuit: &BooleanCircuit) -> Result<(), LayoutError> {
        self.validate_relation(&circuit.relation())
    }

    pub(super) fn validate_relation(&self, circuit: &RelationRef<'_>) -> Result<(), LayoutError> {
        if self.circuit != circuit.id
            || self.value_positions.len() != circuit.value_count
            || self.row_positions.len() != circuit.rows.len()
        {
            return Err(LayoutError::WrongCircuit);
        }
        for positions in [&self.value_positions, &self.row_positions] {
            let mut sorted = positions.clone();
            sorted.sort_unstable();
            if sorted.last().is_some_and(|&p| p >= self.useful_bits)
                || sorted.windows(2).any(|pair| pair[0] == pair[1])
            {
                return Err(LayoutError::InvalidPlacement);
            }
        }
        for column in circuit.columns {
            if column.role != ColumnRole::Witness {
                let alignment_bits = column.alignment_bits;
                let start = self.value_positions[column.values[0].index];
                if !start.is_multiple_of(alignment_bits)
                    || column.values.iter().enumerate().any(|(offset, value)| {
                        start.checked_add(offset) != Some(self.value_positions[value.index])
                    })
                {
                    return Err(LayoutError::InvalidPlacement);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    WrongCircuit,
    PositionOverflow,
    InvalidPlacement,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WrongCircuit => "layout belongs to another circuit",
            Self::PositionOverflow => "physical position overflows usize",
            Self::InvalidPlacement => "invalid witness or equation placement",
        })
    }
}

impl std::error::Error for LayoutError {}
