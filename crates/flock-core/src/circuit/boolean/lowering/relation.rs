//! Borrowed equations shared by sparse lowering and structural walking.
//!
//! This view owns no data and supplies no builder or definition validation.
//! Its owner must validate the relation before exposing it to consumers.

use std::sync::OnceLock;

use crate::circuit::boolean::{
    BooleanCircuit, CircuitId, Expression, LinearExprId, PhysicalLayout, Row, SchemaColumn, ValueId,
};
use crate::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};

use super::{R1csBuildError, checked_capacity};

mod evaluate;

pub(crate) struct RelationRef<'a> {
    pub id: CircuitId,
    pub expressions: &'a [Expression],
    pub rows: &'a [Row],
    pub value_count: usize,
    pub input_values: &'a [ValueId],
    pub columns: &'a [SchemaColumn],
    pub one: ValueId,
}

impl BooleanCircuit {
    pub(crate) fn relation(&self) -> RelationRef<'_> {
        RelationRef {
            id: self.id,
            expressions: &self.expressions,
            rows: &self.rows,
            value_count: self.value_count,
            input_values: &self.input_values,
            columns: &self.columns,
            one: self.one,
        }
    }
}

impl RelationRef<'_> {
    pub(crate) fn normalized_support_terms(&self) -> usize {
        self.expressions.iter().map(|expr| expr.support.len()).sum()
    }

    /// The layout must already have passed uniqueness, shape, and bounds checks.
    pub(crate) fn c_is_identity(&self, layout: &PhysicalLayout) -> bool {
        self.rows.len() == self.value_count
            && self.rows.iter().all(|row| {
                let support = &self.expressions[row.result.index].support;
                support.len() == 1
                    && layout.value_positions[support[0].index()]
                        == layout.row_positions[row.id.index]
            })
    }

    pub(crate) fn to_block_r1cs(
        &self,
        k_log: usize,
        k_skip: usize,
        n_log: usize,
        layout: &PhysicalLayout,
    ) -> Result<BlockR1cs, R1csBuildError> {
        layout
            .validate_relation(self)
            .map_err(R1csBuildError::InvalidLayout)?;
        let k = checked_capacity(k_log).map_err(R1csBuildError::InvalidKLog)?;
        if k_skip > k_log {
            return Err(R1csBuildError::InvalidKSkip { k_log, k_skip });
        }
        let required = layout.useful_bits;
        if k < required {
            return Err(R1csBuildError::Capacity {
                required,
                actual: k,
            });
        }
        let m = k_log
            .checked_add(n_log)
            .ok_or(R1csBuildError::DimensionOverflow)?;
        checked_capacity(m).map_err(|_| R1csBuildError::DimensionOverflow)?;

        let mut a_rows = vec![Vec::new(); k];
        let mut b_rows = vec![Vec::new(); k];
        let mut c_rows = vec![Vec::new(); k];
        let mut occupied_rows = vec![false; required];
        let mut occupied_columns = vec![false; required];
        for &position in &layout.value_positions {
            occupied_columns[position] = true;
        }
        for row in self.rows {
            let physical_row = layout.row_positions[row.id.index];
            occupied_rows[physical_row] = true;
            a_rows[physical_row] = self.support_with_layout(row.lhs, layout);
            b_rows[physical_row] = self.support_with_layout(row.rhs, layout);
            c_rows[physical_row] = self.support_with_layout(row.result, layout);
        }
        // The explicit layout keeps every real row and value in the useful
        // prefix. Its suffix is therefore available for canonical zero-padding
        // rows regardless of holes or permutations inside the prefix.
        for i in 0..k {
            if i >= required || (!occupied_rows[i] && !occupied_columns[i]) {
                c_rows[i] = vec![i];
            }
        }

        Ok(BlockR1cs {
            m,
            k_log,
            k_skip,
            useful_bits: required,
            a_0: SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: a_rows,
            },
            b_0: SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: b_rows,
            },
            c_0: SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                rows: c_rows,
            },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(layout.value_positions[self.one.index]),
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
        })
    }

    fn support_with_layout(&self, id: LinearExprId, layout: &PhysicalLayout) -> Vec<usize> {
        let mut support: Vec<usize> = self.expressions[id.index]
            .support
            .iter()
            .map(|value| layout.value_positions[value.index()])
            .collect();
        support.sort_unstable();
        support
    }
}
