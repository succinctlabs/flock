//! Selected direct or identity-C lowering. Cancellation rows check, rather than define, t.

use std::ops::Range;

use crate::circuit::boolean::{
    BooleanCircuit, CircuitId, Expression, ExpressionNode, Interaction, LayoutError, LinearExprId,
    PhysicalLayout, Row, RowId, SchemaColumn, ValueId, ValueIndex, WalkPlan,
};
use crate::r1cs::BlockR1cs;

use super::{R1csBuildError, relation::RelationRef};

mod build;
#[cfg(test)]
mod tests;
mod validation;
mod witness;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringMode {
    /// Preserve authored equations, including general C sides.
    Direct,
    /// Convert assertions and require exact identity C under the resulting layout.
    RequireIdentityC,
}

/// Two fresh coordinates and replacement rows for one source assertion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssertionAux {
    pub source_row: RowId,
    pub product: ValueId,
    /// Initialized to zero by honest evaluation; either bit is allowed by the relation.
    pub cancellation: ValueId,
    pub product_row: RowId,
    pub cancellation_row: RowId,
}

/// A separately identified selected relation, with no new acceptance pin.
///
/// Source columns remain a logical prefix. Declared columns and interactions use target IDs;
/// operation records stay on the source and use the explicit maps.
#[derive(Debug)]
pub struct LoweredCircuit {
    id: CircuitId,
    source: CircuitId,
    source_value_count: usize,
    source_expression_count: usize,
    expressions: Vec<Expression>,
    rows: Vec<Row>,
    input_values: Vec<ValueId>,
    one: ValueId,
    columns: Vec<SchemaColumn>,
    interactions: Vec<Interaction>,
    source_rows: Vec<Range<usize>>,
    auxiliaries: Vec<AssertionAux>,
    layout: PhysicalLayout,
}

impl BooleanCircuit {
    /// Select the authored equations or enforce identity C. Placement is automatic.
    pub fn lower(&self, mode: LoweringMode) -> Result<LoweredCircuit, LayoutError> {
        LoweredCircuit::build(self, &self.layout()?, mode)
    }

    /// Replace each assertion with two rows and two appended variables.
    /// Source value positions stay unchanged; each C-side value chooses its row position.
    pub fn lower_identity_c(&self) -> Result<LoweredCircuit, LayoutError> {
        self.lower(LoweringMode::RequireIdentityC)
    }
}

impl LoweredCircuit {
    /// Materialized-value references across all retained normalized expression supports.
    pub fn normalized_support_terms(&self) -> usize {
        self.relation().normalized_support_terms()
    }

    /// Compact support payload only, excluding vector headers and spare capacity.
    pub fn normalized_support_bytes(&self) -> usize {
        self.normalized_support_terms() * std::mem::size_of::<ValueIndex>()
    }

    /// Compile this selected relation, including explicit cancellation initialization.
    /// Walk errors carry target row IDs; `source_row` maps them back to authored rows.
    pub fn walk_plan(&self) -> Result<WalkPlan, LayoutError> {
        WalkPlan::compile(
            &self.relation(),
            &self.layout,
            self.auxiliaries.iter().map(|aux| aux.cancellation),
        )
    }

    /// Inspect the actual placed relation, not the originally requested mode.
    pub fn c_is_identity(&self) -> bool {
        self.relation().c_is_identity(&self.layout)
    }

    pub fn value_count(&self) -> usize {
        self.layout.value_positions.len()
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    pub fn one(&self) -> ValueId {
        self.one
    }

    pub fn inputs(&self) -> &[ValueId] {
        &self.input_values
    }

    /// Declared fields with lowered value IDs. Added bits are listed by `auxiliaries()`.
    pub fn schema(&self) -> &[SchemaColumn] {
        &self.columns
    }

    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }

    pub fn auxiliaries(&self) -> &[AssertionAux] {
        &self.auxiliaries
    }

    pub fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    pub fn mapped_value(&self, source: ValueId) -> Option<ValueId> {
        (source.circuit == self.source && source.index < self.source_value_count).then_some(
            ValueId {
                circuit: self.id,
                index: source.index,
            },
        )
    }

    pub fn mapped_expression(&self, source: LinearExprId) -> Option<LinearExprId> {
        (source.circuit == self.source && source.index < self.source_expression_count).then_some(
            LinearExprId {
                circuit: self.id,
                index: source.index,
            },
        )
    }

    /// Target rows in execution order; a source assertion maps to two rows.
    pub fn mapped_rows(&self, source: RowId) -> Option<impl ExactSizeIterator<Item = RowId> + '_> {
        if source.circuit != self.source {
            return None;
        }
        Some(
            self.source_rows
                .get(source.index)?
                .clone()
                .map(|index| RowId {
                    circuit: self.id,
                    index,
                }),
        )
    }

    pub fn source_row(&self, target: RowId) -> Option<RowId> {
        if target.circuit != self.id || target.index >= self.rows.len() {
            return None;
        }
        let index = self
            .source_rows
            .partition_point(|range| range.end <= target.index);
        Some(RowId {
            circuit: self.source,
            index,
        })
    }

    pub fn expression(&self, id: LinearExprId) -> Option<&ExpressionNode> {
        (id.circuit == self.id)
            .then(|| self.expressions.get(id.index))
            .flatten()
            .map(|expr| &expr.node)
    }

    pub fn to_block_r1cs(
        &self,
        k_log: usize,
        k_skip: usize,
        n_log: usize,
    ) -> Result<BlockR1cs, R1csBuildError> {
        self.relation()
            .to_block_r1cs(k_log, k_skip, n_log, &self.layout)
    }

    fn relation(&self) -> RelationRef<'_> {
        RelationRef {
            id: self.id,
            expressions: &self.expressions,
            rows: &self.rows,
            value_count: self.value_count(),
            input_values: &self.input_values,
            columns: &self.columns,
            one: self.one,
        }
    }
}
