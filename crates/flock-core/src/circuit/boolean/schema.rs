//! Typed column reservation and resolution for one circuit invocation.

use std::ops::Range;

use super::{
    Bit, BooleanCircuit, CircuitBuilder, CircuitId, EvaluationError, ExpressionNode, LinearExpr,
    LinearExprId, RowKind, ValueId,
};

/// A reserved column. Resolve it after finalization before using a physical layout.
///
/// ```compile_fail
/// use flock_core::circuit::boolean::Var;
/// fn premature_id(var: Var) { var.value_id(); }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub(super) Bit);

impl From<Var> for LinearExpr {
    fn from(value: Var) -> Self {
        value.0.expr()
    }
}

/// Local ownership of a schema field. Public/outer bindings remain separate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnRole {
    Input,
    Advice(&'static str),
    Witness,
    Output,
    Fixed(bool),
}

impl ColumnRole {
    pub(super) fn is_input(&self) -> bool {
        matches!(self, Self::Input | Self::Advice(_))
    }
}

/// One traversal describes allocation, named inspection, and typed witness access.
/// The schema value holds fixed configuration, such as a row count, never witness data.
pub trait ColumnSchema {
    type Cols<T>;
    fn columns<V: ColumnVisitor>(&self, visitor: &mut V) -> Self::Cols<V::Value>;
}

/// Visits fields in declaration order; word bits are little-endian.
pub trait ColumnVisitor {
    type Value;
    fn bits(
        &mut self,
        name: &str,
        role: ColumnRole,
        width: usize,
        alignment_bits: usize,
    ) -> Vec<Self::Value>;

    fn word<const N: usize>(&mut self, name: &str, role: ColumnRole) -> [Self::Value; N] {
        self.word_aligned(name, role, 1)
    }

    fn word_aligned<const N: usize>(
        &mut self,
        name: &str,
        role: ColumnRole,
        alignment_bits: usize,
    ) -> [Self::Value; N] {
        self.bits(name, role, N, alignment_bits)
            .try_into()
            .ok()
            .expect("schema width changed")
    }

    fn bit(&mut self, name: &str, role: ColumnRole) -> Self::Value {
        let [value] = self.word(name, role);
        value
    }
}

/// One named word, including internal witnesses that are not exported ports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaColumn {
    pub name: String,
    pub role: ColumnRole,
    pub values: Vec<ValueId>,
    pub alignment_bits: usize,
}

/// A named little-endian operation argument or result, referring to the shared DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationWord {
    pub name: String,
    pub expressions: Vec<LinearExprId>,
}

/// Prototype composition metadata; ranges refer to the same canonical circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaOperation {
    pub name: String,
    pub kind: String,
    pub columns: Vec<ValueId>,
    pub inputs: Vec<OperationWord>,
    pub output: OperationWord,
    pub rows: Range<usize>,
    pub expressions: Range<usize>,
    pub interactions: Range<usize>,
}

#[derive(Debug)]
pub(super) struct SchemaState {
    pub columns: Vec<SchemaColumn>,
    pub defined: Vec<bool>,
    pub operations: Vec<SchemaOperation>,
}

/// A canonical circuit and the construction-to-finalized column mapping.
#[derive(Debug)]
pub struct CompiledColumns<S: ColumnSchema> {
    pub(super) construction_id: CircuitId,
    pub(super) circuit: BooleanCircuit,
    pub(super) columns: Vec<SchemaColumn>,
    pub(super) resolved: Vec<ValueId>,
    pub(super) operations: Vec<SchemaOperation>,
    pub(super) schema: S,
}

impl<S: ColumnSchema> CompiledColumns<S> {
    pub fn circuit(&self) -> &BooleanCircuit {
        &self.circuit
    }

    /// Prototype inspection data; this is not a serialized v1 artifact.
    pub fn schema(&self) -> &[SchemaColumn] {
        &self.columns
    }

    pub fn operations(&self) -> &[SchemaOperation] {
        &self.operations
    }

    /// Unconsumed schema bits, reported as (field path, bit offset). Nothing is pruned.
    pub fn unused_values(&self) -> Vec<(&str, usize)> {
        let mut used = vec![false; self.circuit.value_count()];
        let mut seen = vec![false; self.circuit.expression_count()];
        let mut pending = Vec::new();
        for row in &self.circuit.rows {
            if matches!(row.kind, RowKind::One | RowKind::Input) {
                continue;
            }
            pending.extend([row.lhs, row.rhs]);
            if row.kind == RowKind::Constraint {
                pending.push(row.result);
            }
        }
        while let Some(expression) = pending.pop() {
            if std::mem::replace(&mut seen[expression.index], true) {
                continue;
            }
            match &self.circuit.expressions[expression.index].node {
                ExpressionNode::Zero => {}
                ExpressionNode::Value(value) => used[value.index] = true,
                ExpressionNode::Xor(terms) => pending.extend(terms),
            }
        }
        for column in &self.columns {
            if column.role == ColumnRole::Output {
                for value in &column.values {
                    used[value.index] = true;
                }
            }
        }
        for interaction in &self.circuit.interactions {
            used[interaction.selector.index] = true;
            for value in interaction
                .multiplicity
                .iter()
                .chain(interaction.message.iter().flat_map(|field| &field.values))
            {
                used[value.index] = true;
            }
        }
        let mut unused = Vec::new();
        for column in &self.columns {
            for (offset, value) in column.values.iter().enumerate() {
                if !used[value.index] {
                    unused.push((column.name.as_str(), offset));
                }
            }
        }
        unused
    }

    pub fn resolve(&self, var: Var) -> Option<ValueId> {
        if var.0.value.circuit != self.construction_id {
            return None;
        }
        self.resolved.get(var.0.value.index).copied()
    }

    /// Resolved typed columns for layout adapters and inspection.
    pub fn columns(&self) -> S::Cols<ValueId> {
        self.visit(|value| value)
    }

    /// Supply named inputs/advice, initially zero. The evaluator fills all derived fields.
    /// Values written to derived fields by `supply` are ignored.
    pub fn evaluate(
        &self,
        supply: impl FnOnce(S::Cols<&mut bool>),
    ) -> Result<(S::Cols<bool>, Vec<bool>), EvaluationError> {
        let values = self.circuit.evaluate(&self.inputs(supply))?;
        let cols = self.visit(|value| values[value.index()]);
        Ok((cols, values))
    }

    /// Encode zero-initialized inputs/advice without running constraints.
    /// Writes to deterministic fields are ignored, as in `evaluate`.
    pub fn inputs(&self, supply: impl FnOnce(S::Cols<&mut bool>)) -> Vec<bool> {
        let mut supplied = vec![false; self.circuit.value_count()];
        // Schema handles enumerate distinct values in reservation order.
        let mut cells = supplied[1..].iter_mut();
        supply(self.visit(|_| cells.next().expect("schema width changed")));
        let mut inputs = Vec::new();
        let mut offset = 1;
        for column in &self.columns {
            if column.role.is_input() {
                inputs.extend_from_slice(&supplied[offset..offset + column.values.len()]);
            }
            offset += column.values.len();
        }
        inputs
    }

    fn visit<T>(&self, map: impl FnMut(ValueId) -> T) -> S::Cols<T> {
        let mut visitor = ResolvedVisitor {
            columns: self.columns.iter(),
            map,
        };
        let result = self.schema.columns(&mut visitor);
        assert!(visitor.columns.next().is_none(), "schema traversal changed");
        result
    }
}

struct ResolvedVisitor<'a, F> {
    columns: std::slice::Iter<'a, SchemaColumn>,
    map: F,
}

impl<T, F: FnMut(ValueId) -> T> ColumnVisitor for ResolvedVisitor<'_, F> {
    type Value = T;

    fn bits(
        &mut self,
        name: &str,
        role: ColumnRole,
        width: usize,
        alignment_bits: usize,
    ) -> Vec<T> {
        let column = self.columns.next().expect("schema traversal changed");
        assert_eq!(column.name, name, "schema traversal changed");
        assert_eq!(column.role, role, "schema traversal changed");
        assert_eq!(column.values.len(), width, "schema traversal changed");
        assert_eq!(
            column.alignment_bits, alignment_bits,
            "schema traversal changed"
        );
        column
            .values
            .iter()
            .map(|&value| (self.map)(value))
            .collect()
    }
}

impl CircuitBuilder {
    /// Reserve a schema, run its one constraint definition, and finalize it.
    pub fn compile<S: ColumnSchema>(
        schema: S,
        eval: impl FnOnce(&mut Self, &S::Cols<Var>),
    ) -> CompiledColumns<S> {
        let mut builder = Self::new();
        let cols = builder.reserve_columns(&schema);
        eval(&mut builder, &cols);
        builder.finish_columns(schema)
    }
}

#[cfg(test)]
mod tests;
