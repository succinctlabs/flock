//! Circuit inspection, evaluation, digests, and sparse R1CS lowering.

use std::fmt;

pub(super) mod relation;

mod two_variable;
pub use two_variable::{AssertionAux, LoweredCircuit, LoweringMode};

use crate::r1cs::BlockR1cs;

use super::{
    BooleanCircuit, ColumnRole, ExpressionNode, Interaction, LayoutError, LinearExprId,
    PhysicalLayout, Row, RowId, RowKind, SchemaColumn, ValueId, ValueIndex,
};

impl BooleanCircuit {
    /// Number of structural expression nodes, including the canonical zero
    /// node and each materialized value's boundary node.
    pub fn expression_count(&self) -> usize {
        self.expressions.len()
    }

    /// Structural expression nodes in deterministic source order.
    ///
    /// This is the complete authored DAG, including nodes that are not used by
    /// a row. Consumers can use each node's position as its stable local ID.
    pub fn expressions(&self) -> impl ExactSizeIterator<Item = &ExpressionNode> {
        self.expressions.iter().map(|expression| &expression.node)
    }

    /// Total number of materialized-value references retained across all
    /// canonical normalized supports.
    pub fn normalized_support_terms(&self) -> usize {
        self.relation().normalized_support_terms()
    }

    /// Payload bytes occupied by all compact normalized-support entries.
    pub fn normalized_support_bytes(&self) -> usize {
        self.normalized_support_terms() * std::mem::size_of::<ValueIndex>()
    }

    /// Number of materialized values.
    pub fn value_count(&self) -> usize {
        self.value_count
    }

    /// Number of logical constraint rows, including non-definitional rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// The constant-one materialized value.
    pub const fn one(&self) -> ValueId {
        self.one
    }

    /// Input values, in declaration order.
    pub fn inputs(&self) -> &[ValueId] {
        &self.input_values
    }

    /// Declared fields, in declaration order.
    pub fn schema(&self) -> &[SchemaColumn] {
        &self.columns
    }

    /// Find a declared field by name.
    pub fn column(&self, name: &str) -> Option<&SchemaColumn> {
        self.columns.iter().find(|column| column.name == name)
    }

    /// Deferred global interactions referenced by this circuit.
    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }

    /// The row that defines `value`.
    pub fn definition_row(&self, value: ValueId) -> Option<RowId> {
        if value.circuit != self.id {
            return None;
        }
        self.definition_rows.get(value.index).copied()
    }

    /// Place values automatically, keeping input/advice/output/fixed words contiguous and aligned.
    pub fn layout(&self) -> Result<PhysicalLayout, LayoutError> {
        PhysicalLayout::new(self)
    }

    /// Deterministic digest of the authored local relation and input/output/fixed field shape.
    /// Runtime circuit IDs, advice labels, internal field metadata, and interactions are excluded.
    pub fn structure_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"flock-boolean-structure-v1");
        absorb_usize(&mut hasher, self.expressions.len());
        for expression in &self.expressions {
            match &expression.node {
                ExpressionNode::Zero => {
                    hasher.update(&[0]);
                }
                ExpressionNode::Value(value) => {
                    hasher.update(&[1]);
                    absorb_usize(&mut hasher, value.index);
                }
                ExpressionNode::Xor(terms) => {
                    hasher.update(&[2]);
                    absorb_usize(&mut hasher, terms.len());
                    for term in terms {
                        absorb_usize(&mut hasher, term.index);
                    }
                }
            }
            absorb_usize(&mut hasher, expression.support.len());
            for value in &expression.support {
                absorb_usize(&mut hasher, value.index());
            }
        }
        absorb_usize(&mut hasher, self.rows.len());
        for row in &self.rows {
            hasher.update(&[match row.kind {
                RowKind::One => 0,
                RowKind::Input => 1,
                RowKind::And => 2,
                RowKind::Materialize => 3,
                RowKind::Constraint => 4,
            }]);
            absorb_usize(&mut hasher, row.id.index);
            absorb_usize(&mut hasher, row.lhs.index);
            absorb_usize(&mut hasher, row.rhs.index);
            absorb_usize(&mut hasher, row.result.index);
            hasher.update(&[row.defined_value.is_some() as u8]);
            absorb_usize(
                &mut hasher,
                row.defined_value.map_or(0, |value| value.index),
            );
        }
        absorb_usize(&mut hasher, self.input_values.len());
        for value in &self.input_values {
            absorb_usize(&mut hasher, value.index);
        }
        let columns = self
            .columns
            .iter()
            .filter(|column| column.role != ColumnRole::Witness);
        absorb_usize(&mut hasher, columns.clone().count());
        for column in columns {
            absorb_usize(&mut hasher, column.name.len());
            hasher.update(column.name.as_bytes());
            hasher.update(&[match column.role {
                ColumnRole::Input | ColumnRole::Advice(_) => 0,
                ColumnRole::Output => 1,
                ColumnRole::Fixed(_) => 2,
                ColumnRole::Witness => unreachable!(),
            }]);
            hasher.update(&[1]); // Little-endian word tag in the v1 digest.
            absorb_usize(&mut hasher, column.alignment_bits);
            absorb_usize(&mut hasher, column.values.len());
            for value in &column.values {
                absorb_usize(&mut hasher, value.index);
            }
        }
        absorb_usize(&mut hasher, self.one.index);
        *hasher.finalize().as_bytes()
    }

    /// Structural digest extended with one physical layout.
    pub fn structure_layout_digest(
        &self,
        layout: &PhysicalLayout,
    ) -> Result<[u8; 32], LayoutError> {
        layout.validate_for(self)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"flock-boolean-structure-layout-v1");
        hasher.update(&self.structure_digest());
        absorb_usize(&mut hasher, layout.value_positions.len());
        for &position in &layout.value_positions {
            absorb_usize(&mut hasher, position);
        }
        absorb_usize(&mut hasher, layout.row_positions.len());
        for &position in &layout.row_positions {
            absorb_usize(&mut hasher, position);
        }
        absorb_usize(&mut hasher, layout.useful_bits);
        Ok(*hasher.finalize().as_bytes())
    }

    /// Logical rows, in deterministic definition order.
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Inspect one authored structural node.
    pub fn expression(&self, id: LinearExprId) -> Option<&ExpressionNode> {
        if id.circuit != self.id {
            return None;
        }
        self.expressions
            .get(id.index)
            .map(|expression| &expression.node)
    }

    /// The canonical sorted support of one expression over materialized
    /// values. Duplicate terms have cancelled by parity. The returned typed
    /// IDs are reconstructed from compact circuit-local storage.
    pub fn support(&self, id: LinearExprId) -> Option<Vec<ValueId>> {
        if id.circuit != self.id {
            return None;
        }
        self.expressions.get(id.index).map(|expression| {
            expression
                .support
                .iter()
                .map(|value| ValueId {
                    circuit: self.id,
                    index: value.index(),
                })
                .collect()
        })
    }

    /// Evaluate the circuit in logical source order.
    ///
    /// The returned vector is indexed by [`ValueId`]. It is not padded to an
    /// R1CS power of two; use [`Self::evaluate_r1cs`] for that representation.
    pub fn evaluate(&self, inputs: &[bool]) -> Result<Vec<bool>, EvaluationError> {
        self.relation().evaluate(inputs)
    }

    /// Evaluate and pad one block to `2^k_log` bits in automatic layout.
    pub fn evaluate_r1cs(
        &self,
        inputs: &[bool],
        k_log: usize,
    ) -> Result<Vec<bool>, EvaluationError> {
        let layout = self.layout().map_err(EvaluationError::InvalidLayout)?;
        self.evaluate_r1cs_with_layout(inputs, k_log, &layout)
    }

    /// Evaluate and place one block according to an explicit physical layout.
    pub fn evaluate_r1cs_with_layout(
        &self,
        inputs: &[bool],
        k_log: usize,
        layout: &PhysicalLayout,
    ) -> Result<Vec<bool>, EvaluationError> {
        layout
            .validate_for(self)
            .map_err(EvaluationError::InvalidLayout)?;
        let capacity = checked_capacity(k_log).map_err(EvaluationError::InvalidKLog)?;
        let required = layout.useful_bits;
        if capacity < required {
            return Err(EvaluationError::Capacity {
                required,
                actual: capacity,
            });
        }
        let logical = self.evaluate(inputs)?;
        let mut physical = vec![false; capacity];
        for (value, &bit) in logical.iter().enumerate() {
            physical[layout.value_positions[value]] = bit;
        }
        Ok(physical)
    }

    /// Emit automatically placed sparse matrices for one repeated block.
    ///
    /// `n_log` chooses how many identical blocks the returned instance tiles;
    /// it does not alter the base circuit. Padding rows use `0 * 0 = z[i]`.
    /// Definitional rows use their output values on the C side; general rows
    /// retain their authored C expressions. Thus the result may or may not
    /// have identity C. The constant-one wire is verifier-pinned through
    /// [`BlockR1cs::const_pin`].
    pub fn to_block_r1cs(
        &self,
        k_log: usize,
        k_skip: usize,
        n_log: usize,
    ) -> Result<BlockR1cs, R1csBuildError> {
        let layout = self.layout().map_err(R1csBuildError::InvalidLayout)?;
        self.to_block_r1cs_with_layout(k_log, k_skip, n_log, &layout)
    }

    /// Emit sparse matrices using an explicit logical-to-physical layout.
    pub fn to_block_r1cs_with_layout(
        &self,
        k_log: usize,
        k_skip: usize,
        n_log: usize,
        layout: &PhysicalLayout,
    ) -> Result<BlockR1cs, R1csBuildError> {
        self.relation().to_block_r1cs(k_log, k_skip, n_log, layout)
    }

    #[cfg(test)]
    pub(super) fn eval_expression(&self, id: LinearExprId, values: &[bool]) -> bool {
        self.relation().eval_expression(id, values)
    }
}

/// A reference-evaluation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationError {
    InputCount { expected: usize, actual: usize },
    InvalidKLog(usize),
    Capacity { required: usize, actual: usize },
    UnsatisfiedRow(RowId),
    InvalidLayout(LayoutError),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCount { expected, actual } => {
                write!(f, "expected {expected} inputs, received {actual}")
            }
            Self::InvalidKLog(k_log) => write!(f, "2^{k_log} does not fit in usize"),
            Self::Capacity { required, actual } => {
                write!(
                    f,
                    "circuit needs capacity {required} for its values and rows, but capacity is {actual}"
                )
            }
            Self::UnsatisfiedRow(row) => {
                write!(f, "general constraint row {} is not satisfied", row.index())
            }
            Self::InvalidLayout(error) => write!(f, "invalid physical layout: {error}"),
        }
    }
}

impl std::error::Error for EvaluationError {}

/// A sparse-R1CS lowering failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum R1csBuildError {
    InvalidKLog(usize),
    InvalidKSkip { k_log: usize, k_skip: usize },
    Capacity { required: usize, actual: usize },
    DimensionOverflow,
    InvalidLayout(LayoutError),
}

impl fmt::Display for R1csBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKLog(k_log) => write!(f, "2^{k_log} does not fit in usize"),
            Self::InvalidKSkip { k_log, k_skip } => {
                write!(f, "k_skip ({k_skip}) exceeds k_log ({k_log})")
            }
            Self::Capacity { required, actual } => {
                write!(
                    f,
                    "circuit needs capacity {required} for its values and rows, but capacity is {actual}"
                )
            }
            Self::DimensionOverflow => write!(f, "R1CS dimension overflows usize"),
            Self::InvalidLayout(error) => write!(f, "invalid physical layout: {error}"),
        }
    }
}

impl std::error::Error for R1csBuildError {}

fn checked_capacity(k_log: usize) -> Result<usize, usize> {
    let shift = u32::try_from(k_log).map_err(|_| k_log)?;
    1usize.checked_shl(shift).ok_or(k_log)
}

fn absorb_usize(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}
