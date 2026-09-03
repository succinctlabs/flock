//! Typed Boolean circuit definitions with explicit materialization.
//!
//! The language distinguishes virtual linear expressions from materialized
//! witness bits:
//!
//! - [`CircuitBuilder::xor`] records a structural expression, but allocates no
//!   witness value or R1CS row;
//! - [`CircuitBuilder::and`] allocates one value and its defining row; and
//! - [`CircuitBuilder::materialize`] explicitly allocates a linear-copy row.
//!
//! The IR keeps both the structural XOR DAG and normalized linear supports.
//! The DAG drives circuit walking; the supports drive sparse R1CS emission.
//! General `A z * B z = C z` constraints are retained;
//! `C = I` is an optional lowering/prover optimization, not a language rule.
//! Logical identifiers are separate from physical matrix positions. Source
//! order is the default; compatibility layouts may permute it.
//!
//! The public types live here; authoring, layout, and lowering are separate
//! implementation modules.

mod builder;
mod layout;
mod lowering;
mod walk;

pub use builder::CircuitBuilder;
pub use layout::{LayoutBuilder, LayoutError, PhysicalLayout, PositionKind};
pub use lowering::{EvaluationError, R1csBuildError};
pub use walk::{ForwardTrace, WalkError, WalkPlan, WalkStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CircuitId(u64);

/// An identifier for a structural linear-expression node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinearExprId {
    circuit: CircuitId,
    index: usize,
}

impl LinearExprId {
    /// The deterministic source-order index of this expression.
    pub const fn index(self) -> usize {
        self.index
    }
}

/// An identifier for a materialized witness value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId {
    circuit: CircuitId,
    index: usize,
}

impl ValueId {
    /// The deterministic source-order index of this value.
    pub const fn index(self) -> usize {
        self.index
    }
}

/// An identifier for a constraint row, whether definitional or general.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowId {
    circuit: CircuitId,
    index: usize,
}

impl RowId {
    /// The deterministic source-order index of this row.
    pub const fn index(self) -> usize {
        self.index
    }
}

/// A materialized Boolean witness position.
///
/// Copying, naming, putting in an array, or slicing a `Bit` allocates nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bit {
    value: ValueId,
    expression: LinearExprId,
}

impl Bit {
    /// Use this materialized value as a linear expression.
    pub const fn expr(self) -> LinearExpr {
        LinearExpr {
            id: self.expression,
        }
    }

    /// The logical value identifier. Physical R1CS placement is a separate
    /// lowering concern.
    pub const fn value_id(self) -> ValueId {
        self.value
    }
}

/// A virtual linear expression over materialized bits.
///
/// A `LinearExpr` has no witness position. Only [`CircuitBuilder::and`] or
/// [`CircuitBuilder::materialize`] can turn one into a new [`Bit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LinearExpr {
    id: LinearExprId,
}

impl LinearExpr {
    /// The structural expression-node identifier.
    pub const fn id(self) -> LinearExprId {
        self.id
    }
}

impl From<Bit> for LinearExpr {
    fn from(bit: Bit) -> Self {
        bit.expr()
    }
}

impl From<&Bit> for LinearExpr {
    fn from(bit: &Bit) -> Self {
        bit.expr()
    }
}

impl From<&LinearExpr> for LinearExpr {
    fn from(expression: &LinearExpr) -> Self {
        *expression
    }
}

/// The statement-facing role of a named materialized port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortDirection {
    Input,
    Output,
    Fixed,
}

/// The statement-facing encoding of a named port.
///
/// Words always use little-endian bit order: `values()[0]` is the least
/// significant bit. `alignment_bits` constrains that value's physical
/// position; [`LayoutBuilder::place_port`] places the remaining bits
/// contiguously from there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortEncoding {
    /// An ordered collection of bits with no numeric interpretation.
    Bits,
    /// A little-endian word with an explicit physical alignment requirement.
    LittleEndianWord { alignment_bits: usize },
}

/// A named ordered group of materialized bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    name: String,
    direction: PortDirection,
    encoding: PortEncoding,
    values: Vec<ValueId>,
}

impl Port {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn direction(&self) -> PortDirection {
        self.direction
    }

    pub const fn encoding(&self) -> PortEncoding {
        self.encoding
    }

    pub fn values(&self) -> &[ValueId] {
        &self.values
    }
}

/// One node in the authored structural expression DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpressionNode {
    /// The zero linear expression.
    Zero,
    /// A materialized-value boundary.
    Value(ValueId),
    /// An XOR exactly as authored. Nested XOR nodes remain nested here even
    /// when their derived normalized supports cancel or flatten.
    Xor(Vec<LinearExprId>),
}

/// The semantic role of one constraint row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    /// The distinguished `ONE * ONE = ONE` row. The relation also records a
    /// [`crate::r1cs::BlockR1cs::const_pin`]; this row alone proves only
    /// Booleanity.
    One,
    /// A free input tautology: `input * ONE = input`.
    Input,
    /// A nonlinear definition: `lhs * rhs = output`.
    And,
    /// An explicit linear copy: `expr * ONE = output`.
    Materialize,
    /// A general constraint `lhs * rhs = result` that allocates no value.
    Constraint,
}

/// One logical circuit-R1CS row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    id: RowId,
    kind: RowKind,
    lhs: LinearExprId,
    rhs: LinearExprId,
    result: LinearExprId,
    defined_value: Option<ValueId>,
}

impl Row {
    pub const fn id(&self) -> RowId {
        self.id
    }

    pub const fn kind(&self) -> RowKind {
        self.kind
    }

    pub const fn lhs(&self) -> LinearExprId {
        self.lhs
    }

    pub const fn rhs(&self) -> LinearExprId {
        self.rhs
    }

    /// The C-side linear expression.
    pub const fn result(&self) -> LinearExprId {
        self.result
    }

    /// The value defined by this row, when it is a definitional row.
    pub const fn defined_value(&self) -> Option<ValueId> {
        self.defined_value
    }
}

#[derive(Clone, Debug)]
struct Expression {
    node: ExpressionNode,
    /// Sorted, duplicate-free materialized values with odd coefficient.
    support: Vec<ValueId>,
}

/// A completed, deterministic Boolean circuit artifact.
#[derive(Clone, Debug)]
pub struct BooleanCircuit {
    id: CircuitId,
    expressions: Vec<Expression>,
    rows: Vec<Row>,
    definition_rows: Vec<RowId>,
    value_count: usize,
    input_values: Vec<ValueId>,
    ports: Vec<Port>,
    one: ValueId,
}

#[cfg(test)]
#[path = "boolean/tests.rs"]
mod tests;
