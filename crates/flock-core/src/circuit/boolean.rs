//! Typed Boolean circuit definitions with explicit materialization.
//!
//! Declare inputs, advice, witnesses, and outputs through a [`ColumnSchema`].
//! [`CircuitBuilder::compile`] reserves those columns, runs the circuit's eval
//! function, and returns resolved column bindings alongside the circuit.
//! Eval can compose reusable [operations](CircuitBuilder::operation).
//!
//! [`CircuitBuilder::xor`] records a virtual expression without a witness row.
//! [`CircuitBuilder::define_and`] and [`CircuitBuilder::define_linear`] explicitly
//! define reserved columns. Additional backend variables, when needed, belong to
//! [lowering](BooleanCircuit::lower), not the authored column schema.
//!
//! The IR keeps both the structural XOR DAG and normalized linear supports.
//! The DAG drives circuit walking; the supports drive sparse R1CS emission.
//! General `A z * B z = C z` constraints are retained; identity C is optional.
//! Logical identifiers are separate from physical matrix positions. Placement
//! follows evaluation order, adjusted for aligned words and identity-C rows.

mod builder;
mod interface;
mod layout;
mod lowering;
mod schema;
mod walk;

pub use builder::CircuitBuilder;
pub use interface::{
    Interaction, InteractionDirection, InteractionEncoding, InteractionField, InteractionScope,
    Selector,
};
pub use layout::{LayoutError, PhysicalLayout};
pub use lowering::{AssertionAux, EvaluationError, LoweredCircuit, LoweringMode, R1csBuildError};
pub use schema::{
    ColumnRole, ColumnSchema, ColumnVisitor, CompiledColumns, OperationWord, SchemaColumn,
    SchemaOperation, Var,
};
pub use walk::{ForwardTrace, WalkError, WalkLincheckCircuit, WalkPlan, WalkStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CircuitId(u64);

/// Compact circuit-local index used in normalized support storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ValueIndex(u32);

impl ValueIndex {
    fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("Boolean circuits support at most 2^32 values"))
    }

    const fn index(self) -> usize {
        self.0 as usize
    }
}

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
/// A `LinearExpr` has no witness position. Use [`CircuitBuilder::define_linear`]
/// to store its value in a reserved column, or [`CircuitBuilder::define_and`]
/// to store the product of two expressions.
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
pub(crate) struct Expression {
    node: ExpressionNode,
    /// Sorted, duplicate-free materialized values with odd coefficient.
    support: Vec<ValueIndex>,
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
    columns: Vec<SchemaColumn>,
    interactions: Vec<Interaction>,
    one: ValueId,
}

#[cfg(test)]
#[path = "boolean/tests.rs"]
mod tests;
