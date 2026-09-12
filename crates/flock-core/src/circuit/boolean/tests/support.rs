//! Small declared schemas shared by equation, layout, and walk checks.

use crate::circuit::boolean::*;

pub(crate) struct TestSchema {
    pub inputs: usize,
    pub witnesses: usize,
}

pub(crate) struct TestCols<T> {
    pub input: Vec<T>,
    pub witness: Vec<T>,
}

impl ColumnSchema for TestSchema {
    type Cols<T> = TestCols<T>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> TestCols<V::Value> {
        TestCols {
            // One-bit input fields keep placement in evaluation order.
            input: (0..self.inputs)
                .map(|i| v.bit(&format!("input.{i}"), ColumnRole::Input))
                .collect(),
            witness: if self.witnesses == 0 {
                Vec::new()
            } else {
                v.bits("witness", ColumnRole::Witness, self.witnesses, 1)
            },
        }
    }
}

pub(crate) fn compile(
    inputs: usize,
    witnesses: usize,
    eval: impl FnOnce(&mut CircuitBuilder, &TestCols<Var>),
) -> CompiledColumns<TestSchema> {
    CircuitBuilder::compile(TestSchema { inputs, witnesses }, eval)
}

pub(crate) fn circuit(
    inputs: usize,
    witnesses: usize,
    eval: impl FnOnce(&mut CircuitBuilder, &TestCols<Var>),
) -> BooleanCircuit {
    compile(inputs, witnesses, eval).circuit
}

/// Inspect a reserved handle using the finished circuit's IDs.
pub(crate) fn resolved<S: ColumnSchema>(compiled: &CompiledColumns<S>, var: Var) -> Bit {
    Bit {
        value: compiled.resolve(var).unwrap(),
        expression: LinearExprId {
            circuit: compiled.circuit.id,
            index: var.0.expression.index,
        },
    }
}

pub(crate) fn expression(circuit: &BooleanCircuit, expr: LinearExpr) -> LinearExpr {
    LinearExpr {
        id: LinearExprId {
            circuit: circuit.id,
            index: expr.id.index,
        },
    }
}

/// Fields with configurable roles and alignment for interface/layout checks.
pub(crate) struct Fields(pub Vec<(&'static str, ColumnRole, usize, usize)>);
impl ColumnSchema for Fields {
    type Cols<T> = Vec<Vec<T>>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Self::Cols<V::Value> {
        self.0
            .iter()
            .map(|(name, role, width, alignment)| v.bits(name, role.clone(), *width, *alignment))
            .collect()
    }
}
