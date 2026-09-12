//! Author-facing circuit construction.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    Bit, BooleanCircuit, CircuitId, Expression, ExpressionNode, Interaction, LinearExpr,
    LinearExprId, Row, RowId, RowKind, ValueId, ValueIndex,
};

mod interface;
mod operations;
mod schema;
mod validation;

static NEXT_CIRCUIT_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn fresh_circuit_id() -> CircuitId {
    let raw_id = NEXT_CIRCUIT_ID.fetch_add(1, Ordering::Relaxed);
    assert_ne!(raw_id, 0, "Boolean circuit id space exhausted");
    CircuitId(raw_id)
}

/// Builder for one straight-line Boolean circuit.
#[derive(Debug)]
pub struct CircuitBuilder {
    id: CircuitId,
    expressions: Vec<Expression>,
    rows: Vec<Row>,
    value_count: usize,
    input_values: Vec<ValueId>,
    interactions: Vec<Interaction>,
    one: Bit,
    zero: LinearExpr,
    schema: crate::circuit::boolean::schema::SchemaState,
}

impl CircuitBuilder {
    /// Start a circuit with canonical zero and verifier-pinned ONE nodes.
    pub(super) fn new() -> Self {
        let id = fresh_circuit_id();

        let zero = LinearExpr {
            id: LinearExprId {
                circuit: id,
                index: 0,
            },
        };
        let one_value = ValueId {
            circuit: id,
            index: 0,
        };
        let one_expression = LinearExprId {
            circuit: id,
            index: 1,
        };
        let one = Bit {
            value: one_value,
            expression: one_expression,
        };
        Self {
            id,
            expressions: vec![
                Expression {
                    node: ExpressionNode::Zero,
                    support: Vec::new(),
                },
                Expression {
                    node: ExpressionNode::Value(one_value),
                    support: vec![ValueIndex::new(one_value.index)],
                },
            ],
            rows: vec![Row {
                id: RowId {
                    circuit: id,
                    index: 0,
                },
                kind: RowKind::One,
                lhs: one_expression,
                rhs: one_expression,
                result: one_expression,
                defined_value: Some(one_value),
            }],
            value_count: 1,
            input_values: Vec::new(),
            interactions: Vec::new(),
            one,
            zero,
            schema: crate::circuit::boolean::schema::SchemaState {
                columns: Vec::new(),
                defined: vec![true],
                operations: Vec::new(),
            },
        }
    }

    /// The distinguished constant-one materialized bit.
    pub const fn one(&self) -> Bit {
        self.one
    }

    /// The virtual zero expression.
    pub const fn zero(&self) -> LinearExpr {
        self.zero
    }

    /// Record an authored XOR node without allocating a witness value or row.
    pub fn xor<T: Into<LinearExpr>>(&mut self, terms: impl IntoIterator<Item = T>) -> LinearExpr {
        let terms: Vec<LinearExprId> = terms
            .into_iter()
            .map(|term| {
                let term = term.into();
                self.assert_available(term);
                term.id
            })
            .collect();
        let mut support = Vec::new();
        for &term in &terms {
            support = symmetric_difference(&support, &self.expressions[term.index].support);
        }
        let id = LinearExprId {
            circuit: self.id,
            index: self.expressions.len(),
        };
        self.expressions.push(Expression {
            node: ExpressionNode::Xor(terms),
            support,
        });
        LinearExpr { id }
    }

    /// Concise two-term XOR accepting either bits or expressions.
    pub fn xor2<A: Into<LinearExpr>, B: Into<LinearExpr>>(&mut self, a: A, b: B) -> LinearExpr {
        let a = a.into();
        let b = b.into();
        self.xor([a, b])
    }

    /// Concise three-term XOR accepting either bits or expressions.
    pub fn xor3<A: Into<LinearExpr>, B: Into<LinearExpr>, C: Into<LinearExpr>>(
        &mut self,
        a: A,
        b: B,
        c: C,
    ) -> LinearExpr {
        let a = a.into();
        let b = b.into();
        let c = c.into();
        self.xor([a, b, c])
    }

    /// Bitwise XOR of two equally sized words. Results remain virtual.
    pub fn xor2_words<const N: usize, A: Into<LinearExpr> + Copy, B: Into<LinearExpr> + Copy>(
        &mut self,
        a: [A; N],
        b: [B; N],
    ) -> [LinearExpr; N] {
        std::array::from_fn(|i| self.xor2(a[i], b[i]))
    }

    /// Bitwise XOR of three equally sized words. Results remain virtual.
    pub fn xor3_words<
        const N: usize,
        A: Into<LinearExpr> + Copy,
        B: Into<LinearExpr> + Copy,
        C: Into<LinearExpr> + Copy,
    >(
        &mut self,
        a: [A; N],
        b: [B; N],
        c: [C; N],
    ) -> [LinearExpr; N] {
        std::array::from_fn(|i| self.xor3(a[i], b[i], c[i]))
    }

    /// Rotate a word right without allocating values or rows.
    pub fn rotate_right<const N: usize, E: Into<LinearExpr> + Copy>(
        &self,
        word: [E; N],
        amount: usize,
    ) -> [LinearExpr; N] {
        assert!(N > 0, "cannot rotate an empty word");
        let word = word.map(Into::into);
        for expression in &word {
            self.assert_circuit(expression.id.circuit);
        }
        let amount = amount % N;
        std::array::from_fn(|i| word[(i + amount) % N])
    }

    /// Logical right shift of a little-endian bit word. Shifted-in high bits
    /// are the virtual zero expression; no values or rows are allocated.
    pub fn shift_right<const N: usize, E: Into<LinearExpr> + Copy>(
        &self,
        word: [E; N],
        amount: usize,
    ) -> [LinearExpr; N] {
        let word = word.map(Into::into);
        for expression in &word {
            self.assert_circuit(expression.id.circuit);
        }
        std::array::from_fn(|i| {
            i.checked_add(amount)
                .filter(|&source| source < N)
                .map_or(self.zero, |source| word[source])
        })
    }

    /// Add the general R1CS row `lhs * rhs = result` without allocating a
    /// witness value. This is the non-definitional escape hatch needed for
    /// compact `C != I` relations and rejecting assertions.
    pub fn constrain<L: Into<LinearExpr>, R: Into<LinearExpr>, C: Into<LinearExpr>>(
        &mut self,
        lhs: L,
        rhs: R,
        result: C,
    ) -> RowId {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result = result.into();
        self.assert_available(lhs);
        self.assert_available(rhs);
        self.assert_available(result);
        let id = RowId {
            circuit: self.id,
            index: self.rows.len(),
        };
        self.rows.push(Row {
            id,
            kind: RowKind::Constraint,
            lhs: lhs.id,
            rhs: rhs.id,
            result: result.id,
            defined_value: None,
        });
        id
    }

    /// Assert `lhs * rhs = 0` without allocating a product bit.
    pub fn assert_zero_product<L: Into<LinearExpr>, R: Into<LinearExpr>>(
        &mut self,
        lhs: L,
        rhs: R,
    ) -> RowId {
        self.constrain(lhs, rhs, self.zero)
    }

    /// Assert `expression = 0` without allocating a copied result bit.
    pub fn assert_zero(&mut self, expression: impl Into<LinearExpr>) -> RowId {
        self.constrain(expression, self.one.expr(), self.zero)
    }

    /// Number of structural nodes currently recorded.
    pub fn expression_count(&self) -> usize {
        self.expressions.len()
    }

    /// Number of materialized values currently allocated.
    pub fn value_count(&self) -> usize {
        self.value_count
    }

    /// Number of constraint rows currently allocated.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Complete and validate the circuit.
    fn finish(self) -> BooleanCircuit {
        let definition_rows = self.validate();
        BooleanCircuit {
            id: self.id,
            expressions: self.expressions,
            rows: self.rows,
            definition_rows,
            value_count: self.value_count,
            input_values: self.input_values,
            columns: self.schema.columns,
            interactions: self.interactions,
            one: self.one.value,
        }
    }

    fn push_value_expression(&mut self, value: ValueId) -> LinearExprId {
        let id = LinearExprId {
            circuit: self.id,
            index: self.expressions.len(),
        };
        self.expressions.push(Expression {
            node: ExpressionNode::Value(value),
            support: vec![ValueIndex::new(value.index)],
        });
        id
    }

    fn assert_circuit(&self, circuit: CircuitId) {
        assert_eq!(
            circuit, self.id,
            "expression belongs to another circuit builder"
        );
    }
}

/// Symmetric difference of sorted, duplicate-free lists.
pub(super) fn symmetric_difference(lhs: &[ValueIndex], rhs: &[ValueIndex]) -> Vec<ValueIndex> {
    let mut result = Vec::with_capacity(lhs.len() + rhs.len());
    let (mut i, mut j) = (0, 0);
    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            std::cmp::Ordering::Less => {
                result.push(lhs[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(rhs[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&lhs[i..]);
    result.extend_from_slice(&rhs[j..]);
    result
}

#[cfg(test)]
#[path = "builder/tests.rs"]
mod tests;
