//! Author-facing circuit construction.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    Bit, BooleanCircuit, CircuitId, Component, Expression, ExpressionNode, Interaction, LinearExpr,
    LinearExprId, Port, PortDirection, PortEncoding, PortOrigin, Row, RowId, RowKind, ValueId,
    ValueIndex,
};

mod interface;
mod validation;

static NEXT_CIRCUIT_ID: AtomicU64 = AtomicU64::new(1);

/// Builder for one straight-line Boolean circuit.
#[derive(Debug)]
pub struct CircuitBuilder {
    id: CircuitId,
    expressions: Vec<Expression>,
    rows: Vec<Row>,
    value_count: usize,
    input_values: Vec<ValueId>,
    ports: Vec<Port>,
    components: Vec<Component>,
    active_components: Vec<usize>,
    interactions: Vec<Interaction>,
    one: Bit,
    zero: LinearExpr,
}

impl Default for CircuitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBuilder {
    /// Start a circuit with canonical zero and verifier-pinned ONE nodes.
    pub fn new() -> Self {
        let raw_id = NEXT_CIRCUIT_ID.fetch_add(1, Ordering::Relaxed);
        assert_ne!(raw_id, 0, "Boolean circuit id space exhausted");
        let id = CircuitId(raw_id);

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
            ports: Vec::new(),
            components: Vec::new(),
            active_components: Vec::new(),
            interactions: Vec::new(),
            one,
            zero,
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

    /// Allocate a free input using `input * ONE = input`.
    pub fn input(&mut self) -> Bit {
        let value = ValueId {
            circuit: self.id,
            index: self.value_count,
        };
        self.value_count += 1;
        let expression = self.push_value_expression(value);
        let row = RowId {
            circuit: self.id,
            index: self.rows.len(),
        };
        self.rows.push(Row {
            id: row,
            kind: RowKind::Input,
            lhs: expression,
            rhs: self.one.expression,
            result: expression,
            defined_value: Some(value),
        });
        self.input_values.push(value);
        Bit { value, expression }
    }

    /// Allocate a named input bit-array. The returned array is in the same
    /// order recorded by the port.
    pub fn input_bits<const N: usize>(&mut self, name: impl Into<String>) -> [Bit; N] {
        self.input_port(name, PortEncoding::Bits, PortOrigin::Witness)
    }

    /// Allocate a little-endian input word with no alignment beyond one bit.
    pub fn input_word<const N: usize>(&mut self, name: impl Into<String>) -> [Bit; N] {
        self.input_word_aligned(name, 1)
    }

    /// Allocate a little-endian input word with an explicit physical
    /// alignment requirement.
    pub fn input_word_aligned<const N: usize>(
        &mut self,
        name: impl Into<String>,
        alignment_bits: usize,
    ) -> [Bit; N] {
        assert!(alignment_bits > 0, "word alignment must be nonzero");
        self.input_port(
            name,
            PortEncoding::LittleEndianWord { alignment_bits },
            PortOrigin::Witness,
        )
    }

    fn input_port<const N: usize>(
        &mut self,
        name: impl Into<String>,
        encoding: PortEncoding,
        origin: PortOrigin,
    ) -> [Bit; N] {
        let name = name.into();
        self.assert_new_port_name(&name);
        assert!(N > 0, "port `{name}` must contain at least one bit");
        let bits = std::array::from_fn(|_| self.input());
        self.ports.push(Port {
            name,
            direction: PortDirection::Input,
            encoding,
            origin,
            values: bits.iter().map(|bit| bit.value).collect(),
        });
        bits
    }

    /// Allocate a named verifier-derived fixed bit-array. Each bit is
    /// definitionally constrained to ZERO or to the pinned ONE wire.
    pub fn fixed_bits<const N: usize>(
        &mut self,
        name: impl Into<String>,
        values: [bool; N],
    ) -> [Bit; N] {
        self.fixed_port(name, values, PortEncoding::Bits)
    }

    /// Allocate a named little-endian fixed word with no alignment beyond one
    /// bit.
    pub fn fixed_word<const N: usize>(&mut self, name: impl Into<String>, value: u64) -> [Bit; N] {
        self.fixed_word_aligned(name, 1, value)
    }

    /// Allocate a named little-endian fixed word with an explicit physical
    /// alignment requirement.
    pub fn fixed_word_aligned<const N: usize>(
        &mut self,
        name: impl Into<String>,
        alignment_bits: usize,
        value: u64,
    ) -> [Bit; N] {
        assert!(N <= 64, "fixed_word supports at most 64 bits");
        assert!(alignment_bits > 0, "word alignment must be nonzero");
        self.fixed_port(
            name,
            std::array::from_fn(|i| value >> i & 1 == 1),
            PortEncoding::LittleEndianWord { alignment_bits },
        )
    }

    fn fixed_port<const N: usize>(
        &mut self,
        name: impl Into<String>,
        values: [bool; N],
        encoding: PortEncoding,
    ) -> [Bit; N] {
        let name = name.into();
        self.assert_new_port_name(&name);
        assert!(N > 0, "port `{name}` must contain at least one bit");
        let bits = std::array::from_fn(|i| {
            let expression = if values[i] {
                self.one.expr()
            } else {
                self.zero
            };
            self.materialize(expression)
        });
        self.ports.push(Port {
            name,
            direction: PortDirection::Fixed,
            encoding,
            origin: PortOrigin::Fixed,
            values: bits.iter().map(|bit| bit.value).collect(),
        });
        bits
    }

    /// Declare an ordered group of already-materialized bits as a named output.
    /// Virtual expressions must be explicitly materialized before this call.
    pub fn output(&mut self, name: impl Into<String>, bits: impl IntoIterator<Item = Bit>) {
        self.output_port(name, bits, PortEncoding::Bits);
    }

    /// Declare an already-materialized little-endian word as a named output,
    /// with no alignment beyond one bit.
    pub fn output_word<const N: usize>(&mut self, name: impl Into<String>, bits: [Bit; N]) {
        self.output_word_aligned(name, 1, bits);
    }

    /// Declare an already-materialized little-endian word as a named output
    /// with an explicit physical alignment requirement.
    pub fn output_word_aligned<const N: usize>(
        &mut self,
        name: impl Into<String>,
        alignment_bits: usize,
        bits: [Bit; N],
    ) {
        assert!(alignment_bits > 0, "word alignment must be nonzero");
        self.output_port(
            name,
            bits,
            PortEncoding::LittleEndianWord { alignment_bits },
        );
    }

    fn output_port(
        &mut self,
        name: impl Into<String>,
        bits: impl IntoIterator<Item = Bit>,
        encoding: PortEncoding,
    ) {
        let name = name.into();
        self.assert_new_port_name(&name);
        let bits: Vec<Bit> = bits.into_iter().collect();
        assert!(
            !bits.is_empty(),
            "port `{name}` must contain at least one bit"
        );
        for bit in &bits {
            self.assert_circuit(bit.value.circuit);
        }
        let mut values: Vec<ValueId> = bits.iter().map(|bit| bit.value).collect();
        values.sort_unstable();
        assert!(
            values.windows(2).all(|pair| pair[0] != pair[1]),
            "port `{name}` contains the same bit more than once"
        );
        self.ports.push(Port {
            name,
            direction: PortDirection::Output,
            encoding,
            origin: PortOrigin::Derived,
            values: bits.into_iter().map(|bit| bit.value).collect(),
        });
    }

    /// Record an authored XOR node without allocating a witness value or row.
    pub fn xor<T: Into<LinearExpr>>(&mut self, terms: impl IntoIterator<Item = T>) -> LinearExpr {
        let terms: Vec<LinearExprId> = terms
            .into_iter()
            .map(|term| {
                let term = term.into();
                self.assert_circuit(term.id.circuit);
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

    /// Bitwise AND of two equally sized words. Every output is materialized,
    /// making the allocation cost visible in the method's return type.
    pub fn and_words<const N: usize, A: Into<LinearExpr> + Copy, B: Into<LinearExpr> + Copy>(
        &mut self,
        a: [A; N],
        b: [B; N],
    ) -> [Bit; N] {
        std::array::from_fn(|i| self.and(a[i], b[i]))
    }

    /// Explicitly materialize every bit of a virtual word.
    pub fn materialize_word<const N: usize, E: Into<LinearExpr> + Copy>(
        &mut self,
        word: [E; N],
    ) -> [Bit; N] {
        std::array::from_fn(|i| self.materialize(word[i]))
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

    /// Allocate `lhs * rhs = output` and return its materialized output bit.
    pub fn and<L: Into<LinearExpr>, R: Into<LinearExpr>>(&mut self, lhs: L, rhs: R) -> Bit {
        let lhs = lhs.into();
        let rhs = rhs.into();
        self.assert_circuit(lhs.id.circuit);
        self.assert_circuit(rhs.id.circuit);
        self.push_computed(RowKind::And, lhs.id, rhs.id)
    }

    /// Explicitly allocate `expression * ONE = output`.
    pub fn materialize(&mut self, expression: impl Into<LinearExpr>) -> Bit {
        let expression = expression.into();
        self.assert_circuit(expression.id.circuit);
        self.push_computed(RowKind::Materialize, expression.id, self.one.expression)
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
        self.assert_circuit(lhs.id.circuit);
        self.assert_circuit(rhs.id.circuit);
        self.assert_circuit(result.id.circuit);
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
    pub fn finish(self) -> BooleanCircuit {
        let definition_rows = self.validate();
        BooleanCircuit {
            id: self.id,
            expressions: self.expressions,
            rows: self.rows,
            definition_rows,
            value_count: self.value_count,
            input_values: self.input_values,
            ports: self.ports,
            components: self.components,
            interactions: self.interactions,
            one: self.one.value,
        }
    }

    fn push_computed(&mut self, kind: RowKind, lhs: LinearExprId, rhs: LinearExprId) -> Bit {
        let value = ValueId {
            circuit: self.id,
            index: self.value_count,
        };
        self.value_count += 1;
        let expression = self.push_value_expression(value);
        let row = RowId {
            circuit: self.id,
            index: self.rows.len(),
        };
        self.rows.push(Row {
            id: row,
            kind,
            lhs,
            rhs,
            result: expression,
            defined_value: Some(value),
        });
        Bit { value, expression }
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

    fn assert_new_port_name(&self, name: &str) {
        assert!(!name.is_empty(), "port name must not be empty");
        assert!(
            self.ports.iter().all(|port| port.name != name),
            "duplicate port name `{name}`"
        );
    }
}

/// Symmetric difference of sorted, duplicate-free lists.
fn symmetric_difference(lhs: &[ValueIndex], rhs: &[ValueIndex]) -> Vec<ValueIndex> {
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
