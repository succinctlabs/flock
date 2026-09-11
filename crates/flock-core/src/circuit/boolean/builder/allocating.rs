//! Allocating convenience API for circuits without a typed column schema.

use super::*;
use crate::circuit::boolean::Selector;

impl CircuitBuilder {
    /// Allocate a free input using `input * ONE = input`.
    pub fn input(&mut self) -> Bit {
        self.assert_legacy_allocation();
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
        self.assert_legacy_allocation();
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
        self.assert_legacy_allocation();
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

    /// Allocate a named selector input.
    pub fn input_selector(&mut self, name: impl Into<String>) -> Selector {
        let [bit] = self.input_port(name, PortEncoding::Bits, PortOrigin::Witness);
        Selector { bit }
    }

    /// Allocate a named advice bit-array. Advice is untrusted input whose type
    /// identifies its honest generator; constraints must establish validity.
    pub fn advice_bits<const N: usize>(
        &mut self,
        name: impl Into<String>,
        advice_type: impl Into<String>,
    ) -> [Bit; N] {
        self.advice_port(name, advice_type, PortEncoding::Bits)
    }

    /// Allocate a named little-endian advice word.
    pub fn advice_word<const N: usize>(
        &mut self,
        name: impl Into<String>,
        advice_type: impl Into<String>,
    ) -> [Bit; N] {
        self.advice_port(
            name,
            advice_type,
            PortEncoding::LittleEndianWord { alignment_bits: 1 },
        )
    }

    fn advice_port<const N: usize>(
        &mut self,
        name: impl Into<String>,
        advice_type: impl Into<String>,
        encoding: PortEncoding,
    ) -> [Bit; N] {
        let advice_type = advice_type.into();
        assert!(!advice_type.is_empty(), "advice type must not be empty");
        self.input_port(name, encoding, PortOrigin::Advice { advice_type })
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
}
