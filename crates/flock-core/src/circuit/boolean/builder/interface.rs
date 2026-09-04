//! Selector, advice, component, and interaction authoring.

use super::*;
use crate::circuit::boolean::{
    InteractionDirection, InteractionEncoding, InteractionField, InteractionScope, Selector,
};

impl CircuitBuilder {
    /// Treat an existing materialized bit as a constraint/interaction guard.
    /// This is a typed view and allocates nothing.
    pub fn selector(&self, bit: Bit) -> Selector {
        self.assert_circuit(bit.value.circuit);
        Selector { bit }
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

    /// Build one named component and record the ranges it emitted. The closure
    /// runs now; only the resulting ordinary IR and numeric ranges survive.
    pub fn component<T>(
        &mut self,
        name: impl Into<String>,
        build: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let name = name.into();
        assert!(!name.is_empty(), "component name must not be empty");
        assert!(
            self.components
                .iter()
                .all(|component| component.name != name),
            "duplicate component name `{name}`"
        );

        let index = self.components.len();
        let expression_start = self.expressions.len();
        let value_start = self.value_count;
        let row_start = self.rows.len();
        self.components.push(Component {
            name,
            parent: self.active_components.last().copied(),
            expressions: expression_start..expression_start,
            values: value_start..value_start,
            rows: row_start..row_start,
        });
        self.active_components.push(index);
        let result = build(self);
        assert_eq!(self.active_components.pop(), Some(index));

        let component = &mut self.components[index];
        component.expressions.end = self.expressions.len();
        component.values.end = self.value_count;
        component.rows.end = self.rows.len();
        result
    }

    /// Record a deferred global interaction. Its effective multiplicity is
    /// `selector * decode_le(multiplicity)`. This references existing bits and
    /// allocates no values, expressions, or constraint rows.
    pub fn interaction(
        &mut self,
        channel: impl Into<String>,
        kind: impl Into<String>,
        direction: InteractionDirection,
        message: impl IntoIterator<Item = InteractionField>,
        multiplicity: impl IntoIterator<Item = Bit>,
        selector: Selector,
        scope: InteractionScope,
    ) {
        self.assert_circuit(selector.bit.value.circuit);
        let message: Vec<InteractionField> = message.into_iter().collect();
        for field in &message {
            for value in &field.values {
                self.assert_circuit(value.circuit);
            }
        }
        let multiplicity: Vec<ValueId> = multiplicity
            .into_iter()
            .map(|bit| {
                self.assert_circuit(bit.value.circuit);
                bit.value
            })
            .collect();
        self.interactions.push(Interaction {
            channel: channel.into(),
            kind: kind.into(),
            direction,
            message,
            multiplicity,
            selector: selector.bit.value,
            scope,
            component: self.active_components.last().copied(),
        });
    }

    /// Assert `selector * expression = 0`.
    pub fn assert_when_zero(
        &mut self,
        selector: Selector,
        expression: impl Into<LinearExpr>,
    ) -> RowId {
        self.assert_zero_product(selector, expression)
    }

    /// Assert equality only when `selector` is one.
    pub fn assert_when_eq<L: Into<LinearExpr>, R: Into<LinearExpr>>(
        &mut self,
        selector: Selector,
        lhs: L,
        rhs: R,
    ) -> RowId {
        let difference = self.xor2(lhs, rhs);
        self.assert_when_zero(selector, difference)
    }

    pub(super) fn validate_interface(&self) {
        self.validate_ports();
        self.validate_components();
        self.validate_interactions();
    }

    fn validate_ports(&self) {
        for (index, port) in self.ports.iter().enumerate() {
            assert!(!port.name.is_empty(), "port name must not be empty");
            if let PortEncoding::LittleEndianWord { alignment_bits } = port.encoding {
                assert!(alignment_bits > 0, "word alignment must be nonzero");
            }
            assert!(
                self.ports[..index]
                    .iter()
                    .all(|other| other.name != port.name),
                "duplicate port name `{}`",
                port.name
            );
            assert!(!port.values.is_empty(), "port `{}` is empty", port.name);
            let mut values = port.values.clone();
            values.sort_unstable();
            assert!(
                values.windows(2).all(|pair| pair[0] != pair[1]),
                "port `{}` repeats a value",
                port.name
            );
            assert!(
                values
                    .iter()
                    .all(|value| value.circuit == self.id && value.index < self.value_count),
                "port `{}` contains an invalid value",
                port.name
            );
            if port.direction == PortDirection::Input {
                assert!(
                    port.values
                        .iter()
                        .all(|value| self.input_values.contains(value)),
                    "input port `{}` contains a computed value",
                    port.name
                );
            }
            match (&port.direction, &port.origin) {
                (PortDirection::Input, PortOrigin::Witness)
                | (PortDirection::Fixed, PortOrigin::Fixed)
                | (PortDirection::Output, PortOrigin::Derived) => {}
                (PortDirection::Input, PortOrigin::Advice { advice_type }) => {
                    assert!(!advice_type.is_empty(), "advice type must not be empty");
                }
                _ => panic!("port `{}` has an invalid direction/origin pair", port.name),
            }
        }
    }

    fn validate_components(&self) {
        assert!(
            self.active_components.is_empty(),
            "component scope left open"
        );
        for (index, component) in self.components.iter().enumerate() {
            assert!(
                !component.name.is_empty(),
                "component name must not be empty"
            );
            assert!(
                self.components[..index]
                    .iter()
                    .all(|other| other.name != component.name),
                "duplicate component name `{}`",
                component.name
            );
            assert_range(&component.expressions, self.expressions.len(), "expression");
            assert_range(&component.values, self.value_count, "value");
            assert_range(&component.rows, self.rows.len(), "row");
            if let Some(parent) = component.parent {
                assert!(parent < index, "component parent must precede its child");
                let parent = &self.components[parent];
                assert!(
                    contains(&parent.expressions, &component.expressions)
                        && contains(&parent.values, &component.values)
                        && contains(&parent.rows, &component.rows),
                    "component lies outside its parent"
                );
            }
        }
    }

    fn validate_interactions(&self) {
        for interaction in &self.interactions {
            assert!(
                !interaction.channel.is_empty(),
                "interaction channel must not be empty"
            );
            assert!(
                !interaction.kind.is_empty(),
                "interaction kind must not be empty"
            );
            assert!(
                !interaction.message.is_empty(),
                "interaction message must not be empty"
            );
            assert!(
                !interaction.scope.chip().is_empty(),
                "interaction chip scope must not be empty"
            );
            assert!(
                interaction
                    .component
                    .is_none_or(|index| index < self.components.len()),
                "interaction has an invalid component scope"
            );
            assert!(
                interaction.selector.circuit == self.id
                    && interaction.selector.index < self.value_count,
                "interaction selector is invalid"
            );
            assert!(
                !interaction.multiplicity.is_empty(),
                "interaction multiplicity must not be empty"
            );
            assert!(
                interaction
                    .multiplicity
                    .iter()
                    .all(|value| { value.circuit == self.id && value.index < self.value_count }),
                "interaction multiplicity contains an invalid value"
            );
            for (field_index, field) in interaction.message.iter().enumerate() {
                assert!(
                    !field.name.is_empty(),
                    "interaction field name must not be empty"
                );
                assert!(
                    interaction.message[..field_index]
                        .iter()
                        .all(|other| other.name != field.name),
                    "duplicate interaction field `{}`",
                    field.name
                );
                assert!(!field.values.is_empty(), "interaction field is empty");
                assert!(
                    field.values.iter().all(|value| {
                        value.circuit == self.id && value.index < self.value_count
                    }),
                    "interaction field contains an invalid value"
                );
                if let InteractionEncoding::LittleEndian { element_bits } = field.encoding {
                    assert!(
                        element_bits > 0,
                        "interaction element width must be nonzero"
                    );
                    assert_eq!(
                        field.values.len() % element_bits,
                        0,
                        "interaction field width is not a multiple of its element width"
                    );
                }
            }
        }
    }
}

fn assert_range(range: &std::ops::Range<usize>, bound: usize, what: &str) {
    assert!(
        range.start <= range.end && range.end <= bound,
        "component {what} range is invalid"
    );
}

fn contains(outer: &std::ops::Range<usize>, inner: &std::ops::Range<usize>) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}
