//! VM-facing interface metadata that does not alter local arithmetic.

use std::ops::Range;

use super::{Bit, LinearExpr, ValueId, Var};

/// Who supplies or derives the values in a named port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortOrigin {
    Witness,
    Advice { advice_type: String },
    Fixed,
    Derived,
}

/// A materialized Boolean column used to guard constraints or interactions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Selector {
    pub(super) bit: Bit,
}

impl Selector {
    pub const fn bit(self) -> Bit {
        self.bit
    }

    pub const fn expr(self) -> LinearExpr {
        self.bit.expr()
    }
}

impl From<Selector> for LinearExpr {
    fn from(selector: Selector) -> Self {
        selector.expr()
    }
}

/// One named build-time component instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub(super) name: String,
    pub(super) parent: Option<usize>,
    pub(super) expressions: Range<usize>,
    pub(super) values: Range<usize>,
    pub(super) rows: Range<usize>,
}

impl Component {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn parent(&self) -> Option<usize> {
        self.parent
    }

    pub fn expressions(&self) -> Range<usize> {
        self.expressions.clone()
    }

    pub fn values(&self) -> Range<usize> {
        self.values.clone()
    }

    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionDirection {
    /// This component contributes the tuple to the named channel.
    Send,
    /// This component consumes the tuple from the named channel.
    Receive,
}

/// Bit interpretation for one field in an interaction message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionEncoding {
    Bits,
    LittleEndian { element_bits: usize },
}

/// One element of an ordered interaction tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InteractionField {
    pub(super) name: String,
    pub(super) encoding: InteractionEncoding,
    pub(super) values: Vec<ValueId>,
}

impl InteractionField {
    /// An uninterpreted bit message over reserved schema columns.
    pub fn column_bits(name: impl Into<String>, columns: impl IntoIterator<Item = Var>) -> Self {
        Self::bits(name, columns.into_iter().map(|var| var.0))
    }

    /// A little-endian message over reserved schema columns.
    pub fn columns(
        name: impl Into<String>,
        element_bits: usize,
        columns: impl IntoIterator<Item = Var>,
    ) -> Self {
        Self::little_endian(name, element_bits, columns.into_iter().map(|var| var.0))
    }

    pub fn bits(name: impl Into<String>, bits: impl IntoIterator<Item = Bit>) -> Self {
        Self::new(name, InteractionEncoding::Bits, bits)
    }

    pub fn little_endian(
        name: impl Into<String>,
        element_bits: usize,
        bits: impl IntoIterator<Item = Bit>,
    ) -> Self {
        Self::new(
            name,
            InteractionEncoding::LittleEndian { element_bits },
            bits,
        )
    }

    fn new(
        name: impl Into<String>,
        encoding: InteractionEncoding,
        bits: impl IntoIterator<Item = Bit>,
    ) -> Self {
        Self {
            name: name.into(),
            encoding,
            values: bits.into_iter().map(|bit| bit.value).collect(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn encoding(&self) -> InteractionEncoding {
        self.encoding
    }

    pub fn values(&self) -> &[ValueId] {
        &self.values
    }
}

/// Logical location of one interaction in a repeated chip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InteractionScope {
    chip: String,
    row: usize,
}

impl InteractionScope {
    pub fn new(chip: impl Into<String>, row: usize) -> Self {
        Self {
            chip: chip.into(),
            row,
        }
    }

    pub fn chip(&self) -> &str {
        &self.chip
    }

    pub const fn row(&self) -> usize {
        self.row
    }
}

/// A deferred global interaction over values already present in the circuit.
///
/// If `s` is the selector and `m_i` are the little-endian multiplicity bits,
/// its effective multiplicity is `s * sum_i(2^i * m_i)`. The selector is the
/// authoritative activation gate; a zero selector contributes nothing even
/// when the stored multiplicity bits are nonzero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interaction {
    pub(super) channel: String,
    pub(super) kind: String,
    pub(super) direction: InteractionDirection,
    pub(super) message: Vec<InteractionField>,
    pub(super) multiplicity: Vec<ValueId>,
    pub(super) selector: ValueId,
    pub(super) scope: InteractionScope,
    pub(super) component: Option<usize>,
}

impl Interaction {
    pub fn channel(&self) -> &str {
        &self.channel
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub const fn direction(&self) -> InteractionDirection {
        self.direction
    }

    pub fn message(&self) -> &[InteractionField] {
        &self.message
    }

    /// Little-endian unsigned multiplicity bits.
    pub fn multiplicity(&self) -> &[ValueId] {
        &self.multiplicity
    }

    /// Evaluate the selector-gated multiplicity without imposing a host
    /// integer-width limit. The returned bits remain little-endian.
    pub fn effective_multiplicity_bits(&self, values: &[bool]) -> Vec<bool> {
        let selected = *values
            .get(self.selector.index())
            .expect("assignment is missing the interaction selector");
        self.multiplicity
            .iter()
            .map(|value| {
                selected
                    && *values
                        .get(value.index())
                        .expect("assignment is missing an interaction multiplicity bit")
            })
            .collect()
    }

    /// The activation gate in the effective-multiplicity equation.
    pub const fn selector(&self) -> ValueId {
        self.selector
    }

    pub fn scope(&self) -> &InteractionScope {
        &self.scope
    }

    pub const fn component(&self) -> Option<usize> {
        self.component
    }
}

#[cfg(test)]
mod tests;
