//! Logical-to-physical R1CS placement.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use super::{BooleanCircuit, CircuitId, PortEncoding, RowId, ValueId};

/// Deterministic mapping from logical circuit identifiers to physical R1CS
/// columns and rows.
///
/// Circuit authors normally never inspect this type. Compatibility lowering
/// uses it to reproduce an existing SHA/BLAKE layout without exposing raw
/// indices in the arithmetic definition.
#[derive(Clone, Debug)]
pub struct PhysicalLayout {
    pub(super) circuit: CircuitId,
    pub(super) value_positions: Vec<usize>,
    pub(super) row_positions: Vec<usize>,
    pub(super) useful_bits: usize,
}

impl PartialEq for PhysicalLayout {
    fn eq(&self, other: &Self) -> bool {
        // Circuit IDs are runtime guards, not layout data.
        self.value_positions == other.value_positions
            && self.row_positions == other.row_positions
            && self.useful_bits == other.useful_bits
    }
}

impl Eq for PhysicalLayout {}

impl PhysicalLayout {
    /// Logical source-order placement. This is a deterministic candidate: an
    /// aligned word port may make it invalid, which semantic consumers report
    /// as a [`LayoutError`]. When valid, it preserves the pre-layout DSL
    /// behavior and gives identity C when every row defines its same-index
    /// value.
    pub fn source_order(circuit: &BooleanCircuit) -> Self {
        let useful_bits = circuit.value_count().max(circuit.row_count());
        Self {
            circuit: circuit.id,
            value_positions: (0..circuit.value_count()).collect(),
            row_positions: (0..circuit.row_count()).collect(),
            useful_bits,
        }
    }

    pub fn value_position(&self, value: ValueId) -> Option<usize> {
        if value.circuit != self.circuit {
            return None;
        }
        self.value_positions.get(value.index).copied()
    }

    pub fn row_position(&self, row: RowId) -> Option<usize> {
        if row.circuit != self.circuit {
            return None;
        }
        self.row_positions.get(row.index).copied()
    }

    /// Physical prefix containing all placed values and rows. Positions at or
    /// above this boundary are canonical zero padding.
    pub const fn useful_bits(&self) -> usize {
        self.useful_bits
    }

    pub(super) fn validate_for(&self, circuit: &BooleanCircuit) -> Result<(), LayoutError> {
        if self.circuit != circuit.id
            || self.value_positions.len() != circuit.value_count()
            || self.row_positions.len() != circuit.row_count()
        {
            return Err(LayoutError::WrongCircuit);
        }
        validate_unique_positions(&self.value_positions, PositionKind::Column)?;
        validate_unique_positions(&self.row_positions, PositionKind::Row)?;
        for port in &circuit.ports {
            if let PortEncoding::LittleEndianWord { alignment_bits } = port.encoding {
                let start = self.value_positions[port.values[0].index];
                if !start.is_multiple_of(alignment_bits) {
                    return Err(LayoutError::MisalignedPort {
                        name: port.name.clone(),
                        start,
                        alignment_bits,
                    });
                }
                for (offset, value) in port.values.iter().enumerate() {
                    let expected = start
                        .checked_add(offset)
                        .ok_or(LayoutError::PositionOverflow)?;
                    let actual = self.value_positions[value.index];
                    if actual != expected {
                        return Err(LayoutError::NonContiguousPort {
                            name: port.name.clone(),
                            offset,
                            expected,
                            actual,
                        });
                    }
                }
            }
        }
        let required = self
            .value_positions
            .iter()
            .chain(&self.row_positions)
            .copied()
            .max()
            .map_or(Ok(1), |position| {
                position.checked_add(1).ok_or(LayoutError::PositionOverflow)
            })?;
        if self.useful_bits < required {
            return Err(LayoutError::UsefulBitsTooSmall {
                required,
                actual: self.useful_bits,
            });
        }
        Ok(())
    }
}

/// Builder for explicit compatibility layouts.
pub struct LayoutBuilder<'a> {
    circuit: &'a BooleanCircuit,
    value_positions: Vec<Option<usize>>,
    row_positions: Vec<Option<usize>>,
    occupied_columns: BTreeMap<usize, usize>,
    occupied_rows: BTreeMap<usize, usize>,
    reserved: Vec<Range<usize>>,
}

impl<'a> LayoutBuilder<'a> {
    pub fn new(circuit: &'a BooleanCircuit) -> Self {
        Self {
            circuit,
            value_positions: vec![None; circuit.value_count()],
            row_positions: vec![None; circuit.row_count()],
            occupied_columns: BTreeMap::new(),
            occupied_rows: BTreeMap::new(),
            reserved: Vec::new(),
        }
    }

    /// Keep a physical range out of deterministic auto-placement. Explicit
    /// placement may still use it later.
    pub fn reserve(&mut self, range: Range<usize>) -> Result<&mut Self, LayoutError> {
        if range.start > range.end {
            return Err(LayoutError::InvalidRange);
        }
        if !range.is_empty() {
            self.reserved.push(range);
        }
        Ok(self)
    }

    pub fn place_value(
        &mut self,
        value: ValueId,
        position: usize,
    ) -> Result<&mut Self, LayoutError> {
        self.check_circuit(value.circuit)?;
        place_position(
            &mut self.value_positions,
            &mut self.occupied_columns,
            value.index,
            position,
            PositionKind::Column,
        )?;
        Ok(self)
    }

    pub fn place_row(&mut self, row: RowId, position: usize) -> Result<&mut Self, LayoutError> {
        self.check_circuit(row.circuit)?;
        place_position(
            &mut self.row_positions,
            &mut self.occupied_rows,
            row.index,
            position,
            PositionKind::Row,
        )?;
        Ok(self)
    }

    /// Place a value and its defining row at the same physical index, retaining
    /// the identity-C shape for that definition.
    pub fn place_definition(
        &mut self,
        value: ValueId,
        position: usize,
    ) -> Result<&mut Self, LayoutError> {
        self.check_circuit(value.circuit)?;
        let row = self.definition(value)?;
        check_position(
            &self.value_positions,
            &self.occupied_columns,
            value.index,
            position,
            PositionKind::Column,
        )?;
        check_position(
            &self.row_positions,
            &self.occupied_rows,
            row.index,
            position,
            PositionKind::Row,
        )?;
        self.value_positions[value.index] = Some(position);
        self.row_positions[row.index] = Some(position);
        self.occupied_columns.insert(position, value.index);
        self.occupied_rows.insert(position, row.index);
        Ok(self)
    }

    /// Place a named port contiguously and co-locate each bit's defining row.
    pub fn place_port(&mut self, name: &str, start: usize) -> Result<&mut Self, LayoutError> {
        let port = self
            .circuit
            .port(name)
            .ok_or_else(|| LayoutError::UnknownPort(name.to_owned()))?;
        if let PortEncoding::LittleEndianWord { alignment_bits } = port.encoding
            && !start.is_multiple_of(alignment_bits)
        {
            return Err(LayoutError::MisalignedPort {
                name: name.to_owned(),
                start,
                alignment_bits,
            });
        }
        let values = port.values.clone();
        let placements: Vec<(ValueId, usize)> = values
            .into_iter()
            .enumerate()
            .map(|(offset, value)| {
                start
                    .checked_add(offset)
                    .map(|position| (value, position))
                    .ok_or(LayoutError::PositionOverflow)
            })
            .collect::<Result<_, _>>()?;
        for &(value, position) in &placements {
            let row = self.definition(value)?;
            check_position(
                &self.value_positions,
                &self.occupied_columns,
                value.index,
                position,
                PositionKind::Column,
            )?;
            check_position(
                &self.row_positions,
                &self.occupied_rows,
                row.index,
                position,
                PositionKind::Row,
            )?;
        }
        for (value, position) in placements {
            self.place_definition(value, position)?;
        }
        Ok(self)
    }

    /// Deterministically place everything not explicitly positioned. Whenever
    /// possible, a value and its defining row share a position so compatibility
    /// customization does not accidentally lose identity C.
    pub fn finish(mut self) -> Result<PhysicalLayout, LayoutError> {
        let reserved = normalize_ranges(std::mem::take(&mut self.reserved));
        let mut common_cursor = 0;
        for value_index in 0..self.circuit.value_count() {
            let value = ValueId {
                circuit: self.circuit.id,
                index: value_index,
            };
            let row = self.definition(value)?;
            match (
                self.value_positions[value_index],
                self.row_positions[row.index],
            ) {
                (Some(position), None) if !self.occupied_rows.contains_key(&position) => {
                    self.row_positions[row.index] = Some(position);
                    self.occupied_rows.insert(position, row.index);
                }
                (None, Some(position)) if !self.occupied_columns.contains_key(&position) => {
                    self.value_positions[value_index] = Some(position);
                    self.occupied_columns.insert(position, value_index);
                }
                (None, None) => {
                    let position = next_free(&mut common_cursor, &reserved, |position| {
                        self.occupied_columns.contains_key(&position)
                            || self.occupied_rows.contains_key(&position)
                    })?;
                    self.value_positions[value_index] = Some(position);
                    self.row_positions[row.index] = Some(position);
                    self.occupied_columns.insert(position, value_index);
                    self.occupied_rows.insert(position, row.index);
                }
                _ => {}
            }
        }

        fill_unplaced(
            &mut self.value_positions,
            &mut self.occupied_columns,
            &reserved,
        )?;
        fill_unplaced(&mut self.row_positions, &mut self.occupied_rows, &reserved)?;
        let value_positions: Vec<usize> = self
            .value_positions
            .into_iter()
            .map(Option::unwrap)
            .collect();
        let row_positions: Vec<usize> =
            self.row_positions.into_iter().map(Option::unwrap).collect();
        let useful_bits = value_positions
            .iter()
            .chain(&row_positions)
            .copied()
            .max()
            .map_or(Ok(1), |position| {
                position.checked_add(1).ok_or(LayoutError::PositionOverflow)
            })?;
        let layout = PhysicalLayout {
            circuit: self.circuit.id,
            value_positions,
            row_positions,
            useful_bits,
        };
        layout.validate_for(self.circuit)?;
        Ok(layout)
    }

    fn check_circuit(&self, circuit: CircuitId) -> Result<(), LayoutError> {
        if circuit == self.circuit.id {
            Ok(())
        } else {
            Err(LayoutError::WrongCircuit)
        }
    }

    fn definition(&self, value: ValueId) -> Result<RowId, LayoutError> {
        self.circuit
            .definition_row(value)
            .ok_or(LayoutError::InvalidIndex {
                kind: PositionKind::Column,
                index: value.index,
            })
    }
}

/// Whether a physical position is a witness column or a constraint row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionKind {
    Column,
    Row,
}

impl fmt::Display for PositionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column => f.write_str("column"),
            Self::Row => f.write_str("row"),
        }
    }
}

/// A malformed or conflicting physical layout request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    WrongCircuit,
    UnknownPort(String),
    MisalignedPort {
        name: String,
        start: usize,
        alignment_bits: usize,
    },
    NonContiguousPort {
        name: String,
        offset: usize,
        expected: usize,
        actual: usize,
    },
    InvalidIndex {
        kind: PositionKind,
        index: usize,
    },
    InvalidRange,
    PositionOverflow,
    AlreadyPlaced {
        kind: PositionKind,
        index: usize,
        existing: usize,
        requested: usize,
    },
    PositionOccupied {
        kind: PositionKind,
        position: usize,
        occupant: usize,
    },
    UsefulBitsTooSmall {
        required: usize,
        actual: usize,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongCircuit => f.write_str("layout belongs to another circuit shape"),
            Self::UnknownPort(name) => write!(f, "unknown port `{name}`"),
            Self::MisalignedPort {
                name,
                start,
                alignment_bits,
            } => write!(
                f,
                "port `{name}` starts at {start}, which is not aligned to {alignment_bits} bits"
            ),
            Self::NonContiguousPort {
                name,
                offset,
                expected,
                actual,
            } => write!(
                f,
                "port `{name}` bit {offset} is at {actual}, not contiguous position {expected}"
            ),
            Self::InvalidIndex { kind, index } => write!(f, "invalid {kind} id {index}"),
            Self::InvalidRange => f.write_str("layout reservation has start greater than end"),
            Self::PositionOverflow => f.write_str("physical position overflows usize"),
            Self::AlreadyPlaced {
                kind,
                index,
                existing,
                requested,
            } => write!(
                f,
                "{kind} {index} is already at {existing}, not requested position {requested}"
            ),
            Self::PositionOccupied {
                kind,
                position,
                occupant,
            } => write!(
                f,
                "physical {kind} position {position} is already occupied by logical {kind} {occupant}"
            ),
            Self::UsefulBitsTooSmall { required, actual } => write!(
                f,
                "useful-bit prefix {actual} does not cover required position count {required}"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

fn validate_unique_positions(positions: &[usize], kind: PositionKind) -> Result<(), LayoutError> {
    let mut ordered: Vec<(usize, usize)> = positions.iter().copied().enumerate().collect();
    ordered.sort_unstable_by_key(|&(_, position)| position);
    for pair in ordered.windows(2) {
        if pair[0].1 == pair[1].1 {
            return Err(LayoutError::PositionOccupied {
                kind,
                position: pair[0].1,
                occupant: pair[0].0,
            });
        }
    }
    Ok(())
}

fn check_position(
    positions: &[Option<usize>],
    occupied: &BTreeMap<usize, usize>,
    index: usize,
    position: usize,
    kind: PositionKind,
) -> Result<(), LayoutError> {
    let current = positions
        .get(index)
        .ok_or(LayoutError::InvalidIndex { kind, index })?;
    if let Some(existing) = current {
        if *existing == position {
            return Ok(());
        }
        return Err(LayoutError::AlreadyPlaced {
            kind,
            index,
            existing: *existing,
            requested: position,
        });
    }
    if let Some(&occupant) = occupied.get(&position) {
        return Err(LayoutError::PositionOccupied {
            kind,
            position,
            occupant,
        });
    }
    Ok(())
}

fn place_position(
    positions: &mut [Option<usize>],
    occupied: &mut BTreeMap<usize, usize>,
    index: usize,
    position: usize,
    kind: PositionKind,
) -> Result<(), LayoutError> {
    check_position(positions, occupied, index, position, kind)?;
    positions[index] = Some(position);
    occupied.insert(position, index);
    Ok(())
}

fn normalize_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(previous) if range.start <= previous.end => {
                previous.end = previous.end.max(range.end);
            }
            _ => merged.push(range),
        }
    }
    merged
}

fn reserved_end(position: usize, reservations: &[Range<usize>]) -> Option<usize> {
    let index = reservations.partition_point(|range| range.start <= position);
    index
        .checked_sub(1)
        .map(|index| &reservations[index])
        .filter(|range| position < range.end)
        .map(|range| range.end)
}

fn next_free(
    cursor: &mut usize,
    reservations: &[Range<usize>],
    is_occupied: impl Fn(usize) -> bool,
) -> Result<usize, LayoutError> {
    loop {
        if let Some(end) = reserved_end(*cursor, reservations) {
            *cursor = end;
            continue;
        }
        if !is_occupied(*cursor) {
            let position = *cursor;
            *cursor = cursor.checked_add(1).ok_or(LayoutError::PositionOverflow)?;
            return Ok(position);
        }
        *cursor = cursor.checked_add(1).ok_or(LayoutError::PositionOverflow)?;
    }
}

fn fill_unplaced(
    positions: &mut [Option<usize>],
    occupied: &mut BTreeMap<usize, usize>,
    reservations: &[Range<usize>],
) -> Result<(), LayoutError> {
    let mut cursor = 0;
    for index in 0..positions.len() {
        if positions[index].is_none() {
            let position = next_free(&mut cursor, reservations, |position| {
                occupied.contains_key(&position)
            })?;
            positions[index] = Some(position);
            occupied.insert(position, index);
        }
    }
    Ok(())
}
