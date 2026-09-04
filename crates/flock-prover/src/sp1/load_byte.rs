//! Boolean semantic port of one SP1 load-byte row.

use flock_core::circuit::boolean::{
    Bit, BooleanCircuit, CircuitBuilder, InteractionDirection, InteractionField, InteractionScope,
    LinearExpr, Selector, ValueId,
};

const ADVICE_TYPE: &str = "sp1.load-byte.selected-byte/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssertionMode {
    /// Guarded general-C rows reject violations locally.
    Direct,
    /// Definitional rows produce a terminal bit that must be bound to ONE.
    IdentityC,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadByteOpcode {
    Lb,
    Lbu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadByteEvent {
    pub opcode: LoadByteOpcode,
    pub b: u64,
    pub c: u64,
    pub memory_value: u64,
}

impl LoadByteEvent {
    pub const fn result(self) -> u64 {
        let address = self.b.wrapping_add(self.c);
        let byte = (self.memory_value >> (8 * (address & 7))) as u8;
        match self.opcode {
            LoadByteOpcode::Lb => (byte as i8 as i64) as u64,
            LoadByteOpcode::Lbu => byte as u64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoadByteCircuit {
    circuit: BooleanCircuit,
    capacity: usize,
    mode: AssertionMode,
    accept: Option<ValueId>,
}

impl LoadByteCircuit {
    pub fn build(capacity: usize, mode: AssertionMode) -> Self {
        assert!(capacity > 0, "load-byte capacity must be nonzero");
        let mut builder = CircuitBuilder::new();
        let mut checks = Checks::new(mode);
        let mut previous_real = None;

        for row in 0..capacity {
            let real = builder.component(format!("load-byte.row-{row}"), |builder| {
                build_row(builder, &mut checks, previous_real, row)
            });
            previous_real = Some(real);
        }

        let accept = checks.finish(&mut builder);
        Self {
            circuit: builder.finish(),
            capacity,
            mode,
            accept,
        }
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn mode(&self) -> AssertionMode {
        self.mode
    }

    pub fn circuit(&self) -> &BooleanCircuit {
        &self.circuit
    }

    /// The identity-C experiment's terminal bit. It must be checked as ONE by
    /// an enclosing statement; its output name alone has no force.
    pub const fn accept(&self) -> Option<ValueId> {
        self.accept
    }

    /// Encode an explicit capacity-sized row list. `None` is a padding row;
    /// malformed non-prefix lists are useful negative test inputs.
    pub fn encode_rows(&self, rows: &[Option<LoadByteEvent>]) -> Vec<bool> {
        assert_eq!(rows.len(), self.capacity, "wrong load-byte row count");
        let mut inputs = Vec::with_capacity(self.circuit.inputs().len());
        for row in rows {
            encode_row(&mut inputs, *row);
        }
        debug_assert_eq!(inputs.len(), self.circuit.inputs().len());
        inputs
    }

    /// Encode real events followed by canonical zero padding.
    pub fn honest_inputs(&self, events: &[LoadByteEvent]) -> Vec<bool> {
        assert!(events.len() <= self.capacity, "too many load-byte events");
        let rows: Vec<Option<LoadByteEvent>> = events
            .iter()
            .copied()
            .map(Some)
            .chain(std::iter::repeat_n(None, self.capacity - events.len()))
            .collect();
        self.encode_rows(&rows)
    }

    pub fn output(&self, values: &[bool], row: usize) -> u64 {
        assert!(row < self.capacity, "load-byte row is out of range");
        let port = self
            .circuit
            .port(&format!("load-byte.row-{row}.result"))
            .expect("result port exists");
        port.values()
            .iter()
            .enumerate()
            .fold(0u64, |word, (bit, value)| {
                word | (u64::from(values[value.index()]) << bit)
            })
    }
}

fn build_row(
    builder: &mut CircuitBuilder,
    checks: &mut Checks,
    previous_real: Option<Selector>,
    row: usize,
) -> Selector {
    let prefix = format!("load-byte.row-{row}");
    let is_lb = builder.input_selector(format!("{prefix}.is-lb"));
    let is_lbu = builder.input_selector(format!("{prefix}.is-lbu"));
    let b = builder.input_word::<64>(format!("{prefix}.b"));
    let c = builder.input_word::<64>(format!("{prefix}.c"));
    let memory = builder.input_word::<64>(format!("{prefix}.memory-value"));
    let selected = builder.advice_word::<8>(format!("{prefix}.selected-byte"), ADVICE_TYPE);

    checks.zero_when(builder, is_lb, is_lbu.expr());
    let real_expr = builder.xor2(is_lb, is_lbu);
    let real_bit = builder.materialize(real_expr);
    let real = builder.selector(real_bit);
    if let Some(previous) = previous_real {
        let previous_is_zero = builder.xor2(builder.one(), previous);
        checks.zero_when(builder, real, previous_is_zero);
    }

    let address = wrapping_add(builder, b, c);
    for bit in &address[48..] {
        checks.zero_when(builder, real, *bit);
    }
    let address_is_above_guard = or_all(builder, address[16..48].iter().copied());
    checks.equal_when(builder, real, address_is_above_guard, builder.one());

    let expected = select_byte(builder, memory, [address[0], address[1], address[2]]);
    for bit in 0..8 {
        checks.equal_when(builder, real, selected[bit], expected[bit]);
    }

    let sign = builder.and(is_lb, selected[7]);
    let result: [Bit; 64] = std::array::from_fn(|bit| {
        let value = if bit < 8 {
            selected[bit].expr()
        } else {
            sign.expr()
        };
        builder.materialize(value)
    });
    builder.output_word(format!("{prefix}.result"), result);

    let aligned_address: [Bit; 64] = std::array::from_fn(|bit| {
        if bit < 3 {
            let zero = builder.zero();
            builder.materialize(zero)
        } else {
            address[bit]
        }
    });
    // Pilot projections: a complete machine event will add routing and time
    // fields, but it will reference these same computed address/result bits.
    let unit_multiplicity = [builder.one()];
    builder.interaction(
        "memory",
        "read",
        InteractionDirection::Send,
        [
            InteractionField::little_endian("address", 64, aligned_address),
            InteractionField::little_endian("value", 64, memory),
        ],
        unit_multiplicity,
        real,
        InteractionScope::new("load-byte", row),
    );
    builder.interaction(
        "register",
        "write",
        InteractionDirection::Send,
        [
            InteractionField::bits("opcode", [is_lb.bit(), is_lbu.bit()]),
            InteractionField::little_endian("value", 64, result),
        ],
        unit_multiplicity,
        real,
        InteractionScope::new("load-byte", row),
    );
    real
}

fn wrapping_add(builder: &mut CircuitBuilder, lhs: [Bit; 64], rhs: [Bit; 64]) -> [Bit; 64] {
    let mut carry = builder.zero();
    std::array::from_fn(|bit| {
        let propagate = builder.xor2(lhs[bit], rhs[bit]);
        let sum_expr = builder.xor2(propagate, carry);
        let sum = builder.materialize(sum_expr);
        let generated = builder.and(lhs[bit], rhs[bit]);
        let propagated = builder.and(propagate, carry);
        carry = builder.xor2(generated, propagated);
        sum
    })
}

fn select_byte(
    builder: &mut CircuitBuilder,
    memory: [Bit; 64],
    offset: [Bit; 3],
) -> [LinearExpr; 8] {
    let mut candidates: Vec<[LinearExpr; 8]> = (0..8)
        .map(|byte| std::array::from_fn(|bit| memory[8 * byte + bit].expr()))
        .collect();
    for selector in offset {
        candidates = candidates
            .chunks_exact(2)
            .map(|pair| {
                std::array::from_fn(|bit| mux(builder, selector, pair[0][bit], pair[1][bit]))
            })
            .collect();
    }
    candidates[0]
}

fn mux(
    builder: &mut CircuitBuilder,
    selector: Bit,
    when_zero: LinearExpr,
    when_one: LinearExpr,
) -> LinearExpr {
    let difference = builder.xor2(when_zero, when_one);
    let selected_difference = builder.and(selector, difference);
    builder.xor2(when_zero, selected_difference)
}

fn or_all(builder: &mut CircuitBuilder, bits: impl IntoIterator<Item = Bit>) -> LinearExpr {
    let mut bits = bits.into_iter();
    let mut any = bits.next().map_or(builder.zero(), Bit::expr);
    for bit in bits {
        let overlap = builder.and(any, bit);
        any = builder.xor3(any, bit, overlap);
    }
    any
}

fn encode_row(inputs: &mut Vec<bool>, row: Option<LoadByteEvent>) {
    let (is_lb, is_lbu, b, c, memory, selected) = match row {
        Some(event) => {
            let address = event.b.wrapping_add(event.c);
            let selected = (event.memory_value >> (8 * (address & 7))) as u8;
            (
                event.opcode == LoadByteOpcode::Lb,
                event.opcode == LoadByteOpcode::Lbu,
                event.b,
                event.c,
                event.memory_value,
                selected,
            )
        }
        None => (false, false, 0, 0, 0, 0),
    };
    inputs.extend([is_lb, is_lbu]);
    extend_word(inputs, b, 64);
    extend_word(inputs, c, 64);
    extend_word(inputs, memory, 64);
    extend_word(inputs, selected as u64, 8);
}

fn extend_word(bits: &mut Vec<bool>, value: u64, width: usize) {
    bits.extend((0..width).map(|bit| value >> bit & 1 == 1));
}

struct Checks {
    mode: AssertionMode,
    violations: Vec<Bit>,
}

impl Checks {
    fn new(mode: AssertionMode) -> Self {
        Self {
            mode,
            violations: Vec::new(),
        }
    }

    fn zero_when(
        &mut self,
        builder: &mut CircuitBuilder,
        selector: Selector,
        violation: impl Into<LinearExpr>,
    ) {
        match self.mode {
            AssertionMode::Direct => {
                builder.assert_when_zero(selector, violation);
            }
            AssertionMode::IdentityC => {
                self.violations.push(builder.and(selector, violation));
            }
        }
    }

    fn equal_when(
        &mut self,
        builder: &mut CircuitBuilder,
        selector: Selector,
        lhs: impl Into<LinearExpr>,
        rhs: impl Into<LinearExpr>,
    ) {
        let difference = builder.xor2(lhs, rhs);
        self.zero_when(builder, selector, difference);
    }

    fn finish(self, builder: &mut CircuitBuilder) -> Option<ValueId> {
        if self.mode == AssertionMode::Direct {
            return None;
        }

        let mut violations = self.violations.into_iter();
        let mut any = violations.next().map_or(builder.zero(), Bit::expr);
        for violation in violations {
            let overlap = builder.and(any, violation);
            any = builder.xor3(any, violation, overlap);
        }
        let accept_expr = builder.xor2(builder.one(), any);
        let accept = builder.materialize(accept_expr);
        builder.output("load-byte.accept", [accept]);
        Some(accept.value_id())
    }
}

#[cfg(test)]
mod tests;
