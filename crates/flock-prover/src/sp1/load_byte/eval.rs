use flock_core::circuit::boolean::{
    CircuitBuilder, InteractionDirection, InteractionField, InteractionScope, Var,
};

use super::LoadByteCols;

/// All local constraints, including the real-row prefix across invocations.
pub(super) fn eval(b: &mut CircuitBuilder, rows: &[LoadByteCols<Var>]) {
    let mut previous_real = None;
    for (row, cols) in rows.iter().enumerate() {
        let is_lb = b.column_selector(cols.is_lb);
        b.assert_when_zero(is_lb, cols.is_lbu);
        let real_expr = b.xor2(cols.is_lb, cols.is_lbu);
        b.define_linear(cols.real, real_expr);
        let real = b.column_selector(cols.real);
        if let Some(previous) = previous_real {
            let previous_is_zero = b.xor2(b.one(), previous);
            b.assert_when_zero(real, previous_is_zero);
        }
        previous_real = Some(cols.real);

        let address = cols
            .address
            .eval(b, cols.b.map(Into::into), cols.c.map(Into::into));
        for bit in &address[48..] {
            b.assert_when_zero(real, *bit);
        }
        let above_guard = cols
            .address_guard
            .eval(b, address[16..48].try_into().unwrap());
        b.assert_when_eq(real, above_guard, b.one());

        let expected = cols.select.eval(
            b,
            cols.memory_value.map(Into::into),
            [address[0], address[1], address[2]],
        );
        for (selected, expected) in cols.selected_byte.into_iter().zip(expected) {
            b.assert_when_eq(real, selected, expected);
        }

        b.define_and(cols.sign, cols.is_lb, cols.selected_byte[7]);
        for (bit, output) in cols.result.into_iter().enumerate() {
            b.define_linear(
                output,
                if bit < 8 {
                    cols.selected_byte[bit]
                } else {
                    cols.sign
                },
            );
        }
        for low in cols.aligned_low {
            b.define_linear(low, b.zero());
        }
        let aligned_address = cols
            .aligned_low
            .into_iter()
            .chain(cols.address.value[3..].iter().copied());

        // Pilot projections; routing and time fields belong to later machine integration.
        b.interaction(
            "memory",
            "read",
            InteractionDirection::Send,
            [
                InteractionField::columns("address", 64, aligned_address),
                InteractionField::columns("value", 64, cols.memory_value),
            ],
            [b.one()],
            real,
            InteractionScope::new("load-byte", row),
        );
        b.interaction(
            "register",
            "write",
            InteractionDirection::Send,
            [
                InteractionField::column_bits("opcode", [cols.is_lb, cols.is_lbu]),
                InteractionField::columns("value", 64, cols.result),
            ],
            [b.one()],
            real,
            InteractionScope::new("load-byte", row),
        );
    }
}
