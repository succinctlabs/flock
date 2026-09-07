use crate::genus95_curve_code::{
    messages::{BaseMessage, ExtendedMessage, ProductMessage},
    tables::{TABLES, Tables},
};

/// Compute the 222-bit product-code message for the product of two 64-bit base
/// messages.
#[inline(always)]
pub fn product_code_message(left: BaseMessage, right: BaseMessage) -> ProductMessage {
    let tables = &*TABLES;
    let l = extend_base_message_with_tables(tables, left);
    let r = extend_base_message_with_tables(tables, right);

    // order0 is the identity-extended base message — the raw bits themselves.
    let l0 = left.0;
    let r0 = right.0;

    let order0 = l0 & r0;
    let order1 = (l0 & r.order1) ^ (l.order1 & r0);
    let order2 = (l0 & r.order2) ^ (l.order1 & r.order1) ^ (l.order2 & r0);

    // The order-3 section's points are coordinates 0..29, so "order_k at the
    // points" is just the low 30 bits of order_k — taken straight from the full
    // order1/order2/message. `from_sections` masks limbs[3] to 30 bits, so the
    // high bits of the middle terms fall away for free.
    let order3 = (l0 & r.order3 as u64)
        ^ (l.order1 & r.order2)
        ^ (l.order2 & r.order1)
        ^ (l.order3 as u64 & r0);

    ProductMessage::from_sections(order0, order1, order2, order3)
}

#[inline(always)]
fn extend_base_message_with_tables(tables: &Tables, message: BaseMessage) -> ExtendedMessage {
    apply_extended_byte_table(&tables.extended_byte_table, message.0)
}

// On non-aarch64 the only non-test caller chain (`round1::derived_m` → the
// NEON round-1 kernels) is compiled out, so the lib build sees this as dead.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
pub(crate) fn extended_base_product_message(message: BaseMessage) -> ProductMessage {
    let extended = extend_base_message_with_tables(&TABLES, message);
    ProductMessage::from_sections(
        message.0,
        extended.order1,
        extended.order2,
        extended.order3 as u64,
    )
}

#[inline(always)]
fn apply_extended_byte_table(table: &[[ExtendedMessage; 256]; 8], message: u64) -> ExtendedMessage {
    let mut out = table[0][(message & 0xff) as usize];
    out ^= table[1][((message >> 8) & 0xff) as usize];
    out ^= table[2][((message >> 16) & 0xff) as usize];
    out ^= table[3][((message >> 24) & 0xff) as usize];
    out ^= table[4][((message >> 32) & 0xff) as usize];
    out ^= table[5][((message >> 40) & 0xff) as usize];
    out ^= table[6][((message >> 48) & 0xff) as usize];
    out ^= table[7][((message >> 56) & 0xff) as usize];
    out
}
