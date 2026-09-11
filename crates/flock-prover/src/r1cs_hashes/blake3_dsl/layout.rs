//! Legacy positions are separate from BLAKE3 arithmetic and column declaration.
use super::columns::FirstC;
use super::{Blake3Cols, blake3};
use flock_core::circuit::boolean::{BooleanCircuit, PhysicalLayout, ValueId};

pub(super) fn layout(circuit: &BooleanCircuit, cols: &Blake3Cols<ValueId>) -> PhysicalLayout {
    let mut layout = circuit.layout();
    let mut place = |values: &[ValueId], base: usize| {
        for (offset, &value) in values.iter().enumerate() {
            layout
                .place_definition(value, base + offset)
                .expect("unique BLAKE3 placement");
        }
    };
    for i in 0..8 {
        place(&cols.cv[i], blake3::CV_BASE + 32 * i);
        place(&cols.out_lo[i], blake3::OUT_LO_BASE + 32 * i);
        place(&cols.out_hi[i], blake3::OUT_HI_BASE + 32 * i);
    }
    for i in 0..16 {
        place(&cols.message[i], blake3::M_BASE + 32 * i);
    }
    place(&cols.counter[0], blake3::T_LO_BASE);
    place(&cols.counter[1], blake3::T_HI_BASE);
    place(&cols.block_len, blake3::BLEN_BASE);
    place(&cols.flags, blake3::FLAGS_BASE);
    for (g, cols) in cols.rounds.iter().flatten().enumerate() {
        place(
            &cols.a_first.majority_product,
            blake3::g_bit(g, blake3::OFF_MAJ1),
        );
        place(
            &cols.a_first.ripple_product,
            blake3::g_bit(g, blake3::OFF_RIP1),
        );
        match &cols.c_first {
            FirstC::Constant(add) => place(&add.carry_product, blake3::g_bit(g, blake3::OFF_C1)),
            FirstC::Variable(add) => place(&add.carry_product, blake3::g_bit(g, blake3::OFF_C1)),
        }
        place(
            &cols.a_second.majority_product,
            blake3::g_bit(g, blake3::off_maj2(g)),
        );
        place(
            &cols.a_second.ripple_product,
            blake3::g_bit(g, blake3::off_rip2(g)),
        );
        place(
            &cols.c_second.carry_product,
            blake3::g_bit(g, blake3::off_c2(g)),
        );
    }
    place(&[circuit.one()], blake3::Z_CONST_POS);
    let layout = layout
        .finish()
        .expect("complete BLAKE3 compatibility layout");
    assert_eq!(layout.useful_bits(), blake3::USEFUL_BITS);
    layout
}
