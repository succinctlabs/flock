//! Legacy positions live only in this adapter, never in SHA evaluation.
use super::{Sha256Cols, sha2};
use flock_core::circuit::boolean::{BooleanCircuit, PhysicalLayout, ValueId};

pub(super) fn layout(circuit: &BooleanCircuit, cols: &Sha256Cols<ValueId>) -> PhysicalLayout {
    let mut layout = circuit.layout();
    let mut place = |values: &[ValueId], base: usize| {
        for (offset, &value) in values.iter().enumerate() {
            layout
                .place_definition(value, base + offset)
                .expect("unique SHA placement");
        }
    };
    for i in 0..8 {
        place(&cols.h_in[i], sha2::H_BASE + 32 * i);
    }
    for i in 0..16 {
        place(&cols.message[i], sha2::M_BASE + 32 * i);
    }
    for (i, add) in cols.schedule.iter().enumerate() {
        place(
            &add.majority_first_product,
            sha2::sched_bit(i + 16, sha2::SC_MAJ1),
        );
        place(
            &add.majority_second_product,
            sha2::sched_bit(i + 16, sha2::SC_MAJ2),
        );
        place(&add.ripple_product, sha2::sched_bit(i + 16, sha2::SC_RIP));
    }
    for (r, cols) in cols.rounds.iter().enumerate() {
        place(&cols.choose_product, sha2::ch_and_bit(r, 0));
        place(&cols.majority_product, sha2::maj_and_bit(r, 0));
        place(
            &cols.add_constant.carry_product,
            sha2::round_bit(r, sha2::RC_ADDK),
        );
        place(
            &cols.t1.majority_first_product,
            sha2::round_bit(r, sha2::RC_MAJ1),
        );
        place(
            &cols.t1.majority_second_product,
            sha2::round_bit(r, sha2::RC_MAJ2),
        );
        place(&cols.t1.ripple_product, sha2::round_bit(r, sha2::RC_RIP));
        place(
            &cols.a_new.majority_product,
            sha2::round_bit(r, sha2::RC_AMAJ),
        );
        place(
            &cols.a_new.ripple_product,
            sha2::round_bit(r, sha2::RC_ARIP),
        );
        place(&cols.e_new.carry_product, sha2::round_bit(r, sha2::RC_ENEW));
        if let Some(state) = &cols.state {
            place(&state.a, sha2::a_new_bit(r, 0));
            place(&state.e, sha2::e_new_bit(r, 0));
        }
    }
    for i in 0..8 {
        place(&cols.output_add[i].carry_product, sha2::out_carry_bit(i, 0));
        place(&cols.h_out[i], sha2::h_out_bit(i, 0));
    }
    place(&[circuit.one()], sha2::Z_CONST_POS);
    let layout = layout.finish().expect("complete SHA compatibility layout");
    assert_eq!(layout.useful_bits(), sha2::USEFUL_BITS);
    layout
}
