use std::sync::LazyLock;

#[cfg(test)]
use {
    crate::genus95_curve_code::constants::BASE_Y_DEGREE,
    crate::genus95_curve_code::evaluator::eval_poly_mask,
    crate::genus95_curve_code::field::{F128, F128Ext},
    crate::genus95_curve_code::sage_data::ARTIN_SCHREIER_RHS,
};

use crate::genus95_curve_code::{
    artin_schreier::ArtinSchreierSolver,
    constants::{
        BASE_FUNCTIONAL_BITS, BASE_LIMBS, BASE_X_POWER_COUNT, FOUR_RUSSIANS_BLOCK_BITS,
        GAMMA_GROUP_COUNT, PRODUCT_LIMBS, PRODUCT_MESSAGE_BITS, X_POWER_COUNT,
    },
    messages::ExtendedMessage,
    sage_data::{
        BASE_DENOMINATOR, BASE_GAMMA_SLOT_MASKS, GAMMA_SLOT_MASKS, PRODUCT_DENOMINATOR, R_ROWS,
    },
};

/// A coefficient that is a ratio of two GF(2) polynomials in `x`, each stored as
/// a degree-bitmask.  Only used by the test that checks sampled points satisfy
/// the Artin–Schreier equations; the baked values live in
/// [`super::sage_data::ARTIN_SCHREIER_RHS`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RationalMask {
    pub(crate) numerator: u64,
    pub(crate) denominator: u64,
}

#[cfg(test)]
impl RationalMask {
    pub(crate) fn eval<const N: usize>(self, x_powers: &[F128; N]) -> Option<F128> {
        let numerator = eval_poly_mask(self.numerator, x_powers);
        if self.denominator == 1 {
            return Some(numerator);
        }
        let denominator = eval_poly_mask(self.denominator, x_powers);
        Some(numerator * denominator.inverse()?)
    }
}

#[cfg(test)]
fn artin_schreier_rhs() -> [[RationalMask; BASE_Y_DEGREE]; 3] {
    let mut out = [[RationalMask::default(); BASE_Y_DEGREE]; 3];
    for (i, row) in ARTIN_SCHREIER_RHS.iter().enumerate() {
        for (j, &(numerator, denominator)) in row.iter().enumerate() {
            out[i][j] = RationalMask {
                numerator,
                denominator,
            };
        }
    }
    out
}

/// Four-Russians evaluation layout for a functional with `N` GF(2^128)
/// coordinates.  Shared by the product evaluator (`N = 222`) and the base
/// evaluator (`N = 64`); see `FunctionalBuilder` in `evaluator`.
pub(crate) struct FourRussiansLayout<const N: usize> {
    pub(crate) slot_nonempty: Vec<bool>,
    pub(crate) group_degree_counts: [u8; GAMMA_GROUP_COUNT],
    pub(crate) group_degree_offsets: [u16; GAMMA_GROUP_COUNT + 1],
    pub(crate) group_degrees: Vec<u8>,
    pub(crate) block_masks: Vec<[u8; N]>,
    pub(crate) block_coordinate_offsets: Vec<usize>,
    pub(crate) block_coordinates: Vec<u16>,
    pub(crate) input_count: usize,
}

pub(crate) struct Tables {
    pub(crate) extended_byte_table: [[ExtendedMessage; 256]; 8],
    pub(crate) product_denominator: u64,
    pub(crate) product_layout: FourRussiansLayout<PRODUCT_MESSAGE_BITS>,
    pub(crate) base_denominator: u64,
    pub(crate) base_layout: FourRussiansLayout<BASE_FUNCTIONAL_BITS>,
    #[cfg(test)]
    pub(crate) artin_schreier_rhs: [[RationalMask; BASE_Y_DEGREE]; 3],
    pub(crate) as_solver: ArtinSchreierSolver,
}

impl Tables {
    fn load() -> Self {
        const _: () = assert!(R_ROWS.len() == PRODUCT_MESSAGE_BITS);

        let r1: [u64; 64] = R_ROWS[64..128].try_into().unwrap();
        let r2: [u64; 64] = R_ROWS[128..192].try_into().unwrap();
        let r3: [u64; 30] = R_ROWS[192..222].try_into().unwrap();

        let product_layout =
            build_four_russians_tables::<PRODUCT_MESSAGE_BITS, X_POWER_COUNT, PRODUCT_LIMBS>(
                &GAMMA_SLOT_MASKS,
            );
        let base_layout =
            build_four_russians_tables::<BASE_FUNCTIONAL_BITS, BASE_X_POWER_COUNT, BASE_LIMBS>(
                &BASE_GAMMA_SLOT_MASKS,
            );

        Self {
            extended_byte_table: build_extended_byte_table(&r1, &r2, &r3),
            product_denominator: PRODUCT_DENOMINATOR,
            product_layout,
            base_denominator: BASE_DENOMINATOR,
            base_layout,
            #[cfg(test)]
            artin_schreier_rhs: artin_schreier_rhs(),
            as_solver: ArtinSchreierSolver::new(),
        }
    }
}

pub(crate) static TABLES: LazyLock<Tables> = LazyLock::new(Tables::load);

fn build_row_byte_table(rows: &[u64]) -> [[u64; 256]; 8] {
    let mut table = [[0u64; 256]; 8];
    for (byte_index, byte_table) in table.iter_mut().enumerate() {
        let shift = 8 * byte_index;
        for (byte, entry) in byte_table.iter_mut().enumerate() {
            let message = (byte as u64) << shift;
            let mut out = 0u64;
            for (row_index, row) in rows.iter().copied().enumerate() {
                out |= (((row & message).count_ones() as u64) & 1) << row_index;
            }
            *entry = out;
        }
    }
    table
}

fn build_extended_byte_table(
    r1: &[u64; 64],
    r2: &[u64; 64],
    r3: &[u64; 30],
) -> [[ExtendedMessage; 256]; 8] {
    let r1_byte_table = build_row_byte_table(r1);
    let r2_byte_table = build_row_byte_table(r2);
    let r3_byte_table = build_row_byte_table(r3);

    let mut table = [[ExtendedMessage::default(); 256]; 8];
    for byte_index in 0..8 {
        for byte in 0..256 {
            // `order0` (the raw message bits) is the identity under this
            // extension and is reconstructed by the caller. `order3` is 30-bit,
            // hence the `as u32`; the lower orders restricted to the points are
            // the low bits of order1/order2, taken directly at combine time.
            table[byte_index][byte] = ExtendedMessage {
                order1: r1_byte_table[byte_index][byte],
                order2: r2_byte_table[byte_index][byte],
                order3: r3_byte_table[byte_index][byte] as u32,
            };
        }
    }
    table
}

/// Build the four-Russians evaluation layout from the per-slot gamma masks.
///
/// `N` is the functional coordinate count, `XPC` the x-power stride of a gamma
/// group, and `L` the number of `u64` limbs needed to hold `N` coordinate bits.
fn build_four_russians_tables<const N: usize, const XPC: usize, const L: usize>(
    masks: &[[u64; L]],
) -> FourRussiansLayout<N> {
    let slot_count = GAMMA_GROUP_COUNT * XPC;
    debug_assert_eq!(masks.len(), slot_count);

    let mut slot_nonempty = vec![false; slot_count];
    let mut group_degree_counts = [0u8; GAMMA_GROUP_COUNT];
    let mut group_degree_offsets = [0u16; GAMMA_GROUP_COUNT + 1];
    let mut group_degrees = Vec::<u8>::new();
    let mut block_masks = Vec::<[u8; N]>::new();
    let mut input_count = 0usize;

    for (slot, mask) in masks.iter().enumerate() {
        let block_index = input_count / FOUR_RUSSIANS_BLOCK_BITS;
        let block_bit = 1u8 << (input_count % FOUR_RUSSIANS_BLOCK_BITS);
        let mut any = false;

        for (limb_index, mut bits) in mask.iter().copied().enumerate() {
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let index = limb_index * 64 + bit;
                if index < N {
                    if !any {
                        any = true;
                        slot_nonempty[slot] = true;
                        if block_index == block_masks.len() {
                            block_masks.push([0u8; N]);
                        }
                    }
                    block_masks[block_index][index] |= block_bit;
                }
                bits &= bits - 1;
            }
        }

        if any {
            let group = slot / XPC;
            let degree = slot % XPC;
            group_degree_counts[group] = group_degree_counts[group].max((degree + 1) as u8);
            input_count += 1;
        }
    }

    for group in 0..GAMMA_GROUP_COUNT {
        group_degree_offsets[group] = group_degrees.len() as u16;
        let slot_base = group * XPC;
        for degree in 0..group_degree_counts[group] as usize {
            if slot_nonempty[slot_base + degree] {
                group_degrees.push(degree as u8);
            }
        }
    }
    group_degree_offsets[GAMMA_GROUP_COUNT] = group_degrees.len() as u16;

    let mut block_coordinate_offsets = Vec::with_capacity(block_masks.len() + 1);
    let mut block_coordinates = Vec::<u16>::new();
    for block_mask in &block_masks {
        block_coordinate_offsets.push(block_coordinates.len());
        for (coordinate, mask) in block_mask.iter().copied().enumerate() {
            if mask != 0 {
                block_coordinates.push(coordinate as u16);
            }
        }
    }
    block_coordinate_offsets.push(block_coordinates.len());

    FourRussiansLayout {
        slot_nonempty,
        group_degree_counts,
        group_degree_offsets,
        group_degrees,
        block_masks,
        block_coordinate_offsets,
        block_coordinates,
        input_count,
    }
}
