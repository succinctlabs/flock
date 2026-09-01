use std::{mem::MaybeUninit, ops::Index, slice::Iter, sync::OnceLock};

use crate::genus95_curve_code::{
    constants::{
        BASE_Y_DEGREE, COVER_BASIS_LEN, FOUR_RUSSIANS_BLOCK_BITS, FOUR_RUSSIANS_TABLE_SIZE,
        PRODUCT_MESSAGE_BITS, PRODUCT_MESSAGE_BYTES, X_POWER_COUNT,
    },
    field::{F128, F128Ext},
    messages::ProductMessage,
    tables::{FourRussiansLayout, TABLES},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationPoint {
    pub x: F128,
    pub y: F128,
    pub z1: F128,
    pub z2: F128,
    pub z3: F128,
}

pub struct ProductFunctional {
    coordinates: [F128; PRODUCT_MESSAGE_BITS],
    byte_dot_tables: OnceLock<Box<[[F128; 256]; PRODUCT_MESSAGE_BYTES]>>,
}

impl ProductFunctional {
    fn new(coordinates: [F128; PRODUCT_MESSAGE_BITS]) -> Self {
        Self {
            coordinates,
            byte_dot_tables: OnceLock::new(),
        }
    }

    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, F128> {
        self.coordinates.iter()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        PRODUCT_MESSAGE_BITS
    }

    /// Always false — the functional has a fixed nonzero coordinate count.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl Index<usize> for ProductFunctional {
    type Output = F128;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.coordinates[index]
    }
}

/// Accumulates `N` GF(2^128) functional coordinates from a stream of base
/// monomial values, using a four-Russians blocking over the binary gamma
/// coefficient masks.  Generic over the coordinate count so the product
/// evaluator (`N = 222`) and the base evaluator (`N = 64`) share one kernel.
pub(crate) struct FunctionalBuilder<const N: usize> {
    out: [F128; N],
    block_values: [F128; FOUR_RUSSIANS_BLOCK_BITS],
    input_index: usize,
    block_index: usize,
    block_fill: usize,
}

impl<const N: usize> FunctionalBuilder<N> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            out: [F128::ZERO; N],
            block_values: [F128::ZERO; FOUR_RUSSIANS_BLOCK_BITS],
            input_index: 0,
            block_index: 0,
            block_fill: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, value: F128, layout: &FourRussiansLayout<N>) {
        self.block_values[self.block_fill] = value;
        self.input_index += 1;
        self.block_fill += 1;
        if self.block_fill == FOUR_RUSSIANS_BLOCK_BITS {
            self.flush_block(layout);
        }
    }

    #[inline(always)]
    fn flush_block(&mut self, layout: &FourRussiansLayout<N>) {
        let coordinate_start = layout.block_coordinate_offsets[self.block_index];
        let coordinate_end = layout.block_coordinate_offsets[self.block_index + 1];
        apply_four_russians_block(
            &mut self.out,
            &self.block_values,
            &layout.block_masks[self.block_index],
            &layout.block_coordinates[coordinate_start..coordinate_end],
        );
        self.block_index += 1;
        self.block_fill = 0;
    }

    #[inline(always)]
    pub(crate) fn finish(mut self, layout: &FourRussiansLayout<N>) -> [F128; N] {
        if self.block_fill != 0 {
            self.flush_block(layout);
        }
        debug_assert_eq!(self.input_index, layout.input_count);
        self.out
    }
}

/// Stream the monomial values of an evaluation point into a `FunctionalBuilder`,
/// returning the `N` raw functional coordinates and the common denominator.
///
/// The group `(cover_index, y_index) == 0` carries the constant cover monomial
/// `M_0 = 1` at `y^0`, which is where the common denominator `H(x)` lives, so it
/// is folded in along that group's degree loop instead of being recomputed.
#[inline(always)]
pub(crate) fn build_functional<const N: usize, const XPC: usize>(
    point: &EvaluationPoint,
    layout: &FourRussiansLayout<N>,
    denominator_mask: u64,
) -> ([F128; N], F128) {
    let x_powers = x_powers_n::<XPC>(point.x);
    let y_powers = y_powers(point.y);
    let z_monomials = z_monomials(point.z1, point.z2, point.z3);
    let mut denominator = F128::ZERO;
    let mut builder = FunctionalBuilder::<N>::new();

    for (cover_index, z_monomial) in z_monomials.iter().copied().enumerate() {
        for (y_index, y_power) in y_powers.iter().copied().enumerate() {
            let base_value = z_monomial * y_power;
            let group_index = cover_index * BASE_Y_DEGREE + y_index;
            if group_index == 0 {
                let slot_base = group_index * XPC;
                let degree_count = layout.group_degree_counts[group_index] as usize;
                for degree in 0..degree_count {
                    let value = base_value * x_powers[degree];
                    if ((denominator_mask >> degree) & 1) != 0 {
                        denominator += value;
                    }
                    if layout.slot_nonempty[slot_base + degree] {
                        builder.push(value, layout);
                    }
                }
            } else {
                let degree_start = layout.group_degree_offsets[group_index] as usize;
                let degree_end = layout.group_degree_offsets[group_index + 1] as usize;
                for degree in layout.group_degrees[degree_start..degree_end]
                    .iter()
                    .copied()
                    .map(usize::from)
                {
                    builder.push(base_value * x_powers[degree], layout);
                }
            }
        }
    }

    (builder.finish(layout), denominator)
}

/// Compute the 222 GF(2^128) coordinates of the product-code evaluation
/// functional at `point`.  Returns `None` if the common denominator vanishes.
pub fn product_evaluation_functional(point: &EvaluationPoint) -> Option<ProductFunctional> {
    let tables = &*TABLES;
    let (mut out, denominator) = build_functional::<PRODUCT_MESSAGE_BITS, X_POWER_COUNT>(
        point,
        &tables.product_layout,
        tables.product_denominator,
    );

    let denominator_inv = denominator.inverse()?;
    for value in &mut out {
        *value *= denominator_inv;
    }

    Some(ProductFunctional::new(out))
}

/// Evaluate a product-code message against a precomputed product-code
/// evaluation functional.
#[inline(always)]
pub fn evaluate_product_functional(
    functional: &ProductFunctional,
    message: &ProductMessage,
) -> F128 {
    let byte_dot_tables = functional
        .byte_dot_tables
        .get_or_init(|| build_functional_byte_dot_tables(&functional.coordinates));
    let mut out = F128::ZERO;
    xor_full_dot_table_limb(&mut out, byte_dot_tables, 0, message.limbs[0]);
    xor_full_dot_table_limb(&mut out, byte_dot_tables, 8, message.limbs[1]);
    xor_full_dot_table_limb(&mut out, byte_dot_tables, 16, message.limbs[2]);

    let limb = message.limbs[3];
    out += byte_dot_tables[24][(limb & 0xff) as usize];
    out += byte_dot_tables[25][((limb >> 8) & 0xff) as usize];
    out += byte_dot_tables[26][((limb >> 16) & 0xff) as usize];
    out += byte_dot_tables[27][((limb >> 24) & 0xff) as usize];
    out
}

#[inline(always)]
fn xor_full_dot_table_limb(
    out: &mut F128,
    byte_dot_tables: &[[F128; 256]; PRODUCT_MESSAGE_BYTES],
    byte_base: usize,
    limb: u64,
) {
    *out += byte_dot_tables[byte_base][(limb & 0xff) as usize];
    *out += byte_dot_tables[byte_base + 1][((limb >> 8) & 0xff) as usize];
    *out += byte_dot_tables[byte_base + 2][((limb >> 16) & 0xff) as usize];
    *out += byte_dot_tables[byte_base + 3][((limb >> 24) & 0xff) as usize];
    *out += byte_dot_tables[byte_base + 4][((limb >> 32) & 0xff) as usize];
    *out += byte_dot_tables[byte_base + 5][((limb >> 40) & 0xff) as usize];
    *out += byte_dot_tables[byte_base + 6][((limb >> 48) & 0xff) as usize];
    *out += byte_dot_tables[byte_base + 7][((limb >> 56) & 0xff) as usize];
}

/// Build per-byte dot-product tables so a functional can be dotted against a
/// packed binary message a byte at a time.  Generic over the coordinate count
/// `N` and the number of message bytes `BYTES`.
pub(crate) fn build_functional_byte_dot_tables<const N: usize, const BYTES: usize>(
    coordinates: &[F128; N],
) -> Box<[[F128; 256]; BYTES]> {
    let mut tables = Box::new([[F128::ZERO; 256]; BYTES]);
    for byte_index in 0..BYTES {
        let coordinate_base = 8 * byte_index;
        let remaining = N.saturating_sub(coordinate_base);
        let bit_count = remaining.min(8);
        for bit in 0..bit_count {
            let bit_mask = 1usize << bit;
            let coordinate = coordinates[coordinate_base + bit];
            for mask in 0..bit_mask {
                tables[byte_index][bit_mask | mask] = tables[byte_index][mask] + coordinate;
            }
        }
    }
    tables
}

#[inline(always)]
fn apply_four_russians_block<const N: usize>(
    out: &mut [F128; N],
    values: &[F128; FOUR_RUSSIANS_BLOCK_BITS],
    coordinate_masks: &[u8; N],
    nonzero_coordinates: &[u16],
) {
    let mut subset_sums = [MaybeUninit::<F128>::uninit(); FOUR_RUSSIANS_TABLE_SIZE];
    subset_sums[0].write(F128::ZERO);
    for bit in 0..FOUR_RUSSIANS_BLOCK_BITS {
        let bit_mask = 1usize << bit;
        for mask in 0..bit_mask {
            // SAFETY: this loop builds subset sums in increasing powers of two,
            // so every `mask < bit_mask` was initialized by an earlier step.
            let previous = unsafe { subset_sums[mask].assume_init() };
            subset_sums[bit_mask | mask].write(previous + values[bit]);
        }
    }

    for coordinate in nonzero_coordinates.iter().copied() {
        let coordinate = coordinate as usize;
        let mask = coordinate_masks[coordinate];
        debug_assert!(mask != 0);
        // SAFETY: all nonzero masks are below 2^FOUR_RUSSIANS_BLOCK_BITS and
        // were initialized by the subset-sum construction above.
        out[coordinate] += unsafe { subset_sums[mask as usize].assume_init() };
    }
}

#[inline(always)]
pub(crate) fn x_powers(x: F128) -> [F128; X_POWER_COUNT] {
    x_powers_n::<X_POWER_COUNT>(x)
}

#[inline(always)]
pub(crate) fn x_powers_n<const N: usize>(x: F128) -> [F128; N] {
    let mut powers = [F128::ZERO; N];
    powers[0] = F128::ONE;
    for i in 1..N {
        powers[i] = powers[i - 1] * x;
    }
    powers
}

#[inline(always)]
pub(crate) fn y_powers(y: F128) -> [F128; BASE_Y_DEGREE] {
    let y2 = y.square();
    [F128::ONE, y, y2, y2 * y]
}

#[inline(always)]
fn z_monomials(z1: F128, z2: F128, z3: F128) -> [F128; COVER_BASIS_LEN] {
    let z2z3 = z2 * z3;
    let z1z3 = z1 * z3;
    let z1z2 = z1 * z2;
    [F128::ONE, z3, z2, z2z3, z1, z1z3, z1z2, z1z2 * z3]
}

#[inline(always)]
pub(crate) fn eval_poly_mask<const N: usize>(mask: u64, powers: &[F128; N]) -> F128 {
    let mut bits = mask;
    let mut out = F128::ZERO;
    while bits != 0 {
        let degree = bits.trailing_zeros() as usize;
        debug_assert!(degree < N);
        out += powers[degree];
        bits &= bits - 1;
    }
    out
}
