use std::ops::Index;
use std::sync::OnceLock;

use super::constants::{BASE_FUNCTIONAL_BITS, BASE_FUNCTIONAL_BYTES, BASE_X_POWER_COUNT};
use super::evaluator::{EvaluationPoint, build_functional, build_functional_byte_dot_tables};
use super::field::{F128, F128Ext};
use super::messages::BaseMessage;
use super::tables::TABLES;

/// The 64 GF(2^128) coordinates of the direct base-code evaluation functional
/// at a point.  Dotting a 64-bit base message against it gives `C(m)(P)`
/// without first extending `m` to a 222-coordinate product message.
pub struct BaseFunctional {
    coordinates: [F128; BASE_FUNCTIONAL_BITS],
    byte_dot_tables: OnceLock<Box<[[F128; 256]; BASE_FUNCTIONAL_BYTES]>>,
}

impl BaseFunctional {
    fn new(coordinates: [F128; BASE_FUNCTIONAL_BITS]) -> Self {
        Self {
            coordinates,
            byte_dot_tables: OnceLock::new(),
        }
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, F128> {
        self.coordinates.iter()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        BASE_FUNCTIONAL_BITS
    }

    /// Always false — the functional has a fixed nonzero coordinate count.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl Index<usize> for BaseFunctional {
    type Output = F128;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.coordinates[index]
    }
}

/// Compute the 64 GF(2^128) coordinates of the base-code evaluation functional
/// at `point`.  Returns `None` if the common denominator vanishes.
pub fn base_evaluation_functional(point: &EvaluationPoint) -> Option<BaseFunctional> {
    let tables = &*TABLES;
    let (mut out, denominator) = build_functional::<BASE_FUNCTIONAL_BITS, BASE_X_POWER_COUNT>(
        point,
        &tables.base_layout,
        tables.base_denominator,
    );

    let denominator_inv = denominator.inverse()?;
    for value in &mut out {
        *value *= denominator_inv;
    }

    Some(BaseFunctional::new(out))
}

/// Evaluate a base-code message against a precomputed base-code evaluation
/// functional.  Equivalent to `evaluate_product_functional(_, R*m)`.
#[inline(always)]
pub fn evaluate_base_functional(functional: &BaseFunctional, message: &BaseMessage) -> F128 {
    let byte_dot_tables = functional
        .byte_dot_tables
        .get_or_init(|| build_functional_byte_dot_tables(&functional.coordinates));
    let limb = message.0;
    let mut out = F128::ZERO;
    out += byte_dot_tables[0][(limb & 0xff) as usize];
    out += byte_dot_tables[1][((limb >> 8) & 0xff) as usize];
    out += byte_dot_tables[2][((limb >> 16) & 0xff) as usize];
    out += byte_dot_tables[3][((limb >> 24) & 0xff) as usize];
    out += byte_dot_tables[4][((limb >> 32) & 0xff) as usize];
    out += byte_dot_tables[5][((limb >> 40) & 0xff) as usize];
    out += byte_dot_tables[6][((limb >> 48) & 0xff) as usize];
    out += byte_dot_tables[7][((limb >> 56) & 0xff) as usize];
    out
}
