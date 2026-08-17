pub const BASE_MESSAGE_BITS: usize = 64;
pub const PRODUCT_MESSAGE_BITS: usize = 222;

pub(crate) const PRODUCT_LIMBS: usize = 4;
pub(crate) const PRODUCT_MESSAGE_BYTES: usize = PRODUCT_MESSAGE_BITS.div_ceil(8);
pub(crate) const MAX_X_DEGREE: usize = 49;
pub(crate) const X_POWER_COUNT: usize = MAX_X_DEGREE + 1;
pub(crate) const SAMPLE_X_POWER_COUNT: usize = 12;
pub(crate) const COVER_BASIS_LEN: usize = 8;
pub(crate) const BASE_Y_DEGREE: usize = 4;
/// Number of (cover monomial, y power) groups; shared by both evaluators.  The
/// per-evaluator slot count is `GAMMA_GROUP_COUNT * <x-power stride>`.
pub(crate) const GAMMA_GROUP_COUNT: usize = COVER_BASIS_LEN * BASE_Y_DEGREE;
pub(crate) const FOUR_RUSSIANS_BLOCK_BITS: usize = 7;
pub(crate) const FOUR_RUSSIANS_TABLE_SIZE: usize = 1 << FOUR_RUSSIANS_BLOCK_BITS;

// Base-code evaluator (the direct `C(m)(P)` shortcut).  The evaluator dots a
// 64-bit base message against 64 Lagrange functions instead of the 222 product
// functions, and its common denominator has smaller x-degree.
pub(crate) const BASE_FUNCTIONAL_BITS: usize = BASE_MESSAGE_BITS;
pub(crate) const BASE_LIMBS: usize = 1;
pub(crate) const BASE_FUNCTIONAL_BYTES: usize = BASE_FUNCTIONAL_BITS.div_ceil(8);
pub(crate) const BASE_MAX_X_DEGREE: usize = 31;
pub(crate) const BASE_X_POWER_COUNT: usize = BASE_MAX_X_DEGREE + 1;
