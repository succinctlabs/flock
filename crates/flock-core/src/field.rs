//! Binary field arithmetic.
//!
//! - [`F8`]   — GF(2^8) with AES polynomial x^8 + x^4 + x^3 + x + 1
//! - [`F128`] — GF(2^128) in GHASH form, polynomial x^128 + x^7 + x^2 + x + 1
//! - [`F256`] — quadratic extension of F128 defined by u^2 + u + x^-1
//! - [`F256Unreduced`] — 256-bit unreduced GHASH products, for deferred reduction

pub use flock_field::{
    F8, F128, F256, F256Unreduced, PHI_8_TABLE, QUADRATIC_NONRESIDUE, f128_slice, gf2_8, gf2_128,
    gf2_256, mul_by_x, mul_by_x_inv, phi8, test_rng,
};
