//! Binary field arithmetic for Flock protocols.

pub mod f128_slice;
pub mod gf2_128;
pub mod gf2_256;
pub mod gf2_8;
pub mod phi8;

pub use gf2_8::F8;
pub use gf2_128::{F128, F256Unreduced, mul_by_x};
pub use gf2_256::{F256, QUADRATIC_NONRESIDUE, mul_by_x_inv};
pub use phi8::{PHI_8_TABLE, phi8};
pub mod test_rng;
