//! GF(2^128) for the genus-95 routines is the crate-wide [`crate::field::F128`]
//! (GHASH form, irreducible polynomial x^128 + x^7 + x^2 + x + 1) — the same
//! field, with the same hardware-accelerated PMULL/CLMUL arithmetic, that the
//! rest of flock uses.
//!
//! This module previously carried its own standalone `u128`-backed
//! implementation; it now re-exports the shared type and supplies the few
//! helpers the genus-95 code needs on top of flock's API:
//!   - [`F128Ext::inverse`] — `Option`-returning inverse (`None` at zero),
//!     matching the original signature; flock's inherent `inv()` maps 0 → 0.
//!   - [`F128Ext::random`] — uniform sampling from an `RngCore`.
//!   - [`F128Ext::to_bits`]/[`F128Ext::from_bits`] — raw 128-bit access for the
//!     Artin–Schreier GF(2) linear algebra (coefficient i is bit i).
//!
//! `square` is provided by the shared `F128` itself (a dedicated 2-CLMUL
//! square), so the genus-95 code calls `value.square()` directly.
//!
//! Field addition is XOR; the shared `F128` spells it `+` (via `Add`), so the
//! genus-95 code uses `+`/`+=` like the rest of the repository.

use rand_core::RngCore;

pub use crate::field::F128;

/// Genus-95-specific helpers layered onto the shared [`F128`].
pub(crate) trait F128Ext: Sized {
    fn inverse(self) -> Option<Self>;
    fn random(rng: &mut impl RngCore) -> Self;
    fn to_bits(self) -> u128;
    fn from_bits(bits: u128) -> Self;
}

impl F128Ext for F128 {
    #[inline(always)]
    fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.inv())
        }
    }

    #[inline(always)]
    fn random(rng: &mut impl RngCore) -> Self {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    #[inline(always)]
    fn to_bits(self) -> u128 {
        (self.lo as u128) | ((self.hi as u128) << 64)
    }

    #[inline(always)]
    fn from_bits(bits: u128) -> Self {
        F128::new(bits as u64, (bits >> 64) as u64)
    }
}
