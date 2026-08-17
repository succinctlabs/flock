use super::constants::{PRODUCT_LIMBS, PRODUCT_MESSAGE_BITS};
use rand_core::RngCore;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BaseMessage(pub u64);

impl BaseMessage {
    #[inline(always)]
    pub fn random(rng: &mut impl RngCore) -> Self {
        Self(rng.next_u64())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProductMessage {
    pub limbs: [u64; PRODUCT_LIMBS],
}

impl ProductMessage {
    #[inline(always)]
    pub fn random(rng: &mut impl RngCore) -> Self {
        let limbs = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64() & ((1u64 << 30) - 1),
        ];
        Self { limbs }
    }

    #[inline(always)]
    pub fn get_bit(&self, index: usize) -> bool {
        debug_assert!(index < PRODUCT_MESSAGE_BITS);
        ((self.limbs[index >> 6] >> (index & 63)) & 1) != 0
    }

    #[inline(always)]
    pub(crate) fn from_sections(order0: u64, order1: u64, order2: u64, order3: u64) -> Self {
        Self {
            limbs: [order0, order1, order2, order3 & ((1u64 << 30) - 1)],
        }
    }
}

/// One base message exposed at every order, so two of them can be combined
/// bilinearly into a [`ProductMessage`] (see [`super::product`]).
///
/// `order0` (the raw message bits) is the identity under the extension, so it
/// is not stored here — the caller supplies it directly from the `BaseMessage`.
///
/// The order-3 product section lives on the 30 third-derivative points, and the
/// baked tables permute the base coordinates so those points are exactly 0..29.
/// That makes "a lower order restricted to the points" just its **low 30 bits**
/// — derivable on the fly from the full `order1`/`order2`/message — so nothing
/// extra needs storing. These three fields are all that's required:
/// `order1`/`order2` (full 64-bit) and `order3` (30-bit).
#[derive(Clone, Copy, Default)]
pub(crate) struct ExtendedMessage {
    /// First-order coordinates (full 64 bits).
    pub(crate) order1: u64,
    /// Second-order coordinates (full 64 bits).
    pub(crate) order2: u64,
    /// Third-order coordinates: 30 bits on coordinates 0..29.
    pub(crate) order3: u32,
}

impl std::ops::BitXor for ExtendedMessage {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            order1: self.order1 ^ rhs.order1,
            order2: self.order2 ^ rhs.order2,
            order3: self.order3 ^ rhs.order3,
        }
    }
}

impl std::ops::BitXorAssign for ExtendedMessage {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}
