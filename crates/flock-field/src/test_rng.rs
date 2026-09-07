//! The workspace's deterministic test/bench/vector-dump RNG: SplitMix64.
//!
//! Every test, bench and `dump_*_vectors` bin used to carry its own copy of
//! this generator (82 copies, 49 spellings — bloat ledger §G cluster 8; lives in flock-field so every crate above it can use it).
//! They all drew the same SplitMix64 stream, and byte-pinned fixtures
//! (`union_m6_fixtures`, `union_element`, `transcript_shape`, the tower
//! tape pins) depend on the exact draw order, so every method here keeps the
//! derivation the copies shared:
//!
//! - [`Rng::next_u64`] — one SplitMix64 step;
//! - [`Rng::next_u32`] — one step, low 32 bits;
//! - [`Rng::f128`] — two steps, `lo` first then `hi`;
//! - [`Rng::bit`] / [`Rng::bits`] — one step per bit, its low bit;
//! - [`Rng::fill_bytes`] — one step per 8 bytes, little-endian, one extra
//!   step for a partial tail.
//!
//! Sites whose helpers derived differently (packed bit vectors, custom
//! mixers) keep those helpers locally, under different names.
//!
//! NOT a cryptographic generator: never use it for Fiat–Shamir or keys.

use crate::{F128, F256};

/// SplitMix64 state. `Rng(seed)` and [`Rng::new`] are the same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng(pub u64);

impl Rng {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// One SplitMix64 step.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// One step, low 32 bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// One step, its low bit.
    #[inline]
    pub fn bit(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// `n` bits, one step each.
    pub fn bits(&mut self, n: usize) -> Vec<bool> {
        (0..n).map(|_| self.bit()).collect()
    }

    /// Two steps: `lo`, then `hi`.
    #[inline]
    pub fn f128(&mut self) -> F128 {
        F128::new(self.next_u64(), self.next_u64())
    }

    pub fn f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.f128()).collect()
    }

    /// Two [`Self::f128`] draws: the base coefficient first.
    pub fn f256(&mut self) -> F256 {
        F256::new(self.f128(), self.f128())
    }

    /// Rejection-sampled nonzero element.
    pub fn nonzero(&mut self) -> F128 {
        loop {
            let v = self.f128();
            if !v.is_zero() {
                return v;
            }
        }
    }

    /// One step, reduced modulo `n` (biased, as every local copy was).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher–Yates from the top, one step per swap.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }

    /// One step per 8 bytes (little-endian), one more for a partial tail.
    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        let len = buf.len();
        let mut i = 0;
        while i + 8 <= len {
            let v = self.next_u64();
            buf[i..i + 8].copy_from_slice(&v.to_le_bytes());
            i += 8;
        }
        if i < len {
            let v = self.next_u64().to_le_bytes();
            buf[i..].copy_from_slice(&v[..len - i]);
        }
    }
}
