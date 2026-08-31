// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The default `Mul` implementation (`ghash_mul_binius`) is a port of
// `mul_clmul` from binius64
// (https://github.com/binius-zk/binius64, `crates/field/src/arch/shared/ghash.rs`).

//! GF(2^128) in GHASH form: irreducible polynomial x^128 + x^7 + x^2 + x + 1.
//!
//! Layout: `lo` holds coefficients x^0..x^63, `hi` holds x^64..x^127.
//! Hardware: `vmull_p64` (ARM PMULL, AES extension) does a 64×64 carry-less mul
//! in one instruction. Default `Mul` impl uses the binius64 reduction variant
//! (4 PMULL schoolbook + 2-stage recursive reduction, 2 extra PMULL), which
//! benchmarked as the fastest of four variants tried.

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use self::aarch64::{
    ghash_mul_binius as ghash_mul_binius_aarch64,
    ghash_mul_unreduced_neon as ghash_mul_unreduced_aarch64, ghash_square as ghash_square_aarch64,
};
#[cfg(feature = "mul-count")]
use self::op_count::{INVS, MULS};
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
)))]
use self::software::{
    ghash_mul as ghash_mul_software, ghash_mul_unreduced as ghash_mul_unreduced_software,
    ghash_square as ghash_square_software,
};
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
use self::x86_64::{
    ghash_mul_karatsuba_barrett as ghash_mul_x86_64,
    ghash_mul_unreduced_x86 as ghash_mul_unreduced_x86_64, ghash_square_x86 as ghash_square_x86_64,
};
#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use core::arch::x86_64::*;
use core::ops::{Add, AddAssign, BitXor, BitXorAssign, Mul, MulAssign};
#[cfg(feature = "mul-count")]
use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct F128 {
    pub lo: u64,
    pub hi: u64,
}

impl F128 {
    pub const ZERO: Self = Self { lo: 0, hi: 0 };
    pub const ONE: Self = Self { lo: 1, hi: 0 };

    #[inline]
    pub const fn new(lo: u64, hi: u64) -> Self {
        Self { lo, hi }
    }

    /// The generator γ (i.e. the element `x`). `mul_by_x` is a fast shift+fold.
    #[inline]
    pub const fn generator() -> Self {
        Self { lo: 2, hi: 0 }
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    /// Square `self` in GF(2^128). Carry-less squaring has no cross terms —
    /// `(a + b)^2 = a^2 + b^2` over GF(2) — so this needs only the two diagonal
    /// carry-less products (`lo·lo`, `hi·hi`) plus one reduction, half the
    /// PMULL/CLMUL of a general `Mul`. Hot in repeated-squaring paths
    /// (e.g. the genus-95 samplers' x-power chains).
    #[inline]
    pub fn square(self) -> Self {
        ghash_square(self)
    }

    /// 256-bit unreduced product `(self · rhs)`. Caller XORs many of these into
    /// an `F256Unreduced` accumulator and calls `.reduce()` once at the end.
    /// Reduction commutes with XOR, so Σ (aᵢ·bᵢ) mod p = (Σ aᵢ·bᵢ) mod p.
    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> F256Unreduced {
        ghash_mul_unreduced(self, rhs)
    }

    /// Multiplicative inverse via Fermat: x^{2^128 − 2}.
    /// Used in one-time setup (Lagrange weight computation), not in hot paths.
    pub fn inv(self) -> Self {
        #[cfg(feature = "mul-count")]
        INVS.fetch_add(1, Ordering::Relaxed);
        // x^{2^128 - 2} = ∏_{i=1..127} x^{2^i}
        let mut r = Self::ONE;
        let mut cur = self * self; // x^2
        for _ in 1..128 {
            r *= cur;
            cur = cur * cur;
        }
        r
    }
}

impl Add for F128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }
}

impl AddAssign for F128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

/// Global field-operation counters, enabled with `--features mul-count`.
///
/// Answers "where does the verifier's arithmetic go", which has **two
/// different answers** depending on whether you are costing the native
/// verifier or the recursion circuit:
///
/// - *natively* an inversion is `x^(2^128−2)`, i.e. 127 squarings + 127
///   multiplications — see [`F128::inv`]. It is ~255 muls, and a routine with
///   many inversions is dominated by them.
/// - *in-circuit* an inversion is **one constraint**: witness `y`, assert
///   `x·y = 1`. It costs the same as a multiplication.
///
/// So [`Snapshot::native_muls`] and [`Snapshot::circuit_constraints`] can rank
/// the same routines very differently, and both are reported. Ranking circuit
/// work by native profiling is precisely the mistake this exists to prevent.
#[cfg(feature = "mul-count")]
pub mod op_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Every `F128 * F128`, including the ones inside [`super::F128::inv`].
    pub static MULS: AtomicU64 = AtomicU64::new(0);
    /// Every [`super::F128::inv`] call.
    pub static INVS: AtomicU64 = AtomicU64::new(0);

    /// Muls performed inside one `inv`: `cur = self*self`, then 127 iterations
    /// of `r *= cur; cur = cur*cur`. Pinned by `inv_mul_cost_is_stable`.
    pub const MULS_PER_INV: u64 = 255;

    pub fn reset() {
        MULS.store(0, Relaxed);
        INVS.store(0, Relaxed);
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Snapshot {
        /// Multiplications actually executed, inversion internals included.
        pub native_muls: u64,
        pub invs: u64,
    }

    impl Snapshot {
        /// Multiplications excluding inversion internals — the "real" mul
        /// count of the algorithm.
        pub fn muls_excluding_inv(&self) -> u64 {
            self.native_muls.saturating_sub(self.invs * MULS_PER_INV)
        }

        /// What this costs the recursion circuit: one element-class
        /// constraint per multiplication, and one per inversion (witnessed
        /// reciprocal plus `x·y = 1`).
        pub fn circuit_constraints(&self) -> u64 {
            self.muls_excluding_inv() + self.invs
        }
    }

    pub fn snapshot() -> Snapshot {
        Snapshot {
            native_muls: MULS.load(Relaxed),
            invs: INVS.load(Relaxed),
        }
    }

    /// Reset, run `f`, and report what it cost. Not reentrant and not
    /// thread-isolated — the counters are global, so measure one thing at a
    /// time on an otherwise idle process.
    pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Snapshot) {
        reset();
        let out = f();
        (out, snapshot())
    }
}

impl Mul for F128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(feature = "mul-count")]
        MULS.fetch_add(1, Ordering::Relaxed);
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            // SAFETY: aes target feature is enabled at compile time.
            unsafe { ghash_mul_binius_aarch64(self, rhs) }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        {
            // SAFETY: pclmulqdq target feature is enabled at compile time.
            // On Zen4, karatsuba+barrett is ~17% faster in throughput (the
            // dominant mode for the bulk parallel F128 work) than binius, which
            // only wins the latency microbench. (M-series picked binius.)
            unsafe { ghash_mul_x86_64(self, rhs) }
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        {
            ghash_mul_software(self, rhs)
        }
    }
}

impl MulAssign for F128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Multiply by x (the generator). One shift + conditional XOR with 0x87, no PMULL.
/// Used by the sumcheck round when the fixed evaluation point is the generator.
#[inline]
pub const fn mul_by_x(z: F128) -> F128 {
    let carry = z.hi >> 63;
    let mask = 0u64.wrapping_sub(carry); // 0 or all-ones
    F128 {
        lo: (z.lo << 1) ^ (0x87 & mask),
        hi: (z.hi << 1) | (z.lo >> 63),
    }
}

// ---------------------------------------------------------------------------
// Deferred reduction: 256-bit unreduced products that can be XOR-accumulated.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F256Unreduced {
    pub r0: u64,
    pub r1: u64,
    pub r2: u64,
    pub r3: u64,
}

impl F256Unreduced {
    pub const ZERO: Self = Self {
        r0: 0,
        r1: 0,
        r2: 0,
        r3: 0,
    };

    #[inline]
    pub fn reduce(self) -> F128 {
        ghash_reduce(self.r0, self.r1, self.r2, self.r3)
    }
}

impl BitXor for F256Unreduced {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self {
            r0: self.r0 ^ rhs.r0,
            r1: self.r1 ^ rhs.r1,
            r2: self.r2 ^ rhs.r2,
            r3: self.r3 ^ rhs.r3,
        }
    }
}

impl BitXorAssign for F256Unreduced {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.r0 ^= rhs.r0;
        self.r1 ^= rhs.r1;
        self.r2 ^= rhs.r2;
        self.r3 ^= rhs.r3;
    }
}

// ---------------------------------------------------------------------------
// Reduction mod p = x^128 + x^7 + x^2 + x + 1. Works on any target.
// ---------------------------------------------------------------------------

/// Fold the upper 128 bits (r2:r3) into the lower 128 bits (r0:r1) mod p.
/// x^128 ≡ x^7 + x^2 + x + 1, so U·x^128 ≡ U ^ (U<<1) ^ (U<<2) ^ (U<<7).
#[inline]
pub fn ghash_reduce(r0: u64, r1: u64, r2: u64, r3: u64) -> F128 {
    let s1_lo = r2 << 1;
    let s1_hi = (r3 << 1) | (r2 >> 63);
    let s2_lo = r2 << 2;
    let s2_hi = (r3 << 2) | (r2 >> 62);
    let s7_lo = r2 << 7;
    let s7_hi = (r3 << 7) | (r2 >> 57);

    let t_lo = r2 ^ s1_lo ^ s2_lo ^ s7_lo;
    let t_hi = r3 ^ s1_hi ^ s2_hi ^ s7_hi;

    // Bits of r3 that shifted past position 127 (top 7 bits, in 3 shifts).
    let ov = (r3 >> 63) ^ (r3 >> 62) ^ (r3 >> 57);
    let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

    F128 {
        lo: r0 ^ t_lo ^ corr,
        hi: r1 ^ t_hi,
    }
}

// ---------------------------------------------------------------------------
// aarch64 + AES: PMULL-based multiplication variants.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub mod aarch64;

// ---------------------------------------------------------------------------
// x86_64 + PCLMULQDQ: carry-less-multiply-based multiplication.
//
// `_mm_clmulepi64_si128(a, b, 0x00)` is the x86 analogue of ARM `vmull_p64`:
// a 64×64 carry-less mul of the low qwords into a 128-bit `__m128i` laid out
// as {lo = bits 0..63, hi = bits 64..127} — identical to NEON's uint64x2_t.
// So the variants below are direct ports of the `aarch64` module; only the
// primitive and lane-shuffle ops differ. The shared `ghash_reduce` is reused.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub mod x86_64;

// ---------------------------------------------------------------------------
// Software fallback: bit-by-bit clmul64. Slow but portable; also the reference
// the NEON path is checked against in tests.
// ---------------------------------------------------------------------------

#[path = "gf2_128/portable.rs"]
pub mod software;

#[inline]
fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: aes target feature is enabled at compile time.
        unsafe { ghash_mul_unreduced_aarch64(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: pclmulqdq target feature is enabled at compile time.
        unsafe { ghash_mul_unreduced_x86_64(a, b) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        ghash_mul_unreduced_software(a, b)
    }
}

#[inline]
fn ghash_square(a: F128) -> F128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: aes target feature is enabled at compile time.
        unsafe { ghash_square_aarch64(a) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: pclmulqdq target feature is enabled at compile time.
        unsafe { ghash_square_x86_64(a) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        ghash_square_software(a)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    use super::aarch64::{
        ghash_mul_binius as ghash_mul_binius_aarch64,
        ghash_mul_karatsuba as ghash_mul_karatsuba_aarch64,
        ghash_mul_karatsuba_barrett as ghash_mul_karatsuba_barrett_aarch64,
        ghash_mul_schoolbook as ghash_mul_schoolbook_aarch64, ghash_mul_vec2_neon,
    };
    #[cfg(feature = "mul-count")]
    use super::op_count::{MULS_PER_INV, measure};
    use super::software::ghash_mul as ghash_mul_software;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    use super::x86_64::{WideGhashX4, f128x4_loadu, f128x4_set, ghash_mul_x4};
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    use super::x86_64::{
        ghash_mul_binius as ghash_mul_binius_x86_64,
        ghash_mul_karatsuba as ghash_mul_karatsuba_x86_64,
        ghash_mul_karatsuba_barrett as ghash_mul_karatsuba_barrett_x86_64,
        ghash_mul_schoolbook as ghash_mul_schoolbook_x86_64,
        ghash_mul_unreduced_x86 as ghash_mul_unreduced_x86_64,
    };
    use super::*;

    use crate::test_rng::Rng;

    #[test]
    fn add_identities() {
        let mut rng = Rng::new(1);
        for _ in 0..64 {
            let a = rng.f128();
            assert_eq!(a + F128::ZERO, a);
            assert_eq!(a + a, F128::ZERO);
        }
    }

    /// `MULS_PER_INV` is what separates an algorithm's real
    /// multiplication count from the ones buried inside `inv`, so the whole
    /// attribution in `benches/verifier_mul_count.rs` rests on it. Pinned
    /// against the live `inv`, which is where it would silently drift.
    #[cfg(feature = "mul-count")]
    #[test]
    fn inv_mul_cost_is_stable() {
        let (_, s) = measure(|| F128::new(0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321).inv());
        assert_eq!(s.invs, 1);
        assert_eq!(
            s.native_muls, MULS_PER_INV,
            "inv's multiplication count moved; MULS_PER_INV is now wrong"
        );
        assert_eq!(s.muls_excluding_inv(), 0);
        assert_eq!(s.circuit_constraints(), 1, "an inversion is ONE constraint");
    }

    #[test]
    fn mul_identities() {
        let mut rng = Rng::new(2);
        for _ in 0..64 {
            let a = rng.f128();
            assert_eq!(a * F128::ZERO, F128::ZERO);
            assert_eq!(a * F128::ONE, a);
        }
    }

    #[test]
    fn mul_by_x_matches_mul_by_gen() {
        let mut rng = Rng::new(3);
        for _ in 0..256 {
            let a = rng.f128();
            assert_eq!(mul_by_x(a), a * F128::generator());
        }
    }

    #[test]
    fn deferred_reduction_matches_direct() {
        let mut rng = Rng::new(4);
        for _ in 0..64 {
            let a = rng.f128();
            let b = rng.f128();
            let direct = a * b;
            let deferred = a.mul_unreduced(b).reduce();
            assert_eq!(direct, deferred);
        }
    }

    #[test]
    fn deferred_xor_commutes_with_reduction() {
        // Σ aᵢ·bᵢ in F128 must equal reduce(XOR-sum of unreduced products).
        let mut rng = Rng::new(5);
        let n = 16;
        let pairs: Vec<(F128, F128)> = (0..n).map(|_| (rng.f128(), rng.f128())).collect();

        let direct: F128 = pairs.iter().fold(F128::ZERO, |acc, (a, b)| acc + *a * *b);

        let mut acc = F256Unreduced::ZERO;
        for (a, b) in &pairs {
            acc ^= a.mul_unreduced(*b);
        }
        assert_eq!(direct, acc.reduce());
    }

    #[test]
    fn inverse_roundtrip() {
        let mut rng = Rng::new(6);
        for _ in 0..16 {
            let a = rng.f128();
            if a.is_zero() {
                continue;
            }
            assert_eq!(a * a.inv(), F128::ONE);
        }
    }

    #[test]
    fn associativity_random() {
        let mut rng = Rng::new(7);
        for _ in 0..64 {
            let a = rng.f128();
            let b = rng.f128();
            let c = rng.f128();
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn mul_commutativity() {
        let mut rng = Rng::new(91);
        for _ in 0..256 {
            let a = rng.f128();
            let b = rng.f128();
            assert_eq!(a * b, b * a);
        }
    }

    #[test]
    fn ghash_reduction_smoking_gun() {
        // The defining identity of the GHASH polynomial:
        //   x · x^127 = x^128 = x^7 + x^2 + x + 1 = 0x87.
        // If the reduction constant 0x87 is wrong (e.g. 0x86, 0x07, byte-swapped),
        // this test fails immediately and pinpoints the bug.
        let x = F128::generator();
        let x_127 = F128 {
            lo: 0,
            hi: 1u64 << 63,
        };
        assert_eq!(x * x_127, F128 { lo: 0x87, hi: 0 }, "x · x^127");

        // x · x^63 = x^64 — crosses the lo/hi word boundary with no reduction.
        // Catches lo/hi swaps and off-by-one in the 64-bit word split.
        let x_63 = F128 {
            lo: 1u64 << 63,
            hi: 0,
        };
        assert_eq!(x * x_63, F128 { lo: 0, hi: 1 }, "x · x^63 = x^64");

        // x^64 · x^64 = x^128 = 0x87 — reaches the reduction through a different
        // multiplication path (high·high product).
        let x_64 = F128 { lo: 0, hi: 1 };
        assert_eq!(x_64 * x_64, F128 { lo: 0x87, hi: 0 }, "x^64 · x^64");

        // x · x = x^2 (no reduction).
        assert_eq!(x * x, F128 { lo: 4, hi: 0 }, "x^2");
    }

    #[test]
    fn high_bit_inputs_reduce_correctly() {
        // Verify mul still satisfies a^{-1} · a = 1 when both inputs have the
        // top bit (x^127) set — exercising the most overflow-prone code path
        // of `ghash_reduce`. The inverse test naturally lands here for random
        // inputs only by luck; this makes it deterministic.
        let high = F128 {
            lo: 0,
            hi: 1u64 << 63,
        };
        assert_eq!(high * high.inv(), F128::ONE);
        let almost_max = F128 {
            lo: u64::MAX,
            hi: u64::MAX,
        };
        assert_eq!(almost_max * almost_max.inv(), F128::ONE);
        let just_top = F128 {
            lo: 0,
            hi: u64::MAX,
        };
        assert_eq!(just_top * just_top.inv(), F128::ONE);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_mul_vec2_matches_scalar() {
        let mut rng = Rng::new(11);
        for _ in 0..128 {
            let a0 = rng.f128();
            let a1 = rng.f128();
            let b0 = rng.f128();
            let b1 = rng.f128();
            let expected = [a0 * b0, a1 * b1];
            let result = unsafe { ghash_mul_vec2_neon([a0, a1], [b0, b1]) };
            assert_eq!(result[0], expected[0], "lane 0");
            assert_eq!(result[1], expected[1], "lane 1");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn all_neon_variants_agree() {
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.f128();
            let b = rng.f128();
            let sw = ghash_mul_software(a, b);
            let sb = unsafe { ghash_mul_schoolbook_aarch64(a, b) };
            let ka = unsafe { ghash_mul_karatsuba_aarch64(a, b) };
            let kb = unsafe { ghash_mul_karatsuba_barrett_aarch64(a, b) };
            let bi = unsafe { ghash_mul_binius_aarch64(a, b) };
            assert_eq!(sw, sb);
            assert_eq!(sw, ka);
            assert_eq!(sw, kb);
            assert_eq!(sw, bi);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    #[test]
    fn all_x86_variants_agree() {
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.f128();
            let b = rng.f128();
            let sw = ghash_mul_software(a, b);
            let sb = unsafe { ghash_mul_schoolbook_x86_64(a, b) };
            let ka = unsafe { ghash_mul_karatsuba_x86_64(a, b) };
            let kb = unsafe { ghash_mul_karatsuba_barrett_x86_64(a, b) };
            let bi = unsafe { ghash_mul_binius_x86_64(a, b) };
            // Unreduced + deferred reduce must match the direct software product.
            let un = unsafe { ghash_mul_unreduced_x86_64(a, b) }.reduce();
            assert_eq!(sw, sb, "schoolbook");
            assert_eq!(sw, ka, "karatsuba");
            assert_eq!(sw, kb, "karatsuba_barrett");
            assert_eq!(sw, bi, "binius");
            assert_eq!(sw, un, "unreduced");
        }
    }

    /// The 4-lane VPCLMULQDQ multiply must agree, lane for lane, with the
    /// canonical scalar `F128::mul` — the clmul `0x87` reduction reaches the
    /// same field element by a different route, so verify, don't assume.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn ghash_mul_x4_matches_scalar() {
        let mut rng = Rng::new(0x4A4_C0DE);
        for _ in 0..256 {
            let xs = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
            let ys = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
            // SAFETY: vpclmulqdq+avx512f enabled at compile time (cfg gate).
            let got: [F128; 4] = unsafe {
                let x = _mm512_loadu_si512(xs.as_ptr() as *const __m512i);
                let y = _mm512_loadu_si512(ys.as_ptr() as *const __m512i);
                let r = ghash_mul_x4(x, y);
                let mut out = [F128::ZERO; 4];
                _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, r);
                out
            };
            for lane in 0..4 {
                assert_eq!(
                    got[lane],
                    xs[lane] * ys[lane],
                    "lane {lane}: x4 != scalar mul"
                );
            }
        }
    }

    /// The 4-lane deferred-reduction accumulator must equal the scalar
    /// XOR-of-`mul_unreduced` it replaces, both before and after `reduce()`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn wide_ghash_x4_matches_scalar_deferred() {
        let mut rng = Rng::new(0xDEF_E44);
        for _ in 0..128 {
            // SAFETY: vpclmulqdq+avx512f+sse4.1 enabled at compile time.
            let mut wide = unsafe { WideGhashX4::zero() };
            let mut scalar = F256Unreduced::ZERO;
            for _ in 0..5 {
                let xs = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
                let ys = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
                // xs via contiguous load, ys via scalar set — exercises both.
                unsafe {
                    let xv = f128x4_loadu(xs.as_ptr());
                    let yv = f128x4_set(ys[0], ys[1], ys[2], ys[3]);
                    wide.mul_acc(xv, yv);
                }
                for i in 0..4 {
                    scalar ^= xs[i].mul_unreduced(ys[i]);
                }
            }
            let folded = unsafe { wide.fold() };
            assert_eq!(folded, scalar, "wide fold != scalar deferred accumulator");
            assert_eq!(folded.reduce(), scalar.reduce(), "reduced values differ");
        }
    }

    #[test]
    fn square_matches_self_mul() {
        let mut rng = Rng::new(0x5147);
        for _ in 0..1000 {
            let a = rng.f128();
            assert_eq!(a.square(), a * a);
        }
        assert_eq!(F128::ZERO.square(), F128::ZERO);
        assert_eq!(F128::ONE.square(), F128::ONE);
    }
}
