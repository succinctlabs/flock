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

use core::ops::{Add, AddAssign, BitXor, BitXorAssign, Mul, MulAssign};

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

impl Mul for F128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            // SAFETY: aes target feature is enabled at compile time.
            unsafe { aarch64::ghash_mul_binius(self, rhs) }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        {
            // SAFETY: pclmulqdq target feature is enabled at compile time.
            // On Zen4, karatsuba+barrett is ~17% faster in throughput (the
            // dominant mode for the bulk parallel F128 work) than binius, which
            // only wins the latency microbench. (M-series picked binius.)
            unsafe { x86_64::ghash_mul_karatsuba_barrett(self, rhs) }
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        {
            software::ghash_mul(self, rhs)
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
        unsafe { aarch64::ghash_mul_unreduced_neon(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: pclmulqdq target feature is enabled at compile time.
        unsafe { x86_64::ghash_mul_unreduced_x86(a, b) }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    )))]
    {
        software::ghash_mul_unreduced(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn next_f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    #[test]
    fn add_identities() {
        let mut rng = Rng::new(1);
        for _ in 0..64 {
            let a = rng.next_f128();
            assert_eq!(a + F128::ZERO, a);
            assert_eq!(a + a, F128::ZERO);
        }
    }

    #[test]
    fn mul_identities() {
        let mut rng = Rng::new(2);
        for _ in 0..64 {
            let a = rng.next_f128();
            assert_eq!(a * F128::ZERO, F128::ZERO);
            assert_eq!(a * F128::ONE, a);
        }
    }

    #[test]
    fn mul_by_x_matches_mul_by_gen() {
        let mut rng = Rng::new(3);
        for _ in 0..256 {
            let a = rng.next_f128();
            assert_eq!(mul_by_x(a), a * F128::generator());
        }
    }

    #[test]
    fn deferred_reduction_matches_direct() {
        let mut rng = Rng::new(4);
        for _ in 0..64 {
            let a = rng.next_f128();
            let b = rng.next_f128();
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
        let pairs: Vec<(F128, F128)> = (0..n).map(|_| (rng.next_f128(), rng.next_f128())).collect();

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
            let a = rng.next_f128();
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
            let a = rng.next_f128();
            let b = rng.next_f128();
            let c = rng.next_f128();
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn mul_commutativity() {
        let mut rng = Rng::new(91);
        for _ in 0..256 {
            let a = rng.next_f128();
            let b = rng.next_f128();
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
            let a0 = rng.next_f128();
            let a1 = rng.next_f128();
            let b0 = rng.next_f128();
            let b1 = rng.next_f128();
            let expected = [a0 * b0, a1 * b1];
            let result = unsafe { aarch64::ghash_mul_vec2_neon([a0, a1], [b0, b1]) };
            assert_eq!(result[0], expected[0], "lane 0");
            assert_eq!(result[1], expected[1], "lane 1");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn all_neon_variants_agree() {
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let sw = software::ghash_mul(a, b);
            let sb = unsafe { aarch64::ghash_mul_schoolbook(a, b) };
            let ka = unsafe { aarch64::ghash_mul_karatsuba(a, b) };
            let kb = unsafe { aarch64::ghash_mul_karatsuba_barrett(a, b) };
            let bi = unsafe { aarch64::ghash_mul_binius(a, b) };
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
            let a = rng.next_f128();
            let b = rng.next_f128();
            let sw = software::ghash_mul(a, b);
            let sb = unsafe { x86_64::ghash_mul_schoolbook(a, b) };
            let ka = unsafe { x86_64::ghash_mul_karatsuba(a, b) };
            let kb = unsafe { x86_64::ghash_mul_karatsuba_barrett(a, b) };
            let bi = unsafe { x86_64::ghash_mul_binius(a, b) };
            // Unreduced + deferred reduce must match the direct software product.
            let un = unsafe { x86_64::ghash_mul_unreduced_x86(a, b) }.reduce();
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
        use core::arch::x86_64::*;
        let mut rng = Rng::new(0x4A4_C0DE);
        for _ in 0..256 {
            let xs = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            let ys = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            // SAFETY: vpclmulqdq+avx512f enabled at compile time (cfg gate).
            let got: [F128; 4] = unsafe {
                let x = _mm512_loadu_si512(xs.as_ptr() as *const __m512i);
                let y = _mm512_loadu_si512(ys.as_ptr() as *const __m512i);
                let r = x86_64::ghash_mul_x4(x, y);
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
            let mut wide = unsafe { x86_64::WideGhashX4::zero() };
            let mut scalar = F256Unreduced::ZERO;
            for _ in 0..5 {
                let xs = [
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                ];
                let ys = [
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                ];
                // xs via contiguous load, ys via scalar set — exercises both.
                unsafe {
                    let xv = x86_64::f128x4_loadu(xs.as_ptr());
                    let yv = x86_64::f128x4_set(ys[0], ys[1], ys[2], ys[3]);
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
}
