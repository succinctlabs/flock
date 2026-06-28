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
            // SAFETY: pclmulqdq + sse4.1 target features are enabled at compile time.
            unsafe { x86_64::ghash_mul_binius(self, rhs) }
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
pub mod aarch64 {
    use super::{F128, F256Unreduced, ghash_reduce};
    use core::arch::aarch64::*;
    use core::mem::transmute;

    /// 64×64 carry-less product, returned as a 128-bit vector.
    ///
    /// # Safety
    /// Caller must ensure the `aes` target feature is enabled (statically
    /// satisfied here because every caller is itself `#[target_feature(enable = "aes")]`).
    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn pmull(a: u64, b: u64) -> uint64x2_t {
        let prod = vmull_p64(a, b);
        // SAFETY: u128 and uint64x2_t are both 128-bit, 16-byte-aligned values;
        // transmute is a bit-level reinterpret with no UB.
        unsafe { transmute::<u128, uint64x2_t>(prod) }
    }

    /// Schoolbook 4 PMULL — fully independent products, then scalar reduction.
    ///
    /// # Safety
    /// Requires the `aes` target feature (compiles to PMULL); only call where
    /// `aes` is statically enabled or has been runtime-detected.
    #[target_feature(enable = "aes")]
    pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
        // SAFETY: function carries the aes target feature; helper calls below
        // require that and nothing else.
        unsafe {
            let p_ll = pmull(a.lo, b.lo);
            let p_lh = pmull(a.lo, b.hi);
            let p_hl = pmull(a.hi, b.lo);
            let p_hh = pmull(a.hi, b.hi);

            let ll_lo = vgetq_lane_u64::<0>(p_ll);
            let ll_hi = vgetq_lane_u64::<1>(p_ll);
            let hh_lo = vgetq_lane_u64::<0>(p_hh);
            let hh_hi = vgetq_lane_u64::<1>(p_hh);
            let cross = veorq_u64(p_lh, p_hl);
            let cr_lo = vgetq_lane_u64::<0>(cross);
            let cr_hi = vgetq_lane_u64::<1>(cross);

            ghash_reduce(ll_lo, ll_hi ^ cr_lo, hh_lo ^ cr_hi, hh_hi)
        }
    }

    /// Karatsuba 3 PMULL — middle term depends on XOR of inputs (one stall on
    /// CPUs with 2 PMULL units).
    ///
    /// # Safety
    /// Requires the `aes` target feature (compiles to PMULL); only call where
    /// `aes` is statically enabled or has been runtime-detected.
    #[target_feature(enable = "aes")]
    pub unsafe fn ghash_mul_karatsuba(a: F128, b: F128) -> F128 {
        // SAFETY: function carries the aes target feature.
        unsafe {
            let p0 = pmull(a.lo, b.lo);
            let p1 = pmull(a.hi, b.hi);
            let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);

            let p0_lo = vgetq_lane_u64::<0>(p0);
            let p0_hi = vgetq_lane_u64::<1>(p0);
            let p1_lo = vgetq_lane_u64::<0>(p1);
            let p1_hi = vgetq_lane_u64::<1>(p1);
            let pm_lo = vgetq_lane_u64::<0>(pm);
            let pm_hi = vgetq_lane_u64::<1>(pm);

            let cross_lo = pm_lo ^ p0_lo ^ p1_lo;
            let cross_hi = pm_hi ^ p0_hi ^ p1_hi;

            ghash_reduce(p0_lo, p0_hi ^ cross_lo, p1_lo ^ cross_hi, p1_hi)
        }
    }

    /// Karatsuba 3 PMULL + Barrett 2 PMULL = 5 PMULL total.
    /// `r_hi = hi_hi · 0x87` depends only on `d2`, not `d1`, so it can issue
    /// in parallel with the cross-term computation.
    ///
    /// # Safety
    /// Requires the `aes` target feature (compiles to PMULL); only call where
    /// `aes` is statically enabled or has been runtime-detected.
    #[target_feature(enable = "aes")]
    pub unsafe fn ghash_mul_karatsuba_barrett(a: F128, b: F128) -> F128 {
        // SAFETY: function carries the aes target feature.
        unsafe {
            let d0 = pmull(a.lo, b.lo);
            let d2 = pmull(a.hi, b.hi);
            let dm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
            let d1 = veorq_u64(veorq_u64(dm, d0), d2);

            let d0_lo = vgetq_lane_u64::<0>(d0);
            let d0_hi = vgetq_lane_u64::<1>(d0);
            let d1_lo = vgetq_lane_u64::<0>(d1);
            let d1_hi = vgetq_lane_u64::<1>(d1);
            let d2_lo = vgetq_lane_u64::<0>(d2);
            let d2_hi = vgetq_lane_u64::<1>(d2);

            let lo_lo = d0_lo;
            let lo_hi = d0_hi ^ d1_lo;
            let hi_lo = d2_lo ^ d1_hi;
            let hi_hi = d2_hi;

            let r_hi = pmull(hi_hi, 0x87);
            let r_lo = pmull(hi_lo, 0x87);

            let r_lo_lo = vgetq_lane_u64::<0>(r_lo);
            let r_lo_hi = vgetq_lane_u64::<1>(r_lo);
            let r_hi_lo = vgetq_lane_u64::<0>(r_hi);
            let r_hi_hi = vgetq_lane_u64::<1>(r_hi);

            // hi_hi · 0x87 has degree ≤ 70, so r_hi_hi has at most 7 bits.
            let ov = r_hi_hi;
            let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

            F128 {
                lo: lo_lo ^ r_lo_lo ^ corr,
                hi: lo_hi ^ r_lo_hi ^ r_hi_lo,
            }
        }
    }

    /// Binius-style: schoolbook 4 PMULL + recursive 2-stage reduction (2 PMULL).
    /// Each stage keeps the intermediate ≤128 bits — no separate 7-bit overflow
    /// term required. Total 6 PMULL but fewer scalar shifts in the dep chain.
    /// Memory recorded this as the best of the four variants on M-series.
    ///
    /// # Safety
    /// Requires the `aes` target feature (compiles to PMULL); only call where
    /// `aes` is statically enabled or has been runtime-detected.
    #[target_feature(enable = "aes")]
    pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
        // SAFETY: function carries the aes target feature.
        unsafe {
            let zero = vdupq_n_u64(0);

            let t0 = pmull(a.lo, b.lo);
            let t1a = pmull(a.lo, b.hi);
            let t1b = pmull(a.hi, b.lo);
            let t2 = pmull(a.hi, b.hi);
            let mut t1 = veorq_u64(t1a, t1b);

            // First reduce: t1 = t1 + x^64 · t2 (mod p).
            // vextq_u64::<1>(zero, t2) = {0, t2.lo} — places t2.lo into t1.hi.
            let t2_shifted = vextq_u64::<1>(zero, t2);
            t1 = veorq_u64(t1, t2_shifted);
            let t2_hi_s = vgetq_lane_u64::<1>(t2);
            let t2_red = pmull(t2_hi_s, 0x87);
            t1 = veorq_u64(t1, t2_red);

            // Second reduce: t0 = t0 + x^64 · t1 (mod p).
            let mut t0 = t0;
            let t1_shifted = vextq_u64::<1>(zero, t1);
            t0 = veorq_u64(t0, t1_shifted);
            let t1_hi_s = vgetq_lane_u64::<1>(t1);
            let t1_red = pmull(t1_hi_s, 0x87);
            t0 = veorq_u64(t0, t1_red);

            F128 {
                lo: vgetq_lane_u64::<0>(t0),
                hi: vgetq_lane_u64::<1>(t0),
            }
        }
    }

    /// Batch multiply 2× F128 in parallel.
    ///
    /// Strategy: 8 schoolbook PMULLs (4 per mul, all independent), repack the
    /// four unreduced 64-bit words `(r0, r1, r2, r3)` of each product into
    /// lane-paired `uint64x2_t` registers, then run the GHASH shift-XOR
    /// reduction once with each NEON op handling both muls' lanes. Trades
    /// the binius variant's 4 reduction-stage PMULLs (2 per mul × 2 muls)
    /// for a vectorised XOR-based reduction. Worth it because PMULL is the
    /// scarce resource on M-class (2 units, 1/cycle each).
    ///
    /// # Safety
    /// Requires the `aes` target feature (compiles to PMULL); only call where
    /// `aes` is statically enabled or has been runtime-detected.
    #[target_feature(enable = "aes")]
    pub unsafe fn ghash_mul_vec2_neon(a: [F128; 2], b: [F128; 2]) -> [F128; 2] {
        // SAFETY: function carries the aes target feature; pmull requires it.
        unsafe {
            // 8 independent schoolbook PMULLs.
            let p0_ll = pmull(a[0].lo, b[0].lo);
            let p0_lh = pmull(a[0].lo, b[0].hi);
            let p0_hl = pmull(a[0].hi, b[0].lo);
            let p0_hh = pmull(a[0].hi, b[0].hi);
            let p1_ll = pmull(a[1].lo, b[1].lo);
            let p1_lh = pmull(a[1].lo, b[1].hi);
            let p1_hl = pmull(a[1].hi, b[1].lo);
            let p1_hh = pmull(a[1].hi, b[1].hi);

            // Per-mul cross terms (lh + hl).
            let c0 = veorq_u64(p0_lh, p0_hl);
            let c1 = veorq_u64(p1_lh, p1_hl);

            // Lane-paired (mul0, mul1) layout for each word position.
            //   r0 = ll_lo
            //   r1 = ll_hi ^ cross_lo
            //   r2 = hh_lo ^ cross_hi
            //   r3 = hh_hi
            let r0 = vzip1q_u64(p0_ll, p1_ll);
            let ll_hi = vzip2q_u64(p0_ll, p1_ll);
            let c_lo = vzip1q_u64(c0, c1);
            let r1 = veorq_u64(ll_hi, c_lo);
            let hh_lo = vzip1q_u64(p0_hh, p1_hh);
            let c_hi = vzip2q_u64(c0, c1);
            let r2 = veorq_u64(hh_lo, c_hi);
            let r3 = vzip2q_u64(p0_hh, p1_hh);

            // Vectorised GHASH reduction: fold (r2, r3) into (r0, r1) mod p,
            // where p = x^128 + x^7 + x^2 + x + 1. r(x) = x^7 + x^2 + x + 1.
            // Each shift produces (lo_part, overflow); the overflow goes into
            // the next-higher word.
            let s1_lo = vshlq_n_u64::<1>(r2);
            let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
            let s2_lo = vshlq_n_u64::<2>(r2);
            let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
            let s7_lo = vshlq_n_u64::<7>(r2);
            let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));

            let t_lo = veorq_u64(veorq_u64(r2, s1_lo), veorq_u64(s2_lo, s7_lo));
            let t_hi = veorq_u64(veorq_u64(r3, s1_hi), veorq_u64(s2_hi, s7_hi));

            // Bits of r3 that overflowed past position 127 in the three shifts.
            let ov = veorq_u64(
                veorq_u64(vshrq_n_u64::<63>(r3), vshrq_n_u64::<62>(r3)),
                vshrq_n_u64::<57>(r3),
            );
            let corr = veorq_u64(
                veorq_u64(ov, vshlq_n_u64::<1>(ov)),
                veorq_u64(vshlq_n_u64::<2>(ov), vshlq_n_u64::<7>(ov)),
            );

            let final_lo = veorq_u64(veorq_u64(r0, t_lo), corr);
            let final_hi = veorq_u64(r1, t_hi);

            // Unpack: lane 0 → mul0, lane 1 → mul1.
            [
                F128 {
                    lo: vgetq_lane_u64::<0>(final_lo),
                    hi: vgetq_lane_u64::<0>(final_hi),
                },
                F128 {
                    lo: vgetq_lane_u64::<1>(final_lo),
                    hi: vgetq_lane_u64::<1>(final_hi),
                },
            ]
        }
    }

    /// Full 256-bit carry-less product `a · b`, no mod-p reduction. The standard
    /// middle-cross fold is baked in: r1 = ll_hi ^ cross_lo, r2 = hh_lo ^ cross_hi.
    ///
    /// # Safety
    /// Requires the `aes` target feature (compiles to PMULL); only call where
    /// `aes` is statically enabled or has been runtime-detected.
    #[target_feature(enable = "aes")]
    pub unsafe fn ghash_mul_unreduced_neon(a: F128, b: F128) -> F256Unreduced {
        // SAFETY: function carries the aes target feature.
        unsafe {
            let p_ll = pmull(a.lo, b.lo);
            let p_lh = pmull(a.lo, b.hi);
            let p_hl = pmull(a.hi, b.lo);
            let p_hh = pmull(a.hi, b.hi);

            let ll_lo = vgetq_lane_u64::<0>(p_ll);
            let ll_hi = vgetq_lane_u64::<1>(p_ll);
            let hh_lo = vgetq_lane_u64::<0>(p_hh);
            let hh_hi = vgetq_lane_u64::<1>(p_hh);
            let cross = veorq_u64(p_lh, p_hl);
            let cr_lo = vgetq_lane_u64::<0>(cross);
            let cr_hi = vgetq_lane_u64::<1>(cross);

            F256Unreduced {
                r0: ll_lo,
                r1: ll_hi ^ cr_lo,
                r2: hh_lo ^ cr_hi,
                r3: hh_hi,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// x86_64 + PCLMULQDQ: carry-less-multiply path, the direct sibling of the
// aarch64 PMULL code above. `_mm_clmulepi64_si128` is the x86 equivalent of
// `vmull_p64`; the binius schoolbook-4 + 2-stage reduction is identical, only
// the intrinsics differ. Gated on `pclmulqdq` (enabled here via
// `-C target-cpu=native` on Granite Rapids, which also implies `sse4.1`).
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub mod x86_64 {
    use super::{F128, F256Unreduced, ghash_reduce};
    use core::arch::x86_64::*;

    /// Load an `F128` into a 128-bit register: lane 0 = `lo`, lane 1 = `hi`.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load(a: F128) -> __m128i {
        // SAFETY: sse2 is baseline on x86_64.
        unsafe { _mm_set_epi64x(a.hi as i64, a.lo as i64) }
    }

    // imm8 selector for `_mm_clmulepi64_si128`: bit0 picks the qword of the
    // first operand (0 = low / 1 = high), bit4 the second operand. So
    // 0x00 = lo·lo, 0x10 = lo·hi, 0x01 = hi·lo, 0x11 = hi·hi.

    /// Full 256-bit carry-less product `a · b`, no mod-p reduction. The standard
    /// middle-cross fold is baked in (mirrors `aarch64::ghash_mul_unreduced_neon`
    /// and `software::ghash_mul_unreduced` exactly).
    ///
    /// # Safety
    /// Requires `pclmulqdq` + `sse4.1` (statically enabled here).
    #[target_feature(enable = "sse4.1,pclmulqdq")]
    pub unsafe fn ghash_mul_unreduced_clmul(a: F128, b: F128) -> F256Unreduced {
        // SAFETY: function carries the required target features.
        unsafe {
            let va = load(a);
            let vb = load(b);
            let p_ll = _mm_clmulepi64_si128::<0x00>(va, vb); // a.lo · b.lo
            let p_lh = _mm_clmulepi64_si128::<0x10>(va, vb); // a.lo · b.hi
            let p_hl = _mm_clmulepi64_si128::<0x01>(va, vb); // a.hi · b.lo
            let p_hh = _mm_clmulepi64_si128::<0x11>(va, vb); // a.hi · b.hi

            let ll_lo = _mm_extract_epi64::<0>(p_ll) as u64;
            let ll_hi = _mm_extract_epi64::<1>(p_ll) as u64;
            let hh_lo = _mm_extract_epi64::<0>(p_hh) as u64;
            let hh_hi = _mm_extract_epi64::<1>(p_hh) as u64;
            let cross = _mm_xor_si128(p_lh, p_hl);
            let cr_lo = _mm_extract_epi64::<0>(cross) as u64;
            let cr_hi = _mm_extract_epi64::<1>(cross) as u64;

            F256Unreduced {
                r0: ll_lo,
                r1: ll_hi ^ cr_lo,
                r2: hh_lo ^ cr_hi,
                r3: hh_hi,
            }
        }
    }

    /// Schoolbook 4 CLMUL + scalar `ghash_reduce`. Independent reference path
    /// for the cross-check test.
    ///
    /// # Safety
    /// Requires `pclmulqdq` + `sse4.1` (statically enabled here).
    #[target_feature(enable = "sse4.1,pclmulqdq")]
    pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
        // SAFETY: function carries the required target features.
        unsafe {
            let u = ghash_mul_unreduced_clmul(a, b);
            ghash_reduce(u.r0, u.r1, u.r2, u.r3)
        }
    }

    /// Binius-style: schoolbook 4 CLMUL + recursive 2-stage reduction (2 CLMUL),
    /// staying ≤128 bits between stages. Direct translation of
    /// `aarch64::ghash_mul_binius`; this is the default `Mul` path on x86.
    /// `_mm_slli_si128::<8>(v)` produces `{0, v.lo}` (the x86 analogue of
    /// `vextq_u64::<1>(zero, v)`); `_mm_clmulepi64_si128::<0x01>(v, c87)` is
    /// `clmul(v.hi, 0x87)`.
    ///
    /// # Safety
    /// Requires `pclmulqdq` + `sse4.1` (statically enabled here).
    #[target_feature(enable = "sse4.1,pclmulqdq")]
    pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
        // SAFETY: function carries the required target features.
        unsafe {
            let va = load(a);
            let vb = load(b);
            let c87 = _mm_set_epi64x(0, 0x87);

            let t0 = _mm_clmulepi64_si128::<0x00>(va, vb);
            let t1a = _mm_clmulepi64_si128::<0x10>(va, vb);
            let t1b = _mm_clmulepi64_si128::<0x01>(va, vb);
            let t2 = _mm_clmulepi64_si128::<0x11>(va, vb);
            let mut t1 = _mm_xor_si128(t1a, t1b);

            // First reduce: t1 = t1 + x^64 · t2 (mod p).
            let t2_shifted = _mm_slli_si128::<8>(t2); // {0, t2.lo}
            t1 = _mm_xor_si128(t1, t2_shifted);
            let t2_red = _mm_clmulepi64_si128::<0x01>(t2, c87); // clmul(t2.hi, 0x87)
            t1 = _mm_xor_si128(t1, t2_red);

            // Second reduce: t0 = t0 + x^64 · t1 (mod p).
            let mut t0 = t0;
            let t1_shifted = _mm_slli_si128::<8>(t1);
            t0 = _mm_xor_si128(t0, t1_shifted);
            let t1_red = _mm_clmulepi64_si128::<0x01>(t1, c87); // clmul(t1.hi, 0x87)
            t0 = _mm_xor_si128(t0, t1_red);

            F128 {
                lo: _mm_extract_epi64::<0>(t0) as u64,
                hi: _mm_extract_epi64::<1>(t0) as u64,
            }
        }
    }

    // -----------------------------------------------------------------------
    // 4-wide (AVX-512 VPCLMULQDQ): the same binius schoolbook-4 + 2-stage
    // reduction, applied lane-wise across the four 128-bit lanes of a ZMM.
    // `_mm512_clmulepi64_epi128` is the 4-lane VPCLMULQDQ; `_mm512_bslli_epi128`
    // is the per-128-lane analogue of `_mm_slli_si128`. Used by the interleaved
    // NTT to multiply a run of F128 by a shared (broadcast) twiddle.
    // -----------------------------------------------------------------------

    /// 4 independent GF(2¹²⁸) products in registers (ZMM in, ZMM out): each
    /// 128-bit lane `i` holds `a_i · b_i`. Internal hot kernel for the NTT.
    ///
    /// # Safety
    /// Requires avx512f + avx512bw + vpclmulqdq (statically enabled under
    /// `-C target-cpu=native` on Granite Rapids).
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
    pub unsafe fn ghash_mul_zmm(va: __m512i, vb: __m512i) -> __m512i {
        // SAFETY: function carries the required target features.
        unsafe {
            let c87 = _mm512_set1_epi64(0x87);
            let t0 = _mm512_clmulepi64_epi128::<0x00>(va, vb);
            let t1a = _mm512_clmulepi64_epi128::<0x10>(va, vb);
            let t1b = _mm512_clmulepi64_epi128::<0x01>(va, vb);
            let t2 = _mm512_clmulepi64_epi128::<0x11>(va, vb);
            let mut t1 = _mm512_xor_si512(t1a, t1b);

            // First reduce: t1 += x^64 · t2 (mod p), per lane.
            t1 = _mm512_xor_si512(t1, _mm512_bslli_epi128::<8>(t2));
            t1 = _mm512_xor_si512(t1, _mm512_clmulepi64_epi128::<0x01>(t2, c87));

            // Second reduce: t0 += x^64 · t1 (mod p), per lane.
            let mut t0 = t0;
            t0 = _mm512_xor_si512(t0, _mm512_bslli_epi128::<8>(t1));
            t0 = _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, c87));
            t0
        }
    }

    /// Array convenience wrapper around [`ghash_mul_zmm`] for tests / non-hot
    /// callers. `[F128; 4]` is 64 contiguous bytes = one ZMM.
    ///
    /// # Safety
    /// Requires avx512f + avx512bw + vpclmulqdq (statically enabled here).
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    #[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
    pub unsafe fn ghash_mul_vec4(a: [F128; 4], b: [F128; 4]) -> [F128; 4] {
        // SAFETY: function carries the required target features; [F128;4] is
        // 64-byte contiguous, matching a 512-bit load/store.
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr() as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr() as *const __m512i);
            let r = ghash_mul_zmm(va, vb);
            let mut out = [F128::ZERO; 4];
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, r);
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Software fallback: bit-by-bit clmul64. Slow but portable; also the reference
// the NEON path is checked against in tests.
// ---------------------------------------------------------------------------

pub mod software {
    use super::{F128, F256Unreduced, ghash_reduce};

    /// 64×64 carry-less product into 128 bits (lo, hi).
    pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
        let mut lo: u64 = 0;
        let mut hi: u64 = 0;
        let mut i = 0;
        while i < 64 {
            if (a >> i) & 1 != 0 {
                lo ^= b << i;
                if i != 0 {
                    hi ^= b >> (64 - i);
                }
            }
            i += 1;
        }
        (lo, hi)
    }

    pub fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
        let (ll_lo, ll_hi) = clmul64(a.lo, b.lo);
        let (lh_lo, lh_hi) = clmul64(a.lo, b.hi);
        let (hl_lo, hl_hi) = clmul64(a.hi, b.lo);
        let (hh_lo, hh_hi) = clmul64(a.hi, b.hi);
        let cr_lo = lh_lo ^ hl_lo;
        let cr_hi = lh_hi ^ hl_hi;
        F256Unreduced {
            r0: ll_lo,
            r1: ll_hi ^ cr_lo,
            r2: hh_lo ^ cr_hi,
            r3: hh_hi,
        }
    }

    pub fn ghash_mul(a: F128, b: F128) -> F128 {
        let u = ghash_mul_unreduced(a, b);
        ghash_reduce(u.r0, u.r1, u.r2, u.r3)
    }
}

#[inline]
fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: aes target feature is enabled at compile time.
        unsafe { aarch64::ghash_mul_unreduced_neon(a, b) }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: pclmulqdq + sse4.1 target features are enabled at compile time.
        unsafe { x86_64::ghash_mul_unreduced_clmul(a, b) }
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
        // The PCLMULQDQ paths must agree bit-for-bit with the scalar software
        // reference over random inputs — the correctness oracle for the port.
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let sw = software::ghash_mul(a, b);
            let sb = unsafe { x86_64::ghash_mul_schoolbook(a, b) };
            let bi = unsafe { x86_64::ghash_mul_binius(a, b) };
            let unred = unsafe { x86_64::ghash_mul_unreduced_clmul(a, b) }.reduce();
            assert_eq!(sw, sb, "x86 schoolbook");
            assert_eq!(sw, bi, "x86 binius");
            assert_eq!(sw, unred, "x86 unreduced+reduce");
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    #[test]
    fn x86_mul_vec4_matches_scalar() {
        // The AVX-512 4-wide kernel must agree lane-for-lane with the scalar
        // reference — the oracle for the NTT's wide multiply.
        let mut rng = Rng::new(99);
        for _ in 0..128 {
            let a = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            let b = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            let r = unsafe { x86_64::ghash_mul_vec4(a, b) };
            for i in 0..4 {
                assert_eq!(r[i], software::ghash_mul(a[i], b[i]), "lane {i}");
            }
        }
    }
}
