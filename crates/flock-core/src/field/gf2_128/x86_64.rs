use super::{F128, F256Unreduced, ghash_reduce};
use core::arch::x86_64::*;

/// 64×64 carry-less product, returned as a 128-bit vector {lo, hi}.
///
/// # Safety
/// Caller must ensure `pclmulqdq` (and `sse4.1` for the lane extracts in
/// callers) is enabled — statically satisfied since every caller is itself
/// `#[target_feature(enable = "pclmulqdq,sse4.1")]`.
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn pmull(a: u64, b: u64) -> __m128i {
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);
    // IMM8 = 0x00: low qword of a × low qword of b.
    _mm_clmulepi64_si128::<0x00>(va, vb)
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane0(v: __m128i) -> u64 {
    _mm_extract_epi64::<0>(v) as u64
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane1(v: __m128i) -> u64 {
    _mm_extract_epi64::<1>(v) as u64
}

/// Schoolbook 4 CLMUL — fully independent products, then scalar reduction.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        ghash_reduce(
            lane0(p_ll),
            lane1(p_ll) ^ cr_lo,
            lane0(p_hh) ^ cr_hi,
            lane1(p_hh),
        )
    }
}

/// Binius-style: schoolbook 4 CLMUL + recursive 2-stage reduction (2 CLMUL).
/// Direct port of `aarch64::ghash_mul_binius`. `vextq_u64::<1>(zero, t)`
/// (= {0, t.lo}) becomes `_mm_slli_si128::<8>(t)` — an 8-byte left shift
/// that moves the low qword into the high lane and zeroes the low lane.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let t0 = pmull(a.lo, b.lo);
        let t1a = pmull(a.lo, b.hi);
        let t1b = pmull(a.hi, b.lo);
        let t2 = pmull(a.hi, b.hi);
        let mut t1 = _mm_xor_si128(t1a, t1b);

        // First reduce: t1 = t1 + x^64 · t2 (mod p).
        let t2_shifted = _mm_slli_si128::<8>(t2); // {0, t2.lo}
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = pmull(lane1(t2), 0x87);
        t1 = _mm_xor_si128(t1, t2_red);

        // Second reduce: t0 = t0 + x^64 · t1 (mod p).
        let t1_shifted = _mm_slli_si128::<8>(t1); // {0, t1.lo}
        let mut t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = pmull(lane1(t1), 0x87);
        t0 = _mm_xor_si128(t0, t1_red);

        F128 {
            lo: lane0(t0),
            hi: lane1(t0),
        }
    }
}

/// Karatsuba 3 CLMUL — middle term depends on XOR of inputs. Port of
/// `aarch64::ghash_mul_karatsuba`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);

        let p0_lo = lane0(p0);
        let p0_hi = lane1(p0);
        let p1_lo = lane0(p1);
        let p1_hi = lane1(p1);
        let pm_lo = lane0(pm);
        let pm_hi = lane1(pm);

        let cross_lo = pm_lo ^ p0_lo ^ p1_lo;
        let cross_hi = pm_hi ^ p0_hi ^ p1_hi;

        ghash_reduce(p0_lo, p0_hi ^ cross_lo, p1_lo ^ cross_hi, p1_hi)
    }
}

/// Karatsuba 3 CLMUL + Barrett 2 CLMUL = 5 CLMUL total. Port of
/// `aarch64::ghash_mul_karatsuba_barrett`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_barrett(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the required target features.
    unsafe {
        let d0 = pmull(a.lo, b.lo);
        let d2 = pmull(a.hi, b.hi);
        let dm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        let d1 = _mm_xor_si128(_mm_xor_si128(dm, d0), d2);

        let d0_lo = lane0(d0);
        let d0_hi = lane1(d0);
        let d1_lo = lane0(d1);
        let d1_hi = lane1(d1);
        let d2_lo = lane0(d2);
        let d2_hi = lane1(d2);

        let lo_lo = d0_lo;
        let lo_hi = d0_hi ^ d1_lo;
        let hi_lo = d2_lo ^ d1_hi;
        let hi_hi = d2_hi;

        let r_hi = pmull(hi_hi, 0x87);
        let r_lo = pmull(hi_lo, 0x87);

        let r_lo_lo = lane0(r_lo);
        let r_lo_hi = lane1(r_lo);
        let r_hi_lo = lane0(r_hi);
        let r_hi_hi = lane1(r_hi);

        // hi_hi · 0x87 has degree ≤ 70, so r_hi_hi has at most 7 bits.
        let ov = r_hi_hi;
        let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

        F128 {
            lo: lo_lo ^ r_lo_lo ^ corr,
            hi: lo_hi ^ r_lo_hi ^ r_hi_lo,
        }
    }
}

/// 256-bit unreduced schoolbook product, for XOR-accumulation then one
/// deferred `reduce()`. Port of `aarch64::ghash_mul_unreduced_neon`.
///
/// # Safety
/// Requires `pclmulqdq` and `sse4.1`, as declared by the target-feature
/// attribute.
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_unreduced_x86(a: F128, b: F128) -> F256Unreduced {
    // SAFETY: function carries the required target features.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        F256Unreduced {
            r0: lane0(p_ll),
            r1: lane1(p_ll) ^ cr_lo,
            r2: lane0(p_hh) ^ cr_hi,
            r3: lane1(p_hh),
        }
    }
}

// -----------------------------------------------------------------------
// AVX-512 + VPCLMULQDQ: 4 independent GF(2^128) multiplies per instruction.
//
// Lane-parallel port of `ghash_mul_binius` above — same 4 product CLMULs +
// two-stage `0x87` reduction, applied independently in each 128-bit lane of
// a `__m512i`. A `__m512i` holds 4 contiguous `F128` (lane i = {lo_i, hi_i});
// since `F128` is `repr(C, align(16))` little-endian, 4 elements load
// directly with `_mm512_loadu_si512` — no shuffles. The reduction is the
// same field element as the scalar `ghash_mul_binius` (cross-checked in the
// ntt module's tests), reached by the identical operation sequence.
// -----------------------------------------------------------------------

/// Per-lane reduction-poly low word: each 128-bit lane = {lo: 0x87, hi: 0}.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn ghash_poly_x4() -> __m512i {
    _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87)
}

/// Per-128-bit-lane reduce: returns `t0 + x^64 · t1` (mod p) in each lane.
/// Mirrors one stage of `ghash_mul_binius`'s recursive reduction:
/// `t0 ^= (t1 << 64)` then `t0 ^= t1.hi · 0x87` (clmul imm `0x01` = hi qword
/// of `t1` × lo qword of `poly`).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn gf2_128_reduce_x4(mut t0: __m512i, t1: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        let poly = ghash_poly_x4();
        t0 = _mm512_xor_si512(t0, _mm512_bslli_epi128::<8>(t1));
        t0 = _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, poly));
        t0
    }
}

/// 4 independent GF(2^128) products. `x` and `y` each hold 4 contiguous
/// `F128`; the result holds the 4 reduced products. Field-identical to
/// applying `ghash_mul_binius` to each lane.
///
/// # Safety
/// Caller must ensure `avx512f` + `vpclmulqdq` are available (statically
/// satisfied by the cfg gate and target-feature attribute).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4(x: __m512i, y: __m512i) -> __m512i {
    // SAFETY: caller carries avx512f+vpclmulqdq.
    unsafe {
        // Cross terms: x.hi·y.lo (imm 0x01) ^ x.lo·y.hi (imm 0x10), at x^64.
        let t1a = _mm512_clmulepi64_epi128::<0x01>(x, y);
        let t1b = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let mut t1 = _mm512_xor_si512(t1a, t1b);
        // High product x.hi·y.hi (imm 0x11), folded into the cross.
        let t2 = _mm512_clmulepi64_epi128::<0x11>(x, y);
        t1 = gf2_128_reduce_x4(t1, t2);
        // Low product x.lo·y.lo (imm 0x00), then fold t1 down to the result.
        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}

// -----------------------------------------------------------------------
// Deferred-reduction 4-lane accumulator (port of binius `WideGhashProduct`,
// 4 lanes wide). Widen each product with 4 CLMULs but DON'T reduce; XOR many
// into the accumulator; reduce once at the end. Per 128-bit lane the
// unreduced product is `lo + mid·x^64 + hi·x^128` with `lo = x.lo·y.lo`,
// `hi = x.hi·y.hi`, `mid = x.hi·y.lo ⊕ x.lo·y.hi` — the same limb split the
// scalar `mul_unreduced`/`F256Unreduced` uses. `fold()` horizontally XORs
// the 4 lanes into one scalar `F256Unreduced`; since `ghash_reduce` is
// F2-linear, fold-then-reduce equals XOR-of-per-lane-reduce, which equals
// the scalar `Σ mul_unreduced` then `reduce`.
// -----------------------------------------------------------------------

/// Load 4 contiguous `F128` (lane i = `p[i]`) into a `__m512i`.
///
/// # Safety
/// `p` must point to 4 readable `F128`; `avx512f` available (cfg-gated).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_loadu(p: *const F128) -> __m512i {
    // SAFETY: caller guarantees 4 readable F128 at p.
    unsafe { _mm512_loadu_si512(p as *const __m512i) }
}

/// Pack 4 `F128` scalars into a `__m512i` (lane 0 = `a`, …, lane 3 = `d`).
///
/// # Safety
/// Requires `avx512f`, as guaranteed by the cfg gate.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_set(a: F128, b: F128, c: F128, d: F128) -> __m512i {
    // Pure register assembly; avx512f cfg-gated.
    _mm512_set_epi64(
        d.hi as i64,
        d.lo as i64,
        c.hi as i64,
        c.lo as i64,
        b.hi as i64,
        b.lo as i64,
        a.hi as i64,
        a.lo as i64,
    )
}

/// XOR the four 128-bit lanes of `v` into a single `__m128i`.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor4_lanes(v: __m512i) -> __m128i {
    // Register-only lane extracts + XOR; avx512f cfg-gated.
    let l0 = _mm512_extracti32x4_epi32::<0>(v);
    let l1 = _mm512_extracti32x4_epi32::<1>(v);
    let l2 = _mm512_extracti32x4_epi32::<2>(v);
    let l3 = _mm512_extracti32x4_epi32::<3>(v);
    _mm_xor_si128(_mm_xor_si128(l0, l1), _mm_xor_si128(l2, l3))
}

/// 4-lane unreduced GF(2^128) product accumulator (deferred reduction).
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[derive(Clone, Copy)]
pub struct WideGhashX4 {
    lo: __m512i,
    hi: __m512i,
    mid: __m512i,
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
impl WideGhashX4 {
    /// Empty accumulator.
    ///
    /// # Safety
    /// `avx512f` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn zero() -> Self {
        let z = _mm512_setzero_si512();
        Self {
            lo: z,
            hi: z,
            mid: z,
        }
    }

    /// XOR-accumulate the 4 unreduced products `x[i]·y[i]` into self.
    ///
    /// # Safety
    /// `avx512f` + `vpclmulqdq` available (cfg-gated).
    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn mul_acc(&mut self, x: __m512i, y: __m512i) {
        // Register-only widen (4 CLMULs) + XOR-accumulate; cfg-gated.
        self.lo = _mm512_xor_si512(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
        self.hi = _mm512_xor_si512(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
        let m = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x01>(x, y),
            _mm512_clmulepi64_epi128::<0x10>(x, y),
        );
        self.mid = _mm512_xor_si512(self.mid, m);
    }

    /// Horizontally XOR the 4 lanes and assemble a scalar `F256Unreduced`
    /// (NOT yet reduced, so it can be XORed with a scalar tail accumulator).
    ///
    /// # Safety
    /// `avx512f` + `sse4.1` available (cfg-gated + attr).
    #[inline]
    #[target_feature(enable = "avx512f,sse4.1")]
    pub unsafe fn fold(self) -> F256Unreduced {
        // SAFETY: caller carries avx512f+sse4.1.
        unsafe {
            let lo = xor4_lanes(self.lo);
            let hi = xor4_lanes(self.hi);
            let mid = xor4_lanes(self.mid);
            let lo_lo = _mm_extract_epi64::<0>(lo) as u64;
            let lo_hi = _mm_extract_epi64::<1>(lo) as u64;
            let hi_lo = _mm_extract_epi64::<0>(hi) as u64;
            let hi_hi = _mm_extract_epi64::<1>(hi) as u64;
            let mid_lo = _mm_extract_epi64::<0>(mid) as u64;
            let mid_hi = _mm_extract_epi64::<1>(mid) as u64;
            F256Unreduced {
                r0: lo_lo,
                r1: lo_hi ^ mid_lo,
                r2: hi_lo ^ mid_hi,
                r3: hi_hi,
            }
        }
    }
}
