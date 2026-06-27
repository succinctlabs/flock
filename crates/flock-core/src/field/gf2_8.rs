// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The NEON 16-wide multiplier (`gf8_mul_vec16` / `gf8_reduce_vec16`) is a
// port of `packed_aes_16x8b_multiply` from binius64
// (https://github.com/binius-zk/binius64,
// `crates/field/src/arch/aarch64/simd_arithmetic.rs`).

//! GF(2^8) with the AES irreducible polynomial x^8 + x^4 + x^3 + x + 1.
//!
//! Reduction: x^8 ≡ x^4 + x^3 + x + 1, so the upper byte h folds back as
//!   h ^ (h<<1) ^ (h<<3) ^ (h<<4).

use core::ops::{Add, AddAssign, Mul, MulAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F8(pub u8);

impl F8 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    #[inline]
    pub const fn new(v: u8) -> Self {
        Self(v)
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Multiplicative inverse via Fermat: x^254 = x^{-1} in F_{2^8}.
    /// Exponent bit pattern 0xFE = 0b11111110 — 7 squarings + 6 multiplies.
    pub fn inv(self) -> Self {
        let mut result = Self::ONE;
        let mut sq = self;
        for i in 0..8 {
            if (0xFEu8 >> i) & 1 != 0 {
                result *= sq;
            }
            sq *= sq;
        }
        result
    }
}

// In GF(2⁸), addition is bitwise XOR by definition — the `^` is correct, not a
// typo for `+` (which is what these Clippy lints guard against).
#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for F8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl AddAssign for F8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Mul for F8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(gf8_reduce(clmul8(self.0, rhs.0)))
    }
}

impl MulAssign for F8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Carry-less product of two bytes; result fits in 15 bits.
#[inline]
fn clmul8(a: u8, b: u8) -> u16 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: `aes` target feature is enabled at compile time.
        unsafe { clmul8_neon(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        clmul8_software(a, b)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
#[inline]
unsafe fn clmul8_neon(a: u8, b: u8) -> u16 {
    use core::arch::aarch64::*;
    let va = vdup_n_p8(a);
    let vb = vdup_n_p8(b);
    let prod = vmull_p8(va, vb);
    vgetq_lane_u16::<0>(vreinterpretq_u16_p16(prod))
}

/// Software fallback / test oracle. Used when `aes` is off, and as the
/// cross-check oracle inside the `software_matches_neon` unit test.
#[allow(dead_code)]
#[inline]
const fn clmul8_software(a: u8, b: u8) -> u16 {
    let b16 = b as u16;
    let mut acc: u16 = 0;
    let mut i = 0;
    while i < 8 {
        if (a >> i) & 1 != 0 {
            acc ^= b16 << i;
        }
        i += 1;
    }
    acc
}

/// Reduce a polynomial of degree ≤ 14 modulo x^8 + x^4 + x^3 + x + 1.
/// Two-step fold: first turns 15-bit input into ≤12-bit, second into ≤8-bit.
///
/// Exposed `pub(crate)` so the URM shift_reduce inner kernel can reuse it.
#[inline]
pub(crate) const fn gf8_reduce(p: u16) -> u8 {
    let h: u16 = p >> 8;
    let t: u16 = (p & 0xff) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4);
    let h2: u16 = t >> 8;
    ((t & 0xff) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4)) as u8
}

// ---------------------------------------------------------------------------
// aarch64 NEON helpers: 16-lane GF(2^8) mul and reduce.
//
// These are the building blocks for the round-1 URM shift_reduce inner kernel.
//
// `vmull_p8` is a baseline NEON instruction (no aes feature needed), so the
// only cfg gate is `target_arch = "aarch64"`.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use core::arch::aarch64::*;
    use core::mem::transmute;

    /// Reduce 16 polynomial products (in interleaved layout `[lo0,hi0, lo1,hi1, ...]`,
    /// passed as `(c0, c1)`) modulo `x^8 + x^4 + x^3 + x + 1`, returning 16 reduced
    /// GF(2^8) values.
    ///
    /// Two-stage Binius-style reduction:
    ///   Stage 1: ch · QPLUS_RSH1 then ·2 (corrects for /x in QPLUS_RSH1)
    ///   Stage 2: high bytes of stage-1 · QSTAR; take low bytes only.
    ///
    /// Constants:
    ///   QPLUS_RSH1 = (x^8+x^4+x^3+x)/x = 0x8d
    ///   QSTAR      = x^4+x^3+x+1       = 0x1b
    ///
    /// # Safety
    /// Uses `core::arch::aarch64` NEON intrinsics; only call on `aarch64`.
    #[inline]
    pub unsafe fn gf8_reduce_vec16(c0: uint8x16_t, c1: uint8x16_t) -> uint8x16_t {
        unsafe {
            let q_plus_rsh1: poly8x8_t = transmute::<u64, poly8x8_t>(0x8d8d8d8d8d8d8d8d_u64);
            let q_star: poly8x8_t = transmute::<u64, poly8x8_t>(0x1b1b1b1b1b1b1b1b_u64);

            let cl = vuzp1q_u8(c0, c1); // low bytes of all 16 products
            let ch = vuzp2q_u8(c0, c1); // high bytes of all 16 products

            // Stage 1.
            let t0 = vreinterpretq_u8_u16(vshlq_n_u16::<1>(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(ch)),
                q_plus_rsh1,
            ))));
            let t1 = vreinterpretq_u8_u16(vshlq_n_u16::<1>(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(ch)),
                q_plus_rsh1,
            ))));

            // Stage 2.
            let tmp_hi = vuzp2q_u8(t0, t1);
            let r0 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(tmp_hi)),
                q_star,
            )));
            let r1 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(tmp_hi)),
                q_star,
            )));

            veorq_u8(cl, vuzp1q_u8(r0, r1))
        }
    }

    /// Element-wise multiply 16 pairs of GF(2^8) values (binius64 13-op NEON kernel).
    ///
    /// # Safety
    /// Uses `core::arch::aarch64` NEON intrinsics (PMULL); only call on `aarch64`.
    #[inline]
    pub unsafe fn gf8_mul_vec16(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe {
            let c0 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(a)),
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(b)),
            )));
            let c1 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(a)),
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(b)),
            )));
            gf8_reduce_vec16(c0, c1)
        }
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 helpers: 32-lane GF(2^8) operations.
//
// Two functions:
//   gf8_mul_vec32_x86  — scalar × vector (VPSHUFB split-nibble lookup)
//   gf8_mul_vec32_x86_ew — element-wise vector × vector (bitsliced schoolbook)
//
// Both require AVX2. The split-nibble lookup (scalar×vector) uses two
// precomputed 16-entry tables T_lo[i] = a*i mod p, T_hi[i] = a*(i<<4) mod p,
// then VPSHUFB does 32 parallel lookups per table.
//
// The element-wise multiply decomposes each lane's multiply into the
// schoolbook form: accumulate (bit_j(a) AND b) << j, with inline modular
// reduction (xtime) at each step to keep the intermediate in 8 bits.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::{F8, gf8_reduce};

    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Scalar-by-vector GF(2^8) multiply: compute `a_scalar * b[i]` for 32 lanes.
    ///
    /// Uses VPSHUFB split-nibble lookup: decompose each b[i] into low/high nibbles,
    /// look up `a * nibble_val` from precomputed tables, XOR the halves.
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gf8_mul_vec32_x86(a_scalar: u8, b: &[u8; 32]) -> [u8; 32] {
        unsafe {
            // Build the two 16-entry lookup tables.
            let mut tbl_lo = [0u8; 16];
            let mut tbl_hi = [0u8; 16];
            for i in 0u8..16 {
                tbl_lo[i as usize] = (F8(a_scalar) * F8(i)).0;
                tbl_hi[i as usize] = (F8(a_scalar) * F8(i << 4)).0;
            }

            // Broadcast each 16-byte table into both 128-bit lanes of a __m256i.
            // VPSHUFB operates independently on each 128-bit lane, so both lanes
            // need identical table contents.
            let tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128(tbl_lo.as_ptr() as *const _));
            let thi = _mm256_broadcastsi128_si256(_mm_loadu_si128(tbl_hi.as_ptr() as *const _));

            let bv = _mm256_loadu_si256(b.as_ptr() as *const _);
            let nibble_mask = _mm256_set1_epi8(0x0F);

            let b_lo = _mm256_and_si256(bv, nibble_mask);
            let b_hi = _mm256_and_si256(_mm256_srli_epi16(bv, 4), nibble_mask);

            let r_lo = _mm256_shuffle_epi8(tlo, b_lo);
            let r_hi = _mm256_shuffle_epi8(thi, b_hi);

            let result = _mm256_xor_si256(r_lo, r_hi);
            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut _, result);
            out
        }
    }

    /// Element-wise GF(2^8) multiply: compute `a[i] * b[i]` for 32 lanes.
    ///
    /// Uses bitsliced schoolbook multiplication with inline xtime reduction.
    /// At each step j (0..7), if bit j of a[i] is set, XOR the running product
    /// of b into the accumulator. Then apply xtime (multiply b by x, reducing
    /// mod the AES polynomial x^8+x^4+x^3+x+1).
    ///
    /// This avoids the need for wide (16-bit) intermediates by reducing after
    /// every shift.
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gf8_mul_vec32_x86_ew(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        unsafe {
            let av = _mm256_loadu_si256(a.as_ptr() as *const _);
            let bv = _mm256_loadu_si256(b.as_ptr() as *const _);

            let poly = _mm256_set1_epi8(0x1Bu8 as i8); // x^4+x^3+x+1
            let zero = _mm256_setzero_si256();
            let one_bit = _mm256_set1_epi8(1);

            let mut acc = zero;
            let mut shifted_b = bv; // b * x^j at step j

            // Bit 0: if a[i] & 1, XOR b into acc.
            let mask0 = _mm256_cmpeq_epi8(
                _mm256_and_si256(av, one_bit),
                one_bit,
            );
            acc = _mm256_xor_si256(acc, _mm256_and_si256(mask0, shifted_b));

            // Bits 1..7: xtime shifted_b, then conditionally XOR.
            macro_rules! do_bit {
                ($bit:expr) => {{
                    // xtime: shifted_b = (shifted_b << 1) ^ (0x1B if high bit was set)
                    let high_bits = _mm256_cmpgt_epi8(zero, shifted_b); // -1 if byte >= 0x80
                    let sb_shifted = _mm256_add_epi8(shifted_b, shifted_b); // << 1 (mod 256)
                    let reduce = _mm256_and_si256(high_bits, poly);
                    shifted_b = _mm256_xor_si256(sb_shifted, reduce);

                    // Extract bit $bit of each a[i]: shift right, mask, compare.
                    let a_bit = _mm256_and_si256(
                        _mm256_srli_epi16(av, $bit),
                        one_bit,
                    );
                    let mask = _mm256_cmpeq_epi8(a_bit, one_bit);
                    acc = _mm256_xor_si256(acc, _mm256_and_si256(mask, shifted_b));
                }};
            }
            do_bit!(1);
            do_bit!(2);
            do_bit!(3);
            do_bit!(4);
            do_bit!(5);
            do_bit!(6);
            do_bit!(7);

            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut _, acc);
            out
        }
    }

    /// 32-lane GF(2^8) modular reduction: reduce 32 u16 values (stored as
    /// interleaved low/high byte vectors) modulo x^8+x^4+x^3+x+1.
    ///
    /// Input: `lo` contains the low bytes, `hi` contains the high bytes of the
    /// 15-bit carry-less products. Output: 32 reduced GF(2^8) bytes.
    ///
    /// Uses VPSHUFB split-nibble lookup for the reduction polynomial multiply.
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn gf8_reduce_vec32(lo: __m256i, hi: __m256i) -> __m256i {
        unsafe {
            // h folds back as: h ^ (h<<1) ^ (h<<3) ^ (h<<4)
            // This can overflow into bits 8..11, needing a second fold.
            // Use VPSHUFB: build a 16-entry table for the fold of each nibble of h.

            // Stage 1: fold h (7 bits max) into lo.
            // For each possible h (0..127), fold(h) = h ^ (h<<1) ^ (h<<3) ^ (h<<4).
            // The result is at most 12 bits. We keep the low byte and the carry.
            // But h is at most 7 bits (degree 14 product → 7 high bits), so h in 0..127.
            // fold(h) fits in 12 bits; the low 8 bits XOR into lo, bits 8..11 need second fold.

            // Build tables: for nibble values 0..15, compute the fold contribution.
            // T_lo_nib[i] = fold(i) & 0xFF, T_hi_nib[i] = fold(i<<4) & 0xFFFF split into lo/hi bytes.
            // But fold(h) = h ^ (h<<1) ^ (h<<3) ^ (h<<4) where h is 7-bit.
            // We split h into h_lo (bits 0..3) and h_hi (bits 4..6).
            // fold is linear in GF(2), so fold(h) = fold(h_lo) ^ fold(h_hi << 4).

            let mut tbl_lo_lo = [0u8; 16]; // fold(i) low byte, for i = low nibble of h
            let mut tbl_lo_hi = [0u8; 16]; // fold(i) high byte
            let mut tbl_hi_lo = [0u8; 16]; // fold(i<<4) low byte
            let mut tbl_hi_hi = [0u8; 16]; // fold(i<<4) high byte

            for i in 0u16..16 {
                let f1 = i ^ (i << 1) ^ (i << 3) ^ (i << 4);
                tbl_lo_lo[i as usize] = (f1 & 0xFF) as u8;
                tbl_lo_hi[i as usize] = ((f1 >> 8) & 0xFF) as u8;

                let ih = i << 4;
                let f2 = ih ^ (ih << 1) ^ (ih << 3) ^ (ih << 4);
                tbl_hi_lo[i as usize] = (f2 & 0xFF) as u8;
                tbl_hi_hi[i as usize] = ((f2 >> 8) & 0xFF) as u8;
            }

            let t_lo_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(tbl_lo_lo.as_ptr() as *const _));
            let t_lo_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(tbl_lo_hi.as_ptr() as *const _));
            let t_hi_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(tbl_hi_lo.as_ptr() as *const _));
            let t_hi_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(tbl_hi_hi.as_ptr() as *const _));

            let nibble_mask = _mm256_set1_epi8(0x0F);
            let h_lo_nib = _mm256_and_si256(hi, nibble_mask);
            let h_hi_nib = _mm256_and_si256(_mm256_srli_epi16(hi, 4), nibble_mask);

            // Stage-1 fold: 16-bit result split into lo/hi parts.
            let fold_lo = _mm256_xor_si256(
                _mm256_shuffle_epi8(t_lo_lo, h_lo_nib),
                _mm256_shuffle_epi8(t_hi_lo, h_hi_nib),
            );
            let fold_hi = _mm256_xor_si256(
                _mm256_shuffle_epi8(t_lo_hi, h_lo_nib),
                _mm256_shuffle_epi8(t_hi_hi, h_hi_nib),
            );

            // XOR fold low byte into lo to get stage-1 result low byte.
            let s1_lo = _mm256_xor_si256(lo, fold_lo);

            // Stage 2: fold_hi has bits 8..11 (at most 4 bits). Apply same reduction.
            // For 4-bit h2, fold(h2) is at most 12 bits, but since h2 < 16, the
            // high bits after fold are at most 4 more bits. For h2 in 0..15:
            // fold(h2) = h2 ^ (h2<<1) ^ (h2<<3) ^ (h2<<4), max 8 bits for h2<8.
            // Actually h2 can be up to 0x0F. fold(0x0F) = 0x0F ^ 0x1E ^ 0x78 ^ 0xF0 = 0x81.
            // So stage-2 result fits in 8 bits.
            let mut tbl_s2 = [0u8; 16];
            for i in 0u16..16 {
                let f = i ^ (i << 1) ^ (i << 3) ^ (i << 4);
                tbl_s2[i as usize] = (f & 0xFF) as u8;
                // Verify no overflow: assert f < 256
            }
            let t_s2 = _mm256_broadcastsi128_si256(_mm_loadu_si128(tbl_s2.as_ptr() as *const _));
            let s2_fold = _mm256_shuffle_epi8(t_s2, fold_hi);

            _mm256_xor_si256(s1_lo, s2_fold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic splitmix64 PRNG for test reproducibility.
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
    }

    #[test]
    fn add_is_xor() {
        assert_eq!(F8(0x53) + F8(0xCA), F8(0x53 ^ 0xCA));
        assert_eq!(F8(0xFF) + F8(0xFF), F8::ZERO);
    }

    #[test]
    fn mul_identities() {
        for v in 0u8..=255 {
            let a = F8(v);
            assert_eq!(a * F8::ZERO, F8::ZERO);
            assert_eq!(a * F8::ONE, a);
        }
    }

    #[test]
    fn mul_known_values() {
        // x = F8(0x02). x^2 = 0x04. x^4 = 0x10.
        // x^8 mod p = x^4 + x^3 + x + 1 = 0x1B.
        let x = F8(0x02);
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        assert_eq!(x2, F8(0x04));
        assert_eq!(x4, F8(0x10));
        assert_eq!(x8, F8(0x1B));
    }

    #[test]
    fn inv_roundtrip() {
        for v in 1u8..=255 {
            let a = F8(v);
            assert_eq!(a * a.inv(), F8::ONE, "v={}", v);
        }
    }

    #[test]
    fn software_matches_neon() {
        // If we are on aarch64+aes, sanity-check that the software path agrees.
        let mut rng = Rng::new(0xDEADBEEF);
        for _ in 0..1024 {
            let a = (rng.next_u64() & 0xff) as u8;
            let b = (rng.next_u64() & 0xff) as u8;
            assert_eq!(clmul8(a, b), clmul8_software(a, b));
        }
    }

    #[test]
    fn associativity_random() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..256 {
            let a = F8((rng.next_u64() & 0xff) as u8);
            let b = F8((rng.next_u64() & 0xff) as u8);
            let c = F8((rng.next_u64() & 0xff) as u8);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn mul_commutativity_exhaustive() {
        // Trivially symmetric in the formula, but free to assert over all pairs.
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                assert_eq!(F8(a) * F8(b), F8(b) * F8(a));
            }
        }
    }

    #[test]
    fn fips_197_test_vectors() {
        // FIPS 197 § 4.2 (AES specification) publishes these products
        // for the GF(2^8) multiplication used by AES.
        assert_eq!(F8(0x57) * F8(0x13), F8(0xfe), "FIPS-197: 57·13");
        assert_eq!(F8(0x57) * F8(0x83), F8(0xc1), "FIPS-197: 57·83");
        // xtime: a · 0x02 (used by MixColumns), exhaustively cross-check
        // against the spec'd formula: xtime(a) = (a << 1) ^ (0x1B if a high bit).
        for a in 0u8..=255 {
            let expected = if a & 0x80 != 0 {
                (a << 1) ^ 0x1b
            } else {
                a << 1
            };
            assert_eq!(
                (F8(a) * F8(0x02)).0,
                expected,
                "xtime mismatch at a=0x{a:02x}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_gf8_mul_vec16_matches_scalar() {
        use core::arch::aarch64::*;
        use core::mem::transmute;

        let mut rng = Rng::new(0xBADC0FFEE);
        for _ in 0..256 {
            let mut a_arr = [0u8; 16];
            let mut b_arr = [0u8; 16];
            for i in 0..16 {
                a_arr[i] = (rng.next_u64() & 0xff) as u8;
                b_arr[i] = (rng.next_u64() & 0xff) as u8;
            }
            // Scalar reference: lane-wise F8 mul.
            let mut expected = [0u8; 16];
            for i in 0..16 {
                expected[i] = (F8(a_arr[i]) * F8(b_arr[i])).0;
            }
            // NEON result.
            let result_vec = unsafe {
                let a_v = vld1q_u8(a_arr.as_ptr());
                let b_v = vld1q_u8(b_arr.as_ptr());
                neon::gf8_mul_vec16(a_v, b_v)
            };
            let result: [u8; 16] = unsafe { transmute(result_vec) };
            assert_eq!(result, expected, "a={:02x?}, b={:02x?}", a_arr, b_arr);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gf8_vpshufb_matches_scalar() {
        // Exhaustive: for all 256 × 256 input pairs, VPSHUFB scalar×vector
        // must match the scalar reference.
        for a_val in 0u16..=255 {
            let a = a_val as u8;
            // Process b in batches of 32.
            let mut b_arr = [0u8; 32];
            for b_base in (0u16..=255).step_by(32) {
                for j in 0..32u16 {
                    let bv = b_base + j;
                    b_arr[j as usize] = if bv <= 255 { bv as u8 } else { 0 };
                }
                let result = unsafe { super::x86::gf8_mul_vec32_x86(a, &b_arr) };
                for j in 0..32u16 {
                    let bv = b_base + j;
                    if bv <= 255 {
                        let expected = (F8(a) * F8(bv as u8)).0;
                        assert_eq!(
                            result[j as usize], expected,
                            "gf8_mul_vec32_x86 mismatch: a=0x{a:02x}, b=0x{:02x}",
                            bv as u8
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gf8_mul_vec32_x86_matches_scalar() {
        // 32 random pairs, batch multiply, check each matches scalar.
        let mut rng = Rng::new(0xABCD1234);
        for _ in 0..512 {
            let a_scalar = (rng.next_u64() & 0xff) as u8;
            let mut b_arr = [0u8; 32];
            for j in 0..32 {
                b_arr[j] = (rng.next_u64() & 0xff) as u8;
            }
            let result = unsafe { super::x86::gf8_mul_vec32_x86(a_scalar, &b_arr) };
            for j in 0..32 {
                let expected = (F8(a_scalar) * F8(b_arr[j])).0;
                assert_eq!(
                    result[j], expected,
                    "scalar×vec mismatch: a=0x{a_scalar:02x}, b[{j}]=0x{:02x}",
                    b_arr[j]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gf8_mul_vec32_ew_x86_matches_scalar() {
        // Element-wise: 32 random pairs, verify against scalar F8 mul.
        let mut rng = Rng::new(0xF00DCAFE);
        for _ in 0..512 {
            let mut a_arr = [0u8; 32];
            let mut b_arr = [0u8; 32];
            for j in 0..32 {
                a_arr[j] = (rng.next_u64() & 0xff) as u8;
                b_arr[j] = (rng.next_u64() & 0xff) as u8;
            }
            let result = unsafe { super::x86::gf8_mul_vec32_x86_ew(&a_arr, &b_arr) };
            for j in 0..32 {
                let expected = (F8(a_arr[j]) * F8(b_arr[j])).0;
                assert_eq!(
                    result[j], expected,
                    "ew mismatch at lane {j}: a=0x{:02x}, b=0x{:02x}",
                    a_arr[j], b_arr[j]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gf8_mul_vec32_ew_x86_exhaustive_sample() {
        // Sample 256 a-values × 32 random b-values each for broader coverage.
        let mut rng = Rng::new(0xBEEFBEEF);
        for a_val in 0u8..=255 {
            let mut a_arr = [a_val; 32];
            let mut b_arr = [0u8; 32];
            for j in 0..32 {
                b_arr[j] = (rng.next_u64() & 0xff) as u8;
                a_arr[j] = a_val; // keep constant for this row
            }
            let result = unsafe { super::x86::gf8_mul_vec32_x86_ew(&a_arr, &b_arr) };
            for j in 0..32 {
                let expected = (F8(a_val) * F8(b_arr[j])).0;
                assert_eq!(
                    result[j], expected,
                    "ew-exhaust mismatch: a=0x{a_val:02x}, b=0x{:02x}",
                    b_arr[j]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn gf8_reduce_vec32_matches_scalar() {
        // Verify vectorized reduction against the scalar gf8_reduce.
        let mut rng = Rng::new(0xDEAD5678);
        for _ in 0..1024 {
            let mut lo_arr = [0u8; 32];
            let mut hi_arr = [0u8; 32];
            for j in 0..32 {
                let val = (rng.next_u64() & 0x7FFF) as u16; // 15-bit product
                lo_arr[j] = (val & 0xFF) as u8;
                hi_arr[j] = ((val >> 8) & 0xFF) as u8;
            }
            let result = unsafe {
                use core::arch::x86_64::*;
                let lo = _mm256_loadu_si256(lo_arr.as_ptr() as *const _);
                let hi = _mm256_loadu_si256(hi_arr.as_ptr() as *const _);
                let r = super::x86::gf8_reduce_vec32(lo, hi);
                let mut out = [0u8; 32];
                _mm256_storeu_si256(out.as_mut_ptr() as *mut _, r);
                out
            };
            for j in 0..32 {
                let val = (lo_arr[j] as u16) | ((hi_arr[j] as u16) << 8);
                let expected = gf8_reduce(val);
                assert_eq!(
                    result[j], expected,
                    "reduce mismatch at lane {j}: val=0x{val:04x}"
                );
            }
        }
    }

    #[test]
    fn fermat_little_theorem() {
        // F_{2^8}\{0} has order 255, so a^{255} = 1 for every nonzero a.
        // Strong structural check: catches any single-bit error in the
        // reduction logic, since wrong reduction breaks the cyclic group.
        for v in 1u8..=255 {
            let a = F8(v);
            let mut p = F8::ONE;
            for _ in 0..255 {
                p *= a;
            }
            assert_eq!(p, F8::ONE, "a^255 != 1 for a=0x{v:02x}");
        }
    }
}
