//! Round-1 prover message for the **degree-4** zerocheck — optimized
//! (shift_reduce + extract_z, scalar).
//!
//! Companion to [`super::univariate_skip_optimized`] for the degree-2 case.
//! Same layered optimizations apply:
//!
//! 1. **Geometric small-eq + shift_reduce inner** (3 inner-most rest dims).
//!    Uses the same protocol-fixed `r[K_SKIP..K_SKIP+3] = φ_8([0xF7, 0x53, 0xB5])`
//!    so `eq_small[K] = C_s · α^K`. The shift_reduce trick:
//!    `Σ_K eq_small[K] · φ_8(y_K)  =  C_s · φ_8(reduce(Σ_K y_K << K))`
//!    works **unchanged** for `y_K = a·b·c·d ∈ F_8`, because F_8 is closed
//!    under product and the shift_reduce identity is purely about its F_8
//!    elements.
//!
//! 2. **Geometric medium-eq + convert table** (4 next rest dims). Same convert
//!    table as the degree-2 path (re-used via [`super::univariate_skip_optimized::convert_table`]).
//!
//! 3. **D⁻¹ absorbed into eq_lo.** Same as degree-2.
//!
//! ## Output relationship
//!
//! Same `C_s` scaling vs the naive:
//!   `C_s · (res_ABCD[i] + res_Z[i])  ==  naive_p_abcd[i] + naive_p_z[i]`
//! with `C_s = φ_8(0x1C)`. Verified by `optimized_matches_naive_modulo_cs`.
//!
//! ## What changes vs degree-2
//!
//! - **Output domain Λ₄ has 192 lanes** (vs Λ has 64). Constraint poly degree
//!   in λ is < 4·64−3 = 253, requiring ≥ 253 evaluation points; next power
//!   of 2 is 256 = |V₈|, with |Λ₄| = 256−64 = 192 fresh evals.
//! - **Four NTT-extends per row** (a, b, c, d) instead of two (a, b). Each
//!   extends from S (64 bits) to V₈ (256 F8 values).
//! - **F_8 product is 3 chained F_8 mults** per lane (a·b·c·d) instead of 1
//!   (a·b).
//!
//! ## What is intentionally NOT here yet (follow-up patches)
//!
//! - **Lookup-table NTT extension for S→V₈** (analogous to
//!   [`crate::ntt::InvNttTableByteSingleGf8`] but for mismatched input/output k).
//!   This file uses direct `AdditiveNttGf8` calls for clarity — it's a
//!   correctness-first port of the structure.
//! - **NEON fused inner kernel** mirroring `shift_reduce_inner_ab_fused_neon`.
//! - **Rayon parallelism over x_hi** mirroring the degree-2 driver.
//! - **PaddingSpec / build_b_med_counts** for skip-padding-windows.
//!
//! All of these are direct ports of the degree-2 versions and are pure
//! perf-engineering — the math is settled by the tests in this file.

use core::arch::aarch64::uint8x16_t;
use std::sync::OnceLock;

use crate::field::gf2_8::gf8_reduce;
use crate::field::{F8, F128, mul_by_x, phi8};
use crate::ntt::{AdditiveNttGf8, InvNttTableSToV8Gf8};

use super::univariate_skip::build_eq;
use super::univariate_skip_deg4::{K_SKIP, K_V8, LAMBDA4_SIZE, S_SIZE, V8_SIZE};
use super::univariate_skip_optimized::{
    bit_transpose_64bytes, medium_challenges_ghash, small_challenges_ghash,
};

// ---------------------------------------------------------------------------
// Protocol constants. Same shape as the degree-2 path: 3 small + 4 medium dims
// absorbed by the shift_reduce/convert-table optimizations.
// ---------------------------------------------------------------------------

const N_INNER: usize = 7;
const N_MEDIUM: usize = 4;
const N_CHUNKS: usize = S_SIZE / 8; // 8

/// Re-export the small challenges for callers building `r` for cross-check.
pub fn small_challenges_deg4() -> [F128; 3] {
    small_challenges_ghash()
}
/// Re-export the medium challenges for callers building `r` for cross-check.
pub fn medium_challenges_deg4() -> [F128; 4] {
    medium_challenges_ghash()
}

/// D⁻¹ for the geometric medium-eq factorization. Same value as deg-2; we
/// just reuse the cached one via the public accessor pattern.
fn d_inv() -> F128 {
    // Inline of degree-2's d_inv since that's pub(crate)-scoped. The value
    // is purely a function of the medium challenge structure — protocol-fixed.
    use crate::field::F128 as F;
    let g1 = F {
        lo: 1u64 << 1,
        hi: 0,
    };
    let g2 = F {
        lo: 1u64 << 2,
        hi: 0,
    };
    let g4 = F {
        lo: 1u64 << 4,
        hi: 0,
    };
    let g8 = F {
        lo: 1u64 << 8,
        hi: 0,
    };
    ((F::ONE + g1) * (F::ONE + g2) * (F::ONE + g4) * (F::ONE + g8)).inv()
}

/// Convert table γ^b · φ_8(v) for b ∈ [0, 16), v ∈ [0, 256). Same as the
/// degree-2 path — protocol-fixed once. Cached after the first call.
fn build_convert_table() -> Vec<F128> {
    use crate::field::PHI_8_TABLE;
    let mut gamma_pow = [F128::ZERO; 16];
    gamma_pow[0] = F128::ONE;
    for b in 1..16 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }
    let mut table = vec![F128::ZERO; 16 * 256];
    for b in 0..16 {
        let g_b = gamma_pow[b];
        for v in 0..256 {
            table[b * 256 + v] = g_b * PHI_8_TABLE[v];
        }
    }
    table
}

static CONVERT_TABLE_CACHE: OnceLock<Vec<F128>> = OnceLock::new();
fn convert_table() -> &'static [F128] {
    CONVERT_TABLE_CACHE.get_or_init(build_convert_table)
}

// ---------------------------------------------------------------------------
// NTT extension from S (size 64) to V₈ (size 256), single row.
//
// Mirrors what `InvNttTableByteSingleGf8::apply` does for the deg-2 case but
// implemented directly via two NTT instances (no lookup table yet). Output is
// 256 F8 bytes — the first 64 reproduce the input bits on S, the next 192 are
// fresh on Λ₄.
// ---------------------------------------------------------------------------

/// Bundled NTT instances. Build once per session; pass into the round-1 fn.
#[derive(Clone, Debug)]
pub struct NttPairDeg4 {
    pub ntt_s: AdditiveNttGf8,  // size 64
    pub ntt_v8: AdditiveNttGf8, // size 256
}

impl Default for NttPairDeg4 {
    fn default() -> Self {
        Self::new()
    }
}

impl NttPairDeg4 {
    pub fn new() -> Self {
        Self {
            ntt_s: AdditiveNttGf8::new(K_SKIP, F8::ZERO),
            ntt_v8: AdditiveNttGf8::new(K_V8, F8::ZERO),
        }
    }

    /// Extend 64 input bits (one byte per b_chunk position, LSB packed) to 256
    /// F8 evaluations on V₈ = F_8. Output[0..64] reproduces the input bits on
    /// S; output[64..256] is the fresh extension on Λ₄.
    ///
    /// `bits` — 8 bytes, bit `8*b + t` = the input at S-coord `8*b + t`.
    /// `out`  — 256 F8 slot, written entirely.
    pub fn extend_row(&self, bits: &[u8], out: &mut [F8]) {
        debug_assert_eq!(bits.len(), N_CHUNKS);
        debug_assert_eq!(out.len(), V8_SIZE);
        // 1. Unpack 64 bits → 64 F8 in positions 0..64; zero-pad 64..256.
        for s in 0..S_SIZE {
            let bit = (bits[s / 8] >> (s % 8)) & 1;
            out[s] = F8(bit);
        }
        for s in S_SIZE..V8_SIZE {
            out[s] = F8::ZERO;
        }
        // 2. inv on the first 64 → coefficients in 6-dim novel basis.
        self.ntt_s.inverse(&mut out[..S_SIZE]);
        // 3. (positions 64..256 already zero — those are W_64..W_255 coeffs.)
        // 4. fwd on full 256 → evaluations on V₈.
        self.ntt_v8.forward(out);
    }
}

// ---------------------------------------------------------------------------
// Scalar shift_reduce inner kernel — 4-way product.
//
// For one medium-position b_med and the 8 small-positions K ∈ 0..8:
//   1. Extend a,b,c,d row for chunk (chunk_byte_base + (b_med*8 + K)*8) to V₈.
//   2. y_K[lane] = a[lane] · b[lane] · c[lane] · d[lane]  (F_8, 4-way).
//   3. acc[lane] ^= (y_K[lane] as u16) << K   (no reduction yet).
// At end, reduce each acc[lane] back to F_8.
//
// Operates on the Λ₄ lanes (192 of them). The S lanes (0..63) are
// known to the verifier from the witness bits and are computed but
// discarded — the round message only needs the Λ₄ portion.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// F128-valued NTT extension from S (64 lanes) to Λ₄ (192 lanes).
//
// Mirrors `crate::zerocheck::univariate_skip::ntt_extend_f128_vec_ghash` but
// uses the deg-4 lookup table (InvNttTableSToV8Gf8) which produces 256 F8
// outputs per input row. Discards the first 64 (S) lanes; keeps the last 192
// (Λ₄ lanes).
//
// Per bit-plane (128 total, one per F128 bit):
//   1. Pack bit b of each in_s[z] into 8 LSB-first bytes.
//   2. `table.apply` → 256 F8 outputs on V₈ (NEON-fused).
//   3. Lift via φ_8, scale by γ^b, accumulate into out[Λ₄ lane].
//
// Per call: 128 table-applies + 128 × 192 (φ_8 + F128 mul + F128 xor) ≈ ~25k
// F128 ops, NEON-accelerated by the table apply and the F128 mul intrinsics.
// ---------------------------------------------------------------------------

pub fn ntt_extend_f128_vec_ghash_deg4(in_s: &[F128], table: &InvNttTableSToV8Gf8) -> Vec<F128> {
    assert_eq!(in_s.len(), S_SIZE);
    assert_eq!(table.k_in, K_SKIP);
    assert_eq!(table.k_out, K_V8);

    let mut out = vec![F128::ZERO; LAMBDA4_SIZE];

    // γ^b for b ∈ [0, 128).
    let mut gamma_pow = [F128::ZERO; 128];
    gamma_pow[0] = F128::ONE;
    for b in 1..128 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }

    let mut input_bits = vec![0u8; N_CHUNKS]; // 8 bytes = 64 input bits
    let mut out_bytes = vec![F8::ZERO; V8_SIZE]; // 256 F8 outputs

    for b in 0..128 {
        // Pack bit b of each in_s[z] into LSB-first byte form.
        input_bits.iter_mut().for_each(|x| *x = 0);
        for z in 0..S_SIZE {
            let bit = if b < 64 {
                (in_s[z].lo >> b) & 1
            } else {
                (in_s[z].hi >> (b - 64)) & 1
            };
            if bit != 0 {
                input_bits[z / 8] |= 1u8 << (z % 8);
            }
        }

        // NEON-fused bit-input NTT extension to all 256 V₈ lanes.
        table.apply(&input_bits, &mut out_bytes);

        // Lift the Λ₄ lanes via φ_8, scale by γ^b, accumulate.
        let g_b = gamma_pow[b];
        for i in 0..LAMBDA4_SIZE {
            out[i] += g_b * phi8(out_bytes[S_SIZE + i]);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// NEON fused inner kernel — 4-way product, 192-lane output.
//
// Tile structure: process 12 output chunks (192 Λ₄ lanes) as 3 tiles of 4
// chunks each. Per tile, all 8 K iterations run with row-state in registers,
// accumulating into 8 acc regs (4 chunk-pairs × 2 lo/hi). The chunk index
// XOR (b >> 1) maps {4,5,6,7} ↔ {4,5,6,7} within tile 0, similarly for tiles
// 1/2 — so each tile's input chunks stay within its own 4-chunk window.
//
// Per-K register pressure: 4 row-state regs × 4 factors = 16, plus 4 product
// temps, plus 8 acc regs = 28 SIMD regs. Fits ARM's 32-reg file.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_4_regs_deg4<
    const BH: usize,
    const ODD: bool,
    const TILE_BASE: usize,
>(
    table_base: *const u8,
    byte: u8,
    d0: &mut uint8x16_t,
    d1: &mut uint8x16_t,
    d2: &mut uint8x16_t,
    d3: &mut uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        // Row[byte] in the 256-byte-row table.
        let row = table_base.add(byte as usize * V8_SIZE);
        let v0 = vld1q_u8(row.add(((TILE_BASE + 0) ^ BH) * 16));
        let v1 = vld1q_u8(row.add(((TILE_BASE + 1) ^ BH) * 16));
        let v2 = vld1q_u8(row.add(((TILE_BASE + 2) ^ BH) * 16));
        let v3 = vld1q_u8(row.add(((TILE_BASE + 3) ^ BH) * 16));
        let (v0, v1, v2, v3) = if ODD {
            (
                vextq_u8::<8>(v0, v0),
                vextq_u8::<8>(v1, v1),
                vextq_u8::<8>(v2, v2),
                vextq_u8::<8>(v3, v3),
            )
        } else {
            (v0, v1, v2, v3)
        };
        *d0 = veorq_u8(*d0, v0);
        *d1 = veorq_u8(*d1, v1);
        *d2 = veorq_u8(*d2, v2);
        *d3 = veorq_u8(*d3, v3);
    }
}

/// Build the 4-chunk row-state for one factor at one K, given 8 input bytes.
/// b=0: straight load from row[byte_0] at offsets TILE_BASE..TILE_BASE+3.
/// b=1..7: XOR table-row[byte_b] with permutation `(BH(b), ODD(b))`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn build_factor_row_4chunks_deg4<const TILE_BASE: usize>(
    table_base: *const u8,
    factor_row: *const u8,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    use core::arch::aarch64::*;
    unsafe {
        let row0 = table_base.add(*factor_row as usize * V8_SIZE);
        let mut d0 = vld1q_u8(row0.add((TILE_BASE + 0) * 16));
        let mut d1 = vld1q_u8(row0.add((TILE_BASE + 1) * 16));
        let mut d2 = vld1q_u8(row0.add((TILE_BASE + 2) * 16));
        let mut d3 = vld1q_u8(row0.add((TILE_BASE + 3) * 16));

        // b = 1..7, with the b/permutation pattern from the §2.1 collapse.
        xor_apply_byte_into_4_regs_deg4::<0, true, TILE_BASE>(
            table_base,
            *factor_row.add(1),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<1, false, TILE_BASE>(
            table_base,
            *factor_row.add(2),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<1, true, TILE_BASE>(
            table_base,
            *factor_row.add(3),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<2, false, TILE_BASE>(
            table_base,
            *factor_row.add(4),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<2, true, TILE_BASE>(
            table_base,
            *factor_row.add(5),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<3, false, TILE_BASE>(
            table_base,
            *factor_row.add(6),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<3, true, TILE_BASE>(
            table_base,
            *factor_row.add(7),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );

        (d0, d1, d2, d3)
    }
}

/// Run all 8 K iterations for one 4-chunk tile (64 Λ₄ lanes), accumulating
/// into 8 SIMD regs (4 chunk-pairs × 2 lo/hi). Writes the 64 reduced output
/// bytes to `out_tile`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn process_tile_deg4<const TILE_BASE: usize>(
    a_packed: *const u8,
    b_packed: *const u8,
    c_packed: *const u8,
    d_packed: *const u8,
    table_base: *const u8,
    byte_base_b: usize,
    out_tile: *mut u8,
) {
    use crate::field::gf2_8::neon::{gf8_mul_vec16, gf8_reduce_vec16};
    use core::arch::aarch64::*;
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        // Per-K body: load 4 factors' row state for this K's 8 input bytes,
        // chain-multiply, shift-XOR into accs.
        macro_rules! do_k {
            ($k:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                let (a0, a1, a2, a3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, a_packed.add(off));
                let (b0, b1, b2, b3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, b_packed.add(off));
                let (c0, c1, c2, c3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, c_packed.add(off));
                let (d0, d1, d2, d3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, d_packed.add(off));

                // y = (a·b)·(c·d), 3 muls per chunk but in a 2-deep tree (vs
                // the 3-deep chain `((a·b)·c)·d`). Same mul count; shorter
                // dependency chain ⇒ more PMULL ILP on M-series superscalar.
                let ab0 = gf8_mul_vec16(a0, b0);
                let cd0 = gf8_mul_vec16(c0, d0);
                let ab1 = gf8_mul_vec16(a1, b1);
                let cd1 = gf8_mul_vec16(c1, d1);
                let ab2 = gf8_mul_vec16(a2, b2);
                let cd2 = gf8_mul_vec16(c2, d2);
                let ab3 = gf8_mul_vec16(a3, b3);
                let cd3 = gf8_mul_vec16(c3, d3);
                let y0 = gf8_mul_vec16(ab0, cd0);
                let y1 = gf8_mul_vec16(ab1, cd1);
                let y2 = gf8_mul_vec16(ab2, cd2);
                let y3 = gf8_mul_vec16(ab3, cd3);

                // Widen-shift by K, XOR into acc. K is const so vshll_n_u8::<K>
                // specializes per call site.
                acc0_lo = veorq_u16(acc0_lo, vshll_n_u8::<$k>(vget_low_u8(y0)));
                acc0_hi = veorq_u16(acc0_hi, vshll_n_u8::<$k>(vget_high_u8(y0)));
                acc1_lo = veorq_u16(acc1_lo, vshll_n_u8::<$k>(vget_low_u8(y1)));
                acc1_hi = veorq_u16(acc1_hi, vshll_n_u8::<$k>(vget_high_u8(y1)));
                acc2_lo = veorq_u16(acc2_lo, vshll_n_u8::<$k>(vget_low_u8(y2)));
                acc2_hi = veorq_u16(acc2_hi, vshll_n_u8::<$k>(vget_high_u8(y2)));
                acc3_lo = veorq_u16(acc3_lo, vshll_n_u8::<$k>(vget_low_u8(y3)));
                acc3_hi = veorq_u16(acc3_hi, vshll_n_u8::<$k>(vget_high_u8(y3)));
            }};
        }
        do_k!(0);
        do_k!(1);
        do_k!(2);
        do_k!(3);
        do_k!(4);
        do_k!(5);
        do_k!(6);
        do_k!(7);

        // F_8 reduction (u16 → u8) + store 4 × 16 = 64 bytes of output.
        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));
        vst1q_u8(out_tile, r0);
        vst1q_u8(out_tile.add(16), r1);
        vst1q_u8(out_tile.add(32), r2);
        vst1q_u8(out_tile.add(48), r3);
    }
}

/// NEON fused inner — 192-lane output across 3 tiles of 4 chunks each.
///
/// Output: 192 reduced F_8 bytes in `out`, covering Λ₄ = V₈ \ S.
/// Tile mapping (V₈ chunk indices 0..16 with S = chunks 0..4):
///   - Tile 0 (chunks 4..8): out[0..64]
///   - Tile 1 (chunks 8..12): out[64..128]
///   - Tile 2 (chunks 12..16): out[128..192]
#[cfg(target_arch = "aarch64")]
fn shift_reduce_inner_abcd_fused_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    table: &InvNttTableSToV8Gf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; LAMBDA4_SIZE],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let table_base = table.data_ptr();
    let a_p = a_packed.as_ptr();
    let b_p = b_packed.as_ptr();
    let c_p = c_packed.as_ptr();
    let d_p = d_packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    unsafe {
        process_tile_deg4::<4>(a_p, b_p, c_p, d_p, table_base, byte_base_b, out_ptr);
        process_tile_deg4::<8>(a_p, b_p, c_p, d_p, table_base, byte_base_b, out_ptr.add(64));
        process_tile_deg4::<12>(
            a_p,
            b_p,
            c_p,
            d_p,
            table_base,
            byte_base_b,
            out_ptr.add(128),
        );
    }
}

/// Dispatch helper — NEON when available, scalar otherwise.
#[inline]
fn shift_reduce_inner_abcd(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    table: &InvNttTableSToV8Gf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; LAMBDA4_SIZE],
    a_col: &mut [F8; V8_SIZE],
    b_col: &mut [F8; V8_SIZE],
    c_col: &mut [F8; V8_SIZE],
    d_col: &mut [F8; V8_SIZE],
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col, c_col, d_col); // unused in NEON path
        shift_reduce_inner_abcd_fused_neon(
            a_packed,
            b_packed,
            c_packed,
            d_packed,
            table,
            chunk_byte_base,
            b_med,
            out,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        shift_reduce_inner_abcd_scalar(
            a_packed,
            b_packed,
            c_packed,
            d_packed,
            table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
            c_col,
            d_col,
        );
    }
}

#[allow(dead_code)]
fn shift_reduce_inner_abcd_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    table: &InvNttTableSToV8Gf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; LAMBDA4_SIZE],
    a_col: &mut [F8; V8_SIZE],
    b_col: &mut [F8; V8_SIZE],
    c_col: &mut [F8; V8_SIZE],
    d_col: &mut [F8; V8_SIZE],
) {
    let mut acc = [0u16; LAMBDA4_SIZE];
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    for k in 0..8 {
        let chunk_off = byte_base_b + k * N_CHUNKS;
        // Lookup-based NTT extension via the precomputed S→V₈ table: 8 cache-
        // resident byte lookups + an XOR chain, versus an actual inv-NTT-64 +
        // fwd-NTT-256 per factor that the direct path requires.
        table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
        table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
        table.apply(&c_packed[chunk_off..chunk_off + N_CHUNKS], c_col);
        table.apply(&d_packed[chunk_off..chunk_off + N_CHUNKS], d_col);

        // Only the Λ₄ lanes (indices 64..256) feed the round message; skip S.
        for i in 0..LAMBDA4_SIZE {
            let lane = S_SIZE + i;
            let y = (a_col[lane] * b_col[lane] * c_col[lane] * d_col[lane]).0 as u16;
            acc[i] ^= y << k;
        }
    }

    for i in 0..LAMBDA4_SIZE {
        out[i] = gf8_reduce(acc[i]);
    }
}

// ---------------------------------------------------------------------------
// Bit-transpose for z (analog of `bit_transpose_64bytes` for deg-2's C).
//
// Same math as the deg-2 transpose: reorder bits so the K-direction polynomial
// at each lane is contiguous. The deg-2 path operates on 64 packed input bytes
// indexed by (x_small=K=0..8, b_chunk=0..8), bit t. Output is 64 bytes indexed
// by (b_chunk, t), bit K.
//
// For deg-4 z, the same 64-byte transpose works at the S level — z is degree-1
// in z (linear), so its polynomial in λ has degree < 64, fully captured by the
// S evaluations. The transpose then becomes the input to the V₈ NTT extension
// (which lifts to 256 F8 values).
//
// Reuses the deg-2 `bit_transpose_64bytes` directly (just an algorithmic helper).
// ---------------------------------------------------------------------------

// (bit_transpose: now using the NEON-dispatched version from
//  `super::univariate_skip_optimized::bit_transpose_64bytes`.)

// ---------------------------------------------------------------------------
// Top-level driver (scalar, single-thread).
//
// Mirrors round1_shift_reduce_extract_c_packed_padded but for deg-4. No rayon,
// no padding-skip yet — pure correctness scaffolding for the math.
// ---------------------------------------------------------------------------

/// Compute the degree-4 round-1 prover message via shift_reduce + extract_z,
/// in scalar Rust.
///
/// Output relative to `round1_deg4_naive`:
///   `C_s · (res_ABCD[i] + res_Z[i]) = naive_p_abcd[i] + naive_p_z[i]`
///
/// Preconditions:
/// - `m >= K_SKIP + N_INNER` (= 13).
/// - `r.len() == m`. `r[K_SKIP..K_SKIP+7]` must hold the protocol-fixed small
///   + medium constants (see [`small_challenges_deg4`] /
///   [`medium_challenges_deg4`]) for the naive cross-check to line up.
/// - The packed inputs are LSB-first, `2^m / 8` bytes each.
pub fn round1_shift_reduce_extract_z_packed_deg4(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    z_packed: &[u8],
    m: usize,
    r: &[F128],
    ntts: &NttPairDeg4,
    table: &InvNttTableSToV8Gf8,
) -> (Vec<F128>, Vec<F128>) {
    assert!(m >= K_SKIP + N_INNER, "m must be ≥ K_SKIP + N_INNER (=13)");
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(d_packed.len(), total_bytes);
    assert_eq!(z_packed.len(), total_bytes);
    assert_eq!(r.len(), m);

    let convert = convert_table();
    let d_inv_val = d_inv();

    // r[K_SKIP..K_SKIP+7] = 3 small + 4 medium (verified by caller).
    // r[K_SKIP+7..] = the outer dims; eq table built from those, scaled by D⁻¹.
    let n_outer = m - K_SKIP - N_INNER;
    let eq_outer = build_eq(&r[K_SKIP + N_INNER..]);
    let eq_outer_scaled: Vec<F128> = eq_outer.iter().map(|v| *v * d_inv_val).collect();
    let big_outer_size = 1usize << n_outer;

    let mut res_abcd = [F128::ZERO; LAMBDA4_SIZE];
    // z is **linear**: accumulate on S (64 lanes), extend once at end-of-call.
    let mut res_z_on_s = [F128::ZERO; S_SIZE];

    // **16-chunk buffers** for the lane-outer convert+accumulate. abcd: 16×192 = 3 KB;
    // z (on S): 16×64 = 1 KB. Both L1-resident. Restored from the streaming
    // pattern to enable the NEON lane-outer / b_med-inner XOR-fan-in.
    let mut chunk_abcd_bytes: Vec<[u8; LAMBDA4_SIZE]> = vec![[0u8; LAMBDA4_SIZE]; 1 << N_MEDIUM];
    let mut chunk_z_bytes: Vec<[u8; S_SIZE]> = vec![[0u8; S_SIZE]; 1 << N_MEDIUM];
    let mut a_col = [F8::ZERO; V8_SIZE];
    let mut b_col = [F8::ZERO; V8_SIZE];
    let mut c_col = [F8::ZERO; V8_SIZE];
    let mut d_col = [F8::ZERO; V8_SIZE];
    let _ = ntts;

    for x_outer in 0..big_outer_size {
        let chunk_byte_base = (x_outer << N_INNER) * N_CHUNKS;
        let eq_outer_val = eq_outer_scaled[x_outer];

        // ----- Inner: fill 16 b_med chunks of abcd + z -----
        for b_med in 0..(1usize << N_MEDIUM) {
            shift_reduce_inner_abcd(
                a_packed,
                b_packed,
                c_packed,
                d_packed,
                table,
                chunk_byte_base,
                b_med,
                &mut chunk_abcd_bytes[b_med],
                &mut a_col,
                &mut b_col,
                &mut c_col,
                &mut d_col,
            );
            let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
            let z_in: &[u8; 64] = (&z_packed[byte_base_b..byte_base_b + 64])
                .try_into()
                .expect("64 z-bytes per medium position");
            bit_transpose_64bytes(z_in, &mut chunk_z_bytes[b_med]);
        }

        // ----- Convert + accumulate: lane-outer, b_med-inner XOR-fan-in -----
        // Each lane XOR-fans 16 b_med F128 values (raw byte XOR via NEON),
        // then one F128 mul by eq_outer_val. Replaces 16 scalar F128 muls.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let convert_ptr = convert.as_ptr() as *const u8;
            // abcd lanes (192 of them, on Λ₄).
            for lane in 0..LAMBDA4_SIZE {
                let mut cf = vdupq_n_u8(0);
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_abcd_bytes[b_med][lane] as usize;
                    cf = veorq_u8(cf, vld1q_u8(convert_ptr.add((b_med * 256 + v) * 16)));
                }
                let cf_u64 = vreinterpretq_u64_u8(cf);
                let cf_f = F128 {
                    lo: vgetq_lane_u64::<0>(cf_u64),
                    hi: vgetq_lane_u64::<1>(cf_u64),
                };
                res_abcd[lane] += cf_f * eq_outer_val;
            }
            // z lanes (64 of them, on S).
            for lane in 0..S_SIZE {
                let mut cf = vdupq_n_u8(0);
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_z_bytes[b_med][lane] as usize;
                    cf = veorq_u8(cf, vld1q_u8(convert_ptr.add((b_med * 256 + v) * 16)));
                }
                let cf_u64 = vreinterpretq_u64_u8(cf);
                let cf_f = F128 {
                    lo: vgetq_lane_u64::<0>(cf_u64),
                    hi: vgetq_lane_u64::<1>(cf_u64),
                };
                res_z_on_s[lane] += cf_f * eq_outer_val;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for lane in 0..LAMBDA4_SIZE {
                let mut cf = F128::ZERO;
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_abcd_bytes[b_med][lane] as usize;
                    cf += convert[b_med * 256 + v];
                }
                res_abcd[lane] += cf * eq_outer_val;
            }
            for lane in 0..S_SIZE {
                let mut cf = F128::ZERO;
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_z_bytes[b_med][lane] as usize;
                    cf += convert[b_med * 256 + v];
                }
                res_z_on_s[lane] += cf * eq_outer_val;
            }
        }
    }

    // ----- End-of-call: extend res_z_on_s from S to Λ₄ via F128 NTT -----
    let res_z_lifted = ntt_extend_f128_vec_ghash_deg4(&res_z_on_s, table);
    (res_abcd.to_vec(), res_z_lifted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::univariate_skip::pack_bits;
    use super::super::univariate_skip_deg4::round1_deg4_naive;
    use super::super::univariate_skip_optimized::c_s_f128;
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
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    /// Build the protocol r: fix the small + medium dims, randomize the outer.
    fn build_r(m: usize, rng: &mut Rng) -> Vec<F128> {
        let mut r = vec![F128::ZERO; m];
        // First K_SKIP slots are unused (consumed by univariate skip). Set to
        // arbitrary values for completeness.
        for i in 0..K_SKIP {
            r[i] = rng.f128();
        }
        // K_SKIP .. K_SKIP+3: small challenges, protocol-fixed.
        r[K_SKIP..K_SKIP + 3].copy_from_slice(&small_challenges_deg4());
        // K_SKIP+3 .. K_SKIP+7: medium challenges, protocol-fixed.
        r[K_SKIP + 3..K_SKIP + 7].copy_from_slice(&medium_challenges_deg4());
        // K_SKIP+7 .. m: outer (random).
        for i in (K_SKIP + N_INNER)..m {
            r[i] = rng.f128();
        }
        r
    }

    /// Sanity: extend_row on a deterministic input. First 64 outputs reproduce
    /// the input bits.
    #[test]
    fn extend_row_recovers_input_on_s() {
        let ntts = NttPairDeg4::new();
        let mut rng = Rng::new(0xA17);
        let input_bytes: Vec<u8> = (0..N_CHUNKS).map(|_| rng.next_u64() as u8).collect();
        let mut out = vec![F8::ZERO; V8_SIZE];
        ntts.extend_row(&input_bytes, &mut out);
        for s in 0..S_SIZE {
            let expected_bit = (input_bytes[s / 8] >> (s % 8)) & 1;
            assert_eq!(
                out[s],
                F8(expected_bit),
                "extend_row mismatch at s={s} (expected bit {expected_bit}, got {:?})",
                out[s]
            );
        }
    }

    /// Headline cross-check: `C_s · (opt_ABCD + opt_Z) == naive_ABCD + naive_Z`
    /// on the smallest m where the optimization applies (m = K_SKIP + N_INNER).
    #[test]
    fn optimized_matches_naive_modulo_cs() {
        let m = K_SKIP + N_INNER; // 13
        let mut rng = Rng::new(0xBEEF);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let d = rng.bits(1 << m);
        let z = rng.bits(1 << m);
        let r = build_r(m, &mut rng);

        let ntts = NttPairDeg4::new();
        let table = InvNttTableSToV8Gf8::new(&ntts.ntt_s, &ntts.ntt_v8);
        let (a_p, b_p, c_p, d_p, z_p) = (
            pack_bits(&a),
            pack_bits(&b),
            pack_bits(&c),
            pack_bits(&d),
            pack_bits(&z),
        );

        let (opt_abcd, opt_z) = round1_shift_reduce_extract_z_packed_deg4(
            &a_p, &b_p, &c_p, &d_p, &z_p, m, &r, &ntts, &table,
        );
        let (naive_abcd, naive_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);

        let cs = c_s_f128();
        for i in 0..LAMBDA4_SIZE {
            let lhs = cs * (opt_abcd[i] + opt_z[i]);
            let rhs = naive_abcd[i] + naive_z[i];
            assert_eq!(
                lhs, rhs,
                "round1 mismatch at i={i}: C_s·(opt_ab+opt_z) = {lhs:?}, naive sum = {rhs:?}"
            );
        }
    }

    /// Same cross-check, slightly larger m for more outer-dim coverage.
    #[test]
    fn optimized_matches_naive_modulo_cs_m14() {
        let m = K_SKIP + N_INNER + 1; // 14
        let mut rng = Rng::new(0xCAFE);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let d = rng.bits(1 << m);
        let z = rng.bits(1 << m);
        let r = build_r(m, &mut rng);

        let ntts = NttPairDeg4::new();
        let table = InvNttTableSToV8Gf8::new(&ntts.ntt_s, &ntts.ntt_v8);
        let (a_p, b_p, c_p, d_p, z_p) = (
            pack_bits(&a),
            pack_bits(&b),
            pack_bits(&c),
            pack_bits(&d),
            pack_bits(&z),
        );

        let (opt_abcd, opt_z) = round1_shift_reduce_extract_z_packed_deg4(
            &a_p, &b_p, &c_p, &d_p, &z_p, m, &r, &ntts, &table,
        );
        let (naive_abcd, naive_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);

        let cs = c_s_f128();
        for i in 0..LAMBDA4_SIZE {
            let lhs = cs * (opt_abcd[i] + opt_z[i]);
            let rhs = naive_abcd[i] + naive_z[i];
            assert_eq!(lhs, rhs, "mismatch at i={i} (m=14)");
        }
    }
}
