//! Round-1 prover message — fully optimized (shift_reduce + extract_c, scalar).
//!
//! Scalar Rust implementation (no NEON). Three layered optimizations on top of
//! the [`super::round1_extract_c`] scaffold:
//!
//! 1. **Geometric small-eq + shift_reduce inner** (3 inner-most rest-dims).
//!    Protocol fixes the three small challenges to
//!    `r[k_skip..k_skip+3] = φ_8([0xF7, 0x53, 0xB5])`, which makes
//!    `eq_small[K] = C_s · α^K` (geometric in α, the AES root in GHASH).
//!    The shift_reduce trick computes
//!    `Σ_K eq_small[K] · φ_8(y_K)  =  C_s · φ_8(reduce(Σ_K y_K << K))`,
//!    replacing 8 F128 mults per lane with 8 u16 XOR-shifts + one F_8
//!    reduction.
//!
//! 2. **Geometric medium-eq + convert table** (4 next rest-dims).
//!    Protocol fixes the four medium challenges to
//!    `β_i = γ^{2^{i-1}} / (1 + γ^{2^{i-1}})`, which makes
//!    `eq_med[b] = γ^b / D` for `D = ∏(1+γ^{2^{i-1}})`.
//!    Precomputed table `convert[b][v] = γ^b · φ_8(v)` (64 KB) reduces the
//!    per-lane medium-eq sum from 16 F128 mults to 16 lookups + 16 XORs.
//!
//! 3. **D⁻¹ absorbed into eq_lo.**
//!    Pre-scale `eq_lo[i] ← eq_lo[i] · D⁻¹` once before the loop; this cancels
//!    the `1/D` from the medium-eq factorization, leaving only the `C_s`
//!    factor in the relative output scaling.
//!
//! Net output relationship vs the naive / structural versions:
//!   `C_s · (res_AB[i] + res_C_lifted[i])  ==  naive_p_ab[i] + naive_p_c[i]`
//! with `C_s = φ_8(0x1C)`.
//!
//! This variant is hardcoded for `k_skip = 6` (ell=64, n_chunks=8, N_INNER=7).

use std::sync::OnceLock;

use crate::field::gf2_8::gf8_reduce;
use crate::field::{F8, F128, PHI_8_TABLE, mul_by_x, phi8};
use crate::ntt::InvNttTableByteSingleGf8;

use super::PaddingSpec;
use super::univariate_skip::{SplitEqGhash, ntt_extend_f128_vec_ghash, pack_bits};

// ---------------------------------------------------------------------------
// Protocol constants — fixed by the optimization design.
// ---------------------------------------------------------------------------

/// Number of variables folded in round 1 for the shift_reduce variant.
pub const K_SKIP: usize = 6;
const ELL: usize = 64;
const N_CHUNKS: usize = 8;
/// Total inner-most dims absorbed by the optimization: 3 small + 4 medium.
const N_INNER: usize = 7;
const N_MEDIUM: usize = 4;

/// The three small-eq challenges (as F_8 values, then embedded via φ_8).
/// Choosing these specific values is what makes `eq_small[K] = C_s · α^K`.
///
/// **Soundness dependency.** These three constants — together with the
/// four medium constants returned by [`medium_challenges_ghash`] — must be
/// **F₂-linearly independent** in F₁₂₈. Zerocheck soundness relies on this
/// (a witness aligned with the friendly subspace would otherwise let the
/// prover cancel the URM message), and so does Ligerito's L0 list-collapse
/// argument (the SZ bound `(m−7)/|F|` for MLE collisions at `r` requires
/// the seven friendly coords to span a 7-dim F₂-subspace). Asserted by
/// `tests::friendly_challenges_f2_independent`.
pub const SMALL_CHAL_F8: [u8; 3] = [0xF7, 0x53, 0xB5];

/// `C_s` as an F_8 value. Verified empirically by the C++ project.
pub const C_S_F8: u8 = 0x1C;

/// The constant `C_s = φ_8(0x1C) ∈ F_{2^128}` — the relative scaling factor
/// between this optimized output and the naive output.
pub fn c_s_f128() -> F128 {
    phi8(F8(C_S_F8))
}

/// The three F_128 small challenges (embeddings of [`SMALL_CHAL_F8`]) — caller
/// must place these at `r[k_skip..k_skip+3]` for the naive cross-check to
/// produce a result related to the optimized output by exactly `C_s`.
pub fn small_challenges_ghash() -> [F128; 3] {
    [
        phi8(F8(SMALL_CHAL_F8[0])),
        phi8(F8(SMALL_CHAL_F8[1])),
        phi8(F8(SMALL_CHAL_F8[2])),
    ]
}

/// The four F_128 medium challenges `β_i = γ^{2^{i-1}} / (1 + γ^{2^{i-1}})`.
/// Caller must place these at `r[k_skip+3..k_skip+7]` for the naive
/// cross-check.
pub fn medium_challenges_ghash() -> [F128; 4] {
    let g1 = F128 {
        lo: 1u64 << 1,
        hi: 0,
    }; // γ^1
    let g2 = F128 {
        lo: 1u64 << 2,
        hi: 0,
    }; // γ^2
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    }; // γ^4
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    }; // γ^8
    [
        g1 * (F128::ONE + g1).inv(),
        g2 * (F128::ONE + g2).inv(),
        g4 * (F128::ONE + g4).inv(),
        g8 * (F128::ONE + g8).inv(),
    ]
}

/// `C_2 = (1+r_2)(1+r_3)` where `r_2 = φ_8(0x53)` (= `α^2/(1+α^2)`),
/// `r_3 = φ_8(0xB5)` (= `α^4/(1+α^4)`). This is the residual small-eq
/// constant after the first small friendly bit (`b_3[0]`, indexed by
/// `r[k_skip] = φ_8(α)`) has been pulled out for the s_hat_v_c bank split:
///
/// ```text
/// eq([r[k_skip+1], r[k_skip+2]], (b_3[1], b_3[2])) = C_2 · α^{2 b_3[1] + 4 b_3[2]}
/// ```
///
/// Used in [`round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`] to
/// post-scale the raw bank values into canonical `s_hat_v_c` (which
/// `ring_switch::fold_1b_rows` would produce against suffix `r[k_skip+1..m]`).
pub fn c_2_small_f128() -> F128 {
    let r_2 = phi8(F8(SMALL_CHAL_F8[1]));
    let r_3 = phi8(F8(SMALL_CHAL_F8[2]));
    (F128::ONE + r_2) * (F128::ONE + r_3)
}

/// `α⁻¹` in F_128, as a subfield-embedded F_8 element. Used to strip the
/// extra `α` factor from `s_hat_v_c`'s bank 1 (the K-odd lattice's raw
/// contribution is `α · α^{2 b_3[1] + 4 b_3[2]}`; canonical wants just
/// `α^{2 b_3[1] + 4 b_3[2]}`).
pub fn alpha_inv_f128() -> F128 {
    // α in F_8 = byte 0x02 (the polynomial generator). Its inverse is α^254;
    // F8::inv computes it via the standard extended Euclidean / power table.
    phi8(F8(0x02).inv())
}

/// `D = (1+γ)(1+γ^2)(1+γ^4)(1+γ^8)`; `D⁻¹` cancels the medium-eq normalization.
fn compute_d_inv() -> F128 {
    let g1 = F128 {
        lo: 1u64 << 1,
        hi: 0,
    };
    let g2 = F128 {
        lo: 1u64 << 2,
        hi: 0,
    };
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    };
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    };
    ((F128::ONE + g1) * (F128::ONE + g2) * (F128::ONE + g4) * (F128::ONE + g8)).inv()
}

static D_INV_CACHE: OnceLock<F128> = OnceLock::new();
fn d_inv() -> F128 {
    *D_INV_CACHE.get_or_init(compute_d_inv)
}

// ---------------------------------------------------------------------------
// Convert table: γ^b · φ_8(v) for b ∈ [0, 16), v ∈ [0, 256).
// 16 × 256 × 16 bytes = 64 KB. Computed once, cached via OnceLock.
// ---------------------------------------------------------------------------

const CONVERT_TABLE_SIZE: usize = 16 * 256;

static CONVERT_TABLE_CACHE: OnceLock<Vec<F128>> = OnceLock::new();

fn build_convert_table() -> Vec<F128> {
    let mut gamma_pow = [F128::ZERO; 16];
    gamma_pow[0] = F128::ONE;
    for b in 1..16 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }
    let mut table = vec![F128::ZERO; CONVERT_TABLE_SIZE];
    for b in 0..16 {
        let g_b = gamma_pow[b];
        for v in 0..256 {
            table[b * 256 + v] = g_b * PHI_8_TABLE[v];
        }
    }
    table
}

fn convert_table() -> &'static [F128] {
    CONVERT_TABLE_CACHE.get_or_init(build_convert_table)
}

// ---------------------------------------------------------------------------
// Bit transpose for C (scalar form of `bit_transpose_64bytes`).
//
// Input layout :  byte at offset (x_small * 8 + b_chunk) — bit t holds c at
//                 lane = 8*b_chunk + t with inner_K = x_small.
// Output layout:  byte at offset (b_chunk * 8 + t)        — bit K holds c at
//                 lane = 8*b_chunk + t with inner_K = K.
//
// So `out[lane]`'s 8 bits are the inner_K-direction polynomial of c at lane.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn bit_transpose_64bytes_scalar(input: &[u8; 64], output: &mut [u8; 64]) {
    output.iter_mut().for_each(|x| *x = 0);
    for byte_idx in 0..64 {
        let x_small = byte_idx / 8;
        let b_chunk = byte_idx % 8;
        for t in 0..8 {
            let bit = (input[byte_idx] >> t) & 1;
            if bit != 0 {
                output[b_chunk * 8 + t] |= 1u8 << x_small;
            }
        }
    }
}

/// NEON 64-byte bit-transpose. Two-stage:
///   1. `vqtbl4q_u8` reorders the 64 input bytes so each 8-byte group within
///      the output is one byte-chunk's worth of `x_small=0..8` bytes.
///   2. Three rounds of bit-swap at distances 7, 14, 28 across `uint64x2_t`
///      lanes do the actual 8×8 bit transpose.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bit_transpose_64bytes_neon(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::aarch64::*;

    unsafe {
        let in_ptr = input.as_ptr();
        let v0 = vld1q_u8(in_ptr);
        let v1 = vld1q_u8(in_ptr.add(16));
        let v2 = vld1q_u8(in_ptr.add(32));
        let v3 = vld1q_u8(in_ptr.add(48));
        let table = uint8x16x4_t(v0, v1, v2, v3);

        // vqtbl4q indexes that bring bytes belonging to byte-chunk b ∈ 0..8
        // into contiguous 8-byte runs, packed two-chunks-per-Q-reg.
        const IDX0: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57];
        const IDX1: [u8; 16] = [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59];
        const IDX2: [u8; 16] = [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61];
        const IDX3: [u8; 16] = [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63];

        let mut y0 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX0.as_ptr())));
        let mut y1 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX1.as_ptr())));
        let mut y2 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX2.as_ptr())));
        let mut y3 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX3.as_ptr())));

        let mask1 = vdupq_n_u64(0x00AA00AA00AA00AA);
        let mask2 = vdupq_n_u64(0x0000CCCC0000CCCC);
        let mask3 = vdupq_n_u64(0x00000000F0F0F0F0);

        // Round 1: distance 7.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<7>(y0)), mask1);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<7>(y1)), mask1);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<7>(y2)), mask1);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<7>(y3)), mask1);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<7>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<7>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<7>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<7>(t3)));

        // Round 2: distance 14.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<14>(y0)), mask2);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<14>(y1)), mask2);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<14>(y2)), mask2);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<14>(y3)), mask2);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<14>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<14>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<14>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<14>(t3)));

        // Round 3: distance 28.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<28>(y0)), mask3);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<28>(y1)), mask3);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<28>(y2)), mask3);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<28>(y3)), mask3);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<28>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<28>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<28>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<28>(t3)));

        let out_ptr = output.as_mut_ptr();
        vst1q_u8(out_ptr, vreinterpretq_u8_u64(y0));
        vst1q_u8(out_ptr.add(16), vreinterpretq_u8_u64(y1));
        vst1q_u8(out_ptr.add(32), vreinterpretq_u8_u64(y2));
        vst1q_u8(out_ptr.add(48), vreinterpretq_u8_u64(y3));
    }
}

#[inline]
pub fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON.
    unsafe {
        bit_transpose_64bytes_neon(input, output)
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is baseline on x86_64.
    unsafe {
        bit_transpose_64bytes_sse2(input, output)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    bit_transpose_64bytes_scalar(input, output);
}

#[cfg(target_arch = "x86_64")]
unsafe fn bit_transpose_64bytes_sse2(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::x86_64::*;
    unsafe {
        // Step 1: byte-level shuffle. Reorder input so each 8-byte group holds
        // one b_chunk's x_small=0..7 bytes. NEON uses vqtbl4q_u8 (64-byte cross-
        // register table lookup); SSE2 doesn't have that, so we do it in scalar.
        // 64 byte moves — fast because both input and shuffled fit in L1.
        let mut shuffled = [0u8; 64];
        for b_chunk in 0..8 {
            for x_small in 0..8 {
                shuffled[b_chunk * 8 + x_small] = input[x_small * 8 + b_chunk];
            }
        }

        // Step 2: delta-swap 8×8 bit transpose on the shuffled layout.
        // Each 8-byte group is now (shuffled[b_chunk*8 + 0..8]) = the 8 bytes
        // from x_small=0..7 for one b_chunk. The delta-swap turns each byte's
        // 8 bits into 8 output bytes' single-bit contributions.
        let sp = shuffled.as_ptr();
        let mut y0 = _mm_loadu_si128(sp as *const __m128i);
        let mut y1 = _mm_loadu_si128(sp.add(16) as *const __m128i);
        let mut y2 = _mm_loadu_si128(sp.add(32) as *const __m128i);
        let mut y3 = _mm_loadu_si128(sp.add(48) as *const __m128i);

        let mask1 = _mm_set1_epi64x(0x00AA00AA00AA00AAu64 as i64);
        let mask2 = _mm_set1_epi64x(0x0000CCCC0000CCCCu64 as i64);
        let mask3 = _mm_set1_epi64x(0x00000000F0F0F0F0u64 as i64);

        // Round 1: distance 7
        let t0 = _mm_and_si128(_mm_xor_si128(y0, _mm_srli_epi64(y0, 7)), mask1);
        let t1 = _mm_and_si128(_mm_xor_si128(y1, _mm_srli_epi64(y1, 7)), mask1);
        let t2 = _mm_and_si128(_mm_xor_si128(y2, _mm_srli_epi64(y2, 7)), mask1);
        let t3 = _mm_and_si128(_mm_xor_si128(y3, _mm_srli_epi64(y3, 7)), mask1);
        y0 = _mm_xor_si128(y0, _mm_xor_si128(t0, _mm_slli_epi64(t0, 7)));
        y1 = _mm_xor_si128(y1, _mm_xor_si128(t1, _mm_slli_epi64(t1, 7)));
        y2 = _mm_xor_si128(y2, _mm_xor_si128(t2, _mm_slli_epi64(t2, 7)));
        y3 = _mm_xor_si128(y3, _mm_xor_si128(t3, _mm_slli_epi64(t3, 7)));

        // Round 2: distance 14
        let t0 = _mm_and_si128(_mm_xor_si128(y0, _mm_srli_epi64(y0, 14)), mask2);
        let t1 = _mm_and_si128(_mm_xor_si128(y1, _mm_srli_epi64(y1, 14)), mask2);
        let t2 = _mm_and_si128(_mm_xor_si128(y2, _mm_srli_epi64(y2, 14)), mask2);
        let t3 = _mm_and_si128(_mm_xor_si128(y3, _mm_srli_epi64(y3, 14)), mask2);
        y0 = _mm_xor_si128(y0, _mm_xor_si128(t0, _mm_slli_epi64(t0, 14)));
        y1 = _mm_xor_si128(y1, _mm_xor_si128(t1, _mm_slli_epi64(t1, 14)));
        y2 = _mm_xor_si128(y2, _mm_xor_si128(t2, _mm_slli_epi64(t2, 14)));
        y3 = _mm_xor_si128(y3, _mm_xor_si128(t3, _mm_slli_epi64(t3, 14)));

        // Round 3: distance 28
        let t0 = _mm_and_si128(_mm_xor_si128(y0, _mm_srli_epi64(y0, 28)), mask3);
        let t1 = _mm_and_si128(_mm_xor_si128(y1, _mm_srli_epi64(y1, 28)), mask3);
        let t2 = _mm_and_si128(_mm_xor_si128(y2, _mm_srli_epi64(y2, 28)), mask3);
        let t3 = _mm_and_si128(_mm_xor_si128(y3, _mm_srli_epi64(y3, 28)), mask3);
        y0 = _mm_xor_si128(y0, _mm_xor_si128(t0, _mm_slli_epi64(t0, 28)));
        y1 = _mm_xor_si128(y1, _mm_xor_si128(t1, _mm_slli_epi64(t1, 28)));
        y2 = _mm_xor_si128(y2, _mm_xor_si128(t2, _mm_slli_epi64(t2, 28)));
        y3 = _mm_xor_si128(y3, _mm_xor_si128(t3, _mm_slli_epi64(t3, 28)));

        let out_ptr = output.as_mut_ptr();
        _mm_storeu_si128(out_ptr as *mut __m128i, y0);
        _mm_storeu_si128(out_ptr.add(16) as *mut __m128i, y1);
        _mm_storeu_si128(out_ptr.add(32) as *mut __m128i, y2);
        _mm_storeu_si128(out_ptr.add(48) as *mut __m128i, y3);
    }
}

// ---------------------------------------------------------------------------
// Shift_reduce inner kernel (AB only — extract_c handles C separately).
//
// For one medium-position b_med and the 8 small-positions K ∈ 0..8:
//   1. Look up NTT-extended A,B at chunk `chunk_byte_base + (b_med*8 + K)*8`.
//   2. y_K[lane] = ntt_a[lane] · ntt_b[lane]  (in F_8).
//   3. acc[lane] ^= (y_K[lane] as u16) << K   (no reduction yet).
// At the end, reduce each acc[lane] back to a u8 in F_8.
//
// Output `out[lane]` is the F_8 representative of Σ_K x^K · y_K[lane] mod p.
// ---------------------------------------------------------------------------

// Intermediate-stage NEON kernel: scalar `inv_table.apply` writing to
// `a_col`/`b_col` Vecs, then NEON `gf8_mul_vec16` from those Vecs. Superseded
// by `shift_reduce_inner_ab_fused_neon` which keeps everything register-
// resident; kept under `#[allow(dead_code)]` as a cross-check oracle.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
fn shift_reduce_inner_ab_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use crate::field::gf2_8::neon::{gf8_mul_vec16, gf8_reduce_vec16};
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // Four (lo, hi) pairs of u16x8 accumulators = 64 u16 lanes total, matching
    // the 64 lanes of the inv-NTT output.
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        // Per-K step: scalar inv-NTT apply into a_col/b_col, then NEON load +
        // 4× gf8_mul_vec16 + 8× vshll_n_u8::<K> + 8× veorq_u16 into the accs.
        // K is `const` so vshll_n_u8 specializes per call site.
        macro_rules! step_k {
            ($k:literal) => {{
                let chunk_off = byte_base_b + $k * N_CHUNKS;
                inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
                inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
                let a_ptr = a_col.as_ptr() as *const u8;
                let b_ptr = b_col.as_ptr() as *const u8;
                let y0 = gf8_mul_vec16(vld1q_u8(a_ptr), vld1q_u8(b_ptr));
                let y1 = gf8_mul_vec16(vld1q_u8(a_ptr.add(16)), vld1q_u8(b_ptr.add(16)));
                let y2 = gf8_mul_vec16(vld1q_u8(a_ptr.add(32)), vld1q_u8(b_ptr.add(32)));
                let y3 = gf8_mul_vec16(vld1q_u8(a_ptr.add(48)), vld1q_u8(b_ptr.add(48)));
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

        step_k!(0);
        step_k!(1);
        step_k!(2);
        step_k!(3);
        step_k!(4);
        step_k!(5);
        step_k!(6);
        step_k!(7);

        // Final F_8 reduction: each (acc_lo, acc_hi) pair → 16 reduced u8 values.
        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        let out_ptr = out.as_mut_ptr();
        vst1q_u8(out_ptr, r0);
        vst1q_u8(out_ptr.add(16), r1);
        vst1q_u8(out_ptr.add(32), r2);
        vst1q_u8(out_ptr.add(48), r3);
    }
}

// ---------------------------------------------------------------------------
// Fused NEON inner kernel: inv_NTT apply + F_8 mul + shift_reduce, all in
// NEON registers (no Vec<F8> round-trip).
//
// `xor_apply_byte_into_8_regs::<BH, ODD>` handles one byte position (b ≥ 1).
// `BH` (= b >> 1) selects which chunk-index XOR to apply; `ODD` (= b & 1)
// switches on the within-chunk half-swap. Both const-generic so the compiler
// dead-code-eliminates the if-branch and folds the chunk-index XORs.
//
// `fused_apply_one_k::<K>` runs one full K-row: the initial b=0 plain load,
// 7 calls to the byte helper for b=1..7 (with the specific protocol BH/ODD
// pattern), one 16-lane F_8 mul per output chunk, and finally widen-shift-XOR
// into the per-(K, lane) 16-bit accumulators.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_8_regs<const BH: usize, const ODD: bool>(
    table_base: *const u8,
    a_byte: u8,
    b_byte: u8,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
    db0: &mut core::arch::aarch64::uint8x16_t,
    db1: &mut core::arch::aarch64::uint8x16_t,
    db2: &mut core::arch::aarch64::uint8x16_t,
    db3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let ra = table_base.add(a_byte as usize * 64);
        let rb = table_base.add(b_byte as usize * 64);
        let va0 = vld1q_u8(ra.add((0 ^ BH) * 16));
        let va1 = vld1q_u8(ra.add((1 ^ BH) * 16));
        let va2 = vld1q_u8(ra.add((2 ^ BH) * 16));
        let va3 = vld1q_u8(ra.add((3 ^ BH) * 16));
        let vb0 = vld1q_u8(rb.add((0 ^ BH) * 16));
        let vb1 = vld1q_u8(rb.add((1 ^ BH) * 16));
        let vb2 = vld1q_u8(rb.add((2 ^ BH) * 16));
        let vb3 = vld1q_u8(rb.add((3 ^ BH) * 16));
        let (va0, va1, va2, va3, vb0, vb1, vb2, vb3) = if ODD {
            (
                vextq_u8::<8>(va0, va0),
                vextq_u8::<8>(va1, va1),
                vextq_u8::<8>(va2, va2),
                vextq_u8::<8>(va3, va3),
                vextq_u8::<8>(vb0, vb0),
                vextq_u8::<8>(vb1, vb1),
                vextq_u8::<8>(vb2, vb2),
                vextq_u8::<8>(vb3, vb3),
            )
        } else {
            (va0, va1, va2, va3, vb0, vb1, vb2, vb3)
        };
        *da0 = veorq_u8(*da0, va0);
        *da1 = veorq_u8(*da1, va1);
        *da2 = veorq_u8(*da2, va2);
        *da3 = veorq_u8(*da3, va3);
        *db0 = veorq_u8(*db0, vb0);
        *db1 = veorq_u8(*db1, vb1);
        *db2 = veorq_u8(*db2, vb2);
        *db3 = veorq_u8(*db3, vb3);
    }
}

/// Process one K-row: 8 byte positions of `a` and `b` via the inv_NTT table,
/// F_8 multiply, widen-shift by K, XOR into the four `(acc_lo, acc_hi)` pairs.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fused_apply_one_k<const K: i32>(
    table_base: *const u8,
    a_row: *const u8,
    b_row: *const u8,
    acc0_lo: &mut core::arch::aarch64::uint16x8_t,
    acc0_hi: &mut core::arch::aarch64::uint16x8_t,
    acc1_lo: &mut core::arch::aarch64::uint16x8_t,
    acc1_hi: &mut core::arch::aarch64::uint16x8_t,
    acc2_lo: &mut core::arch::aarch64::uint16x8_t,
    acc2_hi: &mut core::arch::aarch64::uint16x8_t,
    acc3_lo: &mut core::arch::aarch64::uint16x8_t,
    acc3_hi: &mut core::arch::aarch64::uint16x8_t,
) {
    use crate::field::gf2_8::neon::gf8_mul_vec16;
    use core::arch::aarch64::*;
    unsafe {
        // b = 0: identity permutation — plain load of the 4 chunks.
        let ra0 = table_base.add(*a_row as usize * 64);
        let rb0 = table_base.add(*b_row as usize * 64);
        let mut da0 = vld1q_u8(ra0);
        let mut da1 = vld1q_u8(ra0.add(16));
        let mut da2 = vld1q_u8(ra0.add(32));
        let mut da3 = vld1q_u8(ra0.add(48));
        let mut db0 = vld1q_u8(rb0);
        let mut db1 = vld1q_u8(rb0.add(16));
        let mut db2 = vld1q_u8(rb0.add(32));
        let mut db3 = vld1q_u8(rb0.add(48));

        // b = 1..7: XOR with table row[bytes[b]], permuted per (BH, ODD).
        xor_apply_byte_into_8_regs::<0, true>(
            table_base,
            *a_row.add(1),
            *b_row.add(1),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<1, false>(
            table_base,
            *a_row.add(2),
            *b_row.add(2),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<1, true>(
            table_base,
            *a_row.add(3),
            *b_row.add(3),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<2, false>(
            table_base,
            *a_row.add(4),
            *b_row.add(4),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<2, true>(
            table_base,
            *a_row.add(5),
            *b_row.add(5),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<3, false>(
            table_base,
            *a_row.add(6),
            *b_row.add(6),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<3, true>(
            table_base,
            *a_row.add(7),
            *b_row.add(7),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );

        // F_8 multiply lane-wise (4 × 16 lanes = 64 total).
        let y0 = gf8_mul_vec16(da0, db0);
        let y1 = gf8_mul_vec16(da1, db1);
        let y2 = gf8_mul_vec16(da2, db2);
        let y3 = gf8_mul_vec16(da3, db3);

        // Widen-shift by K, XOR into the 16-bit accumulators.
        *acc0_lo = veorq_u16(*acc0_lo, vshll_n_u8::<K>(vget_low_u8(y0)));
        *acc0_hi = veorq_u16(*acc0_hi, vshll_n_u8::<K>(vget_high_u8(y0)));
        *acc1_lo = veorq_u16(*acc1_lo, vshll_n_u8::<K>(vget_low_u8(y1)));
        *acc1_hi = veorq_u16(*acc1_hi, vshll_n_u8::<K>(vget_high_u8(y1)));
        *acc2_lo = veorq_u16(*acc2_lo, vshll_n_u8::<K>(vget_low_u8(y2)));
        *acc2_hi = veorq_u16(*acc2_hi, vshll_n_u8::<K>(vget_high_u8(y2)));
        *acc3_lo = veorq_u16(*acc3_lo, vshll_n_u8::<K>(vget_low_u8(y3)));
        *acc3_hi = veorq_u16(*acc3_hi, vshll_n_u8::<K>(vget_high_u8(y3)));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn shift_reduce_inner_ab_fused_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let table_base = inv_table.data_ptr();

    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        // 8 K-iterations — each consumes N_CHUNKS = 8 packed witness bytes
        // for `a` and `b`. K is a const generic so `vshll_n_u8::<K>` specializes.
        macro_rules! do_k {
            ($k:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                fused_apply_one_k::<$k>(
                    table_base,
                    a_packed.as_ptr().add(off),
                    b_packed.as_ptr().add(off),
                    &mut acc0_lo,
                    &mut acc0_hi,
                    &mut acc1_lo,
                    &mut acc1_hi,
                    &mut acc2_lo,
                    &mut acc2_hi,
                    &mut acc3_lo,
                    &mut acc3_hi,
                );
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

        // Reduce 16-bit accs → 16-byte F_8 results (4 × 16 lanes).
        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        let p = out.as_mut_ptr();
        vst1q_u8(p, r0);
        vst1q_u8(p.add(16), r1);
        vst1q_u8(p.add(32), r2);
        vst1q_u8(p.add(48), r3);
    }
}

/// x86_64 AVX2 version of shift_reduce_inner_ab.
///
/// Same algorithm as the scalar version, but uses AVX2 element-wise GF(2^8)
/// multiply (bitsliced schoolbook) for the 64 lane-wise products, and AVX2
/// 16-bit shift-accumulate + vectorized reduction. The 8 k-iterations are
/// macro-unrolled because `_mm256_slli_epi16` requires a compile-time constant.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shift_reduce_inner_ab_avx2(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use core::arch::x86_64::*;
    use crate::field::gf2_8::x86::gf8_mul_vec32_x86_ew;

    unsafe {
        let mut acc0 = _mm256_setzero_si256(); // lanes 0..15 (16 x u16)
        let mut acc1 = _mm256_setzero_si256(); // lanes 16..31
        let mut acc2 = _mm256_setzero_si256(); // lanes 32..47
        let mut acc3 = _mm256_setzero_si256(); // lanes 48..63

        let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

        macro_rules! do_k {
            ($k:literal) => {{
                let chunk_off = byte_base_b + $k * N_CHUNKS;
                inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
                inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);

                let a_ptr = a_col.as_ptr() as *const u8;
                let b_ptr = b_col.as_ptr() as *const u8;

                let a_lo: [u8; 32] = *(a_ptr as *const [u8; 32]);
                let b_lo: [u8; 32] = *(b_ptr as *const [u8; 32]);
                let a_hi: [u8; 32] = *(a_ptr.add(32) as *const [u8; 32]);
                let b_hi: [u8; 32] = *(b_ptr.add(32) as *const [u8; 32]);

                let y_lo = gf8_mul_vec32_x86_ew(&a_lo, &b_lo);
                let y_hi = gf8_mul_vec32_x86_ew(&a_hi, &b_hi);

                let y0 = _mm256_loadu_si256(y_lo.as_ptr() as *const _);
                let y1 = _mm256_loadu_si256(y_hi.as_ptr() as *const _);

                let y0_lo128 = _mm256_castsi256_si128(y0);
                let y0_hi128 = _mm256_extracti128_si256(y0, 1);
                let y1_lo128 = _mm256_castsi256_si128(y1);
                let y1_hi128 = _mm256_extracti128_si256(y1, 1);

                let y0_lo_u16 = _mm256_cvtepu8_epi16(y0_lo128);
                let y0_hi_u16 = _mm256_cvtepu8_epi16(y0_hi128);
                let y1_lo_u16 = _mm256_cvtepu8_epi16(y1_lo128);
                let y1_hi_u16 = _mm256_cvtepu8_epi16(y1_hi128);

                acc0 = _mm256_xor_si256(acc0, _mm256_slli_epi16(y0_lo_u16, $k));
                acc1 = _mm256_xor_si256(acc1, _mm256_slli_epi16(y0_hi_u16, $k));
                acc2 = _mm256_xor_si256(acc2, _mm256_slli_epi16(y1_lo_u16, $k));
                acc3 = _mm256_xor_si256(acc3, _mm256_slli_epi16(y1_hi_u16, $k));
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

        // Reduce: split each u16 accumulator into lo/hi bytes, then vectorized
        // GF(2^8) reduction.
        let mask_lo = _mm256_set1_epi16(0x00FF);
        let lo0 = _mm256_and_si256(acc0, mask_lo);
        let hi0 = _mm256_srli_epi16(acc0, 8);
        let lo1 = _mm256_and_si256(acc1, mask_lo);
        let hi1 = _mm256_srli_epi16(acc1, 8);
        let lo2 = _mm256_and_si256(acc2, mask_lo);
        let hi2 = _mm256_srli_epi16(acc2, 8);
        let lo3 = _mm256_and_si256(acc3, mask_lo);
        let hi3 = _mm256_srli_epi16(acc3, 8);

        // Pack 16 u16 lo/hi pairs into 32 u8 with _mm256_packus_epi16, then fix
        // the AVX2 lane-crossing with _mm256_permute4x64_epi64.
        let lo_packed_01 = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(lo0, lo1), 0b11_01_10_00
        );
        let hi_packed_01 = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(hi0, hi1), 0b11_01_10_00
        );
        let lo_packed_23 = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(lo2, lo3), 0b11_01_10_00
        );
        let hi_packed_23 = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(hi2, hi3), 0b11_01_10_00
        );

        let r01 = crate::field::gf2_8::x86::gf8_reduce_vec32(lo_packed_01, hi_packed_01);
        let r23 = crate::field::gf2_8::x86::gf8_reduce_vec32(lo_packed_23, hi_packed_23);

        let p = out.as_mut_ptr();
        _mm256_storeu_si256(p as *mut _, r01);
        _mm256_storeu_si256(p.add(32) as *mut _, r23);
    }
}

/// Dispatch helper — picks the fused NEON kernel when available, AVX2 on x86_64,
/// otherwise scalar.
#[inline]
fn shift_reduce_inner_ab(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col); // unused in the fused path
        shift_reduce_inner_ab_fused_neon(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: avx2 target_feature is enabled at compile time.
        unsafe {
            shift_reduce_inner_ab_avx2(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
                a_col,
                b_col,
            );
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2"),
    )))]
    {
        shift_reduce_inner_ab_scalar(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
        );
    }
}

/// Kept under `#[allow(dead_code)]` because on aarch64 the dispatcher only
/// reaches `_neon` — but this scalar version remains the non-aarch64 fallback
/// AND the cross-check oracle used by `neon_inner_matches_scalar_inner`.
#[allow(dead_code)]
fn shift_reduce_inner_ab_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    let mut acc: [u16; 64] = [0u16; 64];
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    for k in 0..8 {
        let chunk_off = byte_base_b + k * N_CHUNKS;
        inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
        inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
        for lane in 0..ELL {
            let y = (a_col[lane] * b_col[lane]).0 as u16;
            acc[lane] ^= y << k;
        }
    }
    for lane in 0..ELL {
        out[lane] = gf8_reduce(acc[lane]);
    }
}

// ---------------------------------------------------------------------------
// Main optimized round-1 prover message.
// ---------------------------------------------------------------------------

/// Compute the round-1 prover message via the full shift_reduce + extract_c
/// optimization, in scalar Rust.
///
/// Output relative to [`super::round1_naive`]:
///   `C_s · (res_AB[i] + res_C_lifted[i]) = naive_p_ab[i] + naive_p_c[i]`
///
/// Preconditions:
/// - `k_skip == K_SKIP` (= 6)
/// - `m >= k_skip + N_INNER` (= 13)
/// - `r.len() == m`. `r[k_skip..k_skip+7]` must hold the protocol-fixed small
///   + medium constants (see [`small_challenges_ghash`] /
///   [`medium_challenges_ghash`]) for the naive cross-check to line up. Only
///   `r[k_skip+7..m]` is used internally.
/// - `inv_table.k == k_skip`.
pub fn round1_shift_reduce_extract_c(
    a: &[bool],
    b: &[bool],
    c: &[bool],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert_eq!(c.len(), 1usize << m);
    let a_packed = pack_bits(a);
    let b_packed = pack_bits(b);
    let c_packed = pack_bits(c);
    round1_shift_reduce_extract_c_packed(&a_packed, &b_packed, &c_packed, m, k_skip, r, inv_table)
}

// Per-worker scratch + local accumulator. ~6 KB total, stack-allocated.
struct WorkerState {
    partial_ab: [F128; ELL],
    partial_c: [F128; ELL],
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    chunk_c_bytes: [[u8; 64]; 1 << N_MEDIUM],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s: [F128; ELL],
}

impl WorkerState {
    fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            partial_c: [F128::ZERO; ELL],
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            chunk_c_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s: [F128::ZERO; ELL],
        }
    }
}

/// Process one outer x_hi value: middle-loop over x_outer_lo (reset `partial_ab/c`,
/// run shift_reduce_inner + bit_transpose + convert+apply), then outer fold by
/// `eq_hi_val` into `state.local_res_ab/c_s`.
///
/// Called per-x_hi by both the parallel public function and the serial test oracle.
///
/// `within_outer_mask` and `b_med_counts` together encode the per-block padding
/// pattern (see [`PaddingSpec`]). For each x_outer, `within_hash_outer =
/// x_outer & within_outer_mask` is the position of its 8192-bit window within
/// a block, and `b_med_counts[within_hash_outer]` tells the kernel how many
/// of the 16 b_med 512-bit sub-windows are worth processing — the rest fall
/// entirely in zero padding and are skipped. Pass `within_outer_mask = 0` and
/// `b_med_counts = &[1 << N_MEDIUM]` to disable skipping.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    state: &mut WorkerState,
) {
    state.partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
    state.partial_c.iter_mut().for_each(|p| *p = F128::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;

        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        // Two paths: when n_b_med == 16 (the full case — true for every
        // x_outer_lo on the dense path, and for most of them on the padded
        // path too), use compile-time loop bounds so the SIMD XOR chain
        // unrolls. The slow path handles the rare boundary window where
        // n_b_med < 16.
        if n_b_med == (1 << N_MEDIUM) {
            for b_med in 0..(1 << N_MEDIUM) {
                shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let convert_ptr = convert.as_ptr() as *const u8;
                for lane in 0..ELL {
                    let mut cf_ab = vdupq_n_u8(0);
                    let mut cf_c = vdupq_n_u8(0);
                    for b_med in 0..(1 << N_MEDIUM) {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        cf_ab =
                            veorq_u8(cf_ab, vld1q_u8(convert_ptr.add((b_med * 256 + v_ab) * 16)));
                        cf_c = veorq_u8(cf_c, vld1q_u8(convert_ptr.add((b_med * 256 + v_c) * 16)));
                    }
                    let cf_ab_u64 = vreinterpretq_u64_u8(cf_ab);
                    let cf_c_u64 = vreinterpretq_u64_u8(cf_c);
                    let cf_ab_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_ab_u64),
                        hi: vgetq_lane_u64::<1>(cf_ab_u64),
                    };
                    let cf_c_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_c_u64),
                        hi: vgetq_lane_u64::<1>(cf_c_u64),
                    };
                    state.partial_ab[lane] += cf_ab_f * eq_lo_val;
                    state.partial_c[lane] += cf_c_f * eq_lo_val;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                let convert_ptr = convert.as_ptr() as *const u8;
                for lane in 0..ELL {
                    let mut cf_ab = unsafe { _mm_setzero_si128() };
                    let mut cf_c = unsafe { _mm_setzero_si128() };
                    for b_med in 0..(1 << N_MEDIUM) {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        unsafe {
                            cf_ab = _mm_xor_si128(cf_ab,
                                _mm_loadu_si128(convert_ptr.add((b_med * 256 + v_ab) * 16) as *const __m128i));
                            cf_c = _mm_xor_si128(cf_c,
                                _mm_loadu_si128(convert_ptr.add((b_med * 256 + v_c) * 16) as *const __m128i));
                        }
                    }
                    let cf_ab_f = unsafe {
                        F128 { lo: _mm_extract_epi64(cf_ab, 0) as u64, hi: _mm_extract_epi64(cf_ab, 1) as u64 }
                    };
                    let cf_c_f = unsafe {
                        F128 { lo: _mm_extract_epi64(cf_c, 0) as u64, hi: _mm_extract_epi64(cf_c, 1) as u64 }
                    };
                    state.partial_ab[lane] += cf_ab_f * eq_lo_val;
                    state.partial_c[lane] += cf_c_f * eq_lo_val;
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                for lane in 0..ELL {
                    let mut cf_ab = F128::ZERO;
                    let mut cf_c = F128::ZERO;
                    for b_med in 0..(1 << N_MEDIUM) {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        cf_ab += convert[b_med * 256 + v_ab];
                        cf_c += convert[b_med * 256 + v_c];
                    }
                    state.partial_ab[lane] += cf_ab * eq_lo_val;
                    state.partial_c[lane] += cf_c * eq_lo_val;
                }
            }
        } else {
            // Partial path: n_b_med ∈ (0, 1 << N_MEDIUM). At most one
            // within_hash_outer value per [`PaddingSpec`] lands here (the
            // window straddling the useful/padding boundary), so the tighter
            // loop wins despite losing the SIMD chain unroll.
            for b_med in 0..n_b_med {
                shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let convert_ptr = convert.as_ptr() as *const u8;
                for lane in 0..ELL {
                    let mut cf_ab = vdupq_n_u8(0);
                    let mut cf_c = vdupq_n_u8(0);
                    for b_med in 0..n_b_med {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        cf_ab =
                            veorq_u8(cf_ab, vld1q_u8(convert_ptr.add((b_med * 256 + v_ab) * 16)));
                        cf_c = veorq_u8(cf_c, vld1q_u8(convert_ptr.add((b_med * 256 + v_c) * 16)));
                    }
                    let cf_ab_u64 = vreinterpretq_u64_u8(cf_ab);
                    let cf_c_u64 = vreinterpretq_u64_u8(cf_c);
                    let cf_ab_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_ab_u64),
                        hi: vgetq_lane_u64::<1>(cf_ab_u64),
                    };
                    let cf_c_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_c_u64),
                        hi: vgetq_lane_u64::<1>(cf_c_u64),
                    };
                    state.partial_ab[lane] += cf_ab_f * eq_lo_val;
                    state.partial_c[lane] += cf_c_f * eq_lo_val;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                let convert_ptr = convert.as_ptr() as *const u8;
                for lane in 0..ELL {
                    let mut cf_ab = unsafe { _mm_setzero_si128() };
                    let mut cf_c = unsafe { _mm_setzero_si128() };
                    for b_med in 0..n_b_med {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        unsafe {
                            cf_ab = _mm_xor_si128(cf_ab,
                                _mm_loadu_si128(convert_ptr.add((b_med * 256 + v_ab) * 16) as *const __m128i));
                            cf_c = _mm_xor_si128(cf_c,
                                _mm_loadu_si128(convert_ptr.add((b_med * 256 + v_c) * 16) as *const __m128i));
                        }
                    }
                    let cf_ab_f = unsafe {
                        F128 { lo: _mm_extract_epi64(cf_ab, 0) as u64, hi: _mm_extract_epi64(cf_ab, 1) as u64 }
                    };
                    let cf_c_f = unsafe {
                        F128 { lo: _mm_extract_epi64(cf_c, 0) as u64, hi: _mm_extract_epi64(cf_c, 1) as u64 }
                    };
                    state.partial_ab[lane] += cf_ab_f * eq_lo_val;
                    state.partial_c[lane] += cf_c_f * eq_lo_val;
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                for lane in 0..ELL {
                    let mut cf_ab = F128::ZERO;
                    let mut cf_c = F128::ZERO;
                    for b_med in 0..n_b_med {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        cf_ab += convert[b_med * 256 + v_ab];
                        cf_c += convert[b_med * 256 + v_c];
                    }
                    state.partial_ab[lane] += cf_ab * eq_lo_val;
                    state.partial_c[lane] += cf_c * eq_lo_val;
                }
            }
        }
    }

    // Outer fold by eq_hi.
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        state.local_res_c_s[lane] += eq_hi_val * state.partial_c[lane];
    }
}

// ---------------------------------------------------------------------------
// Fusion: two-bank C accumulator that produces s_hat_v_c alongside round 1.
//
// The only structural change from `process_one_x_hi` is in the C-side inner
// loop: instead of one `cf_c` accumulator collapsing all 3 small bits, we
// keep `b_3[0]` (= bit `k_skip` of the witness, = `b_7` in ring-switch's
// packed-prefix index) as a routing dim. Two `cf_c` banks: bank 0 takes
// the K-even contributions (`v_c & 0x55`), bank 1 takes K-odd (`v_c & 0xAA`).
// By F_2-linearity of φ_8, `PHI_8(v) == PHI_8(v & 0x55) + PHI_8(v & 0xAA)`,
// so summing the two banks reconstructs the original `cf_c` → wire `res_c_s`.
//
// Per chunk-lane-b_med, this costs +1 `vld1q_u8` + +1 `veorq_u8`. Everything
// else (shift_reduce_inner_ab, bit_transpose, partial_ab/c fold, eq_hi
// outer fold) is unchanged.
// ---------------------------------------------------------------------------

/// Per-worker scratch + local accumulator for the two-bank C variant.
/// Identical to [`WorkerState`] except `partial_c` and `local_res_c_s` are
/// split into bank 0 / bank 1.
struct WorkerStateWithSHatV {
    partial_ab: [F128; ELL],
    partial_c_0: [F128; ELL],
    partial_c_1: [F128; ELL],
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    chunk_c_bytes: [[u8; 64]; 1 << N_MEDIUM],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s_0: [F128; ELL],
    local_res_c_s_1: [F128; ELL],
}

impl WorkerStateWithSHatV {
    fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            partial_c_0: [F128::ZERO; ELL],
            partial_c_1: [F128::ZERO; ELL],
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            chunk_c_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s_0: [F128::ZERO; ELL],
            local_res_c_s_1: [F128::ZERO; ELL],
        }
    }
}

/// Two-bank C variant of [`process_one_x_hi`]. AB-side and witness traffic
/// unchanged; the only modification is the C-side inner loop now maintains
/// `cf_c_0` and `cf_c_1` via masked convert-table lookups.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_s_hat_v(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    state: &mut WorkerStateWithSHatV,
) {
    state.partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
    state.partial_c_0.iter_mut().for_each(|p| *p = F128::ZERO);
    state.partial_c_1.iter_mut().for_each(|p| *p = F128::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        if n_b_med == (1 << N_MEDIUM) {
            for b_med in 0..(1 << N_MEDIUM) {
                shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let convert_ptr = convert.as_ptr() as *const u8;
                for lane in 0..ELL {
                    let mut cf_ab = vdupq_n_u8(0);
                    let mut cf_c_0 = vdupq_n_u8(0);
                    let mut cf_c_1 = vdupq_n_u8(0);
                    for b_med in 0..(1 << N_MEDIUM) {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        let v_c_0 = v_c & 0x55;
                        let v_c_1 = v_c & 0xAA;
                        cf_ab =
                            veorq_u8(cf_ab, vld1q_u8(convert_ptr.add((b_med * 256 + v_ab) * 16)));
                        cf_c_0 = veorq_u8(
                            cf_c_0,
                            vld1q_u8(convert_ptr.add((b_med * 256 + v_c_0) * 16)),
                        );
                        cf_c_1 = veorq_u8(
                            cf_c_1,
                            vld1q_u8(convert_ptr.add((b_med * 256 + v_c_1) * 16)),
                        );
                    }
                    let cf_ab_u64 = vreinterpretq_u64_u8(cf_ab);
                    let cf_c_0_u64 = vreinterpretq_u64_u8(cf_c_0);
                    let cf_c_1_u64 = vreinterpretq_u64_u8(cf_c_1);
                    let cf_ab_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_ab_u64),
                        hi: vgetq_lane_u64::<1>(cf_ab_u64),
                    };
                    let cf_c_0_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_c_0_u64),
                        hi: vgetq_lane_u64::<1>(cf_c_0_u64),
                    };
                    let cf_c_1_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_c_1_u64),
                        hi: vgetq_lane_u64::<1>(cf_c_1_u64),
                    };
                    state.partial_ab[lane] += cf_ab_f * eq_lo_val;
                    state.partial_c_0[lane] += cf_c_0_f * eq_lo_val;
                    state.partial_c_1[lane] += cf_c_1_f * eq_lo_val;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::*;
                let convert_ptr = convert.as_ptr() as *const u8;
                for lane in 0..ELL {
                    let mut cf_ab = unsafe { _mm_setzero_si128() };
                    let mut cf_c_0 = unsafe { _mm_setzero_si128() };
                    let mut cf_c_1 = unsafe { _mm_setzero_si128() };
                    for b_med in 0..(1 << N_MEDIUM) {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        let v_c_0 = v_c & 0x55;
                        let v_c_1 = v_c & 0xAA;
                        unsafe {
                            cf_ab = _mm_xor_si128(cf_ab,
                                _mm_loadu_si128(convert_ptr.add((b_med * 256 + v_ab) * 16) as *const __m128i));
                            cf_c_0 = _mm_xor_si128(cf_c_0,
                                _mm_loadu_si128(convert_ptr.add((b_med * 256 + v_c_0) * 16) as *const __m128i));
                            cf_c_1 = _mm_xor_si128(cf_c_1,
                                _mm_loadu_si128(convert_ptr.add((b_med * 256 + v_c_1) * 16) as *const __m128i));
                        }
                    }
                    let cf_ab_f = unsafe {
                        F128 { lo: _mm_extract_epi64(cf_ab, 0) as u64, hi: _mm_extract_epi64(cf_ab, 1) as u64 }
                    };
                    let cf_c_0_f = unsafe {
                        F128 { lo: _mm_extract_epi64(cf_c_0, 0) as u64, hi: _mm_extract_epi64(cf_c_0, 1) as u64 }
                    };
                    let cf_c_1_f = unsafe {
                        F128 { lo: _mm_extract_epi64(cf_c_1, 0) as u64, hi: _mm_extract_epi64(cf_c_1, 1) as u64 }
                    };
                    state.partial_ab[lane] += cf_ab_f * eq_lo_val;
                    state.partial_c_0[lane] += cf_c_0_f * eq_lo_val;
                    state.partial_c_1[lane] += cf_c_1_f * eq_lo_val;
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                for lane in 0..ELL {
                    let mut cf_ab = F128::ZERO;
                    let mut cf_c_0 = F128::ZERO;
                    let mut cf_c_1 = F128::ZERO;
                    for b_med in 0..(1 << N_MEDIUM) {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        cf_ab += convert[b_med * 256 + v_ab];
                        cf_c_0 += convert[b_med * 256 + (v_c & 0x55)];
                        cf_c_1 += convert[b_med * 256 + (v_c & 0xAA)];
                    }
                    state.partial_ab[lane] += cf_ab * eq_lo_val;
                    state.partial_c_0[lane] += cf_c_0 * eq_lo_val;
                    state.partial_c_1[lane] += cf_c_1 * eq_lo_val;
                }
            }
        } else {
            for b_med in 0..n_b_med {
                shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let convert_ptr = convert.as_ptr() as *const u8;
                for lane in 0..ELL {
                    let mut cf_ab = vdupq_n_u8(0);
                    let mut cf_c_0 = vdupq_n_u8(0);
                    let mut cf_c_1 = vdupq_n_u8(0);
                    for b_med in 0..n_b_med {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        let v_c_0 = v_c & 0x55;
                        let v_c_1 = v_c & 0xAA;
                        cf_ab =
                            veorq_u8(cf_ab, vld1q_u8(convert_ptr.add((b_med * 256 + v_ab) * 16)));
                        cf_c_0 = veorq_u8(
                            cf_c_0,
                            vld1q_u8(convert_ptr.add((b_med * 256 + v_c_0) * 16)),
                        );
                        cf_c_1 = veorq_u8(
                            cf_c_1,
                            vld1q_u8(convert_ptr.add((b_med * 256 + v_c_1) * 16)),
                        );
                    }
                    let cf_ab_u64 = vreinterpretq_u64_u8(cf_ab);
                    let cf_c_0_u64 = vreinterpretq_u64_u8(cf_c_0);
                    let cf_c_1_u64 = vreinterpretq_u64_u8(cf_c_1);
                    let cf_ab_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_ab_u64),
                        hi: vgetq_lane_u64::<1>(cf_ab_u64),
                    };
                    let cf_c_0_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_c_0_u64),
                        hi: vgetq_lane_u64::<1>(cf_c_0_u64),
                    };
                    let cf_c_1_f = F128 {
                        lo: vgetq_lane_u64::<0>(cf_c_1_u64),
                        hi: vgetq_lane_u64::<1>(cf_c_1_u64),
                    };
                    state.partial_ab[lane] += cf_ab_f * eq_lo_val;
                    state.partial_c_0[lane] += cf_c_0_f * eq_lo_val;
                    state.partial_c_1[lane] += cf_c_1_f * eq_lo_val;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for lane in 0..ELL {
                    let mut cf_ab = F128::ZERO;
                    let mut cf_c_0 = F128::ZERO;
                    let mut cf_c_1 = F128::ZERO;
                    for b_med in 0..n_b_med {
                        let v_ab = state.chunk_ab_bytes[b_med][lane] as usize;
                        let v_c = state.chunk_c_bytes[b_med][lane] as usize;
                        cf_ab += convert[b_med * 256 + v_ab];
                        cf_c_0 += convert[b_med * 256 + (v_c & 0x55)];
                        cf_c_1 += convert[b_med * 256 + (v_c & 0xAA)];
                    }
                    state.partial_ab[lane] += cf_ab * eq_lo_val;
                    state.partial_c_0[lane] += cf_c_0 * eq_lo_val;
                    state.partial_c_1[lane] += cf_c_1 * eq_lo_val;
                }
            }
        }
    }

    // Outer fold by eq_hi (per bank).
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        state.local_res_c_s_0[lane] += eq_hi_val * state.partial_c_0[lane];
        state.local_res_c_s_1[lane] += eq_hi_val * state.partial_c_1[lane];
    }
}

/// Build the `b_med_counts` table from a [`PaddingSpec`] for use by
/// [`process_one_x_hi`].
///
/// Returns `(within_outer_mask, b_med_counts)`:
///   - `within_outer_mask` masks `x_outer` to the bits identifying the
///     within-block window.
///   - `b_med_counts[w]` is how many of the 16 b_med 512-bit sub-windows of
///     window `w` we should process. Entries past the useful prefix are 0
///     (full skip) — kernels just `continue` past those x_outer_lo iterations.
fn build_b_med_counts(padding: &PaddingSpec) -> (usize, Vec<u8>) {
    const STRIDE: usize = 1 << (K_SKIP + N_INNER); // 8192 bits per within-window
    const B_MED_WINDOW: usize = 1 << (K_SKIP + 3); // 512 bits per b_med
    const N_B_MED_MAX: usize = 1 << N_MEDIUM;

    // For k_log < K_SKIP + N_INNER (= 13) the within-window granularity is
    // coarser than the block itself — skipping at this granularity would be
    // incorrect, so we fall back to "no skip". All hash modules use
    // k_log ∈ {14, 15, 16}.
    if padding.k_log < K_SKIP + N_INNER {
        return (0, vec![N_B_MED_MAX as u8]);
    }
    let within_outer_bits = padding.k_log - K_SKIP - N_INNER;
    let within_outer_count = 1usize << within_outer_bits;
    let within_outer_mask = within_outer_count - 1;
    let useful = padding.useful_bits_per_block;
    let counts: Vec<u8> = (0..within_outer_count)
        .map(|w| {
            let block_start = w * STRIDE;
            if block_start >= useful {
                0u8
            } else {
                let bits_left = useful - block_start;
                let processed = bits_left.div_ceil(B_MED_WINDOW);
                processed.min(N_B_MED_MAX) as u8
            }
        })
        .collect();
    (within_outer_mask, counts)
}

/// Packed-input variant of [`round1_shift_reduce_extract_c`]. **Parallel by
/// default** via rayon — the outer x_hi loop is distributed across workers,
/// each with its own scratch + local accumulator. Reduction is a per-lane
/// F128 XOR across workers (commutative + associative).
///
/// To run single-threaded for debugging, set `RAYON_NUM_THREADS=1`.
pub fn round1_shift_reduce_extract_c_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    round1_shift_reduce_extract_c_packed_padded(
        a_packed,
        b_packed,
        c_packed,
        m,
        k_skip,
        r,
        inv_table,
        &PaddingSpec::dense(m),
    )
}

/// Padding-aware variant of [`round1_shift_reduce_extract_c_packed`]. Skips
/// 512-bit b_med sub-windows that fall entirely in the zero padding of every
/// witness block per `padding`. Output is byte-identical to the dense path
/// when the padding bits are honestly zero.
pub fn round1_shift_reduce_extract_c_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;

    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be ≥ k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let eq_hi = &eq.hi;

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);

    // Parallel fold: each worker accumulates a subset of x_hi values into its
    // own WorkerState. Reduce step combines the per-worker `local_res_*` by
    // per-lane F128 XOR.
    let (res_ab, res_c_s) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerState::new, |mut state, x_hi| {
            let eq_hi_val = eq_hi[x_hi];
            process_one_x_hi(
                x_hi,
                big_lo_size,
                n_lo_and_inner,
                within_outer_mask,
                &b_med_counts,
                a_packed,
                b_packed,
                c_packed,
                inv_table,
                &eq_lo_scaled,
                eq_hi_val,
                convert,
                &mut state,
            );
            state
        })
        .map(|s| (s.local_res_ab, s.local_res_c_s))
        .reduce(
            || ([F128::ZERO; ELL], [F128::ZERO; ELL]),
            |(mut ab1, mut c1), (ab2, c2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                    c1[i] += c2[i];
                }
                (ab1, c1)
            },
        );

    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    (res_ab.to_vec(), res_c_lifted)
}

/// Same as [`round1_shift_reduce_extract_c_packed_padded`] but **also returns
/// `s_hat_v_c`** — the length-128 vector ring-switch would otherwise produce
/// via `fold_1b_rows` for the c-claim's PCS opening at suffix `r[k_skip+1..m]`.
///
/// The wire output `(res_ab, res_c_lifted)` is byte-identical to
/// [`round1_shift_reduce_extract_c_packed_padded`] — same eq weights, same
/// `C_s` drop convention. `s_hat_v_c` is returned in **canonical form**
/// (matches `fold_1b_rows`), with the residual `C_2` and `α⁻¹` scaling
/// applied internally so the caller can feed it straight into
/// `pcs::ring_switch::prove_batched_padded_with_precomputed`.
///
/// Cost vs the original: per chunk-lane-`b_med`, +1 `vld1q_u8` + +1 `veorq_u8`
/// (the bank-split convert lookup). bit_transpose, shift_reduce, eq folds
/// are unchanged. See module-level docs for the F_2-linearity argument that
/// makes `s_hat_v_c[(λ, 0)] + s_hat_v_c[(λ, 1)] · α == res_c_s_opt[λ]`.
pub fn round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;

    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be ≥ k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let eq_hi = &eq.hi;

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);

    let (res_ab, res_c_s_0, res_c_s_1) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerStateWithSHatV::new, |mut state, x_hi| {
            let eq_hi_val = eq_hi[x_hi];
            process_one_x_hi_with_s_hat_v(
                x_hi,
                big_lo_size,
                n_lo_and_inner,
                within_outer_mask,
                &b_med_counts,
                a_packed,
                b_packed,
                c_packed,
                inv_table,
                &eq_lo_scaled,
                eq_hi_val,
                convert,
                &mut state,
            );
            state
        })
        .map(|s| (s.local_res_ab, s.local_res_c_s_0, s.local_res_c_s_1))
        .reduce(
            || ([F128::ZERO; ELL], [F128::ZERO; ELL], [F128::ZERO; ELL]),
            |(mut ab1, mut c0_1, mut c1_1), (ab2, c0_2, c1_2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                    c0_1[i] += c0_2[i];
                    c1_1[i] += c1_2[i];
                }
                (ab1, c0_1, c1_1)
            },
        );

    // Wire output: bank_0 + bank_1 reconstructs the original `res_c_s` (by
    // F_2-linearity of φ_8 over the masked-byte sum).
    let mut res_c_s_combined = [F128::ZERO; ELL];
    for i in 0..ELL {
        res_c_s_combined[i] = res_c_s_0[i] + res_c_s_1[i];
    }
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s_combined, inv_table);

    // s_hat_v_c canonical form: apply residual C_2 (small-eq constant for
    // r[k_skip+1..k_skip+3]) and α⁻¹ (strips bank 1's extra α factor).
    let c_2 = c_2_small_f128();
    let alpha_inv = alpha_inv_f128();
    let c_2_alpha_inv = c_2 * alpha_inv;
    let mut s_hat_v_c = vec![F128::ZERO; 2 * ELL];
    for lane in 0..ELL {
        s_hat_v_c[lane] = c_2 * res_c_s_0[lane];
        s_hat_v_c[ELL + lane] = c_2_alpha_inv * res_c_s_1[lane];
    }

    (res_ab.to_vec(), res_c_lifted, s_hat_v_c)
}

/// Serial reference — same I/O as [`round1_shift_reduce_extract_c_packed`],
/// no rayon. Kept under `#[cfg(test)]` as the cross-check oracle for the
/// parallel version: future "optimizations" to the parallel path must still
/// produce identical output to this straight-line loop.
#[cfg(test)]
fn round1_shift_reduce_extract_c_packed_serial(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(k_skip, K_SKIP);
    assert!(m >= k_skip + N_INNER);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();

    let (within_outer_mask, b_med_counts) = build_b_med_counts(&PaddingSpec::dense(m));

    let mut state = WorkerState::new();
    for x_hi in 0..hi_size {
        process_one_x_hi(
            x_hi,
            big_lo_size,
            n_lo_and_inner,
            within_outer_mask,
            &b_med_counts,
            a_packed,
            b_packed,
            c_packed,
            inv_table,
            &eq_lo_scaled,
            eq.hi[x_hi],
            convert,
            &mut state,
        );
    }

    let res_c_lifted = ntt_extend_f128_vec_ghash(&state.local_res_c_s, inv_table);
    (state.local_res_ab.to_vec(), res_c_lifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::AdditiveNttGf8;
    use crate::zerocheck::univariate_skip::round1_naive;

    /// **Soundness assumption.** Zerocheck and the Ligerito PCS opening at
    /// L0 both depend on the seven "friendly" constants — three small
    /// (`φ_8(SMALL_CHAL_F8[k])`, k ∈ 0..3) and four medium
    /// (`γ^{2^i}/(1+γ^{2^i})`, i ∈ 0..4) — being **F₂-linearly independent**
    /// in F₁₂₈.
    ///
    /// Zerocheck needs this so that the prover's URM message can't be
    /// trivially canceled by a malicious witness aligned with the friendly
    /// subspace. Ligerito's L0 list-collapse argument (which leans on the
    /// zerocheck `(r, v)` claim as an OOD-equivalent) also depends on it
    /// — see the soundness writeup. If any subset of these seven values is
    /// F₂-dependent, the SZ bound `(m−7)/|F|` for collisions between
    /// distinct candidate codewords' MLEs at `r` no longer holds, and a
    /// cheating prover could engineer their witness so two candidates'
    /// MLEs agree at the friendly point with probability 1.
    ///
    /// The check: form the 7×128 binary matrix whose rows are the bit
    /// representations of the seven constants, Gauss-eliminate over F₂,
    /// assert rank = 7.
    #[test]
    fn friendly_challenges_f2_independent() {
        // Pack each F₁₂₈ element into a u128 (lo, hi → 128 bits).
        let mut basis: Vec<u128> = small_challenges_ghash()
            .iter()
            .chain(medium_challenges_ghash().iter())
            .map(|f| ((f.hi as u128) << 64) | (f.lo as u128))
            .collect();
        assert_eq!(
            basis.len(),
            7,
            "expected 3 small + 4 medium friendly values"
        );

        // Row-reduce over F₂. For each column from MSB to LSB, find a row
        // with that bit set (a pivot), swap it into place, and XOR it into
        // every other row to clear that column. Final rank = number of
        // pivots placed.
        let mut rank = 0usize;
        for col in (0..128).rev() {
            let mask = 1u128 << col;
            let pivot = (rank..basis.len()).find(|&i| basis[i] & mask != 0);
            if let Some(p) = pivot {
                basis.swap(rank, p);
                for i in 0..basis.len() {
                    if i != rank && basis[i] & mask != 0 {
                        basis[i] ^= basis[rank];
                    }
                }
                rank += 1;
            }
        }
        assert_eq!(
            rank, 7,
            "friendly challenges must be F₂-linearly independent in F₁₂₈; \
             zerocheck and Ligerito L0 soundness depend on it"
        );
    }

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
        fn bit(&mut self) -> bool {
            (self.next_u64() & 1) != 0
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.bit()).collect()
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    /// Build the full `r` vector with the protocol-fixed constants in the
    /// small/medium slots. Only `r[k_skip + N_INNER..]` is the actual
    /// randomness fed to the optimized URM.
    fn build_protocol_r(m: usize, outer: &[F128]) -> Vec<F128> {
        assert_eq!(outer.len(), m - K_SKIP - N_INNER);
        let mut r = vec![F128::ZERO; m];
        // r[0..K_SKIP]: not used by either function — can be anything.
        for (i, &small) in small_challenges_ghash().iter().enumerate() {
            r[K_SKIP + i] = small;
        }
        for (i, &med) in medium_challenges_ghash().iter().enumerate() {
            r[K_SKIP + 3 + i] = med;
        }
        for (i, &x) in outer.iter().enumerate() {
            r[K_SKIP + N_INNER + i] = x;
        }
        r
    }

    fn make_inv_table() -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    #[test]
    fn output_shape() {
        let m = 14;
        let mut rng = Rng::new(1);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let table = make_inv_table();

        let (ab, c_l) = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        assert_eq!(ab.len(), ELL);
        assert_eq!(c_l.len(), ELL);
    }

    #[test]
    fn deterministic() {
        let m = 14;
        let mut rng = Rng::new(2);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let table = make_inv_table();

        let out1 = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        let out2 = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        assert_eq!(out1, out2);
    }

    /// **The defining cross-check**: `C_s · (opt_AB + opt_C) == naive_AB + naive_C`,
    /// element-wise on Λ. Verifies all three optimization layers compose
    /// correctly — geometric small eq, geometric medium eq, and the D⁻¹
    /// pre-scaling.
    #[test]
    fn matches_naive_with_c_s_factor() {
        let c_s = c_s_f128();
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(100 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();

            let (naive_ab, naive_c) = round1_naive(&a, &b, &c, m, K_SKIP, &r);
            let (opt_ab, opt_c) = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);

            // Combined: C_s · (opt_AB + opt_C) == naive_AB + naive_C
            for i in 0..ELL {
                let lhs = naive_ab[i] + naive_c[i];
                let rhs = c_s * (opt_ab[i] + opt_c[i]);
                assert_eq!(
                    lhs, rhs,
                    "combined mismatch at m={m}, i={i}:\n  naive={lhs:?}\n  C_s·opt={rhs:?}"
                );
            }

            // Stronger: the AB and C pieces match independently (the AB-only
            // shift_reduce and the C bit_transpose both drop the same C_s).
            for i in 0..ELL {
                assert_eq!(naive_ab[i], c_s * opt_ab[i], "AB mismatch at i={i}");
                assert_eq!(naive_c[i], c_s * opt_c[i], "C mismatch at i={i}");
            }
        }
    }

    #[test]
    fn small_and_medium_challenges_sanity() {
        // Reach into the constants and verify their structural identities.
        // Medium: β_i · (1 + γ^{2^{i-1}}) == γ^{2^{i-1}}.
        let med = medium_challenges_ghash();
        let powers = [1u64 << 1, 1u64 << 2, 1u64 << 4, 1u64 << 8];
        for (i, &p) in powers.iter().enumerate() {
            let g = F128 { lo: p, hi: 0 };
            assert_eq!(med[i] * (F128::ONE + g), g, "β_{i} identity");
        }

        // D · D_inv == 1.
        let d_inv_val = d_inv();
        let g1 = F128 {
            lo: 1u64 << 1,
            hi: 0,
        };
        let g2 = F128 {
            lo: 1u64 << 2,
            hi: 0,
        };
        let g4 = F128 {
            lo: 1u64 << 4,
            hi: 0,
        };
        let g8 = F128 {
            lo: 1u64 << 8,
            hi: 0,
        };
        let d = (F128::ONE + g1) * (F128::ONE + g2) * (F128::ONE + g4) * (F128::ONE + g8);
        assert_eq!(d * d_inv_val, F128::ONE);
    }

    #[test]
    fn parallel_matches_serial() {
        use crate::zerocheck::univariate_skip::pack_bits;

        // At small m the parallel overhead dominates, but the *output* must
        // still match the serial version bit-for-bit. F128 XOR-sum reduction
        // is commutative + associative, so any thread-scheduling order yields
        // the same result.
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xCAFE_F00D + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let a_p = pack_bits(&a);
            let b_p = pack_bits(&b);
            let c_p = pack_bits(&c);

            let (par_ab, par_c) =
                round1_shift_reduce_extract_c_packed(&a_p, &b_p, &c_p, m, K_SKIP, &r, &table);
            let (ser_ab, ser_c) = round1_shift_reduce_extract_c_packed_serial(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table,
            );

            assert_eq!(par_ab, ser_ab, "parallel AB ≠ serial AB at m={m}");
            assert_eq!(par_c, ser_c, "parallel C ≠ serial C at m={m}");
        }
    }

    /// **Padding skip is byte-identical to the dense path.** On a witness
    /// where bits `[useful_bits, 2^k_log)` of every block are honestly zero,
    /// the padded URM must produce the exact same `(round1_ab, round1_c)`
    /// vectors as the dense URM — every chunk we skip would have contributed
    /// a literal zero to the dense sum (the convert table maps φ_8(0) = 0).
    ///
    /// Covers the three hash padding shapes:
    ///   - BLAKE3: k_log=14, useful=15409 → b_med_counts ≈ [16, 15]
    ///   - SHA-2:  k_log=15, useful=31401 → b_med_counts ≈ [16, 16, 16, 14]
    ///   - Keccak: k_log=16, useful=42560 → b_med_counts = [16, 16, 16, 16, 16, 4, 0, 0]
    ///     (this is the only shape that exercises the full-skip case.)
    #[test]
    fn padded_matches_dense_with_zero_padding() {
        use crate::zerocheck::PaddingSpec;
        use crate::zerocheck::univariate_skip::pack_bits;

        // (k_log, useful_bits, n_blocks_log) — pick n_blocks_log so
        // m = k_log + n_blocks_log is small enough to keep the test fast
        // while still exercising the kernel's parallel + boundary paths.
        let cases = [
            (14usize, 15_409usize, 0usize), // BLAKE3, m=14
            (15, 31_401, 0),                // SHA-2,  m=15
            (16, 42_560, 0),                // Keccak, m=16
            (16, 42_560, 3),                // Keccak, m=19 (multiple hashes)
        ];

        for (k_log, useful_bits, n_blocks_log) in cases {
            let m = k_log + n_blocks_log;
            assert!(m >= K_SKIP + N_INNER);

            let mut rng = Rng::new(0xBEEF_DEAD_u64.wrapping_add((k_log * 31 + m) as u64));
            let n_blocks = 1usize << n_blocks_log;
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;

            // Random witness, but force bits [useful_bits, 2^k_log) of every
            // block to zero (mirrors the hash-module witness layout).
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    let idx = blk * block_size + j;
                    a[idx] = false;
                    b[idx] = false;
                    c[idx] = false;
                }
            }

            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let a_p = pack_bits(&a);
            let b_p = pack_bits(&b);
            let c_p = pack_bits(&c);

            let (dense_ab, dense_c) =
                round1_shift_reduce_extract_c_packed(&a_p, &b_p, &c_p, m, K_SKIP, &r, &table);
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };
            let (padded_ab, padded_c) = round1_shift_reduce_extract_c_packed_padded(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
            );

            assert_eq!(
                dense_ab, padded_ab,
                "AB mismatch: k_log={k_log}, useful={useful_bits}, m={m}"
            );
            assert_eq!(
                dense_c, padded_c,
                "C mismatch: k_log={k_log}, useful={useful_bits}, m={m}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_bit_transpose_matches_scalar() {
        let mut rng = Rng::new(0xB17_BB17);
        for _ in 0..64 {
            let mut input = [0u8; 64];
            for byte in input.iter_mut() {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_neon = [0u8; 64];
            bit_transpose_64bytes_scalar(&input, &mut out_scalar);
            // SAFETY: on aarch64.
            unsafe { bit_transpose_64bytes_neon(&input, &mut out_neon) };
            assert_eq!(out_scalar, out_neon, "bit_transpose disagreement");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_inner_matches_scalar_inner() {
        // The new register-fused NEON kernel — verify against the same scalar
        // oracle as the intermediate one.
        let mut rng = Rng::new(0xF050D);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_fused = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_fused_neon(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_fused,
            );
            assert_eq!(
                out_scalar, out_fused,
                "fused-neon disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_inner_matches_scalar_inner() {
        // Pin down the NEON kernel directly: same inputs, same output bytes.
        let mut rng = Rng::new(0x5EED);
        let m = 14;
        let table = make_inv_table();
        let n_chunks = 1 << (K_SKIP / 8); // unused; just sanity
        let _ = n_chunks;
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        // A few representative (chunk_byte_base, b_med) values.
        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            // Guard: don't read past the witness.
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_neon = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_neon(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_neon,
                &mut a_col,
                &mut b_col,
            );
            assert_eq!(
                out_scalar, out_neon,
                "scalar/neon inner disagree at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[test]
    fn convert_table_structure() {
        // convert[b][v] == γ^b · φ_8(v); check at a handful of (b, v).
        let t = convert_table();
        let mut g_pow = F128::ONE;
        for b in 0..16 {
            for &v in &[0u8, 1, 0x57, 0xFF] {
                let expected = g_pow * PHI_8_TABLE[v as usize];
                assert_eq!(t[b * 256 + v as usize], expected, "b={b}, v={v}");
            }
            g_pow = mul_by_x(g_pow);
        }
    }

    /// The two-bank fusion variant produces `(res_ab, res_c_lifted)` that
    /// matches the existing optimized output, AND a `s_hat_v_c` that matches
    /// the scalar-oracle's canonical form.
    #[test]
    fn fusion_matches_existing_and_scalar_oracle() {
        use crate::zerocheck::univariate_skip::round1_extract_c_packed_with_s_hat_v;

        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xF00D_u64.wrapping_add(m as u64));
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let mut r = vec![F128::ZERO; m];
            // Friendly inner constants must match the optimization's
            // expectations: 3 small + 4 medium ghash.
            for i in 0..3 {
                r[K_SKIP + i] = phi8(F8(SMALL_CHAL_F8[i]));
            }
            let medium = crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash();
            for i in 0..4 {
                r[K_SKIP + 3 + i] = medium[i];
            }
            for i in 0..K_SKIP {
                r[i] = rng.f128();
            }
            for i in (K_SKIP + N_INNER)..m {
                r[i] = rng.f128();
            }

            let inv_table = {
                let ntt_s = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8::ZERO);
                let ntt_l = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
                InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
            };

            // Reference 1: existing optimized output (no s_hat_v).
            let (ref_ab, ref_c) = round1_shift_reduce_extract_c_packed_padded(
                &a,
                &b,
                &c,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &PaddingSpec::dense(m),
            );

            // Reference 2: scalar oracle (canonical s_hat_v_c).
            let (_, _, oracle_s_hat_v) =
                round1_extract_c_packed_with_s_hat_v(&a, &b, &c, m, K_SKIP, &r, &inv_table);

            // System under test.
            let (got_ab, got_c, got_s_hat_v) =
                round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                    &a,
                    &b,
                    &c,
                    m,
                    K_SKIP,
                    &r,
                    &inv_table,
                    &PaddingSpec::dense(m),
                );

            assert_eq!(got_ab, ref_ab, "res_ab mismatch at m={m}");
            assert_eq!(got_c, ref_c, "res_c_lifted mismatch at m={m}");
            assert_eq!(got_s_hat_v.len(), 2 * ELL, "s_hat_v length at m={m}");
            assert_eq!(
                got_s_hat_v, oracle_s_hat_v,
                "s_hat_v_c mismatch vs scalar oracle at m={m}"
            );
        }
    }
}
