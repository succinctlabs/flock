use core::arch::aarch64::{
    uint8x16_t, uint8x16x4_t, uint16x8_t, vandq_u64, vdupq_n_u8, vdupq_n_u16, vdupq_n_u64,
    veorq_u8, veorq_u16, veorq_u64, vextq_u8, vget_high_u8, vget_low_u8, vgetq_lane_u64, vld1q_u8,
    vqtbl4q_u8, vreinterpretq_u8_u16, vreinterpretq_u8_u64, vreinterpretq_u64_u8, vshll_n_u8,
    vshlq_n_u64, vshrq_n_u64, vst1q_u8,
};

use crate::{
    field::gf2_8::neon::{gf8_mul_vec16, gf8_reduce_vec16},
    zerocheck::univariate_skip_optimized::{F8, F128, InvNttTableByteSingleGf8, N_CHUNKS},
};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c: &mut [F128; 64],
) {
    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        for lane in 0..64 {
            let mut converted_ab = vdupq_n_u8(0);
            let mut converted_c = vdupq_n_u8(0);
            for b_med in 0..n_b_med {
                let ab = chunk_ab_bytes[b_med][lane] as usize;
                let c = chunk_c_bytes[b_med][lane] as usize;
                converted_ab = veorq_u8(
                    converted_ab,
                    vld1q_u8(convert_ptr.add((b_med * 256 + ab) * 16)),
                );
                converted_c = veorq_u8(
                    converted_c,
                    vld1q_u8(convert_ptr.add((b_med * 256 + c) * 16)),
                );
            }
            let ab = vreinterpretq_u64_u8(converted_ab);
            let c = vreinterpretq_u64_u8(converted_c);
            partial_ab[lane] += F128 {
                lo: vgetq_lane_u64::<0>(ab),
                hi: vgetq_lane_u64::<1>(ab),
            } * eq_lo_val;
            partial_c[lane] += F128 {
                lo: vgetq_lane_u64::<0>(c),
                hi: vgetq_lane_u64::<1>(c),
            } * eq_lo_val;
        }
    }
}

/// AB-only drain, two lanes per iteration (see the with_s_hat_v variant for
/// why two): used when the C banks come from the lincheck-stripe fold.
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_ab_only(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
) {
    use core::arch::aarch64::{
        vdupq_n_u8, veorq_u8, vgetq_lane_u64, vld1q_u8, vreinterpretq_u64_u8,
    };
    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        // Four lanes per iteration: with the C side gone each lane carries a
        // single XOR chain of depth n_b_med, so two lanes expose only two
        // chains -- four keeps enough independent gathers in flight to cover
        // the L1 load latency (same mechanism as the earlier two-lane win).
        let mut lane = 0usize;
        while lane + 4 <= 64 {
            let mut acc = [vdupq_n_u8(0); 4];
            for b_med in 0..n_b_med {
                let base = b_med * 256;
                for j in 0..4 {
                    let byte = chunk_ab_bytes[b_med][lane + j] as usize;
                    acc[j] = veorq_u8(acc[j], vld1q_u8(convert_ptr.add((base + byte) * 16)));
                }
            }
            for j in 0..4 {
                let v = vreinterpretq_u64_u8(acc[j]);
                partial_ab[lane + j] += F128 {
                    lo: vgetq_lane_u64::<0>(v),
                    hi: vgetq_lane_u64::<1>(v),
                } * eq_lo_val;
            }
            lane += 4;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c_0: &mut [F128; 64],
    partial_c_1: &mut [F128; 64],
) {
    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        // Two lanes per iteration. Each lane carries three XOR chains of depth
        // n_b_med (16 at the ranked shape), and the chains are serial even
        // though the gathers feeding them are independent -- so a single lane
        // exposes only three chains to the out-of-order engine. Interleaving a
        // second lane doubles that to six without changing the work. The
        // all-ones experiment showed this kernel family is sensitive to how
        // much independent work is in flight, and the drain is gather-bound,
        // so both halves of the trade point the same way here.
        let mut lane = 0usize;
        while lane + 2 <= 64 {
            let l0 = lane;
            let l1 = lane + 1;
            let mut ab_0 = vdupq_n_u8(0);
            let mut c0_0 = vdupq_n_u8(0);
            let mut c1_0 = vdupq_n_u8(0);
            let mut ab_1 = vdupq_n_u8(0);
            let mut c0_1 = vdupq_n_u8(0);
            let mut c1_1 = vdupq_n_u8(0);
            for b_med in 0..n_b_med {
                let base = b_med * 256;
                let a0 = chunk_ab_bytes[b_med][l0] as usize;
                let x0 = chunk_c_bytes[b_med][l0] as usize;
                let a1 = chunk_ab_bytes[b_med][l1] as usize;
                let x1 = chunk_c_bytes[b_med][l1] as usize;
                ab_0 = veorq_u8(ab_0, vld1q_u8(convert_ptr.add((base + a0) * 16)));
                c0_0 = veorq_u8(c0_0, vld1q_u8(convert_ptr.add((base + (x0 & 0x55)) * 16)));
                c1_0 = veorq_u8(c1_0, vld1q_u8(convert_ptr.add((base + (x0 & 0xaa)) * 16)));
                ab_1 = veorq_u8(ab_1, vld1q_u8(convert_ptr.add((base + a1) * 16)));
                c0_1 = veorq_u8(c0_1, vld1q_u8(convert_ptr.add((base + (x1 & 0x55)) * 16)));
                c1_1 = veorq_u8(c1_1, vld1q_u8(convert_ptr.add((base + (x1 & 0xaa)) * 16)));
            }
            macro_rules! drain {
                ($acc:expr, $dst:expr, $l:expr) => {{
                    let v = vreinterpretq_u64_u8($acc);
                    $dst[$l] += F128 {
                        lo: vgetq_lane_u64::<0>(v),
                        hi: vgetq_lane_u64::<1>(v),
                    } * eq_lo_val;
                }};
            }
            drain!(ab_0, partial_ab, l0);
            drain!(c0_0, partial_c_0, l0);
            drain!(c1_0, partial_c_1, l0);
            drain!(ab_1, partial_ab, l1);
            drain!(c0_1, partial_c_0, l1);
            drain!(c1_1, partial_c_1, l1);
            lane += 2;
        }
        #[allow(clippy::needless_range_loop)]
        for lane in lane..64 {
            let mut converted_ab = vdupq_n_u8(0);
            let mut converted_c_0 = vdupq_n_u8(0);
            let mut converted_c_1 = vdupq_n_u8(0);
            for b_med in 0..n_b_med {
                let ab = chunk_ab_bytes[b_med][lane] as usize;
                let c = chunk_c_bytes[b_med][lane] as usize;
                converted_ab = veorq_u8(
                    converted_ab,
                    vld1q_u8(convert_ptr.add((b_med * 256 + ab) * 16)),
                );
                converted_c_0 = veorq_u8(
                    converted_c_0,
                    vld1q_u8(convert_ptr.add((b_med * 256 + (c & 0x55)) * 16)),
                );
                converted_c_1 = veorq_u8(
                    converted_c_1,
                    vld1q_u8(convert_ptr.add((b_med * 256 + (c & 0xaa)) * 16)),
                );
            }
            let ab = vreinterpretq_u64_u8(converted_ab);
            let c_0 = vreinterpretq_u64_u8(converted_c_0);
            let c_1 = vreinterpretq_u64_u8(converted_c_1);
            partial_ab[lane] += F128 {
                lo: vgetq_lane_u64::<0>(ab),
                hi: vgetq_lane_u64::<1>(ab),
            } * eq_lo_val;
            partial_c_0[lane] += F128 {
                lo: vgetq_lane_u64::<0>(c_0),
                hi: vgetq_lane_u64::<1>(c_0),
            } * eq_lo_val;
            partial_c_1[lane] += F128 {
                lo: vgetq_lane_u64::<0>(c_1),
                hi: vgetq_lane_u64::<1>(c_1),
            } * eq_lo_val;
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
pub(crate) unsafe fn bit_transpose_64bytes_neon(input: &[u8; 64], output: &mut [u8; 64]) {
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

// Intermediate-stage NEON kernel: scalar `inv_table.apply` writing to
// `a_col`/`b_col` Vecs, then NEON `gf8_mul_vec16` from those Vecs. Superseded
// by `shift_reduce_inner_ab_fused_neon` which keeps everything register-
// resident; kept under `#[allow(dead_code)]` as a cross-check oracle.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub(crate) fn shift_reduce_inner_ab_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
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

/// Unreduced carry-less products of the 8 low / high lanes (raw PMULL).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_lo_u16(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint16x8_t {
    use core::arch::aarch64::{poly8x8_t, uint8x8_t, vget_low_u8, vmull_p8, vreinterpretq_u16_p16};
    unsafe {
        vreinterpretq_u16_p16(vmull_p8(
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(a)),
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(b)),
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_hi_u16(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint16x8_t {
    use core::arch::aarch64::{
        poly8x8_t, uint8x8_t, vget_high_u8, vmull_p8, vreinterpretq_u16_p16,
    };
    unsafe {
        vreinterpretq_u16_p16(vmull_p8(
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(a)),
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(b)),
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_8_regs<const BH: usize, const ODD: bool>(
    a_table: *const u8,
    b_table: *const u8,
    a_byte: u8,
    b_byte: u8,
    da0: &mut uint8x16_t,
    da1: &mut uint8x16_t,
    da2: &mut uint8x16_t,
    da3: &mut uint8x16_t,
    db0: &mut uint8x16_t,
    db1: &mut uint8x16_t,
    db2: &mut uint8x16_t,
    db3: &mut uint8x16_t,
) {
    unsafe {
        let ra = a_table.add(a_byte as usize * 64);
        let rb = b_table.add(b_byte as usize * 64);
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
    a_table: *const u8,
    table_base: *const u8,
    a_row: *const u8,
    b_row: *const u8,
    acc0_lo: &mut uint16x8_t,
    acc0_hi: &mut uint16x8_t,
    acc1_lo: &mut uint16x8_t,
    acc1_hi: &mut uint16x8_t,
    acc2_lo: &mut uint16x8_t,
    acc2_hi: &mut uint16x8_t,
    acc3_lo: &mut uint16x8_t,
    acc3_hi: &mut uint16x8_t,
) {
    use core::arch::aarch64::{veorq_u16, vld1q_u8, vshlq_n_u16};
    unsafe {
        // Structurally-zero b row: the BLAKE3 circuit pins ~6% of the b
        // operand's 8-byte K-rows to zero (structural zeros of the linear
        // constraints), and a census over 256 word positions x 256 blocks x 3
        // independent witnesses finds them at fixed positions. The inv-NTT
        // transform is F_2-linear so row(0) = 0, hence db_* = 0, hence
        // y_* = gf8_mul(da_*, 0) = 0 and this K-row contributes nothing to any
        // accumulator. Skipping it is exact, and the guard is a compare -- the
        // kernel stays correct for any witness that disagrees.
        let bw = u64::from_le((b_row as *const u64).read_unaligned());
        if bw == 0 {
            return;
        }
        // Read each operand row as ONE word and extract bytes in-register:
        // the byte values only feed table-address arithmetic, so this trades
        // 16 L1 byte-loads per K-row for 2 word loads plus shifts on the
        // 6-wide integer side, freeing load-issue slots for the row gathers.
        let aw = u64::from_le((a_row as *const u64).read_unaligned());
        // b = 0: identity permutation — plain load of the 4 chunks.
        let ra0 = a_table.add((aw & 0xff) as usize * 64);
        let rb0 = table_base.add((bw & 0xff) as usize * 64);
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
            a_table,
            table_base,
            (aw >> 8) as u8,
            (bw >> 8) as u8,
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
            a_table,
            table_base,
            (aw >> 16) as u8,
            (bw >> 16) as u8,
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
            a_table,
            table_base,
            (aw >> 24) as u8,
            (bw >> 24) as u8,
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
            a_table,
            table_base,
            (aw >> 32) as u8,
            (bw >> 32) as u8,
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
            a_table,
            table_base,
            (aw >> 40) as u8,
            (bw >> 40) as u8,
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
            a_table,
            table_base,
            (aw >> 48) as u8,
            (bw >> 48) as u8,
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
            a_table,
            table_base,
            (aw >> 56) as u8,
            (bw >> 56) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );

        // Accumulate the UNREDUCED products, decomposing the x^K row weight
        // as x^4 (the caller passed the x^4-scaled gather table for K >= 4)
        // * x^2 (cheap byte-wise multiply on the reduced a operand)
        // * x^(K&1) (a u16 shift; both reducers are exact over the full
        // 16-bit domain -- gf2_8::tests::*_reduce_full_u16_domain). This
        // deletes the per-K gf8_mul_vec16 reduction (4 of its 6 PMULLs),
        // keeping only the 2 raw product PMULLs; reduction happens once per
        // block in gf8_reduce_vec16 at the end, unchanged.
        let (da0, da1, da2, da3) = if (K >> 1) & 1 == 1 {
            use crate::field::gf2_8::neon::gf8_mul_x2_vec16;
            (
                gf8_mul_x2_vec16(da0),
                gf8_mul_x2_vec16(da1),
                gf8_mul_x2_vec16(da2),
                gf8_mul_x2_vec16(da3),
            )
        } else {
            (da0, da1, da2, da3)
        };
        macro_rules! absorb {
            ($acc:expr, $p:expr) => {{
                let p = $p;
                if K & 1 == 1 {
                    *$acc = veorq_u16(*$acc, vshlq_n_u16::<1>(p));
                } else {
                    *$acc = veorq_u16(*$acc, p);
                }
            }};
        }
        absorb!(acc0_lo, pmull_lo_u16(da0, db0));
        absorb!(acc0_hi, pmull_hi_u16(da0, db0));
        absorb!(acc1_lo, pmull_lo_u16(da1, db1));
        absorb!(acc1_hi, pmull_hi_u16(da1, db1));
        absorb!(acc2_lo, pmull_lo_u16(da2, db2));
        absorb!(acc2_hi, pmull_hi_u16(da2, db2));
        absorb!(acc3_lo, pmull_lo_u16(da3, db3));
        absorb!(acc3_hi, pmull_hi_u16(da3, db3));
    }
}

/// b ≡ 1 shortcut: `y_K = ntt_a`, so each K-row is one a-transform,
/// weight-decomposed exactly like [`fused_apply_one_k`] (x^4 via the scaled
/// table for K ≥ 4, x^2 via the byte multiply, x^(K&1) via a u16 shift),
/// widened and XOR-accumulated with NO product multiplies and NO b gathers.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_a_only_const_b(
    a_packed: &[u8],
    table_base: *const u8,
    table_x4: *const u8,
    byte_base_b: usize,
    out: &mut [u8; 64],
) {
    use crate::field::gf2_8::neon::{gf8_mul_x2_vec16, gf8_reduce_vec16};
    use core::arch::aarch64::{
        vdupq_n_u16, veorq_u16, vget_high_u8, vget_low_u8, vreinterpretq_u8_u16, vshll_n_u8,
        vshlq_n_u16, vst1q_u8,
    };
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);
        macro_rules! do_k {
            ($k:literal) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(
                    a_packed.as_ptr().add(byte_base_b + $k * 8).cast::<u64>(),
                ));
                if aw != 0 {
                    let (d0, d1, d2, d3) =
                        apply_word_into_4_regs(if $k >= 4 { table_x4 } else { table_base }, aw);
                    let (d0, d1, d2, d3) = if ($k >> 1) & 1 == 1 {
                        (
                            gf8_mul_x2_vec16(d0),
                            gf8_mul_x2_vec16(d1),
                            gf8_mul_x2_vec16(d2),
                            gf8_mul_x2_vec16(d3),
                        )
                    } else {
                        (d0, d1, d2, d3)
                    };
                    macro_rules! absorb {
                        ($acc:expr, $half:expr) => {{
                            let widened = vshll_n_u8::<0>($half);
                            if $k & 1 == 1 {
                                *$acc = veorq_u16(*$acc, vshlq_n_u16::<1>(widened));
                            } else {
                                *$acc = veorq_u16(*$acc, widened);
                            }
                        }};
                    }
                    absorb!(&mut acc0_lo, vget_low_u8(d0));
                    absorb!(&mut acc0_hi, vget_high_u8(d0));
                    absorb!(&mut acc1_lo, vget_low_u8(d1));
                    absorb!(&mut acc1_hi, vget_high_u8(d1));
                    absorb!(&mut acc2_lo, vget_low_u8(d2));
                    absorb!(&mut acc2_hi, vget_high_u8(d2));
                    absorb!(&mut acc3_lo, vget_low_u8(d3));
                    absorb!(&mut acc3_hi, vget_high_u8(d3));
                }
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

/// Single-live-K0 shortcut: K-rows 1..7 contribute nothing, so the block is
/// one dual transform + one lane-wise F_8 multiply (K = 0: base table, no
/// weight decomposition, no shift).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_single_k0(table_base: *const u8, aw: u64, bw0: u64, out: &mut [u8; 64]) {
    use crate::field::gf2_8::neon::gf8_mul_vec16;
    use core::arch::aarch64::vst1q_u8;
    unsafe {
        let (a0, a1, a2, a3) = apply_word_into_4_regs(table_base, aw);
        let (b0, b1, b2, b3) = apply_word_into_4_regs(table_base, bw0);
        let y0 = gf8_mul_vec16(a0, b0);
        let y1 = gf8_mul_vec16(a1, b1);
        let y2 = gf8_mul_vec16(a2, b2);
        let y3 = gf8_mul_vec16(a3, b3);
        let p = out.as_mut_ptr();
        vst1q_u8(p, y0);
        vst1q_u8(p.add(16), y1);
        vst1q_u8(p.add(32), y2);
        vst1q_u8(p.add(48), y3);
    }
}

/// Single-operand sibling of [`xor_apply_byte_into_8_regs`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_4_regs<const BH: usize, const ODD: bool>(
    table: *const u8,
    byte: u8,
    d0: &mut core::arch::aarch64::uint8x16_t,
    d1: &mut core::arch::aarch64::uint8x16_t,
    d2: &mut core::arch::aarch64::uint8x16_t,
    d3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::{veorq_u8, vextq_u8, vld1q_u8};
    unsafe {
        let r = table.add(byte as usize * 64);
        let v0 = vld1q_u8(r.add((0 ^ BH) * 16));
        let v1 = vld1q_u8(r.add((1 ^ BH) * 16));
        let v2 = vld1q_u8(r.add((2 ^ BH) * 16));
        let v3 = vld1q_u8(r.add((3 ^ BH) * 16));
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

/// Apply one 8-byte packed row through the inv-NTT table into four 16-lane
/// registers (the a-side of [`fused_apply_one_k`], single operand).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_into_4_regs(
    table: *const u8,
    word: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::vld1q_u8;
    unsafe {
        let r0 = table.add((word & 0xff) as usize * 64);
        let mut d0 = vld1q_u8(r0);
        let mut d1 = vld1q_u8(r0.add(16));
        let mut d2 = vld1q_u8(r0.add(32));
        let mut d3 = vld1q_u8(r0.add(48));
        xor_apply_byte_into_4_regs::<0, true>(
            table,
            (word >> 8) as u8,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<1, false>(
            table,
            (word >> 16) as u8,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<1, true>(
            table,
            (word >> 24) as u8,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<2, false>(
            table,
            (word >> 32) as u8,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<2, true>(
            table,
            (word >> 40) as u8,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<3, false>(
            table,
            (word >> 48) as u8,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<3, true>(
            table,
            (word >> 56) as u8,
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        (d0, d1, d2, d3)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn shift_reduce_inner_ab_fused_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let table_base = inv_table.data_ptr();
    let table_x4 = inv_table.data_x4_ptr();

    unsafe {
        // Structured-b shortcuts (idea from the challenge tree; exact
        // dispatch on runtime row content, no circuit assumptions):
        //  * b ≡ 1 over the whole 8-K block — the NTT-extension of the
        //    constant-one row is constant one, so y_K = ntt_a: skip every
        //    b gather and every product multiply, shift-accumulate the a
        //    transforms directly.
        //  * only the K = 0 word nonzero — K-rows 1..7 have b = 0 and
        //    contribute nothing (row(0) = 0 by F2-linearity): compute the
        //    single K = 0 term.
        let bw = |k: usize| -> u64 {
            u64::from_le(core::ptr::read_unaligned(
                b_packed.as_ptr().add(byte_base_b + k * 8).cast::<u64>(),
            ))
        };
        let and_all = bw(0) & bw(1) & bw(2) & bw(3) & bw(4) & bw(5) & bw(6) & bw(7);
        if and_all == u64::MAX {
            shift_reduce_inner_a_only_const_b(a_packed, table_base, table_x4, byte_base_b, out);
            return;
        }
        if (bw(1) | bw(2) | bw(3) | bw(4) | bw(5) | bw(6) | bw(7)) == 0 {
            let aw = u64::from_le(core::ptr::read_unaligned(
                a_packed.as_ptr().add(byte_base_b).cast::<u64>(),
            ));
            shift_reduce_inner_single_k0(table_base, aw, bw(0), out);
            return;
        }

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
                    if $k >= 4 { table_x4 } else { table_base },
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
