use crate::field::{F128, gf2_128::aarch64::ghash_mul_vec2_neon};

/// Process two butterflies at a time within a block sharing one twiddle.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        // SAFETY: caller guarantees the aes target feature.
        let product = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };
        let new_u_a = F128 {
            lo: u_a.lo ^ product[0].lo,
            hi: u_a.hi ^ product[0].hi,
        };
        let new_u_b = F128 {
            lo: u_b.lo ^ product[1].lo,
            hi: u_b.hi ^ product[1].hi,
        };
        let new_v_a = F128 {
            lo: v_a.lo ^ new_u_a.lo,
            hi: v_a.hi ^ new_u_a.hi,
        };
        let new_v_b = F128 {
            lo: v_b.lo ^ new_u_b.lo,
            hi: v_b.hi ^ new_u_b.hi,
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

/// Process the single pair in each of two adjacent blocks with distinct
/// twiddles.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    // SAFETY: caller guarantees the aes target feature.
    let product = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };
    let new_u_a = F128 {
        lo: u_a.lo ^ product[0].lo,
        hi: u_a.hi ^ product[0].hi,
    };
    let new_u_b = F128 {
        lo: u_b.lo ^ product[1].lo,
        hi: u_b.hi ^ product[1].hi,
    };
    let new_v_a = F128 {
        lo: v_a.lo ^ new_u_a.lo,
        hi: v_a.hi ^ new_u_a.hi,
    };
    let new_v_b = F128 {
        lo: v_b.lo ^ new_u_b.lo,
        hi: v_b.hi ^ new_u_b.hi,
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}

// ---------------------------------------------------------------------------
// Radix-8 (fused-3-layer) top-pass kernels for the streaming commit.
//
// Everything stays in q registers: values load once per row group, run all
// three butterfly levels through `mul_q` (the field lib's q-resident PMULL
// multiply), and store once — no F128 struct round-trips through the GPR
// file. The from-src variant additionally stages its outputs in L1 stack
// tiles and emits each destination row as one sequential non-temporal burst:
// per-lane scatter stores interleave sixteen 16 B streams spaced tens of MB
// apart, which defeats the streaming-store detector and pays a full RFO read
// on the fresh destination lines (this is what sank the first fused-3
// attempt). Ideas from the challenge tree's ranked top; re-derived here.
// ---------------------------------------------------------------------------

/// One forward butterfly, fully in q registers: `u' = u + v·t`, `v' = v + u'`.
#[inline(always)]
unsafe fn butterfly_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::veorq_u64;
    unsafe {
        let new_u = veorq_u64(u, crate::field::gf2_128::aarch64::mul_q(v, t));
        (new_u, veorq_u64(v, new_u))
    }
}

/// Zero-twiddle butterfly: `u' = u`, `v' = v + u`.
#[inline(always)]
unsafe fn butterfly_zero_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe { core::arch::aarch64::veorq_u64(v, u) }
}

/// General fused-3 chain over 8 q values (levels at strides 4, 2, 1;
/// twiddles `t[0]`, `t[1..3]`, `t[3..7]`).
#[inline(always)]
unsafe fn fused3_chain_q(
    v: &mut [core::arch::aarch64::uint64x2_t; 8],
    t: &[core::arch::aarch64::uint64x2_t; 7],
) {
    unsafe {
        for i in 0..4 {
            let (a, b) = butterfly_q(v[i], v[i + 4], t[0]);
            v[i] = a;
            v[i + 4] = b;
        }
        for s in 0..2 {
            for i in 0..2 {
                let (u, w) = (4 * s + i, 4 * s + i + 2);
                let (a, b) = butterfly_q(v[u], v[w], t[1 + s]);
                v[u] = a;
                v[w] = b;
            }
        }
        for s in 0..4 {
            let (a, b) = butterfly_q(v[2 * s], v[2 * s + 1], t[3 + s]);
            v[2 * s] = a;
            v[2 * s + 1] = b;
        }
    }
}

/// Zero-root fused-3 chain: block 0's twiddles at positions 0, 1 and 3 are
/// zero (the block-0 spine), so those butterflies are XOR-only.
#[inline(always)]
unsafe fn fused3_chain_zero_root_q(
    v: &mut [core::arch::aarch64::uint64x2_t; 8],
    t: &[core::arch::aarch64::uint64x2_t; 7],
) {
    unsafe {
        for i in 0..4 {
            v[i + 4] = butterfly_zero_q(v[i], v[i + 4]);
        }
        for i in 0..2 {
            v[i + 2] = butterfly_zero_q(v[i], v[i + 2]);
        }
        for i in 4..6 {
            let (a, b) = butterfly_q(v[i], v[i + 2], t[2]);
            v[i] = a;
            v[i + 2] = b;
        }
        v[1] = butterfly_zero_q(v[0], v[1]);
        for s in 1..4 {
            let (a, b) = butterfly_q(v[2 * s], v[2 * s + 1], t[3 + s]);
            v[2 * s] = a;
            v[2 * s + 1] = b;
        }
    }
}

/// In-place fused-3 rows `[row_start, row_end)` of one block. `zero_root`
/// selects the block-0 chain (caller guarantees `t[0]==t[1]==t[3]==0` there).
///
/// # Safety
/// Row geometry valid; concurrent calls own disjoint row ranges of this
/// block; PMULL available (cfg).
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
    zero_root: bool,
) {
    use core::arch::aarch64::{uint64x2_t, vld1q_u64, vst1q_u64};
    unsafe {
        let t: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const twiddles[i]).cast::<u64>()));
        let step = eighth * num_ntts;
        for r in row_start..row_end {
            for lane in 0..num_ntts {
                let base = ptr.add(r * num_ntts + lane);
                let mut v: [uint64x2_t; 8] =
                    core::array::from_fn(|i| vld1q_u64(base.add(i * step).cast::<u64>()));
                if zero_root {
                    fused3_chain_zero_root_q(&mut v, &t);
                } else {
                    fused3_chain_q(&mut v, &t);
                }
                for (i, value) in v.iter().enumerate() {
                    vst1q_u64(base.add(i * step).cast::<u64>(), *value);
                }
            }
        }
    }
}

/// Rate-1/2 first-pass row kernel: loads the radix-8 row group from `msg`
/// once and evaluates BOTH layer-1 blocks' fused-3 chains on those registers
/// (zero-root for `dst0` = block 0, general for `dst1` = block 1), staging
/// outputs in two L1-resident tiles and emitting each destination row's lane
/// run as one sequential non-temporal burst (see module comment).
///
/// # Safety
/// Geometry valid for all three pointers; disjoint row groups across
/// concurrent calls; `num_ntts` even and ≤ 64; `t_zero[0]==t_zero[1]==
/// t_zero[3]==0`; PMULL available.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_3layer_dual_from_src_row(
    src: *const F128,
    dst0: *mut F128,
    dst1: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    t_zero: &[F128; 7],
    t_gen: &[F128; 7],
) {
    use core::arch::aarch64::{uint64x2_t, vld1q_u64, vst1q_u64};

    #[inline(always)]
    unsafe fn store_pair_nt(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {x:q}, {y:q}, [{dst}]",
                dst = in(reg) dst,
                x = in(vreg) x,
                y = in(vreg) y,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        debug_assert!(num_ntts <= 64 && num_ntts.is_multiple_of(2));
        debug_assert_eq!(t_zero[0], F128::ZERO);
        debug_assert_eq!(t_zero[1], F128::ZERO);
        debug_assert_eq!(t_zero[3], F128::ZERO);
        let tz: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const t_zero[i]).cast::<u64>()));
        let tg: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const t_gen[i]).cast::<u64>()));

        // 8 rows × ≤64 lanes staging tiles, L1-resident.
        let mut stage0 = [F128 { lo: 0, hi: 0 }; 512];
        let mut stage1 = [F128 { lo: 0, hi: 0 }; 512];

        let off = r * num_ntts;
        let step = eighth * num_ntts;
        for lane in 0..num_ntts {
            let src_base = src.add(off + lane);
            let loaded: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(src_base.add(i * step).cast::<u64>()));
            let mut v = loaded;
            fused3_chain_zero_root_q(&mut v, &tz);
            for (i, value) in v.iter().enumerate() {
                vst1q_u64(
                    stage0.as_mut_ptr().add(i * num_ntts + lane).cast::<u64>(),
                    *value,
                );
            }
            let mut v = loaded;
            fused3_chain_q(&mut v, &tg);
            for (i, value) in v.iter().enumerate() {
                vst1q_u64(
                    stage1.as_mut_ptr().add(i * num_ntts + lane).cast::<u64>(),
                    *value,
                );
            }
        }

        for i in 0..8 {
            let s0 = stage0.as_ptr().add(i * num_ntts);
            let s1 = stage1.as_ptr().add(i * num_ntts);
            let d0 = dst0.add(off + i * step);
            let d1 = dst1.add(off + i * step);
            let mut lane = 0;
            while lane < num_ntts {
                let x = vld1q_u64(s0.add(lane).cast::<u64>());
                let y = vld1q_u64(s0.add(lane + 1).cast::<u64>());
                store_pair_nt(d0.add(lane), x, y);
                let x = vld1q_u64(s1.add(lane).cast::<u64>());
                let y = vld1q_u64(s1.add(lane + 1).cast::<u64>());
                store_pair_nt(d1.add(lane), x, y);
                lane += 2;
            }
        }
    }
}
