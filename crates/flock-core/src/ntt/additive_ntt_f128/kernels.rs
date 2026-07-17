//! Compile-time-selected leaf kernels for the F128 additive NTT.
//!
//! Transform scheduling and cache-blocking policy stay in the parent module;
//! this module owns the architecture-specific operations on blocks of data.

use crate::field::F128;

mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod x86_64;

#[inline]
pub(super) fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    debug_assert_eq!(top.len(), bot.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_row_pair(top, bot, twiddle);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_row_pair(top, bot, twiddle);
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), d.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

/// Process one fused-four-layer row group across every interleaved NTT lane.
///
/// # Safety
/// The caller must ensure the 16 row slices selected by `r` are valid and
/// disjoint from any row group being processed concurrently.
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: target features are guaranteed by cfg; the caller owns the row
    // geometry and disjointness contract.
    unsafe {
        x86_64::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, r, twiddles);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, r, twiddles);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { aarch64::butterfly_block(chunk, twiddle, half) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair(
    data: &mut [F128],
    base: usize,
    t_a: F128,
    t_b: F128,
) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { aarch64::butterfly_block_pair(&mut data[base..base + 4], t_a, t_b) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair_chunk(chunk: &mut [F128], t_a: F128, t_b: F128) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { aarch64::butterfly_block_pair(chunk, t_a, t_b) }
}
