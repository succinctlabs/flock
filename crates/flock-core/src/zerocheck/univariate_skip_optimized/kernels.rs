use super::{F8, InvNttTableByteSingleGf8};

mod portable;

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) use portable::bit_transpose_64bytes_scalar;
#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )
))]
pub(super) use portable::shift_reduce_inner_ab_scalar;

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;

#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

#[inline]
pub(super) fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON.
    unsafe {
        aarch64::bit_transpose_64bytes_neon(input, output);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))]
    // SAFETY: all target features required by the kernel are enabled.
    unsafe {
        x86_64::bit_transpose_64bytes_avx512(input, output);
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        )
    )))]
    portable::bit_transpose_64bytes_scalar(input, output);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn shift_reduce_inner_ab(
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
        let _ = (a_col, b_col);
        aarch64::shift_reduce_inner_ab_fused_neon(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        let _ = (a_col, b_col);
        // SAFETY: all required target features are enabled at compile time.
        unsafe {
            x86_64::shift_reduce_inner_ab_x86_avx512(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
            );
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    // SAFETY: gfni is enabled at compile time; SSE2 is baseline on x86_64.
    unsafe {
        x86_64::shift_reduce_inner_ab_x86_sse(
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

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )))]
    portable::shift_reduce_inner_ab_scalar(
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

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::F128; 64],
    partial_c: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // all table-selected loads.
    unsafe {
        aarch64::accumulate_convert(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::F128; 64],
    partial_c_0: &mut [super::F128; 64],
    partial_c_1: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // all table-selected loads.
    unsafe {
        aarch64::accumulate_convert_with_s_hat_v(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c_0,
            partial_c_1,
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the SIMD features and the fixed arrays
    // cover every four-lane load/store.
    unsafe {
        x86_64::accumulate_convert_with_s_hat_v_x86_avx512(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c_0,
            partial_c_1,
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )
    )))]
    portable::accumulate_convert_with_s_hat_v(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c_0,
        partial_c_1,
    );
}
