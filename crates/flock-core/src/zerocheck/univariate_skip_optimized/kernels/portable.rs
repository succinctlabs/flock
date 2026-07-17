#[cfg(not(target_arch = "aarch64"))]
use super::super::F128;
use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};
use crate::field::gf2_8::gf8_reduce;

/// Scalar bit transpose for C.
///
/// Input bytes are indexed by `(x_small * 8 + b_chunk)` with bit `t`
/// selecting a lane. Output bytes are indexed by `(b_chunk * 8 + t)` with
/// bit `x_small` selecting the inner polynomial coefficient.
#[allow(dead_code)]
pub(in super::super) fn bit_transpose_64bytes_scalar(input: &[u8; 64], output: &mut [u8; 64]) {
    output.fill(0);
    for (byte_idx, &input_byte) in input.iter().enumerate() {
        let x_small = byte_idx / 8;
        let b_chunk = byte_idx % 8;
        for t in 0..8 {
            if (input_byte >> t) & 1 != 0 {
                output[b_chunk * 8 + t] |= 1u8 << x_small;
            }
        }
    }
}

/// Scalar shift-reduce kernel and oracle for the architecture backends.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn shift_reduce_inner_ab_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    let mut acc = [0u16; 64];
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    for k in 0..8 {
        let chunk_off = byte_base_b + k * N_CHUNKS;
        inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
        inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
        for lane in 0..64 {
            let y = (a_col[lane] * b_col[lane]).0 as u16;
            acc[lane] ^= y << k;
        }
    }
    for lane in 0..64 {
        out[lane] = gf8_reduce(acc[lane]);
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c: &mut [F128; 64],
) {
    for lane in 0..64 {
        let mut converted_ab = F128::ZERO;
        let mut converted_c = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            converted_ab += convert[table_base + chunk_ab_bytes[b_med][lane] as usize];
            converted_c += convert[table_base + chunk_c_bytes[b_med][lane] as usize];
        }
        partial_ab[lane] += converted_ab * eq_lo_val;
        partial_c[lane] += converted_c * eq_lo_val;
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )
)))]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c_0: &mut [F128; 64],
    partial_c_1: &mut [F128; 64],
) {
    for lane in 0..64 {
        let mut converted_ab = F128::ZERO;
        let mut converted_c_0 = F128::ZERO;
        let mut converted_c_1 = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            let c = chunk_c_bytes[b_med][lane] as usize;
            converted_ab += convert[table_base + chunk_ab_bytes[b_med][lane] as usize];
            converted_c_0 += convert[table_base + (c & 0x55)];
            converted_c_1 += convert[table_base + (c & 0xaa)];
        }
        partial_ab[lane] += converted_ab * eq_lo_val;
        partial_c_0[lane] += converted_c_0 * eq_lo_val;
        partial_c_1[lane] += converted_c_1 * eq_lo_val;
    }
}
