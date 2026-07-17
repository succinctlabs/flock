use crate::field::F128;

/// NEON one-row fold: 8 aligned 16-byte loads + 8 XORs, hand-unrolled for
/// `n_chunks = 8` (the k_skip=6 protocol size). Returns the folded F128.
///
/// The table is `Vec<F128>` with each entry 16-byte aligned (F128 is
/// `repr(C, align(16))`), so every `vld1q_u8` lands on an aligned address.
///
/// # Safety
/// Caller must guarantee `table_data` points to ≥ 8 × 256 × 16 valid bytes
/// (an `n_chunks ≥ 8` table) and `bytes_ptr` to ≥ 8 valid bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fold_one_row_neon_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> F128 {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let mut acc = vld1q_u8(table_data.add((*bytes_ptr) as usize * 16));
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(1 * STRIDE + (*bytes_ptr.add(1)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(2 * STRIDE + (*bytes_ptr.add(2)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(3 * STRIDE + (*bytes_ptr.add(3)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(4 * STRIDE + (*bytes_ptr.add(4)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(5 * STRIDE + (*bytes_ptr.add(5)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(6 * STRIDE + (*bytes_ptr.add(6)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(7 * STRIDE + (*bytes_ptr.add(7)) as usize * 16)),
        );
        let acc_u64 = vreinterpretq_u64_u8(acc);
        F128 {
            lo: vgetq_lane_u64::<0>(acc_u64),
            hi: vgetq_lane_u64::<1>(acc_u64),
        }
    }
}
