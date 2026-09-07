use core::arch::aarch64::{veorq_u8, vgetq_lane_u64, vld1q_u8, vreinterpretq_u64_u8};

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
/// [`fold_one_row_neon_unchecked_8`] without the final lane extraction: the
/// XOR-accumulated row stays in a q register for callers that keep computing
/// on it (the round-2 message chain). Same safety contract.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fold_one_row_neon_q_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::{veorq_u8, vld1q_u8};
    unsafe {
        const STRIDE: usize = 256 * 16;
        // One u64 load + in-register extracts instead of eight byte loads:
        // the bytes only feed gather addresses, so the extraction rides the
        // integer side and the freed load slots go to the table gathers.
        // Same mechanism as the round-1 prep word-extract (-5.1% there).
        let w = u64::from_le((bytes_ptr as *const u64).read_unaligned());
        let mut acc = vld1q_u8(table_data.add((w & 0xff) as usize * 16));
        for j in 1..8usize {
            acc = veorq_u8(
                acc,
                vld1q_u8(table_data.add(j * STRIDE + ((w >> (8 * j)) & 0xff) as usize * 16)),
            );
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
// Production callers moved to the q-returning variant; this remains as the
// extraction-included form the NEON-vs-scalar test exercises.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) unsafe fn fold_one_row_neon_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> F128 {
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
