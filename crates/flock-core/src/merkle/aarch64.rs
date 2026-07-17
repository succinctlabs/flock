use super::{Hash, SHA256_IV, SHA256_K};
use core::arch::aarch64::*;

/// One interleaved compression round over 4 independent states.
/// `blocks[i]` must be ≥ 64 bytes; only the first 64 are consumed.
#[inline(always)]
unsafe fn compress4(
    abcd: &mut [uint32x4_t; 4],
    efgh: &mut [uint32x4_t; 4],
    blocks: [*const u8; 4],
) {
    unsafe {
        let mut msg0 = [vdupq_n_u32(0); 4];
        let mut msg1 = [vdupq_n_u32(0); 4];
        let mut msg2 = [vdupq_n_u32(0); 4];
        let mut msg3 = [vdupq_n_u32(0); 4];
        for i in 0..4 {
            // SHA-256 message words are big-endian.
            msg0[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i])));
            msg1[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(16))));
            msg2[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(32))));
            msg3[i] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blocks[i].add(48))));
        }
        let abcd_save = *abcd;
        let efgh_save = *efgh;

        macro_rules! rounds4 {
            ($msg:expr, $ki:expr) => {{
                let kv = vld1q_u32(SHA256_K.as_ptr().add($ki));
                for i in 0..4 {
                    let wk = vaddq_u32($msg[i], kv);
                    let t = abcd[i];
                    abcd[i] = vsha256hq_u32(abcd[i], efgh[i], wk);
                    efgh[i] = vsha256h2q_u32(efgh[i], t, wk);
                }
            }};
        }
        macro_rules! sched {
            ($m0:expr, $m1:expr, $m2:expr, $m3:expr) => {
                for i in 0..4 {
                    $m0[i] = vsha256su1q_u32(vsha256su0q_u32($m0[i], $m1[i]), $m2[i], $m3[i]);
                }
            };
        }

        rounds4!(msg0, 0);
        rounds4!(msg1, 4);
        rounds4!(msg2, 8);
        rounds4!(msg3, 12);
        for r in 1..4 {
            sched!(msg0, msg1, msg2, msg3);
            sched!(msg1, msg2, msg3, msg0);
            sched!(msg2, msg3, msg0, msg1);
            sched!(msg3, msg0, msg1, msg2);
            rounds4!(msg0, 16 * r);
            rounds4!(msg1, 16 * r + 4);
            rounds4!(msg2, 16 * r + 8);
            rounds4!(msg3, 16 * r + 12);
        }
        for i in 0..4 {
            abcd[i] = vaddq_u32(abcd[i], abcd_save[i]);
            efgh[i] = vaddq_u32(efgh[i], efgh_save[i]);
        }
    }
}

/// Hash 4 equal-length inputs, producing 4 standard SHA-256 digests.
#[inline]
pub fn hash4_equal_len(inputs: [&[u8]; 4], out: &mut [Hash]) {
    let len = inputs[0].len();
    debug_assert!(inputs.iter().all(|x| x.len() == len));
    debug_assert!(out.len() >= 4);

    unsafe {
        let mut abcd = [vld1q_u32(SHA256_IV.as_ptr()); 4];
        let mut efgh = [vld1q_u32(SHA256_IV.as_ptr().add(4)); 4];

        // Full 64-byte blocks.
        let n_full = len / 64;
        for blk in 0..n_full {
            compress4(
                &mut abcd,
                &mut efgh,
                [
                    inputs[0].as_ptr().add(blk * 64),
                    inputs[1].as_ptr().add(blk * 64),
                    inputs[2].as_ptr().add(blk * 64),
                    inputs[3].as_ptr().add(blk * 64),
                ],
            );
        }

        // Tail: remaining bytes + 0x80 + zero pad + 64-bit BE bit length.
        // One extra block when rem ≤ 55, two when 56 ≤ rem ≤ 63.
        let rem = len % 64;
        let bit_len = (len as u64) * 8;
        let n_tail = if rem < 56 { 1 } else { 2 };
        let mut tails = [[0u8; 128]; 4];
        for i in 0..4 {
            tails[i][..rem].copy_from_slice(&inputs[i][len - rem..]);
            tails[i][rem] = 0x80;
            tails[i][n_tail * 64 - 8..n_tail * 64].copy_from_slice(&bit_len.to_be_bytes());
        }
        for blk in 0..n_tail {
            compress4(
                &mut abcd,
                &mut efgh,
                [
                    tails[0].as_ptr().add(blk * 64),
                    tails[1].as_ptr().add(blk * 64),
                    tails[2].as_ptr().add(blk * 64),
                    tails[3].as_ptr().add(blk * 64),
                ],
            );
        }

        // Digest = big-endian a..h.
        for i in 0..4 {
            let be_lo = vrev32q_u8(vreinterpretq_u8_u32(abcd[i]));
            let be_hi = vrev32q_u8(vreinterpretq_u8_u32(efgh[i]));
            vst1q_u8(out[i].as_mut_ptr(), be_lo);
            vst1q_u8(out[i].as_mut_ptr().add(16), be_hi);
        }
    }
}
