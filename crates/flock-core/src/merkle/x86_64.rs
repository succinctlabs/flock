use super::{Hash, SHA256_IV, SHA256_K};
use core::arch::x86_64::*;

#[inline(always)]
unsafe fn schedule(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> __m128i {
    unsafe {
        let t1 = _mm_sha256msg1_epu32(v0, v1);
        let t2 = _mm_alignr_epi8(v3, v2, 4);
        _mm_sha256msg2_epu32(_mm_add_epi32(t1, t2), v3)
    }
}

#[inline(always)]
unsafe fn rounds4(
    abef: &mut [__m128i; 4],
    cdgh: &mut [__m128i; 4],
    words: &[__m128i; 4],
    group: usize,
) {
    unsafe {
        let k = _mm_set_epi32(
            SHA256_K[4 * group + 3] as i32,
            SHA256_K[4 * group + 2] as i32,
            SHA256_K[4 * group + 1] as i32,
            SHA256_K[4 * group] as i32,
        );
        for stream in 0..4 {
            let wk = _mm_add_epi32(words[stream], k);
            cdgh[stream] = _mm_sha256rnds2_epu32(cdgh[stream], abef[stream], wk);
            let wk_hi = _mm_shuffle_epi32(wk, 0x0e);
            abef[stream] = _mm_sha256rnds2_epu32(abef[stream], cdgh[stream], wk_hi);
        }
    }
}

/// Compress one block for each of four independent SHA-256 states.
///
/// # Safety
/// Every block pointer must expose 64 readable bytes. The module cfg
/// guarantees SHA-NI, SSE2, SSSE3, and SSE4.1 on the compiled target.
#[inline(always)]
unsafe fn compress4(abef: &mut [__m128i; 4], cdgh: &mut [__m128i; 4], blocks: [*const u8; 4]) {
    unsafe {
        let endian = _mm_set_epi64x(
            0x0c0d_0e0f_0809_0a0bu64 as i64,
            0x0405_0607_0001_0203u64 as i64,
        );
        let mut w0 = [_mm_setzero_si128(); 4];
        let mut w1 = [_mm_setzero_si128(); 4];
        let mut w2 = [_mm_setzero_si128(); 4];
        let mut w3 = [_mm_setzero_si128(); 4];
        for stream in 0..4 {
            let data = blocks[stream].cast::<__m128i>();
            w0[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data), endian);
            w1[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data.add(1)), endian);
            w2[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data.add(2)), endian);
            w3[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data.add(3)), endian);
        }

        let abef_saved = *abef;
        let cdgh_saved = *cdgh;

        macro_rules! schedule_rounds4 {
            ($dst:ident, $v1:ident, $v2:ident, $v3:ident, $group:expr) => {{
                for stream in 0..4 {
                    $dst[stream] = schedule($dst[stream], $v1[stream], $v2[stream], $v3[stream]);
                }
                rounds4(abef, cdgh, &$dst, $group);
            }};
        }

        rounds4(abef, cdgh, &w0, 0);
        rounds4(abef, cdgh, &w1, 1);
        rounds4(abef, cdgh, &w2, 2);
        rounds4(abef, cdgh, &w3, 3);
        schedule_rounds4!(w0, w1, w2, w3, 4);
        schedule_rounds4!(w1, w2, w3, w0, 5);
        schedule_rounds4!(w2, w3, w0, w1, 6);
        schedule_rounds4!(w3, w0, w1, w2, 7);
        schedule_rounds4!(w0, w1, w2, w3, 8);
        schedule_rounds4!(w1, w2, w3, w0, 9);
        schedule_rounds4!(w2, w3, w0, w1, 10);
        schedule_rounds4!(w3, w0, w1, w2, 11);
        schedule_rounds4!(w0, w1, w2, w3, 12);
        schedule_rounds4!(w1, w2, w3, w0, 13);
        schedule_rounds4!(w2, w3, w0, w1, 14);
        schedule_rounds4!(w3, w0, w1, w2, 15);

        for stream in 0..4 {
            abef[stream] = _mm_add_epi32(abef[stream], abef_saved[stream]);
            cdgh[stream] = _mm_add_epi32(cdgh[stream], cdgh_saved[stream]);
        }
    }
}

/// Hash four equal-length inputs, producing standard SHA-256 digests.
#[inline]
pub fn hash4_equal_len(inputs: [&[u8]; 4], out: &mut [Hash]) {
    let len = inputs[0].len();
    debug_assert!(inputs.iter().all(|input| input.len() == len));
    debug_assert!(out.len() >= 4);

    // SAFETY: the module is compiled only with SHA-NI enabled. Full-block
    // pointers and fixed tail buffers satisfy compress4's 64-byte bound.
    unsafe {
        let state = SHA256_IV.as_ptr().cast::<__m128i>();
        let dcba = _mm_loadu_si128(state);
        let efgh = _mm_loadu_si128(state.add(1));
        let cdab = _mm_shuffle_epi32(dcba, 0xb1);
        let efgh = _mm_shuffle_epi32(efgh, 0x1b);
        let abef_initial = _mm_alignr_epi8(cdab, efgh, 8);
        let cdgh_initial = _mm_blend_epi16(efgh, cdab, 0xf0);
        let mut abef = [abef_initial; 4];
        let mut cdgh = [cdgh_initial; 4];

        for block in 0..(len / 64) {
            compress4(
                &mut abef,
                &mut cdgh,
                [
                    inputs[0].as_ptr().add(block * 64),
                    inputs[1].as_ptr().add(block * 64),
                    inputs[2].as_ptr().add(block * 64),
                    inputs[3].as_ptr().add(block * 64),
                ],
            );
        }

        let rem = len % 64;
        let tail_blocks = if rem < 56 { 1 } else { 2 };
        let mut tails = [[0u8; 128]; 4];
        for stream in 0..4 {
            tails[stream][..rem].copy_from_slice(&inputs[stream][len - rem..]);
            tails[stream][rem] = 0x80;
            tails[stream][tail_blocks * 64 - 8..tail_blocks * 64]
                .copy_from_slice(&((len as u64) * 8).to_be_bytes());
        }
        for block in 0..tail_blocks {
            compress4(
                &mut abef,
                &mut cdgh,
                [
                    tails[0].as_ptr().add(block * 64),
                    tails[1].as_ptr().add(block * 64),
                    tails[2].as_ptr().add(block * 64),
                    tails[3].as_ptr().add(block * 64),
                ],
            );
        }

        let endian = _mm_set_epi64x(
            0x0c0d_0e0f_0809_0a0bu64 as i64,
            0x0405_0607_0001_0203u64 as i64,
        );
        for stream in 0..4 {
            let feba = _mm_shuffle_epi32(abef[stream], 0x1b);
            let dchg = _mm_shuffle_epi32(cdgh[stream], 0xb1);
            let dcba = _mm_blend_epi16(feba, dchg, 0xf0);
            let hgef = _mm_alignr_epi8(dchg, feba, 8);
            _mm_storeu_si128(
                out[stream].as_mut_ptr().cast::<__m128i>(),
                _mm_shuffle_epi8(dcba, endian),
            );
            _mm_storeu_si128(
                out[stream].as_mut_ptr().add(16).cast::<__m128i>(),
                _mm_shuffle_epi8(hgef, endian),
            );
        }
    }
}
