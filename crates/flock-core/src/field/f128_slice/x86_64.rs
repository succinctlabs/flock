use crate::field::F128;

/// Four-lane pair fold using AVX-512 lane deinterleaving and VPCLMULQDQ.
///
/// # Safety
/// Requires `avx512f` and `vpclmulqdq`.
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    // SAFETY: caller guarantees the target features and source bounds.
    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        // u64-element selectors: even 128-bit lanes -> {0,1,4,5,8,9,12,13},
        // odd -> {2,3,6,7,10,11,14,15} over concat(lo, hi).
        let idx_even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let lanes = dst.len() & !3;
        let mut t = 0;
        while t < lanes {
            let s = 2 * (base + t);
            let lo = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
            let hi = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
            let even = _mm512_permutex2var_epi64(lo, idx_even, hi);
            let odd = _mm512_permutex2var_epi64(lo, idx_odd, hi);
            let new = _mm512_xor_si512(even, ghash_mul_x4(r_bcast, _mm512_xor_si512(even, odd)));
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, new);
            t += 4;
        }
        portable_tail(src, base, dst, r, t);
    }
}

#[inline]
fn portable_tail(src: &[F128], base: usize, dst: &mut [F128], r: F128, mut t: usize) {
    let one_plus_r = F128::ONE + r;
    while t < dst.len() {
        let s = 2 * (base + t);
        dst[t] = src[s] * one_plus_r + src[s + 1] * r;
        t += 1;
    }
}
