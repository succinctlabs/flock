use crate::field::F128;

/// Two-lane pair fold using NEON and PMULL.
///
/// # Safety
/// Requires the `aes` target feature.
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let lanes = dst.len() & !1;
    let mut t = 0;
    while t < lanes {
        let s = 2 * (base + t);
        let e0 = src[s];
        let o0 = src[s + 1];
        let e1 = src[s + 2];
        let o1 = src[s + 3];
        let x0 = F128 {
            lo: e0.lo ^ o0.lo,
            hi: e0.hi ^ o0.hi,
        };
        let x1 = F128 {
            lo: e1.lo ^ o1.lo,
            hi: e1.hi ^ o1.hi,
        };
        // SAFETY: caller guarantees the aes target feature.
        let prod = unsafe { ghash_mul_vec2_neon([r, r], [x0, x1]) };
        dst[t] = F128 {
            lo: e0.lo ^ prod[0].lo,
            hi: e0.hi ^ prod[0].hi,
        };
        dst[t + 1] = F128 {
            lo: e1.lo ^ prod[1].lo,
            hi: e1.hi ^ prod[1].hi,
        };
        t += 2;
    }

    let one_plus_r = F128::ONE + r;
    while t < dst.len() {
        let s = 2 * (base + t);
        dst[t] = src[s] * one_plus_r + src[s + 1] * r;
        t += 1;
    }
}
