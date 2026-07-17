use crate::field::F128;

#[inline]
pub(super) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    let one_plus_r = F128::ONE + r;
    for (t, value) in dst.iter_mut().enumerate() {
        let s = 2 * (base + t);
        *value = src[s] * one_plus_r + src[s + 1] * r;
    }
}
