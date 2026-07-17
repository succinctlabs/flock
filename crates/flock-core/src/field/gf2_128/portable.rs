use super::{F128, F256Unreduced, ghash_reduce};

/// 64×64 carry-less product into 128 bits (lo, hi).
pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
    let mut lo: u64 = 0;
    let mut hi: u64 = 0;
    let mut i = 0;
    while i < 64 {
        if (a >> i) & 1 != 0 {
            lo ^= b << i;
            if i != 0 {
                hi ^= b >> (64 - i);
            }
        }
        i += 1;
    }
    (lo, hi)
}

pub fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
    let (ll_lo, ll_hi) = clmul64(a.lo, b.lo);
    let (lh_lo, lh_hi) = clmul64(a.lo, b.hi);
    let (hl_lo, hl_hi) = clmul64(a.hi, b.lo);
    let (hh_lo, hh_hi) = clmul64(a.hi, b.hi);
    let cr_lo = lh_lo ^ hl_lo;
    let cr_hi = lh_hi ^ hl_hi;
    F256Unreduced {
        r0: ll_lo,
        r1: ll_hi ^ cr_lo,
        r2: hh_lo ^ cr_hi,
        r3: hh_hi,
    }
}

pub fn ghash_mul(a: F128, b: F128) -> F128 {
    let u = ghash_mul_unreduced(a, b);
    ghash_reduce(u.r0, u.r1, u.r2, u.r3)
}
