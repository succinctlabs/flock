use crate::field::F128;

/// Process two butterflies at a time within a block sharing one twiddle.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        // SAFETY: caller guarantees the aes target feature.
        let product = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };
        let new_u_a = F128 {
            lo: u_a.lo ^ product[0].lo,
            hi: u_a.hi ^ product[0].hi,
        };
        let new_u_b = F128 {
            lo: u_b.lo ^ product[1].lo,
            hi: u_b.hi ^ product[1].hi,
        };
        let new_v_a = F128 {
            lo: v_a.lo ^ new_u_a.lo,
            hi: v_a.hi ^ new_u_a.hi,
        };
        let new_v_b = F128 {
            lo: v_b.lo ^ new_u_b.lo,
            hi: v_b.hi ^ new_u_b.hi,
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

/// Process the single pair in each of two adjacent blocks with distinct
/// twiddles.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    // SAFETY: caller guarantees the aes target feature.
    let product = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };
    let new_u_a = F128 {
        lo: u_a.lo ^ product[0].lo,
        hi: u_a.hi ^ product[0].hi,
    };
    let new_u_b = F128 {
        lo: u_b.lo ^ product[1].lo,
        hi: u_b.hi ^ product[1].hi,
    };
    let new_v_a = F128 {
        lo: v_a.lo ^ new_u_a.lo,
        hi: v_a.hi ^ new_u_a.hi,
    };
    let new_v_b = F128 {
        lo: v_b.lo ^ new_u_b.lo,
        hi: v_b.hi ^ new_u_b.hi,
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}
