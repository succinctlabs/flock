//! E1 workload — per-instance BLAKE3 compression witness builder (k_log = 14).
//!
//! Faithful copy of the private `blake3::build_block_witness_ab_packed_into`
//! (incl. its private bit-position helpers, reconstructed from the public
//! layout constants). Lockstep-tested against the public driver.

use crate::bit_helpers::{BitRecord, add_carry_parts, or_bit_at, or_u32_at_bit};
use flock_prover::r1cs_hashes::blake3::{
    ADDS_PER_G, BLAKE3_IV, BLEN_BASE, CARRY_BITS_PER_ADD, CV_BASE, Compression, FLAGS_BASE,
    G_LANES, G_MSG_IDX, G_STRIDE, GS_BASE, K, M_BASE, MSG_PERMUTATION, N_G_PER_ROUND, N_ROUNDS,
    OUT_HI_BASE, OUT_LO_BASE, T_HI_BASE, T_LO_BASE, WORD_BITS, Z_CONST_POS,
};

pub const U64_PER_BLOCK: usize = K / 64;

// Record-relative positions (copies of blake3.rs's private consts).
const REC_C0: usize = 0;
const REC_C1: usize = CARRY_BITS_PER_ADD;
const REC_C2: usize = 2 * CARRY_BITS_PER_ADD;
const REC_C3: usize = 3 * CARRY_BITS_PER_ADD;
const REC_C4: usize = 4 * CARRY_BITS_PER_ADD;
const REC_C5: usize = 5 * CARRY_BITS_PER_ADD;
const REC_LIN0: usize = ADDS_PER_G * CARRY_BITS_PER_ADD;
const REC_LIN1: usize = REC_LIN0 + WORD_BITS;

#[inline]
fn cv_bit(w: usize) -> usize {
    CV_BASE + WORD_BITS * w
}
#[inline]
fn m_bit(i: usize) -> usize {
    M_BASE + WORD_BITS * i
}
#[inline]
fn out_lo_bit(w: usize) -> usize {
    OUT_LO_BASE + WORD_BITS * w
}
#[inline]
fn out_hi_bit(w: usize) -> usize {
    OUT_HI_BASE + WORD_BITS * w
}

/// `PERM^r [G_MSG_IDX[g]]` (copy of the private helper).
fn per_round_msg_idx() -> [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] {
    let mut perm = [0usize; 16];
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    let mut out = [[[0usize; 2]; N_G_PER_ROUND]; N_ROUNDS];
    for r in 0..N_ROUNDS {
        for g in 0..N_G_PER_ROUND {
            out[r][g][0] = perm[G_MSG_IDX[g][0]];
            out[r][g][1] = perm[G_MSG_IDX[g][1]];
        }
        let mut next = [0usize; 16];
        for i in 0..16 {
            next[i] = perm[MSG_PERMUTATION[i]];
        }
        perm = next;
    }
    out
}

#[inline]
fn write_lin_word(bit_off: usize, val: u32, z: &mut [u64], a: &mut [u64], b: &mut [u64]) {
    or_u32_at_bit(z, bit_off, val);
    or_u32_at_bit(a, bit_off, val);
    or_u32_at_bit(b, bit_off, 0xFFFF_FFFF);
}

/// Fill one instance's `(z, a, b)` u64 buffers (pre-zeroed).
pub fn build_block_witness(input: &Compression, z: &mut [u64], a: &mut [u64], b: &mut [u64]) {
    let (cv, m, counter, block_len, flags) = input;
    debug_assert_eq!(z.len(), U64_PER_BLOCK);

    or_bit_at(z, Z_CONST_POS);
    or_bit_at(a, Z_CONST_POS);
    or_bit_at(b, Z_CONST_POS);

    let counter_lo = *counter as u32;
    let counter_hi = (*counter >> 32) as u32;
    for w in 0..8 {
        write_lin_word(cv_bit(w), cv[w], z, a, b);
    }
    for i in 0..16 {
        write_lin_word(m_bit(i), m[i], z, a, b);
    }
    write_lin_word(T_LO_BASE, counter_lo, z, a, b);
    write_lin_word(T_HI_BASE, counter_hi, z, a, b);
    write_lin_word(BLEN_BASE, *block_len, z, a, b);
    write_lin_word(FLAGS_BASE, *flags, z, a, b);

    let mut state: [u32; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter_lo,
        counter_hi,
        *block_len,
        *flags,
    ];
    let msg_idx = per_round_msg_idx();
    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_i, my_i] = msg_idx[r][g_in_round];
            let mx = m[mx_i];
            let my = m[my_i];

            let a_val = state[la];
            let b_val = state[lb];
            let c_val = state[lc];
            let d_val = state[ld];

            let mut rz = BitRecord::<4>::new();
            let mut ra = BitRecord::<4>::new();
            let mut rb = BitRecord::<4>::new();

            macro_rules! add_into {
                ($pos:ident, $x:expr, $y:expr) => {{
                    let (sum, left, right, carry) = add_carry_parts($x, $y);
                    rz.push::<$pos>(carry);
                    ra.push::<$pos>(left);
                    rb.push::<$pos>(right);
                    sum
                }};
            }

            let tmp_0 = add_into!(REC_C0, a_val, b_val);
            let a_1 = add_into!(REC_C1, tmp_0, mx);
            let d_1 = (d_val ^ a_1).rotate_right(16);
            let c_1 = add_into!(REC_C2, c_val, d_1);
            let b_1 = (b_val ^ c_1).rotate_right(12);
            let tmp_1 = add_into!(REC_C3, a_1, b_1);
            let a_2 = add_into!(REC_C4, tmp_1, my);
            let d_2 = (d_1 ^ a_2).rotate_right(8);
            let c_2 = add_into!(REC_C5, c_1, d_2);
            let b_new = (b_1 ^ c_2).rotate_right(7);
            let d_new = d_2;
            rz.push::<REC_LIN0>(b_new);
            ra.push::<REC_LIN0>(b_new);
            rb.push::<REC_LIN0>(0xFFFF_FFFF);
            rz.push::<REC_LIN1>(d_new);
            ra.push::<REC_LIN1>(d_new);
            rb.push::<REC_LIN1>(0xFFFF_FFFF);

            let g_base = GS_BASE + G_STRIDE * g;
            rz.flush(z, g_base);
            ra.flush(a, g_base);
            rb.flush(b, g_base);

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    for w in 0..8 {
        let lo = state[w] ^ state[w + 8];
        let hi = state[w + 8] ^ cv[w];
        write_lin_word(out_lo_bit(w), lo, z, a, b);
        write_lin_word(out_hi_bit(w), hi, z, a, b);
    }
}

/// Deterministic pseudo-random compression input.
pub fn random_input(seed: u64) -> Compression {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    };
    let cv: [u32; 8] = std::array::from_fn(|_| next());
    let m: [u32; 16] = std::array::from_fn(|_| next());
    let counter = ((next() as u64) << 32) | next() as u64;
    (cv, m, counter, 64u32, 11u32)
}
