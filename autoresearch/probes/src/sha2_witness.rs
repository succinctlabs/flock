//! E1 workload — per-instance SHA-256 witness builder (k_log = 15).
//!
//! Faithful copy of the private `sha2::build_block_ab_packed_into`, using the
//! module's public layout constants/helpers plus [`crate::bit_helpers`].
//! Lockstep-tested against the public driver.

use crate::bit_helpers::{BitRecord, add_carry_parts, or_bit_at, or_u32_at_bit};
use flock_prover::r1cs_hashes::sha2::{
    CARRIES_PER_ADD, H_WORDS, K, M_WORDS, N_OUT_WORDS, N_ROUNDS, SHA256_K, Z_CONST_POS, ch_and_bit,
    a_new_bit, e_new_bit, h_bit, h_out_bit, m_bit, maj_and_bit, out_carry_bit, round_carry_bit,
    sched_carry_bit, t1_bit, w_bit,
};

pub const U64_PER_BLOCK: usize = K / 64;

/// One SHA-256 compression instance: (chaining value, message block).
pub type Sha2Input = ([u32; 8], [u32; 16]);

#[inline]
fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
#[inline]
fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
#[inline]
fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
#[inline]
fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

#[inline(always)]
fn add_inline_ab(
    x: u32,
    y: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
    carry_base: usize,
) -> u32 {
    let (sum, left, right, carry_aux) = add_carry_parts(x, y);
    or_u32_at_bit(z, carry_base, carry_aux);
    or_u32_at_bit(a, carry_base, left);
    or_u32_at_bit(b, carry_base, right);
    sum
}

#[inline(always)]
fn add_alloc_ab(
    x: u32,
    y: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
    sum_base: usize,
    carry_base: usize,
) -> u32 {
    let sum = add_inline_ab(x, y, z, a, b, carry_base);
    or_u32_at_bit(z, sum_base, sum);
    or_u32_at_bit(a, sum_base, sum);
    or_u32_at_bit(b, sum_base, 0xFFFF_FFFF);
    sum
}

/// Fill one instance's `(z, a, b)` u64 buffers (pre-zeroed) from `(h_in, m)`.
pub fn build_block_witness(input: &Sha2Input, z: &mut [u64], a: &mut [u64], b: &mut [u64]) {
    let (h_in, m) = input;
    debug_assert_eq!(z.len(), U64_PER_BLOCK);

    or_bit_at(z, Z_CONST_POS);
    or_bit_at(a, Z_CONST_POS);
    or_bit_at(b, Z_CONST_POS);

    for w in 0..H_WORDS {
        let off = h_bit(w, 0);
        let v = h_in[w];
        or_u32_at_bit(z, off, v);
        or_u32_at_bit(a, off, v);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
    }
    for i in 0..M_WORDS {
        let off = m_bit(i, 0);
        let v = m[i];
        or_u32_at_bit(z, off, v);
        or_u32_at_bit(a, off, v);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
    }

    // Message schedule.
    let mut w_sched = [0u32; 64];
    w_sched[..16].copy_from_slice(m);
    const SC0: usize = 0;
    const SC1: usize = CARRIES_PER_ADD;
    const SC2: usize = 2 * CARRIES_PER_ADD;
    for t in 16..64 {
        let mut rz = BitRecord::<2>::new();
        let mut ra = BitRecord::<2>::new();
        let mut rb = BitRecord::<2>::new();

        macro_rules! add_into {
            ($pos:ident, $x:expr, $y:expr) => {{
                let (sum, left, right, carry) = add_carry_parts($x, $y);
                rz.push::<$pos>(carry);
                ra.push::<$pos>(left);
                rb.push::<$pos>(right);
                sum
            }};
        }

        let s_0 = add_into!(SC0, small_sigma1(w_sched[t - 2]), w_sched[t - 7]);
        let s_1 = add_into!(SC1, s_0, small_sigma0(w_sched[t - 15]));
        let w_t = add_into!(SC2, s_1, w_sched[t - 16]);

        let sched_base = sched_carry_bit(t, 0, 0);
        rz.flush(z, sched_base);
        ra.flush(a, sched_base);
        rb.flush(b, sched_base);

        let off = w_bit(t, 0);
        or_u32_at_bit(z, off, w_t);
        or_u32_at_bit(a, off, w_t);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        w_sched[t] = w_t;
    }

    // 64 rounds.
    let [
        mut aa,
        mut bb,
        mut cc,
        mut dd,
        mut ee,
        mut ff,
        mut gg,
        mut hh,
    ] = *h_in;
    for r in 0..N_ROUNDS {
        let f_xor_g = ff ^ gg;
        let ch_and_v = ee & f_xor_g;
        let off = ch_and_bit(r, 0);
        or_u32_at_bit(z, off, ch_and_v);
        or_u32_at_bit(a, off, ee);
        or_u32_at_bit(b, off, f_xor_g);
        let ch_out = ch_and_v ^ gg;

        let b_xor_a = bb ^ aa;
        let c_xor_a = cc ^ aa;
        let maj_and_v = b_xor_a & c_xor_a;
        let off = maj_and_bit(r, 0);
        or_u32_at_bit(z, off, maj_and_v);
        or_u32_at_bit(a, off, b_xor_a);
        or_u32_at_bit(b, off, c_xor_a);
        let maj_out = maj_and_v ^ aa;

        const RC0: usize = 0;
        const RC1: usize = CARRIES_PER_ADD;
        const RC2: usize = 2 * CARRIES_PER_ADD;
        const RC3: usize = 3 * CARRIES_PER_ADD;
        const RC4: usize = 4 * CARRIES_PER_ADD;
        const RC5: usize = 5 * CARRIES_PER_ADD;
        const RC6: usize = 6 * CARRIES_PER_ADD;
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

        let t1_0 = add_into!(RC0, hh, big_sigma1(ee));
        let t1_1 = add_into!(RC1, t1_0, ch_out);
        let t1_2 = add_into!(RC2, t1_1, SHA256_K[r]);
        let t1 = add_into!(RC3, t1_2, w_sched[r]);
        let off = t1_bit(r, 0);
        or_u32_at_bit(z, off, t1);
        or_u32_at_bit(a, off, t1);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        let t2 = add_into!(RC4, big_sigma0(aa), maj_out);
        let e_new = add_into!(RC5, dd, t1);
        let off = e_new_bit(r, 0);
        or_u32_at_bit(z, off, e_new);
        or_u32_at_bit(a, off, e_new);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
        let a_new = add_into!(RC6, t1, t2);
        let off = a_new_bit(r, 0);
        or_u32_at_bit(z, off, a_new);
        or_u32_at_bit(a, off, a_new);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        let round_base = round_carry_bit(r, 0, 0);
        rz.flush(z, round_base);
        ra.flush(a, round_base);
        rb.flush(b, round_base);

        hh = gg;
        gg = ff;
        ff = ee;
        ee = e_new;
        dd = cc;
        cc = bb;
        bb = aa;
        aa = a_new;
    }

    // Output feed-forward.
    let final_state = [aa, bb, cc, dd, ee, ff, gg, hh];
    for w in 0..N_OUT_WORDS {
        add_alloc_ab(
            final_state[w],
            h_in[w],
            z,
            a,
            b,
            h_out_bit(w, 0),
            out_carry_bit(w, 0),
        );
    }
}

/// Deterministic pseudo-random compression input.
pub fn random_input(seed: u64) -> Sha2Input {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    };
    (
        std::array::from_fn(|_| next()),
        std::array::from_fn(|_| next()),
    )
}
