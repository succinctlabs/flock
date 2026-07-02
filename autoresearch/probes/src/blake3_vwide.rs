//! C2 — direct-write BLAKE3 producer: V = 8 compressions in lockstep,
//! interleaved L1-resident rows (16 KB per buffer), NT flush. See
//! `direct_common` and `sha2_vwide` (same recipe).

use crate::direct_common::{
    Row, SendPtr, V, add_carry_parts_v, flush_rows_nt, or_bit_row, or_u32_row, stripe_from_rows,
};
use flock_prover::r1cs_hashes::blake3::{
    ADDS_PER_G, BLAKE3_IV, BLEN_BASE, CARRY_BITS_PER_ADD, CV_BASE, Compression, FLAGS_BASE,
    G_LANES, G_MSG_IDX, G_STRIDE, GS_BASE, K, M_BASE, MSG_PERMUTATION, N_G_PER_ROUND, N_ROUNDS,
    OUT_HI_BASE, OUT_LO_BASE, T_HI_BASE, T_LO_BASE, USEFUL_BITS, WORD_BITS, Z_CONST_POS,
};
use rayon::prelude::*;

pub const U64_PER_BLOCK: usize = K / 64; // 256
const USEFUL_WORDS: usize = USEFUL_BITS.div_ceil(64);
pub const USEFUL_CHUNKS: usize = USEFUL_BITS.div_ceil(128);

#[inline]
fn g_add_carry_bit(g: usize, add_idx: usize) -> usize {
    GS_BASE + G_STRIDE * g + CARRY_BITS_PER_ADD * add_idx
}
#[inline]
fn g_lin_bit(g: usize, which: usize) -> usize {
    GS_BASE + G_STRIDE * g + ADDS_PER_G * CARRY_BITS_PER_ADD + WORD_BITS * which
}

/// `PERM^r [G_MSG_IDX[g]]`.
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

#[inline(always)]
fn xor_rotr_v(x: &[u32; V], y: &[u32; V], r: u32) -> [u32; V] {
    std::array::from_fn(|j| (x[j] ^ y[j]).rotate_right(r))
}

struct Rows {
    z: Vec<Row>,
    a: Vec<Row>,
    b: Vec<Row>,
}

#[inline(always)]
fn write_lin(rows: &mut Rows, bit: usize, vals: &[u32; V]) {
    or_u32_row(&mut rows.z, bit, vals);
    or_u32_row(&mut rows.a, bit, vals);
    or_u32_row(&mut rows.b, bit, &[0xFFFF_FFFF; V]);
}

#[inline(always)]
fn add_inline(rows: &mut Rows, x: &[u32; V], y: &[u32; V], carry_bit: usize) -> [u32; V] {
    let (sum, left, right, carry) = add_carry_parts_v(x, y);
    or_u32_row(&mut rows.z, carry_bit, &carry);
    or_u32_row(&mut rows.a, carry_bit, &left);
    or_u32_row(&mut rows.b, carry_bit, &right);
    sum
}

fn build_group_rows(inputs: &[Compression], o0: usize, rows: &mut Rows) {
    let cv: [[u32; V]; 8] = std::array::from_fn(|w| std::array::from_fn(|j| inputs[o0 + j].0[w]));
    let m: [[u32; V]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| inputs[o0 + j].1[i]));
    let counter_lo: [u32; V] = std::array::from_fn(|j| inputs[o0 + j].2 as u32);
    let counter_hi: [u32; V] = std::array::from_fn(|j| (inputs[o0 + j].2 >> 32) as u32);
    let block_len: [u32; V] = std::array::from_fn(|j| inputs[o0 + j].3);
    let flags: [u32; V] = std::array::from_fn(|j| inputs[o0 + j].4);

    or_bit_row(&mut rows.z, Z_CONST_POS);
    or_bit_row(&mut rows.a, Z_CONST_POS);
    or_bit_row(&mut rows.b, Z_CONST_POS);

    for w in 0..8 {
        write_lin(rows, CV_BASE + WORD_BITS * w, &cv[w]);
    }
    for i in 0..16 {
        write_lin(rows, M_BASE + WORD_BITS * i, &m[i]);
    }
    write_lin(rows, T_LO_BASE, &counter_lo);
    write_lin(rows, T_HI_BASE, &counter_hi);
    write_lin(rows, BLEN_BASE, &block_len);
    write_lin(rows, FLAGS_BASE, &flags);

    let mut state: [[u32; V]; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        [BLAKE3_IV[0]; V],
        [BLAKE3_IV[1]; V],
        [BLAKE3_IV[2]; V],
        [BLAKE3_IV[3]; V],
        counter_lo,
        counter_hi,
        block_len,
        flags,
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

            let tmp_0 = add_inline(rows, &a_val, &b_val, g_add_carry_bit(g, 0));
            let a_1 = add_inline(rows, &tmp_0, &mx, g_add_carry_bit(g, 1));
            let d_1 = xor_rotr_v(&d_val, &a_1, 16);
            let c_1 = add_inline(rows, &c_val, &d_1, g_add_carry_bit(g, 2));
            let b_1 = xor_rotr_v(&b_val, &c_1, 12);
            let tmp_1 = add_inline(rows, &a_1, &b_1, g_add_carry_bit(g, 3));
            let a_2 = add_inline(rows, &tmp_1, &my, g_add_carry_bit(g, 4));
            let d_2 = xor_rotr_v(&d_1, &a_2, 8);
            let c_2 = add_inline(rows, &c_1, &d_2, g_add_carry_bit(g, 5));
            let b_new = xor_rotr_v(&b_1, &c_2, 7);
            let d_new = d_2;
            write_lin(rows, g_lin_bit(g, 0), &b_new);
            write_lin(rows, g_lin_bit(g, 1), &d_new);

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    for w in 0..8 {
        let lo: [u32; V] = std::array::from_fn(|j| state[w][j] ^ state[w + 8][j]);
        let hi: [u32; V] = std::array::from_fn(|j| state[w + 8][j] ^ cv[w][j]);
        write_lin(rows, OUT_LO_BASE + WORD_BITS * w, &lo);
        write_lin(rows, OUT_HI_BASE + WORD_BITS * w, &hi);
    }
}

/// Direct L1′ BLAKE3 producer. Dest buffers (`2^n_log · 256` u64) and stripe
/// must be pre-zeroed once; padding is never written.
pub fn build_l1_direct(
    inputs: &[Compression],
    n_log: usize,
    stripe: Option<&mut [u8]>,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    let n = 1usize << n_log;
    assert_eq!(inputs.len(), n);
    assert!(n >= V);
    let total = n * U64_PER_BLOCK;
    assert_eq!(z.len(), total);
    assert_eq!(a.len(), total);
    assert_eq!(b.len(), total);
    let (zp, ap, bp) = (
        SendPtr(z.as_mut_ptr()),
        SendPtr(a.as_mut_ptr()),
        SendPtr(b.as_mut_ptr()),
    );
    let sp = stripe.map(|s| {
        assert_eq!(s.len(), (n / 8) * U64_PER_BLOCK * 64);
        SendPtr(s.as_mut_ptr() as *mut u64)
    });
    let inputs_ref = &inputs[..];

    (0..n / V).into_par_iter().for_each_init(
        || Rows {
            z: vec![[0u64; V]; U64_PER_BLOCK],
            a: vec![[0u64; V]; U64_PER_BLOCK],
            b: vec![[0u64; V]; U64_PER_BLOCK],
        },
        move |rows, g| {
            rows.z[..USEFUL_WORDS].fill([0u64; V]);
            rows.a[..USEFUL_WORDS].fill([0u64; V]);
            rows.b[..USEFUL_WORDS].fill([0u64; V]);
            let o0 = g * V;
            build_group_rows(inputs_ref, o0, rows);
            // SAFETY: disjoint instance ranges per group; dest pre-zeroed.
            unsafe {
                flush_rows_nt(&rows.z, zp.get(), o0, n_log, USEFUL_CHUNKS);
                flush_rows_nt(&rows.a, ap.get(), o0, n_log, USEFUL_CHUNKS);
                flush_rows_nt(&rows.b, bp.get(), o0, n_log, USEFUL_CHUNKS);
                if let Some(p) = sp {
                    stripe_from_rows(
                        &rows.z,
                        p.get() as *mut u8,
                        o0,
                        U64_PER_BLOCK,
                        USEFUL_WORDS,
                    );
                }
            }
        },
    );
}
