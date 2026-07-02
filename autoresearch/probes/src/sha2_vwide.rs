//! C2 — direct-write SHA-256 producer: V = 8 compressions in lockstep,
//! fields OR'd into an interleaved L1-resident row buffer (32 KB per output
//! buffer), NT-flushed to L1′ per useful chunk. See `direct_common`.

use crate::direct_common::{
    Row, SendPtr, V, add_carry_parts_v, flush_rows_nt, or_bit_row, or_u32_row, stripe_from_rows,
};
use crate::sha2_witness::Sha2Input;
use flock_prover::r1cs_hashes::sha2::{
    CARRIES_PER_ADD, H_WORDS, K, M_WORDS, N_OUT_WORDS, N_ROUNDS, SHA256_K, USEFUL_BITS,
    Z_CONST_POS, a_new_bit, ch_and_bit, e_new_bit, h_bit, h_out_bit, m_bit, maj_and_bit,
    out_carry_bit, round_carry_bit, sched_carry_bit, t1_bit, w_bit,
};
use rayon::prelude::*;

pub const U64_PER_BLOCK: usize = K / 64; // 512
const USEFUL_WORDS: usize = USEFUL_BITS.div_ceil(64);
pub const USEFUL_CHUNKS: usize = USEFUL_BITS.div_ceil(128);

#[inline(always)]
fn map_v(x: &[u32; V], f: impl Fn(u32) -> u32) -> [u32; V] {
    std::array::from_fn(|j| f(x[j]))
}
#[inline(always)]
fn xor_v(x: &[u32; V], y: &[u32; V]) -> [u32; V] {
    std::array::from_fn(|j| x[j] ^ y[j])
}
#[inline(always)]
fn and_v(x: &[u32; V], y: &[u32; V]) -> [u32; V] {
    std::array::from_fn(|j| x[j] & y[j])
}

#[inline(always)]
fn big_sigma0_v(x: &[u32; V]) -> [u32; V] {
    map_v(x, |x| x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22))
}
#[inline(always)]
fn big_sigma1_v(x: &[u32; V]) -> [u32; V] {
    map_v(x, |x| x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25))
}
#[inline(always)]
fn small_sigma0_v(x: &[u32; V]) -> [u32; V] {
    map_v(x, |x| x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3))
}
#[inline(always)]
fn small_sigma1_v(x: &[u32; V]) -> [u32; V] {
    map_v(x, |x| x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10))
}

struct Rows {
    z: Vec<Row>,
    a: Vec<Row>,
    b: Vec<Row>,
}

/// z = a = v, b = all-ones (the free-witness tautology rows).
#[inline(always)]
fn write_lin(rows: &mut Rows, bit: usize, vals: &[u32; V]) {
    or_u32_row(&mut rows.z, bit, vals);
    or_u32_row(&mut rows.a, bit, vals);
    or_u32_row(&mut rows.b, bit, &[0xFFFF_FFFF; V]);
}

/// Inline add: carry rows only.
#[inline(always)]
fn add_inline(rows: &mut Rows, x: &[u32; V], y: &[u32; V], carry_bit: usize) -> [u32; V] {
    let (sum, left, right, carry) = add_carry_parts_v(x, y);
    or_u32_row(&mut rows.z, carry_bit, &carry);
    or_u32_row(&mut rows.a, carry_bit, &left);
    or_u32_row(&mut rows.b, carry_bit, &right);
    sum
}

fn build_group_rows(inputs: &[Sha2Input], o0: usize, rows: &mut Rows) {
    // Transpose inputs to lane form.
    let h_in: [[u32; V]; 8] = std::array::from_fn(|w| std::array::from_fn(|j| inputs[o0 + j].0[w]));
    let m: [[u32; V]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| inputs[o0 + j].1[i]));

    or_bit_row(&mut rows.z, Z_CONST_POS);
    or_bit_row(&mut rows.a, Z_CONST_POS);
    or_bit_row(&mut rows.b, Z_CONST_POS);

    for w in 0..H_WORDS {
        write_lin(rows, h_bit(w, 0), &h_in[w]);
    }
    for i in 0..M_WORDS {
        write_lin(rows, m_bit(i, 0), &m[i]);
    }

    // Message schedule.
    let mut w_sched: Vec<[u32; V]> = Vec::with_capacity(64);
    w_sched.extend_from_slice(&m);
    for t in 16..64 {
        let s_0 = add_inline(
            rows,
            &small_sigma1_v(&w_sched[t - 2]),
            &w_sched[t - 7],
            sched_carry_bit(t, 0, 0),
        );
        let s_1 = add_inline(
            rows,
            &s_0,
            &small_sigma0_v(&w_sched[t - 15]),
            sched_carry_bit(t, 1, 0),
        );
        let w_t = add_inline(rows, &s_1, &w_sched[t - 16], sched_carry_bit(t, 2, 0));
        write_lin(rows, w_bit(t, 0), &w_t);
        w_sched.push(w_t);
    }

    // 64 rounds.
    let mut aa = h_in[0];
    let mut bb = h_in[1];
    let mut cc = h_in[2];
    let mut dd = h_in[3];
    let mut ee = h_in[4];
    let mut ff = h_in[5];
    let mut gg = h_in[6];
    let mut hh = h_in[7];
    for r in 0..N_ROUNDS {
        let f_xor_g = xor_v(&ff, &gg);
        let ch_and_v = and_v(&ee, &f_xor_g);
        or_u32_row(&mut rows.z, ch_and_bit(r, 0), &ch_and_v);
        or_u32_row(&mut rows.a, ch_and_bit(r, 0), &ee);
        or_u32_row(&mut rows.b, ch_and_bit(r, 0), &f_xor_g);
        let ch_out = xor_v(&ch_and_v, &gg);

        let b_xor_a = xor_v(&bb, &aa);
        let c_xor_a = xor_v(&cc, &aa);
        let maj_and_v = and_v(&b_xor_a, &c_xor_a);
        or_u32_row(&mut rows.z, maj_and_bit(r, 0), &maj_and_v);
        or_u32_row(&mut rows.a, maj_and_bit(r, 0), &b_xor_a);
        or_u32_row(&mut rows.b, maj_and_bit(r, 0), &c_xor_a);
        let maj_out = xor_v(&maj_and_v, &aa);

        let k_r = [SHA256_K[r]; V];
        let t1_0 = add_inline(rows, &hh, &big_sigma1_v(&ee), round_carry_bit(r, 0, 0));
        let t1_1 = add_inline(rows, &t1_0, &ch_out, round_carry_bit(r, 1, 0));
        let t1_2 = add_inline(rows, &t1_1, &k_r, round_carry_bit(r, 2, 0));
        let t1 = add_inline(rows, &t1_2, &w_sched[r], round_carry_bit(r, 3, 0));
        write_lin(rows, t1_bit(r, 0), &t1);

        let t2 = add_inline(rows, &big_sigma0_v(&aa), &maj_out, round_carry_bit(r, 4, 0));
        let e_new = add_inline(rows, &dd, &t1, round_carry_bit(r, 5, 0));
        write_lin(rows, e_new_bit(r, 0), &e_new);
        let a_new = add_inline(rows, &t1, &t2, round_carry_bit(r, 6, 0));
        write_lin(rows, a_new_bit(r, 0), &a_new);

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
        let sum = add_inline(rows, &final_state[w], &h_in[w], out_carry_bit(w, 0));
        write_lin(rows, h_out_bit(w, 0), &sum);
    }
    let _ = CARRIES_PER_ADD;
}

/// Direct L1′ SHA-256 producer. Dest buffers (`2^n_log · 512` u64) and stripe
/// must be pre-zeroed once; padding is never written.
pub fn build_l1_direct(
    inputs: &[Sha2Input],
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
            // Zero only the useful rows (the tail is never written).
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
