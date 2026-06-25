// Research scratch — bit-position loops and many-arg fused-walk helpers are
// idiomatic for the SHA-256-shaped data flow; iterator/argument-bag refactors
// would obscure the algorithmic structure.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! **Experiment** — compute the lincheck verifier's A-side AND B-side
//! consistency checks by running SHA-256's *linear* data flow on `z_vec`
//! in a single fused walk, sidestepping the sparse matrix multiplications.
//!
//! The standard verifier computes:
//!     v_a == inner_product(sparse_row_fold(A_0, eq_inner), z_vec)
//!     v_b == inner_product(sparse_row_fold(B_0, eq_inner), z_vec)
//!
//! Equivalently (transposed):
//!     v_a == <eq_inner, A_0 · z_vec>
//!     v_b == <eq_inner, B_0 · z_vec>
//!
//! This experiment computes both by walking SHA-256's algorithm on z_vec
//! and emitting the (A·z_vec, B·z_vec) pair at each constraint in lock-step,
//! immediately accumulating into the two `v_*` accumulators. eq_inner is
//! shared between A and B — single table, two scalar accumulators.
//!
//! Verified for correctness against the standard verifier path. Times both
//! at the sha2 R1CS dimensions.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::lincheck::{build_quirky_eq_table, sparse_row_fold};
use flock_prover::r1cs_hashes::sha2::{
    H_WORDS, K, K_LOG, K_SKIP, M_WORDS, N_OUT_WORDS, N_ROUNDS, N_SCHED, SHA256_K, WORD_BITS,
    Z_CONST_POS, a_new_bit, build_matrices, ch_and_bit, e_new_bit, h_bit, h_out_bit, m_bit,
    maj_and_bit, out_carry_bit, round_carry_bit, sched_carry_bit, t1_bit, w_bit,
};

// ───────────────────────────────────────────────────────────────────────────
// FWord: 32 F_{2^128} elements representing one 32-bit SHA-256 "word" worth
// of phantom-witness bits. XOR is per-bit F128 XOR; rotation/shift is just
// index rewiring.
// ───────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct FWord([F128; 32]);

impl FWord {
    fn zero() -> Self {
        Self([F128::ZERO; 32])
    }
    fn read_from(z: &[F128], base: usize) -> Self {
        let mut w = [F128::ZERO; 32];
        w.copy_from_slice(&z[base..base + 32]);
        Self(w)
    }
    fn xor(&self, other: &Self) -> Self {
        let mut w = [F128::ZERO; 32];
        for b in 0..32 {
            w[b] = self.0[b] + other.0[b];
        }
        Self(w)
    }
    fn xor3(a: &Self, b: &Self, c: &Self) -> Self {
        let mut w = [F128::ZERO; 32];
        for i in 0..32 {
            w[i] = a.0[i] + b.0[i] + c.0[i];
        }
        Self(w)
    }
    fn rotr(&self, n: usize) -> Self {
        let mut w = [F128::ZERO; 32];
        for i in 0..32 {
            w[i] = self.0[(i + n) % 32];
        }
        Self(w)
    }
    fn shr(&self, n: usize) -> Self {
        let mut w = [F128::ZERO; 32];
        for i in 0..32 {
            if i + n < 32 {
                w[i] = self.0[i + n];
            }
        }
        Self(w)
    }
}

#[inline]
fn sigma_0(w: &FWord) -> FWord {
    FWord::xor3(&w.rotr(7), &w.rotr(18), &w.shr(3))
}
#[inline]
fn sigma_1(w: &FWord) -> FWord {
    FWord::xor3(&w.rotr(17), &w.rotr(19), &w.shr(10))
}
#[inline]
fn big_sigma_0(w: &FWord) -> FWord {
    FWord::xor3(&w.rotr(2), &w.rotr(13), &w.rotr(22))
}
#[inline]
fn big_sigma_1(w: &FWord) -> FWord {
    FWord::xor3(&w.rotr(6), &w.rotr(11), &w.rotr(25))
}

// ───────────────────────────────────────────────────────────────────────────
// The linear-SHA-256 A-side consistency check.
// For each constraint slot i, compute (A_0 · z_vec)[i] as the A-row's linear
// combination evaluated on z_vec, then accumulate v_a += that · eq_inner[i].
// ───────────────────────────────────────────────────────────────────────────

/// 32-bit add with INLINED carry rows. Accumulates A-side AND B-side
/// contributions for each carry-aux row.
///   A-row at carry-aux[i] = [X[i], cin chain]  → a_val = X[i] ⊕ cin
///   B-row at carry-aux[i] = [Y[i], cin chain]  → b_val = Y[i] ⊕ cin
fn add_inline<F: Fn(usize) -> usize>(
    x: &FWord,
    y: &FWord,
    carry_slot_fn: F,
    z_vec: &[F128],
    eq_inner: &[F128],
    acc_a: &mut F128,
    acc_b: &mut F128,
) -> FWord {
    let mut sum = FWord::zero();
    let mut cin: F128 = F128::ZERO;
    for i in 0..WORD_BITS {
        sum.0[i] = x.0[i] + y.0[i] + cin;
        if i < WORD_BITS - 1 {
            let a_carry = x.0[i] + cin;
            let b_carry = y.0[i] + cin;
            let cs = carry_slot_fn(i);
            let eq_at_cs = eq_inner[cs];
            *acc_a += a_carry * eq_at_cs;
            *acc_b += b_carry * eq_at_cs;
            cin += z_vec[cs];
        }
    }
    sum
}

/// 32-bit add with carry rows AND materialized sum slot. The sum-slot rows
/// have B-row = [Z_CONST] uniformly; A-row carries the full sum expression.
fn add_alloc<F1: Fn(usize) -> usize, F2: Fn(usize) -> usize>(
    x: &FWord,
    y: &FWord,
    carry_slot_fn: F1,
    sum_slot_fn: F2,
    z_vec: &[F128],
    eq_inner: &[F128],
    acc_a: &mut F128,
    acc_b: &mut F128,
) -> FWord {
    let sum = add_inline(x, y, carry_slot_fn, z_vec, eq_inner, acc_a, acc_b);
    let z_const = z_vec[Z_CONST_POS];
    for b in 0..WORD_BITS {
        let ss = sum_slot_fn(b);
        let eq_at_ss = eq_inner[ss];
        *acc_a += sum.0[b] * eq_at_ss;
        *acc_b += z_const * eq_at_ss;
    }
    let mut slotted = [F128::ZERO; 32];
    for b in 0..32 {
        slotted[b] = z_vec[sum_slot_fn(b)];
    }
    FWord(slotted)
}

/// Compute `(v_a, v_b) = (<eq_inner, A_0·z_vec>, <eq_inner, B_0·z_vec>)`
/// in one fused walk through SHA-256's linear data flow on `z_vec`.
fn linear_ab_consistency(z_vec: &[F128], eq_inner: &[F128]) -> (F128, F128) {
    assert_eq!(z_vec.len(), K);
    assert_eq!(eq_inner.len(), K);
    let mut acc_a = F128::ZERO;
    let mut acc_b = F128::ZERO;

    // 1. Z_CONST row: A_row = B_row = [Z_CONST_POS] → both sides = z_vec[0].
    let z0 = z_vec[Z_CONST_POS];
    acc_a += z0 * eq_inner[Z_CONST_POS];
    acc_b += z0 * eq_inner[Z_CONST_POS];

    // 2. H_in, M_in free-witness rows: A=[slot], B=[Z_CONST].
    for w in 0..H_WORDS {
        for b in 0..WORD_BITS {
            let s = h_bit(w, b);
            let eq_s = eq_inner[s];
            acc_a += z_vec[s] * eq_s;
            acc_b += z0 * eq_s;
        }
    }
    for i in 0..M_WORDS {
        for b in 0..WORD_BITS {
            let s = m_bit(i, b);
            let eq_s = eq_inner[s];
            acc_a += z_vec[s] * eq_s;
            acc_b += z0 * eq_s;
        }
    }

    // 3. State words derived from H_in slots.
    let h_in: [FWord; 8] = std::array::from_fn(|w| FWord::read_from(z_vec, h_bit(w, 0)));

    // 4. Message schedule.
    let mut w_words: Vec<FWord> = (0..N_SCHED + 16).map(|_| FWord::zero()).collect();
    for i in 0..16 {
        w_words[i] = FWord::read_from(z_vec, m_bit(i, 0));
    }
    for t in 16..(16 + N_SCHED) {
        let s1 = sigma_1(&w_words[t - 2]);
        let s0 = sigma_0(&w_words[t - 15]);
        let w_m7 = w_words[t - 7];
        let w_m16 = w_words[t - 16];
        let sched_0 = add_inline(
            &s1,
            &w_m7,
            |i| sched_carry_bit(t, 0, i),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        let sched_1 = add_inline(
            &sched_0,
            &s0,
            |i| sched_carry_bit(t, 1, i),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        let w_t = add_alloc(
            &sched_1,
            &w_m16,
            |i| sched_carry_bit(t, 2, i),
            |b| w_bit(t, b),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        w_words[t] = w_t;
    }

    // 5. Working state.
    let mut state: [FWord; 8] = h_in;

    for r in 0..N_ROUNDS {
        let a = state[0];
        let bb = state[1];
        let c = state[2];
        let d = state[3];
        let e = state[4];
        let f = state[5];
        let g = state[6];
        let h_var = state[7];

        // ch_and AND row: A_0 = [e_bit], B_0 = [f_bit, g_bit].
        let mut ch_and = FWord::zero();
        for b in 0..WORD_BITS {
            let s = ch_and_bit(r, b);
            let eq_s = eq_inner[s];
            acc_a += e.0[b] * eq_s;
            acc_b += (f.0[b] + g.0[b]) * eq_s;
            ch_and.0[b] = z_vec[s];
        }
        // maj_and AND row: A_0 = [a, b state bits], B_0 = [a, c state bits].
        let mut maj_and = FWord::zero();
        for b in 0..WORD_BITS {
            let s = maj_and_bit(r, b);
            let eq_s = eq_inner[s];
            acc_a += (a.0[b] + bb.0[b]) * eq_s;
            acc_b += (a.0[b] + c.0[b]) * eq_s;
            maj_and.0[b] = z_vec[s];
        }
        let ch_out = ch_and.xor(&g);
        let maj_out = maj_and.xor(&a);

        let t1_0 = add_inline(
            &h_var,
            &big_sigma_1(&e),
            |i| round_carry_bit(r, 0, i),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        let t1_1 = add_inline(
            &t1_0,
            &ch_out,
            |i| round_carry_bit(r, 1, i),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        // K[r] constant: bit b of K[r] is 1 ⇒ that bit's contribution = z_vec[Z_CONST].
        let k_const = SHA256_K[r];
        let k_word: FWord = {
            let mut w = [F128::ZERO; 32];
            for b in 0..32 {
                if (k_const >> b) & 1 == 1 {
                    w[b] = z0;
                }
            }
            FWord(w)
        };
        let t1_2 = add_inline(
            &t1_1,
            &k_word,
            |i| round_carry_bit(r, 2, i),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        let t1 = add_alloc(
            &t1_2,
            &w_words[r],
            |i| round_carry_bit(r, 3, i),
            |b| t1_bit(r, b),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );

        let t2 = add_inline(
            &big_sigma_0(&a),
            &maj_out,
            |i| round_carry_bit(r, 4, i),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        let e_new = add_alloc(
            &d,
            &t1,
            |i| round_carry_bit(r, 5, i),
            |b| e_new_bit(r, b),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
        let a_new = add_alloc(
            &t1,
            &t2,
            |i| round_carry_bit(r, 6, i),
            |b| a_new_bit(r, b),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );

        state = [a_new, a, bb, c, e_new, e, f, g];
    }

    // 6. Output feed-forward.
    for w in 0..N_OUT_WORDS {
        let _ = add_alloc(
            &state[w],
            &h_in[w],
            |i| out_carry_bit(w, i),
            |b| h_out_bit(w, b),
            z_vec,
            eq_inner,
            &mut acc_a,
            &mut acc_b,
        );
    }

    (acc_a, acc_b)
}

// ───────────────────────────────────────────────────────────────────────────
// Reference path (the standard verifier).
// ───────────────────────────────────────────────────────────────────────────

fn standard_ab_consistency(
    a_0: &flock_prover::r1cs::SparseBinaryMatrix,
    b_0: &flock_prover::r1cs::SparseBinaryMatrix,
    z_vec: &[F128],
    eq_inner: &[F128],
) -> (F128, F128) {
    let a_row = sparse_row_fold(a_0, eq_inner);
    let b_row = sparse_row_fold(b_0, eq_inner);
    (inner_product(&a_row, z_vec), inner_product(&b_row, z_vec))
}

fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += *x * *y;
    }
    acc
}

// ───────────────────────────────────────────────────────────────────────────
// Driver: build a valid z_vec from a real SHA witness, generate eq_inner,
// check both paths agree, time both.
// ───────────────────────────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn main() {
    use flock_prover::r1cs_hashes::sha2::build_block_witness;

    let mut rng = Rng::new(0x00C0_FFEE_5A55);
    let inner_rest_len = K_LOG - K_SKIP;

    // ---- Build a z_vec. We use a real SHA-256 witness (boolean), packed
    // to F128 view (each bit becomes 0 or 1 as F128). This is just a
    // particular valid vector for correctness checking; the linear walk
    // works for ANY F128 vector of length K.
    let h_in = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let m_in: [u32; 16] = std::array::from_fn(|_| rng.next_u64() as u32);
    let z_bool = build_block_witness(&h_in, &m_in);
    let z_vec: Vec<F128> = z_bool
        .iter()
        .map(|&b| if b { F128::ONE } else { F128::ZERO })
        .collect();

    // ---- Build a realistic eq_inner via build_quirky_eq_table.
    let z_skip = rng.f128();
    let x_inner_rest: Vec<F128> = (0..inner_rest_len).map(|_| rng.f128()).collect();
    let eq_inner = build_quirky_eq_table(z_skip, &x_inner_rest, K_SKIP);
    assert_eq!(eq_inner.len(), K);

    // ---- Build A_0, B_0 once.
    let (a_0, b_0) = build_matrices();

    // ---- Correctness: both consistency checks.
    let (v_a_std, v_b_std) = standard_ab_consistency(&a_0, &b_0, &z_vec, &eq_inner);
    let (v_a_lin, v_b_lin) = linear_ab_consistency(&z_vec, &eq_inner);
    println!("=== Correctness (both A and B sides) ===");
    println!("  standard   v_a: {:032x?}", v_a_std);
    println!("  linear-SHA v_a: {:032x?}", v_a_lin);
    println!("  standard   v_b: {:032x?}", v_b_std);
    println!("  linear-SHA v_b: {:032x?}", v_b_lin);
    if v_a_std == v_a_lin && v_b_std == v_b_lin {
        println!("  ✓ both match");
    } else {
        println!("  ✗ MISMATCH");
        std::process::exit(1);
    }

    // ---- Timing (single-threaded, no rayon).
    println!();
    println!("=== Timing (single-threaded, full A+B consistency check) ===");
    const N_ITERS: usize = 200;

    // Warm up both paths.
    let _ = black_box(standard_ab_consistency(&a_0, &b_0, &z_vec, &eq_inner));
    let _ = black_box(linear_ab_consistency(&z_vec, &eq_inner));

    let t0 = Instant::now();
    for _ in 0..N_ITERS {
        let (va, vb) = standard_ab_consistency(&a_0, &b_0, &z_vec, &eq_inner);
        black_box((va, vb));
    }
    let dt_std = t0.elapsed().as_secs_f64() / N_ITERS as f64;

    let t1 = Instant::now();
    for _ in 0..N_ITERS {
        let (va, vb) = linear_ab_consistency(&z_vec, &eq_inner);
        black_box((va, vb));
    }
    let dt_lin = t1.elapsed().as_secs_f64() / N_ITERS as f64;

    println!(
        "  standard (2× sparse_row_fold + 2× inner_product):  {:>8.2} µs",
        dt_std * 1e6
    );
    println!(
        "  linear-SHA walk on z_vec (fused A+B):              {:>8.2} µs",
        dt_lin * 1e6
    );
    let ratio = dt_std / dt_lin;
    if ratio > 1.0 {
        println!("  → linear-SHA is {:.2}× faster", ratio);
    } else {
        println!(
            "  → linear-SHA is {:.2}× slower (1/{:.2}×)",
            1.0 / ratio,
            ratio
        );
    }
}
