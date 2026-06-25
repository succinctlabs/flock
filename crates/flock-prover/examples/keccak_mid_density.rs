//! Density measurement for the midpoint Keccak R1CS encoding.
//!
//! Goal: confirm whether dropping intermediate states (`state_1..11`,
//! `state_13..24`) and keeping only `state_0` + `state_12` + `t_0..t_23`
//! produces tractable A/B row weights after substituting
//! `state_r = L_r(state_0, t_0..t_{r-1})`.
//!
//! For each round r, every t-AND row's A and B side is a single bit of
//! φ(state_r) (XOR of 11 state_r bits via theta_rho_pi_preimage). With
//! state_r expressed symbolically as XOR over state_0 + t-prefix tokens,
//! the row weight is the deduplicated size of that XOR list.
//!
//! Reports avg/max A-row weight per round and total NNZ.
//!
//! Run: `cargo run --release --example keccak_mid_density`

use flock_prover::r1cs_hashes::keccak::{N_ROUNDS, STATE_BITS, state_idx, theta_rho_pi_preimage};

/// One "input token" of the symbolic state. Either a bit of `state_0`
/// (token = j) or a bit of `t_i` (token = 1600 + i*1600 + j). Round constants
/// from ι are ignored for density (they only add at most 1 const-tap per row).
type Token = u32;

#[inline]
fn token_state0(j: usize) -> Token {
    j as Token
}
#[inline]
fn token_t(i: usize, j: usize) -> Token {
    (STATE_BITS + i * STATE_BITS + j) as Token
}

/// 1600-bit deferred-XOR state: bit j = XOR over `bits[j]` tokens.
struct StateExpr {
    bits: Vec<Vec<Token>>,
}

impl StateExpr {
    fn from_state0() -> Self {
        Self {
            bits: (0..STATE_BITS).map(|j| vec![token_state0(j)]).collect(),
        }
    }
    /// `next = phi(self) XOR t_{r_minus_1}` where phi = theta∘rho∘pi.
    /// Round-constant bits are NOT added (negligible density impact).
    fn step(&self, r_minus_1: usize) -> Self {
        let mut next = Vec::with_capacity(STATE_BITS);
        for z in 0..64 {
            for y in 0..5 {
                for x in 0..5 {
                    // φ(state)[x,y,z] = XOR of 11 state bits at theta_rho_pi_preimage.
                    let preim = theta_rho_pi_preimage(x, y, z);
                    let mut acc: Vec<Token> = Vec::with_capacity(64);
                    for &k in preim.iter() {
                        acc.extend_from_slice(&self.bits[k]);
                    }
                    // XOR with t_{r-1}[j] (one new token).
                    acc.push(token_t(r_minus_1, state_idx(x, y, z)));
                    next.push(dedup(acc));
                }
            }
        }
        Self { bits: next }
    }
}

fn dedup(mut v: Vec<Token>) -> Vec<Token> {
    v.sort_unstable();
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        let val = v[i];
        let mut count = 0;
        while i < v.len() && v[i] == val {
            count += 1;
            i += 1;
        }
        if count % 2 == 1 {
            out.push(val);
        }
    }
    out
}

/// For state_r, compute per-round A-side row weights. A row at (x,y,z) is a
/// single bit of φ(state_r) at ((x+1)%5, y, z), i.e., XOR of 11 state_r bits.
fn row_weights(s: &StateExpr) -> Vec<usize> {
    let mut weights = Vec::with_capacity(STATE_BITS);
    for z in 0..64 {
        for y in 0..5 {
            for x in 0..5 {
                let preim = theta_rho_pi_preimage((x + 1) % 5, y, z);
                let mut acc: Vec<Token> = Vec::with_capacity(64);
                for &k in preim.iter() {
                    acc.extend_from_slice(&s.bits[k]);
                }
                let row = dedup(acc);
                weights.push(row.len());
            }
        }
    }
    weights
}

fn summarize(round: usize, weights: &[usize]) -> (f64, usize, usize) {
    let n = weights.len();
    let sum: usize = weights.iter().sum();
    let max = *weights.iter().max().unwrap();
    let min = *weights.iter().min().unwrap();
    let avg = sum as f64 / n as f64;
    println!(
        "  round {round:>2}: avg = {avg:>7.1}  min = {min:>4}  max = {max:>5}  total NNZ = {sum:>10}"
    );
    (avg, max, sum)
}

fn main() {
    println!("Mid-encoded Keccak A/B row density per round.");
    println!(
        "(state_r = L_r(state_0, t_0..t_{{r-1}}); A-row at (x,y,z) is a single bit of φ(state_r))"
    );
    println!();

    // First half: state_r threaded from state_0 (r = 0..12).
    println!("Threading state_r from state_0:");
    let mut s = StateExpr::from_state0();
    let mut total_nnz: u64 = 0;
    let mut total_rows: u64 = 0;
    // r = 0: A-row is on state_0 directly.
    let w0 = row_weights(&s);
    let (_, _, nnz0) = summarize(0, &w0);
    total_nnz += nnz0 as u64;
    total_rows += w0.len() as u64;
    // r = 1..11: thread forward, then measure.
    for r in 1..12 {
        s = s.step(r - 1);
        let w = row_weights(&s);
        let (_, _, nnz) = summarize(r, &w);
        total_nnz += nnz as u64;
        total_rows += w.len() as u64;
    }

    // Second half: state_r threaded from state_12 (r = 12..24). state_12 is
    // a "fresh" input (its taps reference state_12 slots, not state_0/t's),
    // so density restarts at the baseline.
    println!("\nThreading state_r from state_12:");
    let mut s = StateExpr::from_state0(); // symbol meaning: token IDs reference state_12 slots, same shape
    // r = 12: baseline.
    let w12 = row_weights(&s);
    let (_, _, nnz12) = summarize(12, &w12);
    total_nnz += nnz12 as u64;
    total_rows += w12.len() as u64;
    // r = 13..23: thread forward.
    for r in 13..N_ROUNDS {
        s = s.step(r - 1);
        let w = row_weights(&s);
        let (_, _, nnz) = summarize(r, &w);
        total_nnz += nnz as u64;
        total_rows += w.len() as u64;
    }

    println!(
        "\n=== Total ===  A-side rows: {total_rows}  NNZ: {total_nnz}  avg row weight: {:.1}",
        total_nnz as f64 / total_rows as f64
    );
    println!("(B side identical density — symmetric)");

    // Baseline reference: current (full) encoding has ~11 taps per A and B
    // row at every t-AND. With 24 rounds × 1600 = 38,400 A-rows, total NNZ
    // would be 38,400 × 11 ≈ 422,400 per side. Density blow-up factor:
    let baseline_nnz: u64 = 24 * 1600 * 11;
    println!(
        "Baseline (current full encoding, K_LOG=17): {} NNZ  →  blow-up: {:.1}×",
        baseline_nnz,
        total_nnz as f64 / baseline_nnz as f64,
    );
}
