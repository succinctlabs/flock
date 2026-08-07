//! Grand-product permutation check via a **product-circuit GKR**.
//!
//! Proves that `f, g` over `B_μ` (`N = 2^μ`) are related by a permutation `σ`
//! through the grand-product identity
//!
//!   ∏_i (f_i + α·s_id(i) + β)  =  ∏_i (g_i + α·s_σ(i) + β)
//!
//! for random `α, β`, where `s_id` is the injective index tag and
//! `s_σ(i) = s_id(σ(i))`. (Plonk copy-constraints are the `f = g = w` case —
//! the relation the recursive-verifier wiring uses, `∏(w_i + α·i + β) =
//! ∏(w_i + α·σ(i) + β)`.) The two products are equal as polynomials in `α, β`
//! iff the multisets `{(f_i, s_id(i))} = {(g_i, s_σ(i))}` match.
//!
//! ## The GKR circuit (difference from the siblings)
//!
//! [`crate::permutation`] proves one grand product by committing the product
//! tree as a multilinear `v` and opening it. This module proves each grand
//! **product** with the classic product-tree GKR: a binary tree of plain
//! multiplication gates, with **no committed oracle and no field inversions**.
//! For an input vector `V_μ` of `2^μ` values,
//!
//!   V_k[i] = V_{k+1}[i] · V_{k+1}[i + 2^k]   (high-bit pairing),
//!
//! so `V_0` is the total product. Two such circuits are run — one for the LHS
//! vector `lhs_i = f_i + α·s_id(i) + β`, one for the RHS `rhs_i = g_i +
//! α·s_σ(i) + β` — and the verifier checks their roots are equal.
//!
//! ## Protocol
//!
//! Standard GKR, per product circuit. Reduce a claim `V_k(r_k)` (point `r_k ∈
//! F^k`) to a claim at layer `k+1`:
//!
//!   1. `V_k(r_k) = Σ_{x∈B_k} eq(r_k,x)·V_{k+1}(x,0)·V_{k+1}(x,1)` — a `k`-round
//!      eq-weighted **degree-2** sumcheck (Gruen eq-trick, Convention A: send
//!      bare core `(G(1), G(∞))`), reducing to a random `r' ∈ F^k`;
//!   2. the prover sends the two boundary values `V_{k+1}(r',0), V_{k+1}(r',1)`;
//!      the verifier checks the sumcheck's final value equals their product;
//!   3. sample `c_k`; collapse `(r',0),(r',1)` to `(r', c_k)` by linear
//!      interpolation — the next layer's point and claim `V_{k+1}(r', c_k)`.
//!
//! After `μ` layers the claim lands on `V_μ(ρ)` at `ρ ∈ F^μ`, checked against
//! the input value reconstructed from `f(ρ)` (resp. `g(ρ), s_σ(ρ)`) — affine in
//! the witness, so the verifier rebuilds it from the surfaced evals plus the
//! closed-form `s_id(ρ)` (resp. the verifier-known `s_σ(ρ)`).
//!
//! ## Scope & cost
//!
//! PIOP for the witness side, same contract as [`crate::permutation`]: reduces
//! to MLE eval claims on the witness, returned in the claim type. **No PCS
//! commitment, no PCS opening, no inversions** — the proof is just the GKR
//! transcript (`O(μ²)` field elements) plus the witness evals, and the prover
//! is `O(N)` field multiplications.
//!
//! ## Which entry point
//!
//! Prefer [`prove_batched`] / [`verify_batched`]. It runs both circuits in
//! lockstep under one λ-combined sumcheck per layer instead of two independent
//! chains: half the rounds, half the round messages on the wire, and — the part
//! that matters downstream — a **single** reduction point `ρ`, so the witness
//! PCS opens `f, g, s_σ` at one point. [`prove`] keeps the two circuits
//! separate and lands `f` at `ρ_lhs` with `g, s_σ` at `ρ_rhs`. Measured at
//! μ=20 on an M4 Max: batched 7.6 ms vs 15.1 ms, proof 7.4 KiB vs 13.5 KiB.
//!
//! [`prove_batched`] additionally fuses each fold with the next round's message
//! (`fold_and_message`) and reconstructs `f(ρ)`/`g(ρ)` in closed form from the
//! final layer's collapse, so only `s_σ` — which has no closed form — costs an
//! `O(N)` MLE evaluation. [`prove`] does neither.
//!
//! (A sibling `logup_gkr` — a fractional **sum** GKR with an `a/b ⊕ c/d` gate —
//! exists on the `recursive-verifier` branch of `flock-dev`, not ported here.)

use crate::scratch::give_f128;
use crate::scratch::take_f128;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::array::from_fn;
use std::env::var;
#[cfg(test)]
use std::hint::black_box;
use std::mem::replace;
use std::mem::swap;
use std::sync::OnceLock;
use std::time::Instant;

use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::univariate_skip::SplitEqGhash;

const DOMAIN: &[u8] = b"flock-product-gkr-v0";

// ---------------------------------------------------------------------------
// Proof / claim / error types
// ---------------------------------------------------------------------------

/// One product-circuit layer reduction (`layer k → k+1`): the `k`-round sumcheck
/// messages (Convention A `(G(1), G(∞))`) and the two boundary values
/// `V_{k+1}(r', 0), V_{k+1}(r', 1)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerProof {
    /// Per-round `(G(1), G(∞))`, length `k` (empty for the top layer `k=0`).
    pub rounds: Vec<(F128, F128)>,
    pub v0: F128, // V_{k+1}(r', 0)
    pub v1: F128, // V_{k+1}(r', 1)
}

/// Product-GKR permutation proof. Two product transcripts (LHS/RHS), their
/// roots, and the witness evals at the two reduction points.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductGkrProof {
    /// `∏ lhs` and `∏ rhs`; must be equal for a valid permutation.
    pub top_lhs: F128,
    pub top_rhs: F128,
    /// Layer reductions `k = 0…μ-1` for each circuit.
    pub layers_lhs: Vec<LayerProof>,
    pub layers_rhs: Vec<LayerProof>,
    pub f_eval: F128,       // f(ρ_lhs)
    pub g_eval: F128,       // g(ρ_rhs)
    pub s_sigma_eval: F128, // s_σ(ρ_rhs)
}

/// Evaluation claims the verifier outputs, for a downstream witness PCS. The
/// two products reduce to *different* points, so `f` and `g` are claimed at
/// `ρ_lhs` and `ρ_rhs` respectively.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductGkrClaim {
    pub rho_lhs: Vec<F128>,
    pub rho_rhs: Vec<F128>,
    pub f_eval: F128,
    pub g_eval: F128,
    pub s_sigma_eval: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `∏ lhs ≠ ∏ rhs`: the two grand products differ, so the witnesses are
    /// not a valid permuted pair.
    ProductMismatch,
    /// A per-layer sumcheck's final value disagreed with the product of the
    /// claimed boundary values.
    LayerCheckFailed,
    /// A final layer claim disagreed with the witness-derived input value.
    InputMismatch,
    /// The proof is not shaped for this `μ`: wrong number of layers, or a layer
    /// carrying the wrong number of sumcheck rounds. Returned rather than
    /// panicked so an untrusted proof cannot take the verifier down.
    MalformedProof,
}

// ---------------------------------------------------------------------------
// Field / polynomial helpers (shared shape with the sibling GKR module)
// ---------------------------------------------------------------------------

/// Basis for the identity tag `s_id`: `basis[i]` is the field element with bit
/// `i` set (requires `μ ≤ 128`).
fn s_id_basis(mu: usize) -> Vec<F128> {
    assert!(mu <= 128, "s_id needs μ ≤ 128 distinct bit positions");
    (0..mu)
        .map(|i| {
            if i < 64 {
                F128::new(1u64 << i, 0)
            } else {
                F128::new(0, 1u64 << (i - 64))
            }
        })
        .collect()
}

/// `s_id` on the hypercube: the field element whose bit pattern equals `idx`.
#[cfg(test)]
fn s_id_value(idx: usize, basis: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (i, b) in basis.iter().enumerate() {
        if (idx >> i) & 1 == 1 {
            acc += *b;
        }
    }
    acc
}

/// Closed-form MLE of `s_id` at `ρ`: `Σ_i basis_i · ρ_i`.
fn s_id_eval(basis: &[F128], rho: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (b, r) in basis.iter().zip(rho) {
        acc += *b * *r;
    }
    acc
}

/// The whole `s_id` table over `B_μ`, built by doubling in `O(N)`.
fn build_s_id_vec(mu: usize, basis: &[F128]) -> Vec<F128> {
    let n = 1usize << mu;
    let mut v = vec![F128::ZERO; n];
    for (k, &bk) in basis.iter().enumerate() {
        let half = 1usize << k;
        let (lo, hi) = v.split_at_mut(half);
        if half >= (1 << 12) {
            hi[..half]
                .par_iter_mut()
                .zip(lo.par_iter())
                .for_each(|(dst, src)| *dst = *src + bk);
        } else {
            for (dst, src) in hi.iter_mut().zip(lo.iter()) {
                *dst = *src + bk;
            }
        }
    }
    v
}

/// Threshold for the embarrassingly-parallel gate loop. Overridable via
/// `FLOCK_GKR_GATE` for tuning.
///
/// At μ=20 this keeps the four widest layers — ~94% of the gate work — on the
/// pool while the geometrically shrinking tail runs serial instead of paying a
/// rayon dispatch per layer. Measured best-of-3 on an M4 Max: 0.86 ms here
/// against 1.22 ms at the previous `1<<12`.
const PAR_THRESHOLD_DEFAULT: usize = 1 << 16;

fn par_threshold() -> usize {
    static GATE: OnceLock<usize> = OnceLock::new();
    *GATE.get_or_init(|| {
        var("FLOCK_GKR_GATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PAR_THRESHOLD_DEFAULT)
    })
}

/// Phase tracing, enabled by `GKR_TRACE=1` (mirrors `PERM_TRACE` in
/// [`crate::permutation`]). Read once.
fn trace_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| var("GKR_TRACE").is_ok())
}

/// Print `label` with the elapsed time since `t`, then reset `t`. No-op unless
/// [`trace_on`].
fn tp(t: &mut Instant, label: &str) {
    if trace_on() {
        eprintln!(
            "  [prod-gkr] {label:<16} {:8.3} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        *t = Instant::now();
    }
}

/// Bind the low variable at `ρ`: `u[x] ← u[2x]·(1+ρ) + u[2x+1]·ρ`, halving `u`.
fn fold_in_place(u: &mut Vec<F128>, rho: F128) {
    let half = u.len() / 2;
    let one_minus = F128::ONE + rho;
    match crate::fold_min_len(half) {
        Some(min_len) => {
            let mut out = take_f128(half);
            out.par_iter_mut()
                .enumerate()
                .with_min_len(min_len)
                .for_each(|(x, o)| {
                    *o = u[2 * x] * one_minus + u[2 * x + 1] * rho;
                });
            let old = replace(u, out);
            give_f128(old);
        }
        None => {
            for x in 0..half {
                u[x] = u[2 * x] * one_minus + u[2 * x + 1] * rho;
            }
            u.truncate(half);
        }
    }
}

/// Bind the low variable of a **borrowed** slice at `ρ`, returning the folded
/// half — the fold *is* the copy (no separate clone of a layer half).
fn fold_borrowed(src: &[F128], rho: F128) -> Vec<F128> {
    let half = src.len() / 2;
    let one_minus = F128::ONE + rho;
    // Pooled: every slot is written below, so the recycled buffer's stale
    // contents are irrelevant (same contract as `alloc_uninit_f128_vec`).
    // Callers hand these back via `scratch::give_f128` once a layer is done.
    // This is resource hygiene, not a speed win — measured neutral at μ=20,
    // since the fold is bound by memory traffic rather than by the allocator.
    let mut out = take_f128(half);
    match crate::fold_min_len(half) {
        Some(min_len) => {
            out.par_iter_mut()
                .enumerate()
                .with_min_len(min_len)
                .for_each(|(x, o)| {
                    *o = src[2 * x] * one_minus + src[2 * x + 1] * rho;
                });
        }
        None => {
            for (x, o) in out.iter_mut().enumerate() {
                *o = src[2 * x] * one_minus + src[2 * x + 1] * rho;
            }
        }
    }
    out
}

/// Bind the low variable of `src` at `ρ`, writing the folded half into
/// `dst[..src.len()/2]` — a caller-owned destination instead of a fresh
/// allocation.
///
/// [`prove_batched`] hoists its eight working buffers out of the layer loop and
/// ping-pongs them, replacing a per-round `take_f128`/`give_f128` pair. That
/// gives the buffers an explicit lifecycle and one allocation per prove, but it
/// is **not** what made the fold phase scale: measured on its own it was within
/// noise. The fold's poor thread scaling (1.06× on ten cores) turned out to be
/// the sub-gate fan-out rule, not page faults — see [`crate::fold_sqrt_rule`].
fn fold_into(src: &[F128], rho: F128, dst: &mut [F128]) {
    let half = src.len() / 2;
    let one_minus = F128::ONE + rho;
    let out = &mut dst[..half];
    match crate::fold_min_len(half) {
        Some(min_len) => {
            out.par_iter_mut()
                .enumerate()
                .with_min_len(min_len)
                .for_each(|(x, o)| {
                    *o = src[2 * x] * one_minus + src[2 * x + 1] * rho;
                });
        }
        None => {
            for (x, o) in out.iter_mut().enumerate() {
                *o = src[2 * x] * one_minus + src[2 * x + 1] * rho;
            }
        }
    }
}

/// Fold four half-slices at `ρ` into `dst`, **and in the same pass** compute the
/// *next* round's λ-combined message over the values just folded.
///
/// Unfused, each round reads all four vectors twice: once for its message, once
/// to fold them. A layer's eq tables all derive from the *previous* layer's
/// `r_pt`, so round `i+1`'s eq is known before round `i` folds — which lets the
/// fold emit it. Each hi-block folds its outputs and then immediately reads them
/// back while they are still in cache, so the second traversal never returns to
/// memory.
///
/// `eq_next` is `None` on a layer's last round (nothing follows), where this
/// degenerates to a plain fold. Returns the next round's `(G(1), G(∞))`.
///
/// Blocking differs from [`batched_round_message`], but `F128` addition is XOR —
/// exactly associative and commutative — and multiplication distributes over it,
/// so `eh·Σ(el·v)` regroups freely. The transcript is unchanged.
fn fold_and_message(
    src: [&[F128]; 4],
    rho: F128,
    dst: [&mut [F128]; 4],
    lambda: F128,
    eq_next: Option<&SplitEqGhash>,
) -> Option<(F128, F128)> {
    let half = src[0].len() / 2;
    let one_minus = F128::ONE + rho;
    let [s0, s1, s2, s3] = src;
    let [d0, d1, d2, d3] = dst;

    let Some(eq) = eq_next else {
        for (s, d) in [s0, s1, s2, s3].into_iter().zip([d0, d1, d2, d3]) {
            fold_into(s, rho, d);
        }
        return None;
    };

    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len();
    let n_blocks = hi.len();
    debug_assert_eq!(block * n_blocks, half / 2);
    // One hi-block owns `block` pairs of folded values, i.e. `2·block` outputs.
    let chunk = 2 * block;

    let body = |x_hi: usize, c: [&mut [F128]; 4]| -> (F128, F128) {
        let base = x_hi * chunk;
        let [c0, c1, c2, c3] = c;
        for t in 0..c0.len() {
            let x = base + t;
            let (a, b) = (2 * x, 2 * x + 1);
            c0[t] = s0[a] * one_minus + s0[b] * rho;
            c1[t] = s1[a] * one_minus + s1[b] * rho;
            c2[t] = s2[a] * one_minus + s2[b] * rho;
            c3[t] = s3[a] * one_minus + s3[b] * rho;
        }
        // Next round's message over this block's freshly folded values.
        let (mut acc1, mut acc_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let (i0, i1) = (2 * x_lo, 2 * x_lo + 1);
            let v_one = c0[i1] * c1[i1] + lambda * (c2[i1] * c3[i1]);
            let v_inf = (c0[i0] + c0[i1]) * (c1[i0] + c1[i1])
                + lambda * ((c2[i0] + c2[i1]) * (c3[i0] + c3[i1]));
            let el = lo[x_lo];
            acc1 += el * v_one;
            acc_inf += el * v_inf;
        }
        let eh = hi[x_hi];
        (eh * acc1, eh * acc_inf)
    };

    let msg = match crate::sumcheck_round_min_len(block * n_blocks, n_blocks) {
        Some(min_len) => d0[..half]
            .par_chunks_mut(chunk)
            .zip(d1[..half].par_chunks_mut(chunk))
            .zip(d2[..half].par_chunks_mut(chunk))
            .zip(d3[..half].par_chunks_mut(chunk))
            .with_min_len(min_len)
            .enumerate()
            .map(|(x_hi, (((c0, c1), c2), c3))| body(x_hi, [c0, c1, c2, c3]))
            .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d)),
        None => {
            let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
            for (x_hi, (((c0, c1), c2), c3)) in d0[..half]
                .chunks_mut(chunk)
                .zip(d1[..half].chunks_mut(chunk))
                .zip(d2[..half].chunks_mut(chunk))
                .zip(d3[..half].chunks_mut(chunk))
                .enumerate()
            {
                let (o, i) = body(x_hi, [c0, c1, c2, c3]);
                g_one += o;
                g_inf += i;
            }
            (g_one, g_inf)
        }
    };
    Some(msg)
}

/// Direct MLE evaluation of `table` (length `2^k`) at `point` (length `k`),
/// binding the low variable first.
fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
    let Some((&first, rest)) = point.split_first() else {
        return table[0];
    };
    let mut t = fold_borrowed(table, first);
    for &r in rest {
        fold_in_place(&mut t, r);
    }
    t[0]
}

// ---------------------------------------------------------------------------
// Product circuit build + layer sumcheck round message
// ---------------------------------------------------------------------------

/// Build a product-circuit layer from the one below it (high-bit pairing): for
/// `i` in `[0, h)` with `h = 2^k`, `V_k[i] = V_{k+1}[i] · V_{k+1}[i + h]`.
fn build_layer(v_next: &[F128]) -> Vec<F128> {
    let h = v_next.len() / 2;
    let gate = |i: usize| v_next[i] * v_next[i + h];
    if h >= par_threshold() {
        (0..h).into_par_iter().map(gate).collect()
    } else {
        (0..h).map(gate).collect()
    }
}

/// One eq-weighted degree-2 round for the product gate, **excluding** the
/// current variable's eq factor (Convention A). `v0 = V_{k+1}(·,0)`,
/// `v1 = V_{k+1}(·,1)` are the (partially folded) half-slices; the per-element
/// gate value is `v0·v1`. Returns `(G(1), G(∞))` with `eq` supplied split as
/// `eq_lo ⊗ eq_hi`.
fn layer_round_message(v0: &[F128], v1: &[F128], eq: &SplitEqGhash) -> (F128, F128) {
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len();
    let n_blocks = hi.len();
    debug_assert_eq!(block * n_blocks, v0.len() / 2);

    let block_fn = |x_hi: usize| -> (F128, F128) {
        let x_base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let xp = x_base + x_lo;
            let (i0, i1) = (2 * xp, 2 * xp + 1);
            // value at x_i = 1 (odd slice), and the degree-2 leading coeff.
            let v_one = v0[i1] * v1[i1];
            let v_inf = (v0[i0] + v0[i1]) * (v1[i0] + v1[i1]);
            let el = lo[x_lo];
            s1 += el * v_one;
            s_inf += el * v_inf;
        }
        let eh = hi[x_hi];
        (eh * s1, eh * s_inf)
    };

    match crate::sumcheck_round_min_len(block * n_blocks, n_blocks) {
        Some(min_len) => (0..n_blocks)
            .into_par_iter()
            .with_min_len(min_len)
            .map(block_fn)
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(o0, i0), (o1, i1)| (o0 + o1, i0 + i1),
            ),
        None => {
            let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
            for x_hi in 0..n_blocks {
                let (o, i) = block_fn(x_hi);
                g_one += o;
                g_inf += i;
            }
            (g_one, g_inf)
        }
    }
}

// ---------------------------------------------------------------------------
// Single product-circuit GKR (prover + verifier halves)
// ---------------------------------------------------------------------------

/// Prove `∏ v_in` with a product-circuit GKR. Observes the root + per-layer
/// transcript into `ch`, returns `(top, layers, ρ)` where `ρ ∈ F^μ` is the
/// final reduction point (so `V_μ(ρ) = v_in`'s MLE at ρ).
fn prove_product<C: Challenger>(v_in: &[F128], ch: &mut C) -> (F128, Vec<LayerProof>, Vec<F128>) {
    let mu = v_in.len().trailing_zeros() as usize;

    // Build all layers, v_layers[k] has 2^k entries; v_layers[mu] = v_in.
    let mut tt = Instant::now();
    let mut v_layers: Vec<Vec<F128>> = vec![Vec::new(); mu + 1];
    v_layers[mu] = v_in.to_vec();
    for k in (0..mu).rev() {
        v_layers[k] = build_layer(&v_layers[k + 1]);
    }
    let top = v_layers[0][0];
    ch.observe_f128(top);
    tp(&mut tt, "  build-layers");

    let mut r_pt: Vec<F128> = Vec::new();
    let mut layers = Vec::with_capacity(mu);
    for k in 0..mu {
        let h = 1usize << k;
        let (mut s0, mut s1) = (Vec::new(), Vec::new());
        let mut rounds = Vec::with_capacity(k);
        let mut r_prime = Vec::with_capacity(k + 1);
        for i in 0..k {
            let eq = SplitEqGhash::new(&r_pt[i + 1..k]);
            let rho;
            if i == 0 {
                let (v0s, v1s) = v_layers[k + 1].split_at(h);
                let (g1, g_inf) = layer_round_message(v0s, v1s, &eq);
                ch.observe_f128(g1);
                ch.observe_f128(g_inf);
                rho = ch.sample_f128();
                rounds.push((g1, g_inf));
                s0 = fold_borrowed(v0s, rho);
                s1 = fold_borrowed(v1s, rho);
            } else {
                let (g1, g_inf) = layer_round_message(&s0, &s1, &eq);
                ch.observe_f128(g1);
                ch.observe_f128(g_inf);
                rho = ch.sample_f128();
                rounds.push((g1, g_inf));
                fold_in_place(&mut s0, rho);
                fold_in_place(&mut s1, rho);
            }
            r_prime.push(rho);
        }
        let (v0, v1) = if k == 0 {
            (v_layers[1][0], v_layers[1][1])
        } else {
            (s0[0], s1[0])
        };
        ch.observe_f128(v0);
        ch.observe_f128(v1);
        layers.push(LayerProof { rounds, v0, v1 });

        let c_k = ch.sample_f128();
        r_prime.push(c_k);
        r_pt = r_prime;
    }
    tp(&mut tt, "  layer-sumchecks");
    (top, layers, r_pt)
}

/// Verify a single product circuit's GKR transcript. Returns the final input
/// claim `V_μ(ρ)` and the point `ρ` (length `μ`).
fn verify_product<C: Challenger>(
    mu: usize,
    top: F128,
    layers: &[LayerProof],
    ch: &mut C,
) -> Result<(F128, Vec<F128>), VerifyError> {
    if layers.len() != mu {
        return Err(VerifyError::MalformedProof);
    }
    ch.observe_f128(top);

    let mut v_claim = top;
    let mut r_pt: Vec<F128> = Vec::new();
    for (k, layer) in layers.iter().enumerate() {
        if layer.rounds.len() != k {
            return Err(VerifyError::MalformedProof);
        }
        let mut c_run = v_claim;
        let mut r_prime = Vec::with_capacity(k + 1);
        for i in 0..k {
            let (g1, g_inf) = layer.rounds[i];
            let r_eq = r_pt[i];
            let one_plus_r_eq = F128::ONE + r_eq;
            let g0 = (c_run + r_eq * g1) * one_plus_r_eq.inv();
            ch.observe_f128(g1);
            ch.observe_f128(g_inf);
            let rho = ch.sample_f128();
            r_prime.push(rho);
            let one_plus_rho = F128::ONE + rho;
            c_run = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
        }

        let (v0, v1) = (layer.v0, layer.v1);
        ch.observe_f128(v0);
        ch.observe_f128(v1);
        if c_run != v0 * v1 {
            return Err(VerifyError::LayerCheckFailed);
        }

        let c_k = ch.sample_f128();
        let one_plus_c = F128::ONE + c_k;
        v_claim = one_plus_c * v0 + c_k * v1;
        r_prime.push(c_k);
        r_pt = r_prime;
    }
    Ok((v_claim, r_pt))
}

// ---------------------------------------------------------------------------
// Prover / Verifier (the permutation check)
// ---------------------------------------------------------------------------

/// Prove that `f, g` are related by `σ` via the two-product GKR. `f.len() ==
/// g.len() == σ.len() == 2^μ`; `σ` must be a permutation. The caller must have
/// absorbed `f, g, σ` into `ch`.
pub fn prove<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    ch: &mut C,
) -> (ProductGkrProof, ProductGkrClaim) {
    let n = f.len();
    assert_eq!(g.len(), n);
    assert_eq!(sigma.len(), n);
    assert!(n.is_power_of_two() && n >= 2, "need N = 2^μ ≥ 2");
    let mu = n.trailing_zeros() as usize;

    let mut t = Instant::now();
    ch.observe_label(DOMAIN);
    let alpha = ch.sample_f128();
    let beta = ch.sample_f128();

    let basis = s_id_basis(mu);
    let s_id_vec = build_s_id_vec(mu, &basis);
    let s_sig_vec: Vec<F128> = sigma.par_iter().map(|&sx| s_id_vec[sx]).collect();

    // lhs_i = f_i + α·s_id(i) + β,  rhs_i = g_i + α·s_σ(i) + β.
    let lhs: Vec<F128> = f
        .par_iter()
        .zip(&s_id_vec)
        .map(|(fx, sx)| *fx + alpha * *sx + beta)
        .collect();
    let rhs: Vec<F128> = g
        .par_iter()
        .zip(&s_sig_vec)
        .map(|(gx, sx)| *gx + alpha * *sx + beta)
        .collect();

    tp(&mut t, "witness");
    let (top_lhs, layers_lhs, rho_lhs) = prove_product(&lhs, ch);
    tp(&mut t, "gkr(lhs)");
    let (top_rhs, layers_rhs, rho_rhs) = prove_product(&rhs, ch);
    tp(&mut t, "gkr(rhs)");

    let f_eval = mle_eval(f, &rho_lhs);
    let g_eval = mle_eval(g, &rho_rhs);
    let s_sigma_eval = mle_eval(&s_sig_vec, &rho_rhs);
    observe_evals(ch, &[f_eval, g_eval, s_sigma_eval]);
    tp(&mut t, "mle-evals");

    let proof = ProductGkrProof {
        top_lhs,
        top_rhs,
        layers_lhs,
        layers_rhs,
        f_eval,
        g_eval,
        s_sigma_eval,
    };
    let claim = ProductGkrClaim {
        rho_lhs,
        rho_rhs,
        f_eval,
        g_eval,
        s_sigma_eval,
    };
    (proof, claim)
}

/// Verify a product-GKR permutation proof for `N = 2^mu`. The caller must have
/// absorbed the same `f, g, σ` binding into `ch` as the prover did.
pub fn verify<C: Challenger>(
    mu: usize,
    proof: &ProductGkrProof,
    ch: &mut C,
) -> Result<ProductGkrClaim, VerifyError> {
    ch.observe_label(DOMAIN);
    let alpha = ch.sample_f128();
    let beta = ch.sample_f128();

    let (v_lhs, rho_lhs) = verify_product(mu, proof.top_lhs, &proof.layers_lhs, ch)?;
    let (v_rhs, rho_rhs) = verify_product(mu, proof.top_rhs, &proof.layers_rhs, ch)?;

    // The two grand products must agree.
    if proof.top_lhs != proof.top_rhs {
        return Err(VerifyError::ProductMismatch);
    }

    // Input-layer checks: V_μ(ρ) must equal the affine input value.
    let basis = s_id_basis(mu);
    let lhs_in = proof.f_eval + alpha * s_id_eval(&basis, &rho_lhs) + beta;
    let rhs_in = proof.g_eval + alpha * proof.s_sigma_eval + beta;
    if v_lhs != lhs_in || v_rhs != rhs_in {
        return Err(VerifyError::InputMismatch);
    }

    observe_evals(ch, &[proof.f_eval, proof.g_eval, proof.s_sigma_eval]);

    Ok(ProductGkrClaim {
        rho_lhs,
        rho_rhs,
        f_eval: proof.f_eval,
        g_eval: proof.g_eval,
        s_sigma_eval: proof.s_sigma_eval,
    })
}

fn observe_evals<C: Challenger>(ch: &mut C, evals: &[F128; 3]) {
    for e in evals {
        ch.observe_f128(*e);
    }
}

// ---------------------------------------------------------------------------
// Batched (shared-point) variant: run the two product circuits in lockstep, so
// both reduce to the SAME point ρ and the witness is opened ONCE.
// ---------------------------------------------------------------------------
//
// Each layer combines the two circuits' claims with a fresh `λ_k`:
//   V^L_k(r_k) ⊕ λ_k·V^R_k(r_k)
//     = Σ_x eq(r_k,x)·[V^L(x,0)V^L(x,1) ⊕ λ_k·V^R(x,0)V^R(x,1)],
// one `k`-round eq-weighted degree-2 sumcheck (Convention A). The pairing bit is
// shared, so after `μ` layers both circuits land on the same `ρ ∈ F^μ`, and
// `lhs(ρ) = w(ρ)+α·s_id(ρ)+β`, `rhs(ρ) = w(ρ)+α·s_σ(ρ)+β` share the single
// witness eval `w(ρ)` (= `f(ρ) = g(ρ)` for the copy-constraint `f=g=w` case).
// The verifier therefore needs just ONE evaluation of the committed witness —
// batchable as a single `PackedDirectClaim` into flock's opening.

const DOMAIN_BATCHED: &[u8] = b"flock-product-gkr-batched-v0";

/// One batched layer reduction (`layer k → k+1`): the `k`-round sumcheck
/// messages `(G(1), G(∞))` and the four boundary values
/// `V^L(r',0), V^L(r',1), V^R(r',0), V^R(r',1)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchedLayerProof {
    pub rounds: Vec<(F128, F128)>,
    pub vl0: F128,
    pub vl1: F128,
    pub vr0: F128,
    pub vr1: F128,
}

/// Batched product-GKR proof: both grand products' roots, the shared per-layer
/// reductions, and the (single-point) witness evals.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductGkrBatchedProof {
    pub top_lhs: F128,
    pub top_rhs: F128,
    pub layers: Vec<BatchedLayerProof>,
    pub f_eval: F128,       // f(ρ)
    pub g_eval: F128,       // g(ρ)
    pub s_sigma_eval: F128, // s_σ(ρ)
}

/// Evaluation claims at the SINGLE shared point `ρ`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductGkrBatchedClaim {
    pub rho: Vec<F128>,
    pub f_eval: F128,
    pub g_eval: F128,
    pub s_sigma_eval: F128,
}

/// One eq-weighted degree-2 round for the λ-combined pair of product gates
/// `V^L(·,0)·V^L(·,1) ⊕ λ·V^R(·,0)·V^R(·,1)`, Convention A. `l0,l1` and `r0,r1`
/// are the (partially folded) half-slices of the two circuits.
fn batched_round_message(
    l0: &[F128],
    l1: &[F128],
    r0: &[F128],
    r1: &[F128],
    lambda: F128,
    eq: &SplitEqGhash,
) -> (F128, F128) {
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len();
    let n_blocks = hi.len();
    debug_assert_eq!(block * n_blocks, l0.len() / 2);

    let block_fn = |x_hi: usize| -> (F128, F128) {
        let x_base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let xp = x_base + x_lo;
            let (i0, i1) = (2 * xp, 2 * xp + 1);
            let v_one = l0[i1] * l1[i1] + lambda * (r0[i1] * r1[i1]);
            let v_inf = (l0[i0] + l0[i1]) * (l1[i0] + l1[i1])
                + lambda * ((r0[i0] + r0[i1]) * (r1[i0] + r1[i1]));
            let el = lo[x_lo];
            s1 += el * v_one;
            s_inf += el * v_inf;
        }
        let eh = hi[x_hi];
        (eh * s1, eh * s_inf)
    };

    match crate::sumcheck_round_min_len(block * n_blocks, n_blocks) {
        Some(min_len) => (0..n_blocks)
            .into_par_iter()
            .with_min_len(min_len)
            .map(block_fn)
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(o0, i0), (o1, i1)| (o0 + o1, i0 + i1),
            ),
        None => {
            let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
            for x_hi in 0..n_blocks {
                let (o, i) = block_fn(x_hi);
                g_one += o;
                g_inf += i;
            }
            (g_one, g_inf)
        }
    }
}

/// Batched prover: proves `f, g` related by `σ` with both product circuits run
/// in lockstep, reducing to a SINGLE point `ρ`. The caller must have absorbed
/// `f, g, σ` into `ch`.
pub fn prove_batched<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    ch: &mut C,
) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
    let n = f.len();
    assert_eq!(g.len(), n);
    assert_eq!(sigma.len(), n);
    assert!(n.is_power_of_two() && n >= 2, "need N = 2^μ ≥ 2");
    let mu = n.trailing_zeros() as usize;

    let mut t = Instant::now();
    ch.observe_label(DOMAIN_BATCHED);
    let alpha = ch.sample_f128();
    let beta = ch.sample_f128();

    let basis = s_id_basis(mu);
    // `s_id(x)` is the field element whose bit pattern *is* `x`, so it needs no
    // table: `n = 2^μ` must fit a `usize`, hence μ ≤ 64 and every tag is just
    // the index widened. That drops the `O(N)` `s_id` build outright, and turns
    // `s_σ` from a random-access gather through that table into a linear map.
    // (`tag_matches_basis_expansion` pins this against `s_id_value`.)
    //
    // A hard assert, not a debug one: past μ = 64 the `tag` below silently
    // stops being `s_id`, so a release build would produce a proof of the wrong
    // statement rather than fail. (Unreachable in practice — `2^μ` field
    // elements would not fit memory — which is exactly why it must not be the
    // check that is compiled out.)
    assert!(mu <= 64, "s_id-as-index needs μ ≤ 64");
    let tag = |i: usize| F128::new(i as u64, 0);
    let s_sig_vec: Vec<F128> = sigma
        .par_iter()
        .with_min_len(par_threshold())
        .map(|&sx| tag(sx))
        .collect();
    tp(&mut t, "  s_sigma");
    let lhs: Vec<F128> = f
        .par_iter()
        .enumerate()
        .with_min_len(par_threshold())
        .map(|(x, fx)| *fx + alpha * tag(x) + beta)
        .collect();
    let rhs: Vec<F128> = g
        .par_iter()
        .zip(&s_sig_vec)
        .with_min_len(par_threshold())
        .map(|(gx, sx)| *gx + alpha * *sx + beta)
        .collect();
    tp(&mut t, "  lhs,rhs");

    // Build both circuits' layers (index k has 2^k entries; k = mu is input).
    let mut l_layers: Vec<Vec<F128>> = vec![Vec::new(); mu + 1];
    let mut r_layers: Vec<Vec<F128>> = vec![Vec::new(); mu + 1];
    l_layers[mu] = lhs;
    r_layers[mu] = rhs;
    for k in (0..mu).rev() {
        l_layers[k] = build_layer(&l_layers[k + 1]);
        r_layers[k] = build_layer(&r_layers[k + 1]);
    }
    let top_lhs = l_layers[0][0];
    let top_rhs = r_layers[0][0];
    ch.observe_f128(top_lhs);
    ch.observe_f128(top_rhs);
    tp(&mut t, "  build-layers");

    let mut r_pt: Vec<F128> = Vec::new();
    let mut layers = Vec::with_capacity(mu);
    // Mirror the verifier's per-layer collapse so the final values are
    // `lhs(ρ)` and `rhs(ρ)` — see the eval reconstruction after the loop.
    let (mut claim_l, mut claim_r) = (F128::ZERO, F128::ZERO);
    // GKR_TRACE accumulators: eq build vs round message vs fold, summed over
    // every layer's every round.
    let (mut eq_ns, mut msg_ns, mut fold_ns) = (0u128, 0u128, 0u128);
    // Eight working buffers, hoisted for the whole prove and ping-ponged
    // (`cur` folds into `nxt`, then they swap). The widest fold output is the
    // top layer's round 0, `2^(μ-2)`, so that capacity serves every layer and
    // round. Pages are faulted at most once per prove — and across proves the
    // scratch pool hands the same resident buffers straight back.
    let cap = 1usize << mu.saturating_sub(2);
    let mut cur: [Vec<F128>; 4] = from_fn(|_| take_f128(cap));
    let mut nxt: [Vec<F128>; 4] = from_fn(|_| take_f128(cap));
    for k in 0..mu {
        let lambda = ch.sample_f128();
        let h = 1usize << k;
        // Live prefix length of each `cur` buffer; set at round 0, halved after.
        let mut len = 0usize;
        let mut rounds = Vec::with_capacity(k);
        let mut r_prime = Vec::with_capacity(k + 1);
        // Round 0's message is the one read that cannot be fused: nothing has
        // been folded yet, so it comes straight off the layer. Every later
        // round's message is produced by the preceding fold.
        let mut pending = if k > 0 {
            let tr = Instant::now();
            let (l0s, l1s) = l_layers[k + 1].split_at(h);
            let (r0s, r1s) = r_layers[k + 1].split_at(h);
            let eq = SplitEqGhash::new(&r_pt[1..k]);
            let m = batched_round_message(l0s, l1s, r0s, r1s, lambda, &eq);
            if trace_on() {
                msg_ns += tr.elapsed().as_nanos();
            }
            Some(m)
        } else {
            None
        };

        for i in 0..k {
            let (g1, g_inf) = pending.expect("round i's message was produced already");
            ch.observe_f128(g1);
            ch.observe_f128(g_inf);
            let rho = ch.sample_f128();
            rounds.push((g1, g_inf));
            r_prime.push(rho);

            let mut tr = Instant::now();
            // eq for round i+1; `None` on the last round, where the fold has no
            // successor message to emit.
            let eq_next = (i + 1 < k).then(|| SplitEqGhash::new(&r_pt[i + 2..k]));
            if trace_on() {
                eq_ns += tr.elapsed().as_nanos();
                tr = Instant::now();
            }
            if i == 0 {
                let (l0s, l1s) = l_layers[k + 1].split_at(h);
                let (r0s, r1s) = r_layers[k + 1].split_at(h);
                let [d0, d1, d2, d3] = &mut cur;
                pending = fold_and_message(
                    [l0s, l1s, r0s, r1s],
                    rho,
                    [d0, d1, d2, d3].map(|d| d.as_mut_slice()),
                    lambda,
                    eq_next.as_ref(),
                );
                len = h / 2;
            } else {
                let src = [
                    &cur[0][..len],
                    &cur[1][..len],
                    &cur[2][..len],
                    &cur[3][..len],
                ];
                let [d0, d1, d2, d3] = &mut nxt;
                pending = fold_and_message(
                    src,
                    rho,
                    [d0, d1, d2, d3].map(|d| d.as_mut_slice()),
                    lambda,
                    eq_next.as_ref(),
                );
                len /= 2;
                swap(&mut cur, &mut nxt);
            }
            if trace_on() {
                fold_ns += tr.elapsed().as_nanos();
            }
        }
        let (vl0, vl1, vr0, vr1) = if k == 0 {
            (
                l_layers[1][0],
                l_layers[1][1],
                r_layers[1][0],
                r_layers[1][1],
            )
        } else {
            debug_assert_eq!(len, 1, "layer {k}: folds must reduce to one element");
            (cur[0][0], cur[1][0], cur[2][0], cur[3][0])
        };
        for v in [vl0, vl1, vr0, vr1] {
            ch.observe_f128(v);
        }
        layers.push(BatchedLayerProof {
            rounds,
            vl0,
            vl1,
            vr0,
            vr1,
        });
        let c_k = ch.sample_f128();
        let one_plus_c = F128::ONE + c_k;
        claim_l = one_plus_c * vl0 + c_k * vl1;
        claim_r = one_plus_c * vr0 + c_k * vr1;
        r_prime.push(c_k);
        r_pt = r_prime;
    }
    tp(&mut t, "layer-sumchecks");
    if trace_on() {
        eprintln!(
            "  [prod-gkr]   ├ eq-build       {:8.3} ms\n  \
             [prod-gkr]   ├ round-messages {:8.3} ms\n  \
             [prod-gkr]   └ folds          {:8.3} ms",
            eq_ns as f64 / 1e6,
            msg_ns as f64 / 1e6,
            fold_ns as f64 / 1e6,
        );
    }

    let rho = r_pt;
    // After the last layer, `claim_l = lhs(ρ)` and `claim_r = rhs(ρ)` — the same
    // collapse the verifier performs. The witness evals follow in closed form:
    // `lhs = f + α·s_id + β` pointwise, MLE is linear in the table, and `s_id`'s
    // MLE is closed-form, so `f(ρ) = lhs(ρ) + α·s_id(ρ) + β` (char 2:
    // subtraction is addition). Same for `g` via `s_σ`. Only `s_σ` — a permuted
    // table with no closed form — still needs an `O(N)` MLE evaluation, so this
    // is one such pass instead of three.
    let s_sigma_eval = mle_eval(&s_sig_vec, &rho);
    let f_eval = claim_l + alpha * s_id_eval(&basis, &rho) + beta;
    let g_eval = claim_r + alpha * s_sigma_eval + beta;
    observe_evals(ch, &[f_eval, g_eval, s_sigma_eval]);
    // Hand the ping-pong buffers back so the next prove reuses resident pages.
    for u in cur.into_iter().chain(nxt) {
        give_f128(u);
    }
    tp(&mut t, "evals");

    let proof = ProductGkrBatchedProof {
        top_lhs,
        top_rhs,
        layers,
        f_eval,
        g_eval,
        s_sigma_eval,
    };
    let claim = ProductGkrBatchedClaim {
        rho,
        f_eval,
        g_eval,
        s_sigma_eval,
    };
    (proof, claim)
}

/// Verify a batched product-GKR proof for `N = 2^mu`, **trusting
/// `proof.s_sigma_eval`** (sound only if `s_σ` is pinned downstream). Returns
/// the shared claim point `ρ` and the witness evals. The caller must have
/// absorbed the same `f, g, σ` binding into `ch`.
pub fn verify_batched<C: Challenger>(
    mu: usize,
    proof: &ProductGkrBatchedProof,
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, VerifyError> {
    verify_batched_core(mu, proof, None, ch)
}

/// Verify a batched product-GKR proof where **σ is verifier-known**: the
/// verifier computes `s_σ(ρ)` itself from `sigma` and uses it in the final
/// relation instead of trusting `proof.s_sigma_eval` (the recursion / hookup
/// setting). `sigma.len()` must be `2^mu`.
/// The sigma table as the F128 vector the verifier's `s_sigma(rho)` is the
/// MLE of: `s_sig[x] = s_id_vec[sigma[x]]`. `pub` for sigma v2 route B
/// (circuit-wiring-design.tex §sigma): the accumulator's sigma claims are
/// MatrixClaims on this vector reshaped `2^nu × 2^c`, and the root
/// discharge evaluates it once — sourced from here so the encoding cannot
/// drift from the verifier's.
pub fn build_s_sigma_vec(mu: usize, sigma: &[usize]) -> Vec<F128> {
    assert_eq!(sigma.len(), 1usize << mu, "σ length must be 2^mu");
    let basis = s_id_basis(mu);
    let s_id_vec = build_s_id_vec(mu, &basis);
    sigma.iter().map(|&sx| s_id_vec[sx]).collect()
}

pub fn verify_batched_with_sigma<C: Challenger>(
    mu: usize,
    proof: &ProductGkrBatchedProof,
    sigma: &[usize],
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, VerifyError> {
    assert_eq!(sigma.len(), 1usize << mu, "σ length must be 2^mu");
    verify_batched_core(mu, proof, Some(sigma), ch)
}

fn verify_batched_core<C: Challenger>(
    mu: usize,
    proof: &ProductGkrBatchedProof,
    sigma_opt: Option<&[usize]>,
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, VerifyError> {
    if proof.layers.len() != mu {
        return Err(VerifyError::MalformedProof);
    }
    ch.observe_label(DOMAIN_BATCHED);
    let alpha = ch.sample_f128();
    let beta = ch.sample_f128();

    ch.observe_f128(proof.top_lhs);
    ch.observe_f128(proof.top_rhs);
    if proof.top_lhs != proof.top_rhs {
        return Err(VerifyError::ProductMismatch);
    }

    let mut claim_l = proof.top_lhs;
    let mut claim_r = proof.top_rhs;
    let mut r_pt: Vec<F128> = Vec::new();
    for (k, layer) in proof.layers.iter().enumerate() {
        if layer.rounds.len() != k {
            return Err(VerifyError::MalformedProof);
        }
        let lambda = ch.sample_f128();
        let mut c_run = claim_l + lambda * claim_r;
        let mut r_prime = Vec::with_capacity(k + 1);
        for i in 0..k {
            let (g1, g_inf) = layer.rounds[i];
            let r_eq = r_pt[i];
            let one_plus_r_eq = F128::ONE + r_eq;
            let g0 = (c_run + r_eq * g1) * one_plus_r_eq.inv();
            ch.observe_f128(g1);
            ch.observe_f128(g_inf);
            let rho = ch.sample_f128();
            r_prime.push(rho);
            let one_plus_rho = F128::ONE + rho;
            c_run = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
        }
        let (vl0, vl1, vr0, vr1) = (layer.vl0, layer.vl1, layer.vr0, layer.vr1);
        for v in [vl0, vl1, vr0, vr1] {
            ch.observe_f128(v);
        }
        let gate = vl0 * vl1 + lambda * (vr0 * vr1);
        if c_run != gate {
            return Err(VerifyError::LayerCheckFailed);
        }
        let c_k = ch.sample_f128();
        let one_plus_c = F128::ONE + c_k;
        claim_l = one_plus_c * vl0 + c_k * vl1;
        claim_r = one_plus_c * vr0 + c_k * vr1;
        r_prime.push(c_k);
        r_pt = r_prime;
    }

    // Input-layer checks at the shared ρ: both reconstructed affinely, sharing
    // the single witness eval (f_eval = g_eval = w(ρ) when f = g = w).
    let basis = s_id_basis(mu);
    let s_id_rho = s_id_eval(&basis, &r_pt);
    // s_σ(ρ): verifier-computed when σ is known (not trusting the proof), else
    // the proof's claimed value.
    let s_sigma = match sigma_opt {
        Some(sigma) => {
            let s_id_vec = build_s_id_vec(mu, &basis);
            let s_sig: Vec<F128> = sigma.iter().map(|&sx| s_id_vec[sx]).collect();
            mle_eval(&s_sig, &r_pt)
        }
        None => proof.s_sigma_eval,
    };
    let lhs_in = proof.f_eval + alpha * s_id_rho + beta;
    let rhs_in = proof.g_eval + alpha * s_sigma + beta;
    if claim_l != lhs_in || claim_r != rhs_in {
        return Err(VerifyError::InputMismatch);
    }

    observe_evals(ch, &[proof.f_eval, proof.g_eval, s_sigma]);

    Ok(ProductGkrBatchedClaim {
        rho: r_pt,
        f_eval: proof.f_eval,
        g_eval: proof.g_eval,
        s_sigma_eval: s_sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128::new(self.next_u64(), self.next_u64())
        }
        fn permutation(&mut self, n: usize) -> Vec<usize> {
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (self.next_u64() % (i as u64 + 1)) as usize;
                p.swap(i, j);
            }
            p
        }
    }

    fn invert(sigma: &[usize]) -> Vec<usize> {
        let mut inv = vec![0usize; sigma.len()];
        for (x, &sx) in sigma.iter().enumerate() {
            inv[sx] = x;
        }
        inv
    }

    /// Honest instance: random `g`, permutation `σ`, `f(x) = g(σ⁻¹(x))` so the
    /// multiset `{(f, s_id)} = {(g, s_σ)}` holds and the products match.
    fn honest_instance(mu: usize, seed: u64) -> (Vec<F128>, Vec<F128>, Vec<usize>) {
        let n = 1usize << mu;
        let mut rng = Rng::new(seed);
        let g: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let sigma = rng.permutation(n);
        let sinv = invert(&sigma);
        let f: Vec<F128> = (0..n).map(|x| g[sinv[x]]).collect();
        (f, g, sigma)
    }

    fn bind<C: Challenger>(ch: &mut C, f: &[F128], g: &[F128], sigma: &[usize]) {
        ch.observe_f128_slice(f);
        ch.observe_f128_slice(g);
        for &s in sigma {
            ch.observe_f128(F128::new(s as u64, 0));
        }
    }

    fn run_prove(f: &[F128], g: &[F128], sigma: &[usize]) -> (ProductGkrProof, ProductGkrClaim) {
        let mut ch = FsChallenger::new(b"product-gkr-test");
        bind(&mut ch, f, g, sigma);
        prove(f, g, sigma, &mut ch)
    }

    fn run_verify(
        mu: usize,
        f: &[F128],
        g: &[F128],
        sigma: &[usize],
        proof: &ProductGkrProof,
    ) -> Result<ProductGkrClaim, VerifyError> {
        let mut ch = FsChallenger::new(b"product-gkr-test");
        bind(&mut ch, f, g, sigma);
        verify(mu, proof, &mut ch)
    }

    #[test]
    fn honest_roundtrip_and_claim_match() {
        for mu in 1..=10 {
            let (f, g, sigma) = honest_instance(mu, 0xC0FFEE ^ mu as u64);
            let (proof, claim_p) = run_prove(&f, &g, &sigma);
            assert_eq!(proof.top_lhs, proof.top_rhs, "μ={mu}: ∏lhs ≠ ∏rhs");
            let claim_v = run_verify(mu, &f, &g, &sigma, &proof).expect("verify");
            assert_eq!(claim_p, claim_v, "μ={mu}: prover/verifier claim mismatch");
        }
    }

    #[test]
    fn claim_matches_direct_mle() {
        let mu = 8;
        let (f, g, sigma) = honest_instance(mu, 0xABCD);
        let (_proof, claim) = run_prove(&f, &g, &sigma);
        // f at ρ_lhs, g and s_σ at ρ_rhs match direct MLE evals.
        let basis = s_id_basis(mu);
        let s_sig: Vec<F128> = (0..f.len()).map(|x| s_id_value(sigma[x], &basis)).collect();
        assert_eq!(claim.f_eval, mle_eval(&f, &claim.rho_lhs));
        assert_eq!(claim.g_eval, mle_eval(&g, &claim.rho_rhs));
        assert_eq!(claim.s_sigma_eval, mle_eval(&s_sig, &claim.rho_rhs));
    }

    /// Isolated scaling probe for [`fold_into`] — the phase that dominates
    /// `prove_batched` and shows almost no gain from the thread pool in the
    /// end-to-end trace. Run under both thread counts to attribute that:
    ///
    /// ```text
    /// cargo test --release -p flock-core --lib fold_scaling_probe -- --ignored --nocapture
    /// RAYON_NUM_THREADS=1 cargo test --release ... (same)
    /// ```
    #[test]
    #[ignore = "timing probe, not a correctness test"]
    fn fold_scaling_probe() {
        let mut rng = Rng::new(0xF01D);
        let threads = rayon::current_num_threads();
        eprintln!("threads = {threads}");
        // `fold_into` is the plain (last-round) fold; `fold_and_message` is the
        // fused kernel that carries almost all of `prove_batched`'s fold time.
        // Both are reported so the fusion's cost per output is visible.
        eprintln!("  width        plain-fold          fused fold+msg      fused/plain");
        for &log_n in &[14usize, 16, 18, 20] {
            let n = 1usize << log_n;
            let half = n / 2;
            let src: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let rho = rng.f128();
            let lambda = rng.f128();
            // The fused kernel folds four vectors and emits the next round's
            // message, whose eq spans the folded width's pairs.
            let eq_pt: Vec<F128> = (0..log_n.saturating_sub(2)).map(|_| rng.f128()).collect();
            let eq = SplitEqGhash::new(&eq_pt);
            let mut dst: [Vec<F128>; 4] = from_fn(|_| vec![F128::ZERO; half]);

            let time = |iters: usize, mut run: Box<dyn FnMut()>| -> f64 {
                run();
                let t0 = Instant::now();
                for _ in 0..iters {
                    run();
                }
                t0.elapsed().as_secs_f64() * 1e3 / iters as f64
            };

            let plain = {
                let mut d = vec![F128::ZERO; half];
                let s = &src;
                time(20, Box::new(move || fold_into(black_box(s), rho, &mut d)))
            };
            let fused = {
                let s = &src;
                let d = &mut dst;
                let eqr = &eq;
                time(
                    20,
                    Box::new(move || {
                        let [d0, d1, d2, d3] = d;
                        fold_and_message(
                            [s, s, s, s],
                            rho,
                            [d0, d1, d2, d3].map(|x| x.as_mut_slice()),
                            lambda,
                            Some(eqr),
                        );
                    }),
                )
            };
            // Plain folds one vector; fused folds four and emits a message.
            eprintln!(
                "  2^{log_n}->2^{:<3}  {plain:7.3} ms {:5.2} ns/out   {fused:7.3} ms {:5.2} ns/out   {:5.2}x",
                log_n - 1,
                plain * 1e6 / half as f64,
                fused * 1e6 / (4 * half) as f64,
                fused / (4.0 * plain),
            );
        }
    }

    /// `prove_batched` builds its `s_id` tags by widening the index instead of
    /// expanding the basis into an `O(N)` table. Pin the two against each other,
    /// including the `build_s_id_vec` table `prove` still uses.
    #[test]
    fn tag_matches_basis_expansion() {
        for mu in 1..=12 {
            let basis = s_id_basis(mu);
            let table = build_s_id_vec(mu, &basis);
            for x in 0..(1usize << mu) {
                let tag = F128::new(x as u64, 0);
                assert_eq!(tag, s_id_value(x, &basis), "μ={mu}, x={x}: basis");
                assert_eq!(tag, table[x], "μ={mu}, x={x}: table");
            }
        }
    }

    /// `prove_batched` reconstructs `f_eval`/`g_eval` in closed form from the
    /// final collapsed layer claims instead of evaluating them directly. The
    /// roundtrip test cannot catch an error there — the verifier rebuilds
    /// `lhs(ρ)` *from* `f_eval`, so a wrong `f_eval` still satisfies its own
    /// check. Pin all three evals against honest `O(N)` MLE evaluations at the
    /// shared point, across sizes.
    #[test]
    fn batched_claim_matches_direct_mle() {
        for mu in 1..=10 {
            let (f, g, sigma) = honest_instance(mu, 0x5EED ^ mu as u64);
            let mut ch = FsChallenger::new(b"prod-gkr-batched-mle-test");
            bind(&mut ch, &f, &g, &sigma);
            let (_proof, claim) = prove_batched(&f, &g, &sigma, &mut ch);
            let basis = s_id_basis(mu);
            let s_sig: Vec<F128> = (0..f.len()).map(|x| s_id_value(sigma[x], &basis)).collect();
            assert_eq!(claim.f_eval, mle_eval(&f, &claim.rho), "μ={mu}: f");
            assert_eq!(claim.g_eval, mle_eval(&g, &claim.rho), "μ={mu}: g");
            assert_eq!(
                claim.s_sigma_eval,
                mle_eval(&s_sig, &claim.rho),
                "μ={mu}: s_σ"
            );
        }
    }

    #[test]
    fn non_permutation_relation_rejected() {
        // σ = identity but f ≠ g ⇒ the two products differ ⇒ reject.
        let mu = 6;
        let n = 1usize << mu;
        let mut rng = Rng::new(0x1234);
        let f: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let mut g = f.clone();
        g[3] += F128::ONE; // break the multiset equality
        let sigma: Vec<usize> = (0..n).collect();
        let (proof, _) = run_prove(&f, &g, &sigma);
        // The grand products no longer match.
        assert_ne!(proof.top_lhs, proof.top_rhs);
        let res = run_verify(mu, &f, &g, &sigma, &proof);
        assert_eq!(res, Err(VerifyError::ProductMismatch));
    }

    #[test]
    fn mis_permuted_witness_rejected() {
        // A valid permutation but a corrupted witness (not constant on a cycle)
        // ⇒ products differ ⇒ reject.
        let mu = 7;
        let (mut f, g, sigma) = honest_instance(mu, 0x5151);
        f[5] += F128::ONE;
        let (proof, _) = run_prove(&f, &g, &sigma);
        let res = run_verify(mu, &f, &g, &sigma, &proof);
        assert!(res.is_err());
    }

    /// A witness constant on every σ-cycle (so the two grand products match).
    fn cycle_constant_witness(sigma: &[usize], seed: u64) -> Vec<F128> {
        let n = sigma.len();
        let mut rng = Rng::new(seed);
        let mut w = vec![F128::ZERO; n];
        let mut seen = vec![false; n];
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let val = rng.f128();
            let mut i = start;
            loop {
                w[i] = val;
                seen[i] = true;
                i = sigma[i];
                if i == start {
                    break;
                }
            }
        }
        w
    }

    #[test]
    fn batched_honest_roundtrip_shared_point() {
        for mu in 1..=10 {
            let (f, g, sigma) = honest_instance(mu, 0xBA7C ^ mu as u64);
            let mut chp = FsChallenger::new(b"prod-gkr-batched-test");
            bind(&mut chp, &f, &g, &sigma);
            let (proof, claim_p) = prove_batched(&f, &g, &sigma, &mut chp);
            assert_eq!(proof.top_lhs, proof.top_rhs, "μ={mu}: ∏lhs ≠ ∏rhs");
            let mut chv = FsChallenger::new(b"prod-gkr-batched-test");
            bind(&mut chv, &f, &g, &sigma);
            let claim_v = verify_batched(mu, &proof, &mut chv).expect("verify");
            assert_eq!(claim_p, claim_v, "μ={mu}");
            assert_eq!(claim_v.rho.len(), mu, "single shared reduction point");
        }
    }

    #[test]
    fn batched_copy_constraint_single_witness() {
        // f = g = w constant on σ-cycles ⇒ the two evals coincide at the shared
        // point, so ONE witness opening suffices (the hookup's case).
        let mu = 8;
        let n = 1usize << mu;
        let mut rng = Rng::new(0xC0C0);
        let sigma = rng.permutation(n);
        let w = cycle_constant_witness(&sigma, 0xD00D);
        let mut chp = FsChallenger::new(b"prod-gkr-batched-test");
        bind(&mut chp, &w, &w, &sigma);
        let (proof, claim) = prove_batched(&w, &w, &sigma, &mut chp);
        assert_eq!(proof.top_lhs, proof.top_rhs);
        assert_eq!(claim.f_eval, claim.g_eval, "f=g=w ⇒ one witness eval at ρ");
        let mut chv = FsChallenger::new(b"prod-gkr-batched-test");
        bind(&mut chv, &w, &w, &sigma);
        verify_batched(mu, &proof, &mut chv).expect("verify");
    }

    #[test]
    fn batched_rejects_broken_copy_constraint() {
        let mu = 7;
        let n = 1usize << mu;
        let mut rng = Rng::new(0x9999);
        let sigma = rng.permutation(n);
        let mut w = cycle_constant_witness(&sigma, 0x4242);
        w[3] += F128::ONE; // break constancy on a cycle
        let mut chp = FsChallenger::new(b"prod-gkr-batched-test");
        bind(&mut chp, &w, &w, &sigma);
        let (proof, _) = prove_batched(&w, &w, &sigma, &mut chp);
        let mut chv = FsChallenger::new(b"prod-gkr-batched-test");
        bind(&mut chv, &w, &w, &sigma);
        assert!(verify_batched(mu, &proof, &mut chv).is_err());
    }

    /// A proof of the wrong shape must be rejected, not panicked on — matching
    /// `zerocheck::verify_rejects_shape_errors`. These used to be `assert_eq!`,
    /// so a malformed proof from an untrusted source took the verifier down.
    #[test]
    fn verify_rejects_shape_errors() {
        let mu = 6;
        let (f, g, sigma) = honest_instance(mu, 0x5111);

        let mut chp = FsChallenger::new(b"prod-gkr-shape");
        bind(&mut chp, &f, &g, &sigma);
        let (batched, _) = prove_batched(&f, &g, &sigma, &mut chp);

        let verify_shape = |p: &ProductGkrBatchedProof, mu: usize| {
            let mut ch = FsChallenger::new(b"prod-gkr-shape");
            bind(&mut ch, &f, &g, &sigma);
            verify_batched(mu, p, &mut ch)
        };
        // Wrong layer count (both directions).
        let mut short = batched.clone();
        short.layers.pop();
        assert_eq!(verify_shape(&short, mu), Err(VerifyError::MalformedProof));
        assert_eq!(
            verify_shape(&batched, mu - 1),
            Err(VerifyError::MalformedProof)
        );
        // Right layer count, wrong round count inside a layer.
        let mut bad_rounds = batched.clone();
        bad_rounds.layers[mu - 1].rounds.pop();
        assert_eq!(
            verify_shape(&bad_rounds, mu),
            Err(VerifyError::MalformedProof)
        );

        // Same for the unbatched verifier.
        let mut chp = FsChallenger::new(b"prod-gkr-shape");
        bind(&mut chp, &f, &g, &sigma);
        let (plain, _) = prove(&f, &g, &sigma, &mut chp);
        let mut trimmed = plain.clone();
        trimmed.layers_lhs.pop();
        let mut ch = FsChallenger::new(b"prod-gkr-shape");
        bind(&mut ch, &f, &g, &sigma);
        assert_eq!(
            verify(mu, &trimmed, &mut ch),
            Err(VerifyError::MalformedProof)
        );
    }

    /// **Transcript tamper matrix.** Every field of a batched proof is bound:
    /// flipping any one of them must be REJECTED (never accepted, never
    /// panicked on) by the σ-aware verifier — the only one callers may use.
    ///
    /// The single exception is `s_sigma_eval`, which that verifier recomputes
    /// from its own σ and therefore ignores; the assertion below pins that
    /// meaning (and that the returned claim carries the RECOMPUTED value, not
    /// the proof's), which is the whole difference from the trusting variant.
    #[test]
    fn batched_rejects_transcript_tampering() {
        const DOMAIN_TEST: &[u8] = b"prod-gkr-tamper";
        let mu = 5;
        let (f, g, sigma) = honest_instance(mu, 0x7A47);
        let mut chp = FsChallenger::new(DOMAIN_TEST);
        bind(&mut chp, &f, &g, &sigma);
        let (proof, claim) = prove_batched(&f, &g, &sigma, &mut chp);

        let check = |p: &ProductGkrBatchedProof| {
            let mut ch = FsChallenger::new(DOMAIN_TEST);
            bind(&mut ch, &f, &g, &sigma);
            verify_batched_with_sigma(mu, p, &sigma, &mut ch)
        };
        assert!(check(&proof).is_ok(), "the honest proof must verify");

        // Every scalar of the transcript, one at a time.
        let mut mutations: Vec<(String, ProductGkrBatchedProof)> = Vec::new();
        let mut push = |name: String, p: ProductGkrBatchedProof| mutations.push((name, p));
        for (name, sel) in [
            ("top_lhs", 0usize),
            ("top_rhs", 1),
            ("both tops", 2),
            ("f_eval", 3),
            ("g_eval", 4),
        ] {
            let mut p = proof.clone();
            match sel {
                0 => p.top_lhs += F128::ONE,
                1 => p.top_rhs += F128::ONE,
                2 => {
                    p.top_lhs += F128::ONE;
                    p.top_rhs += F128::ONE;
                }
                3 => p.f_eval += F128::ONE,
                _ => p.g_eval += F128::ONE,
            }
            push(name.to_string(), p);
        }
        for k in 0..mu {
            for (name, sel) in [("vl0", 0usize), ("vl1", 1), ("vr0", 2), ("vr1", 3)] {
                let mut p = proof.clone();
                let layer = &mut p.layers[k];
                match sel {
                    0 => layer.vl0 += F128::ONE,
                    1 => layer.vl1 += F128::ONE,
                    2 => layer.vr0 += F128::ONE,
                    _ => layer.vr1 += F128::ONE,
                }
                push(format!("layer {k} {name}"), p);
            }
            for i in 0..proof.layers[k].rounds.len() {
                for half in 0..2 {
                    let mut p = proof.clone();
                    let r = &mut p.layers[k].rounds[i];
                    if half == 0 {
                        r.0 += F128::ONE;
                    } else {
                        r.1 += F128::ONE;
                    }
                    push(format!("layer {k} round {i}.{half}"), p);
                }
            }
        }
        for (name, p) in &mutations {
            assert!(check(p).is_err(), "tampered {name} verified");
        }

        // `s_sigma_eval` is verifier-recomputed under a known σ, so the field is
        // inert there — and the claim reports the honest value.
        let mut inert = proof.clone();
        inert.s_sigma_eval += F128::ONE;
        let out = check(&inert).expect("σ-aware verify ignores the proof's s_σ(ρ)");
        assert_eq!(out.s_sigma_eval, claim.s_sigma_eval);
    }

    /// `verify_batched` reads `s_σ(ρ)` out of the proof, so its input check
    /// `claim_r = g(ρ) + α·s_σ(ρ) + β` has **two** prover-supplied unknowns and
    /// is satisfied by a one-parameter family: shift `g_eval` by δ and
    /// `s_sigma_eval` by `δ/α` and it still holds (char 2, so the two δ's
    /// cancel). Nothing else in the transcript pins either value — the evals are
    /// observed only after every challenge is drawn.
    ///
    /// That is the documented precondition ("sound only if `s_σ` is pinned
    /// downstream"), pinned here as behaviour so it cannot regress silently, and
    /// paired with the demonstration that `verify_batched_with_sigma` — which
    /// recomputes `s_σ(ρ)` from a verifier-known σ — rejects the same forgery.
    #[test]
    fn batched_verify_trusts_s_sigma_unless_sigma_is_known() {
        const DOMAIN_TEST: &[u8] = b"prod-gkr-forge";
        let mu = 7;
        let (f, g, sigma) = honest_instance(mu, 0xD00D);
        let mut chp = FsChallenger::new(DOMAIN_TEST);
        bind(&mut chp, &f, &g, &sigma);
        let (proof, _) = prove_batched(&f, &g, &sigma, &mut chp);

        // Replay the transcript prefix to recover α, exactly as a prover would.
        let mut cha = FsChallenger::new(DOMAIN_TEST);
        bind(&mut cha, &f, &g, &sigma);
        cha.observe_label(DOMAIN_BATCHED);
        let alpha = cha.sample_f128();

        let delta = F128::new(0xDEAD_BEEF, 0x1234);
        let mut forged = proof.clone();
        forged.g_eval += delta;
        forged.s_sigma_eval += delta * alpha.inv();

        // Trusting verifier: accepts, and hands back the forged evals.
        let mut chv = FsChallenger::new(DOMAIN_TEST);
        bind(&mut chv, &f, &g, &sigma);
        let claim = verify_batched(mu, &forged, &mut chv).expect("trusting verifier accepts");
        assert_eq!(claim.g_eval, proof.g_eval + delta);
        assert_ne!(claim.g_eval, proof.g_eval, "forgery really did change g(ρ)");

        // σ-aware verifier: recomputes s_σ(ρ), so the shift no longer cancels.
        let mut chv = FsChallenger::new(DOMAIN_TEST);
        bind(&mut chv, &f, &g, &sigma);
        assert_eq!(
            verify_batched_with_sigma(mu, &forged, &sigma, &mut chv),
            Err(VerifyError::InputMismatch),
        );

        // ...and still accepts the honest proof.
        let mut chv = FsChallenger::new(DOMAIN_TEST);
        bind(&mut chv, &f, &g, &sigma);
        verify_batched_with_sigma(mu, &proof, &sigma, &mut chv).expect("honest proof verifies");
    }
}
