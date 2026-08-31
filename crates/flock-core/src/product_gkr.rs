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
//! The retired `permutation` PIOP proved one grand product by committing the
//! product tree as a multilinear `v` and opening it. This module proves each grand
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
//! PIOP for the witness side, same contract as the retired `permutation`
//! PIOP: reduces
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

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::univariate_skip::SplitEqGhash;

// ---------------------------------------------------------------------------
// Proof / claim / error types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProductGkrError {
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
    /// A Fiat--Shamir grinding witness was missing, superfluous, or did not
    /// satisfy the configured leading-zero predicate.
    InvalidGrinding,
}

// ---------------------------------------------------------------------------
// Field / polynomial helpers (shared shape with the sibling GKR module)
// ---------------------------------------------------------------------------

/// Basis for the identity tag `s_id`: `basis[i]` is the field element with bit
/// `i` set (requires `μ ≤ 128`).
/// `pub`: the recursion circuit's GKR replay computes `s_id(ρ)` from the same
/// closed form the verifier uses, so the two cannot drift.
pub fn s_id_basis(mu: usize) -> Vec<F128> {
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
/// `pub`: see [`s_id_basis`].
pub fn s_id_eval(basis: &[F128], rho: &[F128]) -> F128 {
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
    static GATE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("FLOCK_GKR_GATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PAR_THRESHOLD_DEFAULT)
    })
}

/// Phase tracing, enabled by `GKR_TRACE=1`. Read once.
fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GKR_TRACE").is_ok())
}

/// Print `label` with the elapsed time since `t`, then reset `t`. No-op unless
/// [`trace_on`].
fn tp(t: &mut std::time::Instant, label: &str) {
    if trace_on() {
        eprintln!(
            "  [prod-gkr] {label:<16} {:8.3} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        *t = std::time::Instant::now();
    }
}

/// Bind the low variable at `ρ`: `u[x] ← u[2x]·(1+ρ) + u[2x+1]·ρ`, halving `u`.
fn fold_in_place(u: &mut Vec<F128>, rho: F128) {
    let half = u.len() / 2;
    let one_minus = F128::ONE + rho;
    match crate::fold_min_len(half) {
        Some(min_len) => {
            let mut out = crate::scratch::take_f128(half);
            out.par_iter_mut()
                .enumerate()
                .with_min_len(min_len)
                .for_each(|(x, o)| {
                    *o = u[2 * x] * one_minus + u[2 * x + 1] * rho;
                });
            let old = std::mem::replace(u, out);
            crate::scratch::give_f128(old);
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
    let mut out = crate::scratch::take_f128(half);
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
// The live mask: identity-padded dead cells (the SP1-style 0/1 padding)
// ---------------------------------------------------------------------------

/// The live-row structure of the batched product's leaf space: `counts[ι]`
/// live rows in slot ι's aligned `2^ν`-row subtree (the cell space's row-low
/// layout: leaf `x` lives in slot `x >> ν`, row `x & (2^ν − 1)`).
///
/// With a mask, a DEAD leaf becomes the multiplicative identity —
/// `leaf = live·(w + α·s + β) + (1 − live)` — so the grand product ranges
/// over the live fingerprint multiset only and the prover may skip dead
/// regions outright (each slot subtree is a live prefix over an all-ones
/// tail). σ must fix every dead cell (checked in debug). Soundness is the
/// same permutation argument restricted to the live set; the mask is
/// statement-derived (the declared counts), so no prover freedom enters.
///
/// The input checks stay CLOSED FORM because `w` is already zero on dead
/// cells and the selector multiplies only structural terms (char 2,
/// `−1 = +1`):
///
///   leaf = w + α·(live ⊙ s_id) + (β + 1)·live + 1
///   v̂(ρ) = ŵ(ρ) + α·M̂(ρ) + (β + 1)·livê(ρ) + 1
///
/// with `livê` ([`Self::live_eval`]) and `M̂` ([`Self::masked_id_eval`])
/// evaluated by O(#slots·ν) / O(#slots·ν²) prefix eq-sums over the aligned
/// live prefixes. On the σ side the deferred table becomes `live ⊙ s_σ`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveMask {
    pub nu: usize,
    /// Live rows per slot, `counts.len() = 2^(μ − ν)`.
    pub counts: Vec<usize>,
}

impl LiveMask {
    pub fn is_live(&self, x: usize) -> bool {
        (x & ((1usize << self.nu) - 1)) < self.counts[x >> self.nu]
    }

    /// `Σ_{i<n} eq(p, i)` over `|p|` LSB-first coordinates — the
    /// count-binding prefix sum, O(|p|). The prefix `[0, n)` decomposes
    /// into one aligned subcube per set bit of `n`; partition of unity
    /// makes each subcube's low part contribute exactly 1.
    pub fn eq_prefix_sum(p: &[F128], n: usize) -> F128 {
        if n >= 1usize << p.len() {
            return F128::ONE;
        }
        let mut acc = F128::ZERO;
        let mut run = F128::ONE;
        for j in (0..p.len()).rev() {
            if (n >> j) & 1 == 1 {
                acc += run * (F128::ONE + p[j]);
                run *= p[j];
            } else {
                run *= F128::ONE + p[j];
            }
        }
        acc
    }

    /// `Σ_{i<n} eq(p, i)·(Σ_j basis_j·i_j)` — the prefix eq-sum weighted by
    /// the identity tag's ROW part. Per subcube `C_t` of the prefix
    /// decomposition: bits above `t` are `n`'s (fixed), bit `t` is 0, bits
    /// below are free (contributing `p_j` each by partition of unity).
    fn eq_prefix_id_sum(p: &[F128], n: usize, basis: &[F128]) -> F128 {
        let nv = p.len();
        if n >= 1usize << nv {
            return basis
                .iter()
                .zip(p)
                .fold(F128::ZERO, |acc, (b, x)| acc + *b * *x);
        }
        let mut acc = F128::ZERO;
        let mut run = F128::ONE;
        for t in (0..nv).rev() {
            if (n >> t) & 1 == 1 {
                let cube = run * (F128::ONE + p[t]);
                let mut idsum = F128::ZERO;
                for (j, b) in basis.iter().enumerate().take(nv) {
                    if j > t {
                        if (n >> j) & 1 == 1 {
                            idsum += *b;
                        }
                    } else if j < t {
                        idsum += *b * p[j];
                    }
                }
                acc += cube * idsum;
                run *= p[t];
            } else {
                run *= F128::ONE + p[t];
            }
        }
        acc
    }

    /// `livê(ρ)` — the MLE of the live indicator at `ρ` (`|ρ| = μ`).
    pub fn live_eval(&self, rho: &[F128]) -> F128 {
        let (lo, hi) = rho.split_at(self.nu);
        let eq_hi = crate::zerocheck::univariate_skip::build_eq(hi);
        self.counts
            .iter()
            .zip(&eq_hi)
            .fold(F128::ZERO, |acc, (&n, &e)| {
                acc + e * Self::eq_prefix_sum(lo, n)
            })
    }

    /// `M̂(ρ)` — the MLE of `live ⊙ s_id` at `ρ`.
    pub fn masked_id_eval(&self, basis: &[F128], rho: &[F128]) -> F128 {
        let (lo, hi) = rho.split_at(self.nu);
        let eq_hi = crate::zerocheck::univariate_skip::build_eq(hi);
        let hi_basis = &basis[self.nu..];
        self.counts
            .iter()
            .enumerate()
            .zip(&eq_hi)
            .map(|((iota, &n), &e)| {
                let mut tag_hi = F128::ZERO;
                for (j, b) in hi_basis.iter().enumerate() {
                    if (iota >> j) & 1 == 1 {
                        tag_hi += *b;
                    }
                }
                e * (tag_hi * Self::eq_prefix_sum(lo, n)
                    + Self::eq_prefix_id_sum(lo, n, &basis[..self.nu]))
            })
            .fold(F128::ZERO, |acc, v| acc + v)
    }
}

// ---------------------------------------------------------------------------
// The grouped (live-prefix) layer pipeline — the live mask's phase 2b.
// ---------------------------------------------------------------------------
//
// Under a mask, every leaf beyond a slot's live prefix is EXACTLY 1, and
// constant-1 regions are invariant under both the product-layer build
// (1·1 = 1) and the sumcheck fold (1·(1+ρ) + 1·ρ = 1). So a layer is stored
// as per-group live prefixes over an implicit all-ones tail, and every pass
// touches live-proportional data: the layer build multiplies prefixes
// (merged length = max of the pair), folds halve prefixes, and round
// messages add a closed-form tail term — all-dead pairs contribute
// `(1 + λ)` to `G(1)` (and 0 to `G(∞)`, since `1 + 1 = 0` in char 2),
// weighted by the eq-mass of the tail range (prefix-sum arrays over the
// split eq tables). This is SP1's `padding_adjustment`, adapted to
// per-group tails. Values are IDENTICAL to the dense pipeline (XOR
// addition is exact), pinned by `grouped_matches_dense_masked`.

/// A layer in grouped form: group `g`'s live prefix is `buf[g·rows ..
/// g·rows + lens[g]]`; entries beyond a prefix are 1 and are never written
/// or read. When `rows == 1`, a group IS one entry and `lens[g] ∈ {0, 1}`
/// encodes ones-vs-real.
struct GVec {
    buf: Vec<F128>,
    lens: Vec<usize>,
    rows: usize,
}

impl GVec {
    #[inline]
    fn val(&self, x: usize) -> F128 {
        let (g, r) = (x / self.rows, x % self.rows);
        if r < self.lens[g] {
            self.buf[x]
        } else {
            F128::ONE
        }
    }

    fn view(&self) -> GView<'_> {
        GView {
            buf: &self.buf[..self.lens.len() * self.rows],
            lens: self.lens.clone(),
            rows: self.rows,
        }
    }

    /// The top-bit split into two halves (the batched sumcheck's `V(·,0)`,
    /// `V(·,1)`): multi-group layers split on the top GROUP bit; a
    /// single-group layer splits its rows.
    fn split(&self) -> (GView<'_>, GView<'_>) {
        let n = self.lens.len() * self.rows;
        let h = n / 2;
        if self.lens.len() >= 2 {
            let gh = self.lens.len() / 2;
            (
                GView {
                    buf: &self.buf[..h],
                    lens: self.lens[..gh].to_vec(),
                    rows: self.rows,
                },
                GView {
                    buf: &self.buf[h..],
                    lens: self.lens[gh..].to_vec(),
                    rows: self.rows,
                },
            )
        } else {
            let len = self.lens[0];
            (
                GView {
                    buf: &self.buf[..h],
                    lens: vec![len.min(h)],
                    rows: h,
                },
                GView {
                    buf: &self.buf[h..],
                    lens: vec![len.saturating_sub(h).min(h)],
                    rows: h,
                },
            )
        }
    }
}

/// A borrowed grouped slice (a split half, or a folded working vector seen
/// through its owner).
struct GView<'a> {
    buf: &'a [F128],
    lens: Vec<usize>,
    rows: usize,
}

impl GView<'_> {
    #[inline]
    fn val(&self, x: usize) -> F128 {
        let (g, r) = (x / self.rows, x % self.rows);
        if r < self.lens[g] {
            self.buf[x]
        } else {
            F128::ONE
        }
    }
}

/// Build the product layer below a grouped layer (high-bit pairing).
/// Slice-based: the branch-free bulk runs to the shorter prefix, the
/// remainder multiplies by implicit 1s (a copy).
fn gv_build_prev(v: &GVec) -> GVec {
    if v.lens.len() >= 2 {
        // Merge group g with group g + G/2, row-wise.
        let gh = v.lens.len() / 2;
        let rows = v.rows;
        let lens: Vec<usize> = (0..gh).map(|g| v.lens[g].max(v.lens[g + gh])).collect();
        let mut buf = crate::scratch::take_f128(gh * rows);
        let work: usize = lens.iter().sum();
        let body = |g: usize, out: &mut [F128]| {
            let (la, lb) = (v.lens[g], v.lens[g + gh]);
            let (lo, hi) = (la.min(lb), la.max(lb));
            let a = &v.buf[g * rows..g * rows + la];
            let b = &v.buf[(g + gh) * rows..(g + gh) * rows + lb];
            for i in 0..lo {
                out[i] = a[i] * b[i];
            }
            let longer = if la >= lb { a } else { b };
            out[lo..hi].copy_from_slice(&longer[lo..hi]);
        };
        if work >= par_threshold() {
            // WITHIN-group chunking too — the group axis collapses to one a
            // few layers up, and the widest merges must not run one-core.
            const CH: usize = 1 << 14;
            buf[..gh * rows]
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(g, out)| {
                    let (la, lb) = (v.lens[g], v.lens[g + gh]);
                    let (lo, hi) = (la.min(lb), la.max(lb));
                    let a = &v.buf[g * rows..g * rows + la];
                    let b = &v.buf[(g + gh) * rows..(g + gh) * rows + lb];
                    out[..lo]
                        .par_chunks_mut(CH)
                        .enumerate()
                        .for_each(|(ci, oc)| {
                            let base = ci * CH;
                            for (i, o) in oc.iter_mut().enumerate() {
                                let x = base + i;
                                *o = a[x] * b[x];
                            }
                        });
                    let longer = if la >= lb { a } else { b };
                    out[lo..hi].copy_from_slice(&longer[lo..hi]);
                });
        } else {
            for (g, out) in buf[..gh * rows].chunks_mut(rows).enumerate() {
                body(g, out);
            }
        }
        GVec { buf, lens, rows }
    } else {
        // Single group: pair row i with row i + h. The product prefix
        // parallelizes above the gate — this arm carries the whole build
        // once the groups have merged.
        let h = v.rows / 2;
        let len = v.lens[0];
        let lp = len.min(h);
        let over = len.saturating_sub(h); // rows with BOTH factors real
        let mut buf = crate::scratch::take_f128(h);
        let (head, src) = (&mut buf[..over], &v.buf);
        if over >= par_threshold() {
            const CH: usize = 1 << 14;
            head.par_chunks_mut(CH).enumerate().for_each(|(ci, oc)| {
                let base = ci * CH;
                for (i, o) in oc.iter_mut().enumerate() {
                    let x = base + i;
                    *o = src[x] * src[x + h];
                }
            });
        } else {
            for (i, o) in head.iter_mut().enumerate() {
                *o = src[i] * src[i + h];
            }
        }
        buf[over..lp].copy_from_slice(&v.buf[over..lp]);
        GVec {
            buf,
            lens: vec![lp],
            rows: h,
        }
    }
}

/// Fold the low variable of a grouped view at `rho`. Row folds halve the
/// per-group prefixes (the odd boundary pair mixes in a 1); once
/// `rows == 1`, folds pair adjacent GROUPS (both-ones pairs stay 1
/// without a write).
/// Fold all FOUR grouped views at `rho` in one dispatch. Fold-only — each
/// (vector, group) task walks its OWN live length, so none of the
/// max-length interleaving that made the fold+message fusion net-negative
/// (see the rejection note in the round loop) applies here. Two wins over
/// four [`gv_fold`] calls: one rayon dispatch per round instead of four,
/// and the parallel gate sees the four vectors' COMBINED work, so layers
/// whose single-vector folds sat just below the gate parallelize.
/// Byte-identical: the per-vector fold bodies are `gv_fold`'s verbatim.
fn gv_fold4(vs: [&GView<'_>; 4], rho: F128) -> [GVec; 4] {
    let rows = vs[0].rows;
    debug_assert!(rows >= 2 && vs.iter().all(|v| v.rows == rows));
    let one_minus = F128::ONE + rho;
    let rows_out = rows / 2;
    let ng = vs[0].lens.len();
    let lens_out: [Vec<usize>; 4] =
        std::array::from_fn(|j| vs[j].lens.iter().map(|l| l.div_ceil(2)).collect());
    let mut bufs: [Vec<F128>; 4] =
        std::array::from_fn(|_| crate::scratch::take_f128(ng * rows_out));
    let body = |j: usize, g: usize, out: &mut [F128]| {
        let len = vs[j].lens[g];
        let full = len / 2;
        let src = &vs[j].buf[g * rows..g * rows + len];
        for i in 0..full {
            out[i] = src[2 * i] * one_minus + src[2 * i + 1] * rho;
        }
        if len % 2 == 1 {
            out[full] = src[len - 1] * one_minus + rho;
        }
    };
    let work: usize = (0..4).map(|j| lens_out[j].iter().sum::<usize>()).sum();
    if work >= par_threshold() {
        const CH: usize = 1 << 14;
        bufs.par_iter_mut().enumerate().for_each(|(j, buf)| {
            buf[..ng * rows_out]
                .par_chunks_mut(rows_out)
                .enumerate()
                .for_each(|(g, out)| {
                    let len = vs[j].lens[g];
                    let full = len / 2;
                    let src = &vs[j].buf[g * rows..g * rows + len];
                    out[..full]
                        .par_chunks_mut(CH)
                        .enumerate()
                        .for_each(|(ci, oc)| {
                            let base = ci * CH;
                            for (i, o) in oc.iter_mut().enumerate() {
                                let x = base + i;
                                *o = src[2 * x] * one_minus + src[2 * x + 1] * rho;
                            }
                        });
                    if len % 2 == 1 {
                        out[full] = src[len - 1] * one_minus + rho;
                    }
                });
        });
    } else {
        for (j, buf) in bufs.iter_mut().enumerate() {
            for (g, out) in buf[..ng * rows_out].chunks_mut(rows_out).enumerate() {
                body(j, g, out);
            }
        }
    }
    let [b0, b1, b2, b3] = bufs;
    let [l0, l1, l2, l3] = lens_out;
    [
        GVec {
            buf: b0,
            lens: l0,
            rows: rows_out,
        },
        GVec {
            buf: b1,
            lens: l1,
            rows: rows_out,
        },
        GVec {
            buf: b2,
            lens: l2,
            rows: rows_out,
        },
        GVec {
            buf: b3,
            lens: l3,
            rows: rows_out,
        },
    ]
}

fn gv_fold(v: &GView<'_>, rho: F128) -> GVec {
    let one_minus = F128::ONE + rho;
    if v.rows >= 2 {
        let rows = v.rows / 2;
        let ng = v.lens.len();
        let lens: Vec<usize> = v.lens.iter().map(|l| l.div_ceil(2)).collect();
        let mut buf = crate::scratch::take_f128(ng * rows);
        let work: usize = lens.iter().sum();
        let body = |g: usize, out: &mut [F128]| {
            let len = v.lens[g];
            let full = len / 2;
            let src = &v.buf[g * v.rows..g * v.rows + len];
            for i in 0..full {
                out[i] = src[2 * i] * one_minus + src[2 * i + 1] * rho;
            }
            if len % 2 == 1 {
                out[full] = src[len - 1] * one_minus + rho;
            }
        };
        if work >= par_threshold() {
            // WITHIN-group chunking too: the group axis collapses to one a
            // few layers up (16 → 1 on the wired-leaf shape), so the widest
            // layers would otherwise fold on a single core.
            const CH: usize = 1 << 14;
            buf[..ng * rows]
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(g, out)| {
                    let len = v.lens[g];
                    let full = len / 2;
                    let src = &v.buf[g * v.rows..g * v.rows + len];
                    out[..full]
                        .par_chunks_mut(CH)
                        .enumerate()
                        .for_each(|(ci, oc)| {
                            let base = ci * CH;
                            for (i, o) in oc.iter_mut().enumerate() {
                                let x = base + i;
                                *o = src[2 * x] * one_minus + src[2 * x + 1] * rho;
                            }
                        });
                    if len % 2 == 1 {
                        out[full] = src[len - 1] * one_minus + rho;
                    }
                });
        } else {
            for (g, out) in buf[..ng * rows].chunks_mut(rows).enumerate() {
                body(g, out);
            }
        }
        GVec { buf, lens, rows }
    } else {
        let gh = v.lens.len() / 2;
        let mut lens = Vec::with_capacity(gh);
        let mut buf = crate::scratch::take_f128(gh);
        for j in 0..gh {
            if v.lens[2 * j] == 0 && v.lens[2 * j + 1] == 0 {
                lens.push(0);
            } else {
                buf[j] = v.val(2 * j) * one_minus + v.val(2 * j + 1) * rho;
                lens.push(1);
            }
        }
        GVec { buf, lens, rows: 1 }
    }
}

/// `Σ_{x ∈ [a, b)} lo[x mod B]·hi[x div B]` in O(1) from prefix-sum arrays
/// (`plo[i] = Σ_{j<i} lo[j]`, likewise `phi`; char 2: suffix = total + prefix).
fn range_eq_sum(lo: &[F128], hi: &[F128], plo: &[F128], phi: &[F128], a: usize, b: usize) -> F128 {
    if a >= b {
        return F128::ZERO;
    }
    let bsz = lo.len();
    let lo_total = plo[bsz];
    let (ba, ra) = (a / bsz, a % bsz);
    let (bb, rb) = (b / bsz, b % bsz);
    if ba == bb {
        return hi[ba] * (plo[rb] + plo[ra]);
    }
    let mut acc = hi[ba] * (lo_total + plo[ra]);
    acc += (phi[bb] + phi[ba + 1]) * lo_total;
    if rb > 0 {
        acc += hi[bb] * plo[rb];
    }
    acc
}

/// The batched round message over four grouped views (Convention A —
/// `G(1)`, `G(∞)` of `l0·l1 + λ·r0·r1` under the eq weights). Live pair
/// prefixes are summed directly; all-ones tail ranges contribute
/// `(1 + λ)·eq-mass` to `G(1)` only.
fn gv_message(
    vs: [&GView<'_>; 4],
    lambda: F128,
    eq: &SplitEqGhash,
    plo: &[F128],
    phi: &[F128],
) -> (F128, F128) {
    let (lo, hi) = (&eq.lo, &eq.hi);
    let rows = vs[0].rows;
    let ng = vs[0].lens.len();
    let one_plus_lambda = F128::ONE + lambda;
    if rows >= 2 {
        let pr = rows / 2;
        let bsz = lo.len();
        debug_assert!(bsz.is_power_of_two());
        let (bmask, bshift) = (bsz - 1, bsz.trailing_zeros() as usize);
        #[inline]
        fn rd(s: &[F128], r: usize) -> F128 {
            if r < s.len() { s[r] } else { F128::ONE }
        }
        let body = |g: usize| -> (F128, F128) {
            let lmax = vs.iter().map(|v| v.lens[g]).max().unwrap();
            let lp = lmax.div_ceil(2).min(pr);
            let s0 = &vs[0].buf[g * rows..g * rows + vs[0].lens[g]];
            let s1 = &vs[1].buf[g * rows..g * rows + vs[1].lens[g]];
            let s2 = &vs[2].buf[g * rows..g * rows + vs[2].lens[g]];
            let s3 = &vs[3].buf[g * rows..g * rows + vs[3].lens[g]];
            let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
            for x in 0..lp {
                let (r0, r1) = (2 * x, 2 * x + 1);
                let (a0, a1) = (rd(s0, r0), rd(s0, r1));
                let (b0, b1) = (rd(s1, r0), rd(s1, r1));
                let (c0, c1) = (rd(s2, r0), rd(s2, r1));
                let (d0, d1) = (rd(s3, r0), rd(s3, r1));
                let v_one = a1 * b1 + lambda * (c1 * d1);
                let v_inf = (a0 + a1) * (b0 + b1) + lambda * ((c0 + c1) * (d0 + d1));
                let flat = g * pr + x;
                let el = lo[flat & bmask] * hi[flat >> bshift];
                g_one += el * v_one;
                g_inf += el * v_inf;
            }
            // The all-ones tail: v_one = 1 + λ, v_inf = 0.
            let mass = range_eq_sum(lo, hi, plo, phi, g * pr + lp, (g + 1) * pr);
            (g_one + one_plus_lambda * mass, g_inf)
        };
        let work: usize = (0..ng)
            .map(|g| vs.iter().map(|v| v.lens[g]).max().unwrap() / 2)
            .sum();
        if work >= par_threshold() {
            // Parallel over (group, sub-chunk): group-only parallelism
            // starves once the group axis collapses (the widest layers are
            // single-group on the wired-leaf shape). Partial sums per chunk
            // reduce exactly (XOR reassociation); the O(1) all-ones tail
            // mass is added once per group.
            const CH: usize = 1 << 14;
            (0..ng)
                .into_par_iter()
                .map(|g| {
                    let lmax = vs.iter().map(|v| v.lens[g]).max().unwrap();
                    let lp = lmax.div_ceil(2).min(pr);
                    let s0 = &vs[0].buf[g * rows..g * rows + vs[0].lens[g]];
                    let s1 = &vs[1].buf[g * rows..g * rows + vs[1].lens[g]];
                    let s2 = &vs[2].buf[g * rows..g * rows + vs[2].lens[g]];
                    let s3 = &vs[3].buf[g * rows..g * rows + vs[3].lens[g]];
                    let (g_one, g_inf) = (0..lp.div_ceil(CH))
                        .into_par_iter()
                        .map(|ci| {
                            let (a, b) = (ci * CH, ((ci + 1) * CH).min(lp));
                            let (mut p_one, mut p_inf) = (F128::ZERO, F128::ZERO);
                            for x in a..b {
                                let (r0, r1) = (2 * x, 2 * x + 1);
                                let (a0, a1) = (rd(s0, r0), rd(s0, r1));
                                let (b0, b1) = (rd(s1, r0), rd(s1, r1));
                                let (c0, c1) = (rd(s2, r0), rd(s2, r1));
                                let (d0, d1) = (rd(s3, r0), rd(s3, r1));
                                let v_one = a1 * b1 + lambda * (c1 * d1);
                                let v_inf =
                                    (a0 + a1) * (b0 + b1) + lambda * ((c0 + c1) * (d0 + d1));
                                let flat = g * pr + x;
                                let el = lo[flat & bmask] * hi[flat >> bshift];
                                p_one += el * v_one;
                                p_inf += el * v_inf;
                            }
                            (p_one, p_inf)
                        })
                        .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d));
                    let mass = range_eq_sum(lo, hi, plo, phi, g * pr + lp, (g + 1) * pr);
                    (g_one + one_plus_lambda * mass, g_inf)
                })
                .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d))
        } else {
            let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
            for g in 0..ng {
                let (o, i) = body(g);
                g_one += o;
                g_inf += i;
            }
            (g_one, g_inf)
        }
    } else {
        // Group-pair space: at most 2^c pairs — iterate densely.
        let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
        for x in 0..ng / 2 {
            let (a0, a1) = (vs[0].val(2 * x), vs[0].val(2 * x + 1));
            let (b0, b1) = (vs[1].val(2 * x), vs[1].val(2 * x + 1));
            let (c0, c1) = (vs[2].val(2 * x), vs[2].val(2 * x + 1));
            let (d0, d1) = (vs[3].val(2 * x), vs[3].val(2 * x + 1));
            let v_one = a1 * b1 + lambda * (c1 * d1);
            let v_inf = (a0 + a1) * (b0 + b1) + lambda * ((c0 + c1) * (d0 + d1));
            let el = lo[x % lo.len()] * hi[x / lo.len()];
            g_one += el * v_one;
            g_inf += el * v_inf;
        }
        (g_one, g_inf)
    }
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

// ---------------------------------------------------------------------------
// Single product-circuit GKR (prover + verifier halves)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Prover / Verifier (the permutation check)
// ---------------------------------------------------------------------------

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
    /// PoW witnesses in transcript order: the initial product fingerprint,
    /// then for every layer its lambda, round, and closing challenges.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// Fiat--Shamir grinding for the batched product-GKR permutation argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedGrinding {
    /// Nonzero enables the initial `(alpha, beta)` product fingerprint grind.
    /// Its actual bit count is raised to cover the live product degree.
    pub fingerprint_bits: u32,
    /// Linear batching challenge combining the two product circuits.
    pub lambda_bits: u32,
    /// Quadratic sumcheck challenge in each layer round.
    pub round_bits: u32,
    /// Linear challenge that collapses a layer's two boundary values.
    pub close_bits: u32,
}

impl BatchedGrinding {
    pub const fn disabled() -> Self {
        Self {
            fingerprint_bits: 0,
            lambda_bits: 0,
            round_bits: 0,
            close_bits: 0,
        }
    }

    pub const fn per_challenge_128() -> Self {
        Self {
            fingerprint_bits: 1,
            lambda_bits: 1,
            round_bits: 2,
            close_bits: 1,
        }
    }

    #[inline]
    fn fingerprint_bits_for(self, live_entries: usize) -> u32 {
        if self.fingerprint_bits == 0 {
            0
        } else {
            self.fingerprint_bits
                .max(crate::challenger::grinding_bits_for_degree(
                    live_entries.saturating_sub(1),
                ))
        }
    }

    #[inline]
    fn nonce_count(self, mu: usize, live_entries: usize) -> usize {
        usize::from(self.fingerprint_bits_for(live_entries) != 0)
            + mu * usize::from(self.lambda_bits != 0)
            + (mu * mu.saturating_sub(1) / 2) * usize::from(self.round_bits != 0)
            + mu * usize::from(self.close_bits != 0)
    }
}

#[inline]
fn grind_sample<C: Challenger>(ch: &mut C, nonces: &mut Vec<u64>, bits: u32) -> F128 {
    if bits != 0 {
        let (nonce, challenge) = ch.grind_pow_and_sample_f128(bits);
        nonces.push(nonce);
        challenge
    } else {
        ch.sample_f128()
    }
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
    live: Option<&LiveMask>,
    ch: &mut C,
) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
    prove_batched_with_grinding(f, g, sigma, live, BatchedGrinding::disabled(), ch)
}

/// [`prove_batched`] with an explicit Fiat--Shamir grinding policy.
pub fn prove_batched_with_grinding<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    live: Option<&LiveMask>,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
    prove_batched_impl(f, g, sigma, live, false, grinding, ch)
}

/// The DENSE pipeline forced under a mask — the grouped pipeline's
/// permanent differential oracle (`grouped_matches_dense_masked`).
#[cfg(test)]
pub(crate) fn prove_batched_dense_masked_for_tests<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    live: Option<&LiveMask>,
    ch: &mut C,
) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
    prove_batched_impl(f, g, sigma, live, true, BatchedGrinding::disabled(), ch)
}

/// The sparse masked σ evaluation: `Σ_live eq_lo(row)·eq_hi(slot)·tag(σ(x))`
/// — O(live + 2^ν + 2^c), exact under XOR addition, identical to the dense
/// masked-table MLE.
fn sparse_sigma_eval(sigma: &[usize], m: &LiveMask, rho: &[F128]) -> F128 {
    let (lo, hi) = rho.split_at(m.nu);
    let eq_lo = crate::zerocheck::univariate_skip::build_eq(lo);
    let eq_hi = crate::zerocheck::univariate_skip::build_eq(hi);
    m.counts
        .par_iter()
        .enumerate()
        .map(|(iota, &cnt)| {
            let base = iota << m.nu;
            let mut acc = F128::ZERO;
            for row in 0..cnt {
                acc += eq_lo[row] * F128::new(sigma[base + row] as u64, 0);
            }
            eq_hi[iota] * acc
        })
        .reduce(|| F128::ZERO, |a, b| a + b)
}

/// The GROUPED masked pipeline (phase 2b): live-prefix leaves, prefix-only
/// layer building and folds, and round messages with closed-form all-ones
/// tails. Transcript-identical to the dense masked pipeline (the
/// `grouped_matches_dense_masked` oracle).
fn prove_batched_grouped<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    m: &LiveMask,
    alpha: F128,
    beta: F128,
    grinding: BatchedGrinding,
    grinding_nonces: &mut Vec<u64>,
    ch: &mut C,
) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
    let n = f.len();
    let mu = n.trailing_zeros() as usize;
    assert!(mu <= 64, "s_id-as-index needs μ ≤ 64");
    let tag = |i: usize| F128::new(i as u64, 0);
    let rows = 1usize << m.nu;
    let mut t = std::time::Instant::now();

    // Leaves: live prefixes only (tails are implicit 1s, never written).
    // Parallel over (group, sub-chunk): the σ gather is read-only and the
    // writes are disjoint — this pass was the GKR's single largest line
    // multi-threaded (~2.9M live entries, serial).
    let mut lhs_buf = crate::scratch::take_f128(n);
    let mut rhs_buf = crate::scratch::take_f128(n);
    const LEAF_CH: usize = 1 << 14;
    lhs_buf[..n]
        .par_chunks_mut(rows)
        .zip(rhs_buf[..n].par_chunks_mut(rows))
        .enumerate()
        .for_each(|(iota, (lg, rg))| {
            let cnt = m.counts[iota];
            let base = iota * rows;
            lg[..cnt]
                .par_chunks_mut(LEAF_CH)
                .zip(rg[..cnt].par_chunks_mut(LEAF_CH))
                .enumerate()
                .for_each(|(ci, (lc, rc))| {
                    let start = base + ci * LEAF_CH;
                    for (i, (l, r)) in lc.iter_mut().zip(rc.iter_mut()).enumerate() {
                        let x = start + i;
                        *l = f[x] + alpha * tag(x) + beta;
                        *r = g[x] + alpha * tag(sigma[x]) + beta;
                    }
                });
        });
    let lhs = GVec {
        buf: lhs_buf,
        lens: m.counts.clone(),
        rows,
    };
    let rhs = GVec {
        buf: rhs_buf,
        lens: m.counts.clone(),
        rows,
    };
    tp(&mut t, "  leaves(grouped)");

    // Layer stack, built downward: stack[d] is layer `mu − d`.
    let mut l_layers: Vec<GVec> = Vec::with_capacity(mu + 1);
    let mut r_layers: Vec<GVec> = Vec::with_capacity(mu + 1);
    l_layers.push(lhs);
    r_layers.push(rhs);
    for d in 0..mu {
        l_layers.push(gv_build_prev(&l_layers[d]));
        r_layers.push(gv_build_prev(&r_layers[d]));
    }
    let top_lhs = l_layers[mu].val(0);
    let top_rhs = r_layers[mu].val(0);
    ch.observe_f128(top_lhs);
    ch.observe_f128(top_rhs);
    tp(&mut t, "  build-layers(grouped)");

    let mut r_pt: Vec<F128> = Vec::new();
    let mut layers = Vec::with_capacity(mu);
    let (mut claim_l, mut claim_r) = (F128::ZERO, F128::ZERO);
    // Sub-phase attribution for the layer sumchecks (GKR_TRACE): eq-prep
    // (SplitEqGhash + prefix sums, serial per round), messages, folds+FS,
    // and the per-layer remainder.
    //
    // The message and fold passes stay SEPARATE, measured, not by
    // omission: the zerocheck-style fold-and-round fusion (round i's fold
    // also emitting round i+1's message — valid here because a layer's
    // round weights are eq over suffixes of the PREVIOUS layer's point,
    // known before ρ_i) was built and benchmarked at m32, with a
    // branch-free hot region for the all-live prefix. It measured a WASH
    // multi-threaded (54.8-63.8 vs ~54-60 ms GKR total) and +16 ms
    // single-threaded: the split passes each walk their OWN vector's live
    // length — a fold never visits an empty or short partner at all —
    // while a fused pass walks the MAX length across all four views for
    // the message, dragging fold logic through the implicit-ones regions.
    // On the grouped shapes (live groups split-paired with empty ones)
    // that tax exceeds the saved traversal.
    let (mut d_eq, mut d_msg, mut d_fold) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );
    let t_sc_total = std::time::Instant::now();
    for k in 0..mu {
        let lambda = grind_sample(ch, grinding_nonces, grinding.lambda_bits);
        let mut rounds = Vec::with_capacity(k);
        let mut r_prime = Vec::with_capacity(k + 1);
        let (l0v, l1v) = l_layers[mu - (k + 1)].split();
        let (r0v, r1v) = r_layers[mu - (k + 1)].split();
        let mut cur: Option<[GVec; 4]> = None;
        for i in 0..k {
            let t_ph = std::time::Instant::now();
            let eq = SplitEqGhash::new(&r_pt[i + 1..k]);
            let mut plo = Vec::with_capacity(eq.lo.len() + 1);
            plo.push(F128::ZERO);
            for &e in &eq.lo {
                let last = *plo.last().unwrap();
                plo.push(last + e);
            }
            let mut phi = Vec::with_capacity(eq.hi.len() + 1);
            phi.push(F128::ZERO);
            for &e in &eq.hi {
                let last = *phi.last().unwrap();
                phi.push(last + e);
            }
            let t_ph = {
                d_eq += t_ph.elapsed();
                std::time::Instant::now()
            };
            let msg = match &cur {
                None => gv_message([&l0v, &l1v, &r0v, &r1v], lambda, &eq, &plo, &phi),
                Some([a, b, c, d]) => gv_message(
                    [&a.view(), &b.view(), &c.view(), &d.view()],
                    lambda,
                    &eq,
                    &plo,
                    &phi,
                ),
            };
            let t_ph = {
                d_msg += t_ph.elapsed();
                std::time::Instant::now()
            };
            ch.observe_f128(msg.0);
            ch.observe_f128(msg.1);
            let rho = grind_sample(ch, grinding_nonces, grinding.round_bits);
            rounds.push(msg);
            r_prime.push(rho);
            let cur_rows = match &cur {
                None => l0v.rows,
                Some([a, ..]) => a.rows,
            };
            let next = if cur_rows >= 2 {
                match &cur {
                    None => gv_fold4([&l0v, &l1v, &r0v, &r1v], rho),
                    Some([a, b, c, d]) => {
                        gv_fold4([&a.view(), &b.view(), &c.view(), &d.view()], rho)
                    }
                }
            } else {
                match &cur {
                    None => [
                        gv_fold(&l0v, rho),
                        gv_fold(&l1v, rho),
                        gv_fold(&r0v, rho),
                        gv_fold(&r1v, rho),
                    ],
                    Some([a, b, c, d]) => [
                        gv_fold(&a.view(), rho),
                        gv_fold(&b.view(), rho),
                        gv_fold(&c.view(), rho),
                        gv_fold(&d.view(), rho),
                    ],
                }
            };
            if let Some(old) = cur.take() {
                for v in old {
                    crate::scratch::give_f128(v.buf);
                }
            }
            cur = Some(next);
            d_fold += t_ph.elapsed();
        }
        let (vl0, vl1, vr0, vr1) = match &cur {
            None => (l0v.val(0), l1v.val(0), r0v.val(0), r1v.val(0)),
            Some([a, b, c, d]) => (a.val(0), b.val(0), c.val(0), d.val(0)),
        };
        if let Some(old) = cur.take() {
            for v in old {
                crate::scratch::give_f128(v.buf);
            }
        }
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
        let c_k = grind_sample(ch, grinding_nonces, grinding.close_bits);
        let one_plus_c = F128::ONE + c_k;
        claim_l = one_plus_c * vl0 + c_k * vl1;
        claim_r = one_plus_c * vr0 + c_k * vr1;
        r_prime.push(c_k);
        r_pt = r_prime;
    }
    for v in l_layers.into_iter().chain(r_layers) {
        crate::scratch::give_f128(v.buf);
    }
    if trace_on() {
        let tot = t_sc_total.elapsed();
        eprintln!(
            "  [prod-gkr]     sumcheck split: eq-prep {:7.3} ms | messages {:7.3} | \
             folds+fs {:7.3} | layer misc {:7.3}",
            d_eq.as_secs_f64() * 1e3,
            d_msg.as_secs_f64() * 1e3,
            d_fold.as_secs_f64() * 1e3,
            (tot.saturating_sub(d_eq + d_msg + d_fold)).as_secs_f64() * 1e3,
        );
    }
    tp(&mut t, "layer-sumchecks(grouped)");

    let rho = r_pt;
    let basis = s_id_basis(mu);
    let s_sigma_eval = sparse_sigma_eval(sigma, m, &rho);
    let lv = m.live_eval(&rho);
    let tail = (beta + F128::ONE) * lv + F128::ONE;
    let f_eval = claim_l + alpha * m.masked_id_eval(&basis, &rho) + tail;
    let g_eval = claim_r + alpha * s_sigma_eval + tail;
    observe_evals(ch, &[f_eval, g_eval, s_sigma_eval]);
    tp(&mut t, "evals(grouped)");

    let proof = ProductGkrBatchedProof {
        top_lhs,
        top_rhs,
        layers,
        f_eval,
        g_eval,
        s_sigma_eval,
        grinding_nonces: std::mem::take(grinding_nonces),
    };
    let claim = ProductGkrBatchedClaim {
        rho,
        f_eval,
        g_eval,
        s_sigma_eval,
    };
    (proof, claim)
}

fn prove_batched_impl<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    live: Option<&LiveMask>,
    force_dense: bool,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> (ProductGkrBatchedProof, ProductGkrBatchedClaim) {
    let n = f.len();
    assert_eq!(g.len(), n);
    assert_eq!(sigma.len(), n);
    assert!(n.is_power_of_two() && n >= 2, "need N = 2^μ ≥ 2");
    let mu = n.trailing_zeros() as usize;
    if let Some(m) = live {
        debug_assert_eq!(m.counts.len() << m.nu, n, "mask spans the leaf space");
        // Dead cells must be σ-fixed. Their f/g entries are NEVER READ
        // under a mask (semantically zero) — the caller may leave them
        // unwritten in a pooled buffer, so no value check here.
        debug_assert!(
            (0..n).all(|x| m.is_live(x) || sigma[x] == x),
            "dead cells must be σ-fixed"
        );
    }
    let mut t = std::time::Instant::now();
    ch.observe_label(DOMAIN_BATCHED);
    let live_entries = live.map_or(n, |m| m.counts.iter().sum());
    let mut grinding_nonces = Vec::with_capacity(grinding.nonce_count(mu, live_entries));
    let alpha = grind_sample(
        ch,
        &mut grinding_nonces,
        grinding.fingerprint_bits_for(live_entries),
    );
    let beta = ch.sample_f128();

    if !force_dense && let Some(m) = live {
        return prove_batched_grouped(
            f,
            g,
            sigma,
            m,
            alpha,
            beta,
            grinding,
            &mut grinding_nonces,
            ch,
        );
    }

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
    // With a mask, the σ fingerprint table is never MATERIALIZED: the rhs
    // leaves read `tag(σ(x))` inline on live cells, and the deferred
    // `live ⊙ s_σ` evaluation happens SPARSELY over the live cells after
    // ρ is known (the dead entries are zero by definition).
    let s_sig_vec: Vec<F128> = if live.is_some() {
        Vec::new()
    } else {
        sigma
            .par_iter()
            .with_min_len(par_threshold())
            .map(|&sx| tag(sx))
            .collect()
    };
    tp(&mut t, "  s_sigma");
    let lhs: Vec<F128> = f
        .par_iter()
        .enumerate()
        .with_min_len(par_threshold())
        .map(|(x, fx)| match live {
            Some(m) if !m.is_live(x) => F128::ONE,
            _ => *fx + alpha * tag(x) + beta,
        })
        .collect();
    let rhs: Vec<F128> = g
        .par_iter()
        .enumerate()
        .with_min_len(par_threshold())
        .map(|(x, gx)| match live {
            Some(m) if !m.is_live(x) => F128::ONE,
            _ => *gx + alpha * tag(sigma[x]) + beta,
        })
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
    let mut cur: [Vec<F128>; 4] = std::array::from_fn(|_| crate::scratch::take_f128(cap));
    let mut nxt: [Vec<F128>; 4] = std::array::from_fn(|_| crate::scratch::take_f128(cap));
    for k in 0..mu {
        let lambda = grind_sample(ch, &mut grinding_nonces, grinding.lambda_bits);
        let h = 1usize << k;
        // Live prefix length of each `cur` buffer; set at round 0, halved after.
        let mut len = 0usize;
        let mut rounds = Vec::with_capacity(k);
        let mut r_prime = Vec::with_capacity(k + 1);
        // Round 0's message is the one read that cannot be fused: nothing has
        // been folded yet, so it comes straight off the layer. Every later
        // round's message is produced by the preceding fold.
        let mut pending = if k > 0 {
            let tr = std::time::Instant::now();
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
            let rho = grind_sample(ch, &mut grinding_nonces, grinding.round_bits);
            rounds.push((g1, g_inf));
            r_prime.push(rho);

            let mut tr = std::time::Instant::now();
            // eq for round i+1; `None` on the last round, where the fold has no
            // successor message to emit.
            let eq_next = (i + 1 < k).then(|| SplitEqGhash::new(&r_pt[i + 2..k]));
            if trace_on() {
                eq_ns += tr.elapsed().as_nanos();
                tr = std::time::Instant::now();
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
                std::mem::swap(&mut cur, &mut nxt);
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
        let c_k = grind_sample(ch, &mut grinding_nonces, grinding.close_bits);
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
    // The deferred σ evaluation: dense MLE without a mask; with one, the
    // SPARSE sum over live cells only ([`sparse_sigma_eval`]).
    let s_sigma_eval = match live {
        None => mle_eval(&s_sig_vec, &rho),
        Some(m) => sparse_sigma_eval(sigma, m, &rho),
    };
    // Reconstruct the witness evals from the collapsed claims (char 2:
    // subtraction is addition). With a mask, the leaf's affine form is
    // `w + α·(live⊙s_id) + (β+1)·live + 1` — see [`LiveMask`].
    let (f_eval, g_eval) = match live {
        None => (
            claim_l + alpha * s_id_eval(&basis, &rho) + beta,
            claim_r + alpha * s_sigma_eval + beta,
        ),
        Some(m) => {
            let lv = m.live_eval(&rho);
            let tail = (beta + F128::ONE) * lv + F128::ONE;
            (
                claim_l + alpha * m.masked_id_eval(&basis, &rho) + tail,
                claim_r + alpha * s_sigma_eval + tail,
            )
        }
    };
    observe_evals(ch, &[f_eval, g_eval, s_sigma_eval]);
    // Hand the ping-pong buffers back so the next prove reuses resident pages.
    for u in cur.into_iter().chain(nxt) {
        crate::scratch::give_f128(u);
    }
    tp(&mut t, "evals");

    let proof = ProductGkrBatchedProof {
        top_lhs,
        top_rhs,
        layers,
        f_eval,
        g_eval,
        s_sigma_eval,
        grinding_nonces,
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
    live: Option<&LiveMask>,
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, ProductGkrError> {
    verify_batched_with_grinding(mu, proof, live, BatchedGrinding::disabled(), ch)
}

/// [`verify_batched`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_batched_with_grinding<C: Challenger>(
    mu: usize,
    proof: &ProductGkrBatchedProof,
    live: Option<&LiveMask>,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, ProductGkrError> {
    verify_batched_core(mu, proof, None, live, grinding, ch)
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
    live: Option<&LiveMask>,
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, ProductGkrError> {
    assert_eq!(sigma.len(), 1usize << mu, "σ length must be 2^mu");
    verify_batched_with_sigma_and_grinding(mu, proof, sigma, live, BatchedGrinding::disabled(), ch)
}

/// [`verify_batched_with_sigma`] with an explicit grinding policy.
pub fn verify_batched_with_sigma_and_grinding<C: Challenger>(
    mu: usize,
    proof: &ProductGkrBatchedProof,
    sigma: &[usize],
    live: Option<&LiveMask>,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, ProductGkrError> {
    assert_eq!(sigma.len(), 1usize << mu, "σ length must be 2^mu");
    verify_batched_core(mu, proof, Some(sigma), live, grinding, ch)
}

fn verify_batched_core<C: Challenger>(
    mu: usize,
    proof: &ProductGkrBatchedProof,
    sigma_opt: Option<&[usize]>,
    live: Option<&LiveMask>,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<ProductGkrBatchedClaim, ProductGkrError> {
    if proof.layers.len() != mu {
        return Err(ProductGkrError::MalformedProof);
    }
    let n = 1usize << mu;
    let live_entries = live.map_or(n, |m| m.counts.iter().sum());
    if proof.grinding_nonces.len() != grinding.nonce_count(mu, live_entries) {
        return Err(ProductGkrError::InvalidGrinding);
    }
    let mut nonce_idx = 0usize;
    let mut verify_grind_sample = |ch: &mut C, bits: u32| -> Result<F128, ProductGkrError> {
        if bits != 0 {
            let nonce = proof.grinding_nonces[nonce_idx];
            nonce_idx += 1;
            ch.verify_pow_and_sample_f128(nonce, bits)
                .ok_or(ProductGkrError::InvalidGrinding)
        } else {
            Ok(ch.sample_f128())
        }
    };

    ch.observe_label(DOMAIN_BATCHED);
    let alpha = verify_grind_sample(ch, grinding.fingerprint_bits_for(live_entries))?;
    let beta = ch.sample_f128();

    ch.observe_f128(proof.top_lhs);
    ch.observe_f128(proof.top_rhs);
    if proof.top_lhs != proof.top_rhs {
        return Err(ProductGkrError::ProductMismatch);
    }

    let mut claim_l = proof.top_lhs;
    let mut claim_r = proof.top_rhs;
    let mut r_pt: Vec<F128> = Vec::new();
    for (k, layer) in proof.layers.iter().enumerate() {
        if layer.rounds.len() != k {
            return Err(ProductGkrError::MalformedProof);
        }
        let lambda = verify_grind_sample(ch, grinding.lambda_bits)?;
        let mut c_run = claim_l + lambda * claim_r;
        let mut r_prime = Vec::with_capacity(k + 1);
        for i in 0..k {
            let (g1, g_inf) = layer.rounds[i];
            let r_eq = r_pt[i];
            let one_plus_r_eq = F128::ONE + r_eq;
            let g0 = (c_run + r_eq * g1) * one_plus_r_eq.inv();
            ch.observe_f128(g1);
            ch.observe_f128(g_inf);
            let rho = verify_grind_sample(ch, grinding.round_bits)?;
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
            return Err(ProductGkrError::LayerCheckFailed);
        }
        let c_k = verify_grind_sample(ch, grinding.close_bits)?;
        let one_plus_c = F128::ONE + c_k;
        claim_l = one_plus_c * vl0 + c_k * vl1;
        claim_r = one_plus_c * vr0 + c_k * vr1;
        r_prime.push(c_k);
        r_pt = r_prime;
    }

    // Input-layer checks at the shared ρ: both reconstructed affinely, sharing
    // the single witness eval (f_eval = g_eval = w(ρ) when f = g = w). With a
    // mask, the leaf's affine form is `w + α·(live⊙s_id) + (β+1)·live + 1`
    // and the σ table is the MASKED `live ⊙ s_σ` — see [`LiveMask`].
    let basis = s_id_basis(mu);
    // s_σ(ρ): verifier-computed when σ is known (not trusting the proof), else
    // the proof's claimed value.
    let s_sigma = match (sigma_opt, live) {
        (Some(sigma), Some(m)) => {
            // Sparse masked evaluation over the live cells only — the same
            // sum the prover computes, exact under XOR addition.
            let (lo, hi) = r_pt.split_at(m.nu);
            let eq_lo = crate::zerocheck::univariate_skip::build_eq(lo);
            let eq_hi = crate::zerocheck::univariate_skip::build_eq(hi);
            let mut acc = F128::ZERO;
            for (iota, &cnt) in m.counts.iter().enumerate() {
                let base = iota << m.nu;
                let mut s = F128::ZERO;
                for row in 0..cnt {
                    s += eq_lo[row] * F128::new(sigma[base + row] as u64, 0);
                }
                acc += eq_hi[iota] * s;
            }
            acc
        }
        (Some(sigma), None) => {
            let s_id_vec = build_s_id_vec(mu, &basis);
            let s_sig: Vec<F128> = sigma.iter().map(|&sx| s_id_vec[sx]).collect();
            mle_eval(&s_sig, &r_pt)
        }
        (None, _) => proof.s_sigma_eval,
    };
    let (lhs_in, rhs_in) = match live {
        None => (
            proof.f_eval + alpha * s_id_eval(&basis, &r_pt) + beta,
            proof.g_eval + alpha * s_sigma + beta,
        ),
        Some(m) => {
            let lv = m.live_eval(&r_pt);
            let tail = (beta + F128::ONE) * lv + F128::ONE;
            (
                proof.f_eval + alpha * m.masked_id_eval(&basis, &r_pt) + tail,
                proof.g_eval + alpha * s_sigma + tail,
            )
        }
    };
    if claim_l != lhs_in || claim_r != rhs_in {
        return Err(ProductGkrError::InputMismatch);
    }

    observe_evals(ch, &[proof.f_eval, proof.g_eval, s_sigma]);
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());

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

    /// The GROUPED pipeline is transcript-identical to the dense masked
    /// pipeline — proof and claim byte-equal on random (f, g, σ, mask),
    /// validity not required. The permanent differential oracle for the
    /// live-prefix layer machinery.
    #[test]
    fn grouped_matches_dense_masked() {
        let mut rng = Rng::new(0x6B0B_2026);
        for (nu, c, seed_counts) in [
            (4usize, 3usize, [5usize, 16, 0, 9, 1, 13, 7, 3].as_slice()),
            (
                3,
                4,
                [8, 0, 3, 1, 7, 2, 5, 4, 0, 8, 6, 1, 2, 3, 0, 5].as_slice(),
            ),
            (5, 2, [31, 0, 17, 32].as_slice()),
        ] {
            let mu = nu + c;
            let n = 1usize << mu;
            let mask = LiveMask {
                nu,
                counts: seed_counts.to_vec(),
            };
            // Random live values, zero dead; σ permutes live cells only.
            let f: Vec<F128> = (0..n)
                .map(|x| {
                    if mask.is_live(x) {
                        rng.f128()
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            let g: Vec<F128> = (0..n)
                .map(|x| {
                    if mask.is_live(x) {
                        rng.f128()
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            let live_ix: Vec<usize> = (0..n).filter(|&x| mask.is_live(x)).collect();
            let perm = rng.permutation(live_ix.len());
            let mut sigma: Vec<usize> = (0..n).collect();
            for (a, &pb) in perm.iter().enumerate() {
                sigma[live_ix[a]] = live_ix[pb];
            }
            let mut ch1 = FsChallenger::new(b"grouped-oracle");
            let (p1, c1) = prove_batched(&f, &g, &sigma, Some(&mask), &mut ch1);
            let mut ch2 = FsChallenger::new(b"grouped-oracle");
            let (p2, c2) =
                prove_batched_dense_masked_for_tests(&f, &g, &sigma, Some(&mask), &mut ch2);
            assert_eq!(p1, p2, "grouped == dense masked (proof), nu {nu} c {c}");
            assert_eq!(c1, c2, "grouped == dense masked (claim), nu {nu} c {c}");
        }
    }

    /// The live mask's closed forms match the explicit tables, and a masked
    /// batched roundtrip accepts on BOTH verify arms with dead cells as
    /// identity leaves — the SP1-style padding contract.
    #[test]
    fn live_mask_closed_forms_and_masked_roundtrip() {
        let mut rng = Rng::new(0x11FE_2026);
        let (nu, c) = (4usize, 3usize);
        let mu = nu + c;
        let n = 1usize << mu;
        let counts: Vec<usize> = vec![5, 16, 0, 9, 1, 13, 7, 3];
        let mask = LiveMask {
            nu,
            counts: counts.clone(),
        };
        let basis = s_id_basis(mu);
        let live_vec: Vec<F128> = (0..n)
            .map(|x| {
                if mask.is_live(x) {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            })
            .collect();
        let mid_vec: Vec<F128> = (0..n)
            .map(|x| {
                if mask.is_live(x) {
                    s_id_value(x, &basis)
                } else {
                    F128::ZERO
                }
            })
            .collect();
        for _ in 0..4 {
            let rho: Vec<F128> = (0..mu).map(|_| rng.f128()).collect();
            assert_eq!(
                mask.live_eval(&rho),
                mle_eval(&live_vec, &rho),
                "livê closed form"
            );
            assert_eq!(
                mask.masked_id_eval(&basis, &rho),
                mle_eval(&mid_vec, &rho),
                "M̂ closed form"
            );
        }
        // A live-only permutation (dead cells σ-fixed) over a witness that
        // is constant on live cells and zero on dead — honest for f = g = w.
        let live_ix: Vec<usize> = (0..n).filter(|&x| mask.is_live(x)).collect();
        let perm = rng.permutation(live_ix.len());
        let mut sigma: Vec<usize> = (0..n).collect();
        for (a, &pb) in perm.iter().enumerate() {
            sigma[live_ix[a]] = live_ix[pb];
        }
        let w: Vec<F128> = (0..n)
            .map(|x| {
                if mask.is_live(x) {
                    F128::new(0xD00D, 7)
                } else {
                    F128::ZERO
                }
            })
            .collect();
        let mut chp = FsChallenger::new(b"live-mask-test");
        let (proof, claim_p) = prove_batched(&w, &w, &sigma, Some(&mask), &mut chp);
        // The top product covers ONLY the live multiset (dead leaves = 1).
        let mut chv = FsChallenger::new(b"live-mask-test");
        let claim_v = verify_batched_with_sigma(mu, &proof, &sigma, Some(&mask), &mut chv)
            .expect("masked sigma-aware verify accepts");
        assert_eq!(claim_p, claim_v, "prover and verifier agree on the claim");
        let mut chv2 = FsChallenger::new(b"live-mask-test");
        let claim_t = verify_batched(mu, &proof, Some(&mask), &mut chv2)
            .expect("masked trusting verify accepts");
        assert_eq!(claim_p, claim_t, "the trusting arm agrees");
        // A mask disagreement is caught: verifying with a different count
        // fails the input checks.
        let mut wrong = mask.clone();
        wrong.counts[0] += 1;
        let mut chv3 = FsChallenger::new(b"live-mask-test");
        assert!(
            verify_batched_with_sigma(mu, &proof, &sigma, Some(&wrong), &mut chv3).is_err(),
            "a drifted mask fails"
        );
    }

    use crate::test_rng::Rng;

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
            let (_proof, claim) = prove_batched(&f, &g, &sigma, None, &mut ch);
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
            let (proof, claim_p) = prove_batched(&f, &g, &sigma, None, &mut chp);
            assert_eq!(proof.top_lhs, proof.top_rhs, "μ={mu}: ∏lhs ≠ ∏rhs");
            let mut chv = FsChallenger::new(b"prod-gkr-batched-test");
            bind(&mut chv, &f, &g, &sigma);
            let claim_v = verify_batched(mu, &proof, None, &mut chv).expect("verify");
            assert_eq!(claim_p, claim_v, "μ={mu}");
            assert_eq!(claim_v.rho.len(), mu, "single shared reduction point");
        }
    }

    #[test]
    fn batched_grinding_roundtrip_and_rejects_bad_nonce_shape() {
        let mu = 5;
        let (f, g, sigma) = honest_instance(mu, 0x1280_B17);
        let policy = BatchedGrinding::per_challenge_128();
        let mut chp = FsChallenger::new(b"prod-gkr-batched-grinding-test");
        bind(&mut chp, &f, &g, &sigma);
        let (proof, claim_p) = prove_batched_with_grinding(&f, &g, &sigma, None, policy, &mut chp);
        assert_eq!(proof.grinding_nonces.len(), policy.nonce_count(mu, 1 << mu));

        let mut chv = FsChallenger::new(b"prod-gkr-batched-grinding-test");
        bind(&mut chv, &f, &g, &sigma);
        let claim_v = verify_batched_with_grinding(mu, &proof, None, policy, &mut chv)
            .expect("grinded product-GKR verifies");
        assert_eq!(claim_p, claim_v);

        let mut missing = proof;
        missing.grinding_nonces.pop();
        let mut chv = FsChallenger::new(b"prod-gkr-batched-grinding-test");
        bind(&mut chv, &f, &g, &sigma);
        assert_eq!(
            verify_batched_with_grinding(mu, &missing, None, policy, &mut chv),
            Err(ProductGkrError::InvalidGrinding)
        );
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
        let (proof, claim) = prove_batched(&w, &w, &sigma, None, &mut chp);
        assert_eq!(proof.top_lhs, proof.top_rhs);
        assert_eq!(claim.f_eval, claim.g_eval, "f=g=w ⇒ one witness eval at ρ");
        let mut chv = FsChallenger::new(b"prod-gkr-batched-test");
        bind(&mut chv, &w, &w, &sigma);
        verify_batched(mu, &proof, None, &mut chv).expect("verify");
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
        let (proof, _) = prove_batched(&w, &w, &sigma, None, &mut chp);
        let mut chv = FsChallenger::new(b"prod-gkr-batched-test");
        bind(&mut chv, &w, &w, &sigma);
        assert!(verify_batched(mu, &proof, None, &mut chv).is_err());
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
        let (batched, _) = prove_batched(&f, &g, &sigma, None, &mut chp);

        let verify_shape = |p: &ProductGkrBatchedProof, mu: usize| {
            let mut ch = FsChallenger::new(b"prod-gkr-shape");
            bind(&mut ch, &f, &g, &sigma);
            verify_batched(mu, p, None, &mut ch)
        };
        // Wrong layer count (both directions).
        let mut short = batched.clone();
        short.layers.pop();
        assert_eq!(
            verify_shape(&short, mu),
            Err(ProductGkrError::MalformedProof)
        );
        assert_eq!(
            verify_shape(&batched, mu - 1),
            Err(ProductGkrError::MalformedProof)
        );
        // Right layer count, wrong round count inside a layer.
        let mut bad_rounds = batched.clone();
        bad_rounds.layers[mu - 1].rounds.pop();
        assert_eq!(
            verify_shape(&bad_rounds, mu),
            Err(ProductGkrError::MalformedProof)
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
        let (proof, claim) = prove_batched(&f, &g, &sigma, None, &mut chp);

        let check = |p: &ProductGkrBatchedProof| {
            let mut ch = FsChallenger::new(DOMAIN_TEST);
            bind(&mut ch, &f, &g, &sigma);
            verify_batched_with_sigma(mu, p, &sigma, None, &mut ch)
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
        let (proof, _) = prove_batched(&f, &g, &sigma, None, &mut chp);

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
        let claim = verify_batched(mu, &forged, None, &mut chv).expect("trusting verifier accepts");
        assert_eq!(claim.g_eval, proof.g_eval + delta);
        assert_ne!(claim.g_eval, proof.g_eval, "forgery really did change g(ρ)");

        // σ-aware verifier: recomputes s_σ(ρ), so the shift no longer cancels.
        let mut chv = FsChallenger::new(DOMAIN_TEST);
        bind(&mut chv, &f, &g, &sigma);
        assert_eq!(
            verify_batched_with_sigma(mu, &forged, &sigma, None, &mut chv),
            Err(ProductGkrError::InputMismatch),
        );

        // ...and still accepts the honest proof.
        let mut chv = FsChallenger::new(DOMAIN_TEST);
        bind(&mut chv, &f, &g, &sigma);
        verify_batched_with_sigma(mu, &proof, &sigma, None, &mut chv)
            .expect("honest proof verifies");
    }

    /// The F128 additive-NTT PREFIX-EXTENSION property that a GKR univariate
    /// skip would stand on: inv-NTT(dim 6) of 64 evaluations gives LCH
    /// coefficients whose zero-padded fwd-NTT(dim 7) reproduces the original
    /// 64 values as the first half — the 6-dim basis is a prefix of the
    /// 7-dim one, so the degree-<64 interpolant extends to the 128-point
    /// domain by padding alone.
    #[test]
    fn f128_ntt_prefix_extension_roundtrip() {
        use crate::ntt::AdditiveNttF128;
        let ntt6 = AdditiveNttF128::standard(6);
        let ntt7 = AdditiveNttF128::standard(7);
        let mut rng = Rng::new(0x1717_5C1F);
        let vals: Vec<F128> = (0..64).map(|_| rng.f128()).collect();
        // Roundtrip under dim 6.
        let mut c = vals.clone();
        ntt6.inverse_transform_scalar(&mut c);
        let mut back = c.clone();
        ntt6.forward_transform_scalar(&mut back);
        assert_eq!(back, vals, "inv is the inverse of fwd (dim 6)");
        // Prefix extension: pad coefficients, evaluate on the 128 domain.
        let mut padded = c;
        padded.resize(128, F128::ZERO);
        ntt7.forward_transform_scalar(&mut padded);
        assert_eq!(&padded[..64], &vals[..], "prefix basis: S evals reproduced");
    }
}
