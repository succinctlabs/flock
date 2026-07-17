//! HyperPlonk-style permutation check over GF(2^128).
//!
//! Proves that two multilinear polynomials `f, g` over the boolean hypercube
//! `B_μ` (`N = 2^μ` evaluations) are related by a permutation `σ` of `[0, N)`,
//! via the multiset equality
//!
//!   { (f(x), s_id(x)) : x ∈ B_μ }  =  { (g(x), s_σ(x)) : x ∈ B_μ }
//!
//! where `s_id` is an injective index tag and `s_σ(x) = s_id(σ(x))`. With this
//! choice the relation is equivalent to `f(x) = g(σ⁻¹(x))` for all `x`
//! (equivalently `f(σ(x)) = g(x)`); Plonk copy-constraints are the `f == g`
//! special case. The multiset equality is what makes `∏ p(x) = ∏ q(x)` hold.
//!
//! ## Construction (single fractional product)
//!
//! Sample `β, γ`. Set `p(x) = f(x) + β·s_id(x) + γ`, `q(x) = g(x) + β·s_σ(x) + γ`,
//! and the leaf fractions `ℓ(x) = p(x)·q(x)⁻¹`. Multiset equality ⟺ `∏ ℓ(x) = 1`.
//! We prove this with a binary product tree encoded in **one** auxiliary
//! multilinear `v` (size `2N`): its first half holds the leaves `v(0,x) = ℓ(x)`
//! and its second half the internal products. A single batched zerocheck over
//! `μ` variables combines two relations with powers of a challenge `α`:
//!
//!   A (fraction):  v(0,x)·q(x) ⊕ p(x) = 0      (leaves = p/q)
//!   B (recursion): c(x) ⊕ a(x)·b(x) = 0,  a=v(·,0), b=v(·,1), c=v(1,·)
//!
//! plus the root check `v[2N-2] = ∏ℓ = 1`. Because the leaves live *inside* `v`
//! (its first half), there is no separate `h` polynomial and no separate
//! "leaf-consistency" relation — relation A constrains `v(0,·)` directly.
//!
//! ## Scope
//!
//! This is a PIOP for the witness side (`f, g`): `prove`/`verify` reduce the
//! claim to MLE evaluation claims on `f, g, s_σ` at a random point `ρ` (returned
//! in `PermutationClaim`); a downstream PCS over the witness would open them. The
//! caller must absorb `f, g, σ` into the challenger before calling, exactly as
//! the witness is PCS-committed before `zerocheck::prove_packed`.
//!
//! The single prover-created aux polynomial `v` is **committed with the real
//! PCS** (Binius-style, [`crate::pcs`]): its Merkle root is observed into the
//! transcript before the zerocheck challenges (binding `v` — a known `ρ` would
//! break zerocheck soundness), and after the sumcheck `v` is opened at five
//! points — `(ρ,0), (ρ,1), (1,ρ), (0,ρ)` and the product root `2N-2` — in one
//! batched opening. The witness itself is not committed yet.
//!
//! ### PCS backend
//!
//! The opening uses **Ligerito**, which requires `v` to be large enough for
//! the recursion to be feasible (`ligerito::default_config` succeeds). At
//! `log_batch_size = log_inv_rate = 1` the floor is `log_n ≥ 8` (the L0 block
//! must hold `udr_queries(1) = 243` distinct queries, so `2^log_n ≥ 243`);
//! since `v` has `μ+1` vars, that means **μ ≥ 7**. Committing a single poly
//! (rather than `h` and `v` separately) also keeps it to one opening —
//! Ligerito's succinct verifier is not transcript-balanced for chaining two
//! opens on one challenger.
//!
//! Reuses `build_eq` (`zerocheck::univariate_skip`), the `F128` field arithmetic
//! (incl. `inv`), the `Challenger` Fiat–Shamir trait, the PCS commit/open over
//! F128-packed multilinears, and mirrors the eq-trick sumcheck verifier chain in
//! `zerocheck.rs`.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::merkle::Hash;
use crate::pcs::ligerito::{ProverConfig, VerifierConfig};
use crate::pcs::{
    self, Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
    ProverData, commit,
};
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::univariate_skip::{SplitEqGhash, build_eq};

const DOMAIN: &[u8] = b"flock-perm-v0";

// ---------------------------------------------------------------------------
// Proof / claim / error types
// ---------------------------------------------------------------------------

/// Permutation-check proof. `rounds` carries the per-round `(G(1), G(∞))`
/// messages (Convention A — bare, no `r[i]` prefactor); the eight evals are the
/// MLE openings the sumcheck reduces to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationProof {
    /// Merkle root of the PCS commitment to the grand-product poly `v` (size `2N`).
    pub v_root: Hash,
    /// Claimed grand product `∏ ℓ(x)`; must be `ONE` for an honest permutation.
    pub claimed_product: F128,
    /// Per-round `(G(1), G(∞))`, length `μ`.
    pub rounds: Vec<(F128, F128)>,
    pub f_eval: F128,
    pub g_eval: F128,
    pub s_sigma_eval: F128,
    pub v_x0: F128, // v(ρ, 0)
    pub v_x1: F128, // v(ρ, 1)
    pub v_1x: F128, // v(1, ρ)
    pub v_0x: F128, // v(0, ρ) — the leaf value ℓ(ρ), used in relation A
    /// Ligerito PCS opening of `v` at the five points `(ρ,0), (ρ,1), (1,ρ),
    /// (0,ρ)` and the product root `2N-2`.
    pub v_open: pcs::BatchOpeningProofLigerito,
}

/// Evaluation claims the verifier outputs. `f_eval, g_eval, s_sigma_eval` are
/// for a downstream witness PCS; the `v_*` evals are already proven here by the
/// `v` opening and are surfaced for inspection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationClaim {
    pub rho: Vec<F128>,
    pub f_eval: F128,
    pub g_eval: F128,
    pub s_sigma_eval: F128,
    pub v_x0: F128,
    pub v_x1: F128,
    pub v_1x: F128,
    pub v_0x: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// Final sumcheck consistency `c_running == F(ρ)` failed.
    SumcheckFinalFailed,
    /// Claimed grand product was not `1`.
    RootNotOne,
    /// The PCS opening of `v` failed to verify.
    PcsOpen(pcs::VerifyError),
}

// ---------------------------------------------------------------------------
// Field / polynomial helpers
// ---------------------------------------------------------------------------

/// Montgomery batch inverse, **chunked-parallel**: each chunk runs its own
/// Montgomery pass (one `F128::inv` + ~3·len muls) independently, so the work
/// parallelizes across chunks at the cost of one extra inversion per chunk
/// (negligible: ~N/CHUNK inversions total). Panics (debug) on a zero input —
/// `q(x) = 0` happens with probability `2⁻¹²⁸` per `x` for random `β, γ`.
fn batch_inverse(values: &[F128]) -> Vec<F128> {
    // Chunk large enough that the per-chunk extra inversion is in the noise, yet
    // small enough to give rayon plenty of tasks for load balancing.
    const CHUNK: usize = 1 << 14;
    let mut out = vec![F128::ZERO; values.len()];
    out.par_chunks_mut(CHUNK)
        .zip(values.par_chunks(CHUNK))
        .for_each(|(out_c, val_c)| {
            let mut acc = F128::ONE;
            for (o, v) in out_c.iter_mut().zip(val_c) {
                debug_assert!(!v.is_zero(), "batch_inverse: zero input");
                *o = acc; // prefix product within this chunk
                acc *= *v;
            }
            acc = acc.inv(); // (∏ chunk)⁻¹
            for (o, v) in out_c.iter_mut().zip(val_c).rev() {
                *o *= acc;
                acc *= *v;
            }
        });
    out
}

/// Basis for the identity tag `s_id`: `basis[i]` is the field element with bit
/// `i` set. `{basis_i}` are GF(2)-linearly independent, so `s_id` is injective
/// on `B_μ` (requires `μ ≤ 128`).
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
/// Test-only reference; the prover builds the whole `s_id` table by doubling.
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

/// Closed-form MLE of `s_id` at `ρ`: `Σ_i basis_i · ρ_i` (it is GF(2)-linear).
fn s_id_eval(basis: &[F128], rho: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (b, r) in basis.iter().zip(rho) {
        acc += *b * *r;
    }
    acc
}

/// Build the grand-product poly `v` of size `2N` from leaves `h` (size `N`):
/// `v[i]=h[i]` for `i<N`, `v[N+i]=v[2i]·v[2i+1]`, and `v[2N-1]=0` (padding, so
/// the recursion holds trivially at `i=N-1`). The total product is `v[2N-2]`.
///
/// Computed **level by level**: each level is the products of the previous
/// level's adjacent pairs and parallelizes over its `size/2` outputs; levels run
/// in sequence (the recursion's data dependency), giving `O(N)` work at `O(log N)`
/// depth. Identical output to the naive forward `v[N+i]=v[2i]·v[2i+1]` scan.
fn build_grand_product(h: &[F128]) -> Vec<F128> {
    let n = h.len();
    assert!(n.is_power_of_two() && n >= 2);
    let mut v = vec![F128::ZERO; 2 * n];
    v[..n].copy_from_slice(h);
    // `read` = start of the current level (size `size`); products go to
    // `write..write+size/2`, which is the next level. `read` for the next level
    // is the segment we just wrote.
    let mut read = 0usize;
    let mut size = n;
    let mut write = n;
    while size >= 2 {
        let half = size / 2;
        // src = v[read..read+size] lives in the already-written prefix; dst is
        // the disjoint segment v[write..write+half]. Split so both borrow safely.
        let (done, rest) = v.split_at_mut(write);
        let src = &done[read..read + size];
        rest[..half]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, d)| *d = src[2 * j] * src[2 * j + 1]);
        read = write;
        write += half;
        size = half;
    }
    // v[2n-1] remains ZERO (padding).
    v
}

/// Size at/above which a per-round sumcheck op is worth dispatching to rayon.
/// Below it the serial path avoids parallel-dispatch + alloc overhead — crucial
/// because sumcheck rounds shrink geometrically, so most rounds are tiny.
const PAR_THRESHOLD: usize = 1 << 12;

/// Bind the low variable at `ρ`: `u[x] ← u[2x]·(1+ρ) + u[2x+1]·ρ`, halving `u`.
/// Large folds run in parallel into a **pooled** buffer (reads old `u`, writes
/// new), then swap and recycle the old buffer — no per-round allocation. Small
/// folds stay serial and in place (no dispatch, no buffer).
fn fold_in_place(u: &mut Vec<F128>, rho: F128) {
    let half = u.len() / 2;
    let one_minus = F128::ONE + rho;
    if half >= PAR_THRESHOLD {
        // `take_f128(half)` returns a length-`half` buffer; the map writes every
        // slot (write-before-read contract satisfied).
        let mut out = crate::scratch::take_f128(half);
        out.par_iter_mut().enumerate().for_each(|(x, o)| {
            *o = u[2 * x] * one_minus + u[2 * x + 1] * rho;
        });
        let old = std::mem::replace(u, out);
        crate::scratch::give_f128(old);
    } else {
        for x in 0..half {
            u[x] = u[2 * x] * one_minus + u[2 * x + 1] * rho;
        }
        u.truncate(half);
    }
}

/// Direct MLE evaluation of `table` (length `2^k`) at `point` (length `k`),
/// binding low variable first (matches `fold_in_place`).
#[cfg(test)]
fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
    let mut t = table.to_vec();
    for &r in point {
        fold_in_place(&mut t, r);
    }
    t[0]
}

/// RS inverse rate (log₂) and interleaving batch size (log₂) for the aux-poly
/// commitments. Both backends' L0 commit and the Ligerito `default_config` must
/// agree on these, so they live in one place.
const PCS_LOG_INV_RATE: usize = 1;
const PCS_LOG_BATCH_SIZE: usize = 1;

/// PCS parameters for committing an `F128` multilinear in `num_vars` variables
/// (committed vector length `2^num_vars`). `m = num_vars + LOG_PACKING` so the
/// packed-direct opening point has length `num_vars`. Verifier rebuilds these
/// deterministically from `μ`, so the proof carries only the Merkle roots.
fn pcs_params(num_vars: usize) -> PcsParams {
    PcsParams {
        m: num_vars + LOG_PACKING,
        log_inv_rate: PCS_LOG_INV_RATE,
        log_batch_size: PCS_LOG_BATCH_SIZE,
        profile: Default::default(),
    }
}

/// Ligerito prover config for an `F128` multilinear in `num_vars` variables.
/// Panics when the poly is too small for Ligerito's recursion (`num_vars < 8`,
/// i.e. μ < 7 — see the module docs). Deterministic in `num_vars`, so prover
/// and verifier build the same config.
fn ligerito_prover_config(num_vars: usize) -> ProverConfig {
    pcs::ligerito::default_config(num_vars, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito config for the aux poly v; permutation check requires μ ≥ 7")
}

/// Verifier counterpart to [`ligerito_prover_config`]; agrees with it (both
/// gate on the same feasibility check).
fn ligerito_verifier_config(num_vars: usize) -> VerifierConfig {
    pcs::ligerito::default_verifier_config(num_vars, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito verifier config for the aux poly v; permutation check requires μ ≥ 7")
}

/// Open `poly` (length `2^num_vars`, already committed as `commitment`) at the
/// packed-direct `claims` via Ligerito. `poly` is consumed (Ligerito's prover
/// takes it by value).
fn open_v<C: Challenger>(
    poly: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    claims: &[PackedDirectClaim],
    ch: &mut C,
) -> pcs::BatchOpeningProofLigerito {
    let num_vars = commitment.params.log_msg_len();
    let padding = PaddingSpec::dense(commitment.params.m);
    let cfg = ligerito_prover_config(num_vars);
    pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        poly,
        prover_data,
        commitment,
        &[],
        &[],
        claims,
        &padding,
        &cfg,
        ch,
    )
}

/// Verify an [`open_v`] opening.
fn verify_v<C: Challenger>(
    commitment: &Commitment,
    claims: &[PackedDirectClaimRef<'_>],
    open: &pcs::BatchOpeningProofLigerito,
    ch: &mut C,
) -> Result<(), VerifyError> {
    let num_vars = commitment.params.log_msg_len();
    let cfg = ligerito_verifier_config(num_vars);
    pcs::verify_opening_batch_ligerito_mixed(commitment, &[], &[], &[], claims, open, &cfg, ch)
        .map_err(VerifyError::PcsOpen)
}

/// The five evaluation points of `v` (each length `μ+1`) that the PCS opens, in
/// the fixed order matching `[v_x0, v_x1, v_1x, v_0x, claimed_product]`:
/// `v(ρ,0)`, `v(ρ,1)`, `v(1,ρ)`, `v(0,ρ)`, and the product root `v[2N-2]`.
/// Low bit is bound first, so the leading coord is `b₀` and the trailing coord
/// is the half-selector. The root index `2N-2` has `b₀=0` and all other bits 1.
fn v_open_points(rho: &[F128]) -> [Vec<F128>; 5] {
    let mu = rho.len();
    let with_low = |bit: F128| {
        let mut p = Vec::with_capacity(mu + 1);
        p.push(bit);
        p.extend_from_slice(rho);
        p
    };
    let with_high = |bit: F128| {
        let mut p = rho.to_vec();
        p.push(bit);
        p
    };
    let mut root = vec![F128::ONE; mu + 1];
    root[0] = F128::ZERO;
    [
        with_low(F128::ZERO),  // v(ρ, 0)
        with_low(F128::ONE),   // v(ρ, 1)
        with_high(F128::ONE),  // v(1, ρ)
        with_high(F128::ZERO), // v(0, ρ)
        root,                  // v[2N-2]
    ]
}

// ---------------------------------------------------------------------------
// Batched zerocheck round message
// ---------------------------------------------------------------------------

/// Core of one eq-weighted sumcheck round message for the batched relation
/// `F = F_A + α·F_B`, **excluding** the affine `s_id` contribution to `G(1)`
/// (the prover adds that in closed form — see [`prove`]). Returns `(G(1)_core,
/// G(∞))`. `r_remaining` (the eq weights for the not-yet-bound variables) is
/// supplied **split** as `eq = eq_lo ⊗ eq_hi` ([`SplitEqGhash`]): the inner loop
/// weights by `eq_lo[x_lo]`, the outer block by `eq_hi[x_hi]`, so only
/// `2^n_lo + 2^n_hi` eq entries are built instead of the full `2^(n_lo+n_hi)`.
/// The current variable's eq factor is applied by the verifier (Convention A).
/// Low-bit binding: index `2x'` is `(·,0)`, `2x'+1` is `(·,1)`. `v0` is the leaf
/// view `v(0,·)` (first half of `v`), used by relation A.
#[allow(clippy::too_many_arguments)]
fn round_message(
    f: &[F128],
    g: &[F128],
    s_sig: &[F128],
    a: &[F128],
    b: &[F128],
    c: &[F128],
    v0: &[F128],
    beta: F128,
    gamma: F128,
    alpha: F128,
    eq: &SplitEqGhash,
) -> (F128, F128) {
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len(); // 2^n_lo  (x_lo per x_hi)
    let n_blocks = hi.len(); // 2^n_hi
    debug_assert_eq!(block * n_blocks, f.len() / 2);

    // One outer block (fixed `x_hi`): inner sum weighted by `eq_lo`, then scaled
    // once by `eq_hi[x_hi]`. `p1` omits `β·s_id[i1]` (added in closed form later);
    // `γ` is kept here (an XOR, and its eq-sum is handled the same way).
    let block_fn = |x_hi: usize| -> (F128, F128) {
        let x_base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let xp = x_base + x_lo;
            let (i0, i1) = (2 * xp, 2 * xp + 1);

            let p1 = f[i1] + gamma;
            let q0 = g[i0] + beta * s_sig[i0] + gamma;
            let q1 = g[i1] + beta * s_sig[i1] + gamma;

            // A: v(0,·)·q ⊕ p  (quadratic leaf·q, linear p).
            let (l0, l1) = (v0[i0], v0[i1]);
            let ga1 = l1 * q1 + p1;
            let ga_inf = (l0 + l1) * (q0 + q1);

            // B: c ⊕ a·b  (quadratic a·b, linear c).
            let (a0, a1, b0, b1) = (a[i0], a[i1], b[i0], b[i1]);
            let gb1 = c[i1] + a1 * b1;
            let gb_inf = (a0 + a1) * (b0 + b1);

            let el = lo[x_lo];
            s1 += el * (ga1 + alpha * gb1);
            s_inf += el * (ga_inf + alpha * gb_inf);
        }
        let eh = hi[x_hi];
        (eh * s1, eh * s_inf)
    };

    // Parallelize over the (≤128) outer blocks for big rounds; serial otherwise.
    if block * n_blocks >= PAR_THRESHOLD {
        (0..n_blocks).into_par_iter().map(block_fn).reduce(
            || (F128::ZERO, F128::ZERO),
            |(o0, i0), (o1, i1)| (o0 + o1, i0 + i1),
        )
    } else {
        let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
        for x_hi in 0..n_blocks {
            let (o, i) = block_fn(x_hi);
            g_one += o;
            g_inf += i;
        }
        (g_one, g_inf)
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove that `f, g` are related by `σ` (multiset `{(f,s_id)} = {(g,s_σ)}` with
/// `s_σ(x) = s_id(σ(x))`). `f.len() == g.len() == σ.len() == 2^μ`; `σ` must be a
/// permutation of `[0, 2^μ)`. The caller must have absorbed `f, g, σ` into `ch`.
pub fn prove<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    ch: &mut C,
) -> (PermutationProof, PermutationClaim) {
    let n = f.len();
    assert_eq!(g.len(), n);
    assert_eq!(sigma.len(), n);
    assert!(n.is_power_of_two() && n >= 2, "need N = 2^μ ≥ 2");
    let mu = n.trailing_zeros() as usize;

    // Phase timing (set PERM_TRACE=1). `tp` returns ms since the last reset.
    let trace = std::env::var("PERM_TRACE").is_ok();
    let mut t = std::time::Instant::now();
    let mut tp = |label: &str| {
        if trace {
            eprintln!(
                "  [perm-prove] {label:<14} {:8.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            t = std::time::Instant::now();
        }
    };

    ch.observe_label(DOMAIN);
    let beta = ch.sample_f128();
    let gamma = ch.sample_f128();

    // Tags and the fractional leaves. `s_id_vec[x]` (the field element whose bit
    // pattern is `x`) is built by doubling in O(N) — each entry is one XOR from a
    // lower one — instead of O(N·μ) per-bit recomputation. `s_σ` is then a gather:
    // `s_sig_vec[x] = s_id(σ(x)) = s_id_vec[σ(x)]`.
    let basis = s_id_basis(mu);
    let mut s_id_vec = vec![F128::ZERO; n];
    for (k, &bk) in basis.iter().enumerate() {
        let half = 1usize << k;
        let (lo, hi) = s_id_vec.split_at_mut(half);
        // Big levels parallelize; tiny early ones stay serial (dispatch overhead).
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
    let s_sig_vec: Vec<F128> = sigma.par_iter().map(|&sx| s_id_vec[sx]).collect();
    let p: Vec<F128> = f
        .par_iter()
        .zip(&s_id_vec)
        .map(|(fx, sx)| *fx + beta * *sx + gamma)
        .collect();
    let q: Vec<F128> = g
        .par_iter()
        .zip(&s_sig_vec)
        .map(|(gx, sx)| *gx + beta * *sx + gamma)
        .collect();
    let q_inv = batch_inverse(&q);
    let leaves: Vec<F128> = p.par_iter().zip(&q_inv).map(|(px, qx)| *px * *qx).collect();

    // Grand-product tree over the leaves and its derived views. The first half
    // of `v` IS the leaves (`v(0,x) = ℓ(x)`), so no separate `h` is committed.
    let v = build_grand_product(&leaves);
    let a: Vec<F128> = (0..n).into_par_iter().map(|i| v[2 * i]).collect();
    let b: Vec<F128> = (0..n).into_par_iter().map(|i| v[2 * i + 1]).collect();
    let c: Vec<F128> = v[n..2 * n].to_vec(); // c[i] = v[n+i]
    let v0: Vec<F128> = v[..n].to_vec();
    let claimed_product = v[2 * n - 2];
    tp("witness+v");

    // Commit to the single aux poly `v` (μ+1 vars) and observe its root —
    // binding `v` before the zerocheck challenges (a known `ρ` would break
    // zerocheck soundness).
    let params_v = pcs_params(mu + 1);
    let (commitment_v, pdata_v) = commit(&v, &params_v);
    tp("commit(v)");
    ch.observe_bytes(&commitment_v.root);
    ch.observe_f128(claimed_product);
    let alpha = ch.sample_f128();
    let r = ch.sample_f128_vec(mu);

    // `s_id` is affine, so it is NOT folded as a working vector: its eq-weighted
    // contribution to each round's `G(1)` has the closed form
    // `β·((C_i + basis_i) + Σ_{k>i} basis_k·r_k)`, where `C_i = Σ_{k<i} basis_k·ρ_k`
    // (running). Precompute the suffix sums `S[i] = Σ_{k≥i} basis_k·r_k`.
    let mut sid_suffix = vec![F128::ZERO; mu + 1];
    for i in (0..mu).rev() {
        sid_suffix[i] = basis[i] * r[i] + sid_suffix[i + 1];
    }

    // Working (mutable) copies that get folded each round (7 vectors; s_id excluded).
    let (mut wf, mut wg) = (f.to_vec(), g.to_vec());
    let mut wssig = s_sig_vec;
    let (mut wa, mut wb, mut wc, mut wv0) = (a, b, c, v0);

    let mut rounds = Vec::with_capacity(mu);
    let mut rho = Vec::with_capacity(mu);
    let mut c_prefix = F128::ZERO; // Σ_{k<i} basis_k·ρ_k
    for i in 0..mu {
        let eq = SplitEqGhash::new(&r[i + 1..mu]);
        let (g1_core, g_inf) = round_message(
            &wf, &wg, &wssig, &wa, &wb, &wc, &wv0, beta, gamma, alpha, &eq,
        );
        // Add the affine `s_id` contribution to `G(1)` in closed form.
        let g1 = g1_core + beta * ((c_prefix + basis[i]) + sid_suffix[i + 1]);
        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho_i = ch.sample_f128();
        rho.push(rho_i);
        rounds.push((g1, g_inf));
        c_prefix += basis[i] * rho_i;

        for u in [
            &mut wf, &mut wg, &mut wssig, &mut wa, &mut wb, &mut wc, &mut wv0,
        ] {
            fold_in_place(u, rho_i);
        }
    }
    debug_assert_eq!(c_prefix, s_id_eval(&basis, &rho));
    tp("sumcheck");

    let (f_eval, g_eval, s_sigma_eval) = (wf[0], wg[0], wssig[0]);
    let (v_x0, v_x1, v_1x, v_0x) = (wa[0], wb[0], wc[0], wv0[0]);

    // Bind the evals (so a downstream witness opening is sound), then open `v`
    // at the reduction point.
    observe_evals(ch, &[f_eval, g_eval, s_sigma_eval, v_x0, v_x1, v_1x, v_0x]);

    // `v` at `(ρ,0), (ρ,1), (1,ρ), (0,ρ)` and the product root — five claims.
    let v_points = v_open_points(&rho);
    let v_values = [v_x0, v_x1, v_1x, v_0x, claimed_product];
    let v_claims: Vec<PackedDirectClaim> = v_points
        .iter()
        .zip(v_values)
        .map(|(point, value)| PackedDirectClaim {
            point: point.clone(),
            value,
            eq_ind: DirectEqInd::Dense(build_eq(point)),
        })
        .collect();
    let v_open = open_v(v, &pdata_v, &commitment_v, &v_claims, ch);
    tp("open(v)");

    let proof = PermutationProof {
        v_root: commitment_v.root,
        claimed_product,
        rounds,
        f_eval,
        g_eval,
        s_sigma_eval,
        v_x0,
        v_x1,
        v_1x,
        v_0x,
        v_open,
    };

    let claim = PermutationClaim {
        rho,
        f_eval,
        g_eval,
        s_sigma_eval,
        v_x0,
        v_x1,
        v_1x,
        v_0x,
    };
    (proof, claim)
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify a permutation proof for `N = 2^mu`. The caller must have absorbed the
/// same `f, g, σ` binding into `ch` as the prover did.
pub fn verify<C: Challenger>(
    mu: usize,
    proof: &PermutationProof,
    ch: &mut C,
) -> Result<PermutationClaim, VerifyError> {
    assert_eq!(proof.rounds.len(), mu);

    ch.observe_label(DOMAIN);
    let beta = ch.sample_f128();
    let gamma = ch.sample_f128();

    // Rebuild the `v` commitment from the proof root + params derived from μ, and
    // observe the root at the same transcript position the prover committed.
    let commitment_v = Commitment {
        root: proof.v_root,
        params: pcs_params(mu + 1),
    };
    ch.observe_bytes(&commitment_v.root);
    ch.observe_f128(proof.claimed_product);
    let alpha = ch.sample_f128();
    let r = ch.sample_f128_vec(mu);

    // eq-trick sumcheck chain (mirrors zerocheck.rs): running claim is the bare
    // inner value, ending at F(ρ). Initial zerocheck target is 0.
    let mut c_running = F128::ZERO;
    let mut rho = Vec::with_capacity(mu);
    for i in 0..mu {
        let (g1, g_inf) = proof.rounds[i];
        let r_eq = r[i];
        let one_plus_r_eq = F128::ONE + r_eq;
        // G(0) from consistency c_running = (1+r_eq)·G(0) + r_eq·G(1).
        let g0 = (c_running + r_eq * g1) * one_plus_r_eq.inv();

        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho_i = ch.sample_f128();
        rho.push(rho_i);

        let one_plus_rho = F128::ONE + rho_i;
        // G(ρ) = G(0)(1+ρ) + G(1)ρ + G(∞)ρ(1+ρ).
        c_running = g0 * one_plus_rho + g1 * rho_i + g_inf * rho_i * one_plus_rho;
    }

    // Final consistency: c_running must equal F(ρ) (bare, no eq factor).
    // Relation A uses `v(0,ρ)` (= v_0x) as the leaf value; there is no separate h.
    let basis = s_id_basis(mu);
    let s_id_rho = s_id_eval(&basis, &rho);
    let p_rho = proof.f_eval + beta * s_id_rho + gamma;
    let q_rho = proof.g_eval + beta * proof.s_sigma_eval + gamma;
    let f_a = proof.v_0x * q_rho + p_rho;
    let f_b = proof.v_1x + proof.v_x0 * proof.v_x1;
    let expected = f_a + alpha * f_b;
    if c_running != expected {
        return Err(VerifyError::SumcheckFinalFailed);
    }
    if proof.claimed_product != F128::ONE {
        return Err(VerifyError::RootNotOne);
    }

    // Bind the evals, then verify the PCS opening of `v` at the reduction point —
    // same transcript order as the prover.
    observe_evals(
        ch,
        &[
            proof.f_eval,
            proof.g_eval,
            proof.s_sigma_eval,
            proof.v_x0,
            proof.v_x1,
            proof.v_1x,
            proof.v_0x,
        ],
    );

    let v_points = v_open_points(&rho);
    let v_values = [
        proof.v_x0,
        proof.v_x1,
        proof.v_1x,
        proof.v_0x,
        proof.claimed_product,
    ];
    let v_refs: Vec<PackedDirectClaimRef<'_>> = v_points
        .iter()
        .zip(v_values)
        .map(|(point, value)| PackedDirectClaimRef { point, value })
        .collect();
    verify_v(&commitment_v, &v_refs, &proof.v_open, ch)?;

    Ok(PermutationClaim {
        rho,
        f_eval: proof.f_eval,
        g_eval: proof.g_eval,
        s_sigma_eval: proof.s_sigma_eval,
        v_x0: proof.v_x0,
        v_x1: proof.v_x1,
        v_1x: proof.v_1x,
        v_0x: proof.v_0x,
    })
}

fn observe_evals<C: Challenger>(ch: &mut C, evals: &[F128; 7]) {
    for e in evals {
        ch.observe_f128(*e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    // SplitMix64, matching the repo's test RNG convention.
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

    /// Build an honest instance: random `g`, permutation `σ`, and
    /// `f(x) = g(σ⁻¹(x))` so the multiset `{(f,s_id)} = {(g,s_σ)}` holds.
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

    fn run_prove(f: &[F128], g: &[F128], sigma: &[usize]) -> (PermutationProof, PermutationClaim) {
        let mut ch = FsChallenger::new(b"perm-test");
        bind(&mut ch, f, g, sigma);
        prove(f, g, sigma, &mut ch)
    }

    fn run_verify(
        mu: usize,
        f: &[F128],
        g: &[F128],
        sigma: &[usize],
        proof: &PermutationProof,
    ) -> Result<PermutationClaim, VerifyError> {
        let mut ch = FsChallenger::new(b"perm-test");
        bind(&mut ch, f, g, sigma);
        verify(mu, proof, &mut ch)
    }

    #[test]
    fn honest_roundtrip_and_claim_match() {
        // μ ≥ 7 is the Ligerito floor for `v` (log_n = μ+1 ≥ 8).
        for mu in 7..=8 {
            let (f, g, sigma) = honest_instance(mu, 0xC0FFEE ^ mu as u64);
            let (proof, claim_p) = run_prove(&f, &g, &sigma);
            assert_eq!(proof.claimed_product, F128::ONE, "μ={mu}: ∏ℓ ≠ 1");
            let claim_v = run_verify(mu, &f, &g, &sigma, &proof).expect("verify");
            assert_eq!(claim_p, claim_v, "μ={mu}: prover/verifier claim mismatch");
        }
    }

    #[test]
    fn claim_matches_direct_mle() {
        let mu = 7;
        let (f, g, sigma) = honest_instance(mu, 0xABCD);
        let (_proof, claim) = run_prove(&f, &g, &sigma);

        // Rebuild the witness-derived polys to evaluate their MLEs directly.
        let mut ch = FsChallenger::new(b"perm-test");
        bind(&mut ch, &f, &g, &sigma);
        ch.observe_label(DOMAIN);
        let beta = ch.sample_f128();
        let gamma = ch.sample_f128();
        let n = 1usize << mu;
        let basis = s_id_basis(mu);
        let s_id_vec: Vec<F128> = (0..n).map(|x| s_id_value(x, &basis)).collect();
        let s_sig_vec: Vec<F128> = (0..n).map(|x| s_id_value(sigma[x], &basis)).collect();
        let p: Vec<F128> = (0..n).map(|x| f[x] + beta * s_id_vec[x] + gamma).collect();
        let q: Vec<F128> = (0..n).map(|x| g[x] + beta * s_sig_vec[x] + gamma).collect();
        let q_inv = batch_inverse(&q);
        let leaves: Vec<F128> = (0..n).map(|x| p[x] * q_inv[x]).collect();
        let v = build_grand_product(&leaves);

        let rho = &claim.rho;
        assert_eq!(claim.f_eval, mle_eval(&f, rho));
        assert_eq!(claim.g_eval, mle_eval(&g, rho));
        assert_eq!(claim.s_sigma_eval, mle_eval(&s_sig_vec, rho));
        // The leaf value ℓ(ρ) lives inside v as v(0,ρ) = v_0x (checked below).
        assert_eq!(claim.v_0x, mle_eval(&leaves, rho));

        // v evaluations: low bit selects (·,0)/(·,1); high bit selects (0,·)/(1,·).
        let mut pt_x0 = vec![F128::ZERO];
        pt_x0.extend_from_slice(rho);
        let mut pt_x1 = vec![F128::ONE];
        pt_x1.extend_from_slice(rho);
        let mut pt_1x = rho.clone();
        pt_1x.push(F128::ONE);
        let mut pt_0x = rho.clone();
        pt_0x.push(F128::ZERO);
        assert_eq!(claim.v_x0, mle_eval(&v, &pt_x0));
        assert_eq!(claim.v_x1, mle_eval(&v, &pt_x1));
        assert_eq!(claim.v_1x, mle_eval(&v, &pt_1x));
        assert_eq!(claim.v_0x, mle_eval(&v, &pt_0x));
        assert_eq!(v[2 * n - 2], F128::ONE);
    }

    #[test]
    fn tampered_witness_rejected() {
        let mu = 7;
        let (f, g, sigma) = honest_instance(mu, 0x1234);
        let (proof, _) = run_prove(&f, &g, &sigma);

        // Verifier binds a different f (flip one entry): challenges diverge, so
        // the sumcheck final consistency fails.
        let mut f_bad = f.clone();
        f_bad[3] += F128::ONE;
        let res = run_verify(mu, &f_bad, &g, &sigma, &proof);
        assert!(
            matches!(res, Err(VerifyError::SumcheckFinalFailed)),
            "got {res:?}"
        );
    }

    #[test]
    fn non_permutation_relation_rejected() {
        // f is NOT a permutation of g ⇒ honest prover's ∏h ≠ 1 ⇒ RootNotOne.
        let mu = 7;
        let n = 1usize << mu;
        let mut rng = Rng::new(0x9999);
        let g: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let f: Vec<F128> = (0..n).map(|_| rng.f128()).collect(); // unrelated
        let sigma: Vec<usize> = (0..n).collect(); // identity tag
        let (proof, _) = run_prove(&f, &g, &sigma);
        assert_ne!(proof.claimed_product, F128::ONE, "expected ∏h ≠ 1");
        let res = run_verify(mu, &f, &g, &sigma, &proof);
        assert_eq!(res, Err(VerifyError::RootNotOne));
    }

    #[test]
    fn tampered_pcs_opening_rejected() {
        // Corrupt the opening's sumcheck transcript: the permutation sumcheck
        // and root checks still pass (evals + claimed_product are untouched),
        // so the rejection comes purely from the PCS opening no longer
        // matching the committed `v`.
        let mu = 7;
        let (f, g, sigma) = honest_instance(mu, 0x2468);
        let (mut proof, _) = run_prove(&f, &g, &sigma);
        proof.v_open.ligerito.sumcheck_transcript[0].u_0.lo ^= 1;
        let res = run_verify(mu, &f, &g, &sigma, &proof);
        assert!(matches!(res, Err(VerifyError::PcsOpen(_))), "got {res:?}");
    }
}
