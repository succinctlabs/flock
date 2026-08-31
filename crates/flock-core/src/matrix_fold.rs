//! Accumulation for the lincheck's matrix claims.
//!
//! The lincheck verifier's whole dependence on a base matrix is one bilinear
//! form (`lincheck::MatrixAssertion`) — `O(nnz)`, ~21M nonzeros for BLAKE3,
//! 84% of verify. Discharging it inline is fine natively and fatal in a
//! recursion circuit: arithmetising it is **nnz-preserving** (the gadget's
//! matrix *is* the matrix it evaluates), so a circuit that pays it inline can
//! never verify a proof of itself — the fixed point `nnz(gadget) =
//! Σ_t nnz(table_t) + nnz(gadget)` has no solution.
//!
//! So the circuit does not evaluate the claim: it emits it, and claims are
//! **folded** instead. This module is that fold. The asymmetry it buys is the
//! whole point:
//!
//! * the fold's **prover** pays `O(nnz)` — natively, outside any circuit;
//! * the fold's **verifier** pays `O(κ)` — two sumcheck replays and a handful
//!   of weight evaluations, which is what a circuit replays.
//!
//! Nothing here commits to the matrix. The claims are about a *public,
//! registry-static* polynomial, so the accumulated claim at the root of an
//! aggregation tree is discharged by [`MatrixClaim::check_direct`] in the
//! clear — once, amortised over the tree — and no sparse-matrix commitment,
//! preprocessing or verification key is needed anywhere.
//!
//! ## The reduction
//!
//! A claim is `Σ_{r,c} row(r)·col(c)·M[r,c] = v` for structured weights. `k`
//! claims about one `M` fold to one in two sumchecks:
//!
//! 1. **Column.** With `comb_i(c) = Σ_r row_i(r)·M[r,c]`, each claim reads
//!    `Σ_c col_i(c)·comb_i(c) = v_i`. A degree-2 sumcheck on the
//!    λ-combination binds every claim to one shared `ρ_col`; the prover then
//!    sends the `k` values `comb_i(ρ_col)`.
//! 2. **Row.** Because `ρ_col` is now shared, every `comb_i(ρ_col) =
//!    Σ_r row_i(r)·h(r)` reads against the *same* `h(r) = M̂(r, ρ_col)` — so
//!    the row side collapses to a single sumcheck on `Σ_i μ_i row_i`,
//!    landing on `h(ρ_row) = M̂(ρ_row, ρ_col)`.
//!
//! The order is deliberate and is the difference between a fast fold and a
//! slow one. The `k` per-claim marginals are the dominant `k · nnz` cost,
//! and in THIS order they are **column** marginals — which is exactly what
//! [`crate::lincheck::LincheckCircuit::fold_split`] computes, with each
//! type's tuned walker/CSC kernel. Only the single shared marginal of step 2
//! runs against the raw matrix. Ordered the other way the k-fold work would
//! be row marginals, which no tuned kernel provides.
//!
//! The output is a plain evaluation claim, which folds again by the same
//! machinery (a plain evaluation is just a claim whose weights are `eq`
//! tensors). Fresh μ is what binds the individual `g_i(ρ_row)`: a prover who
//! reports them λ-consistently but individually wrong shifts the column
//! sumcheck's target, and the error lands in the output claim — where the
//! root discharge catches it. That is the accumulation property: **if any
//! input claim is false, the output claim is false with overwhelming
//! probability.**
//!
//! Sumcheck conventions match the rest of the codebase: Convention A messages
//! `(q(1), q(∞))` with `q(0)` re-derived from the running claim, and each
//! round binds the LOW remaining variable.

use crate::element_r1cs::SparseF128Matrix;
use crate::pcs::jagged::JaggedParams;
use crate::pcs::jagged::assist_boundaries;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::challenger::Challenger;
use crate::field::F128;
use crate::r1cs::SparseBinaryMatrix;

const DOMAIN: &[u8] = b"flock-matrix-fold-v0";

/// Rows per parallel block in the two matrix passes. Large enough that the
/// per-block accumulator (one `2^k_col` vector in the scatter pass) is
/// amortised, small enough to keep every worker fed.
const G_ROW_CHUNK: usize = 1 << 10;

/// A structured weight over `2^k` entries: `low ⊗ eq(point)`, with `low`
/// occupying the LOW `log2(low.len())` coordinates.
///
/// Both shapes the lincheck produces are instances — its row weight
/// `λ(z_skip) ⊗ eq(x_inner_rest)` and its column weight
/// `z_partial ⊗ eq(rr)` — as is a plain `eq(point)` (`low = [1]`), which is
/// what a folded claim carries. The verifier evaluates one in `O(2^s + k)`
/// where `2^s = low.len()` is 64 in practice; the prover materialises it in
/// `O(2^k)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Weight {
    /// Explicit factor on the low coordinates; length a power of two.
    pub low: Vec<F128>,
    /// `eq`-tensor point for the remaining high coordinates.
    pub point: Vec<F128>,
}

impl Weight {
    /// A plain `eq(point, ·)` weight — the shape a folded claim carries.
    pub fn eq(point: Vec<F128>) -> Self {
        Self {
            low: vec![F128::ONE],
            point,
        }
    }

    /// `low ⊗ eq(point, ·)`; `low.len()` must be a power of two.
    pub fn low_eq(low: Vec<F128>, point: Vec<F128>) -> Self {
        assert!(
            low.len().is_power_of_two(),
            "the low factor must span whole coordinates"
        );
        Self { low, point }
    }

    /// Number of coordinates this weight spans.
    pub fn n_vars(&self) -> usize {
        self.low.len().trailing_zeros() as usize + self.point.len()
    }

    /// The weight's MLE at `rho` (LSB-first, `rho.len() == n_vars()`).
    pub fn eval(&self, rho: &[F128]) -> F128 {
        assert_eq!(rho.len(), self.n_vars(), "point/weight arity mismatch");
        let s = self.low.len().trailing_zeros() as usize;
        // The low factor is an arbitrary vector: its MLE is a fold, O(2^s).
        let mut acc = self.low.clone();
        for &r in &rho[..s] {
            let half = acc.len() / 2;
            for x in 0..half {
                acc[x] = acc[2 * x] * (F128::ONE + r) + acc[2 * x + 1] * r;
            }
            acc.truncate(half);
        }
        let mut out = acc[0];
        for (&p, &r) in self.point.iter().zip(&rho[s..]) {
            out *= p * r + (F128::ONE + p) * (F128::ONE + r);
        }
        out
    }

    /// The full `2^k` vector — prover side only.
    pub fn materialize(&self) -> Vec<F128> {
        let s = self.low.len().trailing_zeros() as usize;
        let mut out = vec![F128::ZERO; 1usize << self.n_vars()];
        // eq table over the high coordinates, built by doubling.
        let mut eq = vec![F128::ONE];
        for &p in &self.point {
            let mut next = vec![F128::ZERO; eq.len() * 2];
            for (i, &e) in eq.iter().enumerate() {
                next[i] = e * (F128::ONE + p);
                next[i + eq.len()] = e * p;
            }
            eq = next;
        }
        for (hi, &e) in eq.iter().enumerate() {
            for (lo, &l) in self.low.iter().enumerate() {
                out[(hi << s) | lo] = e * l;
            }
        }
        out
    }
}

/// `Σ_{r,c} row(r)·col(c)·M[r,c] = value` for one base matrix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixClaim {
    pub row: Weight,
    pub col: Weight,
    pub value: F128,
}

/// `Σ_{r,c} row(r)·col(c)·M[r,c]` — one pass over the nonzeros, `O(nnz)`.
/// The honest value of a claim, and the root discharge.
pub fn bilinear(row: &Weight, col: &Weight, m: &dyn FoldMatrix) -> F128 {
    let rw = row.materialize();
    let cw = col.materialize();
    assert!(
        rw.len() >= m.n_rows() && cw.len() >= m.n_cols(),
        "weights are too small for the matrix"
    );
    // ⟨row ⊗ col, M⟩ = ⟨row, M·col⟩ — one marginal, then a dot product.
    let g = m.row_marginal(&cw, rw.len());
    rw.iter().zip(&g).fold(F128::ZERO, |a, (x, y)| a + *x * *y)
}

impl MatrixClaim {
    /// An honest claim about `m` at the given weights.
    pub fn honest(row: Weight, col: Weight, m: &dyn FoldMatrix) -> Self {
        let value = bilinear(&row, &col, m);
        Self { row, col, value }
    }

    /// Discharge the claim directly — the `O(nnz)` root check.
    pub fn check_direct(&self, m: &dyn FoldMatrix) -> bool {
        bilinear(&self.row, &self.col, m) == self.value
    }
}

/// Transcript of one fold: the two sumchecks' Convention-A round messages,
/// the `k` bridge values `g_i(ρ_row)`, and the output evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldProof {
    /// Column phase (first): binds every claim to a shared `ρ_col`.
    pub col_rounds: Vec<(F128, F128)>,
    /// `comb_i(ρ_col)` per claim — what lets the row phase collapse to one.
    pub bridge: Vec<F128>,
    /// Row phase (second): one sumcheck against the shared marginal.
    pub row_rounds: Vec<(F128, F128)>,
    pub value: F128,
    /// PoW witnesses in transcript order: lambda-vector, column rounds,
    /// mu-vector, then row rounds.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// Fiat--Shamir grinding schedule shared by dense, sigma, and jagged folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldGrinding {
    /// One grind protects each whole vector of independent linear batching
    /// coefficients (lambda, then mu).
    pub combination_bits: u32,
    /// Each column/row sumcheck challenge evaluates a quadratic polynomial.
    pub round_bits: u32,
}

impl FoldGrinding {
    pub const fn disabled() -> Self {
        Self {
            combination_bits: 0,
            round_bits: 0,
        }
    }

    pub const fn per_challenge_128() -> Self {
        Self {
            combination_bits: 1,
            round_bits: 2,
        }
    }

    #[inline]
    fn nonce_count(self, k_row: usize, k_col: usize) -> usize {
        2 * usize::from(self.combination_bits != 0)
            + (k_row + k_col) * usize::from(self.round_bits != 0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoldError {
    /// No claims to fold, or claims of inconsistent arity.
    Malformed,
    /// A sumcheck's final value disagreed with the claimed factors.
    ConsistencyFailed { which: &'static str },
    /// Grinding witnesses did not have the configured shape or failed PoW.
    InvalidGrinding,
}

/// One degree-2 round over `k` pairs, λ-combined: returns `(q(1), q(∞))` for
/// `Σ_i λ_i·a_i[x]·b_i[x]`, binding the LOW variable.
fn round_message(pairs: &[(Vec<F128>, Vec<F128>)], lambdas: &[F128]) -> (F128, F128) {
    let (mut q1, mut qinf) = (F128::ZERO, F128::ZERO);
    for ((a, b), &lam) in pairs.iter().zip(lambdas) {
        let half = a.len() / 2;
        let (mut s1, mut sinf) = (F128::ZERO, F128::ZERO);
        for x in 0..half {
            let (a0, a1) = (a[2 * x], a[2 * x + 1]);
            let (b0, b1) = (b[2 * x], b[2 * x + 1]);
            s1 += a1 * b1;
            sinf += (a0 + a1) * (b0 + b1);
        }
        q1 += lam * s1;
        qinf += lam * sinf;
    }
    (q1, qinf)
}

/// Bind the LOW variable of `v` at `r`, halving it.
fn fold_low(v: &mut Vec<F128>, r: F128) {
    let half = v.len() / 2;
    let one_minus = F128::ONE + r;
    for x in 0..half {
        v[x] = v[2 * x] * one_minus + v[2 * x + 1] * r;
    }
    v.truncate(half);
}

/// Replay a Convention-A degree-2 sumcheck: `q(0)` comes from the running
/// claim, so each round is two field elements on the wire.
fn replay_rounds<Ch: Challenger>(
    rounds: &[(F128, F128)],
    mut running: F128,
    proof_nonces: &[u64],
    nonce_idx: &mut usize,
    round_bits: u32,
    ch: &mut Ch,
) -> Result<(F128, Vec<F128>), FoldError> {
    let mut rho = Vec::with_capacity(rounds.len());
    for &(q1, qinf) in rounds {
        ch.observe_f128(q1);
        ch.observe_f128(qinf);
        let r = if round_bits != 0 {
            let r = ch
                .verify_pow_and_sample_f128(proof_nonces[*nonce_idx], round_bits)
                .ok_or(FoldError::InvalidGrinding)?;
            *nonce_idx += 1;
            r
        } else {
            ch.sample_f128()
        };
        // char 2: q(0) = running + q(1); q(X) = qinf·X² + c1·X + q(0).
        let q0 = running + q1;
        let c1 = q0 + q1 + qinf;
        running = qinf * r * r + c1 * r + q0;
        rho.push(r);
    }
    Ok((running, rho))
}

#[inline]
fn grind_sample<Ch: Challenger>(ch: &mut Ch, nonces: &mut Vec<u64>, bits: u32) -> F128 {
    if bits != 0 {
        let (nonce, challenge) = ch.grind_pow_and_sample_f128(bits);
        nonces.push(nonce);
        challenge
    } else {
        ch.sample_f128()
    }
}

#[inline]
fn grind_sample_vec<Ch: Challenger>(
    ch: &mut Ch,
    nonces: &mut Vec<u64>,
    bits: u32,
    n: usize,
) -> Vec<F128> {
    if bits != 0 {
        let (nonce, challenge) = ch.grind_pow_and_sample_f128_vec(bits, n);
        nonces.push(nonce);
        challenge
    } else {
        ch.sample_f128_vec(n)
    }
}

#[inline]
fn verify_grind_sample_vec_n<Ch: Challenger>(
    ch: &mut Ch,
    nonces: &[u64],
    nonce_idx: &mut usize,
    bits: u32,
    n: usize,
) -> Result<Vec<F128>, FoldError> {
    if bits != 0 {
        let challenges = ch
            .verify_pow_and_sample_f128_vec(nonces[*nonce_idx], bits, n)
            .ok_or(FoldError::InvalidGrinding)?;
        *nonce_idx += 1;
        Ok(challenges)
    } else {
        Ok(ch.sample_f128_vec(n))
    }
}

/// Bind the claims into the transcript — weights included, not just values.
/// λ and μ must depend on everything being folded: two claim sets that agreed
/// only on their values would otherwise draw the same challenges.
fn observe_claims<Ch: Challenger>(claims: &[MatrixClaim], ch: &mut Ch) {
    ch.observe_label(DOMAIN);
    for c in claims {
        for w in [&c.row, &c.col] {
            ch.observe_f128_slice(&w.low);
            ch.observe_f128_slice(&w.point);
        }
        ch.observe_f128(c.value);
    }
}

/// Fold `k` claims about `m` into one plain evaluation claim.
///
/// `O(k·nnz)` — the only place the matrix is read, and it happens natively,
/// never inside a circuit. The caller must have bound whatever pins the
/// claims (registry digest, statement) into `ch` beforehand.
pub fn prove_fold<Ch: Challenger>(
    m: &dyn FoldMatrix,
    combs: &[Vec<F128>],
    claims: &[MatrixClaim],
    ch: &mut Ch,
) -> (FoldProof, MatrixClaim) {
    prove_fold_with_grinding(m, combs, claims, FoldGrinding::disabled(), ch)
}

/// [`prove_fold`] with an explicit Fiat--Shamir grinding policy.
pub fn prove_fold_with_grinding<Ch: Challenger>(
    m: &dyn FoldMatrix,
    combs: &[Vec<F128>],
    claims: &[MatrixClaim],
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> (FoldProof, MatrixClaim) {
    assert!(!claims.is_empty(), "nothing to fold");
    assert_eq!(combs.len(), claims.len(), "one column marginal per claim");
    let k_row = claims[0].row.n_vars();
    let k_col = claims[0].col.n_vars();
    for c in claims {
        assert_eq!(c.row.n_vars(), k_row, "claims must share the row arity");
        assert_eq!(c.col.n_vars(), k_col, "claims must share the column arity");
    }

    let mut grinding_nonces = Vec::with_capacity(grinding.nonce_count(k_row, k_col));
    observe_claims(claims, ch);
    let lambdas = grind_sample_vec(
        ch,
        &mut grinding_nonces,
        grinding.combination_bits,
        claims.len(),
    );

    // Column phase: Σ_c Σ_i λ_i·col_i(c)·comb_i(c). `combs` is the k·nnz
    // work, done by the caller with the type's tuned kernel.
    let mut pairs: Vec<(Vec<F128>, Vec<F128>)> = claims
        .iter()
        .zip(combs)
        .map(|(claim, comb)| {
            let mut g = comb.clone();
            g.resize(1usize << k_col, F128::ZERO);
            (claim.col.materialize(), g)
        })
        .collect();

    let mut col_rounds = Vec::with_capacity(k_col);
    let mut rho_col = Vec::with_capacity(k_col);
    for _ in 0..k_col {
        let msg = round_message(&pairs, &lambdas);
        ch.observe_f128(msg.0);
        ch.observe_f128(msg.1);
        let r = grind_sample(ch, &mut grinding_nonces, grinding.round_bits);
        col_rounds.push(msg);
        rho_col.push(r);
        for (a, b) in &mut pairs {
            fold_low(a, r);
            fold_low(b, r);
        }
    }
    let bridge: Vec<F128> = pairs.iter().map(|(_, g)| g[0]).collect();
    for &v in &bridge {
        ch.observe_f128(v);
    }

    // Row phase. Every bridge value reads against the same h(r) = M̂(r, ρ_col),
    // so ONE marginal serves all k — the only pass this function makes over
    // the matrix itself.
    let mus = grind_sample_vec(
        ch,
        &mut grinding_nonces,
        grinding.combination_bits,
        claims.len(),
    );
    let eq_col = Weight::eq(rho_col.clone()).materialize();
    let h = m.row_marginal(&eq_col, 1usize << k_row);
    let mut w_mu = vec![F128::ZERO; 1usize << k_row];
    for (claim, &mu) in claims.iter().zip(&mus) {
        for (dst, src) in w_mu.iter_mut().zip(claim.row.materialize()) {
            *dst += mu * src;
        }
    }

    let mut row_pairs = vec![(w_mu, h)];
    let one = [F128::ONE];
    let mut row_rounds = Vec::with_capacity(k_row);
    let mut rho_row = Vec::with_capacity(k_row);
    for _ in 0..k_row {
        let msg = round_message(&row_pairs, &one);
        ch.observe_f128(msg.0);
        ch.observe_f128(msg.1);
        let r = grind_sample(ch, &mut grinding_nonces, grinding.round_bits);
        row_rounds.push(msg);
        rho_row.push(r);
        for (a, b) in &mut row_pairs {
            fold_low(a, r);
            fold_low(b, r);
        }
    }
    let value = row_pairs[0].1[0];
    ch.observe_f128(value);

    (
        FoldProof {
            col_rounds,
            bridge,
            row_rounds,
            value,
            grinding_nonces,
        },
        MatrixClaim {
            row: Weight::eq(rho_row),
            col: Weight::eq(rho_col),
            value,
        },
    )
}

/// A matrix the fold can walk. Implemented for both constraint-system
/// flavours: the boolean class's `GF(2)` supports and the element class's
/// `F128`-coefficient matrices.
///
/// Only the two marginals are needed, in both directions — everything else
/// the fold does lives on `2^κ`-sized vectors.
pub trait FoldMatrix: Sync {
    /// `out[r] = Σ_c w[c]·M[r,c]`.
    fn row_marginal(&self, w: &[F128], n_rows: usize) -> Vec<F128>;
    /// `out[c] = Σ_r w[r]·M[r,c]`. For boolean types prefer
    /// [`crate::lincheck::LincheckCircuit::fold_split`], which is the same
    /// quantity through each type's tuned kernel.
    fn col_marginal(&self, w: &[F128], n_cols: usize) -> Vec<F128>;
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
}

/// A dense column-major matrix over a flat vector: `M[r, c] =
/// vals[(c << n_rows_log) + r]`. The shape sigma v2 route B folds — the
/// sigma table over the cell space (`mu = nu + c` bits, cell index
/// `(col << nu) | row`) reshaped so `s_sigma_hat(rho)` splits as a
/// MatrixClaim with `row = eq(rho[..nu])`, `col = eq(rho[nu..])`.
pub struct DenseMatrix {
    pub vals: Vec<F128>,
    pub n_rows_log: usize,
}

impl FoldMatrix for DenseMatrix {
    fn row_marginal(&self, w: &[F128], n_rows: usize) -> Vec<F128> {
        assert_eq!(n_rows, 1usize << self.n_rows_log);
        let mut out = vec![F128::ZERO; n_rows];
        for (c, &wc) in w.iter().enumerate() {
            let base = c << self.n_rows_log;
            for (r, slot) in out.iter_mut().enumerate() {
                *slot += wc * self.vals[base + r];
            }
        }
        out
    }
    fn col_marginal(&self, w: &[F128], n_cols: usize) -> Vec<F128> {
        let n_rows = 1usize << self.n_rows_log;
        assert_eq!(self.vals.len(), n_cols << self.n_rows_log);
        (0..n_cols)
            .map(|c| {
                let base = c << self.n_rows_log;
                (0..n_rows)
                    .map(|r| w[r] * self.vals[base + r])
                    .fold(F128::ZERO, |a, x| a + x)
            })
            .collect()
    }
    fn n_rows(&self) -> usize {
        1usize << self.n_rows_log
    }
    fn n_cols(&self) -> usize {
        self.vals.len() >> self.n_rows_log
    }
}

impl FoldMatrix for SparseBinaryMatrix {
    fn row_marginal(&self, w: &[F128], n_rows: usize) -> Vec<F128> {
        let mut out = vec![F128::ZERO; n_rows];
        out.par_chunks_mut(G_ROW_CHUNK)
            .enumerate()
            .for_each(|(blk, dst)| {
                let base = blk * G_ROW_CHUNK;
                for (j, slot) in dst.iter_mut().enumerate() {
                    let Some(cs) = self.rows.get(base + j) else {
                        break;
                    };
                    let mut acc = F128::ZERO;
                    for &c in cs {
                        acc += w[c];
                    }
                    *slot = acc;
                }
            });
        out
    }
    fn col_marginal(&self, w: &[F128], n_cols: usize) -> Vec<F128> {
        scatter_cols(
            self.rows.len(),
            n_cols,
            |r, f| {
                for &c in &self.rows[r] {
                    f(c, F128::ONE);
                }
            },
            w,
        )
    }
    fn n_rows(&self) -> usize {
        self.num_rows
    }
    fn n_cols(&self) -> usize {
        self.num_cols
    }
}

impl FoldMatrix for SparseF128Matrix {
    fn row_marginal(&self, w: &[F128], n_rows: usize) -> Vec<F128> {
        let mut out = vec![F128::ZERO; n_rows];
        out.par_chunks_mut(G_ROW_CHUNK)
            .enumerate()
            .for_each(|(blk, dst)| {
                let base = blk * G_ROW_CHUNK;
                for (j, slot) in dst.iter_mut().enumerate() {
                    let Some(cs) = self.rows.get(base + j) else {
                        break;
                    };
                    let mut acc = F128::ZERO;
                    for &(c, coeff) in cs {
                        acc += coeff * w[c];
                    }
                    *slot = acc;
                }
            });
        out
    }
    fn col_marginal(&self, w: &[F128], n_cols: usize) -> Vec<F128> {
        scatter_cols(
            self.rows.len(),
            n_cols,
            |r, f| {
                for &(c, coeff) in &self.rows[r] {
                    f(c, coeff);
                }
            },
            w,
        )
    }
    fn n_rows(&self) -> usize {
        self.num_rows
    }
    fn n_cols(&self) -> usize {
        self.num_cols
    }
}

/// Shared scatter for the column marginals: per-worker accumulators reduced
/// by XOR (addition is XOR, so the reduction is exact and order-free).
fn scatter_cols(
    n_rows: usize,
    n_cols: usize,
    each: impl Fn(usize, &mut dyn FnMut(usize, F128)) + Sync,
    w: &[F128],
) -> Vec<F128> {
    (0..n_rows)
        .into_par_iter()
        .with_min_len(G_ROW_CHUNK)
        .fold(
            || vec![F128::ZERO; n_cols],
            |mut acc, r| {
                let wr = w[r];
                if wr != F128::ZERO {
                    each(r, &mut |c, coeff| acc[c] += wr * coeff);
                }
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; n_cols],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += *y;
                }
                a
            },
        )
}

/// `out[c] = Σ_r w[r]·M[r,c]` — free function kept for existing callers.
pub fn col_marginal(m: &SparseBinaryMatrix, w: &[F128], n_cols: usize) -> Vec<F128> {
    FoldMatrix::col_marginal(m, w, n_cols)
}

/// Replay a fold. `O(k·κ)` — no matrix access at all, which is what lets a
/// circuit run this. Returns the accumulated claim; it is only as true as the
/// inputs were, so something must eventually discharge it
/// ([`MatrixClaim::check_direct`] at the root of the tree).
pub fn verify_fold<Ch: Challenger>(
    claims: &[MatrixClaim],
    proof: &FoldProof,
    ch: &mut Ch,
) -> Result<MatrixClaim, FoldError> {
    verify_fold_with_grinding(claims, proof, FoldGrinding::disabled(), ch)
}

/// [`verify_fold`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_fold_with_grinding<Ch: Challenger>(
    claims: &[MatrixClaim],
    proof: &FoldProof,
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<MatrixClaim, FoldError> {
    if claims.is_empty() || proof.bridge.len() != claims.len() {
        return Err(FoldError::Malformed);
    }
    let k_row = claims[0].row.n_vars();
    let k_col = claims[0].col.n_vars();
    if claims
        .iter()
        .any(|c| c.row.n_vars() != k_row || c.col.n_vars() != k_col)
        || proof.row_rounds.len() != k_row
        || proof.col_rounds.len() != k_col
    {
        return Err(FoldError::Malformed);
    }

    if proof.grinding_nonces.len() != grinding.nonce_count(k_row, k_col) {
        return Err(FoldError::InvalidGrinding);
    }
    let mut nonce_idx = 0usize;

    observe_claims(claims, ch);
    let lambdas = verify_grind_sample_vec_n(
        ch,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.combination_bits,
        claims.len(),
    )?;

    // Column sumcheck: target is the λ-combination of the claimed values.
    let target = claims
        .iter()
        .zip(&lambdas)
        .fold(F128::ZERO, |acc, (c, &l)| acc + l * c.value);
    let (running, rho_col) = replay_rounds(
        &proof.col_rounds,
        target,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.round_bits,
        ch,
    )?;
    for &v in &proof.bridge {
        ch.observe_f128(v);
    }
    let expect = claims
        .iter()
        .zip(&lambdas)
        .zip(&proof.bridge)
        .fold(F128::ZERO, |acc, ((c, &l), &g)| {
            acc + l * c.col.eval(&rho_col) * g
        });
    if running != expect {
        return Err(FoldError::ConsistencyFailed { which: "col" });
    }

    // Row sumcheck: target is the μ-combination of the bridge values.
    let mus = verify_grind_sample_vec_n(
        ch,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.combination_bits,
        claims.len(),
    )?;
    let target = proof
        .bridge
        .iter()
        .zip(&mus)
        .fold(F128::ZERO, |acc, (&g, &m)| acc + m * g);
    let (running, rho_row) = replay_rounds(
        &proof.row_rounds,
        target,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.round_bits,
        ch,
    )?;
    let w_mu = claims
        .iter()
        .zip(&mus)
        .fold(F128::ZERO, |acc, (c, &m)| acc + m * c.row.eval(&rho_row));
    if running != w_mu * proof.value {
        return Err(FoldError::ConsistencyFailed { which: "row" });
    }
    ch.observe_f128(proof.value);
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());

    Ok(MatrixClaim {
        row: Weight::eq(rho_row),
        col: Weight::eq(rho_col),
        value: proof.value,
    })
}

// ───────────────────────────────────────────────────────────────────────────
// The jagged layout table J and its structure-aware fold
// ───────────────────────────────────────────────────────────────────────────
//
// The assist verifier's count-dependent scalar `W(ρ) = Σ_y w_y ·
// Π_ℓ eq(t_{y-1}[ℓ], ρ_{c,ℓ})·eq(t_y[ℓ], ρ_{d,ℓ})` is a bilinear form on the
// LAYOUT: rows indexed by layout column `y`, columns by the interleaved
// boundary-pair space, content the cumulative heights alone. Deferring it
// gives the layout the same treatment the constraint matrices and the wiring
// sigma already get — a claim family folded down the tree and discharged once
// at the root, so no count reaches any circuit's structure.
//
// [`prove_fold`] cannot serve it: the pair space has `2(m+1)` variables
// (`2^44`-sized in real shapes) and the dense column phase would materialize
// it. J has at most one nonzero per row — at most `2^k` distinct pairs — so
// this fold walks the runs instead: a sparse column phase (per-round messages
// from the surviving nonzeros, the claims' eq tensors handled analytically)
// and a dense row phase over the small `2^k` side, reusing the round
// machinery above. Proof shape, transcript conventions and the output claim
// are [`FoldProof`]/[`MatrixClaim`] verbatim, so the folded claim inherits
// onward through either fold path.

const DOMAIN_JAGGED: &[u8] = b"flock-jagged-fold-v0";

/// The jagged layout as a matrix: row `y` (a column of the union's jagged
/// layout) has its single 1 at the INTERLEAVED boundary-pair index of
/// `(t_{y-1}, t_y)` — column-index bit `2ℓ` is `t_{y-1}[ℓ]`, bit `2ℓ+1` is
/// `t_y[ℓ]`, matching the assist point's `(c_0, d_0, c_1, d_1, …)` order.
/// Content is the cumulative heights alone, so the table is a constant of
/// the child circuit's SHAPE — same digest, same heights, same `J` — which
/// is what lets claims key by child digest and fold across a tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaggedTable {
    /// Distinct boundary pairs `(t_{y-1}, t_y)` with run lengths, in column
    /// order — [`crate::pcs::jagged::assist_boundaries`]'s output, covering
    /// all `2^k` columns including zero-height runs and the padded tail.
    pub bounds: Vec<(u64, u64, u32)>,
    /// `log2` of the number of layout columns — J's ROW space.
    pub k: usize,
    /// Boundary values span `0..=2^m`: `m+1` bits each, `2(m+1)` pair-space
    /// variables.
    pub m: usize,
}

impl JaggedTable {
    /// The table of a concrete layout. Everything is derived from
    /// `col_prefix_sums`, the same source the assist verifier reads.
    pub fn from_params(params: &JaggedParams) -> Self {
        let bounds = assist_boundaries(params);
        let covered: u64 = bounds.iter().map(|&(_, _, run)| run as u64).sum();
        assert_eq!(covered, 1u64 << params.k, "runs must cover every column");
        Self {
            bounds,
            k: params.k,
            m: params.m,
        }
    }

    /// Number of pair-space variables (J's column arity).
    pub fn n_col_vars(&self) -> usize {
        2 * (self.m + 1)
    }

    /// The interleaved pair index: bit `2ℓ` from `t_c`, bit `2ℓ+1` from
    /// `t_next`.
    fn pair_index(&self, t_c: u64, t_next: u64) -> u64 {
        let mut idx = 0u64;
        for l in 0..=self.m {
            idx |= ((t_c >> l) & 1) << (2 * l);
            idx |= ((t_next >> l) & 1) << (2 * l + 1);
        }
        idx
    }

    /// `Π_ℓ eq(t_c[ℓ], rho[2ℓ]) · eq(t_next[ℓ], rho[2ℓ+1])` — one pair's eq
    /// factor at a pair-space point.
    fn eq_at_pair(&self, t_c: u64, t_next: u64, rho: &[F128]) -> F128 {
        debug_assert_eq!(rho.len(), self.n_col_vars());
        let mut acc = F128::ONE;
        for l in 0..=self.m {
            let (rc, rd) = (rho[2 * l], rho[2 * l + 1]);
            acc *= if (t_c >> l) & 1 == 1 {
                rc
            } else {
                F128::ONE + rc
            };
            acc *= if (t_next >> l) & 1 == 1 {
                rd
            } else {
                F128::ONE + rd
            };
        }
        acc
    }
}

/// A fresh jagged claim's row weight — over the `2^k` layout columns.
///
/// [`Weight`]'s `low ⊗ eq(point)` cannot express a γ-combination of one-hots
/// at scattered addresses, which is exactly the shape a scalar group's
/// statement has; inherited (already-folded) claims are plain `eq` and use
/// [`JaggedRowWeight::Eq`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JaggedRowWeight {
    /// `scale · eq(point, ·)` — a general statement's `z_col` (scale `1`),
    /// and every inherited folded claim, whose scale is the accumulator
    /// entry's LIVE word.
    ///
    /// The SCALE is the jagged family's ZERO-CLAIM form (wall 3). The
    /// uniform folds get it for free — a `Weight`'s length-1 `low` already
    /// scales `eq` — but a jagged row weight had no such slot, so an
    /// inherited entry could only ever be live. With the scale, a dead
    /// entry (`scale = 0`, value `0`) is a claim that is TRUE about every
    /// layout and costs the fold nothing, which is what lets one spine
    /// node shape serve every level.
    Eq(F128, Vec<F128>),
    /// `Σ_j coeff_j · e(addr_j)` — a scalar group's γ-baked one-hot columns
    /// at build-time-constant addresses (registry-derived, not
    /// count-derived). Fresh only, so no scale: a Combo never rides in as
    /// an inherited claim.
    Combo(Vec<(F128, u32)>),
}

impl JaggedRowWeight {
    /// A live `eq(point, ·)` — scale `1`, the shape every FRESH claim has.
    pub fn eq(point: Vec<F128>) -> Self {
        Self::Eq(F128::ONE, point)
    }

    /// The weight's MLE at `rho` (`rho.len() == k`).
    pub fn eval(&self, rho: &[F128]) -> F128 {
        match self {
            Self::Eq(scale, point) => {
                assert_eq!(point.len(), rho.len(), "point/arity mismatch");
                point.iter().zip(rho).fold(*scale, |acc, (&p, &r)| {
                    acc * (p * r + (F128::ONE + p) * (F128::ONE + r))
                })
            }
            Self::Combo(terms) => terms.iter().fold(F128::ZERO, |acc, &(c, addr)| {
                let e = rho.iter().enumerate().fold(F128::ONE, |e, (l, &r)| {
                    e * if (addr >> l) & 1 == 1 {
                        r
                    } else {
                        F128::ONE + r
                    }
                });
                acc + c * e
            }),
        }
    }

    /// The full `2^k` vector — prover side only.
    pub fn materialize(&self, k: usize) -> Vec<F128> {
        match self {
            Self::Eq(scale, point) => {
                assert_eq!(point.len(), k, "point/arity mismatch");
                let mut out = Weight::eq(point.clone()).materialize();
                if *scale != F128::ONE {
                    for x in &mut out {
                        *x *= *scale;
                    }
                }
                out
            }
            Self::Combo(terms) => {
                let mut out = vec![F128::ZERO; 1usize << k];
                for &(c, addr) in terms {
                    out[addr as usize] += c;
                }
                out
            }
        }
    }

    /// Canonical transcript binding: a tagged header, then the payload.
    fn observe<Ch: Challenger>(&self, ch: &mut Ch) {
        match self {
            Self::Eq(scale, point) => {
                ch.observe_f128(F128::new(0, point.len() as u64));
                ch.observe_f128(*scale);
                ch.observe_f128_slice(point);
            }
            Self::Combo(terms) => {
                ch.observe_f128(F128::new(1, terms.len() as u64));
                for &(c, addr) in terms {
                    ch.observe_f128(c);
                    ch.observe_f128(F128::new(addr as u64, 0));
                }
            }
        }
    }

    /// `true` when every address fits the row space — [`JaggedRowWeight::Eq`]
    /// checks arity instead.
    fn well_formed(&self, k: usize) -> bool {
        match self {
            Self::Eq(_, point) => point.len() == k,
            Self::Combo(terms) => terms.iter().all(|&(_, addr)| (addr as usize) < (1 << k)),
        }
    }
}

/// `Σ_y row(y) · Ĵ(y, col) = value` — one assist statement's `W`-value as a
/// claim on the layout table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JaggedClaim {
    pub row: JaggedRowWeight,
    /// The assist's final point, interleaved — `2(m+1)` coordinates.
    pub col: Vec<F128>,
    pub value: F128,
}

impl JaggedClaim {
    /// An honest claim about the table at the given weights.
    pub fn honest(row: JaggedRowWeight, col: Vec<F128>, t: &JaggedTable) -> Self {
        let value = jagged_bilinear(&row, &col, t);
        Self { row, col, value }
    }

    /// Discharge directly — the root check, `O(2^k + runs·m)`.
    pub fn check_direct(&self, t: &JaggedTable) -> bool {
        jagged_bilinear(&self.row, &self.col, t) == self.value
    }

    /// An inherited claim: a previous fold's plain-eq output re-enters the
    /// next fold. `None` if the claim is not the plain shape a fold emits.
    ///
    /// A published accumulator entry carries its LIVE word as BOTH weights'
    /// length-1 low (wall 3), so the two must agree and their common value
    /// becomes the row weight's scale — `low = [1]` reproduces the old
    /// plain-eq claim exactly, `low = [0]` is the zero claim.
    pub fn from_folded(c: &MatrixClaim) -> Option<Self> {
        (c.row.low.len() == 1 && c.col.low == c.row.low).then(|| Self {
            row: JaggedRowWeight::Eq(c.row.low[0], c.row.point.clone()),
            col: c.col.point.clone(),
            value: c.value,
        })
    }
}

/// One deferred verify's jagged-layout export: every count-dependent
/// `W`-value of the multipoint anchor as a RAW claim on the layout table,
/// structured the way the expect recombination consumes them. The flat
/// [`Self::claims`] view is what a merge node folds; the structure is what
/// a circuit publishes and the recombination assert reads. Coefficients
/// (`γ^{128i}·ĝ(ρ″)`, `γ^{…}·eq(ρ,ρ″)`, the dense members' `γ_pd`) stay
/// OUTSIDE the family — they are transcript-derived, so baking them into a
/// claim identity would make the weights unfoldable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JaggedAssertion {
    /// Layout shape: `k` layout-column bits; boundary values in `0..=2^m`.
    pub k: usize,
    pub m: usize,
    /// Per ring-switch claim, in claim order: the raw `eq(z_col)` claim.
    pub rs: Vec<JaggedClaim>,
    /// Per scalar group, in group order: the γ-baked one-hot combo (absent
    /// when the group has no boolean-column members) and the dense members
    /// as `(γ_pd, raw eq claim)` pairs.
    pub groups: Vec<(Option<JaggedClaim>, Vec<(F128, JaggedClaim)>)>,
}

impl JaggedAssertion {
    /// Every claim, flattened — the fold's fresh-claim view.
    pub fn claims(&self) -> Vec<&JaggedClaim> {
        let mut out: Vec<&JaggedClaim> = self.rs.iter().collect();
        for (combo, dense) in &self.groups {
            out.extend(combo.iter());
            out.extend(dense.iter().map(|(_, c)| c));
        }
        out
    }

    /// Root discharge: every claim against the real layout.
    pub fn check(&self, params: &JaggedParams) -> bool {
        let t = JaggedTable::from_params(params);
        t.k == self.k && t.m == self.m && self.claims().into_iter().all(|c| c.check_direct(&t))
    }
}

/// `Σ_y row(y)·col_eq(pair(y))` over the runs — the honest value of a jagged
/// claim, and the root discharge's evaluator. Never touches the `2^{2(m+1)}`
/// pair space densely.
pub fn jagged_bilinear(row: &JaggedRowWeight, col: &[F128], t: &JaggedTable) -> F128 {
    assert_eq!(col.len(), t.n_col_vars(), "pair-space arity mismatch");
    let rw = row.materialize(t.k);
    let mut acc = F128::ZERO;
    let mut y = 0usize;
    for &(t_c, t_next, run) in &t.bounds {
        let w = rw[y..y + run as usize]
            .iter()
            .fold(F128::ZERO, |a, &x| a + x);
        y += run as usize;
        if w != F128::ZERO {
            acc += w * t.eq_at_pair(t_c, t_next, col);
        }
    }
    debug_assert_eq!(y, 1usize << t.k);
    acc
}

/// Discharge a FOLDED jagged claim (plain eq/eq, the shape
/// [`verify_fold_jagged`] emits) against the real layout — the root's single
/// evaluation for this table.
pub fn discharge_jagged(claim: &MatrixClaim, t: &JaggedTable) -> bool {
    let Some(j) = JaggedClaim::from_folded(claim) else {
        return false;
    };
    j.row.well_formed(t.k) && j.col.len() == t.n_col_vars() && j.check_direct(t)
}

/// Bind the jagged claims into the transcript — weights included, exactly as
/// [`observe_claims`] does for the dense fold, plus a shape header so two
/// claim sets differing only in arity cannot collide.
fn observe_jagged_claims<Ch: Challenger>(k: usize, claims: &[JaggedClaim], ch: &mut Ch) {
    ch.observe_label(DOMAIN_JAGGED);
    ch.observe_f128(F128::new(k as u64, claims.len() as u64));
    for c in claims {
        c.row.observe(ch);
        ch.observe_f128_slice(&c.col);
        ch.observe_f128(c.value);
    }
}

/// Fold `k` jagged claims into one plain evaluation claim on the layout
/// table. The jagged analogue of [`prove_fold`]: same two-phase reduction,
/// same transcript conventions, but the column phase walks J's nonzero pairs
/// instead of materializing the pair space, and the claims' eq tensors enter
/// the round messages analytically. The caller must have bound whatever pins
/// the claims (the table's key — the child circuit digest) into `ch`.
pub fn prove_fold_jagged<Ch: Challenger>(
    t: &JaggedTable,
    claims: &[JaggedClaim],
    ch: &mut Ch,
) -> (FoldProof, MatrixClaim) {
    prove_fold_jagged_with_grinding(t, claims, FoldGrinding::disabled(), ch)
}

/// [`prove_fold_jagged`] with an explicit Fiat--Shamir grinding policy.
pub fn prove_fold_jagged_with_grinding<Ch: Challenger>(
    t: &JaggedTable,
    claims: &[JaggedClaim],
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> (FoldProof, MatrixClaim) {
    assert!(!claims.is_empty(), "nothing to fold");
    let n_col = t.n_col_vars();
    for c in claims {
        assert!(c.row.well_formed(t.k), "row weight outside the layout");
        assert_eq!(c.col.len(), n_col, "claims must share the column arity");
    }

    let mut grinding_nonces = Vec::with_capacity(grinding.nonce_count(t.k, n_col));
    observe_jagged_claims(t.k, claims, ch);
    let lambdas = grind_sample_vec(
        ch,
        &mut grinding_nonces,
        grinding.combination_bits,
        claims.len(),
    );

    // Column phase, sparse: per claim, `comb_i` has one entry per run with a
    // nonzero row-weight mass. The claim's own eq tensor over the pair space
    // stays factored: its bound prefix accumulates into `cur`, its unbound
    // suffix is walked per entry (`rest`), and char-2 collapses the `q(∞)`
    // side's tensor factor to 1.
    struct ColState {
        /// λ_i times the bound prefix `Π_{ℓ<j} eq(p_ℓ, r_ℓ)`.
        cur: F128,
        /// Surviving nonzeros of `comb_i`, indices in the unbound suffix.
        entries: Vec<(u64, F128)>,
    }
    let mut states: Vec<ColState> = claims
        .iter()
        .zip(&lambdas)
        .map(|(c, &lam)| {
            let rw = c.row.materialize(t.k);
            let mut y = 0usize;
            let mut entries = Vec::with_capacity(t.bounds.len());
            for &(t_c, t_next, run) in &t.bounds {
                let w = rw[y..y + run as usize]
                    .iter()
                    .fold(F128::ZERO, |a, &x| a + x);
                y += run as usize;
                if w != F128::ZERO {
                    entries.push((t.pair_index(t_c, t_next), w));
                }
            }
            ColState { cur: lam, entries }
        })
        .collect();

    let mut col_rounds = Vec::with_capacity(n_col);
    let mut rho_col = Vec::with_capacity(n_col);
    for j in 0..n_col {
        let (mut q1, mut qinf) = (F128::ZERO, F128::ZERO);
        for (st, c) in states.iter().zip(claims) {
            let p_j = c.col[j];
            let (mut s1, mut sinf) = (F128::ZERO, F128::ZERO);
            for &(idx, v) in &st.entries {
                // The unbound eq suffix at this entry's remaining bits.
                let mut rest = F128::ONE;
                for (l, &p) in c.col.iter().enumerate().skip(j + 1) {
                    rest *= if (idx >> (l - j)) & 1 == 1 {
                        p
                    } else {
                        F128::ONE + p
                    };
                }
                let vr = v * rest;
                if idx & 1 == 1 {
                    s1 += vr;
                }
                sinf += vr;
            }
            q1 += st.cur * p_j * s1;
            qinf += st.cur * sinf;
        }
        ch.observe_f128(q1);
        ch.observe_f128(qinf);
        let r = grind_sample(ch, &mut grinding_nonces, grinding.round_bits);
        col_rounds.push((q1, qinf));
        rho_col.push(r);
        for (st, c) in states.iter_mut().zip(claims) {
            let p_j = c.col[j];
            st.cur *= p_j * r + (F128::ONE + p_j) * (F128::ONE + r);
            let mut merged: BTreeMap<u64, F128> = BTreeMap::new();
            for &(idx, v) in &st.entries {
                let f = if idx & 1 == 1 { r } else { F128::ONE + r };
                *merged.entry(idx >> 1).or_insert(F128::ZERO) += v * f;
            }
            st.entries = merged.into_iter().collect();
        }
    }
    let bridge: Vec<F128> = states
        .iter()
        .map(|st| st.entries.first().map_or(F128::ZERO, |&(_, v)| v))
        .collect();
    for &v in &bridge {
        ch.observe_f128(v);
    }

    // Row phase — dense over the small `2^k` side, the machinery above
    // verbatim. `h(y) = Ĵ(y, ρ_col)` is one eq factor per run, broadcast.
    let mus = grind_sample_vec(
        ch,
        &mut grinding_nonces,
        grinding.combination_bits,
        claims.len(),
    );
    let mut h = vec![F128::ZERO; 1usize << t.k];
    let mut y = 0usize;
    for &(t_c, t_next, run) in &t.bounds {
        let e = t.eq_at_pair(t_c, t_next, &rho_col);
        h[y..y + run as usize].fill(e);
        y += run as usize;
    }
    let mut w_mu = vec![F128::ZERO; 1usize << t.k];
    for (claim, &mu) in claims.iter().zip(&mus) {
        for (dst, src) in w_mu.iter_mut().zip(claim.row.materialize(t.k)) {
            *dst += mu * src;
        }
    }

    let mut row_pairs = vec![(w_mu, h)];
    let one = [F128::ONE];
    let mut row_rounds = Vec::with_capacity(t.k);
    let mut rho_row = Vec::with_capacity(t.k);
    for _ in 0..t.k {
        let msg = round_message(&row_pairs, &one);
        ch.observe_f128(msg.0);
        ch.observe_f128(msg.1);
        let r = grind_sample(ch, &mut grinding_nonces, grinding.round_bits);
        row_rounds.push(msg);
        rho_row.push(r);
        for (a, b) in &mut row_pairs {
            fold_low(a, r);
            fold_low(b, r);
        }
    }
    let value = row_pairs[0].1[0];
    ch.observe_f128(value);

    (
        FoldProof {
            col_rounds,
            bridge,
            row_rounds,
            value,
            grinding_nonces,
        },
        MatrixClaim {
            row: Weight::eq(rho_row),
            col: Weight::eq(rho_col),
            value,
        },
    )
}

/// Replay a jagged fold — `O(k·κ)`, no table access, which is what a merge
/// node's circuit replays. `k_row` is the layout's column count log, known to
/// the caller from the table's key.
pub fn verify_fold_jagged<Ch: Challenger>(
    k_row: usize,
    claims: &[JaggedClaim],
    proof: &FoldProof,
    ch: &mut Ch,
) -> Result<MatrixClaim, FoldError> {
    verify_fold_jagged_with_grinding(k_row, claims, proof, FoldGrinding::disabled(), ch)
}

/// [`verify_fold_jagged`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_fold_jagged_with_grinding<Ch: Challenger>(
    k_row: usize,
    claims: &[JaggedClaim],
    proof: &FoldProof,
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<MatrixClaim, FoldError> {
    if claims.is_empty() || proof.bridge.len() != claims.len() {
        return Err(FoldError::Malformed);
    }
    let n_col = claims[0].col.len();
    if claims
        .iter()
        .any(|c| c.col.len() != n_col || !c.row.well_formed(k_row))
        || proof.row_rounds.len() != k_row
        || proof.col_rounds.len() != n_col
    {
        return Err(FoldError::Malformed);
    }

    if proof.grinding_nonces.len() != grinding.nonce_count(k_row, n_col) {
        return Err(FoldError::InvalidGrinding);
    }
    let mut nonce_idx = 0usize;

    observe_jagged_claims(k_row, claims, ch);
    let lambdas = verify_grind_sample_vec_n(
        ch,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.combination_bits,
        claims.len(),
    )?;

    let target = claims
        .iter()
        .zip(&lambdas)
        .fold(F128::ZERO, |acc, (c, &l)| acc + l * c.value);
    let (running, rho_col) = replay_rounds(
        &proof.col_rounds,
        target,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.round_bits,
        ch,
    )?;
    for &v in &proof.bridge {
        ch.observe_f128(v);
    }
    let expect =
        claims
            .iter()
            .zip(&lambdas)
            .zip(&proof.bridge)
            .fold(F128::ZERO, |acc, ((c, &l), &g)| {
                let col_eval = c.col.iter().zip(&rho_col).fold(F128::ONE, |e, (&p, &r)| {
                    e * (p * r + (F128::ONE + p) * (F128::ONE + r))
                });
                acc + l * col_eval * g
            });
    if running != expect {
        return Err(FoldError::ConsistencyFailed { which: "col" });
    }

    let mus = verify_grind_sample_vec_n(
        ch,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.combination_bits,
        claims.len(),
    )?;
    let target = proof
        .bridge
        .iter()
        .zip(&mus)
        .fold(F128::ZERO, |acc, (&g, &m)| acc + m * g);
    let (running, rho_row) = replay_rounds(
        &proof.row_rounds,
        target,
        &proof.grinding_nonces,
        &mut nonce_idx,
        grinding.round_bits,
        ch,
    )?;
    let w_mu = claims
        .iter()
        .zip(&mus)
        .fold(F128::ZERO, |acc, (c, &m)| acc + m * c.row.eval(&rho_row));
    if running != w_mu * proof.value {
        return Err(FoldError::ConsistencyFailed { which: "row" });
    }
    ch.observe_f128(proof.value);
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());

    Ok(MatrixClaim {
        row: Weight::eq(rho_row),
        col: Weight::eq(rho_col),
        value: proof.value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::pcs::jagged::JaggedParams;

    const D: &[u8] = b"matrix-fold-test";

    struct Rng(u64);
    impl Rng {
        fn f128(&mut self) -> F128 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let lo = z ^ (z >> 31);
            self.0 = self.0.wrapping_add(0x1234_5678_9ABC_DEF0);
            let mut w = self.0;
            w = (w ^ (w >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            F128::new(lo, w ^ (w >> 32))
        }
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as usize) % n
        }
    }

    /// Column marginals the generic way — tests have no tuned circuit.
    fn gen_combs(m: &SparseBinaryMatrix, claims: &[MatrixClaim]) -> Vec<Vec<F128>> {
        claims
            .iter()
            .map(|c| col_marginal(m, &c.row.materialize(), m.num_cols))
            .collect()
    }

    /// A random sparse binary matrix with ~`per_row` nonzeros per row.
    fn matrix(k: usize, per_row: usize, seed: u64) -> SparseBinaryMatrix {
        let n = 1usize << k;
        let mut rng = Rng(seed);
        let rows = (0..n)
            .map(|_| {
                let mut cs: Vec<usize> = (0..per_row).map(|_| rng.below(n)).collect();
                cs.sort_unstable();
                cs.dedup();
                cs
            })
            .collect();
        SparseBinaryMatrix {
            num_rows: n,
            num_cols: n,
            rows,
        }
    }

    /// An honest claim at the lincheck's weight shape: `low(2^s) ⊗ eq(point)`.
    fn honest_claim(m: &SparseBinaryMatrix, k: usize, s: usize, rng: &mut Rng) -> MatrixClaim {
        let mk = |rng: &mut Rng| {
            Weight::low_eq(
                (0..1usize << s).map(|_| rng.f128()).collect(),
                (0..k - s).map(|_| rng.f128()).collect(),
            )
        };
        MatrixClaim::honest(mk(rng), mk(rng), m)
    }

    /// `materialize` and `eval` must agree — the prover and verifier views of
    /// a weight.
    #[test]
    fn weight_eval_matches_materialization() {
        let mut rng = Rng(7);
        for (k, s) in [(6usize, 0usize), (6, 2), (8, 3), (5, 5)] {
            let w = Weight::low_eq(
                (0..1usize << s).map(|_| rng.f128()).collect(),
                (0..k - s).map(|_| rng.f128()).collect(),
            );
            let dense = w.materialize();
            let rho: Vec<F128> = (0..k).map(|_| rng.f128()).collect();
            // MLE of the dense vector at rho, by folding low-first.
            let mut acc = dense;
            for &r in &rho {
                let half = acc.len() / 2;
                for x in 0..half {
                    acc[x] = acc[2 * x] * (F128::ONE + r) + acc[2 * x + 1] * r;
                }
                acc.truncate(half);
            }
            assert_eq!(w.eval(&rho), acc[0], "k={k} s={s}");
        }
    }

    /// Shapes with a longer eq-point than the early tests used.
    #[test]
    fn fold_at_the_lincheck_shape() {
        for (k, s) in [(9usize, 6usize), (8, 6), (7, 6), (7, 3)] {
            let m = matrix(k, 5, 0x5150 + k as u64);
            let mut rng = Rng(0x9001 + k as u64);
            let claims: Vec<MatrixClaim> =
                (0..2).map(|_| honest_claim(&m, k, s, &mut rng)).collect();
            let mut chp = FsChallenger::new(D);
            let combs = gen_combs(&m, &claims);
            let (proof, _) = prove_fold(&m, &combs, &claims, &mut chp);
            let mut chv = FsChallenger::new(D);
            let out = verify_fold(&claims, &proof, &mut chv)
                .unwrap_or_else(|e| panic!("k={k} s={s}: {e:?}"));
            assert!(out.check_direct(&m), "k={k} s={s}");
        }
    }

    /// The fold accepts honest claims, and its output claim is TRUE — i.e. it
    /// discharges directly against the matrix. That is the property the whole
    /// scheme rests on: the accumulator can be believed at the root.
    #[test]
    fn honest_fold_yields_a_true_claim() {
        let k = 7;
        let m = matrix(k, 5, 0xF01D);
        let mut rng = Rng(0xC1A1);
        for count in [1usize, 2, 4] {
            let claims: Vec<MatrixClaim> = (0..count)
                .map(|_| honest_claim(&m, k, 6, &mut rng))
                .collect();
            for c in &claims {
                assert!(c.check_direct(&m), "test claims must start honest");
            }

            let mut chp = FsChallenger::new(D);
            let combs = gen_combs(&m, &claims);
            let (proof, out_p) = prove_fold(&m, &combs, &claims, &mut chp);
            let mut chv = FsChallenger::new(D);
            let out_v = verify_fold(&claims, &proof, &mut chv).expect("honest fold verifies");
            assert_eq!(out_p, out_v, "prover and verifier must agree (k={count})");
            assert!(
                out_v.check_direct(&m),
                "the folded claim must be true (k={count})"
            );
        }
    }

    /// Folding a folded claim: the output shape (`eq ⊗ eq`) must feed straight
    /// back in, which is what makes an aggregation tree possible.
    #[test]
    fn folded_claims_fold_again() {
        let k = 6;
        let m = matrix(k, 4, 0xBEEF);
        let mut rng = Rng(0xD00D);
        let level0: Vec<MatrixClaim> = (0..4).map(|_| honest_claim(&m, k, 6, &mut rng)).collect();

        // Two 2->1 folds, then one more over their outputs.
        let mut acc = Vec::new();
        for pair in level0.chunks(2) {
            let mut ch = FsChallenger::new(D);
            let combs = gen_combs(&m, pair);
            let (proof, _) = prove_fold(&m, &combs, pair, &mut ch);
            let mut chv = FsChallenger::new(D);
            acc.push(verify_fold(pair, &proof, &mut chv).expect("level-0 fold"));
        }
        let mut ch = FsChallenger::new(D);
        let combs = gen_combs(&m, &acc);
        let (proof, _) = prove_fold(&m, &combs, &acc, &mut ch);
        let mut chv = FsChallenger::new(D);
        let root = verify_fold(&acc, &proof, &mut chv).expect("level-1 fold");
        assert!(root.check_direct(&m), "the root claim must be true");
    }

    #[test]
    fn dense_grinding_roundtrip_and_rejects_bad_nonce_shape() {
        let k = 5;
        let m = matrix(k, 4, 0x1280_D35E);
        let mut rng = Rng(0x1280_F01D);
        let claims: Vec<MatrixClaim> = (0..3).map(|_| honest_claim(&m, k, 3, &mut rng)).collect();
        let combs = gen_combs(&m, &claims);
        let policy = FoldGrinding::per_challenge_128();
        let mut chp = FsChallenger::new(b"matrix-fold-grinding-test");
        let (proof, out_p) = prove_fold_with_grinding(&m, &combs, &claims, policy, &mut chp);
        assert_eq!(proof.grinding_nonces.len(), policy.nonce_count(k, k));

        let mut chv = FsChallenger::new(b"matrix-fold-grinding-test");
        let out_v = verify_fold_with_grinding(&claims, &proof, policy, &mut chv)
            .expect("grinded dense fold verifies");
        assert_eq!(out_p, out_v);
        assert!(out_v.check_direct(&m));

        let mut missing = proof;
        missing.grinding_nonces.pop();
        let mut chv = FsChallenger::new(b"matrix-fold-grinding-test");
        assert_eq!(
            verify_fold_with_grinding(&claims, &missing, policy, &mut chv),
            Err(FoldError::InvalidGrinding)
        );
    }

    #[test]
    fn jagged_grinding_roundtrip_and_rejects_bad_nonce_shape() {
        let params = JaggedParams::from_heights(&[2, 3, 1, 2], 2, 4);
        let table = JaggedTable::from_params(&params);
        let mut rng = Rng(0x1280_1A66);
        let claims = vec![
            JaggedClaim::honest(
                JaggedRowWeight::eq((0..table.k).map(|_| rng.f128()).collect()),
                (0..table.n_col_vars()).map(|_| rng.f128()).collect(),
                &table,
            ),
            JaggedClaim::honest(
                JaggedRowWeight::Combo(vec![(rng.f128(), 0), (rng.f128(), 3)]),
                (0..table.n_col_vars()).map(|_| rng.f128()).collect(),
                &table,
            ),
        ];
        let policy = FoldGrinding::per_challenge_128();
        let mut chp = FsChallenger::new(b"jagged-fold-grinding-test");
        let (proof, out_p) = prove_fold_jagged_with_grinding(&table, &claims, policy, &mut chp);
        let mut chv = FsChallenger::new(b"jagged-fold-grinding-test");
        let out_v = verify_fold_jagged_with_grinding(table.k, &claims, &proof, policy, &mut chv)
            .expect("grinded jagged fold verifies");
        assert_eq!(out_p, out_v);
        assert!(discharge_jagged(&out_v, &table));

        let mut missing = proof;
        missing.grinding_nonces.pop();
        let mut chv = FsChallenger::new(b"jagged-fold-grinding-test");
        assert_eq!(
            verify_fold_jagged_with_grinding(table.k, &claims, &missing, policy, &mut chv),
            Err(FoldError::InvalidGrinding)
        );
    }

    /// A false input claim must not survive: the fold either rejects, or emits
    /// an output claim that fails the root discharge. Both outcomes are sound;
    /// what must never happen is a true-looking accumulator.
    #[test]
    fn a_false_claim_cannot_survive_the_fold() {
        let k = 6;
        let m = matrix(k, 4, 0x1234);
        let mut rng = Rng(0x5678);
        for bad_index in [0usize, 1] {
            let mut claims: Vec<MatrixClaim> =
                (0..2).map(|_| honest_claim(&m, k, 6, &mut rng)).collect();
            claims[bad_index].value += F128::ONE;

            let mut chp = FsChallenger::new(D);
            let combs = gen_combs(&m, &claims);
            let (proof, _) = prove_fold(&m, &combs, &claims, &mut chp);
            let mut chv = FsChallenger::new(D);
            match verify_fold(&claims, &proof, &mut chv) {
                Err(_) => {}
                Ok(out) => assert!(
                    !out.check_direct(&m),
                    "a false claim produced a true accumulator (bad={bad_index})"
                ),
            }
        }
    }

    /// Tampering with the transcript: each of the three proof surfaces must be
    /// caught, either by a sumcheck check or by the root discharge.
    #[test]
    fn tampered_fold_transcripts_are_caught() {
        let k = 6;
        let m = matrix(k, 4, 0xAB01);
        let mut rng = Rng(0xCD02);
        let claims: Vec<MatrixClaim> = (0..2).map(|_| honest_claim(&m, k, 6, &mut rng)).collect();
        let mut ch = FsChallenger::new(D);
        let combs = gen_combs(&m, &claims);
        let (good, _) = prove_fold(&m, &combs, &claims, &mut ch);

        let mut tampers = Vec::new();
        let mut t = good.clone();
        t.row_rounds[0].0 += F128::ONE;
        tampers.push(("row message", t));
        let mut t = good.clone();
        t.bridge[1] += F128::ONE;
        tampers.push(("bridge value", t));
        let mut t = good.clone();
        t.col_rounds[0].1 += F128::ONE;
        tampers.push(("col message", t));
        let mut t = good.clone();
        t.value += F128::ONE;
        tampers.push(("output value", t));

        let mut rejected = 0usize;
        for (what, proof) in tampers {
            let mut chv = FsChallenger::new(D);
            match verify_fold(&claims, &proof, &mut chv) {
                Err(_) => rejected += 1,
                Ok(out) => assert!(
                    !out.check_direct(&m),
                    "tampered {what} produced a true accumulator"
                ),
            }
        }
        // Not vacuous: the sumcheck checks themselves must bite on some of
        // these, rather than every tamper sliding through to the root check.
        assert!(rejected > 0, "no tamper was caught by the fold verifier");
    }

    /// Malformed shapes are rejected, not panicked on — an accumulator may
    /// come from an untrusted source.
    #[test]
    fn malformed_folds_are_rejected() {
        let k = 5;
        let m = matrix(k, 3, 0x99);
        let mut rng = Rng(0x11);
        let claims: Vec<MatrixClaim> = (0..2).map(|_| honest_claim(&m, k, 5, &mut rng)).collect();
        let mut ch = FsChallenger::new(D);
        let combs = gen_combs(&m, &claims);
        let (good, _) = prove_fold(&m, &combs, &claims, &mut ch);

        let mut short = good.clone();
        short.row_rounds.pop();
        let mut chv = FsChallenger::new(D);
        assert_eq!(
            verify_fold(&claims, &short, &mut chv),
            Err(FoldError::Malformed)
        );

        let mut bridge = good.clone();
        bridge.bridge.pop();
        let mut chv = FsChallenger::new(D);
        assert_eq!(
            verify_fold(&claims, &bridge, &mut chv),
            Err(FoldError::Malformed)
        );

        let mut chv = FsChallenger::new(D);
        assert_eq!(verify_fold(&[], &good, &mut chv), Err(FoldError::Malformed));
    }
}
