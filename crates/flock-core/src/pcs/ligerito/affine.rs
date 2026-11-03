//! Affine F128 folds for the Ligerito ladder (docs/affine-fold-f128.tex).
//!
//! PROTOTYPE: the plain materialized-basis path only, with simple kernels.
//! It replaces the F256 fold challenges and the coordinate split of
//! [`super::extension`] with one F128 affine fold per round:
//!
//! `f' = f_0 + α f_1 + β X f_1`, `W' = W_0 + r (W_0 + W_1)`.
//!
//! The prover sends the 2x3 grid `G[u][b] = <u, W_b>` for `u ∈ {f_0, f_1,
//! X f_1}`. The claim fixes `G[f_1][1]`, so five values go on the wire. The
//! folded message gains one coefficient per round. At a code switch the low
//! `2^c` coefficients are committed and the `k` overflow coefficients are
//! sent in the clear. Each committed table is native: a level removes `k`
//! variables, not `k - 1`.
//!
//! Layout: a level's table has `rows = 2^k` rows of `width` coefficients,
//! stored `f[row + rows * col]` (rows on the low bits, as `ligero_commit`
//! expects). The weight `W` keeps `2^c` columns for the whole level and is
//! zero on the overflow columns.

use std::{
    env::var,
    time::{Duration, Instant},
};

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
};

use crate::{
    merkle::cap_layer,
    pcs::{
        ligerito::{
            AdditiveNttF128, Challenger, F128, Hash, LigeroWitness, ProverConfig, RecursiveProof,
            VerifierConfig, build_eq_table, ceil_log2, eval_sk_at_vks, grind_and_sample_queries,
            induce_sumcheck_poly_auto, ligero_commit, merkle_paths_for, next_s,
            verify_and_sample_queries, verify_level_opens,
        },
        ring_switch::build_eq_scaled_parallel,
    },
};

const LABEL: &[u8] = b"flock-ligerito-affine-f128-v0";

/// Proof of the affine ladder. Every vector is in transcript order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineProof {
    /// L0 cap (`2^c₀` nodes at the schedule's cap depth).
    pub initial_cap: Vec<Hash>,
    /// Caps of the committed tables L1..L_r.
    pub recursive_caps: Vec<Vec<Hash>>,
    /// Opened rows + capped paths for L0..L_r (one entry per level).
    pub openings: Vec<RecursiveProof>,
    /// One grid per fold round: `[G00, G01, G10, G20, G21]` with `G[u][b]`,
    /// `u = 0, 1, 2` for `f_0, f_1, X f_1`. `G11 = v - G00` is implied.
    pub grids: Vec<[F128; 5]>,
    /// Per code switch (L0..L_{r-1}): the `k` overflow coefficients.
    pub overflows: Vec<Vec<F128>>,
    /// The final folded message, `2^{c_r} + k_r` coefficients.
    pub yr: Vec<F128>,
    pub ood_values: Vec<F128>,
    pub fold_grinding_nonces: Vec<u64>,
    pub claim_batch_grinding_nonces: Vec<u64>,
    pub consistency_batch_grinding_nonces: Vec<u64>,
    pub grinding_nonces: Vec<u64>,
}

impl AffineProof {
    pub fn size_bytes(&self) -> usize {
        const ELEM: usize = size_of::<F128>();
        let caps: usize =
            self.recursive_caps.iter().map(Vec::len).sum::<usize>() + self.initial_cap.len();
        let opened: usize = self
            .openings
            .iter()
            .map(|p| {
                p.opened_rows.iter().map(Vec::len).sum::<usize>() * ELEM + p.merkle_proof.len() * 32
            })
            .sum();
        let words = self.grids.len() * 5
            + self.overflows.iter().map(Vec::len).sum::<usize>()
            + self.yr.len()
            + self.ood_values.len();
        let nonces = self.fold_grinding_nonces.len()
            + self.claim_batch_grinding_nonces.len()
            + self.consistency_batch_grinding_nonces.len()
            + self.grinding_nonces.len();
        caps * 32 + opened + words * ELEM + nonces * 8
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// Per-level `(k, c)`: level ℓ's table has `2^k` rows of `2^c` columns.
/// Level 0 is the L0 commitment; level ℓ+1's table is level ℓ's folded
/// message, so `k_{ℓ+1} + c_{ℓ+1} = c_ℓ`.
fn level_shapes(log_n: usize, initial_k: usize, recursive_ks: &[usize]) -> Vec<(usize, usize)> {
    assert!(initial_k >= 1 && log_n > initial_k);
    let mut out = vec![(initial_k, log_n - initial_k)];
    for &k in recursive_ks {
        let d = out.last().expect("nonempty").1;
        assert!(
            k >= 1 && d > k,
            "affine level needs c >= 1 (d = {d}, k = {k})"
        );
        out.push((k, d - k));
    }
    for &(k, c) in &out {
        // The overflow coefficients sit on X_{2^c + s}, s < k.
        assert!(
            k <= 1usize << c,
            "overflow k = {k} exceeds 2^c = {}",
            1usize << c
        );
    }
    out
}

/// The affine ladder for a split (F256) config: the same ks, rates and
/// queries per level, with native table shapes. Levels that no longer fit
/// (`c < 1` or `k > 2^c`) are dropped from the tail, since each native level
/// is one variable smaller than its split twin. `fold_grinding_bits` gives
/// the per-level fold grinding (docs/affine-fold-f128.tex, §Parameters).
pub fn affine_configs(
    split: &ProverConfig,
    log_n: usize,
    fold_grinding_bits: &[usize],
) -> (ProverConfig, VerifierConfig) {
    let mut c = log_n - split.initial_k;
    let mut ks = Vec::new();
    let mut cols = Vec::new();
    for &k in &split.recursive_ks {
        if c <= k || k > 1usize << (c - k) {
            break;
        }
        c -= k;
        ks.push(k);
        cols.push(c);
    }
    let levels = ks.len() + 1;
    assert!(
        levels >= 2,
        "affine ladder needs at least one recursive level"
    );
    assert!(fold_grinding_bits.len() >= levels);
    let cut = |v: &[usize]| v[..levels].to_vec();
    let prover = ProverConfig {
        log_inv_rates: cut(&split.log_inv_rates),
        recursive_steps: ks.len(),
        initial_log_msg_cols: log_n - split.initial_k,
        initial_log_num_interleaved: split.initial_k,
        initial_k: split.initial_k,
        recursive_log_msg_cols: cols,
        recursive_ks: ks,
        queries: cut(&split.queries),
        grinding_bits: cut(&split.grinding_bits),
        fold_grinding_bits: cut(fold_grinding_bits),
        claim_batch_grinding_bits: cut(&split.claim_batch_grinding_bits),
        consistency_batch_grinding_bits: cut(&split.consistency_batch_grinding_bits),
        ood_samples: cut(&split.ood_samples),
        merkle_hash: split.merkle_hash,
        stratified: vec![],
    }
    .with_default_stratified();
    let verifier = VerifierConfig {
        log_inv_rates: prover.log_inv_rates.clone(),
        recursive_steps: prover.recursive_steps,
        initial_log_msg_cols: prover.initial_log_msg_cols,
        initial_log_num_interleaved: prover.initial_log_num_interleaved,
        initial_k: prover.initial_k,
        recursive_log_msg_cols: prover.recursive_log_msg_cols.clone(),
        recursive_ks: prover.recursive_ks.clone(),
        queries: prover.queries.clone(),
        grinding_bits: prover.grinding_bits.clone(),
        fold_grinding_bits: prover.fold_grinding_bits.clone(),
        claim_batch_grinding_bits: prover.claim_batch_grinding_bits.clone(),
        consistency_batch_grinding_bits: prover.consistency_batch_grinding_bits.clone(),
        ood_samples: prover.ood_samples.clone(),
        merkle_hash: prover.merkle_hash,
        stratified: prover.stratified.clone(),
    };
    (prover, verifier)
}

// ---------------------------------------------------------------------------
// Novel-basis helpers
// ---------------------------------------------------------------------------

/// Prefix products `P_i = a_0 ⋯ a_{i-1}` of the novel-basis squaring
/// constants `a_i = W_{i+1}(v_{i+1}) / W_i(v_i)^2` for the standard basis
/// `v_i = x^i`. `P` has `dim + 1` entries.
pub fn carry_products(dim: usize) -> Vec<F128> {
    let mut w: Vec<F128> = (0..=dim).map(|i| F128::new(1u64 << i, 0)).collect();
    let mut c = Vec::with_capacity(dim + 1);
    for i in 0..=dim {
        let ci = w[i];
        c.push(ci);
        for wj in w.iter_mut() {
            *wj = *wj * (*wj + ci);
        }
    }
    let mut p = vec![F128::ONE];
    for i in 0..dim {
        let a = c[i + 1] * (c[i] * c[i]).inv();
        p.push(p[i] * a);
    }
    p
}

/// `(T g)_m` for a message `g` of length `len` (the output has `len + 1`
/// coefficients): `P_{t(m-1)} g_{m-1} + P_{z(m)} g_{m + 2^{z(m)} - 1}`, with
/// `t` the trailing ones and `z` the trailing zeros. The second term is
/// present when its index is in range. `g(j)` reads coefficient `j`.
#[inline(always)]
fn t_at(g: impl Fn(usize) -> F128, len: usize, m: usize, p: &[F128]) -> F128 {
    if m == 0 {
        return F128::ZERO;
    }
    let mut out = p[(m - 1).trailing_ones() as usize] * g(m - 1);
    let tz = m.trailing_zeros() as usize;
    let j = m + (1usize << tz) - 1;
    if j < len {
        out += p[tz] * g(j);
    }
    out
}

/// `F_s(g) = <(X^s g)_{<2^c}, eq(z)>` for `s = 0..=max_s`, where `g` has
/// `2^c` coefficients and `z ∈ F^c` (LSB first). This is what the L0 grid
/// needs from each lane: the weight's column factor against `X^s` times the
/// row.
///
/// One top-down fold of `g` gives all of them. Split `g = g_L + Ŵ_{c-1} g_H`.
/// `X` raises a coefficient index by at most one, and coefficients at
/// `2^c` and above never flow back down, so
/// `F_s(g) = F'_s(g_L + z_{c-1}(g_L + g_H)) + z_{c-1} <B_s(g_L + g_H), eq(z')>`,
/// where `B_s(h)` is the overflow of `X^s h` past `2^{c-1}`. It depends only
/// on the top `s` coefficients of `h`. The cost is the fold (`2^c`
/// multiplications) plus `O(c · max_s^2)`, the same as computing `F_0` alone.
pub fn shifted_eq_functionals(g: &[F128], z: &[F128], max_s: usize, p: &[F128]) -> Vec<F128> {
    let c = z.len();
    assert_eq!(g.len(), 1usize << c);
    let mut cur = g.to_vec();
    let mut out = vec![F128::ZERO; max_s + 1];
    // The split identity needs `s <= 2^b`; the last few levels (at most
    // `2 max_s` coefficients) are finished by brute force.
    let mut b = c;
    while b > 0 && (1usize << (b - 1)) >= max_s.max(1) {
        b -= 1;
        let half = 1usize << b;
        let zb = z[b];
        let lo = half - max_s;
        let win: Vec<F128> = (lo..half).map(|j| cur[j] + cur[j + half]).collect();
        split_correction(win, b, z, p, &mut out);
        let (l, h) = cur.split_at(half);
        cur = l.iter().zip(h).map(|(&a, &hh)| a + zb * (a + hh)).collect();
    }
    let eq = build_eq_table(&z[..b]);
    let mut xs = cur;
    for o in out.iter_mut() {
        *o += xs
            .iter()
            .zip(&eq)
            .fold(F128::ZERO, |acc, (&a, &e)| acc + a * e);
        let len = xs.len();
        xs = (0..=len).map(|m| t_at(|j| xs[j], len, m, p)).collect();
    }
    out
}

/// The level-`b` term of [`shifted_eq_functionals`]: `win` holds the top
/// `max_s = out.len() - 1` coefficients of `g_L + g_H` (indices
/// `[2^b - max_s, 2^b)`; inputs below cannot reach `2^b` in `max_s` steps).
/// Adds `z_b <B_s, eq(z_0..z_{b-1})>` to `out[s]` for `s >= 1`.
fn split_correction(mut win: Vec<F128>, b: usize, z: &[F128], p: &[F128], out: &mut [F128]) {
    let max_s = out.len() - 1;
    let half = 1usize << b;
    debug_assert!(half >= max_s && win.len() == max_s);
    let lo = half - max_s;
    let low_eq: Vec<F128> = (0..max_s)
        .map(|j| {
            (0..b).fold(F128::ONE, |e, i| {
                e * if (j >> i) & 1 == 1 {
                    z[i]
                } else {
                    F128::ONE + z[i]
                }
            })
        })
        .collect();
    for s in 1..=max_s {
        let len = lo + win.len();
        let cur = &win;
        win = (lo..=len)
            .map(|m| {
                t_at(
                    |j| if j >= lo { cur[j - lo] } else { F128::ZERO },
                    len,
                    m,
                    p,
                )
            })
            .collect();
        // Overflow coefficient half + j (j < s) carries eq(z_0..z_{b-1}; j).
        let overflow =
            (half..lo + win.len()).fold(F128::ZERO, |acc, j| acc + win[j - lo] * low_eq[j - half]);
        out[s] += z[b] * overflow;
    }
}

/// [`shifted_eq_functionals`] for every live lane of a lane-major witness
/// (`witness[lane · d + col]`) and several column points, in one sweep.
/// The top `c - low_bits` column bits are folded in a single pass against
/// their eq table; the split corrections of those levels need only the
/// `max_s` coefficients below each block boundary, which are gathered
/// directly. The remaining `2^low_bits` vector goes through the recursion.
/// Returns `[term][lane][s]`; lanes `num_lanes..lanes` are zero.
#[allow(clippy::too_many_arguments)]
fn lane_statistics_fused(
    witness: &[F128],
    d: usize,
    lanes: usize,
    num_lanes: usize,
    points: &[&[F128]],
    max_s: usize,
    p: &[F128],
    low_bits: usize,
) -> Vec<Vec<Vec<F128>>> {
    let c = d.trailing_zeros() as usize;
    let low = c.min(low_bits.max(ceil_log2(max_s.max(1)) + 1));
    let blk = 1usize << low;
    let eq_tops: Vec<Vec<F128>> = points.iter().map(|z| build_eq_table(&z[low..])).collect();
    let n_h = 1usize << (c - low);
    const H_CHUNK: usize = 8;
    let terms = points.len();
    let zero = || vec![F128::ZERO; terms * blk];
    let per_lane: Vec<Vec<Vec<F128>>> = (0..num_lanes)
        .into_par_iter()
        .map(|lane| {
            let row = &witness[lane * d..(lane + 1) * d];
            let v = (0..n_h.div_ceil(H_CHUNK))
                .into_par_iter()
                .fold(zero, |mut acc, hc| {
                    for h in hc * H_CHUNK..((hc + 1) * H_CHUNK).min(n_h) {
                        let src = &row[h * blk..(h + 1) * blk];
                        for (t, eq_top) in eq_tops.iter().enumerate() {
                            let w = eq_top[h];
                            for (a, &x) in acc[t * blk..(t + 1) * blk].iter_mut().zip(src) {
                                *a += w * x;
                            }
                        }
                    }
                    acc
                })
                .reduce(zero, |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                });
            points
                .iter()
                .enumerate()
                .map(|(t, z)| {
                    let mut out = vec![F128::ZERO; max_s + 1];
                    if max_s > 0 {
                        for b in (low..c).rev() {
                            let half = 1usize << b;
                            let eq_above = build_eq_table(&z[b + 1..c]);
                            let win = (half - max_s..half)
                                .map(|j| {
                                    eq_above
                                        .iter()
                                        .enumerate()
                                        .fold(F128::ZERO, |acc, (h, &e)| {
                                            let base = h << (b + 1);
                                            acc + e * (row[base + j] + row[base + half + j])
                                        })
                                })
                                .collect();
                            split_correction(win, b, z, p, &mut out);
                        }
                    }
                    let tail =
                        shifted_eq_functionals(&v[t * blk..(t + 1) * blk], &z[..low], max_s, p);
                    for (o, x) in out.iter_mut().zip(tail) {
                        *o += x;
                    }
                    out
                })
                .collect()
        })
        .collect();
    (0..terms)
        .map(|t| {
            let mut stats: Vec<Vec<F128>> = per_lane.iter().map(|l| l[t].clone()).collect();
            stats.resize(lanes, vec![F128::ZERO; max_s + 1]);
            stats
        })
        .collect()
}

/// Normalized subspace polynomials `Ŵ_0(x), …, Ŵ_{count-1}(x)`.
fn subspace_evals(x: F128, count: usize) -> Vec<F128> {
    let sks_vks = eval_sk_at_vks(count);
    let mut w = Vec::with_capacity(count);
    let mut s = x;
    for k in 0..count {
        if k > 0 {
            s = next_s(s, sks_vks[k - 1]);
        }
        w.push(s);
    }
    for (wk, &v) in w.iter_mut().zip(&sks_vks) {
        *wk *= v.inv();
    }
    w
}

/// Novel-basis evaluations `X_0(x), …, X_{len-1}(x)`, `X_j = Π_{b: j_b} Ŵ_b`.
fn novel_basis_evals(x: F128, len: usize) -> Vec<F128> {
    let bits = ceil_log2(len.max(1));
    let w = subspace_evals(x, bits);
    let mut tab = vec![F128::ONE];
    for &wb in &w {
        let ext: Vec<F128> = tab.iter().map(|&t| t * wb).collect();
        tab.extend(ext);
    }
    tab.truncate(len);
    tab
}

/// The fold weights `ω_i(x) = Π_j (α_j + β_j x)^{i_j}` of one level's
/// `2^k` rows (round `j` binds row bit `j`).
fn row_weights(gammas: &[(F128, F128)], x: F128) -> Vec<F128> {
    let mut tab = vec![F128::ONE];
    for &(alpha, beta) in gammas {
        let g = alpha + beta * x;
        let ext: Vec<F128> = tab.iter().map(|&t| t * g).collect();
        tab.extend(ext);
    }
    tab
}

fn query_point(q: usize) -> F128 {
    F128::new(q as u64, 0)
}

/// The value the committed low part `f'_lo` must take at the query point:
/// `Σ_i ω_i(x) leaf_i − Σ_s τ_s X_{2^c + s}(x)`. With no overflow it is
/// `Enc(f')(x)` itself.
fn consistency_value(
    row: &[F128],
    q: usize,
    gammas: &[(F128, F128)],
    overflow: &[F128],
    log_cols: usize,
) -> F128 {
    let x = query_point(q);
    let omega = row_weights(gammas, x);
    debug_assert!(row.len() <= omega.len(), "dead lanes are zero and omitted");
    let folded = row
        .iter()
        .zip(&omega)
        .fold(F128::ZERO, |acc, (&v, &w)| acc + v * w);
    if overflow.is_empty() {
        return folded;
    }
    // X_{2^c + s}(x) = Ŵ_c(x) · X_s(x) for s < k <= 2^c.
    let w = subspace_evals(x, log_cols + 1);
    let low = novel_basis_evals(x, overflow.len());
    let tail = overflow
        .iter()
        .zip(&low)
        .fold(F128::ZERO, |acc, (&t, &b)| acc + t * b);
    folded + w[log_cols] * tail
}

// ---------------------------------------------------------------------------
// Prover state
// ---------------------------------------------------------------------------

const PAR_MIN: usize = 1 << 12;

struct Level {
    f: Vec<F128>,
    rows: usize,
    width: usize,
    w: Vec<F128>,
    claim: F128,
}

impl Level {
    fn new(f: Vec<F128>, w: Vec<F128>, log_rows: usize, claim: F128) -> Self {
        let rows = 1usize << log_rows;
        assert_eq!(f.len(), w.len());
        assert!(f.len().is_power_of_two() && f.len() >= rows);
        let width = f.len() / rows;
        Self {
            f,
            rows,
            width,
            w,
            claim,
        }
    }

    /// `W += β B`, `v += β e`.
    fn add_weight(&mut self, basis: &[F128], beta: F128, value: F128) {
        assert_eq!(basis.len(), self.w.len());
        self.w
            .as_mut_slice()
            .into_par_iter()
            .zip(basis.into_par_iter())
            .with_min_len(PAR_MIN)
            .for_each(|(dst, &b)| *dst += beta * b);
        self.claim += beta * value;
    }

    fn dot(&self, basis: &[F128]) -> F128 {
        let live = self.w.len();
        self.f[..live]
            .into_par_iter()
            .zip(basis.into_par_iter())
            .with_min_len(PAR_MIN)
            .map(|(&a, &b)| a * b)
            .reduce(|| F128::ZERO, |a, b| a + b)
    }

    /// `X f_1` for every row pair: `half * (width + 1)` coefficients.
    fn shifted_odd_rows(&self, p: &[F128]) -> Vec<F128> {
        let half = self.rows / 2;
        let width = self.width;
        let f = &self.f;
        (0..half * (width + 1))
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .map(|q| {
                let (m, col) = (q % half, q / half);
                t_at(|j| f[2 * (m + half * j) + 1], width, col, p)
            })
            .collect()
    }

    /// The 2x3 grid `[G00, G01, G10, G11, G20, G21]` against `X f_1`.
    fn grid(&self, xf1: &[F128]) -> [F128; 6] {
        let live = self.w.len() / 2;
        let (f, w) = (&self.f, &self.w);
        (0..live)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .fold(
                || [F128::ZERO; 6],
                |mut acc, q| {
                    let (f0, f1, t) = (f[2 * q], f[2 * q + 1], xf1[q]);
                    let (w0, w1) = (w[2 * q], w[2 * q + 1]);
                    acc[0] += f0 * w0;
                    acc[1] += f0 * w1;
                    acc[2] += f1 * w0;
                    acc[3] += f1 * w1;
                    acc[4] += t * w0;
                    acc[5] += t * w1;
                    acc
                },
            )
            .reduce(
                || [F128::ZERO; 6],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            )
    }

    /// Fold one round in place of `xf1`, which becomes the new table.
    fn fold(&mut self, mut xf1: Vec<F128>, grid: &[F128; 6], alpha: F128, beta: F128, r: F128) {
        let half = self.rows / 2;
        let base_len = half * self.width;
        let f = &self.f;
        xf1.as_mut_slice()
            .into_par_iter()
            .enumerate()
            .with_min_len(PAR_MIN)
            .for_each(|(q, t)| {
                let base = if q < base_len {
                    f[2 * q] + alpha * f[2 * q + 1]
                } else {
                    F128::ZERO
                };
                *t = base + beta * *t;
            });
        let w = &self.w;
        let next_w: Vec<F128> = (0..w.len() / 2)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .map(|q| w[2 * q] + r * (w[2 * q] + w[2 * q + 1]))
            .collect();
        self.claim = fold_claim(grid, alpha, beta, r);
        self.f = xf1;
        self.w = next_w;
        self.rows = half;
        self.width += 1;
    }
}

/// `v' = Σ_{u,b} a_u e_b G[u][b]`, `a = (1, α, β)`, `e = (1 + r, r)`.
fn fold_claim(g: &[F128; 6], alpha: F128, beta: F128, r: F128) -> F128 {
    let e0 = F128::ONE + r;
    let pair = |x: F128, y: F128| e0 * x + r * y;
    pair(g[0], g[1]) + alpha * pair(g[2], g[3]) + beta * pair(g[4], g[5])
}

struct Commitment<'a> {
    mat: &'a [F128],
    tree: &'a [Hash],
    block_len: usize,
    lanes: usize,
}

impl Commitment<'_> {
    fn row(&self, q: usize) -> &[F128] {
        &self.mat[q * self.lanes..(q + 1) * self.lanes]
    }
}

fn view(w: &LigeroWitness) -> Commitment<'_> {
    Commitment {
        mat: &w.mat,
        tree: &w.tree,
        block_len: w.block_len,
        lanes: w.num_interleaved,
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// The L0 input of the affine ladder.
pub enum AffineL0 {
    /// LSB-lane layout (`ligero_commit`): `witness[col · 2^k0 + lane]`, with a
    /// dense basis in the same layout.
    Dense {
        witness: Vec<F128>,
        basis: Vec<F128>,
    },
    /// Lane-major integer-lane commit (`commit_lane_major`):
    /// `witness[lane · 2^c0 + col]`, lanes `>= num_lanes` zero. The basis is
    /// `gamma · eq(point, ·)` in the same index order, never materialized.
    /// `seeded_stats`, when given, are the per-lane `<lane, eq(point_lo)>`.
    Lanes {
        witness: Vec<F128>,
        point: Vec<F128>,
        gamma: F128,
        seeded_stats: Option<Vec<F128>>,
        num_lanes: usize,
    },
}

/// Prove `Σ_x witness[x] · basis[x] = target` against an L0 commitment of
/// `witness` built by `ligero_commit(witness, log_n - initial_k, initial_k,
/// log_inv_rates[0], …)`. `config` must be an affine ladder
/// ([`affine_configs`]).
pub fn prove_affine<Ch: Challenger>(
    config: &ProverConfig,
    witness: Vec<F128>,
    basis: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    challenger: &mut Ch,
) -> AffineProof {
    prove_affine_l0(
        config,
        AffineL0::Dense { witness, basis },
        target,
        l0_codeword,
        l0_tree,
        challenger,
    )
}

/// [`prove_affine`] for either L0 layout ([`AffineL0`]).
pub fn prove_affine_l0<Ch: Challenger>(
    config: &ProverConfig,
    l0: AffineL0,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    challenger: &mut Ch,
) -> AffineProof {
    let witness_len = match &l0 {
        AffineL0::Dense { witness, .. } | AffineL0::Lanes { witness, .. } => witness.len(),
    };
    let log_n = witness_len.trailing_zeros() as usize;
    let r = config.recursive_steps;
    let shapes = level_shapes(log_n, config.initial_k, &config.recursive_ks);
    assert_eq!(
        config.recursive_log_msg_cols,
        shapes[1..].iter().map(|s| s.1).collect::<Vec<_>>()
    );
    config
        .validate_stratified()
        .expect("stratified schedules invalid (affine prover)");
    let strat = |level: usize| &config.stratified[level];
    let claim_bits = |level: usize| config.claim_batch_grinding_bits[level] as u32;
    let consistency_bits = |level: usize| config.consistency_batch_grinding_bits[level] as u32;

    let trace = var("LIG_PROVE_TRACE").is_ok();
    let t_total = Instant::now();
    let (mut t_ood, mut t_folds, mut t_commits, mut t_grind, mut t_opens, mut t_induce) = (
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );

    challenger.observe_label(LABEL);
    challenger.observe_f128(target);
    let block_len_0 = 1usize << (shapes[0].1 + config.log_inv_rates[0]);
    let lanes_0 = match &l0 {
        AffineL0::Dense { .. } => 1usize << shapes[0].0,
        AffineL0::Lanes { num_lanes, .. } => *num_lanes,
    };
    assert_eq!(l0_codeword.len(), block_len_0 * lanes_0);
    let initial_cap = cap_layer(l0_tree, block_len_0, strat(0).cap_depth()).to_vec();
    challenger.observe_bytes(initial_cap.as_flattened());

    let p = carry_products(log_n + 1);
    let mut proof = AffineProof {
        initial_cap,
        recursive_caps: Vec::new(),
        openings: Vec::new(),
        grids: Vec::new(),
        overflows: Vec::new(),
        yr: Vec::new(),
        ood_values: Vec::new(),
        fold_grinding_nonces: Vec::new(),
        claim_batch_grinding_nonces: Vec::new(),
        consistency_batch_grinding_nonces: Vec::new(),
        grinding_nonces: Vec::new(),
    };

    let t = Instant::now();
    let (mut level, mut l0_gammas) = match l0 {
        AffineL0::Dense { witness, basis } => {
            let mut level = Level::new(witness, basis, shapes[0].0, target);
            for _ in 0..config.ood_samples[0] {
                let z = challenger.sample_f128_vec(log_n);
                let eq_z = build_eq_scaled_parallel(&z, F128::ONE);
                let y = level.dot(&eq_z);
                challenger.observe_f128(y);
                proof.ood_values.push(y);
                let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(0));
                proof.claim_batch_grinding_nonces.push(nonce);
                level.add_weight(&eq_z, beta, y);
            }
            t_ood += t.elapsed();
            (level, None)
        }
        AffineL0::Lanes {
            witness,
            point,
            gamma,
            seeded_stats,
            num_lanes,
        } => {
            let (level, gammas) = lane_rounds(
                config,
                &witness,
                shapes[0],
                num_lanes,
                (point, gamma, seeded_stats),
                target,
                &p,
                &mut proof,
                challenger,
                trace,
            );
            t_folds += t.elapsed();
            (level, Some(gammas))
        }
    };

    let mut committed: Vec<LigeroWitness> = Vec::with_capacity(r);
    for li in 0..=r {
        let (k, c) = shapes[li];
        let t = Instant::now();
        let mut gammas = l0_gammas.take().unwrap_or_default();
        if gammas.is_empty() {
            assert_eq!(level.rows, 1usize << k);
            assert_eq!(level.width, 1usize << c);
        }
        while gammas.len() < k {
            let xf1 = level.shifted_odd_rows(&p);
            let g = level.grid(&xf1);
            debug_assert_eq!(g[0] + g[3], level.claim, "grid must split the claim");
            let sent = [g[0], g[1], g[2], g[4], g[5]];
            challenger.observe_f128_slice(&sent);
            proof.grids.push(sent);
            let (nonce, ch) =
                challenger.grind_pow_and_sample_f128_vec(config.fold_grinding_bits[li] as u32, 3);
            proof.fold_grinding_nonces.push(nonce);
            let (alpha, beta, rr) = (ch[0], ch[1], ch[2]);
            level.fold(xf1, &g, alpha, beta, rr);
            gammas.push((alpha, beta));
        }
        t_folds += t.elapsed();
        let prev = if li == 0 {
            Commitment {
                mat: l0_codeword,
                tree: l0_tree,
                block_len: block_len_0,
                lanes: lanes_0,
            }
        } else {
            view(&committed[li - 1])
        };

        if li == r {
            proof.yr = std::mem::take(&mut level.f);
            challenger.observe_f128_slice(&proof.yr);
            let t = Instant::now();
            let (nonce, queries) = grind_and_sample_queries(
                challenger,
                config.grinding_bits[li] as u32,
                prev.block_len,
                config.queries[li],
                strat(li),
            );
            proof.grinding_nonces.push(nonce);
            t_grind += t.elapsed();
            let t = Instant::now();
            proof.openings.push(RecursiveProof {
                opened_rows: queries.iter().map(|&q| prev.row(q).to_vec()).collect(),
                merkle_proof: merkle_paths_for(prev.tree, prev.block_len, &queries, strat(li)),
            });
            t_opens += t.elapsed();
            break;
        }

        // Code switch: commit the low 2^c coefficients, send the overflow.
        let mut f = std::mem::take(&mut level.f);
        let overflow = f.split_off(1usize << c);
        challenger.observe_f128_slice(&overflow);
        let (k_next, c_next) = shapes[li + 1];
        let rate = config.log_inv_rates[li + 1];
        let t = Instant::now();
        let ntt = AdditiveNttF128::standard(c_next + rate);
        let next = ligero_commit(&f, c_next, k_next, rate, &ntt, config.merkle_hash);
        t_commits += t.elapsed();
        let cap = next.cap(strat(li + 1).cap_depth()).to_vec();
        challenger.observe_bytes(cap.as_flattened());
        proof.recursive_caps.push(cap);
        let w = std::mem::take(&mut level.w);
        level = Level::new(f, w, k_next, level.claim);

        let t = Instant::now();
        for _ in 0..config.ood_samples[li + 1] {
            let z = challenger.sample_f128_vec(c);
            let eq_z = build_eq_scaled_parallel(&z, F128::ONE);
            let y = level.dot(&eq_z);
            challenger.observe_f128(y);
            proof.ood_values.push(y);
            let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(li + 1));
            proof.claim_batch_grinding_nonces.push(nonce);
            level.add_weight(&eq_z, beta, y);
        }
        t_ood += t.elapsed();

        let t = Instant::now();
        let (nonce, queries) = grind_and_sample_queries(
            challenger,
            config.grinding_bits[li] as u32,
            prev.block_len,
            config.queries[li],
            strat(li),
        );
        proof.grinding_nonces.push(nonce);
        let (nonce, alpha) = challenger
            .grind_pow_and_sample_f128_vec(consistency_bits(li), ceil_log2(config.queries[li]));
        proof.consistency_batch_grinding_nonces.push(nonce);
        t_grind += t.elapsed();
        let t = Instant::now();
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&q| prev.row(q).to_vec()).collect();
        proof.openings.push(RecursiveProof {
            merkle_proof: merkle_paths_for(prev.tree, prev.block_len, &queries, strat(li)),
            opened_rows,
        });
        t_opens += t.elapsed();
        let t = Instant::now();
        let rows = &proof.openings.last().expect("just pushed").opened_rows;
        let enforced = batched_consistency(rows, &queries, &gammas, &overflow, c, &alpha);
        let basis = induce_sumcheck_poly_auto(
            c,
            config.log_inv_rates[li],
            &eval_sk_at_vks(c),
            &vec![Vec::new(); queries.len()],
            &[],
            &queries,
            &alpha,
        )
        .0;
        debug_assert_eq!(level.dot(&basis), enforced, "consistency claim must hold");
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(li + 1));
        proof.claim_batch_grinding_nonces.push(nonce);
        level.add_weight(&basis, beta, enforced);
        t_induce += t.elapsed();
        proof.overflows.push(overflow);
        committed.push(next);
    }

    if trace {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        eprintln!("[lig-prove-affine] total = {:.2} ms", ms(t_total.elapsed()));
        eprintln!("  OODs (eq build + eval + fold-in):  {:.2} ms", ms(t_ood));
        eprintln!("  folds (X f1 + grid + fold):        {:.2} ms", ms(t_folds));
        eprintln!(
            "  recursive commits (NTT + merkle):  {:.2} ms",
            ms(t_commits)
        );
        eprintln!("  grinding (query + consistency):    {:.2} ms", ms(t_grind));
        eprintln!("  opens (rows + merkle paths):       {:.2} ms", ms(t_opens));
        eprintln!(
            "  induce (basis + enforced + glue):  {:.2} ms",
            ms(t_induce)
        );
    }
    proof
}

/// One eq term of the L0 weight on the lane-major path:
/// `scale · eq(point_hi, row) · eq(point_lo, col)`, with the per-row
/// statistics `stats[row][s] = <(X^s f[row])_{<2^c}, eq(point_lo)>`.
struct LaneTerm {
    scale: F128,
    col_point: Vec<F128>,
    rows: Vec<F128>,
    stats: Vec<Vec<F128>>,
}

/// Column bits the lane statistics leave to the recursion (`2^12` words
/// per lane and term stay in L1/L2).
const LANE_LOW_BITS: usize = 12;

/// `Σ_i ω_i(X) lane_i` for lane-major lanes and per-lane polynomials `ω_i`
/// (monomial coefficients in `X`): `h_s = Σ_i [X^s]ω_i · lane_i` in one
/// column-chunked pass (stored `[col][s]`), then Horner
/// `f = h_0 + X(h_1 + X(h_2 + …))` with the gather map `T`.
fn materialize_lanes(witness: &[F128], d: usize, omega: &[Vec<F128>], p: &[F128]) -> Vec<F128> {
    let deg = omega.iter().map(|w| w.len() - 1).max().unwrap_or(0);
    let width = deg + 1;
    const COLS: usize = 1 << 10;
    let mut h = vec![F128::ZERO; d * width];
    h.par_chunks_mut(COLS * width)
        .enumerate()
        .for_each(|(chunk, out)| {
            let col0 = chunk * COLS;
            let cols = out.len() / width;
            for (i, w) in omega.iter().enumerate() {
                let src = &witness[i * d + col0..i * d + col0 + cols];
                for (j, &x) in src.iter().enumerate() {
                    let dst = &mut out[j * width..j * width + w.len()];
                    for (o, &cs) in dst.iter_mut().zip(w) {
                        *o += cs * x;
                    }
                }
            }
        });
    let mut f: Vec<F128> = (0..d)
        .into_par_iter()
        .map(|col| h[col * width + deg])
        .collect();
    for s in (0..deg).rev() {
        let len = f.len();
        let cur = &f;
        let hs = &h;
        f = (0..len + 1)
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .map(|m| {
                let low = if m < d { hs[m * width + s] } else { F128::ZERO };
                t_at(|j| cur[j], len, m, p) + low
            })
            .collect();
    }
    f
}

/// The L0 rounds on the lane-major path, from per-lane statistics only:
/// the grid of round j is a combination of `stats` and the row weights, and
/// the fold maps `S'[m][s] = S[2m][s] + α S[2m+1][s] + β S[2m+1][s+1]`.
/// The witness is read once per term (the statistics) and once more to
/// materialize the folded message `Σ_i ω_i(X) lane_i` for the L1 commit.
#[allow(clippy::too_many_arguments)]
fn lane_rounds<Ch: Challenger>(
    config: &ProverConfig,
    witness: &[F128],
    (k, c): (usize, usize),
    num_lanes: usize,
    (point, gamma, seeded_stats): (Vec<F128>, F128, Option<Vec<F128>>),
    target: F128,
    p: &[F128],
    proof: &mut AffineProof,
    challenger: &mut Ch,
    trace: bool,
) -> (Level, Vec<(F128, F128)>) {
    let (lanes, d) = (1usize << k, 1usize << c);
    assert_eq!(witness.len(), lanes * d);
    assert_eq!(point.len(), k + c);
    let claim_bits = config.claim_batch_grinding_bits[0] as u32;
    let t0 = Instant::now();
    // The first OOD point is fixed before any answer is sent, so its
    // statistics share the seeded term's sweep. Later points (no shipped
    // config takes them at L0) depend on the previous answer.
    let first_z = (config.ood_samples[0] > 0).then(|| challenger.sample_f128_vec(k + c));
    let mut sweep_points: Vec<&[F128]> = vec![&point[..c]];
    if let Some(z) = &first_z {
        sweep_points.push(&z[..c]);
    }
    let mut swept = lane_statistics_fused(
        witness,
        d,
        lanes,
        num_lanes,
        &sweep_points,
        k,
        p,
        LANE_LOW_BITS,
    )
    .into_iter();
    let seed_stats = swept.next().expect("seeded term");
    if let Some(given) = &seeded_stats {
        debug_assert!(
            given.iter().zip(&seed_stats).all(|(&a, s)| a == s[0]),
            "seeded statistics disagree with the witness"
        );
    }
    let mut terms = vec![LaneTerm {
        scale: gamma,
        col_point: point[..c].to_vec(),
        rows: build_eq_table(&point[c..]),
        stats: seed_stats,
    }];
    let mut claim = target;
    let mut first = first_z.map(|z| (z, swept.next().expect("first OOD term")));
    for _ in 0..config.ood_samples[0] {
        let (z, stats) = first.take().unwrap_or_else(|| {
            let z = challenger.sample_f128_vec(k + c);
            let stats = lane_statistics_fused(
                witness,
                d,
                lanes,
                num_lanes,
                &[&z[..c]],
                k,
                p,
                LANE_LOW_BITS,
            )
            .remove(0);
            (z, stats)
        });
        let rows = build_eq_table(&z[c..]);
        let y = rows
            .iter()
            .zip(&stats)
            .fold(F128::ZERO, |acc, (&w, s)| acc + w * s[0]);
        challenger.observe_f128(y);
        proof.ood_values.push(y);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits);
        proof.claim_batch_grinding_nonces.push(nonce);
        claim += beta * y;
        terms.push(LaneTerm {
            scale: beta,
            col_point: z[..c].to_vec(),
            rows,
            stats,
        });
    }
    let t_stats = t0.elapsed();

    let mut gammas = Vec::with_capacity(k);
    for _ in 0..k {
        let mut g = [F128::ZERO; 6];
        for term in &terms {
            let mut tg = [F128::ZERO; 6];
            for m in 0..term.rows.len() / 2 {
                let (w0, w1) = (term.rows[2 * m], term.rows[2 * m + 1]);
                let (s0, s1) = (&term.stats[2 * m], &term.stats[2 * m + 1]);
                let us = [s0[0], s1[0], s1[1]];
                for (u, &v) in us.iter().enumerate() {
                    tg[2 * u] += w0 * v;
                    tg[2 * u + 1] += w1 * v;
                }
            }
            for (a, b) in g.iter_mut().zip(tg) {
                *a += term.scale * b;
            }
        }
        debug_assert_eq!(g[0] + g[3], claim, "grid must split the claim");
        let sent = [g[0], g[1], g[2], g[4], g[5]];
        challenger.observe_f128_slice(&sent);
        proof.grids.push(sent);
        let (nonce, ch) =
            challenger.grind_pow_and_sample_f128_vec(config.fold_grinding_bits[0] as u32, 3);
        proof.fold_grinding_nonces.push(nonce);
        let (alpha, beta, r) = (ch[0], ch[1], ch[2]);
        for term in &mut terms {
            let half = term.rows.len() / 2;
            term.stats = (0..half)
                .map(|m| {
                    let (s0, s1) = (&term.stats[2 * m], &term.stats[2 * m + 1]);
                    (0..s0.len() - 1)
                        .map(|s| s0[s] + alpha * s1[s] + beta * s1[s + 1])
                        .collect()
                })
                .collect();
            term.rows = (0..half)
                .map(|m| {
                    let (w0, w1) = (term.rows[2 * m], term.rows[2 * m + 1]);
                    w0 + r * (w0 + w1)
                })
                .collect();
        }
        claim = fold_claim(&g, alpha, beta, r);
        gammas.push((alpha, beta));
    }

    // The folded message Σ_i ω_i(X) lane_i, ω_i = Π_{j: i_j} (α_j + β_j X):
    // h_s = Σ_i [X^s]ω_i · lane_i, then Horner f = h_0 + X(h_1 + X(h_2 + …)).
    let t1 = Instant::now();
    let omega: Vec<Vec<F128>> = (0..num_lanes)
        .map(|i| {
            let mut poly = vec![F128::ONE];
            for (j, &(alpha, beta)) in gammas.iter().enumerate() {
                if (i >> j) & 1 == 1 {
                    let mut next = vec![F128::ZERO; poly.len() + 1];
                    for (s, &a) in poly.iter().enumerate() {
                        next[s] += alpha * a;
                        next[s + 1] += beta * a;
                    }
                    poly = next;
                }
            }
            poly
        })
        .collect();
    let mut f = materialize_lanes(witness, d, &omega, p);
    f.resize(d + k, F128::ZERO);
    let mut w = vec![F128::ZERO; d];
    for term in &terms {
        let eq = build_eq_scaled_parallel(&term.col_point, term.scale * term.rows[0]);
        w.as_mut_slice()
            .into_par_iter()
            .zip(eq.into_par_iter())
            .with_min_len(PAR_MIN)
            .for_each(|(dst, e)| *dst += e);
    }
    if trace {
        eprintln!(
            "  L0 lanes: statistics {:.2} ms, materialize {:.2} ms",
            t_stats.as_secs_f64() * 1e3,
            t1.elapsed().as_secs_f64() * 1e3
        );
    }
    let level = Level {
        f,
        rows: 1,
        width: d + k,
        w,
        claim,
    };
    (level, gammas)
}

/// `Σ_q eq(α)_q · (Enc(f')(x_q) − overflow(x_q))`.
fn batched_consistency(
    rows: &[Vec<F128>],
    queries: &[usize],
    gammas: &[(F128, F128)],
    overflow: &[F128],
    log_cols: usize,
    alpha: &[F128],
) -> F128 {
    let weights = build_eq_table(alpha);
    rows.iter()
        .zip(queries)
        .zip(&weights)
        .fold(F128::ZERO, |acc, ((row, &q), &wt)| {
            acc + wt * consistency_value(row, q, gammas, overflow, log_cols)
        })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// A weight introduced on the table of dimension `dim`: bound by the last
/// `dim - residual_log` fold challenges.
enum Context {
    Ood {
        point: Vec<F128>,
        beta: F128,
    },
    Consistency {
        log_cols: usize,
        queries: Vec<usize>,
        alpha: Vec<F128>,
        beta: F128,
    },
}

/// `scale · eq(point, (fixed, y))` for every residual `y`.
fn eq_at_residual(point: &[F128], fixed: &[F128], residual_log: usize, scale: F128) -> Vec<F128> {
    assert_eq!(fixed.len() + residual_log, point.len());
    let prefix = point
        .iter()
        .zip(fixed)
        .fold(scale, |acc, (&z, &r)| acc * (F128::ONE + z + r));
    build_eq_scaled_parallel(&point[fixed.len()..], prefix)
}

/// The batched query basis `Σ_q eq(α)_q X_·(x_q)` at `(fixed, y)`.
fn induced_at_residual(
    log_cols: usize,
    queries: &[usize],
    alpha: &[F128],
    fixed: &[F128],
    residual_log: usize,
    scale: F128,
) -> Vec<F128> {
    assert_eq!(fixed.len() + residual_log, log_cols);
    let weights = build_eq_table(alpha);
    let mut out = vec![F128::ZERO; 1usize << residual_log];
    for (&q, &wt) in queries.iter().zip(&weights) {
        let w = subspace_evals(query_point(q), log_cols);
        let prefix = fixed.iter().zip(&w).fold(scale * wt, |acc, (&r, &wk)| {
            acc * (F128::ONE + r * (F128::ONE + wk))
        });
        let mut tab = vec![prefix];
        for &wk in &w[fixed.len()..] {
            let ext: Vec<F128> = tab.iter().map(|&t| t * wk).collect();
            tab.extend(ext);
        }
        for (dst, v) in out.iter_mut().zip(tab) {
            *dst += v;
        }
    }
    out
}

/// Verify an [`AffineProof`] for the LSB-lane L0 ([`AffineL0::Dense`]).
/// `eval_b_residual(rs, residual_log)` returns the caller's basis bound at the
/// fold challenges `rs` (in binding order), as a table over the
/// `residual_log` remaining variables.
pub fn verify_affine<Ch, F>(
    config: &VerifierConfig,
    proof: &AffineProof,
    log_n: usize,
    target: F128,
    expected_initial_cap: &[Hash],
    eval_b_residual: F,
    challenger: &mut Ch,
) -> bool
where
    Ch: Challenger,
    F: Fn(&[F128], usize) -> Vec<F128>,
{
    let lanes = 1usize << config.initial_k;
    verify_affine_l0(
        config,
        proof,
        log_n,
        target,
        expected_initial_cap,
        (lanes, false),
        eval_b_residual,
        challenger,
    )
}

/// [`verify_affine`] for either L0 layout: `(l0_num_lanes, lane_major)`.
/// On the lane-major path the L0 rounds bind the lane (high) variables
/// first, so the binding order of a lane-major index is its coordinates
/// rotated left by `log_n - initial_k`; `eval_b_residual` receives the
/// challenges in that order.
#[allow(clippy::too_many_arguments)]
pub fn verify_affine_l0<Ch, F>(
    config: &VerifierConfig,
    proof: &AffineProof,
    log_n: usize,
    target: F128,
    expected_initial_cap: &[Hash],
    (l0_num_lanes, lane_major): (usize, bool),
    eval_b_residual: F,
    challenger: &mut Ch,
) -> bool
where
    Ch: Challenger,
    F: Fn(&[F128], usize) -> Vec<F128>,
{
    let r = config.recursive_steps;
    if r < 1
        || config.recursive_ks.len() != r
        || proof.initial_cap != expected_initial_cap
        || config.validate_stratified().is_err()
    {
        return false;
    }
    let shapes = level_shapes(log_n, config.initial_k, &config.recursive_ks);
    if config.recursive_log_msg_cols != shapes[1..].iter().map(|s| s.1).collect::<Vec<_>>() {
        return false;
    }
    let strat = |level: usize| &config.stratified[level];
    let claim_bits = |level: usize| config.claim_batch_grinding_bits[level] as u32;
    let consistency_bits = |level: usize| config.consistency_batch_grinding_bits[level] as u32;

    challenger.observe_label(LABEL);
    challenger.observe_f128(target);
    challenger.observe_bytes(proof.initial_cap.as_flattened());

    let mut claim = target;
    let mut contexts: Vec<(usize, Context)> = Vec::new();
    let mut all_rs: Vec<F128> = Vec::with_capacity(log_n);
    let (mut ood_i, mut claim_i, mut grid_i) = (0usize, 0usize, 0usize);

    macro_rules! next {
        ($v:expr, $i:ident) => {{
            let Some(x) = $v.get($i) else {
                return false;
            };
            $i += 1;
            x
        }};
    }

    if l0_num_lanes == 0 || l0_num_lanes > 1usize << config.initial_k {
        return false;
    }
    for _ in 0..config.ood_samples[0] {
        let mut point = challenger.sample_f128_vec(log_n);
        if lane_major {
            point.rotate_left(log_n - config.initial_k);
        }
        let y = *next!(proof.ood_values, ood_i);
        challenger.observe_f128(y);
        let nonce = *next!(proof.claim_batch_grinding_nonces, claim_i);
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(0)) else {
            return false;
        };
        claim += beta * y;
        contexts.push((log_n, Context::Ood { point, beta }));
    }

    let mut prev_cap: &[Hash] = &proof.initial_cap;
    for li in 0..=r {
        let (k, c) = shapes[li];
        let mut gammas = Vec::with_capacity(k);
        for _ in 0..k {
            let sent = *next!(proof.grids, grid_i);
            challenger.observe_f128_slice(&sent);
            let g = [sent[0], sent[1], sent[2], claim + sent[0], sent[3], sent[4]];
            let Some(&nonce) = proof.fold_grinding_nonces.get(grid_i - 1) else {
                return false;
            };
            let Some(ch) = challenger.verify_pow_and_sample_f128_vec(
                nonce,
                config.fold_grinding_bits[li] as u32,
                3,
            ) else {
                return false;
            };
            claim = fold_claim(&g, ch[0], ch[1], ch[2]);
            gammas.push((ch[0], ch[1]));
            all_rs.push(ch[2]);
        }
        let block_len = 1usize << (c + config.log_inv_rates[li]);
        let Some(opening) = proof.openings.get(li) else {
            return false;
        };
        let Some(&query_nonce) = proof.grinding_nonces.get(li) else {
            return false;
        };

        if li == r {
            if proof.yr.len() != (1usize << c) + k {
                return false;
            }
            challenger.observe_f128_slice(&proof.yr);
            let Some(queries) = verify_and_sample_queries(
                challenger,
                query_nonce,
                config.grinding_bits[li] as u32,
                block_len,
                config.queries[li],
                strat(li),
            ) else {
                return false;
            };
            if !verify_level_opens(
                prev_cap,
                block_len,
                &queries,
                &opening.opened_rows,
                if li == 0 { l0_num_lanes } else { 1usize << k },
                &opening.merkle_proof,
                config.merkle_hash,
                strat(li),
            ) {
                return false;
            }
            // Direct consistency: Enc(yr)(x) against the folded leaf.
            for (row, &q) in opening.opened_rows.iter().zip(&queries) {
                let x = query_point(q);
                let enc = proof
                    .yr
                    .iter()
                    .zip(novel_basis_evals(x, proof.yr.len()))
                    .fold(F128::ZERO, |acc, (&y, b)| acc + y * b);
                if enc != consistency_value(row, q, &gammas, &[], c) {
                    return false;
                }
            }
            // Final claim: <yr_lo, W_final> = v.
            let residual_log = c;
            if all_rs.len() + residual_log != log_n {
                return false;
            }
            let mut w = eval_b_residual(&all_rs, residual_log);
            if w.len() != 1usize << residual_log {
                return false;
            }
            for (dim, ctx) in &contexts {
                let fixed = &all_rs[log_n - dim..];
                let values = match ctx {
                    Context::Ood { point, beta } => {
                        eq_at_residual(point, fixed, residual_log, *beta)
                    }
                    Context::Consistency {
                        log_cols,
                        queries,
                        alpha,
                        beta,
                    } => induced_at_residual(*log_cols, queries, alpha, fixed, residual_log, *beta),
                };
                for (dst, v) in w.iter_mut().zip(values) {
                    *dst += v;
                }
            }
            let residual_claim = proof.yr[..1usize << c]
                .iter()
                .zip(&w)
                .fold(F128::ZERO, |acc, (&y, &b)| acc + y * b);
            return residual_claim == claim
                && ood_i == proof.ood_values.len()
                && claim_i == proof.claim_batch_grinding_nonces.len()
                && grid_i == proof.grids.len()
                && grid_i == proof.fold_grinding_nonces.len()
                && li + 1 == proof.openings.len()
                && li + 1 == proof.grinding_nonces.len()
                && li == proof.overflows.len()
                && li == proof.recursive_caps.len()
                && li == proof.consistency_batch_grinding_nonces.len();
        }

        let Some(overflow) = proof.overflows.get(li) else {
            return false;
        };
        if overflow.len() != k {
            return false;
        }
        challenger.observe_f128_slice(overflow);
        let Some(cap) = proof.recursive_caps.get(li) else {
            return false;
        };
        challenger.observe_bytes(cap.as_flattened());

        for _ in 0..config.ood_samples[li + 1] {
            let point = challenger.sample_f128_vec(c);
            let y = *next!(proof.ood_values, ood_i);
            challenger.observe_f128(y);
            let nonce = *next!(proof.claim_batch_grinding_nonces, claim_i);
            let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(li + 1))
            else {
                return false;
            };
            claim += beta * y;
            contexts.push((c, Context::Ood { point, beta }));
        }

        let Some(queries) = verify_and_sample_queries(
            challenger,
            query_nonce,
            config.grinding_bits[li] as u32,
            block_len,
            config.queries[li],
            strat(li),
        ) else {
            return false;
        };
        let Some(&nonce) = proof.consistency_batch_grinding_nonces.get(li) else {
            return false;
        };
        let Some(alpha) = challenger.verify_pow_and_sample_f128_vec(
            nonce,
            consistency_bits(li),
            ceil_log2(config.queries[li]),
        ) else {
            return false;
        };
        if !verify_level_opens(
            prev_cap,
            block_len,
            &queries,
            &opening.opened_rows,
            if li == 0 { l0_num_lanes } else { 1usize << k },
            &opening.merkle_proof,
            config.merkle_hash,
            strat(li),
        ) {
            return false;
        }
        let enforced =
            batched_consistency(&opening.opened_rows, &queries, &gammas, overflow, c, &alpha);
        let nonce = *next!(proof.claim_batch_grinding_nonces, claim_i);
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(li + 1)) else {
            return false;
        };
        claim += beta * enforced;
        contexts.push((
            c,
            Context::Consistency {
                log_cols: c,
                queries,
                alpha,
                beta,
            },
        ));
        prev_cap = cap;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{
        AffineL0, AffineProof, F128, affine_configs, carry_products, novel_basis_evals,
        prove_affine, prove_affine_l0, t_at, verify_affine, verify_affine_l0,
    };
    use crate::{
        challenger::{Challenger, FsChallenger, RandomChallenger},
        lincheck::build_eq_table,
        merkle::HashKind,
        ntt::additive_ntt_f128::AdditiveNttF128,
        pcs::ligerito::{ProverConfig, VerifierConfig, ligero_commit, partial_eval_lsb},
    };

    fn random_vec(rng: &mut RandomChallenger, len: usize) -> Vec<F128> {
        (0..len).map(|_| rng.sample_f128()).collect()
    }

    /// `Enc(T g)(x) = x · Enc(g)(x)` on the commit's NTT domain, and the
    /// verifier's novel-basis evaluation matches the NTT.
    #[test]
    fn multiply_by_x_matches_ntt() {
        let mut rng = RandomChallenger::new(0xAFF1_0001);
        for c in 1..8 {
            let len = 1usize << c;
            let p = carry_products(c + 2);
            let g = random_vec(&mut rng, len);
            let tg: Vec<F128> = (0..=len).map(|m| t_at(|j| g[j], len, m, &p)).collect();
            let log_dom = c + 2;
            let ntt = AdditiveNttF128::standard(log_dom);
            let encode = |v: &[F128]| {
                let mut e = v.to_vec();
                e.resize(1 << log_dom, F128::ZERO);
                ntt.forward_transform(&mut e);
                e
            };
            let (eg, etg) = (encode(&g), encode(&tg));
            for i in 0..1usize << log_dom {
                let x = F128::new(i as u64, 0);
                assert_eq!(etg[i], x * eg[i], "c = {c}, point {i}");
                let direct = tg
                    .iter()
                    .zip(novel_basis_evals(x, tg.len()))
                    .fold(F128::ZERO, |acc, (&a, b)| acc + a * b);
                assert_eq!(direct, etg[i], "novel basis eval, c = {c}, point {i}");
            }
        }
    }

    /// The one-fold recursion for `<(X^s g)_{<2^c}, eq(z)>` matches applying
    /// `T` s times and taking the eq inner product.
    #[test]
    fn shifted_eq_functionals_match_brute_force() {
        let mut rng = RandomChallenger::new(0xAFF1_0003);
        let max_s = 7;
        for c in 0..10 {
            let len = 1usize << c;
            let p = carry_products(c + 4);
            let g = random_vec(&mut rng, len);
            let z = random_vec(&mut rng, c);
            let eq = build_eq_table(&z);
            let got = super::shifted_eq_functionals(&g, &z, max_s, &p);
            let mut xs = g.clone();
            for s in 0..=max_s {
                let want = xs
                    .iter()
                    .zip(&eq)
                    .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);
                assert_eq!(got[s], want, "c = {c}, s = {s}");
                let l = xs.len();
                xs = (0..=l).map(|m| t_at(|j| xs[j], l, m, &p)).collect();
            }
        }
    }

    /// The fused lane sweep matches per-lane [`super::shifted_eq_functionals`].
    #[test]
    fn fused_lane_statistics_match_per_lane() {
        let mut rng = RandomChallenger::new(0xAFF1_0005);
        for (c, low_bits) in [(8usize, 3usize), (9, 4), (6, 12), (13, 12)] {
            let d = 1usize << c;
            let (lanes, live) = (8usize, 5usize);
            let mut w = random_vec(&mut rng, live * d);
            w.resize(lanes * d, F128::ZERO);
            let z1 = random_vec(&mut rng, c);
            let z2 = random_vec(&mut rng, c);
            let p = carry_products(c + 4);
            let got =
                super::lane_statistics_fused(&w, d, lanes, live, &[&z1, &z2], 6, &p, low_bits);
            for (t, z) in [&z1, &z2].iter().enumerate() {
                for lane in 0..lanes {
                    let want =
                        super::shifted_eq_functionals(&w[lane * d..(lane + 1) * d], z, 6, &p);
                    assert_eq!(
                        got[t][lane], want,
                        "c {c}, low {low_bits}, term {t}, lane {lane}"
                    );
                }
            }
        }
    }

    struct Fixture {
        prover: ProverConfig,
        verifier: VerifierConfig,
        log_n: usize,
        basis: Vec<F128>,
        target: F128,
        cap: Vec<crate::merkle::Hash>,
        proof: AffineProof,
    }

    fn fixture(ood: usize) -> Fixture {
        let log_n = 14;
        let initial_k = 3;
        let ks = vec![2, 2, 2];
        let levels = ks.len() + 1;
        let split = ProverConfig {
            log_inv_rates: vec![1, 2, 2, 3],
            recursive_steps: ks.len(),
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![0; ks.len()],
            recursive_ks: ks,
            queries: vec![23, 17, 13, 11],
            grinding_bits: vec![2; levels],
            fold_grinding_bits: vec![0; levels],
            claim_batch_grinding_bits: vec![1; levels],
            consistency_batch_grinding_bits: vec![1; levels],
            ood_samples: vec![ood; levels],
            merkle_hash: HashKind::Blake3,
            stratified: vec![],
        };
        let (prover, verifier) = affine_configs(&split, log_n, &[2, 3, 3, 3]);
        assert_eq!(prover.recursive_log_msg_cols, vec![9, 7, 5]);

        let mut rng = RandomChallenger::new(0xAFF1_0002);
        let witness = random_vec(&mut rng, 1 << log_n);
        let z1 = random_vec(&mut rng, log_n);
        let z2 = random_vec(&mut rng, log_n);
        let gamma = rng.sample_f128();
        let basis: Vec<F128> = build_eq_table(&z1)
            .into_iter()
            .zip(build_eq_table(&z2))
            .map(|(a, b)| a + gamma * b)
            .collect();
        let target = witness
            .iter()
            .zip(&basis)
            .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);

        let rate = prover.log_inv_rates[0];
        let ntt = AdditiveNttF128::standard(log_n - initial_k + rate);
        let l0 = ligero_commit(
            &witness,
            log_n - initial_k,
            initial_k,
            rate,
            &ntt,
            HashKind::Blake3,
        );
        let cap = l0.cap(prover.level_zero_cap_depth()).to_vec();
        let mut ch = FsChallenger::new(b"affine-test");
        let proof = prove_affine(
            &prover,
            witness,
            basis.clone(),
            target,
            &l0.mat,
            &l0.tree,
            &mut ch,
        );
        Fixture {
            prover,
            verifier,
            log_n,
            basis,
            target,
            cap,
            proof,
        }
    }

    fn verifies(fx: &Fixture, proof: &AffineProof, target: F128) -> bool {
        let mut ch = FsChallenger::new(b"affine-test");
        verify_affine(
            &fx.verifier,
            proof,
            fx.log_n,
            target,
            &fx.cap,
            |rs, residual_log| {
                let out = partial_eval_lsb(&fx.basis, rs);
                assert_eq!(out.len(), 1usize << residual_log);
                out
            },
            &mut ch,
        )
    }

    /// The lane-major L0 (integer lanes, factored eq basis, statistics-only
    /// rounds) against a real `commit_lane_major` commitment.
    #[test]
    fn lane_major_roundtrip_and_tamper() {
        use crate::{
            merkle::cap_layer,
            pcs::{
                commit::{PcsParams, commit_lane_major},
                ligerito::LigeritoProfile,
            },
        };
        let (log_n, initial_k, live) = (14usize, 3usize, 5usize);
        let c0 = log_n - initial_k;
        let d = 1usize << c0;
        for ood in [0, 1, 2] {
            let levels = 4;
            let split = ProverConfig {
                log_inv_rates: vec![1, 2, 2, 3],
                recursive_steps: 3,
                initial_log_msg_cols: c0,
                initial_log_num_interleaved: initial_k,
                initial_k,
                recursive_log_msg_cols: vec![0; 3],
                recursive_ks: vec![2, 2, 2],
                queries: vec![23, 17, 13, 11],
                grinding_bits: vec![2; levels],
                fold_grinding_bits: vec![0; levels],
                claim_batch_grinding_bits: vec![1; levels],
                consistency_batch_grinding_bits: vec![1; levels],
                ood_samples: vec![ood; levels],
                merkle_hash: HashKind::Blake3,
                stratified: vec![],
            };
            let (prover, verifier) = affine_configs(&split, log_n, &[2, 3, 3, 3]);
            let mut rng = RandomChallenger::new(0xAFF1_0004 + ood as u64);
            let mut q = random_vec(&mut rng, live * d);
            q.resize(1 << log_n, F128::ZERO);
            let point = random_vec(&mut rng, log_n);
            let gamma = rng.sample_f128();
            let eq = build_eq_table(&point);
            let target = q
                .iter()
                .zip(&eq)
                .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b)
                * gamma;
            let low = build_eq_table(&point[..c0]);
            let seeded: Vec<F128> = (0..live)
                .map(|l| {
                    q[l * d..(l + 1) * d]
                        .iter()
                        .zip(&low)
                        .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b)
                })
                .collect();
            let params = PcsParams {
                m: log_n + 7,
                log_inv_rate: 1,
                log_batch_size: initial_k,
                profile: LigeritoProfile::Fast,
                num_lanes: Some(live),
                merkle_hash: HashKind::Blake3,
            };
            let (_, pd) = commit_lane_major(&q, &params);
            let cap = cap_layer(
                &pd.merkle_tree,
                params.n_leaves(),
                prover.level_zero_cap_depth(),
            )
            .to_vec();
            let mut ch = FsChallenger::new(b"affine-lanes");
            let proof = prove_affine_l0(
                &prover,
                AffineL0::Lanes {
                    witness: q.clone(),
                    point: point.clone(),
                    gamma,
                    seeded_stats: Some(seeded),
                    num_lanes: live,
                },
                target,
                &pd.codeword,
                &pd.merkle_tree,
                &mut ch,
            );
            assert_eq!(proof.openings[0].opened_rows[0].len(), live);
            let mut rotated = point.clone();
            rotated.rotate_left(c0);
            let check = |proof: &AffineProof, target: F128| {
                let mut ch = FsChallenger::new(b"affine-lanes");
                verify_affine_l0(
                    &verifier,
                    proof,
                    log_n,
                    target,
                    &cap,
                    (live, true),
                    |rs, residual_log| {
                        let prefix = rotated
                            .iter()
                            .zip(rs)
                            .fold(gamma, |acc, (&z, &r)| acc * (F128::ONE + z + r));
                        let out: Vec<F128> = build_eq_table(&rotated[rs.len()..])
                            .into_iter()
                            .map(|v| v * prefix)
                            .collect();
                        assert_eq!(out.len(), 1usize << residual_log);
                        out
                    },
                    &mut ch,
                )
            };
            assert!(check(&proof, target), "lane-major roundtrip, ood = {ood}");
            assert!(!check(&proof, target + F128::ONE), "wrong target");
            for e in 0..5 {
                let mut bad = proof.clone();
                bad.grids[1][e] += F128::ONE;
                assert!(!check(&bad, target), "L0 grid entry {e}");
            }
            let mut bad = proof.clone();
            bad.openings[0].opened_rows[3][live - 1] += F128::ONE;
            assert!(!check(&bad, target), "L0 opened row");
            let mut bad = proof.clone();
            bad.overflows[0][2] += F128::ONE;
            assert!(!check(&bad, target), "L0 overflow");
        }
    }

    #[test]
    fn roundtrip_with_and_without_ood() {
        for ood in [0, 1, 2] {
            let fx = fixture(ood);
            assert_eq!(fx.proof.grids.len(), 3 + 2 + 2 + 2);
            assert_eq!(fx.proof.overflows.len(), fx.prover.recursive_steps);
            assert_eq!(fx.proof.yr.len(), (1 << 5) + 2);
            assert!(verifies(&fx, &fx.proof, fx.target), "ood = {ood}");
        }
    }

    #[test]
    fn tampered_proofs_reject() {
        let fx = fixture(1);
        let one = F128::ONE;
        assert!(!verifies(&fx, &fx.proof, fx.target + one), "wrong target");
        let mut cases: Vec<(&str, AffineProof)> = Vec::new();
        for gi in [0, 4, 8] {
            for e in 0..5 {
                let mut p = fx.proof.clone();
                p.grids[gi][e] += one;
                cases.push(("grid", p));
            }
        }
        for (li, s) in [(0, 0), (0, 2), (1, 1), (2, 1)] {
            let mut p = fx.proof.clone();
            p.overflows[li][s] += one;
            cases.push(("overflow", p));
        }
        for i in [0, 31, 32, 33] {
            let mut p = fx.proof.clone();
            p.yr[i] += one;
            cases.push(("yr", p));
        }
        for li in 0..fx.proof.openings.len() {
            let mut p = fx.proof.clone();
            p.openings[li].opened_rows[0][0] += one;
            cases.push(("opened row", p));
        }
        for i in 0..fx.proof.ood_values.len() {
            let mut p = fx.proof.clone();
            p.ood_values[i] += one;
            cases.push(("ood value", p));
        }
        let mut p = fx.proof.clone();
        p.grids.pop();
        cases.push(("truncated grids", p));
        for (what, p) in &cases {
            assert!(!verifies(&fx, p, fx.target), "tampered {what} must reject");
        }
    }
}
