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

use serde::{Deserialize, Serialize};

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    ParallelSlice, ParallelSliceMut,
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
        ring_switch::{build_eq_scaled_parallel, build_eq_split},
    },
};

const LABEL: &[u8] = b"flock-ligerito-affine-f128-v0";

/// Proof of the affine ladder. Every vector is in transcript order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
/// At m32_fast the last native level has 8 columns, and
/// `scripts/affine_fold_params.py` finds no fold bound for two of its
/// rounds; use [`affine_ladder`] with three recursive levels there.
pub fn affine_configs(
    split: &ProverConfig,
    log_n: usize,
    fold_grinding_bits: &[usize],
) -> (ProverConfig, VerifierConfig) {
    let ks = native_ks(split, log_n, usize::MAX);
    let levels = ks.len() + 1;
    build_ladder(
        split,
        log_n,
        &ks,
        &split.log_inv_rates[..levels],
        &split.queries[..levels],
        fold_grinding_bits,
    )
}

/// Johnson radius slack of the Fast profile (`η` in the level TOMLs).
const LADDER_ETA: f64 = 0.02;

/// An affine ladder with its own rate schedule: `rates.len()` levels (L0
/// and `rates.len() - 1` recursive levels, the native ks of `split`), the
/// split config's grinding and OOD schedules, and Johnson query counts
/// recomputed for the new rates. Level ℓ takes the fewest queries `t` with
/// `t · log2(1/(√ρ + η)) + grinding_bits ≥ 128` after paying the fold's
/// agreement loss `t · log2(1 + k/(a n))`, `a = √ρ + η`.
pub fn affine_ladder(
    split: &ProverConfig,
    log_n: usize,
    rates: &[usize],
    fold_grinding_bits: &[usize],
) -> (ProverConfig, VerifierConfig) {
    let ks = native_ks(split, log_n, rates.len() - 1);
    assert_eq!(
        ks.len() + 1,
        rates.len(),
        "ladder deeper than the native shapes"
    );
    let shapes = level_shapes(log_n, split.initial_k, &ks);
    let queries: Vec<usize> = shapes
        .iter()
        .zip(rates)
        .enumerate()
        .map(|(li, (&(k, c), &rate))| {
            let a = (-(rate as f64)).exp2().sqrt() + LADDER_ETA;
            let per_query = -a.log2();
            let n = ((c + rate) as f64).exp2();
            let loss = (1.0 + k as f64 / (a * n)).log2();
            let need = 128.0 - split.grinding_bits[li] as f64;
            (need / (per_query - loss) - 1e-9).ceil() as usize
        })
        .collect();
    build_ladder(split, log_n, &ks, rates, &queries, fold_grinding_bits)
}

/// The registered affine ladders for [`super::LigeritoProfile::FastAffine`]:
/// `(log_n, initial_k) -> (rates, fold grinding bits)`, each certified
/// round by round by `scripts/affine_fold_params.py`, on top of the `Fast`
/// config of the same shape.
/// The m = 24 entry is the small shape the end-to-end tests run; at rate
/// 1/8 its level of 2^7 columns has no certificate, at 1/32 it does.
const AFFINE_LADDERS: &[((usize, usize), &[usize], &[usize])] = &[
    ((17, 6), &[1, 5], &[4, 5]),
    ((25, 6), &[1, 3, 5, 7], &[4, 5, 5, 5]),
];

/// The `FastAffine` configs for a shape, or an error if none is registered.
pub fn affine_ladder_for(
    log_n: usize,
    initial_k: usize,
) -> Result<(ProverConfig, VerifierConfig), String> {
    let (_, rates, fold) = AFFINE_LADDERS
        .iter()
        .find(|(shape, _, _)| *shape == (log_n, initial_k))
        .ok_or_else(|| {
            format!(
                "no affine ladder registered for (log_n={log_n}, initial_k={initial_k}); \
                 add one to AFFINE_LADDERS after running scripts/affine_fold_params.py"
            )
        })?;
    let split = super::prover_config_for(log_n, initial_k, super::LigeritoProfile::Fast)?;
    Ok(affine_ladder(&split, log_n, rates, fold))
}

/// The native recursive ks of `split`, at most `max_levels` of them.
fn native_ks(split: &ProverConfig, log_n: usize, max_levels: usize) -> Vec<usize> {
    let mut c = log_n - split.initial_k;
    let mut ks = Vec::new();
    for &k in split.recursive_ks.iter().take(max_levels) {
        if c <= k || k > 1usize << (c - k) {
            break;
        }
        c -= k;
        ks.push(k);
    }
    ks
}

fn build_ladder(
    split: &ProverConfig,
    log_n: usize,
    ks: &[usize],
    rates: &[usize],
    queries: &[usize],
    fold_grinding_bits: &[usize],
) -> (ProverConfig, VerifierConfig) {
    let levels = ks.len() + 1;
    assert!(
        levels >= 2,
        "affine ladder needs at least one recursive level"
    );
    assert!(fold_grinding_bits.len() >= levels);
    let cols = level_shapes(log_n, split.initial_k, ks)[1..]
        .iter()
        .map(|s| s.1)
        .collect();
    let cut = |v: &[usize]| v[..levels].to_vec();
    let prover = ProverConfig {
        log_inv_rates: cut(rates),
        recursive_steps: ks.len(),
        initial_log_msg_cols: log_n - split.initial_k,
        initial_log_num_interleaved: split.initial_k,
        initial_k: split.initial_k,
        recursive_log_msg_cols: cols,
        recursive_ks: ks.to_vec(),
        queries: cut(queries),
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
    // `t_at` relies on `P_0 = 1`.
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
/// Since `P_0 = 1` and `t(m-1) = z(m)` for even `m`, this is `g_{m-1} +
/// g_m` for odd `m` and `P_{z(m)} (g_{m-1} + g_{m+2^{z(m)}-1})` for even
/// `m`: one multiplication per two coefficients.
#[inline(always)]
fn t_at(g: impl Fn(usize) -> F128, len: usize, m: usize, p: &[F128]) -> F128 {
    if m == 0 {
        return F128::ZERO;
    }
    let tz = m.trailing_zeros() as usize;
    let j = m + (1usize << tz) - 1;
    let sum = if j < len { g(m - 1) + g(j) } else { g(m - 1) };
    if tz == 0 { sum } else { p[tz] * sum }
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

/// Largest subspace-polynomial index the cached tables cover.
const SUBSPACE_TABLE: usize = 48;

/// `(s_k(v_k), 1 / s_k(v_k))` for `k < SUBSPACE_TABLE`. The values do not
/// depend on the domain size, so one table serves every level.
fn subspace_constants() -> &'static (Vec<F128>, Vec<F128>) {
    static TABLE: std::sync::OnceLock<(Vec<F128>, Vec<F128>)> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let v = eval_sk_at_vks(SUBSPACE_TABLE);
        let inv = v.iter().map(|x| x.inv()).collect();
        (v, inv)
    })
}

/// Normalized subspace polynomials `Ŵ_0(x), …, Ŵ_{count-1}(x)`.
fn subspace_evals(x: F128, count: usize) -> Vec<F128> {
    assert!(count <= SUBSPACE_TABLE);
    let (sks_vks, inv) = subspace_constants();
    let mut w = Vec::with_capacity(count);
    let mut s = x;
    for k in 0..count {
        if k > 0 {
            s = next_s(s, sks_vks[k - 1]);
        }
        w.push(s * inv[k]);
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

    /// `<f, eq(z)>` from the eq tensor's factors, without the full table.
    fn dot_eq(&self, eq: &EqFactors) -> F128 {
        assert_eq!(eq.len(), self.w.len());
        self.f[..eq.len()]
            .par_chunks(eq.lo.len())
            .zip(eq.hi.par_iter())
            .map(|(block, &h)| {
                h * block
                    .iter()
                    .zip(&eq.lo)
                    .fold(F128::ZERO, |acc, (&a, &l)| acc + a * l)
            })
            .reduce(|| F128::ZERO, |a, b| a + b)
    }

    /// [`Self::add_weight`] for `basis = eq(z)`, from its factors.
    fn add_eq_weight(&mut self, eq: &EqFactors, beta: F128, value: F128) {
        assert_eq!(eq.len(), self.w.len());
        self.w
            .par_chunks_mut(eq.lo.len())
            .zip(eq.hi.par_iter())
            .for_each(|(block, &h)| {
                let bh = beta * h;
                for (dst, &l) in block.iter_mut().zip(&eq.lo) {
                    *dst += bh * l;
                }
            });
        self.claim += beta * value;
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

/// `eq(z)` as its low and high tensor factors: `eq(z)[i] = lo[i mod
/// 2^{n_lo}] · hi[i >> n_lo]`, with `n_lo` about half of `z`.
struct EqFactors {
    lo: Vec<F128>,
    hi: Vec<F128>,
}

impl EqFactors {
    fn new(z: &[F128]) -> Self {
        let (lo, hi) = build_eq_split(z, z.len().div_ceil(2));
        Self { lo, hi }
    }

    fn len(&self) -> usize {
        self.lo.len() * self.hi.len()
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
    ///
    /// The L0 OOD points reuse `point_lo` (the column coordinates) and draw
    /// only the row coordinates, so every L0 term has the same per-lane
    /// statistics. This keeps the OOD bound only if `point_lo` is sampled
    /// uniformly after the L0 commitment: the list of codewords is fixed by
    /// the commitment, and two distinct list members then differ at
    /// `(point_lo, z_hi)` except with probability `log_n / |F|`, as for a
    /// fresh point.
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
                let eq_z = EqFactors::new(&z);
                let y = level.dot_eq(&eq_z);
                challenger.observe_f128(y);
                proof.ood_values.push(y);
                let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(0));
                proof.claim_batch_grinding_nonces.push(nonce);
                level.add_eq_weight(&eq_z, beta, y);
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
            let eq_z = EqFactors::new(&z);
            let y = level.dot_eq(&eq_z);
            challenger.observe_f128(y);
            proof.ood_values.push(y);
            let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(li + 1));
            proof.claim_batch_grinding_nonces.push(nonce);
            level.add_eq_weight(&eq_z, beta, y);
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
        let ti = Instant::now();
        let enforced = batched_consistency(rows, &queries, &gammas, &overflow, c, &alpha);
        let t_enf = ti.elapsed();
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
        if trace {
            eprintln!(
                "  induce L{li}: enforced {:.2} ms, basis+glue {:.2} ms",
                t_enf.as_secs_f64() * 1e3,
                (ti.elapsed() - t_enf).as_secs_f64() * 1e3
            );
        }
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

/// One L0 round's constants for [`fold_lane_pair`]: `f_0 + α f_1 + β T f_1`
/// is `f_0 + (α + β) f_1[m] + β f_1[m-1]` at odd `m` and `f_0 + α f_1[m] +
/// β P_z (f_1[m-1] + f_1[m+2^z-1])` at even `m`, two multiplications per
/// coefficient either way.
struct RoundCoefs {
    alpha: F128,
    beta: F128,
    alpha_beta: F128,
    beta_p: Vec<F128>,
}

/// Fold one lane pair over the columns `col0..col0 + out.len()`, where
/// `col0` is a multiple of `out.len()` (a power of two). Inside the block
/// `T` reads only the block, except at its first column, which reads
/// `f_1[col0 - 1]` (`left`) and `f_1[col0 + 2^z - 1]` (`far`, when outside).
#[inline(always)]
fn fold_lane_pair(
    x0: &[F128],
    x1: &[F128],
    out: &mut [F128],
    col0: usize,
    rc: &RoundCoefs,
    left: F128,
    far: impl FnOnce(usize) -> F128,
) {
    let n = out.len();
    debug_assert!(n.is_power_of_two() && col0.is_multiple_of(n));
    let t0 = if col0 == 0 {
        F128::ZERO
    } else {
        let tz = col0.trailing_zeros() as usize;
        let fi = (1usize << tz) - 1;
        let fv = if fi < n { x1[fi] } else { far(col0 + fi) };
        rc.beta_p[tz] * (left + fv)
    };
    out[0] = x0[0] + rc.alpha * x1[0] + t0;
    let mut j = 1;
    while j + 1 < n {
        out[j] = x0[j] + rc.alpha_beta * x1[j] + rc.beta * x1[j - 1];
        let e = j + 1;
        let tz = e.trailing_zeros() as usize;
        out[e] = x0[e] + rc.alpha * x1[e] + rc.beta_p[tz] * (x1[j] + x1[e + (1usize << tz) - 1]);
        j += 2;
    }
    if j < n {
        out[j] = x0[j] + rc.alpha_beta * x1[j] + rc.beta * x1[j - 1];
    }
}

/// One round over a block of `n` columns at `col0`: `src(i)` is input lane
/// `i` of `lanes_in`, output lane `m` goes to `dst[m n..(m + 1) n]`, and
/// `edge(i, col0)` gives lane `i`'s `left` and `far` edge values.
fn fold_round<'s>(
    src: impl Fn(usize) -> &'s [F128],
    dst: &mut [F128],
    n: usize,
    col0: usize,
    lanes_in: usize,
    rc: &RoundCoefs,
    edge: impl Fn(usize, usize) -> (F128, F128),
) {
    for (m, out) in dst.chunks_mut(n).take(lanes_in.div_ceil(2)).enumerate() {
        if 2 * m + 1 < lanes_in {
            let (left, far) = edge(2 * m + 1, col0);
            fold_lane_pair(src(2 * m), src(2 * m + 1), out, col0, rc, left, |_| far);
        } else {
            out.copy_from_slice(src(2 * m));
        }
    }
}

/// Columns of the window that computes a chunk's last column through all
/// rounds: an error at the window's first column moves right one column
/// per round, so 8 columns cover up to 7 rounds.
const TAIL_COLS: usize = 8;

/// The folded L0 message `Σ_i ω_i(X) lane_i` for lane-major lanes, folded
/// round by round (`f_0 + α f_1 + β T f_1` on lane pairs) in column chunks
/// of `COLS`, so the intermediate lanes stay in cache. A chunk's first
/// column needs the round inputs at the last column of two other chunks;
/// a first pass computes those from an 8-column window per chunk. The
/// `rounds` overflow coefficients past `d` are folded last, on their own.
/// This is `2 (num_lanes - 1)` multiplications per column (90 over 46
/// lanes).
#[doc(hidden)]
pub fn materialize_lanes<const COLS: usize>(
    witness: &[F128],
    d: usize,
    num_lanes: usize,
    gammas: &[(F128, F128)],
    p: &[F128],
) -> Vec<F128> {
    let rounds = gammas.len();
    // Chunks of COLS columns, or one chunk when the lanes are shorter.
    let cols = COLS.min(d);
    assert!(cols.is_power_of_two() && cols >= TAIL_COLS && d.is_multiple_of(cols));
    assert!(rounds < TAIL_COLS && num_lanes <= 1usize << rounds);
    assert_eq!(witness.len() % d, 0);
    let coefs: Vec<RoundCoefs> = gammas
        .iter()
        .map(|&(alpha, beta)| RoundCoefs {
            alpha,
            beta,
            alpha_beta: alpha + beta,
            beta_p: p.iter().map(|&pz| beta * pz).collect(),
        })
        .collect();
    let lane = |i: usize, lo: usize, len: usize| &witness[i * d + lo..i * d + lo + len];
    // Lane counts entering each round.
    let counts: Vec<usize> = (0..=rounds)
        .scan(num_lanes, |n, _| {
            let cur = *n;
            *n = n.div_ceil(2);
            Some(cur)
        })
        .collect();
    let half = num_lanes.div_ceil(2);

    // Pass 1: tails[c][j][i] = round-j input lane i at chunk c's last column.
    let chunks = d / cols;
    let tails: Vec<Vec<Vec<F128>>> = (0..chunks)
        .into_par_iter()
        .map(|c| {
            let w0 = (c + 1) * cols - TAIL_COLS;
            let mut cur: Vec<F128> = (0..num_lanes)
                .flat_map(|i| lane(i, w0, TAIL_COLS).iter().copied())
                .collect();
            let mut next = vec![F128::ZERO; half * TAIL_COLS];
            let mut out = Vec::with_capacity(rounds);
            for (j, rc) in coefs.iter().enumerate() {
                out.push(
                    (0..counts[j])
                        .map(|i| cur[i * TAIL_COLS + TAIL_COLS - 1])
                        .collect::<Vec<_>>(),
                );
                let src = |i: usize| &cur[i * TAIL_COLS..(i + 1) * TAIL_COLS];
                fold_round(src, &mut next, TAIL_COLS, w0, counts[j], rc, |_, _| {
                    (F128::ZERO, F128::ZERO)
                });
                std::mem::swap(&mut cur, &mut next);
            }
            out
        })
        .collect();
    let tail_at = |j: usize, i: usize, col: usize| {
        debug_assert_eq!(col % cols, cols - 1);
        tails[col / cols][j][i]
    };

    // Pass 2: every chunk through all rounds, edges from the tails.
    let mut f = vec![F128::ZERO; d];
    f.par_chunks_mut(cols).enumerate().for_each(|(c, f_chunk)| {
        let col0 = c * cols;
        let mut a = vec![F128::ZERO; half * cols];
        let mut b = vec![F128::ZERO; half.div_ceil(2).max(1) * cols];
        for (j, rc) in coefs.iter().enumerate() {
            let edge = |i: usize, col0: usize| {
                if col0 == 0 {
                    return (F128::ZERO, F128::ZERO);
                }
                let tz = col0.trailing_zeros() as usize;
                (
                    tail_at(j, i, col0 - 1),
                    tail_at(j, i, col0 + (1usize << tz) - 1),
                )
            };
            if j == 0 {
                let src = |i: usize| lane(i, col0, cols);
                fold_round(src, &mut a, cols, col0, counts[0], rc, edge);
            } else {
                let src = |i: usize| &a[i * cols..(i + 1) * cols];
                fold_round(src, &mut b, cols, col0, counts[j], rc, edge);
                std::mem::swap(&mut a, &mut b);
            }
        }
        f_chunk.copy_from_slice(&a[..cols]);
    });

    // The overflow coefficients past d: round j's inputs have j of them.
    let last = chunks - 1;
    let mut ov: Vec<Vec<F128>> = vec![Vec::new(); num_lanes];
    for (j, rc) in coefs.iter().enumerate() {
        let len = d + j;
        ov = (0..counts[j].div_ceil(2))
            .map(|m| {
                let (x0, x1) = (2 * m, 2 * m + 1);
                (d..=len)
                    .map(|pos| {
                        let at = |i: usize, q: usize| {
                            if q == d - 1 {
                                tails[last][j][i]
                            } else {
                                ov[i].get(q - d).copied().unwrap_or(F128::ZERO)
                            }
                        };
                        let v0 = if pos < len { at(x0, pos) } else { F128::ZERO };
                        if x1 >= counts[j] {
                            return v0;
                        }
                        let v1 = if pos < len { at(x1, pos) } else { F128::ZERO };
                        v0 + rc.alpha * v1 + rc.beta * t_at(|q| at(x1, q), len, pos, p)
                    })
                    .collect()
            })
            .collect();
    }
    f.extend_from_slice(&ov[0]);
    f
}

/// Column chunk of the L0 materialize pass.
const MATERIALIZE_COLS: usize = 1 << 12;

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
    // Every L0 OOD point reuses the seeded point's column coordinates and
    // draws only fresh row coordinates, so all terms share one statistics
    // vector and one sweep. The terms then merge into one, whose row
    // weight is `γ eq(point_hi) + Σ_t β_t eq(z_t)`.
    let mut stats = lane_statistics_fused(
        witness,
        d,
        lanes,
        num_lanes,
        &[&point[..c]],
        k,
        p,
        LANE_LOW_BITS,
    )
    .remove(0);
    if let Some(given) = &seeded_stats {
        debug_assert!(
            given.iter().zip(&stats).all(|(&a, s)| a == s[0]),
            "seeded statistics disagree with the witness"
        );
    }
    let mut rows = build_eq_table(&point[c..]);
    rows.iter_mut().for_each(|w| *w *= gamma);
    let mut claim = target;
    for _ in 0..config.ood_samples[0] {
        let ood_rows = build_eq_table(&challenger.sample_f128_vec(k));
        let y = ood_rows
            .iter()
            .zip(&stats)
            .fold(F128::ZERO, |acc, (&w, s)| acc + w * s[0]);
        challenger.observe_f128(y);
        proof.ood_values.push(y);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits);
        proof.claim_batch_grinding_nonces.push(nonce);
        claim += beta * y;
        rows.iter_mut()
            .zip(&ood_rows)
            .for_each(|(w, &o)| *w += beta * o);
    }
    let mut terms = vec![LaneTerm {
        scale: F128::ONE,
        col_point: point[..c].to_vec(),
        rows,
        stats: std::mem::take(&mut stats),
    }];
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

    // The folded message Σ_i ω_i(X) lane_i, ω_i = Π_{j: i_j} (α_j + β_j X).
    let t1 = Instant::now();
    let mut f = materialize_lanes::<MATERIALIZE_COLS>(witness, d, num_lanes, &gammas, p);
    f.resize(d + k, F128::ZERO);
    let t_mat = t1.elapsed();
    // W = Σ_terms scale · row weight · eq(col_point), from the eq tensor's
    // low and high factors in one pass.
    let lo_bits = c / 2;
    let factors: Vec<(Vec<F128>, Vec<F128>)> = terms
        .iter()
        .map(|term| {
            let (lo, mut hi) = build_eq_split(&term.col_point, lo_bits);
            let scale = term.scale * term.rows[0];
            hi.iter_mut().for_each(|h| *h *= scale);
            (lo, hi)
        })
        .collect();
    let mut w = vec![F128::ZERO; d];
    w.par_chunks_mut(1 << lo_bits)
        .enumerate()
        .for_each(|(h, block)| {
            for (lo, hi) in &factors {
                let hv = hi[h];
                for (dst, &l) in block.iter_mut().zip(lo) {
                    *dst += hv * l;
                }
            }
        });
    if trace {
        eprintln!(
            "  L0 lanes: statistics {:.2} ms, materialize {:.2} ms, weight {:.2} ms",
            t_stats.as_secs_f64() * 1e3,
            t_mat.as_secs_f64() * 1e3,
            (t1.elapsed() - t_mat).as_secs_f64() * 1e3
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
    rows.par_iter()
        .zip(queries.par_iter())
        .zip(weights.par_iter())
        .map(|((row, &q), &wt)| wt * consistency_value(row, q, gammas, overflow, log_cols))
        .reduce(|| F128::ZERO, |a, b| a + b)
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
        (lanes, None),
        eval_b_residual,
        challenger,
    )
}

/// [`verify_affine`] for either L0 layout: `(l0_num_lanes, lane_cols)`,
/// with `lane_cols = Some(column point)` on the lane-major path. There the
/// L0 rounds bind the lane (high) variables first, so the binding order of
/// a lane-major index is its coordinates rotated left by `log_n -
/// initial_k`; `eval_b_residual` receives the challenges in that order.
/// The L0 OOD points take their column coordinates from the opening point
/// (`lane_cols`, its low `log_n - initial_k` coordinates) and only their
/// row coordinates from the transcript (see [`AffineL0::Lanes`]).
#[allow(clippy::too_many_arguments)]
pub fn verify_affine_l0<Ch, F>(
    config: &VerifierConfig,
    proof: &AffineProof,
    log_n: usize,
    target: F128,
    expected_initial_cap: &[Hash],
    (l0_num_lanes, lane_cols): (usize, Option<&[F128]>),
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
    if lane_cols.is_some_and(|cols| cols.len() != log_n - config.initial_k) {
        return false;
    }
    for _ in 0..config.ood_samples[0] {
        let point = match lane_cols {
            // Binding order: the fresh row coordinates, then the columns.
            Some(cols) => {
                let mut point = challenger.sample_f128_vec(config.initial_k);
                point.extend_from_slice(cols);
                point
            }
            None => challenger.sample_f128_vec(log_n),
        };
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
        AffineL0, AffineProof, F128, affine_configs, affine_ladder, carry_products,
        novel_basis_evals, prove_affine, prove_affine_l0, t_at, verify_affine, verify_affine_l0,
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

    /// The L0 materialize equals the sequential affine folds of the lanes,
    /// for chunk sizes below and above the lane length.
    #[test]
    fn materialize_lanes_matches_sequential_folds() {
        let mut rng = RandomChallenger::new(0xAFF1_0006);
        let (c, k, live) = (9usize, 6usize, 46usize);
        let d = 1usize << c;
        let w = random_vec(&mut rng, live * d);
        let gammas: Vec<(F128, F128)> = (0..k)
            .map(|_| (rng.sample_f128(), rng.sample_f128()))
            .collect();
        let p = carry_products(c + 4);
        // Reference: fold row pairs round by round, f0 + α f1 + β T f1.
        let mut rows: Vec<Vec<F128>> = (0..1 << k)
            .map(|i| {
                if i < live {
                    w[i * d..(i + 1) * d].to_vec()
                } else {
                    vec![F128::ZERO; d]
                }
            })
            .collect();
        for &(alpha, beta) in &gammas {
            rows = rows
                .chunks(2)
                .map(|pair| {
                    let (a, b) = (&pair[0], &pair[1]);
                    let len = a.len();
                    (0..=len)
                        .map(|m| {
                            let base = if m < len {
                                a[m] + alpha * b[m]
                            } else {
                                F128::ZERO
                            };
                            base + beta * t_at(|j| b[j], len, m, &p)
                        })
                        .collect()
                })
                .collect();
        }
        for got in [
            super::materialize_lanes::<8>(&w, d, live, &gammas, &p),
            super::materialize_lanes::<64>(&w, d, live, &gammas, &p),
            super::materialize_lanes::<512>(&w, d, live, &gammas, &p),
            super::materialize_lanes::<4096>(&w, d, live, &gammas, &p),
        ] {
            let mut got = got;
            got.resize(d + k, F128::ZERO);
            assert_eq!(got, rows[0]);
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
            let check_cols = |proof: &AffineProof, target: F128, cols: &[F128]| {
                let mut ch = FsChallenger::new(b"affine-lanes");
                verify_affine_l0(
                    &verifier,
                    proof,
                    log_n,
                    target,
                    &cap,
                    (live, Some(cols)),
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
            let check = |proof: &AffineProof, target: F128| check_cols(proof, target, &point[..c0]);
            assert!(check(&proof, target), "lane-major roundtrip, ood = {ood}");
            if ood > 0 {
                let mut cols = point[..c0].to_vec();
                cols[0] += F128::ONE;
                assert!(!check_cols(&proof, target, &cols), "OOD column point");
            }
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

    /// At the split ladder's rates, the recomputed Johnson queries are the
    /// shipped m32_fast counts; at L1 rate 1/4 they rise to 119.
    #[test]
    fn ladder_queries_match_the_shipped_analysis() {
        let split = crate::pcs::ligerito::prover_config_for(
            25,
            6,
            crate::pcs::ligerito::LigeritoProfile::Fast,
        )
        .expect("m32 fast");
        let (same, _) = affine_ladder(&split, 25, &split.log_inv_rates[..5], &[0; 5]);
        assert_eq!(same.queries, split.queries[..5]);
        assert_eq!(same.recursive_ks, [4, 4, 4, 4]);
        let (fast, _) = affine_ladder(&split, 25, &[1, 2, 5, 7], &[4, 4, 5, 5]);
        assert_eq!(fast.queries, [244, 119, 48, 35]);
        assert_eq!(fast.recursive_log_msg_cols, [15, 11, 7]);
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
