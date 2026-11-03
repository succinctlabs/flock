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

use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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
fn carry_products(dim: usize) -> Vec<F128> {
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
    debug_assert_eq!(omega.len(), row.len());
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
    let log_n = witness.len().trailing_zeros() as usize;
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
    let lanes_0 = 1usize << shapes[0].0;
    assert_eq!(l0_codeword.len(), block_len_0 * lanes_0);
    let initial_cap = cap_layer(l0_tree, block_len_0, strat(0).cap_depth()).to_vec();
    challenger.observe_bytes(initial_cap.as_flattened());

    let p = carry_products(log_n + 1);
    let mut level = Level::new(witness, basis, shapes[0].0, target);
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

    let mut committed: Vec<LigeroWitness> = Vec::with_capacity(r);
    for li in 0..=r {
        let (k, c) = shapes[li];
        assert_eq!(level.rows, 1usize << k);
        assert_eq!(level.width, 1usize << c);
        let t = Instant::now();
        let mut gammas = Vec::with_capacity(k);
        for _ in 0..k {
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

/// Verify an [`AffineProof`]. `eval_b_residual(rs, residual_log)` returns the
/// caller's basis bound at the fold challenges `rs` (LSB first), as a table
/// over the `residual_log` remaining variables.
#[allow(clippy::too_many_arguments)]
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

    for _ in 0..config.ood_samples[0] {
        let point = challenger.sample_f128_vec(log_n);
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
                1usize << k,
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
            1usize << k,
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
        AffineProof, F128, affine_configs, carry_products, novel_basis_evals, prove_affine, t_at,
        verify_affine,
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
