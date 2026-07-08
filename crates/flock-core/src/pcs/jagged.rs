//! Jagged polynomial commitment — the sparse→dense reduction (standalone core).
//!
//! Implements the "basic jagged" reduction of Hemo–Jue–Rabinovich–Roh–Rothblum
//! ("Jagged Polynomial Commitments", 2025/917) over `F128`. A *jagged function*
//! `p : {0,1}^n × {0,1}^k → F` is a `2^n × 2^k` table in which column `y` is
//! nonzero only below its height `h_y`. Its nonzero entries are flattened, in
//! column-major order, into a single *dense* multilinear `q : {0,1}^m → F`
//! (`2^m ≥ Σ_y h_y`). This module reduces an evaluation claim on the sparse
//! `p̂(z_r, z_c)` to a single evaluation claim `q̂(i*) = α` on the dense `q`,
//! which a downstream multilinear PCS would discharge.
//!
//! This is the **packing-agnostic kernel**: it operates on an abstract dense
//! `F128` multilinear `q`, the cumulative column heights, and points
//! `(z_r, z_c)`. It does *not* wire into ring-switch / ligerito / the
//! arithmetization — that composition is deliberately deferred.
//!
//! ## The reduction (paper §3)
//!
//! With cumulative heights `t_y = h_0 + … + h_y` and the bijection
//! `i ↦ (row_t(i), col_t(i))` between dense indices and nonzero coordinates,
//!
//! ```text
//!   p̂(z_r, z_c) = Σ_{i ∈ {0,1}^m} q(i) · f̂_t(z_r, z_c, i)          (Eq. 3)
//!   f̂_t(z_r, z_c, i) = eq(row_t(i), z_r) · eq(col_t(i), z_c)        (Eq. 4, boolean i only)
//! ```
//!
//! We run a product-of-two-multilinears sumcheck on the right-hand side. The
//! prover materializes `B[i] = eq(row_t(i), z_r)·eq(col_t(i), z_c)` over the
//! boolean cube via two `eq`-tables. At the end the verifier needs
//! `f̂_t(z_r, z_c, i*)` at the *field* point `i*` — where Eq. (4) no longer
//! holds — and computes it through the branching-program evaluator below.
//!
//! ## Evaluating `f̂_t` at a field point (paper §3.1)
//!
//! By Claim 3.2.1, `f̂_t(z_r, z_c, i) = Σ_{y} eq(z_c, y) · ĝ(z_r, i, t_{y-1}, t_y)`,
//! where `g(a,b,c,d) = [b < d ∧ b = a + c]` is computed by a width-4 read-once
//! branching program (registers: an addition carry bit and a "less-than-so-far"
//! bit). `ĝ` is its multilinear extension, evaluated by the Holmgren–Rothblum
//! layer-by-layer DP over the 4 reachable states. Here `a = z_r` (row, `n`
//! bits, zero-padded to `m`), `b = i` (dense index, `m` bits), and
//! `c = t_{y-1}`, `d = t_y` are the (boolean, constant) cumulative heights.
//!
//! ## The jagged assist (paper §1.1.1 / §5)
//!
//! Direct `f̂_t` evaluation costs the verifier `2^k` branching-program DPs —
//! `O(2^k·m)` multiplications with a large constant, and height-dependent
//! control flow that is hostile to recursion. The *assist* delegates it to the
//! prover: with `G(c,d) := ĝ(z_r, i*, c, d)` (row/index points pinned as
//! constants) and the weight multilinear
//!
//! ```text
//!   W(c,d) = Σ_y eq(z_c, y) · eq((t_{y-1}, t_y), (c,d)),
//! ```
//!
//! `β = f̂_t(z_r, z_c, i*) = Σ_{(c,d) ∈ {0,1}^{2(m+1)}} W(c,d)·G(c,d)` — a
//! product-of-two-multilinears sumcheck over only the `2(m+1)` cumulative-height
//! variables. We prove the `eq(z_c,·)`-weighted sum directly (one claim, no
//! per-column values, no batching randomness — the statement is a fixed scalar,
//! so plain sumcheck soundness applies); SP1 Hypercube's `slop/jagged` makes
//! the same choice. Because the `x_y = (t_{y-1}, t_y)` are boolean, each round
//! message needs only one partially-bound `G` evaluation per column
//! (Lemma 5.1's collapse), and columns with equal `(t_{y-1}, t_y)` — zero
//! heights — are merged up front, so the prover pays per *distinct* pair.
//!
//! Variables bind in **layer-interleaved order** `c_0, d_0, c_1, d_1, …`
//! (LSB-first, matching the branching program's read order), which lets the
//! prover use Lemma 4.6 prefix/suffix streaming ([`prove_assist`]): per-column
//! suffix vectors stored layer-major, sparse two-entry transition rows, and an
//! advancing prefix row vector reduce each layer to a single 6-multiplication-
//! per-column bucketing pass from which **both** round messages derive —
//! `O(m·2^k)` total. The naive per-round DP prover is retained as a
//! transcript-identical reference. The verifier finishes with one
//! `Ĝ(ρ)` DP plus `W(ρ)` at `2(m+1)` multiplications per distinct column —
//! `~35×` fewer multiplications than direct `f̂_t` at `m=25, k=10`, and no
//! height-dependent branching. Round messages use the codebase's char-2-safe
//! `(G(1), G(∞))` encoding (SP1's `{0, ½, 1}` interpolation needs `2⁻¹`, which
//! does not exist in `F128`).

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;

/// Configuration of a jagged function: the (zero-padded to `2^k`) column
/// heights, summarized as the cumulative-height prefix sums.
#[derive(Clone, Debug)]
pub struct JaggedParams {
    /// `log2` of the height bound (number of row variables of `p̂`).
    pub n: usize,
    /// `log2` of the number of columns (column variables of `p̂`).
    pub k: usize,
    /// `log2` of the dense area: `q` has `2^m` entries, `Σ_y h_y ≤ 2^m`.
    pub m: usize,
    /// Cumulative heights `[t_{-1}=0, t_0, t_1, …, t_{2^k-1}=area]`, length
    /// `2^k + 1`. Column `c` occupies dense indices `[col_prefix_sums[c],
    /// col_prefix_sums[c+1])`.
    pub col_prefix_sums: Vec<u64>,
}

impl JaggedParams {
    /// Build params from per-column heights. `heights.len()` must be `2^k`
    /// (zero-pad empty columns up to a power of two yourself). Requires each
    /// height `≤ 2^n` and total area `≤ 2^m`.
    pub fn from_heights(heights: &[u64], n: usize, m: usize) -> Self {
        assert!(
            heights.len().is_power_of_two(),
            "number of columns must be a power of two (zero-pad)"
        );
        let k = heights.len().trailing_zeros() as usize;
        let mut col_prefix_sums = Vec::with_capacity(heights.len() + 1);
        let mut acc: u64 = 0;
        col_prefix_sums.push(0);
        for &h in heights {
            assert!(h <= (1u64 << n), "column height exceeds 2^n");
            acc += h;
            col_prefix_sums.push(acc);
        }
        assert!(acc <= (1u64 << m), "total area exceeds 2^m");
        JaggedParams {
            n,
            k,
            m,
            col_prefix_sums,
        }
    }

    /// Total number of nonzero entries `Σ_y h_y`.
    pub fn area(&self) -> u64 {
        *self.col_prefix_sums.last().unwrap()
    }

    /// The bijection `i ↦ (row_t(i), col_t(i))` for a dense index `i < area`:
    /// `col` is the column whose range contains `i`, `row = i - t_{col-1}`.
    pub fn unrank(&self, i: u64) -> (usize, usize) {
        debug_assert!(i < self.area());
        // First prefix-sum strictly greater than `i`, minus one, is the column.
        let col = self.col_prefix_sums.partition_point(|&t| t <= i) - 1;
        let row = i - self.col_prefix_sums[col];
        (row as usize, col)
    }
}

/// Bit `layer` of the field "point" `z`: the coordinate `z[layer]` if present,
/// else `ZERO` (the variable is pinned to 0 — i.e. zero-padded).
#[inline]
fn point_bit(z: &[F128], layer: usize) -> F128 {
    if layer < z.len() {
        z[layer]
    } else {
        F128::ZERO
    }
}

/// Bit `layer` of the integer `t`, as a field element.
#[inline]
fn int_bit(t: u64, layer: usize) -> F128 {
    if (t >> layer) & 1 == 1 {
        F128::ONE
    } else {
        F128::ZERO
    }
}

/// Width-4 branching-program transition for `g(a,b,c,d) = [b<d ∧ b=a+c]`,
/// reading one bit position (LSB→MSB). Input bits: `row=a`, `index=b`,
/// `curr=c`, `next=d`. `state = carry + 2·comparison`. Returns the next state
/// index, or `None` on the rejecting sink (addition inconsistency).
#[inline]
fn transition(row: bool, index: bool, curr: bool, next: bool, state: usize) -> Option<usize> {
    let carry = state & 1;
    let comparison = (state >> 1) & 1;
    // Addition check: index bit must equal LSB of (row + carry + curr).
    let sum = row as usize + carry + curr as usize;
    if (index as usize) != (sum & 1) {
        return None;
    }
    let new_carry = sum >> 1;
    // i < t_{c+1}: if this bit of index and next agree, defer; else the higher
    // bit decides (less-than iff next=1, index=0).
    let new_comparison = if index == next {
        comparison
    } else {
        next as usize
    };
    Some(new_carry + (new_comparison << 1))
}

const STATE_INITIAL: usize = 0; // carry=0, comparison=0
const STATE_SUCCESS: usize = 2; // carry=0, comparison=1

/// Multilinear extension `ĝ(z_r, z_i, c, d)` of the branching program, with
/// the per-layer height coordinates supplied by `cd(layer)` as arbitrary field
/// values. Holmgren–Rothblum layer-by-layer DP over the 4 reachable states;
/// `O(m)` field ops.
fn g_hat_eval_cd(
    z_row: &[F128],
    z_index: &[F128],
    m: usize,
    cd: impl Fn(usize) -> (F128, F128),
) -> F128 {
    // dp[s] = weight, over already-processed (upper) layers, of reaching the
    // accepting sink from state `s`. Seed the accepting state, peel layers from
    // MSB down to LSB, and read off the initial state.
    let mut dp = [F128::ZERO; 4];
    dp[STATE_SUCCESS] = F128::ONE;
    for layer in (0..=m).rev() {
        let (c, d) = cd(layer);
        let eq16 = build_eq_table(&[
            point_bit(z_row, layer),
            point_bit(z_index, layer),
            c,
            d,
        ]);
        let mut new_dp = [F128::ZERO; 4];
        for (s, slot) in new_dp.iter_mut().enumerate() {
            let mut acc = F128::ZERO;
            for (idx, &w) in eq16.iter().enumerate() {
                // idx bit 0 = row, 1 = index, 2 = curr (t_c), 3 = next (t_next).
                let row = idx & 1 != 0;
                let index = (idx >> 1) & 1 != 0;
                let curr = (idx >> 2) & 1 != 0;
                let next = (idx >> 3) & 1 != 0;
                if let Some(out) = transition(row, index, curr, next, s) {
                    acc += w * dp[out];
                }
            }
            *slot = acc;
        }
        dp = new_dp;
    }
    dp[STATE_INITIAL]
}

/// [`g_hat_eval_cd`] specialized to boolean cumulative heights `t_c, t_next`.
fn g_hat_eval(z_row: &[F128], z_index: &[F128], t_c: u64, t_next: u64, m: usize) -> F128 {
    g_hat_eval_cd(z_row, z_index, m, |layer| {
        (int_bit(t_c, layer), int_bit(t_next, layer))
    })
}

/// Evaluate `f̂_t(z_r, z_c, z_i)` at an arbitrary field point, via the
/// branching-program assembly `Σ_y eq(z_c, y)·ĝ(z_r, z_i, t_{y-1}, t_y)`
/// (paper Claim 3.2.1). Cost `O(m · 2^k)`.
pub fn f_hat_t(params: &JaggedParams, z_row: &[F128], z_col: &[F128], z_index: &[F128]) -> F128 {
    assert_eq!(z_row.len(), params.n);
    assert_eq!(z_col.len(), params.k);
    assert_eq!(z_index.len(), params.m);
    let eq_col = build_eq_table(z_col);
    let cols = 1usize << params.k;
    let mut acc = F128::ZERO;
    for c in 0..cols {
        let g = g_hat_eval(
            z_row,
            z_index,
            params.col_prefix_sums[c],
            params.col_prefix_sums[c + 1],
            params.m,
        );
        acc += eq_col[c] * g;
    }
    acc
}

/// Transcript of the jagged sumcheck. Each round sends the degree-2 round
/// polynomial as `(G(1), G(∞))`; `G(0)` is reconstructed by the verifier from
/// the running claim. `q_eval` is the final dense claim `α = q̂(i*)`.
#[derive(Clone, Debug)]
pub struct JaggedSumcheckProof {
    pub rounds: Vec<(F128, F128)>,
    pub q_eval: F128,
}

/// The dense evaluation claim that the jagged reduction produces: prove
/// `q̂(point) = alpha` with a downstream multilinear PCS.
#[derive(Clone, Debug)]
pub struct DenseClaim {
    pub point: Vec<F128>,
    pub alpha: F128,
}

/// Generate the second sumcheck multilinear `B[i] = eq(row_t(i), z_row) ·
/// eq(col_t(i), z_col)` over the boolean cube (zero past `area`), together with
/// the claim `v = Σ_i q(i)·B(i) = p̂(z_row, z_col)` — fused into one parallel
/// pass over the `2^m` entries.
///
/// Each rayon chunk binary-searches its starting column once, then walks the
/// (contiguous, jagged) columns filling `B` and accumulating its share of `v`.
/// The column walk skips height-0 columns naturally and costs O(1) amortized per
/// element, so there is no per-element binary search.
fn generate_f_and_claim(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
) -> (Vec<F128>, F128) {
    use rayon::prelude::*;
    let len = 1usize << params.m;
    let area = params.area() as usize;
    let eq_row = build_eq_table(z_row);
    let eq_col = build_eq_table(z_col);
    let prefix = &params.col_prefix_sums;
    let mut b = crate::alloc_uninit_f128_vec(len);

    // ~1 MB chunks: one binary search amortized over 64K elements.
    const CHUNK: usize = 1 << 16;
    let v = b
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(ci, b_chunk)| {
            let g0 = ci * CHUNK;
            let q_chunk = &q[g0..g0 + b_chunk.len()];
            // Column containing g0 (== num_columns sentinel once g0 ≥ area).
            let mut col = prefix
                .partition_point(|&t| t <= g0 as u64)
                .saturating_sub(1);
            let mut acc = F128::ZERO;
            for (local, slot) in b_chunk.iter_mut().enumerate() {
                let i = g0 + local;
                if i >= area {
                    *slot = F128::ZERO;
                    continue;
                }
                while (i as u64) >= prefix[col + 1] {
                    col += 1;
                }
                let row = i - prefix[col] as usize;
                let bi = eq_row[row] * eq_col[col];
                *slot = bi;
                acc += q_chunk[local] * bi;
            }
            acc
        })
        .reduce(|| F128::ZERO, |x, y| x + y);
    (b, v)
}

/// Prover for the jagged reduction. Given the dense multilinear `q` (length
/// `2^m`, column-major flattening of the jagged function, zero-padded past
/// `area`) and the sparse evaluation point `(z_row, z_col)`, runs the sumcheck
/// and returns the proof together with the sparse claim value
/// `v = p̂(z_row, z_col)`.
pub fn prove<C: Challenger>(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
    challenger: &mut C,
) -> (JaggedSumcheckProof, F128) {
    let (proof, v, _point) = prove_main(params, q, z_row, z_col, challenger);
    (proof, v)
}

/// [`prove`], additionally returning the bound point `i*` (the per-round
/// challenges, low bit first) — needed to continue the transcript into the
/// assist sub-protocol or the downstream dense opening.
fn prove_main<C: Challenger>(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
    challenger: &mut C,
) -> (JaggedSumcheckProof, F128, Vec<F128>) {
    let m = params.m;
    let len = 1usize << m;
    assert_eq!(q.len(), len, "q must have 2^m entries");
    assert_eq!(z_row.len(), params.n);
    assert_eq!(z_col.len(), params.k);
    challenger.observe_label(b"flock-jagged-v0");

    // Second sumcheck multilinear B[i] = eq(row_t(i), z_row)·eq(col_t(i), z_col)
    // over the boolean cube (= f̂_t(z_row, z_col, ·) on {0,1}^m), and the claim
    // v = Σ_i q(i)·B(i) = p̂(z_row, z_col) — one fused parallel pass.
    let (b, v) = generate_f_and_claim(params, q, z_row, z_col);

    // Product-of-two-multilinears sumcheck, binding the low index bit each
    // round — parallel and fused: each fold pass also computes the next round's
    // message, halving passes over the (bandwidth-bound) witness. We ping-pong
    // between `a/bb` and the scratch `sa/sb`. F128 addition is XOR, so the
    // parallel tree reduction is bit-identical to a serial fold.
    let mut a = q.to_vec();
    let mut bb = b;
    let mut sa = crate::alloc_uninit_f128_vec(len / 2);
    let mut sb = crate::alloc_uninit_f128_vec(len / 2);
    let mut cur = len;
    let mut rounds = Vec::with_capacity(m);
    let mut point = Vec::with_capacity(m);
    let (mut g_one, mut g_inf) = round_msg_par(&a[..cur], &bb[..cur]);
    for _ in 0..m {
        let half = cur / 2;
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        rounds.push((g_one, g_inf));
        point.push(r);
        if cur > 2 {
            (g_one, g_inf) =
                fold_and_round_oop_par(&a[..cur], &bb[..cur], r, &mut sa[..half], &mut sb[..half]);
        } else {
            fold_oop_par(&a[..cur], &bb[..cur], r, &mut sa[..half], &mut sb[..half]);
        }
        std::mem::swap(&mut a, &mut sa);
        std::mem::swap(&mut bb, &mut sb);
        cur = half;
    }

    debug_assert_eq!(cur, 1);
    let proof = JaggedSumcheckProof {
        rounds,
        q_eval: a[0],
    };
    (proof, v, point)
}

/// Verifier for the jagged reduction. Replays the sumcheck against the claimed
/// sparse value `claim_v = p̂(z_row, z_col)`, computes `f̂_t` at the final
/// point through the branching program, and on success returns the reduced
/// dense claim `q̂(i*) = alpha`. Returns `None` if the proof is rejected.
pub fn verify<C: Challenger>(
    params: &JaggedParams,
    z_row: &[F128],
    z_col: &[F128],
    claim_v: F128,
    proof: &JaggedSumcheckProof,
    challenger: &mut C,
) -> Option<DenseClaim> {
    challenger.observe_label(b"flock-jagged-v0");
    let (point, claim) = replay_rounds(claim_v, proof, params.m, challenger)?;

    // Final sumcheck relation: claim == q̂(i*) · f̂_t(z_row, z_col, i*).
    let beta = f_hat_t(params, z_row, z_col, &point);
    if claim == proof.q_eval * beta {
        Some(DenseClaim {
            point,
            alpha: proof.q_eval,
        })
    } else {
        None
    }
}

/// Replay the `m` sumcheck rounds against the claimed value, folding the claim
/// and collecting the bound point `i*`. `None` on a length mismatch.
fn replay_rounds<C: Challenger>(
    claim_v: F128,
    proof: &JaggedSumcheckProof,
    m: usize,
    challenger: &mut C,
) -> Option<(Vec<F128>, F128)> {
    if proof.rounds.len() != m {
        return None;
    }
    let mut claim = claim_v;
    let mut point = Vec::with_capacity(m);
    for &(g_one, g_inf) in &proof.rounds {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        claim = fold_round_claim(claim, g_one, g_inf, r);
        point.push(r);
    }
    Some((point, claim))
}

// ───────────────────────────────────────────────────────────────────────────
// The jagged assist (module docs above; paper §5)
// ───────────────────────────────────────────────────────────────────────────

/// Transcript of the assist sumcheck, proving `beta = f̂_t(z_row, z_col, i*)`
/// so the verifier replaces `2^k` branching-program DPs with one. `beta` is
/// the claimed value (observed into the transcript before the rounds); each of
/// the `2(m+1)` rounds sends the degree-2 message `(G(1), G(∞))`.
#[derive(Clone, Debug)]
pub struct JaggedAssistProof {
    pub beta: F128,
    pub rounds: Vec<(F128, F128)>,
}

/// The assist's per-column terms `(w_y, t_{y-1}, t_y)`, with runs of columns
/// sharing the same `(t_{y-1}, t_y)` pair — zero-height columns, including the
/// zero-padded tail — collapsed into one term of summed weight `Σ eq(z_col, y)`.
/// Pure regrouping of identical summands: transcript-invariant.
fn assist_columns(params: &JaggedParams, z_col: &[F128]) -> Vec<(F128, u64, u64)> {
    let eq_col = build_eq_table(z_col);
    let mut out: Vec<(F128, u64, u64)> = Vec::with_capacity(eq_col.len());
    for (y, &w) in eq_col.iter().enumerate() {
        let (t_c, t_next) = (params.col_prefix_sums[y], params.col_prefix_sums[y + 1]);
        match out.last_mut() {
            Some((w_acc, c, d)) if *c == t_c && *d == t_next => *w_acc += w,
            _ => out.push((w, t_c, t_next)),
        }
    }
    out
}

/// The weight multilinear `W` at the assist's final point:
/// `W(ρ) = Σ_y w_y · Π_ℓ eq(t_{y-1}[ℓ], ρ_{c,ℓ}) · eq(t_y[ℓ], ρ_{d,ℓ})`, with
/// `ρ` in the interleaved order `(c_0, d_0, c_1, d_1, …)`. `eq(b, r)` at a
/// boolean `b` is `r` or `1 + r` (char 2), so this is `2(m+1)` multiplications
/// per distinct column — the verifier's only `2^k`-scale work.
fn assist_w_at(cols: &[(F128, u64, u64)], rho: &[F128], m: usize) -> F128 {
    debug_assert_eq!(rho.len(), 2 * (m + 1));
    let mut acc = F128::ZERO;
    for &(w, t_c, t_next) in cols {
        let mut term = w;
        for layer in 0..=m {
            let rc = rho[2 * layer];
            let rd = rho[2 * layer + 1];
            term *= if (t_c >> layer) & 1 == 1 { rc } else { F128::ONE + rc };
            term *= if (t_next >> layer) & 1 == 1 { rd } else { F128::ONE + rd };
        }
        acc += term;
    }
    acc
}

/// Column-chunk size for the assist's parallel passes: coarse enough to
/// amortize rayon task overhead at typical column counts (2^k in the
/// hundreds–thousands), fine enough to load-balance a P-core pool.
const ASSIST_CHUNK: usize = 256;

/// The two surviving transitions of each `(c + 2d, state)` row of a layer
/// matrix: the addition check forces the index bit `b` once `a` is chosen, so
/// each row has exactly two entries `(index into the layer's eq4 table, next
/// state)` — and they are layer-independent (a layer only supplies its eq4
/// table `eq((z_row[ℓ], z_index[ℓ]), ·)`).
fn assist_sparse_transitions() -> [[[(usize, usize); 2]; 4]; 4] {
    let mut table = [[[(0usize, 0usize); 2]; 4]; 4];
    for (cd, rows) in table.iter_mut().enumerate() {
        let (c, d) = (cd & 1 != 0, cd & 2 != 0);
        for (s, row) in rows.iter_mut().enumerate() {
            for (a, entry) in row.iter_mut().enumerate() {
                let b = (a + (s & 1) + c as usize) & 1 == 1;
                let out =
                    transition(a == 1, b, c, d, s).expect("the forced index bit never rejects");
                *entry = (a + 2 * (b as usize), out);
            }
        }
    }
    table
}

/// All columns' suffix vectors `S_y[ℓ] = M_ℓ(bits_y)···M_m(bits_y)·e_S`, laid
/// out **layer-major** (`rows[ℓ·n_cols + y]`) so each sumcheck round streams
/// one contiguous row. Built with one parallel pass per layer, `m → 0`; a
/// column costs 8 multiplications per layer (two surviving transitions per
/// state). Row 0's `INITIAL` entries are the columns' full `ĝ` values.
fn assist_suffix_rows(
    cols: &[(F128, u64, u64)],
    eq4s: &[[F128; 4]],
    sparse: &[[[(usize, usize); 2]; 4]; 4],
    m: usize,
) -> Vec<[F128; 4]> {
    use rayon::prelude::*;
    let n_cols = cols.len();
    let mut rows = vec![[F128::ZERO; 4]; (m + 2) * n_cols];
    for seed in &mut rows[(m + 1) * n_cols..] {
        seed[STATE_SUCCESS] = F128::ONE;
    }
    for layer in (0..=m).rev() {
        let (head, tail) = rows.split_at_mut((layer + 1) * n_cols);
        let dst = &mut head[layer * n_cols..];
        let src = &tail[..n_cols];
        let eq4 = &eq4s[layer];
        dst.par_chunks_mut(ASSIST_CHUNK)
            .zip(src.par_chunks(ASSIST_CHUNK))
            .zip(cols.par_chunks(ASSIST_CHUNK))
            .for_each(|((dc, sc), cc)| {
                for ((dv, sv), &(_, t_c, t_next)) in dc.iter_mut().zip(sc).zip(cc) {
                    let cd = ((t_c >> layer) & 1) as usize + 2 * ((t_next >> layer) & 1) as usize;
                    let rows_cd = &sparse[cd];
                    for (s, slot) in dv.iter_mut().enumerate() {
                        let (i0, o0) = rows_cd[s][0];
                        let (i1, o1) = rows_cd[s][1];
                        *slot = eq4[i0] * sv[o0] + eq4[i1] * sv[o1];
                    }
                }
            });
    }
    rows
}

#[inline]
fn dot4(u: &[F128; 4], v: &[F128; 4]) -> F128 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
}

#[inline]
fn add4(a: &[F128; 4], b: &[F128; 4]) -> [F128; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// `x·a + y·b`, component-wise.
#[inline]
fn comb4(x: F128, a: &[F128; 4], y: F128, b: &[F128; 4]) -> [F128; 4] {
    [
        x * a[0] + y * b[0],
        x * a[1] + y * b[1],
        x * a[2] + y * b[2],
        x * a[3] + y * b[3],
    ]
}

/// Prover for the assist sumcheck: proves `β = f̂_t(z_row, z_col, z_index)` =
/// `Σ_{(c,d)} W(c,d)·ĝ(z_row, z_index, c, d)` over the `2(m+1)` height
/// variables, bound in interleaved order `c_0, d_0, c_1, d_1, …` (LSB first).
///
/// Lemma 4.6 streaming ("assist with storage"), one parallel pass per
/// **layer**: the pass folds the previous layer's two challenges into each
/// column's running weight `we_y = w_y·E_y` and accumulates the four bucketed
/// sums `B[cbit + 2·dbit] = Σ_y we_y·S_y[ℓ+1]` — 6 multiplications per column,
/// streaming one contiguous suffix row ([`assist_suffix_rows`]). Both round
/// messages then come from the buckets alone:
///
/// ```text
///   Ĝ_y(x) = b_ℓᵀ · M_ℓ(mixed with x) · S_y[ℓ+1]
///   c-round:  G(1) = u₁ᵀB₁ + u₃ᵀB₃,   G(∞) = (u₀+u₁)ᵀ(B₀+B₁) + (u₂+u₃)ᵀ(B₂+B₃)
///   d-round:  M_ℓ(r_c, x) is a linear combination of the boolean matrices, so
///             folding r_c into the u's and B's gives the message — no second
///             column pass.
/// ```
///
/// Here `u[cd]ᵀ = b_ℓᵀ·M_ℓ^{(c,d)}` are shared row vectors and the prefix
/// `b_ℓᵀ = e_Iᵀ·M_0(ρ)···M_{ℓ-1}(ρ)` advances once per layer. `O(m·2^k)`
/// multiplications total instead of the naive `O(m²·2^k)`
/// ([`prove_assist_naive`], which produces a bit-identical transcript).
pub fn prove_assist<C: Challenger>(
    params: &JaggedParams,
    z_row: &[F128],
    z_col: &[F128],
    z_index: &[F128],
    challenger: &mut C,
) -> JaggedAssistProof {
    use rayon::prelude::*;
    let m = params.m;
    assert_eq!(z_row.len(), params.n);
    assert_eq!(z_col.len(), params.k);
    assert_eq!(z_index.len(), m);
    let cols = assist_columns(params, z_col);
    let n_cols = cols.len();

    let eq4s: Vec<[F128; 4]> = (0..=m)
        .map(|layer| {
            let t = build_eq_table(&[point_bit(z_row, layer), point_bit(z_index, layer)]);
            [t[0], t[1], t[2], t[3]]
        })
        .collect();
    let sparse = assist_sparse_transitions();
    let sfx = assist_suffix_rows(&cols, &eq4s, &sparse, m);

    // β = Σ_y w_y·ĝ_y — the INITIAL entries of suffix row 0.
    let beta = cols
        .par_iter()
        .zip(sfx[..n_cols].par_iter())
        .map(|(&(w, _, _), s)| w * s[STATE_INITIAL])
        .reduce(|| F128::ZERO, |x, y| x + y);

    challenger.observe_label(b"flock-jagged-assist-v0");
    challenger.observe_f128(beta);

    let mut prefix_row = [F128::ZERO; 4];
    prefix_row[STATE_INITIAL] = F128::ONE;
    let mut we: Vec<F128> = cols.iter().map(|&(w, _, _)| w).collect();
    let mut prev_ch: Option<(F128, F128)> = None;
    let mut rounds = Vec::with_capacity(2 * (m + 1));
    for layer in 0..=m {
        let row = &sfx[(layer + 1) * n_cols..(layer + 2) * n_cols];

        // The layer's only column pass.
        let buckets = we
            .par_chunks_mut(ASSIST_CHUNK)
            .zip(cols.par_chunks(ASSIST_CHUNK))
            .zip(row.par_chunks(ASSIST_CHUNK))
            .map(|((wc, cc), sc)| {
                let mut b = [[F128::ZERO; 4]; 4];
                for ((w_e, &(_, t_c, t_next)), s) in wc.iter_mut().zip(cc).zip(sc) {
                    if let Some((rc, rd)) = prev_ch {
                        let pl = layer - 1;
                        let ec = if (t_c >> pl) & 1 == 1 { rc } else { F128::ONE + rc };
                        let ed = if (t_next >> pl) & 1 == 1 { rd } else { F128::ONE + rd };
                        *w_e *= ec * ed;
                    }
                    let cd = ((t_c >> layer) & 1) as usize + 2 * ((t_next >> layer) & 1) as usize;
                    let v = *w_e;
                    let bk = &mut b[cd];
                    bk[0] += v * s[0];
                    bk[1] += v * s[1];
                    bk[2] += v * s[2];
                    bk[3] += v * s[3];
                }
                b
            })
            .reduce(
                || [[F128::ZERO; 4]; 4],
                |mut x, y| {
                    for (xv, yv) in x.iter_mut().zip(&y) {
                        *xv = add4(xv, yv);
                    }
                    x
                },
            );

        // u[c + 2d]ᵀ = b_ℓᵀ·M_ℓ^{(c,d)}, via the sparse transition rows.
        let eq4 = &eq4s[layer];
        let mut u = [[F128::ZERO; 4]; 4];
        for (cd, uv) in u.iter_mut().enumerate() {
            for (s, &bs) in prefix_row.iter().enumerate() {
                let (i0, o0) = sparse[cd][s][0];
                let (i1, o1) = sparse[cd][s][1];
                uv[o0] += bs * eq4[i0];
                uv[o1] += bs * eq4[i1];
            }
        }

        // c-round.
        let g_one = dot4(&u[1], &buckets[1]) + dot4(&u[3], &buckets[3]);
        let g_inf = dot4(&add4(&u[0], &u[1]), &add4(&buckets[0], &buckets[1]))
            + dot4(&add4(&u[2], &u[3]), &add4(&buckets[2], &buckets[3]));
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let rc = challenger.sample_f128();
        rounds.push((g_one, g_inf));

        // d-round from the same buckets: ud[x]ᵀ = b_ℓᵀ·M_ℓ(rc, x) and
        // D[db] = Σ_{y: dbit=db} we·eq(cbit_y, rc)·S_y, both by folding rc.
        let rc1 = F128::ONE + rc;
        let ud0 = comb4(rc1, &u[0], rc, &u[1]);
        let ud1 = comb4(rc1, &u[2], rc, &u[3]);
        let d0 = comb4(rc1, &buckets[0], rc, &buckets[1]);
        let d1 = comb4(rc1, &buckets[2], rc, &buckets[3]);
        let g_one = dot4(&ud1, &d1);
        let g_inf = dot4(&add4(&ud0, &ud1), &add4(&d0, &d1));
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let rd = challenger.sample_f128();
        rounds.push((g_one, g_inf));

        // Advance the prefix past the now fully-bound layer:
        // b_{ℓ+1}ᵀ = b_ℓᵀ·M_ℓ(rc, rd) = (1+rd)·ud[0] + rd·ud[1].
        prefix_row = comb4(F128::ONE + rd, &ud0, rd, &ud1);
        prev_ch = Some((rc, rd));
    }

    JaggedAssistProof { beta, rounds }
}

/// Naive (SP1-style) reference for [`prove_assist`]: the eq side of each
/// column is maintained incrementally (`prefix_eq`), while the `ĝ` side is
/// re-evaluated per round with the full layer DP — `O(m²·2^k)` multiplications
/// overall. Produces a transcript **bit-identical** to the streaming prover
/// (same algebra over exact field ops); retained as the correctness reference
/// (`assist_streamed_matches_naive`) and for the `runtime_assist_m25`
/// comparison.
#[allow(dead_code)]
fn prove_assist_naive<C: Challenger>(
    params: &JaggedParams,
    z_row: &[F128],
    z_col: &[F128],
    z_index: &[F128],
    challenger: &mut C,
) -> JaggedAssistProof {
    use rayon::prelude::*;
    let m = params.m;
    assert_eq!(z_row.len(), params.n);
    assert_eq!(z_col.len(), params.k);
    assert_eq!(z_index.len(), m);
    let cols = assist_columns(params, z_col);

    // The claimed value β, over the collapsed terms (same value as `f_hat_t`).
    let beta = cols
        .par_iter()
        .map(|&(w, t_c, t_next)| w * g_hat_eval(z_row, z_index, t_c, t_next, m))
        .reduce(|| F128::ZERO, |x, y| x + y);

    challenger.observe_label(b"flock-jagged-assist-v0");
    challenger.observe_f128(beta);

    let total_rounds = 2 * (m + 1);
    let mut rho: Vec<F128> = Vec::with_capacity(total_rounds);
    let mut prefix_eq = vec![F128::ONE; cols.len()];
    let mut rounds = Vec::with_capacity(total_rounds);
    for j in 0..total_rounds {
        let layer = j / 2;
        let bind_c = j % 2 == 0;
        // Round message: G(x) = Σ_y w·E_y·eq(bit_y, x)·Ĝ_y(x), where Ĝ_y(x) is
        // ĝ at (prefix = ρ, current variable = x, suffix = the column's bits)
        // and bit_y is the column's bit of the variable being bound. Both
        // factors are linear in x with eq's x-coefficient 1 (char 2), so
        // G(1) sums the bit_y = 1 columns and G(∞) sums Ĝ_y(0) + Ĝ_y(1).
        let (g_one, g_inf) = cols
            .par_iter()
            .zip(prefix_eq.par_iter())
            .map(|(&(w, t_c, t_next), &e)| {
                let eval = |x: F128| {
                    g_hat_eval_cd(z_row, z_index, m, |l| {
                        use std::cmp::Ordering::*;
                        match l.cmp(&layer) {
                            Less => (rho[2 * l], rho[2 * l + 1]),
                            Equal if bind_c => (x, int_bit(t_next, l)),
                            Equal => (rho[2 * l], x),
                            Greater => (int_bit(t_c, l), int_bit(t_next, l)),
                        }
                    })
                };
                let g0 = eval(F128::ZERO);
                let g1 = eval(F128::ONE);
                let we = w * e;
                let bit = ((if bind_c { t_c } else { t_next }) >> layer) & 1 == 1;
                let one_term = if bit { we * g1 } else { F128::ZERO };
                (one_term, we * (g0 + g1))
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a, b), (c, d)| (a + c, b + d),
            );

        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        rounds.push((g_one, g_inf));
        // Fold the bound bit into each column's running eq prefix:
        // eq(bit, r) = r or 1 + r.
        for (&(_, t_c, t_next), e) in cols.iter().zip(prefix_eq.iter_mut()) {
            let bit = ((if bind_c { t_c } else { t_next }) >> layer) & 1 == 1;
            *e *= if bit { r } else { F128::ONE + r };
        }
        rho.push(r);
    }

    JaggedAssistProof { beta, rounds }
}

/// Verifier for the assist sumcheck: replays the rounds against `proof.beta`
/// and checks the final relation `claim == W(ρ)·ĝ(z_row, z_index, ρ)` — one
/// branching-program DP plus the `assist_w_at` combination. On success returns
/// the now-verified `β = f̂_t(z_row, z_col, z_index)`.
pub fn verify_assist<C: Challenger>(
    params: &JaggedParams,
    z_row: &[F128],
    z_col: &[F128],
    z_index: &[F128],
    proof: &JaggedAssistProof,
    challenger: &mut C,
) -> Option<F128> {
    let m = params.m;
    if proof.rounds.len() != 2 * (m + 1) {
        return None;
    }
    challenger.observe_label(b"flock-jagged-assist-v0");
    challenger.observe_f128(proof.beta);

    let mut claim = proof.beta;
    let mut rho = Vec::with_capacity(2 * (m + 1));
    for &(g_one, g_inf) in &proof.rounds {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        claim = fold_round_claim(claim, g_one, g_inf, r);
        rho.push(r);
    }

    let cols = assist_columns(params, z_col);
    let w = assist_w_at(&cols, &rho, m);
    let g = g_hat_eval_cd(z_row, z_index, m, |l| (rho[2 * l], rho[2 * l + 1]));
    (claim == w * g).then_some(proof.beta)
}

/// [`prove`] followed by the assist sub-protocol at the sumcheck's final point.
/// Companion of [`verify_with_assist`].
pub fn prove_with_assist<C: Challenger>(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
    challenger: &mut C,
) -> (JaggedSumcheckProof, JaggedAssistProof, F128) {
    let (proof, v, point) = prove_main(params, q, z_row, z_col, challenger);
    let assist = prove_assist(params, z_row, z_col, &point, challenger);
    (proof, assist, v)
}

/// [`verify`] with the `f̂_t` evaluation discharged by the assist proof instead
/// of the `O(2^k)` direct computation.
pub fn verify_with_assist<C: Challenger>(
    params: &JaggedParams,
    z_row: &[F128],
    z_col: &[F128],
    claim_v: F128,
    proof: &JaggedSumcheckProof,
    assist: &JaggedAssistProof,
    challenger: &mut C,
) -> Option<DenseClaim> {
    challenger.observe_label(b"flock-jagged-v0");
    let (point, claim) = replay_rounds(claim_v, proof, params.m, challenger)?;
    let beta = verify_assist(params, z_row, z_col, &point, assist, challenger)?;
    if claim == proof.q_eval * beta {
        Some(DenseClaim {
            point,
            alpha: proof.q_eval,
        })
    } else {
        None
    }
}

/// Reduce the running sumcheck claim through one round. The degree-2 round
/// polynomial `G` is given by `G(1) = g_one`, leading coeff `G(∞) = g_inf`, and
/// `G(0) = claim + G(1)` (since `claim = G(0) + G(1)`). Returns `G(r)`.
#[inline]
fn fold_round_claim(claim: F128, g_one: F128, g_inf: F128, r: F128) -> F128 {
    let g0 = claim + g_one; // char-2: G(0) = claim - G(1)
    // G(X) = g0 + (G(1) + g0 + g_inf)·X + g_inf·X²
    g0 + (g_one + g0 + g_inf) * r + g_inf * (r * r)
}

/// Degree-2 round message `(G(1), G(∞))` for `Σ_{x'} a(X,x')·b(X,x')`, low bit
/// bound: `a(0,x') = a[2x']`, `a(1,x') = a[2x'+1]`. Serial reference (the
/// production path uses [`round_msg_par`]); retained for the `runtime_m25`
/// serial-vs-parallel benchmark.
#[allow(dead_code)]
#[inline]
fn round_msg(a: &[F128], b: &[F128]) -> (F128, F128) {
    let half = a.len() / 2;
    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for x in 0..half {
        let (a0, a1) = (a[2 * x], a[2 * x + 1]);
        let (b0, b1) = (b[2 * x], b[2 * x + 1]);
        g_one += a1 * b1;
        g_inf += (a0 + a1) * (b0 + b1);
    }
    (g_one, g_inf)
}

/// Fused round step: fold `(a, b)` at `r` (low bit) **in place** to half size
/// and, in the same pass, compute the next round's message `(G(1), G(∞))` from
/// the freshly folded data. Requires `a.len() >= 4`. The fold is safe in place
/// because output index `2·xp` never exceeds the read index `4·xp` (we overwrite
/// only the front of the buffer), so there is no per-round allocation.
///
/// This makes the loop `m + 1` passes instead of `2m`, but **benchmarks slower
/// single-threaded** (~0.78×): the message muls depend on the just-computed fold
/// muls, exposing PMULL latency that the unfused split avoids. Kept as the
/// building block for the eventual rayon-parallel kernel, where the
/// bandwidth saving from fewer passes should dominate. See `runtime_m25`.
#[allow(dead_code)]
fn fold_and_round_fused(a: &mut Vec<F128>, b: &mut Vec<F128>, r: F128) -> (F128, F128) {
    let n = a.len();
    debug_assert!(n >= 4 && n.is_power_of_two());
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    let pairs = half / 2; // output pairs == input quads
    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for xp in 0..pairs {
        let base = 4 * xp;
        // Fold the two input pairs feeding output pair (2xp, 2xp+1). Read all
        // four inputs into locals before writing (write idx 2xp ≤ read idx 4xp).
        let na0 = a[base] + r * (a[base + 1] + a[base]);
        let na1 = a[base + 2] + r * (a[base + 3] + a[base + 2]);
        let nb0 = b[base] + r * (b[base + 1] + b[base]);
        let nb1 = b[base + 2] + r * (b[base + 3] + b[base + 2]);
        a[2 * xp] = na0;
        a[2 * xp + 1] = na1;
        b[2 * xp] = nb0;
        b[2 * xp + 1] = nb1;
        // Next round's message contribution from this folded pair.
        g_one += na1 * nb1;
        g_inf += (na0 + na1) * (nb0 + nb1);
    }
    a.truncate(half);
    b.truncate(half);
    (g_one, g_inf)
}

/// Parallel degree-2 round message `(G(1), G(∞))`. F128 addition is XOR, so the
/// tree reduction is bit-identical to the serial left fold.
///
/// Iterates contiguous slice chunks with `chunks_exact(2)` rather than indexing
/// `a[2*x]`: eliminating the per-element bounds checks lifts the reduction from
/// ~2.6× to ~6× parallel scaling (hits the memory-bandwidth ceiling). See
/// `scaling_diag`.
fn round_msg_par(a: &[F128], b: &[F128]) -> (F128, F128) {
    use rayon::prelude::*;
    const C: usize = 1 << 14;
    a.par_chunks(C)
        .zip(b.par_chunks(C))
        .map(|(ac, bc)| {
            let mut g1 = F128::ZERO;
            let mut gi = F128::ZERO;
            for (ap, bp) in ac
                .as_chunks::<2>()
                .0
                .iter()
                .zip(bc.as_chunks::<2>().0.iter())
            {
                g1 += ap[1] * bp[1];
                gi += (ap[0] + ap[1]) * (bp[0] + bp[1]);
            }
            (g1, gi)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t))
}

/// Parallel out-of-place fold (no message), `ao/bo` length `a.len()/2`. Used for
/// the final round (size 2 → 1), where there is no successor message.
fn fold_oop_par(a: &[F128], b: &[F128], r: F128, ao: &mut [F128], bo: &mut [F128]) {
    use rayon::prelude::*;
    ao.par_iter_mut()
        .zip(bo.par_iter_mut())
        .enumerate()
        .for_each(|(x, (oa, ob))| {
            *oa = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]);
            *ob = b[2 * x] + r * (b[2 * x + 1] + b[2 * x]);
        });
}

/// Parallel **fused** round: out-of-place fold at `r` + the next round's message
/// in one pass. Requires `a.len() >= 4`. This is the production kernel — in the
/// bandwidth-bound parallel regime the halved pass count is a ~1.4× win (the
/// serial penalty from the fold→message dependency is hidden across cores).
fn fold_and_round_oop_par(
    a: &[F128],
    b: &[F128],
    r: F128,
    ao: &mut [F128],
    bo: &mut [F128],
) -> (F128, F128) {
    use rayon::prelude::*;
    debug_assert_eq!(a.len(), 2 * ao.len());
    debug_assert!(a.len() >= 4);
    // Output chunk of `CO`; the aligned input chunk is `2*CO` (output is half
    // the input). Slice/`chunks_exact` iteration — no per-element bounds checks —
    // so the reduction scales like the fold (~6× vs ~2.6× for indexed access).
    const CO: usize = 1 << 13;
    ao.par_chunks_mut(CO)
        .zip(bo.par_chunks_mut(CO))
        .zip(a.par_chunks(2 * CO))
        .zip(b.par_chunks(2 * CO))
        .map(|(((oa, ob), ain), bin)| {
            let mut g1 = F128::ZERO;
            let mut gi = F128::ZERO;
            for (((op, opb), aq), bq) in oa
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(ob.as_chunks_mut::<2>().0.iter_mut())
                .zip(ain.as_chunks::<4>().0.iter())
                .zip(bin.as_chunks::<4>().0.iter())
            {
                let na0 = aq[0] + r * (aq[1] + aq[0]);
                let na1 = aq[2] + r * (aq[3] + aq[2]);
                let nb0 = bq[0] + r * (bq[1] + bq[0]);
                let nb1 = bq[2] + r * (bq[3] + bq[2]);
                op[0] = na0;
                op[1] = na1;
                opb[0] = nb0;
                opb[1] = nb1;
                g1 += na1 * nb1;
                gi += (na0 + na1) * (nb0 + nb1);
            }
            (g1, gi)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::{FsChallenger, RandomChallenger};
    use crate::zerocheck::multilinear::fold_in_place_pair;

    fn sample_vec(ch: &mut RandomChallenger, n: usize) -> Vec<F128> {
        (0..n).map(|_| ch.sample_f128()).collect()
    }

    /// Direct MLE of `f_t` in the index variable: brute-force reference for
    /// `f̂_t` (paper Eq. 4 summed over the bijection). `O(area · (n+k+m))`.
    fn f_hat_t_bruteforce(
        params: &JaggedParams,
        z_row: &[F128],
        z_col: &[F128],
        z_index: &[F128],
    ) -> F128 {
        let eq_row = build_eq_table(z_row);
        let eq_col = build_eq_table(z_col);
        let eq_idx = build_eq_table(z_index);
        let mut acc = F128::ZERO;
        for i in 0..params.area() {
            let (row, col) = params.unrank(i);
            acc += eq_row[row] * eq_col[col] * eq_idx[i as usize];
        }
        acc
    }

    /// `q̂(point)` directly = ⟨q, eq(point, ·)⟩.
    fn mle_eval(q: &[F128], point: &[F128]) -> F128 {
        let eq = build_eq_table(point);
        q.iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |s, x| s + x)
    }

    /// A small random jagged config + dense data, with total area < 2^m.
    fn random_instance(
        ch: &mut RandomChallenger,
        n: usize,
        k: usize,
        m: usize,
    ) -> (JaggedParams, Vec<F128>) {
        let cols = 1usize << k;
        let cap = 1u64 << m;
        let max_h = 1u64 << n;
        // Pick heights with Σ ≤ 2^m. Pull pseudo-randomness from the challenger.
        let mut heights = vec![0u64; cols];
        let mut remaining = cap;
        for h in heights.iter_mut() {
            let r = ch.sample_f128().lo % (max_h + 1);
            let take = r.min(remaining);
            *h = take;
            remaining -= take;
        }
        let params = JaggedParams::from_heights(&heights, n, m);
        // Dense q: random in [0, area), zero past it.
        let mut q = vec![F128::ZERO; 1usize << m];
        for qi in q.iter_mut().take(params.area() as usize) {
            *qi = ch.sample_f128();
        }
        (params, q)
    }

    #[test]
    fn f_hat_t_matches_bruteforce() {
        let mut ch = RandomChallenger::new(0x1A66_ED12);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6), (5, 1, 5)] {
            for _ in 0..8 {
                let (params, _q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);
                let z_idx = sample_vec(&mut ch, m);
                let got = f_hat_t(&params, &z_row, &z_col, &z_idx);
                let want = f_hat_t_bruteforce(&params, &z_row, &z_col, &z_idx);
                assert_eq!(got, want, "f̂_t mismatch for n={n} k={k} m={m}");
            }
        }
    }

    #[test]
    fn f_hat_t_eq4_at_boolean_points() {
        // At a boolean index i < area, f̂_t = eq(row_t(i), z_r)·eq(col_t(i), z_c).
        let mut ch = RandomChallenger::new(0xB001_2345);
        let (params, _q) = random_instance(&mut ch, 4, 3, 7);
        let z_row = sample_vec(&mut ch, 4);
        let z_col = sample_vec(&mut ch, 3);
        let eq_row = build_eq_table(&z_row);
        let eq_col = build_eq_table(&z_col);
        for i in 0..params.area() {
            let z_idx: Vec<F128> = (0..params.m).map(|bit| int_bit(i, bit)).collect();
            let got = f_hat_t(&params, &z_row, &z_col, &z_idx);
            let (row, col) = params.unrank(i);
            let want = eq_row[row] * eq_col[col];
            assert_eq!(got, want, "Eq.4 failed at boolean i={i}");
        }
    }

    #[test]
    fn sumcheck_roundtrip() {
        let mut ch = RandomChallenger::new(0x5C4E_CC01);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6)] {
            for _ in 0..5 {
                let (params, q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);

                let mut pch = FsChallenger::new(b"flock-jagged-test");
                let (proof, v) = prove(&params, &q, &z_row, &z_col, &mut pch);

                let mut vch = FsChallenger::new(b"flock-jagged-test");
                let claim = verify(&params, &z_row, &z_col, v, &proof, &mut vch)
                    .expect("honest proof must verify");

                // The reduced claim is consistent with the dense polynomial.
                assert_eq!(claim.alpha, mle_eval(&q, &claim.point), "alpha ≠ q̂(i*)");
            }
        }
    }

    #[test]
    fn sumcheck_rejects_wrong_value() {
        let mut ch = RandomChallenger::new(0xBAD0_C1A1);
        let (params, q) = random_instance(&mut ch, 4, 3, 7);
        let z_row = sample_vec(&mut ch, 4);
        let z_col = sample_vec(&mut ch, 3);

        let mut pch = FsChallenger::new(b"flock-jagged-test");
        let (proof, v) = prove(&params, &q, &z_row, &z_col, &mut pch);

        let mut vch = FsChallenger::new(b"flock-jagged-test");
        let bad = v + F128::ONE;
        assert!(
            verify(&params, &z_row, &z_col, bad, &proof, &mut vch).is_none(),
            "verifier must reject a wrong claim value"
        );
    }

    #[test]
    fn assist_beta_matches_f_hat_t() {
        // Standalone assist at an arbitrary z_index: honest roundtrip, and the
        // proven β equals the direct f̂_t evaluation.
        let mut ch = RandomChallenger::new(0xA551_57ED);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6), (5, 1, 5)] {
            for _ in 0..5 {
                let (params, _q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);
                let z_idx = sample_vec(&mut ch, m);

                let mut pch = FsChallenger::new(b"flock-jagged-assist-test");
                let proof = prove_assist(&params, &z_row, &z_col, &z_idx, &mut pch);
                assert_eq!(
                    proof.beta,
                    f_hat_t(&params, &z_row, &z_col, &z_idx),
                    "β ≠ f̂_t for n={n} k={k} m={m}"
                );

                let mut vch = FsChallenger::new(b"flock-jagged-assist-test");
                let beta = verify_assist(&params, &z_row, &z_col, &z_idx, &proof, &mut vch)
                    .expect("honest assist must verify");
                assert_eq!(beta, proof.beta);
            }
        }
    }

    #[test]
    fn assist_streamed_matches_naive() {
        // The Lemma 4.6 streaming prover and the naive per-round-DP prover
        // compute the same polynomials with exact field ops — the transcripts
        // must be bit-identical.
        let mut ch = RandomChallenger::new(0x57EA_4E46);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6), (5, 1, 5)] {
            for _ in 0..5 {
                let (params, _q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);
                let z_idx = sample_vec(&mut ch, m);

                let mut ch_a = FsChallenger::new(b"flock-jagged-assist-test");
                let streamed = prove_assist(&params, &z_row, &z_col, &z_idx, &mut ch_a);
                let mut ch_b = FsChallenger::new(b"flock-jagged-assist-test");
                let naive = prove_assist_naive(&params, &z_row, &z_col, &z_idx, &mut ch_b);

                assert_eq!(streamed.beta, naive.beta, "β mismatch n={n} k={k} m={m}");
                assert_eq!(streamed.rounds, naive.rounds, "rounds mismatch n={n} k={k} m={m}");
            }
        }
    }

    #[test]
    fn assist_handles_degenerate_heights() {
        // Zero-height runs (collapsed terms) and an all-zero instance.
        let mut ch = RandomChallenger::new(0xDE6E_0000);
        for heights in [vec![3u64, 0, 0, 2], vec![0, 0, 0, 0], vec![0, 4, 0, 4]] {
            let params = JaggedParams::from_heights(&heights, 2, 3);
            let z_row = sample_vec(&mut ch, 2);
            let z_col = sample_vec(&mut ch, 2);
            let z_idx = sample_vec(&mut ch, 3);

            let mut pch = FsChallenger::new(b"flock-jagged-assist-test");
            let proof = prove_assist(&params, &z_row, &z_col, &z_idx, &mut pch);
            assert_eq!(proof.beta, f_hat_t(&params, &z_row, &z_col, &z_idx));

            let mut vch = FsChallenger::new(b"flock-jagged-assist-test");
            assert!(
                verify_assist(&params, &z_row, &z_col, &z_idx, &proof, &mut vch).is_some(),
                "assist must verify for heights {heights:?}"
            );
        }
    }

    #[test]
    fn assist_roundtrip() {
        let mut ch = RandomChallenger::new(0x0A55_1CC7);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6)] {
            for _ in 0..5 {
                let (params, q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);

                let mut pch = FsChallenger::new(b"flock-jagged-test");
                let (proof, assist, v) = prove_with_assist(&params, &q, &z_row, &z_col, &mut pch);

                let mut vch = FsChallenger::new(b"flock-jagged-test");
                let claim = verify_with_assist(&params, &z_row, &z_col, v, &proof, &assist, &mut vch)
                    .expect("honest assisted proof must verify");
                assert_eq!(claim.alpha, mle_eval(&q, &claim.point), "alpha ≠ q̂(i*)");

                // Same reduced claim as the assist-free verifier.
                let mut vch2 = FsChallenger::new(b"flock-jagged-test");
                let direct = verify(&params, &z_row, &z_col, v, &proof, &mut vch2)
                    .expect("direct verify of the same transcript");
                assert_eq!(claim.point, direct.point);
                assert_eq!(claim.alpha, direct.alpha);
            }
        }
    }

    #[test]
    fn assist_rejects_tampered_proof() {
        let mut ch = RandomChallenger::new(0xBAD_A5515);
        let (params, q) = random_instance(&mut ch, 4, 3, 7);
        let z_row = sample_vec(&mut ch, 4);
        let z_col = sample_vec(&mut ch, 3);

        let mut pch = FsChallenger::new(b"flock-jagged-test");
        let (proof, assist, v) = prove_with_assist(&params, &q, &z_row, &z_col, &mut pch);

        let check = |proof: &JaggedSumcheckProof, assist: &JaggedAssistProof| {
            let mut vch = FsChallenger::new(b"flock-jagged-test");
            verify_with_assist(&params, &z_row, &z_col, v, proof, assist, &mut vch)
        };
        assert!(check(&proof, &assist).is_some(), "sanity: honest verifies");

        // Wrong β (breaks both the outer relation and the assist sumcheck).
        let mut bad = assist.clone();
        bad.beta += F128::ONE;
        assert!(check(&proof, &bad).is_none(), "tampered β must be rejected");

        // Tampered round message.
        let mut bad = assist.clone();
        bad.rounds[3].0 += F128::ONE;
        assert!(check(&proof, &bad).is_none(), "tampered round must be rejected");

        // Truncated assist.
        let mut bad = assist.clone();
        bad.rounds.pop();
        assert!(check(&proof, &bad).is_none(), "truncated assist must be rejected");

        // Tampered dense claim must break the outer relation against β.
        let mut bad_proof = proof.clone();
        bad_proof.q_eval += F128::ONE;
        assert!(check(&bad_proof, &assist).is_none(), "tampered q_eval must be rejected");
    }

    /// Runtime check at the realistic Option-B size: an m=32-bit trace packed
    /// into F128 (128 bits each) is a dense `q` of `2^25` field elements, so the
    /// jagged sumcheck runs over 25 variables. Mirrors `prove`, split into the
    /// `f̂_t`-sequence generation and the sumcheck rounds.
    ///
    /// `cargo test --release -p flock-core pcs::jagged::tests::runtime_m25 -- --ignored --nocapture`
    #[test]
    #[ignore = "heavy benchmark; run explicitly with --release --ignored --nocapture"]
    fn runtime_m25() {
        use std::time::Instant;

        // Match the full-prover profile (P-core pool) for an apples-to-apples ratio.
        let _ = crate::init_perf_thread_pool();
        let (n, k, m) = (13usize, 12usize, 25usize); // 2^25 dense F128 elements
        let cols = 1usize << k;
        let height = (1u64 << m) / cols as u64; // uniform; total area = 2^m
        let params = JaggedParams::from_heights(&vec![height; cols], n, m);
        assert_eq!(params.area(), 1u64 << m);

        // Cheap deterministic dense data (field-mul cost is data-independent).
        let len = 1usize << m;
        let mut q = vec![F128::ZERO; len];
        for (i, qi) in q.iter_mut().enumerate() {
            *qi = F128 {
                lo: i as u64,
                hi: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            };
        }
        let mut rc = RandomChallenger::new(0x0B7A_4225);
        let z_row = sample_vec(&mut rc, n);
        let z_col = sample_vec(&mut rc, k);

        let mb = (len * std::mem::size_of::<F128>()) as f64 / (1024.0 * 1024.0);
        eprintln!("\n[jagged runtime] m={m} ({len} F128 = {mb:.0} MB), n={n}, k={k}, cols={cols}");

        const REPS: usize = 3;

        // --- Phase 1: B-vector + claim generation, serial vs parallel-fused. ---
        let mut t_gen_ser = std::time::Duration::MAX;
        let mut t_gen_par = std::time::Duration::MAX;
        let (mut b, mut v) = (Vec::new(), F128::ZERO);
        for _ in 0..REPS {
            // Serial reference: column-major build + separate v reduction.
            let t0 = Instant::now();
            let eq_row = build_eq_table(&z_row);
            let eq_col = build_eq_table(&z_col);
            let mut bs = vec![F128::ZERO; len];
            for col in 0..cols {
                let start = params.col_prefix_sums[col] as usize;
                let end = params.col_prefix_sums[col + 1] as usize;
                let ec = eq_col[col];
                for (row, slot) in bs[start..end].iter_mut().enumerate() {
                    *slot = eq_row[row] * ec;
                }
            }
            let mut vs = F128::ZERO;
            for (qi, bi) in q.iter().zip(bs.iter()) {
                vs += *qi * *bi;
            }
            t_gen_ser = t_gen_ser.min(t0.elapsed());
            std::hint::black_box(&bs);

            // Parallel fused helper (the production path).
            let t1 = Instant::now();
            let (bp, vp) = generate_f_and_claim(&params, &q, &z_row, &z_col);
            t_gen_par = t_gen_par.min(t1.elapsed());
            assert_eq!(vs, vp, "parallel gen must match serial");
            b = bp;
            v = vp;
        }
        let _ = v; // prover-side claim value; not needed past phase 1

        // --- Phase 2: 2x2 head-to-head {serial,parallel} x {unfused,fused},
        // min over REPS to suppress thermal / allocator variance. ---

        // Serial: in-place fold; unfused = msg pass + fold pass, fused = both in one.
        let run_serial = |fused: bool| -> std::time::Duration {
            let mut a = q.clone();
            let mut bb = b.clone();
            let mut ch = FsChallenger::new(b"flock-jagged-bench");
            ch.observe_label(b"flock-jagged-v0");
            let t = Instant::now();
            if fused {
                let (mut g1, mut gi) = round_msg(&a, &bb);
                for _ in 0..m {
                    ch.observe_f128(g1);
                    ch.observe_f128(gi);
                    let r = ch.sample_f128();
                    if a.len() > 2 {
                        (g1, gi) = fold_and_round_fused(&mut a, &mut bb, r);
                    } else {
                        fold_in_place_pair(&mut a, &mut bb, r);
                    }
                }
            } else {
                for _ in 0..m {
                    let (g1, gi) = round_msg(&a, &bb);
                    ch.observe_f128(g1);
                    ch.observe_f128(gi);
                    let r = ch.sample_f128();
                    fold_in_place_pair(&mut a, &mut bb, r);
                }
            }
            std::hint::black_box(a[0]);
            t.elapsed()
        };

        // Parallel: rayon kernels, ping-pong between two out-of-place buffers.
        let run_par = |fused: bool| -> std::time::Duration {
            let mut a = q.clone(); // len N
            let mut bb = b.clone();
            let mut sa = vec![F128::ZERO; len / 2];
            let mut sb = vec![F128::ZERO; len / 2];
            let mut cur = len;
            let mut ch = FsChallenger::new(b"flock-jagged-bench");
            ch.observe_label(b"flock-jagged-v0");
            let t = Instant::now();
            let (mut g1, mut gi) = if fused {
                round_msg_par(&a[..cur], &bb[..cur])
            } else {
                (F128::ZERO, F128::ZERO)
            };
            for _ in 0..m {
                let half = cur / 2;
                if !fused {
                    let (m1, mi) = round_msg_par(&a[..cur], &bb[..cur]);
                    g1 = m1;
                    gi = mi;
                }
                ch.observe_f128(g1);
                ch.observe_f128(gi);
                let r = ch.sample_f128();
                if fused && cur > 2 {
                    let (n1, ni) = fold_and_round_oop_par(
                        &a[..cur],
                        &bb[..cur],
                        r,
                        &mut sa[..half],
                        &mut sb[..half],
                    );
                    g1 = n1;
                    gi = ni;
                } else {
                    fold_oop_par(&a[..cur], &bb[..cur], r, &mut sa[..half], &mut sb[..half]);
                }
                std::mem::swap(&mut a, &mut sa);
                std::mem::swap(&mut bb, &mut sb);
                cur = half;
            }
            std::hint::black_box(a[0]);
            t.elapsed()
        };

        let mut s_unf = std::time::Duration::MAX;
        let mut s_fus = std::time::Duration::MAX;
        let mut p_unf = std::time::Duration::MAX;
        let mut p_fus = std::time::Duration::MAX;
        for _ in 0..REPS {
            s_unf = s_unf.min(run_serial(false));
            s_fus = s_fus.min(run_serial(true));
            p_unf = p_unf.min(run_par(false));
            p_fus = p_fus.min(run_par(true));
        }

        // --- Verifier f̂_t eval at a random final point. ---
        let point: Vec<F128> = (0..m).map(|_| rc.sample_f128()).collect();
        let t2 = Instant::now();
        let beta = f_hat_t(&params, &z_row, &z_col, &point);
        std::hint::black_box(beta);
        let t_ver = t2.elapsed();

        let ratio = |unf: std::time::Duration, fus: std::time::Duration| {
            unf.as_secs_f64() / fus.as_secs_f64()
        };
        eprintln!("  threads: {}", rayon::current_num_threads());
        eprintln!(
            "  f̂_t-gen (B + claim) serial {:>8.1?} → parallel {:>8.1?}   ({:.2}x)",
            t_gen_ser,
            t_gen_par,
            ratio(t_gen_ser, t_gen_par)
        );
        eprintln!("                          unfused      fused     fusion");
        eprintln!(
            "  sumcheck serial   : {:>9.1?}  {:>9.1?}   {:.2}x",
            s_unf,
            s_fus,
            ratio(s_unf, s_fus)
        );
        eprintln!(
            "  sumcheck parallel : {:>9.1?}  {:>9.1?}   {:.2}x   (vs serial unfused {:.2}x)",
            p_unf,
            p_fus,
            ratio(p_unf, p_fus),
            ratio(s_unf, p_fus)
        );
        eprintln!("  verifier f̂_t eval            : {:>9.3?}", t_ver);
        let best = p_unf.min(p_fus);
        eprintln!(
            "  best prover total (gen + best sumcheck): {:.1?} ({:.2} ns/elem)\n",
            t_gen_par + best,
            (t_gen_par + best).as_nanos() as f64 / len as f64
        );
    }

    /// The full jagged reduction at the 2^30-bit packed-witness point: a
    /// 2^30-bit trace packed into F128 (128 bits each) is a dense `q` of 2^23
    /// field elements — `m = 23`, with 2^12 uniform columns (`n = 11`).
    /// Best-of-3 for: main sumcheck prover, assist prover, both verifier
    /// paths (direct `f̂_t` vs assist).
    ///
    /// `cargo test --release -p flock-core pcs::jagged::tests::runtime_bits30 -- --ignored --nocapture`
    #[test]
    #[ignore = "heavy benchmark; run explicitly with --release --ignored --nocapture"]
    fn runtime_bits30() {
        use std::time::Instant;

        let _ = crate::init_perf_thread_pool();
        let (n, k, m) = (11usize, 12usize, 23usize); // 2^30 bits / 128 = 2^23 elems
        let cols = 1usize << k;
        let height = (1u64 << m) / cols as u64;
        let params = JaggedParams::from_heights(&vec![height; cols], n, m);
        assert_eq!(params.area(), 1u64 << m);

        let len = 1usize << m;
        let mut q = vec![F128::ZERO; len];
        for (i, qi) in q.iter_mut().enumerate() {
            *qi = F128 {
                lo: i as u64,
                hi: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            };
        }
        let mut rc = RandomChallenger::new(0x0B17_5300);
        let z_row = sample_vec(&mut rc, n);
        let z_col = sample_vec(&mut rc, k);

        let best3 = |f: &mut dyn FnMut() -> std::time::Duration| {
            (0..3).map(|_| f()).min().unwrap()
        };

        // Warm-up (thread pool + page faults).
        let mut ch = FsChallenger::new(b"flock-jagged-bits30");
        let _ = prove(&params, &q, &z_row, &z_col, &mut ch);

        // Main jagged sumcheck prover (B-generation + rounds).
        let t_prove = best3(&mut || {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            let t = Instant::now();
            std::hint::black_box(prove(&params, &q, &z_row, &z_col, &mut ch));
            t.elapsed()
        });

        // Main + assist, and keep one transcript for the verifier runs.
        let mut t_both = std::time::Duration::MAX;
        let mut kept = None;
        for _ in 0..3 {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            let t = Instant::now();
            let out = prove_with_assist(&params, &q, &z_row, &z_col, &mut ch);
            t_both = t_both.min(t.elapsed());
            kept = Some(out);
        }
        let (proof, assist, v) = kept.unwrap();

        // Verifier, direct f̂_t path.
        let t_verify_direct = best3(&mut || {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            let t = Instant::now();
            std::hint::black_box(
                verify(&params, &z_row, &z_col, v, &proof, &mut ch).expect("verify"),
            );
            t.elapsed()
        });

        // Verifier, assist path.
        let t_verify_assist = best3(&mut || {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            let t = Instant::now();
            std::hint::black_box(
                verify_with_assist(&params, &z_row, &z_col, v, &proof, &assist, &mut ch)
                    .expect("verify_with_assist"),
            );
            t.elapsed()
        });

        let main_bytes = (2 * proof.rounds.len() + 1) * 16;
        let assist_bytes = (2 * assist.rounds.len() + 1) * 16;
        eprintln!("  threads: {}", rayon::current_num_threads());
        eprintln!("  witness: 2^{m} F128 = {} MiB (2^30 bits packed)", len * 16 >> 20);
        eprintln!("  sumcheck prover ({m} rounds)          : {t_prove:>9.3?}");
        eprintln!(
            "  + assist prover ({} rounds)          : {:>9.3?}  (assist ≈ {:.3?}, {:.1}% of prover)",
            assist.rounds.len(),
            t_both,
            t_both.saturating_sub(t_prove),
            100.0 * t_both.saturating_sub(t_prove).as_secs_f64() / t_both.as_secs_f64()
        );
        eprintln!("  verifier, direct f̂_t (2^{k} BP evals): {t_verify_direct:>9.3?}");
        eprintln!(
            "  verifier, assist                      : {t_verify_assist:>9.3?}  ({:.1}x)",
            t_verify_direct.as_secs_f64() / t_verify_assist.as_secs_f64()
        );
        eprintln!(
            "  proof: main {main_bytes} B + assist {assist_bytes} B = {} B",
            main_bytes + assist_bytes
        );
    }

    /// Assist runtimes at the realistic size (matches `runtime_m25`: m=25,
    /// 2^12 columns): direct verifier `f̂_t` vs assist prover / assist verifier.
    ///
    /// `cargo test --release -p flock-core pcs::jagged::tests::runtime_assist_m25 -- --ignored --nocapture`
    #[test]
    #[ignore = "heavy benchmark; run explicitly with --release --ignored --nocapture"]
    fn runtime_assist_m25() {
        use std::time::Instant;

        let _ = crate::init_perf_thread_pool();
        let (n, k, m) = (13usize, 12usize, 25usize);
        let cols = 1usize << k;
        let height = (1u64 << m) / cols as u64;
        let params = JaggedParams::from_heights(&vec![height; cols], n, m);

        let mut rc = RandomChallenger::new(0xA551_0B25);
        let z_row = sample_vec(&mut rc, n);
        let z_col = sample_vec(&mut rc, k);
        let z_idx = sample_vec(&mut rc, m);

        let t0 = Instant::now();
        let direct = f_hat_t(&params, &z_row, &z_col, &z_idx);
        let t_direct = t0.elapsed();

        let t1 = Instant::now();
        let mut pch = FsChallenger::new(b"flock-jagged-assist-bench");
        let proof = prove_assist(&params, &z_row, &z_col, &z_idx, &mut pch);
        let t_prove = t1.elapsed();
        assert_eq!(proof.beta, direct);

        let t1n = Instant::now();
        let mut nch = FsChallenger::new(b"flock-jagged-assist-bench");
        let naive = prove_assist_naive(&params, &z_row, &z_col, &z_idx, &mut nch);
        let t_prove_naive = t1n.elapsed();
        assert_eq!(naive.rounds, proof.rounds, "provers must agree");

        let t2 = Instant::now();
        let mut vch = FsChallenger::new(b"flock-jagged-assist-bench");
        let beta = verify_assist(&params, &z_row, &z_col, &z_idx, &proof, &mut vch)
            .expect("honest assist must verify");
        let t_verify = t2.elapsed();
        assert_eq!(beta, direct);

        eprintln!("  threads: {}", rayon::current_num_threads());
        eprintln!("  verifier, direct f̂_t (2^{k} BP evals): {t_direct:>9.3?}");
        eprintln!(
            "  assist prover, streamed ({} rounds)   : {t_prove:>9.3?}  (naive: {t_prove_naive:.3?}, {:.1}x)",
            proof.rounds.len(),
            t_prove_naive.as_secs_f64() / t_prove.as_secs_f64()
        );
        eprintln!("  assist verifier (1 BP eval + W(ρ))    : {t_verify:>9.3?}");
        eprintln!(
            "  verifier speedup: {:.1}x   proof size: {} B",
            t_direct.as_secs_f64() / t_verify.as_secs_f64(),
            (1 + 2 * proof.rounds.len()) * 16
        );
    }

    /// Diagnose the sumcheck's ~4× parallel scaling: is it the memory-bandwidth
    /// ceiling, or fine-grained-kernel inefficiency? Compares a memcpy baseline,
    /// fold and reduction kernels, each fine-grained (current style) vs
    /// coarse-chunked, at 2^25 on the P-core pool.
    ///
    /// `cargo test --release pcs::jagged::tests::scaling_diag -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic; run with --release --ignored --nocapture"]
    fn scaling_diag() {
        use rayon::prelude::*;
        use std::time::{Duration, Instant};
        let _ = crate::init_perf_thread_pool();
        let m = 25usize;
        let len = 1usize << m;
        let half = len / 2;
        let a: Vec<F128> = (0..len)
            .map(|i| F128 {
                lo: i as u64,
                hi: i as u64,
            })
            .collect();
        let b = a.clone();
        let r = F128 {
            lo: 0x9E37,
            hi: 0x1234,
        };
        const REPS: usize = 6;
        const CHUNK: usize = 1 << 13; // coarse: 8K outputs / task

        let bench = |f: &mut dyn FnMut()| {
            let mut t = Duration::MAX;
            for _ in 0..REPS {
                let t0 = Instant::now();
                f();
                t = t.min(t0.elapsed());
            }
            t
        };
        let sp = |s: Duration, p: Duration| s.as_secs_f64() / p.as_secs_f64();
        let gbps = |bytes: usize, t: Duration| bytes as f64 / t.as_secs_f64() / 1e9;

        eprintln!(
            "\n[scaling diag] m={m}, threads={}",
            rayon::current_num_threads()
        );

        // --- memcpy baseline: read len, write len (the raw bandwidth ceiling) ---
        let mut dst = crate::alloc_uninit_f128_vec(len);
        let ts = bench(&mut || dst.copy_from_slice(&a));
        let tp = bench(&mut || {
            dst.par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(ci, d)| d.copy_from_slice(&a[ci * CHUNK..ci * CHUNK + d.len()]));
        });
        let bytes = len * 32; // read+write 16B each
        eprintln!(
            "  memcpy        : serial {:>7.1?} ({:>4.0} GB/s)  parallel {:>7.1?} ({:>4.0} GB/s)  {:.2}x",
            ts,
            gbps(bytes, ts),
            tp,
            gbps(bytes, tp),
            sp(ts, tp)
        );

        // --- fold (read len, write half): fine (par_iter_mut) vs coarse ---
        let mut out = crate::alloc_uninit_f128_vec(half);
        let ts = bench(&mut || {
            for x in 0..half {
                out[x] = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]);
            }
        });
        let tp_fine = bench(&mut || {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(x, o)| *o = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]));
        });
        let tp_coarse = bench(&mut || {
            out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, oc)| {
                let x0 = ci * CHUNK;
                for (j, o) in oc.iter_mut().enumerate() {
                    let x = x0 + j;
                    *o = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]);
                }
            });
        });
        let bytes = half * 16 * 3; // read 2, write 1
        eprintln!(
            "  fold   serial {:>7.1?} ({:>4.0} GB/s)  par.fine {:>7.1?} {:.2}x  par.coarse {:>7.1?} {:.2}x",
            ts,
            gbps(bytes, ts),
            tp_fine,
            sp(ts, tp_fine),
            tp_coarse,
            sp(ts, tp_coarse)
        );

        // --- real round message (contiguous a+b read): round_msg vs round_msg_par ---
        let ts = bench(&mut || {
            std::hint::black_box(round_msg(&a, &b));
        });
        let tp_fine = bench(&mut || {
            std::hint::black_box(round_msg_par(&a, &b));
        });
        // Coarse reduction: per-chunk local accumulator, then combine.
        let tp_coarse = bench(&mut || {
            let acc = (0..half)
                .into_par_iter()
                .with_min_len(CHUNK)
                .fold(
                    || (F128::ZERO, F128::ZERO),
                    |(g1, gi), x| {
                        let (a0, a1) = (a[2 * x], a[2 * x + 1]);
                        let (b0, b1) = (b[2 * x], b[2 * x + 1]);
                        (g1 + a1 * b1, gi + (a0 + a1) * (b0 + b1))
                    },
                )
                .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t));
            std::hint::black_box(acc);
        });
        let rd_bytes = len * 16 * 2; // read all of a and b
        eprintln!(
            "  round_msg serial {:>7.1?} ({:>4.0} GB/s)  par.fine {:>7.1?} {:.2}x  par.coarse {:>7.1?} {:.2}x",
            ts,
            gbps(rd_bytes, ts),
            tp_fine,
            sp(ts, tp_fine),
            tp_coarse,
            sp(ts, tp_coarse)
        );
        // slice/chunks_exact iteration: no per-element bounds checks.
        let tp_slice = bench(&mut || {
            let acc = a
                .par_chunks(2 * CHUNK)
                .zip(b.par_chunks(2 * CHUNK))
                .map(|(ac, bc)| {
                    let mut g1 = F128::ZERO;
                    let mut gi = F128::ZERO;
                    for (ap, bp) in ac
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(bc.as_chunks::<2>().0.iter())
                    {
                        g1 += ap[1] * bp[1];
                        gi += (ap[0] + ap[1]) * (bp[0] + bp[1]);
                    }
                    (g1, gi)
                })
                .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t));
            std::hint::black_box(acc);
        });
        eprintln!(
            "  round_msg par.slice(chunks_exact)   {:>7.1?} ({:>4.0} GB/s)  {:.2}x",
            tp_slice,
            gbps(rd_bytes, tp_slice),
            sp(ts, tp_slice)
        );

        // --- per-round breakdown of the actual parallel-fused sumcheck ---
        let mut sa = crate::alloc_uninit_f128_vec(half);
        let mut sb = crate::alloc_uninit_f128_vec(half);
        let mut round_t = vec![Duration::MAX; m + 1];
        for _ in 0..REPS {
            let mut av = a.clone();
            let mut bv = b.clone();
            let mut cur = len;
            let t0 = Instant::now();
            let (mut _g1, mut _gi) = round_msg_par(&av[..cur], &bv[..cur]);
            round_t[0] = round_t[0].min(t0.elapsed());
            for rd in 0..m {
                let half_r = cur / 2;
                let t = Instant::now();
                if cur > 2 {
                    (_g1, _gi) = fold_and_round_oop_par(
                        &av[..cur],
                        &bv[..cur],
                        r,
                        &mut sa[..half_r],
                        &mut sb[..half_r],
                    );
                } else {
                    fold_oop_par(
                        &av[..cur],
                        &bv[..cur],
                        r,
                        &mut sa[..half_r],
                        &mut sb[..half_r],
                    );
                }
                std::mem::swap(&mut av, &mut sa);
                std::mem::swap(&mut bv, &mut sb);
                round_t[rd + 1] = round_t[rd + 1].min(t.elapsed());
                cur = half_r;
            }
        }
        let total: Duration = round_t.iter().sum();
        let tail: Duration = round_t[6..].iter().sum(); // rounds with cur ≤ 2^20
        eprintln!(
            "  sumcheck per-round: total {:.1?} | r0 {:.1?} r1 {:.1?} r2 {:.1?} | tail(r6+, cur≤2^19) {:.2?} ({:.0}%)",
            total,
            round_t[1],
            round_t[2],
            round_t[3],
            tail,
            100.0 * tail.as_secs_f64() / total.as_secs_f64()
        );
        std::hint::black_box(&out);
        std::hint::black_box(&dst);
    }

    #[test]
    fn sumcheck_rejects_tampered_proof() {
        let mut ch = RandomChallenger::new(0xDEAD_BEEF);
        let (params, q) = random_instance(&mut ch, 3, 3, 6);
        let z_row = sample_vec(&mut ch, 3);
        let z_col = sample_vec(&mut ch, 3);

        let mut pch = FsChallenger::new(b"flock-jagged-test");
        let (mut proof, v) = prove(&params, &q, &z_row, &z_col, &mut pch);
        proof.q_eval += F128::ONE;

        let mut vch = FsChallenger::new(b"flock-jagged-test");
        assert!(
            verify(&params, &z_row, &z_col, v, &proof, &mut vch).is_none(),
            "verifier must reject a tampered q_eval"
        );
    }
}
