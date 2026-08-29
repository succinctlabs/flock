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
use serde::{Deserialize, Serialize};

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

// The two boundary states are `pub`: the recursion circuit's in-circuit
// anchor verifier chains the same 4-state DP (`assist_sparse_transitions`)
// and needs the seed/read-out indices — see the transcription work in
// `flock-prover/tests/`.
pub const STATE_INITIAL: usize = 0; // carry=0, comparison=0
pub const STATE_SUCCESS: usize = 2; // carry=0, comparison=1

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
        let eq16 = build_eq_table(&[point_bit(z_row, layer), point_bit(z_index, layer), c, d]);
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
/// Returns `(B, v, G(1), G(∞))`: the second sumcheck multilinear, the claim,
/// **and the first round message**, all from one pass. Fusing the message in
/// is free traffic-wise (the pass already streams `q` and `B` pair-by-pair)
/// and removes the prover's separate `round_msg_par` pass over 2·2^m elements.
fn generate_f_and_claim(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
) -> (Vec<F128>, F128, F128, F128) {
    use rayon::prelude::*;
    let len = 1usize << params.m;
    let area = params.area() as usize;
    let eq_row = build_eq_table(z_row);
    let eq_col = build_eq_table(z_col);
    let prefix = &params.col_prefix_sums;
    let mut b = crate::alloc_uninit_f128_vec(len);

    if len == 1 {
        // m = 0: single element, no sumcheck rounds (and no pairs).
        let bi = if area == 0 {
            F128::ZERO
        } else {
            let col = prefix.partition_point(|&t| t == 0).saturating_sub(1);
            eq_row[0] * eq_col[col]
        };
        b[0] = bi;
        return (b, q[0] * bi, F128::ZERO, F128::ZERO);
    }

    // ~1 MB chunks: one binary search amortized over 64K elements. CHUNK is
    // even and len is a power of two ≥ 2, so message pairs never straddle
    // chunks.
    const CHUNK: usize = 1 << 16;
    let (v, g_one, g_inf) = b
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(ci, b_chunk)| {
            let g0 = ci * CHUNK;
            let q_chunk = &q[g0..g0 + b_chunk.len()];
            fill_weight_range(b_chunk, g0, area, prefix, &eq_row, &eq_col);
            let mut acc = F128::ZERO;
            let mut m_one = F128::ZERO;
            let mut m_inf = F128::ZERO;
            for (bp, qp) in b_chunk
                .as_chunks::<2>()
                .0
                .iter()
                .zip(q_chunk.as_chunks::<2>().0.iter())
            {
                let t = qp[1] * bp[1];
                acc += qp[0] * bp[0] + t;
                m_one += t;
                m_inf += (qp[0] + qp[1]) * (bp[0] + bp[1]);
            }
            (acc, m_one, m_inf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a, b1, c), (d, e, f)| (a + d, b1 + e, c + f),
        );
    (b, v, g_one, g_inf)
}

/// Fill `out` with the jagged weight `W[e] = eq(row_t(e), z_row)·eq(col_t(e),
/// z_col)` for `e ∈ [g0, g0 + out.len())`, zero past `area` — the single
/// source of truth for the weight formula, shared by the element-paired and
/// block-paired drivers. One binary search per range, then an advancing
/// column cursor.
#[inline]
fn fill_weight_range(
    out: &mut [F128],
    g0: usize,
    area: usize,
    prefix: &[u64],
    eq_row: &[F128],
    eq_col: &[F128],
) {
    if g0 >= area {
        // Wholly past the jagged area — bulk zero instead of an eq product
        // (and a column-cursor step) per element. Under the lane-major
        // commit this is the stack's zero tail, i.e. whole lanes.
        out.fill(F128::ZERO);
        return;
    }
    // Walk COLUMN SEGMENTS, not elements: within a column `eq_col` is constant
    // and the row index runs contiguously, so a segment is a run of `eq_row`
    // scaled by one hoisted constant — the `i >= area` test and the column
    // cursor leave the inner loop entirely. This is what makes the
    // just-in-time basis window fill ([`fill_weight_range`]) competitive:
    // it runs once per element position in the round-0 message, so
    // per-element branching there is the hot path.
    let mut col = prefix
        .partition_point(|&t| t <= g0 as u64)
        .saturating_sub(1);
    let end = (g0 + out.len()).min(area);
    let mut i = g0;
    let mut pos = 0usize;
    while i < end {
        while (i as u64) >= prefix[col + 1] {
            col += 1;
        }
        let seg_end = (prefix[col + 1] as usize).min(end);
        let row0 = i - prefix[col] as usize;
        let ec = eq_col[col];
        let n = seg_end - i;
        for (k, slot) in out[pos..pos + n].iter_mut().enumerate() {
            *slot = eq_row[row0 + k] * ec;
        }
        pos += n;
        i = seg_end;
    }
    out[pos..].fill(F128::ZERO);
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
/// assist sub-protocol ([`prove_with_assist`] pairs the two). Not on the
/// fused opening path (the merged opening — `pcs::open_batch_merged` —
/// discharges the weight-table inner product directly in Ligerito, with no
/// jagged main sumcheck).
pub(crate) fn prove_main<C: Challenger>(
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
    // over the boolean cube (= f̂_t(z_row, z_col, ·) on {0,1}^m), the claim
    // v = Σ_i q(i)·B(i) = p̂(z_row, z_col), AND the first round message — one
    // fused parallel pass, so `q` and `B` are not re-read for round 1's message.
    let (b, v, mut g_one, mut g_inf) = generate_f_and_claim(params, q, z_row, z_col);

    // Product-of-two-multilinears sumcheck, binding the low index bit each
    // round — parallel and fused: each fold pass also computes the next round's
    // message, halving passes over the (bandwidth-bound) witness. Round 1 folds
    // straight out of the borrowed `q` and the owned `b` (q is never copied);
    // rounds 2+ ping-pong `a/bb` (len/4 buffers) with the scratch `sa/sb`
    // (len/2 buffers) — the write always fits the smaller of the pair. F128
    // addition is XOR, so the parallel tree reduction is bit-identical to a
    // serial fold.
    let mut sa = crate::alloc_uninit_f128_vec(len / 2);
    let mut sb = crate::alloc_uninit_f128_vec(len / 2);
    let mut a = crate::alloc_uninit_f128_vec(len / 4);
    let mut bb = crate::alloc_uninit_f128_vec(len / 4);
    let mut cur = len;
    let mut rounds = Vec::with_capacity(m);
    let mut point = Vec::with_capacity(m);
    for round in 0..m {
        let half = cur / 2;
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        rounds.push((g_one, g_inf));
        point.push(r);
        let (a_src, b_src): (&[F128], &[F128]) = if round == 0 { (q, &b) } else { (&a, &bb) };
        if cur > 2 {
            (g_one, g_inf) = fold_and_round_oop_par(
                &a_src[..cur],
                &b_src[..cur],
                r,
                &mut sa[..half],
                &mut sb[..half],
            );
        } else {
            fold_oop_par(
                &a_src[..cur],
                &b_src[..cur],
                r,
                &mut sa[..half],
                &mut sb[..half],
            );
        }
        std::mem::swap(&mut a, &mut sa);
        std::mem::swap(&mut bb, &mut sb);
        cur = half;
    }

    debug_assert_eq!(cur, 1);
    let q_eval = if m == 0 { q[0] } else { a[0] };
    let proof = JaggedSumcheckProof { rounds, q_eval };
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JaggedAssistProof {
    pub beta: F128,
    pub rounds: Vec<(F128, F128)>,
}

/// The distinct boundary pairs `(t_{y-1}, t_y)` in column order, each tagged
/// with the number of original columns it covers. Equal adjacent pairs are the
/// zero-height columns (including the zero-padded tail). Depends on
/// `col_prefix_sums` alone, so a batch of statements over the same params
/// shares one list — and the block tree keyed off it ([`AssistBlocks`]).
/// `pub`: the recursion circuit's anchor gates consume the run structure
/// (per-run boundary pairs) — sourced from here, like
/// [`assist_sparse_transitions`], so the two cannot drift.
pub fn assist_boundaries(params: &JaggedParams) -> Vec<(u64, u64, u32)> {
    let n_col = params.col_prefix_sums.len() - 1;
    let mut out: Vec<(u64, u64, u32)> = Vec::with_capacity(n_col);
    for y in 0..n_col {
        let (t_c, t_next) = (params.col_prefix_sums[y], params.col_prefix_sums[y + 1]);
        match out.last_mut() {
            Some((c, d, run)) if *c == t_c && *d == t_next => *run += 1,
            _ => out.push((t_c, t_next, 1)),
        }
    }
    out
}

/// The assist's per-column terms `(w_y, t_{y-1}, t_y)`, with runs of columns
/// sharing the same `(t_{y-1}, t_y)` pair — zero-height columns, including the
/// zero-padded tail — collapsed into one term of summed weight `Σ eq(z_col, y)`.
/// Pure regrouping of identical summands: transcript-invariant.
fn assist_columns(params: &JaggedParams, z_col: &[F128]) -> Vec<(F128, u64, u64)> {
    assist_columns_at(&assist_boundaries(params), z_col)
}

/// [`assist_columns`] against a prebuilt boundary list: the same summands in
/// the same order, just with the run structure read off instead of rediscovered.
fn assist_columns_at(bounds: &[(u64, u64, u32)], z_col: &[F128]) -> Vec<(F128, u64, u64)> {
    // Boolean column points (the gather claims' `bits(word_col) ‖
    // bits(slot_prefix)`) have a ONE-HOT eq table: the per-run sums are an
    // indicator of the run containing the hot column, so the dense
    // 2^|z_col| build is skipped. Value-identical to the dense path.
    let hot: Option<usize> = z_col.iter().enumerate().try_fold(0usize, |acc, (i, &x)| {
        if x == F128::ZERO {
            Some(acc)
        } else if x == F128::ONE {
            Some(acc | (1 << i))
        } else {
            None
        }
    });
    if let Some(h) = hot {
        let mut out: Vec<(F128, u64, u64)> = Vec::with_capacity(bounds.len());
        let mut y = 0usize;
        for &(t_c, t_next, run) in bounds {
            let w = if (y..y + run as usize).contains(&h) {
                F128::ONE
            } else {
                F128::ZERO
            };
            y += run as usize;
            out.push((w, t_c, t_next));
        }
        debug_assert_eq!(y, 1usize << z_col.len());
        return out;
    }
    let eq_col = build_eq_table(z_col);
    let mut out: Vec<(F128, u64, u64)> = Vec::with_capacity(bounds.len());
    let mut y = 0usize;
    for &(t_c, t_next, run) in bounds {
        let mut w = F128::ZERO;
        for &e in &eq_col[y..y + run as usize] {
            w += e;
        }
        y += run as usize;
        out.push((w, t_c, t_next));
    }
    debug_assert_eq!(y, eq_col.len());
    out
}

/// [`assist_columns_at`] for a PRE-COMBINED dense column-weight vector (a
/// scalar group's γ-baked cols): the per-run sum ranges over the given
/// weights instead of an eq table. Same summands as running the group's
/// members separately, reassociated — value-identical.
fn weights_columns_at(bounds: &[(u64, u64, u32)], weights: &[F128]) -> Vec<(F128, u64, u64)> {
    let mut out: Vec<(F128, u64, u64)> = Vec::with_capacity(bounds.len());
    let mut y = 0usize;
    for &(t_c, t_next, run) in bounds {
        let mut w = F128::ZERO;
        for &e in &weights[y..y + run as usize] {
            w += e;
        }
        y += run as usize;
        out.push((w, t_c, t_next));
    }
    debug_assert_eq!(y, weights.len());
    out
}

/// The weight multilinear `W` at the assist's final point:
/// `W(ρ) = Σ_y w_y · Π_ℓ eq(t_{y-1}[ℓ], ρ_{c,ℓ}) · eq(t_y[ℓ], ρ_{d,ℓ})`, with
/// `ρ` in the interleaved order `(c_0, d_0, c_1, d_1, …)`. `eq(b, r)` at a
/// boolean `b` is `r` or `1 + r` (char 2), so this is `2(m+1)` multiplications
/// per distinct column. Superseded in production by [`assist_w_at_blocked`],
/// which spends one multiply per *run* of columns; retained as that form's
/// correctness reference (`blocked_w_at_matches_dense`).
#[cfg(test)]
fn assist_w_at(cols: &[(F128, u64, u64)], rho: &[F128], m: usize) -> F128 {
    debug_assert_eq!(rho.len(), 2 * (m + 1));
    let mut acc = F128::ZERO;
    for &(w, t_c, t_next) in cols {
        let mut term = w;
        for layer in 0..=m {
            let rc = rho[2 * layer];
            let rd = rho[2 * layer + 1];
            term *= if (t_c >> layer) & 1 == 1 {
                rc
            } else {
                F128::ONE + rc
            };
            term *= if (t_next >> layer) & 1 == 1 {
                rd
            } else {
                F128::ONE + rd
            };
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
/// `pub`: the recursion circuit's anchor gates bake this table into their
/// relation; sourcing it from here (rather than a test-side replica) means
/// a protocol change cannot silently drift the two apart.
pub fn assist_sparse_transitions() -> [[[(usize, usize); 2]; 4]; 4] {
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
/// out **layer-major** (`rows[ℓ·n_cols + y]`), one column per slot. Superseded
/// in production by the block-collapsed [`assist_suffix_rows_blocked`], which
/// stores one slot per *run* of columns agreeing above `ℓ`; retained as that
/// form's correctness reference (`blocked_suffix_rows_match_dense`).
#[cfg(test)]
fn assist_suffix_rows(
    cols: &[(F128, u64, u64)],
    eq4s: &[[F128; 4]],
    sparse: &[[[(usize, usize); 2]; 4]; 4],
    m: usize,
) -> Vec<[F128; 4]> {
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
        for ((dv, sv), &(_, t_c, t_next)) in dst.iter_mut().zip(src).zip(cols) {
            let cd = ((t_c >> layer) & 1) as usize + 2 * ((t_next >> layer) & 1) as usize;
            let rows_cd = &sparse[cd];
            for (s, slot) in dv.iter_mut().enumerate() {
                let (i0, o0) = rows_cd[s][0];
                let (i1, o1) = rows_cd[s][1];
                *slot = eq4[i0] * sv[o0] + eq4[i1] * sv[o1];
            }
        }
    }
    rows
}

// ───────────────────────────────────────────────────────────────────────────
// The assist's block tree: the run structure that collapses BOTH directions
// of its layer recursion from per-column to per-run work.
// ───────────────────────────────────────────────────────────────────────────

/// The laminar family of runs the assist's layer recursion is constant on: for
/// each layer `ℓ`, the maximal runs of consecutive (deduped) columns whose
/// boundary pair `(t_{y-1}, t_y)` agrees on every bit `≥ ℓ`.
///
/// Both quantities the recursion touches at layer `ℓ` are functions of those
/// bits alone, hence constant on such a run:
///
/// - the transition tag `cd_ℓ = bit_ℓ(t_{y-1}) + 2·bit_ℓ(t_y)`, which selects
///   the layer matrix, and
/// - the suffix vector `S[ℓ] = M_ℓ···M_m·e_S`.
///
/// A layer-`ℓ+1` run is a union of layer-`ℓ` runs, so the runs form a tree, and
/// the recursion collapses in both directions: the suffix vectors DESCEND it
/// ([`assist_suffix_rows_blocked`], 8 multiplies per block instead of per
/// column) and the running column weights ASCEND it ([`fold_partials`], one
/// multiply per block where the dense form spent one per column per layer).
///
/// The registry's shape is `k_t` consecutive columns of height `n_t`, so inside
/// a run the pair advances by `n_t` and its bits above `ℓ` change only every
/// `2^ℓ/n_t` columns: `Σ_ℓ blocks(ℓ) = O(n_cols·log n_t + m)` against the dense
/// `(m + 2)·n_cols`. Keyed off `col_prefix_sums` alone, so all `128·K`
/// statements of a Frobenius batch share one tree — only their weights differ.
struct AssistBlocks {
    /// `starts[ℓ]`: ascending block start indices into the deduped columns.
    /// Layer 0 is one block per column (adjacent equal pairs are already merged
    /// by [`assist_boundaries`]); layer `m + 1` is a single block, every pair
    /// being below `2^{m+1}`.
    starts: Vec<Vec<u32>>,
    /// `parent[ℓ][b]`: the layer-`ℓ+1` block containing layer-`ℓ` block `b`.
    /// Non-decreasing in `b`, and `parent[ℓ][b] ≤ b`.
    parent: Vec<Vec<u32>>,
    /// `first_child[ℓ][B]`: the first layer-`ℓ` block inside layer-`ℓ+1` block
    /// `B` — `parent[ℓ]` inverted. Length `blocks(ℓ+1)`; a block's children are
    /// the contiguous range up to the next entry, which is what lets
    /// [`fold_partials`] write each parent independently.
    first_child: Vec<Vec<u32>>,
    /// `cd[ℓ][b]`: the block's constant transition tag.
    cd: Vec<Vec<u8>>,
    /// Flat-array base of each layer's blocks; `off[m + 2]` is the total.
    off: Vec<usize>,
    n_cols: usize,
}

impl AssistBlocks {
    fn new(bounds: &[(u64, u64, u32)], m: usize) -> Self {
        let n_cols = bounds.len();
        let mut starts: Vec<Vec<u32>> = Vec::with_capacity(m + 2);
        let mut parent: Vec<Vec<u32>> = Vec::with_capacity(m + 1);
        let mut first_child: Vec<Vec<u32>> = Vec::with_capacity(m + 1);
        let mut cd: Vec<Vec<u8>> = Vec::with_capacity(m + 1);
        starts.push((0..n_cols as u32).collect());
        for layer in 0..=m {
            let cur = &starts[layer];
            let mut next: Vec<u32> = Vec::new();
            let mut kids: Vec<u32> = Vec::new();
            let mut par: Vec<u32> = Vec::with_capacity(cur.len());
            let mut tag: Vec<u8> = Vec::with_capacity(cur.len());
            let mut last: Option<(u64, u64)> = None;
            for (b, &s) in cur.iter().enumerate() {
                // Any column of the block represents it: the block is defined
                // by agreement above `layer`, and `cd` reads bit `layer`.
                let (t_c, t_next, _) = bounds[s as usize];
                tag.push(((t_c >> layer) & 1) as u8 + 2 * (((t_next >> layer) & 1) as u8));
                let hi = (t_c >> (layer + 1), t_next >> (layer + 1));
                if last != Some(hi) {
                    next.push(s);
                    kids.push(b as u32);
                    last = Some(hi);
                }
                par.push(next.len() as u32 - 1);
            }
            starts.push(next);
            parent.push(par);
            first_child.push(kids);
            cd.push(tag);
        }
        debug_assert_eq!(starts[m + 1].len(), 1, "all pairs are below 2^(m+1)");
        let mut off = Vec::with_capacity(m + 3);
        let mut acc = 0usize;
        for s in &starts {
            off.push(acc);
            acc += s.len();
        }
        off.push(acc);
        AssistBlocks {
            starts,
            parent,
            first_child,
            cd,
            off,
            n_cols,
        }
    }

    #[inline]
    fn n_blocks(&self, layer: usize) -> usize {
        self.starts[layer].len()
    }

    /// Total slots across all layers — the blocked suffix store's size, against
    /// the dense `(m + 2)·n_cols`.
    #[inline]
    fn total(&self) -> usize {
        self.off[self.off.len() - 1]
    }

    /// Layer-0 partials: each block's summed column weight. (Layer 0 blocks are
    /// singletons, but summing the range keeps this independent of that.)
    fn seed(&self, cols: &[(F128, u64, u64)]) -> Vec<F128> {
        debug_assert_eq!(cols.len(), self.n_cols);
        let s = &self.starts[0];
        (0..s.len())
            .map(|b| {
                let lo = s[b] as usize;
                let hi = s.get(b + 1).map_or(self.n_cols, |&x| x as usize);
                cols[lo..hi]
                    .iter()
                    .fold(F128::ZERO, |acc, &(w, _, _)| acc + w)
            })
            .collect()
    }
}

/// The suffix vectors `S[ℓ]`, one slot per block per layer, flat in the block
/// tree's layout. Descends the tree: a block's vector comes from its parent's at
/// 8 multiplications (two surviving transitions per state), so the build costs
/// `Σ_ℓ 8·blocks(ℓ)` against the dense `8·(m + 2)·n_cols`. Layer 0's `INITIAL`
/// entries are still the columns' full `ĝ` values.
///
/// `par` parallelizes within a layer — for the few-statement callers whose
/// statement-level dispatch can't occupy the pool.
fn assist_suffix_rows_blocked(
    blocks: &AssistBlocks,
    eq4s: &[[F128; 4]],
    sparse: &[[[(usize, usize); 2]; 4]; 4],
    m: usize,
    par: bool,
) -> Vec<[F128; 4]> {
    use rayon::prelude::*;
    #[inline]
    fn step(
        dst: &mut [F128; 4],
        src: &[F128; 4],
        cd: u8,
        eq4: &[F128; 4],
        sparse: &[[[(usize, usize); 2]; 4]; 4],
    ) {
        let rows_cd = &sparse[cd as usize];
        for (s, slot) in dst.iter_mut().enumerate() {
            let (i0, o0) = rows_cd[s][0];
            let (i1, o1) = rows_cd[s][1];
            *slot = eq4[i0] * src[o0] + eq4[i1] * src[o1];
        }
    }

    let mut rows = vec![[F128::ZERO; 4]; blocks.total()];
    rows[blocks.off[m + 1]][STATE_SUCCESS] = F128::ONE;
    for layer in (0..=m).rev() {
        let pbase = blocks.off[layer + 1];
        let (head, tail) = rows.split_at_mut(pbase);
        let dst = &mut head[blocks.off[layer]..];
        let src = &*tail;
        let eq4 = &eq4s[layer];
        let (parent, cd) = (&blocks.parent[layer], &blocks.cd[layer]);
        if par {
            dst.par_chunks_mut(ASSIST_CHUNK)
                .zip(parent.par_chunks(ASSIST_CHUNK))
                .zip(cd.par_chunks(ASSIST_CHUNK))
                .for_each(|((dc, pc), cc)| {
                    for ((slot, &p), &t) in dc.iter_mut().zip(pc).zip(cc) {
                        step(slot, &src[p as usize], t, eq4, sparse);
                    }
                });
        } else {
            for ((slot, &p), &t) in dst.iter_mut().zip(parent).zip(cd) {
                step(slot, &src[p as usize], t, eq4, sparse);
            }
        }
    }
    rows
}

/// The **statement-independent** upper part of the blocked suffix store:
/// layers `[lo, m+1]`, in the flat layout shifted down by `off[lo]`.
///
/// The suffix recurrence at layer `ℓ` reads `eq4s[ℓ]`, which is built from
/// `point_bit(z_row, ℓ)` and `point_bit(rho, ℓ)`; `point_bit` zero-pads, so
/// for `ℓ ≥ z_row.len()` the table is a function of the SHARED `rho` alone —
/// identical for every statement of a Frobenius batch (their `z_row`s differ
/// but have equal length). Building these layers once and sharing them
/// across the `128·K` statements is what keeps the per-statement build to
/// the low layers, where (at uniform column heights) most of the blocks
/// live anyway.
///
/// Values are bit-identical to the corresponding slice of
/// [`assist_suffix_rows_blocked`] — same recurrence, same inputs — pinned by
/// `blocked_low_plus_tail_matches_full`.
fn assist_shared_tail_blocked(
    blocks: &AssistBlocks,
    rho: &[F128],
    sparse: &[[[(usize, usize); 2]; 4]; 4],
    m: usize,
    lo: usize,
) -> Vec<[F128; 4]> {
    use rayon::prelude::*;
    debug_assert!(lo >= 1 && lo <= m + 1);
    let base = blocks.off[lo];
    let mut rows = vec![[F128::ZERO; 4]; blocks.total() - base];
    rows[blocks.off[m + 1] - base][STATE_SUCCESS] = F128::ONE;
    for layer in (lo..=m).rev() {
        // `point_bit(z_row, layer)` is 0 for every statement here.
        let t = build_eq_table(&[F128::ZERO, point_bit(rho, layer)]);
        let eq4 = [t[0], t[1], t[2], t[3]];
        let pbase = blocks.off[layer + 1] - base;
        let (head, tail) = rows.split_at_mut(pbase);
        let dst = &mut head[blocks.off[layer] - base..];
        let src = &*tail;
        let (parent, cd) = (&blocks.parent[layer], &blocks.cd[layer]);
        dst.par_chunks_mut(ASSIST_CHUNK)
            .zip(parent.par_chunks(ASSIST_CHUNK))
            .zip(cd.par_chunks(ASSIST_CHUNK))
            .for_each(|((dc, pc), cc)| {
                for ((slot, &p), &t) in dc.iter_mut().zip(pc).zip(cc) {
                    let rows_cd = &sparse[t as usize];
                    let sv = &src[p as usize];
                    for (s, out) in slot.iter_mut().enumerate() {
                        let (i0, o0) = rows_cd[s][0];
                        let (i1, o1) = rows_cd[s][1];
                        *out = eq4[i0] * sv[o0] + eq4[i1] * sv[o1];
                    }
                }
            });
    }
    rows
}

/// The statement's own LOW layers `[0, lo)` of the blocked suffix store,
/// with layer `lo − 1` reading its parents from the shared `tail`
/// ([`assist_shared_tail_blocked`]). Together the two are slot-for-slot the
/// full [`assist_suffix_rows_blocked`] store.
fn assist_suffix_low_blocked(
    blocks: &AssistBlocks,
    eq4s: &[[F128; 4]],
    sparse: &[[[(usize, usize); 2]; 4]; 4],
    lo: usize,
    tail: &[[F128; 4]],
    par: bool,
) -> Vec<[F128; 4]> {
    use rayon::prelude::*;
    #[inline]
    fn step(
        dst: &mut [F128; 4],
        src: &[F128; 4],
        cd: u8,
        eq4: &[F128; 4],
        sparse: &[[[(usize, usize); 2]; 4]; 4],
    ) {
        let rows_cd = &sparse[cd as usize];
        for (s, slot) in dst.iter_mut().enumerate() {
            let (i0, o0) = rows_cd[s][0];
            let (i1, o1) = rows_cd[s][1];
            *slot = eq4[i0] * src[o0] + eq4[i1] * src[o1];
        }
    }

    debug_assert!(lo >= 1);
    let mut rows = vec![[F128::ZERO; 4]; blocks.off[lo]];
    for layer in (0..lo).rev() {
        let eq4 = &eq4s[layer];
        let (parent, cd) = (&blocks.parent[layer], &blocks.cd[layer]);
        let (head, rest) = rows.split_at_mut(blocks.off[layer + 1]);
        let dst = &mut head[blocks.off[layer]..];
        let src: &[[F128; 4]] = if layer + 1 == lo {
            &tail[..blocks.n_blocks(lo)]
        } else {
            &rest[..blocks.n_blocks(layer + 1)]
        };
        if par {
            dst.par_chunks_mut(ASSIST_CHUNK)
                .zip(parent.par_chunks(ASSIST_CHUNK))
                .zip(cd.par_chunks(ASSIST_CHUNK))
                .for_each(|((dc, pc), cc)| {
                    for ((slot, &p), &t) in dc.iter_mut().zip(pc).zip(cc) {
                        step(slot, &src[p as usize], t, eq4, sparse);
                    }
                });
        } else {
            for ((slot, &p), &t) in dst.iter_mut().zip(parent).zip(cd) {
                step(slot, &src[p as usize], t, eq4, sparse);
            }
        }
    }
    rows
}

/// `eq((t_{y-1}, t_y), σ)` per deduped column, by DESCENDING the block tree:
/// a block's value is its parent's times its layer quadrant — one multiply
/// per block, against the dense `2(m+1)` per column. The layer-0 values are
/// the per-column tensor products, shared by every statement of a batch
/// (they depend on the pairs and `σ` alone); pairing them with the
/// statement's weights afterwards is the same field product reassociated,
/// so `Σ_y w_y·eq_y` equals [`assist_w_at_blocked`] exactly.
fn assist_eq_at_blocked(blocks: &AssistBlocks, sigma: &[F128], m: usize) -> Vec<F128> {
    debug_assert_eq!(sigma.len(), 2 * (m + 1));
    let mut vals = vec![F128::ONE]; // the layer-(m+1) root
    for layer in (0..=m).rev() {
        let (rc, rd) = (sigma[2 * layer], sigma[2 * layer + 1]);
        let (rc1, rd1) = (F128::ONE + rc, F128::ONE + rd);
        let e = [rc1 * rd1, rc * rd1, rc1 * rd, rc * rd];
        let (parent, cd) = (&blocks.parent[layer], &blocks.cd[layer]);
        let mut next = Vec::with_capacity(blocks.n_blocks(layer));
        for (&p, &t) in parent.iter().zip(cd) {
            next.push(vals[p as usize] * e[t as usize]);
        }
        vals = next;
    }
    vals
}

/// Ascend the block tree one layer, folding the layer's two challenges into the
/// weight partials: `p[ℓ+1][B] = Σ_{b ⊆ B} ch4[cd_ℓ(b)]·p[ℓ][b]`.
///
/// This is the entirety of the dense form's running-weight fold
/// (`we_y ·= e_c·e_d`, one multiply per column per layer): `cd_ℓ` is constant on
/// a layer-`ℓ` block, so the factor pulls out of the block's sum by
/// distributivity — one multiply per block, exact. Written as a gather over each
/// parent's contiguous child range, so `par` can hand parents to separate
/// threads; `out` is scratch, swapped into `p` on the way out.
fn fold_partials(
    p: &mut Vec<F128>,
    out: &mut Vec<F128>,
    blocks: &AssistBlocks,
    layer: usize,
    ch4: &[F128; 4],
    par: bool,
) {
    use rayon::prelude::*;
    let (cd, kids) = (&blocks.cd[layer], &blocks.first_child[layer]);
    let n_child = blocks.n_blocks(layer);
    debug_assert_eq!(p.len(), n_child);
    out.clear();
    out.resize(kids.len(), F128::ZERO);
    let gather = |b: usize, slot: &mut F128| {
        let lo = kids[b] as usize;
        let hi = kids.get(b + 1).map_or(n_child, |&x| x as usize);
        *slot = (lo..hi).fold(F128::ZERO, |acc, c| acc + ch4[cd[c] as usize] * p[c]);
    };
    // Chunked, not `par_iter_mut`: the latter splits down to single parents, so
    // the layers where only a handful of blocks survive would pay full rayon
    // fork cost for a few multiplies. One chunk runs inline.
    if par {
        out.par_chunks_mut(ASSIST_CHUNK)
            .enumerate()
            .for_each(|(ci, oc)| {
                let base = ci * ASSIST_CHUNK;
                for (i, slot) in oc.iter_mut().enumerate() {
                    gather(base + i, slot);
                }
            });
    } else {
        for (b, slot) in out.iter_mut().enumerate() {
            gather(b, slot);
        }
    }
    std::mem::swap(p, out);
    debug_assert_eq!(p.len(), blocks.n_blocks(layer + 1));
}

/// The layer's only pass over block-scale state: bucket each block's weight
/// partial against its parent's suffix vector,
/// `B[cd] = Σ_{b: cd_ℓ(b) = cd} p[b]·S[ℓ+1][parent(b)]` — 4 multiplies per
/// block, where the dense form spent 4 per column. Both round messages come
/// from these buckets alone.
///
/// `par` chunks over blocks with XOR-reduced partials (value-identical
/// reassociation) — for callers whose statement count can't occupy the pool.
fn assist_buckets(
    p: &[F128],
    sfx: &[[F128; 4]],
    tail: &[[F128; 4]],
    lo_off: usize,
    blocks: &AssistBlocks,
    layer: usize,
    par: bool,
) -> [[F128; 4]; 4] {
    use rayon::prelude::*;
    let pbase = blocks.off[layer + 1];
    // The parent layer's suffix slots: the statement's own store below the
    // shared boundary (`lo_off = off[lo]`), the statement-independent tail
    // above it. A full store passes `lo_off = usize::MAX`.
    let (src, base) = if pbase < lo_off {
        (sfx, pbase)
    } else {
        (tail, pbase - lo_off)
    };
    let (parent, cd) = (&blocks.parent[layer], &blocks.cd[layer]);
    let body = |pc: &[F128], pp: &[u32], cc: &[u8]| {
        let mut b = [[F128::ZERO; 4]; 4];
        for ((&v, &par), &t) in pc.iter().zip(pp).zip(cc) {
            let s = &src[base + par as usize];
            let bk = &mut b[t as usize];
            bk[0] += v * s[0];
            bk[1] += v * s[1];
            bk[2] += v * s[2];
            bk[3] += v * s[3];
        }
        b
    };
    if par {
        p.par_chunks(ASSIST_CHUNK)
            .zip(parent.par_chunks(ASSIST_CHUNK))
            .zip(cd.par_chunks(ASSIST_CHUNK))
            .map(|((pc, pp), cc)| body(pc, pp, cc))
            .reduce(
                || [[F128::ZERO; 4]; 4],
                |mut x, y| {
                    for (xv, yv) in x.iter_mut().zip(&y) {
                        *xv = add4(xv, yv);
                    }
                    x
                },
            )
    } else {
        body(p, parent, cd)
    }
}

/// `u[c + 2d]ᵀ = b_ℓᵀ·M_ℓ^{(c,d)}`, via the sparse transition rows — the
/// statement-local half of a layer's message, independent of the column scale.
fn assist_u_rows(
    prefix_row: &[F128; 4],
    eq4: &[F128; 4],
    sparse: &[[[(usize, usize); 2]; 4]; 4],
) -> [[F128; 4]; 4] {
    let mut u = [[F128::ZERO; 4]; 4];
    for (cd, uv) in u.iter_mut().enumerate() {
        for (s, &bs) in prefix_row.iter().enumerate() {
            let (i0, o0) = sparse[cd][s][0];
            let (i1, o1) = sparse[cd][s][1];
            uv[o0] += bs * eq4[i0];
            uv[o1] += bs * eq4[i1];
        }
    }
    u
}

/// `W(σ) = Σ_y w_y·Π_ℓ eq(t_{y-1}[ℓ], σ_{c,ℓ})·eq(t_y[ℓ], σ_{d,ℓ})` by the same
/// ascent: the layer's four quadrant products `eq(c, σ_c)·eq(d, σ_d)` are
/// constant on a block, so the walk costs one multiply per block instead of
/// [`assist_w_at`]'s `2(m + 1)` per column — the verifier's only `2^k`-scale
/// work. Value-identical (reassociation of the same field product).
fn assist_w_at_blocked(
    blocks: &AssistBlocks,
    cols: &[(F128, u64, u64)],
    sigma: &[F128],
    m: usize,
) -> F128 {
    debug_assert_eq!(sigma.len(), 2 * (m + 1));
    let mut p = blocks.seed(cols);
    let mut scratch = Vec::with_capacity(p.len());
    for layer in 0..=m {
        let (rc, rd) = (sigma[2 * layer], sigma[2 * layer + 1]);
        let (rc1, rd1) = (F128::ONE + rc, F128::ONE + rd);
        fold_partials(
            &mut p,
            &mut scratch,
            blocks,
            layer,
            &[rc1 * rd1, rc * rd1, rc1 * rd, rc * rd],
            false,
        );
    }
    p[0]
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
    let m = params.m;
    assert_eq!(z_row.len(), params.n);
    assert_eq!(z_col.len(), params.k);
    assert_eq!(z_index.len(), m);
    let bounds = assist_boundaries(params);
    let cols = assist_columns_at(&bounds, z_col);
    let blocks = AssistBlocks::new(&bounds, m);

    let eq4s: Vec<[F128; 4]> = (0..=m)
        .map(|layer| {
            let t = build_eq_table(&[point_bit(z_row, layer), point_bit(z_index, layer)]);
            [t[0], t[1], t[2], t[3]]
        })
        .collect();
    let sparse = assist_sparse_transitions();
    // One statement, so both block-scale passes parallelize within the layer.
    let sfx = assist_suffix_rows_blocked(&blocks, &eq4s, &sparse, m, true);

    // β = Σ_y w_y·ĝ_y — the INITIAL entries of suffix layer 0.
    let mut p = blocks.seed(&cols);
    let beta = p
        .iter()
        .zip(&sfx[blocks.off[0]..])
        .fold(F128::ZERO, |acc, (&w, s)| acc + w * s[STATE_INITIAL]);

    challenger.observe_label(b"flock-jagged-assist-v0");
    challenger.observe_f128(beta);

    let mut prefix_row = [F128::ZERO; 4];
    prefix_row[STATE_INITIAL] = F128::ONE;
    let mut scratch = Vec::with_capacity(p.len());
    let mut ch4: Option<[F128; 4]> = None;
    let mut rounds = Vec::with_capacity(2 * (m + 1));
    for layer in 0..=m {
        // Ascend one layer with the previous layer's challenge quadrants, then
        // the layer's only block-scale pass.
        if let Some(c4) = ch4 {
            fold_partials(&mut p, &mut scratch, &blocks, layer - 1, &c4, true);
        }
        let buckets = assist_buckets(&p, &sfx, &[], usize::MAX, &blocks, layer, true);
        let u = assist_u_rows(&prefix_row, &eq4s[layer], &sparse);

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
        let rd1 = F128::ONE + rd;
        prefix_row = comb4(rd1, &ud0, rd, &ud1);
        // The next layer's ascent folds this layer's `ec·ed` into the weight
        // partials — the quadrant products once, not per column.
        ch4 = Some([rc1 * rd1, rc * rd1, rc1 * rd, rc * rd]);
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
            .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d));

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

    let bounds = assist_boundaries(params);
    let cols = assist_columns_at(&bounds, z_col);
    let w = assist_w_at_blocked(&AssistBlocks::new(&bounds, m), &cols, &rho, m);
    let g = g_hat_eval_cd(z_row, z_index, m, |l| (rho[2 * l], rho[2 * l + 1]));
    (claim == w * g).then_some(proof.beta)
}

// ───────────────────────────────────────────────────────────────────────────
// The batched Frobenius assist (design doc §"The batched Frobenius assist,
// in detail"): proves the Φ-twisted jagged weight evaluation
//   V = Ŵ(ρ) = Σ_i Σ_j c_{i,j} · f̂_t(z_row_i^(2^j), z_col_i^(2^j), ρ)
// — an F-combination of ordinary assist statements at Frobenius-powered
// points (Frobenius commutes with the eq-product structure at Boolean
// selectors; c_{i,j} = the linearized-polynomial coefficients of the
// claims' γ-baked fold maps, `ring_switch::linearized_coefficients`) — by
// ONE sumcheck over the 2(m+1) boundary-bit variables with DETERMINISTIC
// weights (the c_{i,j} are transcript-determined; only the combined scalar
// is used, so plain sumcheck soundness on the combined summand suffices).
// ───────────────────────────────────────────────────────────────────────────

/// Materialize the merged reduction's twisted weight over the dense cube:
/// `W[d] = Σ_i fold_one_slot(eq_row_i[row(d)]·eq_col_i[col(d)], table_i)`
/// for `d < area`, ZERO on the power-of-two tail — the definitional
/// zero-extension (`q`'s committed tail is zero, and the Frobenius
/// assist's branching program computes exactly this extension via its
/// comparison state, so prover table and verifier evaluation agree by
/// construction). `claims` = `(z_row, z_col, γ-baked fold table)` views.
/// One claim's (or claim group's) contribution to the merged weight.
pub(crate) enum MergedWeightClaim<'a> {
    /// A ring-switched claim: its F₂-linear fold table applied to
    /// `eq_row ⊗ eq_col` — additive but not F128-homogeneous, so it cannot
    /// join a scalar group.
    Folded {
        z_row: &'a [F128],
        z_col: &'a [F128],
        table: &'a [F128],
    },
    /// A GROUP of γ-scaled (F128-linear) packed-direct claims sharing one
    /// row point: `Σᵢ γᵢ·eq_rowᵢ(row)·eq_colᵢ(col) =
    /// eq_row(row)·(Σᵢ γᵢ·eq_colᵢ(col))`, so the whole group costs ONE
    /// multiply-sweep against the precombined (already γ-summed) column
    /// table. Exact — field multiplication distributes and the sums
    /// reassociate — so the produced `W` is bit-identical to per-claim
    /// fold-table sweeps. This is what keeps the Φ-pass from scaling with
    /// the circuit path's gather-claim count (~2^c claims, one shared
    /// ρ_row). Borrowed: the same groups feed the multipoint protocol
    /// (`ScalarGroupClaim`) and the anchor.
    Scalar { z_row: &'a [F128], cols: &'a [F128] },
}

pub(crate) fn build_merged_weight_and_prime(
    params: &JaggedParams,
    claims: &[MergedWeightClaim<'_>],
    q: &[F128],
) -> (Vec<F128>, (F128, F128)) {
    use rayon::prelude::*;
    let area = params.area() as usize;
    let n_total = 1usize << params.m;
    enum ColSide<'a> {
        Fold(Vec<F128>, &'a [F128]),
        Combined(&'a [F128]),
    }
    let tabs: Vec<(Vec<F128>, ColSide<'_>)> = claims
        .iter()
        .map(|c| match c {
            MergedWeightClaim::Folded {
                z_row,
                z_col,
                table,
            } => (
                build_eq_table(z_row),
                ColSide::Fold(build_eq_table(z_col), table),
            ),
            MergedWeightClaim::Scalar { z_row, cols } => {
                (build_eq_table(z_row), ColSide::Combined(cols))
            }
        })
        .collect();
    assert_eq!(q.len(), n_total);
    let mut w = crate::scratch::take_f128(n_total);
    // Segmented fill (the JaggedWeight lesson): per chunk, ONE cursor into
    // `col_prefix_sums`, then per column segment a claim-OUTER sweep — the
    // column factor hoisted, rows read sequentially, and one claim's 64 KB
    // fold table hot per sweep. The per-element unrank variant measured
    // ~2.5x slower at M = 30.
    // The merged sumcheck's round-0 prime `(u0, u2)` is fused into the same
    // pass (CHUNK is even, so element pairs never straddle chunks); the
    // dead tail past the area contributes zero on both sides.
    //
    // W stays MATERIALIZED, measured, not by omission: this pass is
    // evaluation-bound, not store-bound — a store-skipping probe at m32
    // (2 RS claims + 1 scalar group) timed 266 ms ST / 39 ms MT against
    // ~240-290 / ~50 with the store, i.e. the fold-table applications
    // dominate and the DRAM write-back is ≤20 ms ST / ~10 MT. Both the
    // "recompute W in a fused prime+fold pass" idea and a virtual W (the
    // assist partners' treatment) re-pay that evaluation — once more per
    // pass or once per round — for a saving bounded by the store + one
    // fold read. Net negative at this claim mix; r_0's Fiat-Shamir
    // dependence on the full prime rules out any single-pass scheme.
    const CHUNK: usize = 1 << 14;
    let ps = &params.col_prefix_sums;
    let prime = w
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(ci, out)| {
            let base = (ci * CHUNK) as u64;
            let end = base + out.len() as u64;
            if base >= area as u64 {
                // Wholly past the jagged area: W is identically zero there
                // and the merged sumcheck's trimmed folds never read it —
                // leave the scratch chunk dirty (the caller zeroes the few
                // guard slots its rounded-up round-0 read can touch).
                return (F128::ZERO, F128::ZERO);
            }
            let live_end = end.min(area as u64);
            // Zero the dead tail of this chunk (past the jagged area).
            out[(live_end - base) as usize..].fill(F128::ZERO);
            let mut first_claim = true;
            for (eq_r, side) in tabs.iter() {
                let mut col = ps.partition_point(|&t| t <= base) - 1;
                let mut e = base;
                while e < live_end {
                    while ps[col + 1] <= e {
                        col += 1;
                    }
                    let seg_end = ps[col + 1].min(live_end);
                    let row0 = (e - ps[col]) as usize;
                    let dst = &mut out[(e - base) as usize..(seg_end - base) as usize];
                    let rows = &eq_r[row0..row0 + dst.len()];
                    match side {
                        ColSide::Fold(eq_c, tab) => {
                            let c_hoist = eq_c[col];
                            if first_claim {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot =
                                        crate::pcs::ring_switch::fold_one_slot(r * c_hoist, tab);
                                }
                            } else {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot +=
                                        crate::pcs::ring_switch::fold_one_slot(r * c_hoist, tab);
                                }
                            }
                        }
                        ColSide::Combined(cols) => {
                            let c_hoist = cols[col];
                            if first_claim {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot = r * c_hoist;
                                }
                            } else {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot += r * c_hoist;
                                }
                            }
                        }
                    }
                    e = seg_end;
                }
                first_claim = false;
            }
            let qc = &q[base as usize..end as usize];
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            for (qp, wp) in qc
                .as_chunks::<2>()
                .0
                .iter()
                .zip(out.as_chunks::<2>().0.iter())
            {
                u0 += qp[0] * wp[0];
                u2 += (qp[0] + qp[1]) * (wp[0] + wp[1]);
            }
            (u0, u2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );
    (w, prime)
}

/// One ring-switch claim's inputs to the Frobenius assist: the word-level
/// row/column point split and the 128 linearized coefficients (γ-baked) of
/// its fold map.
pub struct FrobeniusClaim<'a> {
    pub z_row: &'a [F128],
    pub z_col: &'a [F128],
    pub coeffs: &'a [F128],
}

/// A GROUP of γ-scaled (F128-linear) packed-direct claims sharing one row
/// point, entering the multipoint protocol as ONE untwisted claim:
/// `h(d) = eq(z_row, row(d))·cols[col(d)]` with the members' γ's baked into
/// the merged column weights, and fold map the identity (`Φ(x) = x`, so
/// `c_j = 0` past index 0 and the claim needs a single dual value). The
/// column side of `MergedWeightClaim::Scalar`, reused verbatim.
#[derive(Clone, Copy)]
pub struct ScalarGroupClaim<'a> {
    pub z_row: &'a [F128],
    /// Dense per-column weights, length `2^k`.
    pub cols: &'a [F128],
}

/// Transcript of the batched Frobenius assist. `v` is the claimed twisted
/// evaluation (observed before the rounds); each of the `2(m+1)` rounds
/// sends the degree-2 message `(G(1), G(∞))`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrobeniusAssistProof {
    pub v: F128,
    pub rounds: Vec<(F128, F128)>,
    /// PoW witnesses in round order. Empty when grinding is disabled.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// Per-statement prover state: one (claim, Frobenius power) pair, with the
/// coefficient pre-scaled into the column weights. `sfx` and `p` live in the
/// shared block tree's layout — `p` is the running weight partials at the
/// current layer, shrinking as it ascends.
struct FrobeniusStatement {
    cols: Vec<(F128, u64, u64)>,
    eq4s: Vec<[F128; 4]>,
    sfx: Vec<[F128; 4]>,
    p: Vec<F128>,
    scratch: Vec<F128>,
    prefix_row: [F128; 4],
}

/// Build the `128·K` statements: for claim `i` and Frobenius power `j`, the
/// assist objects at coordinate-wise `2^j`-powered `(z_row, z_col)` with
/// column weights scaled by `c_{i,j}`. Statements with `c_{i,j} = 0` are
/// skipped (their contribution is identically zero). Each entry of `groups`
/// is a pre-merged scalar group with a statement-level coefficient: one spec
/// at Frobenius index 0, its column weights read off the group's dense cols
/// (no squaring ever applies — a group's fold map is the identity).
///
/// `prover` is `Some((blocks, tail, lo))` on the prover, which needs the
/// suffix store and the layer-0 weight partials — each statement builds only
/// its low layers `[0, lo)`, sharing `tail` above ([`assist_shared_tail_blocked`]).
/// The verifier takes `None` and touches neither.
fn frobenius_statements(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[(ScalarGroupClaim<'_>, F128)],
    rho: &[F128],
    bounds: &[(u64, u64, u32)],
    prover: Option<(&AssistBlocks, &[[F128; 4]], usize)>,
) -> Vec<FrobeniusStatement> {
    use rayon::prelude::*;
    let m = params.m;
    let sparse = assist_sparse_transitions();
    enum SpecCols<'b> {
        Point(Vec<F128>),
        Weights(&'b [F128]),
    }
    let mut specs: Vec<(Vec<F128>, SpecCols<'_>, F128)> = Vec::new();
    for claim in claims {
        assert_eq!(claim.coeffs.len(), 128);
        let mut zr = claim.z_row.to_vec();
        let mut zc = claim.z_col.to_vec();
        for &c in claim.coeffs.iter() {
            if !c.is_zero() {
                specs.push((zr.clone(), SpecCols::Point(zc.clone()), c));
            }
            for x in zr.iter_mut() {
                *x = *x * *x;
            }
            for x in zc.iter_mut() {
                *x = *x * *x;
            }
        }
    }
    for (g, coeff) in groups {
        specs.push((g.z_row.to_vec(), SpecCols::Weights(g.cols), *coeff));
    }
    // MERGE specs sharing a row point into one statement. A statement's
    // whole contribution — its seed, every layer pass, and the verifier's
    // closed-form expectation — is LINEAR in its per-column weights, and
    // everything else it carries (`eq4s`, the suffix rows) depends only on
    // `(z_row, ρ)`. So specs with identical `z_row` collapse into one
    // statement whose column weights are the γ-weighted sum: exact (field
    // sums and products reassociate), hence transcript-identical — the
    // assist's `V` and round messages are sums over statements of forms
    // linear in the weights. The circuit path's gather claims all share
    // ρ_row, so its ~2^c statements become ONE; a ring-switched claim's 128
    // Frobenius twists have distinct squared rows and stay singletons.
    let mut merged: Vec<(Vec<F128>, Vec<(SpecCols<'_>, F128)>)> = Vec::new();
    for (zr, zc, c) in specs {
        match merged.iter_mut().find(|(g, _)| *g == zr) {
            Some((_, members)) => members.push((zc, c)),
            None => merged.push((zr, vec![(zc, c)])),
        }
    }
    // The suffix build parallelizes within the layer only when there are too
    // few statements for this outer dispatch to occupy the pool.
    let inner = merged.len() < 16;
    merged
        .into_par_iter()
        .map(|(zr, members)| {
            let mut cols: Vec<(F128, u64, u64)> = Vec::new();
            for (i, (zc, c)) in members.iter().enumerate() {
                let mut cs = match zc {
                    SpecCols::Point(zc) => assist_columns_at(bounds, zc),
                    SpecCols::Weights(w) => weights_columns_at(bounds, w),
                };
                for (w, _, _) in cs.iter_mut() {
                    *w *= *c;
                }
                if i == 0 {
                    cols = cs;
                } else {
                    debug_assert_eq!(cols.len(), cs.len());
                    for (dst, src) in cols.iter_mut().zip(cs) {
                        debug_assert_eq!((dst.1, dst.2), (src.1, src.2));
                        dst.0 += src.0;
                    }
                }
            }
            let eq4s: Vec<[F128; 4]> = (0..=m)
                .map(|layer| {
                    let t = build_eq_table(&[point_bit(&zr, layer), point_bit(rho, layer)]);
                    [t[0], t[1], t[2], t[3]]
                })
                .collect();
            let (sfx, p) = match prover {
                Some((b, tail, lo)) => (
                    assist_suffix_low_blocked(b, &eq4s, &sparse, lo, tail, inner),
                    b.seed(&cols),
                ),
                None => (Vec::new(), Vec::new()),
            };
            let scratch = Vec::with_capacity(p.len());
            let mut prefix_row = [F128::ZERO; 4];
            prefix_row[STATE_INITIAL] = F128::ONE;
            FrobeniusStatement {
                cols,
                eq4s,
                sfx,
                p,
                scratch,
                prefix_row,
            }
        })
        .collect()
}

/// One statement's per-layer pass over the block tree: ascend one layer with the
/// PREVIOUS layer's challenge quadrants (nothing else reads the partials in
/// between, so the fold rides this pass rather than costing its own dispatch),
/// then bucket the partials against their parents' suffix vectors and form the
/// u-vectors from the prefix row. Shared by the chunked round dispatch of
/// [`prove_frobenius_assist`]; `par` works within the statement for callers
/// whose statement count can't occupy the pool.
fn frobenius_layer_pass(
    st: &mut FrobeniusStatement,
    blocks: &AssistBlocks,
    tail: &[[F128; 4]],
    lo_off: usize,
    layer: usize,
    prev_ch4: Option<&[F128; 4]>,
    sparse: &[[[(usize, usize); 2]; 4]; 4],
    par: bool,
) -> ([[F128; 4]; 4], [[F128; 4]; 4]) {
    if let Some(ch4) = prev_ch4 {
        fold_partials(&mut st.p, &mut st.scratch, blocks, layer - 1, ch4, par);
    }
    let buckets = assist_buckets(&st.p, &st.sfx, tail, lo_off, blocks, layer, par);
    let u = assist_u_rows(&st.prefix_row, &st.eq4s[layer], sparse);
    (u, buckets)
}

/// Prover for the batched Frobenius assist. Same per-statement algebra as
/// [`prove_assist`] (Lemma 4.6 streaming), with the round messages summed
/// across statements — the sumcheck runs on the combined summand
/// `H(u,v) = Σ_stmt U_stmt(u,v)·ĝ_stmt(u,v)` (coefficients pre-scaled into
/// the `U` weights).
pub fn prove_frobenius_assist<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[(ScalarGroupClaim<'_>, F128)],
    rho: &[F128],
    challenger: &mut C,
) -> FrobeniusAssistProof {
    prove_frobenius_assist_with_grinding(params, claims, groups, rho, 0, challenger)
}

/// [`prove_frobenius_assist`] with a PoW witness before every quadratic
/// sumcheck challenge.
pub fn prove_frobenius_assist_with_grinding<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[(ScalarGroupClaim<'_>, F128)],
    rho: &[F128],
    round_grinding_bits: u32,
    challenger: &mut C,
) -> FrobeniusAssistProof {
    use rayon::prelude::*;
    let m = params.m;
    assert_eq!(rho.len(), m);
    let trace = std::env::var("PCS_TRACE").is_ok();
    let t = std::time::Instant::now();
    let sparse = assist_sparse_transitions();
    let bounds = assist_boundaries(params);
    let blocks = AssistBlocks::new(&bounds, m);
    // Layers ≥ lo of the suffix store are statement-independent (their eq
    // tables read only the shared `rho`); build them once for all 128·K
    // statements. Per statement only the low layers remain.
    let lo = params.n.clamp(1, m + 1);
    let tail = assist_shared_tail_blocked(&blocks, rho, &sparse, m, lo);
    let lo_off = blocks.off[lo];
    let mut sts = frobenius_statements(
        params,
        claims,
        groups,
        rho,
        &bounds,
        Some((&blocks, &tail, lo)),
    );
    if trace {
        eprintln!(
            "    [frobenius] statements + suffix rows (x{}, {} low + {} shared blocks vs {} dense): {:6.2} ms",
            sts.len(),
            lo_off,
            blocks.total() - lo_off,
            (m + 2) * blocks.n_cols,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = std::time::Instant::now();

    let v = sts
        .par_iter()
        .map(|st| {
            st.p.iter()
                .zip(&st.sfx[blocks.off[0]..])
                .map(|(&w, s)| w * s[STATE_INITIAL])
                .fold(F128::ZERO, |a, x| a + x)
        })
        .reduce(|| F128::ZERO, |a, b| a + b);

    challenger.observe_label(b"flock-frobenius-assist-v0");
    challenger.observe_f128(v);

    let mut ch4: Option<[F128; 4]> = None;
    let mut rounds = Vec::with_capacity(2 * (m + 1));
    let mut grinding_nonces = Vec::with_capacity((round_grinding_bits != 0) as usize * 2 * (m + 1));
    for layer in 0..=m {
        // Per-statement block pass: ascend one layer with the previous layer's
        // challenges, bucket the weight partials against their parents' suffix
        // vectors, then the statement's u-vectors from its prefix row. Messages
        // sum. Chunked dispatch: the per-statement work is ~a few thousand
        // multiplies, so per-statement rayon tasks are overhead-bound at
        // 128K statements x 2(m+1) rounds. 8 statements per task keeps
        // ~32 tasks per round.
        let mut per: Vec<([[F128; 4]; 4], [[F128; 4]; 4])> =
            vec![([[F128::ZERO; 4]; 4], [[F128::ZERO; 4]; 4]); sts.len()];
        let c4 = ch4.as_ref();
        // Per-layer barriers cost ~tens of µs against ~5 multiplies per block.
        // Block-parallelism only pays past ~2^14 blocks per statement.
        let inner_par = sts.len() < 16 && blocks.n_blocks(layer) >= 64 * ASSIST_CHUNK;
        if inner_par {
            // Few statements over many blocks (the multipoint anchor's K):
            // parallelize WITHIN each statement — same values,
            // XOR-reassociated.
            for (st, o) in sts.iter_mut().zip(per.iter_mut()) {
                *o = frobenius_layer_pass(st, &blocks, &tail, lo_off, layer, c4, &sparse, true);
            }
        } else {
            sts.par_chunks_mut(8)
                .zip(per.par_chunks_mut(8))
                .for_each(|(stc, oc)| {
                    for (st, o) in stc.iter_mut().zip(oc.iter_mut()) {
                        *o = frobenius_layer_pass(
                            st, &blocks, &tail, lo_off, layer, c4, &sparse, false,
                        );
                    }
                });
        }

        // c-round message, summed across statements.
        let mut g_one = F128::ZERO;
        let mut g_inf = F128::ZERO;
        for (u, buckets) in &per {
            g_one += dot4(&u[1], &buckets[1]) + dot4(&u[3], &buckets[3]);
            g_inf += dot4(&add4(&u[0], &u[1]), &add4(&buckets[0], &buckets[1]))
                + dot4(&add4(&u[2], &u[3]), &add4(&buckets[2], &buckets[3]));
        }
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let rc = if round_grinding_bits != 0 {
            let (nonce, rc) = challenger.grind_pow_and_sample_f128(round_grinding_bits);
            grinding_nonces.push(nonce);
            rc
        } else {
            challenger.sample_f128()
        };
        rounds.push((g_one, g_inf));

        // d-round from the same buckets, folded at rc.
        let rc1 = F128::ONE + rc;
        let mut g_one = F128::ZERO;
        let mut g_inf = F128::ZERO;
        let mut folded: Vec<([F128; 4], [F128; 4])> = Vec::with_capacity(per.len());
        for (u, buckets) in &per {
            let ud0 = comb4(rc1, &u[0], rc, &u[1]);
            let ud1 = comb4(rc1, &u[2], rc, &u[3]);
            let d0 = comb4(rc1, &buckets[0], rc, &buckets[1]);
            let d1 = comb4(rc1, &buckets[2], rc, &buckets[3]);
            g_one += dot4(&ud1, &d1);
            g_inf += dot4(&add4(&ud0, &ud1), &add4(&d0, &d1));
            folded.push((ud0, ud1));
        }
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let rd = if round_grinding_bits != 0 {
            let (nonce, rd) = challenger.grind_pow_and_sample_f128(round_grinding_bits);
            grinding_nonces.push(nonce);
            rd
        } else {
            challenger.sample_f128()
        };
        rounds.push((g_one, g_inf));

        let rd1 = F128::ONE + rd;
        for (st, (ud0, ud1)) in sts.iter_mut().zip(&folded) {
            st.prefix_row = comb4(rd1, ud0, rd, ud1);
        }
        // The next layer's ascent folds `ec·ed` per block; precompute the four
        // quadrant products once (multiplication associativity — value-
        // identical to the two-multiply form).
        ch4 = Some([rc1 * rd1, rc * rd1, rc1 * rd, rc * rd]);
    }

    if trace {
        eprintln!(
            "    [frobenius] v + rounds: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    FrobeniusAssistProof {
        v,
        rounds,
        grinding_nonces,
    }
}

/// What the DEFERRED verify exports from the assist: the final point `σ`
/// (every jagged-layout claim's column point) and each merged statement's
/// count-dependent factor `w_st` (anchor coefficients baked), in statement
/// order — [`frobenius_statements`]'s merge order, which is deterministic:
/// claims in claim order, then groups in group order, specs sharing a row
/// point collapsed into the first occurrence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssistDefer {
    pub sigma: Vec<F128>,
    pub statement_ws: Vec<F128>,
}

/// Verifier for the batched Frobenius assist: replays the rounds against
/// `proof.v` and checks the final relation
/// `claim == Σ_stmt U_stmt(σ)·ĝ_stmt(σ)`. On success returns the verified
/// twisted evaluation `V = Ŵ(ρ)`.
pub fn verify_frobenius_assist<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[(ScalarGroupClaim<'_>, F128)],
    rho: &[F128],
    proof: &FrobeniusAssistProof,
    challenger: &mut C,
) -> Option<F128> {
    verify_frobenius_assist_with_grinding(params, claims, groups, rho, proof, 0, challenger)
}

/// [`verify_frobenius_assist`] with a PoW check before every quadratic
/// sumcheck challenge.
pub fn verify_frobenius_assist_with_grinding<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[(ScalarGroupClaim<'_>, F128)],
    rho: &[F128],
    proof: &FrobeniusAssistProof,
    round_grinding_bits: u32,
    challenger: &mut C,
) -> Option<F128> {
    verify_frobenius_assist_core(
        params,
        claims,
        groups,
        rho,
        proof,
        round_grinding_bits,
        challenger,
        None,
    )
}

fn verify_frobenius_assist_core<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[(ScalarGroupClaim<'_>, F128)],
    rho: &[F128],
    proof: &FrobeniusAssistProof,
    round_grinding_bits: u32,
    challenger: &mut C,
    defer: Option<&mut Option<AssistDefer>>,
) -> Option<F128> {
    use rayon::prelude::*;
    let m = params.m;
    if proof.rounds.len() != 2 * (m + 1) {
        return None;
    }
    let expected_nonces = if round_grinding_bits == 0 {
        0
    } else {
        proof.rounds.len()
    };
    if proof.grinding_nonces.len() != expected_nonces {
        return None;
    }
    // `VERIFY_TRACE` sub-split of the assist — it dominates the Merkle-table
    // verify, so its three phases are worth separating: the transcript replay,
    // building the 128·K statements (a `2^k`-column `eq` table each), and the
    // per-statement `W(σ)` walk + boundary DP.
    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let tfmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };
    challenger.observe_label(b"flock-frobenius-assist-v0");
    challenger.observe_f128(proof.v);

    let t = std::time::Instant::now();
    let mut claim = proof.v;
    let mut sigma = Vec::with_capacity(2 * (m + 1));
    for (round, &(g_one, g_inf)) in proof.rounds.iter().enumerate() {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = if round_grinding_bits != 0 {
            challenger
                .verify_pow_and_sample_f128(proof.grinding_nonces[round], round_grinding_bits)?
        } else {
            challenger.sample_f128()
        };
        claim = fold_round_claim(claim, g_one, g_inf, r);
        sigma.push(r);
    }
    if trace {
        eprintln!(
            "          [fro-v] round replay ({} rounds): {}",
            proof.rounds.len(),
            tfmt(t.elapsed().as_secs_f64())
        );
    }

    let bounds = assist_boundaries(params);
    let blocks = AssistBlocks::new(&bounds, m);
    let t = std::time::Instant::now();
    let sts = frobenius_statements(params, claims, groups, rho, &bounds, None);
    if trace {
        eprintln!(
            "          [fro-v] statements (x{}, {} cols each): {}",
            sts.len(),
            sts.first().map_or(0, |s| s.cols.len()),
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    // `eq(pair, σ)` per column, hoisted out of the per-statement loop — one
    // tree descent shared by all statements; each statement then pays a
    // plain weighted dot. Same field products as the per-statement ascent
    // ([`assist_w_at_blocked`]), reassociated, so `w` is bit-identical.
    let t = std::time::Instant::now();
    let eq_cols = assist_eq_at_blocked(&blocks, &sigma, m);
    if trace {
        eprintln!(
            "          [fro-v] eq descent ({} blocks, once for {} statements): {}",
            blocks.total(),
            sts.len(),
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    let t = std::time::Instant::now();
    // Per statement: (contribution, w) — `w` is the count-dependent factor
    // the deferred path exports; the sequential re-sum is the same XOR fold
    // the parallel reduce performed, so the check is value-identical.
    let parts: Vec<(F128, F128)> = sts
        .par_iter()
        .map(|st| {
            let w = st
                .cols
                .iter()
                .zip(&eq_cols)
                .fold(F128::ZERO, |acc, (&(w, _, _), &e)| acc + w * e);
            let mut g = [F128::ZERO; 4];
            g[STATE_SUCCESS] = F128::ONE;
            let sparse = assist_sparse_transitions();
            for layer in (0..=m).rev() {
                let eq4 = &st.eq4s[layer];
                let rc = sigma[2 * layer];
                let rd = sigma[2 * layer + 1];
                let e = [
                    (F128::ONE + rc) * (F128::ONE + rd),
                    rc * (F128::ONE + rd),
                    (F128::ONE + rc) * rd,
                    rc * rd,
                ];
                let mut prev = [F128::ZERO; 4];
                for (cd, &ecd) in e.iter().enumerate() {
                    for (s, slot) in prev.iter_mut().enumerate() {
                        let (i0, o0) = sparse[cd][s][0];
                        let (i1, o1) = sparse[cd][s][1];
                        *slot += ecd * (eq4[i0] * g[o0] + eq4[i1] * g[o1]);
                    }
                }
                g = prev;
            }
            (w * g[STATE_INITIAL], w)
        })
        .collect();
    let expect = parts.iter().fold(F128::ZERO, |a, &(c, _)| a + c);
    if trace {
        eprintln!(
            "          [fro-v] per-statement dot + boundary DP (x{}): {}",
            sts.len(),
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    if let Some(out) = defer {
        *out = Some(AssistDefer {
            sigma,
            statement_ws: parts.iter().map(|&(_, w)| w).collect(),
        });
    }
    (claim == expect).then_some(proof.v)
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

// ---------------------------------------------------------------------------
// Multipoint twisted evaluation — PROTOTYPE (research note §"Scaling in the
// number of columns"). Replaces the batched Frobenius assist's 128·K
// boundary-DP statements with: the prover SENDS the 128·K dual-form values
// A_{i,j} = f̂_jag(z_i, ρ^{2^-j}) (Remark "dual form"), the verifier
// recombines V = Σ c_{i,j}·A_{i,j}^{2^j} itself, and the values are bound by
// ONE γ-batched product sumcheck over the dense domain against the twisted-eq
// weight g_d = Σ_j γ^j·eq(ρ,d)^{2^-j} — whose MLE the verifier evaluates in
// closed form (twisted EQ collapses by the tensor identity; twisted JAGGED
// does not) — anchored by ONE untwisted assist at the endpoint. Per-column
// proof work drops 128K× → 1×; the suffix-state footprint disappears; what
// remains column-linear is only COMPUTING the sent values (a pure
// computation, no protocol coupling).
// ---------------------------------------------------------------------------

/// Images of the `F₂`-basis under square root (`x ↦ x^{2^127}`, the inverse
/// Frobenius) — square root is `F₂`-linear, so this table defines the map.
fn sqrt_basis() -> &'static [F128; 128] {
    static T: std::sync::OnceLock<[F128; 128]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [F128::ZERO; 128];
        for (b, slot) in t.iter_mut().enumerate() {
            let mut x = if b < 64 {
                F128::new(1u64 << b, 0)
            } else {
                F128::new(0, 1u64 << (b - 64))
            };
            for _ in 0..127 {
                x = x * x;
            }
            *slot = x;
        }
        t
    })
}

/// The inverse Frobenius `x ↦ x^{2^{-1}} = x^{2^127}` (the square root in
/// F₂₁₂₈), via the basis-image table: 128 table adds instead of 127
/// squarings. The recursion tower's walkers use it too; `frob_inv_matches_127_squarings`
/// pins the two derivations equal.
pub fn frob_inv(x: F128) -> F128 {
    let t = sqrt_basis();
    let mut acc = F128::ZERO;
    let mut lo = x.lo;
    while lo != 0 {
        acc += t[lo.trailing_zeros() as usize];
        lo &= lo - 1;
    }
    let mut hi = x.hi;
    while hi != 0 {
        acc += t[64 + hi.trailing_zeros() as usize];
        hi &= hi - 1;
    }
    acc
}

/// `ρ^{2^{-j}}` for `j = 0..128`: coordinate-wise inverse-Frobenius chains.
fn rho_inverse_powers(rho: &[F128]) -> Vec<Vec<F128>> {
    let mut out: Vec<Vec<F128>> = Vec::with_capacity(128);
    out.push(rho.to_vec());
    for j in 1..128 {
        let prev: Vec<F128> = out[j - 1].iter().map(|&x| frob_inv(x)).collect();
        out.push(prev);
    }
    out
}

/// `ĝ(x) = Σ_j γ^j·eq(ρ^{2^{-j}}, x)` in closed form (Lemma "twisted eq"):
/// `128·|x|` multiplications, no polynomial materialized. Shared by the
/// multipoint prover and verifier — both bake it into the anchor's
/// coefficients.
fn twisted_eq_at(gpow: &[F128], rho_pows: &[Vec<F128>], x: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (j, rj) in rho_pows.iter().enumerate() {
        let mut prod = gpow[j];
        for (i, &xi) in x.iter().enumerate() {
            let y = rj[i];
            prod *= xi * y + (F128::ONE + xi) * (F128::ONE + y);
        }
        acc += prod;
    }
    acc
}

/// `eq(a, b) = Π_t (a_t·b_t + (1+a_t)(1+b_t))` — the untwisted partner's
/// endpoint factor.
fn eq_at(a: &[F128], b: &[F128]) -> F128 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(F128::ONE, |acc, (&x, &y)| {
        acc * (x * y + (F128::ONE + x) * (F128::ONE + y))
    })
}

/// Basis images of `x ↦ x^{2^{-j}}` for every `j` (level 0 = identity).
fn inv_frob_basis() -> &'static Vec<[F128; 128]> {
    static T: std::sync::OnceLock<Vec<[F128; 128]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut levels: Vec<[F128; 128]> = Vec::with_capacity(128);
        let mut cur = [F128::ZERO; 128];
        for (b, slot) in cur.iter_mut().enumerate() {
            *slot = if b < 64 {
                F128::new(1u64 << b, 0)
            } else {
                F128::new(0, 1u64 << (b - 64))
            };
        }
        levels.push(cur);
        for j in 1..128 {
            let mut nxt = [F128::ZERO; 128];
            for b in 0..128 {
                nxt[b] = frob_inv(levels[j - 1][b]);
            }
            levels.push(nxt);
        }
        levels
    })
}

/// 16 byte-tables applying an `F₂`-linear map given its basis images (the
/// `fold_one_slot` pattern: 16 lookups + XORs per word).
fn linear_byte_tables(images: &[F128; 128]) -> Vec<[F128; 256]> {
    let mut tables = vec![[F128::ZERO; 256]; 16];
    for (k, table) in tables.iter_mut().enumerate() {
        for v in 1usize..256 {
            let low = crate::bits::lowest_one(v);
            table[v] = table[v ^ low] + images[8 * k + low.trailing_zeros() as usize];
        }
    }
    tables
}

#[inline]
fn apply_linear_tables(tables: &[[F128; 256]], x: F128) -> F128 {
    let mut acc = F128::ZERO;
    for k in 0..8 {
        acc += tables[k][((x.lo >> (8 * k)) & 0xFF) as usize];
        acc += tables[8 + k][((x.hi >> (8 * k)) & 0xFF) as usize];
    }
    acc
}

/// All columns' full BP values `ĝ(y)` at `(z_row, z_index)`, exploiting two
/// layers of structure:
///
/// - INCREMENTAL suffix DP: `S[ℓ]` depends only on boundary bits `≥ ℓ`, and
///   consecutive prefix sums share their high bits, so a column recomputes
///   only from its highest changed bit down — amortized `O(log stride)`
///   layers instead of `m + 1`.
/// - STRIDED low tables: the value is the matrix product
///   `e_I·M_0⋯M_m·e_S`, splittable at any layer `ℓ` into
///   `(low row-vector)·S[ℓ]`. Within a run of equal heights `h` (any
///   integer), the pair's low-`ℓ` bits are a function of `t_c mod 2^ℓ`
///   alone, so ONE table of `2^ℓ` row-vectors (doubling build,
///   `16·2^ℓ` multiplies) serves the whole run; with `ℓ ≈ log h + 2` the
///   suffix side changes rarely, and each column costs one 4-multiply dot.
///
/// Per run the cheaper of the two is chosen. Values are exactly those of
/// the full per-column DP — unchanged layers ARE the stored values, and the
/// table split is a reassociation of the same field product.
fn assist_g_values(
    cols: &[(F128, u64, u64)],
    eq4s: &[[F128; 4]],
    sparse: &[[[(usize, usize); 2]; 4]; 4],
    m: usize,
) -> Vec<F128> {
    let mut sfx = vec![[F128::ZERO; 4]; m + 2];
    sfx[m + 1][STATE_SUCCESS] = F128::ONE;
    let mut out = Vec::with_capacity(cols.len());
    let mut prev: Option<(u64, u64)> = None;
    // sfx[ℓ] is valid (w.r.t. `prev`) for ℓ ≥ valid_floor.
    let mut valid_floor = m + 1;

    // Recompute sfx layers [floor, start] descending for the pair
    // (t_c, t_next); sfx[start + 1] must be valid.
    let refresh = |sfx: &mut Vec<[F128; 4]>, t_c: u64, t_next: u64, start: usize, floor: usize| {
        let mut layer = start + 1;
        while layer > floor {
            layer -= 1;
            let cd = ((t_c >> layer) & 1) as usize + 2 * ((t_next >> layer) & 1) as usize;
            let rows_cd = &sparse[cd];
            let eq4 = &eq4s[layer];
            let (lo, hi) = sfx.split_at_mut(layer + 1);
            let dst = &mut lo[layer];
            let src = &hi[0];
            for (s, slot) in dst.iter_mut().enumerate() {
                let (i0, o0) = rows_cd[s][0];
                let (i1, o1) = rows_cd[s][1];
                *slot = eq4[i0] * src[o0] + eq4[i1] * src[o1];
            }
        }
    };
    // Recompute start: highest changed bit vs `prev`, raised to cover any
    // stale layers between the target floor and the current valid floor.
    let start_for =
        |prev: Option<(u64, u64)>, t_c: u64, t_next: u64, valid_floor: usize, floor: usize| {
            let top = match prev {
                None => m,
                Some((pc, pd)) => {
                    let diff = (t_c ^ pc) | (t_next ^ pd);
                    if diff == 0 {
                        0usize
                    } else {
                        (63 - diff.leading_zeros() as usize).min(m)
                    }
                }
            };
            let stale_top = if valid_floor > floor {
                valid_floor - 1
            } else {
                0
            };
            top.max(stale_top)
        };

    let mut i = 0;
    while i < cols.len() {
        // Maximal run of equal stride starting here (prefix sums are
        // contiguous by construction, so equal height IS the run condition).
        let h = cols[i].2 - cols[i].1;
        let mut end = i + 1;
        while end < cols.len() && cols[end].2 - cols[end].1 == h {
            end += 1;
        }
        let k = end - i;

        // Split layer for the low table, and the adaptive choice: table
        // build 16·2^l + ~6 multiplies/col vs ~8·(l+1) multiplies/col
        // incremental.
        let l = if h == 0 {
            0
        } else {
            (64 - h.leading_zeros() as usize + 2).min(m + 1)
        };
        let table_cost = 16u128 * (1u128 << l) + 6 * (k as u128);
        let inc_cost = 8 * (k as u128) * (l as u128 + 1);
        let use_table = h > 0 && l <= m && table_cost < inc_cost;

        if use_table {
            // Doubling build of the low row-vectors over l bits: bit ℓ of
            // the pair is (v_ℓ, bit ℓ of v + h) — both functions of
            // v = t_c mod 2^l.
            let mut table: Vec<[F128; 4]> = Vec::with_capacity(1 << l);
            let mut seed = [F128::ZERO; 4];
            seed[STATE_INITIAL] = F128::ONE;
            table.push(seed);
            for layer in 0..l {
                let mut next_t: Vec<[F128; 4]> = vec![[F128::ZERO; 4]; 1 << (layer + 1)];
                let eq4 = &eq4s[layer];
                for (v, dst) in next_t.iter_mut().enumerate() {
                    let src = &table[v & ((1 << layer) - 1)];
                    let c_bit = (v >> layer) & 1;
                    let d_bit = ((v as u64 + h) >> layer) & 1;
                    let cd = c_bit + 2 * d_bit as usize;
                    let rows_cd = &sparse[cd];
                    for (s, &sv) in src.iter().enumerate() {
                        let (i0, o0) = rows_cd[s][0];
                        let (i1, o1) = rows_cd[s][1];
                        dst[o0] += sv * eq4[i0];
                        dst[o1] += sv * eq4[i1];
                    }
                }
                table = next_t;
            }
            let mask = (1u64 << l) - 1;
            for &(_, t_c, t_next) in &cols[i..end] {
                let start = start_for(prev, t_c, t_next, valid_floor, l);
                if start >= l {
                    refresh(&mut sfx, t_c, t_next, start, l);
                    valid_floor = l;
                } else {
                    // No refresh: layers touched by the (low) changed bits
                    // become stale; everything above stays valid.
                    valid_floor = valid_floor.max(start + 1);
                }
                prev = Some((t_c, t_next));
                let row = &table[(t_c & mask) as usize];
                let s_l = &sfx[l];
                out.push(dot4(row, s_l));
            }
        } else {
            for &(_, t_c, t_next) in &cols[i..end] {
                if prev == Some((t_c, t_next)) && valid_floor == 0 {
                    out.push(sfx[0][STATE_INITIAL]);
                    continue;
                }
                let start = start_for(prev, t_c, t_next, valid_floor, 0);
                refresh(&mut sfx, t_c, t_next, start, 0);
                valid_floor = 0;
                prev = Some((t_c, t_next));
                out.push(sfx[0][STATE_INITIAL]);
            }
        }
        i = end;
    }
    out
}

/// The 128 dual-form values per claim: `A_j = f̂_jag(z_r, z_c, ρ^{2^{-j}})` —
/// z-points UNpowered (the dual form), so the column weights are shared
/// across `j`; only the ρ-slot of the transition weights varies. Pure
/// computation (no transcript interaction) — parallel over `j`.
fn multipoint_values(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    rho_pows: &[Vec<F128>],
) -> Vec<Vec<F128>> {
    use rayon::prelude::*;
    let m = params.m;
    let sparse = assist_sparse_transitions();
    claims
        .iter()
        .map(|claim| {
            let cols = assist_columns(params, claim.z_col);
            (0..128usize)
                .into_par_iter()
                .map(|j| {
                    let rj = &rho_pows[j];
                    let eq4s: Vec<[F128; 4]> = (0..=m)
                        .map(|layer| {
                            let t = build_eq_table(&[
                                point_bit(claim.z_row, layer),
                                point_bit(rj, layer),
                            ]);
                            [t[0], t[1], t[2], t[3]]
                        })
                        .collect();
                    let g = assist_g_values(&cols, &eq4s, &sparse, m);
                    cols.iter()
                        .zip(&g)
                        .map(|(&(w, _, _), &gv)| w * gv)
                        .fold(F128::ZERO, |a, x| a + x)
                })
                .collect()
        })
        .collect()
}

/// The single dual-form value per scalar group: `B_k = ĥ_k(ρ)` — the
/// group's fold map is the identity, so only the untwisted `j = 0` point
/// exists and one DP sweep replaces a claim's 128. Pure computation
/// (no transcript interaction) — parallel over groups.
fn multipoint_group_values(
    params: &JaggedParams,
    groups: &[ScalarGroupClaim<'_>],
    rho: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;
    let m = params.m;
    let sparse = assist_sparse_transitions();
    let bounds = assist_boundaries(params);
    groups
        .par_iter()
        .map(|g| {
            let cols = weights_columns_at(&bounds, g.cols);
            let eq4s: Vec<[F128; 4]> = (0..=m)
                .map(|layer| {
                    let t = build_eq_table(&[point_bit(g.z_row, layer), point_bit(rho, layer)]);
                    [t[0], t[1], t[2], t[3]]
                })
                .collect();
            let gv = assist_g_values(&cols, &eq4s, &sparse, m);
            cols.iter()
                .zip(&gv)
                .map(|(&(w, _, _), &x)| w * x)
                .fold(F128::ZERO, |a, b| a + b)
        })
        .collect()
}

/// The F128 basis element `2^b` (as an F₂-vector coordinate).
#[inline]
fn basis_elem(b: usize) -> F128 {
    if b < 64 {
        F128::new(1u64 << b, 0)
    } else {
        F128::new(0, 1u64 << (b - 64))
    }
}

/// Split-eq evaluator over the REMAINING coordinates of a low-bit-first
/// fold: after `i` rounds the un-bound tensor is `eq(ρ[i..], ·)`, and two
/// half-size tables give any point in one multiply — `Σ_i 2^((m−i)/2)`
/// total build cost over all rounds, O(√domain), instead of a materialized
/// (and re-folded) `2^(m−i)` vector per round.
struct SplitEq {
    lo: Vec<F128>,
    hi: Vec<F128>,
    n_lo: usize,
    mask: usize,
}

impl SplitEq {
    fn new(coords: &[F128]) -> Self {
        let n_lo = coords.len() / 2;
        Self {
            lo: build_eq_table(&coords[..n_lo]),
            hi: build_eq_table(&coords[n_lo..]),
            n_lo,
            mask: (1usize << n_lo) - 1,
        }
    }

    /// `eq(coords, u)` — exact: the split product is the same field element
    /// as the materialized tensor entry (field ops are exact, so any
    /// correct parenthesization is bitwise identical).
    #[inline]
    fn at(&self, u: usize) -> F128 {
        self.lo[u & self.mask] * self.hi[u >> self.n_lo]
    }
}

/// The VIRTUAL partner side of a two-product pair — never materialized,
/// never folded; evaluated pointwise from [`SplitEq`] tables at whatever
/// positions the messages need. Both closed forms survive low-bit-first
/// folding EXACTLY (docs/multipoint-twisted-assist.tex):
///
/// - `Scaled`: e = eq(ρ,·) folds to `s·eq(ρ[i..], ·)` with
///   `s ← s·(1+ρ_i+r_i)` (char-2 `eq(z,r) = 1+z+r`) — the identity the
///   segment-sparse group pair already exploits for its boundary paddings.
/// - `Twisted`: g = L_γ(eq(ρ,·)) has no tensor factorization (L_γ is
///   F₂-linear, not multiplicative), but multiplication-by-constant IS
///   F₂-linear, so the fold stays "linear map of the remaining tensor":
///   `g_{i+1}[t] = L(c₀·E) + r·L((c₀+c₁)·E) = (M∘mult_{c₀} + mult_r∘M)(E)`
///   with `c₀+c₁ = 1`. The map is carried as its 128 basis images plus the
///   byte-sliced tables, updated per round for ~a few µs.
///
/// This is what makes the ā-side area trim sound where trimming a
/// MATERIALIZED partner was not: a stored partner's tail feeds its own
/// folds (the straddling live block mixes tail values, and the anchor pins
/// the endpoint to the closed form of the FULL vector), but a virtual
/// partner has no stored tail — every evaluation IS the closed form.
#[derive(Clone)]
enum Partner {
    Twisted {
        images: Box<[F128; 128]>,
        tables: Vec<[F128; 256]>,
    },
    Scaled {
        s: F128,
    },
}

impl Partner {
    fn twisted(images: Box<[F128; 128]>) -> Self {
        let tables = linear_byte_tables(&images);
        Partner::Twisted { images, tables }
    }

    /// The partner's value at position `u` of the current round's domain.
    #[inline]
    fn at(&self, eq: &SplitEq, u: usize) -> F128 {
        match self {
            Partner::Twisted { tables, .. } => apply_linear_tables(tables, eq.at(u)),
            Partner::Scaled { s } => *s * eq.at(u),
        }
    }

    /// Advance past one fold: bind the round's coordinate `rho_i` at `r`.
    fn advance(&mut self, rho_i: F128, r: F128) {
        match self {
            Partner::Twisted { images, tables } => {
                let c = F128::ONE + rho_i;
                let mut nxt = Box::new([F128::ZERO; 128]);
                for (b, slot) in nxt.iter_mut().enumerate() {
                    *slot = apply_linear_tables(tables, c * basis_elem(b)) + r * images[b];
                }
                *tables = linear_byte_tables(&nxt);
                *images = nxt;
            }
            Partner::Scaled { s } => *s *= F128::ONE + rho_i + r,
        }
    }
}

/// The combined weight vector of ONE product of the two-product sumcheck —
/// `Σ_s scale_s·eq(z_{s,r}, row(d))·w_s(col(d))`, zero past the area,
/// segmented parallel fill from the prefix sums — AND that product's
/// round-0 message against `partner`, in one traversal: each output chunk
/// is filled (all sides — the chunk stays cache-resident across the
/// per-side cursor walks) and immediately paired against the matching
/// `partner` chunk for the message partial sums. A side's column weights
/// are dense per-column: an RS claim passes its `eq(z_col, ·)` table, a
/// scalar group its γ-baked merged cols — the walk is identical.
fn build_combined_weight_and_msg(
    params: &JaggedParams,
    sides: &[(F128, Vec<F128>, &[F128])],
    partner: &Partner,
    eq: &SplitEq,
) -> (Vec<F128>, (F128, F128)) {
    use rayon::prelude::*;
    let mut a = vec![F128::ZERO; 1usize << params.m];
    let pfx = &params.col_prefix_sums;
    let n_cols = pfx.len() - 1;
    let area = pfx[n_cols];
    const CH: usize = 1 << 14;
    let msg = a
        .par_chunks_mut(CH)
        .enumerate()
        .map(|(ci, chunk)| {
            let start = (ci * CH) as u64;
            // Chunks past the jagged area: the weight is identically zero
            // there (the fill below never reaches them — calloc zeros), so
            // every round-0 message term is zero and the (virtual) partner
            // is never evaluated.
            if start >= area {
                return (F128::ZERO, F128::ZERO);
            }
            let end = start + chunk.len() as u64;
            for (scale, eq_r, cols) in sides {
                let mut y = pfx.partition_point(|&t| t <= start).saturating_sub(1);
                let mut d = start;
                while d < end && y < n_cols {
                    let (t_c, t_next) = (pfx[y], pfx[y + 1]);
                    if t_next <= d {
                        y += 1;
                        continue;
                    }
                    let w = *scale * cols[y];
                    let stop = end.min(t_next);
                    for dd in d..stop {
                        chunk[(dd - start) as usize] += w * eq_r[(dd - t_c) as usize];
                    }
                    d = stop;
                }
            }
            let mut p1 = F128::ZERO;
            let mut pi = F128::ZERO;
            for (j, ap) in chunk.as_chunks::<2>().0.iter().enumerate() {
                let u = start as usize + 2 * j;
                let q0 = partner.at(eq, u);
                let q1 = partner.at(eq, u + 1);
                p1 += ap[1] * q1;
                pi += (ap[0] + ap[1]) * (q0 + q1);
            }
            (p1, pi)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1));
    (a, msg)
}

/// Transcript of the multipoint twisted evaluation: `128` dual-form values
/// per ring-switched claim, ONE per scalar group, the `m` dense-domain
/// two-product sumcheck rounds, and the endpoint's untwisted anchor assist.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultipointTwistedProof {
    pub values: Vec<Vec<F128>>,
    pub group_values: Vec<F128>,
    /// PoW witness after all dual values are bound and before their random
    /// linear-combination coefficient is sampled.
    #[serde(default)]
    pub gamma_grinding_nonce: u64,
    pub rounds: Vec<(F128, F128)>,
    /// PoW witnesses for the two-product quadratic sumcheck, in round order.
    #[serde(default)]
    pub round_grinding_nonces: Vec<u64>,
    pub anchor: FrobeniusAssistProof,
}

/// Grinding policy for the multipoint-twisted transport.  The first field
/// protects the random linear combination of the dual-value claims; the other
/// two protect degree-two sumcheck rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultipointGrinding {
    /// Enables grinding for the gamma batching challenge.  When nonzero this
    /// is a lower bound: the actual schedule is derived from the number `K`
    /// of batched values, whose discrepancy polynomial has degree `K - 1`.
    pub gamma_bits: u32,
    pub round_bits: u32,
    pub anchor_round_bits: u32,
}

impl MultipointGrinding {
    pub const fn disabled() -> Self {
        Self {
            gamma_bits: 0,
            round_bits: 0,
            anchor_round_bits: 0,
        }
    }

    pub const fn per_challenge_128() -> Self {
        Self {
            gamma_bits: 1,
            round_bits: 2,
            anchor_round_bits: 2,
        }
    }

    /// The gamma powers batch `K = 128 * n_rs + n_groups` values, so a false
    /// vector is hidden by a nonzero polynomial of degree at most `K - 1`.
    /// Derive the strict per-site schedule from that degree rather than using
    /// the old fixed one-bit setting (which was insufficient once `K > 2`).
    #[inline]
    pub fn gamma_bits_for(self, n_rs: usize, n_groups: usize) -> u32 {
        if self.gamma_bits == 0 {
            return 0;
        }
        let degree = 128usize
            .checked_mul(n_rs)
            .and_then(|n| n.checked_add(n_groups))
            .and_then(|k| k.checked_sub(1))
            .expect("multipoint batch size must be nonzero and fit usize");
        self.gamma_bits
            .max(crate::challenger::grinding_bits_for_degree(degree))
    }
}

/// Prover for the multipoint twisted evaluation, two-product form
/// (docs/multipoint-twisted-assist.tex §"The two-product grouping").
///
/// `claims` are the ring-switched claims (their 128 `coeffs` define each
/// `Φ_i`; 128 dual values each). `groups` are the γ-baked scalar groups of
/// packed-direct claims — fold map the identity, so ONE untwisted value
/// each, and their sumcheck partner is plain `eq(ρ,·)` instead of the
/// twisted combination `g`. The batching weights stay consecutive powers:
/// `γ^{128 i + j}` for value `(i, j)`, `γ^{128 R + k}` for group `k`.
///
/// The verifier's [`verify_multipoint_twisted`] returns the verified
/// `V = Σ_{i,j} c_{i,j}·A_{i,j}^{2^j} + Σ_k B_k = Ŵ(ρ)`.
pub fn prove_multipoint_twisted<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[ScalarGroupClaim<'_>],
    rho: &[F128],
    challenger: &mut C,
) -> MultipointTwistedProof {
    prove_multipoint_twisted_with_grinding(
        params,
        claims,
        groups,
        rho,
        MultipointGrinding::disabled(),
        challenger,
    )
}

/// [`prove_multipoint_twisted`] with PoW witnesses at its random-linear
/// batching and quadratic sumcheck sites.
pub fn prove_multipoint_twisted_with_grinding<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[ScalarGroupClaim<'_>],
    rho: &[F128],
    grinding: MultipointGrinding,
    challenger: &mut C,
) -> MultipointTwistedProof {
    let m = params.m;
    assert_eq!(rho.len(), m);
    for claim in claims {
        assert_eq!(claim.coeffs.len(), 128);
    }
    for g in groups {
        assert_eq!(
            g.cols.len(),
            1usize << params.k,
            "group cols must be dense over 2^k columns"
        );
    }
    let n_rs = claims.len();
    let n_g = groups.len();
    assert!(n_rs + n_g > 0, "multipoint over zero claims");
    let trace = std::env::var("PCS_TRACE").is_ok();

    let t = std::time::Instant::now();
    // The inverse-Frobenius points exist only for the twisted (RS) side.
    let rho_pows = (n_rs > 0).then(|| rho_inverse_powers(rho));
    let values = match &rho_pows {
        Some(rp) => multipoint_values(params, claims, rp),
        None => Vec::new(),
    };
    let group_values = multipoint_group_values(params, groups, rho);
    if trace {
        eprintln!(
            "    [multipoint] {} + {} values (compute): {:6.2} ms",
            128 * n_rs,
            n_g,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    challenger.observe_label(b"flock-multipoint-twisted-v1");
    for vs in &values {
        for &v in vs {
            challenger.observe_f128(v);
        }
    }
    for &v in &group_values {
        challenger.observe_f128(v);
    }
    let gamma_bits = grinding.gamma_bits_for(n_rs, n_g);
    let (gamma_grinding_nonce, gamma) = if gamma_bits != 0 {
        challenger.grind_pow_and_sample_f128(gamma_bits)
    } else {
        (0, challenger.sample_f128())
    };
    let mut gpow = Vec::with_capacity(128 * n_rs + n_g);
    let mut p = F128::ONE;
    for _ in 0..128 * n_rs + n_g {
        gpow.push(p);
        p *= gamma;
    }

    // The partner sides — e = eq(ρ,·) for the groups, g = L_γ(e) for the RS
    // claims — are VIRTUAL: never materialized, never folded. Their closed
    // forms survive low-bit-first folding exactly (see [`Partner`]), so
    // messages evaluate them pointwise from √n split-eq tables. Only the
    // combined jagged weights (no tensor structure) get buffers, and those
    // fold on the area-trimmed live prefix.
    let t = std::time::Instant::now();
    let eq0 = SplitEq::new(rho);
    let area = params.area() as usize;
    let mut pairs: Vec<Pair> = Vec::with_capacity(2);
    let mut msg0 = (F128::ZERO, F128::ZERO);
    // The aligned full-count shape serves BOTH products' closed forms.
    let aligned = aligned_full_columns(params);
    if n_rs > 0 {
        let basis = inv_frob_basis();
        let mut images = Box::new([F128::ZERO; 128]);
        for j in 0..128 {
            for (b, img) in images.iter_mut().enumerate() {
                *img += gpow[j] * basis[j][b];
            }
        }
        let partner = Partner::twisted(images);
        if let Some(used) = aligned.clone() {
            // Aligned full-count shape: the RS combined weight is, per used
            // column, a sum of plain scaled row-eq tensors — no fold tables
            // (they live in the dual values and the g partner) — so it runs
            // the row rounds in CLOSED FORM: no 2^m fill, no fold traffic,
            // no ping-pong scratch. Same field elements as the materialized
            // path (exact algebra), pinned by the aligned oracle test.
            let coltabs: Vec<Vec<F128>> = claims
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let eq_c = build_eq_table(c.z_col);
                    used.iter().map(|&y| gpow[128 * i] * eq_c[y]).collect()
                })
                .collect();
            let rows: Vec<(Vec<F128>, F128)> = claims
                .iter()
                .map(|c| (c.z_row.to_vec(), F128::ONE))
                .collect();
            let closed = closed_rs_on().then(|| {
                ClosedRs::build(
                    &coltabs,
                    &rows,
                    &gpow,
                    rho_pows.as_ref().expect("RS claims carry the conjugates"),
                    params.n,
                    used.len(),
                )
            });
            let ar = AlignedRsPair {
                coltabs,
                rows,
                partner,
                rho: rho.to_vec(),
                round: 0,
                nu: params.n,
                n_used: used.len(),
                closed,
            };
            let msg = ar.round0_msg(&eq0);
            msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
            pairs.push(Pair::AlignedRs(ar));
        } else if virtual_a_on() && claims.len() <= LAZY_RS_MAX_SIDES {
            // Jagged shape, ā VIRTUAL: no 2^m weight buffer — round 0
            // streams the column walk, the first fold materializes at half
            // size ([`LazyRsPair`]; byte-identical by char-2 reassociation).
            let lz = LazyRsPair::new(params, claims, &gpow, partner, rho);
            let msg = lz.round0_msg(&eq0);
            msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
            pairs.push(Pair::LazyRs(lz));
        } else {
            let eq_cs: Vec<Vec<F128>> = claims.iter().map(|c| build_eq_table(c.z_col)).collect();
            let sides: Vec<(F128, Vec<F128>, &[F128])> = claims
                .iter()
                .zip(&eq_cs)
                .enumerate()
                .map(|(i, (c, eq_c))| (gpow[128 * i], build_eq_table(c.z_row), eq_c.as_slice()))
                .collect();
            let (av, msg) = build_combined_weight_and_msg(params, &sides, &partner, &eq0);
            msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
            pairs.push(Pair::Virtual(VirtualPair::new(av, partner, rho, 0, area)));
        }
    }
    let t_rs_weight = t.elapsed();
    if trace && n_rs > 0 {
        // Which columns could take a closed form, and which cannot: the
        // height histogram is what decides whether the aligned kernel
        // generalizes to this shape at all.
        let pfx = &params.col_prefix_sums;
        let full = 1u64 << params.n;
        let mut hist: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
        for y in 0..pfx.len() - 1 {
            let h = pfx[y + 1] - pfx[y];
            if h > 0 {
                *hist.entry(h).or_default() += 1;
            }
        }
        let n_full: usize = hist.get(&full).copied().unwrap_or(0);
        let n_pow2: usize = hist
            .iter()
            .filter(|(h, _)| h.is_power_of_two())
            .map(|(_, c)| *c)
            .sum();
        let used: usize = hist.values().sum();
        let mut shape: Vec<String> = hist
            .iter()
            .rev()
            .take(6)
            .map(|(h, c)| format!("{h}x{c}"))
            .collect();
        if hist.len() > 6 {
            shape.push(format!("+{} more", hist.len() - 6));
        }
        eprintln!(
            "    [multipoint] rs path: {} (n={} k={} area={} of 2^{m}); columns {used} used, \
             {n_full} full(2^{}), {n_pow2} pow2; heights {}",
            match pairs.first() {
                Some(Pair::AlignedRs(_)) => "ALIGNED closed-form weight",
                Some(Pair::LazyRs(_)) => "dense VIRTUAL weight (half materializes at fold 0)",
                _ => "dense materialized weight",
            },
            params.n,
            params.k,
            params.area(),
            params.n,
            shape.join(" "),
        );
    }
    let mut sparse_support: Option<u64> = None;
    let mut dense_group_support: Option<u64> = None;
    if n_g > 0
        && closed_rs_on()
        && let Some(used) = aligned.as_ref()
    {
        // THE SAME CLOSED FORM, one conjugate. A group's weight is
        // `scale·cols[y]·eq(z_row, row)` — the RS side's shape exactly — and
        // its partner is a SINGLE scaled `eq(ρ,·)` rather than 128 twisted
        // tensors, so [`ClosedRs`] serves it with `pts = [ρ]`, `coef = [1]`
        // and the scale riding the column table. No support scan, no
        // densify, no 2^m weight: the group product's rounds cost
        // `O(#groups)` multiplies.
        let coltabs: Vec<Vec<F128>> = groups
            .iter()
            .enumerate()
            .map(|(k, g)| {
                let scale = gpow[128 * n_rs + k];
                used.iter().map(|&y| scale * g.cols[y]).collect()
            })
            .collect();
        let rows: Vec<(Vec<F128>, F128)> = groups
            .iter()
            .map(|g| (g.z_row.to_vec(), F128::ONE))
            .collect();
        let pts = [rho.to_vec()];
        let closed = ClosedRs::build(&coltabs, &rows, &[F128::ONE], &pts, params.n, used.len());
        let ar = AlignedRsPair {
            coltabs,
            rows,
            partner: Partner::Scaled { s: F128::ONE },
            rho: rho.to_vec(),
            round: 0,
            nu: params.n,
            n_used: used.len(),
            closed: Some(closed),
        };
        let msg = ar.round0_msg(&eq0);
        msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
        pairs.push(Pair::AlignedRs(ar));
    } else if n_g > 0 {
        let sides: Vec<(F128, Vec<F128>, &[F128])> = groups
            .iter()
            .enumerate()
            .map(|(k, g)| (gpow[128 * n_rs + k], build_eq_table(g.z_row), g.cols))
            .collect();
        // The gather-shaped groups are supported on a few columns: fold them
        // segment-sparse instead of materializing b̄ over 2^m.
        let support = SparseGroupPair::support_area(params, groups);
        if (support as usize).saturating_mul(SPARSE_DENSIFY_FACTOR) <= (1usize << m) {
            let (sp, msg) = SparseGroupPair::build(params, &sides, &eq0, rho);
            msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
            pairs.push(Pair::Sparse(sp));
            sparse_support = Some(support);
        } else {
            let partner = Partner::Scaled { s: F128::ONE };
            let (bv, msg) = build_combined_weight_and_msg(params, &sides, &partner, &eq0);
            msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
            pairs.push(Pair::Virtual(VirtualPair::new(bv, partner, rho, 0, area)));
            dense_group_support = Some(support);
        }
    }
    if trace {
        eprintln!(
            "    [multipoint] weight passes (2^{m}, {} products{}, round-0 fused): {:6.2} ms  \
             (rs {:.2} | group {:.2})",
            pairs.len(),
            match (sparse_support, dense_group_support) {
                (Some(s), _) => format!(" — group sparse, support {s} words"),
                (None, Some(s)) => format!(" — group DENSE, support {s} words"),
                (None, None) => String::new(),
            },
            t.elapsed().as_secs_f64() * 1e3,
            t_rs_weight.as_secs_f64() * 1e3,
            (t.elapsed() - t_rs_weight).as_secs_f64() * 1e3,
        );
    }

    // The m-round two-product sumcheck for Σ_d (ā_d·g_d + b̄_d·e_d), low bit
    // first: round-0 messages from the full vectors, later messages fused
    // into the folds ([`fold_and_round_oop_par`], ping-pong scratch halves)
    // and summed across the active products. The final fold never runs —
    // nothing reads the folded scalars; the anchor assist reproves the
    // endpoint.
    let t = std::time::Instant::now();
    let mut rounds = Vec::with_capacity(m);
    let mut round_grinding_nonces = Vec::with_capacity((grinding.round_bits != 0) as usize * m);
    let mut point = Vec::with_capacity(m);
    // Per-PAIR fold attribution: which product's rounds cost what. The two
    // products are not comparable — the RS side folds the whole area against
    // a TWISTED partner (a linearized-map application per element), the group
    // side a sparse support against a scaled `eq` — so the aggregate hides
    // which one any optimisation would actually reach.
    let mut per_pair = vec![std::time::Duration::ZERO; pairs.len()];
    let (mut g_one, mut g_inf) = msg0;
    let mut cur = 1usize << m;
    for i in 0..m {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = if grinding.round_bits != 0 {
            let (nonce, r) = challenger.grind_pow_and_sample_f128(grinding.round_bits);
            round_grinding_nonces.push(nonce);
            r
        } else {
            challenger.sample_f128()
        };
        rounds.push((g_one, g_inf));
        point.push(r);
        if i + 1 == m {
            break;
        }
        // NOT parallelized across pairs: measured on the node arm, a
        // par_iter_mut here moved the round wall from 5.10 to 4.93 ms —
        // rayon's work stealing already interleaves the two pairs' inner
        // parallelism, so the join bought nothing but murkier per-pair
        // attribution.
        let mut nxt = (F128::ZERO, F128::ZERO);
        for (pi, pair) in pairs.iter_mut().enumerate() {
            let t_pair = trace.then(std::time::Instant::now);
            let msg = pair.fold_round(cur, r);
            if let Some(t0) = t_pair {
                per_pair[pi] += t0.elapsed();
            }
            nxt = (nxt.0 + msg.0, nxt.1 + msg.1);
        }
        (g_one, g_inf) = nxt;
        cur /= 2;
    }
    // The virtual pairs' buffers go back to the scratch pool (the half-size
    // ping-pong halves came from it; the full-size weight vectors seed it
    // for the next prove).
    for pair in pairs {
        if let Pair::Virtual(p) = pair {
            p.reclaim();
        }
    }
    if trace {
        let split: Vec<String> = per_pair
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let side = if i == 0 && n_rs > 0 { "rs" } else { "group" };
                format!("{side} {:.2}", d.as_secs_f64() * 1e3)
            })
            .collect();
        eprintln!(
            "    [multipoint] two-product sumcheck ({m} rounds): {:6.2} ms  (folds: {})",
            t.elapsed().as_secs_f64() * 1e3,
            split.join(" | "),
        );
    }

    // Anchor: ONE untwisted batched assist binding the whole endpoint sum
    // `ĝ(ρ'')·ā̂(ρ'') + eq(ρ,ρ'')·b̂(ρ'')` — the closed-form factors are
    // baked into the coefficients (RS claim i: γ^{128 i}·ĝ(ρ''); group k:
    // γ^{128 R + k}·eq(ρ,ρ'')), so the accept check is a plain equality
    // against the running claim and no extra scalar travels.
    let t = std::time::Instant::now();
    let g_at = match &rho_pows {
        Some(rp) => twisted_eq_at(&gpow, rp, &point),
        None => F128::ZERO,
    };
    let e_at = if n_g > 0 {
        eq_at(rho, &point)
    } else {
        F128::ZERO
    };
    let anchor_coeffs: Vec<Vec<F128>> = (0..n_rs)
        .map(|i| {
            let mut c = vec![F128::ZERO; 128];
            c[0] = gpow[128 * i] * g_at;
            c
        })
        .collect();
    let anchor_claims: Vec<FrobeniusClaim<'_>> = claims
        .iter()
        .zip(&anchor_coeffs)
        .map(|(cl, co)| FrobeniusClaim {
            z_row: cl.z_row,
            z_col: cl.z_col,
            coeffs: co,
        })
        .collect();
    let anchor_groups: Vec<(ScalarGroupClaim<'_>, F128)> = groups
        .iter()
        .enumerate()
        .map(|(k, g)| (*g, gpow[128 * n_rs + k] * e_at))
        .collect();
    let anchor = prove_frobenius_assist_with_grinding(
        params,
        &anchor_claims,
        &anchor_groups,
        &point,
        grinding.anchor_round_bits,
        challenger,
    );
    if trace {
        eprintln!(
            "    [multipoint] anchor assist (x{} + x{}): {:6.2} ms",
            n_rs,
            n_g,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    MultipointTwistedProof {
        values,
        group_values,
        gamma_grinding_nonce,
        rounds,
        round_grinding_nonces,
        anchor,
    }
}

/// One product's ping-pong fold state for the two-product sumcheck: the
/// full-size pair, half-size scratch, and which half currently holds the
/// live data.
struct VirtualPair {
    x: Vec<F128>,
    sx: Vec<F128>,
    flip: bool,
    partner: Partner,
    rho: Vec<F128>,
    round: usize,
    /// `x`'s zero-tail bound: folds and messages run on the live prefix
    /// only. EXACT here (unlike for a materialized partner): every skipped
    /// message term carries a folded-`x` factor of zero, folding maps `x`'s
    /// zero tail to a zero tail, and the partner — the only thing with
    /// nonzero tail values — is evaluated in closed form, so it has no
    /// stored tail to go stale. Kept a multiple of 4 (the kernel's exact
    /// chunk width) with zeroed guard slots (the scratch half is
    /// pool-dirty).
    live: usize,
}

impl VirtualPair {
    /// `live`: prefix outside of which the caller guarantees `x` is zero
    /// (jagged weights vanish past the area; every call site hands a vec
    /// whose tail is calloc-zero, which also covers the initial guards).
    fn new(x: Vec<F128>, partner: Partner, rho: &[F128], round: usize, live: usize) -> Self {
        debug_assert!(live <= x.len());
        debug_assert_eq!(x.len(), 1usize << (rho.len() - round));
        let n = x.len();
        // Dirty scratch half: each round fully writes its live prefix and
        // zeroes the guard slots before the next round reads them.
        Self {
            sx: crate::scratch::take_f128(n / 2),
            live: live.next_multiple_of(4).clamp(4, n),
            x,
            flip: false,
            partner,
            rho: rho.to_vec(),
            round,
        }
    }

    /// Fold `x` at `r` into the other half, advance the partner's closed
    /// form, and return the next round's message.
    fn fold_round(&mut self, cur: usize, r: F128) -> (F128, F128) {
        debug_assert_eq!(cur, 1usize << (self.rho.len() - self.round));
        let half = cur / 2;
        self.partner.advance(self.rho[self.round], r);
        self.round += 1;
        let eq = SplitEq::new(&self.rho[self.round..]);
        let live = self.live.min(cur);
        let lhalf = live / 2;
        let msg = if self.flip {
            fold_and_round_virtual_par(
                &self.sx[..live],
                r,
                &mut self.x[..lhalf],
                &self.partner,
                &eq,
            )
        } else {
            fold_and_round_virtual_par(
                &self.x[..live],
                r,
                &mut self.sx[..lhalf],
                &self.partner,
                &eq,
            )
        };
        // Guard slots for the NEXT fold's 4-wide reads over dirty scratch.
        let next = lhalf.next_multiple_of(4).min(half);
        {
            let gx = if self.flip { &mut self.x } else { &mut self.sx };
            for slot in &mut gx[lhalf..next] {
                *slot = F128::ZERO;
            }
        }
        self.live = next;
        self.flip = !self.flip;
        msg
    }

    /// Return both buffers to the scratch pool.
    fn reclaim(self) {
        crate::scratch::give_f128(self.x);
        crate::scratch::give_f128(self.sx);
    }
}

/// One column's message contribution for the aligned RS closed form:
/// pairs over the column's `b` positions of
/// `ā = Σ_i a_i·row_eq_i(row)` against the partner. The row-eq hi factor
/// is constant across each `2^n_lo` run, so `a_i·hi` hoists per run and
/// each position pays one multiply per claim — exact (field algebra), the
/// same field elements as the unhoisted product.
#[inline]
fn column_pair_msg(
    a: &[F128],
    row_eqs: &[SplitEq],
    partner: &Partner,
    eq: &SplitEq,
    base: usize,
    b: usize,
) -> (F128, F128) {
    let mut g1 = F128::ZERO;
    let mut gi = F128::ZERO;
    let n_lo = row_eqs[0].n_lo;
    debug_assert!(row_eqs.iter().all(|e| e.n_lo == n_lo));
    if n_lo >= 1 {
        let mask = (1usize << n_lo) - 1;
        let mut ah: Vec<F128> = vec![F128::ZERO; a.len()];
        let mut cur_hi = usize::MAX;
        for t2 in (0..b).step_by(2) {
            let h = t2 >> n_lo;
            if h != cur_hi {
                cur_hi = h;
                for (dst, (&ai, e)) in ah.iter_mut().zip(a.iter().zip(row_eqs)) {
                    *dst = ai * e.hi[h];
                }
            }
            let mut a0 = F128::ZERO;
            let mut a1 = F128::ZERO;
            for (&ahi, e) in ah.iter().zip(row_eqs) {
                a0 += ahi * e.lo[t2 & mask];
                a1 += ahi * e.lo[(t2 + 1) & mask];
            }
            let p0 = partner.at(eq, base + t2);
            let p1 = partner.at(eq, base + t2 + 1);
            g1 += a1 * p1;
            gi += (a0 + a1) * (p0 + p1);
        }
    } else {
        // One (or zero) remaining row bits: runs are single positions, no
        // hoist — fall back to direct evaluation.
        for t2 in (0..b).step_by(2) {
            let mut a0 = F128::ZERO;
            let mut a1 = F128::ZERO;
            for (&ai, e) in a.iter().zip(row_eqs) {
                a0 += ai * e.at(t2);
                a1 += ai * e.at(t2 + 1);
            }
            let p0 = partner.at(eq, base + t2);
            let p1 = partner.at(eq, base + t2 + 1);
            g1 += a1 * p1;
            gi += (a0 + a1) * (p0 + p1);
        }
    }
    (g1, gi)
}

/// The used-column list when EVERY used column is FULL (height `2^n`) — the
/// full-count shape whose dense stack is column-aligned at `2^n`
/// boundaries: position `d` splits as `(rank j, row) = (d >> n, d & mask)`
/// with `j` indexing the used columns in order.
fn aligned_full_columns(params: &JaggedParams) -> Option<Vec<usize>> {
    let full = 1u64 << params.n;
    let pfx = &params.col_prefix_sums;
    let mut used = Vec::new();
    for y in 0..pfx.len() - 1 {
        match pfx[y + 1] - pfx[y] {
            0 => {}
            h if h == full => used.push(y),
            _ => return None,
        }
    }
    (!used.is_empty() && params.n >= 1).then_some(used)
}

/// The RS product of the two-product sumcheck in CLOSED FORM, for the
/// aligned full-count shape: the combined weight is, per used column `j`,
/// a sum of R plain scaled row-eq tensors —
/// `ā(j, row) = Σ_i coltab_i[j]·s_i·eq(z_row_i[round..], row)` — with NO
/// fold tables (the twisted assist moved the Frobenius structure into the
/// dual values and the g partner), so pointwise evaluation is a couple of
/// multiplies. Column alignment makes the form survive the row-bit folds:
/// each round updates the per-claim scalar `s_i ← s_i·(1+z_row_i[k]+r_k)`
/// exactly like the partners. Nothing is materialized until the row bits
/// are exhausted, at which point the state densifies into a [`VirtualPair`]
/// over the tiny per-column domain. Unaligned shapes (partial counts) keep
/// the materialized path — their column boundaries straddle fold blocks.
/// The twisted partner's CONJUGATE EXPANSION, precomputed — what turns the
/// aligned RS product's round message from a sweep into a handful of
/// multiplies.
///
/// `g = Σ_j γ^j·eq(ρ^{2^{-j}}, ·)` is 128 eq tensors BY DEFINITION (the same
/// form [`twisted_eq_at`] evaluates for the anchor), and on the aligned
/// layout a position splits as `u = rank·2^ν + row`, so every eq factor
/// splits with it. With the weight already in closed form
/// (`ā(rank,row) = Σ_i coltab_i[rank]·s_i·eq(z_row_i, row)`) each message
/// term separates into a COLUMN dot and a ROW product:
///
/// ```text
///   Σ_u ā(u)·eq(ρ⁽ʲ⁾,u) = Σ_i s_i·[Σ_rank coltab_i·eq(ρ⁽ʲ⁾_hi, rank)]
///                                 ·[Π_b (1 + z_i,b + ρ⁽ʲ⁾_b)]
/// ```
///
/// The column dot is CONSTANT across every row round — the folds only ever
/// bind row bits, so `ρ⁽ʲ⁾_hi` never moves — and the row product is a
/// suffix product, so both precompute once. A round then costs `O(128·R)`
/// multiplies instead of one linearized-map application per position
/// (~58M of them at the m32 leaf).
///
/// Char 2 does the rest: `g(∞)`'s bit-0 factors are `(1+x) + x = 1` on both
/// sides, so the infinity leg drops them entirely, and `g(1)` keeps just
/// `z_i,round·ρ⁽ʲ⁾_round`.
struct ClosedRs {
    /// `γ^j` times the accumulated fold factors `Π_{b<round}(1 + ρ⁽ʲ⁾_b + r_b)`
    /// — the same `(1 + point + r)` the weight scalars and
    /// [`Partner::advance`] use.
    coef: Vec<F128>,
    /// The conjugate points `ρ^{2^{-j}}`, full length `m`.
    pts: Vec<Vec<F128>>,
    /// `coldot[i][j] = Σ_rank coltab_i[rank]·eq(ρ⁽ʲ⁾[ν..], rank)`.
    coldot: Vec<Vec<F128>>,
    /// `suf[i][j][b] = Π_{b ≤ b' < ν} (1 + z_row_i[b'] + ρ⁽ʲ⁾[b'])`.
    suf: Vec<Vec<Vec<F128>>>,
}

impl ClosedRs {
    fn build(
        coltabs: &[Vec<F128>],
        rows: &[(Vec<F128>, F128)],
        gpow: &[F128],
        rho_pows: &[Vec<F128>],
        nu: usize,
        n_used: usize,
    ) -> Self {
        let n_j = rho_pows.len();
        let coldot: Vec<Vec<F128>> = coltabs
            .iter()
            .map(|ct| {
                (0..n_j)
                    .map(|j| {
                        let hi = &rho_pows[j][nu..];
                        (0..n_used)
                            .map(|rank| {
                                let mut e = ct[rank];
                                for (b, &p) in hi.iter().enumerate() {
                                    e *= if (rank >> b) & 1 == 1 {
                                        p
                                    } else {
                                        F128::ONE + p
                                    };
                                }
                                e
                            })
                            .fold(F128::ZERO, |a, x| a + x)
                    })
                    .collect()
            })
            .collect();
        let suf: Vec<Vec<Vec<F128>>> = rows
            .iter()
            .map(|(z, _)| {
                (0..n_j)
                    .map(|j| {
                        let mut s = vec![F128::ONE; nu + 1];
                        for b in (0..nu).rev() {
                            s[b] = s[b + 1] * (F128::ONE + z[b] + rho_pows[j][b]);
                        }
                        s
                    })
                    .collect()
            })
            .collect();
        Self {
            coef: gpow[..n_j].to_vec(),
            pts: rho_pows.to_vec(),
            coldot,
            suf,
        }
    }

    /// Bind the round's row bit at `r`: every conjugate tensor drops that
    /// coordinate and scales, exactly as the weight's own scalars do.
    fn advance(&mut self, round: usize, r: F128) {
        for (c, p) in self.coef.iter_mut().zip(&self.pts) {
            *c *= F128::ONE + p[round] + r;
        }
    }

    /// `(g(1), g(∞))` for the state at `round`, given the weight's running
    /// per-claim scalars.
    fn msg(&self, round: usize, rows: &[(Vec<F128>, F128)]) -> (F128, F128) {
        let (mut g1, mut gi) = (F128::ZERO, F128::ZERO);
        for (j, &c) in self.coef.iter().enumerate() {
            let p_r = self.pts[j][round];
            let (mut s1, mut si) = (F128::ZERO, F128::ZERO);
            for (i, (z, s)) in rows.iter().enumerate() {
                let t = self.coldot[i][j] * *s * self.suf[i][j][round + 1];
                si += t;
                s1 += t * z[round] * p_r;
            }
            g1 += c * s1;
            gi += c * si;
        }
        (g1, gi)
    }
}

/// `FLOCK_NO_CLOSED_RS=1` restores the pointwise message (one partner
/// evaluation per position) — the certification knob for the byte-identity
/// A/B, since the two paths must produce the same field elements.
fn closed_rs_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_CLOSED_RS").is_none())
}

struct AlignedRsPair {
    /// Per claim: γ-baked column factors in used-rank order
    /// (`gpow[128 i]·eq_col_i[y_j]`).
    coltabs: Vec<Vec<F128>>,
    /// Per claim: the row point and its running fold scalar.
    rows: Vec<(Vec<F128>, F128)>,
    partner: Partner,
    rho: Vec<F128>,
    round: usize,
    nu: usize,
    n_used: usize,
    /// The closed-form message state. `None` under `FLOCK_NO_CLOSED_RS`.
    closed: Option<ClosedRs>,
}

impl AlignedRsPair {
    /// The round-0 message — the weight-pass replacement: no fill, no
    /// 2^m buffer, pure closed-form evaluation against the partner.
    fn round0_msg(&self, eq0: &SplitEq) -> (F128, F128) {
        use rayon::prelude::*;
        debug_assert_eq!(self.round, 0);
        if let Some(c) = &self.closed {
            return c.msg(0, &self.rows);
        }
        let row_eqs: Vec<SplitEq> = self
            .rows
            .iter()
            .map(|(z, _)| SplitEq::new(&z[..self.nu]))
            .collect();
        let b = 1usize << self.nu;
        (0..self.n_used)
            .into_par_iter()
            .map(|j| {
                let a: Vec<F128> = self.coltabs.iter().map(|ct| ct[j]).collect();
                let base = j * b;
                column_pair_msg(&a, &row_eqs, &self.partner, eq0, base, b)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x1, xi), (y1, yi)| (x1 + y1, xi + yi),
            )
    }

    /// Fold (in closed form: advance the scalars) and return the next
    /// round's message. Only called while `round + 1 < nu` — the dispatch
    /// densifies before the last row fold.
    fn fold_round(&mut self, cur: usize, r: F128) -> (F128, F128) {
        use rayon::prelude::*;
        debug_assert_eq!(cur, 1usize << (self.rho.len() - self.round));
        self.partner.advance(self.rho[self.round], r);
        for (z, s) in self.rows.iter_mut() {
            *s *= F128::ONE + z[self.round] + r;
        }
        if let Some(c) = &mut self.closed {
            c.advance(self.round, r);
        }
        self.round += 1;
        if let Some(c) = &self.closed {
            return c.msg(self.round, &self.rows);
        }
        let s_rem = self.nu - self.round;
        debug_assert!(
            s_rem >= 1,
            "the dispatch densifies before the columns collapse"
        );
        let eq = SplitEq::new(&self.rho[self.round..]);
        let row_eqs: Vec<SplitEq> = self
            .rows
            .iter()
            .map(|(z, _)| SplitEq::new(&z[self.round..self.nu]))
            .collect();
        let b = 1usize << s_rem;
        (0..self.n_used)
            .into_par_iter()
            .map(|j| {
                let a: Vec<F128> = self
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(i, (_, s))| self.coltabs[i][j] * *s)
                    .collect();
                let base = j * b;
                column_pair_msg(&a, &row_eqs, &self.partner, &eq, base, b)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x1, xi), (y1, yi)| (x1 + y1, xi + yi),
            )
    }

    /// Materialize the (by now tiny) per-column state into a
    /// [`VirtualPair`] at the current round — `2^(m−round)` entries, the
    /// last one or two row bits still unexpanded.
    fn densify(&self) -> VirtualPair {
        let s_rem = self.nu - self.round;
        let n = 1usize << (self.rho.len() - self.round);
        let b = 1usize << s_rem;
        let row_eqs: Vec<Vec<F128>> = self
            .rows
            .iter()
            .map(|(z, _)| build_eq_table(&z[self.round..self.nu]))
            .collect();
        let mut x = vec![F128::ZERO; n];
        for j in 0..self.n_used {
            for rb in 0..b {
                let mut v = F128::ZERO;
                for (i, (_, s)) in self.rows.iter().enumerate() {
                    v += self.coltabs[i][j] * *s * row_eqs[i][rb];
                }
                x[j * b + rb] = v;
            }
        }
        VirtualPair::new(
            x,
            self.partner.clone(),
            &self.rho,
            self.round,
            self.n_used * b,
        )
    }
}

/// Fused fold + next-round message for a [`VirtualPair`]: out-of-place fold
/// of the materialized side at `r`, with the message accumulated against
/// the partner's closed form evaluated at the OUTPUT positions (which live
/// in the post-fold domain — the caller advances the partner and builds the
/// post-fold [`SplitEq`] before calling). `a` is the live prefix (multiple
/// of 4); positions are absolute since the prefix starts at 0.
fn fold_and_round_virtual_par(
    a: &[F128],
    r: F128,
    ao: &mut [F128],
    partner: &Partner,
    eq: &SplitEq,
) -> (F128, F128) {
    use rayon::prelude::*;
    debug_assert_eq!(a.len(), 2 * ao.len());
    debug_assert!(a.len() >= 4 && a.len().is_multiple_of(4));
    const CO: usize = 1 << 13;
    ao.par_chunks_mut(CO)
        .zip(a.par_chunks(2 * CO))
        .enumerate()
        .map(|(ci, (oa, ain))| {
            let base = ci * CO;
            let mut g1 = F128::ZERO;
            let mut gi = F128::ZERO;
            for (j, (op, aq)) in oa
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(ain.as_chunks::<4>().0.iter())
                .enumerate()
            {
                let na0 = aq[0] + r * (aq[1] + aq[0]);
                let na1 = aq[2] + r * (aq[3] + aq[2]);
                op[0] = na0;
                op[1] = na1;
                let u = base + 2 * j;
                let p0 = partner.at(eq, u);
                let p1 = partner.at(eq, u + 1);
                g1 += na1 * p1;
                gi += (na0 + na1) * (p0 + p1);
            }
            (g1, gi)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t))
}

/// One live run of the sparse group product: `x` holds the combined group
/// weights `b̄` over dense indices `[start, start + x.len())`, `y` the
/// partner `e = eq(ρ,·)` values at the same indices. `start` and the length
/// stay EVEN, so the sumcheck's `(2t, 2t+1)` pairs never straddle a segment
/// boundary; the padding entries this costs carry `x = 0` (outside b̄'s
/// support) and TRUE `e` values.
struct SparseSeg {
    start: usize,
    x: Vec<F128>,
    y: Vec<F128>,
}

/// One segment's message contribution — the dense pair convention over
/// the segment's `(2t, 2t+1)` pairs (segments keep even start and even
/// length, so pairs never straddle). The message is LINEAR in `x` and
/// overlap `y` values agree, so per-segment sums computed BEFORE the
/// boundary merge XOR to exactly the merged state's message.
#[inline]
fn seg_msg(x: &[F128], y: &[F128]) -> (F128, F128) {
    let mut p1 = F128::ZERO;
    let mut pi = F128::ZERO;
    for (xp, yp) in x.as_chunks::<2>().0.iter().zip(y.as_chunks::<2>().0.iter()) {
        p1 += xp[1] * yp[1];
        pi += (xp[0] + xp[1]) * (yp[0] + yp[1]);
    }
    (p1, pi)
}

/// Segment-sparse fold state for the group product of the two-product
/// sumcheck. The gather-shaped groups are supported on a few columns'
/// live ranges (~9% of the dense area for the wired hash tables), yet the
/// dense [`VirtualPair`] materializes and folds `b̄` and a copy of `e`
/// over the full `2^m`: this state stores only the support runs and folds
/// them locally, so the group product costs O(support) per round instead
/// of O(2^m). Exactness: every message equals the dense path's — `b̄` is
/// zero off-support (off-support pairs contribute nothing to either
/// message sum), and `e` stays a scaled eq tensor under low-bit folding
/// (`e⁽ⁱ⁾ = Πⱼ eq(ρⱼ, rⱼ) · eq(ρ[i..], ·)`, exact in F128), so boundary
/// padding entries are recomputable pointwise at any round. Once the
/// support stops paying (`4·stored > cur`, reached in the tail rounds)
/// the state densifies into a [`VirtualPair`] and proceeds as before.
struct SparseGroupPair {
    /// Sorted by `start`; disjoint (overlaps from boundary padding are
    /// merged after each fold — `x` adds, `y` values agree).
    segs: Vec<SparseSeg>,
    /// The full ρ (all `m` coordinates), for pointwise `e` recomputation.
    rho: Vec<F128>,
    /// `Π_{j<round} eq(ρ_j, r_j)` — the folded-away prefix's scalar.
    c: F128,
    /// Rounds folded so far.
    round: usize,
}

impl SparseGroupPair {
    /// Dense-domain words under the groups' nonzero columns — the sparse
    /// path's cost driver, to compare against the `2^m` the dense path
    /// walks.
    fn support_area(params: &JaggedParams, groups: &[ScalarGroupClaim<'_>]) -> u64 {
        let pfx = &params.col_prefix_sums;
        (0..pfx.len() - 1)
            .filter(|&y| groups.iter().any(|g| g.cols[y] != F128::ZERO))
            .map(|y| pfx[y + 1] - pfx[y])
            .sum()
    }

    /// Build the support segments and the round-0 message. `sides` are the
    /// per-group `(γ-power, eq(z_row,·) table, cols)` triples — the same
    /// shape [`build_combined_weight_and_msg`] takes; `eq0` evaluates the
    /// round-0 `eq(ρ,·)` tensor pointwise (the full vector is never
    /// materialized — support-proportional fills only).
    fn build(
        params: &JaggedParams,
        sides: &[(F128, Vec<F128>, &[F128])],
        eq0: &SplitEq,
        rho: &[F128],
    ) -> (Self, (F128, F128)) {
        use rayon::prelude::*;
        let pfx = &params.col_prefix_sums;
        let n_cols = pfx.len() - 1;
        // Even-extended [start, end) ranges of the support columns; touching
        // ranges merge (adjacent live columns share a boundary word).
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for y in 0..n_cols {
            if sides.iter().all(|(_, _, cols)| cols[y] == F128::ZERO) {
                continue;
            }
            let (t0, t1) = (pfx[y] as usize, pfx[y + 1] as usize);
            if t0 == t1 {
                continue;
            }
            let (s, e) = (t0 & !1, (t1 + 1) & !1);
            match ranges.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => ranges.push((s, e)),
            }
        }
        // Segments are the PARALLEL GRAIN — build fill, message, and folds
        // are serial within one — and gather support clusters into a few
        // contiguous ranges (measured on the node arm: ~737k support words
        // whose group folds ran SLOWER at 10 threads than 1). Split each
        // range at aligned power-of-two cuts: aligned boundaries stay even
        // under the fold's halving, so chunks never overlap-merge back into
        // one serial segment before densify takes over. Same windows, same
        // per-word fills, char-2 sums — bit-identical messages.
        const SEG_CHUNK: usize = 1 << 14;
        let ranges: Vec<(usize, usize)> = ranges
            .into_iter()
            .flat_map(|(s, e)| {
                let mut cuts = Vec::with_capacity((e - s).div_ceil(SEG_CHUNK));
                let mut a = s;
                while a < e {
                    let b = ((a / SEG_CHUNK + 1) * SEG_CHUNK).min(e);
                    cuts.push((a, b));
                    a = b;
                }
                cuts
            })
            .collect();
        let segs = ranges
            .par_iter()
            .map(|&(s, e)| {
                let mut x = vec![F128::ZERO; e - s];
                // The dense fill's cursor walk, restricted to [s, e): only
                // live words get weights (padding words keep x = 0).
                let live_end = e.min(params.area() as usize);
                for (scale, eq_r, cols) in sides {
                    let mut ycol = pfx.partition_point(|&t| t <= s as u64).saturating_sub(1);
                    let mut d = s;
                    while d < live_end && ycol < n_cols {
                        let (t_c, t_next) = (pfx[ycol] as usize, pfx[ycol + 1] as usize);
                        if t_next <= d {
                            ycol += 1;
                            continue;
                        }
                        let w = *scale * cols[ycol];
                        let stop = live_end.min(t_next);
                        if w != F128::ZERO {
                            for dd in d..stop {
                                x[dd - s] += w * eq_r[dd - t_c];
                            }
                        }
                        d = stop;
                    }
                }
                let y: Vec<F128> = (s..e).map(|d| eq0.at(d)).collect();
                // Round-0 message, fused per segment: segments are
                // disjoint and the message is linear in `x`, so the
                // per-segment sums XOR to the whole — one pass, not two.
                let (p1, pi) = seg_msg(&x, &y);
                (SparseSeg { start: s, x, y }, p1, pi)
            })
            .collect::<Vec<_>>();
        let mut msg = (F128::ZERO, F128::ZERO);
        let segs = segs
            .into_iter()
            .map(|(seg, p1, pi)| {
                msg = (msg.0 + p1, msg.1 + pi);
                seg
            })
            .collect();
        let pair = Self {
            segs,
            rho: rho.to_vec(),
            c: F128::ONE,
            round: 0,
        };
        (pair, msg)
    }

    /// Words currently stored (the densify predicate's input).
    fn stored(&self) -> usize {
        self.segs.iter().map(|s| s.x.len()).sum()
    }

    /// `e⁽ʳᵒᵘⁿᵈ⁾[d]` recomputed pointwise: `c · Π_b eq(ρ_{round+b}, bit_b(d))`.
    /// Only boundary-padding entries need this (O(1) per segment per round).
    fn e_at(&self, d: usize) -> F128 {
        let mut v = self.c;
        for (b, &z) in self.rho[self.round..].iter().enumerate() {
            v *= if (d >> b) & 1 == 1 { z } else { F128::ONE + z };
        }
        v
    }

    /// Fold every segment at `r` and return the next round's message.
    ///
    /// The message rides the fold pass — computed per segment on the
    /// freshly folded values BEFORE the boundary merge ([`seg_msg`]'s
    /// linearity argument makes that exact) — so a round is ONE pass over
    /// the support, not two. And once the stored support is small the
    /// whole round runs on the calling thread: the tail rounds' work
    /// shrinks toward nothing while a rayon dispatch does not, and those
    /// dispatches (two per round, ~20 rounds, interleaved with the RS
    /// pair's own parallelism) were the group line's cost AND its
    /// run-to-run variance.
    fn fold_round(&mut self, cur: usize, r: F128) -> (F128, F128) {
        debug_assert_eq!(cur, 1usize << (self.rho.len() - self.round));
        debug_assert!(cur >= 4);
        // In char 2 the multilinear eq(z, r) = 1 + z + r.
        self.c *= F128::ONE + self.rho[self.round] + r;
        self.round += 1;
        let this = &*self;
        let fold_one = |seg: &SparseSeg| -> (SparseSeg, F128, F128) {
            let half = seg.x.len() / 2;
            let mut s = seg.start / 2;
            let mut x = Vec::with_capacity(half + 2);
            let mut y = Vec::with_capacity(half + 2);
            if s & 1 == 1 {
                s -= 1;
                x.push(F128::ZERO);
                y.push(this.e_at(s));
            }
            for q in 0..half {
                x.push(seg.x[2 * q] + r * (seg.x[2 * q] + seg.x[2 * q + 1]));
                y.push(seg.y[2 * q] + r * (seg.y[2 * q] + seg.y[2 * q + 1]));
            }
            if (s + x.len()) & 1 == 1 {
                let d = s + x.len();
                x.push(F128::ZERO);
                y.push(this.e_at(d));
            }
            let (p1, pi) = seg_msg(&x, &y);
            (SparseSeg { start: s, x, y }, p1, pi)
        };
        let folded: Vec<(SparseSeg, F128, F128)> = if this.stored() < SPARSE_SERIAL_WORDS {
            this.segs.iter().map(fold_one).collect()
        } else {
            use rayon::prelude::*;
            this.segs.par_iter().map(fold_one).collect()
        };
        // Boundary padding can make neighbors overlap by up to two entries:
        // merge them. x adds (the true weights are disjointly supported and
        // padding is zero — folding is linear in x); y values agree (every
        // stored e entry is the true folded tensor value).
        let mut msg = (F128::ZERO, F128::ZERO);
        let mut segs: Vec<SparseSeg> = Vec::with_capacity(folded.len());
        for (seg, p1, pi) in folded {
            msg = (msg.0 + p1, msg.1 + pi);
            match segs.last_mut() {
                Some(prev) if seg.start < prev.start + prev.x.len() => {
                    let off = seg.start - prev.start;
                    for (i, (&xv, &yv)) in seg.x.iter().zip(&seg.y).enumerate() {
                        let j = off + i;
                        if j < prev.x.len() {
                            prev.x[j] += xv;
                        } else {
                            prev.x.push(xv);
                            prev.y.push(yv);
                        }
                    }
                }
                _ => segs.push(seg),
            }
        }
        self.segs = segs;
        msg
    }

    /// Materialize the dense [`VirtualPair`] for the tail rounds: scatter
    /// the stored weights, rebuild the partner as the scaled eq tensor —
    /// both exactly equal to what the dense path would hold at this round.
    fn densify(&self, cur: usize) -> VirtualPair {
        debug_assert_eq!(cur, 1usize << (self.rho.len() - self.round));
        let mut x = vec![F128::ZERO; cur];
        let mut live = 0usize;
        for seg in &self.segs {
            for (i, &v) in seg.x.iter().enumerate() {
                x[seg.start + i] += v;
            }
            live = live.max(seg.start + seg.x.len());
        }
        // The e partner continues virtually: the accumulated scalar `c` IS
        // its closed form's coefficient at this round.
        VirtualPair::new(
            x,
            Partner::Scaled { s: self.c },
            &self.rho,
            self.round,
            live,
        )
    }
}

/// Densify once the stored support stops paying against the live length.
/// Sparse-vs-dense gate for the GROUP pair, used BOTH at entry (build
/// segment-sparse when `support * this <= 2^m`) and by the mid-sumcheck
/// self-densify (`stored * this > cur`). The two sites must share one
/// factor: `stored` and `cur` both roughly halve per fold round, so the
/// ratio is round-invariant — a pair admitted under a looser entry gate
/// than the densify gate would densify at its FIRST fold, paying the
/// materialize for nothing (measured: the fold line spiked to 11.8 ms
/// when the gates briefly disagreed).
///
/// Was 4; relaxed to 3 (2026-08-14): the F256 registry's extra element
/// columns pushed the envelope INTERNAL's support to 1,062,614 = 25.3% of
/// 2^22 — 1.3% over the old gate — flipping it to the dense path at ~+4 ms
/// of group weight per open (dense 6.4 ms over 2^22 vs sparse ~3.3 at this
/// support; the measured per-word weight crossover is near 2^m/1.5, so 3
/// keeps real margin). The FL at 793k = 18.9% was and stays sparse.
const SPARSE_DENSIFY_FACTOR: usize = 3;

/// Below this stored-support size a sparse fold round runs on the calling
/// thread: the round's whole work is a few thousand multiplies —
/// dispatch-sized — and the tail rounds' dispatches were the group line's
/// variance under contention with the RS pair's parallelism.
const SPARSE_SERIAL_WORDS: usize = 1 << 12;

/// Cap on the RS claim count the lazy pair's per-position side loop hoists
/// for (a fixed-width array). Real proofs carry R = 2; anything wider falls
/// back to the materialized path.
const LAZY_RS_MAX_SIDES: usize = 8;

/// In-process override for the lazy (virtual-ā) dense RS pair: 0 = env
/// (`FLOCK_NO_VIRTUAL_A` disables), 1 = force on, 2 = force off — the
/// alternating-arm contract of [`crate::pcs::VIRTUAL_B_OVERRIDE`].
pub static VIRTUAL_A_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn virtual_a_on() -> bool {
    match VIRTUAL_A_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => std::env::var_os("FLOCK_NO_VIRTUAL_A").is_none(),
    }
}

/// The dense RS product with its full-size combined weight NEVER
/// materialized. Per column `y` the weight is a scaled prefix of each
/// claim's row-eq table — `ā(t_y + row) = Σ_i w_i[y]·eq_i[row]`, `row <
/// h_y`, and the columns tile the area contiguously — so any position is a
/// few-multiply read. Round 0's message streams those reads, and the FIRST
/// fold writes the HALF-SIZE folded vector straight from the same walk:
/// the `2^m` buffer's allocation, its fill traffic and the first fold's
/// re-read of it never happen. From round 1 on this IS the materialized
/// [`VirtualPair`], at half size, on pool memory.
///
/// EXACT: every difference from the materialized path is a reassociation
/// of char-2 sums (skipped terms all carry a zero `ā` factor), so round
/// messages — and proof bytes — are bit-identical. Pinned by
/// `virtual_a_dense_rs_byte_oracle`.
struct LazyRsPair {
    /// Per claim: (per-column weights `γ-power·eq(z_col, y)` over the `2^k`
    /// columns, row-eq table over `2^n`).
    sides: Vec<(Vec<F128>, Vec<F128>)>,
    partner: Partner,
    rho: Vec<F128>,
    pfx: Vec<u64>,
    area: u64,
}

impl LazyRsPair {
    fn new(
        params: &JaggedParams,
        claims: &[FrobeniusClaim<'_>],
        gpow: &[F128],
        partner: Partner,
        rho: &[F128],
    ) -> Self {
        debug_assert!(claims.len() <= LAZY_RS_MAX_SIDES);
        let sides = claims
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let eq_c = build_eq_table(c.z_col);
                // Same grouping as the materialized fill's `w = scale·cols[y]`.
                let cols = eq_c.iter().map(|&e| gpow[128 * i] * e).collect();
                (cols, build_eq_table(c.z_row))
            })
            .collect();
        Self {
            sides,
            partner,
            rho: rho.to_vec(),
            pfx: params.col_prefix_sums.clone(),
            area: params.area(),
        }
    }

    /// `emit(d, ā(d))` for every `d` in `[d0, min(d1, area))`, strictly in
    /// order — the prefix sums are contiguous, so emission has no gaps.
    #[inline]
    fn walk(&self, d0: u64, d1: u64, mut emit: impl FnMut(u64, F128)) {
        let pfx = &self.pfx;
        let n_cols = pfx.len() - 1;
        let d1 = d1.min(self.area);
        if d0 >= d1 {
            return;
        }
        let mut y = pfx.partition_point(|&t| t <= d0).saturating_sub(1);
        let mut d = d0;
        let mut ws = [F128::ZERO; LAZY_RS_MAX_SIDES];
        while d < d1 && y < n_cols {
            let (t_c, t_next) = (pfx[y], pfx[y + 1]);
            if t_next <= d {
                y += 1;
                continue;
            }
            for (w, (cw, _)) in ws.iter_mut().zip(&self.sides) {
                *w = cw[y];
            }
            let stop = d1.min(t_next);
            for dd in d..stop {
                let row = (dd - t_c) as usize;
                let mut v = F128::ZERO;
                for (s, (_, eq_r)) in self.sides.iter().enumerate() {
                    v += ws[s] * eq_r[row];
                }
                emit(dd, v);
            }
            d = stop;
        }
    }

    /// Round 0's message, streamed — the materialized fill's fused message
    /// pass without the fill: pairs whose `ā` members are both zero (past
    /// the area) contribute nothing and are skipped; an odd area's last
    /// pair carries a zero odd member, exactly as the calloc tail did.
    fn round0_msg(&self, eq: &SplitEq) -> (F128, F128) {
        use rayon::prelude::*;
        const CH: u64 = 1 << 14;
        let n_chunks = self.area.div_ceil(CH) as usize;
        (0..n_chunks)
            .into_par_iter()
            .map(|ci| {
                let start = ci as u64 * CH;
                let end = start + CH;
                let mut p1 = F128::ZERO;
                let mut pi = F128::ZERO;
                let mut even = F128::ZERO;
                self.walk(start, end, |d, v| {
                    if d & 1 == 0 {
                        even = v;
                    } else {
                        let q0 = self.partner.at(eq, (d - 1) as usize);
                        let q1 = self.partner.at(eq, d as usize);
                        p1 += v * q1;
                        pi += (even + v) * (q0 + q1);
                    }
                });
                let stop = end.min(self.area);
                if stop > start && stop & 1 == 1 {
                    let d = (stop - 1) as usize;
                    let q0 = self.partner.at(eq, d);
                    let q1 = self.partner.at(eq, d + 1);
                    pi += even * (q0 + q1);
                }
                (p1, pi)
            })
            .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d))
    }

    /// Round 0's fold: advance the partner past `ρ₀`, write the half-size
    /// folded vector straight from the walk (pool memory, guard-zeroed to
    /// the kernel's 4-wide reads), fuse round 1's message over the freshly
    /// written chunks, and hand the result over as a round-1
    /// [`VirtualPair`] — its `fold_round` contract, without the full-size
    /// vector ever existing.
    fn fold0(&mut self, cur: usize, r: F128) -> (VirtualPair, (F128, F128)) {
        use rayon::prelude::*;
        debug_assert_eq!(cur, 1usize << self.rho.len());
        let half = cur / 2;
        self.partner.advance(self.rho[0], r);
        let eq = SplitEq::new(&self.rho[1..]);
        let mut out = crate::scratch::take_f128(half);
        let written = (self.area as usize).div_ceil(2).min(half);
        const CO: usize = 1 << 13;
        let msg = out[..written]
            .par_chunks_mut(CO)
            .enumerate()
            .map(|(ci, oc)| {
                let t0 = ci * CO;
                let d0 = 2 * t0 as u64;
                let d1 = d0 + 2 * oc.len() as u64;
                let mut even = F128::ZERO;
                self.walk(d0, d1, |d, v| {
                    if d & 1 == 0 {
                        even = v;
                    } else {
                        oc[((d - d0) / 2) as usize] = even + r * (v + even);
                    }
                });
                let dstop = d1.min(self.area);
                if dstop > d0 && dstop & 1 == 1 {
                    // Odd area: the pair's odd member is zero — the fold of
                    // (a0, 0) is a0 + r·a0, as the calloc tail produced.
                    oc[((dstop - d0) / 2) as usize] = even + r * even;
                }
                // Fused round-1 message over this chunk's folded pairs —
                // the kernel's exact term order and position convention.
                let mut g1 = F128::ZERO;
                let mut gi = F128::ZERO;
                for (j, op) in oc.as_chunks::<2>().0.iter().enumerate() {
                    let u = t0 + 2 * j;
                    let p0 = self.partner.at(&eq, u);
                    let p1 = self.partner.at(&eq, u + 1);
                    g1 += op[1] * p1;
                    gi += (op[0] + op[1]) * (p0 + p1);
                }
                if oc.len() & 1 == 1 {
                    // `written` odd: the pair's never-materialized odd
                    // member is zero — only the G(∞) term survives.
                    let na0 = oc[oc.len() - 1];
                    let u = t0 + oc.len() - 1;
                    let p0 = self.partner.at(&eq, u);
                    let p1 = self.partner.at(&eq, u + 1);
                    gi += na0 * (p0 + p1);
                }
                (g1, gi)
            })
            .reduce(|| (F128::ZERO, F128::ZERO), |(a, b), (c, d)| (a + c, b + d));
        // Guard slots: the pool half is dirty past `written`, and the
        // rounds read the live prefix rounded to the kernel's 4-wide
        // chunks — mirror [`VirtualPair::new`]'s rounding exactly.
        let live_end = written.next_multiple_of(4).max(4).min(half);
        for slot in &mut out[written..live_end] {
            *slot = F128::ZERO;
        }
        let partner = std::mem::replace(&mut self.partner, Partner::Scaled { s: F128::ZERO });
        (VirtualPair::new(out, partner, &self.rho, 1, written), msg)
    }
}

/// A product of the two-product sumcheck: a materialized weight with a
/// virtual partner ([`VirtualPair`]), the group product's segment-sparse
/// state ([`SparseGroupPair`]), or the RS product's aligned closed form —
/// eager ([`AlignedRsPair`]) or lazy ([`LazyRsPair`]); each non-virtual
/// variant densifies itself into the first for the tail rounds.
enum Pair {
    Virtual(VirtualPair),
    Sparse(SparseGroupPair),
    AlignedRs(AlignedRsPair),
    LazyRs(LazyRsPair),
}

impl Pair {
    fn fold_round(&mut self, cur: usize, r: F128) -> (F128, F128) {
        // The lazy RS pair materializes at its first fold — half-size,
        // with round 1's message fused into the materializing pass.
        if let Pair::LazyRs(l) = self {
            let (vp, msg) = l.fold0(cur, r);
            *self = Pair::Virtual(vp);
            return msg;
        }
        if let Pair::Sparse(sp) = self
            && sp.stored() * SPARSE_DENSIFY_FACTOR > cur
        {
            *self = Pair::Virtual(sp.densify(cur));
        }
        // The aligned RS form holds only while at least one row bit remains
        // AFTER the fold (its per-column message kernel needs both legs of
        // a pair inside one column): densify at 2·2^k_cols entries.
        if let Pair::AlignedRs(a) = self
            && a.round + 1 >= a.nu
        {
            *self = Pair::Virtual(a.densify());
        }
        match self {
            Pair::Virtual(p) => p.fold_round(cur, r),
            Pair::Sparse(p) => p.fold_round(cur, r),
            Pair::AlignedRs(p) => p.fold_round(cur, r),
            Pair::LazyRs(_) => unreachable!("materialized by the early return above"),
        }
    }
}

/// Verifier for the multipoint twisted evaluation, two-product form. On
/// success returns the verified
/// `V = Σ_{i,j} c_{i,j}·A_{i,j}^{2^j} + Σ_k B_k = Ŵ(ρ)`.
pub fn verify_multipoint_twisted<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[ScalarGroupClaim<'_>],
    rho: &[F128],
    proof: &MultipointTwistedProof,
    challenger: &mut C,
) -> Option<F128> {
    verify_multipoint_twisted_with_grinding(
        params,
        claims,
        groups,
        rho,
        proof,
        MultipointGrinding::disabled(),
        challenger,
    )
}

/// [`verify_multipoint_twisted`] with the matching PoW checks for its
/// batching coefficient, every two-product round, and the anchor rounds.
pub fn verify_multipoint_twisted_with_grinding<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[ScalarGroupClaim<'_>],
    rho: &[F128],
    proof: &MultipointTwistedProof,
    grinding: MultipointGrinding,
    challenger: &mut C,
) -> Option<F128> {
    verify_multipoint_twisted_core(
        params, claims, groups, rho, proof, grinding, challenger, None,
    )
}

/// The deferred export of the whole multipoint anchor: the assist's final
/// point and per-statement factors, plus the anchor-level coefficients the
/// expect recombination applies — everything a parent needs to re-express
/// the count-dependent side as claims on the jagged layout
/// ([`crate::matrix_fold::JaggedClaim`]) instead of evaluating it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultipointDefer {
    /// The anchor sumcheck's bound point `ρ″` (length `m`).
    pub point: Vec<F128>,
    /// The assist's final point `σ` — every jagged claim's column point.
    pub sigma: Vec<F128>,
    /// Merged-statement factors `w_st`, in [`frobenius_statements`]'s
    /// deterministic order (claims first, then groups, row-point merged).
    pub statement_ws: Vec<F128>,
    /// Per RS claim: the anchor's single nonzero coefficient
    /// `γ^{128i}·ĝ(ρ″)`.
    pub rs_coeffs: Vec<F128>,
    /// Per scalar group: the statement-level coefficient
    /// `γ^{128·n_rs+k}·eq(ρ, ρ″)`.
    pub group_coeffs: Vec<F128>,
}

/// Deferred counterpart to [`verify_multipoint_twisted_with_grinding`] —
/// transcript-identical, the export only copies what the plain path
/// computes.
pub fn verify_multipoint_twisted_deferred_with_grinding<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[ScalarGroupClaim<'_>],
    rho: &[F128],
    proof: &MultipointTwistedProof,
    grinding: MultipointGrinding,
    challenger: &mut C,
) -> Option<(F128, MultipointDefer)> {
    let mut out = None;
    let v = verify_multipoint_twisted_core(
        params,
        claims,
        groups,
        rho,
        proof,
        grinding,
        challenger,
        Some(&mut out),
    )?;
    Some((v, out.expect("the deferred core fills the export")))
}

fn verify_multipoint_twisted_core<C: Challenger>(
    params: &JaggedParams,
    claims: &[FrobeniusClaim<'_>],
    groups: &[ScalarGroupClaim<'_>],
    rho: &[F128],
    proof: &MultipointTwistedProof,
    grinding: MultipointGrinding,
    challenger: &mut C,
    defer: Option<&mut Option<MultipointDefer>>,
) -> Option<F128> {
    let m = params.m;
    if proof.values.len() != claims.len()
        || proof.group_values.len() != groups.len()
        || proof.rounds.len() != m
    {
        return None;
    }
    if proof.values.iter().any(|vs| vs.len() != 128) {
        return None;
    }
    let expected_round_nonces = if grinding.round_bits == 0 { 0 } else { m };
    if proof.round_grinding_nonces.len() != expected_round_nonces {
        return None;
    }
    for g in groups {
        assert_eq!(
            g.cols.len(),
            1usize << params.k,
            "group cols must be dense over 2^k columns"
        );
    }
    let n_rs = claims.len();
    let n_g = groups.len();
    challenger.observe_label(b"flock-multipoint-twisted-v1");
    for vs in &proof.values {
        for &v in vs {
            challenger.observe_f128(v);
        }
    }
    for &v in &proof.group_values {
        challenger.observe_f128(v);
    }
    // As with ring switching, this optional operation is absent from a
    // disabled transcript, so check canonical zero without absorbing it.
    let gamma_bits = grinding.gamma_bits_for(n_rs, n_g);
    let gamma = if gamma_bits != 0 {
        challenger.verify_pow_and_sample_f128(proof.gamma_grinding_nonce, gamma_bits)?
    } else {
        if proof.gamma_grinding_nonce != 0 {
            return None;
        }
        challenger.sample_f128()
    };
    let mut gpow = Vec::with_capacity(128 * n_rs + n_g);
    let mut p = F128::ONE;
    for _ in 0..128 * n_rs + n_g {
        gpow.push(p);
        p *= gamma;
    }

    // Sumcheck target from the claimed values; replay the rounds.
    let mut running = F128::ZERO;
    for (i, vs) in proof.values.iter().enumerate() {
        for (j, &v) in vs.iter().enumerate() {
            running += gpow[128 * i + j] * v;
        }
    }
    for (k, &v) in proof.group_values.iter().enumerate() {
        running += gpow[128 * n_rs + k] * v;
    }
    let mut point = Vec::with_capacity(m);
    for (round, &(g_one, g_inf)) in proof.rounds.iter().enumerate() {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = if grinding.round_bits != 0 {
            challenger.verify_pow_and_sample_f128(
                proof.round_grinding_nonces[round],
                grinding.round_bits,
            )?
        } else {
            challenger.sample_f128()
        };
        running = fold_round_claim(running, g_one, g_inf, r);
        point.push(r);
    }

    // The endpoint's closed-form factors — ĝ(ρ'') (Lemma "twisted eq") for
    // the RS product, eq(ρ,ρ'') for the groups' — baked into the anchor's
    // coefficients so ONE assist binds the whole endpoint sum
    // `ĝ(ρ'')·ā̂(ρ'') + eq(ρ,ρ'')·b̂(ρ'')` and the accept check is a plain
    // equality against the running claim.
    let g_at = if n_rs > 0 {
        twisted_eq_at(&gpow, &rho_inverse_powers(rho), &point)
    } else {
        F128::ZERO
    };
    let e_at = if n_g > 0 {
        eq_at(rho, &point)
    } else {
        F128::ZERO
    };
    let anchor_coeffs: Vec<Vec<F128>> = (0..n_rs)
        .map(|i| {
            let mut c = vec![F128::ZERO; 128];
            c[0] = gpow[128 * i] * g_at;
            c
        })
        .collect();
    let anchor_claims: Vec<FrobeniusClaim<'_>> = claims
        .iter()
        .zip(&anchor_coeffs)
        .map(|(cl, co)| FrobeniusClaim {
            z_row: cl.z_row,
            z_col: cl.z_col,
            coeffs: co,
        })
        .collect();
    let anchor_groups: Vec<(ScalarGroupClaim<'_>, F128)> = groups
        .iter()
        .enumerate()
        .map(|(k, g)| (*g, gpow[128 * n_rs + k] * e_at))
        .collect();
    let mut assist_out: Option<AssistDefer> = None;
    let s_at = verify_frobenius_assist_core(
        params,
        &anchor_claims,
        &anchor_groups,
        &point,
        &proof.anchor,
        grinding.anchor_round_bits,
        challenger,
        defer.is_some().then_some(&mut assist_out),
    )?;
    if running != s_at {
        return None;
    }
    if let Some(out) = defer {
        let a = assist_out.expect("the assist core fills the export when asked");
        *out = Some(MultipointDefer {
            point: point.clone(),
            sigma: a.sigma,
            statement_ws: a.statement_ws,
            rs_coeffs: (0..n_rs).map(|i| gpow[128 * i] * g_at).collect(),
            group_coeffs: (0..n_g).map(|k| gpow[128 * n_rs + k] * e_at).collect(),
        });
    }

    // Recombine the verified values:
    // V = Σ_{i,j} c_{i,j}·A_{i,j}^{2^j} + Σ_k B_k (a group's fold map is
    // the identity and its γ's are baked into the cols, so its coefficient
    // is 1).
    let mut total = F128::ZERO;
    for (claim, vs) in claims.iter().zip(&proof.values) {
        for (j, (&c, &v)) in claim.coeffs.iter().zip(vs).enumerate() {
            if c.is_zero() {
                continue;
            }
            let mut x = v;
            for _ in 0..j {
                x = x * x;
            }
            total += c * x;
        }
    }
    for &v in &proof.group_values {
        total += v;
    }
    Some(total)
}

/// Reduce the running sumcheck claim through one round. The degree-2 round
/// polynomial `G` is given by `G(1) = g_one`, leading coeff `G(∞) = g_inf`, and
/// `G(0) = claim + G(1)` (since `claim = G(0) + G(1)`). Returns `G(r)`.
#[inline]
pub(crate) fn fold_round_claim(claim: F128, g_one: F128, g_inf: F128, r: F128) -> F128 {
    let g0 = claim + g_one; // char-2: G(0) = claim - G(1)
    // G(X) = g0 + (G(1) + g0 + g_inf)·X + g_inf·X²
    g0 + (g_one + g0 + g_inf) * r + g_inf * (r * r)
}

/// Degree-2 round message `(G(1), G(∞))` for `Σ_{x'} a(X,x')·b(X,x')`, low bit
/// bound: `a(0,x') = a[2x']`, `a(1,x') = a[2x'+1]`. Serial reference; retained
/// for the `runtime_m25` serial-vs-parallel benchmark. (The production path
/// gets round 1's message fused into [`generate_f_and_claim`] and later
/// messages from the fused fold kernels.)
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
/// ~2.6× to ~6× parallel scaling (hits the memory-bandwidth ceiling);
/// measured with the since-deleted `scaling_diag` probe (bloat ledger §E).
/// No longer on the production path (round 1's message is fused into
/// [`generate_f_and_claim`]); retained for the runtime benchmarks.
#[allow(dead_code)]
pub(crate) fn round_msg_par(a: &[F128], b: &[F128]) -> (F128, F128) {
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
pub(crate) fn fold_oop_par(a: &[F128], b: &[F128], r: F128, ao: &mut [F128], bo: &mut [F128]) {
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
/// Shared with the merged-open sumcheck (`pcs::open_batch_merged`),
/// which runs the same product-sumcheck round structure.
pub(crate) fn fold_and_round_oop_par(
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

    /// The batched Frobenius assist proves exactly the Φ-twisted weight
    /// evaluation: the prover's `V` equals the brute-force
    /// `Σ_e eq(ρ,e)·Σ_i Φ_i(eq_row·eq_col)`, the verifier accepts and
    /// returns it, and tampering with a round message or the claimed `V`
    /// is rejected. Real subset-sum fold tables; two claims plus one
    /// merged-cols scalar group; random non-power-of-two heights.
    #[test]
    fn frobenius_assist_roundtrip_and_tamper() {
        use crate::pcs::ring_switch;
        let mut ch = RandomChallenger::new(0xF12B_A551);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7)] {
            let (params, _q) = random_instance(&mut ch, n, k, m);
            let claims_data: Vec<(Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>)> = (0..2)
                .map(|_| {
                    let zr = sample_vec(&mut ch, n);
                    let zc = sample_vec(&mut ch, k);
                    let eq_r: Vec<F128> = (0..128).map(|_| ch.sample_f128()).collect();
                    let table = ring_switch::build_fold_byte_table(&eq_r);
                    let coeffs = ring_switch::linearized_coefficients(&table);
                    (zr, zc, table, coeffs)
                })
                .collect();
            let g_zr = sample_vec(&mut ch, n);
            let g_cols = sample_vec(&mut ch, 1 << k);
            let g_coeff = ch.sample_f128();
            let rho = sample_vec(&mut ch, m);
            let eq_idx = build_eq_table(&rho);
            let mut v_expect = F128::ZERO;
            for cd in &claims_data {
                let eq_row = build_eq_table(&cd.0);
                let eq_col = build_eq_table(&cd.1);
                for e in 0..params.area() {
                    let (row, col) = params.unrank(e);
                    v_expect += eq_idx[e as usize]
                        * ring_switch::fold_one_slot(eq_row[row] * eq_col[col], &cd.2);
                }
            }
            let g_eq_row = build_eq_table(&g_zr);
            for e in 0..params.area() {
                let (row, col) = params.unrank(e);
                v_expect += eq_idx[e as usize] * g_coeff * g_eq_row[row] * g_cols[col];
            }
            let fclaims: Vec<FrobeniusClaim<'_>> = claims_data
                .iter()
                .map(|c| FrobeniusClaim {
                    z_row: &c.0,
                    z_col: &c.1,
                    coeffs: &c.3,
                })
                .collect();
            let fgroups = [(
                ScalarGroupClaim {
                    z_row: &g_zr,
                    cols: &g_cols,
                },
                g_coeff,
            )];
            let mut chp = FsChallenger::new(b"frobenius-assist-test");
            let proof = prove_frobenius_assist(&params, &fclaims, &fgroups, &rho, &mut chp);
            assert_eq!(proof.v, v_expect, "V must equal the twisted evaluation");
            let mut chv = FsChallenger::new(b"frobenius-assist-test");
            assert_eq!(
                verify_frobenius_assist(&params, &fclaims, &fgroups, &rho, &proof, &mut chv),
                Some(v_expect)
            );
            let mut bad = proof.clone();
            bad.rounds[1].0 += F128::ONE;
            let mut chv = FsChallenger::new(b"frobenius-assist-test");
            assert_eq!(
                verify_frobenius_assist(&params, &fclaims, &fgroups, &rho, &bad, &mut chv),
                None
            );
            let mut bad = proof.clone();
            bad.v += F128::ONE;
            let mut chv = FsChallenger::new(b"frobenius-assist-test");
            assert_eq!(
                verify_frobenius_assist(&params, &fclaims, &fgroups, &rho, &bad, &mut chv),
                None
            );
        }
    }

    /// The Frobenius-twist identity behind the merged-reduction design
    /// sketch (design doc §"Capacity-free ring-switching"): for every j,
    /// `Σ_e eq(ρ,e) · (eq_row[row(e)]·eq_col[col(e)])^(2^j)`
    /// `  = f̂_t(z_row^(2^j), z_col^(2^j), ρ)`
    /// — Frobenius is a field automorphism and commutes with the eq-product
    /// structure at Boolean selectors (`eq(z,b)^(2^j) = eq(z^(2^j), b)`), so
    /// each Frobenius power of the jagged weight IS the ordinary jagged MLE
    /// at Frobenius-powered z-points, with the α-side point ρ untouched.
    /// Combined with the linearized-polynomial form of any F₂-linear map
    /// (`Φ(x) = Σ_j c_j·x^(2^j)`), the Φ-twisted weight evaluation is an
    /// F-combination of 128 ordinary assist statements. Random heights,
    /// non-power-of-two, zero columns included.
    #[test]
    fn frobenius_twist_matches_assist_object() {
        let frob = |x: F128, j: usize| -> F128 {
            let mut y = x;
            for _ in 0..j {
                y = y * y;
            }
            y
        };
        let mut ch = RandomChallenger::new(0xF20B_E415);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6)] {
            for _ in 0..4 {
                let (params, _q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);
                let rho = sample_vec(&mut ch, m);
                let eq_row = build_eq_table(&z_row);
                let eq_col = build_eq_table(&z_col);
                let eq_idx = build_eq_table(&rho);
                for j in [0usize, 1, 2, 7, 40] {
                    // LHS: the j-th Frobenius power of the twisted weight,
                    // summed directly over the dense domain.
                    let mut lhs = F128::ZERO;
                    for e in 0..params.area() {
                        let (row, col) = params.unrank(e);
                        lhs += eq_idx[e as usize] * frob(eq_row[row] * eq_col[col], j);
                    }
                    // RHS: the ordinary jagged MLE at Frobenius-powered
                    // z-points, ρ untouched.
                    let zr: Vec<F128> = z_row.iter().map(|&z| frob(z, j)).collect();
                    let zc: Vec<F128> = z_col.iter().map(|&z| frob(z, j)).collect();
                    let rhs = f_hat_t_bruteforce(&params, &zr, &zc, &rho);
                    assert_eq!(lhs, rhs, "n={n} k={k} m={m} j={j}");
                }
            }
        }
    }

    /// Shared oracle body: prove + verify a two-claim multipoint twisted
    /// evaluation on `params`, compare against the brute-force twisted
    /// weight Ŵ(ρ), and reject a tampered value.
    fn check_multipoint(params: &JaggedParams, ch: &mut RandomChallenger, label: &str) {
        let (n, k, m) = (params.n, params.k, params.m);
        let z1r = sample_vec(ch, n);
        let z1c = sample_vec(ch, k);
        let z2r = sample_vec(ch, n);
        let z2c = sample_vec(ch, k);
        let mut c1 = sample_vec(ch, 128);
        let mut c2 = sample_vec(ch, 128);
        c1[7] = F128::ZERO; // zero coefficients are skipped
        c2[100] = F128::ZERO;
        // Two scalar groups: γ-baked merged cols (a dense one and a one-hot
        // one — the gather-claim shape), fold map the identity.
        let g1r = sample_vec(ch, n);
        let g1cols = sample_vec(ch, 1 << k);
        let g2r = sample_vec(ch, n);
        let mut g2cols = vec![F128::ZERO; 1 << k];
        g2cols[(1usize << k) - 1] = ch.sample_f128();
        let rho = sample_vec(ch, m);
        let claims = [
            FrobeniusClaim {
                z_row: &z1r,
                z_col: &z1c,
                coeffs: &c1,
            },
            FrobeniusClaim {
                z_row: &z2r,
                z_col: &z2c,
                coeffs: &c2,
            },
        ];
        let groups = [
            ScalarGroupClaim {
                z_row: &g1r,
                cols: &g1cols,
            },
            ScalarGroupClaim {
                z_row: &g2r,
                cols: &g2cols,
            },
        ];
        let mut chp = FsChallenger::new(b"multipoint-test");
        let proof = prove_multipoint_twisted(params, &claims, &groups, &rho, &mut chp);
        let mut chv = FsChallenger::new(b"multipoint-test");
        let v = verify_multipoint_twisted(params, &claims, &groups, &rho, &proof, &mut chv)
            .expect("honest multipoint proof verifies");

        // Brute force: Ŵ(ρ) = Σ_d eq(ρ,d)·(Σ_i Φ_i(a_{i,d}) + Σ_k h_{k,d}).
        let eq_idx = build_eq_table(&rho);
        let sides = [
            (build_eq_table(&z1r), build_eq_table(&z1c), &c1),
            (build_eq_table(&z2r), build_eq_table(&z2c), &c2),
        ];
        let gsides = [
            (build_eq_table(&g1r), &g1cols),
            (build_eq_table(&g2r), &g2cols),
        ];
        let mut expect = F128::ZERO;
        for e in 0..params.area() {
            let (row, col) = params.unrank(e);
            for (eq_r, eq_c, cs) in &sides {
                let mut x = eq_r[row] * eq_c[col];
                for &cj in cs.iter() {
                    if !cj.is_zero() {
                        expect += eq_idx[e as usize] * cj * x;
                    }
                    x = x * x;
                }
            }
            for (eq_r, cols) in &gsides {
                expect += eq_idx[e as usize] * eq_r[row] * cols[col];
            }
        }
        assert_eq!(v, expect, "{label}");

        // Tamper: a perturbed value must be rejected — RS and group alike.
        let mut bad = proof.clone();
        bad.values[0][3] += F128::ONE;
        let mut chb = FsChallenger::new(b"multipoint-test");
        assert!(
            verify_multipoint_twisted(params, &claims, &groups, &rho, &bad, &mut chb).is_none(),
            "tampered value accepted ({label})"
        );
        let mut bad = proof.clone();
        bad.group_values[1] += F128::ONE;
        let mut chb = FsChallenger::new(b"multipoint-test");
        assert!(
            verify_multipoint_twisted(params, &claims, &groups, &rho, &bad, &mut chb).is_none(),
            "tampered group value accepted ({label})"
        );

        // The groups-only statement (the element-only shape: no RS claims,
        // single-product sumcheck over eq(ρ,·)) proves and verifies.
        let mut chp = FsChallenger::new(b"multipoint-test-go");
        let go = prove_multipoint_twisted(params, &[], &groups, &rho, &mut chp);
        assert!(go.values.is_empty());
        let mut chv = FsChallenger::new(b"multipoint-test-go");
        let vg = verify_multipoint_twisted(params, &[], &groups, &rho, &go, &mut chv)
            .expect("groups-only multipoint proof verifies");
        let mut expect_g = F128::ZERO;
        for e in 0..params.area() {
            let (row, col) = params.unrank(e);
            for (eq_r, cols) in &gsides {
                expect_g += eq_idx[e as usize] * eq_r[row] * cols[col];
            }
        }
        assert_eq!(vg, expect_g, "groups-only {label}");
    }

    /// The secure multipoint transport has three independent grinding
    /// families: the dual-value batching scalar, each main quadratic round,
    /// and each quadratic anchor round.  Check their transcript placement and
    /// exact proof shapes directly, in addition to the full merged-opening
    /// and recursive-node end-to-end tests.
    #[test]
    fn multipoint_twisted_grinding_roundtrip_and_rejects_malformed_nonces() {
        let mut rng = RandomChallenger::new(0x4D50_4752_494E_44);
        let params = JaggedParams::from_heights(&[8; 4], 3, 5);
        let z_row = sample_vec(&mut rng, params.n);
        let z_col = sample_vec(&mut rng, params.k);
        let coeffs = sample_vec(&mut rng, 128);
        let group_row = sample_vec(&mut rng, params.n);
        let group_cols = sample_vec(&mut rng, 1 << params.k);
        let rho = sample_vec(&mut rng, params.m);
        let claims = [FrobeniusClaim {
            z_row: &z_row,
            z_col: &z_col,
            coeffs: &coeffs,
        }];
        let groups = [ScalarGroupClaim {
            z_row: &group_row,
            cols: &group_cols,
        }];
        let grinding = MultipointGrinding::per_challenge_128();

        let mut ch_p = FsChallenger::new(b"multipoint-grinding-test");
        let proof = prove_multipoint_twisted_with_grinding(
            &params, &claims, &groups, &rho, grinding, &mut ch_p,
        );
        assert_eq!(proof.round_grinding_nonces.len(), params.m);
        assert_eq!(proof.anchor.grinding_nonces.len(), 2 * (params.m + 1));

        let mut ch_v = FsChallenger::new(b"multipoint-grinding-test");
        assert!(
            verify_multipoint_twisted_with_grinding(
                &params, &claims, &groups, &rho, &proof, grinding, &mut ch_v,
            )
            .is_some(),
            "honest grinded multipoint proof verifies"
        );

        let mut missing_round = proof.clone();
        missing_round.round_grinding_nonces.pop();
        let mut ch_v = FsChallenger::new(b"multipoint-grinding-test");
        assert!(
            verify_multipoint_twisted_with_grinding(
                &params,
                &claims,
                &groups,
                &rho,
                &missing_round,
                grinding,
                &mut ch_v,
            )
            .is_none(),
            "a missing main-round PoW must reject"
        );

        let mut missing_anchor = proof.clone();
        missing_anchor.anchor.grinding_nonces.pop();
        let mut ch_v = FsChallenger::new(b"multipoint-grinding-test");
        assert!(
            verify_multipoint_twisted_with_grinding(
                &params,
                &claims,
                &groups,
                &rho,
                &missing_anchor,
                grinding,
                &mut ch_v,
            )
            .is_none(),
            "a missing anchor-round PoW must reject"
        );

        // Find a deterministic invalid witness at the gamma position against
        // the exact transcript prefix.  This pin specifically covers the
        // batching PoW rather than a later sumcheck rejection after gamma.
        let bad_gamma = (0..256u64)
            .find(|&nonce| {
                if nonce == proof.gamma_grinding_nonce {
                    return false;
                }
                let mut ch = FsChallenger::new(b"multipoint-grinding-test");
                ch.observe_label(b"flock-multipoint-twisted-v1");
                for vs in &proof.values {
                    for &v in vs {
                        ch.observe_f128(v);
                    }
                }
                for &v in &proof.group_values {
                    ch.observe_f128(v);
                }
                !ch.verify_pow(nonce, grinding.gamma_bits_for(1, 1))
            })
            .expect("a fixed nonce window contains an invalid gamma PoW witness");
        let mut bad = proof.clone();
        bad.gamma_grinding_nonce = bad_gamma;
        let mut ch_v = FsChallenger::new(b"multipoint-grinding-test");
        assert!(
            verify_multipoint_twisted_with_grinding(
                &params, &claims, &groups, &rho, &bad, grinding, &mut ch_v,
            )
            .is_none(),
            "an invalid multipoint-batching PoW must reject"
        );

        // The gamma PoW is absent, not a zero-bit transcript operation, in a
        // legacy proof.  Its carried optional field must nevertheless stay
        // canonical and therefore cannot become a free proof-malleability
        // knob.
        let mut ch_p = FsChallenger::new(b"multipoint-no-grinding-test");
        let mut legacy = prove_multipoint_twisted(&params, &claims, &groups, &rho, &mut ch_p);
        legacy.gamma_grinding_nonce = 1;
        let mut ch_v = FsChallenger::new(b"multipoint-no-grinding-test");
        assert!(
            verify_multipoint_twisted(&params, &claims, &groups, &rho, &legacy, &mut ch_v)
                .is_none(),
            "disabled optional gamma nonce must be canonical zero"
        );
    }

    #[test]
    fn multipoint_gamma_schedule_tracks_power_boundaries() {
        let grinding = MultipointGrinding::per_challenge_128();
        // K=256 gives degree 255 and needs 8 bits; K=257 gives degree 256
        // and needs 9 under the strict local `< 2^-128` rule.
        assert_eq!(grinding.gamma_bits_for(2, 0), 8);
        assert_eq!(grinding.gamma_bits_for(2, 1), 9);
        assert_eq!(MultipointGrinding::disabled().gamma_bits_for(2, 1), 0);
    }

    /// Sparse-group variant of [`check_multipoint`]: groups supported on
    /// `hot` columns only, so the prover takes the [`SparseGroupPair`]
    /// path (with RS claims and groups-only), against the same brute
    /// force, plus a tampered-value rejection.
    fn check_multipoint_sparse(
        params: &JaggedParams,
        ch: &mut RandomChallenger,
        hot: &[usize],
        label: &str,
    ) {
        let (n, k, m) = (params.n, params.k, params.m);
        let z1r = sample_vec(ch, n);
        let z1c = sample_vec(ch, k);
        let c1 = sample_vec(ch, 128);
        let g1r = sample_vec(ch, n);
        let g2r = sample_vec(ch, n);
        let mut g1cols = vec![F128::ZERO; 1 << k];
        let mut g2cols = vec![F128::ZERO; 1 << k];
        for &y in hot {
            g1cols[y] = ch.sample_f128();
        }
        // The second group is hot on a subset — segments must merge
        // identically whether one or both groups weight a column.
        for &y in hot.iter().step_by(2) {
            g2cols[y] = ch.sample_f128();
        }
        let rho = sample_vec(ch, m);
        let claims = [FrobeniusClaim {
            z_row: &z1r,
            z_col: &z1c,
            coeffs: &c1,
        }];
        let groups = [
            ScalarGroupClaim {
                z_row: &g1r,
                cols: &g1cols,
            },
            ScalarGroupClaim {
                z_row: &g2r,
                cols: &g2cols,
            },
        ];
        let support = SparseGroupPair::support_area(params, &groups);
        assert!(
            (support as usize) * SPARSE_DENSIFY_FACTOR <= (1usize << m),
            "test shape must engage the sparse path ({label})"
        );
        let mut chp = FsChallenger::new(b"multipoint-sparse-test");
        let proof = prove_multipoint_twisted(params, &claims, &groups, &rho, &mut chp);
        let mut chv = FsChallenger::new(b"multipoint-sparse-test");
        let v = verify_multipoint_twisted(params, &claims, &groups, &rho, &proof, &mut chv)
            .expect("honest sparse-group multipoint proof verifies");

        let eq_idx = build_eq_table(&rho);
        let (eq_z1r, eq_z1c) = (build_eq_table(&z1r), build_eq_table(&z1c));
        let gsides = [
            (build_eq_table(&g1r), &g1cols),
            (build_eq_table(&g2r), &g2cols),
        ];
        let mut expect = F128::ZERO;
        let mut expect_g = F128::ZERO;
        for e in 0..params.area() {
            let (row, col) = params.unrank(e);
            let mut x = eq_z1r[row] * eq_z1c[col];
            for &cj in c1.iter() {
                if !cj.is_zero() {
                    expect += eq_idx[e as usize] * cj * x;
                }
                x = x * x;
            }
            for (eq_r, cols) in &gsides {
                expect_g += eq_idx[e as usize] * eq_r[row] * cols[col];
            }
        }
        assert_eq!(v, expect + expect_g, "{label}");

        let mut bad = proof.clone();
        bad.group_values[0] += F128::ONE;
        let mut chb = FsChallenger::new(b"multipoint-sparse-test");
        assert!(
            verify_multipoint_twisted(params, &claims, &groups, &rho, &bad, &mut chb).is_none(),
            "tampered sparse group value accepted ({label})"
        );

        // Groups-only: the sparse pair is the sumcheck's ONLY product.
        let mut chp = FsChallenger::new(b"multipoint-sparse-test-go");
        let go = prove_multipoint_twisted(params, &[], &groups, &rho, &mut chp);
        let mut chv = FsChallenger::new(b"multipoint-sparse-test-go");
        let vg = verify_multipoint_twisted(params, &[], &groups, &rho, &go, &mut chv)
            .expect("groups-only sparse multipoint proof verifies");
        assert_eq!(vg, expect_g, "groups-only {label}");
    }

    /// The segment-sparse group path against the brute force. Odd column
    /// heights put every segment boundary at an odd word (build-time even
    /// extension, fold-time boundary padding and overlap merge, mid-loop
    /// densify all fire); the all-hot tiny-height shape merges the whole
    /// support into one run; hot columns sit at both table edges.
    #[test]
    fn multipoint_twisted_sparse_groups_matches_bruteforce() {
        let mut ch = RandomChallenger::new(0x4D50_59A5);
        // 64 columns of height 37: support 5·37 = 185 ≪ 2^12/4, segments
        // stay sparse for ~7 folds before densifying.
        let heights = vec![37u64; 64];
        let params = JaggedParams::from_heights(&heights, 6, 12);
        for rep in 0..3 {
            check_multipoint_sparse(
                &params,
                &mut ch,
                &[0, 1, 5, 37, 63],
                &format!("odd heights rep={rep}"),
            );
        }
        // Every column hot at height 2: one merged support run of 128.
        let heights = vec![2u64; 64];
        let params = JaggedParams::from_heights(&heights, 6, 12);
        let hot: Vec<usize> = (0..64).collect();
        check_multipoint_sparse(&params, &mut ch, &hot, "all-hot tiny heights");
    }

    /// The multipoint twisted evaluation returns the brute-force twisted
    /// weight Ŵ(ρ) — random jagged shapes, two claims, zero coefficients —
    /// and rejects a tampered value.
    #[test]
    fn multipoint_twisted_matches_bruteforce() {
        let mut ch = RandomChallenger::new(0x4D50_7715);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6)] {
            for rep in 0..3 {
                let (params, _q) = random_instance(&mut ch, n, k, m);
                check_multipoint(&params, &mut ch, &format!("n={n} k={k} m={m} rep={rep}"));
            }
        }
    }

    /// THE VIRTUAL-ā ORACLE: the lazy dense RS pair (no `2^m` weight
    /// buffer, half-size materialization at fold 0) must produce proofs
    /// EQUAL to the materialized path's — the same round messages by exact
    /// char-2 reassociation — across random jagged shapes (odd areas
    /// included), R ∈ {1, 2}, with and without groups. Arms forced via
    /// [`VIRTUAL_A_OVERRIDE`], the alternating in-process pattern.
    #[test]
    fn virtual_a_dense_rs_byte_oracle() {
        use std::sync::atomic::Ordering;
        let mut ch = RandomChallenger::new(0xA11A_5EED);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (5, 2, 8), (2, 4, 6)] {
            for rep in 0..3 {
                let (params, _q) = random_instance(&mut ch, n, k, m);
                let z1r = sample_vec(&mut ch, n);
                let z1c = sample_vec(&mut ch, k);
                let z2r = sample_vec(&mut ch, n);
                let z2c = sample_vec(&mut ch, k);
                let c1 = sample_vec(&mut ch, 128);
                let c2 = sample_vec(&mut ch, 128);
                let g1r = sample_vec(&mut ch, n);
                let g1cols = sample_vec(&mut ch, 1 << k);
                let rho = sample_vec(&mut ch, m);
                let claims_all = [
                    FrobeniusClaim {
                        z_row: &z1r,
                        z_col: &z1c,
                        coeffs: &c1,
                    },
                    FrobeniusClaim {
                        z_row: &z2r,
                        z_col: &z2c,
                        coeffs: &c2,
                    },
                ];
                let groups_all = [ScalarGroupClaim {
                    z_row: &g1r,
                    cols: &g1cols,
                }];
                for (n_claims, n_groups) in [(2usize, 1usize), (1, 1), (2, 0)] {
                    let claims = &claims_all[..n_claims];
                    let groups = &groups_all[..n_groups];
                    let arm = |force: u8| {
                        VIRTUAL_A_OVERRIDE.store(force, Ordering::Relaxed);
                        let mut chp = FsChallenger::new(b"virtual-a-oracle");
                        prove_multipoint_twisted(&params, claims, groups, &rho, &mut chp)
                    };
                    let lazy = arm(1);
                    let dense = arm(2);
                    VIRTUAL_A_OVERRIDE.store(0, Ordering::Relaxed);
                    assert_eq!(
                        lazy,
                        dense,
                        "n={n} k={k} m={m} rep={rep} R={n_claims} G={n_groups} area={}",
                        params.area()
                    );
                    let mut chv = FsChallenger::new(b"virtual-a-oracle");
                    verify_multipoint_twisted(&params, claims, groups, &rho, &lazy, &mut chv)
                        .expect("lazy-arm proof verifies");
                }
            }
        }
    }

    /// Alternating-arm instrument on the SPINE NODE's jagged geometry (687
    /// used columns of 2^11, area ≈2.31M of 2^22, R = 2 + two groups):
    /// lazy vs materialized medians over `MICRO_RUNS` (default 5), byte
    /// equality asserted every run — oracle and instrument in one (the
    /// virtual_b pattern; in-process alternation is the only way to
    /// resolve a few-ms effect on this box).
    #[test]
    #[ignore] // Benchmark + oracle at real scale — run with --nocapture.
    fn virtual_a_node_shape_bench() {
        use std::sync::atomic::Ordering;
        let runs: usize = std::env::var("MICRO_RUNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        // The spine arm's measured height histogram (top runs verbatim, a
        // filler class for the tail) — 687 used columns, 0 full-height.
        let mut heights: Vec<u64> = Vec::new();
        for &(h, c) in &[
            (18640u64, 93usize),
            (12933, 5),
            (6052, 12),
            (5944, 21),
            (2706, 26),
            (2582, 29),
        ] {
            heights.extend(std::iter::repeat_n(h, c));
        }
        while heights.len() < 687 {
            heights.push(341);
        }
        heights.resize(1 << 11, 0);
        let params = JaggedParams::from_heights(&heights, 15, 22);
        let mut ch = RandomChallenger::new(0xA11A_BE4C);
        let z1r = sample_vec(&mut ch, 15);
        let z1c = sample_vec(&mut ch, 11);
        let z2r = sample_vec(&mut ch, 15);
        let z2c = sample_vec(&mut ch, 11);
        let c1 = sample_vec(&mut ch, 128);
        let c2 = sample_vec(&mut ch, 128);
        let g1r = sample_vec(&mut ch, 15);
        let mut g1cols = sample_vec(&mut ch, 1 << 11);
        let g2r = sample_vec(&mut ch, 15);
        let mut g2cols = sample_vec(&mut ch, 1 << 11);
        // Gather-shaped groups, like the real node's: zero the weights on
        // 84 of the 93 tall columns so the union support is ~745k words —
        // the spine arm reads "group sparse, support 752464". Fully dense
        // cols would trip the densify gate and skip the sparse pair.
        for y in 0..84 {
            g1cols[y] = F128::ZERO;
            g2cols[y] = F128::ZERO;
        }
        let rho = sample_vec(&mut ch, 22);
        let claims = [
            FrobeniusClaim {
                z_row: &z1r,
                z_col: &z1c,
                coeffs: &c1,
            },
            FrobeniusClaim {
                z_row: &z2r,
                z_col: &z2c,
                coeffs: &c2,
            },
        ];
        let groups = [
            ScalarGroupClaim {
                z_row: &g1r,
                cols: &g1cols,
            },
            ScalarGroupClaim {
                z_row: &g2r,
                cols: &g2cols,
            },
        ];
        let mut times: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut reference: Option<MultipointTwistedProof> = None;
        for _ in 0..runs {
            for (slot, force) in [(0usize, 2u8), (1, 1)] {
                VIRTUAL_A_OVERRIDE.store(force, Ordering::Relaxed);
                let t = std::time::Instant::now();
                let mut chp = FsChallenger::new(b"virtual-a-bench");
                let proof = prove_multipoint_twisted(&params, &claims, &groups, &rho, &mut chp);
                times[slot].push(t.elapsed().as_secs_f64() * 1e3);
                match &reference {
                    None => reference = Some(proof),
                    Some(rf) => assert_eq!(*rf, proof, "arm divergence (force={force})"),
                }
            }
        }
        VIRTUAL_A_OVERRIDE.store(0, Ordering::Relaxed);
        for (name, ts) in ["dense", "lazy "].iter().zip(times.iter_mut()) {
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ts[ts.len() / 2];
            println!(
                "virtual_a node-shape {name}: median {med:6.2} ms  (runs {:?})",
                ts.iter()
                    .map(|t| (t * 100.0).round() / 100.0)
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Full-count shapes — every used column FULL (height `2^n`) — drive
    /// the `AlignedRsPair` closed-form path (the sibling tests' random
    /// heights keep the materialized fallback covered). Patterns: all
    /// columns full, a used-prefix with a dead tail, and scattered dead
    /// columns; `n = 1` exercises the immediate-densify degenerate.
    #[test]
    fn multipoint_twisted_aligned_columns_matches_bruteforce() {
        let mut ch = RandomChallenger::new(0xA119_ED01);
        let cases: &[(usize, usize, usize, &[usize])] = &[
            (3, 2, 5, &[0, 1, 2, 3]),        // all 4 columns full: area == 2^m
            (4, 3, 7, &[0, 1, 2, 3, 4]),     // used prefix, dead tail
            (3, 4, 8, &[0, 2, 3, 7, 9, 12]), // scattered dead columns
            (1, 3, 4, &[1, 4, 6]),           // n = 1: densify at round 0
        ];
        for &(n, k, m, used) in cases {
            let mut heights = vec![0u64; 1 << k];
            for &y in used {
                heights[y] = 1 << n;
            }
            let params = JaggedParams::from_heights(&heights, n, m);
            assert!(
                aligned_full_columns(&params).is_some(),
                "case must drive the aligned path"
            );
            check_multipoint(
                &params,
                &mut ch,
                &format!("aligned n={n} k={k} m={m} used={}", used.len()),
            );
        }
    }

    /// Strided shapes — long runs of equal NON-power-of-two heights, the
    /// low-table path of [`assist_g_values`] — plus run boundaries, zero
    /// heights, and a power-of-two run, all against the same brute force.
    #[test]
    fn multipoint_twisted_strided_matches_bruteforce() {
        let mut ch = RandomChallenger::new(0x4D50_57F1);
        // 24 columns of height 13, 10 of height 5, one zero column, the
        // rest empty (n = 4, m = 9: area 362 < 512).
        let mut heights = vec![0u64; 64];
        for h in heights.iter_mut().take(24) {
            *h = 13;
        }
        for h in heights[24..34].iter_mut() {
            *h = 5;
        }
        let params = JaggedParams::from_heights(&heights, 4, 9);
        check_multipoint(&params, &mut ch, "strided 24x13 + 10x5");

        // A long odd-stride run: 60 columns of height 6 (n = 3, m = 9).
        let mut heights = vec![0u64; 64];
        for h in heights.iter_mut().take(60) {
            *h = 6;
        }
        let params = JaggedParams::from_heights(&heights, 3, 9);
        check_multipoint(&params, &mut ch, "strided 60x6");

        // Power-of-two run at full utilization (n = 3, m = 9: 64·8 = 512).
        let heights = vec![8u64; 64];
        let params = JaggedParams::from_heights(&heights, 3, 9);
        check_multipoint(&params, &mut ch, "strided 64x8 full");
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

    /// Shapes whose block tree genuinely compresses: the registry's own form
    /// (runs of `k_t` consecutive columns of height `n_t`, zero gaps between
    /// type regions), at power-of-two and odd strides, plus the degenerate ends.
    /// `(heights, n, m)`.
    fn blocked_shapes() -> Vec<(Vec<u64>, usize, usize)> {
        vec![
            // One long uniform run — the deepest compression.
            (vec![5u64; 16], 3, 8),
            // Odd stride, no power-of-two alignment anywhere.
            (vec![13u64; 8], 4, 8),
            // Two type regions of different heights, separated by a zero gap.
            (
                [vec![6u64; 5], vec![0; 3], vec![3; 6], vec![0; 2]].concat(),
                3,
                7,
            ),
            // Height 1: every column is its own block at every layer.
            (vec![1u64; 8], 1, 4),
            // Zero-height tail only, and a single non-empty column.
            (vec![0u64, 0, 0, 7], 3, 4),
            // Heights straddling a power of two (carries into the high bits).
            (vec![7u64; 8], 3, 6),
        ]
    }

    #[test]
    fn blocked_tree_invariants() {
        // The tree the collapse rests on: layer 0 one block per deduped
        // column, layer m+1 a single block, parents non-decreasing and never
        // ahead of the child (what makes `fold_partials` safe in place), and
        // every block genuinely constant in the bits it claims.
        for (heights, n, m) in blocked_shapes() {
            let params = JaggedParams::from_heights(&heights, n, m);
            let bounds = assist_boundaries(&params);
            let blocks = AssistBlocks::new(&bounds, m);
            assert_eq!(blocks.n_blocks(0), bounds.len(), "layer 0 is per-column");
            assert_eq!(blocks.n_blocks(m + 1), 1, "layer m+1 is one block");
            assert_eq!(blocks.total(), blocks.off[m + 2]);
            for layer in 0..=m {
                let (par, cd) = (&blocks.parent[layer], &blocks.cd[layer]);
                assert_eq!(par.len(), blocks.n_blocks(layer));
                let mut last = 0u32;
                for (b, &p) in par.iter().enumerate() {
                    assert!(p <= b as u32, "parent must not run ahead of the child");
                    assert!(p == last || p == last + 1, "parents must be a run index");
                    last = p;
                }
                // Each block is constant in bits ≥ layer, hence in `cd`.
                let starts = &blocks.starts[layer];
                for (b, &s) in starts.iter().enumerate() {
                    let end = starts.get(b + 1).map_or(bounds.len(), |&x| x as usize);
                    let (t_c, t_next, _) = bounds[s as usize];
                    let want = ((t_c >> layer) & 1) as u8 + 2 * (((t_next >> layer) & 1) as u8);
                    assert_eq!(cd[b], want);
                    for &(c, d, _) in &bounds[s as usize..end] {
                        assert_eq!((c >> layer, d >> layer), (t_c >> layer, t_next >> layer));
                    }
                }
            }
        }
    }

    #[test]
    fn blocked_suffix_rows_match_dense() {
        // Every column's dense suffix vector equals the vector stored for the
        // block containing it, at every layer — exactly, both dispatches.
        let mut ch = RandomChallenger::new(0x51F5_B10C);
        let sparse = assist_sparse_transitions();
        for (heights, n, m) in blocked_shapes() {
            let params = JaggedParams::from_heights(&heights, n, m);
            let bounds = assist_boundaries(&params);
            let blocks = AssistBlocks::new(&bounds, m);
            let cols = assist_columns_at(&bounds, &sample_vec(&mut ch, params.k));
            let eq4s: Vec<[F128; 4]> = (0..=m)
                .map(|_| {
                    let v = sample_vec(&mut ch, 2);
                    let t = build_eq_table(&v);
                    [t[0], t[1], t[2], t[3]]
                })
                .collect();
            let dense = assist_suffix_rows(&cols, &eq4s, &sparse, m);
            for par in [false, true] {
                let blk = assist_suffix_rows_blocked(&blocks, &eq4s, &sparse, m, par);
                for layer in 0..=m + 1 {
                    let starts = &blocks.starts[layer];
                    for (b, &s) in starts.iter().enumerate() {
                        let end = starts.get(b + 1).map_or(cols.len(), |&x| x as usize);
                        for y in s as usize..end {
                            assert_eq!(
                                blk[blocks.off[layer] + b],
                                dense[layer * cols.len() + y],
                                "layer {layer} column {y} (par={par}) heights {heights:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The shared-tail split of the blocked store must be slot-for-slot the
    /// monolithic build: layers `≥ lo` from [`assist_shared_tail_blocked`]
    /// (which reconstructs the statement-independent eq tables from `rho`
    /// alone), layers `< lo` from [`assist_suffix_low_blocked`] reading its
    /// boundary parents out of that tail. This is the load-bearing check for
    /// sharing the tail across a Frobenius batch: if the zero-padding
    /// argument were off by one layer, the prover would silently emit a
    /// wrong proof.
    #[test]
    fn blocked_low_plus_tail_matches_full() {
        let mut ch = RandomChallenger::new(0x5A11_7A1B);
        let sparse = assist_sparse_transitions();
        for (heights, n, m) in blocked_shapes() {
            let params = JaggedParams::from_heights(&heights, n, m);
            let bounds = assist_boundaries(&params);
            let blocks = AssistBlocks::new(&bounds, m);
            // eq tables exactly as `frobenius_statements` builds them: the
            // statement's `z_row` below its length, zero-padded above.
            let zr = sample_vec(&mut ch, n);
            let rho = sample_vec(&mut ch, m);
            let eq4s: Vec<[F128; 4]> = (0..=m)
                .map(|layer| {
                    let t = build_eq_table(&[point_bit(&zr, layer), point_bit(&rho, layer)]);
                    [t[0], t[1], t[2], t[3]]
                })
                .collect();
            let full = assist_suffix_rows_blocked(&blocks, &eq4s, &sparse, m, false);
            let lo = n.clamp(1, m + 1);
            let tail = assist_shared_tail_blocked(&blocks, &rho, &sparse, m, lo);
            assert_eq!(
                tail[..],
                full[blocks.off[lo]..],
                "shared tail (lo={lo}) heights {heights:?}"
            );
            for par in [false, true] {
                let low = assist_suffix_low_blocked(&blocks, &eq4s, &sparse, lo, &tail, par);
                assert_eq!(
                    low[..],
                    full[..blocks.off[lo]],
                    "low layers (lo={lo}, par={par}) heights {heights:?}"
                );
            }
        }
    }

    /// The hoisted per-column eq vector (one tree descent, shared by every
    /// statement) dotted with a statement's weights must equal the
    /// per-statement ascent [`assist_w_at_blocked`] — bit-identical, the
    /// same field products reassociated.
    #[test]
    fn hoisted_blocked_eq_matches_w_at() {
        let mut ch = RandomChallenger::new(0x0157_E97A);
        for (heights, n, m) in blocked_shapes() {
            let params = JaggedParams::from_heights(&heights, n, m);
            let bounds = assist_boundaries(&params);
            let blocks = AssistBlocks::new(&bounds, m);
            let cols = assist_columns_at(&bounds, &sample_vec(&mut ch, params.k));
            let sigma = sample_vec(&mut ch, 2 * (m + 1));
            let eq_cols = assist_eq_at_blocked(&blocks, &sigma, m);
            assert_eq!(eq_cols.len(), cols.len(), "heights {heights:?}");
            let dot = cols
                .iter()
                .zip(&eq_cols)
                .fold(F128::ZERO, |acc, (&(w, _, _), &e)| acc + w * e);
            assert_eq!(
                dot,
                assist_w_at_blocked(&blocks, &cols, &sigma, m),
                "heights {heights:?}"
            );
        }
    }

    #[test]
    fn blocked_layer_state_matches_dense() {
        // The two block-scale kernels composed, driven layer by layer with the
        // same challenges as the real prover: `assist_buckets` against the
        // dense per-column bucketing, and `fold_partials` against the dense
        // running-weight fold. This is the state pipeline the Frobenius batch
        // runs (`prove_assist` is separately pinned bit-for-bit against
        // `prove_assist_naive`, which shares no code with either).
        let mut ch = RandomChallenger::new(0x810C_57A7);
        let sparse = assist_sparse_transitions();
        for (heights, n, m) in blocked_shapes() {
            let params = JaggedParams::from_heights(&heights, n, m);
            let bounds = assist_boundaries(&params);
            let blocks = AssistBlocks::new(&bounds, m);
            let cols = assist_columns_at(&bounds, &sample_vec(&mut ch, params.k));
            let eq4s: Vec<[F128; 4]> = (0..=m)
                .map(|_| {
                    let v = sample_vec(&mut ch, 2);
                    let t = build_eq_table(&v);
                    [t[0], t[1], t[2], t[3]]
                })
                .collect();
            let n_cols = cols.len();
            let dense_sfx = assist_suffix_rows(&cols, &eq4s, &sparse, m);
            let blk_sfx = assist_suffix_rows_blocked(&blocks, &eq4s, &sparse, m, false);

            let mut we: Vec<F128> = cols.iter().map(|&(w, _, _)| w).collect();
            let mut p = blocks.seed(&cols);
            // Alternate the fold's dispatch across layers so both paths run.
            let mut scratch = Vec::new();
            for layer in 0..=m {
                let row = &dense_sfx[(layer + 1) * n_cols..(layer + 2) * n_cols];
                let mut want = [[F128::ZERO; 4]; 4];
                for ((&w_e, &(_, t_c, t_next)), s) in we.iter().zip(&cols).zip(row) {
                    let cd = ((t_c >> layer) & 1) as usize + 2 * ((t_next >> layer) & 1) as usize;
                    let bk = &mut want[cd];
                    for (slot, &sv) in bk.iter_mut().zip(s) {
                        *slot += w_e * sv;
                    }
                }
                for par in [false, true] {
                    assert_eq!(
                        assist_buckets(&p, &blk_sfx, &[], usize::MAX, &blocks, layer, par),
                        want,
                        "buckets at layer {layer} (par={par}) heights {heights:?}"
                    );
                }
                // Advance both by the same challenge pair.
                let (rc, rd) = (ch.sample_f128(), ch.sample_f128());
                let (rc1, rd1) = (F128::ONE + rc, F128::ONE + rd);
                let ch4 = [rc1 * rd1, rc * rd1, rc1 * rd, rc * rd];
                for (w_e, &(_, t_c, t_next)) in we.iter_mut().zip(&cols) {
                    let cd = ((t_c >> layer) & 1) as usize + 2 * ((t_next >> layer) & 1) as usize;
                    *w_e *= ch4[cd];
                }
                fold_partials(&mut p, &mut scratch, &blocks, layer, &ch4, layer % 2 == 0);
                // Each block's partial is the exact sum of its columns'.
                let starts = &blocks.starts[layer + 1];
                for (b, &s) in starts.iter().enumerate() {
                    let end = starts.get(b + 1).map_or(n_cols, |&x| x as usize);
                    let want = we[s as usize..end]
                        .iter()
                        .fold(F128::ZERO, |acc, &x| acc + x);
                    assert_eq!(p[b], want, "partial at layer {} block {b}", layer + 1);
                }
            }
        }
    }

    #[test]
    fn blocked_w_at_matches_dense() {
        // The verifier's W(σ) walk: the tree ascent equals the per-column
        // product form exactly (reassociation of the same field product).
        let mut ch = RandomChallenger::new(0x0C57_A7E4);
        for (heights, n, m) in blocked_shapes() {
            let params = JaggedParams::from_heights(&heights, n, m);
            let bounds = assist_boundaries(&params);
            let blocks = AssistBlocks::new(&bounds, m);
            for _ in 0..4 {
                let cols = assist_columns_at(&bounds, &sample_vec(&mut ch, params.k));
                let sigma = sample_vec(&mut ch, 2 * (m + 1));
                assert_eq!(
                    assist_w_at_blocked(&blocks, &cols, &sigma, m),
                    assist_w_at(&cols, &sigma, m),
                    "W(σ) mismatch for heights {heights:?}"
                );
            }
        }
    }

    #[test]
    fn assist_streamed_matches_naive_on_blocked_shapes() {
        // The bit-identity check of `assist_streamed_matches_naive`, on the
        // shapes where the block tree actually collapses layers — the naive
        // prover shares no code with the blocked one.
        let mut ch = RandomChallenger::new(0xB10C_4E46);
        for (heights, n, m) in blocked_shapes() {
            let params = JaggedParams::from_heights(&heights, n, m);
            let z_row = sample_vec(&mut ch, n);
            let z_col = sample_vec(&mut ch, params.k);
            let z_idx = sample_vec(&mut ch, m);

            let mut ch_a = FsChallenger::new(b"flock-jagged-assist-test");
            let streamed = prove_assist(&params, &z_row, &z_col, &z_idx, &mut ch_a);
            let mut ch_b = FsChallenger::new(b"flock-jagged-assist-test");
            let naive = prove_assist_naive(&params, &z_row, &z_col, &z_idx, &mut ch_b);
            assert_eq!(streamed.beta, naive.beta, "β mismatch heights {heights:?}");
            assert_eq!(
                streamed.rounds, naive.rounds,
                "rounds mismatch heights {heights:?}"
            );
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
                assert_eq!(
                    streamed.rounds, naive.rounds,
                    "rounds mismatch n={n} k={k} m={m}"
                );
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
                let claim =
                    verify_with_assist(&params, &z_row, &z_col, v, &proof, &assist, &mut vch)
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
        assert!(
            check(&proof, &bad).is_none(),
            "tampered round must be rejected"
        );

        // Truncated assist.
        let mut bad = assist.clone();
        bad.rounds.pop();
        assert!(
            check(&proof, &bad).is_none(),
            "truncated assist must be rejected"
        );

        // Tampered dense claim must break the outer relation against β.
        let mut bad_proof = proof.clone();
        bad_proof.q_eval += F128::ONE;
        assert!(
            check(&bad_proof, &assist).is_none(),
            "tampered q_eval must be rejected"
        );
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

            // Parallel fused helper (the production path; also emits round 1's
            // message, so it does slightly more work than the serial baseline).
            let t1 = Instant::now();
            let (bp, vp, _g1, _gi) = generate_f_and_claim(&params, &q, &z_row, &z_col);
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

        let best3 = |f: &mut dyn FnMut() -> std::time::Duration| (0..3).map(|_| f()).min().unwrap();

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
        eprintln!(
            "  witness: 2^{m} F128 = {} MiB (2^30 bits packed)",
            (len * 16) >> 20
        );
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

    /// The table-based square root equals 127 squarings on random inputs —
    /// the tower's walkers used the squaring form until 2026-08-29.
    #[test]
    fn frob_inv_matches_127_squarings() {
        let mut rng = crate::test_rng::Rng::new(0xF0B_1A5);
        for _ in 0..256 {
            let x = rng.f128();
            let mut y = x;
            for _ in 0..127 {
                y = y * y;
            }
            assert_eq!(frob_inv(x), y);
            assert_eq!(frob_inv(x) * frob_inv(x), x, "sqrt squares back");
        }
    }
}
