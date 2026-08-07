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
use crate::pcs::ring_switch::fold_one_slot;
use crate::scratch::take_f128;
use serde::{Deserialize, Serialize};
use std::env::var;
#[cfg(test)]
use std::hint::black_box;
#[cfg(test)]
use std::mem::size_of;
use std::mem::swap;
use std::sync::OnceLock;
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

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
    // ~1 MB chunks: one binary search amortized over 64K elements. CHUNK is
    // even and len is a power of two ≥ 2, so message pairs never straddle
    // chunks.
    const CHUNK: usize = 1 << 16;
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
            for (bp, qp) in b_chunk.chunks_exact(2).zip(q_chunk.chunks_exact(2)) {
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
    // just-in-time basis ([`JaggedWeight`]) competitive: it is called twice per
    // element position across the round-0 message and fold, so per-element
    // branching there costs double.
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

/// [`generate_f_and_claim`] whose round-0 message is taken over the BLOCK
/// pairing of block size `d` — output block `c` pairs input blocks `2c` and
/// `2c+1` — which is what L0 binds when the committed stack is lane-major
/// (`ligerito::fold_and_msg_blocked`).
///
/// `d` exceeds the element-paired driver's chunk at production sizes
/// (`2^17` vs `2^16` at `M = 30`), so a chunk can never hold both partners.
/// Each task therefore owns a sub-range of one block PAIR — the two halves
/// carved by `split_at_mut` — which keeps the prime fused into the weight
/// build instead of costing a second 268 MB pass over `(q, W)`.
#[cfg(test)]
fn generate_f_and_claim_blocked(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
    d: usize,
) -> (Vec<F128>, F128, F128, F128) {
    // Sub-range within a block pair; keeps the task count high when there are
    // few pairs (32 pairs × 8 sub-ranges at M = 30).
    const SUB: usize = 1 << 14;
    use rayon::prelude::*;
    let len = 1usize << params.m;
    assert!(d >= 1 && d <= len / 2 && d.is_power_of_two());
    let area = params.area() as usize;
    let eq_row = build_eq_table(z_row);
    let eq_col = build_eq_table(z_col);
    let prefix = &params.col_prefix_sums;
    let mut b = crate::alloc_uninit_f128_vec(len);

    let (v, g_one, g_inf) = b
        .par_chunks_mut(2 * d)
        .enumerate()
        .map(|(c, b_pair)| {
            let (lo, hi) = b_pair.split_at_mut(d);
            let base = 2 * c * d;
            lo.par_chunks_mut(SUB)
                .zip(hi.par_chunks_mut(SUB))
                .enumerate()
                .map(|(si, (lo_c, hi_c))| {
                    let o = si * SUB;
                    let n = lo_c.len();
                    fill_weight_range(lo_c, base + o, area, prefix, &eq_row, &eq_col);
                    fill_weight_range(hi_c, base + d + o, area, prefix, &eq_row, &eq_col);
                    let q_lo = &q[base + o..base + o + n];
                    let q_hi = &q[base + d + o..base + d + o + n];
                    let mut acc = F128::ZERO;
                    let mut m_one = F128::ZERO;
                    let mut m_inf = F128::ZERO;
                    for i in 0..n {
                        let t = q_hi[i] * hi_c[i];
                        acc += q_lo[i] * lo_c[i] + t;
                        m_one += t;
                        m_inf += (q_lo[i] + q_hi[i]) * (lo_c[i] + hi_c[i]);
                    }
                    (acc, m_one, m_inf)
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO, F128::ZERO),
                    |(a, b1, c), (d, e, f)| (a + d, b1 + e, c + f),
                )
        })
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a, b1, c), (d, e, f)| (a + d, b1 + e, c + f),
        );
    (b, v, g_one, g_inf)
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
/// fused opening path (`the removed jagged open` discharges the
/// weight-table inner product directly in Ligerito, with no jagged main
/// sumcheck).
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
        swap(&mut a, &mut sa);
        swap(&mut bb, &mut sb);
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
    swap(p, out);
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
    enum ColSide<'a> {
        Fold(Vec<F128>, &'a [F128]),
        Combined(&'a [F128]),
    }
    // Segmented fill (the JaggedWeight lesson): per chunk, ONE cursor into
    // `col_prefix_sums`, then per column segment a claim-OUTER sweep — the
    // column factor hoisted, rows read sequentially, and one claim's 64 KB
    // fold table hot per sweep. The per-element unrank variant measured
    // ~2.5x slower at M = 30.
    // The merged sumcheck's round-0 prime `(u0, u2)` is fused into the same
    // pass (CHUNK is even, so element pairs never straddle chunks); the
    // dead tail past the area contributes zero on both sides.
    const CHUNK: usize = 1 << 14;
    use rayon::prelude::*;
    let area = params.area() as usize;
    let n_total = 1usize << params.m;

    let tabs: Vec<(Vec<F128>, ColSide<'_>)> = claims
        .iter()
        .map(|c| match c {
            MergedWeightClaim::Folded {
                z_row,
                z_col,
                table,
            } => (
                build_eq_table(z_row),
                ColSide::Fold(build_eq_table(z_col), *table),
            ),
            MergedWeightClaim::Scalar { z_row, cols } => {
                (build_eq_table(z_row), ColSide::Combined(cols))
            }
        })
        .collect();
    assert_eq!(q.len(), n_total);
    let mut w = take_f128(n_total);

    let ps = &params.col_prefix_sums;
    let prime = w
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(ci, out)| {
            let base = (ci * CHUNK) as u64;
            let end = base + out.len() as u64;
            if base >= area as u64 {
                out.fill(F128::ZERO);
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
                                    *slot = fold_one_slot(r * c_hoist, tab);
                                }
                            } else {
                                for (slot, &r) in dst.iter_mut().zip(rows) {
                                    *slot += fold_one_slot(r * c_hoist, tab);
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
    enum SpecCols<'b> {
        Point(Vec<F128>),
        Weights(&'b [F128]),
    }
    use rayon::prelude::*;
    let m = params.m;
    let sparse = assist_sparse_transitions();

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
    use rayon::prelude::*;
    let m = params.m;
    assert_eq!(rho.len(), m);
    let trace = var("PCS_TRACE").is_ok();
    let t = Instant::now();
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
    let t = Instant::now();

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
        let rc = challenger.sample_f128();
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
        let rd = challenger.sample_f128();
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
    FrobeniusAssistProof { v, rounds }
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
    use rayon::prelude::*;
    let m = params.m;
    if proof.rounds.len() != 2 * (m + 1) {
        return None;
    }
    // `VERIFY_TRACE` sub-split of the assist — it dominates the Merkle-table
    // verify, so its three phases are worth separating: the transcript replay,
    // building the 128·K statements (a `2^k`-column `eq` table each), and the
    // per-statement `W(σ)` walk + boundary DP.
    let trace = var("VERIFY_TRACE").is_ok();
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

    let t = Instant::now();
    let mut claim = proof.v;
    let mut sigma = Vec::with_capacity(2 * (m + 1));
    for &(g_one, g_inf) in &proof.rounds {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
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
    let t = Instant::now();
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
    let t = Instant::now();
    let eq_cols = assist_eq_at_blocked(&blocks, &sigma, m);
    if trace {
        eprintln!(
            "          [fro-v] eq descent ({} blocks, once for {} statements): {}",
            blocks.total(),
            sts.len(),
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    let t = Instant::now();
    let expect = sts
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
            w * g[STATE_INITIAL]
        })
        .reduce(|| F128::ZERO, |a, b| a + b);
    if trace {
        eprintln!(
            "          [fro-v] per-statement dot + boundary DP (x{}): {}",
            sts.len(),
            tfmt(t.elapsed().as_secs_f64())
        );
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
    static T: OnceLock<[F128; 128]> = OnceLock::new();
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

/// `x^{2^{-1}}` via the basis-image table.
fn frob_inv(x: F128) -> F128 {
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
    static T: OnceLock<Vec<[F128; 128]>> = OnceLock::new();
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
            let low = v & v.wrapping_neg();
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
    partner: &[F128],
) -> (Vec<F128>, (F128, F128)) {
    const CH: usize = 1 << 14;
    use rayon::prelude::*;
    let mut a = vec![F128::ZERO; 1usize << params.m];
    let pfx = &params.col_prefix_sums;
    let n_cols = pfx.len() - 1;

    let msg = a
        .par_chunks_mut(CH)
        .zip(partner.par_chunks(CH))
        .enumerate()
        .map(|(ci, (chunk, gc))| {
            let start = (ci * CH) as u64;
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
            for (ap, gp) in chunk.chunks_exact(2).zip(gc.chunks_exact(2)) {
                p1 += ap[1] * gp[1];
                pi += (ap[0] + ap[1]) * (gp[0] + gp[1]);
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
    pub rounds: Vec<(F128, F128)>,
    pub anchor: FrobeniusAssistProof,
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
    use rayon::prelude::*;
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
    let trace = var("PCS_TRACE").is_ok();

    let t = Instant::now();
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
    let gamma = challenger.sample_f128();
    let mut gpow = Vec::with_capacity(128 * n_rs + n_g);
    let mut p = F128::ONE;
    for _ in 0..128 * n_rs + n_g {
        gpow.push(p);
        p *= gamma;
    }

    // The dense vectors: e = eq(ρ,·) (the groups' partner), g = L_γ(e) via
    // one byte-table pass (the RS partner), and the two combined weights.
    let t = Instant::now();
    let eq_rho = super::ring_switch::build_eq_parallel(rho);
    let mut pairs: Vec<ProductPair> = Vec::with_capacity(2);
    let mut msg0 = (F128::ZERO, F128::ZERO);
    if n_rs > 0 {
        let basis = inv_frob_basis();
        let mut images = [F128::ZERO; 128];
        for j in 0..128 {
            for (b, img) in images.iter_mut().enumerate() {
                *img += gpow[j] * basis[j][b];
            }
        }
        let tables = linear_byte_tables(&images);
        let mut gv = vec![F128::ZERO; 1usize << m];
        gv.par_iter_mut()
            .zip(eq_rho.par_iter())
            .for_each(|(g, &e)| *g = apply_linear_tables(&tables, e));
        let eq_cs: Vec<Vec<F128>> = claims.iter().map(|c| build_eq_table(c.z_col)).collect();
        let sides: Vec<(F128, Vec<F128>, &[F128])> = claims
            .iter()
            .zip(&eq_cs)
            .enumerate()
            .map(|(i, (c, eq_c))| (gpow[128 * i], build_eq_table(c.z_row), eq_c.as_slice()))
            .collect();
        let (av, msg) = build_combined_weight_and_msg(params, &sides, &gv);
        msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
        pairs.push(ProductPair::new(av, gv));
    }
    if n_g > 0 {
        let sides: Vec<(F128, Vec<F128>, &[F128])> = groups
            .iter()
            .enumerate()
            .map(|(k, g)| (gpow[128 * n_rs + k], build_eq_table(g.z_row), g.cols))
            .collect();
        let (bv, msg) = build_combined_weight_and_msg(params, &sides, &eq_rho);
        msg0 = (msg0.0 + msg.0, msg0.1 + msg.1);
        pairs.push(ProductPair::new(bv, eq_rho));
    }
    if trace {
        eprintln!(
            "    [multipoint] dense weight passes (2^{m}, {} products, round-0 fused): {:6.2} ms",
            pairs.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // The m-round two-product sumcheck for Σ_d (ā_d·g_d + b̄_d·e_d), low bit
    // first: round-0 messages from the full vectors, later messages fused
    // into the folds ([`fold_and_round_oop_par`], ping-pong scratch halves)
    // and summed across the active products. The final fold never runs —
    // nothing reads the folded scalars; the anchor assist reproves the
    // endpoint.
    let t = Instant::now();
    let mut rounds = Vec::with_capacity(m);
    let mut point = Vec::with_capacity(m);
    let (mut g_one, mut g_inf) = msg0;
    let mut cur = 1usize << m;
    for i in 0..m {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        rounds.push((g_one, g_inf));
        point.push(r);
        if i + 1 == m {
            break;
        }
        let mut nxt = (F128::ZERO, F128::ZERO);
        for pair in pairs.iter_mut() {
            let msg = pair.fold_round(cur, r);
            nxt = (nxt.0 + msg.0, nxt.1 + msg.1);
        }
        (g_one, g_inf) = nxt;
        cur /= 2;
    }
    if trace {
        eprintln!(
            "    [multipoint] two-product sumcheck ({m} rounds): {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // Anchor: ONE untwisted batched assist binding the whole endpoint sum
    // `ĝ(ρ'')·ā̂(ρ'') + eq(ρ,ρ'')·b̂(ρ'')` — the closed-form factors are
    // baked into the coefficients (RS claim i: γ^{128 i}·ĝ(ρ''); group k:
    // γ^{128 R + k}·eq(ρ,ρ'')), so the accept check is a plain equality
    // against the running claim and no extra scalar travels.
    let t = Instant::now();
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
    let anchor = prove_frobenius_assist(params, &anchor_claims, &anchor_groups, &point, challenger);
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
        rounds,
        anchor,
    }
}

/// One product's ping-pong fold state for the two-product sumcheck: the
/// full-size pair, half-size scratch, and which half currently holds the
/// live data.
struct ProductPair {
    x: Vec<F128>,
    y: Vec<F128>,
    sx: Vec<F128>,
    sy: Vec<F128>,
    flip: bool,
}

impl ProductPair {
    fn new(x: Vec<F128>, y: Vec<F128>) -> Self {
        debug_assert_eq!(x.len(), y.len());
        let half = x.len() / 2;
        Self {
            sx: vec![F128::ZERO; half],
            sy: vec![F128::ZERO; half],
            x,
            y,
            flip: false,
        }
    }

    /// Fold the live pair of length `cur` at `r` into the other half and
    /// return the next round's message for this product.
    fn fold_round(&mut self, cur: usize, r: F128) -> (F128, F128) {
        let half = cur / 2;
        let msg = if self.flip {
            fold_and_round_oop_par(
                &self.sx[..cur],
                &self.sy[..cur],
                r,
                &mut self.x[..half],
                &mut self.y[..half],
            )
        } else {
            fold_and_round_oop_par(
                &self.x[..cur],
                &self.y[..cur],
                r,
                &mut self.sx[..half],
                &mut self.sy[..half],
            )
        };
        self.flip = !self.flip;
        msg
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
    let gamma = challenger.sample_f128();
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
    for &(g_one, g_inf) in &proof.rounds {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
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
    let s_at = verify_frobenius_assist(
        params,
        &anchor_claims,
        &anchor_groups,
        &point,
        &proof.anchor,
        challenger,
    )?;
    if running != s_at {
        return None;
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
/// ~2.6× to ~6× parallel scaling (hits the memory-bandwidth ceiling). See
/// `scaling_diag`. No longer on the production path (round 1's message is
/// fused into [`generate_f_and_claim`]); retained for the runtime benchmarks.
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
/// Shared with the virtual-opening sumcheck (`the removed jagged open`),
/// which runs the same product-sumcheck round structure.
pub(crate) fn fold_and_round_oop_par(
    a: &[F128],
    b: &[F128],
    r: F128,
    ao: &mut [F128],
    bo: &mut [F128],
) -> (F128, F128) {
    // Output chunk of `CO`; the aligned input chunk is `2*CO` (output is half
    // the input). Slice/`chunks_exact` iteration — no per-element bounds checks —
    // so the reduction scales like the fold (~6× vs ~2.6× for indexed access).
    const CO: usize = 1 << 13;
    use rayon::prelude::*;
    debug_assert_eq!(a.len(), 2 * ao.len());
    debug_assert!(a.len() >= 4);

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

    /// Timing probe for the batched Frobenius assist prover at the m30
    /// merged shape: n = 14, k = 9 (368 live columns at full height),
    /// m = 23, two claims × 128 coefficients = 256 statements.
    /// Informational — prints the min-of-N prove time.
    #[test]
    #[ignore] // Timing probe — run explicitly with --ignored --nocapture
    fn frobenius_assist_bench() {
        let mut ch = RandomChallenger::new(0xBE4C_0011);
        let (n, k, m) = (14usize, 9usize, 23usize);
        let mut heights = vec![0u64; 1 << k];
        for h in heights.iter_mut().take(368) {
            *h = 1 << n;
        }
        let params = JaggedParams::from_heights(&heights, n, m);
        let z_row_a = sample_vec(&mut ch, n);
        let z_col_a = sample_vec(&mut ch, k);
        let coeffs_a = sample_vec(&mut ch, 128);
        let z_row_b = sample_vec(&mut ch, n);
        let z_col_b = sample_vec(&mut ch, k);
        let coeffs_b = sample_vec(&mut ch, 128);
        let claims = [
            FrobeniusClaim {
                z_row: &z_row_a,
                z_col: &z_col_a,
                coeffs: &coeffs_a,
            },
            FrobeniusClaim {
                z_row: &z_row_b,
                z_col: &z_col_b,
                coeffs: &coeffs_b,
            },
        ];
        let rho = sample_vec(&mut ch, m);
        let mut best = f64::INFINITY;
        for _ in 0..12 {
            let mut fs = FsChallenger::new(b"frobenius-assist-bench");
            let t = Instant::now();
            let proof = prove_frobenius_assist(&params, &claims, &[], &rho, &mut fs);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            black_box(proof);
        }
        println!("frobenius assist prove (256 stmts, 368 cols, m = 23): {best:.2} ms (min of 12)");
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

    /// Multipoint twisted evaluation across column counts at m = 23, against
    /// the batched Frobenius assist where the latter fits in memory. The block
    /// tree cut its suffix state by ~1.8x (at 4096 columns, 940 MB across the
    /// 256 statements against 1.7 GB), so 4K is now runnable rather than
    /// hopeless — see the `assist_blocked` probe, which times it at reduced
    /// iterations — but still far too heavy to allocate from a unit test, and
    /// still the losing side at that width (42 ms of assist against this
    /// protocol's 24 ms in total). Informational.
    #[test]
    #[ignore] // Timing probe — run explicitly with --ignored --nocapture
    fn multipoint_twisted_bench() {
        let mut ch = RandomChallenger::new(0x4D50_BE7C);
        let m = 23usize;
        for &(n, k, cols, height, run_assist) in &[
            (14usize, 9usize, 368usize, 1u64 << 14, true),
            (11, 12, 4096, 1 << 11, false),
            (8, 15, 32768, 1 << 8, false),
            // Non-power-of-two stride: the low-table path on an odd-ish run.
            (9, 15, 27962, 300, false),
        ] {
            let mut heights = vec![0u64; 1 << k];
            for h in heights.iter_mut().take(cols) {
                *h = height;
            }
            let params = JaggedParams::from_heights(&heights, n, m);
            let z1r = sample_vec(&mut ch, n);
            let z1c = sample_vec(&mut ch, k);
            let z2r = sample_vec(&mut ch, n);
            let z2c = sample_vec(&mut ch, k);
            let c1 = sample_vec(&mut ch, 128);
            let c2 = sample_vec(&mut ch, 128);
            let rho = sample_vec(&mut ch, m);
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
            let mut best = f64::INFINITY;
            let mut kept = None;
            for _ in 0..3 {
                let mut fs = FsChallenger::new(b"multipoint-bench");
                let t = Instant::now();
                let proof = prove_multipoint_twisted(&params, &claims, &[], &rho, &mut fs);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
                kept = Some(proof);
            }
            let proof = kept.unwrap();
            let t = Instant::now();
            let mut fv = FsChallenger::new(b"multipoint-bench");
            let v = verify_multipoint_twisted(&params, &claims, &[], &rho, &proof, &mut fv)
                .expect("bench proof verifies");
            let verify_ms = t.elapsed().as_secs_f64() * 1e3;
            black_box(v);

            let assist = if run_assist {
                let mut best_a = f64::INFINITY;
                for _ in 0..3 {
                    let mut fa = FsChallenger::new(b"assist-bench");
                    let t = Instant::now();
                    let p = prove_frobenius_assist(&params, &claims, &[], &rho, &mut fa);
                    best_a = best_a.min(t.elapsed().as_secs_f64() * 1e3);
                    black_box(p);
                }
                format!("{best_a:6.2} ms")
            } else {
                "skipped (suffix state in the GBs)".to_string()
            };
            println!(
                "cols = {cols:>6}: multipoint prove {best:7.2} ms  verify {verify_ms:6.2} ms;  \
                 batched assist prove {assist}"
            );
        }
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
        const REPS: usize = 3;
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

        let mb = (len * size_of::<F128>()) as f64 / (1024.0 * 1024.0);
        eprintln!("\n[jagged runtime] m={m} ({len} F128 = {mb:.0} MB), n={n}, k={k}, cols={cols}");

        // --- Phase 1: B-vector + claim generation, serial vs parallel-fused. ---
        let mut t_gen_ser = Duration::MAX;
        let mut t_gen_par = Duration::MAX;
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
            black_box(&bs);

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
        let run_serial = |fused: bool| -> Duration {
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
            black_box(a[0]);
            t.elapsed()
        };

        // Parallel: rayon kernels, ping-pong between two out-of-place buffers.
        let run_par = |fused: bool| -> Duration {
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
                swap(&mut a, &mut sa);
                swap(&mut bb, &mut sb);
                cur = half;
            }
            black_box(a[0]);
            t.elapsed()
        };

        let mut s_unf = Duration::MAX;
        let mut s_fus = Duration::MAX;
        let mut p_unf = Duration::MAX;
        let mut p_fus = Duration::MAX;
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
        black_box(beta);
        let t_ver = t2.elapsed();

        let ratio = |unf: Duration, fus: Duration| unf.as_secs_f64() / fus.as_secs_f64();
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

        let best3 = |f: &mut dyn FnMut() -> Duration| (0..3).map(|_| f()).min().unwrap();

        // Warm-up (thread pool + page faults).
        let mut ch = FsChallenger::new(b"flock-jagged-bits30");
        let _ = prove(&params, &q, &z_row, &z_col, &mut ch);

        // Main jagged sumcheck prover (B-generation + rounds).
        let t_prove = best3(&mut || {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            let t = Instant::now();
            black_box(prove(&params, &q, &z_row, &z_col, &mut ch));
            t.elapsed()
        });

        // Main + assist, and keep one transcript for the verifier runs.
        let mut t_both = Duration::MAX;
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
            black_box(verify(&params, &z_row, &z_col, v, &proof, &mut ch).expect("verify"));
            t.elapsed()
        });

        // Verifier, assist path.
        let t_verify_assist = best3(&mut || {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            let t = Instant::now();
            black_box(
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

    /// Phase breakdown of the jagged sumcheck prover at the 2^30-bit point —
    /// diagnostic companion to `runtime_bits30`. Mirrors `prove_main` with a
    /// timer per phase and implied memory bandwidth (each phase's compulsory
    /// traffic / time) to show how far each sits from the streaming floor.
    ///
    /// `cargo test --release -p flock-core pcs::jagged::tests::bits30_breakdown -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic; run with --release --ignored --nocapture"]
    fn bits30_breakdown() {
        use std::time::Instant;

        let _ = crate::init_perf_thread_pool();
        let (n, k, m) = (11usize, 12usize, 23usize);
        let cols = 1usize << k;
        let height = (1u64 << m) / cols as u64;
        let params = JaggedParams::from_heights(&vec![height; cols], n, m);
        let len = 1usize << m;
        let mut q = vec![F128::ZERO; len];
        for (i, qi) in q.iter_mut().enumerate() {
            *qi = F128 {
                lo: i as u64,
                hi: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            };
        }
        let mut rc = RandomChallenger::new(0x0B17_5301);
        let z_row = sample_vec(&mut rc, n);
        let z_col = sample_vec(&mut rc, k);

        // Warm-up.
        {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            let _ = prove(&params, &q, &z_row, &z_col, &mut ch);
        }

        let mb = |bytes: usize, secs: f64| bytes as f64 / secs / 1e9;
        for trial in 0..3 {
            let mut ch = FsChallenger::new(b"flock-jagged-bits30");
            ch.observe_label(b"flock-jagged-v0");

            let t = Instant::now();
            let (b, _v, mut g_one, mut g_inf) = generate_f_and_claim(&params, &q, &z_row, &z_col);
            let t_gen = t.elapsed();

            let t = Instant::now();
            let mut sa = crate::alloc_uninit_f128_vec(len / 2);
            let mut sb = crate::alloc_uninit_f128_vec(len / 2);
            let mut a = crate::alloc_uninit_f128_vec(len / 4);
            let mut bb = crate::alloc_uninit_f128_vec(len / 4);
            let t_alloc = t.elapsed();

            let mut cur = len;
            let mut per_round = Vec::with_capacity(m);
            for round in 0..m {
                let half = cur / 2;
                ch.observe_f128(g_one);
                ch.observe_f128(g_inf);
                let r = ch.sample_f128();
                let t = Instant::now();
                let (a_src, b_src): (&[F128], &[F128]) =
                    if round == 0 { (&q, &b) } else { (&a, &bb) };
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
                per_round.push(t.elapsed());
                swap(&mut a, &mut sa);
                swap(&mut bb, &mut sb);
                cur = half;
            }
            let t_rounds: Duration = per_round.iter().sum();
            let total = t_gen + t_alloc + t_rounds;

            eprintln!("--- trial {trial}  (total {total:.3?})");
            eprintln!(
                "  gen B + claim + round-1 msg : {t_gen:>9.3?}  ({:>5.1} GB/s of 256 MB r+w)",
                mb(2 * len * 16, t_gen.as_secs_f64())
            );
            eprintln!("  buffer alloc    : {t_alloc:>9.3?}");
            // Fold round j reads 2·cur and writes cur elements (two arrays each).
            let round_bytes = |j: usize| 3 * (len >> j) * 16;
            let tail: Duration = per_round[4..].iter().sum();
            for (j, d) in per_round.iter().take(4).enumerate() {
                eprintln!(
                    "  fold+msg round {:>2}: {d:>8.3?}  ({:>5.1} GB/s of {} MB)",
                    j + 1,
                    mb(round_bytes(j), d.as_secs_f64()),
                    round_bytes(j) >> 20
                );
            }
            eprintln!("  rounds 5..{m}     : {tail:>8.3?}");
        }
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

    /// The two assist entries at ONE shape, so their prover costs are
    /// comparable: the single-statement [`prove_assist`] against the
    /// `128·K`-statement [`prove_frobenius_assist`] that the ring-switched
    /// union path actually runs. The shape is the L0-opening union's
    /// (`k=12` columns, `n_t=218` rows, `m=20`).
    ///
    /// `cargo test --release pcs::jagged::tests::assist_single_vs_frobenius
    /// -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn assist_single_vs_frobenius() {
        use std::time::Instant;

        let _ = crate::init_perf_thread_pool();
        let (n, k, m) = (8usize, 12usize, 20usize);
        let (used, height) = (3709usize, 218u64);
        let mut heights = vec![0u64; 1usize << k];
        for h in &mut heights[..used] {
            *h = height;
        }
        let params = JaggedParams::from_heights(&heights, n, m);

        let mut rc = RandomChallenger::new(0xF20B_1005);
        let z_row = sample_vec(&mut rc, n);
        let z_col = sample_vec(&mut rc, k);
        let z_idx = sample_vec(&mut rc, m);

        // Single statement (no ring switching): one boundary-program walk.
        let reps = 5;
        let mut t_single = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let mut ch = FsChallenger::new(b"flock-assist-cmp");
            let p = prove_assist(&params, &z_row, &z_col, &z_idx, &mut ch);
            black_box(&p);
            t_single = t_single.min(t.elapsed().as_secs_f64());
        }

        // The ring-switched path: K claims, each carrying 128 linearized
        // coefficients, so 128·K statements share one sumcheck.
        let k_claims = 2usize;
        let claims_data: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..k_claims)
            .map(|_| {
                (
                    sample_vec(&mut rc, n),
                    sample_vec(&mut rc, k),
                    sample_vec(&mut rc, 128),
                )
            })
            .collect();
        let claims: Vec<FrobeniusClaim<'_>> = claims_data
            .iter()
            .map(|(zr, zc, c)| FrobeniusClaim {
                z_row: zr,
                z_col: zc,
                coeffs: c,
            })
            .collect();
        let rho = sample_vec(&mut rc, m);
        let mut t_frob = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let mut ch = FsChallenger::new(b"flock-assist-cmp");
            let p = prove_frobenius_assist(&params, &claims, &[], &rho, &mut ch);
            black_box(&p);
            t_frob = t_frob.min(t.elapsed().as_secs_f64());
        }

        let n_stmt = 128 * k_claims;
        eprintln!(
            "  shape: {used} live columns of height {height}, m={m}, {} rounds",
            2 * (m + 1)
        );
        eprintln!(
            "  jagged assist    (1 statement)    : {:>8.2} ms",
            t_single * 1e3
        );
        eprintln!(
            "  frobenius assist ({n_stmt} statements) : {:>8.2} ms",
            t_frob * 1e3
        );
        eprintln!(
            "  ratio: {:.1}x for {n_stmt}x the statements ({:.2} ms marginal per statement)",
            t_frob / t_single,
            (t_frob - t_single) * 1e3 / n_stmt as f64
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
        const REPS: usize = 6;
        const CHUNK: usize = 1 << 13;
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

        // coarse: 8K outputs / task

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
            black_box(round_msg(&a, &b));
        });
        let tp_fine = bench(&mut || {
            black_box(round_msg_par(&a, &b));
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
            black_box(acc);
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
            black_box(acc);
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
                swap(&mut av, &mut sa);
                swap(&mut bv, &mut sb);
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
        black_box(&out);
        black_box(&dst);
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
