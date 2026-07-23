//! Zerocheck PIOP: prove a(y) · b(y) ⊕ c(y) = 0 for all y ∈ {0,1}^m.
//!
//! Inputs are three bit vectors of length 2^m. Output is an evaluation claim
//! on the multilinear extensions â, b̂, ĉ at the protocol-derived point.
//!
//! Protocol shape (m = log_n, k_skip = [`K_SKIP`] = 6):
//!   1. Verifier samples `r ∈ F_{2^128}^m` (the zerocheck challenge).
//!   2. Prover sends `P^{AB}(λ)` and `P^C(λ)` for λ ∈ Λ, |Λ| = 2^k_skip.
//!   3. Verifier samples `z ∈ F_{2^128}` (univariate-skip fold point).
//!   4. For each of the `m - k_skip` multilinear rounds, prover sends
//!      `(P_r(1), P_r(∞))` and verifier samples `ρ_r`.
//!   5. Prover sends final MLE evaluations `(â, b̂, ĉ)` at the resulting point.
//!
//! Both `prove` and `verify` are wired end-to-end. The prove→verify roundtrip
//! is tested on honest witnesses; verify also rejects byte-mutated proofs and
//! shape-corrupted ones.

use crate::challenger::Challenger;
use crate::field::{F8, F128};
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use serde::{Deserialize, Serialize};

pub mod multilinear;
pub mod univariate_skip;
pub mod univariate_skip_deg4;
pub mod univariate_skip_deg4_optimized;
pub mod univariate_skip_optimized;

use multilinear::{
    UniSkipFoldTable, fold_and_compute_round_pair_into, fold_and_round_pair_sparse_into,
    fold_in_place_pair, interpolate_at_z_combined, interpolate_at_z_on_lambda, round_pair_naive,
    uni_skip_fold_and_round_pair_optimized_packed_padded, zero_dead_regions,
};
use univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded,
    small_challenges_ghash,
};

/// Number of variables folded in round 1 via the additive-NTT univariate skip.
/// |Λ| = 2^K_SKIP = 64 elements; the round-1 prover message is two length-64
/// vectors of F128.
pub const K_SKIP: usize = 6;

/// One run of identically-shaped blocks inside a [`PaddingSpec`] run-list.
///
/// A run is `n_blocks` consecutive blocks of `2^k_log` bits each; inside each
/// block, bits `[0, useful_bits_per_block)` carry real data and bits
/// `[useful_bits_per_block, 2^k_log)` are zero padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaddingRun {
    pub k_log: usize,
    pub useful_bits_per_block: usize,
    pub n_blocks: usize,
}

impl PaddingRun {
    /// Address-space extent of the run in bits (= `n_blocks · 2^k_log`).
    pub fn extent_bits(&self) -> usize {
        self.n_blocks << self.k_log
    }
}

/// Witness padding descriptor for URM / fold work-skipping.
///
/// The witness is described by an ordered **run-list**: the [`PaddingRun`]s
/// are laid out back-to-back from address 0, and everything after the last
/// run (up to the instance's `2^m` domain) is an implicit all-zero gap.
/// URM/fold contributions from a chunk of all-zero bits are themselves zero,
/// so kernels may skip any chunk the spec marks as padding or gap and produce
/// byte-identical output — provided those bits are honestly zero.
///
/// Single-table callers build **single-run** specs (one run tiling the whole
/// domain: [`PaddingSpec::dense`], [`PaddingSpec::uniform`], and
/// `BlockR1cs::padding_spec`); the hot kernels detect that case via
/// [`PaddingSpec::as_single_run`] and take exactly the pre-run-list code
/// path. Multi-run specs — the count-derived slot schedules of the
/// multi-table design (`docs/multi-table-design.tex` §5.2, the union prove
/// path) — go through general run-list paths that, since M6, skip dead
/// regions with cost proportional to the declared support
/// ([`Self::useful_block_intervals`] drives the interval-based kernels).
///
/// Use [`PaddingSpec::dense`] when the witness has no padding holes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddingSpec {
    runs: Vec<PaddingRun>,
}

impl PaddingSpec {
    /// "No padding": every bit of the witness is treated as useful. Equivalent
    /// to the legacy URM path with no skipping.
    pub fn dense(m: usize) -> Self {
        Self::uniform(m, 1usize << m, 1)
    }

    /// Single-run spec: `n_blocks` blocks of `2^k_log` bits, each with a
    /// `useful_bits_per_block` useful prefix. With `n_blocks = 2^(m − k_log)`
    /// this is exactly the pre-run-list `PaddingSpec`.
    pub fn uniform(k_log: usize, useful_bits_per_block: usize, n_blocks: usize) -> Self {
        Self::from_runs(vec![PaddingRun {
            k_log,
            useful_bits_per_block,
            n_blocks,
        }])
    }

    /// General run-list constructor. Runs with `n_blocks = 0` cover no address
    /// space and are dropped (canonical form, so `as_single_run` is reliable).
    pub fn from_runs(runs: Vec<PaddingRun>) -> Self {
        for run in &runs {
            assert!(
                run.useful_bits_per_block <= 1usize << run.k_log,
                "useful_bits_per_block {} exceeds block size 2^{}",
                run.useful_bits_per_block,
                run.k_log
            );
        }
        Self {
            runs: runs.into_iter().filter(|r| r.n_blocks > 0).collect(),
        }
    }

    /// The runs, in address order.
    pub fn runs(&self) -> &[PaddingRun] {
        &self.runs
    }

    /// The single run when the list has exactly one — the hot kernels' fast
    /// path. The fast path treats the run as tiling the entire domain
    /// periodically (it ignores `n_blocks`), which matches the pre-run-list
    /// kernels bit-for-bit; a single run with a trailing gap is still handled
    /// correctly because the gap must be honestly zero, like all padding.
    pub fn as_single_run(&self) -> Option<PaddingRun> {
        match self.runs.as_slice() {
            [run] => Some(*run),
            _ => None,
        }
    }

    /// Total extent covered by the runs, in bits. The instance domain `2^m`
    /// may be larger; the difference is the implicit trailing zero gap.
    pub fn covered_bits(&self) -> usize {
        self.runs.iter().map(|r| r.extent_bits()).sum()
    }

    /// [`Self::useful_intervals`] coarsened to `2^log2_block`-bit blocks and
    /// merged: block `x` is listed iff bits `[x·2^log2_block,
    /// (x+1)·2^log2_block)` intersect a useful interval. This is the live set
    /// of a table whose entries each aggregate one block of witness bits
    /// (e.g. the post-URM tables at `log2_block = k_skip`, or packed words at
    /// `log2_block = 7`): outside it the honest table is identically zero.
    pub fn useful_block_intervals(&self, log2_block: usize) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (s, e) in self.useful_intervals() {
            let (s2, e2) = (s >> log2_block, e.div_ceil(1usize << log2_block));
            match out.last_mut() {
                Some((_, prev_e)) if *prev_e >= s2 => *prev_e = (*prev_e).max(e2),
                _ => out.push((s2, e2)),
            }
        }
        out
    }

    /// Sorted, merged list of useful bit intervals `[start, end)` — the
    /// semantic content of the spec (everything outside is declared zero).
    /// Consumed by the general (multi-run) kernel paths and by tests; cost is
    /// O(total blocks), fine off the single-run hot path.
    pub fn useful_intervals(&self) -> Vec<(usize, usize)> {
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        let mut offset = 0usize;
        for run in &self.runs {
            let block_bits = 1usize << run.k_log;
            if run.useful_bits_per_block > 0 {
                for blk in 0..run.n_blocks {
                    let start = offset + blk * block_bits;
                    let end = start + run.useful_bits_per_block;
                    match intervals.last_mut() {
                        Some((_, prev_end)) if *prev_end == start => *prev_end = end,
                        _ => intervals.push((start, end)),
                    }
                }
            }
            offset += run.extent_bits();
        }
        intervals
    }
}

// ---------------------------------------------------------------------------
// Public types: claim, proof, error.
// ---------------------------------------------------------------------------

/// Evaluation claims on the multilinear extensions of a, b, c. **Note that
/// `a_eval`/`b_eval` and `c_eval` are claimed at *different points*** —
/// extract_c separates C from the AB sumcheck:
///
/// - `a_eval`, `b_eval` are at `(z, mlv_challenges)` — the AB sumcheck binds
///   the rest variables one at a time to fresh `ρ_r` challenges.
/// - `c_eval` is at `(z, r_rest)` — C is linear, so its eq-weighted sum
///   collapses immediately to an MLE evaluation at the original eq weights;
///   no per-round folding needed. Here `r_rest = r[K_SKIP..m]` from the
///   zerocheck challenge.
///
/// The downstream caller (R1CS prover + PCS) opens each commitment at its
/// own claim point. Two openings for a, b at the same point; one for c at
/// a different point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckClaim {
    /// Univariate-skip challenge sampled after round 1 (binds the K_SKIP
    /// skip variables).
    pub z: F128,
    /// AB sumcheck bind challenges, one per multilinear round; length = `m - K_SKIP`.
    pub mlv_challenges: Vec<F128>,
    /// Eq weights for the rest variables = the zerocheck challenge restricted
    /// to `r[K_SKIP..m]`. This is the *rest part of the c-claim's point*.
    /// Length = `m - K_SKIP`.
    pub r_rest: Vec<F128>,
    /// `â(z, mlv_challenges)`.
    pub a_eval: F128,
    /// `b̂(z, mlv_challenges)`.
    pub b_eval: F128,
    /// `ĉ(z, r_rest)` — at a *different point* than a_eval, b_eval.
    pub c_eval: F128,
}

/// All round messages the prover sends, in order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZerocheckProof {
    /// Round 1 (univariate skip): `P^{AB}(λ)` for λ ∈ Λ, length 2^K_SKIP.
    pub round1_ab: Vec<F128>,
    /// Round 1 (extract_c): `P^C(λ)` for λ ∈ Λ, length 2^K_SKIP. Sent separately
    /// from `round1_ab` so the verifier can evaluate the C-claim immediately
    /// and skip the C-column in all subsequent rounds.
    pub round1_c: Vec<F128>,
    /// Multilinear sumcheck rounds: each entry is `(P_r(1), P_r(∞))` via the
    /// Karatsuba ∞-trick. Length = `m - K_SKIP`.
    pub multilinear_rounds: Vec<(F128, F128)>,
    /// Final MLE evaluations sent at the end of the protocol.
    pub final_a_eval: F128,
    pub final_b_eval: F128,
    pub final_c_eval: F128,
}

/// Reasons the verifier may reject a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `log_n` doesn't satisfy `log_n >= K_SKIP`.
    LogNTooSmall { log_n: usize, k_skip: usize },
    /// Round-1 messages have the wrong length (expected `2^K_SKIP`).
    BadRound1Length { expected: usize, got: usize },
    /// Wrong number of multilinear-round messages (expected `log_n - K_SKIP`).
    BadMultilinearRoundsLength { expected: usize, got: usize },
    /// `proof.final_c_eval` doesn't match the verifier's reconstruction
    /// `C_s · interpolate_at_z_on_lambda(round1_c, k_skip, z)`. Catches
    /// dishonesty in the round-1 C message or in the final c-eval claim.
    CEvalMismatch,
    /// The AB sumcheck final consistency check failed: the inner running
    /// claim after all rounds should equal `final_a_eval · final_b_eval`.
    /// Any inconsistency in `round1_ab`, in a multilinear round's
    /// `(P_r(1), P_r(∞))`, or in `final_a_eval` / `final_b_eval` propagates
    /// to this check.
    SumcheckFinalFailed,
}

// ---------------------------------------------------------------------------
// API: prove / verify.
// ---------------------------------------------------------------------------

/// Prove that `a(y) · b(y) ⊕ c(y) = 0` for all `y ∈ {0,1}^m`.
///
/// Inputs are LSB-first bit-packed byte vectors (each of length `2^m / 8`).
/// `m ≥ K_SKIP + N_INNER` (= 13). `challenger` supplies all verifier
/// randomness; the prover absorbs each of its messages into the challenger
/// before sampling the next challenge so the verifier (using the same
/// challenger implementation in lockstep) derives identical challenges.
///
/// Returns:
///   - the [`ZerocheckProof`] (raw round messages), and
///   - the [`ZerocheckClaim`] the higher-level caller will pass to its PCS.
pub fn prove_packed<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    prove_packed_padded(
        a_packed,
        b_packed,
        c_packed,
        m,
        &PaddingSpec::dense(m),
        challenger,
    )
}

/// Same as [`prove_packed`] but lets the caller declare a run-list padding
/// pattern so URM can skip work for chunks that fall entirely in zero
/// padding (or in the trailing gap after the last run). Output is
/// byte-identical to the dense path when the padding bits are honestly zero.
pub fn prove_packed_padded<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    let (proof, claim, _) =
        prove_packed_padded_inner(a_packed, b_packed, c_packed, m, padding, false, challenger);
    (proof, claim)
}

/// Variant of [`prove_packed_padded`] that ALSO returns the canonical
/// `s_hat_v_c` produced by the fused two-bank round-1 kernel. The downstream
/// PCS open uses this to skip `fold_1b_rows` for the c-claim — see
/// [`crate::pcs::ring_switch::round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`].
///
/// Wire output `(ZerocheckProof, ZerocheckClaim)` is byte-identical to
/// [`prove_packed_padded`].
pub fn prove_packed_padded_capture_s_hat_v_c<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Vec<F128>) {
    let (proof, claim, captured) =
        prove_packed_padded_inner(a_packed, b_packed, c_packed, m, padding, true, challenger);
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_packed_padded_inner<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    capture_s_hat_v_c: bool,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Option<Vec<F128>>) {
    let k_skip = K_SKIP;
    const N_INNER: usize = 7; // 3 small + 4 medium fixed-constant eq dims
    assert!(
        m >= k_skip + N_INNER,
        "prove requires m >= k_skip + N_INNER (= {})",
        k_skip + N_INNER
    );
    let expected_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected_bytes);
    assert_eq!(b_packed.len(), expected_bytes);
    assert_eq!(c_packed.len(), expected_bytes);
    let n_mlv = m - k_skip;

    challenger.observe_label(b"flock-zerocheck-v0");

    // ---- 1. Sample r (with protocol-fixed constants in the inner 7 dims) ----
    //
    // r layout:
    //   r[0..k_skip]                — sampled (used by verifier for the
    //                                  final check at S; not by the URM)
    //   r[k_skip..k_skip+3]         — protocol small-eq constants φ_8(0xF7..)
    //   r[k_skip+3..k_skip+7]       — protocol medium-eq constants β_i
    //   r[k_skip+7..m]              — sampled (the "outer" eq weights for
    //                                  the URM and multilinear rounds)
    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    // ---- 3. Round 1: URM (extract_c, parallel) ----
    //
    // The optimized URM drops a `C_s = φ_8(0x1C)` scalar from its accumulators
    // (a prover-side optimization tied to the small-eq trick — see the
    // C_s factor analysis in `univariate_skip_optimized`). The wire format
    // must be in "naive" convention so the verifier doesn't need to know
    // about this internal optimization; we restore the C_s factor here.
    let zc_timing = std::env::var_os("FLOCK_ZC_TIMING").is_some();
    let t_round1 = std::time::Instant::now();
    let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
    let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let (round1_ab_opt, round1_c_opt, s_hat_v_c) = if capture_s_hat_v_c {
        let (ab, c, s) =
            crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                a_packed,
                b_packed,
                c_packed,
                m,
                k_skip,
                &r,
                &inv_table,
                padding,
            );
        (ab, c, Some(s))
    } else {
        let (ab, c) = round1_shift_reduce_extract_c_packed_padded(
            a_packed, b_packed, c_packed, m, k_skip, &r, &inv_table, padding,
        );
        (ab, c, None)
    };
    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = round1_ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = round1_c_opt.iter().map(|x| c_s * *x).collect();
    if zc_timing {
        eprintln!(
            "[zc-timing] round1 URM: {:.2} ms",
            t_round1.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- 4. Observe round-1 message, sample z (URM fold point) ----
    challenger.observe_f128_slice(&round1_ab);
    challenger.observe_f128_slice(&round1_c);
    let z = challenger.sample_f128();

    // ---- 5. c_eval = ĉ(z, r_rest) via interpolation of round1_c at z ----
    //
    // round1_c (now in naive convention) carries `P^C(λ) = Σ_x eq(r_rest, x) · ĉ(λ, x)`
    // as its 2^k_skip evaluations on Λ. Interpolating to λ=z gives
    // `ĉ(z, r_rest)` directly (the eq-weighted sum collapses to the MLE
    // evaluation because ĉ is linear). This is **the c-claim** — at point
    // `(z, r_rest)`, *not* `(z, ρ-values)`. ~64 F128 muls + Lagrange weights.
    let final_c_eval = interpolate_at_z_on_lambda(&round1_c, k_skip, z);

    // ---- 6. Round 2: fused fold + first multilinear message ----
    //
    // Convention A wrapping: pass `mlv_arg[0] = ONE` so the function's output
    // `mlv_arg[0] · G(1)` becomes the bare `G(1)` we send on the wire. The
    // verifier samples ρ_1 after observing this message.
    let t_round2 = std::time::Instant::now();
    let fold_table = UniSkipFoldTable::new(k_skip, z);
    let mut mlv_arg = vec![F128::ONE; n_mlv];
    mlv_arg[1..].copy_from_slice(&r[k_skip + 1..]);
    // Support-proportional prover (M6): under a multi-run count-derived spec
    // the post-URM tables are zero outside the declared support (the live
    // interval list). While that support is sparse (live·16 ≤ n), round 2
    // skips its dead-pair zero writes and the tail rounds fold/evaluate over
    // the live intervals only — every skipped term carries an `a·b` factor
    // of zero, so all messages and folded values are byte-identical to the
    // dense path. Once the live fraction crosses the threshold, the dead
    // regions are zeroed once and the dense kernels resume.
    let mut live: Option<Vec<(usize, usize)>> = if padding.as_single_run().is_none() {
        Some(padding.useful_block_intervals(k_skip))
    } else {
        None
    };
    let sparse_from_round2 = live.as_ref().is_some_and(|list| {
        let live_elems: usize = list.iter().map(|&(s, e)| e - s).sum();
        let n_out = 1usize << n_mlv;
        n_out >= 8 && live_elems * 16 <= n_out
    });
    let (mut a_mlv, mut b_mlv, msg_1, msg_inf) = if sparse_from_round2 {
        multilinear::uni_skip_fold_and_round_pair_runs_sparse(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
        )
    } else {
        uni_skip_fold_and_round_pair_optimized_packed_padded(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
        )
    };

    if zc_timing {
        eprintln!(
            "[zc-timing] round2 fused fold: {:.2} ms",
            t_round2.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_tail = std::time::Instant::now();
    let mut multilinear_msgs = Vec::with_capacity(n_mlv);
    multilinear_msgs.push((msg_1, msg_inf));
    challenger.observe_f128(msg_1);
    challenger.observe_f128(msg_inf);
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    mlv_rhos.push(challenger.sample_f128());

    // ---- 7. Rounds 3..(n_mlv + 1) — AB only (c is done) ----
    //
    // Iter i: fold (a, b) at ρ_{i+1}, compute round (i+3) message, sample
    // ρ_{i+2}. Use the fused parallel path while log_n ≥ 10; below that the
    // SplitEqGhash inner can't form lo_size ≥ 2, so we fall back to
    // fold_in_place_pair + round_pair_naive.
    //
    // Ping-pong scratch buffers for the fused path: each fused round folds
    // (a_mlv, b_mlv) of size N into size N/2. Rather than allocating — and,
    // worse, `munmap`-ing, which is single-threaded and caps the tail's
    // parallel speedup — a fresh 64 MB buffer per round, we alternate between
    // two persistent buffers. Scratch capacity = N/2 (the largest fused
    // output); only needed when the first round is actually fused.
    let n_in = a_mlv.len();
    let (mut a_nxt, mut b_nxt) = if n_in >= 1024 {
        (
            crate::scratch::take_f128(n_in / 2),
            crate::scratch::take_f128(n_in / 2),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // See the round-2 comment: `live` drives the tail's sparse rounds;
    // `sparse_dirty` tracks whether the current buffers' dead regions hold
    // unwritten scratch (true from the start when round 2 skipped its
    // dead-pair zero writes).
    let mut sparse_dirty = sparse_from_round2;

    for i in 0..(n_mlv - 1) {
        let rho_prev = mlv_rhos[i];
        let log_n_before = a_mlv.len().trailing_zeros() as usize;

        // r_next for the next round's message: length log_n_before - 1.
        // r_next[0] = ONE (Convention A factor); r_next[1..] are the eq
        // weights for the remaining variables = r[k_skip + i + 2..m].
        let mut r_next = vec![F128::ONE; log_n_before - 1];
        r_next[1..].copy_from_slice(&r[k_skip + i + 2..]);

        let use_sparse = live.as_ref().is_some_and(|list| {
            let live_elems: usize = list.iter().map(|&(s, e)| e - s).sum();
            a_mlv.len() >= 8 && live_elems * 16 <= a_mlv.len()
        });
        if !use_sparse
            && let Some(list) = live.take()
            && sparse_dirty
        {
            let len = a_mlv.len();
            zero_dead_regions(&mut a_mlv, len, &list);
            zero_dead_regions(&mut b_mlv, len, &list);
            sparse_dirty = false;
        }

        let (m1, mi) = if use_sparse {
            let half = a_mlv.len() / 2;
            if a_nxt.len() < half {
                crate::scratch::give_f128(a_nxt);
                crate::scratch::give_f128(b_nxt);
                a_nxt = crate::scratch::take_f128(half);
                b_nxt = crate::scratch::take_f128(half);
            }
            let (m1, mi, live_out) = fold_and_round_pair_sparse_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                rho_prev,
                &r_next,
                live.as_ref().expect("use_sparse implies live"),
            );
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            live = Some(live_out);
            sparse_dirty = true;
            (m1, mi)
        } else if log_n_before >= 10 {
            let half = a_mlv.len() / 2;
            let (m1, mi) = fold_and_compute_round_pair_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                rho_prev,
                &r_next,
            );
            // Swap current <-> scratch, then shrink the new current to the
            // folded size. The old (larger) buffer becomes scratch; we only
            // ever write its leading `half` slots next round, so its stale
            // length is harmless.
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            (m1, mi)
        } else {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            round_pair_naive(&a_mlv, &b_mlv, &r_next)
        };

        multilinear_msgs.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        mlv_rhos.push(challenger.sample_f128());
    }

    // ---- 8. Final binding at ρ_{n_mlv} (the last challenge) ----
    let rho_last = *mlv_rhos.last().expect("at least one ρ sampled");
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_last);
    debug_assert_eq!(a_mlv.len(), 1);
    debug_assert_eq!(b_mlv.len(), 1);

    let final_a_eval = a_mlv[0];
    let final_b_eval = b_mlv[0];

    // ---- Fiat–Shamir: bind the final â, b̂ claims into the transcript ----
    //
    // These two claims are reduced downstream by lincheck via a *single*
    // random-linear-combination check with coefficient α (`target = α·v_a + v_b`,
    // see `lincheck`). That batching is only sound if α is sampled *after*
    // (v_a, v_b) are committed to the transcript — otherwise a prover that knows
    // α can pick (v_a, v_b) to satisfy the one batched equation while violating
    // the individual checks. So observe them here, before any later challenge
    // (the next one drawn is lincheck's α). `final_c_eval` needs no observe — the
    // verifier recomputes it from the already-absorbed `round1_c`/`z` and rejects
    // on mismatch (see `verify`), so it is already transcript-bound.
    challenger.observe_f128(final_a_eval);
    challenger.observe_f128(final_b_eval);

    // Recycle the four tail buffers (the two len-1 survivors still own their
    // full round-2 capacity) for the next phase/prove.
    crate::scratch::give_f128(a_mlv);
    crate::scratch::give_f128(b_mlv);
    crate::scratch::give_f128(a_nxt);
    crate::scratch::give_f128(b_nxt);

    if zc_timing {
        eprintln!(
            "[zc-timing] rounds 3+ tail: {:.2} ms",
            t_tail.elapsed().as_secs_f64() * 1e3
        );
    }

    let r_rest: Vec<F128> = r[k_skip..].to_vec();

    let proof = ZerocheckProof {
        round1_ab,
        round1_c,
        multilinear_rounds: multilinear_msgs,
        final_a_eval,
        final_b_eval,
        final_c_eval,
    };
    let claim = ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: final_a_eval,
        b_eval: final_b_eval,
        c_eval: final_c_eval,
    };
    (proof, claim, s_hat_v_c)
}

/// Verify a zerocheck proof for an instance over `{0,1}^log_n`.
///
/// Walks the challenger in lockstep with the prover, samples the same
/// challenges, and checks every round's consistency equation.
///
/// On accept: returns the [`ZerocheckClaim`] the caller must check against
/// its PCS opening of `â`, `b̂`, `ĉ`.
/// On reject: returns a [`VerifyError`] indicating which check failed.
pub fn verify<C: Challenger>(
    log_n: usize,
    proof: &ZerocheckProof,
    challenger: &mut C,
) -> Result<ZerocheckClaim, VerifyError> {
    let m = log_n;
    let k_skip = K_SKIP;
    const N_INNER: usize = 7;

    if m < k_skip + N_INNER {
        return Err(VerifyError::LogNTooSmall { log_n: m, k_skip });
    }
    let n_mlv = m - k_skip;
    let ell = 1usize << k_skip;

    // ---- Shape checks ----
    if proof.round1_ab.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_ab.len(),
        });
    }
    if proof.round1_c.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_c.len(),
        });
    }
    if proof.multilinear_rounds.len() != n_mlv {
        return Err(VerifyError::BadMultilinearRoundsLength {
            expected: n_mlv,
            got: proof.multilinear_rounds.len(),
        });
    }

    challenger.observe_label(b"flock-zerocheck-v0");

    // ---- Re-derive r (in lockstep with prove_packed) ----
    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    // ---- Observe round-1 messages, sample z ----
    challenger.observe_f128_slice(&proof.round1_ab);
    challenger.observe_f128_slice(&proof.round1_c);
    let z = challenger.sample_f128();

    // ---- Reconstruct ĉ(z, r_rest) from round1_c ----
    //
    // P^C has degree < 2^k_skip in λ (C is linear, summed against eq); ell
    // evaluations on Λ uniquely interpolate to z. round1_c is in naive
    // convention (the prover restored the C_s factor before sending), so
    // `ĉ(z, r_rest) = P^C(z)` directly.
    let computed_c_eval = interpolate_at_z_on_lambda(&proof.round1_c, k_skip, z);
    if computed_c_eval != proof.final_c_eval {
        return Err(VerifyError::CEvalMismatch);
    }

    // ---- Reconstruct the initial AB running claim ----
    //
    // P^{AB}(z) requires the polynomial in λ of degree < 2·ell to be evaluated
    // at z. The prover sent only ell evaluations on Λ — not enough on its own.
    // The verifier uses the **zerocheck assumption** `P^{AB}(λ) + P^C(λ) = 0`
    // for `λ ∈ S`. Together with the ell Λ-evaluations of the combined
    // polynomial, that's 2·ell evaluations — enough to interpolate the
    // combined polynomial at z. Then `P^{AB}(z) = P^{combined}(z) − P^C(z)`,
    // which in char-2 is `P^{combined}(z) + P^C(z)`.
    //
    // If the prover's witness is dishonest the S-zero assumption fails, the
    // reconstructed c_0 is wrong, and the running-claim chain ends at a value
    // inconsistent with `â · b̂`. We catch that at the final sumcheck check.
    let combined_at_lambda: Vec<F128> = proof
        .round1_ab
        .iter()
        .zip(&proof.round1_c)
        .map(|(x, y)| *x + *y)
        .collect();
    let combined_at_z = interpolate_at_z_combined(&combined_at_lambda, k_skip, z);
    let p_c_at_z = interpolate_at_z_on_lambda(&proof.round1_c, k_skip, z);
    let mut c_running = combined_at_z + p_c_at_z;

    // ---- Multilinear sumcheck chain ----
    //
    // The propagated running claim is the *inner* polynomial value G(ρ),
    // not the full per-round polynomial P(ρ) = eq(r_eq, ρ) · G(ρ). The eq
    // factor for the just-bound variable is absorbed by the next round's
    // consistency check via the identity
    //   G_{r-1}(ρ_{r-1}) = (1 + r_eq_r) · G_r(0) + r_eq_r · G_r(1).
    //
    // Round r (0-indexed i = r − 2) binds the i-th rest variable with eq weight
    // r[k_skip + i]. The prover sends `(G(1), G(∞))` (Convention A — no
    // factor). Verifier:
    //   1. reconstruct G(0) from consistency `c_running = (1+r_eq)·G(0) + r_eq·G(1)`,
    //   2. observe message, sample ρ_i,
    //   3. update `c_running ← G(ρ_i)`,
    //      where `G(X) = G(0)·(1+X) + G(1)·X + G(∞)·X·(X+1)` (char-2 quadratic
    //      interpolation through G(0), G(1), G(∞)).
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    for (i, &(msg_1, msg_inf)) in proof.multilinear_rounds.iter().enumerate() {
        let r_eq = r[k_skip + i];
        let one_plus_r_eq = F128::ONE + r_eq;

        let g1 = msg_1;
        let g_inf = msg_inf;
        let g0 = (c_running + r_eq * g1) * one_plus_r_eq.inv();

        challenger.observe_f128(msg_1);
        challenger.observe_f128(msg_inf);
        let rho = challenger.sample_f128();
        mlv_rhos.push(rho);

        let one_plus_rho = F128::ONE + rho;
        // G(ρ) = G(0)·(1+ρ) + G(1)·ρ + G(∞)·ρ·(1+ρ).
        c_running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
    }

    // ---- AB sumcheck final consistency ----
    //
    // After all variables are bound, the inner running claim is just the
    // polynomial without the eq weighting:
    //   G_final(ρ_all) = â(z, ρ) · b̂(z, ρ) = final_a_eval · final_b_eval.
    // (The eq factors were absorbed round-by-round into the consistency checks,
    // never accumulating into the running claim.)
    let r_rest: Vec<F128> = r[k_skip..].to_vec();
    let expected_final = proof.final_a_eval * proof.final_b_eval;
    if c_running != expected_final {
        return Err(VerifyError::SumcheckFinalFailed);
    }

    // ---- Fiat–Shamir: bind the final â, b̂ claims (mirrors `prove_packed_padded_inner`) ----
    //
    // Must observe at the same transcript position as the prover, before the
    // next challenge (lincheck's α) is drawn, so the α-batched reduction of
    // these two claims is sound. `final_c_eval` is already bound via the
    // recompute-and-compare above, so it is not observed.
    challenger.observe_f128(proof.final_a_eval);
    challenger.observe_f128(proof.final_b_eval);

    Ok(ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: proof.final_a_eval,
        b_eval: proof.final_b_eval,
        c_eval: proof.final_c_eval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    /// SplitMix64 PRNG, deterministic.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    /// Pack three Boolean vectors into the (a_packed, b_packed, c_packed)
    /// shape that `prove_packed` consumes.
    fn pack_abc(a: &[bool], b: &[bool], c: &[bool]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use univariate_skip::pack_bits;
        (pack_bits(a), pack_bits(b), pack_bits(c))
    }

    /// `prove` runs end-to-end at the smallest valid m (= k_skip + N_INNER = 13)
    /// without panicking, and produces output of the right shape.
    ///
    /// We can't yet check the proof is *accepted* (verify is a stub), but the
    /// structural sanity here catches:
    ///   - mismatched challenger observe/sample sequence
    ///   - wrong slice lengths in r / mlv_arg / r_next at any round
    ///   - any unreachable assert in the underlying functions
    #[test]
    fn prove_runs_end_to_end() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            // Honest witness: c = a AND b, so a·b ⊕ c = 0 on the hypercube.
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut challenger = FsChallenger::new(b"flock-test-v0");
            let (proof, claim) = prove_packed(&a_p, &b_p, &c_p, m, &mut challenger);

            // Shape checks.
            assert_eq!(proof.round1_ab.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.round1_c.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.multilinear_rounds.len(), m - K_SKIP, "m={m}");
            assert_eq!(claim.mlv_challenges.len(), m - K_SKIP, "m={m}");

            // Claim's eval fields agree with the proof's final evals.
            assert_eq!(claim.a_eval, proof.final_a_eval, "m={m}");
            assert_eq!(claim.b_eval, proof.final_b_eval, "m={m}");
            assert_eq!(claim.c_eval, proof.final_c_eval, "m={m}");
        }
    }

    /// **Prove→verify roundtrip**: an honest proof verifies cleanly, and the
    /// claim returned by `verify` is byte-for-byte equal to the claim returned
    /// by `prove`.
    #[test]
    fn prove_verify_roundtrip_honest() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(1000 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &proof, &mut ch_verify);
            let claim_v = result.unwrap_or_else(|e| panic!("verify rejected at m={m}: {e:?}"));

            assert_eq!(claim_p, claim_v, "claim mismatch at m={m}");
        }
    }

    /// **Verify rejects byte-mutated proofs.** Walk each component of the
    /// proof and flip one F128 entry; the verifier must return an `Err`
    /// (rather than panicking or silently accepting).
    #[test]
    fn verify_rejects_mutations() {
        let m = 14;
        let mut rng = Rng::new(5050);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let _seed: u64 = 0xDEAD_BEEF;
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Each closure returns a mutated copy; verify must reject all of them.
        let mutations: Vec<(&str, Box<dyn Fn(&ZerocheckProof) -> ZerocheckProof>)> = vec![
            (
                "round1_ab[0] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_ab[0].lo ^= 1;
                    q
                }),
            ),
            (
                "round1_c[5] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_c[5].lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[0].0 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.multilinear_rounds[0].0.lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[2].1 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    let last = q.multilinear_rounds.len() / 2;
                    q.multilinear_rounds[last].1.hi ^= 1;
                    q
                }),
            ),
            (
                "final_a_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_a_eval.lo ^= 1;
                    q
                }),
            ),
            (
                "final_c_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_c_eval.hi ^= 1;
                    q
                }),
            ),
        ];

        for (label, mutate) in mutations {
            let bad = mutate(&proof);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &bad, &mut ch);
            assert!(
                result.is_err(),
                "verify accepted mutated proof ({label}) — should have rejected"
            );
        }
    }

    /// Shape rejections: too-short round1, wrong number of multilinear rounds.
    #[test]
    fn verify_rejects_shape_errors() {
        let m = 14;
        let mut rng = Rng::new(606);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Truncate round1_ab.
        let mut bad = proof.clone();
        bad.round1_ab.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadRound1Length { .. })
        ));

        // Truncate multilinear rounds.
        let mut bad = proof.clone();
        bad.multilinear_rounds.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadMultilinearRoundsLength { .. })
        ));

        // log_n too small.
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(K_SKIP + 6, &proof, &mut ch),
            Err(VerifyError::LogNTooSmall { .. })
        ));
    }

    /// AUDIT: a FALSE statement (c ≠ a·b at some hypercube point) must be
    /// rejected, even though the prover follows the honest algorithm on its
    /// (dishonest) witness.
    #[test]
    fn audit_false_statement_rejected() {
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(7777 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            // Correct c, then corrupt ONE bit so a·b ⊕ c ≠ 0 somewhere.
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            c[3] = !c[3];

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "verify ACCEPTED a false statement at m={m}: {res:?}"
            );
        }
    }

    /// AUDIT: flipping any round's `msg_inf` (the degree-2 / ∞ coefficient)
    /// must be rejected. `msg_inf` is observed into the transcript, so the
    /// tamper both reshuffles subsequent ρ challenges and breaks the
    /// running-claim chain — either way the final check fails.
    #[test]
    fn audit_round_msg_inf_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(424242);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // For each round, flip msg_inf to a different value. Because msg_inf
        // is observed into the transcript, this reshuffles subsequent rho's;
        // a sound verifier should reject (overwhelming probability).
        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].1 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &bad, &mut ch);
            assert!(res.is_err(), "msg_inf tamper at round {idx} ACCEPTED");
        }
    }

    /// AUDIT: the LAST round's `msg_inf` must be constrained — a common
    /// off-by-one is to leave the final round's leading coefficient unchecked.
    /// Kept separate from the all-rounds loop above so a regression here points
    /// straight at the final-round binding.
    #[test]
    fn audit_last_round_inf_constrained() {
        let m = 13;
        let mut rng = Rng::new(98765);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        let last = proof.multilinear_rounds.len() - 1;
        let mut bad = proof.clone();
        bad.multilinear_rounds[last].1 += F128::ONE;
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch).is_err(),
            "last-round msg_inf unconstrained"
        );
    }

    /// AUDIT (Fiat–Shamir binding of the final â, b̂ claims). Regression test
    /// for the gap where `final_a_eval`/`final_b_eval` were not observed into
    /// the transcript.
    ///
    /// Downstream, lincheck reduces these two claims via a *single* random-
    /// linear-combination check (`target = α·v_a + v_b`). That batching is only
    /// sound if α is sampled *after* the claims are bound to the transcript —
    /// otherwise a prover that already knows α can pick (v_a, v_b) to satisfy
    /// the one batched equation while violating the individual ties.
    ///
    /// A *product-preserving* tamper `(â, b̂) → (â·t, b̂·t⁻¹)` leaves the
    /// zerocheck's own final check `c_running == â·b̂` satisfied, so `verify`
    /// still returns `Ok` — the zerocheck alone is blind to it. The defense is
    /// that both claims are now observed last in the transcript, so the next
    /// challenge (the slot lincheck draws α from) must diverge from the honest
    /// run. This assertion FAILS before the observe was added (identical
    /// post-state) and passes now.
    #[test]
    fn audit_final_ab_claims_bound_to_transcript() {
        let m = 14;
        let mut rng = Rng::new(0xF1A7_5A11);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Honest verify, then capture the next challenge the transcript feeds
        // downstream — this is exactly the slot lincheck samples α from.
        let mut ch_honest = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &proof, &mut ch_honest).is_ok(),
            "honest verify rejected"
        );
        let alpha_honest = ch_honest.sample_f128();

        // Product-preserving tamper: â' = â·t, b̂' = b̂·t⁻¹ ⇒ â'·b̂' = â·b̂, so the
        // zerocheck's `c_running == â·b̂` check still holds for the tampered pair.
        let t = F128 {
            lo: 0x0123_4567_89ab_cdef,
            hi: 0xfedc_ba98_7654_3210,
        };
        assert!(t != F128::ZERO && t != F128::ONE, "t must be nontrivial");
        let mut bad = proof.clone();
        bad.final_a_eval *= t;
        bad.final_b_eval *= t.inv();
        assert_ne!(bad.final_a_eval, proof.final_a_eval, "tamper must change â");
        assert_ne!(bad.final_b_eval, proof.final_b_eval, "tamper must change b̂");
        assert_eq!(
            bad.final_a_eval * bad.final_b_eval,
            proof.final_a_eval * proof.final_b_eval,
            "tamper must preserve the product",
        );

        // The zerocheck's own checks are blind to a product-preserving tamper:
        // verify still ACCEPTS. This is precisely the gap the FS binding closes —
        // the tamper is caught only because the claims now move the transcript.
        let mut ch_tampered = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch_tampered).is_ok(),
            "product-preserving tamper rejected by zerocheck's own checks (unexpected)",
        );
        let alpha_tampered = ch_tampered.sample_f128();

        // The fix: observing â, b̂ makes the downstream challenge depend on them,
        // so lincheck's α (and everything after) diverges and rejects the
        // tampered pair. Before the fix these challenges were equal.
        assert_ne!(
            alpha_honest, alpha_tampered,
            "final â/b̂ claims are NOT bound into the transcript: a product-preserving \
             tamper leaves the downstream challenge unchanged, breaking lincheck's \
             α-batched reduction of (v_a, v_b)",
        );
    }

    /// AUDIT: many random false witnesses must all be rejected. Stronger than a
    /// single corruption — exercises the full prove→verify path on statements
    /// that are false at varying numbers of hypercube points.
    #[test]
    fn audit_many_false_statements_rejected() {
        let m = 13;
        for seed in 0..20u64 {
            let mut rng = Rng::new(0xBADC0DE ^ seed);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            // Flip a random number of bits (1..=4).
            let nflip = 1 + (rng.next_u64() as usize % 4);
            for _ in 0..nflip {
                let idx = rng.next_u64() as usize % c.len();
                c[idx] = !c[idx];
            }
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "false statement (seed={seed}) ACCEPTED: {res:?}"
            );
        }
    }

    /// AUDIT: tamper msg_1 in each round; must reject.
    #[test]
    fn audit_round_msg_1_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(31415);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].0 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            assert!(
                verify(m, &bad, &mut ch).is_err(),
                "msg_1 tamper round {idx} ACCEPTED"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Run-list PaddingSpec.
    // -----------------------------------------------------------------------

    /// Run-list construction/accessor sanity: canonical forms, extents,
    /// single-run detection, and useful-interval merging.
    #[test]
    fn padding_spec_run_list_accessors() {
        // dense(m) is a single run tiling the domain.
        let dense = PaddingSpec::dense(5);
        assert_eq!(
            dense.as_single_run(),
            Some(PaddingRun {
                k_log: 5,
                useful_bits_per_block: 32,
                n_blocks: 1
            })
        );
        assert_eq!(dense.covered_bits(), 32);
        assert_eq!(dense.useful_intervals(), vec![(0, 32)]);

        // uniform: one interval per block; partial useful prefixes don't merge.
        let uni = PaddingSpec::uniform(4, 10, 3);
        assert_eq!(uni.covered_bits(), 48);
        assert_eq!(uni.useful_intervals(), vec![(0, 10), (16, 26), (32, 42)]);
        assert!(uni.as_single_run().is_some());

        // Fully-useful blocks merge into one interval, across run boundaries
        // too when the next run starts where the previous one's data ends.
        let multi = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 4,
                useful_bits_per_block: 16,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 3,
                useful_bits_per_block: 4,
                n_blocks: 1,
            },
        ]);
        assert!(multi.as_single_run().is_none());
        assert_eq!(multi.covered_bits(), 40);
        assert_eq!(multi.useful_intervals(), vec![(0, 36)]);

        // Zero-block runs are dropped (canonical form), so a list that
        // degenerates to one real run still takes the single-run fast path;
        // zero-useful runs cover address space but contribute no intervals.
        let canon = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 4,
                useful_bits_per_block: 16,
                n_blocks: 0,
            },
            PaddingRun {
                k_log: 3,
                useful_bits_per_block: 0,
                n_blocks: 2,
            },
        ]);
        assert_eq!(canon.runs().len(), 1);
        assert!(canon.as_single_run().is_some());
        assert_eq!(canon.covered_bits(), 16);
        assert_eq!(canon.useful_intervals(), Vec::<(usize, usize)>::new());
    }

    /// A run whose useful prefix exceeds its block size is malformed.
    #[test]
    #[should_panic(expected = "exceeds block size")]
    fn padding_spec_rejects_oversized_useful_prefix() {
        let _ = PaddingSpec::uniform(4, 17, 1);
    }

    /// Zero every bit outside the spec's useful intervals (honest padding).
    fn zero_outside_useful(spec: &PaddingSpec, bits: &mut [bool]) {
        let mut useful = vec![false; bits.len()];
        for (s, e) in spec.useful_intervals() {
            useful[s..e].fill(true);
        }
        for (b, u) in bits.iter_mut().zip(&useful) {
            if !*u {
                *b = false;
            }
        }
    }

    /// **Single-run spec is byte-identical to the dense prover** (same proof,
    /// same claim, same transcript position) on an honestly padded witness —
    /// the run-list generalization must not perturb today's wire format.
    /// Covers the BLAKE3 shape (k_log=14, useful=15409) over several blocks.
    #[test]
    fn prove_padded_single_run_matches_dense() {
        let (m, k_log, useful_bits) = (17usize, 14usize, 15_409usize);
        let padding = PaddingSpec::uniform(k_log, useful_bits, 1 << (m - k_log));

        let mut rng = Rng::new(0x5111_C1E4);
        let mut a = rng.bits(1 << m);
        let mut b = rng.bits(1 << m);
        zero_outside_useful(&padding, &mut a);
        zero_outside_useful(&padding, &mut b);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_dense = FsChallenger::new(b"flock-test-v0");
        let (proof_dense, claim_dense) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_dense);

        let mut ch_padded = FsChallenger::new(b"flock-test-v0");
        let (proof_padded, claim_padded) =
            prove_packed_padded(&a_p, &b_p, &c_p, m, &padding, &mut ch_padded);

        assert_eq!(proof_dense, proof_padded, "proof mismatch");
        assert_eq!(claim_dense, claim_padded, "claim mismatch");
        // Transcript position: the next challenge either prover's caller
        // would draw (lincheck's α slot) must agree.
        assert_eq!(
            ch_dense.sample_f128(),
            ch_padded.sample_f128(),
            "post-proof transcript state diverged"
        );
    }

    /// **Multi-run spec is byte-identical to the dense prover** through the
    /// general kernel paths (full-length b_med_counts table in round 1,
    /// per-pair skip table in round 2), including the `capture_s_hat_v_c`
    /// variant. The spec has two runs of different block shapes plus an
    /// implicit trailing gap — the shape of a multi-table slot schedule.
    #[test]
    fn prove_padded_multi_run_matches_dense() {
        let m = 15usize;
        // Two runs (2×2^13 + 1×2^12 = 20480 bits) + a 12288-bit trailing gap.
        let padding = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 5_000,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 12,
                useful_bits_per_block: 3_000,
                n_blocks: 1,
            },
        ]);
        assert!(padding.as_single_run().is_none(), "must exercise multi-run");
        assert!(padding.covered_bits() < 1 << m, "must exercise the gap");

        let mut rng = Rng::new(0x0417_1157);
        let mut a = rng.bits(1 << m);
        let mut b = rng.bits(1 << m);
        zero_outside_useful(&padding, &mut a);
        zero_outside_useful(&padding, &mut b);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_dense = FsChallenger::new(b"flock-test-v0");
        let (proof_dense, claim_dense, s_hat_v_dense) = prove_packed_padded_capture_s_hat_v_c(
            &a_p,
            &b_p,
            &c_p,
            m,
            &PaddingSpec::dense(m),
            &mut ch_dense,
        );

        let mut ch_padded = FsChallenger::new(b"flock-test-v0");
        let (proof_padded, claim_padded, s_hat_v_padded) =
            prove_packed_padded_capture_s_hat_v_c(&a_p, &b_p, &c_p, m, &padding, &mut ch_padded);

        assert_eq!(proof_dense, proof_padded, "proof mismatch");
        assert_eq!(claim_dense, claim_padded, "claim mismatch");
        assert_eq!(s_hat_v_dense, s_hat_v_padded, "s_hat_v_c mismatch");
        assert_eq!(
            ch_dense.sample_f128(),
            ch_padded.sample_f128(),
            "post-proof transcript state diverged"
        );

        // And the multi-run proof still verifies.
        let mut ch_verify = FsChallenger::new(b"flock-test-v0");
        verify(m, &proof_padded, &mut ch_verify).expect("multi-run proof must verify");
    }

    /// **Sparse multi-run spec is byte-identical to the dense prover** through
    /// the M6 support-proportional tail (`fold_and_round_pair_sparse_into`):
    /// the support here is ~1% of the domain, so the tail's sparse rounds
    /// genuinely run (unlike `prove_padded_multi_run_matches_dense`, whose
    /// support is too dense to trigger them), including the mid-tail
    /// switch-back to the dense kernels (zeroing dead scratch) once the live
    /// fraction crosses the threshold.
    #[test]
    fn prove_padded_sparse_multi_run_matches_dense() {
        let m = 16usize;
        // Two count-derived-shaped runs: blocks of 2^13 bits with a 256-bit
        // declared prefix (n_t = 2 rows of 128), then a gap-shaped zero run,
        // then a smaller block shape — plus the implicit trailing gap.
        let padding = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 256,
                n_blocks: 3,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 0,
                n_blocks: 1,
            },
            PaddingRun {
                k_log: 12,
                useful_bits_per_block: 128,
                n_blocks: 2,
            },
        ]);
        assert!(padding.as_single_run().is_none(), "must exercise multi-run");
        let live = padding.useful_block_intervals(K_SKIP);
        let live_elems: usize = live.iter().map(|&(s, e)| e - s).sum();
        assert!(
            live_elems * 16 <= 1usize << (m - K_SKIP),
            "spec must be sparse enough to drive the sparse tail"
        );

        let mut rng = Rng::new(0x0616_5A9D);
        let mut a = rng.bits(1 << m);
        let mut b = rng.bits(1 << m);
        zero_outside_useful(&padding, &mut a);
        zero_outside_useful(&padding, &mut b);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_dense = FsChallenger::new(b"flock-test-v0");
        let (proof_dense, claim_dense, s_hat_v_dense) = prove_packed_padded_capture_s_hat_v_c(
            &a_p,
            &b_p,
            &c_p,
            m,
            &PaddingSpec::dense(m),
            &mut ch_dense,
        );

        let mut ch_padded = FsChallenger::new(b"flock-test-v0");
        let (proof_padded, claim_padded, s_hat_v_padded) =
            prove_packed_padded_capture_s_hat_v_c(&a_p, &b_p, &c_p, m, &padding, &mut ch_padded);

        assert_eq!(proof_dense, proof_padded, "proof mismatch");
        assert_eq!(claim_dense, claim_padded, "claim mismatch");
        assert_eq!(s_hat_v_dense, s_hat_v_padded, "s_hat_v_c mismatch");
        assert_eq!(
            ch_dense.sample_f128(),
            ch_padded.sample_f128(),
            "post-proof transcript state diverged"
        );

        let mut ch_verify = FsChallenger::new(b"flock-test-v0");
        verify(m, &proof_padded, &mut ch_verify).expect("sparse multi-run proof must verify");
    }

    /// Determinism: same witness + same challenger seed → same proof.
    #[test]
    fn prove_deterministic() {
        let m = 14;
        let mut rng = Rng::new(99);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch1 = FsChallenger::new(b"flock-test-v0");
        let mut ch2 = FsChallenger::new(b"flock-test-v0");
        let (proof1, claim1) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch1);
        let (proof2, claim2) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch2);

        assert_eq!(proof1.round1_ab, proof2.round1_ab);
        assert_eq!(proof1.round1_c, proof2.round1_c);
        assert_eq!(proof1.multilinear_rounds, proof2.multilinear_rounds);
        assert_eq!(proof1.final_a_eval, proof2.final_a_eval);
        assert_eq!(proof1.final_b_eval, proof2.final_b_eval);
        assert_eq!(proof1.final_c_eval, proof2.final_c_eval);
        assert_eq!(claim1.z, claim2.z);
        assert_eq!(claim1.mlv_challenges, claim2.mlv_challenges);
    }
}
