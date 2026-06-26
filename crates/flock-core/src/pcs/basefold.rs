//! BaseFold prover/verifier with multi-arity FRI.
//!
//! BaseFold runs `L = log_msg_len` sumcheck rounds in lockstep with codeword
//! folds. The first `log_batch_size` rounds are **row-batch** rounds (combine
//! 2 adjacent SoA lanes within each codeword position). The remaining
//! `log_dim = L − log_batch_size` rounds are **FRI** rounds that fold pairs
//! of codeword positions via `fold_pair`.
//!
//! ## Two-tree commit + multi-arity FRI
//!
//! Two separate Merkle commitments bind the codeword in stages:
//!
//! - **T₁ (initial)** — built in [`super::commit::commit`] before basefold
//!   runs. Leaves contain ONE codeword position's row-batch lanes
//!   (`2^log_batch_size = num_ntts` F_{2^128} per leaf). Small leaves keep
//!   per-query path proofs short and proof size low.
//! - **T₂ (post-row-batch)** — built **inside** [`prove`] right after the
//!   `log_batch_size` row-batch sumcheck rounds. Multi-arity leaves of
//!   `2^arity_0` F_{2^128} group consecutive post-row-batch positions so
//!   one Merkle opening suffices for the first FRI epoch's `arity_0` folds.
//!
//! Subsequent FRI epochs get their own commits via the multi-arity scheme:
//! `arities = [6, 6, 5]` for `log_dim = 17` → 1 (T₂) + 2 (FRI epoch boundaries)
//! commits inside basefold, plus T₁ from outside.
//!
//! ## Per-query work
//!
//! For each FRI query position:
//! 1. Open the **T₁ leaf** (`num_ntts` F_{2^128} = one position's row-batch
//!    lanes) via one Merkle path. Verify against T₁ root.
//! 2. Row-batch-fold the lanes → a single post-row-batch F_{2^128} value.
//! 3. Open the **T₂ leaf** (`2^arity_0` F_{2^128} = the multi-arity coset
//!    for this position's FRI epoch 0) via one Merkle path. Verify against
//!    T₂ root, then **cross-check** that T₂'s value at the queried offset
//!    matches the row-batch-folded value from step 2.
//! 4. FRI-fold T₂'s `2^arity_0` values via arity_0 challenges → one value at
//!    the post-epoch-0 layer.
//! 5. For each subsequent FRI commit i: open the **epoch leaf**
//!    (`2^arity_{i+1}` F_{2^128} values), verify Merkle, locate the position
//!    inside the leaf, check it matches the prior epoch's folded value, then
//!    fold the leaf via arity_{i+1} challenges to produce the next layer's
//!    expected value.
//! 6. After the last epoch, the expected value must match `final_codeword`
//!    at the corresponding (constant) position.
//!
//! ## Why two trees instead of one big-leaf tree
//!
//! Earlier versions used a single tree whose initial leaves bundled
//! `2^arity_0` consecutive codeword positions × `num_ntts` lanes
//! (= `2^11 = 2 KiB` at default params). One Merkle open per query gave the
//! verifier everything for the first FRI epoch. But this inflated each
//! query's initial-leaf payload by `2^arity_0 ×` more than necessary
//! (sending 64 positions when the verifier only needed one position to
//! do row-batch verification, plus the cross-check against T₂).
//!
//! The two-tree split cuts initial-leaf payload by `2^arity_0`× at the cost
//! of one extra Merkle commit on a 32×-smaller codeword — negligible
//! prover-side, ~4× smaller proofs.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::merkle::{self, Hash};
use crate::ntt::AdditiveNttF128;
use serde::{Deserialize, Serialize};

/// Default FRI query count at **rate 1/2** (= `log_inv_rate = 1`). 243
/// queries give 100 bits of provable soundness in the **unique-decoding
/// regime** (UDR). See [`default_fri_queries`] for the rate-aware lookup
/// used by [`super::open`] / [`super::open_batch_padded`].
///
/// Within distance `γ = (1−ρ)/2 − ε*` of the RS code (strictly inside the
/// unique decoding radius, with proximity loss `ε* = 10⁻³` so BCHKS25
/// Theorem 1.4 covers the folding steps), the prover is consistent with
/// **at most one** codeword — no list, no union bound, no OOD step. Each
/// query catches a γ-far prover with probability ≥ γ:
///
/// ```text
/// soundness error ≤ (1 − γ)^t
/// ```
///
/// For 100 bits we need
///
/// ```text
/// t · (−log₂(1 − γ)) ≥ 100.
/// ```
///
/// The fold-consistency (proximity-gap) term is `a ≤ 2/ε*` by Theorem 1.4,
/// independent of codeword length, so over F128 it sits ≥ 115 bits below the
/// challenge space and needs no grinding. Matches ligerito's `udr_queries` /
/// `UDR_PROXIMITY_LOSS` derivation.
pub const DEFAULT_FRI_QUERIES: usize = 243;

/// FRI query count required for 100 bits of soundness at the given
/// `log_inv_rate`, in the unique-decoding regime documented on
/// [`DEFAULT_FRI_QUERIES`]. Slimmer codes (larger `log_inv_rate`) have
/// larger γ, so each query closes more soundness — but per-query soundness
/// saturates below 1 bit (γ < 1/2 always), unlike the Johnson regime where
/// it grows without bound as the rate drops.
///
/// Panics on unsupported rates so we notice if a new rate is added without
/// updating the table.
pub fn default_fri_queries(log_inv_rate: usize) -> usize {
    match log_inv_rate {
        1 => DEFAULT_FRI_QUERIES, // rate 1/2: γ ≈ 0.249, ~0.413 bits/query
        2 => 148,                 // rate 1/4: γ ≈ 0.374, ~0.676 bits/query
        _ => panic!(
            "default_fri_queries: unsupported log_inv_rate {log_inv_rate} \
             — add a soundness-derived entry to the table"
        ),
    }
}

/// Per-round sumcheck message: `u_0 = u(0)`, `u_2 = u(∞)`. Middle coeff is
/// derived by the verifier from the running claim: `u_1 = T_r + u_2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundMessage {
    pub u_0: F128,
    pub u_2: F128,
}

/// Per-epoch FRI commitment: root of the folded codeword's Merkle tree.
/// (Length = `arities.len() − 1` since the last epoch is sent in plaintext.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundCommitment {
    pub root: Hash,
}

/// A single FRI query opening (multi-arity layout). Leaf payloads only; the
/// Merkle paths binding these leaves to their roots are shared across all
/// queries via [`BaseFoldProof`]'s multi-proof fields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryOpening {
    /// Random initial codeword position in `[0, 2^k_code)`.
    pub position: usize,
    /// Initial Merkle leaf: `2^log_batch_size = num_ntts` F_{2^128} values
    /// — the row-batch lanes for ONE codeword position. Verifier row-batch-
    /// folds these (using `log_batch_size` sumcheck challenges) down to a
    /// single F_{2^128} value, then cross-checks against `post_row_batch_leaf`.
    pub initial_leaf: Vec<F128>,
    /// Multi-arity post-row-batch leaf: `2^arity_0` F_{2^128} values covering
    /// `2^arity_0` consecutive post-row-batch codeword positions (including
    /// the queried one). Enables the verifier to do `arity_0` consecutive
    /// FRI folds with a single Merkle opening.
    pub post_row_batch_leaf: Vec<F128>,
    /// One entry per FRI commit (= `arities.len() − 1` entries; last epoch
    /// sends `final_codeword` in plaintext). Entry `i` is the coset of
    /// `2^arities[i+1]` F_{2^128} values committed at the end of epoch `i`,
    /// which is the input to epoch `i+1`'s arity_{i+1} folds.
    pub epoch_leaves: Vec<Vec<F128>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseFoldProof {
    /// Sumcheck round messages, length `L = log_msg_len`.
    pub round_messages: Vec<RoundMessage>,
    /// Commitment to the **post-row-batch** codeword (= initial codeword after
    /// `log_batch_size` row-batch sumcheck folds). Inserted into the transcript
    /// right after the row-batch rounds and before the first FRI round. Multi-
    /// arity leaves of size `2^arity_0` F_{2^128} support the first FRI epoch
    /// with one Merkle opening per query.
    pub post_row_batch_commit: RoundCommitment,
    /// FRI epoch commitments, length `arities.len() − 1` (last epoch
    /// plaintext).
    pub round_commitments: Vec<RoundCommitment>,
    pub final_a: F128,
    pub final_b: F128,
    /// Final codeword (length `2^log_inv_rate`, must be constant).
    pub final_codeword: Vec<F128>,
    pub queries: Vec<QueryOpening>,
    /// Octopus multi-proof for the T1 (initial) tree: shared sibling hashes
    /// covering every `queries[*].initial_leaf` against `initial_codeword_root`.
    pub initial_multi_proof: Vec<Hash>,
    /// Octopus multi-proof for the T2 (post-row-batch) tree. Empty iff
    /// `arities.is_empty()` (i.e. `log_dim == 0`).
    pub post_row_batch_multi_proof: Vec<Hash>,
    /// One octopus multi-proof per FRI commit (length = `round_commitments.len()`),
    /// aligned with `queries[*].epoch_leaves[i]`.
    pub epoch_multi_proofs: Vec<Vec<Hash>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    SumcheckFinalMismatch,
    FinalCodewordNotConstant,
    SumcheckFriMismatch,
    InitialMerkleFailed { query_index: usize },
    RoundMerkleFailed { query_index: usize, epoch: usize },
    FoldMismatch { query_index: usize, epoch: usize },
    InvalidProofShape,
}

/// One FRI fold step (matches DP24 `fold_pair`):
/// ```text
///   v += u; u += v · twiddle
///   result = u + r · (u + v)
/// ```
fn fold_pair(twiddle: F128, u_in: F128, v_in: F128, r: F128) -> F128 {
    let v = v_in + u_in;
    let u = u_in + v * twiddle;
    u + r * (u + v)
}

/// Fused row-batch fold: collapse each codeword position's `2^k` lanes down to
/// a single value using all `k` row-batch challenges (`r_0..r_{k-1}` in round
/// order) in one streaming pass. Equivalent to `k` successive per-round
/// row-batch folds, but reads the codeword once instead of `k` times — the
/// intermediate lane values never leave registers/L1, so memory traffic drops
/// from ~2× the codeword (read full + write half, ×k rounds) to ~1× (read full
/// once + write `n_positions`). Byte-identical output: each surviving lane is
/// the same nested fold of its position's input lanes.
///
/// Writes `n_positions = codeword.len() / 2^k` outputs into `out[..]`.
fn row_batch_fold_all(codeword: &[F128], out: &mut [F128], challenges: &[F128]) -> usize {
    use rayon::prelude::*;
    let num_ntts = 1usize << challenges.len();
    debug_assert_eq!(codeword.len() % num_ntts, 0);
    let n_positions = codeword.len() / num_ntts;
    // One reusable scratch buffer per parallel chunk (not per position), so the
    // hot inner fold is allocation-free regardless of `num_ntts`.
    const CHUNK: usize = 256;
    out[..n_positions]
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(ci, out_chunk)| {
            let mut buf = vec![F128::ZERO; num_ntts];
            for (k, slot) in out_chunk.iter_mut().enumerate() {
                let base = (ci * CHUNK + k) * num_ntts;
                buf.copy_from_slice(&codeword[base..base + num_ntts]);
                let mut len = num_ntts;
                for &r in challenges {
                    let half = len / 2;
                    for j in 0..half {
                        let u = buf[2 * j];
                        let v = buf[2 * j + 1];
                        buf[j] = u + r * (u + v);
                    }
                    len = half;
                }
                *slot = buf[0];
            }
        });
    n_positions
}

/// FRI fold of a single-lane codeword at the given layer + challenge.
/// Writes `new_len = codeword.len()/2` outputs into `out[..new_len]`.
fn fri_fold_codeword(
    codeword: &[F128],
    out: &mut [F128],
    ntt: &AdditiveNttF128,
    layer: usize,
    challenge: F128,
) -> usize {
    use rayon::prelude::*;
    let new_len = codeword.len() / 2;
    out[..new_len]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            let u = codeword[2 * i];
            let v = codeword[2 * i + 1];
            let twiddle = ntt.twiddle(layer, i);
            *slot = fold_pair(twiddle, u, v, challenge);
        });
    new_len
}

/// Fold one row-batch lanes-stack (length `2^a` for `a = challenges.len()`)
/// down to a single F_{2^128} via `a` row-batch folds.
fn row_batch_fold_one(lanes: &[F128], challenges: &[F128]) -> F128 {
    let mut buf = lanes.to_vec();
    for &r in challenges {
        let half = buf.len() / 2;
        let mut new_buf = Vec::with_capacity(half);
        for j in 0..half {
            let u = buf[2 * j];
            let v = buf[2 * j + 1];
            new_buf.push(u + r * (u + v));
        }
        buf = new_buf;
    }
    debug_assert_eq!(buf.len(), 1);
    buf[0]
}

/// Fold a FRI coset of `2^a` values down to one value via `a` FRI folds.
///
/// - `coset` has length `2^challenges.len()`.
/// - The coset lives at `input_layer` (so the first fold's post-fold layer is
///   `input_layer − 1`).
/// - `coset_idx` is the index of this coset within the `input_layer`-th codeword
///   divided by `2^a`. (For epoch `i` queries, `coset_idx = position >> sum_arities_through_i`.)
fn fri_fold_coset(
    coset: &[F128],
    challenges: &[F128],
    ntt: &AdditiveNttF128,
    input_layer: usize,
    coset_idx: usize,
) -> F128 {
    debug_assert_eq!(coset.len(), 1 << challenges.len());
    let mut buf = coset.to_vec();
    for (k, &r) in challenges.iter().enumerate() {
        // Post-fold layer for this fold step.
        let post_fold_layer = input_layer - k - 1;
        let n = buf.len() / 2;
        let mut new_buf = Vec::with_capacity(n);
        for j in 0..n {
            let u = buf[2 * j];
            let v = buf[2 * j + 1];
            // Position in the post-fold layer of this fold's output.
            // Coset occupies `[coset_idx * 2^(a-k-1) .. (coset_idx+1) * 2^(a-k-1))` in the post-fold layer.
            let pos = coset_idx * n + j;
            let twiddle = ntt.twiddle(post_fold_layer, pos);
            new_buf.push(fold_pair(twiddle, u, v, r));
        }
        buf = new_buf;
    }
    debug_assert_eq!(buf.len(), 1);
    buf[0]
}

/// Serialize a slice of `F128` to little-endian bytes (16 bytes per element).
fn f128_slice_to_bytes(values: &[F128]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 16);
    for f in values {
        bytes.extend_from_slice(&f.lo.to_le_bytes());
        bytes.extend_from_slice(&f.hi.to_le_bytes());
    }
    bytes
}

fn root_to_f128(root: &Hash) -> F128 {
    F128 {
        lo: u64::from_le_bytes(root[0..8].try_into().unwrap()),
        hi: u64::from_le_bytes(root[8..16].try_into().unwrap()),
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove<Ch: Challenger>(
    a_init: &[F128],
    b: Vec<F128>,
    target: F128,
    initial_codeword: &[F128],
    initial_tree: &[Hash],
    ntt: &AdditiveNttF128,
    log_inv_rate: usize,
    log_batch_size: usize,
    n_queries: usize,
    challenger: &mut Ch,
) -> BaseFoldProof {
    prove_with_precomputed_round0_prime(
        a_init,
        b,
        target,
        initial_codeword,
        initial_tree,
        ntt,
        log_inv_rate,
        log_batch_size,
        n_queries,
        None,
        challenger,
    )
}

/// Variant of [`prove`] that accepts an optional pre-computed round-0
/// sumcheck message `(u_0, u_2)`. When `Some`, basefold skips its own
/// round-0 prime computation — the caller fused it with the upstream
/// b_combined construction (see `pcs::open_batch_mixed`'s fused
/// combine + prime path).
#[allow(clippy::too_many_arguments)]
pub fn prove_with_precomputed_round0_prime<Ch: Challenger>(
    a_init: &[F128],
    mut b: Vec<F128>,
    target: F128,
    initial_codeword: &[F128],
    initial_tree: &[Hash],
    ntt: &AdditiveNttF128,
    log_inv_rate: usize,
    log_batch_size: usize,
    n_queries: usize,
    precomputed_round0_prime: Option<(F128, F128)>,
    challenger: &mut Ch,
) -> BaseFoldProof {
    assert_eq!(a_init.len(), b.len());
    assert!(a_init.len().is_power_of_two() && !a_init.is_empty());
    let log_msg_len = a_init.len().trailing_zeros() as usize;
    assert!(log_batch_size <= log_msg_len);
    let log_dim = log_msg_len - log_batch_size;
    let k_code = log_dim + log_inv_rate;
    let num_ntts = 1usize << log_batch_size;
    assert_eq!(initial_codeword.len(), (1 << k_code) * num_ntts);

    challenger.observe_label(b"flock-basefold-v0");

    let arities = crate::pcs::compute_fri_arities(log_dim);
    debug_assert_eq!(arities.iter().sum::<usize>(), log_dim);
    let num_epochs = arities.len();
    let num_fri_commits = num_epochs.saturating_sub(1);

    let mut running_target = target;
    let mut round_messages = Vec::with_capacity(log_msg_len);
    let mut round_commitments = Vec::with_capacity(num_fri_commits);
    // Row-batch challenges (r_0..r_{log_batch_size-1}) are collected across the
    // row-batch rounds and applied in a single fused fold after the last one,
    // rather than folding the codeword once per round (≈3× less traffic).
    let mut rb_challenges: Vec<F128> = Vec::with_capacity(log_batch_size);
    // The post-row-batch tree (T2) is built right after the row-batch rounds.
    // Multi-arity leaves of size 2^arity_0 give the first FRI epoch its
    // single-Merkle-open-per-query property.
    let arity_0 = arities.first().copied().unwrap_or(0);
    let post_row_batch_leaf_f128 = 1usize << arity_0;
    let mut post_row_batch_codeword: Vec<F128> = Vec::new();
    let mut post_row_batch_tree: Vec<Hash> = Vec::new();
    let mut post_row_batch_commit_root: Hash = [0u8; 32];

    // Ping-pong working buffers. Backing memory is uninitialized — basefold
    // writes to every slot before reading from it (par_iter_mut populates
    // *_scratch in round 0 from borrowed `a_init`/`initial_codeword`, then
    // mem::swap promotes scratch → active for subsequent rounds). Skipping
    // the zero-init saves ~47 ms (≈320 MB streaming write) at m=29.
    let t_alloc = std::time::Instant::now();
    let mut a_active: Vec<F128> = crate::scratch::take_f128(a_init.len());
    let mut a_scratch: Vec<F128> = crate::scratch::take_f128(a_init.len());
    let mut a_len = a_init.len();
    let mut b_scratch: Vec<F128> = crate::scratch::take_f128(b.len());
    let mut b_len = b.len();
    let mut codeword_active: Vec<F128> = crate::scratch::take_f128(initial_codeword.len());
    let mut codeword_scratch: Vec<F128> = crate::scratch::take_f128(initial_codeword.len());
    let mut cw_len = initial_codeword.len();
    let mut current_lanes = num_ntts;
    let upfront_alloc_ms = t_alloc.elapsed().as_secs_f64() * 1e3;

    // Per-FRI-commit storage for query opening: the committed codeword + tree
    // + leaf size (in F_{2^128} elements).
    let mut epoch_codewords: Vec<Vec<F128>> = Vec::with_capacity(num_fri_commits);
    let mut epoch_trees: Vec<Vec<Hash>> = Vec::with_capacity(num_fri_commits);
    let mut epoch_leaf_f128s: Vec<usize> = Vec::with_capacity(num_fri_commits);

    use rayon::prelude::*;

    // Track FRI epoch progress.
    let mut current_epoch = 0usize;
    let mut rounds_in_epoch = 0usize;

    // PCS_TRACE per-phase timing (aggregated across all rounds). Each `_ms`
    // accumulates wall time spent in that phase.
    let trace = std::env::var("PCS_TRACE").is_ok();
    let mut sumcheck_msg_ms = 0.0f64;
    let mut fold_ab_ms = 0.0f64;
    let mut row_batch_fold_ms = 0.0f64;
    let mut fri_fold_ms = 0.0f64;
    let mut post_row_batch_merkle_ms = 0.0f64;
    let mut epoch_merkle_ms = 0.0f64;

    // Prime round 0's sumcheck message from the (unfolded) inputs. Every later
    // round's message is then produced *fused* with that round's (a, b) fold:
    // folding at r_round writes exactly the operands round+1's message reads,
    // so a/b are streamed once per round instead of twice (a separate message
    // pass + a separate fold pass). The message value depends only on r_round,
    // so computing it early but observing it at the top of the next iteration
    // (after this round's Merkle-root observation) keeps the transcript — and
    // thus the proof — byte-identical.
    //
    // When `precomputed_round0_prime` is Some, the caller (pcs combine) fused
    // this with the b_combined materialization upstream — skip the redundant
    // pass.
    let t = std::time::Instant::now();
    let (mut cur_u0, mut cur_u2) = if let Some((u_0, u_2)) = precomputed_round0_prime {
        (u_0, u_2)
    } else {
        (0..b_len / 2)
            .into_par_iter()
            .map(|i| {
                let a0 = a_init[2 * i];
                let a1 = a_init[2 * i + 1];
                let b0 = b[2 * i];
                let b1 = b[2 * i + 1];
                (a0 * b0, (a0 + a1) * (b0 + b1))
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
            )
    };
    if trace {
        sumcheck_msg_ms += t.elapsed().as_secs_f64() * 1e3;
    }

    for round in 0..log_msg_len {
        let half = a_len / 2;

        // For round 0, read directly from the borrowed inputs (no clone). For
        // subsequent rounds, read from the active working buffer.
        let a_src: &[F128] = if round == 0 {
            a_init
        } else {
            &a_active[..a_len]
        };
        let b_src: &[F128] = &b[..b_len];

        // --- Observe this round's message (primed for round 0, otherwise
        // computed fused with the previous round's fold) and derive r.
        let u_0 = cur_u0;
        let u_2 = cur_u2;
        challenger.observe_f128(u_0);
        challenger.observe_f128(u_2);
        round_messages.push(RoundMessage { u_0, u_2 });

        let r = challenger.sample_f128();
        let u_1 = running_target + u_2;
        running_target = u_0 + r * u_1 + r * r * u_2;

        // --- Fused fold-at-r + next round's message. Each output *pair*
        // (a'[2j], a'[2j+1]) is folded from a_src[4j..4j+4] in one read, and
        // that pair's contribution to round+1's (u_0, u_2) is accumulated in
        // the same pass. The final round (half == 1) has no next message, so
        // it folds the lone pair directly.
        let t = std::time::Instant::now();
        if half >= 2 {
            let (n0, n2) = a_scratch[..half]
                .par_chunks_mut(2)
                .zip(b_scratch[..half].par_chunks_mut(2))
                .enumerate()
                .map(|(j, (a_out, b_out))| {
                    let base = 4 * j;
                    let a0 = a_src[base];
                    let a1 = a_src[base + 1];
                    let a2 = a_src[base + 2];
                    let a3 = a_src[base + 3];
                    let b0 = b_src[base];
                    let b1 = b_src[base + 1];
                    let b2 = b_src[base + 2];
                    let b3 = b_src[base + 3];
                    let af0 = a0 + r * (a0 + a1);
                    let af1 = a2 + r * (a2 + a3);
                    let bf0 = b0 + r * (b0 + b1);
                    let bf1 = b2 + r * (b2 + b3);
                    a_out[0] = af0;
                    a_out[1] = af1;
                    b_out[0] = bf0;
                    b_out[1] = bf1;
                    (af0 * bf0, (af0 + af1) * (bf0 + bf1))
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
                );
            cur_u0 = n0;
            cur_u2 = n2;
        } else {
            a_scratch[0] = a_src[0] + r * (a_src[0] + a_src[1]);
            b_scratch[0] = b_src[0] + r * (b_src[0] + b_src[1]);
        }
        std::mem::swap(&mut a_active, &mut a_scratch);
        std::mem::swap(&mut b, &mut b_scratch);
        a_len = half;
        b_len = half;
        if trace {
            fold_ab_ms += t.elapsed().as_secs_f64() * 1e3;
        }

        // --- Codeword fold.
        if round < log_batch_size {
            // Deferred row-batch: just record this round's challenge. The
            // codeword is folded once — all `log_batch_size` lanes at a time —
            // after the final row-batch round, so it is streamed through memory
            // once instead of once per round. The per-round folds touch no
            // transcript state (only the post-row-batch T2 root below does), so
            // deferring leaves the proof byte-identical.
            rb_challenges.push(r);
            if round + 1 == log_batch_size {
                let t = std::time::Instant::now();
                cw_len =
                    row_batch_fold_all(initial_codeword, &mut codeword_scratch, &rb_challenges);
                std::mem::swap(&mut codeword_active, &mut codeword_scratch);
                current_lanes = 1;
                if trace {
                    row_batch_fold_ms += t.elapsed().as_secs_f64() * 1e3;
                }

                // Build T2 over the post-row-batch codeword and observe its root.
                if !arities.is_empty() {
                    let t = std::time::Instant::now();
                    let cw_bytes: &[u8] = unsafe {
                        core::slice::from_raw_parts(
                            codeword_active.as_ptr() as *const u8,
                            cw_len * core::mem::size_of::<F128>(),
                        )
                    };
                    let n_leaves = cw_len / post_row_batch_leaf_f128;
                    post_row_batch_tree = merkle::merkle_tree(cw_bytes, n_leaves);
                    post_row_batch_commit_root = *post_row_batch_tree.last().expect("non-empty");
                    challenger.observe_f128(root_to_f128(&post_row_batch_commit_root));
                    post_row_batch_codeword = codeword_active[..cw_len].to_vec();
                    if trace {
                        post_row_batch_merkle_ms += t.elapsed().as_secs_f64() * 1e3;
                    }
                }
            }
        } else {
            // Round 0 reaches this branch only when log_batch_size == 0, in
            // which case it reads the (unfolded) initial codeword directly.
            let cw_src: &[F128] = if round == 0 {
                initial_codeword
            } else {
                &codeword_active[..cw_len]
            };
            debug_assert_eq!(current_lanes, 1);
            let t = std::time::Instant::now();
            let fri_round_idx = round - log_batch_size;
            let layer = k_code - fri_round_idx - 1;
            cw_len = fri_fold_codeword(cw_src, &mut codeword_scratch, ntt, layer, r);
            std::mem::swap(&mut codeword_active, &mut codeword_scratch);
            if trace {
                fri_fold_ms += t.elapsed().as_secs_f64() * 1e3;
            }

            rounds_in_epoch += 1;

            // Epoch boundary?
            if rounds_in_epoch == arities[current_epoch] {
                let is_last_epoch = current_epoch + 1 == num_epochs;
                if !is_last_epoch {
                    let t = std::time::Instant::now();
                    let next_arity = arities[current_epoch + 1];
                    let leaf_f128 = 1usize << next_arity;
                    let n_leaves = cw_len / leaf_f128;
                    let cw_bytes: &[u8] = unsafe {
                        core::slice::from_raw_parts(
                            codeword_active.as_ptr() as *const u8,
                            cw_len * core::mem::size_of::<F128>(),
                        )
                    };
                    let tree = merkle::merkle_tree(cw_bytes, n_leaves);
                    let root = *tree.last().unwrap();
                    challenger.observe_f128(root_to_f128(&root));
                    round_commitments.push(RoundCommitment { root });
                    epoch_codewords.push(codeword_active[..cw_len].to_vec());
                    epoch_trees.push(tree);
                    epoch_leaf_f128s.push(leaf_f128);
                    if trace {
                        epoch_merkle_ms += t.elapsed().as_secs_f64() * 1e3;
                    }
                }
                rounds_in_epoch = 0;
                current_epoch += 1;
            }
        }
    }

    debug_assert_eq!(a_len, 1);
    debug_assert_eq!(b_len, 1);
    let final_a = a_active[0];
    let final_b = b[0];
    let final_codeword = codeword_active[..cw_len].to_vec();

    // --- Sample query positions and gather per-tree leaf indices.
    let t_queries = std::time::Instant::now();
    let mut queries = Vec::with_capacity(n_queries);
    let initial_leaf_f128 = num_ntts;

    let mut initial_positions = Vec::with_capacity(n_queries);
    let mut post_rb_positions = Vec::with_capacity(n_queries);
    let mut epoch_positions: Vec<Vec<usize>> = (0..num_fri_commits)
        .map(|_| Vec::with_capacity(n_queries))
        .collect();

    for _ in 0..n_queries {
        let raw = challenger.sample_f128();
        let position = (raw.lo as usize) & ((1 << k_code) - 1);

        // T1 leaf (= position).
        let initial_start = position * initial_leaf_f128;
        let initial_leaf =
            initial_codeword[initial_start..initial_start + initial_leaf_f128].to_vec();
        initial_positions.push(position);

        // T2 leaf (multi-arity coset of arity_0 consecutive positions).
        let post_row_batch_leaf = if arities.is_empty() {
            Vec::new()
        } else {
            let leaf_idx = position >> arity_0;
            let start = leaf_idx * post_row_batch_leaf_f128;
            post_rb_positions.push(leaf_idx);
            post_row_batch_codeword[start..start + post_row_batch_leaf_f128].to_vec()
        };

        // Per-epoch leaves.
        let mut epoch_leaves = Vec::with_capacity(num_fri_commits);
        let mut cum_arity = arity_0;
        for i in 0..num_fri_commits {
            let p_next = position >> cum_arity;
            let leaf_f128 = epoch_leaf_f128s[i];
            let leaf_idx = p_next / leaf_f128;
            let start = leaf_idx * leaf_f128;
            epoch_leaves.push(epoch_codewords[i][start..start + leaf_f128].to_vec());
            epoch_positions[i].push(leaf_idx);
            cum_arity += arities[i + 1];
        }

        queries.push(QueryOpening {
            position,
            initial_leaf,
            post_row_batch_leaf,
            epoch_leaves,
        });
    }

    // --- Build one multi-proof per tree (shared across all queries).
    let n_initial_leaves = initial_codeword.len() / initial_leaf_f128;
    let initial_multi_proof =
        merkle::merkle_multi_proof(initial_tree, n_initial_leaves, &initial_positions);

    let post_row_batch_multi_proof = if arities.is_empty() {
        Vec::new()
    } else {
        let n_leaves = post_row_batch_codeword.len() / post_row_batch_leaf_f128;
        merkle::merkle_multi_proof(&post_row_batch_tree, n_leaves, &post_rb_positions)
    };

    let mut epoch_multi_proofs = Vec::with_capacity(num_fri_commits);
    for i in 0..num_fri_commits {
        let leaf_f128 = epoch_leaf_f128s[i];
        let n_leaves = epoch_codewords[i].len() / leaf_f128;
        epoch_multi_proofs.push(merkle::merkle_multi_proof(
            &epoch_trees[i],
            n_leaves,
            &epoch_positions[i],
        ));
    }
    let query_openings_ms = t_queries.elapsed().as_secs_f64() * 1e3;

    if trace {
        let total = upfront_alloc_ms
            + sumcheck_msg_ms
            + fold_ab_ms
            + row_batch_fold_ms
            + fri_fold_ms
            + post_row_batch_merkle_ms
            + epoch_merkle_ms
            + query_openings_ms;
        eprintln!(
            "  [basefold::prove] upfront ping-pong vec alloc:        {:6.2} ms",
            upfront_alloc_ms
        );
        eprintln!(
            "  [basefold::prove] sumcheck msg (round-0 prime only):  {:6.2} ms",
            sumcheck_msg_ms
        );
        eprintln!(
            "  [basefold::prove] fused fold+msg (all rounds):        {:6.2} ms",
            fold_ab_ms
        );
        eprintln!(
            "  [basefold::prove] row_batch_fold (rounds < {}):       {:6.2} ms",
            log_batch_size, row_batch_fold_ms
        );
        eprintln!(
            "  [basefold::prove] post-row-batch merkle (one-time):   {:6.2} ms",
            post_row_batch_merkle_ms
        );
        eprintln!(
            "  [basefold::prove] fri_fold_codeword (all FRI rounds): {:6.2} ms",
            fri_fold_ms
        );
        eprintln!(
            "  [basefold::prove] epoch merkle commits ({} epochs):    {:6.2} ms",
            num_fri_commits, epoch_merkle_ms
        );
        eprintln!(
            "  [basefold::prove] query openings ({} queries):       {:6.2} ms",
            n_queries, query_openings_ms
        );
        eprintln!(
            "  [basefold::prove] traced sum:                          {:6.2} ms",
            total
        );
    }

    // Recycle every large transient through the scratch pool. Leaving these
    // to malloc while the early-phase buffers sit in the pool would force
    // fresh page faults here each prove (see scratch.rs docs).
    crate::scratch::give_f128(a_active);
    crate::scratch::give_f128(a_scratch);
    crate::scratch::give_f128(b);
    crate::scratch::give_f128(b_scratch);
    crate::scratch::give_f128(codeword_active);
    crate::scratch::give_f128(codeword_scratch);
    crate::scratch::give_f128(post_row_batch_codeword);
    for cw in epoch_codewords {
        crate::scratch::give_f128(cw);
    }

    BaseFoldProof {
        round_messages,
        post_row_batch_commit: RoundCommitment {
            root: post_row_batch_commit_root,
        },
        round_commitments,
        final_a,
        final_b,
        final_codeword,
        queries,
        initial_multi_proof,
        post_row_batch_multi_proof,
        epoch_multi_proofs,
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// BaseFold verifier. Replays sumcheck + multi-arity FRI consistency and
/// returns the per-round sumcheck challenges so the caller (PCS) can compute
/// `final_b = b(challenges)` and match it against `proof.final_b`.
pub fn verify<Ch: Challenger>(
    target: F128,
    proof: &BaseFoldProof,
    initial_codeword_root: &Hash,
    ntt: &AdditiveNttF128,
    log_inv_rate: usize,
    log_batch_size: usize,
    challenger: &mut Ch,
) -> Result<Vec<F128>, VerifyError> {
    let log_msg_len = proof.round_messages.len();
    if log_batch_size > log_msg_len {
        return Err(VerifyError::InvalidProofShape);
    }
    let log_dim = log_msg_len - log_batch_size;
    let k_code = log_dim + log_inv_rate;
    // `k_code` is the log2 codeword length, derived from the proof-supplied
    // `round_messages` count. It feeds `1 << k_code` masks and FRI twiddle
    // lookups at layers up to `k_code - 1`. The verifier's NTT only carries
    // `log_domain_size()` layers of twiddles (it is built as
    // `standard(params.k_code())` for the committed instance), so a proof
    // claiming a larger codeword would index past the twiddle tables —
    // panicking inside `AdditiveNttF128::twiddle` — before any Merkle check
    // could reject it. Bound it up front and reject as a malformed shape.
    if k_code > ntt.log_domain_size() {
        return Err(VerifyError::InvalidProofShape);
    }
    let num_ntts = 1usize << log_batch_size;
    let arities = crate::pcs::compute_fri_arities(log_dim);
    let num_epochs = arities.len();
    let num_fri_commits = num_epochs.saturating_sub(1);

    challenger.observe_label(b"flock-basefold-v0");

    if proof.round_commitments.len() != num_fri_commits {
        return Err(VerifyError::InvalidProofShape);
    }

    // SECURITY: the number of FRI queries is a soundness parameter, not a
    // prover choice. A malicious prover that sends fewer queries (down to
    // zero) strips the codeword-to-commitment binding and can prove a false
    // evaluation. Enforce the rate-derived count before sampling positions.
    if proof.queries.len() != default_fri_queries(log_inv_rate) {
        return Err(VerifyError::InvalidProofShape);
    }

    let mut running_target = target;
    let mut challenges = Vec::with_capacity(log_msg_len);

    let mut current_epoch = 0usize;
    let mut rounds_in_epoch = 0usize;

    // Replay sumcheck rounds in lockstep with prover; observe T2 (post-row-
    // batch) commit right after the last row-batch round; observe FRI epoch-
    // boundary commits as before.
    for round in 0..log_msg_len {
        let msg = &proof.round_messages[round];
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        let r = challenger.sample_f128();
        challenges.push(r);

        let u_1 = running_target + msg.u_2;
        running_target = msg.u_0 + r * u_1 + r * r * msg.u_2;

        if round + 1 == log_batch_size && !arities.is_empty() {
            challenger.observe_f128(root_to_f128(&proof.post_row_batch_commit.root));
        }

        if round >= log_batch_size {
            rounds_in_epoch += 1;
            if rounds_in_epoch == arities[current_epoch] {
                let is_last_epoch = current_epoch + 1 == num_epochs;
                if !is_last_epoch {
                    let root = proof.round_commitments[current_epoch].root;
                    challenger.observe_f128(root_to_f128(&root));
                }
                rounds_in_epoch = 0;
                current_epoch += 1;
            }
        }
    }

    // Final sumcheck consistency.
    if proof.final_a * proof.final_b != running_target {
        return Err(VerifyError::SumcheckFinalMismatch);
    }

    // Final codeword constancy + equality with final_a.
    if proof.final_codeword.len() != 1 << log_inv_rate {
        return Err(VerifyError::FinalCodewordNotConstant);
    }
    let constant = proof.final_codeword[0];
    for &v in proof.final_codeword.iter().skip(1) {
        if v != constant {
            return Err(VerifyError::FinalCodewordNotConstant);
        }
    }
    if constant != proof.final_a {
        return Err(VerifyError::SumcheckFriMismatch);
    }

    // Resample query positions (challenger state matches prover).
    let n_queries = proof.queries.len();
    let mut positions = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let raw = challenger.sample_f128();
        positions.push((raw.lo as usize) & ((1 << k_code) - 1));
    }

    let arity_0 = arities.first().copied().unwrap_or(0);
    let initial_leaf_f128 = num_ntts; // T1: one position's row-batch lanes
    let post_row_batch_leaf_f128 = 1usize << arity_0;

    if proof.epoch_multi_proofs.len() != num_fri_commits {
        return Err(VerifyError::InvalidProofShape);
    }

    // Per-tree accumulators: leaf indices + leaf hashes, one entry per query.
    let mut initial_positions = Vec::with_capacity(n_queries);
    let mut initial_hashes = Vec::with_capacity(n_queries);
    let mut post_rb_positions = Vec::with_capacity(n_queries);
    let mut post_rb_hashes = Vec::with_capacity(n_queries);
    let mut epoch_positions: Vec<Vec<usize>> = (0..num_fri_commits)
        .map(|_| Vec::with_capacity(n_queries))
        .collect();
    let mut epoch_hashes: Vec<Vec<Hash>> = (0..num_fri_commits)
        .map(|_| Vec::with_capacity(n_queries))
        .collect();

    for (qi, q) in proof.queries.iter().enumerate() {
        if q.position != positions[qi] {
            return Err(VerifyError::FoldMismatch {
                query_index: qi,
                epoch: 0,
            });
        }
        if q.initial_leaf.len() != initial_leaf_f128 {
            return Err(VerifyError::InitialMerkleFailed { query_index: qi });
        }
        if q.epoch_leaves.len() != num_fri_commits {
            return Err(VerifyError::InvalidProofShape);
        }

        // T1: hash the initial leaf; Merkle path verified below in a batch.
        initial_positions.push(q.position);
        initial_hashes.push(merkle::hash_leaf(&f128_slice_to_bytes(&q.initial_leaf)));

        // Row-batch fold T1's lanes to a single post-row-batch F_{2^128}.
        let post_row_batch_value =
            row_batch_fold_one(&q.initial_leaf, &challenges[..log_batch_size]);

        // T2: cross-check the post-row-batch leaf; Merkle path verified in batch.
        let mut expected;
        let fri_challenge_start = log_batch_size;
        let mut cum_arity = arity_0;
        if arities.is_empty() {
            // log_dim = 0: no FRI rounds; the post-row-batch value IS the
            // final fold output.
            expected = post_row_batch_value;
        } else {
            if q.post_row_batch_leaf.len() != post_row_batch_leaf_f128 {
                return Err(VerifyError::InvalidProofShape);
            }
            let post_leaf_idx = q.position >> arity_0;
            post_rb_positions.push(post_leaf_idx);
            post_rb_hashes.push(merkle::hash_leaf(&f128_slice_to_bytes(
                &q.post_row_batch_leaf,
            )));

            // Cross-check: T2 at the queried offset within its leaf must equal
            // the row-batch fold of T1.
            let inner_offset = q.position & ((1usize << arity_0) - 1);
            if q.post_row_batch_leaf[inner_offset] != post_row_batch_value {
                return Err(VerifyError::FoldMismatch {
                    query_index: qi,
                    epoch: 0,
                });
            }

            // FRI fold T2's 2^arity_0 values via the first arity_0 FRI challenges.
            let coset_idx_in_layer = q.position >> arity_0;
            expected = fri_fold_coset(
                &q.post_row_batch_leaf,
                &challenges[fri_challenge_start..fri_challenge_start + arity_0],
                ntt,
                k_code,
                coset_idx_in_layer,
            );
        }

        // Walk through FRI commits.
        for i in 0..num_fri_commits {
            let leaf = &q.epoch_leaves[i];
            let next_arity = arities[i + 1];
            if leaf.len() != 1usize << next_arity {
                return Err(VerifyError::InvalidProofShape);
            }
            let p_at_this_layer = q.position >> cum_arity;
            let leaf_idx = p_at_this_layer >> next_arity;
            let offset = p_at_this_layer & ((1usize << next_arity) - 1);

            epoch_positions[i].push(leaf_idx);
            epoch_hashes[i].push(merkle::hash_leaf(&f128_slice_to_bytes(leaf)));

            // Check the leaf carries the expected value at the relevant offset.
            if leaf[offset] != expected {
                return Err(VerifyError::FoldMismatch {
                    query_index: qi,
                    epoch: i,
                });
            }

            // FRI fold the leaf (2^next_arity values) via next_arity challenges.
            let input_layer = k_code - cum_arity;
            let next_coset_idx = leaf_idx;
            expected = fri_fold_coset(
                leaf,
                &challenges
                    [fri_challenge_start + cum_arity..fri_challenge_start + cum_arity + next_arity],
                ntt,
                input_layer,
                next_coset_idx,
            );
            cum_arity += next_arity;
        }

        // Final check against the plaintext final codeword.
        let p_final = q.position >> cum_arity;
        if proof.final_codeword[p_final] != expected {
            return Err(VerifyError::FoldMismatch {
                query_index: qi,
                epoch: num_fri_commits,
            });
        }
    }

    // Batch-verify the three categories of Merkle paths in one shot per tree.
    if !verify_multi_with_dedup(
        initial_codeword_root,
        1usize << k_code,
        &initial_positions,
        &initial_hashes,
        &proof.initial_multi_proof,
    ) {
        return Err(VerifyError::InitialMerkleFailed { query_index: 0 });
    }
    if !arities.is_empty() {
        let n_post_rb_leaves = 1usize << (k_code - arity_0);
        if !verify_multi_with_dedup(
            &proof.post_row_batch_commit.root,
            n_post_rb_leaves,
            &post_rb_positions,
            &post_rb_hashes,
            &proof.post_row_batch_multi_proof,
        ) {
            return Err(VerifyError::InitialMerkleFailed { query_index: 0 });
        }
    }
    let mut cum_arity_check = arity_0;
    for i in 0..num_fri_commits {
        let next_arity = arities[i + 1];
        let n_leaves = 1usize << (k_code - cum_arity_check - next_arity);
        if !verify_multi_with_dedup(
            &proof.round_commitments[i].root,
            n_leaves,
            &epoch_positions[i],
            &epoch_hashes[i],
            &proof.epoch_multi_proofs[i],
        ) {
            return Err(VerifyError::RoundMerkleFailed {
                query_index: 0,
                epoch: i,
            });
        }
        cum_arity_check += next_arity;
    }

    Ok(challenges)
}

/// Batched Merkle verification with input dedup. The verifier feeds positions
/// + leaf hashes in QUERY order; this helper sorts + dedupes them (rejecting
/// any two queries that claim the same position with different leaf hashes)
/// and forwards to [`merkle::verify_merkle_multi_proof`].
fn verify_multi_with_dedup(
    root: &Hash,
    num_leaves: usize,
    positions: &[usize],
    leaf_hashes: &[Hash],
    proof: &[Hash],
) -> bool {
    if positions.len() != leaf_hashes.len() {
        return false;
    }
    if positions.is_empty() {
        return proof.is_empty();
    }
    let mut paired: Vec<(usize, Hash)> = positions
        .iter()
        .copied()
        .zip(leaf_hashes.iter().copied())
        .collect();
    paired.sort_by_key(|(p, _)| *p);
    let mut deduped: Vec<(usize, Hash)> = Vec::with_capacity(paired.len());
    for (p, h) in paired {
        if let Some(last) = deduped.last()
            && last.0 == p {
                if last.1 != h {
                    return false;
                }
                continue;
            }
        deduped.push((p, h));
    }
    let positions_sorted: Vec<usize> = deduped.iter().map(|(p, _)| *p).collect();
    let hashes_sorted: Vec<Hash> = deduped.iter().map(|(_, h)| *h).collect();
    merkle::verify_merkle_multi_proof(root, num_leaves, &positions_sorted, &hashes_sorted, proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::ntt::AdditiveNttF128;

    /// A proof whose `round_messages` count implies a codeword dimension larger
    /// than the verifier's NTT must be rejected with `InvalidProofShape`, not
    /// panic in the FRI twiddle lookup or the `1 << k_code` mask. Regression
    /// test for the unbounded proof-controlled `k_code` (audit item: verifier
    /// robustness against malformed proofs).
    #[test]
    fn verify_rejects_oversized_codeword_dim_without_panic() {
        let log_inv_rate = 1;
        let log_batch_size = 0;
        // NTT supports only a 4-layer domain (log_domain_size = 4).
        let ntt = AdditiveNttF128::standard(4);
        assert_eq!(ntt.log_domain_size(), 4);

        // round_messages.len() = 10  =>  log_dim = 10, k_code = 11 > 4.
        // Every other field is left empty: the dimension bound must fire before
        // they are ever inspected, so an attacker can't even reach the indexing.
        let proof = BaseFoldProof {
            round_messages: vec![
                RoundMessage {
                    u_0: F128::ZERO,
                    u_2: F128::ZERO,
                };
                10
            ],
            post_row_batch_commit: RoundCommitment { root: [0u8; 32] },
            round_commitments: Vec::new(),
            final_a: F128::ZERO,
            final_b: F128::ZERO,
            final_codeword: Vec::new(),
            queries: Vec::new(),
            initial_multi_proof: Vec::new(),
            post_row_batch_multi_proof: Vec::new(),
            epoch_multi_proofs: Vec::new(),
        };

        let mut ch = FsChallenger::new(b"basefold-oob-test");
        let res = verify(
            F128::ZERO,
            &proof,
            &[0u8; 32],
            &ntt,
            log_inv_rate,
            log_batch_size,
            &mut ch,
        );
        assert_eq!(res, Err(VerifyError::InvalidProofShape));
    }
}
