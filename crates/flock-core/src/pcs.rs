//! Polynomial commitment scheme for the bit-MLE witness `ẑ` over GF(2).
//!
//! Construction: Binius-style PCS with F_{2^128} packing.
//!
//! - **Commit**: pack the 2^m Boolean witness into 2^(m−7) F_{2^128} elements
//!   (one bit per polynomial-basis coordinate of F_{2^128}), batch RS-encode
//!   via additive NTT, Merkle-commit the codeword.
//! - **Open**: at a QuirkyPoint (z_skip, x_outer) from the zerocheck/lincheck:
//!   1. [`ring_switch::prove`] sends 128 partial-evaluations `s_hat_v` and
//!      produces a sumcheck target `(rs_eq_ind, sumcheck_claim)`.
//!   2. [`ligerito::recursive_prover_with_basis`] discharges the combined
//!      claim `⟨packed_witness, b_combined⟩ = target_combined` via the
//!      recursive Ligerito argument, reusing the commit-time codeword and
//!      Merkle tree as Ligerito's L0 commitment.
//! - **Verify**: the verifier replays ring-switching succinctly, then drives
//!   the succinct recursive Ligerito verifier, evaluating the combined basis
//!   at the residual point (see [`verify_opening_batch_ligerito_mixed`]).
//!
//! See [DP24](https://eprint.iacr.org/2024/504) (ring-switching) and the
//! ligerito module docs for the recursion.

pub mod commit;
pub mod jagged;
pub mod ligerito;
pub mod pack;
pub mod ring_switch;
pub mod tensor_algebra;

pub use commit::{
    Commitment, PcsParams, ProverData, commit, commit_into, commit_lane_major, dense_lanes,
    prefault_codeword_during,
};
pub use pack::{LOG_PACKING, pack_witness, unpack_witness};
pub use ring_switch::{RingSwitchProof, SparseEqTensor};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;
use crate::merkle::cap_depth;
use crate::merkle::cap_layer;
#[cfg(test)]
use crate::pcs::ligerito::ProverConfig;
#[cfg(test)]
use crate::pcs::ligerito::VerifierConfig;
#[cfg(test)]
use crate::pcs::ligerito::udr_queries;
use crate::pcs::tensor_algebra::TensorAlgebra;
use crate::scratch::give_f128;
use crate::scratch::take_f128;
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::multilinear::eq_eval_binary_x;
use serde::{Deserialize, Serialize};
use std::env::var;
use std::mem::swap;
use std::time::Instant;

/// Batched opening proof: ring-switching frontend + Ligerito backend.
/// The combined `b_combined` + target_combined feed
/// [`ligerito::recursive_prover_with_basis`] (see ligerito module docs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchOpeningProofLigerito {
    pub ring_switches: Vec<RingSwitchProof>,
    pub ligerito: ligerito::LigeritoProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    RingSwitch(ring_switch::VerifyError),
    /// The Ligerito recursive verifier rejected the proof.
    Ligerito,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyErrorOpen {
    RingSwitch(ring_switch::VerifyError),
    /// The virtual-opening sumcheck rejected (wrong round count, or the final
    /// round does not match `b̂_combined(ρ) · f_eval`).
    VirtualOpen,
    /// The assist layer rejected (the Frobenius-assist replay failed), or a
    /// claim-shape mismatch reached the merged open.
    Assist,
    /// The Ligerito recursive verifier rejected the opening (also raised by
    /// the merged verifier's commitment-params equality check).
    Ligerito,
}

/// `eq_ind` representation for a packed-direct claim. The contributed value at
/// scattered index `j` is the tensor entry — for the dense variant the index
/// is the array offset; for the sparse variant it's reconstructed via
/// [`SparseEqTensor::scatter_idx`].
#[derive(Clone, Debug)]
pub enum DirectEqInd {
    /// Fully-materialized `eq_ind(point)` of length `2^L`.
    Dense(Vec<F128>),
    /// Deferred eq tensor: only the point rides in. Two roles:
    ///
    /// - The merged transport's inner open (a single such claim, no RS
    ///   claims): the combine materializes `γ·eq(point)` directly as
    ///   `b_combined` with a seeded build (no separate eq buffer, no
    ///   re-scale pass).
    /// - The inert carrier for OUTER merged claims and the
    ///   pre-materialization form on the union prover's jagged arm:
    ///   [`open_batch_merged`] derives its weights from `point`/`value`
    ///   alone and never reads `eq_ind`, so deferred claims ride through it
    ///   unbuilt (a claim that WOULD need a materialized tensor trips the
    ///   "EqPoint claims are only supported alone" assert in the combine
    ///   rather than dropping the contribution).
    ///
    /// Transcript-identical to `Dense` of the same point in every role —
    /// the representation is prover-side only.
    EqPoint(Vec<F128>),
    /// Sparse representation — non-zero entries at scattered indices.
    /// Built from a claim point with one or more exactly-zero coords via
    /// [`ring_switch::build_eq_sparse`].
    Sparse(SparseEqTensor),
}

/// A packed-MLE evaluation claim: `ẑ_packed(point) = value`. Unlike a
/// ring-switched claim, this is opened directly without going through the
/// bit-MLE ↔ packed-MLE bridge (no `s_hat_v`, no φ_8 weighting).
///
/// Use case: protocols whose sumcheck output is naturally a packed-MLE
/// evaluation (e.g. the chain shift sumcheck operating on packed columns
/// instead of bit-folded scalars). Skips the ring-switch step for this claim,
/// saving the `fold_1b_rows` + per-opening-tail work at the prover and the
/// ring-switch verify + φ_8 reconstruction at the verifier.
///
/// The claim-combine step adds `γ_k · eq_ind(point)` to `b_combined` and
/// `γ_k · value` to the target; the verifier's residual check contributes
/// `γ_k · eq_eval(point, residual_challenges)`.
#[derive(Clone, Debug)]
pub struct PackedDirectClaim {
    /// Multilinear point of length `L = m − 7`.
    pub point: Vec<F128>,
    /// Claimed `ẑ_packed(point)` value.
    pub value: F128,
    /// `eq_ind(point)` in dense or sparse form. Caller responsibility to
    /// match the claim's `point` — the contribution to `b_combined` is read
    /// directly from this tensor.
    pub eq_ind: DirectEqInd,
}

/// Mixed-claim batched open: supports both **ring-switched** claims (bit-MLE
/// openings reduced via `ring_switch::prove_batched`, with optional per-claim
/// precomputed `s_hat_v`) and **packed-direct** claims (packed-MLE openings
/// that skip ring-switch). Runs the ring_switch + b_combined computation, then
/// routes to [`ligerito::recursive_prover_with_basis`] using the existing
/// `prover_data`'s codeword + tree as Ligerito's L0 commit (no L0 re-commit).
///
/// `lig_config.initial_k` must equal `commitment.params.log_batch_size` so that
/// `prover_data`'s codeword/tree shape matches what Ligerito expects for L0.
#[allow(clippy::too_many_arguments)]
pub fn open_batch_mixed_ligerito_with_precomputed_s_hat_v<Ch: Challenger>(
    packed_witness: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    lig_config: &ligerito::ProverConfig,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    // Belt-and-braces on the cap-depth derivation: the commit-time cap
    // (from `PcsParams::l0_cap_depth`) must be the layer the opener's
    // config implies — a config-source disagreement fails loudly here at
    // prove time instead of as a verifier reject.
    assert_eq!(
        commitment.cap.len(),
        1usize << cap_depth(lig_config.queries[0], commitment.params.k_code(),),
        "commitment cap size disagrees with the opener config's L0 query count"
    );
    debug_assert_eq!(
        commitment.cap.as_slice(),
        cap_layer(
            &prover_data.merkle_tree,
            commitment.params.n_leaves(),
            cap_depth(lig_config.queries[0], commitment.params.k_code()),
        ),
        "commitment cap is not the prover tree's cap layer"
    );
    let trace = var("PCS_TRACE").is_ok();
    let t_total = Instant::now();

    assert_eq!(
        lig_config.initial_k, commitment.params.log_batch_size,
        "ligerito initial_k ({}) must match PcsParams.log_batch_size ({}) for L0 reuse",
        lig_config.initial_k, commitment.params.log_batch_size,
    );
    assert_eq!(
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
        "ligerito log_inv_rates[0] ({}) must match PcsParams.log_inv_rate ({}) for L0 reuse",
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
    );

    // Integer-lane (lane-major) commitments are supported ONLY in the
    // merged transport's inner-open configuration (no RS claims, one
    // EqPoint claim): the eq basis folds honestly over the zero-padding
    // lanes (live-block skip disabled) and the round-0 prime pairs blocks.
    let l0_num_lanes = commitment.params.num_ntts();
    let lane_major = l0_num_lanes < 1usize << lig_config.initial_k;
    let log_n = commitment.params.m - LOG_PACKING;
    if lane_major {
        assert!(
            x_outers.is_empty()
                && packed_direct.len() == 1
                && matches!(packed_direct[0].eq_ind, DirectEqInd::EqPoint(_)),
            "lane-major mixed open: only the merged inner-open configuration is supported"
        );
    }
    let round0_block = if lane_major {
        1usize << (log_n - lig_config.initial_k)
    } else {
        1
    };
    let combined = compute_combined_basis_and_target(
        &packed_witness,
        x_outers,
        precomputed_s_hat_v,
        packed_direct,
        padding,
        round0_block,
        challenger,
        trace,
    );

    let t = Instant::now();
    let ligerito_proof = ligerito::recursive_prover_with_basis_precomputed_round0_lanes(
        lig_config,
        packed_witness,
        combined.b_combined,
        combined.target_combined,
        &prover_data.codeword,
        &prover_data.merkle_tree,
        l0_num_lanes,
        lane_major,
        combined.round0_prime,
        challenger,
    );
    if trace {
        eprintln!(
            "  [open_batch] ligerito::recursive_prover_with_basis: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        eprintln!(
            "  [open_batch] TOTAL: {:6.2} ms",
            t_total.elapsed().as_secs_f64() * 1e3
        );
    }

    BatchOpeningProofLigerito {
        ring_switches: combined.ring_switches,
        ligerito: ligerito_proof,
    }
}

/// What ring_switch + claim-combination produces, fed to the Ligerito backend.
struct CombinedClaim {
    ring_switches: Vec<RingSwitchProof>,
    /// The materialized γ-combined basis — EMPTY when `deferred` is `Some`
    /// (the streaming path never writes the full-domain array).
    b_combined: Vec<F128>,
    target_combined: F128,
    /// Round-0 sumcheck `(u_0, u_2)` prime over `packed_witness · b_combined`,
    /// consumed by `recursive_prover_with_basis_precomputed_round0`.
    round0_prime: (F128, F128),
}

/// Runs ring_switch over RS claims, observes packed-direct claim values +
/// samples their gammas, then builds `b_combined` (the γ-weighted linear
/// combination of all `rs_eq_ind`s and `eq_ind`s) and `target_combined`.
/// Also computes the round-0 prime as a side effect (cheap since it shares
/// the b_combined pass).
///
/// (The M6 support-proportional `live_pairs`/`stream_b` machinery lived
/// here for the jagged transport's virtual-opening sumcheck and was removed
/// with it; the dead-block skip below is the surviving support-awareness.)
#[allow(clippy::too_many_arguments)]
fn compute_combined_basis_and_target<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    // Round-0 pairing block for the EqPoint special path: 1 = adjacent
    // elements (pow2 lanes); 2^(log_n − initial_k) under a lane-major
    // commitment, whose L0 fold pairs BLOCKS. Other paths must pass 1
    // (their primes are adjacent-paired; the jagged path re-derives its
    // blocked prime downstream).
    eqpoint_round0_block: usize,
    challenger: &mut Ch,
    trace: bool,
) -> CombinedClaim {
    use rayon::prelude::*;
    let n_rs = x_outers.len();
    let n_pd = packed_direct.len();
    assert!(n_rs + n_pd > 0, "open_batch_mixed: need at least one claim");
    assert!(
        precomputed_s_hat_v.is_empty() || precomputed_s_hat_v.len() == n_rs,
        "precomputed_s_hat_v: must be empty or length {n_rs}, got {}",
        precomputed_s_hat_v.len(),
    );

    challenger.observe_label(b"flock-pcs-open-batch-v0");

    // 1. Ring-switching for all x_outers.
    let t = Instant::now();
    let (rs_results, gammas_rs): (
        Vec<(RingSwitchProof, ring_switch::RingSwitchBatchOutput)>,
        Vec<F128>,
    ) = if n_rs > 0 {
        ring_switch::prove_batched_padded_with_precomputed(
            packed_witness,
            x_outers,
            precomputed_s_hat_v,
            padding,
            challenger,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    if trace {
        eprintln!(
            "  [open_batch] ring_switch::prove_batched ×{}: {:6.2} ms",
            n_rs,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 2. Observe packed-direct claim values + sample γ_pd.
    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas_pd: Vec<F128> = (0..n_pd).map(|_| challenger.sample_f128()).collect();

    let t = Instant::now();

    let l = if let Some((_, out)) = rs_results.first() {
        out.rs_eq_ind.len()
    } else {
        1usize << packed_direct[0].point.len()
    };
    debug_assert!(rs_results.iter().all(|(_, o)| o.rs_eq_ind.len() == l));
    debug_assert!(
        packed_direct.iter().all(|pd| 1usize << pd.point.len() == l),
        "all packed-direct claims must share L (= packed witness length)"
    );

    let mut target_combined = F128::ZERO;
    for ((_, output), g) in rs_results.iter().zip(gammas_rs.iter()) {
        target_combined += *g * output.sumcheck_claim;
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target_combined += *g * pd.value;
    }

    let rs_baked: Vec<&[F128]> = rs_results
        .iter()
        .filter_map(|(_, o)| match &o.rs_eq_ind {
            ring_switch::RsEqInd::Dense(v) => Some(v.as_slice()),
            _ => None,
        })
        .collect();
    // Deferred-dense claims (fused fast path): the per-claim `γ_k·B_k` buffer
    // was never materialized — fold each slot on the fly below and accumulate
    // straight into `b_combined`, saving a 2^(m-7) materialize + readback per
    // claim. Carries (eq_lo, eq_hi, γ-baked table, log₂ B).
    let rs_deferred: Vec<(&[F128], &[F128], &[F128], usize)> = rs_results
        .iter()
        .filter_map(|(_, o)| match &o.rs_eq_ind {
            ring_switch::RsEqInd::DeferredDense {
                eq_lo,
                eq_hi,
                table,
            } => Some((
                eq_lo.as_slice(),
                eq_hi.as_slice(),
                table.as_slice(),
                eq_lo.len().trailing_zeros() as usize,
            )),
            _ => None,
        })
        .collect();
    let pd_dense: Vec<(&[F128], F128)> = packed_direct
        .iter()
        .zip(gammas_pd.iter())
        .filter_map(|(pd, g)| match &pd.eq_ind {
            DirectEqInd::Dense(v) => Some((v.as_slice(), *g)),
            _ => None,
        })
        .collect();

    // Merged-transport inner open: no RS claims, exactly one EqPoint claim.
    // Build `b_combined = γ·eq(point)` seeded (one pass, no eq buffer) and
    // fuse the round-0 prime. Transcript-identical to the Dense variant of
    // the same point (the eq_ind representation is prover-side only).
    if rs_results.is_empty()
        && packed_direct.len() == 1
        && let DirectEqInd::EqPoint(point) = &packed_direct[0].eq_ind
    {
        assert_eq!(1usize << point.len(), l, "EqPoint length mismatch");
        let b_combined = ring_switch::build_eq_scaled_parallel(point, gammas_pd[0]);
        let blk = eqpoint_round0_block;
        let (round0_u0, round0_u2) = if blk == 1 {
            const C: usize = 1 << 13;
            packed_witness
                .par_chunks(C)
                .zip(b_combined.par_chunks(C))
                .map(|(qc, wc)| {
                    let mut a = F128::ZERO;
                    let mut b = F128::ZERO;
                    for (qp, wp) in qc
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(wc.as_chunks::<2>().0.iter())
                    {
                        a += qp[0] * wp[0];
                        b += (qp[0] + qp[1]) * (wp[0] + wp[1]);
                    }
                    (a, b)
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
                )
        } else {
            // Lane-major round-0: the L0 fold pairs BLOCKS of `blk`
            // elements (lane 2j with lane 2j+1), so the prime pairs
            // element t of block 2j with element t of block 2j+1.
            assert!(blk.is_power_of_two() && l.is_multiple_of(2 * blk));
            (0..l / (2 * blk))
                .into_par_iter()
                .map(|j| {
                    let b0 = 2 * j * blk;
                    let b1 = b0 + blk;
                    let mut u0 = F128::ZERO;
                    let mut u2 = F128::ZERO;
                    for t in 0..blk {
                        let (q0, q1) = (packed_witness[b0 + t], packed_witness[b1 + t]);
                        let (w0, w1) = (b_combined[b0 + t], b_combined[b1 + t]);
                        u0 += q0 * w0;
                        u2 += (q0 + q1) * (w0 + w1);
                    }
                    (u0, u2)
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
                )
        };
        if trace {
            eprintln!(
                "  [open_batch] combine (seeded EqPoint, L={l}): {:6.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        return CombinedClaim {
            ring_switches: Vec::new(),
            b_combined,
            target_combined,
            round0_prime: (round0_u0, round0_u2),
        };
    }

    // Past the special path, EqPoint must not appear — the loops below
    // would silently drop its contribution.
    assert!(
        packed_direct
            .iter()
            .all(|pd| !matches!(pd.eq_ind, DirectEqInd::EqPoint(_))),
        "EqPoint claims are only supported alone, with no RS claims"
    );

    // Fast path (compression-proof open: claims ab, c; also chain/merkle): every
    // RS claim is a fused DeferredDense fold and no DENSE packed-direct claim
    // needs the per-element combine. Fold all claims block-by-block straight into
    // b_combined — each claim's `e_hi` hoisted once per block, exactly as in
    // `fold_b128_elems_split` — and fuse the round-0 prime in the same pass.
    // Neither the per-claim `γ_k·B_k` buffer nor a combine readback is ever
    // materialized (saves ~2·L writes + 2·L reads of the 2^(m-7) basis).
    //
    // SPARSE packed-direct claims (the chain/merkle I/O claim) do NOT disable
    // this path: they're scatter-added onto b_combined after the fold (with an
    // incremental round-0 prime adjustment), so they only require
    // `pd_dense.is_empty()`, not `packed_direct.is_empty()`. This keeps the two
    // big ab/c claims on the fused fold instead of materializing them.
    let use_fast =
        !rs_deferred.is_empty() && rs_deferred.len() == rs_results.len() && pd_dense.is_empty();

    // ---- Build b_combined (γ-weighted sum of all rs_eq_ind + eq_ind) and the
    //      round-0 prime (u_0, u_2 over packed_witness · b_combined).
    let mut b_combined: Vec<F128> = take_f128(l);

    let (mut round0_u0, mut round0_u2) = if use_fast {
        let b = rs_deferred[0].0.len(); // eq_lo.len(); shared across claims (same split)
        debug_assert!(b >= 2 && b.is_multiple_of(2));
        debug_assert!(rs_deferred.iter().all(|d| d.0.len() == b));
        b_combined
            .par_chunks_mut(b)
            .enumerate()
            .map(|(hi, out_block)| {
                // DEAD-BLOCK SKIP. A claim's `e_hi = eq_hi[hi]` is exactly
                // `F128::ZERO` on every block whose high address bits fall
                // outside the claim's own support — which is what a claim point
                // with FROZEN high coordinates produces (`build_eq` of a
                // zero coord kills half the tensor). When EVERY claim's factor
                // vanishes on a block, so does `b_combined` there:
                // `fold_one_slot(0, table) = Σ_k table[k·256 + 0]`, and
                // `build_fold_byte_table` fills `value = 0` with the empty sum,
                // so it is exactly zero. Storing zeros directly is therefore
                // bit-identical to folding, and skips 16 table lookups +
                // 15 XORs + a GF multiply per word.
                //
                // This is where the multi-table design's DISJOINTNESS pays: the
                // boolean class's claim points carry `M − M_bool` frozen-zero
                // high coordinates, so their basis vanishes on the whole
                // element region AND the inter-class gap — half the address
                // space or more. The element class's claims are `Sparse` and
                // only ever touch their own subcube (scatter-added below), so
                // after this skip the combine costs `O(2^M_bool + element
                // support)` rather than `O(2^M)`.
                //
                // Zeros still have to be STORED, not skipped: the
                // virtual-opening sumcheck's round-0 fold reads `b_combined`
                // densely when the witness support is not sparse.
                if rs_deferred.iter().all(|d| d.1[hi] == F128::ZERO) {
                    out_block.fill(F128::ZERO);
                    // The prime's terms here are `a·0`, i.e. zero.
                    return (F128::ZERO, F128::ZERO);
                }
                // Accumulate each claim's block: first claim writes, rest add.
                // `e_hi` is read once per claim per block, then swept over eq_lo.
                for (ci, (eq_lo, eq_hi, table, _)) in rs_deferred.iter().enumerate() {
                    let e_hi = eq_hi[hi];
                    if ci == 0 {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot = ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    } else {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot += ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    }
                }
                // Round-0 prime over this block's pairs (b is even, base is
                // even).
                let base = hi * b;
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
                for t in 0..(b / 2) {
                    let s0 = out_block[2 * t];
                    let s1 = out_block[2 * t + 1];
                    let a0 = packed_witness[base + 2 * t];
                    let a1 = packed_witness[base + 2 * t + 1];
                    u0 += a0 * s0;
                    u2 += (a0 + a1) * (s0 + s1);
                }
                (u0, u2)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
            )
    } else {
        // General path (mixed / sparse / packed-direct): materialize any
        // deferred-dense claims (parallel block fold), then the per-element
        // combine over all dense buffers + packed-direct, matching the
        // original behavior.
        let materialized: Vec<Vec<F128>> = rs_results
            .iter()
            .filter_map(|(_, o)| match &o.rs_eq_ind {
                ring_switch::RsEqInd::DeferredDense {
                    eq_lo,
                    eq_hi,
                    table,
                } => Some(ring_switch::fold_b128_from_table(eq_lo, eq_hi, table)),
                _ => None,
            })
            .collect();
        let mut rs_dense_all: Vec<&[F128]> = rs_baked.clone();
        rs_dense_all.extend(materialized.iter().map(|v| v.as_slice()));
        if rs_dense_all.is_empty() && pd_dense.is_empty() {
            // All-sparse open (e.g. the permutation check's five pinned-coord
            // claims): the per-element loop below would write zeros into every
            // slot and accumulate `a·0` into the prime. Skip straight to the
            // memset — the sparse scatter-adds that follow supply the whole of
            // `b_combined`, and the prime they fold in is the whole prime.
            b_combined.par_chunks_mut(1 << 13).for_each(|c| {
                c.fill(F128::ZERO);
            });
            (F128::ZERO, F128::ZERO)
        } else {
            let prime = b_combined
                .par_chunks_mut(2)
                .enumerate()
                .map(|(i, chunk)| {
                    let mut b0 = F128::ZERO;
                    let mut b1 = F128::ZERO;
                    for v in rs_dense_all.iter() {
                        b0 += v[2 * i];
                        b1 += v[2 * i + 1];
                    }
                    for (v, g) in pd_dense.iter() {
                        b0 += *g * v[2 * i];
                        b1 += *g * v[2 * i + 1];
                    }
                    chunk[0] = b0;
                    chunk[1] = b1;
                    let a0 = packed_witness[2 * i];
                    let a1 = packed_witness[2 * i + 1];
                    (a0 * b0, (a0 + a1) * (b0 + b1))
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
                );
            for v in materialized {
                give_f128(v);
            }
            prime
        }
    };
    let mut adjust_prime_for_delta = |idx: usize, delta: F128| {
        let pair = idx / 2;
        let a0 = packed_witness[2 * pair];
        let a1 = packed_witness[2 * pair + 1];
        if idx & 1 == 0 {
            round0_u0 += a0 * delta;
        }
        round0_u2 += (a0 + a1) * delta;
    };
    for (_, output) in rs_results.iter() {
        if let ring_switch::RsEqInd::Sparse { entries, .. } = &output.rs_eq_ind {
            for &(idx, val) in entries {
                b_combined[idx] += val;
                adjust_prime_for_delta(idx, val);
            }
        }
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        if let DirectEqInd::Sparse(eq) = &pd.eq_ind {
            // Scatter-add the sparse claim and fold its round-0 prime
            // contribution in the SAME pass (O(live positions)), instead of a
            // full O(L) re-pass over b_combined. The prime is linear in
            // b_combined, so the delta from scattering `g·eq` equals
            // Σ adjust_prime_for_delta(idx, g·val) over the live positions.
            let (du0, du2) = sparse_scatter_add_parallel(&mut b_combined, packed_witness, eq, *g);
            round0_u0 += du0;
            round0_u2 += du2;
        }
    }
    if trace {
        eprintln!(
            "  [open_batch] combine rs_eq_ind (L={}, rs×{}, pd×{}): {:6.2} ms",
            l,
            n_rs,
            n_pd,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let ring_switches = rs_results
        .into_iter()
        .map(|(p, o)| {
            // The per-claim rs_eq_ind (L F128s) dies here — recycle it.
            if let ring_switch::RsEqInd::Dense(v) = o.rs_eq_ind {
                give_f128(v);
            }
            p
        })
        .collect();
    CombinedClaim {
        ring_switches,
        b_combined,
        target_combined,
        round0_prime: (round0_u0, round0_u2),
    }
}

/// Parallel sparse scatter-add: `b_combined[scatter_idx(c)] += gamma * eq.live_tensor[c]`
/// for every `c`. Partitions `c`-space across rayon threads; since
/// [`SparseEqTensor::scatter_idx`] is monotonic in `c` (live_positions sorted
/// ascending), each thread's scattered indices fall in a contiguous, disjoint
/// range of `b_combined`. Splits `b_combined` at the chunk boundaries via
/// `split_at_mut`, then writes scatter-adds into the disjoint mutable slices —
/// safe rust, no atomics.
/// Scatter-add `gamma · eq` into `b_combined` and return the resulting
/// round-0 prime delta `(Δu0, Δu2)`. Because the prime is linear in
/// `b_combined`, adding `delta = gamma·val` at index `idx` changes the prime by
/// `Δu0 += a0·delta` (if `idx` even) and `Δu2 += (a0+a1)·delta`, where
/// `a0 = packed_witness[2·pair]`, `a1 = packed_witness[2·pair+1]`,
/// `pair = idx/2`. Computing it here (O(live positions)) avoids a full O(L)
/// re-pass over `b_combined` at the call site.
fn sparse_scatter_add_parallel(
    b_combined: &mut [F128],
    packed_witness: &[F128],
    eq: &SparseEqTensor,
    gamma: F128,
) -> (F128, F128) {
    use rayon::prelude::*;

    let c_total = eq.live_tensor.len();
    if c_total == 0 {
        return (F128::ZERO, F128::ZERO);
    }
    let n_threads = rayon::current_num_threads().max(1);
    let c_per_chunk = c_total.div_ceil(n_threads).max(1);
    let actual_n_chunks = c_total.div_ceil(c_per_chunk);

    // Boundaries in `b_combined` index space. `b_boundaries[i]` is where chunk
    // `i` starts. `b_boundaries[i+1] − b_boundaries[i]` is chunk `i`'s slice
    // length. The last chunk extends to `b_combined.len()` to absorb any tail
    // positions beyond the maximum scatter idx (those contain only dense
    // contributions from the parallel pass).
    let b_boundaries: Vec<usize> = (0..=actual_n_chunks)
        .map(|i| {
            if i == 0 {
                0
            } else if i == actual_n_chunks {
                b_combined.len()
            } else {
                eq.scatter_idx(i * c_per_chunk)
            }
        })
        .collect();
    debug_assert!(b_boundaries.windows(2).all(|w| w[0] <= w[1]));

    // Disjoint mutable slices via repeated split_at_mut.
    let mut remaining: &mut [F128] = b_combined;
    let mut slices: Vec<&mut [F128]> = Vec::with_capacity(actual_n_chunks);
    for i in 1..actual_n_chunks {
        let split_at = b_boundaries[i] - b_boundaries[i - 1];
        let (left, right) = remaining.split_at_mut(split_at);
        slices.push(left);
        remaining = right;
    }
    slices.push(remaining);
    debug_assert_eq!(slices.len(), actual_n_chunks);

    slices
        .into_par_iter()
        .enumerate()
        .map(|(t, slice)| {
            let c_lo = t * c_per_chunk;
            let c_hi = ((t + 1) * c_per_chunk).min(c_total);
            let b_lo = b_boundaries[t];
            let mut du0 = F128::ZERO;
            let mut du2 = F128::ZERO;
            for c in c_lo..c_hi {
                let val = eq.live_tensor[c];
                let idx = eq.scatter_idx(c);
                let delta = gamma * val;
                slice[idx - b_lo] += delta;
                // Round-0 prime delta for this scattered position.
                let pair = idx / 2;
                let a0 = packed_witness[2 * pair];
                let a1 = packed_witness[2 * pair + 1];
                if idx & 1 == 0 {
                    du0 += a0 * delta;
                }
                du2 += (a0 + a1) * delta;
            }
            (du0, du2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        )
}

/// Verifier reference to a packed-direct claim: the multilinear point at
/// which `ẑ_packed` was claimed equal to `value`. The verifier owns the data
/// (it appears in the public statement of whatever produced the claim, e.g.
/// the chain shift sumcheck output).
#[derive(Clone, Copy, Debug)]
pub struct PackedDirectClaimRef<'a> {
    pub point: &'a [F128],
    pub value: F128,
}

/// Verify a mixed-claim batched opening (mirror of
/// [`open_batch_mixed_ligerito_with_precomputed_s_hat_v`]). Uses
/// `ring_switch::verify_succinct` per claim (no dense `rs_eq_ind`
/// materialization), then drives the succinct recursive Ligerito verifier,
/// evaluating the combined basis only at the residual point.
#[allow(clippy::too_many_arguments)]
pub fn verify_opening_batch_ligerito_mixed<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    z_skips: &[F128],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    proof: &BatchOpeningProofLigerito,
    lig_config: &ligerito::VerifierConfig,
    challenger: &mut Ch,
) -> Result<(), VerifyError> {
    let n_rs = claims.len();
    let n_pd = packed_direct.len();
    assert_eq!(z_skips.len(), n_rs);
    assert_eq!(x_outers.len(), n_rs);
    assert_eq!(proof.ring_switches.len(), n_rs);
    assert!(n_rs + n_pd > 0);
    // Lane-major (integer-lane) commitments: supported only for the merged
    // inner-open configuration (packed-direct claims only). The RS claims'
    // residual machinery has no rotated-point path; reject rather than
    // mis-evaluate. (The claim shapes are statement-derived, so this is an
    // assert, not an attacker-reachable rejection.)
    let lane_major = commitment.params.num_ntts() < 1usize << lig_config.initial_k;
    assert!(
        !lane_major || n_rs == 0,
        "lane-major mixed verify: packed-direct claims only"
    );

    challenger.observe_label(b"flock-pcs-open-batch-v0");

    // 1. Ring-switch SUCCINCT verify per claim — gets sumcheck_claim and a
    //    length-128 `eq_r_dprime` instead of the dense `rs_eq_ind`. Saves
    //    ~16 MB allocation at m=29.
    let mut rs_outputs = Vec::with_capacity(n_rs);
    for i in 0..n_rs {
        let out = ring_switch::verify_succinct(
            claims[i],
            z_skips[i],
            x_outers[i],
            &proof.ring_switches[i],
            challenger,
        )
        .map_err(VerifyError::RingSwitch)?;
        rs_outputs.push(out);
    }
    let gammas_rs: Vec<F128> = (0..n_rs).map(|_| challenger.sample_f128()).collect();

    // 2. PD claim values + γ_pd.
    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas_pd: Vec<F128> = (0..n_pd).map(|_| challenger.sample_f128()).collect();

    // 3. target_combined from succinct rs claims + PD values.
    let mut target_combined = F128::ZERO;
    for (out, g) in rs_outputs.iter().zip(gammas_rs.iter()) {
        target_combined += *g * out.sumcheck_claim;
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target_combined += *g * pd.value;
    }

    // 4. Batch evaluator: returns b_combined at all yr positions in one call.
    //    For RS claims, precompute the ring_switch tensor PREFIX once (over
    //    the ris part) and only re-do the yr_log_n-step suffix per y.
    //    For PD claims, precompute eq prefix factors over ris and finish per y.
    //    For BLAKE3 m=30: ris is 19 dims, yr is 4 dims → 19× prefix reuse.
    let log_n = commitment.params.m - LOG_PACKING;
    let eval_b_residual = |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
        // ---- Per-y assembly (parallel over yr positions; each y is independent).
        //      y_suffix is binary (bits of y), so we use the binary-query
        //      specializations of eval_rs_eq_finish / eq_eval — each suffix
        //      step collapses to a single scale_vertical / scalar product.
        use crate::zerocheck::multilinear::eq_eval;
        use rayon::prelude::*;
        let yr_len = 1usize << yr_log_n;
        let prefix_len = ris.len();

        // ---- RS claim prefixes ----
        let rs_prefixes: Vec<TensorAlgebra> = rs_outputs
            .iter()
            .zip(x_outers.iter())
            .map(|(_out, x_outer)| {
                // x_outer[1..] has length log_n; we feed only the ris prefix.
                ring_switch::eval_rs_eq_prefix(&x_outer[1..1 + prefix_len], ris)
            })
            .collect();

        // ---- PD claim prefix scalars ----
        // eq(pd.point, point) factors over coordinates; precompute the prefix
        // product. Under a lane-major commitment the fold challenges bind the
        // ROTATED variable order (lane vars — the high dense vars — first),
        // so pair them against the correspondingly rotated claim point.
        let pd_points_rot: Vec<Vec<F128>> = packed_direct
            .iter()
            .map(|pd| {
                if lane_major {
                    rotate_lane_point(pd.point, pd.point.len() - lig_config.initial_k)
                } else {
                    pd.point.to_vec()
                }
            })
            .collect();
        let pd_prefix_scalars: Vec<F128> = pd_points_rot
            .iter()
            .map(|pt| eq_eval(&pt[..prefix_len], ris))
            .collect();

        debug_assert!(yr_log_n <= 32, "yr_log_n > 32 not supported by binary path");
        (0..yr_len)
            .into_par_iter()
            .map(|y| {
                let y_bits = y as u32;
                let mut sum = F128::ZERO;
                for (((out, g), x_outer), prefix) in rs_outputs
                    .iter()
                    .zip(gammas_rs.iter())
                    .zip(x_outers.iter())
                    .zip(rs_prefixes.iter())
                {
                    sum += *g
                        * ring_switch::eval_rs_eq_finish_from_prefix_binary_q(
                            prefix,
                            &x_outer[1 + prefix_len..],
                            y_bits,
                            &out.eq_r_dprime,
                        );
                }
                for ((pt, g), prefix_scalar) in pd_points_rot
                    .iter()
                    .zip(gammas_pd.iter())
                    .zip(pd_prefix_scalars.iter())
                {
                    sum += *g * *prefix_scalar * eq_eval_binary_x(&pt[prefix_len..], y_bits);
                }
                sum
            })
            .collect()
    };

    // 5. Drive ligerito SUCCINCT verifier — eval_b_residual is called ONCE
    //    at the residual check (returns all yr_len values in one batch).
    let ok = ligerito::recursive_verifier_with_basis_succinct(
        lig_config,
        &proof.ligerito,
        log_n,
        target_combined,
        &commitment.cap,
        commitment.params.num_ntts(),
        eval_b_residual,
        challenger,
    );
    if !ok {
        return Err(VerifyError::Ligerito);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
/// Map a point over the LANE-GRID variables to the corresponding point over
/// the dense-stack variables, for a high-bit-lane commit.
///
/// The grid is `q_grid[p·2^k + l] = q[l·D + p]` (`k = initial_k`), i.e.
/// `q_grid`'s low `k` variables are the lane bits that live at the TOP of
/// `q`'s index. As multilinears that is a pure cyclic rotation of the variable
/// vector, so `q̂_grid(x) = q̂(x_k, …, x_{m−1}, x_0, …, x_{k−1})` — rotate the
/// evaluation point left by `k`.
fn rotate_lane_point(point: &[F128], k: usize) -> Vec<F128> {
    debug_assert!(k <= point.len());
    let mut out = Vec::with_capacity(point.len());
    out.extend_from_slice(&point[k..]);
    out.extend_from_slice(&point[..k]);
    out
}

// ───────────────────────────────────────────────────────────────────────────
// The MERGED jagged/ring-switch opening (design doc §"Capacity-free
// ring-switching: a merged reduction") — THE transport since wire v6 (the
// unmerged jagged transport was removed): ONE sumcheck over the DENSE
// domain replaces the padded-domain virtual-opening sumcheck AND the
// jagged transport — the weight `W[d] = Σ_i Φ_i(eq_row·eq_col at the
// unrank of d)` is simultaneously the ring-switch weight and the
// dense/virtual translation, so the prover's Φ-pass is count-proportional
// (capacity-free). The verifier's twisted-weight evaluation `Ŵ(ρ)` is
// discharged by the batched Frobenius assist, and `q̂(ρ)` by an ordinary
// eq-basis Ligerito opening (a packed-direct claim on the mixed path).
// Handles both power-of-two and integer-lane commitments.
// ───────────────────────────────────────────────────────────────────────────

/// Proof of the merged opening. `merged_rounds` are the dense-domain
/// sumcheck's `(G(1), G(∞))` messages (`m_dense − 7` rounds, LSB first);
/// `q_eval = q̂(ρ)`; the Frobenius assist carries and proves the twisted
/// evaluation `V = Ŵ(ρ)`; `inner` is the eq-basis Ligerito opening of
/// `q̂(ρ)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedOpenProof {
    pub ring_switches: Vec<RingSwitchProof>,
    pub merged_rounds: Vec<(F128, F128)>,
    pub q_eval: F128,
    pub frobenius: jagged::MultipointTwistedProof,
    pub inner: BatchOpeningProofLigerito,
}

/// γ-combine the packed-direct claims into merged-column scalar groups,
/// keyed by shared row point in first-occurrence order: `Σᵢ γᵢ·eq_rowᵢ(row)·
/// eq_colᵢ(col) = eq_row(row)·(Σᵢ γᵢ·eq_colᵢ(col))`, so a group costs ONE
/// sweep wherever it is consumed — the merged `W` build, the multipoint
/// protocol (one dual value per group), and the anchor assist. Built
/// identically on both sides (the prover's claim order is the verifier's).
///
/// A Boolean column point's eq table is a one-hot indicator, so its
/// γ-contribution is a single scatter — the gather claims' column parts are
/// exactly this (`bits(word_col) ‖ bits(slot_prefix)`), and building a
/// `2^k_cols` table per claim is `k_cols`-exponential waste (~2^20 per
/// claim at MVP-5's composite registry). Random column points (the element
/// claims) keep the dense build. Value-identical either way.
fn scalar_claim_groups<'a>(
    points: impl Iterator<Item = &'a [F128]>,
    gammas: &[F128],
    n_log: usize,
    k_cols: usize,
) -> Vec<(&'a [F128], Vec<F128>)> {
    let mut groups: Vec<(&[F128], Vec<F128>)> = Vec::new();
    for (point, &g) in points.zip(gammas) {
        assert_eq!(
            point.len(),
            n_log + k_cols,
            "packed-direct point/row/col split mismatch (no skip coordinate)"
        );
        let (zr, zc) = (&point[..n_log], &point[n_log..]);
        let hot: Option<usize> = zc.iter().enumerate().try_fold(0usize, |acc, (i, &x)| {
            if x == F128::ZERO {
                Some(acc)
            } else if x == F128::ONE {
                Some(acc | (1 << i))
            } else {
                None
            }
        });
        let cols = match groups.iter_mut().find(|(r, _)| *r == zr) {
            Some((_, cols)) => cols,
            None => {
                groups.push((zr, vec![F128::ZERO; 1 << k_cols]));
                &mut groups.last_mut().unwrap().1
            }
        };
        match hot {
            Some(h) => cols[h] += g,
            None => {
                for (dst, e) in cols.iter_mut().zip(build_eq_table(zc)) {
                    *dst += g * e;
                }
            }
        }
    }
    groups
}

#[allow(clippy::too_many_arguments)]
pub fn open_batch_merged<Ch: Challenger>(
    q: Vec<F128>,
    padded_witness: &[F128],
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    heights: &[u64],
    n_log: usize,
    lig_config: &ligerito::ProverConfig,
    challenger: &mut Ch,
) -> MergedOpenProof {
    let trace = var("PCS_TRACE").is_ok();
    let t_total = Instant::now();
    // Belt-and-braces on the cap-depth derivation: the commit-time cap
    // (from `PcsParams::l0_cap_depth`) must be the layer the opener's
    // config implies — a config-source disagreement fails loudly here at
    // prove time instead of as a verifier reject.
    assert_eq!(
        commitment.cap.len(),
        1usize << cap_depth(lig_config.queries[0], commitment.params.k_code(),),
        "commitment cap size disagrees with the opener config's L0 query count"
    );
    debug_assert_eq!(
        commitment.cap.as_slice(),
        cap_layer(
            &prover_data.merkle_tree,
            commitment.params.n_leaves(),
            cap_depth(lig_config.queries[0], commitment.params.k_code()),
        ),
        "commitment cap is not the prover tree's cap layer"
    );
    challenger.observe_label(b"flock-merged-open-v0");
    let t = Instant::now();
    // Element-only registries produce no ring-switched claims; skip the batch
    // entirely (the callee asserts a non-empty batch). This branch DEFINES the
    // element-only merged transcript: nothing is absorbed for the empty batch,
    // exactly as in the mixed open's `n_rs > 0` guard.
    let (rs_results, gammas_rs) = if !x_outers.is_empty() {
        ring_switch::prove_batched_padded_with_precomputed(
            padded_witness,
            x_outers,
            precomputed_s_hat_v,
            padding,
            challenger,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    if trace {
        eprintln!(
            "  [open_merged] ring_switch: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // Packed-direct claims take their γ's from the same stream, AFTER the
    // ring-switched ones — the verifier draws them in the same order.
    let gammas_pd: Vec<F128> = packed_direct
        .iter()
        .map(|c| {
            challenger.observe_f128_slice(&c.point);
            challenger.observe_f128(c.value);
            challenger.sample_f128()
        })
        .collect();

    let mut target = F128::ZERO;
    for ((_, o), g) in rs_results.iter().zip(&gammas_rs) {
        target += *g * o.sumcheck_claim;
    }
    for (c, g) in packed_direct.iter().zip(&gammas_pd) {
        target += *g * c.value;
    }

    let dense_log = commitment.params.m - LOG_PACKING;
    assert_eq!(
        q.len(),
        1usize << dense_log,
        "q must be the committed stack"
    );
    let params = jagged::JaggedParams::from_heights(heights, n_log, dense_log);
    let k_cols = params.k;
    // Packed-direct claims are γ-SCALAR maps (`x ↦ γ·x`), so claims sharing
    // a row point collapse into merged-column scalar groups, built ONCE and
    // consumed by the weight builder, the multipoint protocol, and the
    // anchor ([`scalar_claim_groups`]). A packed-direct point carries NO
    // univariate-skip coordinate, which is the one place the splits differ.
    let pd_groups = scalar_claim_groups(
        packed_direct.iter().map(|c| c.point.as_slice()),
        &gammas_pd,
        n_log,
        k_cols,
    );
    // Ring-switched claims enter the weight builder with their F₂-linear
    // fold tables; the scalar groups with their precombined column tables —
    // `Σᵢ γᵢ·eq_colᵢ` — instead of a per-claim fold-table sweep
    // (bit-identical W; see `MergedWeightClaim::Scalar`). The circuit
    // path's gather claims all share ρ_row, so its ~2^c claims cost one
    // sweep.
    let mut weight_claims: Vec<jagged::MergedWeightClaim<'_>> = rs_results
        .iter()
        .zip(x_outers.iter())
        .map(|((_, o), x)| {
            assert_eq!(x.len(), 1 + n_log + k_cols, "point/row/col split mismatch");
            let table = match &o.rs_eq_ind {
                ring_switch::RsEqInd::DeferredDense { table, .. } => table.as_slice(),
                _ => panic!("merged open requires DeferredDense ring-switch claims"),
            };
            jagged::MergedWeightClaim::Folded {
                z_row: &x[1..1 + n_log],
                z_col: &x[1 + n_log..],
                table,
            }
        })
        .collect();
    weight_claims.extend(
        pd_groups
            .iter()
            .map(|(z_row, cols)| jagged::MergedWeightClaim::Scalar { z_row, cols }),
    );

    // The twisted weight over the dense cube (count-proportional Φ-pass;
    // zero tail past the jagged area).
    let t = Instant::now();
    let (w, (u0, u2)) = jagged::build_merged_weight_and_prime(&params, &weight_claims, &q);
    if trace {
        eprintln!(
            "  [open_merged] W build + round-0 prime (2^{} words): {:6.2} ms",
            dense_log,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Merged sumcheck: Σ_d q[d]·W[d] = target, dense_log rounds, same
    // message/fold conventions as the virtual-opening sumcheck.
    let t = Instant::now();
    let (mut g_one, mut g_inf) = (target + u0, u2);
    let mut merged_rounds = Vec::with_capacity(dense_log);
    let mut rho = Vec::with_capacity(dense_log);
    let l = q.len();
    let mut sa = take_f128(l / 2);
    let mut sb = take_f128(l / 2);
    let mut a = take_f128(l / 4);
    let mut bb = take_f128(l / 4);
    let mut cur = l;
    for round in 0..dense_log {
        let half = cur / 2;
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        merged_rounds.push((g_one, g_inf));
        rho.push(r);
        let (a_src, b_src): (&[F128], &[F128]) = if round == 0 {
            (q.as_slice(), w.as_slice())
        } else {
            (&a, &bb)
        };
        if cur > 2 {
            (g_one, g_inf) = jagged::fold_and_round_oop_par(
                &a_src[..cur],
                &b_src[..cur],
                r,
                &mut sa[..half],
                &mut sb[..half],
            );
        } else {
            jagged::fold_oop_par(
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
    let q_eval = if dense_log == 0 { q[0] } else { a[0] };
    let w_eval = if dense_log == 0 { w[0] } else { bb[0] };
    if trace {
        eprintln!(
            "  [open_merged] merged sumcheck ({dense_log} rounds): {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    give_f128(w);
    give_f128(sa);
    give_f128(sb);
    give_f128(a);
    give_f128(bb);

    // ---- Frobenius assist: proves V = Ŵ(ρ).
    let t = Instant::now();
    let coeffs: Vec<Vec<F128>> = rs_results
        .iter()
        .map(|(_, o)| match &o.rs_eq_ind {
            ring_switch::RsEqInd::DeferredDense { table, .. } => {
                ring_switch::linearized_coefficients(table)
            }
            _ => unreachable!("checked above"),
        })
        .collect();
    let fclaims: Vec<jagged::FrobeniusClaim<'_>> = x_outers
        .iter()
        .zip(&coeffs)
        .map(|(x, c)| jagged::FrobeniusClaim {
            z_row: &x[1..1 + n_log],
            z_col: &x[1 + n_log..],
            coeffs: c,
        })
        .collect();
    // The packed-direct claims enter as the merged-column scalar groups:
    // one untwisted dual value per group instead of 128 per claim.
    let gclaims: Vec<jagged::ScalarGroupClaim<'_>> = pd_groups
        .iter()
        .map(|(z_row, cols)| jagged::ScalarGroupClaim { z_row, cols })
        .collect();
    if trace {
        eprintln!(
            "    [frobenius] linearized_coefficients (x{}): {:6.2} ms",
            coeffs.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_assist = Instant::now();
    // The two-product multipoint replacement
    // (docs/multipoint-twisted-assist.tex): 128R + P claimed dual values +
    // one two-product sumcheck + ONE untwisted anchor, instead of a
    // per-statement assist — family K collapses, and every verifier piece
    // is a shape the recursion circuit already has.
    let frobenius = jagged::prove_multipoint_twisted(&params, &fclaims, &gclaims, &rho, challenger);
    if trace {
        eprintln!(
            "  [open_merged] coeffs + multipoint assist: {:6.2} ms (assist alone {:6.2} ms)",
            t.elapsed().as_secs_f64() * 1e3,
            t_assist.elapsed().as_secs_f64() * 1e3
        );
    }
    #[cfg(debug_assertions)]
    {
        let mut v = F128::ZERO;
        for (cl, vs) in fclaims.iter().zip(&frobenius.values) {
            for (j, &a) in vs.iter().enumerate() {
                let mut t = a;
                for _ in 0..j {
                    t *= t;
                }
                v += cl.coeffs[j] * t;
            }
        }
        for &b in &frobenius.group_values {
            v += b;
        }
        debug_assert_eq!(v, w_eval, "multipoint V must equal the folded weight MLE");
    }
    let _ = w_eval;

    // ---- eq-basis Ligerito opening of q̂(ρ): one packed-direct claim on
    // the existing mixed path (whose verifier evaluates eq residuals in
    // closed form — no b_tilde machinery).
    let pd = PackedDirectClaim {
        point: rho.clone(),
        value: q_eval,
        eq_ind: DirectEqInd::EqPoint(rho.clone()),
    };
    let t = Instant::now();
    let inner = open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        q,
        prover_data,
        commitment,
        &[],
        &[],
        &[pd],
        &PaddingSpec::dense(commitment.params.m),
        lig_config,
        challenger,
    );

    if trace {
        eprintln!(
            "  [open_merged] inner eq-basis open: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        eprintln!(
            "  [open_merged] TOTAL: {:6.2} ms",
            t_total.elapsed().as_secs_f64() * 1e3
        );
    }
    drop(fclaims);
    MergedOpenProof {
        ring_switches: rs_results.into_iter().map(|(p, _)| p).collect(),
        merged_rounds,
        q_eval,
        frobenius,
        inner,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_batch_merged<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    z_skips: &[F128],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    heights: &[u64],
    n_log: usize,
    proof: &MergedOpenProof,
    lig_config: &ligerito::VerifierConfig,
    challenger: &mut Ch,
) -> Result<(), VerifyErrorOpen> {
    let n_rs = claims.len();
    assert_eq!(z_skips.len(), n_rs);
    assert_eq!(x_outers.len(), n_rs);
    if proof.ring_switches.len() != n_rs {
        return Err(VerifyErrorOpen::Assist);
    }
    // `VERIFY_TRACE` phase split. The Ligerito inner verify has its own
    // `LIG_VERIFY_TRACE`, but it is a small tail here — the jagged Frobenius
    // assist is the term that scales with the row/column split.
    let trace = var("VERIFY_TRACE").is_ok();
    let tfmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };
    challenger.observe_label(b"flock-merged-open-v0");
    let t = Instant::now();
    let mut rs_outputs = Vec::with_capacity(n_rs);
    for i in 0..n_rs {
        let out = ring_switch::verify_succinct(
            claims[i],
            z_skips[i],
            x_outers[i],
            &proof.ring_switches[i],
            challenger,
        )
        .map_err(VerifyErrorOpen::RingSwitch)?;
        rs_outputs.push(out);
    }
    if trace {
        eprintln!(
            "        [vbm] ring_switch::verify_succinct ×{n_rs}: {}",
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    let gammas: Vec<F128> = (0..n_rs).map(|_| challenger.sample_f128()).collect();
    // Packed-direct γ's follow the ring-switched ones, in the prover's order.
    let gammas_pd: Vec<F128> = packed_direct
        .iter()
        .map(|c| {
            challenger.observe_f128_slice(c.point);
            challenger.observe_f128(c.value);
            challenger.sample_f128()
        })
        .collect();
    let mut target = F128::ZERO;
    for (out, g) in rs_outputs.iter().zip(&gammas) {
        target += *g * out.sumcheck_claim;
    }
    for (c, g) in packed_direct.iter().zip(&gammas_pd) {
        target += *g * c.value;
    }

    let dense_log = commitment.params.m - LOG_PACKING;
    if proof.merged_rounds.len() != dense_log {
        return Err(VerifyErrorOpen::VirtualOpen);
    }
    let mut running = target;
    let mut rho = Vec::with_capacity(dense_log);
    for &(g_one, g_inf) in &proof.merged_rounds {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        running = jagged::fold_round_claim(running, g_one, g_inf, r);
        rho.push(r);
    }

    let t = Instant::now();
    let params = jagged::JaggedParams::from_heights(heights, n_log, dense_log);
    let k_cols = params.k;
    if trace {
        eprintln!(
            "        [vbm] JaggedParams::from_heights (n_log={n_log}, dense_log={dense_log}, k_cols={k_cols}): {}",
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    let t = Instant::now();
    // The claims' c_{i,j}: derived from the transcript (γ-scaled r''-eq
    // tensors → fold byte tables → linearized coefficients).
    let coeffs: Vec<Vec<F128>> = rs_outputs
        .iter()
        .zip(&gammas)
        .map(|(o, g)| {
            let scaled: Vec<F128> = o.eq_r_dprime.iter().map(|x| *g * *x).collect();
            ring_switch::linearized_coefficients(&ring_switch::build_fold_byte_table(&scaled))
        })
        .collect();
    let fclaims: Vec<jagged::FrobeniusClaim<'_>> = x_outers
        .iter()
        .zip(&coeffs)
        .map(|(x, c)| {
            assert_eq!(x.len(), 1 + n_log + k_cols, "point/row/col split mismatch");
            jagged::FrobeniusClaim {
                z_row: &x[1..1 + n_log],
                z_col: &x[1 + n_log..],
                coeffs: c,
            }
        })
        .collect();
    // The packed-direct claims collapse into the SAME merged-column scalar
    // groups the prover built (identical claim order → identical groups):
    // a group's fold map is the identity with its γ's baked into the cols,
    // so it carries one dual value and no 128-coefficient vector at all.
    for c in packed_direct.iter() {
        if c.point.len() != n_log + k_cols {
            return Err(VerifyErrorOpen::Assist);
        }
    }
    let pd_groups = scalar_claim_groups(
        packed_direct.iter().map(|c| c.point),
        &gammas_pd,
        n_log,
        k_cols,
    );
    let gclaims: Vec<jagged::ScalarGroupClaim<'_>> = pd_groups
        .iter()
        .map(|(z_row, cols)| jagged::ScalarGroupClaim { z_row, cols })
        .collect();
    if trace {
        eprintln!(
            "        [vbm] coeffs (fold byte tables ×{n_rs}) + {} scalar groups: {}",
            gclaims.len(),
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    let t = Instant::now();
    #[cfg(feature = "mul-count")]
    let assist_start = crate::field::gf2_128::op_count::snapshot();
    let v = jagged::verify_multipoint_twisted(
        &params,
        &fclaims,
        &gclaims,
        &rho,
        &proof.frobenius,
        challenger,
    )
    .ok_or(VerifyErrorOpen::Assist)?;
    #[cfg(feature = "mul-count")]
    if std::env::var("MUL_TRACE").is_ok() {
        let e = crate::field::gf2_128::op_count::snapshot();
        let invs = e.invs - assist_start.invs;
        let muls = (e.native_muls - assist_start.native_muls)
            .saturating_sub(invs * crate::field::gf2_128::op_count::MULS_PER_INV);
        println!(
            "  [mul]   of which jagged::verify_frobenius_assist:    {muls:>8} muls {invs:>5} invs \
             = {:>8} constraints",
            muls + invs
        );
    }
    if trace {
        eprintln!(
            "        [vbm] jagged::verify_frobenius_assist: {}",
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    if running != proof.q_eval * v {
        return Err(VerifyErrorOpen::VirtualOpen);
    }

    let t = Instant::now();
    let pd = PackedDirectClaimRef {
        point: &rho,
        value: proof.q_eval,
    };
    verify_opening_batch_ligerito_mixed(
        commitment,
        &[],
        &[],
        &[],
        &[pd],
        &proof.inner,
        lig_config,
        challenger,
    )
    .map_err(|_| VerifyErrorOpen::Ligerito)?;
    if trace {
        eprintln!(
            "        [vbm] verify_opening_batch_ligerito_mixed: {}",
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::zerocheck::multilinear::lagrange_weights_naive;
    use crate::zerocheck::univariate_skip::build_eq;

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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    fn zhat_skip_reference(z: &[bool], m: usize, z_skip: F128, x_outer: &[F128]) -> F128 {
        const K_SKIP: usize = 6;
        let ell = 1usize << K_SKIP;
        let lambda = lagrange_weights_naive(K_SKIP, z_skip);
        let eq_outer = build_eq(x_outer);
        let mut acc = F128::ZERO;
        for i_outer in 0..(1usize << (m - K_SKIP)) {
            let base = i_outer * ell;
            let mut inner = F128::ZERO;
            for i_skip in 0..ell {
                if z[base + i_skip] {
                    inner += lambda[i_skip];
                }
            }
            acc += eq_outer[i_outer] * inner;
        }
        acc
    }

    /// End-to-end Ligerito backend roundtrip through pcs::open_batch_mixed_ligerito
    /// and verify_opening_batch_ligerito_mixed. Single ring-switched claim
    /// (no PD — PD path is task #11).
    #[test]
    #[ignore] // Heavier — ~50-100 ms; run with `cargo test pcs_ligerito_roundtrip -- --ignored --nocapture`
    fn pcs_ligerito_backend_roundtrip() {
        let m = 22usize;
        let mut rng = Rng::new(0x11_6E_2170);
        let z = rng.bits(1 << m);
        let z_skip = rng.f128();
        let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
        let rs_claim = zhat_skip_reference(&z, m, z_skip, &x_outer);

        // PcsParams MUST set log_batch_size = ligerito_initial_k for L0 reuse.
        let initial_k = 6;
        let params = PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: initial_k,
            profile: Default::default(),
            num_lanes: None,
            merkle_hash: Default::default(),
        };
        let z_packed = pack_witness(&z, m);
        let (commitment, prover_data) = commit(&z_packed, &params);

        let recursive_ks = vec![3usize, 3, 3];
        let log_inv_rates = vec![1usize, 3, 4, 6];
        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![0usize; log_inv_rates.len()];
        let n_levels = log_inv_rates.len();
        let lig_p_cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: (m - LOG_PACKING) - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![6, 3, 0],
            recursive_ks: recursive_ks.clone(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; n_levels],
            ood_samples: vec![0; n_levels],
            merkle_hash: Default::default(),
        };
        let lig_v_cfg = VerifierConfig {
            log_inv_rates,
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: (m - LOG_PACKING) - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![6, 3, 0],
            recursive_ks,
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; n_levels],
            ood_samples: vec![0; n_levels],
            merkle_hash: Default::default(),
        };

        let mut ch_p = FsChallenger::new(b"flock-test-lig-v0");
        let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v(
            z_packed.clone(),
            &prover_data,
            &commitment,
            &[x_outer.as_slice()],
            &[],
            &[],
            &PaddingSpec::dense(m),
            &lig_p_cfg,
            &mut ch_p,
        );

        let mut ch_v = FsChallenger::new(b"flock-test-lig-v0");
        verify_opening_batch_ligerito_mixed(
            &commitment,
            &[rs_claim],
            &[z_skip],
            &[x_outer.as_slice()],
            &[],
            &proof,
            &lig_v_cfg,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("ligerito verify rejected honest proof: {e:?}"));
    }
}
