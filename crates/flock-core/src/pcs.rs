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
    Commitment, PcsParams, ProverData, commit, commit_into, prefault_codeword_during,
};
pub use pack::{LOG_PACKING, pack_witness, unpack_witness};
pub use ring_switch::{RingSwitchProof, SparseEqTensor};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::PaddingSpec;
use serde::{Deserialize, Serialize};

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

/// Batched opening proof for the **jagged transport** path (the
/// three-polynomial pipeline of `docs/multi-table-design.tex` §"The
/// commitment layer"): the ring-switching frontend exactly as
/// [`BatchOpeningProofLigerito`], then the virtual-opening sumcheck
/// converting the γ-combined inner-product claim into a single evaluation
/// claim `f̂(ρ) = f_eval`, and the **fused** Ligerito opening discharging
/// `⟨q, W_ρ⟩ = f_eval` directly against the jagged weight table
/// `W_ρ = f̂_t(ρ_row, ρ_col, ·)` as the basis (no jagged main sumcheck). The
/// weight-table evaluations Ligerito's final check needs at `ris ‖ bits(y)`
/// are prover-supplied (`b_tilde`) and bound to the true `Ŵ_ρ` by a
/// send-and-spot-check reduction: a fresh challenge `r_extra` is squeezed
/// after `b_tilde`, and the (unmodified) jagged assist proves
/// `Ŵ_ρ(ris ‖ r_extra)`, which must equal the MLE of the received `b_tilde`
/// at `r_extra` (Schwartz–Zippel). Produced by [`open_batch_jagged_ligerito`],
/// checked by [`verify_opening_batch_jagged_ligerito`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchOpeningProofJaggedLigerito {
    pub ring_switches: Vec<RingSwitchProof>,
    /// Virtual-opening sumcheck round messages `(G(1), G(∞))` — one per
    /// packed-word variable (`m − 7` rounds, LSB bound first).
    pub virtual_open_rounds: Vec<(F128, F128)>,
    /// `f̂(ρ)` — the packed witness folded at the virtual-opening challenges.
    pub f_eval: F128,
    pub ligerito: ligerito::LigeritoProof,
    /// `b_tilde[y] = Ŵ_ρ(ris ‖ bits(y))` for all `y ∈ {0,1}^yr_log_n`, where
    /// `ris` are Ligerito's fold challenges — the basis evaluations its final
    /// check consumes, sent by the prover and spot-checked via the assist.
    pub b_tilde: Vec<F128>,
    pub jagged_assist: jagged::JaggedAssistProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyErrorJagged {
    RingSwitch(ring_switch::VerifyError),
    /// The virtual-opening sumcheck rejected (wrong round count, or the final
    /// round does not match `b̂_combined(ρ) · f_eval`).
    VirtualOpen,
    /// The jagged transport rejected: the assist replay failed, or the
    /// assist-verified `β = Ŵ_ρ(ris ‖ r_extra)` does not match the MLE of the
    /// proof's `b_tilde` at `r_extra` (the spot-check binding `b_tilde` to the
    /// true weight table).
    Jagged,
    /// The Ligerito recursive verifier rejected the fused opening (including
    /// its final check, which consumes the proof's `b_tilde`).
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
    let trace = std::env::var("PCS_TRACE").is_ok();
    let t_total = std::time::Instant::now();

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

    let combined = compute_combined_basis_and_target(
        &packed_witness,
        x_outers,
        precomputed_s_hat_v,
        packed_direct,
        padding,
        challenger,
        trace,
    );

    let t = std::time::Instant::now();
    let ligerito_proof = ligerito::recursive_prover_with_basis_precomputed_round0(
        lig_config,
        packed_witness,
        combined.b_combined,
        combined.target_combined,
        &prover_data.codeword,
        &prover_data.merkle_tree,
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
#[allow(clippy::too_many_arguments)]
fn compute_combined_basis_and_target<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    challenger: &mut Ch,
    trace: bool,
) -> CombinedClaim {
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
    let t = std::time::Instant::now();
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

    let t = std::time::Instant::now();
    use rayon::prelude::*;

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

    // ---- Build b_combined (γ-weighted sum of all rs_eq_ind + eq_ind) and the
    //      round-0 prime (u_0, u_2 over packed_witness · b_combined).
    let mut b_combined: Vec<F128> = crate::scratch::take_f128(l);

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

    let (mut round0_u0, mut round0_u2) = if use_fast {
        let b = rs_deferred[0].0.len(); // eq_lo.len(); shared across claims (same split)
        debug_assert!(b >= 2 && b.is_multiple_of(2));
        debug_assert!(rs_deferred.iter().all(|d| d.0.len() == b));
        b_combined
            .par_chunks_mut(b)
            .enumerate()
            .map(|(hi, out_block)| {
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
                // Round-0 prime over this block's pairs (b is even, base is even).
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
            crate::scratch::give_f128(v);
        }
        prime
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

    CombinedClaim {
        ring_switches: rs_results
            .into_iter()
            .map(|(p, o)| {
                // The per-claim rs_eq_ind (L F128s) dies here — recycle it.
                if let ring_switch::RsEqInd::Dense(v) = o.rs_eq_ind {
                    crate::scratch::give_f128(v);
                }
                p
            })
            .collect(),
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
        use crate::zerocheck::multilinear::eq_eval;
        let yr_len = 1usize << yr_log_n;
        let prefix_len = ris.len();

        // ---- RS claim prefixes ----
        let rs_prefixes: Vec<crate::pcs::tensor_algebra::TensorAlgebra> = rs_outputs
            .iter()
            .zip(x_outers.iter())
            .map(|(_out, x_outer)| {
                // x_outer[1..] has length log_n; we feed only the ris prefix.
                ring_switch::eval_rs_eq_prefix(&x_outer[1..1 + prefix_len], ris)
            })
            .collect();

        // ---- PD claim prefix scalars ----
        // eq(pd.point, point) factors over coordinates; precompute the prefix product.
        let pd_prefix_scalars: Vec<F128> = packed_direct
            .iter()
            .map(|pd| eq_eval(&pd.point[..prefix_len], ris))
            .collect();

        // ---- Per-y assembly (parallel over yr positions; each y is independent).
        //      y_suffix is binary (bits of y), so we use the binary-query
        //      specializations of eval_rs_eq_finish / eq_eval — each suffix
        //      step collapses to a single scale_vertical / scalar product.
        use rayon::prelude::*;
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
                for ((pd, g), prefix_scalar) in packed_direct
                    .iter()
                    .zip(gammas_pd.iter())
                    .zip(pd_prefix_scalars.iter())
                {
                    sum += *g
                        * *prefix_scalar
                        * crate::zerocheck::multilinear::eq_eval_binary_x(
                            &pd.point[prefix_len..],
                            y_bits,
                        );
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
        &commitment.root,
        eval_b_residual,
        challenger,
    );
    if !ok {
        return Err(VerifyError::Ligerito);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The jagged opening path (Phase 1 of docs/multi-table-design.tex §"The
// commitment layer"): claim assembly exactly as the mixed path, then
// virtual-opening sumcheck → fused Ligerito opening against the jagged weight
// table W_ρ → b_tilde send-and-spot-check (with the unmodified assist).
// Additive — the mixed path above is untouched.
// ---------------------------------------------------------------------------

/// Mixed-claim batched open through the **jagged transport**. Runs the exact
/// claim assembly of [`open_batch_mixed_ligerito_with_precomputed_s_hat_v`]
/// (ring-switch batched prove + γ-combination — transcript-identical up to and
/// including the combined claim), then:
///
/// 1. **Virtual-opening sumcheck** (`flock-virtual-open-v0`): a product
///    sumcheck over the `m − 7` packed-word variables proving
///    `Σ_x f(x)·b_combined(x) = target_combined` (`f` = packed witness),
///    with the char-2-safe `(G(1), G(∞))` round encoding of `pcs::jagged`.
///    Converts the inner-product claim into the single evaluation claim
///    `f̂(ρ) = f_eval`.
/// 2. **Fused Ligerito opening**: materializes the jagged weight table
///    `W_ρ[e] = eq(ρ_row, row(e))·eq(ρ_col, col(e))` over the dense domain
///    (`z_row = ρ[0..n_log]`, `z_col = ρ[n_log..]` — BatchMajor suffix order
///    is `[batch | chunk]`; zero past the jagged area, so
///    `Σ_e q[e]·W_ρ[e] = f_eval`) and opens `q` = the packed witness against
///    `W_ρ` as the basis with target `f_eval`, reusing the commit-time
///    codeword/Merkle tree as L0 exactly like the mixed path. There is no
///    jagged main sumcheck on this path.
/// 3. **Send-and-spot-check** (`b_tilde` + assist): the verifier cannot
///    evaluate `Ŵ_ρ` succinctly at Ligerito's residual points
///    `ris ‖ bits(y)`, so the prover sends `b_tilde[y] = Ŵ_ρ(ris ‖ bits(y))`
///    for all `y`, the transcript squeezes `yr_log_n` fresh challenges
///    `r_extra`, and the existing, unmodified [`jagged::prove_assist`] proves
///    `Ŵ_ρ(ris ‖ r_extra)` — which the verifier compares against the MLE of
///    the received `b_tilde` at `r_extra` (Schwartz–Zippel binding).
///
/// `heights` are the per-chunk-column word counts of the jagged grid
/// (`2^(k_log−7)` entries; see `BlockR1cs::jagged_heights`), `n_log` the
/// number of batch (row) variables. The witness must be zero past the jagged
/// area (`Σ heights` packed words) — the BatchMajor buffer layout guarantees
/// this.
///
/// **True dense-stack commit (M4):** `dense_witness` is the committed stack
/// `q` — the padded `packed_witness` with the height-0 (dropped) columns
/// deleted and the total zero-padded to a power of two (see
/// `UnionInstance::compact_witness`; `col_prefix_sums` of `heights` IS the
/// compaction map). When `Some(q)`: the commitment, `lig_config`, and the
/// jagged `W_ρ`/`b_tilde`/assist all live on `q`'s (possibly smaller)
/// `2^dense_log` domain, while the claim assembly and the virtual-opening
/// sumcheck keep running over the padded `packed_witness` — the identity
/// `⟨q, W_ρ⟩ = f̂(ρ)` holds because the padded buffer is zero on every
/// dropped column. When `None`, `q` is `packed_witness` itself (the
/// single-table paths, whose compaction map is the identity).
#[allow(clippy::too_many_arguments)]
pub fn open_batch_jagged_ligerito<Ch: Challenger>(
    packed_witness: Vec<F128>,
    dense_witness: Option<Vec<F128>>,
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
) -> BatchOpeningProofJaggedLigerito {
    let trace = std::env::var("PCS_TRACE").is_ok();
    let t_total = std::time::Instant::now();

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

    // ---- Claim assembly: shared with (and transcript-identical to) the
    // mixed path up to the γ-combined `(b_combined, target_combined)`.
    let combined = compute_combined_basis_and_target(
        &packed_witness,
        x_outers,
        precomputed_s_hat_v,
        packed_direct,
        padding,
        challenger,
        trace,
    );

    let l = packed_witness.len();
    let log_l = l.trailing_zeros() as usize;
    assert_eq!(l, 1usize << log_l);
    assert!(n_log <= log_l, "n_log exceeds packed-word variable count");

    // ---- Virtual-opening sumcheck: Σ_x f(x)·b_combined(x) = target_combined,
    // binding the low packed-word variable each round. Round 0's message falls
    // out of the already-computed round-0 prime: `u_0 = G(0)` and
    // `target = G(0) + G(1)` (char 2) give `G(1) = target + u_0`.
    let t = std::time::Instant::now();
    challenger.observe_label(b"flock-virtual-open-v0");
    let b0 = combined.b_combined;
    let (u0, u2) = combined.round0_prime;
    let (mut g_one, mut g_inf) = (combined.target_combined + u0, u2);
    let mut virtual_open_rounds = Vec::with_capacity(log_l);
    let mut rho = Vec::with_capacity(log_l);
    // Ping-pong fold buffers, exactly as jagged::prove_main: round 0 folds out
    // of the borrowed (packed_witness, b0); rounds 1+ alternate (a, bb) with
    // the scratch (sa, sb).
    let mut sa = crate::scratch::take_f128(l / 2);
    let mut sb = crate::scratch::take_f128(l / 2);
    let mut a = crate::scratch::take_f128(l / 4);
    let mut bb = crate::scratch::take_f128(l / 4);
    let mut cur = l;
    for round in 0..log_l {
        let half = cur / 2;
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        virtual_open_rounds.push((g_one, g_inf));
        rho.push(r);
        let (a_src, b_src): (&[F128], &[F128]) = if round == 0 {
            (packed_witness.as_slice(), b0.as_slice())
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
        std::mem::swap(&mut a, &mut sa);
        std::mem::swap(&mut bb, &mut sb);
        cur = half;
    }
    let f_eval = if log_l == 0 { packed_witness[0] } else { a[0] };
    challenger.observe_f128(f_eval);
    crate::scratch::give_f128(b0);
    crate::scratch::give_f128(sa);
    crate::scratch::give_f128(sb);
    crate::scratch::give_f128(a);
    crate::scratch::give_f128(bb);
    if trace {
        eprintln!(
            "  [open_jagged] virtual-opening sumcheck ({log_l} rounds): {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Switch from the padded buffer to the committed dense stack q
    // (identical when no dense_witness is given). Everything from here on —
    // W_ρ, the fused Ligerito opening, b_tilde, the assist — lives on q's
    // 2^dense_log domain; the padded buffer is dead and recycled.
    let q: Vec<F128> = match dense_witness {
        Some(q) => {
            crate::scratch::give_f128(packed_witness);
            q
        }
        None => packed_witness,
    };
    let dense_log = q.len().trailing_zeros() as usize;
    assert_eq!(q.len(), 1usize << dense_log, "q must be a power of two");
    assert_eq!(
        dense_log,
        commitment.params.m - LOG_PACKING,
        "dense witness length must match the commitment's PcsParams.m"
    );
    assert!(dense_log <= log_l, "dense domain exceeds the padded domain");

    // ---- Jagged weight table W_ρ over the dense domain + round-0 prime.
    // W_ρ[e] = eq(ρ_row, row(e))·eq(ρ_col, col(e)) (zero past the jagged
    // area), so ⟨q, W_ρ⟩ = f̂(ρ) = f_eval — the fused Ligerito opening below
    // discharges this inner product directly. (With a compacted q the
    // identity holds because the padded buffer is zero on every dropped
    // column, so the deleted terms of f̂(ρ) were all zero.)
    let t = std::time::Instant::now();
    let params = jagged::JaggedParams::from_heights(heights, n_log, dense_log);
    debug_assert!(
        q[params.area() as usize..].iter().all(|&w| w == F128::ZERO),
        "committed stack must be zero past the jagged area"
    );
    let (w_rho, claim_v, round0) =
        jagged::weight_table_claim_and_round0(&params, &q, &rho[..n_log], &rho[n_log..]);
    debug_assert_eq!(
        claim_v, f_eval,
        "⟨q, W_ρ⟩ must equal the virtual-opening output (witness zero past area)"
    );
    if trace {
        eprintln!(
            "  [open_jagged] W_rho weight table + round0: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Fused Ligerito: open q against W_ρ with target f_eval, reusing the
    // commit-time codeword/tree as L0. The fused entry also returns the fold
    // challenges `ris` and mirrors the succinct verifier's trailing samples so
    // the transcript can continue below.
    let t = std::time::Instant::now();
    let (ligerito_proof, ris) = ligerito::recursive_prover_with_basis_precomputed_round0_fused(
        lig_config,
        q,
        w_rho,
        f_eval,
        &prover_data.codeword,
        &prover_data.merkle_tree,
        round0,
        challenger,
    );
    if trace {
        eprintln!(
            "  [open_jagged] ligerito::recursive_prover_with_basis (fused): {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Send-and-spot-check: b_tilde[y] = Ŵ_ρ(ris ‖ bits(y)) for all y in
    // {0,1}^yr_log_n (Ŵ_ρ = f̂_t(ρ_row, ρ_col, ·), so 2^yr_log_n branching-
    // program evaluations — O(2^yr_log_n · 2^k · dense_log) muls, no pass over
    // the 2^dense_log table), observed in index order; then squeeze r_extra
    // and run the existing, unmodified assist at z_index = ris ‖ r_extra.
    let t = std::time::Instant::now();
    let yr_log_n = dense_log - ris.len();
    let b_tilde: Vec<F128> = {
        use rayon::prelude::*;
        (0..1usize << yr_log_n)
            .into_par_iter()
            .map(|y| {
                let mut z_index = Vec::with_capacity(dense_log);
                z_index.extend_from_slice(&ris);
                for j in 0..yr_log_n {
                    z_index.push(if (y >> j) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    });
                }
                jagged::f_hat_t(&params, &rho[..n_log], &rho[n_log..], &z_index)
            })
            .collect()
    };
    for &v in &b_tilde {
        challenger.observe_f128(v);
    }
    let r_extra = challenger.sample_f128_vec(yr_log_n);
    let mut z_index = ris;
    z_index.extend_from_slice(&r_extra);
    let jagged_assist =
        jagged::prove_assist(&params, &rho[..n_log], &rho[n_log..], &z_index, challenger);
    if trace {
        eprintln!(
            "  [open_jagged] b_tilde (2^{yr_log_n}) + assist: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        eprintln!(
            "  [open_jagged] TOTAL: {:6.2} ms",
            t_total.elapsed().as_secs_f64() * 1e3
        );
    }

    BatchOpeningProofJaggedLigerito {
        ring_switches: combined.ring_switches,
        virtual_open_rounds,
        f_eval,
        ligerito: ligerito_proof,
        b_tilde,
        jagged_assist,
    }
}

/// Verify a jagged-path batched opening (mirror of
/// [`open_batch_jagged_ligerito`]). Runs the per-claim
/// `ring_switch::verify_succinct` + target reconstruction exactly as
/// [`verify_opening_batch_ligerito_mixed`], replays the virtual-opening
/// sumcheck and checks its final round against `b̂_combined(ρ) · f_eval`
/// (evaluating `b̂_combined` itself via the same residual machinery —
/// `eval_rs_eq` per ring-switched claim, `eq_eval` per packed-direct claim —
/// at the arbitrary field point `ρ`), then drives the succinct Ligerito
/// verifier on the fused opening `⟨q, W_ρ⟩ = f_eval` with the residual basis
/// values at `ris ‖ bits(y)` **taken from the proof's `b_tilde`** (the
/// induced/OOD terms are computed as usual), and finally binds `b_tilde` to
/// the true weight table: observe `b_tilde`, squeeze `r_extra`, run the
/// unmodified [`jagged::verify_assist`] at `z_index = ris ‖ r_extra`, and
/// require the verified `β = Ŵ_ρ(ris ‖ r_extra)` to equal the MLE of the
/// received `b_tilde` at `r_extra` (Schwartz–Zippel spot-check).
///
/// `virtual_vars` is the packed-word variable count of the VIRTUAL (padded)
/// polynomial the PIOP ran over — the virtual-opening sumcheck's round
/// count. The committed dense stack's variable count is
/// `commitment.params.m − 7 ≤ virtual_vars`; the two coincide on the
/// single-table paths (identity compaction) and split under the true
/// dense-stack commit (M4).
#[allow(clippy::too_many_arguments)]
pub fn verify_opening_batch_jagged_ligerito<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    z_skips: &[F128],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    heights: &[u64],
    n_log: usize,
    virtual_vars: usize,
    proof: &BatchOpeningProofJaggedLigerito,
    lig_config: &ligerito::VerifierConfig,
    challenger: &mut Ch,
) -> Result<(), VerifyErrorJagged> {
    let n_rs = claims.len();
    let n_pd = packed_direct.len();
    assert_eq!(z_skips.len(), n_rs);
    assert_eq!(x_outers.len(), n_rs);
    assert_eq!(proof.ring_switches.len(), n_rs);
    assert!(n_rs + n_pd > 0);

    challenger.observe_label(b"flock-pcs-open-batch-v0");

    // 1.–3. Ring-switch succinct verify + γ-batching — identical to
    // `verify_opening_batch_ligerito_mixed` steps 1–3.
    let mut rs_outputs = Vec::with_capacity(n_rs);
    for i in 0..n_rs {
        let out = ring_switch::verify_succinct(
            claims[i],
            z_skips[i],
            x_outers[i],
            &proof.ring_switches[i],
            challenger,
        )
        .map_err(VerifyErrorJagged::RingSwitch)?;
        rs_outputs.push(out);
    }
    let gammas_rs: Vec<F128> = (0..n_rs).map(|_| challenger.sample_f128()).collect();

    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas_pd: Vec<F128> = (0..n_pd).map(|_| challenger.sample_f128()).collect();

    let mut target_combined = F128::ZERO;
    for (out, g) in rs_outputs.iter().zip(gammas_rs.iter()) {
        target_combined += *g * out.sumcheck_claim;
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target_combined += *g * pd.value;
    }

    // 4. Virtual-opening sumcheck replay over the PADDED word domain
    //    (`virtual_vars` rounds). The `(G(1), G(∞))` encoding folds the
    //    per-round sum check into the running claim (`G(0)` is
    //    reconstructed from it); the final round is checked against
    //    `b̂_combined(ρ) · f_eval` below.
    let log_l = virtual_vars;
    let dense_log = commitment.params.m - LOG_PACKING;
    assert!(
        dense_log <= log_l,
        "committed dense domain exceeds the virtual domain"
    );
    challenger.observe_label(b"flock-virtual-open-v0");
    if proof.virtual_open_rounds.len() != log_l {
        return Err(VerifyErrorJagged::VirtualOpen);
    }
    let mut running = target_combined;
    let mut rho = Vec::with_capacity(log_l);
    for &(g_one, g_inf) in &proof.virtual_open_rounds {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        running = jagged::fold_round_claim(running, g_one, g_inf, r);
        rho.push(r);
    }
    // b̂_combined(ρ): the same residual-evaluation machinery as the mixed
    // path's `eval_b_residual`, at the (arbitrary-field) point ρ.
    let mut b_at_rho = F128::ZERO;
    for ((out, g), x_outer) in rs_outputs.iter().zip(gammas_rs.iter()).zip(x_outers.iter()) {
        b_at_rho += *g * ring_switch::eval_rs_eq(&x_outer[1..], &rho, &out.eq_r_dprime);
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        b_at_rho += *g * crate::zerocheck::multilinear::eq_eval(pd.point, &rho);
    }
    if running != b_at_rho * proof.f_eval {
        return Err(VerifyErrorJagged::VirtualOpen);
    }
    challenger.observe_f128(proof.f_eval);

    // 5. Succinct Ligerito verify of the fused opening ⟨q, W_ρ⟩ = f_eval,
    //    over the committed DENSE domain (`dense_log` variables; the
    //    jagged params' `col_prefix_sums` are the compaction map). The
    //    residual basis values at ris ‖ bits(y) are TAKEN FROM the
    //    proof's b_tilde (the closure returns them verbatim; the induced/OOD
    //    terms are computed as usual, and the final check consumes b_tilde).
    //    The closure also captures Ligerito's fold challenges `ris` for the
    //    spot-check below — it runs exactly once, at the residual.
    assert!(n_log <= log_l, "n_log exceeds packed-word variable count");
    let params = jagged::JaggedParams::from_heights(heights, n_log, dense_log);
    let ris_cell: std::cell::RefCell<Vec<F128>> = std::cell::RefCell::new(Vec::new());
    let eval_b_residual = |ris: &[F128], _yr_log_n: usize| -> Vec<F128> {
        ris_cell.borrow_mut().extend_from_slice(ris);
        proof.b_tilde.clone()
    };
    let ok = ligerito::recursive_verifier_with_basis_succinct(
        lig_config,
        &proof.ligerito,
        dense_log,
        proof.f_eval,
        &commitment.root,
        eval_b_residual,
        challenger,
    );
    if !ok {
        return Err(VerifyErrorJagged::Ligerito);
    }
    let ris = ris_cell.into_inner();
    let yr_log_n = dense_log - ris.len();
    // The succinct verifier accepted, so it reached the residual (populating
    // `ris`) and checked the closure's output length against 2^yr_log_n.
    debug_assert_eq!(proof.b_tilde.len(), 1usize << yr_log_n);

    // 6. Send-and-spot-check binding of b_tilde to the true Ŵ_ρ. Transcript
    //    mirror of the prover: observe b_tilde (index order), squeeze
    //    r_extra, then the unmodified assist at z_index = ris ‖ r_extra. The
    //    assist-verified β = Ŵ_ρ(ris ‖ r_extra) must equal the MLE of the
    //    received b_tilde at r_extra — otherwise b_tilde differs from the
    //    true basis vector and the proof is rejected (Schwartz–Zippel).
    for &v in &proof.b_tilde {
        challenger.observe_f128(v);
    }
    let r_extra = challenger.sample_f128_vec(yr_log_n);
    let b_tilde_at_r_extra = ligerito::partial_eval_lsb(&proof.b_tilde, &r_extra)[0];
    let mut z_index = ris;
    z_index.extend_from_slice(&r_extra);
    let beta = jagged::verify_assist(
        &params,
        &rho[..n_log],
        &rho[n_log..],
        &z_index,
        &proof.jagged_assist,
        challenger,
    )
    .ok_or(VerifyErrorJagged::Jagged)?;
    if beta != b_tilde_at_r_extra {
        return Err(VerifyErrorJagged::Jagged);
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
        };
        let z_packed = pack_witness(&z, m);
        let (commitment, prover_data) = commit(&z_packed, &params);

        let recursive_ks = vec![3usize, 3, 3];
        let log_inv_rates = vec![1usize, 3, 4, 6];
        let queries: Vec<usize> = log_inv_rates
            .iter()
            .map(|&r| crate::pcs::ligerito::udr_queries(r))
            .collect();
        let grinding_bits = vec![0usize; log_inv_rates.len()];
        let n_levels = log_inv_rates.len();
        let lig_p_cfg = crate::pcs::ligerito::ProverConfig {
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
        };
        let lig_v_cfg = crate::pcs::ligerito::VerifierConfig {
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

    /// End-to-end roundtrip through the jagged opening path
    /// (`open_batch_jagged_ligerito` / `verify_opening_batch_jagged_ligerito`)
    /// on a synthetic single-table instance with dead chunk-columns, plus
    /// tamper-rejection on every new proof component.
    #[test]
    #[ignore] // Heavier — run with `cargo test pcs_jagged_backend -- --ignored`
    fn pcs_jagged_backend_roundtrip_and_tamper() {
        let m = 22usize; // log_l = 15 packed-word variables
        let n_log = 8usize; // 2^8 rows per chunk-column, 2^7 chunk-columns
        let n_chunks = 1usize << (m - 7 - n_log);
        let useful_chunks = 100usize; // 28 dead (zero) chunk-columns
        let area_words = useful_chunks << n_log;

        let mut rng = Rng::new(0x1A66_ED01);
        let mut z = rng.bits(1 << m);
        // BatchMajor stacking: the useful chunk-columns are the contiguous
        // word prefix [0, area); zero everything past it.
        for bit in z.iter_mut().skip(area_words * 128) {
            *bit = false;
        }
        let z_skip = rng.f128();
        let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
        let rs_claim = zhat_skip_reference(&z, m, z_skip, &x_outer);
        let heights: Vec<u64> = (0..n_chunks)
            .map(|c| if c < useful_chunks { 1u64 << n_log } else { 0 })
            .collect();

        let initial_k = 6;
        let params = PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: initial_k,
            profile: Default::default(),
        };
        let z_packed = pack_witness(&z, m);
        let (commitment, prover_data) = commit(&z_packed, &params);

        // Production embedded configs (the hand-rolled ad-hoc config of the
        // test above predates the query-count derivation and is stale).
        let lig_p_cfg =
            crate::pcs::ligerito::prover_config_for(m - LOG_PACKING, initial_k, params.profile)
                .expect("embedded Ligerito config for m=22");
        let lig_v_cfg =
            crate::pcs::ligerito::verifier_config_for(m - LOG_PACKING, initial_k, params.profile)
                .expect("embedded Ligerito verifier config for m=22");

        let mut ch_p = FsChallenger::new(b"flock-test-jagged-v0");
        let proof = open_batch_jagged_ligerito(
            z_packed.clone(),
            None,
            &prover_data,
            &commitment,
            &[x_outer.as_slice()],
            &[],
            &[],
            &PaddingSpec::dense(m),
            &heights,
            n_log,
            &lig_p_cfg,
            &mut ch_p,
        );

        let verify = |proof: &BatchOpeningProofJaggedLigerito,
                      heights: &[u64]|
         -> Result<(), VerifyErrorJagged> {
            let mut ch_v = FsChallenger::new(b"flock-test-jagged-v0");
            verify_opening_batch_jagged_ligerito(
                &commitment,
                &[rs_claim],
                &[z_skip],
                &[x_outer.as_slice()],
                &[],
                heights,
                n_log,
                m - LOG_PACKING,
                proof,
                &lig_v_cfg,
                &mut ch_v,
            )
        };

        verify(&proof, &heights)
            .unwrap_or_else(|e| panic!("jagged verify rejected honest proof: {e:?}"));

        // Tamper: corrupted f_eval → virtual-opening final check fails.
        {
            let mut bad = proof.clone();
            bad.f_eval.lo ^= 1;
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::VirtualOpen));
        }
        // Tamper: corrupted virtual-opening round.
        {
            let mut bad = proof.clone();
            bad.virtual_open_rounds[3].0.lo ^= 1;
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::VirtualOpen));
        }
        // Tamper: wrong virtual-opening round count.
        {
            let mut bad = proof.clone();
            bad.virtual_open_rounds.pop();
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::VirtualOpen));
        }
        // Tamper: single b_tilde element → Ligerito's final check consumes
        // b_tilde and fails (its yr-weighted sum shifts by yr[0]·δ ≠ 0).
        {
            let mut bad = proof.clone();
            bad.b_tilde[0].lo ^= 1;
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::Ligerito));
        }
        // Tamper: a b_tilde PAIR crafted to keep Ligerito's final check
        // satisfied — δ_0 = yr[1]·c, δ_1 = yr[0]·c makes the yr-weighted
        // shift yr[0]·yr[1]·c + yr[1]·yr[0]·c = 0 (char 2) — so it must be
        // caught downstream by the spot-check/assist comparison: the changed
        // b_tilde re-randomizes r_extra, and the assist replay and/or the
        // β = MLE_{b_tilde}(r_extra) check rejects.
        {
            let mut bad = proof.clone();
            let c = F128 {
                lo: 0xD1CE,
                hi: 0x5EED,
            };
            let yr0 = proof.ligerito.final_proof.yr[0];
            let yr1 = proof.ligerito.final_proof.yr[1];
            let (d0, d1) = (yr1 * c, yr0 * c);
            assert!(d0 != F128::ZERO && d1 != F128::ZERO, "degenerate yr");
            bad.b_tilde[0] += d0;
            bad.b_tilde[1] += d1;
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::Jagged));
        }
        // Tamper: corrupted assist claim β.
        {
            let mut bad = proof.clone();
            bad.jagged_assist.beta.lo ^= 1;
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::Jagged));
        }
        // Tamper: corrupted assist round.
        {
            let mut bad = proof.clone();
            bad.jagged_assist.rounds[5].0.lo ^= 1;
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::Jagged));
        }
        // Tamper: corrupted ring-switch message → claim check fails.
        {
            let mut bad = proof.clone();
            bad.ring_switches[0].s_hat_v[0].lo ^= 1;
            assert!(matches!(
                verify(&bad, &heights),
                Err(VerifyErrorJagged::RingSwitch(_))
            ));
        }
        // Tamper: corrupted Ligerito final message.
        {
            let mut bad = proof.clone();
            bad.ligerito.final_proof.yr[0].lo ^= 1;
            assert_eq!(verify(&bad, &heights), Err(VerifyErrorJagged::Ligerito));
        }
        // Wrong heights vector (one fewer useful column) → the jagged
        // transport's f̂_t no longer matches the proof.
        {
            let mut bad_heights = heights.clone();
            bad_heights[useful_chunks - 1] = 0;
            assert_eq!(verify(&proof, &bad_heights), Err(VerifyErrorJagged::Jagged));
        }
    }

    /// True dense-stack commit (M4) at the PCS level: the committed stack
    /// `q` is strictly SMALLER than the padded buffer (2^15 vs 2^16 words —
    /// 2x fewer Merkle leaves), the virtual-opening sumcheck runs over the
    /// padded domain while Ligerito/W_ρ/b_tilde run over `q`, and the
    /// roundtrip verifies. Synthetic single-table shape whose used columns
    /// are the word prefix, so the compaction map is a pure truncation.
    #[test]
    #[ignore] // Heavier — run with `cargo test pcs_jagged_dense -- --ignored`
    fn pcs_jagged_dense_stack_smaller_commit_roundtrip() {
        let m_virtual = 23usize; // padded: 2^16 packed words
        let n_log = 8usize; // 2^8 rows per chunk-column, 2^8 chunk-columns
        let n_chunks = 1usize << (m_virtual - 7 - n_log);
        let useful_chunks = 100usize; // area 100·2^8 = 25 600 words → q = 2^15
        let area_words = useful_chunks << n_log;
        let committed_words = area_words.next_power_of_two();
        let m_dense = committed_words.trailing_zeros() as usize + LOG_PACKING;
        assert_eq!(m_dense, 22, "test shape must land on the m22 config");

        let mut rng = Rng::new(0xDE_5E_57AC);
        let mut z = rng.bits(1 << m_virtual);
        for bit in z.iter_mut().skip(area_words * 128) {
            *bit = false;
        }
        let z_skip = rng.f128();
        let x_outer: Vec<F128> = (0..(m_virtual - 6)).map(|_| rng.f128()).collect();
        let rs_claim = zhat_skip_reference(&z, m_virtual, z_skip, &x_outer);
        let heights: Vec<u64> = (0..n_chunks)
            .map(|c| if c < useful_chunks { 1u64 << n_log } else { 0 })
            .collect();

        let initial_k = 6;
        let params = PcsParams {
            m: m_dense,
            log_inv_rate: 1,
            log_batch_size: initial_k,
            profile: Default::default(),
        };
        let z_packed = pack_witness(&z, m_virtual);
        // The dense stack: used columns are the contiguous word prefix, so
        // compaction = truncation to the committed power of two.
        let q: Vec<F128> = z_packed[..committed_words].to_vec();
        assert!(
            q.len() < z_packed.len(),
            "committed words must be fewer than padded words"
        );
        let (commitment, prover_data) = commit(&q, &params);

        let lig_p_cfg = crate::pcs::ligerito::prover_config_for(
            m_dense - LOG_PACKING,
            initial_k,
            params.profile,
        )
        .expect("embedded Ligerito config for m=22");
        let lig_v_cfg = crate::pcs::ligerito::verifier_config_for(
            m_dense - LOG_PACKING,
            initial_k,
            params.profile,
        )
        .expect("embedded Ligerito verifier config for m=22");

        let mut ch_p = FsChallenger::new(b"flock-test-jagged-dense-v0");
        let proof = open_batch_jagged_ligerito(
            z_packed.clone(),
            Some(q),
            &prover_data,
            &commitment,
            &[x_outer.as_slice()],
            &[],
            &[],
            &PaddingSpec::dense(m_virtual),
            &heights,
            n_log,
            &lig_p_cfg,
            &mut ch_p,
        );
        assert_eq!(
            proof.virtual_open_rounds.len(),
            m_virtual - LOG_PACKING,
            "virtual-opening sumcheck must span the padded domain"
        );

        let verify = |proof: &BatchOpeningProofJaggedLigerito| -> Result<(), VerifyErrorJagged> {
            let mut ch_v = FsChallenger::new(b"flock-test-jagged-dense-v0");
            verify_opening_batch_jagged_ligerito(
                &commitment,
                &[rs_claim],
                &[z_skip],
                &[x_outer.as_slice()],
                &[],
                &heights,
                n_log,
                m_virtual - LOG_PACKING,
                proof,
                &lig_v_cfg,
                &mut ch_v,
            )
        };
        verify(&proof)
            .unwrap_or_else(|e| panic!("dense-stack jagged verify rejected honest proof: {e:?}"));

        // Tamper smoke on the new split: f_eval (virtual-open final check)
        // and a Ligerito final message (dense-domain opening).
        {
            let mut bad = proof.clone();
            bad.f_eval.lo ^= 1;
            assert_eq!(verify(&bad), Err(VerifyErrorJagged::VirtualOpen));
        }
        {
            let mut bad = proof.clone();
            bad.ligerito.final_proof.yr[0].lo ^= 1;
            assert_eq!(verify(&bad), Err(VerifyErrorJagged::Ligerito));
        }
    }
}
