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
pub mod stratified;
pub mod tensor_algebra;

pub use commit::{
    Commitment, PcsParams, ProverData, commit, commit_encode, commit_encode_into, commit_into,
    commit_lane_major, commit_leaf_pipeline_shape, commit_merkle, dense_lanes,
    prefault_codeword_during,
};
pub use jagged::rectangular_prefix_columns;
pub use pack::{LOG_PACKING, pack_witness};
pub use ring_switch::{RingSwitchProof, SparseEqTensor};

use crate::challenger::Challenger;
use crate::field::{F128, F256};
use crate::zerocheck::PaddingSpec;
use serde::{Deserialize, Serialize};

/// Batched opening proof: ring-switching frontend + Ligerito backend.
/// The combined `b_combined` + target_combined feed
/// [`ligerito::recursive_prover_with_basis`] (see ligerito module docs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchOpeningProofLigerito {
    pub ring_switches: Vec<RingSwitchProof>,
    /// Fiat--Shamir PoW witnesses for the random-linear-combination
    /// coefficients, in claim order (ring-switched claims first, then packed
    /// direct claims). Empty when opening grinding is disabled or when the
    /// batch has only one claim.
    #[serde(default)]
    pub batching_nonces: Vec<u64>,
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

/// In-process A/B override for the VIRTUAL BASIS (see
/// [`ligerito::VirtualEqBasis`]): `0` follows the `FLOCK_NO_VIRTUAL_B` env
/// knob, `1` forces the virtual basis on, `2` forces the pre-virtualization
/// path (round-0-only JIT fill). A few-ms effect only resolves under an
/// ALTERNATING in-process instrument over identical inputs — process-level
/// arms on this box carry ±4-8 ms of interference per sample.
pub static VIRTUAL_B_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// In-process A/B override for the STATISTICS LADDER (see
/// `ligerito::extension::init_phase_statistics`): `0` follows the
/// `FLOCK_NO_STATS_LADDER` env knob, `1` forces it on, `2` forces the
/// incremental L0 fold ladder. Both arms produce byte-identical proofs.
pub static STATS_LADDER_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(crate) fn virtual_b_enabled() -> bool {
    match VIRTUAL_B_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => std::env::var_os("FLOCK_NO_VIRTUAL_B").is_none(),
    }
}

pub(crate) fn stats_ladder_enabled() -> bool {
    match STATS_LADDER_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => std::env::var_os("FLOCK_NO_STATS_LADDER").is_none(),
    }
}

/// Fiat--Shamir grinding policy for the PCS transport that sits before the
/// Ligerito opening.  Each nonzero field is applied immediately after its
/// prover message(s) are bound and immediately before the challenge it
/// protects is sampled.
///
/// The Secure schedule turns a degree-/numerator-`D` algebraic event over
/// `F128` into a computational work-factor term of about
/// `D * 2^-bits / 2^128` at that site:
///
/// * ring switching has degree at most seven;
/// * each random linear-combination coefficient has numerator one;
/// * the merged, multipoint, and anchor sumchecks have quadratic rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningGrinding {
    /// Seven-coordinate ring-switch point `r''`.
    pub ring_switch_bits: u32,
    /// Random coefficients used to batch independent opening claims.
    pub claim_batch_bits: u32,
    /// Each quadratic round of the dense merged sumcheck.
    pub merged_round_bits: u32,
    /// The multipoint dual-value batching coefficient and its sumcheck /
    /// anchor-round policies.
    pub multipoint: jagged::MultipointGrinding,
}

impl OpeningGrinding {
    pub const fn disabled() -> Self {
        Self {
            ring_switch_bits: 0,
            claim_batch_bits: 0,
            merged_round_bits: 0,
            multipoint: jagged::MultipointGrinding::disabled(),
        }
    }

    pub const fn per_challenge_128() -> Self {
        Self {
            // 7 / 2^128, then a 2^-3 PoW filter.
            ring_switch_bits: 3,
            // The transport coefficient is sampled before Ligerito's OOD
            // points bind one candidate from the Johnson list. Its bad event
            // therefore unions over L0's list (log2 L < 6 in every embedded
            // strict profile); six PoW bits make L/2^(128+6) < 2^-128.
            claim_batch_bits: 6,
            // Quadratic sumcheck rounds: 2 / 2^128, then 2^-2.
            merged_round_bits: 2,
            multipoint: jagged::MultipointGrinding::per_challenge_128(),
        }
    }

    #[inline]
    fn claim_batch_bits_for(self, claim_count: usize) -> u32 {
        if claim_count > 1 {
            self.claim_batch_bits
        } else {
            0
        }
    }
}

/// Mixed-claim batched open: supports both **ring-switched** claims (bit-MLE
/// openings reduced via `ring_switch::prove_batched`, with optional per-claim
/// precomputed `s_hat_v`) and **packed-direct** claims (packed-MLE openings
/// that skip ring-switch). Runs the ring_switch + b_combined computation, then
/// routes to [`ligerito::recursive_prover_with_basis`] using the existing
/// `prover_data`'s codeword + tree as Ligerito's L0 commit (no L0 re-commit),
/// with PoW witnesses for ring switching and nontrivial claim batching
/// (pass [`OpeningGrinding::disabled`] for a grind-free transcript).
///
/// `lig_config.initial_k` must equal `commitment.params.log_batch_size` so that
/// `prover_data`'s codeword/tree shape matches what Ligerito expects for L0.
#[allow(clippy::too_many_arguments)]
pub fn open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding<Ch: Challenger>(
    packed_witness: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    lig_config: &ligerito::ProverConfig,
    grinding: OpeningGrinding,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    open_batch_mixed_ligerito_seeded(
        packed_witness,
        prover_data,
        commitment,
        x_outers,
        precomputed_s_hat_v,
        packed_direct,
        padding,
        lig_config,
        grinding,
        None,
        challenger,
    )
}

/// [`open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding`] with
/// the inner open's SEEDED BLOCK STATISTICS supplied by the caller: the
/// merged transport's inner claim sits at the merged sumcheck's own point,
/// so that sumcheck's folded witness after `log_n − initial_k` rounds IS
/// `Σ_h f[e,h]·eq(ρ_{low}, h)` per lane block — the statistics ladder's
/// seeded term, which it would otherwise sweep the witness to recompute.
#[allow(clippy::too_many_arguments)]
pub fn open_batch_mixed_ligerito_seeded<Ch: Challenger>(
    packed_witness: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    lig_config: &ligerito::ProverConfig,
    grinding: OpeningGrinding,
    seeded_stats: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    // Belt-and-braces on the cap-depth derivation: the commit-time cap
    // (from `PcsParams::l0_cap_depth`) must be the layer the opener's
    // config implies — a config-source disagreement fails loudly here at
    // prove time instead of as a verifier reject.
    assert_eq!(
        commitment.cap.len(),
        1usize << lig_config.l0_cap_depth(),
        "commitment cap size disagrees with the opener config's L0 query count"
    );
    debug_assert_eq!(
        commitment.cap.as_slice(),
        crate::merkle::cap_layer(
            &prover_data.merkle_tree,
            commitment.params.n_leaves(),
            lig_config.l0_cap_depth(),
        ),
        "commitment cap is not the prover tree's cap layer"
    );
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
        grinding,
        challenger,
        trace,
    );

    let t = std::time::Instant::now();
    let CombinedClaim {
        ring_switches,
        batching_nonces,
        b_combined,
        eq_basis,
        eq_gamma,
        target_combined,
        round0_prime,
        round1_lookahead,
    } = combined;
    // Factored EqPoint basis. DEFAULT (virtual): the basis stays factored
    // across EVERY L0 fold — an eq tensor folds to an eq tensor, so no round
    // writes or reads a half-size basis array; it materializes once, at the
    // last L0 round, at the size the recursion takes over
    // ([`ligerito::VirtualEqBasis`]). `FLOCK_NO_VIRTUAL_B=1` falls back to
    // the round-0-only JIT fill (the pre-virtualization behavior) — the
    // CERTIFICATION knob for alternating A/B runs; both settings produce the
    // same b values, hence byte-identical proofs.
    // Lane-major only: the virtual basis rides the BLOCKED L0 fold, which is
    // what every shipped merged-transport open runs (a pow2-lane inner keeps
    // the tuned element-pairing kernel and its round-0 JIT).
    let virtual_b = eq_basis.is_some() && lane_major && virtual_b_enabled();
    let vbasis = if virtual_b {
        // The point IS the claim's, and γ is its single transcript scalar —
        // the same (γ, ρ) the split tables above were seeded with.
        Some(ligerito::VirtualEqBasis::new(
            match &packed_direct[0].eq_ind {
                DirectEqInd::EqPoint(point) => point.clone(),
                _ => unreachable!("the factored basis is built only for EqPoint"),
            },
            eq_gamma.expect("an eq basis carries its γ"),
        ))
    } else {
        None
    };
    let jit_fill;
    let jit: Option<ligerito::BasisWindowFn<'_>> = match (&eq_basis, virtual_b) {
        (Some((lo, hi, n_lo)), false) => {
            let mask = (1usize << n_lo) - 1;
            let n_lo = *n_lo;
            jit_fill = move |out: &mut [F128], g0: usize| {
                for (i, slot) in out.iter_mut().enumerate() {
                    let u = g0 + i;
                    *slot = lo[u & mask] * hi[u >> n_lo];
                }
            };
            Some(&jit_fill)
        }
        _ => None,
    };
    let ligerito_proof = ligerito::recursive_prover_with_basis_precomputed_round0_lanes(
        lig_config,
        packed_witness,
        b_combined,
        target_combined,
        &prover_data.codeword,
        &prover_data.merkle_tree,
        l0_num_lanes,
        lane_major,
        round0_prime,
        round1_lookahead,
        jit,
        vbasis,
        seeded_stats,
        None,
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
        ring_switches,
        batching_nonces,
        ligerito: ligerito_proof,
    }
}

/// What ring_switch + claim-combination produces, fed to the Ligerito backend.
struct CombinedClaim {
    ring_switches: Vec<RingSwitchProof>,
    /// PoW witnesses for random linear-combination coefficients.
    batching_nonces: Vec<u64>,
    /// The materialized γ-combined basis — EMPTY when `eq_basis` is `Some`
    /// (the factored path never writes the full-domain array).
    b_combined: Vec<F128>,
    /// The single-EqPoint fast path's FACTORED basis: `(γ-scaled eq_lo,
    /// eq_hi, n_lo)` with `b[u] = lo[u & mask]·hi[u >> n_lo]`. γ is one
    /// fixed transcript scalar (exactly one packed-direct claim), so the
    /// whole basis is a scaled eq tensor — two √L tables instead of a
    /// 2^(m−7)-word array. The L0 first fold sources windows from it
    /// just-in-time ([`ligerito::BasisWindowFn`]); later folds carry the
    /// half-size folded basis as before.
    eq_basis: Option<(Vec<F128>, Vec<F128>, usize)>,
    /// The single γ behind `eq_basis` — the virtual basis re-seeds its own
    /// tables from `(γ, point)` as it folds.
    eq_gamma: Option<F128>,
    target_combined: F128,
    /// Round-0 sumcheck `(u_0, u_2)` prime over `packed_witness · b_combined`,
    /// consumed by `recursive_prover_with_basis_precomputed_round0`.
    round0_prime: (F128, F128),
    /// Quadratic coefficients of Ligerito's ROUND-1 message in the round-0
    /// fold challenge, accumulated in the same combine pass. Three paths
    /// emit them: the plain fast path (LSB-pair coefficients), the fast
    /// path WITH sparse packed-direct claims (scatter deltas corrected by
    /// linearity), and the seeded EqPoint path (BLOCKED coefficients, the
    /// fold's own block pairing). Lets the recursive prover's first lane
    /// fold be an O(1) skip round — see [`ligerito::FoldLookahead`].
    round1_lookahead: Option<ligerito::FoldLookahead>,
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
    grinding: OpeningGrinding,
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
    let batch_bits = grinding.claim_batch_bits_for(n_rs + n_pd);
    let mut rs_results = if n_rs > 0 {
        ring_switch::prove_batched_padded_with_precomputed_unbatched_and_grinding(
            packed_witness,
            x_outers,
            precomputed_s_hat_v,
            padding,
            grinding.ring_switch_bits,
            challenger,
        )
    } else {
        Vec::new()
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
    let mut batching_nonces = Vec::with_capacity(usize::from(batch_bits != 0));
    let gammas = if batch_bits != 0 {
        let (nonce, gammas) = challenger.grind_pow_and_sample_f128_vec(batch_bits, n_rs + n_pd);
        batching_nonces.push(nonce);
        gammas
    } else {
        challenger.sample_f128_vec(n_rs + n_pd)
    };
    let (gammas_rs, gammas_pd) = gammas.split_at(n_rs);
    for ((_, output), &gamma) in rs_results.iter_mut().zip(gammas_rs) {
        output.rs_eq_ind.scale_in_place(gamma);
    }

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

    // Merged-transport inner open: no RS claims, exactly one EqPoint claim.
    // Build `b_combined = γ·eq(point)` seeded (one pass, no eq buffer) and
    // fuse the round-0 prime. Transcript-identical to the Dense variant of
    // the same point (the eq_ind representation is prover-side only).
    if rs_results.is_empty()
        && packed_direct.len() == 1
        && let DirectEqInd::EqPoint(point) = &packed_direct[0].eq_ind
    {
        assert_eq!(1usize << point.len(), l, "EqPoint length mismatch");
        // FACTORED basis: b = γ·eq(point,·) is one fixed γ times an eq
        // tensor, so two √L tables replace the 2^(m−7)-word array — γ baked
        // into the lo half. Exact: field ops are exact, so the split
        // product is bitwise the materialized entry.
        let n_lo = point.len() / 2;
        let lo = ring_switch::build_eq_scaled_parallel(&point[..n_lo], gammas_pd[0]);
        let hi = ring_switch::build_eq_scaled_parallel(&point[n_lo..], F128::ONE);
        let mask = (1usize << n_lo) - 1;
        let bs = |u: usize| lo[u & mask] * hi[u >> n_lo];
        let blk = eqpoint_round0_block;
        // STATISTICS LADDER: the lane-major inner open derives its round-0
        // message and every later L0 round from block statistics inside the
        // recursive prover (one sweep there covers this term AND the L0 OOD
        // term), so the O(L) prime + lookahead sweep below is skipped. Same
        // gate as the prover's: `virtual_b` = eq basis + lane-major, which
        // is exactly this branch with `blk > 1`.
        if blk > 1 && virtual_b_enabled() && stats_ladder_enabled() {
            if trace {
                eprintln!("  [open_batch] combine (seeded EqPoint, statistics ladder): no sweep");
            }
            return CombinedClaim {
                ring_switches: Vec::new(),
                batching_nonces,
                b_combined: Vec::new(),
                eq_basis: Some((lo, hi, n_lo)),
                eq_gamma: Some(gammas_pd[0]),
                target_combined,
                round0_prime: (F128::ZERO, F128::ZERO),
                round1_lookahead: None,
            };
        }
        if blk == 1 {
            // The ladder statically rejects a lookahead on this shape:
            // `fold_block == 1` comes with the factored (JIT) basis, so
            // `kind_ok` in extension.rs never consumes it. Run the lean
            // prime-only pass (2 muls/pair) instead of paying the doubled
            // lookahead accumulation for coefficients nothing reads.
            const C: usize = 1 << 13;
            let (round0_u0, round0_u2) = packed_witness
                .par_chunks(C)
                .enumerate()
                .map(|(ci, qc)| {
                    let base = ci * C;
                    let mut a = F128::ZERO;
                    let mut b = F128::ZERO;
                    for (j, qp) in qc.as_chunks::<2>().0.iter().enumerate() {
                        let u = base + 2 * j;
                        let (w0, w1) = (bs(u), bs(u + 1));
                        a += qp[0] * w0;
                        b += (qp[0] + qp[1]) * (w0 + w1);
                    }
                    (a, b)
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
                );
            if trace {
                eprintln!(
                    "  [open_batch] combine (seeded EqPoint, L={l}): {:6.2} ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
            }
            return CombinedClaim {
                ring_switches: Vec::new(),
                batching_nonces,
                b_combined: Vec::new(),
                eq_basis: Some((lo, hi, n_lo)),
                eq_gamma: Some(gammas_pd[0]),
                target_combined,
                round0_prime: (round0_u0, round0_u2),
                round1_lookahead: None,
            };
        }
        // The same seeded pass now also accumulates the ROUND-1 message's
        // quadratic coefficients in the round-0 fold challenge, under the
        // fold's own pairing: quad `q` covers the four consecutive
        // `blk`-blocks the ladder's first two folds combine, so
        // `ligerito::lookahead_accum_group` applies verbatim with blocked
        // gathering (+4 unreduced muls per quad on a pass that already runs;
        // it buys the ladder a full fold-0 pass). Needs `l ≥ 4·blk`
        // (initial_k ≥ 2), which every shipped ladder satisfies.
        use crate::field::F256Unreduced;
        assert!(blk.is_power_of_two() && l.is_multiple_of(4 * blk));
        let quad = |q: usize, acc: &mut [F256Unreduced; 8]| {
            let i0 = 4 * q * blk;
            for k in 0..blk {
                let i = i0 + k;
                let fq = [
                    packed_witness[i],
                    packed_witness[i + blk],
                    packed_witness[i + 2 * blk],
                    packed_witness[i + 3 * blk],
                ];
                let bq = [bs(i), bs(i + blk), bs(i + 2 * blk), bs(i + 3 * blk)];
                ligerito::lookahead_accum_group(&fq, &bq, acc);
            }
        };
        // Parallelize on quads; for the lane-major shape (huge blk, few
        // quads) split each quad's inner `k` range instead.
        let n_quads = l / (4 * blk);
        let acc = if n_quads >= 64 {
            // ~2^13 elements per task, whatever the block size.
            let qc = ((1usize << 13) / (4 * blk)).max(1);
            (0..n_quads.div_ceil(qc))
                .into_par_iter()
                .map(|qq| {
                    let mut acc = [F256Unreduced::ZERO; 8];
                    for q in (qq * qc)..((qq + 1) * qc).min(n_quads) {
                        quad(q, &mut acc);
                    }
                    acc
                })
                .reduce(|| [F256Unreduced::ZERO; 8], ligerito::xor_acc8)
        } else {
            const KC: usize = 1 << 12;
            (0..n_quads * blk.div_ceil(KC))
                .into_par_iter()
                .map(|item| {
                    let chunks_per_quad = blk.div_ceil(KC);
                    let (q, kc) = (item / chunks_per_quad, item % chunks_per_quad);
                    let i0 = 4 * q * blk;
                    let mut acc = [F256Unreduced::ZERO; 8];
                    for k in (kc * KC)..((kc + 1) * KC).min(blk) {
                        let i = i0 + k;
                        let fq = [
                            packed_witness[i],
                            packed_witness[i + blk],
                            packed_witness[i + 2 * blk],
                            packed_witness[i + 3 * blk],
                        ];
                        let bq = [bs(i), bs(i + blk), bs(i + 2 * blk), bs(i + 3 * blk)];
                        ligerito::lookahead_accum_group(&fq, &bq, &mut acc);
                    }
                    acc
                })
                .reduce(|| [F256Unreduced::ZERO; 8], ligerito::xor_acc8)
        };
        let (round0, la) = ligerito::lookahead_finish(acc);
        if trace {
            eprintln!(
                "  [open_batch] combine (seeded EqPoint + lookahead, L={l}): {:6.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        return CombinedClaim {
            ring_switches: Vec::new(),
            batching_nonces,
            b_combined: Vec::new(),
            eq_basis: Some((lo, hi, n_lo)),
            eq_gamma: Some(gammas_pd[0]),
            target_combined,
            round0_prime: (round0.u_0, round0.u_2),
            round1_lookahead: Some(la),
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
    let mut b_combined: Vec<F128> = crate::scratch::take_f128(l);

    // The combine is compute-bound (open_combine_probe: ~4.3 ms traffic floor
    // vs ~18 ms total at m=30 on 4 P-threads), and its flat block-parallel
    // shape drains cleanly around slow cores — run it on the all-core (P+E)
    // pool (−29% on the probe on 4P+4E; a wash-to-slight-loss on 10P+4E, so
    // gated on [`crate::ecore_rich_topology`], `FLOCK_ALLCORE=1` overrides).
    // PCS_COMBINE_PCORES_ONLY=1 keeps it on the caller's pool (A/B toggle).
    // Thread count never changes the output bits: every slot is written
    // deterministically and the prime is an XOR reduction (associative +
    // commutative, exact).
    let combine_all_cores =
        std::env::var("PCS_COMBINE_PCORES_ONLY").is_err() && crate::ecore_rich_topology();
    // The fast path's block tail can also accumulate Ligerito's round-1
    // message coefficients (groups of 4; +1 unreduced mul per slot) — the
    // round-0 prime falls out of the same accumulators. Sparse post-combine
    // scatter-adds no longer disable this: the coefficients are linear in
    // the basis, so each scattered delta corrects them below. See
    // `CombinedClaim`.
    let want_lookahead = use_fast;
    let mut round1_lookahead: Option<ligerito::FoldLookahead> = None;
    let b_combined_ref = &mut b_combined;
    let la_ref = &mut round1_lookahead;
    let mut combine = || {
        if use_fast {
            use crate::field::F256Unreduced;
            let b = rs_deferred[0].0.len(); // eq_lo.len(); shared across claims (same split)
            debug_assert!(b >= 2 && b.is_multiple_of(2));
            debug_assert!(rs_deferred.iter().all(|d| d.0.len() == b));
            // Fold one block of every claim into `out_block`; returns true when
            // the block is DEAD (stored as zeros — no prime or lookahead terms
            // to accumulate).
            let fold_block = |hi: usize, out_block: &mut [F128]| -> bool {
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
                    // The prime's (and lookahead's) terms here are `a·0`.
                    return true;
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
                false
            };
            if want_lookahead && b.is_multiple_of(4) {
                // Fused prime + round-1 lookahead tail (groups of 4).
                let acc = b_combined_ref
                    .par_chunks_mut(b)
                    .enumerate()
                    .map(|(hi, out_block)| {
                        if fold_block(hi, out_block) {
                            return [F256Unreduced::ZERO; 8];
                        }
                        let base = hi * b;
                        let mut acc = [F256Unreduced::ZERO; 8];
                        for g in 0..(b / 4) {
                            let i = 4 * g;
                            let fq = [
                                packed_witness[base + i],
                                packed_witness[base + i + 1],
                                packed_witness[base + i + 2],
                                packed_witness[base + i + 3],
                            ];
                            let bq = [
                                out_block[i],
                                out_block[i + 1],
                                out_block[i + 2],
                                out_block[i + 3],
                            ];
                            ligerito::lookahead_accum_group(&fq, &bq, &mut acc);
                        }
                        acc
                    })
                    .reduce(|| [F256Unreduced::ZERO; 8], ligerito::xor_acc8);
                let (msg, la) = ligerito::lookahead_finish(acc);
                *la_ref = Some(la);
                (msg.u_0, msg.u_2)
            } else {
                let (u0, u2) = b_combined_ref
                    .par_chunks_mut(b)
                    .enumerate()
                    .map(|(hi, out_block)| {
                        if fold_block(hi, out_block) {
                            return (F256Unreduced::ZERO, F256Unreduced::ZERO);
                        }
                        // Round-0 prime over this block's pairs (b is even, base is
                        // even). Unreduced 256-bit accumulation, one reduction at
                        // the very end (XOR-linear, bit-identical to reducing per
                        // term).
                        let base = hi * b;
                        let mut u0 = F256Unreduced::ZERO;
                        let mut u2 = F256Unreduced::ZERO;
                        for t in 0..(b / 2) {
                            let s0 = out_block[2 * t];
                            let s1 = out_block[2 * t + 1];
                            let a0 = packed_witness[base + 2 * t];
                            let a1 = packed_witness[base + 2 * t + 1];
                            u0 ^= a0.mul_unreduced(s0);
                            u2 ^= (a0 + a1).mul_unreduced(s0 + s1);
                        }
                        (u0, u2)
                    })
                    .reduce(
                        || (F256Unreduced::ZERO, F256Unreduced::ZERO),
                        |(x0, x2), (y0, y2)| (x0 ^ y0, x2 ^ y2),
                    );
                (u0.reduce(), u2.reduce())
            }
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
                b_combined_ref.par_chunks_mut(1 << 13).for_each(|c| {
                    c.fill(F128::ZERO);
                });
                return (F128::ZERO, F128::ZERO);
            }
            let prime = b_combined_ref
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
        }
    };
    let (mut round0_u0, mut round0_u2) = if combine_all_cores {
        crate::all_core_pool().install(combine)
    } else {
        combine()
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
    // Post-combine b_combined mutations. The round-1 lookahead coefficients
    // are LINEAR in the basis, so every scattered delta corrects them with
    // its own quad contribution (`bq` = the delta at one slot, `fq` = the
    // quad's witness words — 8 unreduced muls per live entry); the prime
    // keeps its existing incremental adjustment.
    let mut la_acc = [crate::field::F256Unreduced::ZERO; 8];
    let la_active = round1_lookahead.is_some();
    let la_correct = |idx: usize, delta: F128, acc: &mut [crate::field::F256Unreduced; 8]| {
        let g = idx & !3usize;
        let fq = [
            packed_witness[g],
            packed_witness[g + 1],
            packed_witness[g + 2],
            packed_witness[g + 3],
        ];
        let mut bq = [F128::ZERO; 4];
        bq[idx & 3] = delta;
        ligerito::lookahead_accum_group(&fq, &bq, acc);
    };
    for (_, output) in rs_results.iter() {
        if let ring_switch::RsEqInd::Sparse { entries, .. } = &output.rs_eq_ind {
            for &(idx, val) in entries {
                b_combined[idx] += val;
                adjust_prime_for_delta(idx, val);
                if la_active {
                    la_correct(idx, val, &mut la_acc);
                }
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
            if la_active {
                for c in 0..eq.live_tensor.len() {
                    la_correct(eq.scatter_idx(c), *g * eq.live_tensor[c], &mut la_acc);
                }
            }
        }
    }
    if let Some(la) = round1_lookahead.as_mut() {
        // The group kernel's message slots duplicate the prime deltas
        // already applied above — only the coefficient half is consumed.
        let (_, delta) = ligerito::lookahead_finish(la_acc);
        la.add(&delta);
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
                crate::scratch::give_f128(v);
            }
            p
        })
        .collect();
    CombinedClaim {
        ring_switches,
        batching_nonces,
        b_combined,
        eq_basis: None,
        eq_gamma: None,
        target_combined,
        round0_prime: (round0_u0, round0_u2),
        round1_lookahead,
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
/// [`open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding`]). Uses
/// `ring_switch::verify_succinct` per claim (no dense `rs_eq_ind`
/// materialization), then drives the succinct recursive Ligerito verifier,
/// evaluating the combined basis only at the residual point, with the
/// matching PCS-transport PoW checks.
#[allow(clippy::too_many_arguments)]
pub fn verify_opening_batch_ligerito_mixed_with_grinding<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    skip_weights: &[&[F128]],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    proof: &BatchOpeningProofLigerito,
    lig_config: &ligerito::VerifierConfig,
    grinding: OpeningGrinding,
    challenger: &mut Ch,
) -> Result<(), VerifyError> {
    let n_rs = claims.len();
    let n_pd = packed_direct.len();
    assert_eq!(skip_weights.len(), n_rs);
    assert_eq!(x_outers.len(), n_rs);
    if proof.ring_switches.len() != n_rs {
        return Err(VerifyError::RingSwitch(
            ring_switch::VerifyError::MalformedProof,
        ));
    }
    assert!(n_rs + n_pd > 0);
    let batch_bits = grinding.claim_batch_bits_for(n_rs + n_pd);
    let expected_batch_nonces = usize::from(batch_bits != 0);
    if proof.batching_nonces.len() != expected_batch_nonces {
        return Err(VerifyError::RingSwitch(
            ring_switch::VerifyError::InvalidGrinding,
        ));
    }
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
        let out = ring_switch::verify_succinct_with_grinding(
            claims[i],
            skip_weights[i],
            x_outers[i],
            &proof.ring_switches[i],
            grinding.ring_switch_bits,
            challenger,
        )
        .map_err(VerifyError::RingSwitch)?;
        rs_outputs.push(out);
    }
    // 2. Bind every PD value, then protect the whole mixed linear batching
    // vector with one PoW. A discrepancy is total-degree one in this vector.
    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas = if batch_bits != 0 {
        challenger
            .verify_pow_and_sample_f128_vec(proof.batching_nonces[0], batch_bits, n_rs + n_pd)
            .ok_or(VerifyError::RingSwitch(
                ring_switch::VerifyError::InvalidGrinding,
            ))?
    } else {
        challenger.sample_f128_vec(n_rs + n_pd)
    };
    let (gammas_rs, gammas_pd) = gammas.split_at(n_rs);

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
    let eval_b_residual = |ris: &[F256], yr_log_n: usize| -> Vec<F256> {
        let yr_len = 1usize << yr_log_n;
        let prefix_len = ris.len();

        // ---- RS claim prefixes ----
        let rs_prefixes: Vec<crate::pcs::tensor_algebra::TensorAlgebra256> = rs_outputs
            .iter()
            .zip(x_outers.iter())
            .map(|(_out, x_outer)| {
                // x_outer[1..] has length log_n; we feed only the ris prefix.
                ring_switch::eval_rs_eq_prefix_f256(&x_outer[1..1 + prefix_len], ris)
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
        let pd_prefix_scalars: Vec<F256> = pd_points_rot
            .iter()
            .map(|pt| {
                pt[..prefix_len]
                    .iter()
                    .zip(ris)
                    .fold(F256::ONE, |acc, (&p, &r)| {
                        acc * (F256::ONE + F256::from(p) + r)
                    })
            })
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
                let mut sum = F256::ZERO;
                for (((out, g), x_outer), prefix) in rs_outputs
                    .iter()
                    .zip(gammas_rs.iter())
                    .zip(x_outers.iter())
                    .zip(rs_prefixes.iter())
                {
                    sum += *g
                        * ring_switch::eval_rs_eq_finish_from_prefix_binary_q_f256(
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
                    let suffix =
                        pt[prefix_len..]
                            .iter()
                            .enumerate()
                            .fold(F128::ONE, |acc, (j, &p)| {
                                acc * if (y_bits >> j) & 1 == 1 {
                                    p
                                } else {
                                    F128::ONE + p
                                }
                            });
                    sum += *prefix_scalar * (*g * suffix);
                }
                sum
            })
            .collect()
    };

    // 5. Drive ligerito SUCCINCT verifier — eval_b_residual is called ONCE
    //    at the residual check (returns all yr_len values in one batch).
    let ok = ligerito::extension::recursive_verifier_with_basis_succinct(
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
    /// PoW witnesses for the outer opening-claim batching coefficients:
    /// ring-switched claims first, then packed-direct claims.
    #[serde(default)]
    pub batching_nonces: Vec<u64>,
    pub merged_rounds: Vec<(F128, F128)>,
    /// PoW witnesses for the dense merged sumcheck, in round order.
    #[serde(default)]
    pub merged_round_nonces: Vec<u64>,
    pub q_eval: F128,
    /// The Frobenius assist. `None` under the fused transport (a full-height
    /// column prefix on a lane-major commitment), where the merged weight is
    /// the inner open's own basis and its evaluation is in closed form.
    #[serde(default)]
    pub frobenius: Option<jagged::MultipointTwistedProof>,
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
                for (dst, e) in cols.iter_mut().zip(crate::lincheck::build_eq_table(zc)) {
                    *dst += g * e;
                }
            }
        }
    }
    groups
}

/// The fused weight's level-1 image in closed form (docs, "full fusion"):
/// given `eq(σ_blk, ·)` over the blocks (F256), `W′[(c0, r)] =
/// Σ_{e live for c0} eq(σ_blk, e)·W[e, (c0, r)]` — the block factor folds
/// into the linearized coefficients (now F256), `c0` into per-`c0` byte
/// tables, exactly as `block_first_phase` does in F128.
fn fused_level1_builder(
    rs: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)>,
    pd: Vec<(Vec<F128>, Vec<F128>)>,
    n_log: usize,
    k_cols: usize,
) -> Box<dyn Fn(&[F256]) -> Vec<F256> + Send + Sync> {
    Box::new(move |eq_blk: &[F256]| {
        use rayon::prelude::*;
        const NB: usize = 1 << LOG_PACKING;
        let n_e = 1usize << (k_cols - 1);
        assert_eq!(eq_blk.len(), n_e);
        let blk = 1usize << (n_log + 1);
        let n_rows = 1usize << n_log;
        let pw = frobenius_bit_powers();
        let eq_rows: Vec<Vec<F128>> = rs
            .iter()
            .map(|(z_row, _, _)| crate::lincheck::build_eq_table(z_row))
            .collect();
        let tabs: Vec<[(Vec<F128>, Vec<F128>); 2]> = rs
            .iter()
            .map(|(_, z_col, coeffs)| {
                let z0 = z_col[0];
                let z_hi = &z_col[1..];
                let mut zj: Vec<F128> = z_hi.to_vec();
                let eq_hi_j: Vec<Vec<F128>> = (0..NB)
                    .map(|_| {
                        let t = crate::lincheck::build_eq_table(&zj);
                        for z in zj.iter_mut() {
                            *z = z.square();
                        }
                        t
                    })
                    .collect();
                [0usize, 1].map(|c0| {
                    let mut kappa = if c0 == 0 { F128::ONE + z0 } else { z0 };
                    let cp: Vec<F256> = (0..NB)
                        .map(|j| {
                            let s =
                                (0..n_e).fold(F256::ZERO, |acc, e| acc + eq_blk[e] * eq_hi_j[j][e]);
                            let out = s * (coeffs[j] * kappa);
                            kappa = kappa.square();
                            out
                        })
                        .collect();
                    // The F256 weights split into two F128 byte tables (the
                    // `c0`/`c1` components), so the fill is two base-field
                    // folds per entry and no extension arithmetic.
                    let w: Vec<F256> = (0..NB)
                        .map(|b| (0..NB).fold(F256::ZERO, |acc, j| acc + cp[j] * pw[b][j]))
                        .collect();
                    let w0: Vec<F128> = w.iter().map(|x| x.c0).collect();
                    let w1: Vec<F128> = w.iter().map(|x| x.c1).collect();
                    (
                        ring_switch::build_fold_byte_table(&w0),
                        ring_switch::build_fold_byte_table(&w1),
                    )
                })
            })
            .collect();
        let pdv: Vec<(Vec<F128>, [F256; 2])> = pd
            .iter()
            .map(|(z_row, cols)| {
                let eq_row = crate::lincheck::build_eq_table(z_row);
                let s = [0usize, 1].map(|c0| {
                    (0..n_e).fold(F256::ZERO, |acc, e| acc + eq_blk[e] * cols[2 * e + c0])
                });
                (eq_row, s)
            })
            .collect();
        let mut wp = vec![F256::ZERO; blk];
        wp.par_chunks_mut(2048).enumerate().for_each(|(ci, chunk)| {
            let off = ci * 2048;
            for (i, slot) in chunk.iter_mut().enumerate() {
                let h = off + i;
                let c0 = h >> n_log;
                let r = h & (n_rows - 1);
                let mut acc = F256::ZERO;
                for (eq_row, tab) in eq_rows.iter().zip(&tabs) {
                    let (t0, t1) = &tab[c0];
                    acc += F256 {
                        c0: ring_switch::fold_one_slot(eq_row[r], t0),
                        c1: ring_switch::fold_one_slot(eq_row[r], t1),
                    };
                }
                for (eq_row, s) in &pdv {
                    acc += s[c0] * eq_row[r];
                }
                *slot = acc;
            }
        });
        wp
    })
}

/// The fused transport's inner open: the Ligerito open of the committed
/// stack whose level-0 basis is the merged weight (`FusedL0`), proving
/// `Σ_d q[d]·W[d] = target` directly.
fn open_fused_ligerito<Ch: Challenger>(
    q: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    target: F128,
    fused: ligerito::extension::FusedL0,
    lig_config: &ligerito::ProverConfig,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    assert_eq!(
        commitment.cap.len(),
        1usize << lig_config.l0_cap_depth(),
        "commitment cap size disagrees with the opener config's L0 query count"
    );
    assert_eq!(
        lig_config.initial_k, commitment.params.log_batch_size,
        "ligerito initial_k must match PcsParams.log_batch_size for L0 reuse"
    );
    assert_eq!(
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
        "ligerito log_inv_rates[0] must match PcsParams.log_inv_rate for L0 reuse"
    );
    let l0_num_lanes = commitment.params.num_ntts();
    assert!(
        l0_num_lanes < 1usize << lig_config.initial_k,
        "the fused open is lane-major"
    );
    challenger.observe_label(b"flock-pcs-open-fused-v0");
    let ligerito = ligerito::recursive_prover_with_basis_precomputed_round0_lanes(
        lig_config,
        q,
        Vec::new(),
        target,
        &prover_data.codeword,
        &prover_data.merkle_tree,
        l0_num_lanes,
        true,
        (F128::ZERO, F128::ZERO),
        None,
        None,
        None,
        None,
        Some(fused),
        challenger,
    );
    BatchOpeningProofLigerito {
        ring_switches: Vec::new(),
        batching_nonces: Vec::new(),
        ligerito,
    }
}

/// What the verifier needs to evaluate the fused weight in closed form:
/// per ring-switch claim its row/column points and γ-scaled linearized
/// coefficients; per scalar group its row point and γ-scaled column table.
struct FusedWeightSpec<'a> {
    n_log: usize,
    k_cols: usize,
    rs: Vec<(&'a [F128], &'a [F128], Vec<F128>)>,
    pd: Vec<(&'a [F128], Vec<F128>)>,
}

/// `Ŵ` at the inner open's residual positions: the bound coordinates `ris`
/// (F256, in the ladder's ROTATED order — block coordinates first, then
/// the low dense coordinates) and the `yr_log_n` residual coordinates as
/// the bits of each position. The fused basis is the full-cube weight, so
/// this is a plain product of twisted eq factors —
/// `Σ_j c_j·κ_{c0}^{2^j}·eq(z_row^{2^j}, x_row)·eq(z_col[1..]^{2^j}, x_blk)` —
/// with the Frobenius twists on the base-field claim point only; nothing
/// here depends on the layout's counts.
fn fused_weight_residual(spec: &FusedWeightSpec<'_>, ris: &[F256], yr_log_n: usize) -> Vec<F256> {
    const NB: usize = 1 << LOG_PACKING;
    let n = spec.n_log;
    let kb = spec.k_cols - 1;
    let total = n + spec.k_cols;
    let prefix_len = ris.len();
    assert_eq!(prefix_len + yr_log_n, total, "bound + residual = dense");
    assert!(prefix_len >= kb, "the block coordinates are bound first");
    let ris_blk = &ris[..kb];
    let n_bound_rows = (prefix_len - kb).min(n);
    let c0_bound = prefix_len == total;
    let yr_len = 1usize << yr_log_n;
    // Column bit `c0` (dense coordinate `n`, rotated `total − 1`): a bound
    // coordinate weights both families by eq(σ, c0); a residual one is a
    // bit of the position.
    let c0_terms = |y: usize| -> [(usize, F256); 2] {
        if c0_bound {
            let x = ris[total - 1];
            [(0, F256::ONE + x), (1, x)]
        } else {
            let c0 = (y >> (total - 1 - prefix_len)) & 1;
            [(c0, F256::ONE), (c0, F256::ZERO)]
        }
    };
    let res_row = |zr: &[F128], y: usize| -> F128 {
        (n_bound_rows..n).fold(F128::ONE, |acc, k| {
            let bit = (y >> (kb + k - prefix_len)) & 1;
            acc * if bit == 1 { zr[k] } else { F128::ONE + zr[k] }
        })
    };
    let eq_bound = |z: &[F128], sigma: &[F256]| -> F256 {
        z.iter().zip(sigma).fold(F256::ONE, |acc, (&zk, &sk)| {
            acc * (F256::ONE + F256::from(zk) + sk)
        })
    };
    let mut out = vec![F256::ZERO; yr_len];
    for (z_row, z_col, coeffs) in &spec.rs {
        let z0 = z_col[0];
        let z_hi = &z_col[1..];
        let mut s = [vec![F256::ZERO; NB], vec![F256::ZERO; NB]];
        let mut zj_rows: Vec<Vec<F128>> = Vec::with_capacity(NB);
        // Frobenius powers of the claim point, one squaring per coordinate
        // per `j` (`x^(2^(j+1)) = (x^(2^j))^2`).
        let mut zj_row: Vec<F128> = z_row.to_vec();
        let mut zj_hi: Vec<F128> = z_hi.to_vec();
        let mut kappa_j = [F128::ONE + z0, z0];
        for j in 0..NB {
            if coeffs[j] != F128::ZERO {
                let bound =
                    eq_bound(&zj_hi, ris_blk) * eq_bound(&zj_row[..n_bound_rows], &ris[kb..]);
                for c0 in 0..2 {
                    s[c0][j] = bound * (coeffs[j] * kappa_j[c0]);
                }
            }
            zj_rows.push(zj_row.clone());
            for z in zj_row
                .iter_mut()
                .chain(zj_hi.iter_mut())
                .chain(kappa_j.iter_mut())
            {
                *z = z.square();
            }
        }
        for (y, slot) in out.iter_mut().enumerate() {
            for (c0, w) in c0_terms(y) {
                if w == F256::ZERO {
                    continue;
                }
                let mut t = F256::ZERO;
                for j in 0..NB {
                    if coeffs[j] != F128::ZERO {
                        t += s[c0][j] * res_row(&zj_rows[j], y);
                    }
                }
                *slot += w * t;
            }
        }
    }
    if !spec.pd.is_empty() {
        let eq_blk = ligerito::extension::build_eq_table256(ris_blk);
        for (z_row, cols) in &spec.pd {
            let colfac = [0usize, 1].map(|c0| {
                (0..eq_blk.len()).fold(F256::ZERO, |acc, e| acc + eq_blk[e] * cols[2 * e + c0])
            });
            let br = eq_bound(&z_row[..n_bound_rows], &ris[kb..]);
            for (y, slot) in out.iter_mut().enumerate() {
                for (c0, w) in c0_terms(y) {
                    if w == F256::ZERO {
                        continue;
                    }
                    *slot += w * colfac[c0] * br * res_row(z_row, y);
                }
            }
        }
    }
    out
}

/// The fused transport's inner verify (see `open_fused_ligerito`).
fn verify_fused_ligerito<Ch: Challenger>(
    commitment: &Commitment,
    target: F128,
    spec: &FusedWeightSpec<'_>,
    proof: &BatchOpeningProofLigerito,
    lig_config: &ligerito::VerifierConfig,
    challenger: &mut Ch,
) -> Result<(), VerifyErrorOpen> {
    if !proof.ring_switches.is_empty() || !proof.batching_nonces.is_empty() {
        return Err(VerifyErrorOpen::Ligerito);
    }
    challenger.observe_label(b"flock-pcs-open-fused-v0");
    let log_n = commitment.params.m - LOG_PACKING;
    let eval = |ris: &[F256], yr_log_n: usize| fused_weight_residual(spec, ris, yr_log_n);
    let ok = ligerito::extension::recursive_verifier_with_basis_succinct(
        lig_config,
        &proof.ligerito,
        log_n,
        target,
        &commitment.cap,
        commitment.params.num_ntts(),
        eval,
        challenger,
    );
    if ok {
        Ok(())
    } else {
        Err(VerifyErrorOpen::Ligerito)
    }
}

/// One ring-switch claim's inputs to the block-first phase.
struct BlockFirstClaim<'a> {
    z_row: &'a [F128],
    z_col: &'a [F128],
    /// `linearized_coefficients` of its γ-scaled fold table.
    coeffs: &'a [F128],
    /// Column bit-bank at `z_row`:
    /// `bank[c·128 + b] = Σ_r eq(z_row, r)·bit_b(q[c·2^n + r])`.
    bank: &'a [F128],
}

/// `x^(2^j)`.
#[inline]
fn frob(mut x: F128, j: usize) -> F128 {
    for _ in 0..j {
        x = x.square();
    }
    x
}

/// Round message of the product sumcheck from rank-1 terms `Σ_t A_t(e)·S_t(e)`
/// (LSB-first pairs): `(Σ_odd A·S, Σ (A0+A1)(S0+S1))`.
fn round_message_terms(terms: &[(Vec<F128>, Vec<F128>)]) -> (F128, F128) {
    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for (a, s) in terms {
        for i in 0..a.len() / 2 {
            let (a0, a1, s0, s1) = (a[2 * i], a[2 * i + 1], s[2 * i], s[2 * i + 1]);
            g_one += a1 * s1;
            g_inf += (a0 + a1) * (s0 + s1);
        }
    }
    (g_one, g_inf)
}

/// `v[i] ← v[2i] + r·(v[2i] + v[2i+1])`, halving `v`.
fn fold_pairs_in_place(v: &mut Vec<F128>, r: F128) {
    let half = v.len() / 2;
    for i in 0..half {
        let (x0, x1) = (v[2 * i], v[2 * i + 1]);
        v[i] = x0 + r * (x0 + x1);
    }
    v.truncate(half);
}

/// Round message of the product sumcheck over a dense pair (see
/// `round_message_terms`).
fn round_message_pairs(a: &[F128], b: &[F128]) -> (F128, F128) {
    use rayon::prelude::*;
    a.par_chunks(4096)
        .zip(b.par_chunks(4096))
        .map(|(ac, bc)| {
            let mut g_one = F128::ZERO;
            let mut g_inf = F128::ZERO;
            for i in 0..ac.len() / 2 {
                let (a0, a1, b0, b1) = (ac[2 * i], ac[2 * i + 1], bc[2 * i], bc[2 * i + 1]);
                g_one += a1 * b1;
                g_inf += (a0 + a1) * (b0 + b1);
            }
            (g_one, g_inf)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |x, y| (x.0 + y.0, x.1 + y.1))
}

/// Reference column bit-bank of the dense stack at `z_row` (the union prover
/// supplies these from its stripe kernels; this is the slow oracle).
fn column_bitbank_reference(q: &[F128], z_row: &[F128], n_log: usize, k_cols: usize) -> Vec<F128> {
    use rayon::prelude::*;
    let eq_row = crate::lincheck::build_eq_table(z_row);
    let n_rows = 1usize << n_log;
    (0..1usize << k_cols)
        .into_par_iter()
        .flat_map_iter(|c| {
            let mut acc = vec![F128::ZERO; 1 << LOG_PACKING];
            for (r, w) in q[c * n_rows..(c + 1) * n_rows].iter().enumerate() {
                let e = eq_row[r];
                let mut lo = w.lo;
                while lo != 0 {
                    acc[lo.trailing_zeros() as usize] += e;
                    lo &= lo - 1;
                }
                let mut hi = w.hi;
                while hi != 0 {
                    acc[64 + hi.trailing_zeros() as usize] += e;
                    hi &= hi - 1;
                }
            }
            acc
        })
        .collect()
}

/// Frobenius powers of the bit basis: `pw[b][k] = β_b^(2^k)`.
fn frobenius_bit_powers() -> Vec<Vec<F128>> {
    const NB: usize = 1 << LOG_PACKING;
    (0..NB)
        .map(|b| {
            let mut row = vec![F128::ZERO; NB];
            row[0] = ring_switch::bit_basis(b);
            for k in 1..NB {
                row[k] = row[k - 1].square();
            }
            row
        })
        .collect()
}

/// The rank-1 terms of the merged weight on the block cube (see
/// `block_first_phase`): per ring-switch claim, Frobenius power `j` and
/// column bit `c0`, the block factor `A_{j,c0}(e)` (masked to the live
/// prefix) and the statistics `S_{j,c0}(e)` from the column bit-bank; per
/// scalar group, one term per `c0`. `Σ_terms Σ_e A(e)·S(e)` is the target.
fn block_first_terms(
    q: &[F128],
    n_log: usize,
    k_cols: usize,
    live_cols: usize,
    // Mask the block factors to the live column prefix (the zero-tail
    // weight the assist evaluates); the fused transport uses the full-cube
    // weight and passes `false`.
    masked: bool,
    rs: &[BlockFirstClaim<'_>],
    pd_groups: &[(&[F128], Vec<F128>)],
    trace: bool,
) -> Vec<(Vec<F128>, Vec<F128>)> {
    use rayon::prelude::*;
    const NB: usize = 1 << LOG_PACKING;
    let t = std::time::Instant::now();
    let kb = k_cols - 1;
    let n_e = 1usize << kb;
    let n_rows = 1usize << n_log;
    // Live blocks per column bit: column `2e + c0` is live iff `2e + c0 < p`.
    let live_e = |c0: usize| {
        if masked {
            (live_cols + 1 - c0) / 2
        } else {
            n_e
        }
    };
    let pw = frobenius_bit_powers();
    let mut terms: Vec<(Vec<F128>, Vec<F128>)> = Vec::new();
    for cl in rs {
        let z0 = cl.z_col[0];
        let z_hi = &cl.z_col[1..];
        let per_j: Vec<(Vec<F128>, Vec<F128>)> = (0..NB)
            .into_par_iter()
            .filter(|&j| cl.coeffs[j] != F128::ZERO)
            .flat_map_iter(|j| {
                let pw = &pw;
                let zj: Vec<F128> = z_hi.iter().map(|&z| frob(z, j)).collect();
                let eq_hi = crate::lincheck::build_eq_table(&zj);
                let kinv = (NB - j) % NB;
                (0..2).map(move |c0| {
                    let kappa = if c0 == 0 { F128::ONE + z0 } else { z0 };
                    let live = live_e(c0);
                    let a: Vec<F128> = (0..n_e)
                        .map(|e| {
                            if e < live {
                                cl.coeffs[j] * eq_hi[e]
                            } else {
                                F128::ZERO
                            }
                        })
                        .collect();
                    let s: Vec<F128> = (0..n_e)
                        .map(|e| {
                            let bank = &cl.bank[(2 * e + c0) * NB..(2 * e + c0 + 1) * NB];
                            let mut acc = F128::ZERO;
                            for b in 0..NB {
                                acc += pw[b][kinv] * bank[b];
                            }
                            frob(kappa * acc, j)
                        })
                        .collect();
                    (a, s)
                })
            })
            .collect();
        terms.extend(per_j);
    }
    for (z_row, cols) in pd_groups {
        let eq_row = crate::lincheck::build_eq_table(z_row);
        let f: Vec<F128> = (0..1usize << k_cols)
            .into_par_iter()
            .map(|c| {
                if c < live_cols {
                    q[c * n_rows..(c + 1) * n_rows]
                        .iter()
                        .zip(&eq_row)
                        .fold(F128::ZERO, |acc, (&x, &y)| acc + x * y)
                } else {
                    F128::ZERO
                }
            })
            .collect();
        for c0 in 0..2 {
            let live = live_e(c0);
            let a: Vec<F128> = (0..n_e)
                .map(|e| {
                    if e < live {
                        cols[2 * e + c0]
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            let s: Vec<F128> = (0..n_e).map(|e| f[2 * e + c0]).collect();
            terms.push((a, s));
        }
    }
    if trace {
        eprintln!(
            "  [open_merged] block statistics ({} terms × {n_e}): {:6.2} ms",
            terms.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    terms
}

/// The block-first phase of the merged sumcheck (docs, "direct transport").
///
/// The dense index is `d = e·2^(n+1) + c0·2^n + r` (block `e` of two lane
/// columns, column bit `c0`, row `r`). Every weight term is rank-1 across the
/// `(e | c0, r)` split once the fold map is linearized:
/// `W_i[e, (c0, r)] = Σ_j c_ij·u_i(e)^(2^j)·(eq(z_col[0], c0)·eq(z_row, r))^(2^j)`
/// with `u_i(e) = eq(z_col[1..], e)`, masked to the live column prefix. The
/// `k−1` block rounds therefore need only, per term, the block statistics
/// `S_ij[e] = (eq(z_col[0], c0)·Σ_b β_b^(2^-j)·bank[(2e+c0)·128 + b])^(2^j)`
/// — a Frobenius transform of the column bit-bank — against
/// `A_ij[e] = c_ij·eq(z_col[1..]^(2^j), e)`. After them, `q` is folded over
/// the bound block coordinates and the row-side weight `W'` is written in
/// closed form (the block factor folds into the linearized coefficients, the
/// `c0` bit into the byte tables), ready for the ordinary row rounds.
#[allow(clippy::too_many_arguments)]
fn block_first_phase<Ch: Challenger>(
    q: &[F128],
    n_log: usize,
    k_cols: usize,
    live_cols: usize,
    rs: &[BlockFirstClaim<'_>],
    pd_groups: &[(&[F128], Vec<F128>)],
    grinding: OpeningGrinding,
    challenger: &mut Ch,
    merged_rounds: &mut Vec<(F128, F128)>,
    merged_round_nonces: &mut Vec<u64>,
    r_blk: &mut Vec<F128>,
    trace: bool,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    const NB: usize = 1 << LOG_PACKING;
    let kb = k_cols - 1;
    let blk = 1usize << (n_log + 1);
    let n_rows = 1usize << n_log;
    let live_e = |c0: usize| (live_cols + 1 - c0) / 2;
    let pw = frobenius_bit_powers();
    let mut terms = block_first_terms(q, n_log, k_cols, live_cols, true, rs, pd_groups, trace);

    // The block rounds.
    let t = std::time::Instant::now();
    let (mut g_one, mut g_inf) = round_message_terms(&terms);
    for round in 0..kb {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = if grinding.merged_round_bits != 0 {
            let (nonce, r) = challenger.grind_pow_and_sample_f128(grinding.merged_round_bits);
            merged_round_nonces.push(nonce);
            r
        } else {
            challenger.sample_f128()
        };
        merged_rounds.push((g_one, g_inf));
        r_blk.push(r);
        for (a, s) in terms.iter_mut() {
            fold_pairs_in_place(a, r);
            fold_pairs_in_place(s, r);
        }
        if round + 1 < kb {
            (g_one, g_inf) = round_message_terms(&terms);
        }
    }

    // q' = q folded over the bound block coordinates.
    let eq_blk = crate::lincheck::build_eq_table(r_blk);
    let n_live_e = live_cols.div_ceil(2);
    let mut qp = vec![F128::ZERO; blk];
    qp.par_chunks_mut(2048).enumerate().for_each(|(ci, chunk)| {
        let off = ci * 2048;
        for e in 0..n_live_e {
            let w = eq_blk[e];
            let src = &q[e * blk + off..e * blk + off + chunk.len()];
            for (d, &x) in chunk.iter_mut().zip(src) {
                *d += w * x;
            }
        }
    });

    // W'[(c0, r)] = Σ_{e live for c0} eq(r_blk, e)·W[e, (c0, r)], in closed
    // form: the block factor folds into the linearized coefficients,
    // c'_{j,c0} = c_j·Σ_e eq(r_blk, e)·eq(z_col[1..]^(2^j), e), and the
    // column bit bakes eq(z_col[0], c0) into the per-c0 byte tables.
    let eq_rows: Vec<Vec<F128>> = rs
        .iter()
        .map(|cl| crate::lincheck::build_eq_table(cl.z_row))
        .collect();
    let tabs: Vec<[Vec<F128>; 2]> = rs
        .iter()
        .map(|cl| {
            let z0 = cl.z_col[0];
            let z_hi = &cl.z_col[1..];
            let eq_hi_j: Vec<Vec<F128>> = (0..NB)
                .map(|j| {
                    let zj: Vec<F128> = z_hi.iter().map(|&z| frob(z, j)).collect();
                    crate::lincheck::build_eq_table(&zj)
                })
                .collect();
            [0usize, 1].map(|c0| {
                let kappa = if c0 == 0 { F128::ONE + z0 } else { z0 };
                let live = live_e(c0);
                let cp: Vec<F128> = (0..NB)
                    .map(|j| {
                        let s =
                            (0..live).fold(F128::ZERO, |acc, e| acc + eq_blk[e] * eq_hi_j[j][e]);
                        cl.coeffs[j] * s * frob(kappa, j)
                    })
                    .collect();
                let w: Vec<F128> = (0..NB)
                    .map(|b| (0..NB).fold(F128::ZERO, |acc, j| acc + cp[j] * pw[b][j]))
                    .collect();
                ring_switch::build_fold_byte_table(&w)
            })
        })
        .collect();
    let pd: Vec<(Vec<F128>, [F128; 2])> = pd_groups
        .iter()
        .map(|(z_row, cols)| {
            let eq_row = crate::lincheck::build_eq_table(z_row);
            let s = [0usize, 1].map(|c0| {
                (0..live_e(c0)).fold(F128::ZERO, |acc, e| acc + eq_blk[e] * cols[2 * e + c0])
            });
            (eq_row, s)
        })
        .collect();
    let mut wp = vec![F128::ZERO; blk];
    wp.par_chunks_mut(2048).enumerate().for_each(|(ci, chunk)| {
        let off = ci * 2048;
        for (i, slot) in chunk.iter_mut().enumerate() {
            let h = off + i;
            let c0 = h >> n_log;
            let r = h & (n_rows - 1);
            let mut acc = F128::ZERO;
            for (eq_row, tab) in eq_rows.iter().zip(&tabs) {
                acc += ring_switch::fold_one_slot(eq_row[r], &tab[c0]);
            }
            for (eq_row, s) in &pd {
                acc += s[c0] * eq_row[r];
            }
            *slot = acc;
        }
    });
    if trace {
        eprintln!(
            "  [open_merged] block rounds ({kb}) + q'/W' (2^{}): {:6.2} ms",
            n_log + 1,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    (qp, wp)
}

/// The remaining rounds of the merged product sumcheck over `(a0, b0)`
/// (LSB-first pairs), starting from the caller's round message. `live` bounds
/// the non-zero prefix of `a0` (fold ranges shrink with it; `b0` must be zero
/// on `[area, live)`). Appends messages, grinding nonces and challenges;
/// returns the fully folded pair and the statistics seed captured at
/// `seed_len` (see `open_batch_mixed_ligerito_seeded`).
#[allow(clippy::too_many_arguments)]
fn merged_sumcheck_rounds<Ch: Challenger>(
    a0: &[F128],
    b0: &[F128],
    live: usize,
    mut g_one: F128,
    mut g_inf: F128,
    seed_len: usize,
    grinding: OpeningGrinding,
    challenger: &mut Ch,
    merged_rounds: &mut Vec<(F128, F128)>,
    merged_round_nonces: &mut Vec<u64>,
    rho: &mut Vec<F128>,
) -> (F128, F128, Option<Vec<F128>>) {
    let l = a0.len();
    debug_assert_eq!(b0.len(), l);
    let n_rounds = l.trailing_zeros() as usize;
    debug_assert_eq!(1usize << n_rounds, l);
    if n_rounds == 0 {
        return (a0[0], b0[0], None);
    }
    let mut live = live.min(l);
    let mut sa = crate::scratch::take_f128(l / 2);
    let mut sb = crate::scratch::take_f128(l / 2);
    let mut a = crate::scratch::take_f128(l / 4);
    let mut bb = crate::scratch::take_f128(l / 4);
    let mut cur = l;
    let mut seeded_stats: Option<Vec<F128>> = None;
    for round in 0..n_rounds {
        let half = cur / 2;
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = if grinding.merged_round_bits != 0 {
            let (nonce, r) = challenger.grind_pow_and_sample_f128(grinding.merged_round_bits);
            merged_round_nonces.push(nonce);
            r
        } else {
            challenger.sample_f128()
        };
        merged_rounds.push((g_one, g_inf));
        rho.push(r);
        let (a_src, b_src): (&[F128], &[F128]) = if round == 0 { (a0, b0) } else { (&a, &bb) };
        if cur == seed_len && seed_len < l {
            let lv = live.min(cur);
            let mut g = a_src[..lv].to_vec();
            g.resize(cur, F128::ZERO);
            seeded_stats = Some(g);
        }
        if cur > 2 {
            let lv = live.min(cur);
            let lhalf = lv / 2;
            (g_one, g_inf) = jagged::fold_and_round_oop_par(
                &a_src[..lv],
                &b_src[..lv],
                r,
                &mut sa[..lhalf],
                &mut sb[..lhalf],
            );
            let next = lhalf.next_multiple_of(4).min(half).max(lhalf);
            for i in lhalf..next {
                sa[i] = F128::ZERO;
                sb[i] = F128::ZERO;
            }
            live = next;
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
    let out = (a[0], bb[0], seeded_stats);
    crate::scratch::give_f128(sa);
    crate::scratch::give_f128(sb);
    crate::scratch::give_f128(a);
    crate::scratch::give_f128(bb);
    out
}

#[allow(clippy::too_many_arguments)]
pub fn open_batch_merged<Ch: Challenger>(
    q: Vec<F128>,
    // The M-variable padded buffer the ring switch reads. `None` means it IS
    // `q` — identity compaction, where the caller moved one buffer in rather
    // than cloning it to satisfy the by-value/by-ref split.
    padded_witness: Option<&[F128]>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    // Per ring-switch claim, the column bit-bank at its row point (see
    // `lincheck::union_bitbank_fold`); consumed by the block-first transport
    // when the stack is a full-height column prefix. Missing entries are
    // rebuilt from `q` (slow reference fold — tests, not the union prover).
    column_bitbanks: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    heights: &[u64],
    n_log: usize,
    lig_config: &ligerito::ProverConfig,
    grinding: OpeningGrinding,
    challenger: &mut Ch,
) -> MergedOpenProof {
    let trace = std::env::var("PCS_TRACE").is_ok();
    let t_total = std::time::Instant::now();
    // Belt-and-braces on the cap-depth derivation: the commit-time cap
    // (from `PcsParams::l0_cap_depth`) must be the layer the opener's
    // config implies — a config-source disagreement fails loudly here at
    // prove time instead of as a verifier reject.
    assert_eq!(
        commitment.cap.len(),
        1usize << lig_config.l0_cap_depth(),
        "commitment cap size disagrees with the opener config's L0 query count"
    );
    debug_assert_eq!(
        commitment.cap.as_slice(),
        crate::merkle::cap_layer(
            &prover_data.merkle_tree,
            commitment.params.n_leaves(),
            lig_config.l0_cap_depth(),
        ),
        "commitment cap is not the prover tree's cap layer"
    );
    challenger.observe_label(b"flock-merged-open-v1");
    let t = std::time::Instant::now();
    // Element-only registries produce no ring-switched claims; skip the batch
    // entirely (the callee asserts a non-empty batch). This branch DEFINES the
    // element-only merged transcript: nothing is absorbed for the empty batch,
    // exactly as in the mixed open's `n_rs > 0` guard.
    let padded_witness = padded_witness.unwrap_or(&q);
    let batch_bits = grinding.claim_batch_bits_for(x_outers.len() + packed_direct.len());
    let mut rs_results = if !x_outers.is_empty() {
        ring_switch::prove_batched_padded_with_precomputed_unbatched_and_grinding(
            padded_witness,
            x_outers,
            precomputed_s_hat_v,
            padding,
            grinding.ring_switch_bits,
            challenger,
        )
    } else {
        Vec::new()
    };
    if trace {
        eprintln!(
            "  [open_merged] ring_switch: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // Packed-direct claims take their γ's from the same stream, AFTER the
    // ring-switched ones — the verifier draws them in the same order.
    //
    // VALUE-ONLY absorb (v1): a claim's POINT is never the prover's to choose
    // — every packed-direct caller derives it from earlier transcript
    // challenges plus statement constants (gather points from the wiring
    // GKR's ρ and the cell space, element points from the region PIOP's own
    // challenges and the frozen prefix), and the verifier RECOMPUTES it
    // rather than reading it from the proof. Absorbing a deterministic
    // public function of (statement, transcript-prefix) adds no binding, and
    // at ~24 words × hundreds of claims it was ~92 KB of transcript the
    // recursion circuit re-hashed per child. The γ's still bind the VALUES,
    // which are genuine prover messages. (The non-merged `open_batch` has
    // absorbed value-only since birth — this aligns the merged intake.)
    for c in packed_direct {
        challenger.observe_f128(c.value);
    }
    let mut batching_nonces = Vec::with_capacity(usize::from(batch_bits != 0));
    let n_gammas = x_outers.len() + packed_direct.len();
    let gammas = if batch_bits != 0 {
        let (nonce, gammas) = challenger.grind_pow_and_sample_f128_vec(batch_bits, n_gammas);
        batching_nonces.push(nonce);
        gammas
    } else {
        challenger.sample_f128_vec(n_gammas)
    };
    let (gammas_rs, gammas_pd) = gammas.split_at(x_outers.len());
    for ((_, output), &gamma) in rs_results.iter_mut().zip(gammas_rs) {
        output.rs_eq_ind.scale_in_place(gamma);
    }

    let mut target = F128::ZERO;
    for ((_, o), g) in rs_results.iter().zip(gammas_rs.iter()) {
        target += *g * o.sumcheck_claim;
    }
    for (c, g) in packed_direct.iter().zip(gammas_pd.iter()) {
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
        gammas_pd,
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

    // Linearized coefficients of the (γ-scaled) fold tables: the block-first
    // transport needs them now; the Frobenius assist below reuses them.
    let coeffs: Vec<Vec<F128>> = rs_results
        .iter()
        .map(|(_, o)| match &o.rs_eq_ind {
            ring_switch::RsEqInd::DeferredDense { table, .. } => {
                ring_switch::linearized_coefficients(table)
            }
            _ => unreachable!("checked above"),
        })
        .collect();

    // The block-first family (docs, "direct transport"): a full-height
    // column prefix makes every weight term rank-1 across the
    // (lane block | column bit, row) split, so the block coordinates can be
    // bound from O(2^k) block statistics without a 2^m weight vector. The
    // column bit-banks are the statistics' source.
    let rect = jagged::rectangular_prefix_columns(heights, n_log);
    let banks: Vec<std::borrow::Cow<'_, [F128]>> = if rect.is_some() {
        x_outers
            .iter()
            .enumerate()
            .map(|(i, x)| match column_bitbanks.get(i).copied().flatten() {
                Some(b) if b.len() == 1usize << (k_cols + LOG_PACKING) => {
                    std::borrow::Cow::Borrowed(b)
                }
                _ => {
                    if trace {
                        eprintln!(
                            "  [open_merged] claim {i}: no column bit-bank supplied, \
                             reference fold"
                        );
                    }
                    std::borrow::Cow::Owned(column_bitbank_reference(
                        &q,
                        &x[1..1 + n_log],
                        n_log,
                        k_cols,
                    ))
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let rs_bf: Vec<BlockFirstClaim<'_>> = x_outers
        .iter()
        .zip(&coeffs)
        .zip(&banks)
        .map(|((x, c), b)| BlockFirstClaim {
            z_row: &x[1..1 + n_log],
            z_col: &x[1 + n_log..],
            coeffs: c,
            bank: b,
        })
        .collect();

    // Full fusion (docs, "full fusion"): on a lane-major commitment whose
    // ladder binds exactly the block coordinates, the weight's rank-1 terms
    // ARE the inner open's level-0 basis — no merged sumcheck, no assist.
    // The verifier evaluates Ŵ at the ladder's point in closed form.
    let lane_major = commitment.params.num_ntts() < 1usize << lig_config.initial_k;
    if let Some(live_cols) = rect.filter(|_| lane_major && lig_config.initial_k + 1 == k_cols) {
        let t = std::time::Instant::now();
        // The fused basis is the FULL-cube weight (q vanishes on dead columns,
        // so the target is the same); its evaluation is then count-independent.
        let terms = block_first_terms(
            &q, n_log, k_cols, live_cols, false, &rs_bf, &pd_groups, trace,
        );
        let terms: Vec<(Vec<F256>, Vec<F256>)> = terms
            .into_iter()
            .map(|(a, s)| {
                (
                    a.into_iter().map(F256::from).collect(),
                    s.into_iter().map(F256::from).collect(),
                )
            })
            .collect();
        let level1 = fused_level1_builder(
            x_outers
                .iter()
                .zip(&coeffs)
                .map(|(x, c)| (x[1..1 + n_log].to_vec(), x[1 + n_log..].to_vec(), c.clone()))
                .collect(),
            pd_groups
                .iter()
                .map(|(z, cols)| (z.to_vec(), cols.clone()))
                .collect(),
            n_log,
            k_cols,
        );
        if trace {
            eprintln!(
                "  [open_merged] fused L0 terms ({}): {:6.2} ms",
                terms.len(),
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let t = std::time::Instant::now();
        let inner = open_fused_ligerito(
            q,
            prover_data,
            commitment,
            target,
            ligerito::extension::FusedL0 { terms, level1 },
            lig_config,
            challenger,
        );
        if trace {
            eprintln!(
                "  [open_merged] fused inner open: {:6.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            eprintln!(
                "  [open_merged] TOTAL: {:6.2} ms",
                t_total.elapsed().as_secs_f64() * 1e3
            );
        }
        return MergedOpenProof {
            ring_switches: rs_results.into_iter().map(|(p, _)| p).collect(),
            batching_nonces,
            merged_rounds: Vec::new(),
            merged_round_nonces: Vec::new(),
            q_eval: F128::ZERO,
            frobenius: None,
            inner,
        };
    }

    let t = std::time::Instant::now();
    let mut merged_rounds = Vec::with_capacity(dense_log);
    let mut merged_round_nonces =
        Vec::with_capacity((grinding.merged_round_bits != 0) as usize * dense_log);
    let mut rho = Vec::with_capacity(dense_log);
    let seed_len = 1usize << lig_config.initial_k;
    let (q_eval, w_eval, seeded_stats) = if let Some(live_cols) = rect {
        // Block-first transport: the verifier rotates ρ back into
        // coordinate order under the same predicate.
        let rs = rs_bf;
        let (qp, wp) = block_first_phase(
            &q,
            n_log,
            k_cols,
            live_cols,
            &rs,
            &pd_groups,
            grinding,
            challenger,
            &mut merged_rounds,
            &mut merged_round_nonces,
            &mut rho,
            trace,
        );
        let (g_one, g_inf) = round_message_pairs(&qp, &wp);
        let (q_eval, w_eval, _) = merged_sumcheck_rounds(
            &qp,
            &wp,
            qp.len(),
            g_one,
            g_inf,
            usize::MAX,
            grinding,
            challenger,
            &mut merged_rounds,
            &mut merged_round_nonces,
            &mut rho,
        );
        // Round order → coordinate order: the first k−1 rounds bound the
        // block coordinates n+1.. of the dense index.
        rho.rotate_left(k_cols - 1);
        (q_eval, w_eval, None)
    } else {
        // The twisted weight over the dense cube (count-proportional
        // Φ-pass; zero tail past the jagged area).
        let (mut w, (u0, u2)) = jagged::build_merged_weight_and_prime(&params, &weight_claims, &q);
        if trace {
            eprintln!(
                "  [open_merged] W build + round-0 prime (2^{} words): {:6.2} ms",
                dense_log,
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        let l = q.len();
        let area = (params.area() as usize).min(l);
        let live = area.next_multiple_of(4).clamp(4, l);
        for slot in &mut w[area..live] {
            *slot = F128::ZERO;
        }
        let out = merged_sumcheck_rounds(
            &q,
            &w,
            live,
            target + u0,
            u2,
            seed_len,
            grinding,
            challenger,
            &mut merged_rounds,
            &mut merged_round_nonces,
            &mut rho,
        );
        crate::scratch::give_f128(w);
        out
    };
    if trace {
        eprintln!(
            "  [open_merged] merged sumcheck ({dense_log} rounds): {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Frobenius assist: proves V = Ŵ(ρ).
    let t = std::time::Instant::now();
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
    let t_assist = std::time::Instant::now();
    // ---- eq-basis Ligerito opening of q̂(ρ): one packed-direct claim on
    // the existing mixed path (whose verifier evaluates eq residuals in
    // closed form — no b_tilde machinery).
    let pd = PackedDirectClaim {
        point: rho.clone(),
        value: q_eval,
        eq_ind: DirectEqInd::EqPoint(rho.clone()),
    };
    // FORK/JOIN: the assist and the inner open are SIBLINGS, not a chain.
    // Both consume only `rho` and `q_eval` from the merged sumcheck above,
    // and neither reads the other's output — the assist's `V` is checked
    // against the sumcheck's folded claim, not against the opening. So they
    // run CONCURRENTLY on domain-separated chains: the assist takes a child
    // seeded from the parent, the inner open continues on the parent, and
    // the child's closing digest merges after. Sound for the same reason
    // the wiring fork is: both branches bind only the prefix through ρ,
    // which the fork point already covers.
    //
    // The merge is what binds the child's chain into the parent even though
    // nothing samples after it here — keep it, so composing this open into
    // a longer transcript stays sound without revisiting the fork.
    let mut ch_a = challenger.fork(b"flock-par-assist-v1");
    // The two-product multipoint replacement
    // (docs/multipoint-twisted-assist.tex): 128R + P claimed dual values +
    // one two-product sumcheck + ONE untwisted anchor, instead of a
    // per-statement assist — family K collapses, and every verifier piece
    // is a shape the recursion circuit already has.
    let (frobenius, inner) = rayon::join(
        || {
            jagged::prove_multipoint_twisted_with_grinding(
                &params,
                &fclaims,
                &gclaims,
                &rho,
                grinding.multipoint,
                &mut ch_a,
            )
        },
        || {
            open_batch_mixed_ligerito_seeded(
                q,
                prover_data,
                commitment,
                &[],
                &[],
                &[pd],
                &PaddingSpec::dense(commitment.params.m),
                lig_config,
                grinding,
                seeded_stats,
                challenger,
            )
        },
    );
    challenger.merge_child(ch_a);
    if trace {
        eprintln!(
            "  [open_merged] coeffs + (multipoint assist ∥ inner open): {:6.2} ms \
             (join wall {:6.2} ms = max of the two branches)",
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

    if trace {
        eprintln!(
            "  [open_merged] TOTAL: {:6.2} ms",
            t_total.elapsed().as_secs_f64() * 1e3
        );
    }
    drop(fclaims);
    MergedOpenProof {
        ring_switches: rs_results.into_iter().map(|(p, _)| p).collect(),
        batching_nonces,
        merged_rounds,
        merged_round_nonces,
        q_eval,
        frobenius: Some(frobenius),
        inner,
    }
}

#[allow(clippy::too_many_arguments)]
/// Rebuild the member decomposition [`scalar_claim_groups`] packed, express
/// every count-dependent `W`-value as a RAW claim on the layout table, and
/// tie the decomposition to the anchor's own statement factors — exactly,
/// since bilinear forms are linear in their weights and `GF(2^128)` sums
/// reassociate. Panics on any mismatch: an export that does not recombine
/// to what the verifier itself checked must never leave the process.
fn assemble_jagged_assertion(
    params: &jagged::JaggedParams,
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    gammas_pd: &[F128],
    pd_groups: &[(&[F128], Vec<F128>)],
    n_log: usize,
    mp: &jagged::MultipointDefer,
) -> crate::matrix_fold::JaggedAssertion {
    use crate::matrix_fold::{JaggedClaim, JaggedRowWeight, JaggedTable};
    let table = JaggedTable::from_params(params);

    // RS claims: raw eq(z_col) at σ, claim order.
    let rs: Vec<JaggedClaim> = x_outers
        .iter()
        .map(|x| {
            JaggedClaim::honest(
                JaggedRowWeight::eq(x[1 + n_log..].to_vec()),
                mp.sigma.clone(),
                &table,
            )
        })
        .collect();

    // The pd members re-grouped by row point in scalar_claim_groups' exact
    // order: boolean columns are one-hot addresses, everything else stays a
    // dense member. The regrouping must mirror the groups the anchor saw.
    struct G<'a> {
        z_row: &'a [F128],
        combo: Vec<(F128, u32)>,
        dense: Vec<(F128, &'a [F128])>,
    }
    let mut gs: Vec<G<'_>> = Vec::new();
    for (c, &g) in packed_direct.iter().zip(gammas_pd) {
        let (zr, zc) = (&c.point[..n_log], &c.point[n_log..]);
        let hot: Option<usize> = zc.iter().enumerate().try_fold(0usize, |acc, (i, &x)| {
            if x == F128::ZERO {
                Some(acc)
            } else if x == F128::ONE {
                Some(acc | (1 << i))
            } else {
                None
            }
        });
        let slot = match gs.iter_mut().find(|m| m.z_row == zr) {
            Some(m) => m,
            None => {
                gs.push(G {
                    z_row: zr,
                    combo: Vec::new(),
                    dense: Vec::new(),
                });
                gs.last_mut().unwrap()
            }
        };
        match hot {
            Some(h) => slot.combo.push((g, h as u32)),
            None => slot.dense.push((g, zc)),
        }
    }
    assert_eq!(
        gs.len(),
        pd_groups.len(),
        "the regrouping must mirror scalar_claim_groups"
    );
    for (m, (zr, _)) in gs.iter().zip(pd_groups) {
        assert_eq!(m.z_row, *zr, "group order must mirror scalar_claim_groups");
    }

    let groups: Vec<(Option<JaggedClaim>, Vec<(F128, JaggedClaim)>)> = gs
        .iter()
        .map(|m| {
            let combo = (!m.combo.is_empty()).then(|| {
                JaggedClaim::honest(
                    JaggedRowWeight::Combo(m.combo.clone()),
                    mp.sigma.clone(),
                    &table,
                )
            });
            let dense = m
                .dense
                .iter()
                .map(|&(g, zc)| {
                    (
                        g,
                        JaggedClaim::honest(
                            JaggedRowWeight::eq(zc.to_vec()),
                            mp.sigma.clone(),
                            &table,
                        ),
                    )
                })
                .collect();
            (combo, dense)
        })
        .collect();

    // THE TIE: the decomposition recombines to the anchor's own statement
    // factors. Statement order is frobenius_statements' — RS claims with a
    // live coefficient first, then the groups. A row-point collision would
    // merge statements and break the mapping, so its absence is asserted.
    for (i, xa) in x_outers.iter().enumerate() {
        for xb in &x_outers[i + 1..] {
            assert!(
                xa[1..1 + n_log] != xb[1..1 + n_log],
                "RS row points must be distinct"
            );
        }
        for (zr, _) in pd_groups {
            assert!(
                xa[1..1 + n_log] != **zr,
                "an RS row point collides with a group's"
            );
        }
    }
    let mut ws = mp.statement_ws.iter();
    for (rc, &coeff) in rs.iter().zip(&mp.rs_coeffs) {
        if coeff.is_zero() {
            continue;
        }
        assert_eq!(
            *ws.next().expect("a statement per live RS claim"),
            coeff * rc.value,
            "RS statement recombination"
        );
    }
    for ((combo, dense), &coeff) in groups.iter().zip(&mp.group_coeffs) {
        let raw = combo.as_ref().map_or(F128::ZERO, |c| c.value)
            + dense
                .iter()
                .fold(F128::ZERO, |a, &(g, ref c)| a + g * c.value);
        assert_eq!(
            *ws.next().expect("a statement per group"),
            coeff * raw,
            "group statement recombination"
        );
    }
    assert!(ws.next().is_none(), "every statement accounted for");

    crate::matrix_fold::JaggedAssertion {
        k: params.k,
        m: params.m,
        rs,
        groups,
    }
}

pub fn verify_batch_merged<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    z_skips: &[crate::lincheck::SkipPoint],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    heights: &[u64],
    n_log: usize,
    proof: &MergedOpenProof,
    lig_config: &ligerito::VerifierConfig,
    grinding: OpeningGrinding,
    challenger: &mut Ch,
) -> Result<(), VerifyErrorOpen> {
    verify_batch_merged_core(
        commitment,
        claims,
        z_skips,
        x_outers,
        packed_direct,
        heights,
        n_log,
        proof,
        lig_config,
        grinding,
        challenger,
        None,
    )
}

/// [`verify_batch_merged`] plus the jagged-layout export — the count win's
/// deferral seam. The anchor's count-dependent `W`-values come back as RAW
/// claims on the layout table ([`crate::matrix_fold::JaggedAssertion`]),
/// each tied to the verifier's own expect by an EXACT recombination assert
/// (field sums and products distribute, so the tie is equality, not
/// tolerance). Transcript-identical to the plain entry.
#[allow(clippy::too_many_arguments)]
pub fn verify_batch_merged_deferred<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    z_skips: &[crate::lincheck::SkipPoint],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    heights: &[u64],
    n_log: usize,
    proof: &MergedOpenProof,
    lig_config: &ligerito::VerifierConfig,
    grinding: OpeningGrinding,
    challenger: &mut Ch,
) -> Result<crate::matrix_fold::JaggedAssertion, VerifyErrorOpen> {
    let mut out = None;
    verify_batch_merged_core(
        commitment,
        claims,
        z_skips,
        x_outers,
        packed_direct,
        heights,
        n_log,
        proof,
        lig_config,
        grinding,
        challenger,
        Some(&mut out),
    )?;
    Ok(out.expect("the deferred core fills the export"))
}

#[allow(clippy::too_many_arguments)]
fn verify_batch_merged_core<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    z_skips: &[crate::lincheck::SkipPoint],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    heights: &[u64],
    n_log: usize,
    proof: &MergedOpenProof,
    lig_config: &ligerito::VerifierConfig,
    grinding: OpeningGrinding,
    challenger: &mut Ch,
    defer: Option<&mut Option<crate::matrix_fold::JaggedAssertion>>,
) -> Result<(), VerifyErrorOpen> {
    let n_rs = claims.len();
    let n_pd = packed_direct.len();
    assert_eq!(z_skips.len(), n_rs);
    assert_eq!(x_outers.len(), n_rs);
    if proof.ring_switches.len() != n_rs {
        return Err(VerifyErrorOpen::Assist);
    }
    let batch_bits = grinding.claim_batch_bits_for(n_rs + n_pd);
    let expected_batch_nonces = usize::from(batch_bits != 0);
    if proof.batching_nonces.len() != expected_batch_nonces {
        return Err(VerifyErrorOpen::Assist);
    }
    // `VERIFY_TRACE` phase split. The Ligerito inner verify has its own
    // `LIG_VERIFY_TRACE`, but it is a small tail here — the jagged Frobenius
    // assist is the term that scales with the row/column split.
    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let tfmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };
    challenger.observe_label(b"flock-merged-open-v1");
    let t = std::time::Instant::now();
    let mut rs_outputs = Vec::with_capacity(n_rs);
    for i in 0..n_rs {
        let skip_w = z_skips[i].weights(LOG_PACKING - 1);
        let out = ring_switch::verify_succinct_with_grinding(
            claims[i],
            &skip_w,
            x_outers[i],
            &proof.ring_switches[i],
            grinding.ring_switch_bits,
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
    // Packed-direct values are bound before one PoW protects the entire mixed
    // coefficient vector (total discrepancy degree one).
    // VALUE-ONLY absorb (v1) — see the prover-side intake for the argument:
    // the points are transcript-derived and verifier-recomputed, never prover
    // messages.
    for c in packed_direct {
        challenger.observe_f128(c.value);
    }
    let gammas_all = if batch_bits != 0 {
        challenger
            .verify_pow_and_sample_f128_vec(proof.batching_nonces[0], batch_bits, n_rs + n_pd)
            .ok_or(VerifyErrorOpen::Assist)?
    } else {
        challenger.sample_f128_vec(n_rs + n_pd)
    };
    let (gammas, gammas_pd) = gammas_all.split_at(n_rs);
    let mut target = F128::ZERO;
    for (out, g) in rs_outputs.iter().zip(gammas.iter()) {
        target += *g * out.sumcheck_claim;
    }
    for (c, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target += *g * c.value;
    }

    let dense_log = commitment.params.m - LOG_PACKING;
    // Full fusion (see the prover): the inner open's basis is the weight
    // itself; no merged rounds, no assist, Ŵ in closed form at its point.
    let k_cols_pub = dense_log - n_log;
    let lane_major = commitment.params.num_ntts() < 1usize << lig_config.initial_k;
    if jagged::rectangular_prefix_columns(heights, n_log)
        .filter(|_| lane_major && lig_config.initial_k + 1 == k_cols_pub)
        .is_some()
    {
        // Every field of the transport is bound: the fused proof carries no
        // rounds, no assist and a zero `q_eval`.
        if !proof.merged_rounds.is_empty()
            || !proof.merged_round_nonces.is_empty()
            || proof.frobenius.is_some()
            || proof.q_eval != F128::ZERO
        {
            return Err(VerifyErrorOpen::VirtualOpen);
        }
        let coeffs: Vec<Vec<F128>> = rs_outputs
            .iter()
            .zip(gammas.iter())
            .map(|(o, g)| {
                let scaled: Vec<F128> = o.eq_r_dprime.iter().map(|x| *g * *x).collect();
                ring_switch::linearized_coefficients(&ring_switch::build_fold_byte_table(&scaled))
            })
            .collect();
        for c in packed_direct.iter() {
            if c.point.len() != n_log + k_cols_pub {
                return Err(VerifyErrorOpen::Assist);
            }
        }
        let pd_groups = scalar_claim_groups(
            packed_direct.iter().map(|c| c.point),
            gammas_pd,
            n_log,
            k_cols_pub,
        );
        let spec = FusedWeightSpec {
            n_log,
            k_cols: k_cols_pub,
            rs: x_outers
                .iter()
                .zip(coeffs)
                .map(|(x, c)| {
                    assert_eq!(
                        x.len(),
                        1 + n_log + k_cols_pub,
                        "point/row/col split mismatch"
                    );
                    (&x[1..1 + n_log], &x[1 + n_log..], c)
                })
                .collect(),
            pd: pd_groups,
        };
        return verify_fused_ligerito(
            commitment,
            target,
            &spec,
            &proof.inner,
            lig_config,
            challenger,
        );
    }
    if proof.merged_rounds.len() != dense_log {
        return Err(VerifyErrorOpen::VirtualOpen);
    }
    let expected_merged_nonces = if grinding.merged_round_bits == 0 {
        0
    } else {
        dense_log
    };
    if proof.merged_round_nonces.len() != expected_merged_nonces {
        return Err(VerifyErrorOpen::VirtualOpen);
    }
    let mut running = target;
    let mut rho = Vec::with_capacity(dense_log);
    for (round, &(g_one, g_inf)) in proof.merged_rounds.iter().enumerate() {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = if grinding.merged_round_bits != 0 {
            challenger
                .verify_pow_and_sample_f128(
                    proof.merged_round_nonces[round],
                    grinding.merged_round_bits,
                )
                .ok_or(VerifyErrorOpen::VirtualOpen)?
        } else {
            challenger.sample_f128()
        };
        running = jagged::fold_round_claim(running, g_one, g_inf, r);
        rho.push(r);
    }

    let t = std::time::Instant::now();
    let params = jagged::JaggedParams::from_heights(heights, n_log, dense_log);
    let k_cols = params.k;
    if jagged::rectangular_prefix_columns(heights, n_log).is_some() {
        // Block-first transport: the prover bound the k−1 block coordinates
        // first; put ρ back into coordinate order.
        rho.rotate_left(k_cols - 1);
    }
    if trace {
        eprintln!(
            "        [vbm] JaggedParams::from_heights (n_log={n_log}, dense_log={dense_log}, k_cols={k_cols}): {}",
            tfmt(t.elapsed().as_secs_f64())
        );
    }
    let t = std::time::Instant::now();
    // The claims' c_{i,j}: derived from the transcript (γ-scaled r''-eq
    // tensors → fold byte tables → linearized coefficients).
    let coeffs: Vec<Vec<F128>> = rs_outputs
        .iter()
        .zip(gammas.iter())
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
        gammas_pd,
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
    let t = std::time::Instant::now();
    #[cfg(feature = "mul-count")]
    let assist_start = crate::field::gf2_128::op_count::snapshot();
    // Mirror the prover's FORK/JOIN (see the prover-side note): the assist
    // replays on a domain-separated child seeded here, the inner opening on
    // the parent, and the child merges after. The verifier stays sequential
    // — only the transcript forks — so the assist runs first and its `v` is
    // held for the `running` check below, which moved past the merge.
    let mut ch_a = challenger.fork(b"flock-par-assist-v1");
    let mut mp_defer = None;
    let frobenius = proof.frobenius.as_ref().ok_or(VerifyErrorOpen::Assist)?;
    let v = if defer.is_some() {
        let (v, mp) = jagged::verify_multipoint_twisted_deferred_with_grinding(
            &params,
            &fclaims,
            &gclaims,
            &rho,
            frobenius,
            grinding.multipoint,
            &mut ch_a,
        )
        .ok_or(VerifyErrorOpen::Assist)?;
        mp_defer = Some(mp);
        v
    } else {
        jagged::verify_multipoint_twisted_with_grinding(
            &params,
            &fclaims,
            &gclaims,
            &rho,
            frobenius,
            grinding.multipoint,
            &mut ch_a,
        )
        .ok_or(VerifyErrorOpen::Assist)?
    };
    if let (Some(out), Some(mp)) = (defer, mp_defer) {
        *out = Some(assemble_jagged_assertion(
            &params,
            x_outers,
            packed_direct,
            gammas_pd,
            &pd_groups,
            n_log,
            &mp,
        ));
    }
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
    let t = std::time::Instant::now();
    let pd = PackedDirectClaimRef {
        point: &rho,
        value: proof.q_eval,
    };
    verify_opening_batch_ligerito_mixed_with_grinding(
        commitment,
        &[],
        &[],
        &[],
        &[pd],
        &proof.inner,
        lig_config,
        grinding,
        challenger,
    )
    .map_err(|_| VerifyErrorOpen::Ligerito)?;
    challenger.merge_child(ch_a);
    // The assist's claim against the merged sumcheck's folded target. A pure
    // arithmetic check — it consumes no challenges, so the fork moved it here
    // without touching the transcript.
    if running != proof.q_eval * v {
        return Err(VerifyErrorOpen::VirtualOpen);
    }
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

    use crate::test_rng::Rng;

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
    #[ignore] // Heavier — ~50-100 ms; run with `cargo test pcs_ligerito_backend_roundtrip -- --ignored --nocapture`
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

        let log_n = m - LOG_PACKING;
        let lig_p_cfg = crate::pcs::ligerito::prover_config_for(
            log_n,
            initial_k,
            crate::pcs::ligerito::LigeritoProfile::Fast,
        )
        .expect("m22 Fast prover config");
        let lig_v_cfg = crate::pcs::ligerito::verifier_config_for(
            log_n,
            initial_k,
            crate::pcs::ligerito::LigeritoProfile::Fast,
        )
        .expect("m22 Fast verifier config");

        let mut ch_p = FsChallenger::new(b"flock-test-lig-v0");
        let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
            z_packed.clone(),
            &prover_data,
            &commitment,
            &[x_outer.as_slice()],
            &[],
            &[],
            &PaddingSpec::dense(m),
            &lig_p_cfg,
            OpeningGrinding::disabled(),
            &mut ch_p,
        );

        let mut ch_v = FsChallenger::new(b"flock-test-lig-v0");
        verify_opening_batch_ligerito_mixed_with_grinding(
            &commitment,
            &[rs_claim],
            &[&lagrange_weights_naive(6, z_skip)],
            &[x_outer.as_slice()],
            &[],
            &proof,
            &lig_v_cfg,
            OpeningGrinding::disabled(),
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("ligerito verify rejected honest proof: {e:?}"));

        let mut malformed = proof.clone();
        malformed.ring_switches.clear();
        let mut ch_v = FsChallenger::new(b"flock-test-lig-v0");
        assert!(matches!(
            verify_opening_batch_ligerito_mixed_with_grinding(
                &commitment,
                &[rs_claim],
                &[&lagrange_weights_naive(6, z_skip)],
                &[x_outer.as_slice()],
                &[],
                &malformed,
                &lig_v_cfg,
                OpeningGrinding::disabled(),
                &mut ch_v,
            ),
            Err(VerifyError::RingSwitch(
                ring_switch::VerifyError::MalformedProof
            ))
        ));
    }

    #[test]
    fn strict_transport_grinds_over_the_johnson_l0_list() {
        let policy = OpeningGrinding::per_challenge_128();
        assert_eq!(policy.claim_batch_bits_for(1), 0);
        assert_eq!(policy.claim_batch_bits_for(2), 6);
    }
}
