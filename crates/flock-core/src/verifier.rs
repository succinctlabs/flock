//! Top-level R1CS verifier: walks the challenger in lockstep with the
//! prover, runs `zerocheck::verify` and `lincheck::verify`, derives the two
//! ZClaims, and verifies the PCS openings at those points against the
//! witness commitment.

use crate::challenger::Challenger;
use crate::circuit::Circuit;
use crate::circuit::WiringError;
use crate::circuit::WiringProof;
use crate::circuit::verify_wiring;
use crate::element_r1cs::union::Claims;
use crate::element_r1cs::union::ElementAssertion;
use crate::element_r1cs::union::Proof;
use crate::element_r1cs::union::VerifyError as ElementVerifyError;
use crate::element_r1cs::union::verify_deferred;
use crate::field::F128;
use crate::lincheck;
use crate::pcs::MergedOpenProof;
use crate::pcs::PcsParams;
use crate::pcs::VerifyErrorOpen;
use crate::pcs::{self, Commitment};
use crate::proof::BooleanPiopProof;
use crate::proof::R1csProofCircuitMerged;
use crate::proof::R1csProofMergedLigerito;
use crate::proof::R1csProofMixedClassMerged;
use crate::proof::UnionClassClaims;
use crate::proof::bind_statement;
use crate::proof::{R1csClaim, R1csProofLigerito, ZClaim};
use crate::r1cs::BlockR1cs;
use crate::union::UnionInstance;
use crate::zerocheck;
use std::env::var;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    Zerocheck(zerocheck::VerifyError),
    Lincheck(lincheck::VerifyError),
    PcsAb(pcs::VerifyError),
    PcsC(pcs::VerifyError),
    /// The jagged-path batched opening rejected (see [`verify_ligerito_jagged`]).
    PcsOpen(pcs::VerifyErrorOpen),
    /// The element-region PIOP rejected.
    Element(ElementVerifyError),
    /// A mixed-class proof carries a class sub-proof the registry has no type
    /// for, or omits one it does — the statement and the proof disagree on
    /// which PIOPs ran.
    ClassMismatch,
    /// The wiring (copy-constraint) argument rejected.
    Wiring(WiringError),
    /// The circuit and the union instance are not the same statement: a
    /// different registry, or gate counts that are not the union's declared
    /// counts. A rejection, not a panic — both come from the caller.
    CircuitMismatch,
}

/// Per-phase wall-clock timings (seconds) of a verify, for benchmark cost
/// breakdowns. Produced by [`verify_ligerito_timed`] (direct) and
/// [`verify_ligerito_jagged_union_timed`] (union). Benchmark-only.
#[derive(Clone, Copy, Debug, Default)]
pub struct VerifyPhaseTimings {
    /// `zerocheck::verify` — the zerocheck PIOP replay.
    pub zerocheck_s: f64,
    /// The full lincheck verify (`lincheck::verify` / `lincheck::verify_union`)
    /// including the per-type comb construction below.
    pub lincheck_s: f64,
    /// UNION only: the per-type α-batched comb construction inside the union
    /// lincheck verify — the `O(Σ_t nnz_t)` `fold_alpha_batched` over every
    /// slot, i.e. the multi-slot circuit replay. `0.0` on the direct path,
    /// whose single-table lincheck folds the comb in lockstep with the
    /// sumcheck (no separable phase).
    pub lincheck_comb_s: f64,
    /// The batched PCS opening verify (`verify_claims_ligerito` /
    /// `verify_claims_jagged_ligerito`).
    pub open_s: f64,
}

/// Dedicated single-thread rayon pool that the verifier runs inside.
///
/// The verifier is intentionally single-threaded — matching the convention of
/// comparable provers (binius64, plonky3, hashcaster all ship serial
/// verifiers) and keeping reported verify times honest single-core numbers.
/// The verify path shares several `par_*` helpers with the (multi-threaded)
/// prover — e.g. `lincheck::fold_alpha_batched`, `sumcheck_bind_top_in_place_par`,
/// and the Ligerito residual eval — so rather than fork every shared helper, the
/// reusable verify cores (`verify_core`, `verify_claims_ligerito`)
/// run their body via `verifier_pool().install(..)`. Any `par_iter` reached from
/// there uses this 1-thread pool and collapses onto a single worker, without
/// touching the prover's use of the global pool.
/// Thread count for [`verifier_pool`]. **1 in production** — the override
/// (`FLOCK_VERIFY_THREADS`) exists only so a benchmark can ask what the verify
/// would cost with the prover's parallelism, since the pool below is otherwise
/// the one place that decision is made. Read once per process, so a single run
/// measures a single configuration.
fn verifier_threads() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        var("FLOCK_VERIFY_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1)
    })
}

fn verifier_pool() -> &'static rayon::ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(verifier_threads())
            // The whole verify body runs on this worker — including the deep
            // recursive Ligerito verifier — so give it an ample stack. A rayon
            // worker otherwise defaults to ~2 MiB (vs the 8 MiB main thread),
            // which the recursion overflows.
            .stack_size(64 * 1024 * 1024)
            .thread_name(|_| "flock-verify".to_string())
            .build()
            .expect("build single-thread verifier pool")
    })
}

/// Verify an R1CS proof: replay zerocheck + lincheck → the two base z-claims,
/// then verify the batched Ligerito PCS opening covering both.
pub fn verify_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofLigerito,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    let (ab, c) = verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )?;
    verify_claims_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsAb)?;
    Ok(R1csClaim { ab, c })
}

/// [`verify_ligerito`] with per-phase timers — the direct-path counterpart
/// of [`verify_ligerito_jagged_union_timed`]. Splits the verify into
/// zerocheck-verify / lincheck-verify / opening-verify (the single-table
/// lincheck has no separable comb phase, so `lincheck_comb_s == 0`). Kept in
/// lockstep with `verify_ligerito`; benchmark-only, production path
/// undisturbed.
pub fn verify_ligerito_timed<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofLigerito,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(R1csClaim, VerifyPhaseTimings), VerifyError> {
    use std::time::Instant;
    // PIOP replay (bind + zerocheck + lincheck) on the 1-thread pool.
    let (ab, c, zerocheck_s, lincheck_s) =
        verifier_pool().install(|| -> Result<(ZClaim, ZClaim, f64, f64), VerifyError> {
            bind_statement(challenger, r1cs, commitment);
            let t0 = Instant::now();
            let zc_claim = zerocheck::verify(r1cs.m, &proof.zerocheck, challenger)
                .map_err(VerifyError::Zerocheck)?;
            let zerocheck_s = t0.elapsed().as_secs_f64();
            let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
            let t0 = Instant::now();
            let lc_claim = lincheck::verify(
                r1cs.m,
                r1cs.k_log,
                r1cs.k_skip,
                lincheck_circuit,
                &x_ab,
                zc_claim.a_eval,
                zc_claim.b_eval,
                &proof.lincheck,
                challenger,
            )
            .map_err(VerifyError::Lincheck)?;
            let lincheck_s = t0.elapsed().as_secs_f64();
            let ab = ZClaim {
                point: r1cs.ab_claim_point(
                    lc_claim.r_inner_skip,
                    &lc_claim.r_inner_rest,
                    &x_ab.x_outer,
                ),
                value: lc_claim.w,
            };
            let c = ZClaim {
                point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
                value: zc_claim.c_eval,
            };
            Ok((ab, c, zerocheck_s, lincheck_s))
        })?;

    let t0 = Instant::now();
    verify_claims_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(VerifyError::PcsAb)?;
    let open_s = t0.elapsed().as_secs_f64();

    let t = VerifyPhaseTimings {
        zerocheck_s,
        lincheck_s,
        lincheck_comb_s: 0.0,
        open_s,
    };
    Ok((R1csClaim { ab, c }, t))
}

/// Statement-binding selector for the union verify path. Private: the two
/// public entries below fix the variant (mirror of the prove-side enum in
/// `flock_prover::prover`).
enum UnionVerifyBinding<'a> {
    /// The protocol binding: `flock-mixed-v1` over the registry digest, the
    /// counts vector, and the commitment root
    /// ([`crate::union::UnionInstance::bind_statement`]).
    Mixed,
    /// The circuit binding: [`UnionVerifyBinding::Mixed`] plus the circuit
    /// digest and the public words.
    Circuit {
        circuit: &'a Circuit,
        public: &'a [F128],
    },
}

/// The MERGED-transport union verifier (wire v6) — the Mixed protocol's
/// verify entry for BOOLEAN-only registries: a thin wrapper over
/// [`verify_ligerito_union_mixed_class`] (the one shared
/// body). Handles both lane-major and power-of-two commitments (dispatched
/// on `commitment.params.num_lanes`, which the shared body's
/// params-equality check pins to the count-derived value).
pub fn verify_ligerito_union<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofMergedLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    // Mirror of the prove-side guard: this entry consumes `R1csClaim` —
    // structurally boolean-only.
    assert!(
        !union.has_element(),
        "this entry is boolean-only; element registries go through \
         verify_ligerito_union_mixed_class"
    );
    // Repackage as a boolean-only mixed-class proof and run the one shared
    // verify body (the two-body split died with the jagged transport). The
    // clone is a few hundred KB against a multi-ms verify.
    let mixed = R1csProofMixedClassMerged {
        boolean: Some(BooleanPiopProof {
            zerocheck: proof.zerocheck.clone(),
            lincheck: proof.lincheck.clone(),
        }),
        element: None,
        pcs_open: proof.pcs_open.clone(),
    };
    let claims = verify_ligerito_union_mixed_class(
        union, circuits, commitment, &mixed, pcs_params, challenger,
    )?;
    Ok(claims.boolean.expect("asserted boolean-only above"))
}

/// The **circuit** verify entry over the MERGED transport — the production
/// shape, and the mirror of
/// `flock_prover::prover::prove_fast_ligerito_union_circuit`.
///
/// Same replay as the jagged variant: both class PIOPs, then the wiring
/// argument over the circuit's cell space (σ-aware GKR plus the
/// recombination and `f_eval == g_eval` bindings). Only the opening differs
/// — the wiring's gather claims are packed-direct, which the merged
/// transport carries the same way it carries the element class's.
#[allow(clippy::too_many_arguments)]
pub fn verify_ligerito_union_circuit<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuit: &Circuit,
    public: &[F128],
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofCircuitMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionClassClaims, VerifyError> {
    if !circuit.check_instance(union) || public.len() != circuit.num_public() {
        return Err(VerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        Some(&proof.wiring),
        pcs_params,
        challenger,
    )?;
    if let Some(a) = matrix {
        a.check(union, circuits).map_err(VerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(VerifyError::Element)?;
    }
    verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
}

/// [`verify_ligerito_union_circuit`] with the matrix work left
/// undischarged — what a merge node runs on each child proof.
///
/// Everything else is verified: both class PIOPs, the wiring argument, and
/// the single merged opening. What comes back alongside the claims is the two
/// classes' [`DeferredMatrixWork`], for the caller to fold into an
/// accumulator ([`crate::aggregate`]) rather than evaluate.
///
/// No base matrix is read anywhere in it — that is what lets a recursion
/// circuit replay it. There is deliberately NO jagged counterpart: the merged
/// transport is the production path, and building deferred machinery on the
/// legacy one would be work aimed at something being retired.
///
/// **The claims are conditional on the returned work**: a proof whose
/// lincheck is simply wrong still returns `Ok` here. Callers that are not
/// accumulating must use [`verify_ligerito_union_circuit`].
#[allow(clippy::too_many_arguments)]
pub fn verify_ligerito_union_circuit_deferred<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuit: &Circuit,
    public: &[F128],
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofCircuitMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(UnionClassClaims, DeferredMatrixWork), VerifyError> {
    if !circuit.check_instance(union) || public.len() != circuit.num_public() {
        return Err(VerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        Some(&proof.wiring),
        pcs_params,
        challenger,
    )?;
    let claims = verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
    )?;
    Ok((
        claims,
        DeferredMatrixWork {
            boolean: matrix,
            element: el_matrix,
        },
    ))
}

/// The merged transport's verification, shared by the mixed-class and circuit
/// entries: the boolean pair ring-switched, everything else packed-direct.
fn verify_merged_opening<Ch: Challenger>(
    union: &UnionInstance<'_>,
    commitment: &Commitment,
    claims: &UnionClassClaims,
    packed_direct_points: &[(Vec<F128>, F128)],
    pcs_open: &MergedOpenProof,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionClassClaims, VerifyError> {
    let cl: Vec<ZClaim> = match &claims.boolean {
        Some(c) => vec![c.ab.clone(), c.c.clone()],
        None => Vec::new(),
    };
    let values: Vec<F128> = cl.iter().map(|z| z.value).collect();
    let z_skips: Vec<F128> = cl.iter().map(|z| z.point.z_skip).collect();
    let x_fulls: Vec<Vec<F128>> = cl
        .iter()
        .map(|z| {
            let mut v = z.point.x_inner_rest.clone();
            v.extend_from_slice(&z.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pd: Vec<pcs::PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| pcs::PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    verifier_pool()
        .install(|| {
            pcs::verify_batch_merged(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                pcs_open,
                &lig_v_config,
                challenger,
            )
        })
        .map_err(VerifyError::PcsOpen)?;
    Ok(claims.clone())
}

/// [`verify_ligerito_jagged_union_mixed_class`] over the MERGED transport.
///
/// Same statement, same PIOP replay; only the opening differs. The element
/// class's two claims ride as packed-direct claims, which the merged
/// transport carries by expressing each weight as the `F₂`-linear map
/// `x ↦ γ·x` — indistinguishable, to its per-claim weight builder, from a
/// ring-switched claim's Φ-fold.
pub fn verify_ligerito_union_mixed_class<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofMixedClassMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionClassClaims, VerifyError> {
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        None,
        pcs_params,
        challenger,
    )?;
    // Both classes' matrix work comes back undischarged from the PIOP
    // replay; this is a non-deferred entry, so discharge here — after the
    // replay and BEFORE the opening, as the sibling entries do, so an
    // inconsistent lincheck is rejected as Lincheck and before the expensive
    // PCS work.
    if let Some(a) = matrix {
        a.check(union, circuits).map_err(VerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(VerifyError::Element)?;
    }

    // Same construction as the boolean-only merged verifier: the PCS point
    // is `x_inner_rest ‖ x_outer`, with the skip coordinate carried
    // separately in `z_skip`.
    let cl: Vec<ZClaim> = match &claims.boolean {
        Some(c) => vec![c.ab.clone(), c.c.clone()],
        None => Vec::new(),
    };
    let values: Vec<F128> = cl.iter().map(|z| z.value).collect();
    let z_skips: Vec<F128> = cl.iter().map(|z| z.point.z_skip).collect();
    let x_fulls: Vec<Vec<F128>> = cl
        .iter()
        .map(|z| {
            let mut v = z.point.x_inner_rest.clone();
            v.extend_from_slice(&z.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pd: Vec<pcs::PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| pcs::PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    verifier_pool()
        .install(|| {
            pcs::verify_batch_merged(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                &proof.pcs_open,
                &lig_v_config,
                challenger,
            )
        })
        .map_err(VerifyError::PcsOpen)?;
    Ok(claims)
}

/// [`verify_ligerito_union_mixed_class`] with the matrix work left
/// undischarged: everything else is verified, and both classes' assertions
/// come back as [`DeferredMatrixWork`] for the caller to discharge or
/// accumulate ([`crate::aggregate`]).
///
/// This is the "succinct verify" of the accumulation route — no base matrix
/// is read anywhere in it, which is what lets a recursion circuit replay it.
/// **The returned claims are conditional on the assertions**: a proof whose
/// lincheck is simply wrong still returns `Ok` here, so a caller that is not
/// accumulating must use [`verify_ligerito_union_mixed_class`].
pub fn verify_ligerito_union_mixed_class_deferred<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofMixedClassMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(UnionClassClaims, DeferredMatrixWork), VerifyError> {
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(VerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof.boolean.as_ref(),
        proof.element.as_ref(),
        None,
        pcs_params,
        challenger,
    )?;
    // DEFERRED: both classes' matrix work rides out undischarged for the
    // caller to check or accumulate (`crate::aggregate`). The returned
    // claims are CONDITIONAL on the assertions.
    let deferred = DeferredMatrixWork {
        boolean: matrix,
        element: el_matrix,
    };

    // Same construction as the boolean-only merged verifier: the PCS point
    // is `x_inner_rest ‖ x_outer`, with the skip coordinate carried
    // separately in `z_skip`.
    let cl: Vec<ZClaim> = match &claims.boolean {
        Some(c) => vec![c.ab.clone(), c.c.clone()],
        None => Vec::new(),
    };
    let values: Vec<F128> = cl.iter().map(|z| z.value).collect();
    let z_skips: Vec<F128> = cl.iter().map(|z| z.point.z_skip).collect();
    let x_fulls: Vec<Vec<F128>> = cl
        .iter()
        .map(|z| {
            let mut v = z.point.x_inner_rest.clone();
            v.extend_from_slice(&z.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pd: Vec<pcs::PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| pcs::PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    verifier_pool()
        .install(|| {
            pcs::verify_batch_merged(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                &proof.pcs_open,
                &lig_v_config,
                challenger,
            )
        })
        .map_err(VerifyError::PcsOpen)?;
    Ok((claims, deferred))
}

/// Shared PIOP replay for both union verify shapes: statement binding, the
/// boolean class's zerocheck + lincheck over the `M_bool` prefix subcube, then
/// the element region's PIOP. Returns the per-class claims and the element
/// class's `(point, value)` pairs for the packed-direct intake.
///
/// Runs on the 1-thread verifier pool, like every other verify core.
#[allow(clippy::too_many_arguments)]
fn verify_union_piops<Ch: Challenger>(
    union: &UnionInstance<'_>,
    binding: UnionVerifyBinding<'_>,
    circuits: &[&dyn lincheck::LincheckCircuit],
    commitment: &Commitment,
    boolean: Option<&BooleanPiopProof>,
    element: Option<&Proof>,
    wiring: Option<&WiringProof>,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionPiopOut, VerifyError> {
    // The commitment is to the DENSE stack q (M4/M5): PcsParams.m is the
    // dense variable count — count-dependent under height-n_t stacking,
    // derived from the declared counts — while the PIOP and the
    // virtual-opening sumcheck run over the M-variable padded address space.
    assert_eq!(
        pcs_params.m,
        union.dense_m(),
        "PcsParams.m must equal the union's dense_m (committed stack size)"
    );
    // The proof carries `commitment.params`, and the opening reads its
    // `num_ntts()` for the L0 leaf width and the lane-grid rotation — but the
    // transcript binds only the commitment ROOT, so those params are
    // ATTACKER-CONTROLLED. The honest lane count is count-derived
    // (`UnionInstance::commit_lanes`, like `dense_m`), so require the
    // commitment to carry exactly it; a mismatch is a rejection, not a panic.
    if commitment.params.m != pcs_params.m
        || commitment.params.log_batch_size != pcs_params.log_batch_size
        || commitment.params.log_inv_rate != pcs_params.log_inv_rate
        || commitment.params.num_ntts() != pcs_params.num_ntts()
    {
        return Err(VerifyError::PcsOpen(VerifyErrorOpen::Ligerito));
    }
    // Verification is single-threaded; run the PIOP replay on the dedicated
    // 1-thread pool (verify_claims_jagged_ligerito installs it itself).
    verifier_pool().install(|| -> Result<UnionPiopOut, VerifyError> {
        match binding {
            UnionVerifyBinding::Mixed => union.bind_statement(challenger, commitment),
            UnionVerifyBinding::Circuit { circuit, public } => {
                union.bind_statement_circuit(challenger, commitment, &circuit.digest(), public)
            }
        }

        let mut matrix: Option<lincheck::MatrixAssertion> = None;
        let mut el_matrix: Option<ElementAssertion> = None;
        let bool_claim = match boolean {
            Some(piop) => {
                // The boolean PIOP runs over the BOOLEAN REGION only — the
                // prefix subcube `[0, 2^M_bool)`, `M_bool = M` for a
                // boolean-only registry. (The element region cannot join this
                // sum: `c = z` there.)
                let zc_claim = zerocheck::verify(union.m_bool(), &piop.zerocheck, challenger)
                    .map_err(VerifyError::Zerocheck)?;
                let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
                // The union-column lincheck (one circuit per BOOLEAN slot, in
                // slot order); the declared counts additionally bind through
                // the per-type const-pin target terms.
                // DEFERRED: the matrix work leaves as an assertion instead of
                // being discharged here. Callers that are not accumulating get
                // it discharged for them by the wrappers below.
                let (lc_claim, assertion) = lincheck::verify_union_deferred(
                    union,
                    circuits,
                    &x_ab,
                    zc_claim.a_eval,
                    zc_claim.b_eval,
                    &piop.lincheck,
                    challenger,
                )
                .map_err(VerifyError::Lincheck)?;
                matrix = Some(assertion);
                Some(R1csClaim {
                    ab: ZClaim {
                        point: union.ab_claim_point(
                            lc_claim.r_inner_skip,
                            &lc_claim.r_inner_rest,
                            &x_ab.x_outer,
                        ),
                        value: lc_claim.w,
                    },
                    c: ZClaim {
                        point: union.c_claim_point(zc_claim.z, &zc_claim.r_rest),
                        value: zc_claim.c_eval,
                    },
                })
            }
            None => None,
        };

        // DEFERRED on this side too: the element class's matrix work leaves
        // as its own assertion rather than being evaluated here, so a
        // `*_deferred` entry really does defer BOTH classes.
        let el_claim = match element {
            Some(p) => {
                let (c, a) = verify_deferred(union, p, challenger).map_err(VerifyError::Element)?;
                el_matrix = Some(a);
                Some(c)
            }
            None => None,
        };
        let mut packed_direct = el_claim
            .as_ref()
            .map(|c: &Claims| {
                vec![
                    (c.c_point.clone(), c.c_value),
                    (c.lc_point.clone(), c.lc_value),
                ]
            })
            .unwrap_or_default();

        // The wiring argument replays AFTER both classes' PIOPs, at the
        // prover's transcript position; its gather claims join the same
        // packed-direct intake the element claims ride.
        if let UnionVerifyBinding::Circuit { circuit, public } = binding {
            let proof = wiring.ok_or(VerifyError::CircuitMismatch)?;
            #[cfg(feature = "mul-count")]
            let wiring_start = crate::field::gf2_128::op_count::snapshot();
            let gather =
                verify_wiring(circuit, public, proof, challenger).map_err(VerifyError::Wiring)?;
            #[cfg(feature = "mul-count")]
            if std::env::var("MUL_TRACE").is_ok() {
                let e = crate::field::gf2_128::op_count::snapshot();
                let invs = e.invs - wiring_start.invs;
                let muls = (e.native_muls - wiring_start.native_muls)
                    .saturating_sub(invs * crate::field::gf2_128::op_count::MULS_PER_INV);
                println!(
                    "  [mul] wiring GKR (grand product + sigma):             \
                     {muls:>8} muls {invs:>5} invs = {:>8} constraints",
                    muls + invs
                );
            }
            packed_direct.extend(gather);
        }

        Ok((
            UnionClassClaims {
                boolean: bool_claim,
                element: el_claim,
            },
            packed_direct,
            matrix,
            el_matrix,
        ))
    })
}

/// What the union PIOP replay yields: the per-class claims, the element
/// class's packed-direct claims for the opening, and — when a boolean PIOP
/// ran — the [`lincheck::MatrixAssertion`] carrying its undischarged matrix
/// work.
type UnionPiopOut = (
    UnionClassClaims,
    Vec<(Vec<F128>, F128)>,
    Option<lincheck::MatrixAssertion>,
    Option<ElementAssertion>,
);

/// Both classes' undischarged matrix work, as a `*_deferred` entry returns
/// it. Either half is `None` when that class has no types in the registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredMatrixWork {
    pub boolean: Option<lincheck::MatrixAssertion>,
    pub element: Option<ElementAssertion>,
}

/// Verify a batched PCS opening over an arbitrary list of `ẑ`-claims — the
/// mirror of `flock_prover::prover::open_claims_with_precomputed_ligerito`.
/// Relation wrappers (e.g. the hash chain) reuse this with their own appended
/// claims. Must run at the same transcript position as the prover's open.
pub fn verify_claims_ligerito<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &pcs::BatchOpeningProofLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), pcs::VerifyError> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || {
        verify_claims_ligerito_inner(commitment, claims, pcs_open, pcs_params, challenger)
    })
}

fn verify_claims_ligerito_inner<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &pcs::BatchOpeningProofLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), pcs::VerifyError> {
    let z_skips: Vec<F128> = claims.iter().map(|c| c.point.z_skip).collect();
    let values: Vec<F128> = claims.iter().map(|c| c.value).collect();
    let x_fulls: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| {
            let mut v = c.point.x_inner_rest.clone();
            v.extend_from_slice(&c.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    pcs::verify_opening_batch_ligerito_mixed(
        commitment,
        &values,
        &z_skips,
        &x_refs,
        &[],
        pcs_open,
        &lig_v_config,
        challenger,
    )
}

/// Replay bind → zerocheck → lincheck and reconstruct the two base z-claims
/// (`ab`, `c`), stopping before the PCS open. Mirror of
/// `flock_prover::prover::prove_fast_core`; relation wrappers reuse this then call
/// [`verify_claims_ligerito`] over `[ab, c, …]`.
pub fn verify_core<Ch: Challenger>(
    r1cs: &BlockR1cs,
    zerocheck_proof: &zerocheck::ZerocheckProof,
    lincheck_proof: &lincheck::LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), VerifyError> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || {
        verify_core_inner(
            r1cs,
            zerocheck_proof,
            lincheck_proof,
            commitment,
            lincheck_circuit,
            challenger,
        )
    })
}

fn verify_core_inner<Ch: Challenger>(
    r1cs: &BlockR1cs,
    zerocheck_proof: &zerocheck::ZerocheckProof,
    lincheck_proof: &lincheck::LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), VerifyError> {
    let trace = var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    // ---- Bind FS transcript to the statement (mirrors prover::prove).
    let t = Instant::now();
    bind_statement(challenger, r1cs, commitment);
    if trace {
        eprintln!(
            "      [vco] bind_statement: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Zerocheck.
    let t = Instant::now();
    let zc_claim =
        zerocheck::verify(r1cs.m, zerocheck_proof, challenger).map_err(VerifyError::Zerocheck)?;
    if trace {
        eprintln!(
            "      [vco] zerocheck::verify: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Build lincheck's shared quirky point from the zerocheck output
    // (layout-aware: the mlv challenges are address-ordered).
    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    // ---- Lincheck. v_a, v_b come from the zerocheck's final â, b̂ evals.
    let t = Instant::now();
    let lc_claim = lincheck::verify(
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        lincheck_circuit,
        &x_ab,
        zc_claim.a_eval,
        zc_claim.b_eval,
        lincheck_proof,
        challenger,
    )
    .map_err(VerifyError::Lincheck)?;
    if trace {
        eprintln!(
            "      [vco] lincheck::verify: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Build the two z-claims (must match what `prove` returned).
    // Layout-aware: the ZClaim points are address-ordered for the PCS.
    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    // c-claim is already a z-claim since `C = I` ⇒ ĉ = ẑ.
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    Ok((ab, c))
}

#[cfg(test)]
mod tests {
    /// The verifier is intentionally single-threaded: every `par_*` reached
    /// from a verify core must collapse onto the one-thread `verifier_pool`.
    /// Guard the invariant so a future `ThreadPoolBuilder` tweak can't silently
    /// re-parallelize verification.
    ///
    /// (The end-to-end prove → verify roundtrip and tamper-rejection tests live
    /// in `flock-prover`'s `tests/verifier_roundtrip.rs`, since they need the
    /// prove path.)
    #[test]
    fn verifier_pool_is_single_threaded() {
        let n = super::verifier_pool().install(rayon::current_num_threads);
        assert_eq!(n, 1, "verifier_pool must have exactly one worker thread");
    }
}
