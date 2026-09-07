//! Top-level R1CS verifier: walks the challenger in lockstep with the
//! prover, runs `zerocheck::verify` and `verify_lincheck`, derives the two
//! ZClaims, and verifies the PCS openings at those points against the
//! witness commitment.

use std::{env::var, sync::OnceLock, time::Instant};

use lincheck::{
    LincheckCircuit, LincheckError, LincheckGrinding, LincheckProof, MatrixAssertion,
    verify as verify_lincheck, verify_union_deferred_with_grinding,
    verify_with_grinding as verify_lincheck_with_grinding,
};
use pcs::{
    BatchOpeningProofLigerito, LOG_PACKING, PackedDirectClaimRef, PcsError, verify_batch_merged,
    verify_batch_merged_deferred, verify_opening_batch_ligerito_mixed_with_grinding,
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use zerocheck::{
    ZerocheckError, ZerocheckGrinding, ZerocheckProof,
    ag_skip::{
        AgProof, AgVerifyError, K_SKIP, verify as verify_ag,
        verify_with_grinding as verify_ag_with_grinding,
    },
    verify_with_grinding as verify_zerocheck_with_grinding,
};
#[cfg(feature = "mul-count")]
use {crate::field::gf2_128::op_count::MULS_PER_INV, crate::field::gf2_128::op_count::snapshot};

use crate::{
    challenger::Challenger,
    circuit::{
        Circuit, SigmaAssertion, WiringError, WiringProof, verify_wiring_deferred_with_grinding,
        verify_wiring_with_grinding,
    },
    element_r1cs::union::{
        Claims, ElementAssertion, ElementUnionError, Proof, verify_deferred_with_grinding,
    },
    field::F128,
    lincheck,
    lincheck::SkipPoint,
    matrix_fold::JaggedAssertion,
    pcs::{self, Commitment, MergedOpenProof, PcsOpenError, PcsParams},
    proof::{
        BooleanPiopProof, BooleanPiopProofAg, R1csClaim, R1csProofCircuitMerged,
        R1csProofCircuitMergedAg, R1csProofLigerito, R1csProofLigeritoAg, R1csProofMergedLigerito,
        R1csProofMergedLigeritoAg, R1csProofMixedClassMerged, UnionClassClaims, ZClaim,
        bind_statement,
    },
    r1cs::BlockR1cs,
    union::UnionInstance,
    zerocheck,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlockVerifyError {
    Zerocheck(ZerocheckError),
    /// AG-skip zerocheck round replay failed (`verify_core_ag`).
    Ag(AgVerifyError),
    Lincheck(LincheckError),
    PcsAb(PcsError),
    PcsC(PcsError),
    /// The jagged-path batched opening rejected (see [`verify_ligerito_jagged`]).
    PcsOpen(PcsOpenError),
    /// The element-region PIOP rejected.
    Element(ElementUnionError),
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
    /// The single-table entries build the c-claim as a direct z-claim, which
    /// is sound only for the circuit-R1CS shape `C = I` — an R1CS with any
    /// other `c_0` must be rejected here, not silently misverified. (The
    /// union path has no such check by design: registry table types carry
    /// stub `c_0` matrices under the walker-encoder convention, and the
    /// statement is bound through the registry digest instead.)
    NonIdentityC,
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
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        var("FLOCK_VERIFY_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1)
    })
}

fn verifier_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
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

/// The verifier's PCS parameters are public policy, not proof-selected
/// inputs. Compare every serialized parameter before accepting an opening so
/// a commitment cannot downgrade the requested profile or redirect Merkle
/// verification.
fn commitment_params_match_expected(commitment: &Commitment, expected: &PcsParams) -> bool {
    &commitment.params == expected
}

/// Verify an R1CS proof: replay zerocheck + lincheck → the two base z-claims,
/// then verify the batched Ligerito PCS opening covering both.
pub fn verify_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofLigerito,
    lincheck_circuit: &dyn LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, FlockVerifyError> {
    let (ab, c) = verify_core_with_grinding(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        pcs_params.zerocheck_grinding(),
        pcs_params.lincheck_grinding(),
        challenger,
    )?;
    verify_claims_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(FlockVerifyError::PcsAb)?;
    Ok(R1csClaim { ab, c })
}

/// Statement-binding selector for the union verify path. Private: the two
/// public entries below fix the variant (mirror of the prove-side enum in
/// `flock_prover::prover`).
enum UnionVerifyBinding<'a> {
    /// The protocol binding: `flock-mixed-v1` over the registry digest, the
    /// counts vector, and the commitment cap layer
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
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofMergedLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, FlockVerifyError> {
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

/// [`verify_ligerito_union`] with the **AG-skip** boolean zerocheck — the
/// mirror of `flock_prover::prover::prove_fast_ligerito_union_ag`. Same
/// statement binding, lincheck, and merged opening; only the zerocheck's
/// round-1 replay differs (and the claim points ride [`SkipPoint::Ag`]).
pub fn verify_ligerito_union_ag<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofMergedLigeritoAg,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, FlockVerifyError> {
    assert!(
        !union.has_element(),
        "the AG union route is boolean-only (the element region's PIOP is \
         independent of the zerocheck flavor, but no mixed AG proof shape \
         exists yet)"
    );
    if union.num_boolean() == 0 {
        return Err(FlockVerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, _el_matrix, _sigma) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        Some(BooleanPiopRef::Ag(&proof.boolean)),
        None,
        None,
        false,
        pcs_params,
        challenger,
    )?;
    if let Some(a) = matrix {
        a.check(union, circuits)
            .map_err(FlockVerifyError::Lincheck)?;
    }
    let claims = verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
        None,
    )?;
    Ok(claims.boolean.expect("boolean-only registry checked above"))
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
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofCircuitMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionClassClaims, FlockVerifyError> {
    if !circuit.check_instance(union) || !circuit.check_public(public) {
        return Err(FlockVerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(FlockVerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix, _sigma) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref().map(BooleanPiopRef::Rs),
        proof.element.as_ref(),
        Some(&proof.wiring),
        false,
        pcs_params,
        challenger,
    )?;
    if let Some(a) = matrix {
        a.check(union, circuits)
            .map_err(FlockVerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(FlockVerifyError::Element)?;
    }
    verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
        None,
    )
}

/// [`verify_ligerito_union_circuit`] with the matrix work left
/// undischarged — what a merge node runs on each child proof.
///
/// Everything else is verified: both class PIOPs, the wiring argument, and
/// the single merged opening. What comes back alongside the claims is the two
/// classes' [`DeferredMatrixWork`] AND the wiring's
/// [`SigmaAssertion`](crate::circuit::SigmaAssertion) (route B: the
/// `s_sigma(rho)` evaluation leaves as a foldable claim instead of costing
/// its O(2^mu) discharge here), for the caller to fold into an accumulator
/// ([`crate::aggregate`]) rather than evaluate. Sigma never travels alone —
/// it accumulates together with the matrix assertions of the same proof.
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
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofCircuitMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(UnionClassClaims, DeferredMatrixWork, SigmaAssertion), FlockVerifyError> {
    if !circuit.check_instance(union) || !circuit.check_public(public) {
        return Err(FlockVerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(FlockVerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix, sigma) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref().map(BooleanPiopRef::Rs),
        proof.element.as_ref(),
        Some(&proof.wiring),
        true,
        pcs_params,
        challenger,
    )?;
    let mut jagged = None;
    let claims = verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
        Some(&mut jagged),
    )?;
    Ok((
        claims,
        DeferredMatrixWork {
            boolean: matrix,
            element: el_matrix,
            jagged: jagged.expect("the deferred opening fills the export"),
        },
        sigma.expect("a circuit binding always verifies wiring"),
    ))
}

/// [`verify_ligerito_union_circuit`] with the **AG-skip** boolean zerocheck —
/// the mirror of `flock_prover::prover::prove_fast_ligerito_union_circuit_ag`.
/// Same replay; only the boolean zerocheck's round 1 differs.
#[allow(clippy::too_many_arguments)]
pub fn verify_ligerito_union_circuit_ag<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuit: &Circuit,
    public: &[F128],
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofCircuitMergedAg,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionClassClaims, FlockVerifyError> {
    if !circuit.check_instance(union) || !circuit.check_public(public) {
        return Err(FlockVerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(FlockVerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix, _sigma) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref().map(BooleanPiopRef::Ag),
        proof.element.as_ref(),
        Some(&proof.wiring),
        false,
        pcs_params,
        challenger,
    )?;
    if let Some(a) = matrix {
        a.check(union, circuits)
            .map_err(FlockVerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(FlockVerifyError::Element)?;
    }
    verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
        None,
    )
}

/// [`verify_ligerito_union_circuit_deferred`] with the **AG-skip** boolean
/// zerocheck — the succinct entry the recursion tower records and replays for
/// AG-flavored children. Same conditional-claims contract as the RS twin.
#[allow(clippy::too_many_arguments)]
pub fn verify_ligerito_union_circuit_ag_deferred<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuit: &Circuit,
    public: &[F128],
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofCircuitMergedAg,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(UnionClassClaims, DeferredMatrixWork, SigmaAssertion), FlockVerifyError> {
    if !circuit.check_instance(union) || !circuit.check_public(public) {
        return Err(FlockVerifyError::CircuitMismatch);
    }
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(FlockVerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix, sigma) = verify_union_piops(
        union,
        UnionVerifyBinding::Circuit { circuit, public },
        circuits,
        commitment,
        proof.boolean.as_ref().map(BooleanPiopRef::Ag),
        proof.element.as_ref(),
        Some(&proof.wiring),
        true,
        pcs_params,
        challenger,
    )?;
    let mut jagged = None;
    let claims = verify_merged_opening(
        union,
        commitment,
        &claims,
        &packed_direct_points,
        &proof.pcs_open,
        pcs_params,
        challenger,
        Some(&mut jagged),
    )?;
    Ok((
        claims,
        DeferredMatrixWork {
            boolean: matrix,
            element: el_matrix,
            jagged: jagged.expect("the deferred opening fills the export"),
        },
        sigma.expect("a circuit binding always verifies wiring"),
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
    defer: Option<&mut Option<JaggedAssertion>>,
) -> Result<UnionClassClaims, FlockVerifyError> {
    let cl: Vec<ZClaim> = match &claims.boolean {
        Some(c) => vec![c.ab.clone(), c.c.clone()],
        None => Vec::new(),
    };
    let values: Vec<F128> = cl.iter().map(|z| z.value).collect();
    let z_skips: Vec<SkipPoint> = cl.iter().map(|z| z.point.z_skip).collect();
    let x_fulls: Vec<Vec<F128>> = cl
        .iter()
        .map(|z| {
            let mut v = z.point.x_inner_rest.clone();
            v.extend_from_slice(&z.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pd: Vec<PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    verifier_pool()
        .install(|| match defer {
            Some(out) => verify_batch_merged_deferred(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                pcs_open,
                &lig_v_config,
                pcs_params.opening_grinding(),
                challenger,
            )
            .map(|a| *out = Some(a)),
            None => verify_batch_merged(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                pcs_open,
                &lig_v_config,
                pcs_params.opening_grinding(),
                challenger,
            ),
        })
        .map_err(FlockVerifyError::PcsOpen)?;
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
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofMixedClassMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionClassClaims, FlockVerifyError> {
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(FlockVerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix, _sigma) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof.boolean.as_ref().map(BooleanPiopRef::Rs),
        proof.element.as_ref(),
        None,
        false,
        pcs_params,
        challenger,
    )?;
    // Both classes' matrix work comes back undischarged from the PIOP
    // replay; this is a non-deferred entry, so discharge here — after the
    // replay and BEFORE the opening, as the sibling entries do, so an
    // inconsistent lincheck is rejected as Lincheck and before the expensive
    // PCS work.
    if let Some(a) = matrix {
        a.check(union, circuits)
            .map_err(FlockVerifyError::Lincheck)?;
    }
    if let Some(a) = el_matrix {
        a.check_reported(union).map_err(FlockVerifyError::Element)?;
    }

    // Same construction as the boolean-only merged verifier: the PCS point
    // is `x_inner_rest ‖ x_outer`, with the skip coordinate carried
    // separately in `z_skip`.
    let cl: Vec<ZClaim> = match &claims.boolean {
        Some(c) => vec![c.ab.clone(), c.c.clone()],
        None => Vec::new(),
    };
    let values: Vec<F128> = cl.iter().map(|z| z.value).collect();
    let z_skips: Vec<SkipPoint> = cl.iter().map(|z| z.point.z_skip).collect();
    let x_fulls: Vec<Vec<F128>> = cl
        .iter()
        .map(|z| {
            let mut v = z.point.x_inner_rest.clone();
            v.extend_from_slice(&z.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pd: Vec<PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    verifier_pool()
        .install(|| {
            verify_batch_merged(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                &proof.pcs_open,
                &lig_v_config,
                pcs_params.opening_grinding(),
                challenger,
            )
        })
        .map_err(FlockVerifyError::PcsOpen)?;
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
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    proof: &R1csProofMixedClassMerged,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(UnionClassClaims, DeferredMatrixWork), FlockVerifyError> {
    if proof.boolean.is_some() != (union.num_boolean() > 0)
        || proof.element.is_some() != union.has_element()
    {
        return Err(FlockVerifyError::ClassMismatch);
    }
    let (claims, packed_direct_points, matrix, el_matrix, _sigma) = verify_union_piops(
        union,
        UnionVerifyBinding::Mixed,
        circuits,
        commitment,
        proof.boolean.as_ref().map(BooleanPiopRef::Rs),
        proof.element.as_ref(),
        None,
        false,
        pcs_params,
        challenger,
    )?;
    // Same construction as the boolean-only merged verifier: the PCS point
    // is `x_inner_rest ‖ x_outer`, with the skip coordinate carried
    // separately in `z_skip`.
    let cl: Vec<ZClaim> = match &claims.boolean {
        Some(c) => vec![c.ab.clone(), c.c.clone()],
        None => Vec::new(),
    };
    let values: Vec<F128> = cl.iter().map(|z| z.value).collect();
    let z_skips: Vec<SkipPoint> = cl.iter().map(|z| z.point.z_skip).collect();
    let x_fulls: Vec<Vec<F128>> = cl
        .iter()
        .map(|z| {
            let mut v = z.point.x_inner_rest.clone();
            v.extend_from_slice(&z.point.x_outer);
            v
        })
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pd: Vec<PackedDirectClaimRef<'_>> = packed_direct_points
        .iter()
        .map(|(point, value)| PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();
    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config");
    // DEFERRED: both classes' matrix work rides out undischarged for the
    // caller to check or accumulate (`crate::aggregate`), the layout's
    // W-claims beside them. The returned claims are CONDITIONAL on the
    // assertions.
    let jagged = verifier_pool()
        .install(|| {
            verify_batch_merged_deferred(
                commitment,
                &values,
                &z_skips,
                &x_refs,
                &pd,
                &union.jagged_heights(),
                union.n_log(),
                &proof.pcs_open,
                &lig_v_config,
                pcs_params.opening_grinding(),
                challenger,
            )
        })
        .map_err(FlockVerifyError::PcsOpen)?;
    Ok((
        claims,
        DeferredMatrixWork {
            boolean: matrix,
            element: el_matrix,
            jagged,
        },
    ))
}

/// The boolean sub-proof as [`verify_union_piops`] consumes it — either
/// zerocheck flavor. Downstream of the zerocheck the two are identical (the
/// lincheck and both claim points are generic over [`SkipPoint`]).
#[derive(Clone, Copy)]
enum BooleanPiopRef<'a> {
    Rs(&'a BooleanPiopProof),
    Ag(&'a BooleanPiopProofAg),
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
    circuits: &[&dyn LincheckCircuit],
    commitment: &Commitment,
    boolean: Option<BooleanPiopRef<'_>>,
    element: Option<&Proof>,
    wiring: Option<&WiringProof>,
    defer_sigma: bool,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<UnionPiopOut, FlockVerifyError> {
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
    if !commitment_params_match_expected(commitment, pcs_params) {
        return Err(FlockVerifyError::PcsOpen(PcsOpenError::Ligerito));
    }
    // Verification is single-threaded; run the PIOP replay on the dedicated
    // 1-thread pool (verify_claims_jagged_ligerito installs it itself).
    verifier_pool().install(|| -> Result<UnionPiopOut, FlockVerifyError> {
        match binding {
            UnionVerifyBinding::Mixed => union.bind_statement(challenger, commitment),
            UnionVerifyBinding::Circuit { circuit, public } => {
                union.bind_statement_circuit(challenger, commitment, &circuit.digest(), public)
            }
        }

        let mut matrix: Option<MatrixAssertion> = None;
        let mut el_matrix: Option<ElementAssertion> = None;
        // Mirror the prover's FORK/JOIN transcript, which every circuit
        // binding uses: the wiring replays on a domain-separated child, run
        // before the element class so its closing digest merges at the
        // prover's position. Same labels, same order.
        let par_transcript =
            matches!(binding, UnionVerifyBinding::Circuit { .. }) && boolean.is_some();
        // ONE-SIDED fork (the prover's shape): the boolean PIOP replays on
        // the PARENT transcript; only the wiring gets a child, forked
        // before the zerocheck and merged after it.
        let mut ch_w = par_transcript.then(|| challenger.fork(b"flock-par-wiring-v1"));
        let bool_claim = match boolean {
            Some(piop) => {
                // The boolean PIOP runs over the BOOLEAN REGION only — the
                // prefix subcube `[0, 2^M_bool)`, `M_bool = M` for a
                // boolean-only registry. (The element region cannot join this
                // sum: `c = z` there.) The zerocheck flavor dispatches here;
                // everything downstream is shared, generic over the flavor's
                // skip point.
                let (z_skip, mlv_challenges, a_eval, b_eval, c_eval, r_rest, lincheck_proof) =
                    match piop {
                        BooleanPiopRef::Rs(p) => {
                            let zc = verify_zerocheck_with_grinding(
                                union.m_bool(),
                                &p.zerocheck,
                                pcs_params.zerocheck_grinding(),
                                challenger,
                            )
                            .map_err(FlockVerifyError::Zerocheck)?;
                            (
                                SkipPoint::Phi8(zc.z),
                                zc.mlv_challenges,
                                zc.a_eval,
                                zc.b_eval,
                                zc.c_eval,
                                zc.r_rest,
                                &p.lincheck,
                            )
                        }
                        BooleanPiopRef::Ag(p) => {
                            let ag = verify_ag_with_grinding(
                                union.m_bool(),
                                &p.ag,
                                pcs_params.zerocheck_grinding(),
                                challenger,
                            )
                            .map_err(FlockVerifyError::Ag)?;
                            (
                                SkipPoint::Ag(ag.r1),
                                ag.mlv_challenges,
                                ag.a_eval,
                                ag.b_eval,
                                ag.c_eval,
                                ag.r_rest,
                                &p.lincheck,
                            )
                        }
                    };
                let x_ab = union.x_ab_from_mlv(z_skip, &mlv_challenges);
                // The union-column lincheck (one circuit per BOOLEAN slot, in
                // slot order); the declared counts additionally bind through
                // the per-type const-pin target terms.
                // DEFERRED: the matrix work leaves as an assertion instead of
                // being discharged here. Callers that are not accumulating get
                // it discharged for them by the wrappers below.
                let (lc_claim, assertion) = verify_union_deferred_with_grinding(
                    union,
                    circuits,
                    &x_ab,
                    a_eval,
                    b_eval,
                    lincheck_proof,
                    pcs_params.lincheck_grinding(),
                    challenger,
                )
                .map_err(FlockVerifyError::Lincheck)?;
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
                        point: union.c_claim_point(z_skip, &r_rest),
                        value: c_eval,
                    },
                })
            }
            None => None,
        };

        // FORK/JOIN variant: the wiring replays NOW on its child (its
        // transcript is independent of the boolean's), then both children's
        // closing digests merge before the element class — the prover's
        // exact positions. The gather claims are held and appended at the
        // sequential position below, so the packed-direct order is
        // unchanged.
        let mut par_gather: Option<Vec<(Vec<F128>, F128)>> = None;
        let mut sigma: Option<SigmaAssertion> = None;
        if par_transcript {
            let UnionVerifyBinding::Circuit { circuit, public } = binding else {
                unreachable!("par_transcript requires a circuit binding");
            };
            let proof = wiring.ok_or(FlockVerifyError::CircuitMismatch)?;
            let ch = ch_w.as_mut().expect("forked above");
            let gather = if defer_sigma {
                let (gather, sig) = verify_wiring_deferred_with_grinding(
                    circuit,
                    public,
                    proof,
                    pcs_params.product_gkr_grinding(),
                    ch,
                )
                .map_err(FlockVerifyError::Wiring)?;
                sigma = Some(sig);
                gather
            } else {
                verify_wiring_with_grinding(
                    circuit,
                    public,
                    proof,
                    pcs_params.product_gkr_grinding(),
                    ch,
                )
                .map_err(FlockVerifyError::Wiring)?
            };
            par_gather = Some(gather);
            challenger.merge_child(ch_w.take().expect("forked above"));
        }

        // DEFERRED on this side too: the element class's matrix work leaves
        // as its own assertion rather than being evaluated here, so a
        // `*_deferred` entry really does defer BOTH classes.
        let el_claim = match element {
            Some(p) => {
                let (c, a) = verify_deferred_with_grinding(
                    union,
                    p,
                    pcs_params.element_grinding(),
                    challenger,
                )
                .map_err(FlockVerifyError::Element)?;
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
        // packed-direct intake the element claims ride. Deferred callers
        // get the sigma evaluation back as a claim (route B) instead of
        // paying its O(2^mu) discharge here — same transcript either way.
        if let Some(gather) = par_gather {
            packed_direct.extend(gather);
        } else if let UnionVerifyBinding::Circuit { circuit, public } = binding {
            let proof = wiring.ok_or(FlockVerifyError::CircuitMismatch)?;
            #[cfg(feature = "mul-count")]
            let wiring_start = snapshot();
            let gather = if defer_sigma {
                let (gather, sig) = verify_wiring_deferred_with_grinding(
                    circuit,
                    public,
                    proof,
                    pcs_params.product_gkr_grinding(),
                    challenger,
                )
                .map_err(FlockVerifyError::Wiring)?;
                sigma = Some(sig);
                gather
            } else {
                verify_wiring_with_grinding(
                    circuit,
                    public,
                    proof,
                    pcs_params.product_gkr_grinding(),
                    challenger,
                )
                .map_err(FlockVerifyError::Wiring)?
            };
            #[cfg(feature = "mul-count")]
            if var("MUL_TRACE").is_ok() {
                let e = snapshot();
                let invs = e.invs - wiring_start.invs;
                let muls =
                    (e.native_muls - wiring_start.native_muls).saturating_sub(invs * MULS_PER_INV);
                println!(
                    "  [mul] wiring GKR (grand product + sigma):             \
                     {muls:>8} muls {invs:>5} invs = {:>8} constraints",
                    muls + invs
                );
            }
            packed_direct.extend(gather);
        }

        // The circuit-structure accumulator also binds the succinct
        // verifier's remaining static helper evaluations. These values enter
        // Product-GKR/lincheck arithmetic above, but their truth depends only
        // on the digest-bound child circuit and are therefore folded under
        // the same key as sigma.
        if let Some(sig) = sigma.as_mut() {
            if let Some(a) = matrix.as_ref() {
                sig.boolean_pins = a
                    .pin_evals
                    .iter()
                    .enumerate()
                    .filter_map(|(t, value)| value.map(|v| (t, a.pin_point.clone(), v)))
                    .collect();
            }
            if let Some(a) = el_matrix.as_ref() {
                sig.element_constants = Some((a.r_con.clone(), a.a_const_eval, a.b_const_eval));
            }
        }

        Ok((
            UnionClassClaims {
                boolean: bool_claim,
                element: el_claim,
            },
            packed_direct,
            matrix,
            el_matrix,
            sigma,
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
    Option<MatrixAssertion>,
    Option<ElementAssertion>,
    Option<SigmaAssertion>,
);

/// Both classes' undischarged matrix work, as a `*_deferred` entry returns
/// it. Either half is `None` when that class has no types in the registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredMatrixWork {
    pub boolean: Option<MatrixAssertion>,
    pub element: Option<ElementAssertion>,
    /// The layout's count-dependent `W`-values as raw foldable claims on the
    /// jagged table (the count win). Always present — every merged opening
    /// runs the multipoint anchor — and tied to the verifier's own expect by
    /// the export's exact recombination assert.
    pub jagged: JaggedAssertion,
}

/// AG-skip mirror of [`verify_ligerito`]: replays the AG zerocheck
/// ([`verify_ag`], including the single-attempt r₁ nonce
/// re-derivation) + lincheck → base claims, then the standard ring-switch
/// Ligerito open with AG base-code skip weights. Counterpart of
/// `flock_prover::prover::prove_fast_ligerito_ag_from_witness`.
pub fn verify_ligerito_ag<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofLigeritoAg,
    lincheck_circuit: &dyn LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<R1csClaim, FlockVerifyError> {
    let (ab, c) = verify_core_ag(
        r1cs,
        &proof.ag,
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
    .map_err(FlockVerifyError::PcsAb)?;
    Ok(R1csClaim { ab, c })
}

/// AG-skip mirror of [`verify_core`]: replay bind → AG zerocheck → lincheck
/// and reconstruct the two base z-claims, stopping before the PCS open.
pub fn verify_core_ag<Ch: Challenger>(
    r1cs: &BlockR1cs,
    ag_proof: &AgProof,
    lincheck_proof: &LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), FlockVerifyError> {
    verifier_pool().install(move || {
        verify_core_ag_inner(
            r1cs,
            ag_proof,
            lincheck_proof,
            commitment,
            lincheck_circuit,
            challenger,
        )
    })
}

fn verify_core_ag_inner<Ch: Challenger>(
    r1cs: &BlockR1cs,
    ag_proof: &AgProof,
    lincheck_proof: &LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), FlockVerifyError> {
    assert_eq!(r1cs.k_skip, K_SKIP, "AG skip is k_skip=6");
    // The c-claim below is a direct z-claim (ĉ = ẑ) — sound only for C = I.
    if !r1cs.c0_is_identity() {
        return Err(FlockVerifyError::NonIdentityC);
    }

    // ---- Bind FS transcript to the statement (mirrors the AG prover).
    bind_statement(challenger, r1cs, commitment);

    // ---- Replay the AG-skip zerocheck rounds.
    let ag_claim = verify_ag(r1cs.m, ag_proof, challenger).map_err(FlockVerifyError::Ag)?;

    // ---- Lincheck on the AG quirky point (layout-aware constructors).
    let x_ab = r1cs.x_ab_from_mlv(SkipPoint::Ag(ag_claim.r1), &ag_claim.mlv_challenges);
    let lc_claim = verify_lincheck(
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        lincheck_circuit,
        &x_ab,
        ag_claim.a_eval,
        ag_claim.b_eval,
        lincheck_proof,
        challenger,
    )
    .map_err(FlockVerifyError::Lincheck)?;

    // ---- Build the two z-claims (must match what the AG prover returned).
    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(SkipPoint::Ag(ag_claim.r1), &ag_claim.r_rest),
        value: ag_claim.c_eval,
    };
    Ok((ab, c))
}

/// Verify a batched PCS opening over an arbitrary list of `ẑ`-claims — the
/// mirror of `flock_prover::prover::open_claims_with_precomputed_ligerito`.
/// Relation wrappers (e.g. the hash chain) reuse this with their own appended
/// claims. Must run at the same transcript position as the prover's open.
pub fn verify_claims_ligerito<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &BatchOpeningProofLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), PcsError> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || {
        verify_claims_ligerito_inner(commitment, claims, pcs_open, pcs_params, challenger)
    })
}

fn verify_claims_ligerito_inner<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &BatchOpeningProofLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), PcsError> {
    if !commitment_params_match_expected(commitment, pcs_params) {
        return Err(PcsError::Ligerito);
    }
    let skip_weight_vecs: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| c.point.z_skip.weights(LOG_PACKING - 1))
        .collect();
    let skip_weights: Vec<&[F128]> = skip_weight_vecs.iter().map(|v| v.as_slice()).collect();
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
    verify_opening_batch_ligerito_mixed_with_grinding(
        commitment,
        &values,
        &skip_weights,
        &x_refs,
        &[],
        pcs_open,
        &lig_v_config,
        pcs_params.opening_grinding(),
        challenger,
    )
}

/// Replay bind → zerocheck → lincheck and reconstruct the two base z-claims
/// (`ab`, `c`), stopping before the PCS open. Mirror of
/// `flock_prover::prover::prove_fast_core`; relation wrappers reuse this then call
/// [`verify_claims_ligerito`] over `[ab, c, …]`.
pub fn verify_core<Ch: Challenger>(
    r1cs: &BlockR1cs,
    zerocheck_proof: &ZerocheckProof,
    lincheck_proof: &LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), FlockVerifyError> {
    verify_core_with_grinding(
        r1cs,
        zerocheck_proof,
        lincheck_proof,
        commitment,
        lincheck_circuit,
        ZerocheckGrinding::disabled(),
        LincheckGrinding::disabled(),
        challenger,
    )
}

/// [`verify_core`] with explicit Boolean zerocheck and lincheck grinding
/// policies.
///
/// Relation-specific callers that do not carry [`crate::pcs::PcsParams`]
/// retain the legacy wrapper above.  The standard proof entries pass the
/// policy selected by their PCS profile.
pub fn verify_core_with_grinding<Ch: Challenger>(
    r1cs: &BlockR1cs,
    zerocheck_proof: &ZerocheckProof,
    lincheck_proof: &LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn LincheckCircuit,
    zerocheck_grinding: ZerocheckGrinding,
    lincheck_grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), FlockVerifyError> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || {
        verify_core_inner(
            r1cs,
            zerocheck_proof,
            lincheck_proof,
            commitment,
            lincheck_circuit,
            zerocheck_grinding,
            lincheck_grinding,
            challenger,
        )
    })
}

fn verify_core_inner<Ch: Challenger>(
    r1cs: &BlockR1cs,
    zerocheck_proof: &ZerocheckProof,
    lincheck_proof: &LincheckProof,
    commitment: &Commitment,
    lincheck_circuit: &dyn LincheckCircuit,
    zerocheck_grinding: ZerocheckGrinding,
    lincheck_grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), FlockVerifyError> {
    // The c-claim below is a direct z-claim (ĉ = ẑ) — sound only for C = I.
    if !r1cs.c0_is_identity() {
        return Err(FlockVerifyError::NonIdentityC);
    }
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
        verify_zerocheck_with_grinding(r1cs.m, zerocheck_proof, zerocheck_grinding, challenger)
            .map_err(FlockVerifyError::Zerocheck)?;
    if trace {
        eprintln!(
            "      [vco] zerocheck::verify: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Build lincheck's shared quirky point from the zerocheck output
    // (layout-aware: the mlv challenges are address-ordered).
    let x_ab = r1cs.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);

    // ---- Lincheck. v_a, v_b come from the zerocheck's final â, b̂ evals.
    let t = Instant::now();
    let lc_claim = verify_lincheck_with_grinding(
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        lincheck_circuit,
        &x_ab,
        zc_claim.a_eval,
        zc_claim.b_eval,
        lincheck_proof,
        lincheck_grinding,
        challenger,
    )
    .map_err(FlockVerifyError::Lincheck)?;
    if trace {
        eprintln!(
            "      [vco] verify_lincheck: {}",
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
        point: r1cs.c_claim_point(SkipPoint::Phi8(zc_claim.z), &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    Ok((ab, c))
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use rayon::current_num_threads;

    use crate::{
        challenger::FsChallenger,
        field::F128,
        lincheck::LincheckProof,
        merkle::HashKind,
        pcs::{Commitment, PcsParams, ligerito::LigeritoProfile},
        r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
        verifier::{
            FlockVerifyError, commitment_params_match_expected, verifier_pool, verify_core,
            verify_core_ag,
        },
        zerocheck::{
            ZerocheckProof,
            ag_skip::{AgProof, K_SKIP},
        },
    };

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
        let n = verifier_pool().install(current_num_threads);
        assert_eq!(n, 1, "verifier_pool must have exactly one worker thread");
    }

    #[test]
    fn commitment_params_cannot_select_profile_or_merkle_hash() {
        let expected = PcsParams {
            m: 22,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: None,
            merkle_hash: HashKind::Sha256,
        };
        let mut commitment = Commitment {
            cap: Vec::new(),
            params: expected.clone(),
        };
        assert!(commitment_params_match_expected(&commitment, &expected));

        commitment.params.profile = LigeritoProfile::Fast100;
        assert!(!commitment_params_match_expected(&commitment, &expected));
        commitment.params.profile = expected.profile;
        commitment.params.merkle_hash = HashKind::Blake3;
        assert!(!commitment_params_match_expected(&commitment, &expected));
    }

    /// Both single-table entries build the c-claim as a direct z-claim, which
    /// assumes `C = I`; an R1CS with any other `c_0` must be REJECTED (a
    /// structured error), not silently misverified. The guard fires before
    /// any proof inspection, so empty proofs suffice.
    #[test]
    fn non_identity_c_is_rejected_by_both_single_table_entries() {
        let k_log = 6;
        let k = 1usize << k_log;
        let identity = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: (0..k).map(|i| vec![i]).collect(),
        };
        // c_0 = a shift-by-one permutation: a valid matrix, not the identity.
        let shifted = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: (0..k).map(|i| vec![(i + 1) % k]).collect(),
        };
        let r1cs = BlockR1cs {
            m: 12,
            k_log,
            k_skip: K_SKIP,
            useful_bits: k,
            a_0: identity.clone(),
            b_0: identity.clone(),
            c_0: shifted,
            layout: WitnessLayout::RowMajor,
            const_pin: None,
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
        };
        let commitment = Commitment {
            cap: Vec::new(),
            params: PcsParams {
                m: 12,
                log_inv_rate: 1,
                log_batch_size: 6,
                profile: LigeritoProfile::Fast,
                num_lanes: None,
                merkle_hash: HashKind::Sha256,
            },
        };
        let circuit = r1cs.csc_lincheck_circuit();

        let zc = ZerocheckProof {
            round1_ab: Vec::new(),
            round1_c: Vec::new(),
            multilinear_rounds: Vec::new(),
            final_a_eval: F128::ZERO,
            final_b_eval: F128::ZERO,
            final_c_eval: F128::ZERO,
            grinding_nonces: Vec::new(),
        };
        let lc = LincheckProof {
            rounds: Vec::new(),
            z_partial: Vec::new(),
            matrix_evals: Vec::new(),
            grinding_nonces: Vec::new(),
        };
        let mut ch = FsChallenger::new(b"non-identity-c-test");
        assert!(matches!(
            verify_core(&r1cs, &zc, &lc, &commitment, circuit, &mut ch),
            Err(FlockVerifyError::NonIdentityC)
        ));

        let ag = AgProof {
            round1_ab: Vec::new(),
            round1_c: Vec::new(),
            r1_nonce: 0,
            multilinear_rounds: Vec::new(),
            final_a_eval: F128::ZERO,
            final_b_eval: F128::ZERO,
            final_c_eval: F128::ZERO,
            grinding_nonces: Vec::new(),
        };
        let mut ch = FsChallenger::new(b"non-identity-c-test");
        assert!(matches!(
            verify_core_ag(&r1cs, &ag, &lc, &commitment, circuit, &mut ch),
            Err(FlockVerifyError::NonIdentityC)
        ));
    }
}
