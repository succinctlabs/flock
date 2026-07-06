//! Top-level R1CS verifier: walks the challenger in lockstep with the
//! prover, runs `zerocheck::verify` and `lincheck::verify`, derives the two
//! ZClaims, and verifies the PCS openings at those points against the
//! witness commitment.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck;
use crate::pcs::{self, Commitment};
use crate::proof::{R1csClaim, R1csProof, R1csProofLigerito, ZClaim};
use crate::r1cs::BlockR1cs;
use crate::zerocheck;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    Zerocheck(zerocheck::VerifyError),
    Lincheck(lincheck::VerifyError),
    PcsAb(pcs::VerifyError),
    PcsC(pcs::VerifyError),
    /// The R1CS instance has a non-identity `C_0`. The pipeline assumes the
    /// circuit-R1CS shape `C = I` (the c-claim is taken as a direct z-claim),
    /// so any other instance would be checked against the wrong relation —
    /// reject it rather than verify unsoundly.
    NonIdentityC0,
}

/// Dedicated single-thread rayon pool that the verifier runs inside.
///
/// The verifier is intentionally single-threaded — matching the convention of
/// comparable provers (binius64, plonky3, hashcaster all ship serial
/// verifiers) and keeping reported verify times honest single-core numbers.
/// The verify path shares several `par_*` helpers with the (multi-threaded)
/// prover — e.g. `lincheck::fold_alpha_batched`, `sumcheck_bind_top_in_place_par`,
/// and the Ligerito residual eval — so rather than fork every shared helper, the
/// reusable verify cores (`verify_core`, `verify_claims`, `verify_claims_ligerito`)
/// run their body via `verifier_pool().install(..)`. Any `par_iter` reached from
/// there uses this 1-thread pool and collapses onto a single worker, without
/// touching the prover's use of the global pool.
fn verifier_pool() -> &'static rayon::ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
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

pub fn verify<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProof,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<R1csClaim, VerifyError> {
    // ---- Replay zerocheck + lincheck → the two base claims.
    let (ab, c) = verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )?;

    // ---- Verify the batched PCS opening covering both z-claims.
    verify_claims(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        challenger,
    )
    .map_err(VerifyError::PcsAb)?;

    Ok(R1csClaim { ab, c })
}

/// Ligerito-backend mirror of [`verify`]. Same FS protocol replay; only the
/// final PCS verification step differs.
pub fn verify_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProofLigerito,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    pcs_params: &crate::pcs::PcsParams,
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

/// Ligerito-backend mirror of [`verify_claims`].
pub fn verify_claims_ligerito<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &pcs::BatchOpeningProofLigerito,
    pcs_params: &crate::pcs::PcsParams,
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
    pcs_params: &crate::pcs::PcsParams,
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
    let log_n = pcs_params.m - pcs::LOG_PACKING;
    let lig_v_config = crate::pcs::ligerito::verifier_config_for(
        log_n,
        pcs_params.log_batch_size,
        pcs_params.profile,
    )
    .map_err(pcs::VerifyError::UnsupportedConfig)?;
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
/// [`verify_claims`] over `[ab, c, …]`.
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
    // The verifier assumes the circuit-R1CS shape `C = I`: the c-claim below is
    // built as a direct z-claim, which only matches the prover when `C_0` is
    // the identity. Reject any other instance up front instead of silently
    // checking it against the wrong relation.
    if !r1cs.c0_is_identity() {
        return Err(VerifyError::NonIdentityC0);
    }
    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    // ---- Bind FS transcript to the statement (mirrors prover::prove).
    let t = std::time::Instant::now();
    crate::proof::bind_statement(challenger, r1cs, commitment);
    if trace {
        eprintln!(
            "      [vco] bind_statement: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Zerocheck.
    let t = std::time::Instant::now();
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
    let t = std::time::Instant::now();
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

/// Verify a batched PCS opening over an arbitrary list of `ẑ`-claims — the
/// mirror of `flock_prover::prover::open_claims`. Relation wrappers (e.g. the hash
/// chain) reuse this with their own appended claims. Must run at the same
/// transcript position as the prover's open.
pub fn verify_claims<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &pcs::BatchOpeningProof,
    challenger: &mut Ch,
) -> Result<(), pcs::VerifyError> {
    // Verification is single-threaded; run the body on the dedicated 1-thread pool.
    verifier_pool().install(move || verify_claims_inner(commitment, claims, pcs_open, challenger))
}

fn verify_claims_inner<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[ZClaim],
    pcs_open: &pcs::BatchOpeningProof,
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
    pcs::verify_opening_batch(commitment, &values, &z_skips, &x_refs, pcs_open, challenger)
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

    /// `verify_core` must reject an R1CS whose `C_0` is not the identity rather
    /// than build the c-claim against the wrong relation. The guard fires before
    /// any proof field is touched, so junk (empty) sub-proofs are fine.
    #[test]
    fn verify_core_rejects_non_identity_c0() {
        use super::*;
        use crate::lincheck::LincheckProof;
        use crate::pcs::PcsParams;
        use crate::r1cs::{BlockR1cs, SparseBinaryMatrix};
        use crate::zerocheck::ZerocheckProof;
        use std::sync::OnceLock;

        let k_log = 2;
        let k = 1usize << k_log;
        let zero = || SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: vec![Vec::new(); k],
        };
        // c_0 = zero matrix  =>  c0_is_identity() == false.
        let r1cs = BlockR1cs {
            m: 4,
            k_log,
            k_skip: 1,
            useful_bits: k,
            a_0: zero(),
            b_0: zero(),
            c_0: zero(),
            const_pin: None,
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
        };
        assert!(!r1cs.c0_is_identity());

        let commitment = Commitment {
            root: [0u8; 32],
            params: PcsParams {
                m: 4,
                log_inv_rate: 1,
                log_batch_size: 0,
                profile: Default::default(),
            },
        };
        let zc = ZerocheckProof {
            round1_ab: Vec::new(),
            round1_c: Vec::new(),
            multilinear_rounds: Vec::new(),
            final_a_eval: F128::ZERO,
            final_b_eval: F128::ZERO,
            final_c_eval: F128::ZERO,
        };
        let lc = LincheckProof {
            rounds: Vec::new(),
            z_partial: Vec::new(),
        };
        let lc_circuit = r1cs.sparse_lincheck_circuit();
        let mut ch = crate::challenger::RandomChallenger::new(0);

        let res = super::verify_core(&r1cs, &zc, &lc, &commitment, &lc_circuit, &mut ch);
        assert_eq!(res.unwrap_err(), VerifyError::NonIdentityC0);
    }

    /// A commitment whose `PcsParams` have no embedded Ligerito security config
    /// (`m = 8` is outside the registered 22..=35 range) must surface a
    /// structured `UnsupportedConfig` error, not panic in `.expect`. The bound
    /// fires before the opening proof is inspected, so a junk proof is fine.
    #[test]
    fn verify_claims_ligerito_rejects_unsupported_config() {
        use super::*;
        use crate::pcs::ligerito::{FinalProof, LigeritoProof, RecursiveProof};
        use crate::pcs::{BatchOpeningProofLigerito, PcsParams};

        let commitment = Commitment {
            root: [0u8; 32],
            params: PcsParams {
                m: 8,
                log_inv_rate: 1,
                log_batch_size: 0,
                profile: Default::default(),
            },
        };
        let junk = LigeritoProof {
            initial_root: [0u8; 32],
            initial_proof: RecursiveProof {
                opened_rows: Vec::new(),
                merkle_proof: Vec::new(),
            },
            recursive_roots: Vec::new(),
            recursive_proofs: Vec::new(),
            final_proof: FinalProof {
                yr: Vec::new(),
                opened_rows: Vec::new(),
                merkle_proof: Vec::new(),
            },
            sumcheck_transcript: Vec::new(),
            grinding_nonces: Vec::new(),
            ood_values: Vec::new(),
            fold_grinding_nonces: Vec::new(),
        };
        let pcs_open = BatchOpeningProofLigerito {
            ring_switches: Vec::new(),
            ligerito: junk,
        };
        let params = commitment.params.clone();
        let mut ch = crate::challenger::RandomChallenger::new(0);

        let res = verify_claims_ligerito(&commitment, &[], &pcs_open, &params, &mut ch);
        assert!(matches!(res, Err(pcs::VerifyError::UnsupportedConfig(_))));
    }
}
