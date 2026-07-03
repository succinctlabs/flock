//! E6 — end-to-end R1CS prove/verify on an L1′-committed witness, generic
//! over the hash encoder (keccak / sha2 / blake3) and the PCS backend
//! (BaseFold / Ligerito).
//!
//! Mirrors `flock_prover::prover::prove_fast_*` / `flock_core::verifier`,
//! with the L1′ bookkeeping differences:
//!
//! - the witness is produced directly in L1′ and committed as-is; the
//!   producers fully write the useful chunk-column prefix, so recycled
//!   buffers only need their **suffix** zeroed (padding columns);
//! - zerocheck runs on the L1′ buffers with the suffix `PaddingSpec`
//!   (`k_log = m`, useful prefix), via the s_hat_v_c capture kernel;
//! - the zerocheck→lincheck point translation splits `mlv_challenges` in
//!   L1′ address order `[dim6 | batch | chunk]`;
//! - PCS claim points are assembled in address order
//!   (`x_full_ab = [r_inner_rest[0]] ++ x_outer ++ r_inner_rest[1..]`,
//!   `x_full_c = zc.r_rest` verbatim) and fed through the unmodified
//!   `verifier::verify_claims{,_ligerito}`;
//! - both `s_hat_v` precomputes are reused verbatim: the zerocheck two-bank
//!   `s_hat_v_c` capture is address-generic, and `s_hat_v_from_z_vec`
//!   computes the semantic (layout-independent) bit-slice evaluations from
//!   lincheck's `z_vec`, whose index layout is unchanged under L1′.
//!
//! Lincheck itself is untouched: its inputs (byte-stripe, circuit walker,
//! semantic quirky point) are layout-independent.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::{self, LincheckCircuit, QuirkyPoint};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::{R1csProof, R1csProofLigerito, ZClaim, bind_statement};
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier::{verify_claims, verify_claims_ligerito};
use flock_core::zerocheck::{self, PaddingSpec};

/// Everything hash-specific the L1′ pipeline needs.
pub struct L1HashSpec<'a, S> {
    pub r1cs: &'a BlockR1cs,
    pub circuit: &'a dyn LincheckCircuit,
    #[allow(clippy::type_complexity)]
    pub direct: &'a (dyn Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64])
             + Sync),
}

/// L1′ suffix padding: one giant block with a useful chunk-column prefix.
pub fn l1_padding_spec(m: usize, k_log: usize, useful_bits: usize) -> PaddingSpec {
    let n_log = m - k_log;
    PaddingSpec {
        k_log: m,
        useful_bits_per_block: useful_bits.div_ceil(128) << (7 + n_log),
    }
}

fn as_u8(v: &[F128]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn as_u64_mut(v: &mut [F128]) -> &mut [u64] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, v.len() * 2) }
}

/// Split the zerocheck's address-ordered `mlv_challenges` into the semantic
/// quirky point under L1′.
fn x_ab_from_zerocheck(zc: &zerocheck::ZerocheckClaim, k_log: usize, n_log: usize) -> QuirkyPoint {
    let mlv = &zc.mlv_challenges;
    assert_eq!(mlv.len(), 1 + n_log + (k_log - 7));
    let mut x_inner_rest = Vec::with_capacity(k_log - 6);
    x_inner_rest.push(mlv[0]);
    x_inner_rest.extend_from_slice(&mlv[1 + n_log..]);
    QuirkyPoint {
        z_skip: zc.z,
        x_inner_rest,
        x_outer: mlv[1..1 + n_log].to_vec(),
    }
}

/// Address-ordered PCS point for the AB claim.
fn x_full_ab(r_inner_rest: &[F128], x_outer: &[F128]) -> Vec<F128> {
    let mut v = Vec::with_capacity(r_inner_rest.len() + x_outer.len());
    v.push(r_inner_rest[0]);
    v.extend_from_slice(x_outer);
    v.extend_from_slice(&r_inner_rest[1..]);
    v
}

/// `ZClaim` carrying an address-ordered point (x_outer segment empty — the
/// verifier's PCS step concatenates the segments, so this reproduces the
/// L1′ ordering through the unmodified `verify_claims`).
fn flat_claim(z_skip: F128, x_full: Vec<F128>, value: F128) -> ZClaim {
    ZClaim {
        point: QuirkyPoint {
            z_skip,
            x_inner_rest: x_full,
            x_outer: Vec::new(),
        },
        value,
    }
}

/// Shared prove core: witness → commit → bind → zerocheck → lincheck →
/// claims + s_hat_v precomputes, stopping before the backend-specific open.
struct L1Core {
    commitment: Commitment,
    prover_data: pcs::ProverData,
    z: Vec<F128>,
    zc_proof: zerocheck::ZerocheckProof,
    lc_proof: lincheck::LincheckProof,
    ab: ZClaim,
    c: ZClaim,
    padding: PaddingSpec,
    s_hat_v_ab: Option<Vec<F128>>,
    s_hat_v_c: Option<Vec<F128>>,
}

fn prove_l1_core<S, Ch: Challenger>(
    spec: &L1HashSpec<'_, S>,
    pcs_params: &PcsParams,
    inputs: &[S],
    use_precomputes: bool,
    challenger: &mut Ch,
) -> L1Core {
    let r1cs = spec.r1cs;
    let m = r1cs.m;
    let k_log = r1cs.k_log;
    let n_log = m - k_log;
    let total_f128 = 1usize << (m - 7);
    let useful_chunks = r1cs.useful_bits.div_ceil(128);
    let timing = std::env::var_os("E6_TIMING").is_some();
    let mut t0 = std::time::Instant::now();
    let phase = |name: &str, t0: &mut std::time::Instant| {
        if timing {
            eprintln!("[e6-l1] {name}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        *t0 = std::time::Instant::now();
    };

    // ---- Witness directly in L1′ (+ lincheck stripe), scratch-recycled
    // buffers. The producers fully write chunk-columns [0, useful_chunks),
    // so only the padding suffix needs zeroing.
    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut stripe = vec![0u8; 1usize << (m - 3)];
    {
        use rayon::prelude::*;
        let tail_start = useful_chunks << n_log;
        for buf in [&mut z, &mut a, &mut b] {
            buf[tail_start..]
                .par_chunks_mut(1 << 16)
                .for_each(|c| c.fill(F128::ZERO));
        }
    }
    phase("alloc+zero-suffix", &mut t0);
    (spec.direct)(
        inputs,
        n_log,
        Some(&mut stripe),
        as_u64_mut(&mut z),
        as_u64_mut(&mut a),
        as_u64_mut(&mut b),
    );
    phase("witness", &mut t0);

    // ---- Commit + bind.
    let (commitment, prover_data) = pcs::commit(&z, pcs_params);
    bind_statement(challenger, r1cs, &commitment);
    phase("commit", &mut t0);

    // ---- Zerocheck on the L1′ buffers (c aliases z), suffix padding.
    let padding = l1_padding_spec(m, k_log, r1cs.useful_bits);
    let (zc_proof, zc_claim, s_hat_v_c) = if use_precomputes {
        let (p, cl, s) = zerocheck::prove_packed_padded_capture_s_hat_v_c(
            as_u8(&a),
            as_u8(&b),
            as_u8(&z),
            m,
            &padding,
            challenger,
        );
        (p, cl, Some(s))
    } else {
        let (p, cl) = zerocheck::prove_packed_padded(
            as_u8(&a),
            as_u8(&b),
            as_u8(&z),
            m,
            &padding,
            challenger,
        );
        (p, cl, None)
    };
    flock_core::scratch::give_f128(a);
    flock_core::scratch::give_f128(b);
    phase("zerocheck", &mut t0);

    // ---- Lincheck (layout-independent). Capture z_vec for the AB s_hat_v.
    let x_ab = x_ab_from_zerocheck(&zc_claim, k_log, n_log);
    let (lc_proof, lc_claim, z_vec) = lincheck::prove_padded_capture_z_vec(
        &stripe,
        m,
        k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        spec.circuit,
        &x_ab,
        challenger,
    );
    drop(stripe);
    phase("lincheck", &mut t0);

    // ---- Address-ordered PCS claims + AB s_hat_v (semantic, so the
    // row-major derivation from z_vec applies verbatim under L1′).
    let ab = flat_claim(
        lc_claim.r_inner_skip,
        x_full_ab(&lc_claim.r_inner_rest, &x_ab.x_outer),
        lc_claim.w,
    );
    let c = flat_claim(zc_claim.z, zc_claim.r_rest.clone(), zc_claim.c_eval);
    let s_hat_v_ab = if use_precomputes && k_log >= pcs::LOG_PACKING {
        Some(pcs::ring_switch::s_hat_v_from_z_vec(
            &z_vec,
            &lc_claim.r_inner_rest[1..],
        ))
    } else {
        None
    };
    phase("claims+s_hat_v", &mut t0);

    L1Core {
        commitment,
        prover_data,
        z,
        zc_proof,
        lc_proof,
        ab,
        c,
        padding,
        s_hat_v_ab,
        s_hat_v_c,
    }
}

pub struct L1Prove {
    pub proof: R1csProof,
    pub commitment: Commitment,
    pub ab: ZClaim,
    pub c: ZClaim,
}

pub struct L1ProveLigerito {
    pub proof: R1csProofLigerito,
    pub commitment: Commitment,
    pub ab: ZClaim,
    pub c: ZClaim,
}

/// End-to-end L1′ prove, BaseFold backend.
pub fn prove_l1_basefold<S, Ch: Challenger>(
    spec: &L1HashSpec<'_, S>,
    pcs_params: &PcsParams,
    inputs: &[S],
    use_precomputes: bool,
    challenger: &mut Ch,
) -> L1Prove {
    let core = prove_l1_core(spec, pcs_params, inputs, use_precomputes, challenger);
    let x_refs: Vec<&[F128]> = vec![&core.ab.point.x_inner_rest, &core.c.point.x_inner_rest];
    let pre: Vec<Option<&[F128]>> =
        vec![core.s_hat_v_ab.as_deref(), core.s_hat_v_c.as_deref()];
    let pcs_open = pcs::open_batch_padded_with_precomputed_s_hat_v(
        &core.z,
        &core.prover_data,
        &core.commitment,
        &x_refs,
        &pre,
        &core.padding,
        challenger,
    );
    flock_core::scratch::give_f128(core.z);
    L1Prove {
        proof: R1csProof {
            zerocheck: core.zc_proof,
            lincheck: core.lc_proof,
            pcs_open,
        },
        commitment: core.commitment,
        ab: core.ab,
        c: core.c,
    }
}

/// End-to-end L1′ prove, Ligerito backend (production PCS).
pub fn prove_l1_ligerito<S, Ch: Challenger>(
    spec: &L1HashSpec<'_, S>,
    pcs_params: &PcsParams,
    inputs: &[S],
    use_precomputes: bool,
    challenger: &mut Ch,
) -> L1ProveLigerito {
    let core = prove_l1_core(spec, pcs_params, inputs, use_precomputes, challenger);
    let log_n = pcs_params.m - pcs::LOG_PACKING;
    let lig_config =
        pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
            .expect("Ligerito config for this m");
    let x_fulls = [
        core.ab.point.x_inner_rest.clone(),
        core.c.point.x_inner_rest.clone(),
    ];
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let pre: Vec<Option<&[F128]>> =
        vec![core.s_hat_v_ab.as_deref(), core.s_hat_v_c.as_deref()];
    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        core.z,
        &core.prover_data,
        &core.commitment,
        &x_refs,
        &pre,
        &[],
        &core.padding,
        &lig_config,
        challenger,
    );
    L1ProveLigerito {
        proof: R1csProofLigerito {
            zerocheck: core.zc_proof,
            lincheck: core.lc_proof,
            pcs_open,
        },
        commitment: core.commitment,
        ab: core.ab,
        c: core.c,
    }
}

#[derive(Debug)]
pub enum L1VerifyError {
    Zerocheck(zerocheck::VerifyError),
    Lincheck(lincheck::VerifyError),
    Pcs(pcs::VerifyError),
}

/// Replay bind → zerocheck → lincheck with the L1′ point assembly; returns
/// the two address-ordered claims (backend-agnostic).
fn verify_l1_core<Ch: Challenger>(
    r1cs: &BlockR1cs,
    circuit: &dyn LincheckCircuit,
    commitment: &Commitment,
    zc_proof: &zerocheck::ZerocheckProof,
    lc_proof: &lincheck::LincheckProof,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), L1VerifyError> {
    let m = r1cs.m;
    let k_log = r1cs.k_log;
    let n_log = m - k_log;

    bind_statement(challenger, r1cs, commitment);
    let zc_claim =
        zerocheck::verify(m, zc_proof, challenger).map_err(L1VerifyError::Zerocheck)?;
    let x_ab = x_ab_from_zerocheck(&zc_claim, k_log, n_log);
    let lc_claim = lincheck::verify(
        m,
        k_log,
        r1cs.k_skip,
        circuit,
        &x_ab,
        zc_claim.a_eval,
        zc_claim.b_eval,
        lc_proof,
        challenger,
    )
    .map_err(L1VerifyError::Lincheck)?;

    let ab = flat_claim(
        lc_claim.r_inner_skip,
        x_full_ab(&lc_claim.r_inner_rest, &x_ab.x_outer),
        lc_claim.w,
    );
    let c = flat_claim(zc_claim.z, zc_claim.r_rest.clone(), zc_claim.c_eval);
    Ok((ab, c))
}

/// Mirror verifier, BaseFold backend.
pub fn verify_l1_basefold<Ch: Challenger>(
    r1cs: &BlockR1cs,
    circuit: &dyn LincheckCircuit,
    commitment: &Commitment,
    proof: &R1csProof,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), L1VerifyError> {
    let (ab, c) = verify_l1_core(
        r1cs,
        circuit,
        commitment,
        &proof.zerocheck,
        &proof.lincheck,
        challenger,
    )?;
    verify_claims(commitment, &[ab.clone(), c.clone()], &proof.pcs_open, challenger)
        .map_err(L1VerifyError::Pcs)?;
    Ok((ab, c))
}

/// Mirror verifier, Ligerito backend.
pub fn verify_l1_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    circuit: &dyn LincheckCircuit,
    commitment: &Commitment,
    proof: &R1csProofLigerito,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), L1VerifyError> {
    let (ab, c) = verify_l1_core(
        r1cs,
        circuit,
        commitment,
        &proof.zerocheck,
        &proof.lincheck,
        challenger,
    )?;
    verify_claims_ligerito(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        pcs_params,
        challenger,
    )
    .map_err(L1VerifyError::Pcs)?;
    Ok((ab, c))
}

/// (r1cs, pcs_params) mirroring the production `*Setup` types
/// (log_inv_rate = 1, Fast profile).
pub fn setup(r1cs: BlockR1cs) -> (BlockR1cs, PcsParams) {
    let pcs_params = PcsParams {
        m: r1cs.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    (r1cs, pcs_params)
}
