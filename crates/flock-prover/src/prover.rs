//! Top-level R1CS prover: composes zerocheck + lincheck for block-diagonal
//! circuit R1CS instances. Outputs **two** z-claims at different quirky
//! points that the PCS layer (when it lands) will verify against `z`'s
//! commitment.
//!
//! Flow:
//! ```text
//!     witness z ──► pack ──► a = A·z, b = B·z, c = z (since C=I)
//!         │
//!         │       ┌─────────────┐
//!         │       │  zerocheck  │  reduces a·b ⊕ c = 0 to MLE claims:
//!         │       │             │  • â(z, mlv_challenges) = v_a
//!         │       │             │  • b̂(z, mlv_challenges) = v_b
//!         │       │             │  • ĉ(z, r_rest)         = v_c  ← directly a z-claim
//!         │       └─────────────┘
//!         │
//!         │       ┌─────────────┐
//!         │ ─► z ─►  lincheck   │  reduces â, b̂ claims (same point) to a
//!         │       │             │  single z-claim at (r_inner_skip,
//!         │       │             │                      r_inner_rest,
//!         │       │             │                      x_ab.x_outer).
//!         │       └─────────────┘
//!         │
//!         ▼
//!     R1csClaim { ab: z-claim from lincheck,  c: z-claim from extract_c }
//! ```

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::{self, QuirkyPoint, pack_z_lincheck_from_packed};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProof, R1csProofLigerito, ZClaim, bind_statement};
use flock_core::r1cs::BlockR1cs;
use flock_core::zerocheck;

/// Construct a multilinear `x_outer_full` of length `m − k_skip` from a
/// QuirkyPoint: concatenate `x_inner_rest` and `x_outer`. This is the format
/// the PCS expects (k_skip = 6 absorbed via `z_skip`; everything else is
/// multilinear).
pub(crate) fn quirky_x_outer_full(point: &QuirkyPoint) -> Vec<F128> {
    let mut v = Vec::with_capacity(point.x_inner_rest.len() + point.x_outer.len());
    v.extend_from_slice(&point.x_inner_rest);
    v.extend_from_slice(&point.x_outer);
    v
}

/// Batched PCS open over an arbitrary list of `ẑ`-evaluation claims, in one
/// shared BaseFold/FRI. This is the generic seam: the base R1CS proof opens
/// `[ab, c]`; relation wrappers (e.g. the hash chain) append their own claims
/// and open `[ab, c, …]`. The claims' `value`s aren't needed here (the opening
/// is over the points); the verifier supplies them to [`open_claims`]'s mirror.
///
/// Must be called at the same transcript position as the verifier's
/// [`flock_core::verifier::verify_claims`].
/// Per-claim optional precomputed `s_hat_v` is passed through to ring-switch:
/// when `Some(v)`, the claim skips `fold_1b_rows` and uses `v` directly.
/// Caller responsibility: each `Some(v)` MUST equal what `fold_1b_rows` would
/// produce on `z_packed` against the claim's suffix — see
/// [`pcs::ring_switch::s_hat_v_from_z_vec`] for the AB-claim derivation.
pub(crate) fn open_claims_with_precomputed<Ch: Challenger>(
    z_packed: &[F128],
    prover_data: &pcs::ProverData,
    commitment: &Commitment,
    claims: &[ZClaim],
    precomputed_s_hat_v: &[Option<&[F128]>],
    padding: &zerocheck::PaddingSpec,
    challenger: &mut Ch,
) -> pcs::BatchOpeningProof {
    let x_fulls: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| quirky_x_outer_full(&c.point))
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    pcs::open_batch_padded_with_precomputed_s_hat_v(
        z_packed,
        prover_data,
        commitment,
        &x_refs,
        precomputed_s_hat_v,
        padding,
        challenger,
    )
}

/// Ligerito-backend counterpart to [`open_claims_with_precomputed`]. Same
/// transcript shape from the caller's POV; just routes through Ligerito
/// instead of BaseFold at the final PCS step.
pub(crate) fn open_claims_with_precomputed_ligerito<Ch: Challenger>(
    z_packed: Vec<F128>,
    prover_data: &pcs::ProverData,
    commitment: &Commitment,
    claims: &[ZClaim],
    precomputed_s_hat_v: &[Option<&[F128]>],
    padding: &zerocheck::PaddingSpec,
    lig_config: &pcs::ligerito::ProverConfig,
    challenger: &mut Ch,
) -> pcs::BatchOpeningProofLigerito {
    let x_fulls: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| quirky_x_outer_full(&c.point))
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        prover_data,
        commitment,
        &x_refs,
        precomputed_s_hat_v,
        &[],
        padding,
        lig_config,
        challenger,
    )
}

/// Run the full R1CS proof on an F_{2^128}-packed witness.
///
/// The witness is in the canonical packed form (polynomial basis: bit `r` of
/// `z_packed[i]` = logical bit `i·128 + r`), length `2^(m - 7)`. The prover
/// never unpacks; downstream R1CS/zerocheck/lincheck/PCS all consume packed
/// representations.
///
/// Returns the proof bundle, the witness commitment, and the two claims (which
/// the verifier needs to know to check the openings).
pub fn prove<Ch: Challenger>(
    r1cs: &BlockR1cs,
    z_packed: &[F128],
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> (R1csProof, Commitment, R1csClaim) {
    assert_eq!(z_packed.len(), 1usize << (r1cs.m - 7));
    assert_eq!(pcs_params.m, r1cs.m);

    // ---- PCS commit to z (already packed). Pass by reference — commit copies
    // the witness into the codeword buffer and doesn't retain it.
    let (commitment, prover_data) = pcs::commit(z_packed, pcs_params);

    // ---- Bind FS transcript to the statement (R1CS instance + commitment).
    bind_statement(challenger, r1cs, &commitment);

    // ---- Compute a = A·z, b = B·z in packed form. For the circuit-R1CS
    // convention C = I (every production instance), c = C·z = z — alias it
    // instead of running a third block-diagonal apply.
    let a_packed_f128 = r1cs.apply_a_packed(z_packed);
    let b_packed_f128 = r1cs.apply_b_packed(z_packed);
    let c_packed_f128: Vec<F128> = if r1cs.c0_is_identity() {
        Vec::new() // unused — c aliases z below
    } else {
        r1cs.apply_c_packed(z_packed)
    };

    // ---- View a/b/c as LSB-first bytes for zerocheck. `F128` is
    // `repr(C, align(16))` with two little-endian u64s; that byte layout is
    // identical to the bit packing the zerocheck consumes, so a raw cast
    // suffices — no allocation, no copy.
    let cast = |v: &[F128]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    let a_packed: &[u8] = cast(&a_packed_f128);
    let b_packed: &[u8] = cast(&b_packed_f128);
    let c_packed: &[u8] = if c_packed_f128.is_empty() {
        cast(z_packed)
    } else {
        cast(&c_packed_f128)
    };

    // ---- Stripe-pack z from F_{2^128}-packed for lincheck.
    let z_packed_lincheck = pack_z_lincheck_from_packed(z_packed, r1cs.m, r1cs.k_log);

    // ---- Zerocheck.
    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let (zc_proof, zc_claim, s_hat_v_c) = zerocheck::prove_packed_padded_capture_s_hat_v_c(
        a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
    );

    // ---- Translate zerocheck output → lincheck input.
    //
    // Zerocheck's claim point for (â, b̂) is `(z, mlv_challenges)` where:
    //   - `z = zc_claim.z` is the URM challenge (binds k_skip skip vars)
    //   - `mlv_challenges[0..k_log-k_skip]` binds the inner-rest bits (between
    //      the skip dim and the lincheck inner boundary).
    //   - `mlv_challenges[k_log-k_skip..]` binds the outer bits.
    let inner_rest_len = r1cs.k_log - r1cs.k_skip;
    let x_ab = QuirkyPoint {
        z_skip: zc_claim.z,
        x_inner_rest: zc_claim.mlv_challenges[..inner_rest_len].to_vec(),
        x_outer: zc_claim.mlv_challenges[inner_rest_len..].to_vec(),
    };

    // ---- Lincheck. Capture pre-sumcheck z_vec to skip fold_1b_rows for AB
    // at PCS-open time.
    let lc_circuit = r1cs.csc_lincheck_circuit();
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        lc_circuit,
        &x_ab,
        challenger,
    );

    // ---- Build the two z-claims for the PCS.
    let ab = ZClaim {
        point: QuirkyPoint {
            z_skip: lc_claim.r_inner_skip,
            x_inner_rest: lc_claim.r_inner_rest.clone(),
            x_outer: x_ab.x_outer.clone(),
        },
        value: lc_claim.w,
    };
    // c-claim: zerocheck gives `ĉ(point_c) = c_eval` where ĉ is the MLE of
    // `C·z`. Since `C = I`, ĉ = ẑ and the c-claim is already a z-claim.
    let c = ZClaim {
        point: QuirkyPoint {
            z_skip: zc_claim.z,
            x_inner_rest: zc_claim.r_rest[..inner_rest_len].to_vec(),
            x_outer: zc_claim.r_rest[inner_rest_len..].to_vec(),
        },
        value: zc_claim.c_eval,
    };

    // ---- Single batched PCS open covering both z-claims via one BaseFold.
    // AB s_hat_v derived from lincheck's pre-sumcheck z_vec; skips
    // fold_1b_rows for the AB claim. Skip when k_log < LOG_PACKING.
    let s_hat_v_ab = if r1cs.k_log >= pcs::LOG_PACKING {
        Some(pcs::ring_switch::s_hat_v_from_z_vec(
            &z_vec_pre,
            &lc_claim.r_inner_rest[1..],
        ))
    } else {
        None
    };
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = open_claims_with_precomputed(
        z_packed,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        challenger,
    );

    let proof = R1csProof {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim)
}

/// Ligerito-backend mirror of [`prove`]. Drop-in replacement returning
/// `R1csProofLigerito` instead of `R1csProof`. Same FS protocol upstream
/// (commit, zerocheck, lincheck); only the final PCS step differs.
pub fn prove_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    z_packed: Vec<F128>,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    assert_eq!(z_packed.len(), 1usize << (r1cs.m - 7));
    assert_eq!(pcs_params.m, r1cs.m);

    let log_n = r1cs.m - pcs::LOG_PACKING;
    let lig_config =
        pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
            .expect("Ligerito default config; bump m or use prove (BaseFold) for tiny instances");

    let (commitment, prover_data) = pcs::commit(&z_packed, pcs_params);
    bind_statement(challenger, r1cs, &commitment);

    // a = A·z, b = B·z; for the C = I convention c aliases z (see prove()).
    let a_packed_f128 = r1cs.apply_a_packed(&z_packed);
    let b_packed_f128 = r1cs.apply_b_packed(&z_packed);
    let c_packed_f128: Vec<F128> = if r1cs.c0_is_identity() {
        Vec::new()
    } else {
        r1cs.apply_c_packed(&z_packed)
    };
    let cast = |v: &[F128]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    let a_packed: &[u8] = cast(&a_packed_f128);
    let b_packed: &[u8] = cast(&b_packed_f128);
    let c_packed: &[u8] = if c_packed_f128.is_empty() {
        cast(&z_packed)
    } else {
        cast(&c_packed_f128)
    };
    let z_packed_lincheck = pack_z_lincheck_from_packed(&z_packed, r1cs.m, r1cs.k_log);

    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let (zc_proof, zc_claim, s_hat_v_c) = zerocheck::prove_packed_padded_capture_s_hat_v_c(
        a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
    );

    let inner_rest_len = r1cs.k_log - r1cs.k_skip;
    let x_ab = QuirkyPoint {
        z_skip: zc_claim.z,
        x_inner_rest: zc_claim.mlv_challenges[..inner_rest_len].to_vec(),
        x_outer: zc_claim.mlv_challenges[inner_rest_len..].to_vec(),
    };

    let lc_circuit =
        lincheck::SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        &lc_circuit,
        &x_ab,
        challenger,
    );

    let ab = ZClaim {
        point: QuirkyPoint {
            z_skip: lc_claim.r_inner_skip,
            x_inner_rest: lc_claim.r_inner_rest.clone(),
            x_outer: x_ab.x_outer.clone(),
        },
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: QuirkyPoint {
            z_skip: zc_claim.z,
            x_inner_rest: zc_claim.r_rest[..inner_rest_len].to_vec(),
            x_outer: zc_claim.r_rest[inner_rest_len..].to_vec(),
        },
        value: zc_claim.c_eval,
    };

    let s_hat_v_ab = if r1cs.k_log >= pcs::LOG_PACKING {
        Some(pcs::ring_switch::s_hat_v_from_z_vec(
            &z_vec_pre,
            &lc_claim.r_inner_rest[1..],
        ))
    } else {
        None
    };
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = open_claims_with_precomputed_ligerito(
        z_packed,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &lig_config,
        challenger,
    );

    let proof = R1csProofLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim)
}

/// Shared `prove_fast` pipeline for the monolithic hash R1CS modules. Takes
/// the four packed buffers produced by the per-hash
/// `generate_witness_with_ab_packed_and_lincheck` and runs commit → zerocheck
/// → lincheck → PCS-open. Uses the c-aliasing trick (`C = I` → `c == z`
/// byte-for-byte).
pub fn prove_fast_from_witness<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (R1csProof, Commitment, R1csClaim) {
    let core = prove_fast_core(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        lincheck_circuit,
        challenger,
    );

    // ---- Single batched PCS open over the two base claims. AB-claim's
    // s_hat_v is precomputed from lincheck's pre-sumcheck z_vec, skipping
    // fold_1b_rows for that claim.
    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let pre_ab: Option<&[F128]> = core.s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(core.s_hat_v_c.as_slice());
    let pcs_open = open_claims_with_precomputed(
        &core.z_packed,
        &core.prover_data,
        &core.commitment,
        &[core.ab.clone(), core.c.clone()],
        &[pre_ab, pre_c],
        &padding,
        challenger,
    );

    let proof = R1csProof {
        zerocheck: core.zc_proof,
        lincheck: core.lc_proof,
        pcs_open,
    };
    let claim = R1csClaim {
        ab: core.ab,
        c: core.c,
    };
    // Recycle the packed witness (the open was its last reader).
    flock_core::scratch::give_f128(core.z_packed);
    (proof, core.commitment, claim)
}

/// Ligerito-backend mirror of [`prove_fast_from_witness`]. Used by per-hash
/// modules' `prove_fast_ligerito` methods.
pub fn prove_fast_ligerito_from_witness<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    let log_n = r1cs.m - pcs::LOG_PACKING;
    let lig_config =
        pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
            .expect(
                "Ligerito default config; bump m or use prove_fast (BaseFold) for tiny instances",
            );

    let ProveCore {
        zc_proof,
        lc_proof,
        ab,
        c,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
    } = prove_fast_core_with_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    );

    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = open_claims_with_precomputed_ligerito(
        z_packed,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &lig_config,
        challenger,
    );

    let proof = R1csProofLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim)
}

/// Everything the prover produces *before* the PCS open: the zerocheck +
/// lincheck sub-proofs, the two base z-claims (`ab`, `c`), and the retained
/// commitment / prover-data / packed witness needed to open more claims.
///
/// The generic seam: `prove_fast_from_witness` = `prove_fast_core` +
/// `open_claims([ab, c])`; a relation wrapper (e.g. the hash chain) runs the
/// same core, derives extra z-claims, and calls `open_claims([ab, c, …])`.
pub struct ProveCore {
    pub zc_proof: zerocheck::ZerocheckProof,
    pub lc_proof: lincheck::LincheckProof,
    pub ab: ZClaim,
    pub c: ZClaim,
    pub commitment: Commitment,
    pub prover_data: pcs::ProverData,
    pub z_packed: Vec<F128>,
    /// Precomputed `s_hat_v` for the AB claim — derived from lincheck's
    /// pre-sumcheck `z_vec` via [`pcs::ring_switch::s_hat_v_from_z_vec`].
    /// Skips `fold_1b_rows` for the AB claim at PCS-open time.
    ///
    /// `None` when `k_log < LOG_PACKING` (the kernel needs `z_vec.len() ==
    /// 2^LOG_PACKING * 2^tail.len()`, which requires `k_log >= LOG_PACKING`).
    /// Real R1CS instances have `k_log >= 16` so this branch only fires in
    /// tiny test setups.
    pub s_hat_v_ab: Option<Vec<F128>>,
    /// Precomputed `s_hat_v` for the C claim — produced by zerocheck round 1's
    /// two-bank fusion kernel (one extra `vld1q+veorq` per chunk-lane-b_med
    /// vs the original single-bank C-side). Skips `fold_1b_rows` for the C
    /// claim at PCS-open time.
    pub s_hat_v_c: Vec<F128>,
}

/// Run commit → bind → zerocheck → lincheck and build the base claims, stopping
/// just before the PCS open. See [`ProveCore`].
pub fn prove_fast_core<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> ProveCore {
    prove_fast_core_with_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        lincheck_circuit,
        None,
        challenger,
    )
}

/// [`prove_fast_core`] with an optional pre-faulted codeword buffer (see
/// [`pcs::prefault_codeword_during`]). When `Some`, the commit reuses it via
/// [`pcs::commit_into`] instead of allocating — the alloc was already done,
/// overlapped with witness generation. When `None`, behaves exactly like
/// [`prove_fast_core`] (commit allocates inline).
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_core_with_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> ProveCore {
    let (commitment, prover_data) = match prefaulted_codeword {
        Some(buf) => pcs::commit_into(&z_packed, pcs_params, buf),
        None => pcs::commit(&z_packed, pcs_params),
    };
    bind_statement(challenger, r1cs, &commitment);

    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let (zc_proof, zc_claim, s_hat_v_c) = {
        // Zero-cost &[u8] views of the F128 buffers; c aliases z (C = I).
        let a_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                a_packed_f128.as_ptr() as *const u8,
                a_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let b_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                b_packed_f128.as_ptr() as *const u8,
                b_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let c_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                z_packed.as_ptr() as *const u8,
                z_packed.len() * core::mem::size_of::<F128>(),
            )
        };
        zerocheck::prove_packed_padded_capture_s_hat_v_c(
            a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
        )
    };
    // Nothing downstream reads a/b (zerocheck consumed them in rounds 1–2);
    // recycle the two buffers (2 × 2^(m-3) bytes — 128 MB at m = 29) instead
    // of carrying them through lincheck and the PCS open.
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);

    let inner_rest_len = r1cs.k_log - r1cs.k_skip;
    let x_ab = QuirkyPoint {
        z_skip: zc_claim.z,
        x_inner_rest: zc_claim.mlv_challenges[..inner_rest_len].to_vec(),
        x_outer: zc_claim.mlv_challenges[inner_rest_len..].to_vec(),
    };

    // Capture lincheck's pre-sumcheck z_vec so the PCS open can derive the
    // AB-claim's `s_hat_v` from it (skips fold_1b_rows for AB).
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        lincheck_circuit,
        &x_ab,
        challenger,
    );
    // The lincheck stripe copy of z is dead from here on; free it before the
    // PCS open (2^(m-3) bytes — 64 MB at m = 29).
    drop(z_packed_lincheck);

    let ab = ZClaim {
        point: QuirkyPoint {
            z_skip: lc_claim.r_inner_skip,
            x_inner_rest: lc_claim.r_inner_rest.clone(),
            x_outer: x_ab.x_outer.clone(),
        },
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: QuirkyPoint {
            z_skip: zc_claim.z,
            x_inner_rest: zc_claim.r_rest[..inner_rest_len].to_vec(),
            x_outer: zc_claim.r_rest[inner_rest_len..].to_vec(),
        },
        value: zc_claim.c_eval,
    };

    // Strided fold of z_vec_pre against the AB-claim suffix's inner-rest tail
    // (everything past prefix0). Byte-identical to `fold_1b_rows` on the AB
    // suffix tensor — see `s_hat_v_from_z_vec`. Skip when k_log < LOG_PACKING
    // (only test setups; real R1CS has k_log >= 16).
    let s_hat_v_ab = if r1cs.k_log >= pcs::LOG_PACKING {
        Some(pcs::ring_switch::s_hat_v_from_z_vec(
            &z_vec_pre,
            &lc_claim.r_inner_rest[1..],
        ))
    } else {
        None
    };

    ProveCore {
        zc_proof,
        lc_proof,
        ab,
        c,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
    }
}

/// Per-phase wall-clock timings (seconds) of the Ligerito fast prover, for
/// benchmark cost breakdowns. See [`prove_fast_ligerito_timed`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ProvePhaseTimings {
    /// Witness generation. Filled by the per-setup `prove_fast_timed` wrappers
    /// (not by [`prove_fast_ligerito_timed`], which takes the witness as input).
    pub witness_s: f64,
    pub commit_s: f64,
    pub zerocheck_s: f64,
    /// Lincheck prove + the small post-lincheck base-claim / `s_hat_v` setup.
    pub lincheck_s: f64,
    /// The real Ligerito recursive PCS open (`open_claims_…_ligerito`).
    pub open_s: f64,
}

/// [`prove_fast_ligerito_from_witness`] with per-phase timers. Inlines the same
/// calls in the same order as `prove_fast_core_with_codeword` + the Ligerito
/// open, wrapping each phase in an `Instant`, so the returned
/// [`ProvePhaseTimings`] decompose the *real* Ligerito prover --- including its
/// recursive opening --- not a BaseFold-style reconstruction. Kept in lockstep
/// with `prove_fast_ligerito_from_witness`; benchmark-only.
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_timed<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim, ProvePhaseTimings) {
    use std::time::Instant;
    let mut t = ProvePhaseTimings::default();

    let log_n = r1cs.m - pcs::LOG_PACKING;
    let lig_config =
        pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
            .expect(
                "Ligerito default config; bump m or use prove_fast (BaseFold) for tiny instances",
            );

    // --- PCS commit ---
    let t0 = Instant::now();
    let (commitment, prover_data) = match prefaulted_codeword {
        Some(buf) => pcs::commit_into(&z_packed, pcs_params, buf),
        None => pcs::commit(&z_packed, pcs_params),
    };
    t.commit_s = t0.elapsed().as_secs_f64();
    bind_statement(challenger, r1cs, &commitment);

    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };

    // --- zerocheck ---
    let t0 = Instant::now();
    let (zc_proof, zc_claim, s_hat_v_c) = {
        let a_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                a_packed_f128.as_ptr() as *const u8,
                a_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let b_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                b_packed_f128.as_ptr() as *const u8,
                b_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let c_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                z_packed.as_ptr() as *const u8,
                z_packed.len() * core::mem::size_of::<F128>(),
            )
        };
        zerocheck::prove_packed_padded_capture_s_hat_v_c(
            a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
        )
    };
    t.zerocheck_s = t0.elapsed().as_secs_f64();
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);

    let inner_rest_len = r1cs.k_log - r1cs.k_skip;
    let x_ab = QuirkyPoint {
        z_skip: zc_claim.z,
        x_inner_rest: zc_claim.mlv_challenges[..inner_rest_len].to_vec(),
        x_outer: zc_claim.mlv_challenges[inner_rest_len..].to_vec(),
    };

    // --- lincheck + base-claim / s_hat_v setup ---
    let t0 = Instant::now();
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        lincheck_circuit,
        &x_ab,
        challenger,
    );
    drop(z_packed_lincheck);
    let ab = ZClaim {
        point: QuirkyPoint {
            z_skip: lc_claim.r_inner_skip,
            x_inner_rest: lc_claim.r_inner_rest.clone(),
            x_outer: x_ab.x_outer.clone(),
        },
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: QuirkyPoint {
            z_skip: zc_claim.z,
            x_inner_rest: zc_claim.r_rest[..inner_rest_len].to_vec(),
            x_outer: zc_claim.r_rest[inner_rest_len..].to_vec(),
        },
        value: zc_claim.c_eval,
    };
    let s_hat_v_ab = if r1cs.k_log >= pcs::LOG_PACKING {
        Some(pcs::ring_switch::s_hat_v_from_z_vec(
            &z_vec_pre,
            &lc_claim.r_inner_rest[1..],
        ))
    } else {
        None
    };
    t.lincheck_s = t0.elapsed().as_secs_f64();

    // --- Ligerito recursive PCS open ---
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let t0 = Instant::now();
    let pcs_open = open_claims_with_precomputed_ligerito(
        z_packed,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &lig_config,
        challenger,
    );
    t.open_s = t0.elapsed().as_secs_f64();

    let proof = R1csProofLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim, t)
}
