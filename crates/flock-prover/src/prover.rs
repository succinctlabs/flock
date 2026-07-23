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
use flock_core::proof::{
    R1csClaim, R1csProofJaggedLigerito, R1csProofLigerito, ZClaim, bind_statement,
};
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

/// Batched PCS open over an arbitrary list of `ẑ`-evaluation claims. This is
/// the generic seam: the base R1CS proof opens `[ab, c]`; relation wrappers
/// (e.g. the hash chain) append their own claims and open `[ab, c, …]`.
/// Per-claim optional precomputed `s_hat_v` is passed through to ring-switch:
/// when `Some(v)`, the claim skips `fold_1b_rows` and uses `v` directly.
/// Caller responsibility: each `Some(v)` MUST equal what `fold_1b_rows` would
/// produce on `z_packed` against the claim's suffix — see
/// [`pcs::ring_switch::s_hat_v_from_z_vec`] for the AB-claim derivation.
///
/// Must be called at the same transcript position as the verifier's
/// [`flock_core::verifier::verify_claims_ligerito`].
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
pub fn prove_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    z_packed: Vec<F128>,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    assert_eq!(
        r1cs.layout,
        flock_core::r1cs::WitnessLayout::RowMajor,
        "the generic matrix-driven provers assume the row-major layout \
         (block-diagonal apply + lincheck stripe packing); batch-major \
         setups must use the per-hash prove_fast paths"
    );
    assert_eq!(z_packed.len(), 1usize << (r1cs.m - 7));
    assert_eq!(pcs_params.m, r1cs.m);

    let log_n = r1cs.m - pcs::LOG_PACKING;
    let lig_config =
        pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
            .expect("Ligerito default config; bump m for tiny instances");

    let (commitment, prover_data) = pcs::commit(&z_packed, pcs_params);
    bind_statement(challenger, r1cs, &commitment);

    // a = A·z, b = B·z; for the C = I convention c aliases z.
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

    let padding = r1cs.padding_spec();
    let (zc_proof, zc_claim, s_hat_v_c) = zerocheck::prove_packed_padded_capture_s_hat_v_c(
        a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
    );

    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

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
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
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
/// byte-for-byte). Used by per-hash modules' `prove_fast_ligerito` methods.
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
            .expect("Ligerito default config; bump m for tiny instances");

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

    let padding = r1cs.padding_spec();
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

/// Jagged-path counterpart of [`open_claims_with_precomputed_ligerito`]:
/// batched PCS open over `ẑ`-claims routed through the virtual-opening
/// sumcheck + jagged transport (`pcs::open_batch_jagged_ligerito`).
/// `heights` / `n_log` describe the committed jagged grid (see
/// [`flock_core::r1cs::BlockR1cs::jagged_heights`]); `dense_witness` is the
/// committed dense stack `q` when it differs from the padded buffer (the M4
/// dense-stack commit — `UnionInstance::compact_witness`), `None` when the
/// compaction map is the identity. Must be called at the same transcript
/// position as the verifier's
/// [`flock_core::verifier::verify_claims_jagged_ligerito`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn open_claims_with_precomputed_jagged_ligerito<Ch: Challenger>(
    z_packed: Vec<F128>,
    dense_witness: Option<Vec<F128>>,
    prover_data: &pcs::ProverData,
    commitment: &Commitment,
    claims: &[ZClaim],
    precomputed_s_hat_v: &[Option<&[F128]>],
    padding: &zerocheck::PaddingSpec,
    heights: &[u64],
    n_log: usize,
    lig_config: &pcs::ligerito::ProverConfig,
    challenger: &mut Ch,
) -> pcs::BatchOpeningProofJaggedLigerito {
    let x_fulls: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| quirky_x_outer_full(&c.point))
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    pcs::open_batch_jagged_ligerito(
        z_packed,
        dense_witness,
        prover_data,
        commitment,
        &x_refs,
        precomputed_s_hat_v,
        &[],
        padding,
        heights,
        n_log,
        lig_config,
        challenger,
    )
}

/// [`prove_fast_ligerito_from_witness`] with the opening routed through the
/// **jagged transport** (Phase 1 of the multi-table design): identical
/// commit → zerocheck → lincheck pipeline ([`prove_fast_core_with_codeword`],
/// so the PIOP transcript prefix is byte-identical to the direct path on the
/// same statement + witness), then `pcs::open_batch_jagged_ligerito` instead
/// of the mixed Ligerito open. Requires the BatchMajor witness layout — the
/// jagged grid's columns are the buffer's chunk-columns. Verify with
/// [`flock_core::verifier::verify_ligerito_jagged`].
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_jagged_from_witness<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofJaggedLigerito, Commitment, R1csClaim) {
    assert_eq!(
        r1cs.layout,
        flock_core::r1cs::WitnessLayout::BatchMajor,
        "the jagged opening path requires the BatchMajor witness layout"
    );
    let log_n = r1cs.m - pcs::LOG_PACKING;
    let lig_config =
        pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
            .expect("Ligerito default config; bump m for tiny instances");

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

    let padding = r1cs.padding_spec();
    let heights = r1cs.jagged_heights();
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = open_claims_with_precomputed_jagged_ligerito(
        z_packed,
        None,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &heights,
        r1cs.n_log(),
        &lig_config,
        challenger,
    );

    let proof = R1csProofJaggedLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim)
}

/// One slot's prover inputs for the union prove entry: the packed witness
/// bundle plus the slot's lincheck stripe and circuit. One per registry
/// type, in slot order.
pub struct UnionSlotProverInput<'a> {
    /// The slot's `(z, a, b)` packed buffers (see
    /// [`flock_core::union::SlotWitness`]).
    pub witness: flock_core::union::SlotWitness,
    /// The slot's lincheck stripe copy of `z` (the drivers' fourth output).
    pub z_lincheck: Vec<u8>,
    /// The slot's lincheck circuit (e.g. `BlockR1cs::csc_lincheck_circuit`).
    pub lincheck_circuit: &'a dyn lincheck::LincheckCircuit,
}

impl<'a> UnionSlotProverInput<'a> {
    /// Wrap one slot's driver output — the `(z, a, b, stripe)` tuple of the
    /// existing batch-major witness generators (e.g.
    /// `blake3::generate_witness_batch_major`) — plus its lincheck circuit.
    pub fn new(
        (z_packed, a_packed, b_packed, z_lincheck): (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
        lincheck_circuit: &'a dyn lincheck::LincheckCircuit,
    ) -> Self {
        Self {
            witness: flock_core::union::SlotWitness {
                z_packed,
                a_packed,
                b_packed,
            },
            z_lincheck,
            lincheck_circuit,
        }
    }
}

/// Statement-binding selector for the union prove path. Private: the two
/// public entries below fix the variant.
enum UnionProveBinding<'a> {
    /// The protocol binding: `flock-mixed-v1` over the registry digest, the
    /// counts vector, and the commitment root
    /// ([`flock_core::union::UnionInstance::bind_statement`]).
    Mixed,
    /// The M1/M2 differential-harness binding: the slot's single-table
    /// `BlockR1cs` statement digest, transcript-identical to the direct
    /// jagged path. Single-type registries only; not a protocol mode.
    SingleTypeHarness(&'a BlockR1cs),
}

/// Prove a registry instance through the **union address space** — Phase 2
/// of the multi-table design, since M3 under the real multi-table statement
/// binding: assemble the per-slot witnesses into the union buffers, bind
/// the statement as `flock-mixed-v1` (registry digest + counts vector +
/// commitment root, [`flock_core::union::UnionInstance::bind_statement`]),
/// and drive the EXISTING jagged path with the
/// [`flock_core::union::UnionInstance`]-derived quantities (count-derived
/// run-list padding, union jagged heights, `n_log = nu`, union claim
/// points) and the union-column lincheck. Verify with
/// [`flock_core::verifier::verify_ligerito_jagged_union`].
///
/// `slots` are one per registry type, **in slot order** — the registry's
/// order, i.e. sorted by capacity area descending (under uniform capacity:
/// by `k_log` descending; e.g. SHA-256 (κ = 15) before BLAKE3 (κ = 14)).
/// Mis-ordered inputs cannot produce a proof: the witness-assembly and
/// lincheck layers assert each slot's buffer sizes and circuit shape
/// against the registry type.
///
/// A single-type instance proved here roundtrips with the union verifier
/// but is deliberately **not** byte-identical to
/// [`prove_fast_ligerito_jagged_from_witness`] — the statement bindings are
/// domain-separated. The byte-identity regression anchor is
/// [`prove_fast_ligerito_jagged_union_harness`].
///
/// Witness contract: rows `[n_t, 2^nu)` of each slot must be identically
/// zero — the run-list padding lets the kernels skip them (only sound, and
/// only byte-identical to the dense computation, for honest zeros), the
/// dense-stack transport commits them at capacity height, and the union
/// lincheck's count-derived const-pin target requires the pin at 0 on every
/// dummy row. Use the per-hash `generate_witness_batch_major_partial`
/// drivers (M4), which honor any `n_t ≤ 2^nu` and zero the remainder; the
/// full-utilization `generate_witness_batch_major` drivers instead fill
/// padding rows with real dummy invocations (pin = 1) and are only valid
/// here at `n_t = 2^nu`.
pub fn prove_fast_ligerito_jagged_union<Ch: Challenger>(
    union: &flock_core::union::UnionInstance<'_>,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofJaggedLigerito, Commitment, R1csClaim) {
    prove_union_with_binding(
        union,
        UnionProveBinding::Mixed,
        pcs_params,
        slots,
        challenger,
    )
}

/// [`prove_fast_ligerito_jagged_union`] under the M1/M2 **harness** binding
/// (the slot's single-table `BlockR1cs` statement digest): on a single-type
/// registry at full utilization, the proof is **byte-identical** to
/// [`prove_fast_ligerito_jagged_from_witness`] on the same statement +
/// witness — the differential oracle in `tests/union_roundtrip.rs`, kept as
/// the regression anchor for the union plumbing. Verify with
/// [`flock_core::verifier::verify_ligerito_jagged_union_harness`].
/// Test/differential harness only — not a protocol mode.
pub fn prove_fast_ligerito_jagged_union_harness<Ch: Challenger>(
    union: &flock_core::union::UnionInstance<'_>,
    slot_r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofJaggedLigerito, Commitment, R1csClaim) {
    prove_union_with_binding(
        union,
        UnionProveBinding::SingleTypeHarness(slot_r1cs),
        pcs_params,
        slots,
        challenger,
    )
}

/// Shared body of the two union prove entries; `binding` selects the
/// statement binding, everything else is identical.
fn prove_union_with_binding<Ch: Challenger>(
    union: &flock_core::union::UnionInstance<'_>,
    binding: UnionProveBinding<'_>,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofJaggedLigerito, Commitment, R1csClaim) {
    // Harness guard + slot statement consistency (also asserts one type) —
    // before doing anything heavy.
    if let UnionProveBinding::SingleTypeHarness(slot_r1cs) = binding {
        union.expect_single_type_slot(slot_r1cs);
    }
    let m = union.m_total();
    // The commitment is to the DENSE stack q (M4): PcsParams.m is the dense
    // variable count; the PIOP and the virtual-opening sumcheck keep the
    // M-variable padded address space.
    assert_eq!(
        pcs_params.m,
        union.dense_m(),
        "PcsParams.m must equal the union's dense_m (committed stack size)"
    );
    assert_eq!(
        slots.len(),
        union.registry().num_types(),
        "need one prover input per registry type"
    );

    let log_n = union.dense_m() - pcs::LOG_PACKING;
    let lig_config =
        pcs::ligerito::prover_config_for(log_n, pcs_params.log_batch_size, pcs_params.profile)
            .expect("Ligerito default config; bump m for tiny instances");

    // Union witness assembly (single slot: zero-copy passthrough), keeping
    // the per-slot lincheck inputs aside.
    let mut witnesses = Vec::with_capacity(slots.len());
    let mut linchecks = Vec::with_capacity(slots.len());
    for slot in slots {
        witnesses.push(slot.witness);
        linchecks.push((slot.z_lincheck, slot.lincheck_circuit));
    }
    let (z_packed, a_packed_f128, b_packed_f128) = union.assemble_witness(witnesses);

    // True dense-stack commit: commit the compacted stack q (used
    // chunk-columns at capacity height, useless columns and gaps dropped,
    // padded to a power of two). When the compaction map is the identity
    // (single-slot registries — the byte-identity anchors), q IS the padded
    // buffer and no copy is made.
    let dense_q: Option<Vec<F128>> = if union.compaction_is_identity() {
        None
    } else {
        Some(union.compact_witness(&z_packed))
    };
    let (commitment, prover_data) = match &dense_q {
        Some(q) => pcs::commit(q, pcs_params),
        None => pcs::commit(&z_packed, pcs_params),
    };
    match binding {
        UnionProveBinding::Mixed => union.bind_statement(challenger, &commitment),
        UnionProveBinding::SingleTypeHarness(slot_r1cs) => {
            union.bind_statement_single_type(challenger, slot_r1cs, &commitment)
        }
    }

    // Zerocheck over the union address space, driven by the count-derived
    // run-list (the existing kernels' general multi-run paths — value-
    // identical to the single-run spec on honestly-zero padding).
    let padding = union.padding_spec();
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
            a_packed, b_packed, c_packed, m, &padding, challenger,
        )
    };
    // a/b are consumed; recycle the buffers as in `prove_fast_core`.
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);

    let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    // M2: the union-column lincheck — one sumcheck over the union column
    // domain against the per-slot stripes and circuits. On the M1
    // single-type registries it is byte-identical to invoking the slot's
    // own lincheck (the union of one slot has m = M).
    let (lc_proof, lc_claim, z_vec_pre) = {
        let lc_slots: Vec<lincheck::UnionLincheckSlot<'_>> = linchecks
            .iter()
            .map(|(stripe, circuit)| lincheck::UnionLincheckSlot {
                z_lincheck: stripe,
                circuit: *circuit,
            })
            .collect();
        lincheck::prove_union_capture_z_vec(union, &lc_slots, &x_ab, challenger)
    };
    drop(linchecks);

    let ab = ZClaim {
        point: union.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: union.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    // `s_hat_v_from_z_vec` needs `z_vec.len() = 2^LOG_PACKING · 2^tail`;
    // the union fold has `len = 2^(M−ν)` and `tail = M−ν−LOG_PACKING`, so
    // the condition is `M−ν ≥ LOG_PACKING` — for a single-type registry
    // exactly the old `k_log ≥ LOG_PACKING`, and always true for real
    // registries (every `k_log ≥ 7`).
    let s_hat_v_ab = if m - union.n_log() >= pcs::LOG_PACKING {
        Some(pcs::ring_switch::s_hat_v_from_z_vec(
            &z_vec_pre,
            &lc_claim.r_inner_rest[1..],
        ))
    } else {
        None
    };

    let heights = union.jagged_heights();
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = open_claims_with_precomputed_jagged_ligerito(
        z_packed,
        dense_q,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &heights,
        union.n_log(),
        &lig_config,
        challenger,
    );

    let proof = R1csProofJaggedLigerito {
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
/// The generic seam: `prove_fast_ligerito_from_witness` = `prove_fast_core` +
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

    let padding = r1cs.padding_spec();
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

    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

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
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
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
/// recursive opening. Kept in lockstep
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
            .expect("Ligerito default config; bump m for tiny instances");

    // --- PCS commit ---
    let t0 = Instant::now();
    let (commitment, prover_data) = match prefaulted_codeword {
        Some(buf) => pcs::commit_into(&z_packed, pcs_params, buf),
        None => pcs::commit(&z_packed, pcs_params),
    };
    t.commit_s = t0.elapsed().as_secs_f64();
    bind_statement(challenger, r1cs, &commitment);

    let padding = r1cs.padding_spec();

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

    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

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
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
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
