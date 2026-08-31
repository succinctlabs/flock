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

use core::mem::size_of;
use flock_core::circuit::Circuit;
use flock_core::circuit::WiringProof;
use flock_core::circuit::prove_wiring_with_grinding;
use flock_core::element_r1cs::ElementTableType;
use flock_core::element_r1cs::union::Claims;
use flock_core::element_r1cs::union::Proof;
use flock_core::element_r1cs::union::copy_live_region;
use flock_core::element_r1cs::union::dead_rows_unread;
use flock_core::element_r1cs::union::fill_slot;
use flock_core::element_r1cs::union::give_back_live_region;
use flock_core::element_r1cs::union::prove_with_grinding;
use flock_core::lincheck::SkipPoint;
use flock_core::lincheck::{self, QuirkyPoint, pack_z_lincheck_from_packed};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::BooleanPiopProof;
use flock_core::proof::BooleanPiopProofAg;
use flock_core::proof::R1csProofCircuitMerged;
use flock_core::proof::R1csProofCircuitMergedAg;
#[cfg(target_arch = "aarch64")]
use flock_core::proof::R1csProofLigeritoAg;
use flock_core::proof::R1csProofMergedLigerito;
use flock_core::proof::R1csProofMergedLigeritoAg;
use flock_core::proof::R1csProofMixedClassMerged;
use flock_core::proof::UnionClassClaims;
use flock_core::proof::{R1csClaim, R1csProofLigerito, ZClaim, bind_statement};
use flock_core::r1cs::{BlockR1cs, WitnessLayout};
use flock_core::schedule::TableClass;
use flock_core::scratch::give_f128;
use flock_core::scratch::give_u8;
use flock_core::union::SlotWitness;
use flock_core::union::SlotWitnessDest;
use flock_core::union::UnionInstance;
use flock_core::union::WitnessBufMode;
use flock_core::zerocheck;
use flock_core::zerocheck::ag_skip::AgProof;
use flock_field::F128;
use flock_transcript::challenger::Challenger;
use lincheck::LincheckCircuit;
use lincheck::LincheckProof;
use lincheck::SparseMatrixCircuit;
use lincheck::UnionLincheckSlot;
use lincheck::prove_padded_capture_z_vec;
use lincheck::prove_padded_capture_z_vec_with_grinding;
use lincheck::prove_union_capture_z_vec_with_grinding;
use pcs::BatchOpeningProofLigerito;
use pcs::DirectEqInd;
use pcs::LOG_PACKING;
use pcs::MergedOpenProof;
use pcs::OpeningGrinding;
use pcs::PackedDirectClaim;
use pcs::ProverData;
use pcs::commit;
use pcs::commit_into;
use pcs::commit_lane_major;
use pcs::ligerito::ProverConfig;
use pcs::open_batch_merged;
use pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding;
use pcs::ring_switch::s_hat_v_from_z_vec;
use rayon::join;
use std::env::var;
use std::mem::size_of_val;
use std::slice::from_raw_parts;
use std::sync::Arc;
use std::time::Instant;
use zerocheck::PaddingSpec;
use zerocheck::ZerocheckProof;
use zerocheck::ag_skip::K_SKIP;
use zerocheck::ag_skip::prove_capture_s_hat_v_c;
use zerocheck::ag_skip::prove_capture_s_hat_v_c_with_grinding;
use zerocheck::prove_packed_padded_capture_s_hat_v_c_with_grinding;

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
    prover_data: &ProverData,
    commitment: &Commitment,
    claims: &[ZClaim],
    precomputed_s_hat_v: &[Option<&[F128]>],
    padding: &PaddingSpec,
    lig_config: &ProverConfig,
    opening_grinding: OpeningGrinding,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    let x_fulls: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| quirky_x_outer_full(&c.point))
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
        z_packed,
        prover_data,
        commitment,
        &x_refs,
        precomputed_s_hat_v,
        &[],
        padding,
        lig_config,
        opening_grinding,
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
        WitnessLayout::RowMajor,
        "the generic matrix-driven provers assume the row-major layout \
         (block-diagonal apply + lincheck stripe packing); batch-major \
         setups must use the per-hash prove_fast paths"
    );
    assert_eq!(z_packed.len(), 1usize << (r1cs.m - 7));
    assert_eq!(pcs_params.m, r1cs.m);

    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config; bump m for tiny instances");

    let (commitment, prover_data) = commit(&z_packed, pcs_params);
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
        unsafe { from_raw_parts(v.as_ptr() as *const u8, size_of_val(v)) }
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
    let (zc_proof, zc_claim, s_hat_v_c) = prove_packed_padded_capture_s_hat_v_c_with_grinding(
        a_packed,
        b_packed,
        c_packed,
        r1cs.m,
        &padding,
        pcs_params.zerocheck_grinding(),
        challenger,
    );

    let x_ab = r1cs.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);

    let lc_circuit = SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
    let (lc_proof, lc_claim, z_vec_pre) = prove_padded_capture_z_vec_with_grinding(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        &lc_circuit,
        &x_ab,
        pcs_params.lincheck_grinding(),
        challenger,
    );

    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(SkipPoint::Phi8(zc_claim.z), &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    let s_hat_v_ab = if r1cs.k_log >= LOG_PACKING {
        Some(s_hat_v_from_z_vec(&z_vec_pre, &lc_claim.r_inner_rest[1..]))
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
        pcs_params.opening_grinding(),
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
    lincheck_circuit: &dyn LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    let lig_config = pcs_params
        .ligerito_prover_config()
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
        pcs_params.opening_grinding(),
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

/// One slot's prover inputs for the union prove entry: where the slot's
/// packed witness comes from, plus its lincheck circuit. One per registry
/// type, in slot order.
pub struct UnionSlotProverInput<'a> {
    source: UnionSlotWitnessSource<'a>,
    /// The slot's lincheck circuit (e.g. `BlockR1cs::csc_lincheck_circuit`).
    pub lincheck_circuit: &'a dyn LincheckCircuit,
}

/// How a slot's packed witness reaches the padded union buffers.
enum UnionSlotWitnessSource<'a> {
    /// Already generated into the slot's own buffers — the union assembly
    /// COPIES them to the slot's aligned block.
    Prebuilt {
        witness: SlotWitness,
        z_lincheck: Vec<u8>,
    },
    /// Generated in place: the closure is handed the slot's block of the
    /// union buffers and writes it directly, returning the lincheck stripe.
    /// No copy — see [`flock_core::union::SlotWitnessDest`].
    InPlace(Box<dyn FnOnce(SlotWitnessDest<'_>) -> Vec<u8> + Send + 'a>),
    /// An ELEMENT slot: the closure writes the slot's committed element words
    /// into the `z` view and `element_r1cs::union::fill_slot` derives `a`/`b`
    /// from them by sparse gather. There is no lincheck stripe — the element
    /// lincheck folds the committed region itself.
    Element {
        ty: Arc<ElementTableType>,
        generate: Box<dyn FnOnce(&mut [F128]) + Send + 'a>,
    },
}

/// One ELEMENT slot's prover input: a closure that writes the slot's committed
/// element words. One per registry element type, in slot order.
///
/// **Contract** (same as [`flock_core::union::SlotWitnessDest`]): the closure
/// must write EVERY word of its `2^{ν+κ}`-word block — real rows from the
/// generator, dummy rows `[n_t, 2^ν)` and padding columns as zeros. The block
/// comes from the recycled scratch pool and starts out holding stale data. The
/// element PIOP sums over the WHOLE region, so a stale word is not merely
/// uncommitted, it would break the zerocheck.
pub struct UnionElementSlotInput<'a> {
    generate: Box<dyn FnOnce(&mut [F128]) + Send + 'a>,
}

impl<'a> UnionElementSlotInput<'a> {
    /// `generate` receives the slot's `2^{ν+κ}`-word block in the BatchMajor,
    /// rows-low layout the element class fixes: word `(c << ν) + row` is
    /// (column `c`, row `row`).
    pub fn new(generate: impl FnOnce(&mut [F128]) + Send + 'a) -> Self {
        Self {
            generate: Box::new(generate),
        }
    }
}

impl<'a> UnionSlotProverInput<'a> {
    /// Wrap one slot's driver output — the `(z, a, b, stripe)` tuple of the
    /// existing batch-major witness generators (e.g.
    /// `blake3::generate_witness_batch_major`) — plus its lincheck circuit.
    ///
    /// The witness is copied into the union buffers. Prefer
    /// [`Self::in_place`] on the hot path: at `M = 30` the scatter this
    /// incurs is ~10 ms of pure memory traffic.
    pub fn new(
        (z_packed, a_packed, b_packed, z_lincheck): (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
        lincheck_circuit: &'a dyn LincheckCircuit,
    ) -> Self {
        Self {
            source: UnionSlotWitnessSource::Prebuilt {
                witness: SlotWitness {
                    z_packed,
                    a_packed,
                    b_packed,
                },
                z_lincheck,
            },
            lincheck_circuit,
        }
    }

    /// Generate this slot's witness DIRECTLY into the union buffers — the
    /// copy-free assembly path. `generate` receives the slot's aligned
    /// `2^{m_t−7}`-word block of `z`, `a`, `b` and must write every word of
    /// it (the `*_into` drivers do), returning the lincheck stripe:
    ///
    /// ```ignore
    /// UnionSlotProverInput::in_place(
    ///     |dst| blake3::generate_witness_batch_major_partial_into(blocks, nu, dst),
    ///     circuit,
    /// )
    /// ```
    ///
    /// Produces the same padded buffers as [`Self::new`] on the same witness
    /// — a slot's BatchMajor layout IS its aligned union sub-block — so the
    /// proof is byte-identical, only the copy is gone.
    pub fn in_place(
        generate: impl FnOnce(SlotWitnessDest<'_>) -> Vec<u8> + Send + 'a,
        lincheck_circuit: &'a dyn LincheckCircuit,
    ) -> Self {
        Self {
            source: UnionSlotWitnessSource::InPlace(Box::new(generate)),
            lincheck_circuit,
        }
    }
}

/// Build the padded union witness buffers from the per-slot sources,
/// returning them with each slot's lincheck stripe (in slot order).
///
/// All-prebuilt input takes the existing [`flock_core::union::UnionInstance::
/// assemble_witness`] path (single-slot passthrough included). Otherwise the
/// buffers are allocated once and each slot is materialized into its own
/// aligned block — generated there directly for in-place slots, copied there
/// for prebuilt ones. Either way the result is the same padded buffer.
///
/// Element slots always take the in-place path (their `z` block is generated
/// there and `a`/`b` derived from it by sparse gather), so the all-prebuilt
/// fast path below only ever sees a boolean-only registry. The returned stripe
/// vector has one entry per slot, EMPTY for element slots — the element
/// lincheck folds the committed region directly and has no stripe.
fn build_union_witness(
    union: &UnionInstance<'_>,
    sources: Vec<UnionSlotWitnessSource<'_>>,
    padding_unread: bool,
) -> (
    Vec<F128>,
    Vec<F128>,
    Vec<F128>,
    Vec<Vec<u8>>,
    WitnessBufMode,
) {
    assert_eq!(
        sources.len(),
        union.registry().num_types(),
        "need one prover input per registry type"
    );
    if sources
        .iter()
        .all(|s| matches!(s, UnionSlotWitnessSource::Prebuilt { .. }))
    {
        let mut witnesses = Vec::with_capacity(sources.len());
        let mut stripes = Vec::with_capacity(sources.len());
        for s in sources {
            match s {
                UnionSlotWitnessSource::Prebuilt {
                    witness,
                    z_lincheck,
                } => {
                    witnesses.push(witness);
                    stripes.push(z_lincheck);
                }
                _ => unreachable!("checked above"),
            }
        }
        let (z, a, b) = union.assemble_witness(witnesses);
        return (z, a, b, stripes, WitnessBufMode::PooledZeroed);
    }

    let (mut z, mut a, mut b, mode) = union.take_witness_buffers(padding_unread);
    let elide = mode != WitnessBufMode::PooledZeroed;
    let nu = union.n_log();
    // Live-only element `pa`/`pb` derivation: when the region zerocheck will
    // take its sparse arm, dead rows of `a`/`b` are unread everywhere (their
    // values are substituted analytically), so the gather skips them — the
    // same pay-per-live discipline the boolean side's run-lists apply. Gated
    // by the zerocheck's OWN predicate so the two cannot drift.
    let elem_live = union.has_element() && dead_rows_unread(union);
    let stripes = union
        .slot_dests(&mut z, &mut a, &mut b, elide)
        .into_iter()
        .zip(sources)
        .enumerate()
        .map(|(i, (dst, source))| match source {
            UnionSlotWitnessSource::InPlace(generate) => generate(dst),
            UnionSlotWitnessSource::Prebuilt {
                witness,
                z_lincheck,
            } => {
                dst.z.copy_from_slice(&witness.z_packed);
                dst.a.copy_from_slice(&witness.a_packed);
                dst.b.copy_from_slice(&witness.b_packed);
                z_lincheck
            }
            UnionSlotWitnessSource::Element { ty, generate } => {
                let live = elem_live.then(|| union.counts()[i]);
                fill_slot(&ty, nu, live, dst.z, dst.a, dst.b, generate);
                Vec::new()
            }
        })
        .collect();
    (z, a, b, stripes, mode)
}

/// Statement-binding selector for the union prove path. Private: the public
/// entries below fix the variant.
enum UnionProveBinding<'a> {
    /// The protocol binding: `flock-mixed-v1` over the registry digest, the
    /// counts vector, and the commitment root
    /// ([`flock_core::union::UnionInstance::bind_statement`]).
    Mixed,
    /// The circuit binding: [`UnionProveBinding::Mixed`] plus the circuit
    /// digest and the public words, and the wiring GKR after the class PIOPs.
    Circuit(CircuitProverInput<'a>),
}

/// A circuit's prover input: the circuit (whose gate counts must be the
/// union's declared counts) and its public words, in public-segment order.
#[derive(Clone, Copy)]
struct CircuitProverInput<'a> {
    circuit: &'a Circuit,
    public: &'a [F128],
}

/// The **circuit** prove entry over the MERGED transport — the production
/// shape (and, since the jagged transport's removal, the only one): the
/// class PIOPs, the wiring argument, and one merged opening carrying the
/// class claims AND the wiring's gather claims (packed-direct, the same
/// intake the element class's claims use).
///
/// Fiat–Shamir order: commit → `bind_statement_circuit` (statement +
/// circuit digest + public words) → boolean τ/ZC/LC → element τ'/ZC/α'/LC →
/// wiring GKR (α, β at its entry) → gather values observed → γ-batched
/// opening.
///
/// Verify with [`flock_core::verifier::verify_ligerito_union_circuit`].
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_union_circuit<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuit: &Circuit,
    public: &[F128],
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    element_slots: Vec<UnionElementSlotInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofCircuitMerged, Commitment, UnionClassClaims) {
    assert!(
        circuit.check_instance(union),
        "the circuit and the union instance must be the same statement \
         (same registry, and the circuit's gate counts ARE the union's counts)"
    );
    assert!(
        circuit.check_public(public),
        "the public segment must have the circuit's declared length and fixed constants"
    );
    let (out, commitment) = prove_union_with_binding(
        union,
        UnionProveBinding::Circuit(CircuitProverInput { circuit, public }),
        pcs_params,
        slots,
        element_slots,
        challenger,
    );
    let UnionProveOutput {
        boolean,
        element,
        wiring,
        pcs_open,
    } = out;
    let (bool_proof, bool_claim) = match boolean {
        Some((p, c)) => (Some(p.expect_rs()), Some(c)),
        None => (None, None),
    };
    let (el_proof, el_claim) = match element {
        Some((p, c)) => (Some(p), Some(c)),
        None => (None, None),
    };
    (
        R1csProofCircuitMerged {
            boolean: bool_proof,
            element: el_proof,
            wiring: wiring.expect("the circuit binding runs the wiring argument"),
            pcs_open,
        },
        commitment,
        UnionClassClaims {
            boolean: bool_claim,
            element: el_claim,
        },
    )
}

/// Proves a Boolean-only registry with the merged opening transport.
///
/// This wraps [`prove_union_with_binding`] and packages the Boolean subproofs as
/// [`flock_core::proof::R1csProofMergedLigerito`].
///
/// Witness contract: rows `[n_t, 2^nu)` of each slot must be identically
/// zero — the count-derived run-list padding lets the kernels skip them
/// (only sound, and only byte-identical to the dense computation, for
/// honest zeros), the height-`n_t` dense-stack transport DROPS them from
/// the committed stack, and the union lincheck's count-derived const-pin
/// target requires the pin at 0 on every dummy row. Use the per-hash
/// `generate_witness_batch_major_partial` drivers, which honor any
/// `n_t ≤ 2^nu` and zero the remainder; the full-utilization
/// `generate_witness_batch_major` drivers fill padding rows with real dummy
/// invocations (pin = 1) and are only valid here at `n_t = 2^nu`.
pub fn prove_fast_ligerito_union<Ch: Challenger>(
    union: &UnionInstance<'_>,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofMergedLigerito, Commitment, R1csClaim) {
    // This entry returns `R1csClaim` — structurally boolean-only. Element
    // registries go through the mixed-class entry, whose merged open carries
    // their claims packed-direct.
    assert!(
        !union.has_element(),
        "this entry is boolean-only; element registries go through \
         prove_fast_ligerito_union_mixed_class"
    );
    let (out, commitment) = prove_union_with_binding(
        union,
        UnionProveBinding::Mixed,
        pcs_params,
        slots,
        Vec::new(),
        challenger,
    );
    let (piop, claim) = out.boolean.expect("asserted boolean-only above");
    let piop = piop.expect_rs();
    (
        R1csProofMergedLigerito {
            zerocheck: piop.zerocheck,
            lincheck: piop.lincheck,
            pcs_open: out.pcs_open,
        },
        commitment,
        claim,
    )
}

/// Which zerocheck the boolean class runs inside
/// [`prove_union_with_binding_zc`]. The lincheck, claims, and opening are
/// flavor-generic ([`SkipPoint`] carries the difference).
#[derive(Clone, Copy, PartialEq, Eq)]
enum BooleanZcKind {
    /// RS additive-NTT univariate skip (the default).
    Rs,
    /// Genus-95 AG multiplication-code skip. aarch64-only (NEON SLP
    /// kernel); run-list read-exact like the RS flavor — see the padding
    /// contract on `flock_core::proof::BooleanPiopProofAg`.
    #[cfg(target_arch = "aarch64")]
    Ag,
}

/// [`prove_fast_ligerito_union_circuit`] with the **AG-skip** boolean
/// zerocheck — verify with
/// [`flock_core::verifier::verify_ligerito_union_circuit_ag`] (or the
/// `_deferred` twin). Same class PIOP order, wiring argument, and merged
/// opening; only the boolean zerocheck's round 1 differs. The element class
/// may be present (its PIOP is flavor-independent). Run-list read-exact —
/// the padding contract on [`flock_core::proof::BooleanPiopProofAg`] is
/// the owning statement. aarch64-only (the AG round-1 kernel is NEON).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_union_circuit_ag<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuit: &Circuit,
    public: &[F128],
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    element_slots: Vec<UnionElementSlotInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofCircuitMergedAg, Commitment, UnionClassClaims) {
    assert!(
        circuit.check_instance(union),
        "the circuit and the union instance must be the same statement \
         (same registry, and the circuit's gate counts ARE the union's counts)"
    );
    assert!(
        circuit.check_public(public),
        "the public segment must have the circuit's declared length and fixed constants"
    );
    let (out, commitment) = prove_union_with_binding_zc(
        union,
        UnionProveBinding::Circuit(CircuitProverInput { circuit, public }),
        BooleanZcKind::Ag,
        pcs_params,
        slots,
        element_slots,
        challenger,
    );
    let UnionProveOutput {
        boolean,
        element,
        wiring,
        pcs_open,
    } = out;
    let (bool_proof, bool_claim) = match boolean {
        Some((p, c)) => {
            let UnionBooleanProof::Ag(p) = p else {
                unreachable!("the Ag flavor produces an Ag boolean proof")
            };
            (Some(p), Some(c))
        }
        None => (None, None),
    };
    let (el_proof, el_claim) = match element {
        Some((p, c)) => (Some(p), Some(c)),
        None => (None, None),
    };
    (
        R1csProofCircuitMergedAg {
            boolean: bool_proof,
            element: el_proof,
            wiring: wiring.expect("the circuit binding runs the wiring argument"),
            pcs_open,
        },
        commitment,
        UnionClassClaims {
            boolean: bool_claim,
            element: el_claim,
        },
    )
}

/// The zerocheck transcript alone, before the lincheck joins it in the
/// closure's assembly step.
enum UnionZcProof {
    Rs(ZerocheckProof),
    #[cfg(target_arch = "aarch64")]
    Ag(AgProof),
}

/// The boolean sub-proof [`prove_union_with_binding_zc`] hands back — the
/// prove-side counterpart of the verifier's `BooleanPiopRef`.
enum UnionBooleanProof {
    Rs(BooleanPiopProof),
    #[cfg(target_arch = "aarch64")]
    Ag(BooleanPiopProofAg),
}

impl UnionBooleanProof {
    fn expect_rs(self) -> BooleanPiopProof {
        match self {
            UnionBooleanProof::Rs(p) => p,
            #[cfg(target_arch = "aarch64")]
            UnionBooleanProof::Ag(_) => {
                unreachable!("an RS-flavor entry received an AG boolean proof")
            }
        }
    }
}

/// [`prove_fast_ligerito_union`] with the **AG-skip** boolean zerocheck —
/// verify with [`flock_core::verifier::verify_ligerito_union_ag`]. Same
/// transport, lincheck, and merged opening; only the zerocheck's round 1
/// differs (the genus-95 AG multiplication code replaces the RS
/// additive-NTT skip, and the claim points ride [`SkipPoint::Ag`]).
///
/// aarch64-only (the AG round-1 kernel is NEON SLP). Run-list read-exact —
/// the padding contract on [`flock_core::proof::BooleanPiopProofAg`] is
/// the owning statement; dirty pooled padding is legal for both flavors.
#[cfg(target_arch = "aarch64")]
pub fn prove_fast_ligerito_union_ag<Ch: Challenger>(
    union: &UnionInstance<'_>,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofMergedLigeritoAg, Commitment, R1csClaim) {
    assert!(
        !union.has_element(),
        "the AG union route is boolean-only; element registries go through          the RS mixed-class entry"
    );
    let (out, commitment) = prove_union_with_binding_zc(
        union,
        UnionProveBinding::Mixed,
        BooleanZcKind::Ag,
        pcs_params,
        slots,
        Vec::new(),
        challenger,
    );
    let (piop, claim) = out.boolean.expect("asserted boolean-only above");
    let UnionBooleanProof::Ag(boolean) = piop else {
        unreachable!("the Ag flavor produces an Ag boolean proof")
    };
    (
        R1csProofMergedLigeritoAg {
            boolean,
            pcs_open: out.pcs_open,
        },
        commitment,
        claim,
    )
}

/// What [`prove_union_with_binding`] produces: each class's PIOP sub-proof
/// paired with its claims (`None` when the registry has no type of that class),
/// plus the single opening covering all of them.
struct UnionProveOutput {
    boolean: Option<(UnionBooleanProof, R1csClaim)>,
    element: Option<(Proof, Claims)>,
    /// The wiring argument's transcript — `Some` exactly under
    /// [`UnionProveBinding::Circuit`].
    wiring: Option<WiringProof>,
    pcs_open: MergedOpenProof,
}

/// Shared body of the union prove entries; `binding` selects the statement
/// binding, everything else is identical.
///
/// Runs the two class PIOPs over their DISJOINT regions in the Fiat–Shamir
/// order documented on [`prove_fast_ligerito_union_mixed_class`],
/// then batches all four claims into one merged opening.
fn prove_union_with_binding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    binding: UnionProveBinding,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    element_slots: Vec<UnionElementSlotInput<'_>>,
    challenger: &mut Ch,
) -> (UnionProveOutput, Commitment) {
    prove_union_with_binding_zc(
        union,
        binding,
        BooleanZcKind::Rs,
        pcs_params,
        slots,
        element_slots,
        challenger,
    )
}

/// [`prove_union_with_binding`] with the boolean zerocheck flavor explicit.
fn prove_union_with_binding_zc<Ch: Challenger>(
    union: &UnionInstance<'_>,
    binding: UnionProveBinding,
    bool_zc: BooleanZcKind,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    element_slots: Vec<UnionElementSlotInput<'_>>,
    challenger: &mut Ch,
) -> (UnionProveOutput, Commitment) {
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
        union.num_boolean(),
        "need one prover input per BOOLEAN registry type"
    );
    assert_eq!(
        element_slots.len(),
        union.num_element(),
        "need one element prover input per ELEMENT registry type"
    );

    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config; bump m for tiny instances");

    // Union witness assembly, in slot order (booleans first, then elements —
    // the class-major sort): in-place slots generate straight into the union
    // buffers, prebuilt ones are copied (single slot: zero-copy passthrough),
    // element slots generate their `z` block and derive `a`/`b` from it. The
    // per-slot lincheck stripes come back alongside (empty for element slots).
    let mut sources = Vec::with_capacity(slots.len() + element_slots.len());
    let mut circuits = Vec::with_capacity(slots.len());
    for slot in slots {
        sources.push(slot.source);
        circuits.push(slot.lincheck_circuit);
    }
    for (ty, input) in union.registry().element_types().iter().zip(element_slots) {
        let element = match &ty.class {
            TableClass::LargeField(el) => el.clone(),
            TableClass::Boolean => {
                unreachable!("element_types() are LargeField")
            }
        };
        sources.push(UnionSlotWitnessSource::Element {
            ty: element,
            generate: input.generate,
        });
    }
    let trace = var("PCS_TRACE").is_ok();
    let t_all = Instant::now();
    let t = Instant::now();
    // BOOLEAN-only registries never read dropped words: the zerocheck is
    // run-list-gated, the union lincheck is count-proportional, compaction
    // reads declared rows only, and (when s_hat_v is precomputed) the
    // ring-switch succinct step reads nothing bulk. Padding may therefore
    // stay dirty in pooled resident buffers. NOT extended to the element
    // class for `z`: the element zerocheck debug-asserts `z` is zero on
    // every dead word and the region PIOP folds the committed words, so
    // the element `z` blocks must be honestly written in full. (`a`/`b`
    // dead rows ARE elided on the sparse-zerocheck arm — see
    // `build_union_witness`.)
    // And NOT under IDENTITY compaction: there q IS the padded buffer, so
    // its padding words are committed and must be honest zeros — dirty
    // pooling would put garbage into the committed stack (a latent hazard
    // of the pre-unification standalone body, never exercised there).
    // Flavor-independent since the AG round 1 + fold went run-list-gated
    // (Dead blocks skipped, Partial blocks cleansed — no declared-dead bit
    // is read on either flavor).
    let padding_unread = !union.has_element()
        && !union.compaction_is_identity()
        && union.m_total() - union.n_log() >= LOG_PACKING;
    let (z_packed, a_packed_f128, b_packed_f128, stripes, buf_mode) =
        build_union_witness(union, sources, padding_unread);
    // Where the witness buffers return: the dirty scratch pool for the pooled
    // modes, the ZERO pool (via `give_back_witness_buffer`) for FreshZeroed.
    let give_back = buf_mode != WitnessBufMode::FreshZeroed;
    if trace {
        eprintln!(
            "  [prove_union] witgen (padded 2^{}, {:?}): {:7.2} ms",
            union.m_total() - 7,
            buf_mode,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // Element slots' stripes are empty; `zip` truncates to the boolean prefix.
    let mut stripes = stripes;
    let element_stripes = stripes.split_off(union.num_boolean());
    debug_assert!(element_stripes.iter().all(|s| s.is_empty()));
    let linchecks: Vec<(Vec<u8>, &dyn LincheckCircuit)> =
        stripes.into_iter().zip(circuits).collect();

    // True dense-stack commit (height-n_t stacking): commit the compacted
    // stack q — the declared n_t-row prefix of every used chunk-column;
    // dummy rows, useless columns and gaps dropped; padded to a power of
    // two with the m22 config floor. The stack is OWNED (the merged open
    // consumes it for the inner eq-basis opening): identity compaction
    // (single-slot registries at full utilization) copies — a prototype
    // cost only. Under PooledDirty, dropped words are dirty by design —
    // and never read — so the compaction skips the honest-zeros
    // debug_assert.
    let t = Instant::now();
    // IDENTITY COMPACTION COPIES NOTHING. There is no compaction to do — the
    // dense stack IS the padded buffer, byte for byte — and `open_batch_merged`
    // wants both `q` (by value) and the padded witness (by reference), which
    // under identity are the same 512 MB at m32. Cloning to satisfy that was a
    // pure memcpy: 43 ms and a doubled residency, and single-threaded, so it
    // read 1.0x on the scaling table while everything around it read 6-7x.
    //
    // Instead the buffer is MOVED into the open (see `open_witness` below) and
    // the open aliases the padded witness to it. Pool-neutral: the Ligerito
    // sumcheck's Drop hands `f` back to `scratch`, which is exactly where this
    // caller's own give-back sends it — so only the FreshZeroed mode, whose
    // buffer belongs to the union's own pool instead, keeps the copy.
    let alias_q = union.compaction_is_identity() && give_back;
    let q_owned: Option<Vec<F128>> = if alias_q {
        None
    } else if union.compaction_is_identity() {
        Some(z_packed.clone())
    } else if buf_mode == WitnessBufMode::PooledDirty {
        Some(union.compact_witness_unchecked(&z_packed))
    } else {
        Some(union.compact_witness(&z_packed))
    };
    let q: &[F128] = q_owned.as_deref().unwrap_or(&z_packed);
    if trace {
        eprintln!(
            "  [prove_union] compact q (2^{} dense{}): {:7.2} ms",
            union.dense_m() - 7,
            if alias_q { ", ALIASED — no copy" } else { "" },
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // Integer-lane commit: when the dense stack leaves whole high-bit lanes
    // empty (`UnionInstance::commit_lanes`), encode + hash only the real ones.
    //
    // This applies to IDENTITY compaction too: identity means the dense stack
    // IS the padded buffer, not that the buffer is full — its useless
    // chunk-columns are still a contiguous zero tail (BLAKE3 commits 121 of
    // 128, so t = 61 of 64 lanes at M = 30). Both arms therefore dispatch on
    // `num_lanes` alone.
    let t = Instant::now();
    let (commitment, prover_data) = if pcs_params.num_lanes.is_some() {
        commit_lane_major(q, pcs_params)
    } else {
        commit(q, pcs_params)
    };
    if trace {
        eprintln!(
            "  [prove_union] commit: {:7.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = Instant::now();
    match binding {
        UnionProveBinding::Mixed => union.bind_statement(challenger, &commitment),
        UnionProveBinding::Circuit(ci) => {
            union.bind_statement_circuit(challenger, &commitment, &ci.circuit.digest(), ci.public)
        }
    }

    if trace {
        eprintln!(
            "  [prove_union] bind statement: {:7.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // Zerocheck over the BOOLEAN REGION of the union address space — the
    // prefix subcube `[0, 2^M_bool)` — driven by the count-derived run-list
    // (the existing kernels' general multi-run paths, value-identical to the
    // single-run spec on honestly-zero padding).
    //
    // The element region is NOT part of this sum. It cannot be: the union
    // zerocheck passes `c = z`, so on the element region the honest summand is
    // `0·0 − z ≠ 0` and the global sum would not vanish. `M_bool = M` for a
    // boolean-only registry, so nothing here changes for one.
    let padding = union.padding_spec();
    let bool_padding = union.boolean_padding_spec();
    let m_bool = union.m_bool();
    let bool_words = union.boolean_packed_len();

    // The element region's copies of `a`/`b`, taken BEFORE the boolean
    // zerocheck recycles those buffers. `2^(M_elem−7)` words each — the element
    // area, not the capacity.
    //
    // On the sparse-zerocheck arm (`dead_rows_unread` — the same predicate
    // that gated the live-only derivation above, so dead rows of `a`/`b`
    // were never even written) the copy takes LIVE SPANS ONLY into
    // lazy-zeroed buffers: the sparse row rounds read live prefixes and
    // substitute dead values analytically from `RowSupport::{a,b}_dead`,
    // so the zeros left behind are unread. The test
    // `dummy_row_is_structurally_invisible_under_the_union` checks this.
    //
    // On the dense arm (> 50% region utilization) the zerocheck reads the
    // whole region, so the copy stays faithful — including whatever the
    // full derivation wrote on dead rows (the per-column constants).
    let t = Instant::now();
    let mut live_arm = false;
    let element_ab: Option<(Vec<F128>, Vec<F128>)> = union.has_element().then(|| {
        let r = union.element_word_range();
        if dead_rows_unread(union) {
            live_arm = true;
            copy_live_region(union, &a_packed_f128[r.clone()], &b_packed_f128[r])
        } else {
            (a_packed_f128[r.clone()].to_vec(), b_packed_f128[r].to_vec())
        }
    });

    if trace && element_ab.is_some() {
        eprintln!(
            "  [prove_union] element a/b copies (2^{} words, {}): {:7.2} ms",
            union.m_elem() - 7,
            if live_arm { "LIVE spans" } else { "DENSE" },
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- The boolean class's PIOP pair, over the prefix subcube.
    //
    // FORK/JOIN transcript — THE protocol for a circuit binding, not an
    // option: the boolean PIOP and the wiring argument run CONCURRENTLY on
    // domain-separated chains, merged before anything downstream samples (the
    // element PIOP, the opening's γ's). Sound because both the zerocheck's r
    // and the wiring's α/β bind only the commitment+statement prefix, which
    // the fork point already covers; lincheck-after-zerocheck and
    // gather-after-GKR are data orderings WITHIN their own forks. The
    // verifier forks identically, and the recursion circuit binds both chains
    // (`merge_chain` in the tower's tape tests).
    let par_transcript =
        matches!(&binding, UnionProveBinding::Circuit(_)) && union.num_boolean() > 0;
    let t_bool = Instant::now();
    let run_boolean = |challenger: &mut Ch| {
        (union.num_boolean() > 0).then(|| {
            let (zc_proof, z_skip, mlv_challenges, zc_r_rest, zc_c_eval, s_hat_v_c) = {
                // Zero-cost &[u8] views of the F128 buffers; c aliases z (C = I).
                let view = |v: &[F128]| -> &[u8] {
                    unsafe {
                        from_raw_parts(v.as_ptr() as *const u8, bool_words * size_of::<F128>())
                    }
                };
                let a_packed = view(&a_packed_f128);
                let b_packed = view(&b_packed_f128);
                let c_packed = view(&z_packed);
                match bool_zc {
                    BooleanZcKind::Rs => {
                        let (p, cl, sv) = prove_packed_padded_capture_s_hat_v_c_with_grinding(
                            a_packed,
                            b_packed,
                            c_packed,
                            m_bool,
                            &bool_padding,
                            pcs_params.zerocheck_grinding(),
                            challenger,
                        );
                        (
                            UnionZcProof::Rs(p),
                            SkipPoint::Phi8(cl.z),
                            cl.mlv_challenges,
                            cl.r_rest,
                            cl.c_eval,
                            sv,
                        )
                    }
                    #[cfg(target_arch = "aarch64")]
                    BooleanZcKind::Ag => {
                        // Run-list-gated like the RS twin: round 1 and the
                        // fold skip Dead code blocks and cleanse Partial
                        // ones, reading no declared-dead bit — PooledDirty
                        // witnesses are legal here too.
                        let (p, cl, sv) = prove_capture_s_hat_v_c_with_grinding(
                            a_packed,
                            b_packed,
                            c_packed,
                            m_bool,
                            &bool_padding,
                            pcs_params.zerocheck_grinding(),
                            challenger,
                        );
                        (
                            UnionZcProof::Ag(p),
                            SkipPoint::Ag(cl.r1),
                            cl.mlv_challenges,
                            cl.r_rest,
                            cl.c_eval,
                            sv,
                        )
                    }
                }
            };

            let x_ab = union.x_ab_from_mlv(z_skip, &mlv_challenges);

            // M2: the union-column lincheck — one sumcheck over the boolean
            // column domain against the per-slot stripes and circuits. On the M1
            // single-type registries it is byte-identical to invoking the slot's
            // own lincheck (the union of one slot has m = M_bool = M).
            let (lc_proof, lc_claim, z_vec_pre) = {
                let lc_slots: Vec<UnionLincheckSlot<'_>> = linchecks
                    .iter()
                    .map(|(stripe, circuit)| UnionLincheckSlot {
                        z_lincheck: stripe,
                        circuit: *circuit,
                    })
                    .collect();
                prove_union_capture_z_vec_with_grinding(
                    union,
                    &lc_slots,
                    &x_ab,
                    pcs_params.lincheck_grinding(),
                    challenger,
                )
            };

            let ab = ZClaim {
                point: union.ab_claim_point(
                    lc_claim.r_inner_skip,
                    &lc_claim.r_inner_rest,
                    &x_ab.x_outer,
                ),
                value: lc_claim.w,
            };
            let c = ZClaim {
                point: union.c_claim_point(z_skip, &zc_r_rest),
                value: zc_c_eval,
            };

            // `s_hat_v_from_z_vec` needs `z_vec.len() = 2^LOG_PACKING · 2^tail`;
            // the boolean fold has `len = 2^(M_bool−ν)` and
            // `tail = M_bool−ν−LOG_PACKING`, so the condition is
            // `M_bool−ν ≥ LOG_PACKING` — for a single-type registry exactly the
            // old `k_log ≥ LOG_PACKING`, and always true for real registries
            // (every `k_log ≥ 7`).
            //
            // The precomputed value stays honest even though the AB claim's point
            // now carries `M − M_bool` frozen ZERO high coordinates:
            // `s_hat_v[b] = Σ_j eq(suffix, j)·bit_b(w[j])` and those zeros kill
            // every `j` outside the boolean region, so the full-buffer fold equals
            // this boolean-region one term for term.
            let s_hat_v_ab = if m_bool - union.n_log() >= LOG_PACKING {
                let t_sv = Instant::now();
                let sv = Some(s_hat_v_from_z_vec(&z_vec_pre, &lc_claim.r_inner_rest[1..]));
                if var("PCS_TRACE").is_ok() {
                    eprintln!(
                        "  [prove_union] s_hat_v_ab fold (z_vec 2^{}): {:6.2} ms",
                        m_bool - union.n_log(),
                        t_sv.elapsed().as_secs_f64() * 1e3
                    );
                }
                sv
            } else {
                None
            };
            let piop = match zc_proof {
                UnionZcProof::Rs(zerocheck) => UnionBooleanProof::Rs(BooleanPiopProof {
                    zerocheck,
                    lincheck: lc_proof,
                }),
                #[cfg(target_arch = "aarch64")]
                UnionZcProof::Ag(ag) => UnionBooleanProof::Ag(BooleanPiopProofAg {
                    ag,
                    lincheck: lc_proof,
                }),
            };
            (piop, R1csClaim { ab, c }, s_hat_v_ab, s_hat_v_c)
        })
    };
    let (boolean, wiring_pre) = if par_transcript {
        let UnionProveBinding::Circuit(ci) = &binding else {
            unreachable!("par_transcript requires a circuit binding");
        };
        // ONE-SIDED fork: only the wiring leaves the main chain — the
        // boolean PIOP continues on the parent transcript exactly as in
        // the sequential protocol. The child's seed sample advances the
        // parent BEFORE the zerocheck begins, so both branches bind the
        // same commitment+statement prefix; the merge (the child's closing
        // digest) lands after the boolean PIOP, before anything samples
        // against the wiring's messages. In the chained-BLAKE3 discipline
        // the circuit-side cost of this shape is ~one compression row: the
        // child chain can continue from the fork-point CV under a domain
        // byte, and the merge absorbs its 256-bit final CV in one block.
        //
        // The ZERO-row alternative — one shared chain interleaving both
        // protocols' absorbs, each shared point squeezing both challenges —
        // is equally sound (challenges bind a superset) but trades the one
        // row for a SCHEDULE: the GKR keeps one squeeze point per message
        // (~231) against the zerocheck's ~26, naive alignment overlaps only
        // the GKR's tiny layers, and a work-balanced interleave becomes
        // part of the protocol (tape + circuit, per (m, μ) shape) with a
        // rendezvous at every shared point. One row buys free-running and
        // two clean tape segments.
        let mut ch_w = challenger.fork(b"flock-par-wiring-v1");
        let (boolean, w) = join(
            || run_boolean(challenger),
            || {
                let r = prove_wiring_with_grinding(
                    ci.circuit,
                    &z_packed,
                    ci.public,
                    pcs_params.product_gkr_grinding(),
                    &mut ch_w,
                );
                (r, ch_w)
            },
        );
        let (wiring, ch_w) = w;
        challenger.merge_child(ch_w);
        (boolean, Some(wiring))
    } else {
        (run_boolean(challenger), None)
    };

    if trace && union.num_boolean() > 0 {
        eprintln!(
            "  [prove_union] boolean zerocheck + lincheck (M_bool = {}){}: {:7.2} ms",
            union.m_bool(),
            if par_transcript { " ∥ wiring" } else { "" },
            t_bool.elapsed().as_secs_f64() * 1e3
        );
    }
    // a/b are consumed; recycle the buffers as in `prove_fast_core`. The
    // FreshZeroed (all-zero) ones return to the ZERO pool with their slot
    // areas re-zeroed rather than being dropped — unmapping multi-GiB per
    // prove is what turned late-process `alloc_zeroed` into a real memset
    // (level-2 witgen 35 → 590 ms before this).
    if give_back {
        give_f128(a_packed_f128);
        give_f128(b_packed_f128);
    } else {
        union.give_back_witness_buffer(a_packed_f128);
        union.give_back_witness_buffer(b_packed_f128);
    }
    // Recycle the stripes (as large as the witness itself) rather than
    // unmapping them — the drivers take them from the same pool. Mode-
    // independent: the byte pool's write-or-zero-before-read contract does
    // not care how the witness buffers were sourced.
    for (stripe, _) in linchecks {
        give_u8(stripe);
    }

    // ---- The element class's PIOP pair, over the element region. Runs AFTER
    // the boolean pair, so its τ' is drawn from a transcript that already
    // absorbed every boolean message (and vice versa is impossible).
    let t = Instant::now();
    let element = element_ab.map(|(pa, pb)| {
        let r = union.element_word_range();
        let out = prove_with_grinding(
            union,
            &z_packed[r],
            &pa,
            &pb,
            pcs_params.element_grinding(),
            challenger,
        );
        // Recycle the region pair (the PIOP borrows, never writes): the live
        // arm's buffers came from the zero pool via `copy_live_region` and
        // return there with only their live spans dirty; the dense arm's
        // faithful full copies go to the dirty pool.
        if live_arm {
            give_back_live_region(union, pa, pb);
        } else {
            give_f128(pa);
            give_f128(pb);
        }
        out
    });
    if trace && element.is_some() {
        eprintln!(
            "  [prove_union] element region PIOP (2^{} words): {:7.2} ms",
            union.m_elem() - 7,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- The wiring argument over the circuit's cell space, AFTER both
    // classes' PIOPs (so its α, β come from a transcript that already absorbed
    // every class message) and BEFORE the opening its gather claims join.
    //
    // It reads `z_packed` — the padded buffer the commitment was built from,
    // whose dummy rows are zero by the union's witness contract, which is what
    // makes the dummy cells' `w = 0` honest.
    let t = Instant::now();
    let wiring = match (wiring_pre, &binding) {
        // FORK/JOIN variant: the wiring already ran concurrently with the
        // boolean PIOP on its own child transcript; only the claims flow on.
        (Some(w), _) => Some(w),
        (None, UnionProveBinding::Circuit(ci)) => Some(prove_wiring_with_grinding(
            ci.circuit,
            &z_packed,
            ci.public,
            pcs_params.product_gkr_grinding(),
            challenger,
        )),
        (None, _) => None,
    };
    if trace && let Some((_, claims)) = &wiring {
        eprintln!(
            "  [prove_union] wiring GKR + gather (μ = {}, {} claims): {:7.2} ms",
            binding_circuit_mu(&binding),
            claims.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- One opening over every claim: the boolean pair ring-switched (as
    // quirky points), the element pair and the wiring's gather claims
    // PACKED-DIRECT — carried unbuilt by the merged open (it derives its
    // identity-fold weights from `point`/`value` alone and never reads
    // `eq_ind`).
    let t = Instant::now();
    let heights = union.jagged_heights();
    let t_h = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let (z_claims, pre): (Vec<ZClaim>, Vec<Option<&[F128]>>) = match &boolean {
        Some((_, claim, s_hat_v_ab, s_hat_v_c)) => (
            vec![claim.ab.clone(), claim.c.clone()],
            vec![s_hat_v_ab.as_deref(), Some(s_hat_v_c.as_slice())],
        ),
        None => (Vec::new(), Vec::new()),
    };
    let t_z = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let mut packed_direct: Vec<PackedDirectClaim> = match &element {
        Some((_, claims)) => element_packed_direct_claims(claims),
        None => Vec::new(),
    };
    let t_e = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let wiring = wiring.map(|(proof, gather_claims)| {
        packed_direct.extend(gather_claims);
        proof
    });
    let t_w = t.elapsed().as_secs_f64() * 1e3;
    if trace {
        eprintln!(
            "  [prove_union] claim assembly: {:7.2} ms  (heights {t_h:.2} + z_claims \
             {t_z:.2} + element pd {t_e:.2} + wiring pd {t_w:.2})",
            t_h + t_z + t_e + t_w
        );
    }
    let t = Instant::now();
    let x_fulls: Vec<Vec<F128>> = z_claims
        .iter()
        .map(|cl| quirky_x_outer_full(&cl.point))
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    // Under the alias, `z_packed` IS `q`: hand it over and let the open serve
    // the padded witness from it. Otherwise the two are distinct buffers and
    // the padded one stays borrowed.
    let gb_words = z_packed.len();
    let (open_witness, padded_owner) = match q_owned {
        Some(v) => (v, Some(z_packed)),
        None => (z_packed, None),
    };
    let pcs_open = open_batch_merged(
        open_witness,
        padded_owner.as_deref(),
        &prover_data,
        &commitment,
        &x_refs,
        &pre,
        &packed_direct,
        &padding,
        &heights,
        union.n_log(),
        &lig_config,
        pcs_params.opening_grinding(),
        challenger,
    );
    let t_gb = Instant::now();
    // Under the alias there is nothing left to hand back: the open consumed
    // the one buffer and the Ligerito sumcheck's Drop already returned it to
    // `scratch`, which is where this arm would have sent it anyway.
    if let Some(zp) = padded_owner {
        if give_back {
            give_f128(zp);
        } else {
            union.give_back_witness_buffer(zp);
        }
    }
    if trace {
        eprintln!(
            "  [prove_union] witness buffer give-back (2^{} words, zero dirty spans): {:6.2} ms",
            gb_words.trailing_zeros(),
            t_gb.elapsed().as_secs_f64() * 1e3
        );
        eprintln!(
            "  [prove_union] open (rs×{}, pd×{}): {:7.2} ms",
            z_claims.len(),
            packed_direct.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
        // Self-delimiting: one line per prove, tagging the arm, so a trace
        // reader never has to guess which prove a phase belonged to.
        eprintln!(
            "  [prove_union] TOTAL {:7.2} ms === done (element: {}) ===",
            t_all.elapsed().as_secs_f64() * 1e3,
            element.is_some()
        );
    }

    (
        UnionProveOutput {
            boolean: boolean.map(|(piop, claim, _, _)| (piop, claim)),
            element,
            wiring,
            pcs_open,
        },
        commitment,
    )
}

/// `μ` of the circuit binding's cell space, for the trace line only.
fn binding_circuit_mu(binding: &UnionProveBinding<'_>) -> usize {
    match binding {
        UnionProveBinding::Circuit(ci) => ci.circuit.cells().mu(),
        _ => 0,
    }
}

/// The element class's two claims as packed-direct PCS claims, in the fixed
/// order `[C at r, LC at (r_row, r'_col)]` — the order the verifier rebuilds.
///
/// The claims ride as DEFERRED `DirectEqInd::EqPoint`: the shipped (merged)
/// transport never reads `eq_ind` — `open_batch_merged` builds its own
/// identity-fold weights from `point`/`value` — so materializing an eq
/// tensor here would be `2^(m_elem−7)` F128 per claim of pure waste. (A
/// forgotten conversion on a path that DID need a tensor would trip the
/// combine's "EqPoint claims are only supported alone" assert rather than
/// silently dropping the contribution.)
fn element_packed_direct_claims(claims: &Claims) -> Vec<PackedDirectClaim> {
    [
        (&claims.c_point, claims.c_value),
        (&claims.lc_point, claims.lc_value),
    ]
    .into_iter()
    .map(|(point, value)| PackedDirectClaim {
        point: point.clone(),
        value,
        eq_ind: DirectEqInd::EqPoint(point.clone()),
    })
    .collect()
}

/// AG-skip mirror of [`prove_ligerito`]: same commit → bind → zerocheck →
/// lincheck → ring-switch open pipeline, with round 1 of the zerocheck run on
/// the genus-95 AG multiplication code. The `(ab, c)` claims carry AG
/// base-code skip weights (`SkipPoint::Ag`) instead of φ₈ (`SkipPoint::Phi8`);
/// the c-claim point is `(skip = r₁, rest = friendly ‖ outer)`. aarch64-only
/// (the AG round-1 kernel is NEON).
#[cfg(target_arch = "aarch64")]
pub fn prove_ligerito_ag<Ch: Challenger>(
    r1cs: &BlockR1cs,
    z_packed: Vec<F128>,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> (R1csProofLigeritoAg, Commitment, R1csClaim) {
    assert_eq!(
        r1cs.layout,
        WitnessLayout::RowMajor,
        "the generic matrix-driven provers assume the row-major layout"
    );
    assert_eq!(z_packed.len(), 1usize << (r1cs.m - 7));
    assert_eq!(pcs_params.m, r1cs.m);
    assert_eq!(r1cs.k_skip, K_SKIP, "AG skip is k_skip=6");
    assert!(
        r1cs.c0_is_identity(),
        "prove_ligerito_ag: C = I convention required (c aliases z)"
    );

    // a = A·z, b = B·z; for the C = I convention c aliases z (see `prove_ligerito`).
    let a_packed_f128 = r1cs.apply_a_packed(&z_packed);
    let b_packed_f128 = r1cs.apply_b_packed(&z_packed);
    let z_packed_lincheck = pack_z_lincheck_from_packed(&z_packed, r1cs.m, r1cs.k_log);
    let lc_circuit = SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
    prove_fast_ligerito_ag_from_witness(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        &lc_circuit,
        None,
        challenger,
    )
}

/// AG-skip mirror of [`prove_fast_ligerito_from_witness`]: commit → bind →
/// **AG zerocheck** ([`zerocheck::ag_skip`]) → lincheck → ring-switch Ligerito
/// open. The zerocheck's `s_hat_v_c` capture and lincheck's `z_vec` capture
/// feed the open exactly as in the RS path; claim points are built through the
/// layout-aware [`BlockR1cs`] constructors, so both witness layouts work.
/// aarch64-only (the AG round-1 kernel is NEON).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_ag_from_witness<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigeritoAg, Commitment, R1csClaim) {
    assert_eq!(r1cs.k_skip, K_SKIP, "AG skip is k_skip=6");
    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config; bump m for tiny instances");

    let (commitment, prover_data) = match prefaulted_codeword {
        Some(buf) => commit_into(&z_packed, pcs_params, buf),
        None => commit(&z_packed, pcs_params),
    };
    bind_statement(challenger, r1cs, &commitment);

    // ---- AG-skip zerocheck (round 1 = genus-95 AG code; tail = shared MLV).
    // Capture s_hat_v_c so the open skips fold_1b_rows for the c-claim.
    let (ag_proof, ag_claim, s_hat_v_c) = {
        let cast = |v: &[F128]| -> &[u8] {
            unsafe { from_raw_parts(v.as_ptr() as *const u8, size_of_val(v)) }
        };
        prove_capture_s_hat_v_c(
            cast(&a_packed_f128),
            cast(&b_packed_f128),
            cast(&z_packed),
            r1cs.m,
            challenger,
        )
    };
    give_f128(a_packed_f128);
    give_f128(b_packed_f128);

    // ---- Translate AG zerocheck output → lincheck input. Structurally
    // identical to the RS path (`mlv_challenges` binds the m−k_skip non-skip
    // bits low→high, address-ordered), only the skip basis differs (Ag vs Phi8).
    let x_ab = r1cs.x_ab_from_mlv(SkipPoint::Ag(ag_claim.r1), &ag_claim.mlv_challenges);

    let (lc_proof, lc_claim, z_vec_pre) = prove_padded_capture_z_vec(
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
    // c-claim: ĉ = ẑ (C = I). Skip = r₁ (AG), rest = ag_claim.r_rest
    // (friendly ‖ outer); value = ⟨base(r₁), w̄⟩ (κ already in w̄).
    let c = ZClaim {
        point: r1cs.c_claim_point(SkipPoint::Ag(ag_claim.r1), &ag_claim.r_rest),
        value: ag_claim.c_eval,
    };

    // AB s_hat_v from lincheck's pre-sumcheck z_vec (basis-agnostic: it folds
    // the witness bits, the skip weights enter only at claim_check). The c
    // s_hat_v was captured during the AG round-1 c-scan.
    let s_hat_v_ab = if r1cs.k_log >= LOG_PACKING {
        Some(s_hat_v_from_z_vec(&z_vec_pre, &lc_claim.r_inner_rest[1..]))
    } else {
        None
    };

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
        pcs_params.opening_grinding(),
        challenger,
    );

    let proof = R1csProofLigeritoAg {
        ag: ag_proof,
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
    pub zc_proof: ZerocheckProof,
    pub lc_proof: LincheckProof,
    pub ab: ZClaim,
    pub c: ZClaim,
    pub commitment: Commitment,
    pub prover_data: ProverData,
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
    lincheck_circuit: &dyn LincheckCircuit,
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
fn prove_fast_core_with_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> ProveCore {
    let (commitment, prover_data) = match prefaulted_codeword {
        Some(buf) => commit_into(&z_packed, pcs_params, buf),
        None => commit(&z_packed, pcs_params),
    };
    bind_statement(challenger, r1cs, &commitment);

    let padding = r1cs.padding_spec();
    let (zc_proof, zc_claim, s_hat_v_c) = {
        // Zero-cost &[u8] views of the F128 buffers; c aliases z (C = I).
        let a_packed: &[u8] = unsafe {
            from_raw_parts(
                a_packed_f128.as_ptr() as *const u8,
                a_packed_f128.len() * size_of::<F128>(),
            )
        };
        let b_packed: &[u8] = unsafe {
            from_raw_parts(
                b_packed_f128.as_ptr() as *const u8,
                b_packed_f128.len() * size_of::<F128>(),
            )
        };
        let c_packed: &[u8] = unsafe {
            from_raw_parts(
                z_packed.as_ptr() as *const u8,
                z_packed.len() * size_of::<F128>(),
            )
        };
        prove_packed_padded_capture_s_hat_v_c_with_grinding(
            a_packed,
            b_packed,
            c_packed,
            r1cs.m,
            &padding,
            pcs_params.zerocheck_grinding(),
            challenger,
        )
    };
    // Nothing downstream reads a/b (zerocheck consumed them in rounds 1–2);
    // recycle the two buffers (2 × 2^(m-3) bytes — 128 MB at m = 29) instead
    // of carrying them through lincheck and the PCS open.
    give_f128(a_packed_f128);
    give_f128(b_packed_f128);

    let x_ab = r1cs.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);

    // Capture lincheck's pre-sumcheck z_vec so the PCS open can derive the
    // AB-claim's `s_hat_v` from it (skips fold_1b_rows for AB).
    let (lc_proof, lc_claim, z_vec_pre) = prove_padded_capture_z_vec_with_grinding(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        lincheck_circuit,
        &x_ab,
        pcs_params.lincheck_grinding(),
        challenger,
    );
    // The lincheck stripe copy of z is dead from here on; free it before the
    // PCS open (2^(m-3) bytes — 64 MB at m = 29).
    give_u8(z_packed_lincheck);

    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(SkipPoint::Phi8(zc_claim.z), &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    // Strided fold of z_vec_pre against the AB-claim suffix's inner-rest tail
    // (everything past prefix0). Byte-identical to `fold_1b_rows` on the AB
    // suffix tensor — see `s_hat_v_from_z_vec`. Skip when k_log < LOG_PACKING
    // (only test setups; real R1CS has k_log >= 16).
    let s_hat_v_ab = if r1cs.k_log >= LOG_PACKING {
        Some(s_hat_v_from_z_vec(&z_vec_pre, &lc_claim.r_inner_rest[1..]))
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

/// The **mixed-class** union prove entry over the MERGED transport (the
/// only transport since the jagged one was removed): the boolean PIOP, the
/// element PIOP, and one merged opening carrying all four claims — the
/// boolean pair ring-switched, the element pair packed-direct.
///
/// Fiat–Shamir order (every prover message observed before the challenge
/// that depends on it): commit → `bind_statement` → boolean τ → boolean
/// zerocheck → boolean lincheck (α, β_t) → element τ' → element zerocheck →
/// element α' → element lincheck → γ-batched merged opening. Either class
/// may be absent: a boolean-only registry produces `element: None` (and is
/// transcript-identical to [`prove_fast_ligerito_union`] —
/// only the proof struct differs), an element-only one `boolean: None` and
/// an opening with no ring-switched claims at all.
pub fn prove_fast_ligerito_union_mixed_class<Ch: Challenger>(
    union: &UnionInstance<'_>,
    pcs_params: &PcsParams,
    slots: Vec<UnionSlotProverInput<'_>>,
    element_slots: Vec<UnionElementSlotInput<'_>>,
    challenger: &mut Ch,
) -> (R1csProofMixedClassMerged, Commitment, UnionClassClaims) {
    let (out, commitment) = prove_union_with_binding(
        union,
        UnionProveBinding::Mixed,
        pcs_params,
        slots,
        element_slots,
        challenger,
    );
    let UnionProveOutput {
        boolean,
        element,
        wiring,
        pcs_open,
    } = out;
    debug_assert!(wiring.is_none(), "the Mixed binding runs no wiring");
    let (bool_proof, bool_claim) = match boolean {
        Some((p, c)) => (Some(p.expect_rs()), Some(c)),
        None => (None, None),
    };
    let (el_proof, el_claim) = match element {
        Some((p, c)) => (Some(p), Some(c)),
        None => (None, None),
    };
    (
        R1csProofMixedClassMerged {
            boolean: bool_proof,
            element: el_proof,
            pcs_open,
        },
        commitment,
        UnionClassClaims {
            boolean: bool_claim,
            element: el_claim,
        },
    )
}
