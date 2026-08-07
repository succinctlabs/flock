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
use flock_core::challenger::Challenger;
use flock_core::circuit::Circuit;
use flock_core::circuit::WiringProof;
use flock_core::circuit::prove_wiring;
use flock_core::element_r1cs::ElementTableType;
use flock_core::element_r1cs::union::Claims;
use flock_core::element_r1cs::union::Proof;
use flock_core::element_r1cs::union::copy_live_region;
use flock_core::element_r1cs::union::dead_rows_unread;
use flock_core::element_r1cs::union::fill_slot;
use flock_core::element_r1cs::union::prove;
use flock_core::field::F128;
use flock_core::lincheck::{self, QuirkyPoint, pack_z_lincheck_from_packed};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::BooleanPiopProof;
use flock_core::proof::R1csProofCircuitMerged;
use flock_core::proof::R1csProofMergedLigerito;
use flock_core::proof::R1csProofMixedClassMerged;
use flock_core::proof::UnionClassClaims;
use flock_core::proof::{R1csClaim, R1csProofLigerito, ZClaim, bind_statement};
use flock_core::r1cs::BlockR1cs;
use flock_core::r1cs::WitnessLayout;
use flock_core::schedule::TableClass;
use flock_core::scratch::give_f128;
use flock_core::scratch::give_u8;
use flock_core::union::SlotWitness;
use flock_core::union::SlotWitnessDest;
use flock_core::union::UnionInstance;
use flock_core::union::WitnessBufMode;
use flock_core::zerocheck;
use std::env::var;
use std::mem::size_of_val;
use std::slice::from_raw_parts;
use std::sync::Arc;
use std::time::Instant;

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
    pub lincheck_circuit: &'a dyn lincheck::LincheckCircuit,
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
        lincheck_circuit: &'a dyn lincheck::LincheckCircuit,
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
        lincheck_circuit: &'a dyn lincheck::LincheckCircuit,
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
pub struct CircuitProverInput<'a> {
    pub circuit: &'a Circuit,
    pub public: &'a [F128],
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
        Some((p, c)) => (Some(p), Some(c)),
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

/// The MERGED-transport union prover (wire v6; design doc §"Capacity-free
/// ring-switching") — the Mixed protocol's prove entry for BOOLEAN-only
/// registries: a thin wrapper over [`prove_union_with_binding`] (the one
/// shared body, since the two-body split died with the jagged transport),
/// repackaging the boolean sub-proofs as the wire's
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

/// What [`prove_union_with_binding`] produces: each class's PIOP sub-proof
/// paired with its claims (`None` when the registry has no type of that class),
/// plus the single opening covering all of them.
struct UnionProveOutput {
    boolean: Option<(BooleanPiopProof, R1csClaim)>,
    element: Option<(Proof, Claims)>,
    /// The wiring argument's transcript — `Some` exactly under
    /// [`UnionProveBinding::Circuit`].
    wiring: Option<WiringProof>,
    pcs_open: pcs::MergedOpenProof,
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
    // `build_union_witness` — that part of the old follow-up is done.)
    // And NOT under IDENTITY compaction: there q IS the padded buffer, so
    // its padding words are committed and must be honest zeros — dirty
    // pooling would put garbage into the committed stack (a latent hazard
    // of the pre-unification standalone body, never exercised there).
    let padding_unread = !union.has_element()
        && !union.compaction_is_identity()
        && union.m_total() - union.n_log() >= pcs::LOG_PACKING;
    let (z_packed, a_packed_f128, b_packed_f128, stripes, buf_mode) =
        build_union_witness(union, sources, padding_unread);
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
    let linchecks: Vec<(Vec<u8>, &dyn lincheck::LincheckCircuit)> =
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
    let q: Vec<F128> = if union.compaction_is_identity() {
        z_packed.clone()
    } else if buf_mode == WitnessBufMode::PooledDirty {
        union.compact_witness_unchecked(&z_packed)
    } else {
        union.compact_witness(&z_packed)
    };
    if trace {
        eprintln!(
            "  [prove_union] compact q (2^{} dense): {:7.2} ms",
            union.dense_m() - 7,
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
        pcs::commit_lane_major(&q, pcs_params)
    } else {
        pcs::commit(&q, pcs_params)
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
    // so the zeros left behind are unread — byte-identical, pinned by
    // `dummy_row_is_structurally_invisible_under_the_union`. (An earlier
    // note here forbade compacting this copy outright, citing the jagged-
    // era rejection posture; on the merged transport dead words are
    // structurally invisible and the DENSE arm below keeps the faithful
    // full copy.)
    //
    // On the dense arm (> 50% region utilization) the zerocheck reads the
    // whole region, so the copy stays faithful — including whatever the
    // full derivation wrote on dead rows (the per-column constants).
    let t = Instant::now();
    let element_ab: Option<(Vec<F128>, Vec<F128>)> = union.has_element().then(|| {
        let r = union.element_word_range();
        if dead_rows_unread(union) {
            copy_live_region(union, &a_packed_f128[r.clone()], &b_packed_f128[r])
        } else {
            (a_packed_f128[r.clone()].to_vec(), b_packed_f128[r].to_vec())
        }
    });

    if trace && element_ab.is_some() {
        eprintln!(
            "  [prove_union] element a/b copies (2^{} words): {:7.2} ms",
            union.m_elem() - 7,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- The boolean class's PIOP pair, over the prefix subcube.
    let t_bool = Instant::now();
    let boolean = (union.num_boolean() > 0).then(|| {
        let (zc_proof, zc_claim, s_hat_v_c) = {
            // Zero-cost &[u8] views of the F128 buffers; c aliases z (C = I).
            let view = |v: &[F128]| -> &[u8] {
                unsafe { from_raw_parts(v.as_ptr() as *const u8, bool_words * size_of::<F128>()) }
            };
            let a_packed = view(&a_packed_f128);
            let b_packed = view(&b_packed_f128);
            let c_packed = view(&z_packed);
            zerocheck::prove_packed_padded_capture_s_hat_v_c(
                a_packed,
                b_packed,
                c_packed,
                m_bool,
                &bool_padding,
                challenger,
            )
        };

        let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

        // M2: the union-column lincheck — one sumcheck over the boolean
        // column domain against the per-slot stripes and circuits. On the M1
        // single-type registries it is byte-identical to invoking the slot's
        // own lincheck (the union of one slot has m = M_bool = M).
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

        let ab = ZClaim {
            point: union.ab_claim_point(
                lc_claim.r_inner_skip,
                &lc_claim.r_inner_rest,
                &x_ab.x_outer,
            ),
            value: lc_claim.w,
        };
        let c = ZClaim {
            point: union.c_claim_point(zc_claim.z, &zc_claim.r_rest),
            value: zc_claim.c_eval,
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
        let s_hat_v_ab = if m_bool - union.n_log() >= pcs::LOG_PACKING {
            Some(pcs::ring_switch::s_hat_v_from_z_vec(
                &z_vec_pre,
                &lc_claim.r_inner_rest[1..],
            ))
        } else {
            None
        };
        (
            BooleanPiopProof {
                zerocheck: zc_proof,
                lincheck: lc_proof,
            },
            R1csClaim { ab, c },
            s_hat_v_ab,
            s_hat_v_c,
        )
    });

    if trace && union.num_boolean() > 0 {
        eprintln!(
            "  [prove_union] boolean zerocheck + lincheck (M_bool = {}): {:7.2} ms",
            union.m_bool(),
            t_bool.elapsed().as_secs_f64() * 1e3
        );
    }
    // a/b are consumed; recycle the buffers as in `prove_fast_core`.
    if give_back {
        give_f128(a_packed_f128);
        give_f128(b_packed_f128);
    }
    // Recycle the stripes (as large as the witness itself) rather than
    // unmapping them — the drivers take them from the same pool.
    for (stripe, _) in linchecks {
        if give_back {
            give_u8(stripe);
        }
    }

    // ---- The element class's PIOP pair, over the element region. Runs AFTER
    // the boolean pair, so its τ' is drawn from a transcript that already
    // absorbed every boolean message (and vice versa is impossible).
    let t = Instant::now();
    let element = element_ab.map(|(pa, pb)| {
        let r = union.element_word_range();
        prove(union, &z_packed[r], pa, pb, challenger)
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
    let wiring = match &binding {
        UnionProveBinding::Circuit(ci) => {
            Some(prove_wiring(ci.circuit, &z_packed, ci.public, challenger))
        }
        _ => None,
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
    let mut packed_direct: Vec<pcs::PackedDirectClaim> = match &element {
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
    let pcs_open = pcs::open_batch_merged(
        q,
        &z_packed,
        &prover_data,
        &commitment,
        &x_refs,
        &pre,
        &packed_direct,
        &padding,
        &heights,
        union.n_log(),
        &lig_config,
        challenger,
    );
    give_f128(z_packed);
    if trace {
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
fn element_packed_direct_claims(claims: &Claims) -> Vec<pcs::PackedDirectClaim> {
    [
        (&claims.c_point, claims.c_value),
        (&claims.lc_point, claims.lc_value),
    ]
    .into_iter()
    .map(|(point, value)| pcs::PackedDirectClaim {
        point: point.clone(),
        value,
        eq_ind: pcs::DirectEqInd::EqPoint(point.clone()),
    })
    .collect()
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
        zerocheck::prove_packed_padded_capture_s_hat_v_c(
            a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
        )
    };
    // Nothing downstream reads a/b (zerocheck consumed them in rounds 1–2);
    // recycle the two buffers (2 × 2^(m-3) bytes — 128 MB at m = 29) instead
    // of carrying them through lincheck and the PCS open.
    give_f128(a_packed_f128);
    give_f128(b_packed_f128);

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
    give_u8(z_packed_lincheck);

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
    /// SUB-phase of `witness_s` (union paths only): building the padded
    /// union buffers — the scatter (`UnionInstance::assemble_witness`) for
    /// prebuilt slots, or the drivers' in-place generation for
    /// [`UnionSlotProverInput::in_place`] ones (where it therefore ALSO
    /// covers witness generation). Do not add to the total.
    pub witness_place_s: f64,
    /// SUB-phase of `witness_s` (union paths only): the dense-stack gather
    /// (`UnionInstance::compact_witness`). Do not add to the total.
    pub witness_compact_s: f64,
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

    let lig_config = pcs_params
        .ligerito_prover_config()
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
        zerocheck::prove_packed_padded_capture_s_hat_v_c(
            a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
        )
    };
    t.zerocheck_s = t0.elapsed().as_secs_f64();
    give_f128(a_packed_f128);
    give_f128(b_packed_f128);

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
    give_u8(z_packed_lincheck);
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
        Some((p, c)) => (Some(p), Some(c)),
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
