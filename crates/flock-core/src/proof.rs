//! Shared R1CS proof types and the Fiat-Shamir statement binding.
//!
//! These live in a backend-neutral module (rather than in `prover`) so the
//! verifier can name them without depending on the prove path. The prover
//! produces these structs; the verifier consumes them.

use lincheck::LincheckProof;
use pcs::{BatchOpeningProofLigerito, MergedOpenProof};
use serde::{Deserialize, Serialize};
use zerocheck::{ZerocheckProof, ag_skip::AgProof};

use crate::{
    challenger::Challenger,
    circuit::WiringProof,
    element_r1cs::union::{Claims, Proof},
    field::F128,
    lincheck::{self, QuirkyPoint},
    pcs::{self, Commitment},
    r1cs::BlockR1cs,
    zerocheck,
};

/// Top-level R1CS proof: zerocheck + lincheck transcripts, plus one batched
/// Ligerito PCS opening covering both the `ab` and `c` z-claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofLigerito {
    pub zerocheck: ZerocheckProof,
    pub lincheck: LincheckProof,
    pub pcs_open: BatchOpeningProofLigerito,
}

/// Top-level R1CS proof with the **AG-skip** zerocheck + Ligerito PCS backend.
/// Identical downstream of the zerocheck (lincheck + the same standard
/// ring-switch open on the std pack); only round 1 of the zerocheck differs
/// (the genus-95 AG multiplication code replaces the RS additive-NTT skip), so
/// `ag` carries the AG round messages instead of a `ZerocheckProof`. The skip
/// stays in the packing prefix `[skip 6 | bit6]`, so both `(ab, c)` claims open
/// via the unchanged RS path with AG base-code skip weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofLigeritoAg {
    pub ag: AgProof,
    pub lincheck: LincheckProof,
    pub pcs_open: BatchOpeningProofLigerito,
}

/// R1CS proof with a merged jagged and ring-switch opening.
///
/// The PIOP subproofs match [`R1csProofLigerito`]. The opening transport differs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMergedLigerito {
    pub zerocheck: ZerocheckProof,
    pub lincheck: LincheckProof,
    pub pcs_open: MergedOpenProof,
}

/// A **mixed-class** union proof over the MERGED (Frobenius) transport: the
/// boolean PIOP, the element-region PIOP, and ONE merged opening covering
/// all four claims (boolean AB + C ring-switched, element C + LC
/// packed-direct — each expressed, to the weight builder, as the
/// `F₂`-linear map `x ↦ γ·x`, indistinguishable from a ring-switched
/// claim's Φ-fold).
///
/// Each class's sub-proof is `Option`: a boolean-only registry has no element
/// half, an element-only one no boolean half.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMixedClassMerged {
    /// Boolean zerocheck + lincheck over the `M_bool` prefix subcube.
    pub boolean: Option<BooleanPiopProof>,
    /// The element-region zerocheck + lincheck.
    pub element: Option<Proof>,
    pub pcs_open: MergedOpenProof,
}

/// A **circuit** proof over the MERGED (Frobenius) transport — a
/// mixed-class union proof plus the wiring argument over the circuit's cell
/// space. What it attests, in one proof: every gate row satisfies its
/// table's relation, the circuit's wiring equalities hold, and the
/// designated cells equal the statement's public words.
///
/// The wiring argument's gather claims are packed-direct, which the merged
/// transport carries by expressing each weight as the `F₂`-linear map
/// `x ↦ γ·x` — the same intake the element class's claims use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofCircuitMerged {
    pub boolean: Option<BooleanPiopProof>,
    pub element: Option<Proof>,
    pub wiring: WiringProof,
    pub pcs_open: MergedOpenProof,
}

/// The boolean class's two PIOP sub-proofs, as they appear inside
/// [`R1csProofMixedClassMerged`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanPiopProof {
    pub zerocheck: ZerocheckProof,
    pub lincheck: LincheckProof,
}

/// [`BooleanPiopProof`] with the **AG-skip** zerocheck: the genus-95 AG
/// multiplication code replaces the RS additive-NTT round 1; the lincheck and
/// every claim downstream are unchanged (the skip point rides as
/// [`lincheck::SkipPoint::Ag`]). PADDING CONTRACT (the owning statement —
/// other sites point here): the AG union entries are run-list READ-EXACT,
/// exactly like the RS kernels — Dead code blocks are skipped, Partial
/// blocks are cleansed into zeroed scratch (`zerocheck::cleanse_block`),
/// and no declared-dead bit is ever read — so dirty pooled padding
/// (`PooledDirty`) is legal for both flavors. Only the DENSE direct-route
/// entries (`ag_skip::prove`, `prove_capture_s_hat_v_c`) still sum the
/// full region and need honestly zero padding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanPiopProofAg {
    pub ag: AgProof,
    pub lincheck: LincheckProof,
}

/// [`R1csProofMergedLigerito`] with the **AG-skip** boolean zerocheck — the
/// boolean-only union proof over the MERGED transport, AG flavor. Same
/// transport, same lincheck, same merged opening; only the zerocheck's round
/// 1 differs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofMergedLigeritoAg {
    pub boolean: BooleanPiopProofAg,
    pub pcs_open: MergedOpenProof,
}

/// [`R1csProofCircuitMerged`] with the **AG-skip** boolean zerocheck — the
/// circuit proof, AG flavor. The element class and the wiring argument are
/// flavor-independent; only the boolean zerocheck's round 1 differs (and the
/// boolean claim points ride [`lincheck::SkipPoint::Ag`]). MIGRATION shape:
/// when the RS skip is removed this struct is renamed to primary and
/// [`R1csProofCircuitMerged`] is deleted (docs/ag-recursion-plan.md).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1csProofCircuitMergedAg {
    pub boolean: Option<BooleanPiopProofAg>,
    pub element: Option<Proof>,
    pub wiring: WiringProof,
    pub pcs_open: MergedOpenProof,
}

/// The claims a verified mixed-class union proof leaves behind, per class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionClassClaims {
    /// Boolean AB + C — `None` when the registry has no boolean types.
    pub boolean: Option<R1csClaim>,
    /// Element C + LC, in union word coordinates — `None` when the registry
    /// has no element types.
    pub element: Option<Claims>,
}

/// A claim of the form `ẑ(point) = value` for the witness `z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZClaim {
    pub point: QuirkyPoint,
    pub value: F128,
}

/// Two MLE evaluation claims on `z` that the PCS layer must verify.
///
/// Both `point.x_outer` parts differ; both `point.z_skip` and
/// `point.x_inner_rest` shapes match (one univariate-skip coord + multilinear
/// inner-rest), so this is "two quirky-shaped openings of `z`."
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csClaim {
    /// From lincheck: `ẑ(ab.point) = ab.value` — covers both `â` and `b̂` at
    /// the same point (their lincheck claims collapsed to a shared z-claim
    /// at a fresh quirky inner point).
    pub ab: ZClaim,
    /// From the zerocheck's extract_c interpolation: `ẑ(c.point) = c.value`.
    /// Bypasses lincheck because `C = I` ⇒ ĉ-claim is a direct z-claim.
    pub c: ZClaim,
}

/// Bind the Fiat-Shamir transcript to the statement: the R1CS instance digest
/// + the PCS commitment cap. Call once at the top of every R1CS prove/verify
/// path, before any sub-protocol challenge is drawn. RandomChallenger ignores
/// these observations; FsChallenger uses them to defeat statement substitution.
pub fn bind_statement<Ch: Challenger>(
    challenger: &mut Ch,
    r1cs: &BlockR1cs,
    commitment: &Commitment,
) {
    challenger.observe_label(b"flock-r1cs-v0");
    challenger.observe_bytes(&r1cs.statement_digest());
    challenger.observe_bytes(commitment.cap.as_flattened());
}
