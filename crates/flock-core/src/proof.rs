//! Shared R1CS proof types and the Fiat-Shamir statement binding.
//!
//! These live in a backend-neutral module (rather than in `prover`) so the
//! verifier can name them without depending on the prove path. The prover
//! produces these structs; the verifier consumes them.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::{self, QuirkyPoint};
use crate::pcs::{self, Commitment};
use crate::r1cs::BlockR1cs;
use crate::zerocheck;
use serde::{Deserialize, Serialize};

/// Top-level R1CS proof: zerocheck + lincheck transcripts, plus one batched
/// Ligerito PCS opening covering both the `ab` and `c` z-claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofLigerito {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
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
    pub ag: zerocheck::ag_skip::AgProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
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
/// + the PCS commitment root. Call once at the top of every R1CS prove/verify
/// path, before any sub-protocol challenge is drawn. RandomChallenger ignores
/// these observations; FsChallenger uses them to defeat statement substitution.
pub fn bind_statement<Ch: Challenger>(
    challenger: &mut Ch,
    r1cs: &BlockR1cs,
    commitment: &Commitment,
) {
    challenger.observe_label(b"flock-r1cs-v0");
    challenger.observe_bytes(&r1cs.statement_digest());
    challenger.observe_bytes(&commitment.root);
}
