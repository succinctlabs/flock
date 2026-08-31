//! Aggregating many proofs' matrix work into one accumulator.
//!
//! This is the driver that composes the two halves of the accumulation
//! route: [`lincheck::verify_union_deferred`] replays a proof succinctly and
//! hands back a [`MatrixAssertion`] instead of reading the base matrices,
//! and [`crate::matrix_fold`] folds those assertions' claims into a running
//! [`Accumulator`]. Nobody reads a matrix until somebody discharges the
//! accumulator — once, at the end.
//!
//! **This does not make native verification faster** — measured, batching is
//! several times SLOWER. Folding `k` claims costs `k · nnz` (the row phase
//! builds `g_i` per claim), which is exactly what checking those `k` claims
//! directly costs, plus the final discharge on top. No random-linear-
//! combination avoids it: the combined weight is a sum of `k` rank-1 terms,
//! so every nonzero still needs `k` multiplications. `k` claims at `k`
//! distinct points cost `k` passes over the matrix, full stop.
//!
//! What the fold buys is an ASYMMETRY, and it is worth only one thing:
//! recursion. The fold's PROVER pays that `k · nnz`; the fold's VERIFIER
//! pays `O(κ)` and reads no matrix at all. That moves the matrix work from
//! inside a circuit — where it is nnz-preserving and the fixed point cannot
//! close — to a native prover, where it is ordinary. So this module is the
//! thing a recursion circuit arithmetises.
//! [`verify_aggregate`] touches no matrix, so it is exactly what a merge
//! circuit replays: verify the children succinctly, fold their claims plus
//! the accumulators they carried, output one accumulator. The proof that
//! comes out has the same shape as the ones that went in — a proof plus an
//! accumulator — which is what lets the recursion close.
//!
//! ## What folds with what
//!
//! One accumulator per `(boolean type, matrix)`: `A₀` and `B₀` never mix,
//! because only their α-combination appears in a proof's target and α is
//! per-proof, so a claim about `α·A₀ + B₀` names a different polynomial in
//! every proof. Within one accumulator the fold takes
//!
//! * the claim carried in by each prior accumulator (a merge node's
//!   children each bring one), in order, then
//! * the claim each verified proof emitted (one per proof).
//!
//! So a leaf over two proofs folds `2 → 1`, and a `2 → 1` merge of two
//! recursive proofs folds `4 → 1` (two inherited, two fresh).

use crate::circuit::Circuit;
use crate::circuit::SigmaAssertion;
use crate::element_r1cs::SparseF128Matrix;
use crate::element_r1cs::union::ElementUnionError;
use crate::lincheck::LincheckCircuit;
use crate::matrix_fold::bilinear;
use crate::pcs::jagged::JaggedParams;
use crate::union::UnionInstance;
use matrix_fold::FoldError;
use matrix_fold::FoldGrinding;
use matrix_fold::FoldMatrix;
use matrix_fold::JaggedAssertion;
use matrix_fold::JaggedClaim;
use matrix_fold::JaggedTable;
use matrix_fold::col_marginal;
use matrix_fold::discharge_jagged;
use matrix_fold::prove_fold_jagged_with_grinding;
use matrix_fold::prove_fold_with_grinding;
use matrix_fold::verify_fold_jagged_with_grinding;
use matrix_fold::verify_fold_with_grinding;
use serde::{Deserialize, Serialize};

use crate::challenger::Challenger;
use crate::element_r1cs::union::ElementAssertion;
use crate::field::F128;
use crate::lincheck::{LincheckError, MatrixAssertion};
use crate::matrix_fold::{self, FoldProof, MatrixClaim};
use crate::r1cs::SparseBinaryMatrix;
use crate::schedule::Registry;

const DOMAIN: &[u8] = b"flock-aggregate-v0";

/// The base matrices of one boolean type, `(A₀, B₀)`.
pub type TypeMatrices<'a> = (&'a SparseBinaryMatrix, &'a SparseBinaryMatrix);

/// The base matrices of one element type, `(A₀, B₀)` — `F128` coefficients,
/// not `GF(2)` supports.
pub type ElementMatrices<'a> = (&'a SparseF128Matrix, &'a SparseF128Matrix);

/// Accumulated matrix claims: one `(A₀, B₀)` pair per boolean type, in slot
/// order, tied to the registry they are about.
///
/// The digest is load-bearing rather than decorative: claims fold only if
/// they name the same matrices, so an accumulator from a different registry
/// must be rejected, not silently folded into one it does not belong to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accumulator {
    pub registry_digest: [u8; 32],
    /// Boolean types, in slot order.
    pub per_type: Vec<(MatrixClaim, MatrixClaim)>,
    /// Element types, in slot order — a SEPARATE group, because they name
    /// different matrices.
    ///
    /// The key is really `(registry, type, matrix)`. The registry digest
    /// covers the first component; the class split is the second. Folding a
    /// boolean type's claim with an element type's would be folding claims
    /// about different polynomials, which is meaningless — so the groups
    /// never mix, exactly as `A₀` and `B₀` never mix within a group.
    pub per_element: Vec<(MatrixClaim, MatrixClaim)>,
    /// The wiring-sigma group (sigma v2 route B, wiring doc §sigma):
    /// per circuit SHAPE, one folded claim on that circuit's sigma table,
    /// keyed by the circuit digest. PER-DIGEST entries (wall 3): a tree
    /// mixes circuit shapes — a spine node's main fold inherits an
    /// FL-keyed prior entry beside its internal-keyed fresh claims — and
    /// each key's claims fold only among themselves because they name
    /// different permutations. Empty until a circuit proof's wiring
    /// assertion joins.
    pub sigma: Vec<([u8; 32], MatrixClaim)>,
    /// The jagged-layout group (the count win): per child SHAPE, one folded
    /// claim on that shape's layout table `J`, keyed by the digest whose
    /// circuit determines the heights — the same per-digest discipline as
    /// sigma. Empty until a deferred verify's jagged assertion joins.
    pub jagged: Vec<([u8; 32], MatrixClaim)>,
}

impl Accumulator {
    // Circuit-specific folds took 17.3 ms for an N=4 BLAKE3 batch. This raw
    // matrix pass took 15.3 ms because A and B use different row points.

    /// Discharge every accumulated claim against the raw matrices — the
    /// generic `O(Σ_t nnz_t)` root check, for callers without circuits.
    pub fn discharge(&self, mats: &[TypeMatrices<'_>]) -> bool {
        self.per_type.len() == mats.len()
            && self
                .per_type
                .iter()
                .zip(mats)
                .all(|((ca, cb), (a, b))| ca.check_direct(*a) && cb.check_direct(*b))
    }

    /// Discharge the element group against its `F128`-coefficient matrices.
    pub fn discharge_element(&self, mats: &[ElementMatrices<'_>]) -> bool {
        self.per_element.len() == mats.len()
            && self
                .per_element
                .iter()
                .zip(mats)
                .all(|((ca, cb), (a, b))| ca.check_direct(*a) && cb.check_direct(*b))
    }

    /// The jagged group's root discharge: each entry's folded claim against
    /// its own layout — `O(2^k + runs·m)` per key, once. The caller supplies
    /// the layout per digest (the heights are shape constants of that
    /// digest's circuit). `true` when nothing jagged was accumulated; an
    /// entry whose key is missing from `tables` fails, never skips.
    pub fn discharge_jagged(&self, tables: &[([u8; 32], &JaggedParams)]) -> bool {
        self.jagged.iter().all(|(d, c)| {
            tables
                .iter()
                .find(|(k, _)| k == d)
                .is_some_and(|(_, p)| discharge_jagged(c, &JaggedTable::from_params(p)))
        })
    }

    /// The sigma group's root discharge: each entry's folded claim against
    /// its own circuit's sigma table — `O(2^mu)` per key, once. The caller
    /// supplies the circuits; an entry whose digest is missing FAILS,
    /// never skips. `true` when nothing sigma was accumulated.
    pub fn discharge_sigma(&self, circuits: &[&Circuit]) -> bool {
        self.sigma.iter().all(|(d, claim)| {
            circuits.iter().find(|c| c.digest() == *d).is_some_and(|c| {
                bilinear(&claim.row, &claim.col, &SigmaAssertion::matrix(c)) == claim.value
            })
        })
    }
}

/// Per boolean type, the two folds `(A₀, B₀)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateProof {
    /// Per boolean type, the two folds `(A₀, B₀)`.
    pub folds: Vec<(FoldProof, FoldProof)>,
    /// Per element type, likewise.
    pub el_folds: Vec<(FoldProof, FoldProof)>,
    /// The sigma group's folds, one per digest key, in the caller's key
    /// order.
    pub sigma_folds: Vec<FoldProof>,
    /// The jagged group's folds, one per digest key, in the caller's key
    /// order.
    pub jagged_folds: Vec<FoldProof>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregateError {
    /// Nothing to fold.
    Empty,
    /// A prior accumulator names a different registry, or has the wrong
    /// number of per-type claims.
    RegistryMismatch,
    /// The proof is not shaped for this registry.
    Malformed,
    /// An assertion's reported matrix evaluations do not reproduce its
    /// target (`MatrixAssertion::check_reported`).
    Reported(LincheckError),
    /// Likewise on the element side.
    ReportedElement(ElementUnionError),
    /// A fold did not verify.
    Fold(FoldError),
    /// The accumulated claims did not hold against the real matrices.
    Discharge,
}

/// Claims to fold for one type, in a fixed order: the prior accumulators'
/// first (in the order given), then one per assertion. Prover and verifier
/// build this the same way, so the fold transcripts line up.
fn gather(
    registry: &Registry,
    assertions: &[MatrixAssertion],
    priors: &[&Accumulator],
    t: usize,
) -> (Vec<MatrixClaim>, Vec<MatrixClaim>) {
    let mut a = Vec::with_capacity(assertions.len() + priors.len());
    let mut b = Vec::with_capacity(assertions.len() + priors.len());
    for p in priors {
        a.push(p.per_type[t].0.clone());
        b.push(p.per_type[t].1.clone());
    }
    for assertion in assertions {
        let (ca, cb) = assertion.claims(registry).swap_remove(t);
        a.push(ca);
        b.push(cb);
    }
    (a, b)
}

/// The prior COUNT is one transcript byte — `0`/`1` coincide with the old
/// `is_some` flag, so pre-existing transcripts are unchanged.
fn bind<Ch: Challenger>(registry: &Registry, priors: &[&Accumulator], ch: &mut Ch) {
    assert!(priors.len() < 256, "at most 255 prior accumulators");
    ch.observe_label(DOMAIN);
    ch.observe_bytes(&registry.digest());
    ch.observe_bytes(&[priors.len() as u8]);
}

/// The shape checks every entry runs on its priors: same registry, and a
/// claim pair for every type of BOTH classes.
fn check_priors(
    registry: &Registry,
    priors: &[&Accumulator],
    n_element: usize,
) -> Result<(), AggregateError> {
    for p in priors {
        if p.registry_digest != registry.digest()
            || p.per_type.len() != registry.num_boolean()
            || p.per_element.len() != n_element
        {
            return Err(AggregateError::RegistryMismatch);
        }
    }
    Ok(())
}

/// Fold `assertions` (and `priors`, if this is not a leaf) into one
/// accumulator. `O(k · Σ_t nnz_t)` — the matrices are read here, natively,
/// so that no circuit ever has to.
pub fn prove_aggregate<Ch: Challenger>(
    registry: &Registry,
    mats: &[TypeMatrices<'_>],
    circuits: &[&dyn LincheckCircuit],
    assertions: &[MatrixAssertion],
    priors: &[&Accumulator],
    ch: &mut Ch,
) -> Result<(AggregateProof, Accumulator), AggregateError> {
    prove_aggregate_classes(
        registry,
        mats,
        circuits,
        assertions,
        &[],
        &[],
        &[],
        &[],
        priors,
        ch,
    )
}

/// [`prove_aggregate`] over BOTH classes: the boolean assertions against
/// their `GF(2)` matrices, the element ones against their `F128` matrices,
/// each group folding independently because they name different
/// polynomials.
#[allow(clippy::too_many_arguments)]
pub fn prove_aggregate_classes<Ch: Challenger>(
    registry: &Registry,
    mats: &[TypeMatrices<'_>],
    circuits: &[&dyn LincheckCircuit],
    assertions: &[MatrixAssertion],
    el_mats: &[ElementMatrices<'_>],
    el_assertions: &[(&UnionInstance<'_>, ElementAssertion)],
    sigma: &[SigmaKey<'_>],
    jagged: &[JaggedKeyProve<'_>],
    priors: &[&Accumulator],
    ch: &mut Ch,
) -> Result<(AggregateProof, Accumulator), AggregateError> {
    prove_aggregate_classes_with_grinding(
        registry,
        mats,
        circuits,
        assertions,
        el_mats,
        el_assertions,
        sigma,
        jagged,
        priors,
        FoldGrinding::disabled(),
        ch,
    )
}

/// [`prove_aggregate_classes`] with explicit grinding for every dense,
/// element, sigma, and jagged matrix fold.
#[allow(clippy::too_many_arguments)]
pub fn prove_aggregate_classes_with_grinding<Ch: Challenger>(
    registry: &Registry,
    mats: &[TypeMatrices<'_>],
    circuits: &[&dyn LincheckCircuit],
    assertions: &[MatrixAssertion],
    el_mats: &[ElementMatrices<'_>],
    el_assertions: &[(&UnionInstance<'_>, ElementAssertion)],
    sigma: &[SigmaKey<'_>],
    jagged: &[JaggedKeyProve<'_>],
    priors: &[&Accumulator],
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<(AggregateProof, Accumulator), AggregateError> {
    // Sigma never travels alone: a circuit proof's deferred verify yields
    // the matrix assertions AND the sigma assertion together, so the
    // boolean group always has work when the sigma group does.
    if assertions.is_empty() && priors.is_empty() {
        return Err(AggregateError::Empty);
    }
    if mats.len() != registry.num_boolean() {
        return Err(AggregateError::Malformed);
    }
    check_priors(registry, priors, el_mats.len())?;

    bind(registry, priors, ch);
    let mut folds = Vec::with_capacity(registry.num_boolean());
    let mut per_type = Vec::with_capacity(registry.num_boolean());
    for (t, (ma, mb)) in mats.iter().enumerate() {
        let (ca, cb) = gather(registry, assertions, priors, t);
        // The k·nnz work. ONE `fold_split` per claim yields the column
        // marginals for BOTH matrices, so the A- and B-folds share it — and
        // it runs on the type's tuned kernel rather than a generic sparse
        // walk. Claims share their row weight (A and B are reported at the
        // same point), so `ca` and `cb` agree here row-wise.
        let n_cols = 1usize << registry.boolean_types()[t].k_log;
        let mut combs_a = Vec::with_capacity(ca.len());
        let mut combs_b = Vec::with_capacity(cb.len());
        for (qa, qb) in ca.iter().zip(&cb) {
            let (xa, xb) = if qa.row == qb.row {
                circuits[t].fold_split(&qa.row.materialize())
            } else {
                (
                    col_marginal(ma, &qa.row.materialize(), n_cols),
                    col_marginal(mb, &qb.row.materialize(), n_cols),
                )
            };
            combs_a.push(xa);
            combs_b.push(xb);
        }
        let (pa, out_a) = prove_fold_with_grinding(*ma, &combs_a, &ca, grinding, ch);
        let (pb, out_b) = prove_fold_with_grinding(*mb, &combs_b, &cb, grinding, ch);
        folds.push((pa, pb));
        per_type.push((out_a, out_b));
    }

    // The element group: same fold, different matrices. Its claims are plain
    // eq ⊗ eq (no univariate skip), so no tuned column-marginal kernel
    // applies and the generic one is used.
    let mut el_folds = Vec::with_capacity(el_mats.len());
    let mut per_element = Vec::with_capacity(el_mats.len());
    for (t, (ma, mb)) in el_mats.iter().enumerate() {
        let (ca, cb) = gather_element(el_assertions, priors, t);
        let n_cols = ma.num_cols;
        let combs_a: Vec<Vec<F128>> = ca
            .iter()
            .map(|q| FoldMatrix::col_marginal(*ma, &q.row.materialize(), n_cols))
            .collect();
        let combs_b: Vec<Vec<F128>> = cb
            .iter()
            .map(|q| FoldMatrix::col_marginal(*mb, &q.row.materialize(), n_cols))
            .collect();
        let (pa, out_a) = prove_fold_with_grinding(*ma, &combs_a, &ca, grinding, ch);
        let (pb, out_b) = prove_fold_with_grinding(*mb, &combs_b, &cb, grinding, ch);
        el_folds.push((pa, pb));
        per_element.push((out_a, out_b));
    }

    let (sigma_folds, sigma_out) = fold_sigma_prove(sigma, priors, grinding, ch)?;
    let (jagged_folds, jagged_out) = fold_jagged_prove(jagged, priors, grinding, ch)?;

    Ok((
        AggregateProof {
            folds,
            el_folds,
            sigma_folds,
            jagged_folds,
        },
        Accumulator {
            registry_digest: registry.digest(),
            per_type,
            per_element,
            sigma: sigma_out,
            jagged: jagged_out,
        },
    ))
}

/// One sigma key: the circuit whose digest keys the group and whose table
/// the fold's prover reads, plus the fresh assertions about it. The
/// verifier's replay reads NO table — it needs the circuit only for the
/// digest and the shape checks.
pub type SigmaKey<'a> = (&'a Circuit, Vec<&'a SigmaAssertion>);

// THE ZERO CLAIM lives in the PUBLISHED BLOCK, not here (wall 3, the
// spine): an accumulator entry carries its own LIVE word as both weights'
// length-1 low, so a slot a node had no fold for is published as zeros and
// decodes as a claim with weights identically zero and value zero — true
// about every table, no table access, transcript-visible like any claim.
// That is what pins a fold group's claim COUNT without any capacity
// parameter here; an earlier `FoldCaps` sketch that padded claim lists
// inside this module was superseded by it and removed.

const DOMAIN_SIGMA_GROUP: &[u8] = b"flock-aggregate-sigma-v1";

/// The claims of one sigma key, in the fixed order every group uses: the
/// priors' entries with this digest first (in prior order), then every
/// canonical circuit-structure claim per assertion (shape-checked against
/// the circuit).
fn gather_sigma(
    circuit: &Circuit,
    asserts: &[&SigmaAssertion],
    priors: &[&Accumulator],
) -> Result<Vec<MatrixClaim>, AggregateError> {
    let digest = circuit.digest();
    let mut claims: Vec<MatrixClaim> = Vec::new();
    for p in priors {
        for (d, c) in &p.sigma {
            if *d == digest {
                claims.push(c.clone());
            }
        }
    }
    for a in asserts {
        let matrix = SigmaAssertion::matrix(circuit);
        let base_bits = (FoldMatrix::n_cols(&matrix) / 8).trailing_zeros() as usize;
        if a.nu != circuit.cells().nu()
            || a.rho.len() != circuit.cells().mu()
            || a.base_bits != base_bits
        {
            return Err(AggregateError::Malformed);
        }
        claims.extend(a.claims());
    }
    if claims.is_empty() {
        return Err(AggregateError::Malformed);
    }
    Ok(claims)
}

/// The sigma group's fold, prover side (route B, PER-DIGEST — wall 3): for
/// each key in the caller's order, [priors' entries with that digest |
/// fresh assertions] fold to one claim, the group bound by its label +
/// digest exactly as the jagged group binds. The caller's key list must
/// cover every prior entry's digest — a prior claim that cannot fold must
/// fail loudly, the rule every keyed group shares.
fn fold_sigma_prove<Ch: Challenger>(
    sigma: &[SigmaKey<'_>],
    priors: &[&Accumulator],
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<(Vec<FoldProof>, Vec<([u8; 32], MatrixClaim)>), AggregateError> {
    for p in priors {
        for (d, _) in &p.sigma {
            if !sigma.iter().any(|(c, _)| c.digest() == *d) {
                return Err(AggregateError::RegistryMismatch);
            }
        }
    }
    let mut folds = Vec::with_capacity(sigma.len());
    let mut out = Vec::with_capacity(sigma.len());
    for (i, (circuit, asserts)) in sigma.iter().enumerate() {
        let digest = circuit.digest();
        if sigma[..i].iter().any(|(c, _)| c.digest() == digest) {
            return Err(AggregateError::Malformed);
        }
        let claims = gather_sigma(circuit, asserts, priors)?;
        ch.observe_label(DOMAIN_SIGMA_GROUP);
        ch.observe_bytes(&digest);
        let m = SigmaAssertion::matrix(circuit);
        let n_cols = FoldMatrix::n_cols(&m);
        // Several circuit-structure claims share row points. Reuse each
        // expensive marginal instead of streaming the child table again.
        let mut combs: Vec<Vec<F128>> = Vec::with_capacity(claims.len());
        for (i, q) in claims.iter().enumerate() {
            if let Some(j) = claims[..i].iter().position(|p| p.row == q.row) {
                combs.push(combs[j].clone());
            } else {
                combs.push(FoldMatrix::col_marginal(&m, &q.row.materialize(), n_cols));
            }
        }
        let (pf, folded) = prove_fold_with_grinding(&m, &combs, &claims, grinding, ch);
        folds.push(pf);
        out.push((digest, folded));
    }
    Ok((folds, out))
}

/// The sigma group's replay — no table read anywhere.
fn fold_sigma_verify<Ch: Challenger>(
    sigma: &[SigmaKey<'_>],
    priors: &[&Accumulator],
    proofs: &[FoldProof],
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<Vec<([u8; 32], MatrixClaim)>, AggregateError> {
    for p in priors {
        for (d, _) in &p.sigma {
            if !sigma.iter().any(|(c, _)| c.digest() == *d) {
                return Err(AggregateError::RegistryMismatch);
            }
        }
    }
    if proofs.len() != sigma.len() {
        return Err(AggregateError::Malformed);
    }
    let mut out = Vec::with_capacity(sigma.len());
    for (i, ((circuit, asserts), pf)) in sigma.iter().zip(proofs).enumerate() {
        let digest = circuit.digest();
        if sigma[..i].iter().any(|(c, _)| c.digest() == digest) {
            return Err(AggregateError::Malformed);
        }
        let claims = gather_sigma(circuit, asserts, priors)?;
        ch.observe_label(DOMAIN_SIGMA_GROUP);
        ch.observe_bytes(&digest);
        let folded =
            verify_fold_with_grinding(&claims, pf, grinding, ch).map_err(AggregateError::Fold)?;
        out.push((digest, folded));
    }
    Ok(out)
}

/// The jagged group's per-key key list entry, prover side: the digest, the
/// layout it names, and the fresh assertions carrying claims about it.
pub type JaggedKeyProve<'a> = ([u8; 32], &'a JaggedParams, Vec<&'a JaggedAssertion>);

/// The verifier's view of a key: digest and fresh assertions — no layout
/// anywhere, which is what lets a circuit replay this half.
pub type JaggedKeyVerify<'a> = ([u8; 32], Vec<&'a JaggedAssertion>);

/// The jagged group's fold, prover side: for each key in the caller's order,
/// [priors' entries with that key, in prior order | fresh claims, in
/// assertion order] fold to one claim through the structure-aware jagged
/// fold. The caller's key list must cover every prior entry's key — a prior
/// claim that cannot fold must fail loudly, exactly as sigma's rule reads.
fn fold_jagged_prove<Ch: Challenger>(
    jagged: &[JaggedKeyProve<'_>],
    priors: &[&Accumulator],
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<(Vec<FoldProof>, Vec<([u8; 32], MatrixClaim)>), AggregateError> {
    for p in priors {
        for (d, _) in &p.jagged {
            if !jagged.iter().any(|(k, _, _)| k == d) {
                return Err(AggregateError::RegistryMismatch);
            }
        }
    }
    let mut folds = Vec::with_capacity(jagged.len());
    let mut out = Vec::with_capacity(jagged.len());
    for (i, (digest, params, asserts)) in jagged.iter().enumerate() {
        if jagged[..i].iter().any(|(k, _, _)| k == digest) {
            return Err(AggregateError::Malformed);
        }
        let table = JaggedTable::from_params(params);
        let claims = gather_jagged(digest, asserts, priors, table.k, table.m)?;
        ch.observe_label(DOMAIN_JAGGED_GROUP);
        ch.observe_bytes(digest);
        let (pf, folded) = prove_fold_jagged_with_grinding(&table, &claims, grinding, ch);
        folds.push(pf);
        out.push((*digest, folded));
    }
    Ok((folds, out))
}

/// The claims of one jagged key, in the fixed order every group uses.
/// `k`/`m` hold fresh assertions to the key's layout shape; inherited
/// claims are plain-eq by construction and their arity is checked by the
/// fold itself.
fn gather_jagged(
    digest: &[u8; 32],
    asserts: &[&JaggedAssertion],
    priors: &[&Accumulator],
    k: usize,
    m: usize,
) -> Result<Vec<JaggedClaim>, AggregateError> {
    let mut claims: Vec<JaggedClaim> = Vec::new();
    for p in priors {
        for (d, c) in &p.jagged {
            if d == digest {
                claims.push(JaggedClaim::from_folded(c).ok_or(AggregateError::Malformed)?);
            }
        }
    }
    for a in asserts {
        if a.k != k || a.m != m {
            return Err(AggregateError::Malformed);
        }
        claims.extend(a.claims().into_iter().cloned());
    }
    if claims.is_empty() {
        return Err(AggregateError::Malformed);
    }
    Ok(claims)
}

const DOMAIN_JAGGED_GROUP: &[u8] = b"flock-aggregate-jagged-v0";

/// The jagged group's replay — no layout read anywhere.
fn fold_jagged_verify<Ch: Challenger>(
    jagged: &[JaggedKeyVerify<'_>],
    priors: &[&Accumulator],
    proofs: &[FoldProof],
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<Vec<([u8; 32], MatrixClaim)>, AggregateError> {
    for p in priors {
        for (d, _) in &p.jagged {
            if !jagged.iter().any(|(k, _)| k == d) {
                return Err(AggregateError::RegistryMismatch);
            }
        }
    }
    if proofs.len() != jagged.len() {
        return Err(AggregateError::Malformed);
    }
    let mut out = Vec::with_capacity(jagged.len());
    for (i, ((digest, asserts), pf)) in jagged.iter().zip(proofs).enumerate() {
        if jagged[..i].iter().any(|(k, _)| k == digest) {
            return Err(AggregateError::Malformed);
        }
        // The key's shape: from a fresh assertion, else from an inherited
        // claim's own arity (folded claims are plain eq).
        let (k, m) = match asserts.first() {
            Some(a) => (a.k, a.m),
            None => {
                let c = priors
                    .iter()
                    .flat_map(|p| p.jagged.iter())
                    .find(|(d, _)| d == digest)
                    .ok_or(AggregateError::Malformed)?;
                let k = c.1.row.point.len();
                let n_col = c.1.col.point.len();
                if n_col < 2 || n_col % 2 != 0 {
                    return Err(AggregateError::Malformed);
                }
                (k, n_col / 2 - 1)
            }
        };
        let claims = gather_jagged(digest, asserts, priors, k, m)?;
        ch.observe_label(DOMAIN_JAGGED_GROUP);
        ch.observe_bytes(digest);
        let folded = verify_fold_jagged_with_grinding(k, &claims, pf, grinding, ch)
            .map_err(AggregateError::Fold)?;
        out.push((*digest, folded));
    }
    Ok(out)
}

/// Element claims to fold for one type: the priors' first (in order), then
/// one per assertion — the same fixed order the boolean side uses.
fn gather_element(
    assertions: &[(&UnionInstance<'_>, ElementAssertion)],
    priors: &[&Accumulator],
    t: usize,
) -> (Vec<MatrixClaim>, Vec<MatrixClaim>) {
    let mut a = Vec::with_capacity(assertions.len() + priors.len());
    let mut b = Vec::with_capacity(assertions.len() + priors.len());
    for p in priors {
        a.push(p.per_element[t].0.clone());
        b.push(p.per_element[t].1.clone());
    }
    for (union, assertion) in assertions {
        let (ca, cb) = assertion.claims(union).swap_remove(t);
        a.push(ca);
        b.push(cb);
    }
    (a, b)
}

/// Replay an aggregation. **Reads no matrix** — this is the half a merge
/// circuit arithmetises.
///
/// It also checks each assertion's reported evaluations against its target
/// ([`MatrixAssertion::check_reported`]), so a caller cannot forget the one
/// step that ties a proof's reported matrix work to the proof itself.
///
/// The accumulator it returns is conditional, like everything else on this
/// path: it is true only if the inputs were, and something must eventually
/// call [`Accumulator::discharge`].
pub fn verify_aggregate<Ch: Challenger>(
    registry: &Registry,
    assertions: &[MatrixAssertion],
    priors: &[&Accumulator],
    proof: &AggregateProof,
    ch: &mut Ch,
) -> Result<Accumulator, AggregateError> {
    verify_aggregate_classes(registry, assertions, &[], &[], &[], priors, proof, ch)
}

/// [`verify_aggregate`] over BOTH classes. Reads no matrix of either kind.
#[allow(clippy::too_many_arguments)]
pub fn verify_aggregate_classes<Ch: Challenger>(
    registry: &Registry,
    assertions: &[MatrixAssertion],
    el_assertions: &[(&UnionInstance<'_>, ElementAssertion)],
    sigma: &[SigmaKey<'_>],
    jagged: &[JaggedKeyVerify<'_>],
    priors: &[&Accumulator],
    proof: &AggregateProof,
    ch: &mut Ch,
) -> Result<Accumulator, AggregateError> {
    verify_aggregate_classes_with_grinding(
        registry,
        assertions,
        el_assertions,
        sigma,
        jagged,
        priors,
        proof,
        FoldGrinding::disabled(),
        ch,
    )
}

/// [`verify_aggregate_classes`] with explicit grinding for every fold.
#[allow(clippy::too_many_arguments)]
pub fn verify_aggregate_classes_with_grinding<Ch: Challenger>(
    registry: &Registry,
    assertions: &[MatrixAssertion],
    el_assertions: &[(&UnionInstance<'_>, ElementAssertion)],
    sigma: &[SigmaKey<'_>],
    jagged: &[JaggedKeyVerify<'_>],
    priors: &[&Accumulator],
    proof: &AggregateProof,
    grinding: FoldGrinding,
    ch: &mut Ch,
) -> Result<Accumulator, AggregateError> {
    // Sigma never travels alone: a circuit proof's deferred verify yields
    // the matrix assertions AND the sigma assertion together, so the
    // boolean group always has work when the sigma group does.
    if assertions.is_empty() && priors.is_empty() {
        return Err(AggregateError::Empty);
    }
    if proof.folds.len() != registry.num_boolean() {
        return Err(AggregateError::Malformed);
    }
    check_priors(registry, priors, proof.el_folds.len())?;
    for assertion in assertions {
        assertion
            .check_reported(registry)
            .map_err(AggregateError::Reported)?;
    }

    bind(registry, priors, ch);
    let mut per_type = Vec::with_capacity(registry.num_boolean());
    for (t, (pa, pb)) in proof.folds.iter().enumerate() {
        let (ca, cb) = gather(registry, assertions, priors, t);
        let out_a =
            verify_fold_with_grinding(&ca, pa, grinding, ch).map_err(AggregateError::Fold)?;
        let out_b =
            verify_fold_with_grinding(&cb, pb, grinding, ch).map_err(AggregateError::Fold)?;
        per_type.push((out_a, out_b));
    }

    // The element group, replayed the same way. Its fold count is the
    // number of element types, which `check_priors` already held every
    // accumulator to.
    for (union, assertion) in el_assertions {
        assertion
            .check_reported(union)
            .map_err(AggregateError::ReportedElement)?;
    }
    let mut per_element = Vec::with_capacity(proof.el_folds.len());
    for (t, (pa, pb)) in proof.el_folds.iter().enumerate() {
        let (ca, cb) = gather_element(el_assertions, priors, t);
        let out_a =
            verify_fold_with_grinding(&ca, pa, grinding, ch).map_err(AggregateError::Fold)?;
        let out_b =
            verify_fold_with_grinding(&cb, pb, grinding, ch).map_err(AggregateError::Fold)?;
        per_element.push((out_a, out_b));
    }

    // The sigma group, replayed the same way — the verifier reads no
    // sigma table here; the fold verifies against the CLAIMS alone, and
    // the table is only touched at the root discharge.
    let sigma_out = fold_sigma_verify(sigma, priors, &proof.sigma_folds, grinding, ch)?;

    let jagged_out = fold_jagged_verify(jagged, priors, &proof.jagged_folds, grinding, ch)?;

    Ok(Accumulator {
        registry_digest: registry.digest(),
        per_type,
        per_element,
        sigma: sigma_out,
        jagged: jagged_out,
    })
}
