//! The element PIOP over the union's **element region** — the large-field half
//! of a mixed proof.
//!
//! The class-major layout ([`crate::schedule::Registry::new`]) puts every
//! element slot inside one aligned subcube of the union address space, the
//! *element region*. This module runs the standalone element PIOP
//! ([`super::zerocheck`] + a per-slot-collapsing lincheck) over exactly that
//! region — over `E = M_elem − 7` **word** variables, since an element *is* a
//! word and there is no in-word structure to fold.
//!
//! ## Why a separate domain, not a "dead" sub-region
//!
//! The boolean union zerocheck passes `c = z` (the `C = I` convention aliases
//! the C side to the witness itself). On the element region `a = b = 0` — there
//! are no boolean constraints there — but `c = z ≠ 0`, so the honest global sum
//! `Σ eq·(ab − c)` would be NON-ZERO and an honest proof would fail. Any "skip
//! the element region" hack that makes the prover pass also removes those terms
//! from the verifier's statement, which *is* the disjoint-domain formulation.
//! So the two classes get two honest PIOPs over two disjoint domains.
//!
//! ## Region addressing
//!
//! Region word `w ∈ [0, 2^E)`. Element slot `t` occupies the aligned block
//! `[off_t, off_t + 2^{ν+κ_t})`, and inside it word `(c << ν) + j` is
//! (column `c`, row `j`) — the standalone rows-low BatchMajor layout. So
//!
//! ```text
//! w = (q_t << (ν + κ_t)) | (c << ν) | j,     q_t = off_t >> (ν + κ_t)
//! ```
//!
//! and the region splits as `[row (ν low) | col (κ_t) | slot prefix q_t]`.
//! Gaps — between slots, and the region's tail past the last slot — hold zeros
//! in all three tables, so they contribute nothing to either phase.
//!
//! ## Phase 1 — one zerocheck across every element slot
//!
//! `Σ_w eq(τ, w)·((Az+a)(w)·(Bz+b)(w) + z(w)) = 0` over the whole region:
//! plain [`super::zerocheck`] at `m_words = E`. The constraint system is
//! block-diagonal *across* slots as well as across rows, so the eq weight
//! batches the slots implicitly — the same trick the boolean union zerocheck
//! uses. Honest at any counts: real rows satisfy the relation, dummy rows are
//! zero and satisfy it by the `a_const ⊙ b_const = 0` rule, padding columns are
//! pinned to zero by their all-zero constraint rows, and gaps are zero.
//!
//! The affine constants' MLE has a closed form the verifier evaluates in
//! `O(Σ_t 2^{κ_t})` — see [`strip_constants`]: within a slot they are uniform
//! in the row coordinates, so those sum away by partition of unity.
//!
//! ## Phase 2 — one lincheck over the region's column domain
//!
//! The region weight `M̂(r, y)` factors per slot exactly as in the standalone
//! case, with the slot-prefix eq factor out front:
//!
//! ```text
//! Σ_y M̂(r,y)·ẑ(y) = Σ_t w_t · Σ_c comb_t[c] · ẑ_t(r_row, c),
//!   w_t       = eq(r[ν+κ_t..], q_t)              (a fixed public scalar)
//!   comb_t[c] = Σ_con eq(r[ν..ν+κ_t], con)·(A_0 + α·B_0)_t[con, c]
//! ```
//!
//! Placing `Comb[u] = w_t·comb_t[c]` and `G[u] = ẑ_region(r_row, u)` on the
//! region **column domain** `u = w >> ν` (length `2^{E−ν}`; `Comb` is zero on
//! gaps) turns that into ONE product sumcheck of `E − ν` rounds — the same
//! shape as the standalone one, so it runs the same shared core
//! ([`super::lincheck::column_sumcheck_prove`]). `G` is literally the row
//! collapse of the region witness, so the output claim
//! `ẑ_region(r_row, r'_col)` is an evaluation of the committed polynomial.
//!
//! As in the standalone lincheck the row coordinates are **reused**, not
//! resampled, so both element claims share `r_row`: the C-claim sits at
//! `(r_row, r_con-side of r)` — i.e. at `r` itself — and the LC claim at
//! `(r_row, r'_col)`.
//!
//! ## Claim points
//!
//! Both claims come out in region-word coordinates (length `E`) and are lifted
//! to the union word space by appending the region's frozen prefix
//! coordinates ([`crate::union::UnionInstance::element_prefix_coords`]) — a
//! fixed Boolean pattern, which is why they open as **packed-direct** claims
//! with a `Sparse` eq tensor and no ring-switching.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::lincheck::{column_sumcheck_prove, column_sumcheck_replay};
use super::{ElementTableType, Grinding, zerocheck};
use crate::challenger::Challenger;
use crate::field::F128;
use crate::matrix_fold::MatrixClaim;
use crate::union::{ElementSlotLayout, UnionInstance};
use crate::zerocheck::univariate_skip::build_eq;

/// Domain labels of the region PIOP's two phases — distinct from the
/// standalone single-table labels, so a region proof can never be replayed as
/// a standalone one.
const ZC_LABEL: &[u8] = b"flock-element-union-zc-v0";
const LC_LABEL: &[u8] = b"flock-element-union-lc-v0";

/// One element slot as the region PIOP sees it: its base block plus its
/// geometry inside the region. Built by [`region_slots`] from the registry.
#[derive(Clone, Copy, Debug)]
pub struct RegionSlot<'a> {
    pub ty: &'a ElementTableType,
    pub layout: ElementSlotLayout,
}

/// The element slots of a union instance, in slot order — the region PIOP's
/// statement (the base blocks and their region offsets; the counts bind
/// elsewhere, through `bind_statement` and the jagged heights).
pub fn region_slots<'r>(union: &UnionInstance<'r>) -> Vec<RegionSlot<'r>> {
    let nb = union.num_boolean();
    union.registry().types()[nb..]
        .iter()
        .zip(union.element_slot_layout())
        .map(|(ty, layout)| RegionSlot {
            ty: ty.element_type().expect("element_types are LargeField"),
            layout,
        })
        .collect()
}

/// The region's declared row support, per region COLUMN: `n_t` on slot `t`'s
/// used columns and 0 on its padding columns, on inter-slot gaps and on the
/// region tail — plus the affine constants those columns' dead rows carry.
///
/// This is what makes the element zerocheck's row rounds count-proportional
/// (see [`zerocheck::RowSupport`]). It is derived from the SAME per-slot counts
/// the jagged heights use, so prover and verifier cannot disagree about which
/// rows exist — and it changes no round message, so it is invisible to the
/// verifier entirely.
fn row_support(slots: &[RegionSlot<'_>], nu: usize, e_vars: usize) -> zerocheck::RowSupport {
    let n_cols = 1usize << (e_vars - nu);
    let mut live = vec![0usize; n_cols];
    let mut a_dead = vec![F128::ZERO; n_cols];
    let mut b_dead = vec![F128::ZERO; n_cols];
    for s in slots {
        let off = s.layout.column_offset(nu);
        // Only the `k` REAL columns carry data; the padding columns past them
        // are pinned to zero by their all-zero constraint rows, so they are
        // dead at every row (and their constants are zero anyway).
        for y in 0..s.ty.k() {
            live[off + y] = s.layout.n_t;
            a_dead[off + y] = s.ty.a_const()[y];
            b_dead[off + y] = s.ty.b_const()[y];
        }
    }
    zerocheck::RowSupport {
        live,
        a_dead,
        b_dead,
    }
}

/// The region PIOP's proof: the two phases' round messages and final values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    pub zerocheck: zerocheck::Proof,
    pub lincheck: super::lincheck::Proof,
}

/// The two witness evaluation claims a verified region PIOP leaves behind, in
/// **union word coordinates** (LSB-first, region point followed by the region's
/// frozen prefix bits) — ready to become packed-direct claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claims {
    /// `r ‖ prefix` — the C-claim point. Direct, because `C = I`.
    pub c_point: Vec<F128>,
    /// `ẑ(r ‖ prefix)`.
    pub c_value: F128,
    /// `(r_row, r'_col) ‖ prefix` — the lincheck's output point. Shares its
    /// row coordinates with `c_point`.
    pub lc_point: Vec<F128>,
    /// `ẑ(lc_point)`.
    pub lc_value: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    Zerocheck(zerocheck::VerifyError),
    /// Wrong number of lincheck round messages (expected `E − nu`).
    LincheckRoundCount {
        expected: usize,
        got: usize,
    },
    /// The element lincheck's α/round PoW witness vector has the wrong
    /// transcript-determined length.
    LincheckGrindingNonceCount {
        expected: usize,
        got: usize,
    },
    /// A lincheck PoW witness did not satisfy the difficulty that protects
    /// its following Fiat--Shamir challenge.
    LincheckGrindingInvalid {
        which: &'static str,
    },
    /// The lincheck's final consistency check `running == Ĉomb(r'_col)·z_eval`
    /// failed.
    LincheckFinalFailed,
}

/// Prove the region PIOP.
///
/// `z`, `pa`, `pb` are the element REGION's slices of the padded union
/// buffers: the committed element words, `A_0·z + a_const`, and
/// `B_0·z + b_const`, each `2^E` words, with gaps and dummy rows honestly
/// zero. All three stay borrowed and unwritten — the zerocheck ping-pongs
/// pooled scratch halves instead of folding `pa`/`pb` in place, so a caller
/// that built them with [`copy_live_region`] can hand them back to the zero
/// pool afterwards ([`give_back_live_region`]).
pub fn prove<C: Challenger>(
    union: &UnionInstance<'_>,
    z: &[F128],
    pa: &[F128],
    pb: &[F128],
    ch: &mut C,
) -> (Proof, Claims) {
    prove_with_grinding(union, z, pa, pb, Grinding::disabled(), ch)
}

pub fn prove_with_grinding<C: Challenger>(
    union: &UnionInstance<'_>,
    z: &[F128],
    pa: &[F128],
    pb: &[F128],
    grinding: Grinding,
    ch: &mut C,
) -> (Proof, Claims) {
    let slots = region_slots(union);
    let (nu, e_vars) = (union.n_log(), union.m_elem() - 7);
    assert!(
        !slots.is_empty(),
        "region PIOP needs at least one element slot"
    );
    assert_eq!(z.len(), 1usize << e_vars, "region witness length");

    // ---- Phase 1: one zerocheck over the whole region, with the declared row
    // support so the row rounds cost `O(Σ n_t · used_cols)` instead of
    // `O(2^E)`. Bit-identical to the dense path — see `zerocheck::RowSupport`.
    let support = row_support(&slots, nu, e_vars);
    let (zc_proof, zc) = zerocheck::prove_with_support_with_grinding(
        ZC_LABEL,
        pa,
        pb,
        z,
        e_vars,
        nu,
        Some(&support),
        grinding,
        ch,
    );
    let (va, vb, _, _) = strip_constants(&slots, nu, &zc);

    // ---- Phase 2: the column-domain lincheck with the per-slot collapse.
    ch.observe_label(LC_LABEL);
    let mut grinding_nonces = Vec::with_capacity(grinding.lincheck_nonce_count(e_vars - nu));
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let (nonce, alpha) = ch.grind_pow_and_sample_f128(bits);
        grinding_nonces.push(nonce);
        alpha
    } else {
        ch.sample_f128()
    };
    let mut comb = region_comb(&slots, nu, e_vars, alpha, &zc.r);
    let mut g = collapse_rows(z, &zc.r[..nu], Some(&support.live));
    debug_assert_eq!(
        comb.iter()
            .zip(&g)
            .fold(F128::ZERO, |a, (x, y)| a + *x * *y),
        va + alpha * vb,
        "region lincheck target must be the honest weighted inner product"
    );
    let (lc_rounds, bind_order) =
        column_sumcheck_prove(&mut comb, &mut g, grinding, &mut grinding_nonces, ch);
    debug_assert_eq!(g.len(), 1);
    // The matrix work, reported rather than left for the verifier to redo:
    // per slot the UNSCALED pair (⟨W,A_0⟩, ⟨W,B_0⟩) at the row point the
    // zerocheck fixed and the column point the sumcheck just bound. The
    // slot's two prefix weights are verifier scalars and stay out — what
    // accumulates must name the static matrix alone.
    let r_col: Vec<F128> = bind_order.iter().rev().copied().collect();
    let matrix_evals = slot_matrix_evals(&slots, nu, &zc.r, &r_col);
    let lc_proof = super::lincheck::Proof {
        rounds: lc_rounds,
        z_eval: g[0],
        matrix_evals,
        grinding_nonces,
    };

    let claims = assemble_claims(union, &zc, &bind_order, g[0]);
    (
        Proof {
            zerocheck: zc_proof,
            lincheck: lc_proof,
        },
        claims,
    )
}

/// Verify the region PIOP, walking the challenger in lockstep with [`prove`].
pub fn verify<C: Challenger>(
    union: &UnionInstance<'_>,
    proof: &Proof,
    ch: &mut C,
) -> Result<Claims, VerifyError> {
    verify_with_grinding(union, proof, Grinding::disabled(), ch)
}

pub fn verify_with_grinding<C: Challenger>(
    union: &UnionInstance<'_>,
    proof: &Proof,
    grinding: Grinding,
    ch: &mut C,
) -> Result<Claims, VerifyError> {
    let (claims, assertion) = verify_deferred_with_grinding(union, proof, grinding, ch)?;
    assertion.check_reported(union)?;
    Ok(claims)
}

/// [`verify`] with the matrix work left undischarged — the element class's
/// half of the accumulation route. Reads no base matrix.
pub fn verify_deferred<C: Challenger>(
    union: &UnionInstance<'_>,
    proof: &Proof,
    ch: &mut C,
) -> Result<(Claims, ElementAssertion), VerifyError> {
    verify_deferred_with_grinding(union, proof, Grinding::disabled(), ch)
}

pub fn verify_deferred_with_grinding<C: Challenger>(
    union: &UnionInstance<'_>,
    proof: &Proof,
    grinding: Grinding,
    ch: &mut C,
) -> Result<(Claims, ElementAssertion), VerifyError> {
    let slots = region_slots(union);
    let (nu, e_vars) = (union.n_log(), union.m_elem() - 7);
    assert!(
        !slots.is_empty(),
        "region PIOP needs at least one element slot"
    );
    let lc_rounds = e_vars - nu;
    if proof.lincheck.rounds.len() != lc_rounds {
        return Err(VerifyError::LincheckRoundCount {
            expected: lc_rounds,
            got: proof.lincheck.rounds.len(),
        });
    }

    let zc =
        zerocheck::verify_with_label_and_grinding(ZC_LABEL, e_vars, &proof.zerocheck, grinding, ch)
            .map_err(VerifyError::Zerocheck)?;
    let (va, vb, a_const_eval, b_const_eval) = strip_constants(&slots, nu, &zc);

    ch.observe_label(LC_LABEL);
    let expected_nonces = grinding.lincheck_nonce_count(lc_rounds);
    if proof.lincheck.grinding_nonces.len() != expected_nonces {
        return Err(VerifyError::LincheckGrindingNonceCount {
            expected: expected_nonces,
            got: proof.lincheck.grinding_nonces.len(),
        });
    }
    let mut nonce_idx = 0;
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let alpha = ch
            .verify_pow_and_sample_f128(proof.lincheck.grinding_nonces[nonce_idx], bits)
            .ok_or(VerifyError::LincheckGrindingInvalid { which: "alpha" })?;
        nonce_idx += 1;
        alpha
    } else {
        ch.sample_f128()
    };
    let (running, bind_order) = column_sumcheck_replay(
        va + alpha * vb,
        &proof.lincheck.rounds,
        grinding,
        &proof.lincheck.grinding_nonces,
        &mut nonce_idx,
        ch,
    )
    .map_err(|err| match err {
        super::lincheck::VerifyError::InvalidGrindingNonce { which } => {
            VerifyError::LincheckGrindingInvalid { which }
        }
        _ => VerifyError::LincheckFinalFailed,
    })?;
    debug_assert_eq!(nonce_idx, proof.lincheck.grinding_nonces.len());

    // Final check: `Ĉomb(r'_col) · z_eval`. Evaluated by the closed form —
    // per-slot comb MLE times the "the bound point addresses slot t" prefix-eq
    // factor — in `O(Σ_t (2^{κ_t} + nnz_t))`, never anything region-sized.
    let r_col: Vec<F128> = bind_order.iter().rev().copied().collect();
    // DEFERRED: `running = Ĉomb(r_col)·z_eval` is the only place the base
    // matrices enter, and Ĉomb is exactly the scaled combination of the
    // reported per-slot bilinear forms. The assertion keeps both sides and
    // checks `Ĉomb·z_eval == running` rather than dividing out `z_eval` —
    // which would be wrong precisely when `z_eval = 0` (an all-dummy
    // instance), where the relation constrains the matrices not at all.
    let assertion = ElementAssertion {
        alpha,
        r_con: zc.r[nu..].to_vec(),
        r_col,
        evals: proof.lincheck.matrix_evals.clone(),
        z_eval: proof.lincheck.z_eval,
        target: running,
        a_const_eval,
        b_const_eval,
    };

    Ok((
        assemble_claims(union, &zc, &bind_order, proof.lincheck.z_eval),
        assertion,
    ))
}

/// Lift the two region claims into union word coordinates by appending the
/// region's frozen prefix bits. Shared by prover and verifier — any divergence
/// would be a transcript break.
fn assemble_claims(
    union: &UnionInstance<'_>,
    zc: &zerocheck::Claim,
    bind_order: &[F128],
    lc_value: F128,
) -> Claims {
    let nu = union.n_log();
    let prefix = union.element_prefix_coords();
    let lift = |mut point: Vec<F128>| -> Vec<F128> {
        point.extend_from_slice(&prefix);
        point
    };
    // The lincheck reuses `r`'s row coordinates and resamples only the column
    // half; `bind_order` bound the TOP column variable first, so reversing it
    // puts the column point LSB-first.
    let mut lc = Vec::with_capacity(zc.r.len());
    lc.extend_from_slice(&zc.r[..nu]);
    lc.extend(bind_order.iter().rev().copied());
    debug_assert_eq!(lc.len(), zc.r.len());
    Claims {
        c_point: lift(zc.r.clone()),
        c_value: zc.ec,
        lc_point: lift(lc),
        lc_value,
    }
}

/// Turn the zerocheck's `(ea, eb)` — claims on `A_0 z + a_const` and
/// `B_0 z + b_const` at the region point `r` — into the pure `Âz(r)`, `B̂z(r)`
/// claims the lincheck reduces, by subtracting (char 2: adding) the constants'
/// closed-form MLEs.
///
/// Within slot `t` the constants are uniform in the row coordinates, so those
/// sum away by partition of unity and the slot contributes
/// `eq(r[ν+κ_t..], q_t) · Σ_c eq(r[ν..ν+κ_t], c)·a_const_t[c]`. Gaps
/// contribute nothing (no slot owns them). `O(Σ_t 2^{κ_t})`.
fn strip_constants(
    slots: &[RegionSlot<'_>],
    nu: usize,
    zc: &zerocheck::Claim,
) -> (F128, F128, F128, F128) {
    let mut a_sum = F128::ZERO;
    let mut b_sum = F128::ZERO;
    for s in slots {
        let kappa = s.layout.kappa;
        let eq_con = build_eq(&zc.r[nu..nu + kappa]);
        let w = eq_prefix_weight(&zc.r[nu + kappa..], s.layout.region_prefix(nu));
        let dot = |c: &[F128]| -> F128 {
            eq_con
                .iter()
                .zip(c)
                .fold(F128::ZERO, |acc, (e, v)| acc + *e * *v)
        };
        a_sum += w * dot(s.ty.a_const());
        b_sum += w * dot(s.ty.b_const());
    }
    (zc.ea + a_sum, zc.eb + b_sum, a_sum, b_sum)
}

/// `Π_j eq(coords[j], bit_j(bits))` — the eq factor freezing `coords` to the
/// Boolean pattern `bits`, LSB-first. The region-prefix weights `w_t` and the
/// verifier's bound-point subcube factors are both instances. (Mirror of
/// `crate::lincheck::union::eq_prefix_weight`, over word coordinates.)
fn eq_prefix_weight(coords: &[F128], bits: usize) -> F128 {
    debug_assert!(bits < 1usize << coords.len() || coords.is_empty() && bits == 0);
    let mut acc = F128::ONE;
    for (j, &x) in coords.iter().enumerate() {
        acc *= if (bits >> j) & 1 == 1 {
            x
        } else {
            F128::ONE + x
        };
    }
    acc
}

/// One slot's α-batched comb `comb_t[c] = Σ_con eq_con[con]·(A_0 + α·B_0)[con, c]`
/// — the eq-weighted column marginal of the base block, `O(nnz)`. Both sides
/// call this, so there is one definition to disagree with.
fn slot_comb(ty: &ElementTableType, alpha: F128, eq_con: &[F128]) -> Vec<F128> {
    debug_assert_eq!(eq_con.len(), ty.width());
    let mut comb = vec![F128::ZERO; ty.width()];
    for (m, scale) in [(ty.a_0(), F128::ONE), (ty.b_0(), alpha)] {
        for (con, row) in m.rows.iter().enumerate() {
            if row.is_empty() {
                continue;
            }
            let w = scale * eq_con[con];
            for &(c, coeff) in row {
                comb[c] += w * coeff;
            }
        }
    }
    comb
}

/// The dense region-column weight vector `Comb[u]`: each slot's `w_t`-scaled
/// comb at its aligned column block, zero on gaps. Length `2^{E−ν}`.
fn region_comb(
    slots: &[RegionSlot<'_>],
    nu: usize,
    e_vars: usize,
    alpha: F128,
    r: &[F128],
) -> Vec<F128> {
    let mut out = vec![F128::ZERO; 1usize << (e_vars - nu)];
    for s in slots {
        let kappa = s.layout.kappa;
        let comb = slot_comb(s.ty, alpha, &build_eq(&r[nu..nu + kappa]));
        let w = eq_prefix_weight(&r[nu + kappa..], s.layout.region_prefix(nu));
        let off = s.layout.column_offset(nu);
        for (dst, &src) in out[off..off + comb.len()].iter_mut().zip(&comb) {
            *dst = w * src;
        }
    }
    out
}

/// Per slot, `(⟨eq_con ⊗ eq_col, A_0⟩, ⟨eq_con ⊗ eq_col, B_0⟩)` — unscaled.
///
/// Element weights are plain `eq ⊗ eq`: unlike the boolean class there is no
/// univariate skip, so no `λ` factor appears. That makes an element claim
/// the simplest shape [`crate::matrix_fold::Weight`] has.
fn slot_matrix_evals(
    slots: &[RegionSlot<'_>],
    nu: usize,
    r: &[F128],
    r_col: &[F128],
) -> Vec<(F128, F128)> {
    slots
        .iter()
        .map(|s| {
            let kappa = s.layout.kappa;
            let row = crate::matrix_fold::Weight::eq(r[nu..nu + kappa].to_vec());
            let col = crate::matrix_fold::Weight::eq(r_col[..kappa].to_vec());
            (
                crate::matrix_fold::bilinear(&row, &col, s.ty.a_0()),
                crate::matrix_fold::bilinear(&row, &col, s.ty.b_0()),
            )
        })
        .collect()
}

/// The element class's undischarged matrix work — its counterpart of
/// [`crate::lincheck::MatrixAssertion`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementAssertion {
    pub alpha: F128,
    /// Row point (`eq` over the constraint coordinates) and column point.
    pub r_con: Vec<F128>,
    pub r_col: Vec<F128>,
    /// Per slot, the reported `(⟨W,A_0⟩, ⟨W,B_0⟩)`.
    pub evals: Vec<(F128, F128)>,
    /// The witness evaluation the combination multiplies.
    pub z_eval: F128,
    /// What `Ĉomb·z_eval` must equal — the sumcheck's running claim.
    pub target: F128,
    /// Evaluations of the registry-static affine constant vectors at
    /// `r_con`. They are folded into the circuit-structure accumulator on
    /// recursive paths, binding the zerocheck-to-lincheck strip.
    pub a_const_eval: F128,
    pub b_const_eval: F128,
}

impl ElementAssertion {
    /// The per-slot `(A_0, B_0)` claims, ready for
    /// [`crate::matrix_fold::prove_fold`].
    pub fn claims(&self, union: &UnionInstance<'_>) -> Vec<(MatrixClaim, MatrixClaim)> {
        region_slots(union)
            .iter()
            .zip(&self.evals)
            .map(|(s, &(va, vb))| {
                let kappa = s.layout.kappa;
                let row = crate::matrix_fold::Weight::eq(self.r_con[..kappa].to_vec());
                let col = crate::matrix_fold::Weight::eq(self.r_col[..kappa].to_vec());
                (
                    MatrixClaim {
                        row: row.clone(),
                        col: col.clone(),
                        value: va,
                    },
                    MatrixClaim {
                        row,
                        col,
                        value: vb,
                    },
                )
            })
            .collect()
    }

    /// Check the reported values reproduce the target — scalars only, no
    /// matrix read.
    pub fn check_reported(&self, union: &UnionInstance<'_>) -> Result<(), VerifyError> {
        let nu = union.n_log();
        let slots = region_slots(union);
        if self.evals.len() != slots.len() {
            return Err(VerifyError::LincheckFinalFailed);
        }
        let mut acc = F128::ZERO;
        for (s, &(va, vb)) in slots.iter().zip(&self.evals) {
            let kappa = s.layout.kappa;
            let w_r = eq_prefix_weight(&self.r_con[kappa..], s.layout.region_prefix(nu));
            let w_col = eq_prefix_weight(&self.r_col[kappa..], s.layout.region_prefix(nu));
            acc += w_r * w_col * (va + self.alpha * vb);
        }
        if acc * self.z_eval != self.target {
            return Err(VerifyError::LincheckFinalFailed);
        }
        Ok(())
    }
}

/// Closed-form `Ĉomb(r_col)` — what the verifier used to evaluate inline,
/// now the differential oracle for the reported path
/// (`reported_evals_match_the_inline_comb`). Reads the base matrices, which
/// is exactly why the verifier no longer calls it.
#[cfg(test)]
fn region_comb_at_oracle(
    slots: &[RegionSlot<'_>],
    nu: usize,
    alpha: F128,
    r: &[F128],
    r_col: &[F128],
) -> F128 {
    region_comb_at(slots, nu, alpha, r, r_col)
}

/// Closed-form `Ĉomb(r_col)` — the verifier's counterpart of
/// [`region_comb`], without materializing anything region-sized: each slot
/// contributes its own comb MLE at the bound point times the subcube prefix-eq
/// factor "the bound point addresses slot `t`".
#[cfg_attr(not(test), allow(dead_code))]
fn region_comb_at(
    slots: &[RegionSlot<'_>],
    nu: usize,
    alpha: F128,
    r: &[F128],
    r_col: &[F128],
) -> F128 {
    let mut acc = F128::ZERO;
    for s in slots {
        let kappa = s.layout.kappa;
        let comb = slot_comb(s.ty, alpha, &build_eq(&r[nu..nu + kappa]));
        let w_r = eq_prefix_weight(&r[nu + kappa..], s.layout.region_prefix(nu));
        // The bound COLUMN point must address this slot's block: its low
        // `kappa` coords index the comb, its high coords freeze to `q_t`.
        let w_col = eq_prefix_weight(&r_col[kappa..], s.layout.region_prefix(nu));
        let eq_col = build_eq(&r_col[..kappa]);
        let inner = comb
            .iter()
            .zip(&eq_col)
            .fold(F128::ZERO, |a, (c, e)| a + *c * *e);
        acc += w_r * w_col * inner;
    }
    acc
}

/// The row collapse: `G[u] = Σ_j eq(r_row, j)·z[(u << ν) + j] = ẑ_region(r_row, u)`,
/// a length-`2^{E−ν}` vector. Rows-low layout makes each column's rows
/// contiguous, so this is one chunked dot product over the whole region in a
/// single pass — and it is the row collapse of the REGION, gaps included
/// (where it is zero), which is what makes the output claim an evaluation of
/// the committed polynomial.
fn collapse_rows(z: &[F128], r_row: &[F128], live: Option<&[usize]>) -> Vec<F128> {
    let eq_row = crate::pcs::ring_switch::build_eq_parallel(r_row);
    let rows = eq_row.len();
    debug_assert_eq!(z.len() % rows, 0);
    z.par_chunks(rows)
        .enumerate()
        .map(|(c, col)| {
            // Count-proportional: the witness is zero past the declared rows,
            // so truncating the dot product drops only zero terms — the sum is
            // bit-identical.
            let n = live.map_or(rows, |l| l[c]);
            col[..n]
                .iter()
                .zip(&eq_row)
                .fold(F128::ZERO, |acc, (v, e)| acc + *v * *e)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Witness assembly
// ---------------------------------------------------------------------------

/// Fill one element slot's block of the padded union `(z, a, b)` buffers:
/// `generate` writes the slot's `2^{ν+κ}` committed element words (BatchMajor,
/// `z[(c << ν) + row]`, zero on dummy rows and padding columns), then
/// [`ElementTableType::affine_products_into`] derives `pa`/`pb` in place by
/// sparse gather.
///
/// `live = Some(n_t)` derives only the declared rows' `pa`/`pb`, leaving dead
/// rows unwritten — pass it exactly when [`dead_rows_unread`] holds (the
/// region zerocheck will run its sparse path, which substitutes dead halves
/// from `RowSupport::{a,b}_dead` analytically and never reads them; pinned
/// byte-identical by `dummy_row_is_structurally_invisible_under_the_union`).
///
/// This is the union's in-place witness path for element slots — the
/// `SlotWitnessDest` counterpart of the boolean drivers, minus the lincheck
/// stripe (the element lincheck folds the committed region directly, so there
/// is no separate stripe copy).
pub fn fill_slot(
    ty: &ElementTableType,
    nu: usize,
    live: Option<usize>,
    z: &mut [F128],
    pa: &mut [F128],
    pb: &mut [F128],
    generate: impl FnOnce(&mut [F128]),
) {
    let words = ty.width() << nu;
    assert_eq!(z.len(), words, "element slot z block");
    generate(z);
    ty.affine_products_into(z, nu, live, pa, pb);
}

/// Whether the region zerocheck will take its SPARSE row rounds — the arm
/// that never reads dead rows of `pa`/`pb` (their values are substituted
/// analytically from the per-column constants). This is the gate for
/// live-only `pa`/`pb` derivation ([`fill_slot`]'s `live`) and for the
/// live-span region copy ([`copy_live_region`]): both are byte-identical to
/// the full versions exactly when this holds, because the words they leave
/// unwritten are unread everywhere.
pub fn dead_rows_unread(union: &UnionInstance<'_>) -> bool {
    let slots = region_slots(union);
    if slots.is_empty() {
        return false;
    }
    let (nu, e_vars) = (union.n_log(), union.m_elem() - 7);
    row_support(&slots, nu, e_vars).worth_skipping(nu)
}

/// The element region's `a`/`b` tables copied LIVE SPANS ONLY into
/// lazy-zeroed buffers: per slot, per used column, rows `[0, n_t)`. Only
/// valid under [`dead_rows_unread`] — the sparse zerocheck reads live
/// prefixes and substitutes dead values analytically, so the zeros this
/// leaves where the full copy would have carried constants (or a dirty
/// buffer's stale words) are never observed. `a_region`/`b_region` are the
/// element word ranges of the padded union buffers.
pub fn copy_live_region(
    union: &UnionInstance<'_>,
    a_region: &[F128],
    b_region: &[F128],
) -> (Vec<F128>, Vec<F128>) {
    let words = 1usize << (union.m_elem() - 7);
    assert_eq!(a_region.len(), words, "element region a length");
    assert_eq!(b_region.len(), words, "element region b length");
    // Zero-pool buffers: all-zero without a memset, and — since [`prove`]
    // never writes its inputs — returnable via [`give_back_live_region`]
    // with exactly the spans below declared dirty.
    let mut pa = crate::scratch::take_zeroed_f128(words);
    let mut pb = crate::scratch::take_zeroed_f128(words);
    for span in live_spans(union) {
        pa[span.clone()].copy_from_slice(&a_region[span.clone()]);
        pb[span.clone()].copy_from_slice(&b_region[span]);
    }
    (pa, pb)
}

/// The element region's live word spans — per slot, per used column, rows
/// `[0, n_t)`: exactly what [`copy_live_region`] writes, and therefore the
/// dirty ranges its give-back must re-zero.
fn live_spans(union: &UnionInstance<'_>) -> Vec<core::ops::Range<usize>> {
    let nu = union.n_log();
    let mut spans = Vec::new();
    for s in region_slots(union) {
        let off = s.layout.column_offset(nu);
        let n_t = s.layout.n_t;
        for y in 0..s.ty.k() {
            let base = (off + y) << nu;
            spans.push(base..base + n_t);
        }
    }
    spans
}

/// Return a [`copy_live_region`] pair to the zero pool, re-zeroing exactly
/// the live spans that function wrote. Only valid for buffers it produced
/// (zero outside the spans) that [`prove`] borrowed — anything else would
/// poison the pool's all-zero invariant (debug builds verify it outright).
pub fn give_back_live_region(union: &UnionInstance<'_>, pa: Vec<F128>, pb: Vec<F128>) {
    let spans = live_spans(union);
    crate::scratch::give_zeroed_f128(pa, &spans);
    crate::scratch::give_zeroed_f128(pb, &spans);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::element_r1cs::tests::{mixed_gate, mixed_witness, mult_gate, mult_witness};
    use crate::element_r1cs::{ElementTableBuilder, ElementTableType};
    use crate::schedule::{Registry, TableType};
    use crate::test_rng::Rng;
    use crate::zerocheck::multilinear::{eq_eval, fold_in_place_single};
    use std::sync::Arc;

    /// Direct MLE evaluation at `point`, binding the low variable first.
    fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
        let mut t = table.to_vec();
        for &p in point {
            fold_in_place_single(&mut t, p);
        }
        t[0]
    }

    fn bits(v: usize, n: usize) -> Vec<F128> {
        (0..n)
            .map(|i| {
                if (v >> i) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            })
            .collect()
    }

    /// A boolean stub type, for mixed registries whose boolean half is never
    /// proven here (only its address space matters to the layout).
    fn bool_ty(k_log: usize, useful_bits: usize) -> TableType {
        TableType {
            k_log,
            useful_bits,
            a_0: crate::r1cs::SparseBinaryMatrix {
                num_rows: 0,
                num_cols: 0,
                rows: Vec::new(),
            },
            b_0: crate::r1cs::SparseBinaryMatrix {
                num_rows: 0,
                num_cols: 0,
                rows: Vec::new(),
            },
            c_0: crate::r1cs::SparseBinaryMatrix {
                num_rows: 0,
                num_cols: 0,
                rows: Vec::new(),
            },
            const_pin: None,
            class: crate::schedule::TableClass::Boolean,
            io_schema: Vec::new(),
        }
    }

    /// One element type + its satisfying witness generator.
    struct Case {
        ty: Arc<ElementTableType>,
        make: Box<dyn Fn(usize, usize, &mut Rng) -> Vec<F128>>,
    }

    fn mult_case(kappa: usize) -> Case {
        let ty = Arc::new(mult_gate(kappa));
        let t = ty.clone();
        Case {
            ty,
            make: Box::new(move |nu, n, rng| mult_witness(&t, nu, n, rng)),
        }
    }

    /// An all-free-wire block of width `2^kappa` — the tautology row
    /// `(z_y)(1) = z_y`, valid at ANY kappa (unlike `mult_gate`, which needs a
    /// product column). Its witness is arbitrary on declared rows.
    fn free_case(kappa: usize) -> Case {
        let mut b = ElementTableBuilder::new(kappa);
        for y in 0..1usize << kappa {
            b.free_wire(y);
        }
        let ty = Arc::new(b.build().expect("free wires are valid"));
        let width = ty.width();
        Case {
            ty,
            make: Box::new(move |nu, n, rng| {
                let mut z = vec![F128::ZERO; width << nu];
                for c in 0..width {
                    for j in 0..n {
                        z[(c << nu) + j] = rng.f128();
                    }
                }
                z
            }),
        }
    }

    fn mixed_case(rng: &mut Rng) -> Case {
        let ty = Arc::new(mixed_gate(rng));
        let t = ty.clone();
        Case {
            ty,
            make: Box::new(move |nu, n, rng| mixed_witness(&t, nu, n, rng)),
        }
    }

    /// Build a registry over the given element cases (plus optional boolean
    /// stubs), assemble the padded union `(z, pa, pb)` buffers via
    /// [`fill_slot`], and return everything the region PIOP needs.
    struct Harness {
        registry: Registry,
        z: Vec<F128>,
        pa: Vec<F128>,
        pb: Vec<F128>,
        counts: Vec<usize>,
    }

    fn build(
        bools: Vec<(usize, usize)>,
        cases: &[Case],
        nu: usize,
        counts_elem: &[usize],
        rng: &mut Rng,
    ) -> Harness {
        let mut types: Vec<TableType> = bools
            .iter()
            .map(|&(k_log, useful)| bool_ty(k_log, useful))
            .collect();
        let n_bool = types.len();
        for c in cases {
            types.push(TableType::element(c.ty.clone()));
        }
        let registry = Registry::new(types, nu);
        assert_eq!(registry.num_boolean(), n_bool);
        // Element cases were pushed in the order `Registry::new` sorts them
        // into only if they are already area-descending; assert rather than
        // guess, so the test's witness↔slot pairing is pinned.
        for (i, c) in cases.iter().enumerate() {
            assert_eq!(
                registry.element_types()[i].k_log,
                c.ty.kappa() + 7,
                "element case {i} did not land in slot {i}"
            );
        }
        let mut counts = vec![0usize; n_bool];
        counts.extend_from_slice(counts_elem);
        let union = UnionInstance::new(&registry, counts.clone());

        let words = union.packed_len();
        let mut z = vec![F128::ZERO; words];
        let mut pa = vec![F128::ZERO; words];
        let mut pb = vec![F128::ZERO; words];
        for (t, c) in cases.iter().enumerate() {
            let slot = n_bool + t;
            let range = union.slot_word_range(slot);
            let n_t = counts_elem[t];
            let w = c.ty.width() << nu;
            assert_eq!(range.len(), w);
            let rows = (c.make)(nu, n_t, rng);
            assert!(
                c.ty.satisfies(&rows, nu, n_t),
                "generated witness must satisfy"
            );
            fill_slot(
                &c.ty,
                nu,
                None,
                &mut z[range.clone()],
                &mut pa[range.clone()],
                &mut pb[range.clone()],
                |dst| dst.copy_from_slice(&rows),
            );
        }
        Harness {
            registry,
            z,
            pa,
            pb,
            counts,
        }
    }

    /// The region view of the three buffers.
    fn region<'a>(union: &UnionInstance<'_>, v: &'a [F128]) -> &'a [F128] {
        let r = union.element_word_range();
        &v[r]
    }

    /// **The honesty anchor.** For a satisfying witness the region zerocheck's
    /// eq-weighted sum is ZERO at any τ — across slots, across gaps, at partial
    /// counts, with padding columns. If the region assembly or the padding
    /// convention were wrong this fails before any sumcheck runs.
    #[test]
    fn satisfying_region_has_zero_eq_weighted_sum() {
        let mut rng = Rng::new(0xE1E_2E20);
        let shapes: Vec<(Vec<(usize, usize)>, Vec<Case>, usize, Vec<usize>)> = vec![
            // One element slot filling the region (no prefix at all).
            (vec![], vec![mult_case(2)], 3, vec![5]),
            // Two element slots of different widths → real slot prefixes.
            (
                vec![],
                vec![mixed_case(&mut rng), mult_case(2)],
                3,
                vec![6, 3],
            ),
            // Mixed with a boolean slot in front → a region prefix too.
            (vec![(10, 700)], vec![mult_case(2)], 3, vec![0]),
            (vec![(10, 700), (9, 300)], vec![mult_case(3)], 2, vec![4]),
        ];
        for (bools, cases, nu, counts_elem) in shapes {
            let h = build(bools, &cases, nu, &counts_elem, &mut rng);
            let union = UnionInstance::new(&h.registry, h.counts.clone());
            let (z, pa, pb) = (
                region(&union, &h.z),
                region(&union, &h.pa),
                region(&union, &h.pb),
            );
            let e_vars = union.m_elem() - 7;
            assert_eq!(z.len(), 1usize << e_vars);
            let tau: Vec<F128> = (0..e_vars).map(|_| rng.f128()).collect();
            let mut acc = F128::ZERO;
            for w in 0..z.len() {
                acc += eq_eval(&tau, &bits(w, e_vars)) * (pa[w] * pb[w] + z[w]);
            }
            assert_eq!(acc, F128::ZERO, "counts {counts_elem:?} nu={nu}");
        }
    }

    /// The lincheck's load-bearing identity: the collapsed inner product
    /// `Σ_u Comb[u]·G[u]` equals the brute-force block-diagonal weighted sum
    /// over the region, AND equals `va + α·vb` for the true `Âz(r)`, `B̂z(r)`.
    /// The sumcheck only ever sees `E − ν` variables, so nothing else would
    /// catch an error in the per-slot collapse.
    #[test]
    fn region_collapse_matches_brute_force() {
        let mut rng = Rng::new(0xC0_11A_25E);
        for (bools, cases, nu, counts_elem) in [
            (vec![], vec![mult_case(2)], 2, vec![3]),
            (
                vec![(10, 700)],
                vec![mult_case(2), free_case(1)],
                2,
                vec![0, 2, 1],
            ),
        ] {
            let cases: Vec<Case> = cases;
            let n_elem = cases.len();
            let counts_elem = &counts_elem[counts_elem.len() - n_elem..];
            let h = build(bools, &cases, nu, counts_elem, &mut rng);
            let union = UnionInstance::new(&h.registry, h.counts.clone());
            let slots = region_slots(&union);
            let e_vars = union.m_elem() - 7;
            // Random z: the identity is about the WEIGHTS, not satisfaction.
            let z: Vec<F128> = (0..1usize << e_vars).map(|_| rng.f128()).collect();
            let r: Vec<F128> = (0..e_vars).map(|_| rng.f128()).collect();
            let alpha = rng.f128();

            let comb = region_comb(&slots, nu, e_vars, alpha, &r);
            let g = collapse_rows(&z, &r[..nu], None);
            let collapsed = comb
                .iter()
                .zip(&g)
                .fold(F128::ZERO, |a, (x, y)| a + *x * *y);

            // Brute force from the UNFACTORED definition: walk every (x, y)
            // pair of the block-diagonal region system explicitly.
            let mut bf = F128::ZERO;
            let mut az = vec![F128::ZERO; z.len()];
            let mut bz = vec![F128::ZERO; z.len()];
            for s in &slots {
                let width = s.ty.width();
                let off = s.layout.region_word_offset;
                for j in 0..1usize << nu {
                    let zj: Vec<F128> = (0..width).map(|c| z[off + (c << nu) + j]).collect();
                    for y in 0..width {
                        az[off + (y << nu) + j] = s.ty.a_0().row_dot(y, &zj);
                        bz[off + (y << nu) + j] = s.ty.b_0().row_dot(y, &zj);
                    }
                }
            }
            for x in 0..z.len() {
                bf += eq_eval(&r, &bits(x, e_vars)) * (az[x] + alpha * bz[x]);
            }
            assert_eq!(collapsed, bf, "collapsed weights vs brute force");
            // …and that is exactly `Âz(r) + α·B̂z(r)`.
            assert_eq!(collapsed, mle_eval(&az, &r) + alpha * mle_eval(&bz, &r));
        }
    }

    /// `collapse_rows` really is the region witness restricted to `r_row`, so
    /// the lincheck's output claim is an evaluation of the COMMITTED
    /// polynomial: `G[u] = ẑ_region(r_row, u)` at every boolean `u`, hence
    /// `Ĝ(r_col) = ẑ_region(r_row, r_col)`.
    #[test]
    fn collapse_is_the_region_witness_at_r_row() {
        let mut rng = Rng::new(0x40_11AB5E);
        for (nu, cols) in [(3usize, 4usize), (2, 8), (4, 2)] {
            let z: Vec<F128> = (0..cols << nu).map(|_| rng.f128()).collect();
            let r_row: Vec<F128> = (0..nu).map(|_| rng.f128()).collect();
            let g = collapse_rows(&z, &r_row, None);
            assert_eq!(g.len(), cols);
            let col_vars = cols.trailing_zeros() as usize;
            for u in 0..cols {
                let mut point = r_row.clone();
                point.extend(bits(u, col_vars));
                assert_eq!(g[u], mle_eval(&z, &point), "u={u}");
            }
            let r_col: Vec<F128> = (0..col_vars).map(|_| rng.f128()).collect();
            let mut point = r_row.clone();
            point.extend_from_slice(&r_col);
            assert_eq!(mle_eval(&g, &r_col), mle_eval(&z, &point));
        }
    }

    /// The verifier's closed-form `Ĉomb(r_col)` matches folding the dense
    /// region-column vector — the multi-slot placement algebra, isolated.
    #[test]
    fn region_comb_closed_form_matches_dense_fold() {
        let mut rng = Rng::new(0xC0_3B_E1);
        let cases = vec![mixed_case(&mut rng), mult_case(2), mult_case(2)];
        let h = build(vec![(10, 700)], &cases, 3, &[2, 3, 4], &mut rng);
        let union = UnionInstance::new(&h.registry, h.counts.clone());
        let slots = region_slots(&union);
        let (nu, e_vars) = (union.n_log(), union.m_elem() - 7);
        for _ in 0..8 {
            let r: Vec<F128> = (0..e_vars).map(|_| rng.f128()).collect();
            let r_col: Vec<F128> = (0..e_vars - nu).map(|_| rng.f128()).collect();
            let alpha = rng.f128();
            let dense = region_comb(&slots, nu, e_vars, alpha, &r);
            assert_eq!(
                region_comb_at(&slots, nu, alpha, &r, &r_col),
                mle_eval(&dense, &r_col)
            );
        }
    }

    /// The reported per-slot evaluations reproduce the comb the verifier used
    /// to build itself — so deferring changed what is COMPUTED, not what is
    /// CHECKED. Also pins that the deferred verify leaves the matrix work
    /// genuinely undone: a corrupted report sails past it and is caught only
    /// by `check_reported`.
    #[test]
    fn reported_evals_match_the_inline_comb() {
        let mut rng = Rng::new(0x5EED_0BEE);
        for counts_elem in [vec![5usize], vec![6, 3]] {
            let cases: Vec<Case> = if counts_elem.len() == 1 {
                vec![mult_case(2)]
            } else {
                vec![mixed_case(&mut rng), mult_case(2)]
            };
            let h = build(vec![], &cases, 3, &counts_elem, &mut rng);
            let union = UnionInstance::new(&h.registry, h.counts.clone());
            let z_region = region(&union, &h.z).to_vec();
            let pa = region(&union, &h.pa).to_vec();
            let pb = region(&union, &h.pb).to_vec();
            let mut ch_p = FsChallenger::new(b"element-report-rt");
            let (proof, _) = prove(&union, &z_region, &pa, &pb, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"element-report-rt");
            let (_, assertion) =
                verify_deferred(&union, &proof, &mut ch_v).expect("deferred accepts");
            assert!(assertion.check_reported(&union).is_ok(), "honest report");

            // The reported values, recombined with the verifier's own
            // scalars, ARE the old inline comb.
            let slots = region_slots(&union);
            let nu = union.n_log();
            let inline = region_comb_at_oracle(
                &slots,
                nu,
                assertion.alpha,
                &{
                    let mut r = vec![F128::ZERO; nu];
                    r.extend_from_slice(&assertion.r_con);
                    r
                },
                &assertion.r_col,
            );
            let mut recombined = F128::ZERO;
            for (s, &(va, vb)) in slots.iter().zip(&assertion.evals) {
                let kappa = s.layout.kappa;
                let w_r = eq_prefix_weight(&assertion.r_con[kappa..], s.layout.region_prefix(nu));
                let w_col = eq_prefix_weight(&assertion.r_col[kappa..], s.layout.region_prefix(nu));
                recombined += w_r * w_col * (va + assertion.alpha * vb);
            }
            assert_eq!(recombined, inline, "counts {counts_elem:?}");

            // A corrupted report is invisible to the deferred verify.
            let mut bad = proof.clone();
            bad.lincheck.matrix_evals[0].0 += F128::ONE;
            let mut ch = FsChallenger::new(b"element-report-rt");
            let (_, a2) = verify_deferred(&union, &bad, &mut ch).expect("defers");
            assert!(
                a2.check_reported(&union).is_err(),
                "check_reported must catch it"
            );
            let mut ch = FsChallenger::new(b"element-report-rt");
            assert!(verify(&union, &bad, &mut ch).is_err(), "composed rejects");
        }
    }

    /// Prove → verify roundtrip on satisfying witnesses at several shapes:
    /// one slot / two slots, with and without a boolean region in front, at
    /// partial, full and ZERO counts.
    #[test]
    fn prove_verify_roundtrip_honest() {
        let mut rng = Rng::new(0x2044_1A11);
        let shapes: Vec<(Vec<(usize, usize)>, Vec<Case>, usize, Vec<usize>)> = vec![
            (vec![], vec![mult_case(2)], 3, vec![5]),
            (vec![], vec![mult_case(2)], 3, vec![8]),
            (vec![], vec![mult_case(2)], 3, vec![0]),
            (
                vec![],
                vec![mixed_case(&mut rng), mult_case(2)],
                3,
                vec![6, 3],
            ),
            (vec![(10, 700)], vec![mult_case(3)], 2, vec![0, 3]),
            (
                vec![(10, 700), (9, 300)],
                vec![mixed_case(&mut rng), free_case(1)],
                2,
                vec![0, 0, 4, 2],
            ),
        ];
        for (bools, cases, nu, counts) in shapes {
            let n_elem = cases.len();
            let counts_elem: Vec<usize> = counts[counts.len() - n_elem..].to_vec();
            let h = build(bools, &cases, nu, &counts_elem, &mut rng);
            let union = UnionInstance::new(&h.registry, h.counts.clone());
            let z_region = region(&union, &h.z).to_vec();
            let pa = region(&union, &h.pa).to_vec();
            let pb = region(&union, &h.pb).to_vec();

            let mut ch_p = FsChallenger::new(b"element-region-rt");
            let (proof, claims_p) = prove(&union, &z_region, &pa, &pb, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"element-region-rt");
            let claims_v = verify(&union, &proof, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected counts {counts_elem:?}: {e:?}"));
            assert_eq!(claims_p, claims_v, "counts {counts_elem:?}");

            // Both claims are evaluations of the UNION witness MLE at their
            // points — the property the packed-direct opening discharges.
            let m_words = union.m_total() - 7;
            for (point, value) in [
                (&claims_v.c_point, claims_v.c_value),
                (&claims_v.lc_point, claims_v.lc_value),
            ] {
                assert_eq!(point.len(), m_words, "claim point addresses the union");
                assert_eq!(mle_eval(&h.z, point), value, "counts {counts_elem:?}");
            }
            // The two points share their row coordinates (r_row reused).
            assert_eq!(&claims_v.c_point[..nu], &claims_v.lc_point[..nu]);
            // The lincheck costs E − nu rounds, not E.
            assert_eq!(proof.lincheck.rounds.len(), union.m_elem() - 7 - nu);
            // The region prefix coordinates are Boolean, the rest random.
            let prefix = union.element_prefix_coords();
            let lo = m_words - prefix.len();
            assert_eq!(&claims_v.c_point[lo..], &prefix[..]);
        }
    }

    /// The union-region path is what the production mixed prover uses.  Pin
    /// its Secure policy separately from the standalone API: the element
    /// zerocheck protects tau and every round, while the lincheck protects
    /// alpha and every column sumcheck round.
    #[test]
    fn grinded_region_roundtrip_and_rejects_missing_nonce() {
        let mut rng = Rng::new(0x128_E1E_2044);
        let nu = 3usize;
        let cases = vec![mixed_case(&mut rng), mult_case(2)];
        let h = build(vec![(10, 700)], &cases, nu, &[5, 6], &mut rng);
        let union = UnionInstance::new(&h.registry, h.counts.clone());
        let z = region(&union, &h.z).to_vec();
        let pa = region(&union, &h.pa).to_vec();
        let pb = region(&union, &h.pb).to_vec();
        let grinding = Grinding::per_challenge_128();
        let e_vars = union.m_elem() - 7;
        let lc_rounds = e_vars - nu;

        let mut ch_p = FsChallenger::new(b"element-region-grinding");
        let (proof, claims_p) = prove_with_grinding(&union, &z, &pa, &pb, grinding, &mut ch_p);
        assert_eq!(
            proof.zerocheck.grinding_nonces.len(),
            grinding.zerocheck_nonce_count(e_vars)
        );
        assert_eq!(
            proof.lincheck.grinding_nonces.len(),
            grinding.lincheck_nonce_count(lc_rounds)
        );

        let mut ch_v = FsChallenger::new(b"element-region-grinding");
        assert_eq!(
            verify_with_grinding(&union, &proof, grinding, &mut ch_v).expect("honest proof"),
            claims_p
        );

        let mut bad = proof.clone();
        bad.lincheck.grinding_nonces.pop();
        let mut ch_v = FsChallenger::new(b"element-region-grinding");
        assert!(matches!(
            verify_with_grinding(&union, &bad, grinding, &mut ch_v),
            Err(VerifyError::LincheckGrindingNonceCount { .. })
        ));

        // The α nonce is checked before alpha is squeezed.  The scan finds a
        // nonce that fails that 1-bit predicate without relying on a lucky
        // hard-coded value for this transcript state.
        let original = proof.lincheck.grinding_nonces[0];
        let mut rejected_bad_nonce = false;
        for delta in 1..=64u64 {
            let mut bad = proof.clone();
            bad.lincheck.grinding_nonces[0] = original.wrapping_add(delta);
            let mut ch_v = FsChallenger::new(b"element-region-grinding");
            if matches!(
                verify_with_grinding(&union, &bad, grinding, &mut ch_v),
                Err(VerifyError::LincheckGrindingInvalid { which: "alpha" })
            ) {
                rejected_bad_nonce = true;
                break;
            }
        }
        assert!(rejected_bad_nonce, "a changed alpha PoW nonce must reject");
    }

    /// A witness violating ONE constraint in ONE row of ONE slot is rejected —
    /// including in the SECOND element slot, which the region zerocheck only
    /// sees through the eq weight's implicit batching.
    #[test]
    fn violated_constraint_is_rejected() {
        let mut rng = Rng::new(0xBAD_2044);
        let nu = 3usize;
        for bad_slot in [0usize, 1] {
            let cases = vec![mult_case(2), mult_case(2)];
            let h = build(vec![(10, 700)], &cases, nu, &[5, 6], &mut rng);
            let mut h = h;
            let union = UnionInstance::new(&h.registry, h.counts.clone());
            // Break the product column of one real row, then RE-derive pa/pb
            // so the prover follows the honest algorithm on a bad witness.
            let range = union.slot_word_range(1 + bad_slot);
            let rows = 1usize << nu;
            h.z[range.start + 2 * rows + 1] += F128::ONE;
            let zc = h.z[range.clone()].to_vec();
            cases[bad_slot].ty.affine_products_into(
                &zc,
                nu,
                None,
                &mut h.pa[range.clone()],
                &mut h.pb[range.clone()],
            );

            let z_region = region(&union, &h.z).to_vec();
            let pa = region(&union, &h.pa).to_vec();
            let pb = region(&union, &h.pb).to_vec();
            let mut ch_p = FsChallenger::new(b"element-region-bad");
            let (proof, _) = prove(&union, &z_region, &pa, &pb, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"element-region-bad");
            assert_eq!(
                verify(&union, &proof, &mut ch_v),
                Err(VerifyError::Zerocheck(
                    zerocheck::VerifyError::SumcheckFinalFailed
                )),
                "slot {bad_slot}"
            );
        }
    }

    /// **THE TRUST BOUNDARY of the support-proportional row rounds.** Padding
    /// columns are pinned to zero by their all-zero constraint rows, so a
    /// non-zero padding column is a relation violation — and the DENSE zerocheck
    /// catches it, because the dirty word sits inside its sum.
    ///
    /// The count-proportional row rounds do NOT, and cannot: they skip every
    /// word the declared support calls dead, so a dirty dead word is simply
    /// never read. That is exactly the boolean side's run-list contract ("only
    /// sound, and only bit-identical to the dense computation, for honest
    /// zeros"), and under the union the enforcer is the TRANSPORT — a dead word
    /// is not committed, so a non-zero one breaks `⟨q, W_ρ⟩ = f̂(ρ)` and the
    /// opening rejects. End-to-end coverage lives in
    /// `union_element::{satisfying_dummy_row_is_rejected_under_the_union,
    /// element_only_agrees_with_the_standalone_proof}`.
    ///
    /// This test pins both halves by picking the utilization that decides which
    /// path runs: FULL utilization keeps the gate closed (dense, caught), half
    /// utilization opens it (skipped, and the PIOP alone accepts).
    #[test]
    fn padding_column_boundary_dense_catches_sparse_delegates() {
        let nu = 3usize;
        // kappa 2, k = 3: column 3 is padding.
        for (n_t, dense_path) in [(1usize << nu, true), (4, false)] {
            // The delegation leg is release-only: a debug build's dead-word
            // guard refuses the dirty witness before the sparse rounds can
            // skip it (pinned by
            // dirty_dead_word_trips_the_sparse_rounds_debug_guard).
            if cfg!(debug_assertions) && !dense_path {
                continue;
            }
            let mut rng = Rng::new(0x9AD_0 + n_t as u64);
            let cases = vec![mult_case(2)];
            let mut h = build(vec![], &cases, nu, &[n_t], &mut rng);
            let union = UnionInstance::new(&h.registry, h.counts.clone());
            let range = union.slot_word_range(0);
            h.z[range.start + 3 * (1 << nu)] = F128::ONE;
            let zc = h.z[range.clone()].to_vec();
            cases[0].ty.affine_products_into(
                &zc,
                nu,
                None,
                &mut h.pa[range.clone()],
                &mut h.pb[range.clone()],
            );
            let z_region = region(&union, &h.z).to_vec();
            let pa = region(&union, &h.pa).to_vec();
            let pb = region(&union, &h.pb).to_vec();
            let mut ch_p = FsChallenger::new(b"element-region-pad");
            let (proof, _) = prove(&union, &z_region, &pa, &pb, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"element-region-pad");
            let rejected = verify(&union, &proof, &mut ch_v).is_err();
            assert_eq!(
                rejected, dense_path,
                "n_t={n_t}: the dense zerocheck must catch a dirty padding \
                 column; the support-proportional one delegates it to the \
                 transport"
            );
        }
    }

    /// Debug builds close the delegation window loudly instead: the sparse
    /// rounds' dead-word guard scans the declared-dead region (affordable
    /// only under `debug_assertions`) and refuses a dirty witness outright.
    /// Release pins the delegation above; this pins the guard.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "zero on every dead word")]
    fn dirty_dead_word_trips_the_sparse_rounds_debug_guard() {
        let nu = 3usize;
        let n_t = 4usize; // half utilization: the support-proportional path
        let mut rng = Rng::new(0x9AD_0 + n_t as u64);
        let cases = vec![mult_case(2)];
        let mut h = build(vec![], &cases, nu, &[n_t], &mut rng);
        let union = UnionInstance::new(&h.registry, h.counts.clone());
        let range = union.slot_word_range(0);
        h.z[range.start + 3 * (1 << nu)] = F128::ONE;
        let zc = h.z[range.clone()].to_vec();
        cases[0].ty.affine_products_into(
            &zc,
            nu,
            None,
            &mut h.pa[range.clone()],
            &mut h.pb[range.clone()],
        );
        let z_region = region(&union, &h.z).to_vec();
        let pa = region(&union, &h.pa).to_vec();
        let pb = region(&union, &h.pb).to_vec();
        let mut ch_p = FsChallenger::new(b"element-region-pad");
        let _ = prove(&union, &z_region, &pa, &pb, &mut ch_p);
    }

    /// Tamper matrix on the region proof: every round message and both final
    /// values are pinned.
    #[test]
    fn verify_rejects_mutations() {
        let mut rng = Rng::new(0x7A_3E_2044);
        let nu = 3usize;
        let cases = vec![mixed_case(&mut rng), mult_case(2)];
        let h = build(vec![(10, 700)], &cases, nu, &[4, 5], &mut rng);
        let union = UnionInstance::new(&h.registry, h.counts.clone());
        let z_region = region(&union, &h.z).to_vec();
        let mut ch_p = FsChallenger::new(b"element-region-mut");
        let (proof, _) = prove(
            &union,
            &z_region,
            region(&union, &h.pa),
            region(&union, &h.pb),
            &mut ch_p,
        );
        let mut ch = FsChallenger::new(b"element-region-mut");
        assert!(verify(&union, &proof, &mut ch).is_ok(), "honest proof");

        let mut cases_t: Vec<(String, Proof)> = Vec::new();
        for i in 0..proof.zerocheck.rounds.len() {
            for w in 0..2 {
                let mut bad = proof.clone();
                if w == 0 {
                    bad.zerocheck.rounds[i].0 += F128::ONE;
                } else {
                    bad.zerocheck.rounds[i].1 += F128::ONE;
                }
                cases_t.push((format!("zc round {i} msg {w}"), bad));
            }
        }
        for i in 0..proof.lincheck.rounds.len() {
            for w in 0..2 {
                let mut bad = proof.clone();
                if w == 0 {
                    bad.lincheck.rounds[i].0 += F128::ONE;
                } else {
                    bad.lincheck.rounds[i].1 += F128::ONE;
                }
                cases_t.push((format!("lc round {i} msg {w}"), bad));
            }
        }
        for (name, f) in [
            ("ea", 0usize),
            ("eb", 1),
            ("ec", 2),
            ("z_eval", 3),
            ("zc rounds truncated", 4),
            ("lc rounds truncated", 5),
        ] {
            let mut bad = proof.clone();
            match f {
                0 => bad.zerocheck.ea += F128::ONE,
                1 => bad.zerocheck.eb += F128::ONE,
                2 => bad.zerocheck.ec += F128::ONE,
                3 => bad.lincheck.z_eval += F128::ONE,
                4 => {
                    bad.zerocheck.rounds.pop();
                }
                _ => {
                    bad.lincheck.rounds.pop();
                }
            }
            cases_t.push((name.to_string(), bad));
        }
        for (name, bad) in cases_t {
            let mut ch = FsChallenger::new(b"element-region-mut");
            assert!(
                verify(&union, &bad, &mut ch).is_err(),
                "verify accepted mutation: {name}"
            );
        }
    }

    /// A count vector the prover did not use makes the verifier read a
    /// different region geometry, so it rejects. (Counts also bind in the
    /// statement, one layer up; this pins that the PIOP itself is not
    /// count-agnostic when the geometry moves.)
    #[test]
    fn region_geometry_binds() {
        let mut rng = Rng::new(0x6E0_2044);
        let nu = 3usize;
        let cases = vec![mult_case(2)];
        let h = build(vec![(10, 700)], &cases, nu, &[4], &mut rng);
        let union = UnionInstance::new(&h.registry, h.counts.clone());
        let mut ch_p = FsChallenger::new(b"element-region-geo");
        let (proof, _) = prove(
            &union,
            region(&union, &h.z),
            region(&union, &h.pa),
            region(&union, &h.pb),
            &mut ch_p,
        );
        // A registry with a WIDER element block has a bigger region, hence a
        // different round count — rejected on shape.
        let other = Registry::new(
            vec![bool_ty(10, 700), TableType::element(Arc::new(mult_gate(3)))],
            nu,
        );
        let other_union = UnionInstance::new(&other, vec![0, 4]);
        let mut ch = FsChallenger::new(b"element-region-geo");
        assert!(verify(&other_union, &proof, &mut ch).is_err());
    }

    /// **THE optimization's own oracle.** The support-proportional row rounds
    /// must produce a BIT-IDENTICAL proof to the dense ones — same round
    /// messages, same finals — at every utilization, and be cheaper.
    ///
    /// Bit-identity is what lets the pinned mixed-class fixtures survive the
    /// change, and it rests on two facts: a fully-dead pair contributes nothing
    /// to either round message (`G(1)`'s summand is `a_const·b_const + 0 = 0`
    /// by the validity rule, and `G(∞)`'s factor `wa[i0] + wa[i1]` vanishes in
    /// characteristic 2 because both halves hold the same constant), and a
    /// boundary pair's dead half is exactly `a_dead[c]` because folding
    /// preserves a constant column.
    ///
    /// Arms alternate in one process; timings are a median over reps and
    /// nothing asserts on the clock.
    #[test]
    fn support_proportional_rounds_are_bit_identical() {
        use crate::element_r1cs::zerocheck;
        use std::time::Instant;

        let mut rng = Rng::new(0x5044_0A17);
        let nu = 14usize;
        let cases = vec![mult_case(3)]; // kappa 3: 8 columns, 3 real + padding
        let rows = 1usize << nu;
        eprintln!("\n[element-rows] nu={nu}, region 2^17 words, median [min – max] ms");
        for div in [1usize, 2, 4, 16, 64] {
            let n = rows / div;
            let h = build(vec![], &cases, nu, &[n], &mut rng);
            let union = UnionInstance::new(&h.registry, h.counts.clone());
            let e_vars = union.m_elem() - 7;
            let z = region(&union, &h.z).to_vec();
            let pa = region(&union, &h.pa).to_vec();
            let pb = region(&union, &h.pb).to_vec();
            let slots = region_slots(&union);
            let sup = row_support(&slots, nu, e_vars);

            let reps = 5usize;
            let (mut t_dense, mut t_sparse) = (Vec::new(), Vec::new());
            for rep in 0..=reps {
                let mut ch = FsChallenger::new(b"element-rows-ab");
                let t = Instant::now();
                let (dense, dclaim) = zerocheck::prove_with_support(
                    zerocheck::LABEL,
                    &pa,
                    &pb,
                    &z,
                    e_vars,
                    nu,
                    None,
                    &mut ch,
                );
                let ms_d = t.elapsed().as_secs_f64() * 1e3;

                let mut ch = FsChallenger::new(b"element-rows-ab");
                let t = Instant::now();
                let (sparse, sclaim) = zerocheck::prove_with_support(
                    zerocheck::LABEL,
                    &pa,
                    &pb,
                    &z,
                    e_vars,
                    nu,
                    Some(&sup),
                    &mut ch,
                );
                let ms_s = t.elapsed().as_secs_f64() * 1e3;

                assert_eq!(dense, sparse, "n={n}: round messages must be identical");
                assert_eq!(dclaim, sclaim, "n={n}: claims must be identical");
                if rep > 0 {
                    t_dense.push(ms_d);
                    t_sparse.push(ms_s);
                }
            }
            let med = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[0], v[v.len() / 2], v[v.len() - 1])
            };
            let (d, sp) = (med(t_dense), med(t_sparse));
            eprintln!(
                "[element-rows]  n={n:>5} of {rows} ({:>4.1}% rows, {:>4.1}% words)  \
dense {:6.2} [{:5.2} – {:5.2}]  support {:6.2} [{:5.2} – {:5.2}]  {:4.1}x",
                100.0 * n as f64 / rows as f64,
                100.0 * (cases[0].ty.k() * n) as f64 / (1usize << e_vars) as f64,
                d.1,
                d.0,
                d.2,
                sp.1,
                sp.0,
                sp.2,
                d.1 / sp.1,
            );
        }
    }

    /// `fill_slot` derives `pa`/`pb` exactly as the standalone prover's
    /// `apply` + `broadcast_add` — the one place the union and standalone
    /// witness paths could drift.
    #[test]
    fn fill_slot_matches_the_standalone_preparation() {
        let mut rng = Rng::new(0xF111_5107);
        let nu = 4usize;
        let ty = mixed_gate(&mut rng);
        let n = 11usize;
        let rows = mixed_witness(&ty, nu, n, &mut rng);

        let (mut apply_a, mut apply_b) = ty.apply(&rows, nu);
        crate::element_r1cs::broadcast_add(&mut apply_a, ty.a_const(), nu);
        crate::element_r1cs::broadcast_add(&mut apply_b, ty.b_const(), nu);

        let words = ty.width() << nu;
        let mut z = vec![F128::ZERO; words];
        let mut pa = vec![F128::ZERO; words];
        let mut pb = vec![F128::ZERO; words];
        fill_slot(&ty, nu, None, &mut z, &mut pa, &mut pb, |dst| {
            dst.copy_from_slice(&rows)
        });
        assert_eq!(z, rows);
        assert_eq!(pa, apply_a);
        assert_eq!(pb, apply_b);
    }
}
