//! Union-column lincheck for multi-table proofs.
//!
//! Generalizes the single-table lincheck (`super`) from one slot to the `T`
//! aligned slot subcubes of a registry instance. One product sumcheck over
//! the **union column domain** — the
//! seven in-word coordinates plus every chunk-and-prefix coordinate, i.e.
//! everything except the shared row block `R` — proves the α-batched claim
//! `α·â(r) + b̂(r)` left by the union zerocheck.
//!
//! ## The identity being proven (doc §6.2–6.4, uniform capacity)
//!
//! ```text
//!   α·â(r) + b̂(r) = Σ_t w_t(r) · Σ_{c'} (α·comb_t^A + comb_t^B)(c') · y_t(c')
//! ```
//!
//! - `w_t(r) = eq(r[m_t:M], p_t)` — the slot's prefix-eq weight, a *fixed
//!   public scalar* determined by `r` and the registry (no cross-slot
//!   batching randomness; the single α batches only the A- and B-claims);
//! - `comb_t` — the type's α-batched comb via its [`LincheckCircuit`] with
//!   the quirky table of the slot's own column coordinates of `r` (the
//!   slot's chunk bits are the LOW union chunk coordinates; the frozen
//!   prefix is the high ones) — exactly today's single-table comb;
//! - `y_t(c') = ẑ_t(r|_R, c')` — slot `t`'s witness folded over the SHARED
//!   row coordinates: one standard eq table serves every slot, by uniform
//!   capacity. Dummy rows are honest zeros, so folding the full capacity
//!   equals folding the declared rows.
//!
//! The prover materializes `Comb(c) = w_t(r)·comb_t(c')` and `g(c) = y_t(c')`
//! on slot `t`'s aligned column subcube — zero on gaps — as dense
//! union-column-domain vectors and runs the existing sumcheck core
//! (`column_sumcheck_prove`) over them. A single-slot registry reproduces
//! today's two vectors verbatim (`w_1 = 1`, offset `0`), so the `T = 1`
//! union lincheck is **byte-identical** to the single-table one (the M2
//! differential property). Dense union-column vectors are the simple, correct
//! choice at current scales; the per-slot factorized / scalar-tail
//! iteration (collapsed slots contribute `O(1)` per remaining round, doc
//! §"One sumcheck over the union column domain") is a later performance
//! item.
//!
//! The verifier replays the rounds without materializing anything
//! union-sized: at the end it evaluates the Comb-hat skip-block collapse by
//! the closed form — per-type comb MLEs at the bound point, times the
//! slot's r-side prefix weight and the bound-point subcube prefix-eq factor
//! ([`union_comb_partial`]).
//!
//! Output: a SINGLE witness claim `ẑ(c*-columns, r|_R-rows)`, structurally
//! identical to today's ([`LincheckClaim`] with `r_inner_rest` spanning the
//! union column coordinates); `UnionInstance::ab_claim_point` converts it to
//! the address-ordered `ZClaim` point.
//!
//! ## Constant-wire pins under counts
//!
//! Pins generalize per type: β_t is sampled after α, in slot order, and
//! added (unscaled) at the pin's union column. Declared rows carry the pin
//! at `1`; dummy rows are all-zero *including the pin* (doc Remark "Dummy
//! rows are complete" — the pin at 0 is what marks a row as dummy), so the
//! honest fold of the pin column is `Σ_{row<n_t} eq(r|_R, row)` and the
//! verifier's target gains `β_t · eq_prefix_sum(r|_R, n_t)` — the counts
//! bind inside the lincheck. At full utilization the sum is exactly `1`
//! (the eq-sum identity), reproducing today's `target += β` byte for byte.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::matrix_fold::{MatrixClaim, Weight};
use crate::schedule::Registry;
use crate::union::UnionInstance;
use crate::zerocheck::K_SKIP;

use super::{
    LincheckCircuit, LincheckClaim, LincheckError, LincheckGrinding, LincheckProof, QuirkyPoint,
    SkipPoint, build_eq_table, build_quirky_eq_table_from_weights, column_sumcheck_prove,
    inner_product, partial_fold_packed_z_rows_best,
};

/// One slot's lincheck inputs, in slot order — the union counterpart of the
/// `(z_packed_lincheck, circuit)` pair the single-table lincheck consumes.
pub struct UnionLincheckSlot<'a> {
    /// The slot's lincheck stripe copy of `z` — `pack_z_lincheck` layout over
    /// the slot's own `(m_slot, k_log)`, i.e. the batch-major witness
    /// drivers' fourth output. Length `2^{m_slot−3}` bytes.
    pub z_lincheck: &'a [u8],
    /// The slot's lincheck circuit; its `const_pin_col` must match the
    /// registry type's `const_pin`.
    pub circuit: &'a dyn LincheckCircuit,
}

/// `Π_j eq(coords[j], bit_j(bits))` — the eq factor freezing `coords` to the
/// Boolean pattern `bits` (LSB-first). The slot prefix weights `w_t(r)` and
/// the closed form's bound-point subcube factors are both instances.
pub fn eq_prefix_weight(coords: &[F128], bits: usize) -> F128 {
    debug_assert!(coords.len() >= usize::BITS as usize - bits.leading_zeros() as usize);
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

/// `Σ_{i<n} eq(point, i)` in `O(|point|)` field ops — the verifier's
/// const-pin expected value at declared count `n`. `n = 2^{|point|}` returns
/// exactly `F128::ONE` (the eq-sum identity), with no arithmetic, so the
/// full-utilization target matches today's `target += β` byte for byte.
pub fn eq_prefix_sum(point: &[F128], n: usize) -> F128 {
    let d = point.len();
    assert!(n <= 1usize << d, "prefix length {n} exceeds domain 2^{d}");
    if n == 1usize << d {
        return F128::ONE;
    }
    // Indices i < n partition by the highest bit where i drops below n: for
    // each j with bit_j(n) = 1, the block {i: i matches n above j, bit_j(i) =
    // 0, free below} contributes Π_{i>j} eq(point[i], bit_i(n)) · eq(point[j],
    // 0) · 1 (the free bits sum to 1 by the eq-sum identity).
    let mut acc = F128::ZERO;
    let mut high = F128::ONE; // running Π_{i>j} eq(point[i], bit_i(n))
    for j in (0..d).rev() {
        if (n >> j) & 1 == 1 {
            acc += high * (F128::ONE + point[j]);
            high *= point[j];
        } else {
            high *= F128::ONE + point[j];
        }
    }
    acc
}

/// Closed-form skip-block collapse of the union Comb-hat at the bound column
/// point (doc §"One sumcheck over the union column domain"): with `combs[t]`
/// the type's length-`2^{κ_t}` comb vector — already carrying the r-side
/// prefix weight `w_t(r)` and any const-pin β terms — and `rr` the LSB-first
/// bound challenges of the multilinear column coordinates (`rr[i]` binds
/// coordinate `k_skip + i`), returns the length-`2^{k_skip}` vector
///
/// ```text
///   out[s] = Σ_t eq(rr[κ_t−k_skip..], p_t)
///              · Σ_j eq-tensor(rr[..κ_t−k_skip])[j] · combs[t][s + (j << k_skip)]
/// ```
///
/// — each slot contributes its own comb MLE at the bound point times the
/// subcube prefix-eq factor "the bound point addresses slot t"; nothing
/// union-sized is materialized. Cost `O(Σ_t 2^{κ_t})`, the same as the
/// per-type comb construction itself.
pub fn union_comb_partial(
    registry: &Registry,
    combs: &[Vec<F128>],
    rr: &[F128],
    k_skip: usize,
) -> Vec<F128> {
    let nb = registry.num_boolean();
    assert_eq!(combs.len(), nb);
    assert_eq!(rr.len(), registry.m_bool() - registry.nu() - k_skip);
    let n_skip = 1usize << k_skip;
    let mut out = vec![F128::ZERO; n_skip];
    for ((ty, slot), comb) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(combs)
    {
        assert_eq!(comb.len(), 1usize << ty.k_log);
        let inner = ty.k_log - k_skip;
        let p_t = eq_prefix_weight(&rr[inner..], slot.prefix);
        let eq_rho = build_eq_table(&rr[..inner]);
        for (j, &e) in eq_rho.iter().enumerate() {
            let s = p_t * e;
            let block = &comb[(j << k_skip)..((j + 1) << k_skip)];
            for (o, &c) in out.iter_mut().zip(block) {
                *o += s * c;
            }
        }
    }
    out
}

/// Prove the union-column lincheck and capture the pre-sumcheck union fold
/// `g` (the per-slot `y_t` vectors at their aligned column offsets, length
/// `2^{M−ν}`) for downstream reuse — the union counterpart of
/// [`super::prove_padded_capture_z_vec`]. `x_ab` is the union semantic
/// quirky point from `UnionInstance::x_ab_from_mlv`; `k_skip` is the
/// BatchMajor-fixed [`K_SKIP`]. On a single-type registry this computes
/// exactly the single-table lincheck — same vectors, same rounds, same
/// bytes.
pub fn prove_union_capture_z_vec<Ch: Challenger>(
    union: &UnionInstance<'_>,
    slots: &[UnionLincheckSlot<'_>],
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    prove_union_capture_z_vec_with_grinding(
        union,
        slots,
        x_ab,
        LincheckGrinding::disabled(),
        challenger,
    )
}

/// [`prove_union_capture_z_vec`] with an explicit Fiat--Shamir grinding
/// policy.
pub fn prove_union_capture_z_vec_with_grinding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    slots: &[UnionLincheckSlot<'_>],
    x_ab: &QuirkyPoint,
    grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    let registry = union.registry();
    let k_skip = K_SKIP;
    let nu = union.n_log();
    let col_vars = union.m_bool() - nu;
    let inner_rest_len = col_vars - k_skip;
    assert!(nu >= 3, "lincheck stripe fold needs n_outer ≥ 8 (nu ≥ 3)");
    assert_eq!(
        slots.len(),
        registry.num_boolean(),
        "one lincheck input per BOOLEAN registry type (element slots have their own PIOP)"
    );
    assert_eq!(x_ab.x_inner_rest.len(), inner_rest_len);
    assert_eq!(x_ab.x_outer.len(), nu);
    for (ty, slot_in) in registry.boolean_types().iter().zip(slots) {
        assert_eq!(
            slot_in.circuit.n_cols(),
            1usize << ty.k_log,
            "circuit width must match the registry type"
        );
        assert_eq!(
            slot_in.circuit.const_pin_col(),
            ty.const_pin,
            "circuit const_pin must match the registry type"
        );
        assert_eq!(
            slot_in.z_lincheck.len(),
            1usize << (nu + ty.k_log - 3),
            "slot lincheck stripe length mismatch"
        );
    }

    challenger.observe_label(b"flock-lincheck-v0");
    let trace = std::env::var("LINCHECK_TRACE").is_ok();
    let mut grinding_nonces = Vec::with_capacity(
        grinding.nonce_count(
            inner_rest_len,
            slots
                .iter()
                .filter(|slot| slot.circuit.const_pin_col().is_some())
                .count(),
            k_skip,
        ),
    );

    // 1. Sample α (matches verifier's order). ONE α batches the A- and
    //    B-claims for every slot (doc §"The B-claim, and batching the two
    //    sumchecks"); the cross-slot weights w_t(r) are fixed scalars, not
    //    randomness.
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let (nonce, alpha) = challenger.grind_pow_and_sample_f128(bits);
        grinding_nonces.push(nonce);
        alpha
    } else {
        challenger.sample_f128()
    };

    // 2. Per-type α-batched combs via each type's quirky table — the slot's
    //    column coordinates of r are z_skip, dim6, and the LOW `κ_t − 7`
    //    union chunk coordinates — scaled by the slot prefix weight w_t(r)
    //    and placed at the slot's aligned column offset `o_t / 2^ν`.
    //    Dense union-column placement (see module docs); a single slot is
    //    today's comb vector, moved.
    // The T = 1 fast path (today's vectors verbatim) needs the one boolean
    // slot to FILL the boolean region — true for every single-type registry,
    // and for no multi-slot one.
    let single = registry.num_boolean() == 1 && registry.slots()[0].m_slot == registry.m_bool();
    let t_comb = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut comb_vec: Vec<F128> = if single {
        Vec::new()
    } else {
        vec![F128::ZERO; 1usize << col_vars]
    };
    // Per-type (ξ_A, ξ_B), kept for the matrix-eval report after the sumcheck.
    let mut split_combs: Vec<(Vec<F128>, Vec<F128>)> = Vec::with_capacity(registry.num_boolean());
    for ((ty, slot), slot_in) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(slots)
    {
        let inner = ty.k_log - k_skip;
        let eq_inner = build_quirky_eq_table_from_weights(
            &x_ab.z_skip.weights(k_skip),
            &x_ab.x_inner_rest[..inner],
        );
        // Split, so the per-matrix bilinear values can be reported below.
        // Same nonzeros as the α-batched fold; only the accumulation differs.
        let (comb_a, comb_b) = slot_in.circuit.fold_split(&eq_inner);
        let comb_t: Vec<F128> = comb_a
            .iter()
            .zip(&comb_b)
            .map(|(a, b)| alpha * *a + *b)
            .collect();
        split_combs.push((comb_a, comb_b));
        if single {
            comb_vec = comb_t; // T = 1: w_1 = 1, offset 0 — today's vector.
        } else {
            let w_t = eq_prefix_weight(&x_ab.x_inner_rest[inner..], slot.prefix);
            let off = slot.prefix << ty.k_log;
            for (dst, &src) in comb_vec[off..off + comb_t.len()].iter_mut().zip(&comb_t) {
                *dst = w_t * src;
            }
        }
    }

    // 2b. Constant-wire pins, per type in slot order: β_t sampled after α,
    //     added UNSCALED at the pin's union column. The verifier's target
    //     gains the count-derived β_t · Σ_{row<n_t} eq(x_outer, row) — see
    //     `verify_union` and the module docs.
    for (slot, slot_in) in registry.slots().iter().zip(slots) {
        if let Some(col) = slot_in.circuit.const_pin_col() {
            let beta = if let Some(bits) = grinding.beta_bits() {
                let (nonce, beta) = challenger.grind_pow_and_sample_f128(bits);
                grinding_nonces.push(nonce);
                beta
            } else {
                challenger.sample_f128()
            };
            let off = slot.prefix << (slot.m_slot - nu);
            comb_vec[off + col] += beta;
        }
    }
    if let Some(t) = t_comb {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "union comb build",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 3. Per-slot row folds against the SHARED standard eq table of the row
    //    block r|_R (uniform capacity: one table, all slots), placed at the
    //    slots' aligned column offsets — the union generalization of today's
    //    single partial fold. Dummy rows are honest zeros, so folding only
    //    the DECLARED rows equals the full-capacity fold byte for byte —
    //    the row-aware dispatch makes the fold count-proportional (M6).
    let t_fold = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let eq_x_outer = build_eq_table(&x_ab.x_outer);
    let mut z_vec: Vec<F128> = if single {
        Vec::new()
    } else {
        vec![F128::ZERO; 1usize << col_vars]
    };
    for (((ty, slot), slot_in), &n_t) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(slots)
        .zip(union.counts())
    {
        let y_t = partial_fold_packed_z_rows_best(
            slot_in.z_lincheck,
            nu + ty.k_log,
            ty.k_log,
            ty.useful_bits,
            &eq_x_outer,
            n_t,
        );
        if single {
            z_vec = y_t;
        } else {
            let off = slot.prefix << ty.k_log;
            z_vec[off..off + y_t.len()].copy_from_slice(&y_t);
        }
    }

    if let Some(t) = t_fold {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "union partial folds",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 3b. Capture the pre-sumcheck union fold g for downstream reuse
    //     (mirrors `prove_padded_capture_z_vec`).
    let captured = z_vec.clone();

    // 4.–9. The existing sumcheck core, over the (longer) union column
    //       domain.
    let (mut proof, claim) = column_sumcheck_prove(
        comb_vec,
        z_vec,
        k_skip,
        &x_ab.z_skip,
        trace,
        grinding,
        &mut grinding_nonces,
        challenger,
    );

    // 10. The matrix work, reported rather than folded into the target: for
    //     each type the UNSCALED pair (⟨W_t, A_0⟩, ⟨W_t, B_0⟩) with the
    //     column weight W_col_t = z_partial ⊗ eq(rr[..inner_t]) that the
    //     sumcheck just fixed. The slot's prefix weights and β are verifier
    //     scalars, so they stay out — what accumulates must be a claim about
    //     the static matrix alone.
    proof.matrix_evals = registry
        .boolean_types()
        .iter()
        .zip(&split_combs)
        .map(|(ty, (comb_a, comb_b))| {
            let inner = ty.k_log - k_skip;
            let w_col = col_weight(&proof.z_partial, &claim.r_inner_rest[..inner], k_skip);
            (inner_product(&w_col, comb_a), inner_product(&w_col, comb_b))
        })
        .collect();

    (proof, claim, captured)
}

/// `z_partial ⊗ eq(rr)` over a type's own `2^{k_skip + rr.len()}` column
/// domain — the lincheck's column weight, shared by the prover's report and
/// the verifier's check so the two cannot drift.
fn col_weight(z_partial: &[F128], rr: &[F128], k_skip: usize) -> Vec<F128> {
    let n_skip = 1usize << k_skip;
    debug_assert_eq!(z_partial.len(), n_skip);
    let eq = build_eq_table(rr);
    let mut out = Vec::with_capacity(n_skip * eq.len());
    for &e in &eq {
        for &zp in z_partial {
            out.push(e * zp);
        }
    }
    out
}

/// The `O(nnz)` residue of the lincheck verifier: everything its final
/// consistency check needs EXCEPT the per-type comb folds.
///
/// The verifier's whole dependence on the base matrices is this one
/// assertion — a bilinear form in `α·Â₀ + B̂₀` at a rank-1 tensor point,
/// with the row weight `λ(z_skip) ⊗ eq(x_inner_rest)` and the column weight
/// `z_partial ⊗ eq(rr)`. Everything else the verifier does is succinct.
/// [`verify_union`] discharges it immediately (identical behaviour to
/// before this split); [`verify_union_deferred`] hands it back undischarged
/// so a caller can instead
///
/// * batch it across many proofs of the same registry and pay `O(nnz)`
///   once rather than per proof, or
/// * carry it as a public claim into an accumulator — the recursion path,
///   where a circuit replays only the succinct verifier and never pays the
///   matrix work at all.
///
/// Two of the four tensor factors (`λ(z_skip)` from the univariate skip,
/// and the proof-supplied `z_partial`) are not `eq`-tensors, so this is a
/// weighted sum rather than a plain MLE evaluation; a folding protocol's
/// first sumcheck is what turns it into one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixAssertion {
    /// The challenge batching the A- and B-matrices.
    pub alpha: F128,
    /// Row-side weight: `λ(z_skip) ⊗ eq(x_inner_rest)` — `λ` is the skip
    /// basis's evaluation functional ([`SkipPoint::weights`]), φ₈ Lagrange
    /// on the RS route, the AG base-code functional on the AG route.
    pub z_skip: SkipPoint,
    pub x_inner_rest: Vec<F128>,
    /// Column-side weight: `z_partial ⊗ eq(rr)`.
    pub rr: Vec<F128>,
    pub z_partial: Vec<F128>,
    /// Const-wire pin challenges, one entry per boolean slot in slot order
    /// (`None` where the type has no pin). Applied to the comb at the pin
    /// column; the target-side half of the pin is already folded into
    /// [`Self::target`].
    pub betas: Vec<Option<F128>>,
    /// Outer-row point and the count-prefix evaluation used by each const
    /// pin. Recursive verification carries these as claims on the static
    /// circuit-structure table instead of accepting prefix sums as advice.
    pub pin_point: Vec<F128>,
    pub pin_evals: Vec<Option<F128>>,
    /// What the weighted bilinear sum must equal.
    pub target: F128,
    /// The prover's per-type `(⟨W_t, A_0⟩, ⟨W_t, B_0⟩)`, in slot order —
    /// unscaled, so each is a claim about a registry-static matrix and
    /// nothing else. [`Self::check`] verifies they reproduce
    /// [`Self::target`]; [`Self::claims`] hands them out for accumulation.
    pub evals: Vec<(F128, F128)>,
}

impl MatrixAssertion {
    /// The per-`(type, matrix)` claims this assertion asserts, in slot order
    /// as `[(A_0 claim, B_0 claim); T]` — ready for
    /// [`crate::matrix_fold::prove_fold`].
    ///
    /// A and B stay separate: the target only ever carries their
    /// α-combination, and α is per-proof, so a claim about `α·A_0 + B_0`
    /// would name a different polynomial in every proof. These name the
    /// static matrices, and fold forever.
    pub fn claims(&self, registry: &Registry) -> Vec<(MatrixClaim, MatrixClaim)> {
        let k_skip = K_SKIP;
        registry
            .boolean_types()
            .iter()
            .zip(&self.evals)
            .map(|(ty, &(v_a, v_b))| {
                let inner = ty.k_log - k_skip;
                let row = Weight::low_eq(
                    self.z_skip.weights(k_skip),
                    self.x_inner_rest[..inner].to_vec(),
                );
                let col = Weight::low_eq(self.z_partial.clone(), self.rr[..inner].to_vec());
                (
                    MatrixClaim {
                        row: row.clone(),
                        col: col.clone(),
                        value: v_a,
                    },
                    MatrixClaim {
                        row,
                        col,
                        value: v_b,
                    },
                )
            })
            .collect()
    }

    /// Discharge the assertion: rebuild the per-type combs (`O(Σ_t nnz_t)`,
    /// the only place the base matrices are read) and check the closed-form
    /// Comb-hat collapse against [`Self::target`].
    pub fn check_reported(&self, registry: &Registry) -> Result<(), LincheckError> {
        let k_skip = K_SKIP;
        if self.evals.len() != registry.num_boolean() {
            return Err(LincheckError::BadVectorLength {
                which: "matrix_evals",
                expected: registry.num_boolean(),
                got: self.evals.len(),
            });
        }
        // target = Σ_t P_t·w_t·(α·⟨W_t,A⟩ + ⟨W_t,B⟩) + Σ_t P_t·β_t·W_col_t[pin]
        // — the slot prefix weights on both sides (w_t from the row point,
        // P_t from the bound column point) and β are all verifier scalars.
        let mut acc = F128::ZERO;
        for (((ty, slot), &(v_a, v_b)), beta) in registry
            .boolean_types()
            .iter()
            .zip(registry.slots())
            .zip(&self.evals)
            .zip(&self.betas)
        {
            let inner = ty.k_log - k_skip;
            let w_t = eq_prefix_weight(&self.x_inner_rest[inner..], slot.prefix);
            let p_t = eq_prefix_weight(&self.rr[inner..], slot.prefix);
            acc += p_t * w_t * (self.alpha * v_a + v_b);
            if let (Some(col), Some(beta)) = (ty.const_pin, beta) {
                let w_col = col_weight(&self.z_partial, &self.rr[..inner], k_skip);
                acc += p_t * *beta * w_col[col];
            }
        }
        if acc != self.target {
            return Err(LincheckError::ConsistencyFailed {
                which: "matrix-evals",
            });
        }
        Ok(())
    }

    /// Discharge against the real matrices — the `O(Σ_t nnz_t)` check. Used
    /// by [`verify_union`], and the differential oracle for the accumulated
    /// path: this and [`Self::check_reported`] + discharging the
    /// [`Self::claims`] must accept exactly the same proofs.
    pub fn check(
        &self,
        union: &UnionInstance<'_>,
        circuits: &[&dyn LincheckCircuit],
    ) -> Result<(), LincheckError> {
        let registry = union.registry();
        let k_skip = K_SKIP;
        assert_eq!(
            circuits.len(),
            registry.num_boolean(),
            "one circuit per BOOLEAN registry type"
        );
        let mut combs: Vec<Vec<F128>> = registry
            .boolean_types()
            .iter()
            .zip(registry.slots())
            .zip(circuits)
            .map(|((ty, slot), circuit)| {
                let inner = ty.k_log - k_skip;
                let eq_inner = build_quirky_eq_table_from_weights(
                    &self.z_skip.weights(k_skip),
                    &self.x_inner_rest[..inner],
                );
                let mut comb = circuit.fold_alpha_batched(self.alpha, &eq_inner);
                if slot.prefix_bits > 0 {
                    let w_t = eq_prefix_weight(&self.x_inner_rest[inner..], slot.prefix);
                    for v in &mut comb {
                        *v *= w_t;
                    }
                }
                comb
            })
            .collect();

        for ((circuit, comb), beta) in circuits.iter().zip(&mut combs).zip(&self.betas) {
            if let (Some(col), Some(beta)) = (circuit.const_pin_col(), beta) {
                comb[col] += *beta;
            }
        }

        let comb_partial = union_comb_partial(registry, &combs, &self.rr, k_skip);
        if self.target != inner_product(&comb_partial, &self.z_partial) {
            return Err(LincheckError::ConsistencyFailed {
                which: "sumcheck-final",
            });
        }
        Ok(())
    }
}

/// Verify a union-column lincheck proof: walk the challenger in lockstep
/// with [`prove_union_capture_z_vec`], replay the rounds, and check the
/// final consistency against the closed-form Comb-hat collapse
/// ([`union_comb_partial`]) — per-type `O(nnz)` comb machinery plus
/// `O(T·M)` prefix weights, never anything union-sized. `circuits` are the
/// per-type lincheck circuits in slot order. Returns the single witness
/// claim `ẑ(c*-columns, r|_R-rows)`.
///
/// This is [`verify_union_deferred`] followed immediately by
/// [`MatrixAssertion::check`] — identical accept/reject, and the entry
/// point every existing caller keeps using.
pub fn verify_union<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    challenger: &mut Ch,
) -> Result<LincheckClaim, LincheckError> {
    verify_union_with_grinding(
        union,
        circuits,
        x_ab,
        v_a,
        v_b,
        proof,
        LincheckGrinding::disabled(),
        challenger,
    )
}

/// [`verify_union`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_union_with_grinding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> Result<LincheckClaim, LincheckError> {
    let (claim, assertion) = verify_union_deferred_with_grinding(
        union, circuits, x_ab, v_a, v_b, proof, grinding, challenger,
    )?;
    assertion.check(union, circuits)?;
    Ok(claim)
}

/// [`verify_union`] with the matrix work left undischarged: replays the
/// whole protocol succinctly and returns the witness claim alongside the
/// [`MatrixAssertion`] the caller must discharge (or accumulate) for the
/// proof to mean anything.
///
/// **The returned claim is conditional.** Nothing here reads the base
/// matrices, so a proof whose lincheck is simply wrong still yields
/// `Ok(..)`; only [`MatrixAssertion::check`], or a folding protocol that
/// eventually discharges it, rejects that. Callers that are not
/// accumulating must use [`verify_union`].
pub fn verify_union_deferred<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    challenger: &mut Ch,
) -> Result<(LincheckClaim, MatrixAssertion), LincheckError> {
    verify_union_deferred_with_grinding(
        union,
        circuits,
        x_ab,
        v_a,
        v_b,
        proof,
        LincheckGrinding::disabled(),
        challenger,
    )
}

/// [`verify_union_deferred`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_union_deferred_with_grinding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> Result<(LincheckClaim, MatrixAssertion), LincheckError> {
    let registry = union.registry();
    let k_skip = K_SKIP;
    let nu = union.n_log();
    let col_vars = union.m_bool() - nu;
    let inner_rest_len = col_vars - k_skip;
    let n_skip = 1usize << k_skip;

    assert_eq!(
        circuits.len(),
        registry.num_boolean(),
        "one circuit per BOOLEAN registry type (element slots have their own PIOP)"
    );
    for (ty, circuit) in registry.boolean_types().iter().zip(circuits) {
        let k = 1usize << ty.k_log;
        if circuit.n_cols() != k {
            return Err(LincheckError::BadMatrixShape {
                which: "circuit",
                expected: k,
                got_rows: k,
                got_cols: circuit.n_cols(),
            });
        }
        assert_eq!(
            circuit.const_pin_col(),
            ty.const_pin,
            "circuit const_pin must match the registry type"
        );
    }
    if x_ab.x_inner_rest.len() != inner_rest_len {
        return Err(LincheckError::BadInnerRestLength {
            which: "x_ab",
            expected: inner_rest_len,
            got: x_ab.x_inner_rest.len(),
        });
    }
    if x_ab.x_outer.len() != nu {
        return Err(LincheckError::BadOuterLength {
            which: "x_ab",
            expected: nu,
            got: x_ab.x_outer.len(),
        });
    }
    if proof.rounds.len() != inner_rest_len {
        return Err(LincheckError::BadVectorLength {
            which: "rounds",
            expected: inner_rest_len,
            got: proof.rounds.len(),
        });
    }
    if proof.z_partial.len() != n_skip {
        return Err(LincheckError::BadVectorLength {
            which: "z_partial",
            expected: n_skip,
            got: proof.z_partial.len(),
        });
    }
    let expected_nonces = grinding.nonce_count(
        inner_rest_len,
        circuits
            .iter()
            .filter(|circuit| circuit.const_pin_col().is_some())
            .count(),
        k_skip,
    );
    if proof.grinding_nonces.len() != expected_nonces {
        return Err(LincheckError::BadGrindingNonceCount {
            expected: expected_nonces,
            got: proof.grinding_nonces.len(),
        });
    }

    challenger.observe_label(b"flock-lincheck-v0");
    let mut nonce_idx = 0;

    // 1. Sample α (matches prover's order).
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let alpha = challenger
            .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], bits)
            .ok_or(LincheckError::InvalidGrindingNonce { which: "alpha" })?;
        nonce_idx += 1;
        alpha
    } else {
        challenger.sample_f128()
    };

    // 2. The per-type α-batched combs are NOT built here — they are the
    //    only `O(nnz)` step, and they are needed nowhere before the final
    //    consistency check. They ride out in the `MatrixAssertion` instead.
    //    Nothing between here and there touches the challenger on their
    //    account, so lifting them out leaves the transcript untouched.
    //
    //    Merge note (`merkle-path-r1cs`): that branch added a `VERIFY_TRACE`
    //    split of this function into "comb phase" and "everything after".
    //    Deferral removed the comb phase from here entirely, so the
    //    instrumentation had nothing left to time and is not carried over.
    //    The measurement it existed for lived in `verify_union_timed` (which
    //    built combs inline on purpose) until that probe was deleted in the
    //    bloat sweep (ledger §E).

    // 2b. Constant-wire pins (mirror of prove): β_t sampled after α in slot
    //     order, the comb gains +β_t at the pin, and the target gains the
    //     count-derived β_t · Σ_{row<n_t} eq(x_outer, row) — declared rows
    //     carry pin = 1, dummy rows are all-zero (pin included). At full
    //     utilization the sum is exactly 1, reproducing today's target += β.
    //     Only the target half is count-derived work; the comb half is
    //     carried in the assertion.
    let mut target = alpha * v_a + v_b;
    let mut betas: Vec<Option<F128>> = Vec::with_capacity(circuits.len());
    let mut pin_evals: Vec<Option<F128>> = Vec::with_capacity(circuits.len());
    for (circuit, &n_t) in circuits.iter().zip(union.counts()) {
        if circuit.const_pin_col().is_some() {
            let beta = if let Some(bits) = grinding.beta_bits() {
                let beta = challenger
                    .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], bits)
                    .ok_or(LincheckError::InvalidGrindingNonce { which: "beta" })?;
                nonce_idx += 1;
                beta
            } else {
                challenger.sample_f128()
            };
            betas.push(Some(beta));
            let eps = eq_prefix_sum(&x_ab.x_outer, n_t);
            pin_evals.push(Some(eps));
            target += beta * eps;
        } else {
            betas.push(None);
            pin_evals.push(None);
        }
    }

    // 3. Replay the multilinear product-sumcheck rounds. Unlike the
    //    single-table verifier there is no comb fold in lockstep — the
    //    skip-block collapse is evaluated once at the end, in closed form.
    let mut running = target;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    for &(e1, einf) in &proof.rounds {
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = if let Some(bits) = grinding.multilinear_round_bits() {
            let r = challenger
                .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], bits)
                .ok_or(LincheckError::InvalidGrindingNonce {
                    which: "sumcheck-round",
                })?;
            nonce_idx += 1;
            r
        } else {
            challenger.sample_f128()
        };
        // q(0) = claim + q(1) in char 2; q(X) = einf·X² + c1·X + e0.
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
        r_rounds.push(r);
    }

    // 4. Observe z_partial AFTER the sumcheck rounds (matches prover order).
    challenger.observe_f128_slice(&proof.z_partial);

    // 5. The final consistency check — the closed-form Comb-hat collapse
    //    against `running` — is exactly the deferred assertion. `rr` is
    //    LSB-first: the loop bound the TOP coordinate first, so rr[i] binds
    //    column coordinate k_skip + i.
    let mut rr = r_rounds;
    rr.reverse();
    let assertion = MatrixAssertion {
        alpha,
        z_skip: x_ab.z_skip,
        x_inner_rest: x_ab.x_inner_rest.clone(),
        rr: rr.clone(),
        z_partial: proof.z_partial.clone(),
        betas,
        pin_point: x_ab.x_outer.clone(),
        pin_evals,
        target: running,
        evals: proof.matrix_evals.clone(),
    };

    // 6.–7. Fresh skip challenge after z_partial, in the SAME basis as the
    //       input point (identical to the single-table verifier); claim
    //       value via the basis's evaluation functional. Mirrors
    //       `column_sumcheck_prove`'s grinding composition exactly: the φ₈
    //       basis keeps the FUSED PoW+squeeze (byte-pinned), the AG basis
    //       verifies the fused nonce one-shot (`sample_fresh_pow_verifier`).
    let r_inner_skip = match x_ab.z_skip {
        SkipPoint::Phi8(_) => match grinding.skip_bits(k_skip) {
            Some(bits) => {
                let nonce = proof.grinding_nonces[nonce_idx];
                nonce_idx += 1;
                let r = challenger.verify_pow_and_sample_f128(nonce, bits).ok_or(
                    LincheckError::InvalidGrindingNonce {
                        which: "inner-skip",
                    },
                )?;
                SkipPoint::Phi8(r)
            }
            None => x_ab.z_skip.sample_fresh(challenger),
        },
        // AG basis: ONE-SHOT fused-nonce check — 3 explicit bits + the
        // sampler's 5 provable bits = bits_for(158) for the base-code bad set.
        SkipPoint::Ag(_) => match grinding.ag_skip_bits(k_skip) {
            Some(bits) => {
                let nonce = proof.grinding_nonces[nonce_idx];
                nonce_idx += 1;
                x_ab.z_skip
                    .sample_fresh_pow_verifier(challenger, nonce, bits)
                    .ok_or(LincheckError::InvalidGrindingNonce {
                        which: "inner-skip",
                    })?
            }
            None => x_ab.z_skip.sample_fresh(challenger),
        },
    };
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());
    let lambda = r_inner_skip.weights(k_skip);
    let w = inner_product(&lambda, &proof.z_partial);

    Ok((
        LincheckClaim {
            r_inner_skip,
            r_inner_rest: rr,
            w,
        },
        assertion,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::SparseBinaryMatrix;
    use crate::schedule::{TableClass, TableType};

    use crate::test_rng::Rng;

    fn stub() -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            num_rows: 0,
            num_cols: 0,
            rows: Vec::new(),
        }
    }

    fn ty(k_log: usize, useful_bits: usize) -> TableType {
        TableType {
            k_log,
            useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: None,
            class: TableClass::Boolean,
            io_schema: Vec::new(),
        }
    }

    /// `eq_prefix_weight` against the eq-table entry it freezes.
    #[test]
    fn eq_prefix_weight_matches_eq_table() {
        let mut rng = Rng::new(0xE0_11);
        for d in 1..=6usize {
            let coords = rng.f128_vec(d);
            let table = build_eq_table(&coords);
            for bits in 0..(1usize << d) {
                assert_eq!(
                    eq_prefix_weight(&coords, bits),
                    table[bits],
                    "d={d}, bits={bits}"
                );
            }
        }
        // Empty coordinate set: the weight is the empty product.
        assert_eq!(eq_prefix_weight(&[], 0), F128::ONE);
    }

    /// `eq_prefix_sum` against the naive eq-table prefix sum, including the
    /// n = 0 and n = 2^d edges (the latter must be exactly ONE).
    #[test]
    fn eq_prefix_sum_matches_naive() {
        let mut rng = Rng::new(0xE0_55);
        for d in 1..=6usize {
            let point = rng.f128_vec(d);
            let table = build_eq_table(&point);
            let mut naive = F128::ZERO;
            for n in 0..=(1usize << d) {
                assert_eq!(eq_prefix_sum(&point, n), naive, "d={d}, n={n}");
                if n < 1usize << d {
                    naive += table[n];
                }
            }
            assert_eq!(eq_prefix_sum(&point, 1 << d), F128::ONE);
        }
    }

    /// The closed-form [`union_comb_partial`] against a dense union-column
    /// vector folded with the full eq tensor, on a two-type registry with
    /// random per-type "comb" vectors (no circuits needed — the closed form
    /// is pure placement algebra).
    #[test]
    fn union_comb_partial_matches_dense_fold() {
        let k_skip = K_SKIP;
        // κ = 9/8, ν = 2 → areas 2^11 + 2^10, M = 12, column domain 2^10.
        let reg = Registry::new(vec![ty(9, 1 << 9), ty(8, 1 << 8)], 2);
        assert_eq!(reg.m_total(), 12);
        let col_vars = reg.m_total() - reg.nu();
        let mut rng = Rng::new(0xC0_3B);

        let combs: Vec<Vec<F128>> = reg
            .types()
            .iter()
            .map(|t| rng.f128_vec(1usize << t.k_log))
            .collect();
        let rr = rng.f128_vec(col_vars - k_skip);

        // Dense: place each comb at its slot's aligned column offset, fold
        // the multilinear coords with the full eq tensor.
        let mut dense = vec![F128::ZERO; 1usize << col_vars];
        for ((t, slot), comb) in reg.types().iter().zip(reg.slots()).zip(&combs) {
            let off = slot.prefix << t.k_log;
            dense[off..off + comb.len()].copy_from_slice(comb);
        }
        let eq_rr = build_eq_table(&rr);
        let n_skip = 1usize << k_skip;
        let mut expected = vec![F128::ZERO; n_skip];
        for (j, &e) in eq_rr.iter().enumerate() {
            for (i, x) in expected.iter_mut().enumerate() {
                *x += e * dense[(j << k_skip) + i];
            }
        }

        assert_eq!(union_comb_partial(&reg, &combs, &rr, k_skip), expected);
    }
}
