//! The affine inner opening's QUERY-PHASE CHECKS.
//!
//! At every non-final level the affine verifier
//! (`flock_core::pcs::ligerito::affine::verify_affine_l0`) checks the opened
//! rows against the next level's committed low part through ONE batched
//! value,
//!
//! ```text
//!   enforced = Σ_q eq(α)_q · cv_q,
//!   cv_q     = Σ_i ω_i(x_q)·row_q[i]  +  Ŵ_c(x_q) · Σ_{s<k} τ_s·X_s(x_q),
//!   ω_i(x)   = Π_j (α_j + β_j·x)^{i_j},
//! ```
//!
//! where `x_q` is the query position embedded in F128, `(α_j, β_j)` are the
//! level's fold challenges, `τ` is the absorbed overflow, `Ŵ_b` are the
//! normalized subspace polynomials and `X_s = Π_{b: s_b} Ŵ_b` is the novel
//! basis. Both sums are NOVEL folds ([`Arith::fold`]): the row over
//! `g_j = α_j + β_j·x` (one MAC each), the overflow over `Ŵ_0..`.
//! The FINAL level has no overflow and no batching: each query's folded row
//! must equal `Enc(yr)(x_q) = Σ_s yr_s X_s(x_q)`, the novel fold of `yr`
//! over `Ŵ_0..Ŵ_c`; the sum of both sides is connected to an assert-zero
//! wire.
//!
//! Every weight is computed in the circuit from wires: the `Ŵ` ladder by
//! the shared [`ResidualWeightsGate`](crate::tower::ResidualWeightsGate)
//! relation from the position wire, the `eq(α)` tensor by MAC rows from the
//! squeezed alpha wires. No weight enters as checker-tier advice.

use flock_core::circuit::builder::{ShapeBuilder, SlotId, Wire};

use crate::tower::affine_gates::{Arith, FoldMode};

/// `eq(α, ·)` over `2^|α|` entries, LSB-first (`build_eq_table`'s order),
/// from the squeezed alpha wires: two MAC rows per entry.
pub(super) fn emit_eq_tensor(sb: &mut ShapeBuilder, ar: &Arith, alpha: &[Wire]) -> Vec<Wire> {
    let mut aw = vec![ar.ow];
    for &r in alpha {
        let old = std::mem::take(&mut aw);
        aw.extend(old.iter().map(|&x| ar.mac(sb, x, x, r)));
        aw.extend(old.iter().map(|&x| ar.mac(sb, ar.zw, x, r)));
    }
    aw
}

/// The normalized subspace ladder `Ŵ_0(x) .. Ŵ_{n-1}(x)` at a position wire:
/// `x` itself, then one `ResidualWeightsGate::with_weights` row.
pub(super) fn emit_subspace_ladder(
    sb: &mut ShapeBuilder,
    ladder: SlotId,
    x: Wire,
    ow: Wire,
) -> Vec<Wire> {
    let mut w = vec![x];
    w.extend(sb.gate(ladder, &[x, ow]));
    w
}

/// `g_j = α_j + β_j·x`, one MAC each.
fn row_challenges(
    sb: &mut ShapeBuilder,
    ar: &Arith,
    gammas: &[(Wire, Wire)],
    x: Wire,
) -> Vec<Wire> {
    gammas
        .iter()
        .map(|&(alpha, beta)| ar.mac(sb, alpha, beta, x))
        .collect()
}

/// One non-final level's query-phase wires.
pub(super) struct AffineQueryLevel {
    /// Per query: the opened row's live lanes.
    pub(super) rows: Vec<Vec<Wire>>,
    /// Per query: the position wire (`x_q`).
    pub(super) xs: Vec<Wire>,
    /// The level's fold challenges `(α_j, β_j)`, in binding order.
    pub(super) gammas: Vec<(Wire, Wire)>,
    /// The absorbed overflow `τ`.
    pub(super) overflow: Vec<Wire>,
    /// The level's own `c` (its message's log column count).
    pub(super) log_cols: usize,
    /// The consistency alpha squeeze, `ceil_log2(q)` wires.
    pub(super) alpha: Vec<Wire>,
}

/// What [`emit_affine_consistency`] leaves for the claim fold and the
/// residual: the `enforced` wire (the value the claim fold joins), each
/// query's ladder `Ŵ_0..`, and the `eq(α)` tensor (one weight per query).
pub(super) struct AffineConsistency {
    pub(super) enforced: Wire,
    pub(super) ladders: Vec<Vec<Wire>>,
    pub(super) eq: Vec<Wire>,
}

/// Emit one level's batched consistency: the level's `eq(α)` tensor, then
/// per query a ladder row, `k` MACs for the row challenges, the row fold
/// (weight `eq(α)_q`) and the overflow fold (weight `eq(α)_q · Ŵ_c`), both
/// accumulating into the level's `enforced`.
pub(super) fn emit_affine_consistency(
    sb: &mut ShapeBuilder,
    ar: &Arith,
    ladder: SlotId,
    lvl: &AffineQueryLevel,
) -> AffineConsistency {
    assert_eq!(lvl.rows.len(), lvl.xs.len());
    let mut eq = emit_eq_tensor(sb, ar, &lvl.alpha);
    assert!(
        eq.len() >= lvl.xs.len(),
        "the alpha tensor covers every query"
    );
    eq.truncate(lvl.xs.len());
    let bits = lvl.overflow.len().next_power_of_two().trailing_zeros() as usize;
    let mut acc = ar.zw;
    let mut ladders = Vec::with_capacity(lvl.xs.len());
    for ((row, &x), &wt) in lvl.rows.iter().zip(&lvl.xs).zip(&eq) {
        let w = emit_subspace_ladder(sb, ladder, x, ar.ow);
        let g = row_challenges(sb, ar, &lvl.gammas, x);
        acc = ar.fold(sb, FoldMode::Novel, row, &g, wt, acc);
        let wc = ar.mac(sb, ar.zw, wt, w[lvl.log_cols]);
        acc = ar.fold(sb, FoldMode::Novel, &lvl.overflow, &w[..bits], wc, acc);
        ladders.push(w);
    }
    AffineConsistency {
        enforced: acc,
        ladders,
        eq,
    }
}

/// The final level's query-phase wires.
pub(super) struct AffineFinalLevel {
    /// Per query: the opened row's live lanes.
    pub(super) rows: Vec<Vec<Wire>>,
    /// Per query: the position wire (`x_q`).
    pub(super) xs: Vec<Wire>,
    /// The level's fold challenges `(α_j, β_j)`, in binding order.
    pub(super) gammas: Vec<(Wire, Wire)>,
    /// The absorbed `yr`, `2^c + k` words.
    pub(super) yr: Vec<Wire>,
    /// The final level's own `c`.
    pub(super) log_cols: usize,
}

/// Emit the final level's direct checks: per query a ladder row, the row
/// fold and `Enc(yr)(x_q)` summed into one value CONNECTED to `zassert`
/// (a zero wire no gate consumes). Returns each query's ladder.
pub(super) fn emit_affine_final_check(
    sb: &mut ShapeBuilder,
    ar: &Arith,
    ladder: SlotId,
    lvl: &AffineFinalLevel,
    zassert: Wire,
) -> Vec<Vec<Wire>> {
    assert_eq!(lvl.rows.len(), lvl.xs.len());
    let c = lvl.log_cols;
    assert_eq!(lvl.yr.len(), (1 << c) + lvl.gammas.len());
    assert!(
        lvl.gammas.len() <= 1 << c,
        "the overflow fits under X_{{2^c}}"
    );
    let mut ladders = Vec::with_capacity(lvl.xs.len());
    for (row, &x) in lvl.rows.iter().zip(&lvl.xs) {
        let w = emit_subspace_ladder(sb, ladder, x, ar.ow);
        let g = row_challenges(sb, ar, &lvl.gammas, x);
        let folded = ar.fold(sb, FoldMode::Novel, row, &g, ar.ow, ar.zw);
        let check = ar.fold(sb, FoldMode::Novel, &lvl.yr, &w[..=c], ar.ow, folded);
        sb.connect(check, zassert);
        ladders.push(w);
    }
    ladders
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::{
        F128, MacGate, PrefixGate, ResidualWeightsGate,
        affine_fixture::affine_fixture,
        affine_gates::{FoldChunkGate, NovelPrefixGate, PREFIX},
    };

    fn input(sb: &mut ShapeBuilder, ins: &mut Vec<F128>, v: F128) -> Wire {
        ins.push(v);
        sb.input()
    }

    /// A direct affine opening at the registered m = 24 shape, verified
    /// through the recording challenger. Every non-final level's emitted
    /// consistency outputs exactly the verifier's own `enforced` value, and
    /// every final-level check runs (its connect to zero agrees); a wrong
    /// `yr` word breaks that connect.
    #[test]
    fn query_checks_match_the_verifier() {
        let fx = affine_fixture(b"tower-affine-query");
        let (proof, trace, ks, cs, levels) = (&fx.proof, &fx.trace, &fx.ks, &fx.cs, &fx.levels);
        let (vals, chals) = (&fx.vals, &fx.chals);
        let build = |yr_tweak: Option<usize>| {
            let mut sb = ShapeBuilder::new(12);
            let max_c = *cs.iter().max().unwrap();
            let ladder = sb.slot(ResidualWeightsGate::with_weights(max_c + 1));
            let mut ins: Vec<F128> = Vec::new();
            let ar = Arith {
                mac: sb.slot(MacGate::new()),
                chunk: sb.slot(FoldChunkGate::new()),
                nprefix: sb.slot(NovelPrefixGate::new()),
                prefix: sb.slot(PrefixGate::new(PREFIX)),
                zw: input(&mut sb, &mut ins, F128::ZERO),
                ow: input(&mut sb, &mut ins, F128::ONE),
            };
            let zassert = input(&mut sb, &mut ins, F128::ZERO);
            let fl = levels.len() - 1;
            let mut outs = Vec::new();
            for (li, lvl) in levels.iter().enumerate() {
                let (k, c) = (ks[li], cs[li]);
                let rows: Vec<Vec<Wire>> = proof.openings[li]
                    .opened_rows
                    .iter()
                    .map(|r| r.iter().map(|&v| input(&mut sb, &mut ins, v)).collect())
                    .collect();
                let xs: Vec<Wire> = trace.queries[li]
                    .iter()
                    .map(|&q| input(&mut sb, &mut ins, F128::new(q as u64, 0)))
                    .collect();
                let gammas: Vec<(Wire, Wire)> = lvl
                    .rounds
                    .iter()
                    .map(|rd| {
                        (
                            input(&mut sb, &mut ins, chals[rd.ch]),
                            input(&mut sb, &mut ins, chals[rd.ch + 1]),
                        )
                    })
                    .collect();
                if li == fl {
                    let yr_v = lvl.yr_v.expect("final yr");
                    let yr = (0..(1 << c) + k)
                        .map(|s| {
                            let bump = if yr_tweak == Some(s) {
                                F128::ONE
                            } else {
                                F128::ZERO
                            };
                            input(&mut sb, &mut ins, vals[yr_v + s] + bump)
                        })
                        .collect();
                    let f = AffineFinalLevel {
                        rows,
                        xs,
                        gammas,
                        yr,
                        log_cols: c,
                    };
                    emit_affine_final_check(&mut sb, &ar, ladder, &f, zassert);
                    continue;
                }
                let ov = lvl.overflow_v.expect("non-final overflow");
                let overflow = (0..k)
                    .map(|s| input(&mut sb, &mut ins, vals[ov + s]))
                    .collect();
                let (_, a_ch, a_count) = lvl.alpha.expect("non-final alpha");
                let alpha = (0..a_count)
                    .map(|j| input(&mut sb, &mut ins, chals[a_ch + j]))
                    .collect();
                let q = AffineQueryLevel {
                    rows,
                    xs,
                    gammas,
                    overflow,
                    log_cols: c,
                    alpha,
                };
                outs.push(emit_affine_consistency(&mut sb, &ar, ladder, &q).enforced);
            }
            for &o in &outs {
                sb.publish(o);
            }
            (
                sb.finish().expect("the query-check circuit builds"),
                ins,
                ladder,
            )
        };
        let (shape, ins, ladder) = build(None);
        let built = shape.run(&ins, &[]);
        assert_eq!(
            built.public, trace.enforced,
            "every level's enforced = verifier"
        );
        let total_q: usize = trace.queries.iter().map(Vec::len).sum();
        let ladder_rows = built.rows::<ResidualWeightsGate>(ladder).len();
        assert_eq!(ladder_rows, total_q, "one ladder row per query");

        let (shape, ins, _) = build(Some(2));
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shape.run(&ins, &[])));
        assert!(res.is_err(), "a wrong yr word breaks the final check");
    }
}
