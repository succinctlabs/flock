//! The affine inner opening's RESIDUAL CHECK.
//!
//! The affine verifier ends on `⟨yr_lo, W⟩ = claim`, where `W` over the
//! final `2^c` positions is the caller's basis plus every introduced weight
//! bound at the later fold challenges `r` (`verify_affine_l0`'s contexts).
//! Every term has one of two shapes, and the inner product of either with
//! `yr_lo` folds to a scalar without ever building `W`:
//!
//! ```text
//!   EQ    (basis, OOD):   s · Π_{i<f} (1 + z_i + r_i)        · MLE(yr_lo)(z_f..)
//!   NOVEL (consistency):  wt_q · Π_{i<f} (1 + r_i(1 + Ŵ_i))  · Σ_y yr_y Π_{j: y_j} Ŵ_{f+j}
//! ```
//!
//! with `f = dim − c` the weight's fixed coordinates (`r` = the LAST `f`
//! fold challenges), `Ŵ_i = Ŵ_i(x_q)` the query's ladder and `wt_q =
//! eq(α)_q`. Each term is a prefix product ([`Arith::eq_prefix`] /
//! [`Arith::novel_prefix`]) then a fold of `yr_lo` in the matching mode
//! weighted by it ([`Arith::fold`]); a level's NOVEL terms accumulate
//! unscaled and join with that level's consistency `β` on one MAC row.
//! The caller CONNECTS the returned total to the claim fold's output.

use flock_core::circuit::builder::{ShapeBuilder, Wire};

use crate::tower::affine_gates::{Arith, FoldMode};

/// An `eq(point, ·)` weight with scale `seed` (the basis: `one`; an OOD
/// claim: its `β`). `dim = point.len()`.
pub(super) struct EqContext {
    pub(super) seed: Wire,
    pub(super) point: Vec<Wire>,
}

/// One level's consistency weight: its `β`, and per query the ladder
/// (`Ŵ_0..`, at least `dim` long) and `eq(α)_q`.
pub(super) struct QueryContext {
    pub(super) beta: Wire,
    pub(super) dim: usize,
    pub(super) ladders: Vec<Vec<Wire>>,
    pub(super) weights: Vec<Wire>,
}

/// Emit the residual inner product `⟨yr_lo, W⟩`. `all_rs` are every fold
/// round's `r` in binding order; `yr_lo` the first `2^c` absorbed `yr`
/// words. Returns the total's wire.
pub(super) fn emit_affine_residual(
    sb: &mut ShapeBuilder,
    ar: &Arith,
    all_rs: &[Wire],
    yr_lo: &[Wire],
    eqs: &[EqContext],
    queries: &[QueryContext],
) -> Wire {
    assert!(yr_lo.len().is_power_of_two());
    let c = yr_lo.len().trailing_zeros() as usize;
    let log_n = all_rs.len() + c;
    let fixed = |dim: usize| -> &[Wire] {
        assert!(
            c <= dim && dim <= log_n,
            "a weight's dimension spans the residual"
        );
        &all_rs[log_n - dim..]
    };
    let mut total = ar.zw;
    for e in eqs {
        let dim = e.point.len();
        let f = dim - c;
        let pre = ar.eq_prefix(sb, e.seed, &e.point[..f], fixed(dim));
        total = ar.fold(sb, FoldMode::Eq, yr_lo, &e.point[f..], pre, total);
    }
    for qc in queries {
        assert_eq!(qc.ladders.len(), qc.weights.len());
        let f = qc.dim - c;
        let mut acc = ar.zw;
        for (w, &wt) in qc.ladders.iter().zip(&qc.weights) {
            let pre = ar.novel_prefix(sb, wt, fixed(qc.dim), &w[..f]);
            acc = ar.fold(sb, FoldMode::Novel, yr_lo, &w[f..qc.dim], pre, acc);
        }
        total = ar.mac(sb, total, qc.beta, acc);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::{
        F128, MacGate, PrefixGate, ResidualWeightsGate,
        affine_fixture::affine_fixture,
        affine_fold::{AffineFoldGate, AffineLevelWires, JoinWires, emit_affine_claim_fold},
        affine_gates::{FoldChunkGate, NovelPrefixGate, PREFIX},
        affine_query::{
            AffineFinalLevel, AffineQueryLevel, emit_affine_consistency, emit_affine_final_check,
        },
        tape::AffineOodRec,
    };

    fn input(sb: &mut ShapeBuilder, ins: &mut Vec<F128>, v: F128) -> Wire {
        ins.push(v);
        sb.input()
    }

    /// The whole affine verifier as one circuit over the m = 24 direct
    /// opening's tape values: claim fold, per-level consistency feeding its
    /// joins, the final Enc(yr) checks and the residual, whose total is
    /// CONNECTED to the claim. The circuit runs (every connect agrees) and
    /// the total equals the verifier's final claim; a wrong yr word breaks
    /// a connect.
    #[test]
    fn full_affine_circuit_closes_on_the_claim() {
        let fx = affine_fixture(b"tower-affine-residual");
        let build = |yr_tweak: Option<usize>| {
            let (vals, chals) = (&fx.vals, &fx.chals);
            let mut sb = ShapeBuilder::new(13);
            let fold = sb.slot(AffineFoldGate::new());
            let max_c = *fx.cs.iter().max().unwrap();
            let ladder = sb.slot(ResidualWeightsGate::with_weights(max_c + 1));
            let mut ins = Vec::new();
            let ar = Arith {
                mac: sb.slot(MacGate::new()),
                chunk: sb.slot(FoldChunkGate::new()),
                nprefix: sb.slot(NovelPrefixGate::new()),
                prefix: sb.slot(PrefixGate::new(PREFIX)),
                zw: input(&mut sb, &mut ins, F128::ZERO),
                ow: input(&mut sb, &mut ins, F128::ONE),
            };
            let zassert = input(&mut sb, &mut ins, F128::ZERO);
            let chw =
                |sb: &mut ShapeBuilder, ins: &mut Vec<F128>, i: usize| input(sb, ins, chals[i]);
            let target = input(&mut sb, &mut ins, vals[fx.target_v]);
            let z_w: Vec<Wire> = fx.z.iter().map(|&v| input(&mut sb, &mut ins, v)).collect();
            let mut eqs = vec![EqContext {
                seed: ar.ow,
                point: z_w,
            }];
            let mut ood_join = |sb: &mut ShapeBuilder, ins: &mut Vec<F128>, o: &AffineOodRec| {
                let beta = chw(sb, ins, o.beta_ch);
                let y = input(sb, ins, vals[o.y_v]);
                let point = (0..o.z_len).map(|j| chw(sb, ins, o.z_ch + j)).collect();
                eqs.push(EqContext { seed: beta, point });
                (beta, y)
            };
            let init: Vec<JoinWires> = fx
                .initial_ood
                .iter()
                .map(|o| ood_join(&mut sb, &mut ins, o))
                .collect();
            let fl = fx.levels.len() - 1;
            let mut lvl_w = Vec::new();
            let mut all_rs = Vec::new();
            let mut qctx = Vec::new();
            let mut yr_lo = Vec::new();
            for (li, lvl) in fx.levels.iter().enumerate() {
                let (k, c) = (fx.ks[li], fx.cs[li]);
                let mut rounds = Vec::new();
                let mut gammas = Vec::new();
                for rd in &lvl.rounds {
                    let g: [Wire; 5] =
                        std::array::from_fn(|e| input(&mut sb, &mut ins, vals[rd.grid_v + e]));
                    let ch: [Wire; 3] = std::array::from_fn(|e| chw(&mut sb, &mut ins, rd.ch + e));
                    rounds.push((g, ch));
                    gammas.push((ch[0], ch[1]));
                    all_rs.push(ch[2]);
                }
                let rows: Vec<Vec<Wire>> = fx.proof.openings[li]
                    .opened_rows
                    .iter()
                    .map(|r| r.iter().map(|&v| input(&mut sb, &mut ins, v)).collect())
                    .collect();
                let xs: Vec<Wire> = fx.trace.queries[li]
                    .iter()
                    .map(|&q| input(&mut sb, &mut ins, F128::new(q as u64, 0)))
                    .collect();
                if li == fl {
                    let yr_v = lvl.yr_v.unwrap();
                    let yr: Vec<Wire> = (0..(1 << c) + k)
                        .map(|s| {
                            let mut v = vals[yr_v + s];
                            if yr_tweak == Some(s) {
                                v += F128::ONE;
                            }
                            input(&mut sb, &mut ins, v)
                        })
                        .collect();
                    yr_lo = yr[..1 << c].to_vec();
                    let f = AffineFinalLevel {
                        rows,
                        xs,
                        gammas,
                        yr,
                        log_cols: c,
                    };
                    emit_affine_final_check(&mut sb, &ar, ladder, &f, zassert);
                    lvl_w.push(AffineLevelWires {
                        rounds,
                        ood: vec![],
                        consistency: None,
                    });
                    continue;
                }
                let ov = lvl.overflow_v.unwrap();
                let overflow = (0..k)
                    .map(|s| input(&mut sb, &mut ins, vals[ov + s]))
                    .collect();
                let ood: Vec<JoinWires> = lvl
                    .ood
                    .iter()
                    .map(|o| ood_join(&mut sb, &mut ins, o))
                    .collect();
                let (_, a_ch, a_count) = lvl.alpha.unwrap();
                let alpha = (0..a_count)
                    .map(|j| chw(&mut sb, &mut ins, a_ch + j))
                    .collect();
                let q = AffineQueryLevel {
                    rows,
                    xs,
                    gammas,
                    overflow,
                    log_cols: c,
                    alpha,
                };
                let out = emit_affine_consistency(&mut sb, &ar, ladder, &q);
                let beta = chw(&mut sb, &mut ins, lvl.beta.unwrap().1);
                qctx.push(QueryContext {
                    beta,
                    dim: c,
                    ladders: out.ladders,
                    weights: out.eq,
                });
                lvl_w.push(AffineLevelWires {
                    rounds,
                    ood,
                    consistency: Some((beta, out.enforced)),
                });
            }
            let claim = emit_affine_claim_fold(&mut sb, fold, ar.mac, target, &init, &lvl_w);
            assert_eq!(all_rs.len() + fx.cs[fl], fx.log_n);
            let total = emit_affine_residual(&mut sb, &ar, &all_rs, &yr_lo, &eqs, &qctx);
            sb.connect(total, claim);
            sb.publish(total);
            (sb.finish().expect("the affine circuit builds"), ins)
        };

        let (shape, ins) = build(None);
        let built = shape.run(&ins, &[]);
        assert_eq!(
            built.public,
            vec![fx.trace.final_claim],
            "residual total = verifier claim"
        );

        // A wrong yr_lo word shifts the residual total off the claim.
        let (shape, ins) = build(Some(3));
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shape.run(&ins, &[])));
        assert!(res.is_err(), "a wrong yr word breaks the claim connect");
    }
}
