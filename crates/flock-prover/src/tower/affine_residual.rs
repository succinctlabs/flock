//! The affine inner opening's RESIDUAL CHECK — the F128 replacement for the
//! split walker's residual region (`query::emit_residual_region`).
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
//! eq(α)_q`. One [`AffineResidualGate`] row per term; a level's NOVEL terms
//! accumulate unscaled and join with that level's consistency `β` on one
//! [`MacGate`] row. The caller CONNECTS the returned total to the claim
//! fold's output.
//!
//! [`MacGate`]: crate::tower::MacGate
#![cfg_attr(not(test), allow(dead_code))]

use std::sync::Arc;

use flock_core::{
    circuit::builder::{GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
    element_r1cs::{ElementTableBuilder, ElementTableType},
    schedule::{IoWord, TableType},
};
use flock_field::F128;

/// Which weight shape a residual row folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResidualMode {
    /// `eq(point, ·)`: the basis and every OOD claim.
    Eq,
    /// The novel-basis query weight of a consistency claim.
    Novel,
}

/// The column layout of an [`AffineResidualGate`]. Inputs: `yr_lo[2^c]`,
/// `pt[f + c]` (the point, or the query's ladder `Ŵ_0..`), `r[f]`, `seed`,
/// `one`, `acc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResidualLayout {
    pub(super) mode: ResidualMode,
    pub(super) f: usize,
    pub(super) c: usize,
}

impl ResidualLayout {
    pub(super) fn n_in(&self) -> usize {
        (1 << self.c) + 2 * self.f + self.c + 3
    }
    fn pt(&self, i: usize) -> usize {
        (1 << self.c) + i
    }
    fn r(&self, i: usize) -> usize {
        self.pt(self.f + self.c) + i
    }
    fn seed(&self) -> usize {
        self.r(self.f)
    }
    fn one(&self) -> usize {
        self.seed() + 1
    }
    fn acc(&self) -> usize {
        self.seed() + 2
    }
}

/// One residual term folded into a running accumulator, `acc' = acc +
/// prefix · fold` (the module docs give both shapes). The prefix is a chain
/// of products from `seed`; the fold keeps its values as linear forms, one
/// product column per node:
///
/// ```text
///   EQ:     prefix' = prefix·(1 + z_i + r_i)          new = lo + z·(lo + hi)
///   NOVEL:  p = r_i·Ŵ_i, prefix' = prefix·(1 + r_i + p)   new = lo + Ŵ·hi
/// ```
pub(super) struct AffineResidualGate {
    pub(super) ty: Arc<ElementTableType>,
    pub(super) lay: ResidualLayout,
    /// `(out, a, b)`: `out = Σa · Σb`, or `Σa` when `b` is empty.
    rows: Vec<(usize, Vec<usize>, Vec<usize>)>,
    width: usize,
}

impl AffineResidualGate {
    pub(super) fn new(lay: ResidualLayout) -> Self {
        let mut rows: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
        let mut next = lay.n_in();
        let mut col = |rows: &mut Vec<_>, a: Vec<usize>, b: Vec<usize>| -> usize {
            rows.push((next, a, b));
            next += 1;
            next - 1
        };
        let mut prefix = lay.seed();
        for i in 0..lay.f {
            let factor = match lay.mode {
                ResidualMode::Eq => vec![lay.one(), lay.pt(i), lay.r(i)],
                ResidualMode::Novel => {
                    let p = col(&mut rows, vec![lay.r(i)], vec![lay.pt(i)]);
                    vec![lay.one(), lay.r(i), p]
                }
            };
            prefix = col(&mut rows, vec![prefix], factor);
        }
        let mut cur: Vec<Vec<usize>> = (0..1usize << lay.c).map(|y| vec![y]).collect();
        for j in 0..lay.c {
            let g = vec![lay.pt(lay.f + j)];
            cur = cur
                .chunks(2)
                .map(|p| {
                    let b = match lay.mode {
                        ResidualMode::Eq => [p[0].clone(), p[1].clone()].concat(),
                        ResidualMode::Novel => p[1].clone(),
                    };
                    let e = col(&mut rows, g.clone(), b);
                    let mut f = p[0].clone();
                    f.push(e);
                    f
                })
                .collect();
        }
        let fold = cur.pop().expect("a folded value");
        let m = col(&mut rows, vec![prefix], fold);
        let out = col(&mut rows, vec![lay.acc(), m], vec![]);
        let width = out + 1;

        let one =
            |f: &[usize]| -> Vec<(usize, F128)> { f.iter().map(|&c| (c, F128::ONE)).collect() };
        let kappa = width.next_power_of_two().trailing_zeros().max(2) as usize;
        let mut b = ElementTableBuilder::new(kappa);
        for w in 0..lay.n_in() {
            b.free_wire(w);
        }
        for (out, a, bb) in &rows {
            if bb.is_empty() {
                b.linear(*out, &one(a));
            } else {
                b.mult_lin(*out, &one(a), &one(bb));
            }
        }
        Self {
            ty: Arc::new(b.build().expect("affine residual gate")),
            lay,
            rows,
            width,
        }
    }

    pub(super) fn out(&self) -> usize {
        self.width - 1
    }
}

impl GateType for AffineResidualGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..self.lay.n_in()).map(IoWord::input).collect();
        schema.push(IoWord::output(self.out()));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; self.width];
        z[..self.lay.n_in()].copy_from_slice(&inputs[..self.lay.n_in()]);
        let sum = |z: &[F128], f: &[usize]| f.iter().fold(F128::ZERO, |acc, &c| acc + z[c]);
        for (out, a, b) in &self.rows {
            let av = sum(&z, a);
            z[*out] = if b.is_empty() { av } else { av * sum(&z, b) };
        }
        outputs.push(z[self.out()]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

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

/// Emit the residual inner product `⟨yr_lo, W⟩`: one [`AffineResidualGate`]
/// row per EQ context and per query, slots cached by layout in `cache`.
/// `all_rs` are every fold round's `r` in binding order; `yr_lo` the first
/// `2^c` absorbed `yr` words. Returns the total's wire.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_affine_residual(
    sb: &mut ShapeBuilder,
    cache: &mut Vec<(ResidualLayout, SlotId)>,
    mac: SlotId,
    all_rs: &[Wire],
    yr_lo: &[Wire],
    eqs: &[EqContext],
    queries: &[QueryContext],
    zw: Wire,
    ow: Wire,
) -> Wire {
    assert!(yr_lo.len().is_power_of_two());
    let c = yr_lo.len().trailing_zeros() as usize;
    let log_n = all_rs.len() + c;
    let mut slot = |sb: &mut ShapeBuilder, lay: ResidualLayout| -> SlotId {
        if let Some(&(_, s)) = cache.iter().find(|e| e.0 == lay) {
            return s;
        }
        let s = sb.slot(AffineResidualGate::new(lay));
        cache.push((lay, s));
        s
    };
    let mut row = |sb: &mut ShapeBuilder, mode, pt: &[Wire], seed: Wire, acc: Wire| -> Wire {
        let dim = pt.len();
        assert!(
            c <= dim && dim <= log_n,
            "a weight's dimension spans the residual"
        );
        let lay = ResidualLayout {
            mode,
            f: dim - c,
            c,
        };
        let s = slot(sb, lay);
        let mut ins: Vec<Wire> = yr_lo.to_vec();
        ins.extend_from_slice(pt);
        ins.extend_from_slice(&all_rs[log_n - dim..]);
        ins.extend([seed, ow, acc]);
        sb.gate(s, &ins)[0]
    };
    let mut total = zw;
    for e in eqs {
        total = row(sb, ResidualMode::Eq, &e.point, e.seed, total);
    }
    for qc in queries {
        assert_eq!(qc.ladders.len(), qc.weights.len());
        let mut acc = zw;
        for (w, &wt) in qc.ladders.iter().zip(&qc.weights) {
            acc = row(sb, ResidualMode::Novel, &w[..qc.dim], wt, acc);
        }
        total = sb.gate(mac, &[total, qc.beta, acc])[0];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::{
        MacGate, ResidualWeightsGate256,
        affine_fixture::affine_fixture,
        affine_fold::{AffineFoldGate, AffineLevelWires, JoinWires, emit_affine_claim_fold},
        affine_query::{
            AffineFinalGate, AffineFinalLayout, AffineFinalLevel, AffineLeafGate, AffineLeafLayout,
            AffineQueryLevel, emit_affine_consistency, emit_affine_final_check,
        },
    };

    fn input(sb: &mut ShapeBuilder, ins: &mut Vec<F128>, v: F128) -> Wire {
        ins.push(v);
        sb.input()
    }

    /// The whole affine verifier as one circuit over the m = 24 direct
    /// opening's tape values: claim fold (step 1), per-level consistency
    /// feeding its joins (step 2), the final Enc(yr) checks (step 3) and the
    /// residual (this module), whose total is CONNECTED to the claim. The
    /// circuit runs (every connect agrees), every residual row satisfies its
    /// constraints, and the total equals the verifier's final claim. A
    /// wrong yr word breaks the connect.
    #[test]
    fn full_affine_circuit_closes_on_the_claim() {
        let fx = affine_fixture(b"tower-affine-residual");
        let build = |yr_tweak: Option<usize>| {
            let (vals, chals) = (&fx.vals, &fx.chals);
            let mut sb = ShapeBuilder::new(11);
            let mac = sb.slot(MacGate::new());
            let fold = sb.slot(AffineFoldGate::new());
            let max_c = *fx.cs.iter().max().unwrap();
            let ladder = sb.slot(ResidualWeightsGate256::with_weights(max_c + 1));
            let mut ins = Vec::new();
            let zw = input(&mut sb, &mut ins, F128::ZERO);
            let ow = input(&mut sb, &mut ins, F128::ONE);
            let chw =
                |sb: &mut ShapeBuilder, ins: &mut Vec<F128>, i: usize| input(sb, ins, chals[i]);
            let target = input(&mut sb, &mut ins, vals[fx.target_v]);
            let z_w: Vec<Wire> = fx.z.iter().map(|&v| input(&mut sb, &mut ins, v)).collect();
            let mut eqs = vec![EqContext {
                seed: ow,
                point: z_w,
            }];
            let mut ood_join =
                |sb: &mut ShapeBuilder,
                 ins: &mut Vec<F128>,
                 o: &crate::tower::tape::AffineOodRec| {
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
            let mut leaf_slots: Vec<(usize, usize, SlotId)> = Vec::new();
            let mut final_parts = None;
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
                let rows_v = &fx.proof.openings[li].opened_rows;
                let rows: Vec<Vec<Wire>> = rows_v
                    .iter()
                    .map(|r| r.iter().map(|&v| input(&mut sb, &mut ins, v)).collect())
                    .collect();
                let xs: Vec<Wire> = fx.trace.queries[li]
                    .iter()
                    .map(|&q| input(&mut sb, &mut ins, F128::new(q as u64, 0)))
                    .collect();
                if li == fl {
                    let yr_v = lvl.yr_v.unwrap();
                    let n = (1 << c) + k;
                    let yr: Vec<Wire> = (0..n)
                        .map(|s| {
                            let mut v = vals[yr_v + s];
                            if yr_tweak == Some(s) {
                                v += F128::ONE;
                            }
                            input(&mut sb, &mut ins, v)
                        })
                        .collect();
                    final_parts = Some((rows, xs, gammas, yr));
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
                let lanes = rows_v[0].len();
                let leaf = match leaf_slots.iter().find(|s| s.0 == lanes && s.1 == k) {
                    Some(s) => s.2,
                    None => {
                        let s = sb.slot(AffineLeafGate::new(AffineLeafLayout::new(lanes, k, k)));
                        leaf_slots.push((lanes, k, s));
                        s
                    }
                };
                let q = AffineQueryLevel {
                    rows,
                    xs,
                    gammas,
                    overflow,
                    log_cols: c,
                    alpha,
                };
                let out = emit_affine_consistency(&mut sb, leaf, ladder, mac, &q, zw, ow);
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
            let claim = emit_affine_claim_fold(&mut sb, fold, mac, target, &init, &lvl_w);
            let (rows, xs, gammas, yr) = final_parts.unwrap();
            let (fk, fc) = (fx.ks[fl], fx.cs[fl]);
            let fin = sb.slot(AffineFinalGate::new(AffineFinalLayout::new(
                rows[0].len(),
                fk,
                fc,
            )));
            let yr_lo = yr[..1 << fc].to_vec();
            let final_lvl = AffineFinalLevel {
                rows,
                xs,
                gammas,
                yr,
                log_cols: fc,
            };
            emit_affine_final_check(&mut sb, fin, ladder, &final_lvl, zw, ow);
            assert_eq!(all_rs.len() + fc, fx.log_n);
            let mut cache = Vec::new();
            let total = emit_affine_residual(
                &mut sb, &mut cache, mac, &all_rs, &yr_lo, &eqs, &qctx, zw, ow,
            );
            sb.connect(total, claim);
            let shape = sb.finish().expect("the affine circuit builds");
            (shape, ins, cache, mac)
        };

        let (shape, ins, cache, mac) = build(None);
        let built = shape.run(&ins, &[]);
        for &(lay, s) in &cache {
            let ty = AffineResidualGate::new(lay).ty;
            for r in built.rows::<AffineResidualGate>(s) {
                let mut z = r.clone();
                z.resize(ty.width(), F128::ZERO);
                assert!(ty.satisfies(&z, 0, 1), "{lay:?} row");
            }
        }
        let last_mac = built.rows::<MacGate>(mac).last().unwrap();
        assert_eq!(
            last_mac[4], fx.trace.final_claim,
            "residual total = verifier claim"
        );
        let n_eq = 1 + fx.initial_ood.len() + fx.levels.iter().map(|l| l.ood.len()).sum::<usize>();
        let n_q: usize = fx.trace.queries[..fx.levels.len() - 1]
            .iter()
            .map(Vec::len)
            .sum();
        let residual_rows: usize = cache
            .iter()
            .map(|&(_, s)| built.rows::<AffineResidualGate>(s).len())
            .sum();
        assert_eq!(residual_rows, n_eq + n_q, "one residual row per context");
        println!(
            "affine m24 circuit: {} eq + {} query residual rows over {} layouts",
            n_eq,
            n_q,
            cache.len()
        );

        // A wrong yr_lo word shifts the residual total off the claim.
        let (shape, ins, ..) = build(Some(3));
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shape.run(&ins, &[])));
        assert!(res.is_err(), "a wrong yr word breaks the claim connect");
    }

    /// The residual gate's sizes at the tower's affine shapes (`--nocapture`).
    #[test]
    fn residual_gate_sizes_at_the_tower_shapes() {
        use flock_core::pcs::ligerito::{LigeritoProfile, affine::affine_ladder_for};
        for (name, profile, log_n, initial_k) in [
            ("slim29", LigeritoProfile::SlimAffine, 22, 5),
            ("fast32", LigeritoProfile::FastAffine, 25, 6),
        ] {
            let (pc, _) = affine_ladder_for(profile, log_n, initial_k).expect("registered");
            let c = *pc.recursive_log_msg_cols.last().unwrap();
            let mut dims = vec![(ResidualMode::Eq, log_n)];
            dims.push((ResidualMode::Novel, log_n - initial_k));
            for &d in &pc.recursive_log_msg_cols[..pc.recursive_log_msg_cols.len() - 1] {
                dims.push((ResidualMode::Novel, d));
            }
            for (mode, dim) in dims {
                let g = AffineResidualGate::new(ResidualLayout {
                    mode,
                    f: dim - c,
                    c,
                });
                println!(
                    "{name} {mode:?} dim {dim} (f {}, c {c}): {} inputs, {} cols, kappa {}",
                    dim - c,
                    g.lay.n_in(),
                    g.out() + 1,
                    g.ty.kappa()
                );
            }
        }
    }
}
