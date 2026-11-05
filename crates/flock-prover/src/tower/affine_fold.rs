//! The affine inner opening's CLAIM FOLD — the F128 replacement for the
//! F256 Ligerito sumcheck spine (`gates_spine::SpineGate256`,
//! `walker_common::emit_ligerito_spine_walk`).
//!
//! The affine verifier (`flock_core::pcs::ligerito::affine::verify_affine_l0`)
//! runs no sumcheck: each fold round sends a 5-word grid
//! `[G00, G01, G10, G20, G21]` (`G11 = v + G00` is implied) and folds the
//! running claim linearly,
//!
//! ```text
//!   v' = p(G00, G01) + α·p(G10, G11) + β·p(G20, G21),   p(x, y) = x + r·(x + y),
//! ```
//!
//! and each OOD claim and each level's batched consistency value join the
//! claim as `v += β·y`. [`AffineFoldGate`] is one fold round in ONE element
//! row; the joins are rows of the shared [`MacGate`]. The claim that leaves
//! the last round is the one the residual check binds.
//!
//! Not yet consumed outside tests: the affine child walker (query phase,
//! final check, residual region) is the next step of the tower port.
#![cfg_attr(not(test), allow(dead_code))]

use std::sync::Arc;

use flock_core::{
    circuit::builder::{GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
    element_r1cs::{ElementTableBuilder, ElementTableType},
    pcs::ligerito::affine::fold_claim,
    schedule::{IoWord, TableType},
};
use flock_field::F128;

use crate::tower::tape::{AffineLevel, AffineOodRec};

/// One affine fold round: inputs `claim(0), G00(1), G01(2), G10(3), G20(4),
/// G21(5), α(6), β(7), r(8)`, output `v'`(14). Each product of two linear
/// forms is one row (`mult_lin`), so the round costs five products and one
/// linear join:
///
/// ```text
///   z9  = r·(G00 + G01)          z12 = α·(G10 + z10)
///   z10 = r·(G10 + claim + G00)  z13 = β·(G20 + z11)
///   z11 = r·(G20 + G21)          z14 = G00 + z9 + z12 + z13
/// ```
pub(super) struct AffineFoldGate {
    pub(super) ty: Arc<ElementTableType>,
}

/// The output column of [`AffineFoldGate`].
pub(super) const FOLD_OUT: usize = 14;

impl AffineFoldGate {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        let mut b = ElementTableBuilder::new(4);
        for w in 0..9 {
            b.free_wire(w);
        }
        b.mult_lin(9, &[(1, o), (2, o)], &[(8, o)]);
        b.mult_lin(10, &[(3, o), (0, o), (1, o)], &[(8, o)]);
        b.mult_lin(11, &[(4, o), (5, o)], &[(8, o)]);
        b.mult_lin(12, &[(6, o)], &[(3, o), (10, o)]);
        b.mult_lin(13, &[(7, o)], &[(4, o), (11, o)]);
        b.linear(FOLD_OUT, &[(1, o), (9, o), (12, o), (13, o)]);
        Self {
            ty: Arc::new(b.build().expect("affine fold gate")),
        }
    }
}

impl GateType for AffineFoldGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..9).map(IoWord::input).collect();
        schema.push(IoWord::output(FOLD_OUT));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; self.ty.width()];
        z[..9].copy_from_slice(&inputs[..9]);
        z[9] = z[8] * (z[1] + z[2]);
        z[10] = z[8] * (z[3] + z[0] + z[1]);
        z[11] = z[8] * (z[4] + z[5]);
        z[12] = z[6] * (z[3] + z[10]);
        z[13] = z[7] * (z[4] + z[11]);
        z[FOLD_OUT] = z[1] + z[9] + z[12] + z[13];
        outputs.push(z[FOLD_OUT]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// The native claim fold over a parsed affine tape: from the opening's
/// target, through every L0 OOD join, fold round, recursive OOD join and
/// consistency join, to the claim the residual check binds. `enforced[li]`
/// is level `li`'s batched consistency value (the query phase computes it).
pub(super) fn replay_affine_claim(
    target: F128,
    initial_ood: &[AffineOodRec],
    levels: &[AffineLevel],
    vals: &[F128],
    chals: &[F128],
    enforced: &[F128],
) -> F128 {
    let join = |claim: &mut F128, o: &AffineOodRec| *claim += chals[o.beta_ch] * vals[o.y_v];
    let mut claim = target;
    initial_ood.iter().for_each(|o| join(&mut claim, o));
    for (li, lvl) in levels.iter().enumerate() {
        for rd in &lvl.rounds {
            let g = &vals[rd.grid_v..rd.grid_v + 5];
            let grid = [g[0], g[1], g[2], claim + g[0], g[3], g[4]];
            claim = fold_claim(&grid, chals[rd.ch], chals[rd.ch + 1], chals[rd.ch + 2]);
        }
        lvl.ood.iter().for_each(|o| join(&mut claim, o));
        if let Some((_, beta_ch)) = lvl.beta {
            claim += chals[beta_ch] * enforced[li];
        }
    }
    claim
}

/// One OOD or consistency join: `(β, value)`.
pub(super) type JoinWires = (Wire, Wire);

/// One level's claim-fold wires, in transcript order: its fold rounds
/// (`[G00, G01, G10, G20, G21]` and `[α, β, r]`), then its OOD joins, then
/// (non-final levels) its consistency join.
pub(super) struct AffineLevelWires {
    pub(super) rounds: Vec<([Wire; 5], [Wire; 3])>,
    pub(super) ood: Vec<JoinWires>,
    pub(super) consistency: Option<JoinWires>,
}

/// Emit the claim fold: `initial_ood` joins onto `target`, then each
/// level's rounds as [`AffineFoldGate`] rows and its joins as [`MacGate`]
/// rows on `mac`. Returns the final claim's wire.
///
/// [`MacGate`]: crate::tower::MacGate
pub(super) fn emit_affine_claim_fold(
    sb: &mut ShapeBuilder,
    fold: SlotId,
    mac: SlotId,
    target: Wire,
    initial_ood: &[JoinWires],
    levels: &[AffineLevelWires],
) -> Wire {
    let join = |sb: &mut ShapeBuilder, claim: Wire, (beta, y): JoinWires| -> Wire {
        sb.gate(mac, &[claim, beta, y])[0]
    };
    let mut claim = target;
    for &j in initial_ood {
        claim = join(sb, claim, j);
    }
    for lvl in levels {
        for &(g, [alpha, beta, r]) in &lvl.rounds {
            claim = sb.gate(fold, &[claim, g[0], g[1], g[2], g[3], g[4], alpha, beta, r])[0];
        }
        for &j in &lvl.ood {
            claim = join(sb, claim, j);
        }
        if let Some(j) = lvl.consistency {
            claim = join(sb, claim, j);
        }
    }
    claim
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::{MacGate, affine_fixture::affine_fixture};

    /// A direct affine opening at the registered m = 24 shape, verified
    /// through the RECORDING challenger: the native replay over the parsed
    /// tape lands on the verifier's own final claim, and the emitted rows
    /// (one [`AffineFoldGate`] per round, one MAC per join) satisfy their
    /// constraint systems and output that same claim. A tampered fold row
    /// fails its constraints.
    #[test]
    fn claim_fold_matches_the_verifier() {
        let fx = affine_fixture(b"tower-affine-claim-fold");
        let (target, trace, ks) = (fx.target, &fx.trace, &fx.ks);
        let (initial_ood, levels) = (&fx.initial_ood, &fx.levels);
        let (vals, chals) = (&fx.vals[..], &fx.chals[..]);
        let target_v = fx.target_v;
        assert_eq!(vals[target_v], target);
        assert_eq!(trace.enforced.len(), levels.len() - 1);
        let native = replay_affine_claim(target, initial_ood, levels, vals, chals, &trace.enforced);
        assert_eq!(native, trace.final_claim, "native replay = verifier");

        // The circuit: every value and challenge enters as an input wire.
        let mut sb = ShapeBuilder::new(7);
        let fold = sb.slot(AffineFoldGate::new());
        let mac = sb.slot(MacGate::new());
        let mut ins: Vec<F128> = Vec::new();
        fn input(sb: &mut ShapeBuilder, ins: &mut Vec<F128>, v: F128) -> Wire {
            ins.push(v);
            sb.input()
        }
        let join = |sb: &mut ShapeBuilder, ins: &mut Vec<F128>, o: &AffineOodRec| -> JoinWires {
            (
                input(sb, ins, chals[o.beta_ch]),
                input(sb, ins, vals[o.y_v]),
            )
        };
        let target_w = input(&mut sb, &mut ins, target);
        let init_w: Vec<JoinWires> = initial_ood
            .iter()
            .map(|o| join(&mut sb, &mut ins, o))
            .collect();
        let mut lvl_w = Vec::new();
        for (li, lvl) in levels.iter().enumerate() {
            let mut rounds = Vec::new();
            for rd in &lvl.rounds {
                let g: [Wire; 5] =
                    std::array::from_fn(|e| input(&mut sb, &mut ins, vals[rd.grid_v + e]));
                let c: [Wire; 3] =
                    std::array::from_fn(|e| input(&mut sb, &mut ins, chals[rd.ch + e]));
                rounds.push((g, c));
            }
            let ood = lvl.ood.iter().map(|o| join(&mut sb, &mut ins, o)).collect();
            let consistency = lvl.beta.map(|(_, beta_ch)| {
                (
                    input(&mut sb, &mut ins, chals[beta_ch]),
                    input(&mut sb, &mut ins, trace.enforced[li]),
                )
            });
            lvl_w.push(AffineLevelWires {
                rounds,
                ood,
                consistency,
            });
        }
        emit_affine_claim_fold(&mut sb, fold, mac, target_w, &init_w, &lvl_w);
        let shape = sb.finish().expect("the claim-fold circuit builds");
        let built = shape.run(&ins, &[]);

        let fold_rows = built.rows::<AffineFoldGate>(fold);
        let mac_rows = built.rows::<MacGate>(mac);
        let rounds: usize = ks.iter().sum();
        let joins = initial_ood.len()
            + levels.iter().map(|l| l.ood.len()).sum::<usize>()
            + trace.enforced.len();
        assert_eq!(fold_rows.len(), rounds, "one row per fold round");
        assert_eq!(mac_rows.len(), joins, "one MAC per join");
        // The final level ends on fold rounds, so the last fold row's output
        // is the claim the residual check binds.
        assert_eq!(fold_rows.last().unwrap()[FOLD_OUT], trace.final_claim);

        let fold_ty = AffineFoldGate::new().ty;
        let mac_ty = MacGate::new().ty;
        let ok = |ty: &ElementTableType, row: &[F128]| {
            let mut z = row.to_vec();
            z.resize(ty.width(), F128::ZERO);
            ty.satisfies(&z, 0, 1)
        };
        assert!(fold_rows.iter().all(|r| ok(&fold_ty, r)));
        assert!(mac_rows.iter().all(|r| ok(&mac_ty, r)));
        for col in [9, 10, 12, FOLD_OUT] {
            let mut bad = fold_rows[0].clone();
            bad[col] += F128::ONE;
            assert!(!ok(&fold_ty, &bad), "a wrong column {col} fails the gate");
        }
    }
}
