//! The affine inner opening's QUERY-PHASE CONSISTENCY — the F128
//! replacement for the split walker's leaf-eval accumulators
//! (`gates_leaf::LeafEvalGate256`, `query::emit_query_phase`).
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
//! basis. Every weight is computed in the circuit from wires: the
//! `Ŵ` ladder by the shared [`ResidualWeightsGate256`](crate::tower::ResidualWeightsGate256) relation from the
//! position wire, the `eq(α)` tensor by [`MacGate`] rows from the squeezed
//! alpha wires, and the per-query sum by ONE [`AffineLeafGate`] row. No
//! weight enters as checker-tier advice.
//!
//! Not yet consumed outside tests: the affine child walker wires these to
//! the opening's leaf words and the FS chain's squeezes.
#![cfg_attr(not(test), allow(dead_code))]

use std::sync::Arc;

use flock_core::{
    circuit::builder::{GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
    element_r1cs::{ElementTableBuilder, ElementTableType},
    schedule::{IoWord, TableType},
};
use flock_field::F128;

/// A linear form over a gate's columns, all coefficients one.
type Form = Vec<usize>;

/// One row of an [`AffineLeafGate`]: `out = (Σ a)·(Σ b)`, or `out = Σ a`
/// when `b` is empty.
struct Row {
    out: usize,
    a: Form,
    b: Form,
}

/// The column layout of an [`AffineLeafGate`].
#[derive(Clone, Copy, Debug)]
pub(super) struct AffineLeafLayout {
    /// Live lanes of the opened row (the dead top lanes are zero).
    pub(super) lanes: usize,
    /// Fold rounds at the level: the row has `2^k` slots.
    pub(super) k: usize,
    /// Overflow words (`k` at a non-final level).
    pub(super) over: usize,
    /// Novel-basis bits the overflow needs: `ceil_log2(over)`.
    pub(super) bits: usize,
}

impl AffineLeafLayout {
    pub(super) fn new(lanes: usize, k: usize, over: usize) -> Self {
        assert!(lanes >= 1 && lanes <= 1 << k, "a row has at most 2^k lanes");
        assert!(over >= 1, "a non-final level absorbs its overflow");
        Self {
            lanes,
            k,
            over,
            bits: over.next_power_of_two().trailing_zeros() as usize,
        }
    }

    /// The input order: `row[lanes]`, `α[k]`, `β[k]`, `x`, `Ŵ_1..Ŵ_{bits}`,
    /// `Ŵ_c`, `τ[over]`, `eq(α)_q`, `acc`. (`Ŵ_0 = x`.)
    pub(super) fn n_in(&self) -> usize {
        self.lanes + 2 * self.k + 1 + self.bits.saturating_sub(1) + 1 + self.over + 2
    }
    fn alpha(&self, j: usize) -> usize {
        self.lanes + j
    }
    fn beta(&self, j: usize) -> usize {
        self.lanes + self.k + j
    }
    fn x(&self) -> usize {
        self.lanes + 2 * self.k
    }
    /// `Ŵ_b` for `b < bits`: `x` itself at `b = 0`.
    fn w(&self, b: usize) -> usize {
        self.x() + b
    }
    fn wc(&self) -> usize {
        self.x() + self.bits.max(1)
    }
    fn tau(&self, s: usize) -> usize {
        self.wc() + 1 + s
    }
    fn weight(&self) -> usize {
        self.tau(self.over)
    }
    fn acc(&self) -> usize {
        self.weight() + 1
    }
}

/// One query's consistency term at one level, folded into a running
/// accumulator: `acc' = acc + eq(α)_q · cv_q` (the module docs give `cv`).
///
/// Every product is one `mult_lin` row and every folded value stays a
/// LINEAR FORM until a product consumes it, so the row fold costs one column
/// per tree node (`lanes − 1` at a full row) and the forms grow by one term
/// per round:
///
/// ```text
///   d_j    = β_j · x                         (g_j = α_j + d_j, a form)
///   e      = g_j · hi        new = lo + e    (each fold node)
///   t      = Ŵ_b · hi        T  = lo + t     (the overflow's novel basis)
///   o      = Ŵ_c · T
///   m      = eq(α)_q · (folded + o)
///   acc'   = acc + m                         (the output column)
/// ```
pub(super) struct AffineLeafGate {
    pub(super) ty: Arc<ElementTableType>,
    pub(super) lay: AffineLeafLayout,
    rows: Vec<Row>,
    width: usize,
}

/// Fold `vals` (absent = zero) LSB-first, one round per entry of `g`:
/// `new[i] = lo + g·hi`. Appends the product rows; returns the folded form.
fn fold_forms(vals: Vec<Option<Form>>, g: &[Form], rows: &mut Vec<Row>, next: &mut usize) -> Form {
    assert_eq!(vals.len(), 1 << g.len());
    let mut cur = vals;
    for gj in g {
        cur = cur
            .chunks(2)
            .map(|p| match (&p[0], &p[1]) {
                (lo, None) => lo.clone(),
                (lo, Some(hi)) => {
                    let e = *next;
                    *next += 1;
                    rows.push(Row {
                        out: e,
                        a: gj.clone(),
                        b: hi.clone(),
                    });
                    let mut f = lo.clone().unwrap_or_default();
                    f.push(e);
                    Some(f)
                }
            })
            .collect();
    }
    cur.pop().flatten().unwrap_or_default()
}

impl AffineLeafGate {
    pub(super) fn new(lay: AffineLeafLayout) -> Self {
        let mut rows = Vec::new();
        let mut next = lay.n_in();
        fn col(rows: &mut Vec<Row>, next: &mut usize, a: Form, b: Form) -> usize {
            let out = *next;
            *next += 1;
            rows.push(Row { out, a, b });
            out
        }
        let g: Vec<Form> = (0..lay.k)
            .map(|j| {
                let d = col(&mut rows, &mut next, vec![lay.beta(j)], vec![lay.x()]);
                vec![lay.alpha(j), d]
            })
            .collect();
        let lanes: Vec<Option<Form>> = (0..1usize << lay.k)
            .map(|i| (i < lay.lanes).then(|| vec![i]))
            .collect();
        let folded = fold_forms(lanes, &g, &mut rows, &mut next);
        let wb: Vec<Form> = (0..lay.bits).map(|b| vec![lay.w(b)]).collect();
        let taus: Vec<Option<Form>> = (0..1usize << lay.bits)
            .map(|s| (s < lay.over).then(|| vec![lay.tau(s)]))
            .collect();
        let tail = fold_forms(taus, &wb, &mut rows, &mut next);
        let o = col(&mut rows, &mut next, vec![lay.wc()], tail);
        let mut cv = folded;
        cv.push(o);
        let m = col(&mut rows, &mut next, vec![lay.weight()], cv);
        let out = col(&mut rows, &mut next, vec![lay.acc(), m], vec![]);
        let width = out + 1;

        let one = |f: &Form| -> Vec<(usize, F128)> { f.iter().map(|&c| (c, F128::ONE)).collect() };
        let kappa = width.next_power_of_two().trailing_zeros().max(2) as usize;
        let mut b = ElementTableBuilder::new(kappa);
        for w in 0..lay.n_in() {
            b.free_wire(w);
        }
        for r in &rows {
            if r.b.is_empty() {
                b.linear(r.out, &one(&r.a));
            } else {
                b.mult_lin(r.out, &one(&r.a), &one(&r.b));
            }
        }
        Self {
            ty: Arc::new(b.build().expect("affine leaf gate")),
            lay,
            rows,
            width,
        }
    }

    /// The output column (`acc'`).
    pub(super) fn out(&self) -> usize {
        self.width - 1
    }
}

impl GateType for AffineLeafGate {
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
        let sum = |z: &[F128], f: &Form| f.iter().fold(F128::ZERO, |acc, &c| acc + z[c]);
        for r in &self.rows {
            let a = sum(&z, &r.a);
            z[r.out] = if r.b.is_empty() { a } else { a * sum(&z, &r.b) };
        }
        outputs.push(z[self.out()]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// `eq(α, ·)` over `2^|α|` entries, LSB-first (`build_eq_table`'s order),
/// from the squeezed alpha wires: two [`MacGate`] rows per entry, as the
/// split residual region builds it.
///
/// [`MacGate`]: crate::tower::MacGate
pub(super) fn emit_eq_tensor(
    sb: &mut ShapeBuilder,
    mac: SlotId,
    alpha: &[Wire],
    zw: Wire,
    ow: Wire,
) -> Vec<Wire> {
    let mut aw = vec![ow];
    for &r in alpha {
        let old = std::mem::take(&mut aw);
        aw.extend(old.iter().map(|&x| sb.gate(mac, &[x, x, r])[0]));
        aw.extend(old.iter().map(|&x| sb.gate(mac, &[zw, x, r])[0]));
    }
    aw
}

/// The normalized subspace ladder `Ŵ_0(x) .. Ŵ_{n-1}(x)` at a position wire:
/// `x` itself, then one [`ResidualWeightsGate256::with_weights`](crate::tower::ResidualWeightsGate256::with_weights) row.
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

/// Emit one level's batched consistency: per query a ladder row and an
/// [`AffineLeafGate`] row on `leaf`, plus the level's `eq(α)` tensor.
/// Returns the `enforced` wire (the value the claim fold joins) and each
/// query's ladder, which the residual region reuses.
pub(super) fn emit_affine_consistency(
    sb: &mut ShapeBuilder,
    leaf: SlotId,
    ladder: SlotId,
    mac: SlotId,
    lvl: &AffineQueryLevel,
    zw: Wire,
    ow: Wire,
) -> (Wire, Vec<Vec<Wire>>) {
    assert_eq!(lvl.rows.len(), lvl.xs.len());
    let eq = emit_eq_tensor(sb, mac, &lvl.alpha, zw, ow);
    assert!(
        eq.len() >= lvl.xs.len(),
        "the alpha tensor covers every query"
    );
    let bits = lvl.overflow.len().next_power_of_two().trailing_zeros() as usize;
    let mut acc = zw;
    let mut ladders = Vec::with_capacity(lvl.xs.len());
    for (q, (row, &x)) in lvl.rows.iter().zip(&lvl.xs).enumerate() {
        let w = emit_subspace_ladder(sb, ladder, x, ow);
        let mut ins: Vec<Wire> = row.clone();
        ins.extend(lvl.gammas.iter().map(|g| g.0));
        ins.extend(lvl.gammas.iter().map(|g| g.1));
        ins.push(x);
        ins.extend_from_slice(&w[1..bits.max(1)]);
        ins.push(w[lvl.log_cols]);
        ins.extend_from_slice(&lvl.overflow);
        ins.push(eq[q]);
        ins.push(acc);
        acc = sb.gate(leaf, &ins)[0];
        ladders.push(w);
    }
    (acc, ladders)
}

#[cfg(test)]
mod tests {
    use flock_core::{
        challenger::FsChallenger,
        lincheck::build_eq_table,
        ntt::additive_ntt_f128::AdditiveNttF128,
        pcs::ligerito::{
            LigeritoProfile,
            affine::{affine_ladder_for, prove_affine, verify_affine_l0_traced},
            ligero_commit,
        },
    };
    use flock_transcript::transcript_record::RecordingChallenger;

    use super::*;
    use crate::tower::{
        MacGate, ResidualWeightsGate256, fs_chain::flatten_ops, tape::parse_affine_direct,
    };

    fn rand_vec(len: usize, seed: u64) -> Vec<F128> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let lo = s;
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                F128::new(lo, s)
            })
            .collect()
    }

    fn satisfied(ty: &ElementTableType, row: &[F128]) -> bool {
        let mut z = row.to_vec();
        z.resize(ty.width(), F128::ZERO);
        ty.satisfies(&z, 0, 1)
    }

    /// A direct affine opening at the registered m = 24 shape, verified
    /// through the recording challenger. Every non-final level's emitted
    /// consistency (ladder rows, eq tensor, one leaf row per query) outputs
    /// exactly the verifier's own `enforced` value, every row satisfies its
    /// constraints, and a tampered intermediate column fails the leaf gate.
    #[test]
    fn consistency_matches_the_verifier() {
        const D: &[u8] = b"tower-affine-query";
        let (log_n, initial_k) = (17, 6);
        let (pc, vc) = affine_ladder_for(LigeritoProfile::FastAffine, log_n, initial_k)
            .expect("m24 fast-affine");
        let witness = rand_vec(1 << log_n, 0xA11E);
        let z = rand_vec(log_n, 0x5EED);
        let basis = build_eq_table(&z);
        let target = witness
            .iter()
            .zip(&basis)
            .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);
        let log_cols = log_n - initial_k;
        let rate = pc.log_inv_rates[0];
        let ntt = AdditiveNttF128::standard(log_cols + rate);
        let l0 = ligero_commit(&witness, log_cols, initial_k, rate, &ntt, pc.merkle_hash);
        let mut ch = FsChallenger::with_chained_blake3(D);
        let proof = prove_affine(&pc, witness, basis, target, &l0.mat, &l0.tree, &mut ch);
        let cap = l0.cap(pc.level_zero_cap_depth()).to_vec();
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(D));
        let trace = verify_affine_l0_traced(
            &vc,
            &proof,
            log_n,
            target,
            &cap,
            (1 << initial_k, None),
            |rs, residual_log| {
                let prefix = z
                    .iter()
                    .zip(rs)
                    .fold(F128::ONE, |acc, (&zi, &ri)| acc * (F128::ONE + zi + ri));
                let mut out = build_eq_table(&z[rs.len()..]);
                assert_eq!(out.len(), 1usize << residual_log);
                out.iter_mut().for_each(|v| *v *= prefix);
                out
            },
            &mut rec,
        )
        .expect("the recorded affine verify accepts");

        let mut ks = vec![pc.initial_k];
        ks.extend(&pc.recursive_ks);
        let mut cs = vec![log_cols];
        cs.extend(&pc.recursive_log_msg_cols);
        let yr_len = (1usize << cs[cs.len() - 1]) + ks[ks.len() - 1];
        let ops = flatten_ops(rec.shape().ops());
        let (_, _, levels) = parse_affine_direct(&ops, &ks, yr_len);
        let (vals, chals) = (rec.values(), rec.challenges());
        assert_eq!(trace.queries.len(), levels.len());

        let max_c = *cs.iter().max().unwrap();
        let mut sb = ShapeBuilder::new(11);
        let mac = sb.slot(MacGate::new());
        let ladder = sb.slot(ResidualWeightsGate256::with_weights(max_c + 1));
        let mut ins: Vec<F128> = Vec::new();
        fn input(sb: &mut ShapeBuilder, ins: &mut Vec<F128>, v: F128) -> Wire {
            ins.push(v);
            sb.input()
        }
        let zw = input(&mut sb, &mut ins, F128::ZERO);
        let ow = input(&mut sb, &mut ins, F128::ONE);
        let mut leaf_slots: Vec<(usize, usize, SlotId)> = Vec::new();
        let mut outs = Vec::new();
        for (li, lvl) in levels.iter().enumerate().take(levels.len() - 1) {
            let (k, c) = (ks[li], cs[li]);
            let rows_v = &proof.openings[li].opened_rows;
            let lanes = rows_v[0].len();
            let slot = match leaf_slots.iter().find(|s| s.0 == lanes && s.1 == k) {
                Some(s) => s.2,
                None => {
                    let s = sb.slot(AffineLeafGate::new(AffineLeafLayout::new(lanes, k, k)));
                    leaf_slots.push((lanes, k, s));
                    s
                }
            };
            let rows: Vec<Vec<Wire>> = rows_v
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
            let ov = lvl.overflow_v.expect("non-final overflow");
            let overflow: Vec<Wire> = (0..k)
                .map(|s| input(&mut sb, &mut ins, vals[ov + s]))
                .collect();
            let (_, a_ch, a_count) = lvl.alpha.expect("non-final alpha");
            let alpha: Vec<Wire> = (0..a_count)
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
            outs.push((
                slot,
                emit_affine_consistency(&mut sb, slot, ladder, mac, &q, zw, ow).0,
            ));
        }
        let shape = sb.finish().expect("the consistency circuit builds");
        let built = shape.run(&ins, &[]);

        // Each level's last leaf row carries its enforced value; levels that
        // share a slot are laid out in emission order.
        let mut used: Vec<(SlotId, usize)> = Vec::new();
        for (li, &(slot, _)) in outs.iter().enumerate() {
            let (lanes, k) = (proof.openings[li].opened_rows[0].len(), ks[li]);
            let gate = AffineLeafGate::new(AffineLeafLayout::new(lanes, k, k));
            let rows = built.rows::<AffineLeafGate>(slot);
            let before: usize = used.iter().filter(|u| u.0 == slot).map(|u| u.1).sum();
            let q = trace.queries[li].len();
            let gate_rows = &rows[before..before + q];
            assert!(
                gate_rows.iter().all(|r| satisfied(&gate.ty, r)),
                "L{li} rows"
            );
            let out = gate.out();
            assert_eq!(
                gate_rows[q - 1][out],
                trace.enforced[li],
                "L{li} enforced = verifier"
            );
            for col in [gate.lay.n_in(), out - 1, out] {
                let mut bad = gate_rows[0].clone();
                bad[col] += F128::ONE;
                assert!(
                    !satisfied(&gate.ty, &bad),
                    "a wrong column {col} fails L{li}"
                );
            }
            used.push((slot, q));
        }
        let mac_ty = MacGate::new().ty;
        assert!(
            built
                .rows::<MacGate>(mac)
                .iter()
                .all(|r| satisfied(&mac_ty, r))
        );
        let ladder_ty = ResidualWeightsGate256::with_weights(max_c + 1).ty;
        let ladder_rows = built.rows::<ResidualWeightsGate256>(ladder);
        assert!(ladder_rows.iter().all(|r| satisfied(&ladder_ty, r)));
        let total_q: usize = trace.queries[..levels.len() - 1].iter().map(Vec::len).sum();
        assert_eq!(ladder_rows.len(), total_q, "one ladder row per query");
    }

    /// The leaf gate's size at the tower's affine shapes (`--nocapture`).
    #[test]
    fn leaf_gate_sizes_at_the_tower_shapes() {
        for (name, profile, log_n, initial_k, l0_lanes) in [
            ("slim29", LigeritoProfile::SlimAffine, 22, 5, 31),
            ("fast32", LigeritoProfile::FastAffine, 25, 6, 46),
        ] {
            let (pc, _) = affine_ladder_for(profile, log_n, initial_k).expect("registered");
            let mut ks = vec![pc.initial_k];
            ks.extend(&pc.recursive_ks);
            for li in 0..ks.len() - 1 {
                let lanes = if li == 0 { l0_lanes } else { 1 << ks[li] };
                let g = AffineLeafGate::new(AffineLeafLayout::new(lanes, ks[li], ks[li]));
                println!(
                    "{name} L{li}: lanes {lanes} k {} q {}: {} inputs, {} cols, kappa {}",
                    ks[li],
                    pc.queries[li],
                    g.lay.n_in(),
                    g.out() + 1,
                    g.ty.kappa(),
                );
                assert!(g.ty.kappa() <= if li == 0 { 8 } else { 7 });
            }
        }
    }
}
