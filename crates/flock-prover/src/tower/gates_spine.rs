use std::sync::Arc;

use flock_core::{
    circuit::builder::SlotId,
    element_r1cs::{ElementTableBuilder, ElementTableType},
    pcs::{jagged::assist_sparse_transitions, ligerito::eval_sk_at_vks},
    schedule::IoWord,
};
use flock_field::QUADRATIC_NONRESIDUE;

use crate::{
    prover::UnionElementSlotInput,
    tower::{F128, F256, GateType, ShapeBuilder, SlotWitness, TableType, Wire, build_mac256},
};

/// The sumcheck-spine gate: one fold-and-eval step of the verifier's running
/// quadratic, `RoundQuad` in circuit form (char-2, so `u1 = t + u0` is the
/// linear coefficient trick):
///
///   c' = c + beta*u0     b' = b + beta*(y + u2)     a' = a + beta*u2
///   tr' = tr + beta*y    t' = c' + r*b' + r^2*a'
///
/// Three degenerate uses cover every verifier step with ONE table type:
/// BUILD `from_msg` (zero quad in, beta = 1, y = the running target),
/// EVAL a held quad (beta = 0; only t' consumed), and INTRO-FOLD an OOD or
/// enforced-sum claim (consume c', b', a', tr'; t' unwired).
pub(super) struct SpineGate {
    pub(super) ty: Arc<ElementTableType>,
}

pub(super) const SP_IN: usize = 9; // c b a tr u0 u2 y beta r
pub(super) const SP_K: usize = 21;

impl SpineGate {
    pub(super) fn new() -> Self {
        let one = F128::ONE;
        let (c, b, a, tr, u0, u2, y, beta, r) = (0, 1, 2, 3, 4, 5, 6, 7, 8);
        let (pc, pb, pa, pt, co, bo, ao, tro, r2, m1, m2, to) =
            (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
        let mut bld = ElementTableBuilder::new(5);
        for w in 0..SP_IN {
            bld.free_wire(w);
        }
        bld.mult(pc, beta, u0)
            .mult_lin(pb, &[(y, one), (u2, one)], &[(beta, one)])
            .mult(pa, beta, u2)
            .mult(pt, beta, y)
            .linear(co, &[(c, one), (pc, one)])
            .linear(bo, &[(b, one), (pb, one)])
            .linear(ao, &[(a, one), (pa, one)])
            .linear(tro, &[(tr, one), (pt, one)])
            .mult(r2, r, r)
            .mult(m1, r, bo)
            .mult(m2, r2, ao)
            .linear(to, &[(co, one), (m1, one), (m2, one)]);
        Self {
            ty: Arc::new(bld.build().expect("spine gate is valid")),
        }
    }
}

impl GateType for SpineGate {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..SP_IN).map(IoWord::input).collect();
        for o in [13, 14, 15, 16, 20] {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; SP_K];
        z[..SP_IN].copy_from_slice(&inputs[..SP_IN]);
        let (c, b, a, tr, u0, u2, y, beta, r) =
            (z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8]);
        z[9] = beta * u0;
        z[10] = (y + u2) * beta;
        z[11] = beta * u2;
        z[12] = beta * y;
        z[13] = c + z[9];
        z[14] = b + z[10];
        z[15] = a + z[11];
        z[16] = tr + z[12];
        z[17] = r * r;
        z[18] = r * z[14];
        z[19] = z[17] * z[15];
        z[20] = z[13] + z[18] + z[19];
        outputs.extend_from_slice(&[z[13], z[14], z[15], z[16], z[20]]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// The Ligerito sumcheck spine over F256. The batching coefficient `beta`
/// remains a base-field scalar, while the quadratic, messages, fold
/// challenge, running target, and outputs are pairs of base-field wires.
pub(super) struct SpineGate256 {
    pub(super) ty: Arc<ElementTableType>,
}

pub(super) const SP256_IN: usize = 17;
pub(super) const SP256_K: usize = 50;

impl SpineGate256 {
    pub(super) fn new() -> Self {
        let one = F128::ONE;
        let pair = |at| [at, at + 1];
        let (c, qb, qa, tr, u0, u2, y, beta, r) = (
            pair(0),
            pair(2),
            pair(4),
            pair(6),
            pair(8),
            pair(10),
            pair(12),
            14,
            pair(15),
        );
        let (pc, pb, pa, pt) = (pair(17), pair(19), pair(21), pair(23));
        let (co, bo, ao, tro) = (pair(25), pair(27), pair(29), pair(31));
        let mut b = ElementTableBuilder::new(6);
        for w in 0..SP256_IN {
            b.free_wire(w);
        }
        b.mult(pc[0], beta, u0[0]);
        b.mult(pc[1], beta, u0[1]);
        b.mult_lin(pb[0], &[(y[0], one), (u2[0], one)], &[(beta, one)]);
        b.mult_lin(pb[1], &[(y[1], one), (u2[1], one)], &[(beta, one)]);
        b.mult(pa[0], beta, u2[0]);
        b.mult(pa[1], beta, u2[1]);
        b.mult(pt[0], beta, y[0]);
        b.mult(pt[1], beta, y[1]);
        for (out, lhs, rhs) in [(co, c, pc), (bo, qb, pb), (ao, qa, pa), (tro, tr, pt)] {
            b.linear(out[0], &[(lhs[0], one), (rhs[0], one)]);
            b.linear(out[1], &[(lhs[1], one), (rhs[1], one)]);
        }
        let r2_at = 33;
        build_mac256(&mut b, r2_at, None, r, r);
        let r2 = pair(r2_at + 3);
        let rb_at = 38;
        build_mac256(&mut b, rb_at, None, r, bo);
        let rb = pair(rb_at + 3);
        let ra_at = 43;
        build_mac256(&mut b, ra_at, None, r2, ao);
        let ra = pair(ra_at + 3);
        b.linear(48, &[(co[0], one), (rb[0], one), (ra[0], one)]);
        b.linear(49, &[(co[1], one), (rb[1], one), (ra[1], one)]);
        Self {
            ty: Arc::new(b.build().expect("extension spine gate is valid")),
        }
    }
}

impl GateType for SpineGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..SP256_IN).map(IoWord::input).collect();
        for o in [25, 26, 27, 28, 29, 30, 31, 32, 48, 49] {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let get = |z: &[F128], at| F256::new(z[at], z[at + 1]);
        let put = |z: &mut [F128], at, v: F256| {
            z[at] = v.c0;
            z[at + 1] = v.c1;
        };
        let mut z = vec![F128::ZERO; SP256_K];
        z[..SP256_IN].copy_from_slice(&inputs[..SP256_IN]);
        let (c, b, a, tr, u0, u2, y, beta, r) = (
            get(&z, 0),
            get(&z, 2),
            get(&z, 4),
            get(&z, 6),
            get(&z, 8),
            get(&z, 10),
            get(&z, 12),
            z[14],
            get(&z, 15),
        );
        let (pc, pb, pa, pt) = (u0 * beta, (y + u2) * beta, u2 * beta, y * beta);
        put(&mut z, 17, pc);
        put(&mut z, 19, pb);
        put(&mut z, 21, pa);
        put(&mut z, 23, pt);
        let (co, bo, ao, tro) = (c + pc, b + pb, a + pa, tr + pt);
        put(&mut z, 25, co);
        put(&mut z, 27, bo);
        put(&mut z, 29, ao);
        put(&mut z, 31, tro);
        let r2 = r * r;
        let rb = r * bo;
        let ra = r2 * ao;
        for (at, x, lhs, rhs) in [(33, r2, r, r), (38, rb, r, bo), (43, ra, r2, ao)] {
            let p0 = lhs.c0 * rhs.c0;
            let p1 = lhs.c1 * rhs.c1;
            let p2 = (lhs.c0 + lhs.c1) * (rhs.c0 + rhs.c1);
            z[at] = p0;
            z[at + 1] = p1;
            z[at + 2] = p2;
            put(&mut z, at + 3, x);
        }
        let to = co + rb + ra;
        put(&mut z, 48, to);
        outputs.extend_from_slice(&[
            co.c0, co.c1, bo.c0, bo.c1, ao.c0, ao.c1, tro.c0, tro.c1, to.c0, to.c1,
        ]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

pub(super) fn emit_spine256(
    sb: &mut ShapeBuilder,
    slot: SlotId,
    c: [Wire; 2],
    b: [Wire; 2],
    a: [Wire; 2],
    tr: [Wire; 2],
    u0: [Wire; 2],
    u2: [Wire; 2],
    y: [Wire; 2],
    beta: Wire,
    r: [Wire; 2],
) -> [[Wire; 2]; 5] {
    let inputs: Vec<Wire> = [c, b, a, tr, u0, u2, y]
        .into_iter()
        .flatten()
        .chain([beta])
        .chain(r)
        .collect();
    let out = sb.gate(slot, &inputs);
    [
        [out[0], out[1]],
        [out[2], out[3]],
        [out[4], out[5]],
        [out[6], out[7]],
        [out[8], out[9]],
    ]
}

/// The residual-basis gate (step 2b): one query's contribution to a level's
/// `induce_sumcheck_evaluate_at_residual`, at every residual position `y`.
///
/// From `q_field` the novel-basis chain runs `s_{k+1} = s_k (s_k + c_k)`
/// (`c_k = s_k(v_k)`, a constant; the `1/s_k(v_k)` normalizations fold into
/// downstream weights). The level's post-intro fold challenges `ris` build
/// `prefix = prod_k (1 + ris_k (1 + W_k))`, the suffix `W`s form subset
/// products over the `2^yr` residual positions (`1 + p_j(1+w) = w` iff the
/// bit is set), and `aw * prefix * subset(y)` accumulates into `2^yr` running
/// sums. One gate row per (level, query); the accumulators chain across
/// queries like `LeafEvalGate`'s.
///
/// `q_field` is a public input, bound at the boundary: the checker masks the
/// (already published) challenge word natively — same pattern as the cap
/// select.
/// Smallest kappa whose column budget holds `c_need` (floored at MacGate's
/// 3). Tight envelopes matter: the element region's size is the sum of
/// 2^kappa envelopes rounded to a power of two, and the union's column
/// domain (claim-point lengths, the eq-dot loops, run counts) follows it.
pub(super) fn gate_kappa(c_need: usize) -> usize {
    assert!(c_need <= 256, "gate spills kappa=8 ({c_need} cols)");
    (c_need.next_power_of_two().trailing_zeros() as usize).max(3)
}

/// Residual-basis accumulation for extension-valued fold challenges. The
/// novel-basis chain and query weights are base-field values; only the
/// products involving later fold challenges and the running accumulators
/// need two limbs.
pub(super) struct ResidualWeightsGate256 {
    pub(super) ty: Arc<ElementTableType>,
    pub(super) coeffs: Vec<F128>,
}

impl ResidualWeightsGate256 {
    /// Sized to the deepest walked ladder, like `spread_w`: the
    /// m32 FAST chain leaf's L0 needs `pl 15 + yr_log 4 = 19` (the residual
    /// domain is 16 entries = two 8-chunks; the chunk-high extension reads
    /// `weights[pl + yr_log - 1]`). The m29 outer ladders stay below this;
    /// anything deeper fails the `lmc` assert loudly at build.
    pub(super) const N_WEIGHTS: usize = 19;

    pub(super) fn new() -> Self {
        let o = F128::ONE;
        // The subspace-polynomial chain constants `s_k(v_k)`, straight from
        // ligerito (the residual boundary check pins the spine against them).
        let sks = eval_sk_at_vks(Self::N_WEIGHTS);
        let coeffs: Vec<F128> = (0..Self::N_WEIGHTS - 1)
            .map(|k| {
                assert_ne!(sks[k + 1], F128::ZERO, "novel-basis normalizer is nonzero");
                sks[k] * sks[k] * sks[k + 1].inv()
            })
            .collect();
        // in: W_0=q, one. out: W_1..W_18.
        let mut b = ElementTableBuilder::new(5);
        b.free_wire(0).free_wire(1);
        let mut prev = 0;
        for (j, &d) in coeffs.iter().enumerate() {
            let out = 2 + j;
            b.mult_lin(out, &[(prev, d)], &[(prev, o), (1, o)]);
            prev = out;
        }
        Self {
            ty: Arc::new(b.build().expect("normalized residual weights gate")),
            coeffs,
        }
    }
}

impl GateType for ResidualWeightsGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema = vec![IoWord::input(0), IoWord::input(1)];
        schema.extend((2..2 + self.coeffs.len()).map(IoWord::output));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 2 + self.coeffs.len()];
        z[..2].copy_from_slice(&inputs[..2]);
        let mut prev = 0;
        for (j, &d) in self.coeffs.iter().enumerate() {
            let out = 2 + j;
            z[out] = d * z[prev] * (z[prev] + z[1]);
            prev = out;
        }
        outputs.extend_from_slice(&z[2..]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// Three consecutive F256 residual-prefix factors. Ligerito introduces
/// three post-introduction fold challenges per level, so every active
/// residual prefix is a chain of this one relation:
///
/// `P' = P product_i (1 + R_i (1 + W_i))`, for `i=0,1,2`.
pub(super) struct ResidualPrefix3Gate256 {
    pub(super) ty: Arc<ElementTableType>,
    out: [usize; 2],
}

impl ResidualPrefix3Gate256 {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        let nr = QUADRATIC_NONRESIDUE;
        // in: prefix pair, three challenge pairs, three base weights, one.
        let (n_in, one) = (12usize, 11usize);
        let mut b = ElementTableBuilder::new(6);
        for col in 0..n_in {
            b.free_wire(col);
        }
        let mut c = n_in;
        let mut pr = [0, 1];
        for i in 0..3 {
            let r = [2 + 2 * i, 2 + 2 * i + 1];
            let w = 8 + i;
            b.mult_lin(c, &[(r[0], o)], &[(one, o), (w, o)]);
            b.mult_lin(c + 1, &[(r[1], o)], &[(one, o), (w, o)]);
            let pk = [c, c + 1];
            c += 2;
            b.mult_lin(c, &[(pr[0], o)], &[(one, o), (pk[0], o)]);
            b.mult(c + 1, pr[1], pk[1]);
            b.mult_lin(
                c + 2,
                &[(pr[0], o), (pr[1], o)],
                &[(one, o), (pk[0], o), (pk[1], o)],
            );
            b.linear(c + 3, &[(c, o), (c + 1, nr)]);
            b.linear(c + 4, &[(c + 2, o), (c, o)]);
            pr = [c + 3, c + 4];
            c += 5;
        }
        assert_eq!(c, 33, "three residual-prefix factors use 33 columns");
        Self {
            ty: Arc::new(b.build().expect("three-factor residual prefix gate")),
            out: pr,
        }
    }
}

impl GateType for ResidualPrefix3Gate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..12).map(IoWord::input).collect();
        schema.push(IoWord::output(self.out[0]));
        schema.push(IoWord::output(self.out[1]));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 33];
        z[..12].copy_from_slice(&inputs[..12]);
        let mut c = 12;
        let mut pr = F256::new(z[0], z[1]);
        for i in 0..3 {
            let r = F256::new(z[2 + 2 * i], z[2 + 2 * i + 1]);
            let pk = r * (z[11] + z[8 + i]);
            z[c] = pk.c0;
            z[c + 1] = pk.c1;
            c += 2;
            let factor = F256::new(z[11] + pk.c0, pk.c1);
            let product = pr * factor;
            z[c] = pr.c0 * factor.c0;
            z[c + 1] = pr.c1 * factor.c1;
            z[c + 2] = (pr.c0 + pr.c1) * (factor.c0 + factor.c1);
            z[c + 3] = product.c0;
            z[c + 4] = product.c1;
            pr = product;
            c += 5;
        }
        outputs.extend_from_slice(&[pr.c0, pr.c1]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// Add one residual query to all eight low-coordinate accumulators:
/// `acc_y' = acc_y + aw * prefix * product_{j:y_j=1} W_j`.
pub(super) struct ResidualAccGate256 {
    pub(super) ty: Arc<ElementTableType>,
    acc_out: [[usize; 2]; 8],
}

impl ResidualAccGate256 {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        // in: aw, prefix pair, three low weights, eight accumulator pairs.
        let (n_in, acc0) = (22usize, 6usize);
        let mut b = ElementTableBuilder::new(6);
        for col in 0..n_in {
            b.free_wire(col);
        }
        let mut c = n_in;
        b.mult(c, 0, 1).mult(c + 1, 0, 2);
        let t = [c, c + 1];
        c += 2;
        b.mult(c, 3, 4).mult(c + 1, 3, 5).mult(c + 2, 4, 5);
        b.mult(c + 3, c, 5);
        let weights = [
            None,
            Some(3),
            Some(4),
            Some(c),
            Some(5),
            Some(c + 1),
            Some(c + 2),
            Some(c + 3),
        ];
        c += 4;
        let mut contributions = [t; 8];
        for y in 1..8 {
            let w = weights[y].expect("a nonzero subset has a weight");
            b.mult(c, t[0], w).mult(c + 1, t[1], w);
            contributions[y] = [c, c + 1];
            c += 2;
        }
        let mut acc_out = [[0usize; 2]; 8];
        for y in 0..8 {
            b.linear(c, &[(acc0 + 2 * y, o), (contributions[y][0], o)]);
            b.linear(c + 1, &[(acc0 + 2 * y + 1, o), (contributions[y][1], o)]);
            acc_out[y] = [c, c + 1];
            c += 2;
        }
        assert_eq!(c, 58, "the residual accumulator uses 58 columns");
        Self {
            ty: Arc::new(b.build().expect("residual accumulator gate")),
            acc_out,
        }
    }
}

impl GateType for ResidualAccGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..22).map(IoWord::input).collect();
        for out in self.acc_out {
            schema.push(IoWord::output(out[0]));
            schema.push(IoWord::output(out[1]));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 58];
        z[..22].copy_from_slice(&inputs[..22]);
        let prefix = F256::new(z[1], z[2]);
        let t = prefix * z[0];
        z[22] = t.c0;
        z[23] = t.c1;
        let low = [z[3], z[4], z[5]];
        z[24] = low[0] * low[1];
        z[25] = low[0] * low[2];
        z[26] = low[1] * low[2];
        z[27] = z[24] * low[2];
        let weights = [
            F128::ONE,
            low[0],
            low[1],
            z[24],
            low[2],
            z[25],
            z[26],
            z[27],
        ];
        let mut c = 28;
        let mut contributions = [t; 8];
        for y in 1..8 {
            contributions[y] = t * weights[y];
            z[c] = contributions[y].c0;
            z[c + 1] = contributions[y].c1;
            c += 2;
        }
        for y in 0..8 {
            z[c] = z[6 + 2 * y] + contributions[y].c0;
            z[c + 1] = z[6 + 2 * y + 1] + contributions[y].c1;
            outputs.push(z[c]);
            outputs.push(z[c + 1]);
            c += 2;
        }
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// [`live_element_input`] without the packed intermediate: the slot's rows
/// (from a DEFERRED run, [`CircuitWitness::take_rows_of`]) scatter straight
/// into the union block — the same `dst[(col << nu) + j] = row[col]` write
/// every element gate's `witness()` makes, minus the full-capacity buffer it
/// makes it into. `dst` arrives zeroed and a row shorter than the slot's
/// width leaves implicit zero columns, exactly as the packed path did.
pub(super) fn live_element_input_from_rows(
    rows: Vec<Vec<F128>>,
    nu: usize,
) -> UnionElementSlotInput<'static> {
    UnionElementSlotInput::new(move |dst: &mut [F128]| {
        debug_assert!(rows.len() <= 1usize << nu);
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                dst[(col << nu) + j] = v;
            }
        }
    })
}

/// 2b stage 2: PrefixGate computes `seed * prod_j (1 + a_j + b_j)` — the
/// char-2 eq prefix of a packed-direct claim (seed = gamma, a = point,
/// b = fold challenges) or an OOD claim (seed = beta, a = z), and the eq
/// FACTORS of the close-out's per-position tensor (bit set → factor
/// `coord`, clear → `1 + coord`, pad → 1). The former SuffixGate/
/// PartialCombineGate/FinalDotGate close-out types are DISSOLVED (Round
/// 3): their tensor/combine/dot work rides prefix rows + the shared
/// MacGate — 51 schema words (each a cell slot AND a gather claim) for
/// ~30 rows of work became ~250 cheap rows and zero types.
pub(super) struct PrefixGate {
    pub(super) ty: Arc<ElementTableType>,
    pub(super) pl: usize,
    pub(super) n_in: usize,
    pub(super) k: usize,
}

impl PrefixGate {
    pub(super) fn new(pl: usize) -> Self {
        let o = F128::ONE;
        let n_in = 2 + 2 * pl; // seed, a[pl], b[pl], one
        let one = n_in - 1;
        // FUSED: each factor is ONE mult_lin cell, pr' = pr·(1 + a + b) —
        // the B side is a linear combination (the envelope program).
        let c_need = n_in + pl;
        let kappa = gate_kappa(c_need);
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(kappa);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut pr = 0;
        for j in 0..pl {
            bl.mult_lin(c, &[(pr, o)], &[(one, o), (1 + j, o), (1 + pl + j, o)]);
            pr = c;
            c += 1;
        }
        assert_eq!(c, c_need, "the prefix column count is the counted one");
        Self {
            ty: Arc::new(bl.build().expect("prefix gate")),
            pl,
            n_in,
            k: c,
        }
    }
}

impl GateType for PrefixGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.k - 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut pr = z[0];
        for j in 0..self.pl {
            z[c] = pr * (F128::ONE + z[1 + j] + z[1 + self.pl + j]);
            pr = z[c];
            c += 1;
        }
        outputs.extend_from_slice(&[pr]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// Extension-field prefix product `seed * product_j (1 + a_j + b_j)`.
/// Every value occupies two base-field wires; `one` and `zero` inputs make
/// the constant extension element `(1, 0)` explicit and preserve the
/// all-zero padding-row convention.
pub(super) struct PrefixGate256 {
    pub(super) ty: Arc<ElementTableType>,
    pub(super) pl: usize,
    pub(super) n_in: usize,
    out: [usize; 2],
    pub(super) k: usize,
}

impl PrefixGate256 {
    pub(super) fn new(pl: usize) -> Self {
        let o = F128::ONE;
        let n_in = 4 + 4 * pl; // seed pair, a pairs, b pairs, one, zero
        let one = n_in - 2;
        let zero = n_in - 1;
        let mut b = ElementTableBuilder::new(gate_kappa(n_in + 5 * pl));
        for w in 0..n_in {
            b.free_wire(w);
        }
        let mut c = n_in;
        let mut pr = [0, 1];
        for j in 0..pl {
            let a = [2 + 2 * j, 2 + 2 * j + 1];
            let bs = 2 + 2 * pl + 2 * j;
            let factor0 = vec![(one, o), (a[0], o), (bs, o)];
            let factor1 = vec![(zero, o), (a[1], o), (bs + 1, o)];
            let nr = QUADRATIC_NONRESIDUE;
            b.mult_lin(c, &[(pr[0], o)], &factor0);
            b.mult_lin(c + 1, &[(pr[1], o)], &factor1);
            b.mult_lin(
                c + 2,
                &[(pr[0], o), (pr[1], o)],
                &[
                    (one, o),
                    (zero, o),
                    (a[0], o),
                    (a[1], o),
                    (bs, o),
                    (bs + 1, o),
                ],
            );
            b.linear(c + 3, &[(c, o), (c + 1, nr)]);
            b.linear(c + 4, &[(c + 2, o), (c, o)]);
            pr = [c + 3, c + 4];
            c += 5;
        }
        Self {
            ty: Arc::new(b.build().expect("extension prefix gate")),
            pl,
            n_in,
            out: pr,
            k: c,
        }
    }
}

impl GateType for PrefixGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.out[0]));
        schema.push(IoWord::output(self.out[1]));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let one = self.n_in - 2;
        let zero = self.n_in - 1;
        let mut c = self.n_in;
        let mut pr = F256::new(z[0], z[1]);
        for j in 0..self.pl {
            let a = F256::new(z[2 + 2 * j], z[2 + 2 * j + 1]);
            let bs = 2 + 2 * self.pl + 2 * j;
            let factor = F256::new(z[one], z[zero]) + a + F256::new(z[bs], z[bs + 1]);
            let product = pr * factor;
            z[c] = pr.c0 * factor.c0;
            z[c + 1] = pr.c1 * factor.c1;
            z[c + 2] = (pr.c0 + pr.c1) * (factor.c0 + factor.c1);
            z[c + 3] = product.c0;
            z[c + 4] = product.c1;
            pr = product;
            c += 5;
        }
        outputs.extend_from_slice(&[pr.c0, pr.c1]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// One merged W-round of the verifier (`jagged::fold_round_claim`):
/// `t' = (t + g1) + (t + gi) r + gi r^2` — messages `(G(1), G(inf))` wire
/// from the absorbed stream, `r` from the chain squeeze; the chain of these
/// binds rho and carries the outer gamma-combination down to `running`.
pub(super) struct MergedRoundGate {
    pub(super) ty: Arc<ElementTableType>,
}

impl MergedRoundGate {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        // in: t(0), g1(1), gi(2), r(3)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..4 {
            b.free_wire(w);
        }
        b.mult_lin(4, &[(0, o), (2, o)], &[(3, o)]); // (t+gi) r
        b.mult(5, 3, 3); // r^2
        b.mult(6, 5, 2); // gi r^2
        b.linear(7, &[(0, o), (1, o), (4, o), (6, o)]);
        Self {
            ty: Arc::new(b.build().expect("merged round gate")),
        }
    }
}

impl GateType for MergedRoundGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..4).map(IoWord::input).collect();
        schema.push(IoWord::output(7));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 8];
        z[..4].copy_from_slice(&inputs[..4]);
        z[4] = (z[0] + z[2]) * z[3];
        z[5] = z[3] * z[3];
        z[6] = z[5] * z[2];
        z[7] = z[0] + z[1] + z[4] + z[6];
        outputs.extend_from_slice(&[z[7]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// Multiply-accumulate: `out = acc + x·y` — the workhorse of the multipoint
/// intake (gamma-power chains, the `T0`/`V` sums, and zero-delta joins:
/// `mac(a, b, one) = a + b` is the char-2 equality delta).
pub(super) struct MacGate {
    pub(super) ty: Arc<ElementTableType>,
}

impl MacGate {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        // in: acc(0), x(1), y(2)
        let mut b = ElementTableBuilder::new(3);
        for w in 0..3 {
            b.free_wire(w);
        }
        b.mult(3, 1, 2);
        b.linear(4, &[(0, o), (3, o)]);
        Self {
            ty: Arc::new(b.build().expect("mac gate")),
        }
    }
}

impl GateType for MacGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..3).map(IoWord::input).collect();
        schema.push(IoWord::output(4));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 5];
        z[..3].copy_from_slice(&inputs[..3]);
        z[3] = z[1] * z[2];
        z[4] = z[0] + z[3];
        outputs.extend_from_slice(&[z[4]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// Extension-field multiply-accumulate `out = acc + x*y`.
pub(super) struct MacGate256 {
    pub(super) ty: Arc<ElementTableType>,
}

impl MacGate256 {
    pub(super) fn new() -> Self {
        let mut b = ElementTableBuilder::new(4);
        for w in 0..6 {
            b.free_wire(w);
        }
        build_mac256(&mut b, 6, Some([0, 1]), [2, 3], [4, 5]);
        Self {
            ty: Arc::new(b.build().expect("extension mac gate")),
        }
    }
}

impl GateType for MacGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..6).map(IoWord::input).collect();
        schema.push(IoWord::output(9));
        schema.push(IoWord::output(10));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let acc = F256::new(inputs[0], inputs[1]);
        let x = F256::new(inputs[2], inputs[3]);
        let y = F256::new(inputs[4], inputs[5]);
        let product = x * y;
        let out = acc + product;
        let mut z = vec![F128::ZERO; 11];
        z[..6].copy_from_slice(&inputs[..6]);
        z[6] = x.c0 * y.c0;
        z[7] = x.c1 * y.c1;
        z[8] = (x.c0 + x.c1) * (y.c0 + y.c1);
        z[9] = out.c0;
        z[10] = out.c1;
        outputs.extend_from_slice(&[out.c0, out.c1]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

pub(super) fn emit_mac256(
    sb: &mut ShapeBuilder,
    slot: SlotId,
    acc: [Wire; 2],
    x: [Wire; 2],
    y: [Wire; 2],
) -> [Wire; 2] {
    let out = sb.gate(slot, &[acc[0], acc[1], x[0], x[1], y[0], y[1]]);
    [out[0], out[1]]
}

/// One layer of the multipoint anchor's four-state boundary computation.
pub(super) struct AssistLayerGate {
    pub(super) ty: Arc<ElementTableType>,
}

pub(super) const AL_IN: usize = 9; // g0..g3, za, rb, rc, rd, one
pub(super) const AL_OUT0: usize = 49;

impl AssistLayerGate {
    pub(super) fn new() -> Self {
        let one = F128::ONE;
        let sparse = assist_sparse_transitions();
        let mut b = ElementTableBuilder::new(6);
        for w in 0..AL_IN {
            b.free_wire(w);
        }
        b.mult(9, 4, 5)
            .linear(10, &[(8, one), (4, one), (5, one), (9, one)])
            .linear(11, &[(4, one), (9, one)])
            .linear(12, &[(5, one), (9, one)]);
        let eq4 = [10usize, 11, 12, 9];
        b.mult(13, 6, 7)
            .linear(14, &[(8, one), (6, one), (7, one), (13, one)])
            .linear(15, &[(6, one), (13, one)])
            .linear(16, &[(7, one), (13, one)]);
        let e = [14usize, 15, 16, 13];
        let p = |i: usize, o: usize| 17 + 4 * i + o;
        for i in 0..4 {
            for o in 0..4 {
                b.mult(p(i, o), eq4[i], o);
            }
        }
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                b.mult_lin(
                    33 + 4 * cd + s,
                    &[(p(i0, o0), one), (p(i1, o1), one)],
                    &[(e[cd], one)],
                );
            }
        }
        for s in 0..4 {
            b.linear(
                AL_OUT0 + s,
                &[(33 + s, one), (37 + s, one), (41 + s, one), (45 + s, one)],
            );
        }
        Self {
            ty: Arc::new(b.build().expect("assist layer gate")),
        }
    }
}

impl GateType for AssistLayerGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..AL_IN).map(IoWord::input).collect();
        for s in 0..4 {
            schema.push(IoWord::output(AL_OUT0 + s));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let sparse = assist_sparse_transitions();
        let mut z = vec![F128::ZERO; 53];
        z[..AL_IN].copy_from_slice(&inputs[..AL_IN]);
        // z[8] is the ONE input wire the table's linear rows read — eval
        // must mirror the constraint, not shortcut it with a literal one:
        // the counts* padding rows run this eval on all-zero inputs, and a
        // literal would produce a row the zerocheck rejects.
        z[9] = z[4] * z[5];
        z[10] = z[8] + z[4] + z[5] + z[9];
        z[11] = z[4] + z[9];
        z[12] = z[5] + z[9];
        let eq4 = [10usize, 11, 12, 9];
        z[13] = z[6] * z[7];
        z[14] = z[8] + z[6] + z[7] + z[13];
        z[15] = z[6] + z[13];
        z[16] = z[7] + z[13];
        let e = [14usize, 15, 16, 13];
        let p = |i: usize, o: usize| 17 + 4 * i + o;
        for i in 0..4 {
            for o in 0..4 {
                z[p(i, o)] = z[eq4[i]] * z[o];
            }
        }
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                z[33 + 4 * cd + s] = z[e[cd]] * (z[p(i0, o0)] + z[p(i1, o1)]);
            }
        }
        for s in 0..4 {
            z[AL_OUT0 + s] = z[33 + s] + z[37 + s] + z[41 + s] + z[45 + s];
        }
        outputs.extend_from_slice(&z[AL_OUT0..AL_OUT0 + 4]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// One element-zerocheck round (degree-3, convention A): `g0` rides as
/// ADVICE (a public input) and the gate enforces its defining identity as a
/// published-zero delta — the family-I pattern, no in-circuit inversion:
///
///   delta = g0 (1+t) + running + t g1          (must be 0)
///   running' = g0 (1+rho) + g1 rho + g_inf rho (1+rho)
pub(super) struct ZcRoundGate {
    pub(super) ty: Arc<ElementTableType>,
}

impl ZcRoundGate {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        // in: running(0) g1(1) gi(2) t(3) rho(4) g0(5) one(6)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..7 {
            b.free_wire(w);
        }
        b.mult_lin(7, &[(5, o)], &[(6, o), (3, o)]); // g0(1+t)
        b.mult(8, 3, 1); // t g1
        b.linear(9, &[(7, o), (0, o), (8, o)]); // delta
        b.mult_lin(10, &[(5, o)], &[(6, o), (4, o)]); // g0(1+rho)
        b.mult(11, 1, 4); // g1 rho
        b.mult_lin(12, &[(4, o)], &[(6, o), (4, o)]); // rho(1+rho)
        b.mult(13, 2, 12); // gi rho(1+rho)
        b.linear(14, &[(10, o), (11, o), (13, o)]);
        Self {
            ty: Arc::new(b.build().expect("zc round gate")),
        }
    }
}

impl GateType for ZcRoundGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..7).map(IoWord::input).collect();
        schema.push(IoWord::output(9));
        schema.push(IoWord::output(14));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 15];
        z[..7].copy_from_slice(&inputs[..7]);
        z[7] = z[5] * (z[6] + z[3]);
        z[8] = z[3] * z[1];
        z[9] = z[7] + z[0] + z[8];
        z[10] = z[5] * (z[6] + z[4]);
        z[11] = z[1] * z[4];
        z[12] = z[4] * (z[6] + z[4]);
        z[13] = z[2] * z[12];
        z[14] = z[10] + z[11] + z[13];
        outputs.extend_from_slice(&[z[9], z[14]]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}
