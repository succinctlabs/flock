use std::sync::Arc;

use flock_core::{
    circuit::builder::SlotId,
    element_r1cs::{ElementTableBuilder, ElementTableType},
    pcs::{jagged::assist_sparse_transitions, ligerito::eval_sk_at_vks},
    schedule::IoWord,
};

use crate::{
    prover::UnionElementSlotInput,
    tower::{F128, GateType, ShapeBuilder, SlotWitness, TableType, Wire},
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

/// The normalized novel-basis ladder at one point: from `W_0 = x` and
/// `one`, the chain `W_{k+1} = d_k · W_k (W_k + 1)` with the subspace
/// constants `d_k = s_k(v_k)^2 / s_{k+1}(v_{k+1})`. The affine query phase
/// reads `Ŵ_i(x_q)` off one row per query.
pub(super) struct ResidualWeightsGate {
    pub(super) ty: Arc<ElementTableType>,
    pub(super) coeffs: Vec<F128>,
}

impl ResidualWeightsGate {
    /// The ladder emitting `W_1 .. W_{n-1}`. The affine query phase needs
    /// `W_c` at the level's own `c`, which reaches 19 at the m32 L0.
    pub(super) fn with_weights(n: usize) -> Self {
        let o = F128::ONE;
        // The subspace-polynomial chain constants `s_k(v_k)`, straight from
        // ligerito (the residual boundary check pins the spine against them).
        let sks = eval_sk_at_vks(n);
        let coeffs: Vec<F128> = (0..n - 1)
            .map(|k| {
                assert_ne!(sks[k + 1], F128::ZERO, "novel-basis normalizer is nonzero");
                sks[k] * sks[k] * sks[k + 1].inv()
            })
            .collect();
        // in: W_0=q, one. out: W_1..W_{n-1}.
        let mut b = ElementTableBuilder::new((n + 1).next_power_of_two().trailing_zeros() as usize);
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

impl GateType for ResidualWeightsGate {
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

/// Squarings per [`SquareChainGate`] row.
pub(super) const SQUARINGS: usize = 8;

/// Eight chained squarings: input `x`, outputs `x^2, x^4, .., x^(2^8)`.
/// Family H's Frobenius powers `a^(2^j)` ride `ceil(j/8)` rows each
/// ([`frobenius_power`]), and every output is readable, so one chain also
/// yields every intermediate power.
pub(super) struct SquareChainGate {
    pub(super) ty: Arc<ElementTableType>,
}

impl SquareChainGate {
    pub(super) fn new() -> Self {
        let mut b = ElementTableBuilder::new(4);
        b.free_wire(0);
        for i in 1..=SQUARINGS {
            b.mult(i, i - 1, i - 1);
        }
        Self {
            ty: Arc::new(b.build().expect("square chain gate")),
        }
    }
}

impl GateType for SquareChainGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema = vec![IoWord::input(0)];
        schema.extend((1..=SQUARINGS).map(IoWord::output));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![inputs[0]; SQUARINGS + 1];
        for i in 1..=SQUARINGS {
            z[i] = z[i - 1] * z[i - 1];
        }
        outputs.extend_from_slice(&z[1..]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// `x^(2^j)` on `ceil(j/8)` [`SquareChainGate`] rows (`x` itself at `j = 0`).
pub(super) fn frobenius_power(sb: &mut ShapeBuilder, sq: SlotId, x: Wire, j: usize) -> Wire {
    let mut x = x;
    let mut left = j;
    while left > 0 {
        let step = left.min(SQUARINGS);
        x = sb.gate(sq, &[x])[step - 1];
        left -= step;
    }
    x
}

/// `[x^(2^1), .., x^(2^n)]`: every power along one chain.
pub(super) fn frobenius_orbit(sb: &mut ShapeBuilder, sq: SlotId, x: Wire, n: usize) -> Vec<Wire> {
    let mut out = Vec::with_capacity(n);
    let mut x = x;
    while out.len() < n {
        let o = sb.gate(sq, &[x]);
        let take = (n - out.len()).min(SQUARINGS);
        out.extend_from_slice(&o[..take]);
        x = o[SQUARINGS - 1];
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::affine_fixture::rand_vec;

    /// `frobenius_power` and `frobenius_orbit` match native squaring at
    /// every chain length, including a partial last row, and a wrong
    /// square fails the row relation.
    #[test]
    fn square_chains_match_native_squaring() {
        let pow2 = |x: F128, j: usize| (0..j).fold(x, |y, _| y * y);
        let mut sb = ShapeBuilder::new(8);
        let sq = sb.slot(SquareChainGate::new());
        let xs = rand_vec(3, 0x5C0A);
        let mut ins = Vec::new();
        let mut want = Vec::new();
        for (i, &x) in xs.iter().enumerate() {
            ins.push(x);
            let xw = sb.input();
            for j in [0, 1, 7, 8, 9, 16, 127] {
                let p = frobenius_power(&mut sb, sq, xw, j);
                sb.publish(p);
                want.push(pow2(x, j));
            }
            let n = [5, 8, 127][i];
            for w in frobenius_orbit(&mut sb, sq, xw, n) {
                sb.publish(w);
            }
            want.extend((1..=n).map(|j| pow2(x, j)));
        }
        let shape = sb.finish().expect("the square-chain circuit builds");
        let built = shape.run(&ins, &[]);
        assert_eq!(built.public, want, "every Frobenius power");
        let ty = SquareChainGate::new().ty;
        let ok = |row: &[F128]| {
            let mut z = row.to_vec();
            z.resize(ty.width(), F128::ZERO);
            ty.satisfies(&z, 0, 1)
        };
        let rows = built.rows::<SquareChainGate>(sq);
        assert!(rows.iter().all(|r| ok(r)));
        let mut bad = rows[0].clone();
        bad[3] += F128::ONE;
        assert!(!ok(&bad), "a wrong square fails the row");
    }
}
