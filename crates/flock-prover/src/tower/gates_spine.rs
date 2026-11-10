use std::sync::Arc;

use flock_core::{
    element_r1cs::{ElementTableBuilder, ElementTableType},
    pcs::{jagged::assist_sparse_transitions, ligerito::eval_sk_at_vks},
    schedule::IoWord,
};

use crate::{
    prover::UnionElementSlotInput,
    tower::{F128, GateType, SlotWitness, TableType},
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

/// Squarings per [`SquareChainGate`] row: one input plus 31 squares fills
/// the 32-column (`kappa = 5`) envelope exactly.
pub(super) const SQ_N: usize = 31;

/// A Frobenius chain: `x, x^2, x^4, …, x^(2^SQ_N)` — one `mult` column per
/// squaring, every power an output. Rows chain (the next row's input is
/// this row's last output), so `n` consecutive powers cost `ceil(n/31)`
/// rows. Family H reads `x^(2^-j) = x^(2^(128-j))` off these chains: on
/// F_{2^128} the Frobenius has order 128, so the inverse powers it needs are
/// forward squarings and no square-root advice is involved.
pub(super) struct SquareChainGate {
    pub(super) ty: Arc<ElementTableType>,
}

impl SquareChainGate {
    pub(super) fn new() -> Self {
        let mut b = ElementTableBuilder::new(gate_kappa(1 + SQ_N));
        b.free_wire(0);
        for i in 1..=SQ_N {
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
        schema.extend((1..=SQ_N).map(IoWord::output));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; 1 + SQ_N];
        z[0] = inputs[0];
        for i in 1..=SQ_N {
            z[i] = z[i - 1] * z[i - 1];
        }
        outputs.extend_from_slice(&z[1..]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

// HornerStepGate columns.
const HS_G: usize = 7;
const HS_V: usize = 8;
const HS_HP: usize = 9;
const HS_ONE: usize = 10;
const HS_IN: usize = 11;
const HS_M01: usize = 11;
const HS_M02: usize = 12;
const HS_M12: usize = 13;
const HS_M012: usize = 14;
const HS_P2: usize = 15; // p2..p6 at 15..=19
const HS_E1: usize = 20; // e1..e3 at 20..=22
const HS_X: usize = 23;
const HS_GV: usize = 24;
const HS_OUT: usize = 25;
const HS_H2: usize = 26;
const HS_H: usize = 27;
const HS_K: usize = 28;

/// One step of family H's Frobenius-twisted sum, by Horner:
///
/// ```text
///   h = g · MLE_0(s) · v + hp²
/// ```
///
/// where `MLE_0` is the multilinear extension of row 0 of the inverse Moore
/// matrix (the trace-dual basis `d_t`) over its 7 index bits. Every row of
/// that matrix is row 0 Frobenius-shifted, which is what lets ONE fixed row
/// serve all 128 steps — so its constants live in the table's A/B rows
/// instead of riding wires. Inputs: `s_0..s_6, g, v, hp, one`.
///
/// `d_t = g0·ratio^t + corr_t·[t < 7]` (the GHASH dual basis is geometric
/// above index 6; [`family_h_dual_decomposition`]), so
///
/// ```text
///   MLE_0(s) = g0·Π_q (1 + c_q·s_q)  +  E(s_3..s_6) · Q(s_0, s_1, s_2)
/// ```
///
/// with `c_q = 1 + ratio^(2^q)`, `E = Π_{q≥3} (1 + s_q)`, and `Q` the
/// correction polynomial in monomial form (`Q_S = Σ_{t ⊆ S, t<7} corr_t`).
/// The constant `1` is the `one` input, so the all-zero padding row
/// satisfies the relation.
///
/// [`family_h_dual_decomposition`]: crate::tower::real_walker::family_h_dual_decomposition
pub(super) struct HornerStepGate {
    pub(super) ty: Arc<ElementTableType>,
    g0: F128,
    c: [F128; 7],
    /// Monomial coefficients of `Q`, indexed by the subset bitmask of
    /// `{s_0, s_1, s_2}`.
    q: [F128; 8],
}

impl HornerStepGate {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        let (g0, ratio, corr) = crate::tower::real_walker::family_h_dual_decomposition();
        let mut c = [F128::ZERO; 7];
        let mut rq = ratio;
        for cq in &mut c {
            *cq = o + rq;
            rq = rq * rq;
        }
        let mut q = [F128::ZERO; 8];
        for (s, qs) in q.iter_mut().enumerate() {
            for (t, &ct) in corr.iter().enumerate() {
                if t & !s == 0 {
                    *qs += ct;
                }
            }
        }
        let mut b = ElementTableBuilder::new(gate_kappa(HS_K));
        for w in 0..HS_IN {
            b.free_wire(w);
        }
        let one = HS_ONE;
        b.mult(HS_M01, 0, 1)
            .mult(HS_M02, 0, 2)
            .mult(HS_M12, 1, 2)
            .mult(HS_M012, HS_M01, 2);
        // L0·L1 is linear once s0·s1 is a column.
        let l01 = vec![(one, o), (0, c[0]), (1, c[1]), (HS_M01, c[0] * c[1])];
        b.mult_lin(HS_P2, &l01, &[(one, o), (2, c[2])]);
        for qi in 3..7 {
            let at = HS_P2 + qi - 2;
            b.mult_lin(at, &[(at - 1, o)], &[(one, o), (qi, c[qi])]);
        }
        b.mult_lin(HS_E1, &[(one, o), (3, o)], &[(one, o), (4, o)]);
        b.mult_lin(HS_E1 + 1, &[(HS_E1, o)], &[(one, o), (5, o)]);
        b.mult_lin(HS_E1 + 2, &[(HS_E1 + 1, o)], &[(one, o), (6, o)]);
        let q_cols = [one, 0, 1, HS_M01, 2, HS_M02, HS_M12, HS_M012];
        let q_terms: Vec<(usize, F128)> = q_cols
            .iter()
            .zip(&q)
            .filter(|(_, qs)| **qs != F128::ZERO)
            .map(|(&col, &qs)| (col, qs))
            .collect();
        b.mult_lin(HS_X, &q_terms, &[(HS_E1 + 2, o)]);
        b.mult(HS_GV, HS_G, HS_V);
        b.mult_lin(HS_OUT, &[(HS_GV, o)], &[(HS_P2 + 4, g0), (HS_X, o)]);
        b.mult(HS_H2, HS_HP, HS_HP);
        b.linear(HS_H, &[(HS_OUT, o), (HS_H2, o)]);
        Self {
            ty: Arc::new(b.build().expect("Horner step gate")),
            g0,
            c,
            q,
        }
    }
}

impl GateType for HornerStepGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..HS_IN).map(IoWord::input).collect();
        schema.push(IoWord::output(HS_H));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; HS_K];
        z[..HS_IN].copy_from_slice(&inputs[..HS_IN]);
        let (c, one) = (&self.c, z[HS_ONE]);
        z[HS_M01] = z[0] * z[1];
        z[HS_M02] = z[0] * z[2];
        z[HS_M12] = z[1] * z[2];
        z[HS_M012] = z[HS_M01] * z[2];
        let l01 = one + c[0] * z[0] + c[1] * z[1] + c[0] * c[1] * z[HS_M01];
        z[HS_P2] = l01 * (one + c[2] * z[2]);
        for qi in 3..7 {
            let at = HS_P2 + qi - 2;
            z[at] = z[at - 1] * (one + c[qi] * z[qi]);
        }
        z[HS_E1] = (one + z[3]) * (one + z[4]);
        z[HS_E1 + 1] = z[HS_E1] * (one + z[5]);
        z[HS_E1 + 2] = z[HS_E1 + 1] * (one + z[6]);
        let mono = [
            one, z[0], z[1], z[HS_M01], z[2], z[HS_M02], z[HS_M12], z[HS_M012],
        ];
        let qv = mono
            .iter()
            .zip(&self.q)
            .fold(F128::ZERO, |acc, (&m, &qs)| acc + qs * m);
        z[HS_X] = qv * z[HS_E1 + 2];
        z[HS_GV] = z[HS_G] * z[HS_V];
        z[HS_OUT] = z[HS_GV] * (self.g0 * z[HS_P2 + 4] + z[HS_X]);
        z[HS_H2] = z[HS_HP] * z[HS_HP];
        z[HS_H] = z[HS_OUT] + z[HS_H2];
        outputs.push(z[HS_H]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// Partial words summed per [`TransposeDotGate`] row: the 16 destination
/// bytes of one transposed word.
pub(super) const TD_PARTS: usize = 16;
const TD_EQ: usize = 1;
const TD_P: usize = 2;
const TD_M: usize = TD_P + TD_PARTS;
const TD_OUT: usize = TD_M + 1;

/// One transposed word of family H's bit-transpose dot:
///
/// ```text
///   out = acc + eq · (p_0 + … + p_15)
/// ```
///
/// The 8×8 transpose tiles hand back transposed word `t` as 16 disjoint
/// byte pieces (one per destination byte), so the word is their sum, and
/// the sum is free inside one product of linear forms. Inputs: `acc, eq,
/// p_0..p_15`; output `out`. 128 rows per claim, where multiplying every
/// piece into the dot separately took 2,048 MACs.
pub(super) struct TransposeDotGate {
    pub(super) ty: Arc<ElementTableType>,
}

impl TransposeDotGate {
    pub(super) fn new() -> Self {
        let o = F128::ONE;
        let mut b = ElementTableBuilder::new(gate_kappa(TD_OUT + 1));
        for w in 0..TD_M {
            b.free_wire(w);
        }
        let parts: Vec<(usize, F128)> = (TD_P..TD_M).map(|w| (w, o)).collect();
        b.mult_lin(TD_M, &[(TD_EQ, o)], &parts);
        b.linear(TD_OUT, &[(0, o), (TD_M, o)]);
        Self {
            ty: Arc::new(b.build().expect("transpose dot gate")),
        }
    }
}

impl GateType for TransposeDotGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..TD_M).map(IoWord::input).collect();
        schema.push(IoWord::output(TD_OUT));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let mut z = vec![F128::ZERO; TD_OUT + 1];
        z[..TD_M].copy_from_slice(&inputs[..TD_M]);
        let sum = z[TD_P..TD_M].iter().fold(F128::ZERO, |a, &p| a + p);
        z[TD_M] = z[TD_EQ] * sum;
        z[TD_OUT] = z[0] + z[TD_M];
        outputs.push(z[TD_OUT]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

#[cfg(test)]
mod horner_tests {
    use flock_core::pcs::ring_switch::moore_inverse;

    use super::*;

    /// splitmix64 — deterministic test values without a rand dependency.
    struct StdRng(u64);
    impl StdRng {
        fn seed_from_u64(s: u64) -> Self {
            Self(s)
        }
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    fn rand_f(rng: &mut StdRng) -> F128 {
        F128::new(rng.next(), rng.next())
    }

    fn frob(mut x: F128, j: usize) -> F128 {
        for _ in 0..j {
            x = x * x;
        }
        x
    }

    /// `Σ_t eq(s, t) · inv[row·128 + t]`, brute force.
    fn moore_row_mle(row: usize, s: &[F128; 7]) -> F128 {
        let minv = moore_inverse();
        (0..128).fold(F128::ZERO, |acc, t| {
            let eq = (0..7).fold(F128::ONE, |p, q| {
                p * if (t >> q) & 1 == 1 {
                    s[q]
                } else {
                    F128::ONE + s[q]
                }
            });
            acc + eq * minv[row * 128 + t]
        })
    }

    /// Lay `rows` out column-major at `n_log`, then check that the honest
    /// layout satisfies and that `+1` on ANY used column of row 0 (a random
    /// row) does not: the outputs are bound by their defining constraint and
    /// every input feeds one. The regression Ron's probe asked to keep: the
    /// honest-row tests above would pass a relation that left a column free.
    fn assert_every_column_bound(ty: &ElementTableType, rows: &[Vec<F128>], n_log: usize) {
        let layout = |rows: &[Vec<F128>]| {
            let mut z = vec![F128::ZERO; ty.width() << n_log];
            for (j, row) in rows.iter().enumerate() {
                for (col, &x) in row.iter().enumerate() {
                    z[(col << n_log) + j] = x;
                }
            }
            z
        };
        assert!(
            ty.satisfies(&layout(rows), n_log, rows.len()),
            "the honest rows satisfy"
        );
        for col in 0..rows[0].len() {
            let mut bad = rows.to_vec();
            bad[0][col] += F128::ONE;
            assert!(
                !ty.satisfies(&layout(&bad), n_log, rows.len()),
                "+1 on column {col} must break the relation"
            );
        }
    }

    #[test]
    fn horner_step_rows_satisfy_and_evaluate_row_zero() {
        let g = HornerStepGate::new();
        let mut rng = StdRng::seed_from_u64(7);
        let mut rows = Vec::new();
        for _ in 0..16 {
            let s: [F128; 7] = std::array::from_fn(|_| rand_f(&mut rng));
            let (gg, v, hp) = (rand_f(&mut rng), rand_f(&mut rng), rand_f(&mut rng));
            let mut inp = s.to_vec();
            inp.extend([gg, v, hp, F128::ONE]);
            let mut out = Vec::new();
            rows.push(g.eval(&inp, &(), &mut out));
            assert_eq!(out[0], gg * moore_row_mle(0, &s) * v + hp * hp);
        }
        // The all-zero padding row.
        let mut out = Vec::new();
        rows.push(g.eval(&[F128::ZERO; HS_IN], &(), &mut out));
        assert_eq!(out[0], F128::ZERO);
        let n_log = 5;
        let mut z = vec![F128::ZERO; g.ty.width() << n_log];
        for (j, row) in rows.iter().enumerate() {
            for (col, &x) in row.iter().enumerate() {
                z[(col << n_log) + j] = x;
            }
        }
        assert!(g.ty.satisfies(&z, n_log, rows.len()), "Horner rows satisfy");
        assert_every_column_bound(&g.ty, &rows, n_log);
    }

    #[test]
    fn square_chain_rows_satisfy() {
        let g = SquareChainGate::new();
        let mut rng = StdRng::seed_from_u64(9);
        let x = rand_f(&mut rng);
        let mut out = Vec::new();
        let row = g.eval(&[x], &(), &mut out);
        assert_eq!(out[SQ_N - 1], frob(x, SQ_N));
        let n_log = 1;
        let mut z = vec![F128::ZERO; g.ty.width() << n_log];
        for (col, &v) in row.iter().enumerate() {
            z[col << n_log] = v;
        }
        assert!(g.ty.satisfies(&z, n_log, 2));
        // A second chained row: its input is the first row's last power.
        let row2 = g.eval(&[out[SQ_N - 1]], &(), &mut Vec::new());
        assert_every_column_bound(&g.ty, &[row, row2], n_log);
    }

    /// The identity family H's Horner chain rests on, end to end at the
    /// gate level: `Σ_j γ·MLE_j(r)·v_j^(2^j)` equals the Horner chain over
    /// `e_j = φ^-j(γ)·MLE_0(φ^-j(r))` with `φ^-j = φ^(128-j)`.
    #[test]
    fn horner_chain_matches_frobenius_twisted_sum() {
        let g = HornerStepGate::new();
        let mut rng = StdRng::seed_from_u64(11);
        let r: [F128; 7] = std::array::from_fn(|_| rand_f(&mut rng));
        let gamma = rand_f(&mut rng);
        let v: Vec<F128> = (0..128).map(|_| rand_f(&mut rng)).collect();
        let mut brute = F128::ZERO;
        for j in 0..128 {
            brute += gamma * moore_row_mle(j, &r) * frob(v[j], j);
        }
        let mut h = F128::ZERO;
        for j in (0..128).rev() {
            let i = (128 - j) % 128;
            let mut inp: Vec<F128> = r.iter().map(|&x| frob(x, i)).collect();
            inp.extend([frob(gamma, i), v[j], h, F128::ONE]);
            let mut out = Vec::new();
            g.eval(&inp, &(), &mut out);
            h = out[0];
        }
        assert_eq!(h, brute);
    }

    #[test]
    fn transpose_dot_rows_satisfy() {
        let g = TransposeDotGate::new();
        let mut rng = StdRng::seed_from_u64(13);
        let mut rows = Vec::new();
        for _ in 0..8 {
            let inp: Vec<F128> = (0..TD_M).map(|_| rand_f(&mut rng)).collect();
            let mut out = Vec::new();
            rows.push(g.eval(&inp, &(), &mut out));
            let sum = inp[TD_P..].iter().fold(F128::ZERO, |a, &p| a + p);
            assert_eq!(out[0], inp[0] + inp[TD_EQ] * sum);
        }
        let mut out = Vec::new();
        rows.push(g.eval(&[F128::ZERO; TD_M], &(), &mut out));
        let n_log = 4;
        let mut z = vec![F128::ZERO; g.ty.width() << n_log];
        for (j, row) in rows.iter().enumerate() {
            for (col, &x) in row.iter().enumerate() {
                z[(col << n_log) + j] = x;
            }
        }
        assert!(
            g.ty.satisfies(&z, n_log, rows.len()),
            "transpose dot rows satisfy"
        );
        assert_every_column_bound(&g.ty, &rows, n_log);
    }
}
