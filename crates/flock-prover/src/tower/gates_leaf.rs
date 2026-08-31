use super::*;
use flock_core::element_r1cs::ElementTableBuilder;
use flock_core::schedule::IoWord;

/// What the verifier actually computes from the opened leaves, at one level:
///
/// ```text
/// enforced_sum = Σ_i  α_i · ⟨row_i, eq(v_challenges, ·)⟩
/// ```
///
/// (`ligerito::induce_sumcheck_enforced_sum`.) The inner product against an
/// `eq` table IS the multilinear evaluation of the leaf's 64 lanes at the
/// 6-dimensional point `v`, so this gate evaluates it by folding, one variable
/// per level, and folds the α-weighted result into a running accumulator so
/// the whole level's sum falls out of the last row.
///
/// Layout (`kappa = 8`, 256 columns, 200 real):
///
/// ```text
///   0  .. 64   leaf lanes         (In — the SAME wires the Merkle gate reads)
///  64  .. 70   v challenges       (In)
///  70          alpha_i            (In)
///  71          prev accumulator   (In)
///  72  ..198   fold tree          (2 columns per node: d, then the folded value)
/// 198          alpha_i · y
/// 199          accumulator out    (Out)
/// ```
///
/// **The fold, and why 2 columns per node.** `build_eq_table` is LSB-first, so
/// variable `j` is bit `j` of the lane index and folding pairs `(2i, 2i+1)`:
///
/// ```text
///   new[i] = (1+v)·f[2i] + v·f[2i+1] = f[2i] + v·(f[2i] + f[2i+1])
/// ```
///
/// A row's left-hand side is a product of two linear forms, so `v·(f+f)` is
/// one `mult_lin` — the addition rides the multiplication's `A_0` row for
/// free — but the trailing `+ f[2i]` is outside the product and costs a
/// `linear` row of its own. Hence 2 rows per node, 126 for the 63 nodes.
///
/// That is not the floor. Materializing only `d[i] = v·(f[2i]+f[2i+1])` and
/// leaving `new[i]` as the *linear form* `f[2i] + d[i]` would let the next
/// level's `mult_lin` absorb it, giving 1 row per node — at the price of `A_0`
/// rows that grow to 127 terms at the last level. Left for later on purpose:
/// this is the MVP.
pub(super) struct LeafEvalGate {
    pub(super) ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    lay: LeafLayout,
}

/// The column layout of a [`LeafEvalGate`] over `lanes` leaf words.
///
/// Parameterised because the levels differ: L0's leaves are 1 KiB (64 lanes
/// at `log_batch_size = 6`) and every recursive level's are 128 B (8 lanes).
/// Same shape, different width — and two levels with the same lane count
/// share one table type, hence one slot.
#[derive(Clone, Copy)]
pub(super) struct LeafLayout {
    pub(super) lanes: usize,
    vars: usize,
    pub(super) v: usize,
    pub(super) alpha: usize,
    pub(super) prev: usize,
    pub(super) fold: usize,
    pub(super) n_in: usize,
    pub(super) t: usize,
    pub(super) acc: usize,
    pub(super) k: usize,
    pub(super) kappa: usize,
}

impl LeafLayout {
    fn new(lanes: usize) -> Self {
        assert!(lanes.is_power_of_two() && lanes >= 2);
        let vars = lanes.trailing_zeros() as usize;
        let (v, alpha) = (lanes, lanes + vars);
        let (prev, fold) = (alpha + 1, alpha + 2);
        let t = fold + 2 * (lanes - 1);
        let k = t + 2;
        Self {
            lanes,
            vars,
            v,
            alpha,
            prev,
            fold,
            n_in: fold,
            t,
            acc: t + 1,
            k,
            kappa: k.next_power_of_two().trailing_zeros().max(2) as usize,
        }
    }

    /// First column of fold level `l` (`1..=vars`); level `l` has
    /// `lanes >> l` nodes and each node owns two columns.
    fn base(&self, l: usize) -> usize {
        (1..l).fold(self.fold, |acc, k| acc + 2 * (self.lanes >> k))
    }

    /// The column holding entry `j` of the array entering fold level `l`.
    fn prev_col(&self, l: usize, j: usize) -> usize {
        if l == 1 {
            j
        } else {
            self.base(l - 1) + 2 * j + 1
        }
    }

    /// The fully folded value: the last level's single node.
    fn y(&self) -> usize {
        self.base(self.vars) + 1
    }
}

impl LeafEvalGate {
    pub(super) fn new(lanes: usize) -> Self {
        let one = F128::ONE;
        let lay = LeafLayout::new(lanes);
        let mut b = ElementTableBuilder::new(lay.kappa);
        for c in 0..lay.n_in {
            b.free_wire(c);
        }
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let (p0, p1) = (lay.prev_col(l, 2 * i), lay.prev_col(l, 2 * i + 1));
                let d = lay.base(l) + 2 * i;
                b.mult_lin(d, &[(p0, one), (p1, one)], &[(lay.v + l - 1, one)]);
                b.linear(d + 1, &[(p0, one), (d, one)]);
            }
        }
        b.mult(lay.t, lay.alpha, lay.y());
        b.linear(lay.acc, &[(lay.prev, one), (lay.t, one)]);
        Self {
            ty: std::sync::Arc::new(b.build().expect("leaf-eval block is valid")),
            lay,
        }
    }
}

impl GateType for LeafEvalGate {
    /// The row's committed columns, verbatim.
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..self.lay.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.lay.acc));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let lay = self.lay;
        let mut z = vec![F128::ZERO; lay.k];
        z[..lay.n_in].copy_from_slice(&inputs[..lay.n_in]);
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let (p0, p1) = (z[lay.prev_col(l, 2 * i)], z[lay.prev_col(l, 2 * i + 1)]);
                let d = lay.base(l) + 2 * i;
                z[d] = (p0 + p1) * z[lay.v + l - 1];
                z[d + 1] = p0 + z[d];
            }
        }
        z[lay.t] = z[lay.alpha] * z[lay.y()];
        z[lay.acc] = z[lay.prev] + z[lay.t];
        outputs.extend_from_slice(&[z[lay.acc]]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// Extension-field version of the opened-row evaluation. Each extension
/// value is represented by its `(c0, c1)` base-field wires. A product
///
/// ```text
/// (a0 + a1 u)(b0 + b1 u),  u^2 = u + x^-1,
/// ```
///
/// is constrained with the three Karatsuba products `a0 b0`, `a1 b1`, and
/// `(a0+a1)(b0+b1)`; the two output limbs are linear combinations of those
/// products. L0 words enter as `(word, 0)`, while recursive commitment rows
/// already contain adjacent `(c0, c1)` words.
pub(super) struct LeafEvalGate256 {
    pub(super) ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
    lay: LeafLayout256,
}

#[derive(Clone, Copy)]
pub(super) struct LeafLayout256 {
    pub(super) lanes: usize,
    vars: usize,
    pub(super) v: usize,
    pub(super) alpha: usize,
    pub(super) prev: usize,
    pub(super) fold: usize,
    pub(super) n_in: usize,
    pub(super) t: usize,
    pub(super) acc: usize,
    pub(super) k: usize,
    pub(super) kappa: usize,
}

impl LeafLayout256 {
    fn new(lanes: usize) -> Self {
        assert!(lanes.is_power_of_two() && lanes >= 2);
        let vars = lanes.trailing_zeros() as usize;
        let v = 2 * lanes;
        let alpha = v + 2 * vars;
        let prev = alpha + 2;
        let fold = prev + 2;
        let t = fold + 5 * (lanes - 1);
        let acc = t + 3;
        let k = t + 5;
        Self {
            lanes,
            vars,
            v,
            alpha,
            prev,
            fold,
            n_in: fold,
            t,
            acc,
            k,
            kappa: k.next_power_of_two().trailing_zeros().max(2) as usize,
        }
    }

    fn base(&self, l: usize) -> usize {
        (1..l).fold(self.fold, |acc, k| acc + 5 * (self.lanes >> k))
    }

    fn prev_pair(&self, l: usize, j: usize) -> [usize; 2] {
        if l == 1 {
            [2 * j, 2 * j + 1]
        } else {
            let base = self.base(l - 1) + 5 * j;
            [base + 3, base + 4]
        }
    }

    fn y(&self) -> [usize; 2] {
        let base = self.base(self.vars);
        [base + 3, base + 4]
    }
}

/// Emit `out = add + a*b` over F256, returning the next unused column.
/// The five emitted columns are the three Karatsuba products and two limbs.
pub(super) fn build_mac256(
    b: &mut flock_core::element_r1cs::ElementTableBuilder,
    at: usize,
    add: Option<[usize; 2]>,
    a: [usize; 2],
    rhs: [usize; 2],
) -> usize {
    let one = F128::ONE;
    let nr = flock_field::gf2_256::QUADRATIC_NONRESIDUE;
    b.mult(at, a[0], rhs[0]);
    b.mult(at + 1, a[1], rhs[1]);
    b.mult_lin(
        at + 2,
        &[(a[0], one), (a[1], one)],
        &[(rhs[0], one), (rhs[1], one)],
    );
    let mut c0 = vec![(at, one), (at + 1, nr)];
    let mut c1 = vec![(at + 2, one), (at, one)];
    if let Some(add) = add {
        c0.push((add[0], one));
        c1.push((add[1], one));
    }
    b.linear(at + 3, &c0);
    b.linear(at + 4, &c1);
    at + 5
}

pub(super) fn eval_mac256(add: F256, a: F256, b: F256) -> F256 {
    add + a * b
}

impl LeafEvalGate256 {
    pub(super) fn new(lanes: usize) -> Self {
        let one = F128::ONE;
        let lay = LeafLayout256::new(lanes);
        let mut b = ElementTableBuilder::new(lay.kappa);
        for c in 0..lay.n_in {
            b.free_wire(c);
        }
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let left = lay.prev_pair(l, 2 * i);
                let right = lay.prev_pair(l, 2 * i + 1);
                let challenge = [lay.v + 2 * (l - 1), lay.v + 2 * (l - 1) + 1];
                let at = lay.base(l) + 5 * i;
                let nr = flock_field::gf2_256::QUADRATIC_NONRESIDUE;
                b.mult_lin(
                    at,
                    &[(left[0], one), (right[0], one)],
                    &[(challenge[0], one)],
                );
                b.mult_lin(
                    at + 1,
                    &[(left[1], one), (right[1], one)],
                    &[(challenge[1], one)],
                );
                b.mult_lin(
                    at + 2,
                    &[
                        (left[0], one),
                        (right[0], one),
                        (left[1], one),
                        (right[1], one),
                    ],
                    &[(challenge[0], one), (challenge[1], one)],
                );
                b.linear(at + 3, &[(left[0], one), (at, one), (at + 1, nr)]);
                b.linear(at + 4, &[(left[1], one), (at + 2, one), (at, one)]);
            }
        }
        build_mac256(
            &mut b,
            lay.t,
            Some([lay.prev, lay.prev + 1]),
            [lay.alpha, lay.alpha + 1],
            lay.y(),
        );
        Self {
            ty: std::sync::Arc::new(b.build().expect("extension leaf-eval block is valid")),
            lay,
        }
    }
}

impl GateType for LeafEvalGate256 {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..self.lay.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.lay.acc));
        schema.push(IoWord::output(self.lay.acc + 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let lay = self.lay;
        let mut z = vec![F128::ZERO; lay.k];
        z[..lay.n_in].copy_from_slice(&inputs[..lay.n_in]);
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let lp = lay.prev_pair(l, 2 * i);
                let rp = lay.prev_pair(l, 2 * i + 1);
                let left = F256::new(z[lp[0]], z[lp[1]]);
                let right = F256::new(z[rp[0]], z[rp[1]]);
                let r = F256::new(z[lay.v + 2 * (l - 1)], z[lay.v + 2 * (l - 1) + 1]);
                let out = eval_mac256(left, left + right, r);
                let at = lay.base(l) + 5 * i;
                let p0 = (left.c0 + right.c0) * r.c0;
                let p1 = (left.c1 + right.c1) * r.c1;
                let p2 = (left.c0 + right.c0 + left.c1 + right.c1) * (r.c0 + r.c1);
                z[at] = p0;
                z[at + 1] = p1;
                z[at + 2] = p2;
                z[at + 3] = out.c0;
                z[at + 4] = out.c1;
            }
        }
        let y = lay.y();
        let alpha = F256::new(z[lay.alpha], z[lay.alpha + 1]);
        let yv = F256::new(z[y[0]], z[y[1]]);
        let prev = F256::new(z[lay.prev], z[lay.prev + 1]);
        let p0 = alpha.c0 * yv.c0;
        let p1 = alpha.c1 * yv.c1;
        let p2 = (alpha.c0 + alpha.c1) * (yv.c0 + yv.c1);
        z[lay.t] = p0;
        z[lay.t + 1] = p1;
        z[lay.t + 2] = p2;
        let acc = prev + alpha * yv;
        z[lay.acc] = acc.c0;
        z[lay.acc + 1] = acc.c1;
        outputs.extend_from_slice(&z[lay.acc..lay.acc + 2]);
        z
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}
