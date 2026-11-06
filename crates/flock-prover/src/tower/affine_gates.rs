//! The narrow arithmetic the affine inner verifier rides.
//!
//! Every gate type's IO schema is a cell slot of the outer's wiring space
//! (and a gather claim), and the whole envelope shares ~512 of them
//! (`mu = nu + 9`). So the affine verifier is built from a few SHAPE-FREE
//! types whose width does not grow with lanes, `yr` or the residual
//! point, and pays for size in rows instead:
//!
//! - [`FoldChunkGate`]: `acc' = acc + w · fold(v[16]; g[4])`, LSB-first,
//!   in either fold mode (a 0/1 `mode` input). [`emit_fold`] runs a fold
//!   of any length as a TREE of chunk rows (each chunk's value, then the
//!   chunk values folded by the higher-round challenges);
//! - [`NovelPrefixGate`]: `seed · Π_j (1 + r_j (1 + Ŵ_j))`, eight factors
//!   a row (the eq prefix `seed · Π (1 + z + r)` is the shared
//!   [`PrefixGate`](crate::tower::PrefixGate));
//! - the shared [`MacGate`](crate::tower::MacGate).
//!
//! Fold modes, per round with challenge `g` over the pair `(lo, hi)`:
//!
//! ```text
//!   NOVEL (mode 0):  lo + g·hi            Σ_i v_i Π_j g_j^{i_j}
//!   EQ    (mode 1):  lo + g·(lo + hi)     Σ_i v_i Π_j eq(g_j, i_j)
//! ```
//!
//! Zero padding is exact in both modes: an absent slot is a zero value,
//! and a padded round (`g = 0`) passes `lo` through.

use std::sync::Arc;

use flock_core::{
    circuit::builder::{GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
    element_r1cs::{ElementTableBuilder, ElementTableType},
    schedule::{IoWord, TableType},
};
use flock_field::F128;

/// Values per [`FoldChunkGate`] row.
pub(super) const CHUNK: usize = 16;
/// Rounds per [`FoldChunkGate`] row (`CHUNK = 2^ROUNDS`).
pub(super) const ROUNDS: usize = 4;
/// Factors per [`NovelPrefixGate`] row.
pub(super) const PREFIX: usize = 8;

/// Fold modes ([`FoldChunkGate`]'s `mode` input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FoldMode {
    Novel,
    Eq,
}

/// One fold chunk. Inputs `v[16]` (0..16), `g[4]` (16..20), `mode` (20),
/// `w` (21), `acc` (22); output `acc' = acc + w · fold` (last column).
/// Per tree node, `t = mode·lo` and `e = g·(t + hi)`, the node's value
/// staying the linear form `lo + e`:
pub(super) struct FoldChunkGate {
    pub(super) ty: Arc<ElementTableType>,
    width: usize,
}

const V: usize = 0;
const G: usize = CHUNK;
const MODE: usize = CHUNK + ROUNDS;
const W: usize = MODE + 1;
const ACC: usize = MODE + 2;
const N_IN: usize = MODE + 3;

/// `(out, a, b)`: `z[out] = Σa · Σb`, or `Σa` when `b` is empty.
type Rows = Vec<(usize, Vec<usize>, Vec<usize>)>;

fn chunk_rows() -> (Rows, usize) {
    let mut rows: Rows = Vec::new();
    let mut next = N_IN;
    let mut col = |rows: &mut Rows, a: Vec<usize>, b: Vec<usize>| {
        rows.push((next, a, b));
        next += 1;
        next - 1
    };
    let mut cur: Vec<Vec<usize>> = (0..CHUNK).map(|i| vec![V + i]).collect();
    for j in 0..ROUNDS {
        cur = cur
            .chunks(2)
            .map(|p| {
                let t = col(&mut rows, vec![MODE], p[0].clone());
                let mut hi = p[1].clone();
                hi.push(t);
                let e = col(&mut rows, vec![G + j], hi);
                let mut f = p[0].clone();
                f.push(e);
                f
            })
            .collect();
    }
    let fold = cur.pop().unwrap();
    let m = col(&mut rows, vec![W], fold);
    let out = col(&mut rows, vec![ACC, m], vec![]);
    (rows, out + 1)
}

fn build(n_in: usize, rows: &Rows, width: usize, what: &str) -> Arc<ElementTableType> {
    let one = |f: &[usize]| -> Vec<(usize, F128)> { f.iter().map(|&c| (c, F128::ONE)).collect() };
    let kappa = width.next_power_of_two().trailing_zeros().max(2) as usize;
    let mut b = ElementTableBuilder::new(kappa);
    for w in 0..n_in {
        b.free_wire(w);
    }
    for (out, a, bb) in rows {
        if bb.is_empty() {
            b.linear(*out, &one(a));
        } else {
            b.mult_lin(*out, &one(a), &one(bb));
        }
    }
    Arc::new(b.build().expect(what))
}

fn run(rows: &Rows, width: usize, n_in: usize, inputs: &[F128]) -> Vec<F128> {
    let mut z = vec![F128::ZERO; width];
    z[..n_in].copy_from_slice(&inputs[..n_in]);
    let sum = |z: &[F128], f: &[usize]| f.iter().fold(F128::ZERO, |acc, &c| acc + z[c]);
    for (out, a, b) in rows {
        let av = sum(&z, a);
        z[*out] = if b.is_empty() { av } else { av * sum(&z, b) };
    }
    z
}

impl FoldChunkGate {
    pub(super) fn new() -> Self {
        let (rows, width) = chunk_rows();
        Self {
            ty: build(N_IN, &rows, width, "fold chunk gate"),
            width,
        }
    }
}

impl GateType for FoldChunkGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..N_IN).map(IoWord::input).collect();
        schema.push(IoWord::output(self.width - 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let (rows, _) = chunk_rows();
        let z = run(&rows, self.width, N_IN, inputs);
        outputs.push(z[self.width - 1]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// `seed · Π_{j<8} (1 + r_j·(1 + Ŵ_j))`: inputs `seed` (0), `r[8]`
/// (1..9), `Ŵ[8]` (9..17), `one` (17). Two columns a factor: `p =
/// r·(one + Ŵ)`, `pr' = pr·(one + p)`. A padded factor (`r = 0`) is one.
pub(super) struct NovelPrefixGate {
    pub(super) ty: Arc<ElementTableType>,
    width: usize,
}

const NP_IN: usize = 2 + 2 * PREFIX;

fn prefix_rows() -> (Rows, usize) {
    let one = NP_IN - 1;
    let mut rows: Rows = Vec::new();
    let mut c = NP_IN;
    let mut pr = 0;
    for j in 0..PREFIX {
        rows.push((c, vec![1 + j], vec![one, 1 + PREFIX + j]));
        rows.push((c + 1, vec![pr], vec![one, c]));
        pr = c + 1;
        c += 2;
    }
    (rows, c)
}

impl NovelPrefixGate {
    pub(super) fn new() -> Self {
        let (rows, width) = prefix_rows();
        Self {
            ty: build(NP_IN, &rows, width, "novel prefix gate"),
            width,
        }
    }
}

impl GateType for NovelPrefixGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        let mut schema: Vec<IoWord> = (0..NP_IN).map(IoWord::input).collect();
        schema.push(IoWord::output(self.width - 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let (rows, _) = prefix_rows();
        let z = run(&rows, self.width, NP_IN, inputs);
        outputs.push(z[self.width - 1]);
        z
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        SlotWitness::element_from_rows(self.ty.width(), nu, rows)
    }
}

/// The slots and constant wires the affine arithmetic emits with.
#[derive(Clone, Copy)]
pub(super) struct Arith {
    pub(super) mac: SlotId,
    pub(super) chunk: SlotId,
    pub(super) nprefix: SlotId,
    /// The shared eq [`PrefixGate`](crate::tower::PrefixGate) of width 8.
    pub(super) prefix: SlotId,
    pub(super) zw: Wire,
    pub(super) ow: Wire,
}

impl Arith {
    /// `acc + x·y`.
    pub(super) fn mac(&self, sb: &mut ShapeBuilder, acc: Wire, x: Wire, y: Wire) -> Wire {
        sb.gate(self.mac, &[acc, x, y])[0]
    }

    fn mode(&self, m: FoldMode) -> Wire {
        match m {
            FoldMode::Novel => self.zw,
            FoldMode::Eq => self.ow,
        }
    }

    /// `acc + w · fold(v; g)` for a fold of `g.len()` rounds over
    /// `v.len() <= 2^g.len()` values (the rest zero), as a tree of
    /// [`FoldChunkGate`] rows.
    pub(super) fn fold(
        &self,
        sb: &mut ShapeBuilder,
        mode: FoldMode,
        v: &[Wire],
        g: &[Wire],
        w: Wire,
        acc: Wire,
    ) -> Wire {
        assert!(
            !v.is_empty() && v.len() <= 1usize << g.len(),
            "a fold domain"
        );
        if g.len() <= ROUNDS {
            let mut ins: Vec<Wire> = v.to_vec();
            ins.resize(CHUNK, self.zw);
            ins.extend_from_slice(g);
            ins.resize(CHUNK + ROUNDS, self.zw);
            ins.extend([self.mode(mode), w, acc]);
            return sb.gate(self.chunk, &ins)[0];
        }
        let parts: Vec<Wire> = v
            .chunks(CHUNK)
            .map(|c| self.fold(sb, mode, c, &g[..ROUNDS], self.ow, self.zw))
            .collect();
        self.fold(sb, mode, &parts, &g[ROUNDS..], w, acc)
    }

    /// `seed · Π_j (1 + a_j + b_j)`, eight factors a [`PrefixGate`] row.
    ///
    /// [`PrefixGate`]: crate::tower::PrefixGate
    pub(super) fn eq_prefix(
        &self,
        sb: &mut ShapeBuilder,
        seed: Wire,
        a: &[Wire],
        b: &[Wire],
    ) -> Wire {
        assert_eq!(a.len(), b.len());
        let mut s = seed;
        for (ca, cb) in a.chunks(PREFIX).zip(b.chunks(PREFIX)) {
            let mut ins = vec![s];
            ins.extend_from_slice(ca);
            ins.resize(1 + PREFIX, self.zw);
            ins.extend_from_slice(cb);
            ins.resize(1 + 2 * PREFIX, self.zw);
            ins.push(self.ow);
            s = sb.gate(self.prefix, &ins)[0];
        }
        s
    }

    /// `seed · Π_j (1 + r_j (1 + Ŵ_j))`, eight factors a
    /// [`NovelPrefixGate`] row.
    pub(super) fn novel_prefix(
        &self,
        sb: &mut ShapeBuilder,
        seed: Wire,
        r: &[Wire],
        w: &[Wire],
    ) -> Wire {
        assert_eq!(r.len(), w.len());
        let mut s = seed;
        for (cr, cw) in r.chunks(PREFIX).zip(w.chunks(PREFIX)) {
            let mut ins = vec![s];
            ins.extend_from_slice(cr);
            ins.resize(1 + PREFIX, self.zw);
            ins.extend_from_slice(cw);
            ins.resize(1 + 2 * PREFIX, self.zw);
            ins.push(self.ow);
            s = sb.gate(self.nprefix, &ins)[0];
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use flock_core::lincheck::build_eq_table;

    use super::*;
    use crate::tower::{MacGate, PrefixGate, affine_fixture::rand_vec};

    fn novel_fold(v: &[F128], g: &[F128]) -> F128 {
        let mut tab = vec![F128::ONE];
        for &gj in g {
            let ext: Vec<F128> = tab.iter().map(|&t| t * gj).collect();
            tab.extend(ext);
        }
        v.iter().zip(&tab).fold(F128::ZERO, |a, (&x, &w)| a + x * w)
    }

    /// Tree folds of every length against the direct sums, in both modes;
    /// both prefix products against their definitions; every row
    /// satisfies its relation and a wrong derived column rejects.
    #[test]
    fn folds_and_prefixes_match_their_definitions() {
        let mut sb = ShapeBuilder::new(10);
        let a = Arith {
            mac: sb.slot(MacGate::new()),
            chunk: sb.slot(FoldChunkGate::new()),
            nprefix: sb.slot(NovelPrefixGate::new()),
            prefix: sb.slot(PrefixGate::new(PREFIX)),
            zw: sb.input(),
            ow: sb.input(),
        };
        let mut ins = vec![F128::ZERO, F128::ONE];
        let mut want = Vec::new();
        let mut outs = Vec::new();
        let mut input = |sb: &mut ShapeBuilder, v: F128| {
            ins.push(v);
            sb.input()
        };
        for (n, rounds, seed) in [
            (1, 0, 1),
            (5, 3, 2),
            (16, 4, 3),
            (36, 6, 4),
            (128, 7, 5),
            (46, 6, 6),
        ] {
            let v = rand_vec(n, seed);
            let g = rand_vec(rounds, seed + 100);
            let (w, acc) = (rand_vec(1, seed + 200)[0], rand_vec(1, seed + 300)[0]);
            let vw: Vec<Wire> = v.iter().map(|&x| input(&mut sb, x)).collect();
            let gw: Vec<Wire> = g.iter().map(|&x| input(&mut sb, x)).collect();
            let (ww, aw) = (input(&mut sb, w), input(&mut sb, acc));
            outs.push(a.fold(&mut sb, FoldMode::Novel, &vw, &gw, ww, aw));
            want.push(acc + w * novel_fold(&v, &g));
            if n == 1 << rounds {
                let eq = build_eq_table(&g);
                let mle = v.iter().zip(&eq).fold(F128::ZERO, |s, (&x, &e)| s + x * e);
                outs.push(a.fold(&mut sb, FoldMode::Eq, &vw, &gw, ww, aw));
                want.push(acc + w * mle);
            }
        }
        for len in [1usize, 8, 12, 19] {
            let (x, y) = (rand_vec(len, 7 + len as u64), rand_vec(len, 9 + len as u64));
            let seed = rand_vec(1, 11)[0];
            let xw: Vec<Wire> = x.iter().map(|&v| input(&mut sb, v)).collect();
            let yw: Vec<Wire> = y.iter().map(|&v| input(&mut sb, v)).collect();
            let sw = input(&mut sb, seed);
            outs.push(a.eq_prefix(&mut sb, sw, &xw, &yw));
            want.push(
                x.iter()
                    .zip(&y)
                    .fold(seed, |p, (&u, &v)| p * (F128::ONE + u + v)),
            );
            outs.push(a.novel_prefix(&mut sb, sw, &xw, &yw));
            want.push(
                x.iter()
                    .zip(&y)
                    .fold(seed, |p, (&r, &w)| p * (F128::ONE + r * (F128::ONE + w))),
            );
        }
        for &o in &outs {
            sb.publish(o);
        }
        let shape = sb.finish().expect("the fold circuit builds");
        let built = shape.run(&ins, &[]);
        assert_eq!(built.public, want, "every fold and prefix");
        let ok = |ty: &ElementTableType, row: &[F128]| {
            let mut z = row.to_vec();
            z.resize(ty.width(), F128::ZERO);
            ty.satisfies(&z, 0, 1)
        };
        let chunk_ty = FoldChunkGate::new().ty;
        let rows = built.rows::<FoldChunkGate>(a.chunk);
        assert!(rows.iter().all(|r| ok(&chunk_ty, r)));
        for col in [N_IN, N_IN + 5, rows[0].len() - 1] {
            let mut bad = rows[0].clone();
            bad[col] += F128::ONE;
            assert!(!ok(&chunk_ty, &bad), "a wrong column {col} fails the chunk");
        }
        let np_ty = NovelPrefixGate::new().ty;
        assert!(
            built
                .rows::<NovelPrefixGate>(a.nprefix)
                .iter()
                .all(|r| ok(&np_ty, r))
        );
        println!(
            "fold chunk: {} IO words, {} cols; novel prefix: {} IO words",
            N_IN + 1,
            rows[0].len(),
            NP_IN + 1
        );
    }
}
