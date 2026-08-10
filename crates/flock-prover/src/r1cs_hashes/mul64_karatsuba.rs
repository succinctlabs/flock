//! u64 × u64 → u128 batch multiplication R1CS, **subtractive Karatsuba**
//! (K_LOG=17, dynamic products-per-block at ~99% fill).
//!
//! Per product the free wires are p, q (identity rows); the 128 product bits
//! are linear combinations of committed bits, correct in any satisfying
//! assignment. The recursion splits n into two m = n/2 halves and uses the
//! *absolute difference* for the middle operand:
//!
//!   z0 = x0·y0,  z2 = x1·y1,  z1 = |x0−x1|·|y0−y1|,  s = sx ⊕ sy
//!   x·y = z0 + 2^m·(z0 + z2 − (−1)^s·z1) + 2^{2m}·z2
//!
//! Because |x0−x1| stays m bits (where x0+x1 would need m+1), every width in
//! the tree is balanced: 64 → 3×32 → 9×16 → 27×8, with no 17- or 18-bit
//! leaves. That is worth 329 ANDs in the leaves against ~150 for the two
//! absolute differences per node.
//!
//! Three ingredients make the arithmetic affordable over GF(2):
//!
//! - **Complement trick**: `−X ≡ Σ_i ¬x_i·2^{c_i} + const (mod 2^n)`.
//!   Complemented bits are *affine* (x ⊕ 1), so they enter carry-save pools
//!   for free; every complement's additive correction accumulates into one
//!   integer constant whose set bits join the pool as constant entries.
//! - **Conditional sign without sign extension**: the ±z1 term enters the
//!   recombination pool as (z1 ⊕ c) plus exactly two dynamic entries — c at
//!   position m and ¬c at m+w — and one static −2^{m+w}. No word-wide
//!   conditional negation, and nothing propagates across the accumulator.
//! - **Const-1 wire**: affine row sides need a pinned constant-one column
//!   (block slot 0, `const_pin`, the sha2 pattern). Consequently padding
//!   blocks/slots must carry real (p,q) = (0,0) transcripts — those are NOT
//!   all-zero once complements exist.
//!
//! Sub-product outputs are left unbound ([`BIND_DEPTH`] = 0): binding trades
//! committed bits against nnz, and nnz turned out nearly free at this scale.
//!
//! Measured against the previous additive construction (TR 7970X, m = 28):
//! 5140 vs 5911 committed bits, 25 vs 22 products/block, 7.05 vs 8.16 µs/mul
//! single-threaded, 0.93 vs 1.06 µs multithreaded at m = 30.
//!
//! Block layout: slot 0 = const-1 wire, slots [1, 64) reserved zero, then
//! `muls_per_block()` products at a 64-bit-aligned stride.

use std::sync::OnceLock;

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

use super::mul64::{Node, transpose_64x64};

/// Schoolbook base width of the Karatsuba recursion. 8 measured best: at 16
/// the leaves cost 9·496 and at 4 the recombination overhead outgrows the
/// smaller leaves (5536 / 5140 / 5356 committed bits respectively).
pub const LEAF_WIDTH: usize = 8;
/// Levels whose sub-product outputs are re-bound to single wires. 0 measured
/// best: binding trades committed bits for nnz, and nnz is nearly free here
/// (a 2.1× nnz spread moved wall clock by 2%).
pub const BIND_DEPTH: usize = 0;

pub const K_LOG: usize = 17;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;

/// Block header: slot 0 is the const-1 wire; slots [1, 64) reserved (zero).
pub const HEADER_BITS: usize = 64;
pub const CONST_SLOT: usize = 0;

/// Ligerito configs ship for m ≥ 22.
pub const MIN_N_BLOCKS_LOG: usize = 22 - K_LOG; // 5

// Node ids: [0,64) p bits, [64,128) q bits, 128 = const-one, 129 = const-zero.
const CONST_ONE_NODE: u32 = 128;
const CONST_ZERO_NODE: u32 = 129;
const N_INPUT_NODES: usize = 130;

// ───────────────────────────────────────────────────────────────────────────
// Circuit builder
// ───────────────────────────────────────────────────────────────────────────

/// Affine expression over committed (product-relative) slots:
/// value = ⊕_{s ∈ supp} z[s] ⊕ cst, with `node` evaluating it.
#[derive(Clone, Debug)]
struct Expr {
    node: u32,
    supp: Vec<u32>,
    cst: bool,
}

impl Expr {
    fn weight(&self) -> usize {
        self.supp.len() + self.cst as usize
    }
}

fn symdiff(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// One committed row: (A-support, A-const, B-support, B-const), supports in
/// product-relative slot ids.
type Row = (Vec<u32>, bool, Vec<u32>, bool);

struct Builder {
    nodes: Vec<Node>,
    /// Per committed slot: value node (an `And`).
    slot_nodes: Vec<u32>,
    /// Per committed slot: (A-side node, B-side node) for fused witness gen.
    row_ab_nodes: Vec<(u32, u32)>,
    rows: Vec<Row>,
}

impl Builder {
    fn new() -> Self {
        let mut nodes = Vec::with_capacity(60_000);
        for _ in 0..N_INPUT_NODES {
            nodes.push(Node::Input);
        }
        let mut b = Self {
            nodes,
            slot_nodes: Vec::with_capacity(8_000),
            row_ab_nodes: Vec::with_capacity(8_000),
            rows: Vec::with_capacity(8_000),
        };
        // Identity rows for the 128 free input wires (slots 0..128 = p, q).
        for s in 0..128u32 {
            b.slot_nodes.push(s);
            b.row_ab_nodes.push((s, s));
            b.rows.push((vec![s], false, vec![s], false));
        }
        b
    }

    fn one(&self) -> Expr {
        Expr {
            node: CONST_ONE_NODE,
            supp: Vec::new(),
            cst: true,
        }
    }
    fn zero(&self) -> Expr {
        Expr {
            node: CONST_ZERO_NODE,
            supp: Vec::new(),
            cst: false,
        }
    }

    fn input_bits(&self) -> (Vec<Expr>, Vec<Expr>) {
        let mk = |base: u32| {
            (0..64u32)
                .map(|i| Expr {
                    node: base + i,
                    supp: vec![base + i],
                    cst: false,
                })
                .collect::<Vec<_>>()
        };
        (mk(0), mk(64))
    }

    fn xor(&mut self, x: &Expr, y: &Expr) -> Expr {
        let node = self.nodes.len() as u32;
        self.nodes.push(Node::Xor(x.node, y.node));
        Expr {
            node,
            supp: symdiff(&x.supp, &y.supp),
            cst: x.cst ^ y.cst,
        }
    }

    fn not(&mut self, x: &Expr) -> Expr {
        let node = self.nodes.len() as u32;
        self.nodes.push(Node::Xor(x.node, CONST_ONE_NODE));
        Expr {
            node,
            supp: x.supp.clone(),
            cst: !x.cst,
        }
    }

    /// Commit `x ∧ y` as a new slot; returns the (support-1) result expr.
    fn and_row(&mut self, x: &Expr, y: &Expr) -> Expr {
        let slot = self.slot_nodes.len() as u32;
        let node = self.nodes.len() as u32;
        self.nodes.push(Node::And {
            a: x.node,
            b: y.node,
            slot,
        });
        self.slot_nodes.push(node);
        self.row_ab_nodes.push((x.node, y.node));
        self.rows
            .push((x.supp.clone(), x.cst, y.supp.clone(), y.cst));
        Expr {
            node,
            supp: vec![slot],
            cst: false,
        }
    }

    /// Full adder: (sum, carry) with one committed AND `t = (x⊕c)(y⊕c)`.
    fn fa(&mut self, x: &Expr, y: &Expr, c: &Expr) -> (Expr, Expr) {
        let a_side = self.xor(x, c);
        let b_side = self.xor(y, c);
        let t = self.and_row(&a_side, &b_side);
        let sum = self.xor(&a_side, y);
        let carry = self.xor(&t, c);
        (sum, carry)
    }

    /// Half adder: (sum, carry = u∧v committed).
    fn ha(&mut self, u: &Expr, v: &Expr) -> (Expr, Expr) {
        let t = self.and_row(u, v);
        let sum = self.xor(u, v);
        (sum, t)
    }

    /// Ripple add of two little-endian bit vectors; result has
    /// max(len)+1 bits. One committed AND per position with ≥ 2 operands.
    fn add(&mut self, xs: &[Expr], ys: &[Expr]) -> Vec<Expr> {
        let n = xs.len().max(ys.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry: Option<Expr> = None;
        for i in 0..n {
            let mut items: Vec<&Expr> = Vec::with_capacity(3);
            if let Some(x) = xs.get(i) {
                items.push(x);
            }
            if let Some(y) = ys.get(i) {
                items.push(y);
            }
            let c = carry.take();
            if let Some(ref c) = c {
                items.push(c);
            }
            match items.len() {
                3 => {
                    let (s, co) = self.fa(&items[0].clone(), &items[1].clone(), &items[2].clone());
                    out.push(s);
                    carry = Some(co);
                }
                2 => {
                    let (s, co) = self.ha(&items[0].clone(), &items[1].clone());
                    out.push(s);
                    carry = Some(co);
                }
                1 => out.push(items[0].clone()),
                _ => out.push(self.zero()),
            }
        }
        if let Some(c) = carry {
            out.push(c);
        }
        out
    }

    /// Compress weight-column pools into `ncols` result bits, mod 2^ncols.
    /// `const_acc` is the accumulated integer correction (complements etc.);
    /// its residue's set bits join the pools as constant entries. The top
    /// column is a plain XOR (mod-2^ncols semantics discard its carries).
    fn compress(&mut self, mut pools: Vec<Vec<Expr>>, ncols: usize, const_acc: i128) -> Vec<Expr> {
        assert_eq!(pools.len(), ncols);
        let c = if ncols == 128 {
            const_acc as u128
        } else {
            (const_acc as u128) & ((1u128 << ncols) - 1)
        };
        for (k, pool) in pools.iter_mut().enumerate() {
            if (c >> k) & 1 == 1 {
                pool.push(Expr {
                    node: CONST_ONE_NODE,
                    supp: Vec::new(),
                    cst: true,
                });
            }
        }

        let mut out = Vec::with_capacity(ncols);
        for k in 0..ncols {
            let mut pool = std::mem::take(&mut pools[k]);
            if k == ncols - 1 {
                // Top column: XOR-fold, discard overflow.
                let bit = match pool.len() {
                    0 => self.zero(),
                    _ => {
                        let mut acc = pool[0].clone();
                        for e in &pool[1..] {
                            acc = self.xor(&acc, e);
                        }
                        acc
                    }
                };
                out.push(bit);
                continue;
            }
            // Huffman-style: keep sorted by descending weight; combine the
            // smallest three (c = smallest, shared by both AND sides).
            pool.sort_by(|x, y| y.weight().cmp(&x.weight()));
            while pool.len() >= 3 {
                let c3 = pool.pop().unwrap();
                let x = pool.pop().unwrap();
                let y = pool.pop().unwrap();
                let (sum, carry) = self.fa(&x, &y, &c3);
                let pos = pool.partition_point(|e| e.weight() > sum.weight());
                pool.insert(pos, sum);
                pools[k + 1].push(carry);
            }
            let bit = match pool.len() {
                0 => self.zero(),
                1 => pool.pop().unwrap(),
                2 => {
                    let v = pool.pop().unwrap();
                    let u = pool.pop().unwrap();
                    let (sum, carry) = self.ha(&u, &v);
                    pools[k + 1].push(carry);
                    sum
                }
                _ => unreachable!(),
            };
            out.push(bit);
        }
        out
    }

    /// Schoolbook product of two little-endian affine bit vectors (full
    /// width, no truncation).
    fn mul_schoolbook(&mut self, xs: &[Expr], ys: &[Expr]) -> Vec<Expr> {
        let ncols = xs.len() + ys.len();
        let mut pools: Vec<Vec<Expr>> = (0..ncols).map(|_| Vec::new()).collect();
        for (i, x) in xs.iter().enumerate() {
            for (j, y) in ys.iter().enumerate() {
                let x = x.clone();
                let y = y.clone();
                let pp = self.and_row(&x, &y);
                pools[i + j].push(pp);
            }
        }
        self.compress(pools, ncols, 0)
    }

    /// Bind each bit to a fresh committed slot via `z = expr ∧ 1`, resetting
    /// its support to 1.
    fn bind(&mut self, bits: Vec<Expr>) -> Vec<Expr> {
        let one = self.one();
        bits.into_iter()
            .map(|b| self.and_row(&b, &one))
            .collect()
    }

    /// |xs − ys| (equal lengths) plus the sign bit (1 iff xs < ys).
    /// Cost: m ANDs for the borrow chain + (m−1) for the conditional negate.
    fn abs_diff(&mut self, xs: &[Expr], ys: &[Expr]) -> (Vec<Expr>, Expr) {
        let m = xs.len();
        assert_eq!(m, ys.len());
        // xs + ¬ys + 1 (mod 2^m); final carry = 1 ⟺ xs ≥ ys.
        let mut carry = self.one();
        let mut d = Vec::with_capacity(m);
        for i in 0..m {
            let ny = self.not(&ys[i]);
            let x = xs[i].clone();
            let (sum, c) = self.fa(&x, &ny, &carry);
            d.push(sum);
            carry = c;
        }
        let sign = self.not(&carry);
        // out = (d ⊕ sign) + sign
        let mut out = Vec::with_capacity(m);
        let mut cin = sign.clone();
        for i in 0..m {
            let y = self.xor(&d[i], &sign);
            let o = self.xor(&y, &cin);
            out.push(o);
            if i + 1 < m {
                cin = self.and_row(&y, &cin);
            }
        }
        (out, sign)
    }

    /// Subtractive Karatsuba: the middle operand is |x0−x1|, which stays m
    /// bits, so all three sub-products have equal width and the 17/18-bit
    /// leaves of the additive form disappear.
    ///
    ///   M = z0 + z2 − (−1)^s·z1,  s = sx ⊕ sy,  z1 = |x0−x1|·|y0−y1|
    ///
    /// with the conditional negation folded into the recombination pool as
    /// (z1 ⊕ c) plus two dynamic correction entries (c at m, ¬c at m+w) and
    /// one static −2^{m+w}; no sign extension across the accumulator.
    fn karatsuba_sub(
        &mut self,
        xs: &[Expr],
        ys: &[Expr],
        depth: usize,
        base: usize,
        bind_depth: usize,
        bind_z1: bool,
    ) -> Vec<Expr> {
        let n = xs.len();
        assert_eq!(n, ys.len());
        if n <= base {
            return self.mul_schoolbook(xs, ys);
        }
        assert!(n % 2 == 0, "subtractive path needs even widths (64/32/16/8)");
        let m = n / 2;
        let (x0, x1) = (xs[..m].to_vec(), xs[m..].to_vec());
        let (y0, y1) = (ys[..m].to_vec(), ys[m..].to_vec());

        let (dx, sx) = self.abs_diff(&x0, &x1);
        let (dy, sy) = self.abs_diff(&y0, &y1);

        let mut z0 = self.karatsuba_sub(&x0, &y0, depth + 1, base, bind_depth, bind_z1);
        let mut z2 = self.karatsuba_sub(&x1, &y1, depth + 1, base, bind_depth, bind_z1);
        let mut z1 = self.karatsuba_sub(&dx, &dy, depth + 1, base, bind_depth, bind_z1);
        if depth < bind_depth {
            z0 = self.bind(z0);
            z2 = self.bind(z2);
            if bind_z1 {
                z1 = self.bind(z1);
            }
        }

        let s = self.xor(&sx, &sy);
        let c = self.not(&s); // c = 1 ⇒ subtract z1

        let ncols = 2 * n;
        let mut pools: Vec<Vec<Expr>> = (0..ncols).map(|_| Vec::new()).collect();
        for (i, e) in z0.iter().enumerate() {
            pools[i].push(e.clone());
            pools[m + i].push(e.clone());
        }
        for (i, e) in z2.iter().enumerate() {
            pools[2 * m + i].push(e.clone());
            pools[m + i].push(e.clone());
        }
        let w = z1.len();
        for i in 0..w {
            let z = z1[i].clone();
            let t = self.xor(&z, &c);
            pools[m + i].push(t);
        }
        pools[m].push(c.clone());
        let nc = self.not(&c);
        pools[m + w].push(nc);
        let acc: i128 = -(1i128 << (m + w));
        self.compress(pools, ncols, acc)
    }

    /// Karatsuba product (full width). `depth` 0 is the top call; the three
    /// depth-1 sub-product outputs are bound.
    fn karatsuba(&mut self, xs: &[Expr], ys: &[Expr], depth: usize) -> Vec<Expr> {
        let n = xs.len();
        assert_eq!(n, ys.len());
        if n <= 18 {
            return self.mul_schoolbook(xs, ys);
        }
        let m = n / 2;
        let (x0, x1) = (&xs[..m], &xs[m..]);
        let (y0, y1) = (&ys[..m], &ys[m..]);

        let ps = self.add(x0, x1);
        let qs = self.add(y0, y1);

        let mut z0 = self.karatsuba(x0, y0, depth + 1);
        let mut z2 = self.karatsuba(x1, y1, depth + 1);
        let mut z1 = self.karatsuba(&ps, &qs, depth + 1);
        if depth == 0 {
            z0 = self.bind(z0);
            z2 = self.bind(z2);
            z1 = self.bind(z1);
        }

        // r = z0 + (z1 − z0 − z2)·2^m + z2·2^2m   (mod 2^ncols)
        let ncols = xs.len() + ys.len();
        let mut pools: Vec<Vec<Expr>> = (0..ncols).map(|_| Vec::new()).collect();
        let mut acc: i128 = 0;
        for (i, e) in z0.iter().enumerate() {
            pools[i].push(e.clone());
        }
        for (i, e) in z2.iter().enumerate() {
            pools[2 * m + i].push(e.clone());
        }
        for (i, e) in z1.iter().enumerate() {
            if m + i < ncols {
                pools[m + i].push(e.clone());
            } else {
                // z1's top bits can only carry information ≥ 2^ncols when the
                // subtraction cancels it; mod 2^ncols they still contribute.
                unreachable!("z1 exceeds ncols: {} + {} >= {}", m, i, ncols);
            }
        }
        // −z0·2^m and −z2·2^m via complements.
        for (label, z) in [(0usize, &z0), (1, &z2)] {
            let _ = label;
            for (i, e) in z.iter().enumerate() {
                let ne = self.not(e);
                pools[m + i].push(ne);
            }
            acc += (1i128 << m) - (1i128 << (m + z.len()));
        }
        self.compress(pools, ncols, acc)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Built circuit + layout
// ───────────────────────────────────────────────────────────────────────────

pub struct KMulCircuit {
    pub nodes: Vec<Node>,
    pub slot_nodes: Vec<u32>,
    pub row_ab_nodes: Vec<(u32, u32)>,
    rows: Vec<Row>,
    pub product_bit_nodes: [u32; 128],
    /// Committed slots per product.
    pub sub_bits: usize,
    /// Product stride in the block (64-bit aligned).
    pub stride: usize,
    /// Products per K_LOG=17 block.
    pub muls_per_block: usize,
    pub useful_bits: usize,
    /// Total A₀+B₀ nonzeros per product (excl. const column refs).
    pub nnz: usize,
}

fn build_circuit() -> KMulCircuit {
    // Default: subtractive Karatsuba at the tuned parameters. Two escapes,
    // both for A/B measurement only:
    //   MUL64K_ADDITIVE=1        the previous additive construction
    //   MUL64K_SUB=base,bind,bz1 other points in the subtractive family
    if let Ok(cfg) = std::env::var("MUL64K_SUB") {
        let v: Vec<usize> = cfg
            .split(',')
            .map(|x| x.trim().parse().expect("MUL64K_SUB=base,bind,bindz1"))
            .collect();
        assert_eq!(v.len(), 3, "MUL64K_SUB=base,bind,bindz1");
        return build_circuit_sub(v[0], v[1], v[2] != 0);
    }
    if std::env::var("MUL64K_ADDITIVE").is_err() {
        return build_circuit_sub(LEAF_WIDTH, BIND_DEPTH, true);
    }
    let mut b = Builder::new();
    let (p, q) = b.input_bits();
    let r = b.karatsuba(&p, &q, 0);
    assert_eq!(r.len(), 128);
    let mut product_bit_nodes = [0u32; 128];
    for (k, e) in r.iter().enumerate() {
        product_bit_nodes[k] = e.node;
    }

    let sub_bits = b.slot_nodes.len();
    let stride = sub_bits.next_multiple_of(64);
    let muls_per_block = (K - HEADER_BITS) / stride;
    let useful_bits = HEADER_BITS + muls_per_block * stride;
    let nnz: usize = b.rows.iter().map(|(a, _, bs, _)| a.len() + bs.len()).sum();

    KMulCircuit {
        nodes: b.nodes,
        slot_nodes: b.slot_nodes,
        row_ab_nodes: b.row_ab_nodes,
        rows: b.rows,
        product_bit_nodes,
        sub_bits,
        stride,
        muls_per_block,
        useful_bits,
        nnz,
    }
}

/// Build the subtractive-Karatsuba circuit with a configurable schoolbook
/// base width and bind depth (levels 0..bind_depth materialize their three
/// sub-product outputs).
fn build_circuit_sub(base: usize, bind_depth: usize, bind_z1: bool) -> KMulCircuit {
    let mut b = Builder::new();
    let (p, q) = b.input_bits();
    let r = b.karatsuba_sub(&p, &q, 0, base, bind_depth, bind_z1);
    assert_eq!(r.len(), 128);
    let mut product_bit_nodes = [0u32; 128];
    for (k, e) in r.iter().enumerate() {
        product_bit_nodes[k] = e.node;
    }
    let sub_bits = b.slot_nodes.len();
    let stride = sub_bits.next_multiple_of(64);
    let muls_per_block = (K - HEADER_BITS) / stride;
    let useful_bits = HEADER_BITS + muls_per_block * stride;
    let nnz: usize = b.rows.iter().map(|(a, _, bs, _)| a.len() + bs.len()).sum();
    KMulCircuit {
        nodes: b.nodes,
        slot_nodes: b.slot_nodes,
        row_ab_nodes: b.row_ab_nodes,
        rows: b.rows,
        product_bit_nodes,
        sub_bits,
        stride,
        muls_per_block,
        useful_bits,
        nnz,
    }
}

/// (committed bits, nnz, muls/block, self-check passed) for the subtractive
/// variant. The self-check evaluates the node graph on 64 random lanes and
/// compares against native u128 multiplication.
pub fn stats_sub(base: usize, bind_depth: usize, bind_z1: bool) -> (usize, usize, usize, bool, usize) {
    let circ = build_circuit_sub(base, bind_depth, bind_z1);
    let mut st: u64 = 0x1234_5678_9abc_def0;
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    // Edge lanes exercise the abs_diff paths: x0==x1 (dx=0), x0<x1, x0>x1,
    // all-ones, single bits, and the 2^32 boundary.
    let edges: [u64; 12] = [
        0, 1, u64::MAX, 1 << 63, 1 << 32, (1 << 32) - 1,
        0x0000_0001_0000_0001, 0xffff_ffff_0000_0000,
        0x0000_0000_ffff_ffff, 0x8000_0000_8000_0000,
        0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555,
    ];
    let ps: [u64; 64] = std::array::from_fn(|i| if i < 12 { edges[i] } else { next() });
    let qs: [u64; 64] = std::array::from_fn(|i| if i < 12 { edges[11 - i] } else { next() });
    let mut p_lanes = ps;
    let mut q_lanes = qs;
    transpose_64x64(&mut p_lanes);
    transpose_64x64(&mut q_lanes);
    let mut vals = vec![0u64; circ.nodes.len()];
    eval_lanes(&circ, &p_lanes, &q_lanes, &mut vals);
    let mut ok = true;
    for l in 0..64 {
        let want = (ps[l] as u128) * (qs[l] as u128);
        for k in 0..128 {
            let got = (vals[circ.product_bit_nodes[k] as usize] >> l) & 1;
            if got as u128 != ((want >> k) & 1) {
                ok = false;
            }
        }
    }
    (circ.sub_bits, circ.nnz, circ.muls_per_block, ok, circ.nodes.len())
}

pub fn circuit() -> &'static KMulCircuit {
    static CIRCUIT: OnceLock<KMulCircuit> = OnceLock::new();
    CIRCUIT.get_or_init(build_circuit)
}

pub fn muls_per_block() -> usize {
    circuit().muls_per_block
}

// ───────────────────────────────────────────────────────────────────────────
// R1CS
// ───────────────────────────────────────────────────────────────────────────

pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let circ = circuit();
    let mut a_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
    let mut b_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
    // Const-1 wire: identity row (pinned to 1 by the lincheck const_pin).
    a_rows[CONST_SLOT] = vec![CONST_SLOT];
    b_rows[CONST_SLOT] = vec![CONST_SLOT];
    for sub in 0..circ.muls_per_block {
        let off = HEADER_BITS + sub * circ.stride;
        for (r, (sa, ca, sb, cb)) in circ.rows.iter().enumerate() {
            let map = |supp: &[u32], cst: bool| -> Vec<usize> {
                let mut v: Vec<usize> = Vec::with_capacity(supp.len() + cst as usize);
                if cst {
                    v.push(CONST_SLOT);
                }
                v.extend(supp.iter().map(|&c| off + c as usize));
                v
            };
            a_rows[off + r] = map(sa, *ca);
            b_rows[off + r] = map(sb, *cb);
        }
    }
    (
        SparseBinaryMatrix {
            num_rows: K,
            num_cols: K,
            rows: a_rows,
        },
        SparseBinaryMatrix {
            num_rows: K,
            num_cols: K,
            rows: b_rows,
        },
    )
}

/// `BlockR1cs` for `2^n_blocks_log` blocks of [`muls_per_block`] products.
/// The const-1 wire (block slot 0) is pinned via `const_pin`; padding slots
/// and blocks carry (0,0) transcripts (NOT all-zero — complements).
pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    let circ = circuit();
    let (a_0, b_0) = build_matrices();
    super::common::build_block_r1cs_with_matrices(
        n_blocks_log,
        K_LOG,
        K_SKIP,
        circ.useful_bits,
        a_0,
        b_0,
        Some(CONST_SLOT),
    )
}

// ───────────────────────────────────────────────────────────────────────────
// Witness generation (bit-sliced, 64 blocks per lane group)
// ───────────────────────────────────────────────────────────────────────────

fn eval_lanes(circ: &KMulCircuit, p_lanes: &[u64; 64], q_lanes: &[u64; 64], vals: &mut [u64]) {
    vals[..64].copy_from_slice(p_lanes);
    vals[64..128].copy_from_slice(q_lanes);
    vals[CONST_ONE_NODE as usize] = !0u64;
    vals[CONST_ZERO_NODE as usize] = 0;
    for (id, node) in circ.nodes.iter().enumerate().skip(N_INPUT_NODES) {
        vals[id] = match *node {
            Node::Input => unreachable!("inputs are the first {N_INPUT_NODES} nodes"),
            Node::Xor(x, y) => vals[x as usize] ^ vals[y as usize],
            Node::And { a, b, .. } => vals[a as usize] & vals[b as usize],
        };
    }
}

/// Fused (z, A·z, B·z, lincheck-stripe) builder, RowMajor layout. Every
/// block and every product slot is materialized (padding products are (0,0)
/// — their transcripts are non-zero because of the complement constants, and
/// the const wire must be 1 in every block).
pub fn generate_witness_with_ab_packed_and_lincheck(
    pairs: &[(u64, u64)],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    use rayon::prelude::*;

    let circ = circuit();
    let mpb = circ.muls_per_block;
    let n_total = 1usize << n_blocks_log;
    assert!(pairs.len() <= n_total * mpb);
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "lincheck stripe layout requires n_total ≥ 8 and divisible by 8"
    );

    let f128_per_block = K / 128;
    let u64_per_block = K / 64;
    let sub_words = circ.sub_bits.div_ceil(64);
    let total_f128 = n_total * f128_per_block;

    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut stripe = vec![0u8; (n_total / 8) * K];

    let group_f128 = 64 * f128_per_block;
    let group_stripe = 8 * K;

    z.par_chunks_mut(group_f128)
        .zip(a.par_chunks_mut(group_f128))
        .zip(b.par_chunks_mut(group_f128))
        .zip(stripe.par_chunks_mut(group_stripe))
        .enumerate()
        .for_each_init(
            || vec![0u64; circ.nodes.len()],
            |vals, (g, (((z_grp, a_grp), b_grp), stripe_grp))| {
                // SAFETY: F128 = repr(C, align(16)) two LE u64s; zero bytes
                // are F128::ZERO (buffers come from the uninit scratch pool).
                unsafe {
                    std::ptr::write_bytes(z_grp.as_mut_ptr(), 0, z_grp.len());
                    std::ptr::write_bytes(a_grp.as_mut_ptr(), 0, a_grp.len());
                    std::ptr::write_bytes(b_grp.as_mut_ptr(), 0, b_grp.len());
                }
                let base_block = g * 64;
                let blocks_here = z_grp.len() / f128_per_block;

                let z_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(z_grp.as_mut_ptr() as *mut u64, z_grp.len() * 2)
                };
                let a_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(a_grp.as_mut_ptr() as *mut u64, a_grp.len() * 2)
                };
                let b_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(b_grp.as_mut_ptr() as *mut u64, b_grp.len() * 2)
                };

                // Const-1 wire (block bit 0): z = a = b = 1 in every block.
                for l in 0..blocks_here {
                    z_u64[l * u64_per_block] |= 1;
                    a_u64[l * u64_per_block] |= 1;
                    b_u64[l * u64_per_block] |= 1;
                }

                for sub in 0..mpb {
                    let mut p_lanes = [0u64; 64];
                    let mut q_lanes = [0u64; 64];
                    for l in 0..blocks_here {
                        let idx = (base_block + l) * mpb + sub;
                        if let Some(&(p, q)) = pairs.get(idx) {
                            p_lanes[l] = p;
                            q_lanes[l] = q;
                        }
                    }
                    transpose_64x64(&mut p_lanes);
                    transpose_64x64(&mut q_lanes);
                    eval_lanes(circ, &p_lanes, &q_lanes, vals);

                    let word_off = (HEADER_BITS + sub * circ.stride) / 64;
                    let mut zc = [0u64; 64];
                    let mut ac = [0u64; 64];
                    let mut bc = [0u64; 64];
                    for c in 0..sub_words {
                        let s0 = c * 64;
                        let n_slots = 64.min(circ.sub_bits.saturating_sub(s0));
                        for o in 0..n_slots {
                            let s = s0 + o;
                            zc[o] = vals[circ.slot_nodes[s] as usize];
                            let (an, bn) = circ.row_ab_nodes[s];
                            ac[o] = vals[an as usize];
                            bc[o] = vals[bn as usize];
                        }
                        for o in n_slots..64 {
                            zc[o] = 0;
                            ac[o] = 0;
                            bc[o] = 0;
                        }
                        transpose_64x64(&mut zc);
                        transpose_64x64(&mut ac);
                        transpose_64x64(&mut bc);
                        for l in 0..blocks_here {
                            z_u64[l * u64_per_block + word_off + c] = zc[l];
                            a_u64[l * u64_per_block + word_off + c] = ac[l];
                            b_u64[l * u64_per_block + word_off + c] = bc[l];
                        }
                    }
                }

                for sub8 in 0..blocks_here / 8 {
                    let out = &mut stripe_grp[sub8 * K..(sub8 + 1) * K];
                    for w in 0..u64_per_block {
                        let lanes: [u64; 8] =
                            std::array::from_fn(|s| z_u64[(sub8 * 8 + s) * u64_per_block + w]);
                        transpose_8_u64s_to_64_bytes(&lanes, &mut out[w * 64..w * 64 + 64]);
                    }
                }
            },
        );

    (z, a, b, stripe)
}

// ───────────────────────────────────────────────────────────────────────────
// Setup (mirrors mul64::Mul64Setup)
// ───────────────────────────────────────────────────────────────────────────

pub fn min_n_blocks_log(n_muls: usize) -> usize {
    assert!(n_muls >= 1);
    let blocks = n_muls
        .div_ceil(muls_per_block())
        .max(1 << MIN_N_BLOCKS_LOG);
    blocks.next_power_of_two().trailing_zeros() as usize
}

#[derive(Clone, Debug)]
pub struct Mul64KaratsubaSetup {
    pub n_muls: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: flock_core::pcs::PcsParams,
}

impl Mul64KaratsubaSetup {
    pub fn new(n_muls: usize) -> Self {
        Self::with_profile(n_muls, flock_core::pcs::ligerito::LigeritoProfile::Fast)
    }

    pub fn with_profile(
        n_muls: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        let n_log = min_n_blocks_log(n_muls);
        let r1cs = build_block_r1cs(n_log);
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let pcs_params = flock_core::pcs::PcsParams {
            m: r1cs.m,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: 6,
            profile,
            merkle_hash: Default::default(),
        };
        Self {
            n_muls,
            r1cs,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn capacity(&self) -> usize {
        muls_per_block() << self.n_blocks_log()
    }

    fn generate_witness_ab(
        &self,
        pairs: &[(u64, u64)],
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        generate_witness_with_ab_packed_and_lincheck(pairs, self.n_blocks_log())
    }

    pub fn generate_witness_packed(&self, pairs: &[(u64, u64)]) -> Vec<F128> {
        let (z, _a, _b, _stripe) = self.generate_witness_ab(pairs);
        z
    }

    pub fn prove_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        pairs: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        let z = self.generate_witness_packed(pairs);
        crate::prover::prove_ligerito(&self.r1cs, z, &self.pcs_params, challenger)
    }

    pub fn prove_fast<Ch: flock_core::challenger::Challenger>(
        &self,
        pairs: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        assert_eq!(pairs.len(), self.n_muls);
        let (codeword, (z, a, b, stripe)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                self.generate_witness_ab(pairs)
            });
        crate::prover::prove_fast_ligerito_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z,
            a,
            b,
            stripe,
            self.r1cs.csc_lincheck_circuit(),
            codeword,
            challenger,
        )
    }

    pub fn prove_fast_timed<Ch: flock_core::challenger::Challenger>(
        &self,
        pairs: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(pairs.len(), self.n_muls);
        let t0 = std::time::Instant::now();
        let (z, a, b, stripe) = self.generate_witness_ab(pairs);
        let witness_s = t0.elapsed().as_secs_f64();
        let (proof, commitment, claim, mut timings) = crate::prover::prove_fast_ligerito_timed(
            &self.r1cs,
            &self.pcs_params,
            z,
            a,
            b,
            stripe,
            self.r1cs.csc_lincheck_circuit(),
            None,
            challenger,
        );
        timings.witness_s = witness_s;
        (proof, commitment, claim, timings)
    }

    pub fn verify<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::VerifyError> {
        flock_core::verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn circuit_stats() {
        let c = circuit();
        println!(
            "mul64-karatsuba: sub_bits = {}, stride = {}, {}/block, useful = {} / {K} \
             ({:.2}%), nodes = {}, nnz = {} (vs schoolbook 8256 bits, 147340 nnz)",
            c.sub_bits,
            c.stride,
            c.muls_per_block,
            c.useful_bits,
            100.0 * c.useful_bits as f64 / K as f64,
            c.nodes.len(),
            c.nnz,
        );
        assert!(c.sub_bits < 8256, "Karatsuba should beat schoolbook");
        assert_eq!(c.slot_nodes.len(), c.sub_bits);
        assert_eq!(c.rows.len(), c.sub_bits);
    }

    #[test]
    fn product_bits_match_native() {
        let circ = circuit();
        let mut rng = Rng(0xDEADBEEF);
        let mut p_lanes = [0u64; 64];
        let mut q_lanes = [0u64; 64];
        let pairs: Vec<(u64, u64)> = (0..64)
            .map(|i| match i {
                0 => (0, 0),
                1 => (u64::MAX, u64::MAX),
                2 => (u64::MAX, 1),
                3 => (1 << 63, 2),
                4 => (0xFFFF_FFFF, 0xFFFF_FFFF),
                5 => (0x1_0000_0000, 0x1_0000_0000),
                6 => (u64::MAX, 0),
                _ => (rng.next_u64(), rng.next_u64()),
            })
            .collect();
        for (l, &(p, q)) in pairs.iter().enumerate() {
            p_lanes[l] = p;
            q_lanes[l] = q;
        }
        transpose_64x64(&mut p_lanes);
        transpose_64x64(&mut q_lanes);
        let mut vals = vec![0u64; circ.nodes.len()];
        eval_lanes(circ, &p_lanes, &q_lanes, &mut vals);
        for (l, &(p, q)) in pairs.iter().enumerate() {
            let want = (p as u128) * (q as u128);
            for k in 0..128 {
                let got = (vals[circ.product_bit_nodes[k] as usize] >> l) & 1;
                assert_eq!(
                    got,
                    ((want >> k) & 1) as u64,
                    "product bit {k} of {p:#x} * {q:#x}"
                );
            }
        }
    }

    /// Fused (z, a, b) vs matrices at a small non-Ligerito size, partial
    /// fill (exercises the (0,0) padding transcripts and the const wire).
    #[test]
    fn fused_ab_matches_matrices() {
        let n_log = 3;
        let r1cs = build_block_r1cs(n_log);
        let mut rng = Rng(0xC0FFEE);
        let n_muls = 5 * muls_per_block() + 7;
        let pairs: Vec<(u64, u64)> = (0..n_muls)
            .map(|_| (rng.next_u64(), rng.next_u64()))
            .collect();
        let (z, a, b, _stripe) = generate_witness_with_ab_packed_and_lincheck(&pairs, n_log);
        assert!(r1cs.satisfies_packed(&z), "R1CS unsatisfied");
        let a_ref = r1cs.apply_a_packed(&z);
        let b_ref = r1cs.apply_b_packed(&z);
        assert_eq!(a, a_ref, "fused A-side disagrees with A₀·z");
        assert_eq!(b, b_ref, "fused B-side disagrees with B₀·z");
    }

    #[test]
    fn prove_verify_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let n_muls = muls_per_block() << MIN_N_BLOCKS_LOG;
        let setup = Mul64KaratsubaSetup::new(n_muls);
        assert_eq!(setup.m(), 22);
        let mut rng = Rng(0x5EED);
        let pairs: Vec<(u64, u64)> = (0..n_muls)
            .map(|_| (rng.next_u64(), rng.next_u64()))
            .collect();
        let mut ch_p = FsChallenger::new(b"mul64k-test-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&pairs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"mul64k-test-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .expect("verification failed");
        assert_eq!(claim_p, claim_v);
    }
}
