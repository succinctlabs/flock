//! The Boolean gadgets around the hashes, written in the circuit DSL and
//! wired through [`BooleanGate`].
//!
//! Every wired value is one 128-bit word, so a one-bit control (a chain's
//! step selector, a Merkle direction) travels as a word read at a fixed bit:
//! a table's relation cannot read a different bit position in each row.

use flock_core::circuit::boolean::{
    BooleanCircuit, BooleanGate, BooleanGateError, CircuitBuilder, ColumnRole, ColumnSchema,
    ColumnVisitor, LinearExpr as Expr, Var,
};
use flock_field::F128;

fn u(v: F128) -> u128 {
    v.lo as u128 | (v.hi as u128) << 64
}

fn f(v: u128) -> F128 {
    F128::new(v as u64, (v >> 64) as u64)
}

use super::reference::{
    CHAIN_END, HEIGHT, TAG_CHAIN, TAG_LEAF, TAG_MSG, TAG_NODE, TARGET_SUM, V, W,
};

/// A wired word: 128 bits on a 128-bit boundary of the block.
fn wired<V: ColumnVisitor>(v: &mut V, name: &str, role: ColumnRole) -> [V::Value; 128] {
    v.word_aligned(name, role, 128)
}

// ---------------------------------------------------------------------------
// Select: one chain step
// ---------------------------------------------------------------------------

/// `y = sel[step] ? h : x`: chain value `x` advances to the hash `h` of the
/// step leaving position `step` exactly when that step is taken.
pub struct SelectCols<T> {
    pub sel: [T; 128],
    pub x: [T; 128],
    pub h: [T; 128],
    pub d: [T; 128],
    pub y: [T; 128],
}

struct SelectSchema;

impl ColumnSchema for SelectSchema {
    type Cols<T> = SelectCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> SelectCols<V::Value> {
        SelectCols {
            sel: wired(v, "sel", ColumnRole::Input),
            x: wired(v, "x", ColumnRole::Input),
            h: wired(v, "h", ColumnRole::Input),
            d: v.word("d", ColumnRole::Witness),
            y: wired(v, "y", ColumnRole::Output),
        }
    }
}

pub fn select_circuit(step: usize) -> BooleanCircuit {
    CircuitBuilder::compile(SelectSchema, |b, c| {
        for j in 0..128 {
            let diff = b.xor2(c.x[j], c.h[j]);
            b.define_and(c.d[j], c.sel[step], diff);
            let y = b.xor2(c.x[j], c.d[j]);
            b.define_linear(c.y[j], y);
        }
    })
    .circuit()
    .clone()
}

/// Inputs `[sel, x, h]`, output `[y]`.
pub fn select_gate(step: usize, nu: usize) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(
        &format!("leanxmss.select{step}"),
        &select_circuit(step),
        &["sel", "x", "h"],
        &["y"],
        nu,
    )
    .map(|g| {
        g.with_native(move |i, o| {
            o.push(if (u(i[0]) >> step) & 1 == 1 {
                i[2]
            } else {
                i[1]
            })
        })
    })
}

// ---------------------------------------------------------------------------
// Swap: one Merkle level
// ---------------------------------------------------------------------------

/// `(left, right) = dir[0] ? (sibling, node) : (node, sibling)`.
pub struct SwapCols<T> {
    pub dir: [T; 128],
    pub node: [T; 128],
    pub sibling: [T; 128],
    pub d: [T; 128],
    pub left: [T; 128],
    pub right: [T; 128],
}

struct SwapSchema;

impl ColumnSchema for SwapSchema {
    type Cols<T> = SwapCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> SwapCols<V::Value> {
        SwapCols {
            dir: wired(v, "dir", ColumnRole::Input),
            node: wired(v, "node", ColumnRole::Input),
            sibling: wired(v, "sibling", ColumnRole::Input),
            d: v.word("d", ColumnRole::Witness),
            left: wired(v, "left", ColumnRole::Output),
            right: wired(v, "right", ColumnRole::Output),
        }
    }
}

pub fn swap_circuit() -> BooleanCircuit {
    CircuitBuilder::compile(SwapSchema, |b, c| {
        for j in 0..128 {
            let diff = b.xor2(c.node[j], c.sibling[j]);
            b.define_and(c.d[j], c.dir[0], diff);
            let left = b.xor2(c.node[j], c.d[j]);
            b.define_linear(c.left[j], left);
            let right = b.xor2(c.sibling[j], c.d[j]);
            b.define_linear(c.right[j], right);
        }
    })
    .circuit()
    .clone()
}

/// Inputs `[dir, node, sibling]`, outputs `[left, right]`.
pub fn swap_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(
        "leanxmss.swap",
        &swap_circuit(),
        &["dir", "node", "sibling"],
        &["left", "right"],
        nu,
    )
    .map(|g| {
        g.with_native(|i, o| {
            if i[0].lo & 1 == 1 {
                o.extend([i[2], i[1]]);
            } else {
                o.extend([i[1], i[2]]);
            }
        })
    })
}

// ---------------------------------------------------------------------------
// Codeword: the digit checks, and each chain's step selectors
// ---------------------------------------------------------------------------
//
// Every slot has the same 2^nu rows, so a table's address space is 2^nu times
// its block size, whatever its row count. Per-signature gadgets are therefore
// kept to small blocks and chained through words: the check below has no
// outputs, and the selectors come from eight small spread steps per layer.

/// Bits of a running count of up to 63 ones.
const COUNT_BITS: usize = 6;

/// One running bit count: after input `i`, `sum[i]` is the count so far and
/// `carry[i]` the carries of adding that input.
pub struct CountCols<T> {
    pub sum: Vec<[T; COUNT_BITS]>,
    pub carry: Vec<[T; COUNT_BITS]>,
}

impl CountCols<Var> {
    fn columns<V: ColumnVisitor>(v: &mut V, p: &str, n: usize) -> CountCols<V::Value> {
        CountCols {
            sum: (0..n)
                .map(|i| v.word(&format!("{p}.sum-{i}"), ColumnRole::Witness))
                .collect(),
            carry: (0..n)
                .map(|i| v.word(&format!("{p}.carry-{i}"), ColumnRole::Witness))
                .collect(),
        }
    }

    /// The count of ones in `bits`, `COUNT_BITS` little-endian bits.
    fn eval(&self, b: &mut CircuitBuilder, bits: &[Expr]) -> [Expr; COUNT_BITS] {
        let mut acc: [Expr; COUNT_BITS] = [b.zero(); COUNT_BITS];
        for (i, &bit) in bits.iter().enumerate() {
            let mut carry = bit;
            for k in 0..COUNT_BITS {
                let s = b.xor2(acc[k], carry);
                b.define_linear(self.sum[i][k], s);
                b.define_and(self.carry[i][k], acc[k], carry);
                carry = self.carry[i][k].into();
            }
            acc = self.sum[i].map(Into::into);
        }
        acc
    }
}

/// Constrains a layer's digest `D` to be a codeword: digits 58..63 zero and
/// digits 0..57 summing to [`TARGET_SUM`]. No outputs.
pub struct CodewordCols<T> {
    pub digest: [T; 128],
    pub low: CountCols<T>,
    pub high: CountCols<T>,
    /// `low + 2·high`, an 8-bit ripple sum: majority products per bit.
    pub total: [T; 8],
}

struct CodewordSchema;

impl ColumnSchema for CodewordSchema {
    type Cols<T> = CodewordCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> CodewordCols<V::Value> {
        CodewordCols {
            digest: wired(v, "digest", ColumnRole::Input),
            low: CountCols::columns(v, "low", V),
            high: CountCols::columns(v, "high", V),
            total: v.word("total", ColumnRole::Witness),
        }
    }
}

pub fn codeword_circuit() -> BooleanCircuit {
    CircuitBuilder::compile(CodewordSchema, |b, c| {
        let one: Expr = b.one().into();
        for bit in W * V..128 {
            b.assert_zero(c.digest[bit]);
        }
        let x0: Vec<Expr> = (0..V).map(|i| c.digest[W * i].into()).collect();
        let x1: Vec<Expr> = (0..V).map(|i| c.digest[W * i + 1].into()).collect();
        // Σ x_i = (number of low bits set) + 2·(number of high bits set).
        let low = c.low.eval(b, &x0);
        let high = c.high.eval(b, &x1);
        let mut carry = b.zero();
        for k in 0..8 {
            let a = if k < COUNT_BITS { low[k] } else { b.zero() };
            let h = if (1..=COUNT_BITS).contains(&k) {
                high[k - 1]
            } else {
                b.zero()
            };
            let sum = b.xor3(a, h, carry);
            let target = if (TARGET_SUM >> k) & 1 == 1 {
                one
            } else {
                b.zero()
            };
            let miss = b.xor2(sum, target);
            b.assert_zero(miss);
            let (ac, hc) = (b.xor2(a, carry), b.xor2(h, carry));
            b.define_and(c.total[k], ac, hc);
            carry = b.xor2(c.total[k], carry);
        }
    })
    .circuit()
    .clone()
}

/// Input `[digest]`, no outputs.
pub fn codeword_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(
        "leanxmss.codeword",
        &codeword_circuit(),
        &["digest"],
        &[],
        nu,
    )
    .map(|g| {
        g.with_native(|i, _| {
            assert!(
                super::reference::codeword(&u(i[0]).to_le_bytes()).is_some(),
                "leanxmss.codeword: not a codeword"
            );
        })
    })
}

/// Digits handled by one spread step.
pub const SPREAD_DIGITS: usize = 8;
/// Spread steps per layer (the last covers digits 56..63).
pub const SPREADS: usize = V.div_ceil(SPREAD_DIGITS);

/// From `r = D >> 16j`, the selector words of chains `8j..8j+8` and
/// `r >> 16` for the next step. Selector `t`'s bit `µ` is set exactly when
/// the chain takes the step leaving position `µ`, i.e. its digit `x ≤ µ`;
/// bits 3..127 are zero.
pub struct SpreadCols<T> {
    pub r: [T; 128],
    pub both: [T; SPREAD_DIGITS],
    pub sel: Vec<[T; 128]>,
    pub next: [T; 128],
}

struct SpreadSchema;

pub fn spread_sel_name(t: usize) -> String {
    format!("sel-{t}")
}

impl ColumnSchema for SpreadSchema {
    type Cols<T> = SpreadCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> SpreadCols<V::Value> {
        SpreadCols {
            r: wired(v, "r", ColumnRole::Input),
            both: v.word("both", ColumnRole::Witness),
            sel: (0..SPREAD_DIGITS)
                .map(|t| wired(v, &spread_sel_name(t), ColumnRole::Output))
                .collect(),
            next: wired(v, "next", ColumnRole::Output),
        }
    }
}

pub fn spread_circuit() -> BooleanCircuit {
    CircuitBuilder::compile(SpreadSchema, |b, c| {
        let one: Expr = b.one().into();
        // The digit-is-3 products first: placement follows definition order,
        // and a loose bit between two aligned words costs a word of padding.
        for t in 0..SPREAD_DIGITS {
            b.define_and(c.both[t], c.r[W * t], c.r[W * t + 1]);
        }
        for t in 0..SPREAD_DIGITS {
            let (x0, x1) = (c.r[W * t], c.r[W * t + 1]);
            let (n0, n1) = (b.xor2(one, x0), b.xor2(one, x1));
            b.define_and(c.sel[t][0], n0, n1); // x = 0
            b.define_linear(c.sel[t][1], n1); // x ≤ 1
            let not_three = b.xor2(one, c.both[t]);
            b.define_linear(c.sel[t][2], not_three); // x ≤ 2
            for bit in CHAIN_END..128 {
                b.define_linear(c.sel[t][bit], b.zero());
            }
        }
        let shift = W * SPREAD_DIGITS;
        for j in 0..128 {
            let v = if j + shift < 128 {
                c.r[j + shift].into()
            } else {
                b.zero()
            };
            b.define_linear(c.next[j], v);
        }
    })
    .circuit()
    .clone()
}

/// Input `[r]`, outputs `[sel_0, …, sel_7, next]`.
pub fn spread_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    let names: Vec<String> = (0..SPREAD_DIGITS).map(spread_sel_name).collect();
    let outputs: Vec<&str> = names.iter().map(String::as_str).chain(["next"]).collect();
    BooleanGate::new("leanxmss.spread", &spread_circuit(), &["r"], &outputs, nu).map(|g| {
        g.with_native(|i, o| {
            let r = u(i[0]);
            for t in 0..SPREAD_DIGITS {
                let x = (r >> (W * t)) & 3;
                o.push(f((x == 0) as u128
                    | ((x <= 1) as u128) << 1
                    | ((x <= 2) as u128) << 2));
            }
            o.push(f(r >> (W * SPREAD_DIGITS)));
        })
    })
}

// ---------------------------------------------------------------------------
// Addresses: a shift chain over the leaf index
// ---------------------------------------------------------------------------
//
// Every address is `LE32(lo) ‖ C` where `C` (bits 32..63: `hi`, the tag and
// the step) is a per-position constant and `lo` is `idx >> k` for some k.
// A chain of shift steps carries `R_k = idx >> k` (31 bits after the first
// step); bit 0 of `R_k` is also the Merkle direction at that level, and
// `R_16 = idx >> 16` is the top layer's key index.

/// `a = lo32(e) ⊕ c`: an address from its key index and constant part.
pub struct AddressCols<T> {
    pub e: [T; 128],
    pub c: [T; 128],
    pub a: [T; 128],
}

struct AddressSchema;

impl ColumnSchema for AddressSchema {
    type Cols<T> = AddressCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> AddressCols<V::Value> {
        AddressCols {
            e: wired(v, "e", ColumnRole::Input),
            c: wired(v, "c", ColumnRole::Input),
            a: wired(v, "a", ColumnRole::Output),
        }
    }
}

pub fn address_circuit() -> BooleanCircuit {
    CircuitBuilder::compile(AddressSchema, |b, c| {
        for j in 0..128 {
            let v = if j < 32 {
                b.xor2(c.e[j], c.c[j])
            } else {
                c.c[j].into()
            };
            b.define_linear(c.a[j], v);
        }
    })
    .circuit()
    .clone()
}

/// Inputs `[e, c]`, output `[a]`.
pub fn address_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(
        "leanxmss.address",
        &address_circuit(),
        &["e", "c"],
        &["a"],
        nu,
    )
    .map(|g| g.with_native(|i, o| o.push(f((u(i[0]) & 0xFFFF_FFFF) ^ u(i[1])))))
}

/// `next = (r >> 1) mod 2^31` and `a = next ⊕ c`: one level up the tree.
pub struct AddressStepCols<T> {
    pub r: [T; 128],
    pub c: [T; 128],
    pub next: [T; 128],
    pub a: [T; 128],
}

struct AddressStepSchema;

impl ColumnSchema for AddressStepSchema {
    type Cols<T> = AddressStepCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> AddressStepCols<V::Value> {
        AddressStepCols {
            r: wired(v, "r", ColumnRole::Input),
            c: wired(v, "c", ColumnRole::Input),
            next: wired(v, "next", ColumnRole::Output),
            a: wired(v, "a", ColumnRole::Output),
        }
    }
}

pub fn address_step_circuit() -> BooleanCircuit {
    CircuitBuilder::compile(AddressStepSchema, |b, c| {
        for j in 0..128 {
            let v = if j < 31 { c.r[j + 1].into() } else { b.zero() };
            b.define_linear(c.next[j], v);
        }
        for j in 0..128 {
            let v = if j < 32 {
                b.xor2(c.next[j], c.c[j])
            } else {
                c.c[j].into()
            };
            b.define_linear(c.a[j], v);
        }
    })
    .circuit()
    .clone()
}

/// Inputs `[r, c]`, outputs `[next, a]`.
pub fn address_step_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(
        "leanxmss.address-step",
        &address_step_circuit(),
        &["r", "c"],
        &["next", "a"],
        nu,
    )
    .map(|g| {
        g.with_native(|i, o| {
            let next = (u(i[0]) >> 1) & 0x7FFF_FFFF;
            o.extend([f(next), f(next ^ u(i[1]))]);
        })
    })
}

/// The constant part of an address: `enc(tag, step, hi, 0)`.
pub fn address_constant(tag: u8, step: u8, hi: u32) -> [u8; 16] {
    let mut a = [0u8; 16];
    a[4..7].copy_from_slice(&hi.to_le_bytes()[..3]);
    a[7] = tag | step << 5;
    a
}

/// Layer `layer`'s constant parts: the message digest, the three chain
/// steps, the leaf, then the nodes at levels 1..=16.
pub fn layer_address_constants(layer: usize) -> Vec<[u8; 16]> {
    let hi = |a: u32| a + ((layer as u32) << 23);
    let mut out = vec![address_constant(TAG_MSG, 0, hi(0))];
    for step in 0..CHAIN_END {
        out.push(address_constant(TAG_CHAIN, step as u8, hi(0)));
    }
    out.push(address_constant(TAG_LEAF, 0, hi(0)));
    for level in 1..=HEIGHT {
        out.push(address_constant(TAG_NODE, 0, hi(level as u32)));
    }
    out
}

// ---------------------------------------------------------------------------
// The routing network: Beneš blocks and the digit lookup
// ---------------------------------------------------------------------------

/// Bits of a chain index (positions `0..64`).
pub const INDEX_BITS: usize = 6;

/// One block of the routing network ([`super::routing`]): eight positions
/// through three stages, local stage `s` swapping the pairs differing in
/// local bit `s` where control bit `4s + q` says so. The forward block
/// carries each position's value and index word (only the index's six bits
/// switch; the rest are zero); the reverse block carries values only and
/// applies the stages in reverse order, which undoes the forward block with
/// the same control word.
pub struct BenesCols<T> {
    pub ctrl: [T; 128],
    pub v_in: Vec<[T; 128]>,
    pub m_in: Vec<[T; 128]>,
    pub dv: Vec<[T; 128]>,
    pub dm: Vec<[T; INDEX_BITS]>,
    pub v_out: Vec<[T; 128]>,
    pub m_out: Vec<[T; 128]>,
}

struct BenesSchema {
    reverse: bool,
}

pub fn benes_in_names(reverse: bool) -> Vec<String> {
    let mut names = vec!["ctrl".to_string()];
    names.extend((0..8).map(|e| format!("v-in-{e}")));
    if !reverse {
        names.extend((0..8).map(|e| format!("m-in-{e}")));
    }
    names
}

pub fn benes_out_names(reverse: bool) -> Vec<String> {
    let mut names: Vec<String> = (0..8).map(|e| format!("v-out-{e}")).collect();
    if !reverse {
        names.extend((0..8).map(|e| format!("m-out-{e}")));
    }
    names
}

impl ColumnSchema for BenesSchema {
    type Cols<T> = BenesCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> BenesCols<V::Value> {
        let fwd = !self.reverse;
        let words = |v: &mut V, p: &str, n: usize, role: ColumnRole| -> Vec<[V::Value; 128]> {
            (0..n)
                .map(|e| wired(v, &format!("{p}-{e}"), role.clone()))
                .collect()
        };
        BenesCols {
            ctrl: wired(v, "ctrl", ColumnRole::Input),
            v_in: words(v, "v-in", 8, ColumnRole::Input),
            m_in: words(v, "m-in", if fwd { 8 } else { 0 }, ColumnRole::Input),
            dv: (0..12)
                .map(|t| v.word(&format!("dv-{t}"), ColumnRole::Witness))
                .collect(),
            dm: (0..if fwd { 12 } else { 0 })
                .map(|t| v.word(&format!("dm-{t}"), ColumnRole::Witness))
                .collect(),
            v_out: words(v, "v-out", 8, ColumnRole::Output),
            m_out: words(v, "m-out", if fwd { 8 } else { 0 }, ColumnRole::Output),
        }
    }
}

pub fn benes_circuit(reverse: bool) -> BooleanCircuit {
    CircuitBuilder::compile(BenesSchema { reverse }, |b, c| {
        let mut v: Vec<Vec<Expr>> = c.v_in.iter().map(|w| w.map(Into::into).to_vec()).collect();
        let mut m: Vec<Vec<Expr>> = c.m_in.iter().map(|w| w.map(Into::into).to_vec()).collect();
        let stages: Vec<usize> = if reverse {
            vec![2, 1, 0]
        } else {
            vec![0, 1, 2]
        };
        for s in stages {
            for q in 0..4 {
                let lo = super::routing::pair(s, q);
                let hi = lo | 1 << s;
                let t = 4 * s + q;
                let ctrl = c.ctrl[t];
                for j in 0..128 {
                    let diff = b.xor2(v[lo][j], v[hi][j]);
                    b.define_and(c.dv[t][j], ctrl, diff);
                    v[lo][j] = b.xor2(v[lo][j], c.dv[t][j]);
                    v[hi][j] = b.xor2(v[hi][j], c.dv[t][j]);
                }
                if !reverse {
                    for j in 0..INDEX_BITS {
                        let diff = b.xor2(m[lo][j], m[hi][j]);
                        b.define_and(c.dm[t][j], ctrl, diff);
                        m[lo][j] = b.xor2(m[lo][j], c.dm[t][j]);
                        m[hi][j] = b.xor2(m[hi][j], c.dm[t][j]);
                    }
                }
            }
        }
        for (out, val) in c.v_out.iter().zip(&v) {
            for j in 0..128 {
                b.define_linear(out[j], val[j]);
            }
        }
        for (out, val) in c.m_out.iter().zip(&m) {
            for j in 0..128 {
                let bit = if j < INDEX_BITS { val[j] } else { b.zero() };
                b.define_linear(out[j], bit);
            }
        }
    })
    .circuit()
    .clone()
}

/// Forward: inputs `[ctrl, v_0..v_7, m_0..m_7]`, outputs `[v_0..v_7, m_0..m_7]`.
/// Reverse: inputs `[ctrl, v_0..v_7]`, outputs `[v_0..v_7]`.
pub fn benes_gate(reverse: bool, nu: usize) -> Result<BooleanGate, BooleanGateError> {
    let ins = benes_in_names(reverse);
    let outs = benes_out_names(reverse);
    let ins: Vec<&str> = ins.iter().map(String::as_str).collect();
    let outs: Vec<&str> = outs.iter().map(String::as_str).collect();
    let name = if reverse {
        "leanxmss.benes-reverse"
    } else {
        "leanxmss.benes"
    };
    BooleanGate::new(name, &benes_circuit(reverse), &ins, &outs, nu).map(|g| {
        g.with_native(move |i, o| {
            let ctrl = i[0].lo;
            let mut items: [(F128, u64); 8] =
                std::array::from_fn(|e| (i[1 + e], if reverse { 0 } else { i[9 + e].lo & 63 }));
            let stages: [usize; 3] = if reverse { [2, 1, 0] } else { [0, 1, 2] };
            for s in stages {
                for q in 0..4 {
                    if (ctrl >> (4 * s + q)) & 1 == 1 {
                        let lo = super::routing::pair(s, q);
                        items.swap(lo, lo | 1 << s);
                    }
                }
            }
            o.extend(items.iter().map(|&(v, _)| v));
            if !reverse {
                o.extend(items.iter().map(|&(_, m)| F128::new(m, 0)));
            }
        })
    })
}

/// The digit of the chain whose index word is `index`, read from the layer's
/// digest `D` (indices 58..63 are the network's dummies and read as digit
/// 3), as the selector word `sel` (bit 0: digit 0; bit 1: digit ≤ 1). Bit 0
/// of `class` asserts the digit is not 0, bit 1 that it is at least 2.
pub struct LookupCols<T> {
    pub digest: [T; 128],
    pub index: [T; 128],
    pub class: [T; 128],
    /// The multiplexer tree's products, two bits per node.
    pub mux: Vec<T>,
    pub sel: [T; 128],
}

struct LookupSchema;

impl ColumnSchema for LookupSchema {
    type Cols<T> = LookupCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> LookupCols<V::Value> {
        LookupCols {
            digest: wired(v, "digest", ColumnRole::Input),
            index: wired(v, "index", ColumnRole::Input),
            class: wired(v, "class", ColumnRole::Input),
            mux: v.bits(
                "mux",
                ColumnRole::Witness,
                2 * (super::routing::WIDTH - 1),
                1,
            ),
            sel: wired(v, "sel", ColumnRole::Output),
        }
    }
}

pub fn lookup_circuit() -> BooleanCircuit {
    CircuitBuilder::compile(LookupSchema, |b, c| {
        let one: Expr = b.one().into();
        // Two bits per entry: digit i of D, or 3 for a dummy.
        let mut entries: Vec<[Expr; 2]> = (0..super::routing::WIDTH)
            .map(|i| {
                if i < V {
                    [c.digest[W * i].into(), c.digest[W * i + 1].into()]
                } else {
                    [one, one]
                }
            })
            .collect();
        let mut mux = c.mux.iter();
        for level in 0..INDEX_BITS {
            let sel = c.index[level];
            entries = entries
                .as_chunks::<2>()
                .0
                .iter()
                .map(|[e0, e1]| {
                    std::array::from_fn(|bit| {
                        let product = *mux.next().unwrap();
                        let diff = b.xor2(e0[bit], e1[bit]);
                        b.define_and(product, sel, diff);
                        b.xor2(e0[bit], product)
                    })
                })
                .collect();
        }
        let [x0, x1] = entries[0];
        let (n0, n1) = (b.xor2(one, x0), b.xor2(one, x1));
        b.define_and(c.sel[0], n0, n1);
        b.define_linear(c.sel[1], n1);
        for bit in 2..128 {
            b.define_linear(c.sel[bit], b.zero());
        }
        b.assert_zero_product(c.class[0], c.sel[0]);
        b.assert_zero_product(c.class[1], c.sel[1]);
    })
    .circuit()
    .clone()
}

/// Inputs `[digest, index, class]`, output `[sel]`.
pub fn lookup_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(
        "leanxmss.lookup",
        &lookup_circuit(),
        &["digest", "index", "class"],
        &["sel"],
        nu,
    )
    .map(|g| {
        g.with_native(|i, o| {
            let index = (i[1].lo & 63) as usize;
            let digit = if index < V {
                (u(i[0]) >> (W * index)) & 3
            } else {
                3
            };
            let class = i[2].lo;
            assert!(
                !(class & 1 == 1 && digit == 0 || class & 2 == 2 && digit < 2),
                "leanxmss.lookup: digit {digit} at index {index} breaks class {class}"
            );
            o.push(f((digit == 0) as u128 | ((digit <= 1) as u128) << 1));
        })
    })
}

#[cfg(test)]
mod tests;
