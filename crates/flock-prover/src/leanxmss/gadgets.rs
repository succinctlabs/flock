//! The Boolean gadgets around the hashes, written in the circuit DSL and
//! wired through [`BooleanGate`].
//!
//! Every wired value is one 128-bit word, so a one-bit control (a Merkle
//! direction) travels as a word read at a fixed bit: a table's relation
//! cannot read a different bit position in each row.

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

use super::reference::{HEIGHT, TAG_CHAIN, TAG_LEAF, TAG_MSG, TAG_NODE, TARGET_SUM, V, W};

/// A wired word: 128 bits on a 128-bit boundary of the block.
fn wired<V: ColumnVisitor>(v: &mut V, name: &str, role: ColumnRole) -> [V::Value; 128] {
    v.word_aligned(name, role, 128)
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
// Codeword: the digit checks
// ---------------------------------------------------------------------------
//
// Every slot has the same 2^nu rows, so a table's address space is 2^nu times
// its block size, whatever its row count. Per-signature gadgets are therefore
// kept to small blocks.

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

/// Layer `layer`'s constant parts: the message digest, the chain step
/// leaving position 0 (the machine sets the step field for the others), the
/// leaf, then the nodes at levels 1..=16.
pub fn layer_address_constants(layer: usize) -> Vec<[u8; 16]> {
    let hi = |a: u32| a + ((layer as u32) << 23);
    let mut out = vec![
        address_constant(TAG_MSG, 0, hi(0)),
        address_constant(TAG_CHAIN, 0, hi(0)),
        address_constant(TAG_LEAF, 0, hi(0)),
    ];
    for level in 1..=HEIGHT {
        out.push(address_constant(TAG_NODE, 0, hi(level as u32)));
    }
    out
}

#[cfg(test)]
mod tests;
