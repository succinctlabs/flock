//! The chain machine: one layer's Winternitz chains and one-time-key leaf as
//! a sequence of identical rows, each one BLAKE3 compression.
//!
//! A codeword's digits sum to [`TARGET_SUM`], so a layer always takes
//! `3·58 − 106 = 68` chain steps; its leaf, `H(P ‖ A_leaf ‖ end_0 ‖ … ‖
//! end_57)`, is 15 more compressions. The machine walks the chains in order
//! and absorbs their ends into the leaf as they finish, so a layer is exactly
//! [`ROWS`] = 83 rows, whatever the digits. Which rows step and which absorb
//! depends on the digits; the wiring does not: row `r`'s state outputs feed
//! row `r + 1`'s state inputs.
//!
//! A row is a STEP (`Th(P, A(µ), v ‖ LE128(c))` of chain `c` at position `µ`)
//! or a LEAF (one block of the leaf hash). Its state, carried in seven words:
//!
//! - `v`: the chain value about to be stepped, or the end of the chain in
//!   leaf slot 3 while a LEAF is pending;
//! - `b0..b2`: leaf slots 0..2 of the block being filled (chain `c` lands in
//!   slot `(c + 2) mod 4`: block 0 starts with `P ‖ A_leaf`);
//! - `h0, h1`: the leaf's chaining value;
//! - `ctl`: the digits of the chains not yet started (bits `0..116`, shifted
//!   down as chains are consumed), then `p` (the current chain, 6 bits), `µ`
//!   (2), `active` and `pending` (a full block awaits its LEAF).
//!
//! Before hashing, a row with no active chain and no pending block skips the
//! chains with digit 3 (they take no step; their end is their revealed
//! value) and either starts the next chain at its digit or, if the skipped
//! chains fill the block, becomes a LEAF. After a STEP, a chain reaching
//! position 3 writes its end into its slot (slot 3 stays in `v`).
//!
//! A row whose `ctl` is a [`plain_ctl`] constant is a plain compression
//! instead: `cv = (h0, h1)`, the block `b0 ‖ b1 ‖ b2 ‖ v`, the flags and block
//! length from `ctl`. Every BLAKE3 call of a signature (digests, Merkle nodes)
//! runs on these rows, so one slot kind holds all of them.
//!
//! The prover chooses only values, never control: `y` (a starting chain's
//! value, or a skipped chain's end in slot 3) and `f0..f2` (after a LEAF, the
//! next block's slots, which matter only for skipped chains). The initial
//! `ctl` is the digest itself (its bits 116..127 are zero for a codeword), and
//! the last row's `ctl` must be [`final_ctl`]: all 58 chains consumed and no
//! block pending, which also fixes the digit sum.

use flock_core::circuit::boolean::{
    BooleanCircuit, BooleanGate, BooleanGateError, CircuitBuilder, ColumnRole, ColumnSchema,
    ColumnVisitor, LinearExpr as Expr, SchemaOperation, Var,
};
use flock_field::F128;

use super::reference::{CHAIN_END, TARGET_SUM, V, Value, W};
use crate::r1cs_hashes::{
    blake3::BLAKE3_IV,
    blake3_dsl::{GCols, compress_op, round_columns},
    dsl::{Word, boolean_gate},
};

/// Chain steps per layer: every codeword takes exactly this many.
pub const STEPS: usize = V * CHAIN_END - TARGET_SUM;
/// Leaf blocks: `P ‖ A_leaf` and the 58 ends, four 16-byte words a block.
pub const LEAF_BLOCKS: usize = (2 + V).div_ceil(4);
/// Rows per layer.
pub const ROWS: usize = STEPS + LEAF_BLOCKS;

/// Bits of `ctl` holding digits.
const DIGIT_BITS: usize = W * V;
/// `ctl` fields after the digits.
const P_BASE: usize = DIGIT_BITS;
const P_BITS: usize = 6;
const MU_BASE: usize = P_BASE + P_BITS;
const ACTIVE: usize = MU_BASE + 2;
const PENDING: usize = ACTIVE + 1;
/// A plain compression's `CHUNK_START` and `CHUNK_END | ROOT`: zero on every
/// machine row.
const PLAIN_START: usize = 126;
const PLAIN_END: usize = 127;

// The leaf flags below read block 14 as p = 58 = 0b111010.
const _: () = assert!(PENDING < PLAIN_START && V == 58 && (2 + V).is_multiple_of(4));

/// Leaf blocks are compressed with these flags.
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const ROOT: u32 = 8;

/// The state words, in order: the first inputs, and the outputs (primed).
const STATE: [&str; 7] = ["v", "b0", "b1", "b2", "h0", "h1", "ctl"];
/// Inputs after the state: `P`, the layer's chain address at step 0, and the
/// prover's values.
const EXTRA: [&str; 6] = ["pp", "a0", "f0", "f1", "f2", "y"];

fn out_name(state: &str) -> String {
    format!("{state}'")
}

/// The `ctl` word every layer must end on.
pub fn final_ctl() -> F128 {
    F128::new(0, (V as u64) << (P_BASE - 64))
}

/// A row's columns: the wired input words, the transition's products, the
/// compression's G blocks, and the wired output words.
pub struct MachineCols<T> {
    pub state: Vec<[T; 128]>,
    pub extra: Vec<[T; 128]>,
    pub rounds: Vec<[GCols<T>; 8]>,
    pub settle: T,
    pub three: [T; 4],
    pub run: [T; 3],
    pub slot3: T,
    pub fill: [T; 4],
    pub leaf_at: [T; 4],
    pub start_at: [T; 4],
    pub shift: Vec<[T; DIGIT_BITS]>,
    pub p_carry: [T; P_BITS - 1],
    pub mu_start: [T; 2 * 4],
    pub at_two: T,
    pub end: T,
    pub mu_next: [T; 2],
    pub end_slot3: T,
    pub end_slot: [T; 3],
    pub leaf_slot3: T,
    pub p_inc: [T; P_BITS - 1],
    pub leaf_low: T,
    pub first: [T; 3],
    pub last: [T; 3],
    pub short: T,
    pub keep: [T; 128],
    pub cv: [T; 256],
    pub m: Vec<[T; 128]>,
    pub out_lo: [T; 256],
    pub h: [T; 256],
    pub reset: Vec<[T; 128]>,
    pub write: Vec<[T; 128]>,
    pub out: Vec<[T; 128]>,
}

struct MachineSchema;

impl ColumnSchema for MachineSchema {
    type Cols<T> = MachineCols<T>;

    fn columns<Vis: ColumnVisitor>(&self, v: &mut Vis) -> MachineCols<Vis::Value> {
        let wired = |v: &mut Vis, name: &str, role: ColumnRole| v.word_aligned(name, role, 128);
        let state = STATE
            .iter()
            .map(|s| wired(v, s, ColumnRole::Input))
            .collect();
        let extra = EXTRA
            .iter()
            .map(|s| wired(v, s, ColumnRole::Input))
            .collect();
        MachineCols {
            state,
            extra,
            settle: v.bit("settle", ColumnRole::Witness),
            three: v.word("three", ColumnRole::Witness),
            run: v.word("run", ColumnRole::Witness),
            slot3: v.bit("slot3", ColumnRole::Witness),
            fill: v.word("fill", ColumnRole::Witness),
            leaf_at: v.word("leaf-at", ColumnRole::Witness),
            start_at: v.word("start-at", ColumnRole::Witness),
            shift: (0..3)
                .map(|t| v.word(&format!("shift-{t}"), ColumnRole::Witness))
                .collect(),
            p_carry: v.word("p-carry", ColumnRole::Witness),
            mu_start: v.word("mu-start", ColumnRole::Witness),
            at_two: v.bit("at-two", ColumnRole::Witness),
            end: v.bit("end", ColumnRole::Witness),
            mu_next: v.word("mu-next", ColumnRole::Witness),
            end_slot3: v.bit("end-slot3", ColumnRole::Witness),
            end_slot: v.word("end-slot", ColumnRole::Witness),
            leaf_slot3: v.bit("leaf-slot3", ColumnRole::Witness),
            p_inc: v.word("p-inc", ColumnRole::Witness),
            leaf_low: v.bit("leaf-low", ColumnRole::Witness),
            first: v.word("first", ColumnRole::Witness),
            last: v.word("last", ColumnRole::Witness),
            short: v.bit("short", ColumnRole::Witness),
            keep: v.word("keep", ColumnRole::Witness),
            cv: v.word("cv", ColumnRole::Witness),
            m: (0..4)
                .map(|j| v.word(&format!("m-{j}"), ColumnRole::Witness))
                .collect(),
            rounds: round_columns(v, "rounds"),
            out_lo: v.word("out-lo", ColumnRole::Witness),
            h: v.word("h", ColumnRole::Witness),
            reset: (0..3)
                .map(|j| v.word(&format!("reset-{j}"), ColumnRole::Witness))
                .collect(),
            write: (0..3)
                .map(|j| v.word(&format!("write-{j}"), ColumnRole::Witness))
                .collect(),
            out: STATE
                .iter()
                .map(|s| wired(v, &out_name(s), ColumnRole::Output))
                .collect(),
        }
    }
}

/// `x·y` stored in `col`, as an expression.
fn and(b: &mut CircuitBuilder, col: Var, x: impl Into<Expr>, y: impl Into<Expr>) -> Expr {
    b.define_and(col, x, y);
    col.into()
}

/// The XOR of `terms`.
fn sum(b: &mut CircuitBuilder, terms: &[Expr]) -> Expr {
    b.xor(terms.iter().copied())
}

/// A 128-bit word as four little-endian `u32` words.
fn split(word: &[Expr; 128]) -> [Word; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| word[32 * i + j]))
}

/// A constant 128-bit word.
fn constant(b: &CircuitBuilder, v: u128) -> [Expr; 128] {
    std::array::from_fn(|j| {
        if (v >> j) & 1 == 1 {
            b.one().into()
        } else {
            b.zero()
        }
    })
}

/// `IV[4h..4h+4]` as one word.
fn iv_word(h: usize) -> u128 {
    (0..4).fold(0, |w, i| w | (BLAKE3_IV[4 * h + i] as u128) << (32 * i))
}

/// `select ? x : y` per bit, the product stored in `cols`.
fn mux<const N: usize>(
    b: &mut CircuitBuilder,
    cols: &[Var; N],
    select: Expr,
    x: &[Expr; N],
    y: &[Expr; N],
) -> [Expr; N] {
    std::array::from_fn(|j| {
        let diff = b.xor2(x[j], y[j]);
        let d = and(b, cols[j], select, diff);
        b.xor2(y[j], d)
    })
}

/// The row's circuit and its operations.
pub fn machine_circuit() -> (BooleanCircuit, Vec<SchemaOperation>) {
    let compiled = CircuitBuilder::compile(MachineSchema, |b, c| {
        let one: Expr = b.one().into();
        let zero = b.zero();
        let word = |w: &[Var; 128]| -> [Expr; 128] { w.map(Into::into) };
        let [v, b0, b1, b2, h0, h1, ctl] = std::array::from_fn(|i| word(&c.state[i]));
        let [pp, a0, f0, f1, f2, y] = std::array::from_fn(|i| word(&c.extra[i]));
        let not = |b: &mut CircuitBuilder, x: Expr| b.xor2(one, x);

        // Control.
        let p: [Expr; P_BITS] = std::array::from_fn(|j| ctl[P_BASE + j]);
        let mu = [ctl[MU_BASE], ctl[MU_BASE + 1]];
        let (active, pending) = (ctl[ACTIVE], ctl[PENDING]);
        let (na, np) = (not(b, active), not(b, pending));
        let settle = and(b, c.settle, na, np);
        // t[k]: the next k chains all have digit 3.
        let three: [Expr; 4] =
            std::array::from_fn(|k| and(b, c.three[k], ctl[W * k], ctl[W * k + 1]));
        let mut t = [one; 5];
        t[1] = three[0];
        for k in 2..=4 {
            t[k] = and(b, c.run[k - 2], t[k - 1], three[k - 1]);
        }
        // The current chain's slot s = (p + 2) mod 4, one-hot.
        let (s0, s1) = (p[0], not(b, p[1]));
        let sigma3 = and(b, c.slot3, s0, s1);
        let sigma = [
            sum(b, &[one, s0, s1, sigma3]),
            b.xor2(s0, sigma3),
            b.xor2(s1, sigma3),
            sigma3,
        ];
        // leaf_at[k]: settling skips k = 4 − s chains with digit 3, which
        // fill the block, so this row is a LEAF.
        let mut leaf_at = [zero; 5];
        for k in 1..=4 {
            let fill = and(b, c.fill[k - 1], sigma[4 - k], t[k]);
            leaf_at[k] = and(b, c.leaf_at[k - 1], settle, fill);
        }
        let leaf_now = sum(b, &leaf_at[1..]);
        let leaf = b.xor2(pending, leaf_now);
        let start = b.xor2(settle, leaf_now);
        let step = b.xor2(active, start);
        // start_at[k]: settling skips k chains and starts the next.
        let start_at: [Expr; 4] = std::array::from_fn(|k| {
            let exactly = b.xor2(t[k], t[k + 1]);
            and(b, c.start_at[k], start, exactly)
        });
        // Chains consumed by settling, m (shifts the digits), and chains
        // skipped before this row's chain or block, k (advances p).
        let consumed: [Expr; 5] = std::array::from_fn(|m| match m {
            0 => zero,
            _ => b.xor2(start_at[m - 1], leaf_at[m]),
        });
        let m_bits = [
            b.xor2(consumed[1], consumed[3]),
            b.xor2(consumed[2], consumed[3]),
            consumed[4],
        ];
        let skipped: [Expr; 5] = std::array::from_fn(|k| match k {
            0 => start_at[0],
            4 => leaf_at[4],
            _ => b.xor2(start_at[k], leaf_at[k]),
        });
        let k_bits = [
            b.xor2(skipped[1], skipped[3]),
            b.xor2(skipped[2], skipped[3]),
            skipped[4],
        ];
        // p_pre = p + k.
        let mut p_pre = [zero; P_BITS];
        let mut carry = zero;
        for j in 0..P_BITS {
            let kj = if j < 3 { k_bits[j] } else { zero };
            p_pre[j] = b.xor3(p[j], kj, carry);
            if j + 1 < P_BITS {
                let lhs = b.xor2(p[j], carry);
                let rhs = b.xor2(kj, carry);
                let prod = and(b, c.p_carry[j], lhs, rhs);
                carry = b.xor2(prod, carry);
            }
        }
        // The started chain's digit, and the position this row steps from.
        let mut mu_start = [zero; 2];
        for (k, &sel) in start_at.iter().enumerate() {
            for bit in 0..2 {
                let prod = and(b, c.mu_start[2 * k + bit], sel, ctl[W * k + bit]);
                mu_start[bit] = b.xor2(mu_start[bit], prod);
            }
        }
        let mu_eff = [b.xor2(mu[0], mu_start[0]), b.xor2(mu[1], mu_start[1])];
        // After a STEP: does the chain reach position 3?
        let nm0 = not(b, mu_eff[0]);
        let at_two = and(b, c.at_two, mu_eff[1], nm0);
        let end = and(b, c.end, step, at_two);
        let cont = b.xor2(step, end);
        let mu_inc = b.xor2(mu_eff[1], mu_eff[0]);
        let mu_next = [
            and(b, c.mu_next[0], cont, nm0),
            and(b, c.mu_next[1], cont, mu_inc),
        ];
        // The ending chain's slot, (p_pre + 2) mod 4.
        let (q0, q1) = (p_pre[0], not(b, p_pre[1]));
        let tau3 = and(b, c.end_slot3, q0, q1);
        let tau = [
            sum(b, &[one, q0, q1, tau3]),
            b.xor2(q0, tau3),
            b.xor2(q1, tau3),
        ];
        let end_slot: [Expr; 3] = std::array::from_fn(|j| and(b, c.end_slot[j], end, tau[j]));
        let pending_next = and(b, c.leaf_slot3, end, tau3);
        // p' = p_pre + end.
        let mut p_next = [zero; P_BITS];
        let mut inc = end;
        for j in 0..P_BITS {
            p_next[j] = b.xor2(p_pre[j], inc);
            if j + 1 < P_BITS {
                inc = and(b, c.p_inc[j], p_pre[j], inc);
            }
        }
        // The digits of the chains not yet started.
        let mut digits: [Expr; DIGIT_BITS] = std::array::from_fn(|j| ctl[j]);
        for (stage, &bit) in m_bits.iter().enumerate() {
            let by = W << stage;
            let shifted: [Expr; DIGIT_BITS] = std::array::from_fn(|j| {
                if j + by < DIGIT_BITS {
                    digits[j + by]
                } else {
                    zero
                }
            });
            digits = mux(b, &c.shift[stage], bit, &shifted, &digits);
        }
        // A LEAF's flags: CHUNK_START on block 0 (p_pre = 2), CHUNK_END |
        // ROOT on block 14 (p_pre = 58).
        let np2 = not(b, p_pre[2]);
        let leaf_low = and(b, c.leaf_low, leaf, np2);
        let mut first = leaf_low;
        let mut last = leaf_low;
        for j in 3..P_BITS {
            let nj = not(b, p_pre[j]);
            first = and(b, c.first[j - 3], first, nj);
            last = and(b, c.last[j - 3], last, p_pre[j]);
        }
        let not_leaf = not(b, leaf);
        let start_flag = b.xor3(not_leaf, first, ctl[PLAIN_START]);
        let end_flag = b.xor3(not_leaf, last, ctl[PLAIN_END]);
        // The one 16-byte block: a plain CHUNK_END without CHUNK_START.
        let not_start = not(b, ctl[PLAIN_START]);
        let short = and(b, c.short, not_start, ctl[PLAIN_END]);
        let long = not(b, short);
        let flags: Word = std::array::from_fn(|j| match j {
            0 => start_flag,
            1 | 3 => end_flag,
            _ => zero,
        });

        // The compression's inputs.
        let keep = b.xor2(active, pending);
        let v_eff = mux(b, &c.keep, keep, &v, &y);
        let mut address = a0;
        address[61] = b.xor2(a0[61], mu_eff[0]);
        address[62] = b.xor2(a0[62], mu_eff[1]);
        let index: [Expr; 128] = std::array::from_fn(|j| if j < P_BITS { p_pre[j] } else { zero });
        let step_msg = [pp, address, v_eff, index];
        let leaf_msg = [b0, b1, b2, v_eff];
        let m: Vec<[Expr; 128]> = (0..4)
            .map(|j| {
                let cols: &[Var; 128] = &c.m[j];
                mux(b, cols, leaf, &leaf_msg[j], &step_msg[j])
            })
            .collect();
        let h: [Expr; 256] = std::array::from_fn(|j| if j < 128 { h0[j] } else { h1[j - 128] });
        let iv: [Expr; 256] = {
            let lo = constant(b, iv_word(0));
            let hi = constant(b, iv_word(1));
            std::array::from_fn(|j| if j < 128 { lo[j] } else { hi[j - 128] })
        };
        let cv = mux(b, &c.cv, leaf, &h, &iv);
        let cv_words: [Word; 8] = std::array::from_fn(|i| std::array::from_fn(|j| cv[32 * i + j]));
        let msg_words: [Word; 16] = std::array::from_fn(|i| split(&m[i / 4])[i % 4]);
        let block_len: Word = std::array::from_fn(|j| match j {
            4 => short,
            6 => long,
            _ => zero,
        });
        // One compression, packed by the hand-written BLAKE3 generator.
        let out_lo = compress_op(
            b,
            &c.rounds,
            &c.out_lo,
            cv_words,
            msg_words,
            [[zero; 32]; 2],
            block_len,
            flags,
        );
        let out0: [Expr; 128] = std::array::from_fn(|j| out_lo[j]);

        // The next state.
        let h_next = mux(b, &c.h, leaf, &out_lo, &h);
        let slots = [b0, b1, b2];
        let fresh = [f0, f1, f2];
        let b_next: Vec<[Expr; 128]> = (0..3)
            .map(|j| {
                let reset = mux(b, &c.reset[j], leaf, &fresh[j], &slots[j]);
                mux(b, &c.write[j], end_slot[j], &out0, &reset)
            })
            .collect();
        let ctl_next: [Expr; 128] = std::array::from_fn(|j| match j {
            j if j < DIGIT_BITS => digits[j],
            j if j < MU_BASE => p_next[j - P_BASE],
            j if j < ACTIVE => mu_next[j - MU_BASE],
            ACTIVE => cont,
            PENDING => pending_next,
            _ => zero,
        });
        let next: [[Expr; 128]; 7] = [
            out0,
            b_next[0],
            b_next[1],
            b_next[2],
            std::array::from_fn(|j| h_next[j]),
            std::array::from_fn(|j| h_next[128 + j]),
            ctl_next,
        ];
        for (cols, value) in c.out.iter().zip(next) {
            for (&col, e) in cols.iter().zip(value) {
                b.define_linear(col, e);
            }
        }
    });
    (compiled.circuit().clone(), compiled.operations().to_vec())
}

fn u(v: F128) -> u128 {
    v.lo as u128 | (v.hi as u128) << 64
}

fn f(v: u128) -> F128 {
    F128::new(v as u64, (v >> 64) as u64)
}

fn bit(x: u128, j: usize) -> bool {
    (x >> j) & 1 == 1
}

fn u32s(w: u128) -> [u32; 4] {
    std::array::from_fn(|i| (w >> (32 * i)) as u32)
}

fn word_of(w: &[u32]) -> u128 {
    w.iter()
        .enumerate()
        .fold(0, |acc, (i, &x)| acc | (x as u128) << (32 * i))
}

/// The row's outputs from its inputs, the same Boolean formulas as
/// [`machine_circuit`] on every input (reachable or not).
pub fn native(inputs: &[F128], outputs: &mut Vec<F128>) {
    let w: [u128; 13] = std::array::from_fn(|k| u(inputs[k]));
    let [v, b0, b1, b2, h0, h1, ctl, pp, a0, f0, f1, f2, y] = w;
    let p = ((ctl >> P_BASE) & 63) as u32;
    let mu = (ctl >> MU_BASE) & 3;
    let (active, pending) = (bit(ctl, ACTIVE), bit(ctl, PENDING));
    let settle = !active & !pending;
    let three: [bool; 4] = std::array::from_fn(|k| bit(ctl, W * k) & bit(ctl, W * k + 1));
    let mut t = [true; 5];
    for k in 1..=4 {
        t[k] = t[k - 1] & three[k - 1];
    }
    let (s0, s1) = (p & 1 == 1, (p >> 1) & 1 == 0);
    let sigma3 = s0 & s1;
    let sigma = [true ^ s0 ^ s1 ^ sigma3, s0 ^ sigma3, s1 ^ sigma3, sigma3];
    let mut leaf_at = [false; 5];
    for k in 1..=4 {
        leaf_at[k] = settle & sigma[4 - k] & t[k];
    }
    let leaf_now = leaf_at[1] ^ leaf_at[2] ^ leaf_at[3] ^ leaf_at[4];
    let leaf = pending ^ leaf_now;
    let start = settle ^ leaf_now;
    let step = active ^ start;
    let mut start_at = [false; 5];
    for k in 0..4 {
        start_at[k] = start & (t[k] ^ t[k + 1]);
    }
    let consumed: [bool; 5] = std::array::from_fn(|m| m > 0 && (start_at[m - 1] ^ leaf_at[m]));
    let skipped: [bool; 5] = std::array::from_fn(|k| start_at[k] ^ leaf_at[k]);
    let shift = 2 * (consumed[1] ^ consumed[3]) as u32
        + 4 * (consumed[2] ^ consumed[3]) as u32
        + 8 * consumed[4] as u32;
    let k = (skipped[1] ^ skipped[3]) as u32
        + 2 * (skipped[2] ^ skipped[3]) as u32
        + 4 * skipped[4] as u32;
    let p_pre = (p + k) & 63;
    let mut mu_start = 0u128;
    for (kk, &sel) in start_at[..4].iter().enumerate() {
        if sel {
            mu_start ^= (ctl >> (W * kk)) & 3;
        }
    }
    let mu_eff = mu ^ mu_start;
    let (me0, me1) = (bit(mu_eff, 0), bit(mu_eff, 1));
    let end = step & me1 & !me0;
    let cont = step ^ end;
    let mu_next = (cont & !me0) as u128 | ((cont & (me1 ^ me0)) as u128) << 1;
    let (q0, q1) = (p_pre & 1 == 1, (p_pre >> 1) & 1 == 0);
    let tau3 = q0 & q1;
    let tau = [true ^ q0 ^ q1 ^ tau3, q0 ^ tau3, q1 ^ tau3];
    let pending_next = end & tau3;
    let p_next = (p_pre + end as u32) & 63;
    let digits = (ctl & ((1u128 << DIGIT_BITS) - 1)) >> shift;
    let pb = |j: u32| (p_pre >> j) & 1 == 1;
    let leaf_low = leaf & !pb(2);
    let first = leaf_low & !pb(3) & !pb(4) & !pb(5);
    let last = leaf_low & pb(3) & pb(4) & pb(5);
    let (plain_start, plain_end) = (bit(ctl, PLAIN_START), bit(ctl, PLAIN_END));
    let flags = ((!leaf ^ first ^ plain_start) as u32 * CHUNK_START)
        | ((!leaf ^ last ^ plain_end) as u32 * (CHUNK_END | ROOT));
    let block_len = if !plain_start & plain_end { 16 } else { 64 };
    let keep = active ^ pending;
    let v_eff = if keep { v } else { y };
    let address = a0 ^ (me0 as u128) << 61 ^ (me1 as u128) << 62;
    let m = if leaf {
        [b0, b1, b2, v_eff]
    } else {
        [pp, address, v_eff, p_pre as u128]
    };
    let cv = if leaf {
        [h0, h1]
    } else {
        [iv_word(0), iv_word(1)]
    };
    let mut cv32 = [0u32; 8];
    cv32[..4].copy_from_slice(&u32s(cv[0]));
    cv32[4..].copy_from_slice(&u32s(cv[1]));
    let m32: [u32; 16] = std::array::from_fn(|i| u32s(m[i / 4])[i % 4]);
    let out = flock_hash::blake3_compress(&cv32, &m32, 0, block_len, flags);
    let (out0, out1) = (word_of(&out[..4]), word_of(&out[4..8]));
    let h_next = if leaf { [out0, out1] } else { [h0, h1] };
    let slots = [b0, b1, b2];
    let fresh = [f0, f1, f2];
    let b_next: [u128; 3] = std::array::from_fn(|j| {
        if end & tau[j] {
            out0
        } else if leaf {
            fresh[j]
        } else {
            slots[j]
        }
    });
    let ctl_next = digits
        | (p_next as u128) << P_BASE
        | mu_next << MU_BASE
        | (cont as u128) << ACTIVE
        | (pending_next as u128) << PENDING;
    outputs.extend(
        [
            out0, b_next[0], b_next[1], b_next[2], h_next[0], h_next[1], ctl_next,
        ]
        .map(f),
    );
}

/// The `ctl` of a plain compression with these flags and block length: one
/// block (`CHUNK_START | CHUNK_END | ROOT`, 64 bytes), a first block
/// (`CHUNK_START`, 64) or a 16-byte last block (`CHUNK_END | ROOT`, 16).
pub fn plain_ctl(flags: u32, block_len: u32) -> F128 {
    let (start, end) = match (flags, block_len) {
        (f, 64) if f == CHUNK_START | CHUNK_END | ROOT => (1, 1),
        (CHUNK_START, 64) => (1, 0),
        (f, 16) if f == CHUNK_END | ROOT => (0, 1),
        _ => panic!("no plain compression with flags {flags:#x}, block length {block_len}"),
    };
    // A pending LEAF selects the leaf block and chaining value; p = 4 is no
    // leaf block's, so the leaf flags stay clear.
    f(1 << PENDING | 4 << P_BASE | start << PLAIN_START | end << PLAIN_END)
}

/// The machine row as a gate type: inputs the state words then
/// `[P, A_chain(step 0), f0, f1, f2, y]`, outputs the next state words.
pub fn machine_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    static CIRCUIT: std::sync::OnceLock<(BooleanCircuit, Vec<SchemaOperation>)> =
        std::sync::OnceLock::new();
    let (circuit, operations) = CIRCUIT.get_or_init(machine_circuit);
    let ins: Vec<&str> = STATE.iter().chain(&EXTRA).copied().collect();
    let outs: Vec<String> = STATE.iter().map(|s| out_name(s)).collect();
    let outs: Vec<&str> = outs.iter().map(String::as_str).collect();
    boolean_gate("leanxmss.machine", circuit, operations, &ins, &outs, nu)
        .map(|g| g.with_native(native))
}

/// The prover's values for one layer: the initial slot 2 (chain 0's end if
/// it takes no step), then per row `[f0, f1, f2, y]`.
pub struct LayerFresh {
    pub b2: F128,
    pub rows: Vec<[F128; 4]>,
}

/// Plans one layer's run from its codeword digits and revealed chain values.
pub fn layer_fresh(digits: &[u8; V], y: &[Value; V]) -> LayerFresh {
    let word = |v: &Value| f(u128::from_le_bytes(*v));
    // A skipped chain's slot holds its revealed value; any other slot is
    // overwritten by its chain's end.
    let free = |c: usize| {
        if c < V && digits[c] as usize == CHAIN_END {
            word(&y[c])
        } else {
            F128::ZERO
        }
    };
    let slot = |c: usize| (c + 2) % 4;
    let (mut p, mut active, mut mu, mut pending) = (0usize, false, 0usize, false);
    let mut rows = Vec::with_capacity(ROWS);
    for _ in 0..ROWS {
        let mut row = [F128::ZERO; 4];
        let mut leaf = pending;
        if !active && !pending {
            loop {
                if digits[p] as usize == CHAIN_END {
                    p += 1;
                    if slot(p - 1) == 3 {
                        row[3] = word(&y[p - 1]);
                        leaf = true;
                        break;
                    }
                } else {
                    row[3] = word(&y[p]);
                    active = true;
                    mu = digits[p] as usize;
                    break;
                }
            }
        }
        if leaf {
            pending = false;
            for (j, fj) in row[..3].iter_mut().enumerate() {
                *fj = free(p + j);
            }
        } else {
            mu += 1;
            if mu == CHAIN_END {
                active = false;
                pending = slot(p) == 3;
                p += 1;
            }
        }
        rows.push(row);
    }
    assert!(
        p == V && !active && !pending,
        "the digits are not a codeword"
    );
    LayerFresh { b2: free(0), rows }
}

#[cfg(test)]
mod tests;
