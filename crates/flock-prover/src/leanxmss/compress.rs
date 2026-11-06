//! One BLAKE3 compression per row, everything wired: the chaining value, the
//! block and a control word in, the output chaining value out. Every
//! compression a signature spends runs on this gate.
//!
//! A row also outputs `next`: its block word `m1` with the chain step `µ`
//! (bits 61..62, from the address byte `tag + 32µ`) advanced by one, mod 4.
//! On a chain step `m1` is the address `A(µ)` and `next` is `A(µ + 1)`,
//! which the tuple bus needs to hand the result to the following step
//! ([`super::circuit`]); other rows ignore it.

use flock_core::circuit::boolean::{
    BooleanCircuit, BooleanGate, BooleanGateError, CircuitBuilder, ColumnRole, ColumnSchema,
    ColumnVisitor, LinearExpr as Expr, SchemaOperation, Var,
};
use flock_field::F128;

use crate::r1cs_hashes::{
    blake3_dsl::{GCols, compress_op, round_columns},
    dsl::{Word, boolean_gate},
};

/// Wired inputs, in io-schema order.
const INPUTS: [&str; 7] = ["cv0", "cv1", "m0", "m1", "m2", "m3", "ctl"];
/// Wired outputs, after the inputs.
const OUTPUTS: [&str; 3] = ["o0", "o1", "next"];

/// io-schema positions: the block words, the first output word and `next`.
pub const M: usize = 2;
pub const O0: usize = 7;
pub const NEXT: usize = 9;

/// The chain step's two bits in an address word.
pub const STEP_LO: usize = 61;
pub const STEP_HI: usize = 62;

/// The control word: BLAKE3 `flags` in bits 0..32, the block length in bits
/// 32..64 (the counter is always zero).
pub fn ctl(flags: u32, block_len: u32) -> F128 {
    F128::new(flags as u64 | (block_len as u64) << 32, 0)
}

/// `m1` with its step advanced: `(s0, s1) ↦ (¬s0, s1 ⊕ s0)`.
pub fn next_step(m1: F128) -> F128 {
    let (s0, s1) = ((m1.lo >> STEP_LO) & 1, (m1.lo >> STEP_HI) & 1);
    let lo = m1.lo & !(3 << STEP_LO) | (s0 ^ 1) << STEP_LO | (s1 ^ s0) << STEP_HI;
    F128::new(lo, m1.hi)
}

pub struct CompressCols<T> {
    /// Zero bits that end the rounds on a word boundary, so the output words
    /// can follow them contiguously (the BLAKE3 kernel writes the rounds and
    /// the output as one run).
    pub pad: Vec<T>,
    pub rounds: Vec<[GCols<T>; 8]>,
    pub out: [[T; 128]; 2],
    pub inputs: Vec<[T; 128]>,
    pub next: [T; 128],
}

struct CompressSchema {
    pad: usize,
}

impl ColumnSchema for CompressSchema {
    type Cols<T> = CompressCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> CompressCols<V::Value> {
        CompressCols {
            pad: if self.pad == 0 {
                Vec::new()
            } else {
                v.bits("pad", ColumnRole::Witness, self.pad, 1)
            },
            rounds: round_columns(v, "rounds"),
            out: [
                v.word_aligned(OUTPUTS[0], ColumnRole::Output, 128),
                v.word_aligned(OUTPUTS[1], ColumnRole::Output, 128),
            ],
            inputs: INPUTS
                .iter()
                .map(|name| v.word_aligned(name, ColumnRole::Input, 128))
                .collect(),
            next: v.word_aligned(OUTPUTS[2], ColumnRole::Output, 128),
        }
    }
}

fn compile(pad: usize) -> (BooleanCircuit, Vec<SchemaOperation>) {
    let compiled = CircuitBuilder::compile(CompressSchema { pad }, |b, c| {
        let zero = b.zero();
        for &bit in &c.pad {
            b.define_linear(bit, zero);
        }
        let word =
            |w: &[Var; 128], i: usize| -> Word { std::array::from_fn(|j| w[32 * i + j].into()) };
        let [cv0, cv1, m0, m1, m2, m3, ctl]: [&[Var; 128]; 7] =
            std::array::from_fn(|i| &c.inputs[i]);
        let cv: [Word; 8] = std::array::from_fn(|i| word([cv0, cv1][i / 4], i % 4));
        let message: [Word; 16] = std::array::from_fn(|i| word([m0, m1, m2, m3][i / 4], i % 4));
        let out: [Var; 256] = std::array::from_fn(|j| c.out[j / 128][j % 128]);
        compress_op(
            b,
            &c.rounds,
            &out,
            cv,
            message,
            [[zero; 32]; 2],
            word(ctl, 1),
            word(ctl, 0),
        );
        let one: Expr = b.one().into();
        for j in 0..128 {
            let bit: Expr = match j {
                STEP_LO => b.xor2(m1[STEP_LO], one),
                STEP_HI => b.xor2(m1[STEP_HI], m1[STEP_LO]),
                _ => m1[j].into(),
            };
            b.define_linear(c.next[j], bit);
        }
    });
    (compiled.circuit().clone(), compiled.operations().to_vec())
}

/// The compression row's circuit and its operation. Built once without
/// padding to find where the rounds end, then with the padding that puts
/// that end on a word boundary.
pub fn compress_circuit() -> &'static (BooleanCircuit, Vec<SchemaOperation>) {
    static CIRCUIT: std::sync::OnceLock<(BooleanCircuit, Vec<SchemaOperation>)> =
        std::sync::OnceLock::new();
    CIRCUIT.get_or_init(|| {
        let (probe, operations) = compile(0);
        let layout = probe.layout().expect("the compression row lays out");
        let columns = &operations
            .iter()
            .find(|op| op.kind == crate::r1cs_hashes::blake3_dsl::OPERATION)
            .expect("the compression is an operation")
            .columns;
        let last_round = columns[columns.len() - 257];
        let end = layout.value_position(last_round).expect("placed") + 1;
        compile((128 - end % 128) % 128)
    })
}

fn native(inputs: &[F128], outputs: &mut Vec<F128>) {
    let u32s = |w: F128| {
        [
            w.lo as u32,
            (w.lo >> 32) as u32,
            w.hi as u32,
            (w.hi >> 32) as u32,
        ]
    };
    let mut cv = [0u32; 8];
    cv[..4].copy_from_slice(&u32s(inputs[0]));
    cv[4..].copy_from_slice(&u32s(inputs[1]));
    let m: [u32; 16] = std::array::from_fn(|i| u32s(inputs[M + i / 4])[i % 4]);
    let ctl = inputs[6];
    let out = flock_hash::blake3_compress(&cv, &m, 0, (ctl.lo >> 32) as u32, ctl.lo as u32);
    let word = |w: &[u32]| {
        F128::new(
            w[0] as u64 | (w[1] as u64) << 32,
            w[2] as u64 | (w[3] as u64) << 32,
        )
    };
    outputs.extend([word(&out[..4]), word(&out[4..8]), next_step(inputs[M + 1])]);
}

/// Inputs `[cv0, cv1, m0, m1, m2, m3, ctl]`, outputs `[o0, o1, next]`.
pub fn compress_gate(nu: usize) -> Result<BooleanGate, BooleanGateError> {
    let (circuit, operations) = compress_circuit();
    boolean_gate(
        "leanxmss.compress",
        circuit,
        operations,
        &INPUTS,
        &OUTPUTS,
        nu,
    )
    .map(|g| g.with_native(native))
}

#[cfg(test)]
mod tests;
