//! One BLAKE3 compression per row, everything wired: the chaining value, the
//! block and a control word in, the output chaining value out. Every
//! compression a signature spends runs on this gate.
//!
//! A row also outputs `next`: its block word `m1` with the chain step `µ`
//! (bits 61..62, from the address byte `tag + 32µ`) advanced by one, mod 4.
//! On a chain step `m1` is the address `A(µ)` and `next` is `A(µ + 1)`,
//! which the tuple bus needs to hand the result to the following step
//! ([`super::circuit`]); other rows ignore it.

use flock_core::{
    circuit::{
        boolean::{
            BooleanCircuit, BooleanGate, BooleanGateError, CircuitBuilder, ColumnRole,
            ColumnSchema, ColumnVisitor, LinearExpr as Expr, OperationKernel, SchemaOperation, Var,
        },
        builder::GateType,
    },
    union::SlotWitnessDest,
};
use flock_field::F128;

use crate::r1cs_hashes::{
    blake3::{Blake3Kernel, G_REGION_BITS},
    blake3_dsl::{GCols, OPERATION, compress_op, round_columns},
    common::{BM_V, BmRow, drive_witness_batch_major_partial_into},
    dsl::{Word, boolean_gate},
};

/// The gate's name.
pub const NAME: &str = "leanxmss.compress";

/// The compression's first bit within its block word: the padding that ends
/// the rounds on a word boundary starts them here. A constant so the kernel's
/// fields compile to fixed shifts ([`Blake3Kernel::fill8_inline`]).
const KERNEL_BIT: usize = 6;

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
            .find(|op| op.kind == OPERATION)
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
    boolean_gate(NAME, circuit, operations, &INPUTS, &OUTPUTS, nu).map(|g| g.with_native(native))
}

/// The compression slots' witness driver: [`BooleanGate::pack_into`] for
/// this gate, written the way the hand-written BLAKE3 generator writes its
/// own slot ([`drive_witness_batch_major_partial_into`]). Eight rows at a
/// time, every word goes straight into row form: the wired inputs and
/// `next` as whole words (identity and linear rows, `z = A·z`, `B·z = 1`),
/// the compression by [`Blake3Kernel`], and the rest — the pin and the
/// padding's rows, the same in every row — from a template. No walk, no
/// plane transposes. Writes the same slot as the gate's packer.
pub struct CompressPacker {
    k_log: usize,
    useful_bits: usize,
    /// First block `u64` word of each wired input, and of `next`.
    inputs: [usize; 7],
    next: usize,
    /// The kernel's first word: it starts at [`KERNEL_BIT`] in it.
    kernel_word: usize,
    /// Every other word as `z`, `A·z`, `B·z`, its kernel bits clear.
    fixed: Vec<(usize, [u64; 3])>,
}

impl CompressPacker {
    pub fn new(gate: &BooleanGate) -> Self {
        assert_eq!(gate.name(), NAME);
        let (circuit, operations) = compress_circuit();
        let layout = circuit.layout().expect("the compression row lays out");
        let op = operations.iter().find(|op| op.kind == OPERATION).unwrap();
        let base = layout.value_position(op.columns[0]).expect("placed");
        assert_eq!(
            base % 64,
            KERNEL_BIT,
            "the compression moved within its word"
        );
        assert!(
            (base + G_REGION_BITS).is_multiple_of(64),
            "the rounds end mid-word"
        );
        let end = base + Blake3Kernel.width();
        let io: Vec<usize> = gate
            .table()
            .io_schema
            .iter()
            .map(|w| 2 * w.word_col)
            .collect();
        let inputs: [usize; 7] = io[..7].try_into().unwrap();
        let next = io[NEXT];
        // The bits computed per row; the template supplies the rest.
        let useful_words = gate.useful_bits().div_ceil(64);
        let mut computed = vec![0u64; useful_words];
        for &w in inputs.iter().chain([&next]) {
            computed[w..w + 2].fill(!0);
        }
        for p in base..end {
            computed[p / 64] |= 1 << (p % 64);
        }
        // The template: the gate's own packer on one group of zero rows.
        let len = BM_V << (gate.k_log() - 7);
        let mut z = vec![F128::ZERO; len];
        let (mut a, mut b) = (z.clone(), z.clone());
        let zero: Vec<Box<[F128]>> = vec![vec![F128::ZERO; 7].into(); BM_V];
        gate.pack_into(
            &zero,
            SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
                dead_padding_unread: false,
            },
        );
        let word = |buf: &[F128], w: usize| {
            let v = buf[(w / 2) * BM_V];
            (if w.is_multiple_of(2) { v.lo } else { v.hi }) & !computed[w]
        };
        let fixed = (0..useful_words)
            .filter(|&w| computed[w] != !0)
            .map(|w| (w, [word(&z, w), word(&a, w), word(&b, w)]))
            .collect();
        Self {
            k_log: gate.k_log(),
            useful_bits: gate.useful_bits(),
            inputs,
            next,
            kernel_word: base / 64,
            fixed,
        }
    }

    /// The slot's `z`, `A·z`, `B·z` into `dst`; returns the lincheck stripe.
    pub fn pack_into(&self, rows: &[Box<[F128]>], dst: SlotWitnessDest<'_>) -> Vec<u8> {
        let capacity = dst.z.len() >> (self.k_log - 7);
        drive_witness_batch_major_partial_into(
            rows,
            capacity.trailing_zeros() as usize,
            self.k_log,
            self.useful_bits,
            dst,
            |rows, rz, ra, rb| self.group(rows.map(|r| &**r), rz, ra, rb),
        )
    }

    fn group(&self, rows: [&[F128]; BM_V], rz: &mut [BmRow], ra: &mut [BmRow], rb: &mut [BmRow]) {
        // The kernel stores every word of its run whole, keeping the fixed
        // bits below it (its rounds end on a word boundary), so the rows
        // need no clearing.
        for &(w, [z, a, b]) in &self.fixed {
            (rz[w], ra[w], rb[w]) = ([z; BM_V], [a; BM_V], [b; BM_V]);
        }
        // Arguments: cv, the block, the counter (zero), block length, flags.
        let args: [[u64; 14]; BM_V] = rows.map(|r| {
            let mut args = [0; 14];
            for (i, w) in r[..6].iter().enumerate() {
                (args[2 * i], args[2 * i + 1]) = (w.lo, w.hi);
            }
            args[13] = r[6].lo.rotate_left(32);
            args
        });
        let base = 64 * self.kernel_word + KERNEL_BIT;
        Blake3Kernel.fill8_inline(args.each_ref().map(|a| &a[..]), base, rz, ra, rb);
        let mut store = |w: usize, v: [F128; BM_V]| {
            for (w, half) in [(w, v.map(|x| x.lo)), (w + 1, v.map(|x| x.hi))] {
                (rz[w], ra[w], rb[w]) = (half, half, [!0; BM_V]);
            }
        };
        for (i, &w) in self.inputs.iter().enumerate() {
            store(w, rows.map(|r| r[i]));
        }
        store(self.next, rows.map(|r| next_step(r[M + 1])));
    }
}

#[cfg(test)]
mod tests;
