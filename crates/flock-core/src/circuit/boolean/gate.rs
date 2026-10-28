//! A compiled Boolean circuit as a wiring-layer gate type.
//!
//! [`BooleanGate`] turns one gadget written with [`CircuitBuilder`] into a
//! [`GateType`] that [`ShapeBuilder`] can instantiate and wire. Named 128-bit
//! columns become the gate's wired words: `inputs` in order as `In` words,
//! then `outputs` as `Out` words. The table is the gadget lowered with
//! identity `C`, and both the online evaluation and the packed prover witness
//! come from the gadget's own [`WalkPlan`], so the constraints, the values on
//! the wires and the committed bits are one description.
//!
//! **Identity `C` is enforced here.** The union prover and verifier use the
//! `C` claim directly as a `z` claim; nothing downstream checks that `C = I`.
//! A gadget with general constraints must therefore be lowered with
//! [`LoweringMode::RequireIdentityC`], and [`BooleanGate::new`] refuses any
//! relation whose `C` is not the identity.
//!
//! [`CircuitBuilder`]: super::CircuitBuilder
//! [`ShapeBuilder`]: crate::circuit::builder::ShapeBuilder

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSlice,
    ParallelSliceMut,
};

use super::{
    BooleanCircuit, ColumnRole, LayoutError, LoweringMode, R1csBuildError, ValueId, WalkError,
    WalkPlan,
};
use crate::{
    circuit::builder::{GateType, SlotWitness},
    field::F128,
    r1cs::BlockR1cs,
    schedule::{IoWord, TableType},
    scratch::take_u8,
    union::SlotWitnessDest,
    zerocheck::K_SKIP,
};

/// Why a compiled circuit cannot be used as a gate type.
#[derive(Debug)]
pub enum BooleanGateError {
    Layout(LayoutError),
    R1cs(R1csBuildError),
    Walk(WalkError),
    /// The lowered relation's `C` is not the identity.
    NotIdentityC,
    /// No column has this name.
    UnknownColumn(String),
    /// A wired column must be exactly 128 bits.
    NotAWord {
        name: String,
        width: usize,
    },
    /// A wired input column must have [`ColumnRole::Input`].
    NotAnInput(String),
    /// A wired column must occupy one whole 128-bit word of the block, in bit
    /// order. Declare it with `word_aligned::<128>(name, role, 128)`.
    Misaligned(String),
    /// Every circuit input must arrive on a wired input word.
    UnwiredInput(String),
}

impl From<LayoutError> for BooleanGateError {
    fn from(e: LayoutError) -> Self {
        Self::Layout(e)
    }
}

impl From<R1csBuildError> for BooleanGateError {
    fn from(e: R1csBuildError) -> Self {
        Self::R1cs(e)
    }
}

impl From<WalkError> for BooleanGateError {
    fn from(e: WalkError) -> Self {
        Self::Walk(e)
    }
}

/// A Boolean gadget as a gate type: one gadget invocation per row. Cloning
/// shares the relation and the walk plan.
#[derive(Clone)]
pub struct BooleanGate {
    name: String,
    r1cs: Arc<BlockR1cs>,
    plan: Arc<WalkPlan>,
    io_schema: Vec<IoWord>,
    /// The source of each walk input, in the circuit's input order: (wired
    /// input word, bit).
    input_bits: Vec<(u32, u8)>,
    n_in: usize,
    /// The block word column of each wired output, in order.
    output_cols: Vec<usize>,
    /// A direct evaluation of the outputs, if the gadget has one.
    native: Option<NativeEval>,
}

/// Computes a gadget's wired outputs from its wired inputs without walking
/// the circuit (see [`BooleanGate::with_native`]).
pub type NativeEval = Arc<dyn Fn(&[F128], &mut Vec<F128>) + Send + Sync>;

thread_local! {
    static SCRATCH: RefCell<SlicedScratch> = RefCell::new(SlicedScratch::default());
}

impl BooleanGate {
    /// Lower `circuit` with identity `C` into a table of `2^nu` rows, wiring
    /// the named 128-bit columns: `inputs` (role [`ColumnRole::Input`]), then
    /// `outputs` (any materialized word).
    pub fn new(
        name: &str,
        circuit: &BooleanCircuit,
        inputs: &[&str],
        outputs: &[&str],
        nu: usize,
    ) -> Result<Self, BooleanGateError> {
        let lowered = circuit.lower(LoweringMode::RequireIdentityC)?;
        let layout = lowered.layout();
        let k_log = (layout.useful_bits().next_power_of_two().trailing_zeros() as usize).max(7);
        let r1cs = lowered.to_block_r1cs(k_log, K_SKIP, nu)?;
        let plan = lowered.walk_plan()?;
        if !r1cs.c0_is_identity() || !plan.c_is_identity() {
            return Err(BooleanGateError::NotIdentityC);
        }
        // A wired column: its source bits and its block word column.
        let word = |column: &str| -> Result<(Vec<ValueId>, usize), BooleanGateError> {
            let col = circuit
                .column(column)
                .ok_or_else(|| BooleanGateError::UnknownColumn(column.into()))?;
            if col.values.len() != 128 {
                return Err(BooleanGateError::NotAWord {
                    name: column.into(),
                    width: col.values.len(),
                });
            }
            let position = |bit: usize| {
                lowered
                    .mapped_value(col.values[bit])
                    .and_then(|v| layout.value_position(v))
            };
            let first = position(0).ok_or_else(|| BooleanGateError::Misaligned(column.into()))?;
            if first % 128 != 0 || (0..128).any(|bit| position(bit) != Some(first + bit)) {
                return Err(BooleanGateError::Misaligned(column.into()));
            }
            Ok((col.values.clone(), first / 128))
        };
        let mut io_schema = Vec::with_capacity(inputs.len() + outputs.len());
        let mut source = HashMap::new();
        for (w, &column) in inputs.iter().enumerate() {
            if circuit.column(column).map(|c| &c.role) != Some(&ColumnRole::Input) {
                return Err(BooleanGateError::NotAnInput(column.into()));
            }
            let (bits, col) = word(column)?;
            io_schema.push(IoWord::input(col));
            for (bit, value) in bits.into_iter().enumerate() {
                source.insert(value, (w as u32, bit as u8));
            }
        }
        let mut output_cols = Vec::with_capacity(outputs.len());
        for &column in outputs {
            let (_, col) = word(column)?;
            io_schema.push(IoWord::output(col));
            output_cols.push(col);
        }
        let input_bits = circuit
            .inputs()
            .iter()
            .map(|value| {
                source.get(value).copied().ok_or_else(|| {
                    let column = circuit
                        .schema()
                        .iter()
                        .find(|c| c.values.contains(value))
                        .map_or_else(|| format!("{value:?}"), |c| c.name.clone());
                    BooleanGateError::UnwiredInput(column)
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            name: name.into(),
            r1cs: Arc::new(r1cs),
            plan: Arc::new(plan),
            io_schema,
            input_bits,
            n_in: inputs.len(),
            output_cols,
            native: None,
        })
    }

    /// Evaluate the outputs with `native` in the online phase instead of the
    /// walk. The packer still derives every committed bit from the circuit,
    /// so a `native` that disagreed with it could only break proofs, never
    /// make a false one; builds with debug assertions check the two agree on
    /// every evaluation. `native` must panic on inputs the circuit rejects.
    pub fn with_native(
        mut self,
        native: impl Fn(&[F128], &mut Vec<F128>) + Send + Sync + 'static,
    ) -> Self {
        self.native = Some(Arc::new(native));
        self
    }

    /// Whether the online phase uses a native evaluation.
    pub fn has_native(&self) -> bool {
        self.native.is_some()
    }

    /// The outputs as the circuit computes them: one lane of the bit-sliced
    /// walk. Panics if the circuit rejects the inputs.
    pub fn eval_walk(&self, inputs: &[F128], outputs: &mut Vec<F128>) {
        assert_eq!(inputs.len(), self.n_in, "{}: wrong input count", self.name);
        SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let sc: &mut SlicedScratch = &mut guard;
            sc.inputs.clear();
            sc.inputs.extend(self.input_bits.iter().map(|&(w, bit)| {
                let v = inputs[w as usize];
                (if bit < 64 {
                    v.lo >> bit
                } else {
                    v.hi >> (bit - 64)
                }) & 1
            }));
            if let Err(e) = self.plan.forward_sliced_into(
                &sc.inputs,
                1,
                self.r1cs.k_log,
                &mut sc.z,
                &mut sc.a,
                &mut sc.b,
                &mut sc.temporaries,
            ) {
                panic!(
                    "{}: the gadget's constraints reject these inputs ({e:?})",
                    self.name
                );
            }
            for &col in &self.output_cols {
                let bits = &sc.z[col * 128..col * 128 + 128];
                let half = |bits: &[u64]| {
                    bits.iter()
                        .enumerate()
                        .fold(0u64, |w, (j, &b)| w | (b & 1) << j)
                };
                outputs.push(F128::new(half(&bits[..64]), half(&bits[64..])));
            }
        });
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The block relation, for the prover's and verifier's lincheck circuit.
    pub fn r1cs(&self) -> &BlockR1cs {
        &self.r1cs
    }

    pub fn k_log(&self) -> usize {
        self.r1cs.k_log
    }

    pub fn useful_bits(&self) -> usize {
        self.r1cs.useful_bits
    }

    /// The prover's in-place driver for this slot: packs every row's `z`,
    /// `A·z` and `B·z` into the BatchMajor buffers of `dst` (rows past
    /// `rows.len()` stay zero) and returns the lincheck stripe.
    ///
    /// Rows are walked 64 at a time, bit-sliced ([`WalkPlan::forward_sliced_into`]):
    /// each position's 64 lanes are the stripe's bytes for 8 row groups as
    /// they stand, and a 64×64 bit transpose turns them into row words.
    pub fn pack_into(&self, rows: &[Box<[F128]>], dst: SlotWitnessDest<'_>) -> Vec<u8> {
        let k_log = self.r1cs.k_log;
        let k = 1usize << k_log;
        let useful = self.r1cs.useful_bits;
        let w = useful.div_ceil(128);
        let n = rows.len();
        let SlotWitnessDest {
            z,
            a,
            b,
            elide_padding_writes,
            ..
        } = dst;
        let capacity = z.len() >> (k_log - 7);
        assert!(
            capacity.is_power_of_two() && n <= capacity,
            "{}: {n} rows",
            self.name
        );
        let nu = capacity.trailing_zeros() as usize;
        if !elide_padding_writes {
            for buf in [&mut *z, &mut *a, &mut *b] {
                buf.par_chunks_mut(1 << 16).for_each(|c| c.fill(F128::ZERO));
            }
        }
        let mut stripe = take_u8((capacity / 8) * k);
        let live_groups = n.div_ceil(8);
        let zeroed = if elide_padding_writes {
            live_groups
        } else {
            capacity / 8
        };
        stripe
            .par_chunks_mut(k)
            .take(zeroed)
            .for_each(|g| g.fill(0));
        // Each 64-row group writes its own run of every column: rows
        // `64g..64g + 64` of column `col` are `buf[(col << nu) + 64g..]`.
        let mut runs: Vec<[Vec<&mut [F128]>; 3]> = (0..n.div_ceil(64))
            .map(|_| std::array::from_fn(|_| Vec::with_capacity(w)))
            .collect();
        for (side, buf) in [z, a, b].into_iter().enumerate() {
            for column in buf.chunks_mut(1 << nu).take(w) {
                for (g, run) in column[..n].chunks_mut(64).enumerate() {
                    runs[g][side].push(run);
                }
            }
        }
        runs.par_iter_mut()
            .zip(stripe[..live_groups * k].par_chunks_mut(8 * k))
            .zip(rows.par_chunks(64))
            .for_each_init(SlicedScratch::default, |sc, ((runs, stripe), rows)| {
                self.walk_sliced(rows, sc);
                for (planes, runs) in [&sc.z, &sc.a, &sc.b].into_iter().zip(runs.iter_mut()) {
                    for (col, run) in runs.iter_mut().enumerate() {
                        let lo = transpose64(planes[128 * col..128 * col + 64].try_into().unwrap());
                        let hi = transpose64(
                            planes[128 * col + 64..128 * col + 128].try_into().unwrap(),
                        );
                        for (r, o) in run.iter_mut().enumerate() {
                            *o = F128::new(lo[r], hi[r]);
                        }
                    }
                }
                for (g, group) in stripe.chunks_mut(k).enumerate() {
                    for (byte, &plane) in group[..useful].iter_mut().zip(&sc.z) {
                        *byte = (plane >> (8 * g)) as u8;
                    }
                }
            });
        stripe
    }

    /// Up to 64 rows at once: their `z`, `A·z` and `B·z` as bit planes in `sc`.
    fn walk_sliced(&self, rows: &[Box<[F128]>], sc: &mut SlicedScratch) {
        let lanes = rows.len();
        let live = if lanes == 64 { !0 } else { (1u64 << lanes) - 1 };
        // Input word `w`, bit `j`, across the rows: plane `128w + j`.
        sc.in_planes.clear();
        for w in 0..self.n_in {
            let mut lo = [0u64; 64];
            let mut hi = [0u64; 64];
            for (r, row) in rows.iter().enumerate() {
                lo[r] = row[w].lo;
                hi[r] = row[w].hi;
            }
            sc.in_planes.extend(transpose64(lo));
            sc.in_planes.extend(transpose64(hi));
        }
        sc.inputs.clear();
        sc.inputs.extend(
            self.input_bits
                .iter()
                .map(|&(w, bit)| sc.in_planes[128 * w as usize + bit as usize]),
        );
        if let Err(e) = self
            .plan
            .forward_sliced_compact(&sc.inputs, live, &mut sc.z, &mut sc.a, &mut sc.b)
        {
            panic!(
                "{}: the gadget's constraints reject a row ({e:?})",
                self.name
            );
        }
    }
}

/// Per-worker buffers of the bit-sliced packer.
#[derive(Default)]
struct SlicedScratch {
    in_planes: Vec<u64>,
    inputs: Vec<u64>,
    z: Vec<u64>,
    a: Vec<u64>,
    b: Vec<u64>,
    temporaries: Vec<u64>,
}

/// Transpose a 64×64 bit matrix: bit `c` of `out[r]` is bit `r` of `m[c]`.
fn transpose64(mut m: [u64; 64]) -> [u64; 64] {
    let mut width = 32;
    let mut mask: u64 = 0x0000_0000_FFFF_FFFF;
    while width > 0 {
        let mut base = 0;
        while base < 64 {
            for r in base..base + width {
                let t = ((m[r] >> width) ^ m[r + width]) & mask;
                m[r] ^= t << width;
                m[r + width] ^= t;
            }
            base += 2 * width;
        }
        width >>= 1;
        mask ^= mask << width;
    }
    m
}

impl GateType for BooleanGate {
    /// The wired input words; the packer re-walks them.
    type Row = Box<[F128]>;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&self.r1cs).with_io_schema(self.io_schema.clone())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        match &self.native {
            Some(native) => {
                let start = outputs.len();
                native(inputs, outputs);
                if cfg!(debug_assertions) {
                    let mut walked = Vec::new();
                    self.eval_walk(inputs, &mut walked);
                    assert_eq!(
                        outputs[start..],
                        walked[..],
                        "{}: native evaluation disagrees",
                        self.name
                    );
                }
            }
            None => self.eval_walk(inputs, outputs),
        }
        inputs.into()
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn transpose64_transposes() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let m: [u64; 64] = std::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        });
        let t = super::transpose64(m);
        for r in 0..64 {
            for c in 0..64 {
                assert_eq!((t[r] >> c) & 1, (m[c] >> r) & 1, "({r}, {c})");
            }
        }
    }
}
