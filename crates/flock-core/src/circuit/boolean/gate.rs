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

use std::{
    cell::RefCell,
    collections::HashMap,
    ops::Range,
    sync::{Arc, OnceLock},
};

use rayon::prelude::{ParallelIterator, ParallelSliceMut};

use super::{
    BooleanCircuit, ColumnRole, ExpressionNode, LayoutError, LoweringMode, OperationKernel,
    R1csBuildError, SchemaOperation, ValueId, WalkError, WalkPlan,
    walk::{KernelSite, SlicedProgram},
};
use crate::{
    bits::transpose_8_u64s_to_64_bytes,
    circuit::builder::{GateType, SlotWitness},
    field::F128,
    r1cs::BlockR1cs,
    run_hetero_chunks_stateful,
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
    /// An operation does not fit the kernel attached to its kind.
    Kernel(String),
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
    /// The block word column of each wired input, in order.
    input_cols: Vec<usize>,
    /// The block word column of each wired output, in order.
    output_cols: Vec<usize>,
    /// A direct evaluation of the outputs, if the gadget has one.
    native: Option<NativeEval>,
    /// Operations whose rows native kernels write ([`Self::with_kernel`]).
    kernels: Option<Arc<Kernels>>,
    /// How [`Self::pack_into`] writes a row group eight rows at a time, once
    /// worked out; `None` if a kernel's result feeds the walk.
    batched: Arc<OnceLock<Option<Batched>>>,
}

/// The operations left to kernels, and the walk without them.
struct Kernels {
    program: SlicedProgram,
    sites: Vec<Site>,
    /// Per 64-bit block word: whether a site covers all of it, part of it,
    /// or none.
    words: Vec<Cover>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cover {
    None,
    Part,
    Full,
}

/// The packer's plan for writing a 64-row group eight rows at a time: the
/// walk fills only the words the kernels leave, then each batch of eight
/// rows is zeroed, filled (wired inputs copied straight from the rows, the
/// walk's words transposed in, the kernels run) and streamed out on its own,
/// as the hand-written generators do. Needs every kernel's result to be
/// unread by the walk, so the kernels can run after it.
struct Batched {
    /// The walk without its identity rows
    /// ([`SlicedProgram::without_identity_rows`]).
    program: SlicedProgram,
    /// Block `u64` words holding identity rows, and which bits: there
    /// `A·z = z` and `B·z = 1`.
    identity: Vec<(usize, u64)>,
    /// Per wired input word: whether the walk (or a kernel argument the
    /// rows cannot supply directly) reads it, so its planes are needed.
    walked_inputs: Vec<bool>,
    /// Wired input words copied straight from the rows: (input, first block
    /// `u64` word of its column).
    direct: Vec<(usize, usize)>,
    /// Block `u64` words taken from the walk's planes.
    walked: Vec<Walked>,
    /// Block `u64` words a kernel writes, all or part: zeroed per batch.
    covered: Vec<usize>,
    /// Plane ranges the walk must zero each run: the words of `walked`.
    clear: Vec<Range<usize>>,
    /// Per kernel site, its arguments as runs of input bits, if every
    /// argument bit is one input bit or zero.
    args: Vec<Option<Vec<Vec<ArgRun>>>>,
}

/// A block `u64` word the walk writes.
#[derive(Clone, Copy)]
struct Walked {
    word: usize,
    /// A kernel writes the rest of the word: the walk's bits are OR-ed in.
    part: bool,
    /// `z` comes from the planes too; a wired input word only needs `A·z`
    /// and `B·z` from them (rows that check inputs sit on the inputs).
    z: bool,
}

/// A run of kernel argument bits copied from one half of an input word:
/// bits `from..` of it, as many as `mask` (the low bits) has, land at `to..`.
#[derive(Clone, Copy)]
struct ArgRun {
    input: usize,
    hi: bool,
    from: u32,
    to: u32,
    mask: u64,
}

/// One operation call left to a kernel.
#[derive(Clone)]
struct Site {
    kernel: Arc<dyn OperationKernel>,
    region: Range<usize>,
    /// Each argument bit: the block positions whose XOR it is.
    args: Vec<Vec<u32>>,
    /// The positions of its materialized result.
    results: Vec<usize>,
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
        let mut input_cols = Vec::with_capacity(inputs.len());
        for (w, &column) in inputs.iter().enumerate() {
            if circuit.column(column).map(|c| &c.role) != Some(&ColumnRole::Input) {
                return Err(BooleanGateError::NotAnInput(column.into()));
            }
            let (bits, col) = word(column)?;
            io_schema.push(IoWord::input(col));
            input_cols.push(col);
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
            input_cols,
            output_cols,
            native: None,
            kernels: None,
            batched: Arc::default(),
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

    /// Hand every `kind` operation of `circuit` (as `operations` lists them,
    /// [`CompiledColumns::operations`](super::CompiledColumns::operations))
    /// to `kernel` when packing rows: the walk skips its rows and the kernel
    /// writes them in row form. Each such operation must own `kernel.width()`
    /// contiguous columns in order, its result materialized among them, and
    /// take `kernel.arg_bits()` argument bits; the rest of the gadget must
    /// read nothing of it but that result. `circuit` must be the one this
    /// gate was built from. A circuit without such operations is unchanged.
    pub fn with_kernel(
        mut self,
        circuit: &BooleanCircuit,
        operations: &[SchemaOperation],
        kind: &str,
        kernel: Arc<dyn OperationKernel>,
    ) -> Result<Self, BooleanGateError> {
        let err = |what: &str| BooleanGateError::Kernel(format!("{}: {kind}: {what}", self.name));
        let lowered = circuit.lower(LoweringMode::RequireIdentityC)?;
        let layout = lowered.layout();
        let position = |value: ValueId| {
            lowered
                .mapped_value(value)
                .and_then(|v| layout.value_position(v))
        };
        let mut sites = self
            .kernels
            .as_ref()
            .map_or_else(Vec::new, |k| k.sites.clone());
        let before = sites.len();
        for op in operations.iter().filter(|op| op.kind == kind) {
            if op.columns.len() != kernel.width() {
                return Err(err("operation width differs from the kernel's"));
            }
            let base = position(op.columns[0]).ok_or_else(|| err("unplaced column"))?;
            if op
                .columns
                .iter()
                .enumerate()
                .any(|(i, &v)| position(v) != Some(base + i))
            {
                return Err(err("columns are not contiguous in order"));
            }
            let args = op
                .inputs
                .iter()
                .flat_map(|word| &word.expressions)
                .map(|e| {
                    circuit.expressions[e.index]
                        .support
                        .iter()
                        .map(|v| {
                            position(ValueId {
                                circuit: circuit.id,
                                index: v.index(),
                            })
                            .map(|p| p as u32)
                        })
                        .collect::<Option<Vec<u32>>>()
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| err("argument outside the layout"))?;
            if args.len() != kernel.arg_bits() {
                return Err(err("argument count differs from the kernel's"));
            }
            let results = op
                .output
                .expressions
                .iter()
                .map(|e| match circuit.expressions[e.index].node {
                    ExpressionNode::Value(v) if op.columns.contains(&v) => position(v),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| err("result is not materialized among its columns"))?;
            sites.push(Site {
                kernel: kernel.clone(),
                region: base..base + kernel.width(),
                args,
                results,
            });
        }
        if sites.len() == before {
            return Ok(self);
        }
        for site in &sites {
            let inside = |p: u32| sites.iter().any(|s| s.region.contains(&(p as usize)));
            if site.args.iter().flatten().any(|&p| inside(p)) {
                return Err(err("an argument reads a kernel's columns"));
            }
        }
        let program = SlicedProgram::with_kernels(
            &self.plan,
            &sites
                .iter()
                .map(|s| KernelSite {
                    region: s.region.clone(),
                    results: s.results.clone(),
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|e| err(&e))?;
        let words = (0..self.r1cs.useful_bits.div_ceil(64))
            .map(|w| {
                let covered = (64 * w..64 * w + 64)
                    .filter(|p| sites.iter().any(|s| s.region.contains(p)))
                    .count();
                match covered {
                    0 => Cover::None,
                    64 => Cover::Full,
                    _ => Cover::Part,
                }
            })
            .collect();
        self.kernels = Some(Arc::new(Kernels {
            program,
            sites,
            words,
        }));
        self.batched = Arc::default();
        Ok(self)
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
    /// Rows are walked 64 at a time, bit-sliced, and 64×64 bit transposes
    /// turn the walk's planes into row words; the rows then stream out to
    /// the column runs and become the stripe, as the hand-written
    /// generators' rows do. Unless a kernel's result feeds the walk, the
    /// walk covers only what the kernels and the wired inputs leave
    /// ([`Batched`]): input words are copied from the rows, kernel arguments
    /// that are input bits are read from them, and the kernels run eight
    /// rows at a time after the walk.
    pub fn pack_into(&self, rows: &[Box<[F128]>], dst: SlotWitnessDest<'_>) -> Vec<u8> {
        self.pack_into_with(rows, dst, self.batched())
    }

    /// [`Self::pack_into`] as it packs when a kernel's result feeds the
    /// walk: every group walked whole, the kernels inside the walk. Writes
    /// the same slot; kept as the reference the batched packer is tested
    /// against.
    #[doc(hidden)]
    pub fn pack_into_unbatched(&self, rows: &[Box<[F128]>], dst: SlotWitnessDest<'_>) -> Vec<u8> {
        self.pack_into_with(rows, dst, None)
    }

    fn pack_into_with(
        &self,
        rows: &[Box<[F128]>],
        dst: SlotWitnessDest<'_>,
        batched: Option<&Batched>,
    ) -> Vec<u8> {
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
            dead_padding_unread,
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
        // The groups write the live stripe; dummy groups are zero unless
        // their writes may be elided.
        if !elide_padding_writes {
            stripe[live_groups * k..]
                .par_chunks_mut(1 << 16)
                .for_each(|c| c.fill(0));
        }
        // Past each group's useful words the stripe is never read when the
        // caller certifies it, and zero otherwise.
        let stripe_tail = (!dead_padding_unread).then_some(64 * useful.div_ceil(64));
        let (program, sites, words) = match &self.kernels {
            Some(k) => (&k.program, &k.sites[..], &k.words[..]),
            None => (self.plan.sliced(), &[][..], &[][..]),
        };
        let slot = SlotPtrs {
            bufs: [z, a, b].map(|buf| SendPtr(buf.as_mut_ptr())),
            stripe: SendPtr(stripe.as_mut_ptr()),
        };
        // Groups drain on the P-cores and two E-core helpers, as the
        // hand-written generators' do.
        run_hetero_chunks_stateful(n.div_ceil(64), SlicedScratch::default, |sc, g| {
            let rows = &rows[64 * g..n.min(64 * g + 64)];
            // Rows `64g..` of column `col` are `buf[(col << nu) + 64g..]`;
            // the group's stripe is its batches' `k` bytes each.
            // SAFETY: every group takes its own rows of each of the first
            // `w` columns (`n <= capacity` rows each, asserted above) and
            // its own stripe bytes, all inside the buffers, once.
            let (mut runs, stripe) = unsafe {
                let runs: [Vec<&mut [F128]>; 3] = std::array::from_fn(|side| {
                    (0..w)
                        .map(|col| {
                            std::slice::from_raw_parts_mut(
                                slot.bufs[side].get().add((col << nu) + 64 * g),
                                rows.len(),
                            )
                        })
                        .collect()
                });
                let stripe = std::slice::from_raw_parts_mut(
                    slot.stripe.get().add(8 * g * k),
                    rows.len().div_ceil(8) * k,
                );
                (runs, stripe)
            };
            match batched {
                Some(plan) => self.pack_group_batched(rows, sc, plan, sites, &mut runs, stripe),
                None => self.pack_group(rows, sc, program, sites, words, &mut runs, stripe),
            }
            if let Some(tail) = stripe_tail {
                for batch in stripe.chunks_mut(k) {
                    batch[tail..].fill(0);
                }
            }
        });
        stripe
    }

    /// The batched packing plan, worked out on first use.
    fn batched(&self) -> Option<&Batched> {
        self.batched.get_or_init(|| self.plan_batched()).as_ref()
    }

    fn plan_batched(&self) -> Option<Batched> {
        let (sites, covers) = match &self.kernels {
            Some(k) => (&k.sites[..], &k.words[..]),
            None => (&[][..], &[][..]),
        };
        let program = SlicedProgram::without_identity_rows(
            &self.plan,
            &sites
                .iter()
                .map(|s| KernelSite {
                    region: s.region.clone(),
                    results: s.results.clone(),
                })
                .collect::<Vec<_>>(),
        )
        .expect("the program compiled with its identity rows");
        if (0..sites.len()).any(|s| program.reads_result(s)) {
            return None;
        }
        let mut identity: Vec<(usize, u64)> = Vec::new();
        for &p in program.identity_rows() {
            match identity.iter_mut().find(|(w, _)| *w == p / 64) {
                Some((_, mask)) => *mask |= 1 << (p % 64),
                None => identity.push((p / 64, 1 << (p % 64))),
            }
        }
        let words = 2 * self.r1cs.useful_bits.div_ceil(128);
        let cover = |w: usize| covers.get(w).copied().unwrap_or(Cover::None);
        let input_at: HashMap<usize, (usize, usize)> = self
            .input_cols
            .iter()
            .enumerate()
            .flat_map(|(i, &col)| (0..128).map(move |bit| (128 * col + bit, (i, bit))))
            .collect();
        let args: Vec<_> = sites
            .iter()
            .map(|site| arg_runs(&site.args, &input_at))
            .collect();
        let mut walked_inputs: Vec<bool> = self
            .input_cols
            .iter()
            .map(|&col| (128 * col..128 * col + 128).any(|p| program.reads_position(p)))
            .collect();
        for (site, args) in sites.iter().zip(&args) {
            if args.is_none() {
                for &p in site.args.iter().flatten() {
                    if let Some(&(i, _)) = input_at.get(&(p as usize)) {
                        walked_inputs[i] = true;
                    }
                }
            }
        }
        let mut input_word = vec![false; words];
        let direct = self
            .input_cols
            .iter()
            .enumerate()
            .map(|(i, &col)| {
                debug_assert!(cover(2 * col) == Cover::None && cover(2 * col + 1) == Cover::None);
                input_word[2 * col] = true;
                input_word[2 * col + 1] = true;
                (i, 2 * col)
            })
            .collect();
        let (mut walked, mut covered, mut clear) =
            (Vec::new(), Vec::new(), Vec::<Range<usize>>::new());
        for w in 0..words {
            let c = cover(w);
            if c != Cover::None {
                covered.push(w);
            }
            let rows = (64 * w..64 * w + 64).any(|p| program.has_row_at(p));
            if c == Cover::Full || input_word[w] && !rows {
                continue;
            }
            walked.push(Walked {
                word: w,
                part: c == Cover::Part,
                z: !input_word[w],
            });
            match clear.last_mut() {
                Some(r) if r.end == 64 * w => r.end += 64,
                _ => clear.push(64 * w..64 * w + 64),
            }
        }
        Some(Batched {
            program,
            identity,
            walked_inputs,
            direct,
            walked,
            covered,
            clear,
            args,
        })
    }

    /// Up to 64 rows by [`Batched`]: the walk, then eight rows at a time.
    #[allow(clippy::too_many_arguments)]
    fn pack_group_batched(
        &self,
        rows: &[Box<[F128]>],
        sc: &mut SlicedScratch,
        plan: &Batched,
        sites: &[Site],
        runs: &mut [Vec<&mut [F128]>; 3],
        stripe: &mut [u8],
    ) {
        let lanes = rows.len();
        let live = if lanes == 64 { !0 } else { (1u64 << lanes) - 1 };
        let k = 1usize << self.r1cs.k_log;
        let per_block = k / 64;
        // Planes of the inputs the walk reads; the others read as zero.
        sc.in_planes.resize(128 * self.n_in, 0);
        for (w, _) in plan
            .walked_inputs
            .iter()
            .enumerate()
            .filter(|(_, walked)| **walked)
        {
            let mut lo = [0u64; 64];
            let mut hi = [0u64; 64];
            for (r, row) in rows.iter().enumerate() {
                lo[r] = row[w].lo;
                hi[r] = row[w].hi;
            }
            sc.in_planes[128 * w..128 * w + 64].copy_from_slice(&transpose64(lo));
            sc.in_planes[128 * w + 64..128 * w + 128].copy_from_slice(&transpose64(hi));
        }
        let SlicedScratch {
            in_planes,
            inputs,
            z,
            a,
            b,
            args,
            arg_rows,
            batch_rows,
            walked_rows,
            site_args,
            ..
        } = sc;
        inputs.clear();
        inputs.extend(self.input_bits.iter().map(|&(w, bit)| {
            if plan.walked_inputs[w as usize] {
                in_planes[128 * w as usize + bit as usize]
            } else {
                0
            }
        }));
        site_args.resize_with(sites.len(), Vec::new);
        // Kernel arguments the rows cannot supply are read off the planes
        // when the walk reaches the kernel.
        let walked = plan.program.run_clearing(
            inputs,
            live,
            z,
            a,
            b,
            &mut |s, z| {
                if plan.args[s].is_some() {
                    return;
                }
                let site = &sites[s];
                let arg_words = site.args.len().div_ceil(64);
                args.clear();
                args.extend(
                    site.args
                        .iter()
                        .map(|support| support.iter().fold(0, |acc, &p| acc ^ z[p as usize])),
                );
                args.resize(64 * arg_words, 0);
                let out = &mut site_args[s];
                out.resize(64 * arg_words, 0);
                for c in 0..arg_words {
                    let t = transpose64(args[64 * c..64 * c + 64].try_into().unwrap());
                    for (r, &word) in t.iter().enumerate() {
                        out[r * arg_words + c] = word;
                    }
                }
            },
            Some(&plan.clear),
        );
        if let Err(e) = walked {
            panic!(
                "{}: the gadget's constraints reject a row ({e:?})",
                self.name
            );
        }
        // The walked words in row form, side by side: `z`, `A·z`, `B·z`.
        walked_rows.clear();
        for walked in &plan.walked {
            let w = walked.word;
            for (side, planes) in [&*z, &*a, &*b].into_iter().enumerate() {
                let plane: &[u64; 64] = planes[64 * w..64 * w + 64].try_into().unwrap();
                walked_rows.push(if (side > 0 || walked.z) && plane.iter().any(|&x| x != 0) {
                    transpose64(*plane)
                } else {
                    [0; 64]
                });
            }
        }
        let [rz_all, ra_all, rb_all] = batch_rows;
        for side in [&mut *rz_all, &mut *ra_all, &mut *rb_all] {
            side.resize(8 * per_block, [0; 8]);
        }
        let useful = 64 * self.r1cs.useful_bits.div_ceil(64);
        for (batch, stripe) in stripe.chunks_mut(k).enumerate() {
            let at = 8 * batch;
            let rows = &rows[at..lanes.min(at + 8)];
            let n = rows.len();
            let window = batch * per_block..(batch + 1) * per_block;
            let (rz, ra, rb) = (
                &mut rz_all[window.clone()],
                &mut ra_all[window.clone()],
                &mut rb_all[window],
            );
            for side in [&mut *rz, &mut *ra, &mut *rb] {
                for &w in &plan.covered {
                    side[w] = [0; 8];
                }
            }
            for &(input, w) in &plan.direct {
                let half = |hi: bool| -> [u64; 8] {
                    std::array::from_fn(|j| {
                        rows.get(j)
                            .map_or(0, |row| if hi { row[input].hi } else { row[input].lo })
                    })
                };
                rz[w] = half(false);
                rz[w + 1] = half(true);
                for side in [&mut *ra, &mut *rb] {
                    side[w] = [0; 8];
                    side[w + 1] = [0; 8];
                }
            }
            for (t, walked) in walked_rows.chunks(3).zip(&plan.walked) {
                let (w, part) = (walked.word, walked.part);
                let sides = [&mut *rz, &mut *ra, &mut *rb];
                for (side, t) in sides.into_iter().zip(t).skip(!walked.z as usize) {
                    let t: &[u64; 8] = t[at..at + 8].try_into().unwrap();
                    if part {
                        for (x, &y) in side[w].iter_mut().zip(t) {
                            *x |= y;
                        }
                    } else {
                        side[w] = *t;
                    }
                }
            }
            for &(w, mask) in &plan.identity {
                for j in 0..n {
                    ra[w][j] |= rz[w][j] & mask;
                    rb[w][j] |= mask;
                }
            }
            for (s, site) in sites.iter().enumerate() {
                let arg_words = site.args.len().div_ceil(64);
                if let Some(runs) = &plan.args[s] {
                    arg_rows.clear();
                    arg_rows.resize(8 * arg_words, 0);
                    for (row, out) in rows.iter().zip(arg_rows.chunks_mut(arg_words)) {
                        for (out, runs) in out.iter_mut().zip(runs) {
                            *out = runs.iter().fold(0, |acc, r| {
                                let src = if r.hi {
                                    row[r.input].hi
                                } else {
                                    row[r.input].lo
                                };
                                acc | (src >> r.from & r.mask) << r.to
                            });
                        }
                    }
                }
                let source: &[u64] = match plan.args[s] {
                    Some(_) => arg_rows,
                    None => &site_args[s][at * arg_words..],
                };
                let row_args: [&[u64]; 8] =
                    std::array::from_fn(|j| &source[j * arg_words..][..arg_words]);
                site.kernel.fill8(row_args, site.region.start, rz, ra, rb);
                // Rows past the slot's end stay zero, kernel bits included.
                if n < 8 {
                    for side in [&mut *rz, &mut *ra, &mut *rb] {
                        for w in site.region.start / 64..site.region.end.div_ceil(64) {
                            side[w][n..].fill(0);
                        }
                    }
                }
            }
            for (w, bytes) in stripe[..useful]
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .enumerate()
            {
                transpose_8_u64s_to_64_bytes(&rz[w], bytes);
            }
        }
        // Each column's 64 rows go out as one contiguous run: a few long
        // bursts instead of a short store to every column per batch.
        for (runs, side) in runs.iter_mut().zip([&*rz_all, &*ra_all, &*rb_all]) {
            for (col, run) in runs.iter_mut().enumerate() {
                for (batch, run) in run.chunks_mut(8).enumerate() {
                    let lo = &side[batch * per_block + 2 * col];
                    let hi = &side[batch * per_block + 2 * col + 1];
                    let out: [F128; 8] = std::array::from_fn(|j| F128::new(lo[j], hi[j]));
                    if let Ok(run) = <&mut [F128; 8]>::try_from(&mut *run) {
                        store_streaming(run, &out);
                    } else {
                        run.copy_from_slice(&out[..run.len()]);
                    }
                }
            }
        }
    }

    /// Up to 64 rows: the walk fills `sc`'s planes but for the kernel
    /// sites, whose rows the kernels write in row form into `sc.rows`.
    /// Every word the kernels do not fully cover (`words`, missing words
    /// uncovered) is then transposed into row form too, and the rows go out
    /// to the column runs and the stripe.
    #[allow(clippy::too_many_arguments)]
    fn pack_group(
        &self,
        rows: &[Box<[F128]>],
        sc: &mut SlicedScratch,
        program: &SlicedProgram,
        sites: &[Site],
        words: &[Cover],
        runs: &mut [Vec<&mut [F128]>; 3],
        stripe: &mut [u8],
    ) {
        let lanes = rows.len();
        let live = self.input_planes(rows, sc);
        let k = 1usize << self.r1cs.k_log;
        let per_block = k / 64;
        let batches = lanes.div_ceil(8);
        let SlicedScratch {
            inputs,
            z,
            a,
            b,
            rows: rm,
            args,
            arg_rows,
            ..
        } = sc;
        for side in rm.iter_mut() {
            side.resize(8 * per_block, [0; 8]);
            for batch in side.chunks_mut(per_block) {
                for (word, cover) in batch.iter_mut().zip(words) {
                    if *cover != Cover::None {
                        *word = [0; 8];
                    }
                }
            }
        }
        let [rz, ra, rb] = rm;
        let walked = program.run(inputs, live, z, a, b, &mut |s, z| {
            let site = &sites[s];
            let arg_words = site.args.len().div_ceil(64);
            args.clear();
            args.extend(
                site.args
                    .iter()
                    .map(|support| support.iter().fold(0, |acc, &p| acc ^ z[p as usize])),
            );
            args.resize(64 * arg_words, 0);
            arg_rows.resize(64 * arg_words, 0);
            for c in 0..arg_words {
                let t = transpose64(args[64 * c..64 * c + 64].try_into().unwrap());
                for (r, &word) in t.iter().enumerate() {
                    arg_rows[r * arg_words + c] = word;
                }
            }
            for batch in 0..batches {
                let row_args: [&[u64]; 8] =
                    std::array::from_fn(|j| &arg_rows[(8 * batch + j) * arg_words..][..arg_words]);
                let window = batch * per_block..(batch + 1) * per_block;
                site.kernel.fill8(
                    row_args,
                    site.region.start,
                    &mut rz[window.clone()],
                    &mut ra[window.clone()],
                    &mut rb[window],
                );
            }
            // Rows past the slot's end stay zero, kernel bits included.
            if !lanes.is_multiple_of(8) {
                let batch = lanes / 8;
                for side in [&mut *rz, &mut *ra, &mut *rb] {
                    for w in site.region.start / 64..site.region.end.div_ceil(64) {
                        for lane in side[batch * per_block + w][lanes % 8..].iter_mut() {
                            *lane = 0;
                        }
                    }
                }
            }
            // The result, back into the walk's planes, a word at a time.
            let mut planes = (usize::MAX, [0u64; 64]);
            for &p in &site.results {
                let w = p / 64;
                if planes.0 != w {
                    planes = (w, transpose64(std::array::from_fn(|r| row_word(rz, r, w))));
                }
                z[p] = planes.1[p % 64];
            }
        });
        if let Err(e) = walked {
            panic!(
                "{}: the gadget's constraints reject a row ({e:?})",
                self.name
            );
        }
        let columns = runs[0].len();
        for (planes, side) in [&*z, &*a, &*b]
            .into_iter()
            .zip([&mut *rz, &mut *ra, &mut *rb])
        {
            for w in 0..2 * columns {
                let cover = *words.get(w).unwrap_or(&Cover::None);
                if cover == Cover::Full {
                    continue;
                }
                let plane: &[u64; 64] = planes[64 * w..64 * w + 64].try_into().unwrap();
                let t = if plane.iter().any(|&x| x != 0) {
                    transpose64(*plane)
                } else {
                    [0; 64]
                };
                for (batch, t) in t.as_chunks::<8>().0.iter().enumerate() {
                    let word = &mut side[batch * per_block + w];
                    if cover == Cover::None {
                        *word = *t;
                    } else {
                        for (word, &t) in word.iter_mut().zip(t) {
                            *word |= t;
                        }
                    }
                }
            }
        }
        for (runs, side) in runs.iter_mut().zip([&*rz, &*ra, &*rb]) {
            for (col, run) in runs.iter_mut().enumerate() {
                for (batch, run) in run.chunks_mut(8).enumerate() {
                    let lo = &side[batch * per_block + 2 * col];
                    let hi = &side[batch * per_block + 2 * col + 1];
                    let out: [F128; 8] = std::array::from_fn(|j| F128::new(lo[j], hi[j]));
                    if let Ok(run) = <&mut [F128; 8]>::try_from(&mut *run) {
                        store_streaming(run, &out);
                    } else {
                        run.copy_from_slice(&out[..run.len()]);
                    }
                }
            }
        }
        let useful = 64 * self.r1cs.useful_bits.div_ceil(64);
        for (g, group) in stripe.chunks_mut(k).enumerate() {
            let rows = &rz[g * per_block..];
            for (w, bytes) in group[..useful]
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .enumerate()
            {
                transpose_8_u64s_to_64_bytes(&rows[w], bytes);
            }
        }
    }

    /// The rows' wired input words as walk input planes in `sc.inputs`;
    /// returns the live-lane mask.
    fn input_planes(&self, rows: &[Box<[F128]>], sc: &mut SlicedScratch) -> u64 {
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
        live
    }
}

/// A raw pointer the group drivers share; each writes only its own part.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
// SAFETY: the drivers write disjoint ranges (see `pack_into_with`).
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// The slot's buffers (`z`, `A·z`, `B·z`) and its stripe.
#[derive(Clone, Copy)]
struct SlotPtrs {
    bufs: [SendPtr<F128>; 3],
    stripe: SendPtr<u8>,
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
    /// Kernel rows in row form, eight rows interleaved per word, eight
    /// batches: `z`, `A·z`, `B·z`.
    rows: [Vec<[u64; 8]>; 3],
    /// A kernel's argument planes, then its arguments row by row.
    args: Vec<u64>,
    arg_rows: Vec<u64>,
    /// The batched packer's rows of one batch: `z`, `A·z`, `B·z`.
    batch_rows: [Vec<[u64; 8]>; 3],
    /// The batched packer's walked words in row form, three sides each.
    walked_rows: Vec<[u64; 64]>,
    /// Per kernel site, the group's arguments read off the planes, row by
    /// row.
    site_args: Vec<Vec<u64>>,
}

/// A kernel site's arguments as runs of input bits per argument word, if
/// each argument bit is one input bit or zero; `input_at` maps a block
/// position to its (input, bit).
fn arg_runs(
    args: &[Vec<u32>],
    input_at: &HashMap<usize, (usize, usize)>,
) -> Option<Vec<Vec<ArgRun>>> {
    let mut words: Vec<Vec<ArgRun>> = vec![Vec::new(); args.len().div_ceil(64)];
    for (i, support) in args.iter().enumerate() {
        let (input, bit) = match support[..] {
            [] => continue,
            [p] => *input_at.get(&(p as usize))?,
            _ => return None,
        };
        let (hi, from, to) = (bit >= 64, (bit % 64) as u32, (i % 64) as u32);
        let runs = &mut words[i / 64];
        match runs.last_mut() {
            Some(r)
                if r.input == input
                    && r.hi == hi
                    && r.from + r.mask.count_ones() == from
                    && r.to + r.mask.count_ones() == to =>
            {
                r.mask = r.mask << 1 | 1;
            }
            _ => runs.push(ArgRun {
                input,
                hi,
                from,
                to,
                mask: 1,
            }),
        }
    }
    Some(words)
}

/// `*dst = *src` for a run the caller will not read back soon: on aarch64 a
/// non-temporal store, which skips reading the destination lines in.
#[inline(always)]
fn store_streaming(dst: &mut [F128; 8], src: &[F128; 8]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: both are 128 valid bytes, and `dst` is exclusively borrowed.
    unsafe {
        std::arch::asm!(
            "ldp {t0:q}, {t1:q}, [{s}]",
            "stnp {t0:q}, {t1:q}, [{d}]",
            "ldp {t0:q}, {t1:q}, [{s}, #32]",
            "stnp {t0:q}, {t1:q}, [{d}, #32]",
            "ldp {t0:q}, {t1:q}, [{s}, #64]",
            "stnp {t0:q}, {t1:q}, [{d}, #64]",
            "ldp {t0:q}, {t1:q}, [{s}, #96]",
            "stnp {t0:q}, {t1:q}, [{d}, #96]",
            s = in(reg) src.as_ptr(),
            d = in(reg) dst.as_mut_ptr(),
            t0 = out(vreg) _,
            t1 = out(vreg) _,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        *dst = *src;
    }
}

/// Row `r`'s word `w` of 8-row interleaved row words, `per_block` words a
/// batch.
fn row_word(rows: &[[u64; 8]], r: usize, w: usize) -> u64 {
    rows[(r / 8) * (rows.len() / 8) + w][r % 8]
}

/// Transpose a 64×64 bit matrix: bit `c` of `out[r]` is bit `r` of `m[c]`.
///
/// Six rounds, each swapping one bit of the row index with the same bit of
/// the column index; the rounds commute.
#[cfg(not(target_arch = "aarch64"))]
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

/// Transpose a 64×64 bit matrix: bit `c` of `out[r]` is bit `r` of `m[c]`.
///
/// Six rounds, each swapping one bit of the row index with the same bit of
/// the column index; the rounds commute. Register `i` holds rows `i` and
/// `i + 32`, so the 32-wide round is one byte shuffle per register and
/// every other round pairs registers: first the 16- and 8-wide rounds over
/// each set of four registers they mix, then the rest over each run of
/// eight, so either pass fits in registers.
#[cfg(target_arch = "aarch64")]
fn transpose64(m: [u64; 64]) -> [u64; 64] {
    use std::arch::aarch64::{
        uint64x2_t, vandq_u64, vcombine_u64, vdupq_n_u64, veorq_u64, vld1_u64, vld1q_u8,
        vqtbl1q_u8, vreinterpretq_u8_u64, vreinterpretq_u64_u8, vshlq_n_u64, vshrq_n_u64,
        vst1q_u64, vzip1q_u64, vzip2q_u64,
    };
    #[inline(always)]
    unsafe fn swap<const WIDTH: i32>(a: &mut uint64x2_t, b: &mut uint64x2_t, mask: u64) {
        unsafe {
            let t = vandq_u64(veorq_u64(vshrq_n_u64::<WIDTH>(*a), *b), vdupq_n_u64(mask));
            *a = veorq_u64(*a, vshlq_n_u64::<WIDTH>(t));
            *b = veorq_u64(*b, t);
        }
    }
    // SAFETY: NEON is baseline on aarch64, and every load and store stays
    // inside `m`, `v`, `out` or the shuffle table.
    unsafe {
        let mut v = [vdupq_n_u64(0); 32];
        for i in 0..8 {
            let at = |j: usize| m.as_ptr().add(i + 8 * j);
            let mut r: [uint64x2_t; 4] =
                std::array::from_fn(|j| vcombine_u64(vld1_u64(at(j)), vld1_u64(at(j).add(32))));
            let [r0, r1, r2, r3] = &mut r;
            swap::<16>(r0, r2, 0x0000_FFFF_0000_FFFF);
            swap::<16>(r1, r3, 0x0000_FFFF_0000_FFFF);
            swap::<8>(r0, r1, 0x00FF_00FF_00FF_00FF);
            swap::<8>(r2, r3, 0x00FF_00FF_00FF_00FF);
            for (j, r) in r.into_iter().enumerate() {
                v[i + 8 * j] = r;
            }
        }
        // `[a_lo, a_hi, b_lo, b_hi]` as 32-bit halves becomes
        // `[a_lo, b_lo, a_hi, b_hi]`.
        let halves: [u8; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15];
        let halves = vld1q_u8(halves.as_ptr());
        let mut out = [0u64; 64];
        for (b, block) in v.as_chunks::<8>().0.iter().enumerate() {
            let mut r = *block;
            for x in r.iter_mut() {
                *x = vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(*x), halves));
            }
            for i in [0, 1, 2, 3] {
                let (lo, hi) = r.split_at_mut(4);
                swap::<4>(&mut lo[i], &mut hi[i], 0x0F0F_0F0F_0F0F_0F0F);
            }
            for i in [0, 1, 4, 5] {
                let (lo, hi) = r.split_at_mut(i + 2);
                swap::<2>(&mut lo[i], &mut hi[0], 0x3333_3333_3333_3333);
            }
            for i in [0, 2, 4, 6] {
                let (lo, hi) = r.split_at_mut(i + 1);
                swap::<1>(&mut lo[i], &mut hi[0], 0x5555_5555_5555_5555);
            }
            let first = 8 * b;
            for j in 0..4 {
                let (a, b) = (r[2 * j], r[2 * j + 1]);
                vst1q_u64(out.as_mut_ptr().add(first + 2 * j), vzip1q_u64(a, b));
                vst1q_u64(out.as_mut_ptr().add(32 + first + 2 * j), vzip2q_u64(a, b));
            }
        }
        out
    }
}

#[cfg(test)]
fn transpose64_scalar(mut m: [u64; 64]) -> [u64; 64] {
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

    #[test]
    #[ignore = "timing; run explicitly"]
    fn transpose64_cost() {
        let mut state = 7u64;
        let mut m: [u64; 64] = std::array::from_fn(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        });
        for (name, f) in [
            ("fast", super::transpose64 as fn([u64; 64]) -> [u64; 64]),
            ("scalar", super::transpose64_scalar),
        ] {
            let t = std::time::Instant::now();
            for _ in 0..1_000_000 {
                m = std::hint::black_box(f(std::hint::black_box(m)));
            }
            eprintln!("{name}: {:.1} ns", t.elapsed().as_nanos() as f64 / 1e6);
        }
    }
}
