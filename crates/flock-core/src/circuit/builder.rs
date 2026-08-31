//! Describing circuits by *building* them, rather than by hand-writing cells.
//!
//! [`Circuit::new`] takes the wiring as raw equivalence classes of [`Cell`],
//! and the witness is built by separate code. Two artifacts that must agree
//! with nothing enforcing it — a mismatch is a silently wrong *statement*, not
//! a compile error. See `docs/circuit-wiring-design.tex`
//! §"Describing circuits: the builder".
//!
//! The fix is the principle behind [`crate::transcript_record`]: **one
//! description, both views**. A [`GateType`] carries its constraints and its
//! native evaluation together, so instantiating a gate emits the row, the
//! wiring and the witness from a single call and they cannot drift. `counts`
//! and the equivalence classes fall out of construction instead of being
//! passed in.
//!
//! ```ignore
//! let mut b = CircuitBuilder::new(nu);
//! let mult = b.slot(MultGate { kappa });
//! let mut acc = b.public_value(seed);
//! for &a in &multipliers {
//!     let a_w = b.public_value(a);
//!     acc = b.gate(mult, &[a_w, acc])[0];
//! }
//! b.publish(acc);
//! let built = b.finish();
//! ```
//!
//! ## Determinism
//!
//! Rows are allocated in gate-instantiation order and cell-slots enumerate in
//! registry order, so the same sequence of calls always produces the same
//! [`Circuit::digest`]. That matters because the digest is statement-binding:
//! a *regenerated* circuit must be the SAME circuit, not merely an equivalent
//! one.

use crate::alloc_zeroed_vec;
use core::ops::Range;
use std::any::{Any, TypeId, type_name};
use std::cmp::Reverse;
use std::env::var;
use std::mem::take;
use std::sync::Arc;
use std::time::Instant;

use crate::field::F128;
use crate::schedule::{IoDirection, Registry, TableType};

use super::{Cell, Circuit, CircuitError};
use rayon::prelude::*;

/// A value in the circuit, and the cells that must hold it.
///
/// A wire IS an equivalence class under construction: binding it as a gate
/// input appends that gate's input cell to the class, and the builder hands
/// the finished classes to [`Circuit::new`].
///
/// Wires are usable before their producer is emitted — a value can be consumed
/// by a gate declared earlier in the program than the one that defines it,
/// because a class is just a set. The Fiat–Shamir chain needs this: a value a
/// hash row derives is consumed by gates emitted earlier in the program than
/// that row. In the online
/// phase such a wire takes its value from the input that supplies it, and the
/// producing gate's output is then *checked* against it rather than overwriting
/// it — see [`CircuitShape::run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Wire(usize);

/// A declared gate slot. Indexes the builder's DECLARATION order, which is not
/// the registry's slot order — see [`CircuitShape::registry_slot`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId(usize);

/// One slot's committed witness, in the form its class's prover input wants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlotWitness {
    /// Element slot: the committed words in the BatchMajor rows-low layout,
    /// `word[(col << nu) + row]`, length `width << nu`. Feeds
    /// `UnionElementSlotInput`'s closure directly.
    Element(Vec<F128>),
    /// The gate type does not pack its own witness. Boolean slots are
    /// bit-packed by the hash modules' `generate_witness_batch_major*`, which
    /// lives in `flock-prover`, above this crate — so the builder cannot
    /// produce those buffers and does not pretend to. Recover the typed rows
    /// with [`CircuitWitness::rows`] and hand them to that generator.
    DeferredToRows,
}

impl SlotWitness {
    /// The element-slot witness of per-row words: `word[(col << nu) + row]`
    /// over a `width << nu` buffer, rows in declaration order. Every
    /// element gate's `witness()` is this call.
    pub fn element_from_rows<R: AsRef<[F128]>>(width: usize, nu: usize, rows: &[R]) -> Self {
        let mut z = alloc_zeroed_vec::<F128>(width << nu);
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.as_ref().iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// A gate type: its constraint system and its native evaluation, together.
///
/// The pairing is the whole point. A type that could describe its constraints
/// but not evaluate them would put the witness back in separate code, which is
/// the failure this module exists to remove.
pub trait GateType {
    /// What one gate contributes to its slot's witness. Kept abstract because
    /// witnesses do not decompose uniformly per row: element slots are plain
    /// `F128` words, while boolean slots are bit-packed in bulk by their own
    /// `generate_witness`. The builder collects `Row`s in order and lets the
    /// gate type emit the slot's witness once.
    type Row;

    /// Nondeterministic advice for [`eval`](GateType::eval): data the gate
    /// needs to run that does not travel on a wire.
    ///
    /// Wires carry whole 128-bit words at word-aligned schema positions, so
    /// only word-aligned data is wireable at all — and of that, only data some
    /// other gate produces or consumes has any reason to be. A Merkle opening
    /// is the motivating case: its leaf, index and root are wired, but its
    /// sibling digests are read by nothing else and sit unaligned in each
    /// node's padding. They are supplied here instead.
    ///
    /// A hint is invisible to the statement. The constraints still pin
    /// everything the relation depends on, so a wrong hint yields a row that
    /// fails to satisfy them — it cannot buy a false proof, only a broken one.
    /// Gates that need no advice set this to `()` and are instantiated with
    /// [`ShapeBuilder::gate`].
    type Hint;

    /// The registry type: constraints, width, and the `io_schema` whose order
    /// defines this gate's input and output positions.
    fn table(&self) -> TableType;

    /// Evaluate one gate. `inputs` are the schema's `In` words in schema
    /// order; the `Out` words are PUSHED onto `outputs` in schema order (the
    /// caller hands a cleared scratch — a fresh Vec per call was a measurable
    /// slice of the online phase at ~10^5 calls per proof); returns the row
    /// record. `hint` is this instance's advice — see [`Hint`](GateType::Hint).
    fn eval(&self, inputs: &[F128], hint: &Self::Hint, outputs: &mut Vec<F128>) -> Self::Row;

    /// The slot's committed witness, given every row in instantiation order
    /// and the uniform capacity `nu`. Rows `[rows.len(), 2^nu)` are dummy and
    /// must be written as zeros — the PIOP sums over the whole region.
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness;
}

/// Object-safe view of a declared slot, erasing `GateType::Row` and
/// `GateType::Hint`.
///
/// **Stateless.** Rows accumulate in the online phase, not here, so one shape
/// can be run many times concurrently — the whole point of the split.
trait SlotBuild: Any + Send + Sync {
    /// MOVE the table out. Called once, by `finish`, on its way into the
    /// registry — which then owns it. Cloning here instead cost 2 deep copies
    /// of every table's matrices; BLAKE3's are ~21M nonzeros, so that was
    /// ~300 ms of pure memcpy per circuit.
    fn take_table(&mut self) -> TableType;
    fn n_in(&self) -> usize;
    fn n_out(&self) -> usize;
    /// A fresh, empty `Vec<G::Row>` for one online run.
    fn new_rows(&self) -> Box<dyn Any + Send>;
    /// Evaluate one gate, appending its row; outputs land on the scratch.
    fn push(&self, rows: &mut dyn Any, inputs: &[F128], hint: &dyn Any, outputs: &mut Vec<F128>);
    /// Evaluate `n` gates of this slot against the value tape in one
    /// monomorphic loop — the [`FillPlan`] runner. `in_idx`/`out_idx` are the
    /// gates' pre-resolved tape indices (`n_in`/`n_out` per gate, gate-major);
    /// an output index with [`FILL_CHECK`] set addresses a class that already
    /// has a value, and is asserted against instead of written. Hinted slots
    /// consume `hints[hint_base..hint_base + n]`.
    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        rows: &mut dyn Any,
        values: &mut [F128],
        n: usize,
        in_idx: &[u32],
        out_idx: &[u32],
        hints: &[&(dyn Any + Sync)],
        hint_base: usize,
        hinted: bool,
        scratch_in: &mut Vec<F128>,
        scratch_out: &mut Vec<F128>,
    );
    fn witness(&self, rows: &dyn Any, nu: usize) -> SlotWitness;
    /// Append `src`'s rows (an island accumulator) onto `dst`, preserving
    /// instantiation order.
    fn merge_rows(&self, dst: &mut dyn Any, src: Box<dyn Any + Send>);
}

struct GateSlot<G: GateType> {
    gate: G,
    /// `None` after `finish` has moved it into the registry.
    table: Option<TableType>,
    n_in: usize,
    n_out: usize,
}

impl<G: GateType + Send + Sync + 'static> SlotBuild for GateSlot<G>
where
    G::Row: Send + 'static,
    G::Hint: 'static,
{
    fn take_table(&mut self) -> TableType {
        self.table
            .take()
            .expect("table already moved into the registry")
    }
    fn n_in(&self) -> usize {
        self.n_in
    }
    fn n_out(&self) -> usize {
        self.n_out
    }
    fn new_rows(&self) -> Box<dyn Any + Send> {
        Box::new(Vec::<G::Row>::new())
    }
    fn push(&self, rows: &mut dyn Any, inputs: &[F128], hint: &dyn Any, outputs: &mut Vec<F128>) {
        let rows = rows
            .downcast_mut::<Vec<G::Row>>()
            .expect("row store belongs to another slot");
        let hint = hint.downcast_ref::<G::Hint>().unwrap_or_else(|| {
            panic!(
                "gate expects a hint of type {}; use gate_hinted and supply one",
                type_name::<G::Hint>()
            )
        });
        let row = self.gate.eval(inputs, hint, outputs);
        assert_eq!(
            outputs.len(),
            self.n_out,
            "gate returned {} outputs, schema declares {}",
            outputs.len(),
            self.n_out
        );
        rows.push(row);
    }
    fn run_batch(
        &self,
        rows: &mut dyn Any,
        values: &mut [F128],
        n: usize,
        in_idx: &[u32],
        out_idx: &[u32],
        hints: &[&(dyn Any + Sync)],
        hint_base: usize,
        hinted: bool,
        scratch_in: &mut Vec<F128>,
        scratch_out: &mut Vec<F128>,
    ) {
        let rows = rows
            .downcast_mut::<Vec<G::Row>>()
            .expect("row store belongs to another slot");
        let unit = ();
        for g in 0..n {
            scratch_in.clear();
            for &i in &in_idx[g * self.n_in..(g + 1) * self.n_in] {
                scratch_in.push(values[i as usize]);
            }
            let hint_any: &dyn Any = if hinted { hints[hint_base + g] } else { &unit };
            let hint = hint_any.downcast_ref::<G::Hint>().unwrap_or_else(|| {
                panic!(
                    "gate expects a hint of type {}; use gate_hinted and supply one",
                    type_name::<G::Hint>()
                )
            });
            scratch_out.clear();
            let row = self.gate.eval(scratch_in, hint, scratch_out);
            assert_eq!(
                scratch_out.len(),
                self.n_out,
                "gate returned {} outputs, schema declares {}",
                scratch_out.len(),
                self.n_out
            );
            for (k, &v) in scratch_out.iter().enumerate() {
                let oi = out_idx[g * self.n_out + k];
                let r = (oi & !FILL_CHECK) as usize;
                if oi & FILL_CHECK != 0 {
                    assert_eq!(
                        values[r], v,
                        "a connected wire disagrees with the gate output that produces it"
                    );
                } else {
                    values[r] = v;
                }
            }
            rows.push(row);
        }
    }
    fn witness(&self, rows: &dyn Any, nu: usize) -> SlotWitness {
        self.gate.witness(
            rows.downcast_ref::<Vec<G::Row>>()
                .expect("row store belongs to another slot"),
            nu,
        )
    }
    fn merge_rows(&self, dst: &mut dyn Any, src: Box<dyn Any + Send>) {
        let dst = dst
            .downcast_mut::<Vec<G::Row>>()
            .expect("row store belongs to another slot");
        let mut src = src
            .downcast::<Vec<G::Row>>()
            .expect("row store belongs to another slot");
        dst.append(&mut src);
    }
}

/// One recorded gate instantiation. Wire indices, not values — this is the
/// value-independent half.
#[derive(Clone)]
struct Step {
    slot: usize,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    hinted: bool,
}

/// Marks a [`FillPlan`] output index whose class already holds a value at
/// that point in the program (the Fiat–Shamir forward reference, or a
/// [`ShapeBuilder::connect`] between two producers): the runner asserts
/// equality instead of writing.
const FILL_CHECK: u32 = 1 << 31;

/// A maximal run of consecutive same-slot, same-hintedness steps — one
/// monomorphic [`SlotBuild::run_batch`] call. Batches never straddle an
/// island boundary.
struct FillBatch {
    slot: u32,
    n: u32,
    in_off: u32,
    out_off: u32,
    /// Hinted steps before this batch: the ordinal of the batch's first hint.
    hint_base: u32,
    hinted: bool,
}

/// One declared island, compiled for parallel execution on a COMPACT tape of
/// its own: every root the island touches gets a local index, its batches'
/// `in_idx`/`out_idx` entries are rewritten to those, and the global tape is
/// only touched by the gather before and the scatter after. This replaces the
/// walk's full value-state clone and full-width merge scan with copies
/// proportional to what the island actually reads and writes — and the
/// independence contract moves to compile time: an island reading (or
/// checking against) a wire another island writes fails `fill_plan`, not the
/// proof.
struct FillIsland {
    /// This island's batches: `batches[range.0..range.1]`.
    batches: (u32, u32),
    /// Pre-island values copied onto the local tape: `(local, global)`.
    gather: Vec<(u32, u32)>,
    /// The island's writes, copied back after: `(local, global)`.
    scatter: Vec<(u32, u32)>,
    tape_len: u32,
}

/// **The index-fill runner's program**: one shape's wire traffic, resolved to
/// value-tape indices at setup.
///
/// The generic walk ([`CircuitShape::run`]) pays value-independent work per
/// gate per proof — resolving wires, tracking definedness, downcasting the
/// row store and hint, checking output arity. All of it is a function of the
/// shape alone, so [`CircuitShape::fill_plan`] pays it once: inputs become
/// `(root, ordinal)` copy pairs, every gate's reads and writes become flat
/// index arrays, and consecutive same-slot gates coalesce into batches that
/// [`CircuitShape::run_filled`] streams through one monomorphic loop each.
/// Definedness moves entirely to compile time — a read before any write
/// fails `fill_plan`, not the proof, and an output cell whose class is
/// already valued is marked [`FILL_CHECK`] so the runner asserts equality
/// exactly where the walk would.
///
/// The plan is data about the shape, not a second semantics: `run` stays as
/// the differential oracle, and the two produce identical
/// [`CircuitWitness`]es — publics, rows and witnesses.
pub struct FillPlan {
    batches: Vec<FillBatch>,
    /// Concatenated input-cell tape indices, `n_in` per gate, batch-major.
    /// Island batches address their island's LOCAL tape.
    in_idx: Vec<u32>,
    /// Concatenated output tape indices, `n_out` per gate; see [`FILL_CHECK`].
    out_idx: Vec<u32>,
    /// First supply of each input class: `(root, input ordinal)`.
    input_fills: Vec<(u32, u32)>,
    /// Later supplies of an already-filled class — the walk's "connected
    /// inputs" equality, precomputed to just the duplicated pairs.
    input_checks: Vec<(u32, u32)>,
    /// Compiled islands, run in parallel like the walk's. Empty when fewer
    /// than two were declared (the walk runs those sequentially too).
    islands: Vec<FillIsland>,
    /// Fingerprint of the shape this plan was compiled from.
    n_steps: usize,
    n_wires: usize,
}

// ---------------------------------------------------------------------------
// Setup phase
// ---------------------------------------------------------------------------

/// Builds the value-independent half of a circuit: which gates exist, how they
/// are wired, and what is public. No field arithmetic happens here.
///
/// See [`CircuitShape`] for why the split exists. For a one-shot circuit where
/// separating the phases buys nothing, [`CircuitBuilder`] is the same thing
/// with values supplied inline.
pub struct ShapeBuilder {
    nu: usize,
    slots: Vec<Box<dyn SlotBuild>>,
    slot_types: Vec<TypeId>,
    /// Cells per wire. A wire's value lives in the online phase, not here.
    wires: Vec<Vec<Cell>>,
    /// Union-find over wires, so [`ShapeBuilder::connect`] can merge two
    /// equivalence classes that were created independently.
    parent: Vec<usize>,
    public: Vec<Wire>,
    fixed_public: Vec<Option<F128>>,
    inputs: Vec<Wire>,
    steps: Vec<Step>,
    rows_per_slot: Vec<usize>,
    n_hints: usize,
    /// Step spans whose gate subgraphs are mutually independent (each reads
    /// only wires written before the first island or inside itself) — the
    /// online phase evaluates them in parallel. Declared by the caller via
    /// [`ShapeBuilder::begin_island`]/[`ShapeBuilder::end_island`].
    islands: Vec<(usize, usize)>,
}

impl ShapeBuilder {
    pub fn new(nu: usize) -> Self {
        Self {
            nu,
            slots: Vec::new(),
            slot_types: Vec::new(),
            wires: Vec::new(),
            parent: Vec::new(),
            public: Vec::new(),
            fixed_public: Vec::new(),
            inputs: Vec::new(),
            steps: Vec::new(),
            rows_per_slot: Vec::new(),
            n_hints: 0,
            islands: Vec::new(),
        }
    }

    fn find(&mut self, w: Wire) -> usize {
        let mut r = w.0;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        let mut c = w.0;
        while self.parent[c] != r {
            let next = self.parent[c];
            self.parent[c] = r;
            c = next;
        }
        r
    }

    fn new_wire(&mut self, cells: Vec<Cell>) -> Wire {
        self.wires.push(cells);
        self.parent.push(self.wires.len() - 1);
        Wire(self.wires.len() - 1)
    }

    /// Declare a gate slot. Every gate of this type shares the slot, and the
    /// slot's row capacity is the registry's uniform `2^nu`.
    pub fn slot<G>(&mut self, gate: G) -> SlotId
    where
        G: GateType + Send + Sync + 'static,
        G::Row: Send + 'static,
        G::Hint: 'static,
    {
        let table = gate.table();
        let n_in = table
            .io_schema
            .iter()
            .filter(|w| w.dir == IoDirection::In)
            .count();
        let n_out = table.io_schema.len() - n_in;
        assert!(
            !table.io_schema.is_empty(),
            "a gate slot needs an io_schema; a type with none is unwireable"
        );
        self.slots.push(Box::new(GateSlot {
            gate,
            table: Some(table),
            n_in,
            n_out,
        }));
        self.slot_types.push(TypeId::of::<G>());
        self.rows_per_slot.push(0);
        SlotId(self.slots.len() - 1)
    }

    /// A declared slot's input arity — what a caller emitting gates
    /// generically (e.g. count padding: fixed declared counts reached by
    /// emitting zero-input gates) needs to size the input list.
    pub fn slot_inputs(&self, s: SlotId) -> usize {
        self.slots[s.0].n_in()
    }

    /// A free value entering the circuit. It gets no producing cell, so it
    /// must be constrained by something — published, or consumed by a gate
    /// whose relation pins it.
    ///
    /// The online phase supplies one `F128` per `input()` call, in call order.
    pub fn input(&mut self) -> Wire {
        let w = self.new_wire(Vec::new());
        self.inputs.push(w);
        w
    }

    /// A value that is both free and public — the common case for circuit
    /// inputs.
    pub fn public_input(&mut self) -> Wire {
        let w = self.input();
        self.publish(w);
        w
    }

    /// A public input whose value is part of the circuit definition. The
    /// value is committed by [`Circuit::digest`](super::Circuit::digest) and
    /// checked at the prove/verify boundary.
    pub fn fixed_public_input(&mut self, value: F128) -> Wire {
        let w = self.input();
        self.public.push(w);
        self.fixed_public.push(Some(value));
        w
    }

    /// Instantiate a gate: allocate a row, bind `inputs` to its input cells,
    /// and return wires for its outputs. For a gate type whose
    /// [`Hint`](GateType::Hint) is `()`; use [`gate_hinted`] otherwise.
    ///
    /// [`gate_hinted`]: ShapeBuilder::gate_hinted
    pub fn gate(&mut self, slot: SlotId, inputs: &[Wire]) -> Vec<Wire> {
        self.emit(slot, inputs, false)
    }

    /// Instantiate a gate that consumes advice. The online phase supplies one
    /// hint per `gate_hinted` call, in call order. See [`GateType::Hint`].
    pub fn gate_hinted(&mut self, slot: SlotId, inputs: &[Wire]) -> Vec<Wire> {
        self.n_hints += 1;
        self.emit(slot, inputs, true)
    }

    fn emit(&mut self, slot: SlotId, inputs: &[Wire], hinted: bool) -> Vec<Wire> {
        let s = &self.slots[slot.0];
        assert_eq!(
            inputs.len(),
            s.n_in(),
            "gate takes {} inputs, got {}",
            s.n_in(),
            inputs.len()
        );
        let n_in = s.n_in();
        let n_out = s.n_out();
        let row = self.rows_per_slot[slot.0];
        assert!(
            row < (1usize << self.nu),
            "slot {} exceeded its 2^{} row capacity",
            slot.0,
            self.nu
        );

        // Cells are assigned once the registry order is known; record the
        // (declared slot, schema index, row) triple and resolve in `finish`.
        for (k, w) in inputs.iter().enumerate() {
            self.wires[w.0].push(Cell::new(encode(slot.0, k), row));
        }
        let outputs: Vec<Wire> = (0..n_out)
            .map(|k| self.new_wire(vec![Cell::new(encode(slot.0, n_in + k), row)]))
            .collect();
        self.rows_per_slot[slot.0] += 1;
        self.steps.push(Step {
            slot: slot.0,
            inputs: inputs.iter().map(|w| w.0).collect(),
            outputs: outputs.iter().map(|w| w.0).collect(),
            hinted,
        });
        outputs
    }

    /// Publish a wire: it joins the public segment, in call order.
    pub fn publish(&mut self, w: Wire) {
        self.public.push(w);
        self.fixed_public.push(None);
    }

    /// How many public entries exist so far — the index the NEXT
    /// [`Self::publish`] or [`Self::public_input`] will land at. Lets a
    /// caller emitting several independent regions into one builder record
    /// where each region's public block starts, instead of reconstructing
    /// offsets from counts after the fact.
    pub fn public_len(&self) -> usize {
        self.public.len()
    }

    /// Rows emitted so far into slot `s` — lets census instrumentation
    /// attribute a SHARED slot's rows to the regions that emitted them,
    /// the same way [`Self::public_len`] brackets public blocks.
    pub fn rows_in_slot(&self, s: SlotId) -> usize {
        self.rows_per_slot[s.0]
    }

    /// Assert two wires carry the same value: merge their classes, so the
    /// wiring argument enforces it.
    ///
    /// This is the circuit's `assert_eq`. It is also how an inverse is
    /// expressed — witness `y`, emit `x·y`, and connect that product to a
    /// public cell holding 1 — so no inversion gate is needed.
    ///
    /// The value check this used to make eagerly now happens in
    /// [`CircuitShape::run`], because there are no values here.
    pub fn connect(&mut self, a: Wire, b: Wire) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let cells = take(&mut self.wires[rb]);
        self.wires[ra].extend(cells);
        self.parent[rb] = ra;
    }

    /// Open an independence island: the steps recorded until the matching
    /// [`Self::end_island`] must read only wires valued BEFORE the first
    /// island (inputs or earlier non-island gate outputs) or produced inside
    /// this island. The online phase runs islands in parallel; a violated
    /// contract fails loudly there (a missing value is a deterministic
    /// assert, and conflicting writes are caught at the merge), never
    /// silently.
    pub fn begin_island(&mut self) -> usize {
        self.steps.len()
    }

    /// Close the island opened at `start` (its value is what
    /// [`Self::begin_island`] returned).
    pub fn end_island(&mut self, start: usize) {
        let end = self.steps.len();
        if let Some(&(_, prev_end)) = self.islands.last() {
            assert!(start >= prev_end, "islands must not overlap");
        }
        assert!(start <= end);
        self.islands.push((start, end));
    }

    pub fn finish(mut self) -> Result<CircuitShape, CircuitError> {
        // Registry::new sorts class-major, area-descending, with a STABLE sort
        // on `(is_element, Reverse(k_log))`. Replicate that key to learn where
        // each declared slot landed, then assert the result agrees with the
        // registry we actually get — so a change to the registry's ordering
        // fails loudly here rather than silently mis-wiring every circuit.
        // Tables MOVE from the slots into the registry: no clone at any point,
        // because a table's matrices are the largest thing here by orders of
        // magnitude. Order and cell-slot bases come from cheap metadata read
        // before the move.
        let mut taken: Vec<Option<TableType>> = self
            .slots
            .iter_mut()
            .map(|s| Some(s.take_table()))
            .collect();
        let meta: Vec<(bool, usize, usize)> = taken
            .iter()
            .map(|t| {
                let t = t.as_ref().expect("just taken");
                (t.is_element(), t.k_log, t.io_schema.len())
            })
            .collect();
        let mut order: Vec<usize> = (0..meta.len()).collect();
        order.sort_by_key(|&i| (meta[i].0, Reverse(meta[i].1)));

        let registry = Registry::new(
            order
                .iter()
                .map(|&i| taken[i].take().expect("each slot moved once"))
                .collect(),
            self.nu,
        );
        for (reg_idx, &declared) in order.iter().enumerate() {
            assert_eq!(
                registry.types()[reg_idx].k_log,
                meta[declared].1,
                "builder's slot ordering disagrees with Registry::new"
            );
        }
        let mut registry_slot = vec![0usize; meta.len()];
        for (reg_idx, &declared) in order.iter().enumerate() {
            registry_slot[declared] = reg_idx;
        }

        // Cell-slots enumerate in registry order, each type contributing its
        // io_schema words in schema order, then the public slots.
        let mut iota_base = vec![0usize; meta.len()];
        let mut acc = 0usize;
        for &declared in &order {
            iota_base[declared] = acc;
            acc += meta[declared].2;
        }
        let num_gate_slots = acc;
        let rows_per_public_slot = 1usize << self.nu;

        // Resolve the placeholder cells to real cell-slot indices.
        for cells in &mut self.wires {
            for c in cells.iter_mut() {
                let (declared, k) = decode(c.slot);
                *c = Cell::new(iota_base[declared] + k, c.row);
            }
        }
        // Public cells.
        let pubs: Vec<usize> = self.public.clone().iter().map(|&w| self.find(w)).collect();
        for (p, &r) in pubs.iter().enumerate() {
            let slot = num_gate_slots + p / rows_per_public_slot;
            self.wires[r].push(Cell::new(slot, p % rows_per_public_slot));
        }

        // The online phase addresses values by class ROOT, so resolve every
        // wire's root once here rather than walking the union-find per proof.
        let n_wires = self.wires.len();
        let root_of: Vec<usize> = (0..n_wires).map(|i| self.find(Wire(i))).collect();
        let input_roots: Vec<usize> = self.inputs.iter().map(|&w| root_of[w.0]).collect();
        // Pre-resolve every step's wires to class roots: the online phase
        // runs ~10^5 gate calls per proof, and the per-access root_of
        // indirection was a measurable slice of its time.
        let mut steps = self.steps;
        for st in &mut steps {
            for w in &mut st.inputs {
                *w = root_of[*w];
            }
            for w in &mut st.outputs {
                *w = root_of[*w];
            }
        }

        // A class of one cell needs no copy constraint.
        let mut wires: Vec<Vec<Cell>> = self
            .wires
            .into_iter()
            .filter(|c| c.len() > 1)
            .collect::<Vec<_>>();
        for c in &mut wires {
            c.sort_unstable();
        }
        wires.sort_unstable();

        let counts: Vec<usize> = order.iter().map(|&d| self.rows_per_slot[d]).collect();

        // NOTE `Circuit::new` computes the REGISTRY digest, which hashes every
        // table's matrices. At the recursion shape that is ~280 ms, dominated
        // by BLAKE3's ~21M nonzeros — the single largest item in setup. It is
        // a pure function of the table types, and `TableType` caches it, so a
        // caller that reuses one `TableType` across circuits pays it once.
        debug_assert_eq!(self.fixed_public.len(), pubs.len());
        let circuit =
            Circuit::new_with_fixed_public(&registry, counts.clone(), self.fixed_public, wires)?;
        Ok(CircuitShape {
            registry,
            circuit,
            counts,
            nu: self.nu,
            order,
            registry_slot,
            slots: self.slots.into_iter().map(Arc::from).collect(),
            slot_types: self.slot_types,
            steps,
            n_wires,
            inputs: input_roots,
            publics: pubs,
            n_hints: self.n_hints,
            islands: self.islands,
        })
    }
}

// ---------------------------------------------------------------------------
// The shape: setup output, online input
// ---------------------------------------------------------------------------

/// The value-independent half of a circuit: the statement, plus the program
/// needed to replay it against fresh values.
///
/// **Why the split.** The statement is the same for every proof of the same
/// circuit — `Circuit::digest` binds the registry, the cell space and σ, none
/// of which depend on a value. Building it is therefore setup, paid once; only
/// evaluating the gates is per-proof. Measured at the recursion L0 shape (218
/// depth-13 Merkle openings) the two are ~46 ms and ~4 ms, so keeping them
/// together put an order of magnitude more work on the proving path than
/// belonged there.
///
/// The shape is immutable and [`run`](Self::run) takes `&self`, so one shape
/// serves any number of concurrent proofs.
#[derive(Clone)]
pub struct CircuitShape {
    pub registry: Registry,
    pub circuit: Circuit,
    /// Declared counts per slot, in REGISTRY order — what `UnionInstance::new`
    /// wants.
    pub counts: Vec<usize>,
    nu: usize,
    /// `order[registry index] = declared slot`.
    order: Vec<usize>,
    /// `registry_slot[declared] = registry index`.
    registry_slot: Vec<usize>,
    /// `Arc`, not `Box`: the finished shape only ever reads its slots
    /// (`run` takes `&self`; `take_table`'s `&mut` happened in the builder,
    /// before `finish` moved them here) — and shared slots are what makes
    /// the shape `Clone`, so a statement-independent shape can be built
    /// once and reused across proofs.
    slots: Vec<Arc<dyn SlotBuild>>,
    slot_types: Vec<TypeId>,
    steps: Vec<Step>,
    n_wires: usize,
    /// Class root per declared input, in declaration order.
    inputs: Vec<usize>,
    /// Class root per published cell, in publication order.
    publics: Vec<usize>,
    n_hints: usize,
    islands: Vec<(usize, usize)>,
}

impl CircuitShape {
    /// Where a declared slot landed in the registry. `Registry::new` sorts
    /// class-major, area-descending, so declaration order is not slot order.
    pub fn registry_slot(&self, s: SlotId) -> usize {
        self.registry_slot[s.0]
    }

    /// **The online phase.** Evaluate every gate against `inputs` and `hints`,
    /// producing this proof's witness and public segment.
    ///
    /// `inputs` are the values of the [`ShapeBuilder::input`] wires in
    /// declaration order; `hints` the advice for the
    /// [`ShapeBuilder::gate_hinted`] calls in call order.
    ///
    /// Gates run in instantiation order, which is the order the caller wrote
    /// them, so a gate's inputs must already have values — either supplied, or
    /// produced by an earlier gate. A wire whose class holds *both* a supplied
    /// input and a gate output (the forward reference the Fiat–Shamir chain
    /// needs) takes the supplied value, and the gate's output is then asserted
    /// equal to it rather than overwriting it. That assertion is what
    /// [`ShapeBuilder::connect`] promises.
    pub fn run(&self, inputs: &[F128], hints: &[&(dyn Any + Sync)]) -> CircuitWitness {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "circuit takes {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );
        assert_eq!(
            hints.len(),
            self.n_hints,
            "circuit takes {} hints, got {}",
            self.n_hints,
            hints.len()
        );

        let mut values = vec![F128::ZERO; self.n_wires];
        let mut set = vec![false; self.n_wires];
        for (&root, &v) in self.inputs.iter().zip(inputs) {
            if set[root] {
                assert_eq!(
                    values[root], v,
                    "connected inputs were given different values"
                );
            }
            values[root] = v;
            set[root] = true;
        }

        let mut rows: Vec<Box<dyn Any + Send>> = self.slots.iter().map(|s| s.new_rows()).collect();

        if self.islands.len() >= 2 {
            // Declared-independent islands run in parallel on copied value state.
            let hinted_before = |at: usize| self.steps[..at].iter().filter(|s| s.hinted).count();
            for w in self.islands.windows(2) {
                assert_eq!(
                    w[0].1, w[1].0,
                    "islands must be contiguous (steps between islands have \
                     no defined order against the parallel evaluation)"
                );
            }
            let prefix_end = self.islands[0].0;
            let suffix_start = self.islands.last().expect("nonempty").1;
            self.exec_steps(0..prefix_end, &mut values, &mut set, &mut rows, hints, 0);
            let results: Vec<(Vec<F128>, Vec<bool>, Vec<Box<dyn Any + Send>>)> = self
                .islands
                .par_iter()
                .map(|&(a, b)| {
                    let mut v = values.clone();
                    let mut st = set.clone();
                    let mut rw: Vec<Box<dyn Any + Send>> =
                        self.slots.iter().map(|s| s.new_rows()).collect();
                    self.exec_steps(a..b, &mut v, &mut st, &mut rw, hints, hinted_before(a));
                    (v, st, rw)
                })
                .collect();
            for (iv, ist, irw) in results {
                for r in 0..self.n_wires {
                    if ist[r] {
                        if set[r] {
                            assert_eq!(
                                values[r], iv[r],
                                "islands (or an island and the prefix) disagree on a wire"
                            );
                        } else {
                            values[r] = iv[r];
                            set[r] = true;
                        }
                    }
                }
                for (d, src) in irw.into_iter().enumerate() {
                    self.slots[d].merge_rows(rows[d].as_mut(), src);
                }
            }
            self.exec_steps(
                suffix_start..self.steps.len(),
                &mut values,
                &mut set,
                &mut rows,
                hints,
                hinted_before(suffix_start),
            );
        } else {
            self.exec_steps(
                0..self.steps.len(),
                &mut values,
                &mut set,
                &mut rows,
                hints,
                0,
            );
        }

        let public: Vec<F128> = self
            .publics
            .iter()
            .map(|&r| {
                assert!(set[r], "a published wire was never given a value");
                values[r]
            })
            .collect();
        let witnesses: Vec<SlotWitness> = self
            .order
            .iter()
            .map(|&d| self.slots[d].witness(rows[d].as_ref(), self.nu))
            .collect();

        CircuitWitness {
            public,
            witnesses,
            rows,
            slot_types: self.slot_types.clone(),
        }
    }

    /// Compile the shape's [`FillPlan`].
    ///
    /// Setup work, once per shape. Walks the steps in instantiation order —
    /// the order [`run`](Self::run)'s island mode is asserted equivalent to —
    /// resolving every read and write to a tape index and proving definedness
    /// as it goes; then compiles each declared island onto a compact local
    /// tape of its own so [`run_filled`](Self::run_filled) can execute them
    /// in parallel without cloning the value state.
    pub fn fill_plan(&self) -> FillPlan {
        assert!(
            self.n_wires < FILL_CHECK as usize,
            "the fill plan packs its check flag into bit 31 of a wire index"
        );
        // Definition order per class: inputs precede every step.
        const UNDEF: u32 = u32::MAX;
        const DEF_INPUT: u32 = u32::MAX - 1;
        assert!(self.steps.len() < DEF_INPUT as usize);
        let mut def_at = vec![UNDEF; self.n_wires];
        let mut input_fills = Vec::new();
        let mut input_checks = Vec::new();
        for (ord, &root) in self.inputs.iter().enumerate() {
            if def_at[root] == UNDEF {
                def_at[root] = DEF_INPUT;
                input_fills.push((root as u32, ord as u32));
            } else {
                input_checks.push((root as u32, ord as u32));
            }
        }

        // Islands compile to parallel execution exactly when the walk would
        // parallelize them, under the same contiguity contract.
        let par_islands = self.islands.len() >= 2;
        if par_islands {
            for w in self.islands.windows(2) {
                assert_eq!(
                    w[0].1, w[1].0,
                    "islands must be contiguous (steps between islands have \
                     no defined order against the parallel evaluation)"
                );
            }
        }
        // Step ordinals where a batch must break, descending so `pop`
        // consumes them in step order. Contiguity makes starts + final end
        // the complete boundary set.
        let mut boundaries: Vec<usize> = if par_islands {
            let mut b: Vec<usize> = self.islands.iter().map(|&(a, _)| a).collect();
            b.push(self.islands.last().expect("nonempty").1);
            b.reverse();
            b
        } else {
            Vec::new()
        };

        let mut batches: Vec<FillBatch> = Vec::new();
        let mut batch_step0: Vec<usize> = Vec::new();
        let mut in_idx = Vec::new();
        let mut out_idx = Vec::new();
        let mut n_hinted = 0usize;
        for (s, step) in self.steps.iter().enumerate() {
            let boundary = boundaries.last() == Some(&s);
            if boundary {
                boundaries.pop();
            }
            let coalesce = !boundary
                && batches
                    .last()
                    .is_some_and(|b| b.slot as usize == step.slot && b.hinted == step.hinted);
            if !coalesce {
                batches.push(FillBatch {
                    slot: step.slot as u32,
                    n: 0,
                    in_off: in_idx.len() as u32,
                    out_off: out_idx.len() as u32,
                    hint_base: n_hinted as u32,
                    hinted: step.hinted,
                });
                batch_step0.push(s);
            }
            batches.last_mut().expect("just pushed").n += 1;
            if step.hinted {
                n_hinted += 1;
            }
            for &r in &step.inputs {
                assert!(
                    def_at[r] != UNDEF,
                    "gate input has no value yet: a gate was instantiated before \
                     the gate producing one of its inputs"
                );
                in_idx.push(r as u32);
            }
            for &r in &step.outputs {
                if def_at[r] == UNDEF {
                    def_at[r] = s as u32;
                    out_idx.push(r as u32);
                } else {
                    out_idx.push(r as u32 | FILL_CHECK);
                }
            }
        }
        for &r in &self.publics {
            assert!(
                def_at[r] != UNDEF,
                "a published wire was never given a value"
            );
        }

        // Compile each island onto its own compact tape: intern every root
        // the island touches in first-touch order, rewriting its batches'
        // indices in place. A root defined before the islands is a GATHER; a
        // root this island defines is a SCATTER; anything else is another
        // island's value, and reading or checking against it is the
        // independence violation the walk catches at run time — here it
        // fails compilation.
        let mut islands: Vec<FillIsland> = Vec::new();
        if par_islands {
            let first_start = self.islands[0].0;
            let batch_at = |step: usize| batch_step0.partition_point(|&s0| s0 < step);
            let mut local_of = vec![UNDEF; self.n_wires];
            for &(a, b) in &self.islands {
                let (ba, bb) = (batch_at(a), batch_at(b));
                let mut touched: Vec<usize> = Vec::new();
                let mut gather: Vec<(u32, u32)> = Vec::new();
                let mut scatter: Vec<(u32, u32)> = Vec::new();
                let mut tape_len = 0u32;
                let intern_read = |r: usize,
                                   local_of: &mut Vec<u32>,
                                   touched: &mut Vec<usize>,
                                   gather: &mut Vec<(u32, u32)>,
                                   tape_len: &mut u32| {
                    if local_of[r] != UNDEF {
                        return local_of[r];
                    }
                    assert!(
                        def_at[r] == DEF_INPUT || (def_at[r] as usize) < first_start,
                        "an island reads a wire another island writes"
                    );
                    let l = *tape_len;
                    *tape_len += 1;
                    local_of[r] = l;
                    touched.push(r);
                    gather.push((l, r as u32));
                    l
                };
                for bi in ba..bb {
                    let bt = &batches[bi];
                    let slot = bt.slot as usize;
                    let (n_in, n_out) = (self.slots[slot].n_in(), self.slots[slot].n_out());
                    for g in 0..bt.n as usize {
                        let i0 = bt.in_off as usize + g * n_in;
                        for e in &mut in_idx[i0..i0 + n_in] {
                            *e = intern_read(
                                *e as usize,
                                &mut local_of,
                                &mut touched,
                                &mut gather,
                                &mut tape_len,
                            );
                        }
                        let o0 = bt.out_off as usize + g * n_out;
                        for e in &mut out_idx[o0..o0 + n_out] {
                            let check = *e & FILL_CHECK != 0;
                            let r = (*e & !FILL_CHECK) as usize;
                            let l = if check {
                                intern_read(
                                    r,
                                    &mut local_of,
                                    &mut touched,
                                    &mut gather,
                                    &mut tape_len,
                                )
                            } else {
                                debug_assert_eq!(local_of[r], UNDEF, "a write is a first def");
                                let l = tape_len;
                                tape_len += 1;
                                local_of[r] = l;
                                touched.push(r);
                                scatter.push((l, r as u32));
                                l
                            };
                            *e = if check { l | FILL_CHECK } else { l };
                        }
                    }
                }
                for r in touched {
                    local_of[r] = UNDEF;
                }
                islands.push(FillIsland {
                    batches: (ba as u32, bb as u32),
                    gather,
                    scatter,
                    tape_len,
                });
            }
        }

        FillPlan {
            batches,
            in_idx,
            out_idx,
            input_fills,
            input_checks,
            islands,
            n_steps: self.steps.len(),
            n_wires: self.n_wires,
        }
    }

    /// **The online phase, plan-driven.** Same contract and same output as
    /// [`run`](Self::run) — `run` is the differential oracle for this — with
    /// the walk's per-gate bookkeeping replaced by the plan's index copies
    /// and per-slot batch loops.
    pub fn run_filled(
        &self,
        plan: &FillPlan,
        inputs: &[F128],
        hints: &[&(dyn Any + Sync)],
    ) -> CircuitWitness {
        let mut w = self.run_filled_deferred(plan, inputs, hints);
        let t_pack = Instant::now();
        self.pack_witnesses(&mut w);
        if var("FILL_TRACE").is_ok() {
            eprintln!(
                "  [run_filled] pack_witnesses {:.2} ms",
                t_pack.elapsed().as_secs_f64() * 1e3
            );
        }
        w
    }

    /// [`run_filled`](Self::run_filled) without the witness packing: every
    /// slot's entry is left [`SlotWitness::DeferredToRows`] and only the
    /// rows and publics are produced.
    ///
    /// This is the fast path for a caller that feeds the prover in place
    /// from the typed rows ([`CircuitWitness::take_rows_of`]): a packed
    /// element witness is a full-capacity column-major buffer whose live
    /// cells are copied again immediately after, so materializing it is
    /// pure overhead there. Call [`pack_witnesses`](Self::pack_witnesses)
    /// to get the eager result later; the two paths are value-identical by
    /// construction (packing is a pure function of the rows).
    pub fn run_filled_deferred(
        &self,
        plan: &FillPlan,
        inputs: &[F128],
        hints: &[&(dyn Any + Sync)],
    ) -> CircuitWitness {
        assert_eq!(
            (plan.n_steps, plan.n_wires),
            (self.steps.len(), self.n_wires),
            "the plan was compiled from a different shape"
        );
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "circuit takes {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );
        assert_eq!(
            hints.len(),
            self.n_hints,
            "circuit takes {} hints, got {}",
            self.n_hints,
            hints.len()
        );

        let t_run = Instant::now();
        let mut values = vec![F128::ZERO; self.n_wires];
        for &(r, ord) in &plan.input_fills {
            values[r as usize] = inputs[ord as usize];
        }
        for &(r, ord) in &plan.input_checks {
            assert_eq!(
                values[r as usize], inputs[ord as usize],
                "connected inputs were given different values"
            );
        }

        let mut rows: Vec<Box<dyn Any + Send>> = self.slots.iter().map(|s| s.new_rows()).collect();
        let mut scratch_in: Vec<F128> = Vec::with_capacity(16);
        let mut scratch_out: Vec<F128> = Vec::with_capacity(16);
        if plan.islands.is_empty() {
            self.exec_batches(
                plan,
                0..plan.batches.len(),
                &mut values,
                &mut rows,
                hints,
                &mut scratch_in,
                &mut scratch_out,
            );
        } else {
            // Each island evaluates its compact local tape in parallel.
            let pre = plan.islands[0].batches.0 as usize;
            let suf = plan.islands.last().expect("nonempty").batches.1 as usize;
            self.exec_batches(
                plan,
                0..pre,
                &mut values,
                &mut rows,
                hints,
                &mut scratch_in,
                &mut scratch_out,
            );
            let t_isl = Instant::now();
            let results: Vec<(Vec<F128>, Vec<Box<dyn Any + Send>>)> = plan
                .islands
                .par_iter()
                .map(|isl| {
                    let mut local = vec![F128::ZERO; isl.tape_len as usize];
                    for &(l, g) in &isl.gather {
                        local[l as usize] = values[g as usize];
                    }
                    let mut irows: Vec<Box<dyn Any + Send>> =
                        self.slots.iter().map(|s| s.new_rows()).collect();
                    let mut si: Vec<F128> = Vec::with_capacity(16);
                    let mut so: Vec<F128> = Vec::with_capacity(16);
                    if var("FILL_CENSUS").is_ok() {
                        // Per-slot time attribution inside this island,
                        // batch-serial so the numbers are exact. DECLARED
                        // slot indices — map them with the harness's own
                        // census (e.g. NODE_CENSUS). This is what caught the
                        // residual gates' per-row constant inversions.
                        let mut per_slot = vec![(0f64, 0usize); self.slots.len()];
                        for bi in isl.batches.0 as usize..isl.batches.1 as usize {
                            let t = Instant::now();
                            self.exec_batches(
                                plan,
                                bi..bi + 1,
                                &mut local,
                                &mut irows,
                                hints,
                                &mut si,
                                &mut so,
                            );
                            let b = &plan.batches[bi];
                            per_slot[b.slot as usize].0 += t.elapsed().as_secs_f64() * 1e3;
                            per_slot[b.slot as usize].1 += b.n as usize;
                        }
                        let mut v: Vec<(usize, f64, usize)> = per_slot
                            .iter()
                            .enumerate()
                            .map(|(s, &(ms, n))| (s, ms, n))
                            .collect();
                        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        for (s, ms, n) in v.iter().take(12) {
                            eprintln!(
                                "    [census] declared slot {s:3}: {ms:7.2} ms over {n:6} rows"
                            );
                        }
                    } else {
                        self.exec_batches(
                            plan,
                            isl.batches.0 as usize..isl.batches.1 as usize,
                            &mut local,
                            &mut irows,
                            hints,
                            &mut si,
                            &mut so,
                        );
                    }
                    (local, irows)
                })
                .collect();
            let t_merge = Instant::now();
            for (isl, (local, irows)) in plan.islands.iter().zip(results) {
                for &(l, g) in &isl.scatter {
                    values[g as usize] = local[l as usize];
                }
                for (d, src) in irows.into_iter().enumerate() {
                    self.slots[d].merge_rows(rows[d].as_mut(), src);
                }
            }
            let t_suf = Instant::now();
            self.exec_batches(
                plan,
                suf..plan.batches.len(),
                &mut values,
                &mut rows,
                hints,
                &mut scratch_in,
                &mut scratch_out,
            );
            if var("FILL_TRACE").is_ok() {
                eprintln!(
                    "  [run_filled] prefix {:.2} | islands {:.2} | merge {:.2} | suffix {:.2} ms",
                    t_isl.duration_since(t_run).as_secs_f64() * 1e3,
                    t_merge.duration_since(t_isl).as_secs_f64() * 1e3,
                    t_suf.duration_since(t_merge).as_secs_f64() * 1e3,
                    t_suf.elapsed().as_secs_f64() * 1e3,
                );
            }
        }

        let public: Vec<F128> = self.publics.iter().map(|&r| values[r]).collect();
        if var("FILL_TRACE").is_ok() {
            eprintln!(
                "  [run_filled] batches+islands+publics {:.2} ms",
                t_run.elapsed().as_secs_f64() * 1e3,
            );
        }
        CircuitWitness {
            public,
            witnesses: vec![SlotWitness::DeferredToRows; self.order.len()],
            rows,
            slot_types: self.slot_types.clone(),
        }
    }

    /// Pack every slot's committed witness from its rows, in registry order —
    /// the eager half [`run_filled_deferred`](Self::run_filled_deferred)
    /// skips. Idempotent; a pure function of the rows.
    pub fn pack_witnesses(&self, w: &mut CircuitWitness) {
        assert_eq!(
            w.witnesses.len(),
            self.order.len(),
            "witness produced by a different shape"
        );
        for (reg, &d) in self.order.iter().enumerate() {
            w.witnesses[reg] = self.slots[d].witness(w.rows[d].as_ref(), self.nu);
        }
    }

    /// Run the plan's batches in `range` against one value tape — global for
    /// the prefix/suffix, an island's local tape inside one.
    #[allow(clippy::too_many_arguments)]
    fn exec_batches(
        &self,
        plan: &FillPlan,
        range: Range<usize>,
        values: &mut [F128],
        rows: &mut [Box<dyn Any + Send>],
        hints: &[&(dyn Any + Sync)],
        scratch_in: &mut Vec<F128>,
        scratch_out: &mut Vec<F128>,
    ) {
        for b in &plan.batches[range] {
            let slot = b.slot as usize;
            let s = &self.slots[slot];
            let n = b.n as usize;
            let (i0, o0) = (b.in_off as usize, b.out_off as usize);
            s.run_batch(
                rows[slot].as_mut(),
                values,
                n,
                &plan.in_idx[i0..i0 + n * s.n_in()],
                &plan.out_idx[o0..o0 + n * s.n_out()],
                hints,
                b.hint_base as usize,
                b.hinted,
                scratch_in,
                scratch_out,
            );
        }
    }

    /// Execute the steps in `range` against the given value state and row
    /// accumulators. `hint_base` is the number of hinted steps before the
    /// range (hints are consumed in absolute instantiation order).
    fn exec_steps(
        &self,
        range: Range<usize>,
        values: &mut [F128],
        set: &mut [bool],
        rows: &mut [Box<dyn Any + Send>],
        hints: &[&(dyn Any + Sync)],
        hint_base: usize,
    ) {
        let unit = ();
        let mut next_hint = hint_base;
        // One scratch buffer for every step's input values — a fresh Vec per
        // gate call was a measurable slice of the online phase. Step wires
        // are pre-resolved to class roots by `finish`.
        let mut vals: Vec<F128> = Vec::with_capacity(16);
        let mut outs: Vec<F128> = Vec::with_capacity(16);
        for (step_i, step) in self.steps[range].iter().enumerate() {
            vals.clear();
            for &r in &step.inputs {
                assert!(
                    set[r],
                    "gate input has no value yet: a gate was instantiated before \
                     the gate producing one of its inputs (or an island read a \
                     wire another island writes)"
                );
                vals.push(values[r]);
            }
            let hint: &dyn Any = if step.hinted {
                let h = hints[next_hint];
                next_hint += 1;
                h
            } else {
                &unit
            };
            outs.clear();
            self.slots[step.slot].push(rows[step.slot].as_mut(), &vals, hint, &mut outs);
            for (&r, &v) in step.outputs.iter().zip(&outs) {
                if set[r] {
                    assert_eq!(
                        values[r], v,
                        "a connected wire disagrees with the gate output that produces it \
                         (slot {}, class root {r}, step {step_i})",
                        step.slot
                    );
                } else {
                    values[r] = v;
                    set[r] = true;
                }
            }
        }
    }
}

/// **The online phase's output**: one proof's worth of witness.
pub struct CircuitWitness {
    /// The public segment, in publication order.
    pub public: Vec<F128>,
    /// Per-slot witnesses, in REGISTRY order.
    pub witnesses: Vec<SlotWitness>,
    /// Per-slot rows, in DECLARED order.
    rows: Vec<Box<dyn Any + Send>>,
    slot_types: Vec<TypeId>,
}

impl CircuitWitness {
    /// A slot's rows in instantiation order, with their concrete type
    /// recovered.
    ///
    /// The escape hatch for witnesses the builder cannot pack — a boolean slot
    /// hands back its `&[Compression]` here, and the caller feeds it to
    /// `generate_witness_batch_major_partial`. Row ORDER is the builder's
    /// contract: row `j` of this slice is row `j` of the committed trace, which
    /// is what makes the wiring the builder emitted correct for that witness.
    ///
    /// Panics if `s` was not declared with `G`.
    pub fn rows<G>(&self, s: SlotId) -> &[G::Row]
    where
        G: GateType + 'static,
        G::Row: 'static,
    {
        assert_eq!(
            self.slot_types[s.0],
            TypeId::of::<G>(),
            "slot was declared with a different GateType"
        );
        self.rows[s.0]
            .downcast_ref::<Vec<G::Row>>()
            .expect("slot type matched but its rows did not")
    }

    /// MOVE a slot's rows out, keyed by their ROW type rather than the gate
    /// type — the element-assembly escape for feeding the prover in place
    /// (with [`CircuitShape::run_filled_deferred`]) without the packed
    /// intermediate. Every element gate in a harness shares one row shape,
    /// and the assembly loops over its element slots generically, so a
    /// gate-typed accessor cannot serve it. A wrong `R` still fails loudly
    /// on the downcast.
    pub fn take_rows_of<R: Send + 'static>(&mut self, s: SlotId) -> Vec<R> {
        take(
            self.rows[s.0]
                .downcast_mut::<Vec<R>>()
                .expect("slot rows are not of this row type"),
        )
    }
}

// ---------------------------------------------------------------------------
// One-shot front door
// ---------------------------------------------------------------------------

/// Build a circuit and evaluate it in one pass, supplying values inline.
///
/// Convenience over [`ShapeBuilder`] + [`CircuitShape::run`] for circuits
/// proved once — tests, and any caller that does not reuse the shape. It is
/// exactly those two steps: `finish` builds the shape, then runs it. A caller
/// that proves the same circuit repeatedly should use the two directly and
/// keep the shape.
pub struct CircuitBuilder {
    shape: ShapeBuilder,
    values: Vec<F128>,
    hints: Vec<Box<dyn Any + Sync>>,
}

impl CircuitBuilder {
    pub fn new(nu: usize) -> Self {
        Self {
            shape: ShapeBuilder::new(nu),
            values: Vec::new(),
            hints: Vec::new(),
        }
    }

    pub fn slot<G>(&mut self, gate: G) -> SlotId
    where
        G: GateType + Send + Sync + 'static,
        G::Row: Send + 'static,
        G::Hint: 'static,
    {
        self.shape.slot(gate)
    }

    /// A free value entering the circuit. See [`ShapeBuilder::input`].
    pub fn value(&mut self, value: F128) -> Wire {
        self.values.push(value);
        self.shape.input()
    }

    /// A value that is both free and public — the common case for circuit
    /// inputs.
    pub fn public_value(&mut self, value: F128) -> Wire {
        let w = self.value(value);
        self.publish(w);
        w
    }

    pub fn gate(&mut self, slot: SlotId, inputs: &[Wire]) -> Vec<Wire> {
        self.shape.gate(slot, inputs)
    }

    /// Instantiate a gate, supplying this instance's nondeterministic advice.
    /// See [`GateType::Hint`]; `hint` must be that exact type.
    pub fn gate_with_hint<H: Any + Sync>(
        &mut self,
        slot: SlotId,
        inputs: &[Wire],
        hint: H,
    ) -> Vec<Wire> {
        self.hints.push(Box::new(hint));
        self.shape.gate_hinted(slot, inputs)
    }

    pub fn publish(&mut self, w: Wire) {
        self.shape.publish(w);
    }

    /// See [`ShapeBuilder::connect`]. The value check happens in `finish`,
    /// when the gates are evaluated.
    pub fn connect(&mut self, a: Wire, b: Wire) {
        self.shape.connect(a, b);
    }

    pub fn finish(self) -> Result<BuiltCircuit, CircuitError> {
        let shape = self.shape.finish()?;
        let hints: Vec<&(dyn Any + Sync)> = self.hints.iter().map(|b| b.as_ref()).collect();
        let witness = shape.run(&self.values, &hints);
        Ok(BuiltCircuit { shape, witness })
    }
}

/// Everything [`CircuitBuilder::finish`] produces: the statement and the
/// witness, from one description.
pub struct BuiltCircuit {
    pub shape: CircuitShape,
    pub witness: CircuitWitness,
}

impl BuiltCircuit {
    pub fn registry_slot(&self, s: SlotId) -> usize {
        self.shape.registry_slot(s)
    }

    /// See [`CircuitWitness::rows`].
    pub fn rows<G>(&self, s: SlotId) -> &[G::Row]
    where
        G: GateType + 'static,
        G::Row: 'static,
    {
        self.witness.rows::<G>(s)
    }
}

// Placeholder cell-slot encoding, used only between `gate` and `finish`: the
// real cell-slot index needs the registry order, which is not known until
// every slot has been declared.
fn encode(slot: usize, k: usize) -> usize {
    slot << 32 | k
}

fn decode(c: usize) -> (usize, usize) {
    (c >> 32, c & 0xFFFF_FFFF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_r1cs::{ElementTableBuilder, ElementTableType};
    use crate::schedule::IoWord;
    use std::sync::Arc;

    #[test]
    fn fixed_public_values_are_digest_bound_and_checked() {
        let build = |constant: F128| {
            let mut b = ShapeBuilder::new(3);
            let mult = b.slot(MultGate::new(3));
            let c = b.fixed_public_input(constant);
            let x = b.input();
            let y = b.gate(mult, &[c, x])[0];
            b.publish(y);
            b.finish().expect("fixed-public circuit builds")
        };

        let seven = F128::new(7, 0);
        let shape = build(seven);
        let witness = shape.run(&[seven, F128::new(9, 0)], &[]);
        assert!(shape.circuit.check_public(&witness.public));
        let mut changed = witness.public.clone();
        changed[0] = F128::new(6, 0);
        assert!(!shape.circuit.check_public(&changed));

        let other = build(F128::new(8, 0));
        assert_ne!(shape.circuit.digest(), other.circuit.digest());
    }

    /// The element `mult` gate from `circuit_wiring.rs`: columns 0,1 free
    /// wires in, column 2 = z0·z1 out.
    struct MultGate {
        ty: Arc<ElementTableType>,
    }

    impl MultGate {
        fn new(kappa: usize) -> Self {
            let mut b = ElementTableBuilder::new(kappa);
            b.free_wire(0).free_wire(1).mult(2, 0, 1);
            Self {
                ty: Arc::new(b.build().expect("mult block is valid")),
            }
        }
    }

    impl GateType for MultGate {
        type Row = (F128, F128, F128);
        type Hint = ();

        fn table(&self) -> TableType {
            TableType::element(self.ty.clone()).with_io_schema(vec![
                IoWord::input(0),
                IoWord::input(1),
                IoWord::output(2),
            ])
        }

        fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
            let (a, b) = (inputs[0], inputs[1]);
            let c = a * b;
            outputs.push(c);
            (a, b, c)
        }

        fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
            let at = |c: usize, j: usize| (c << nu) + j;
            let mut z = vec![F128::ZERO; self.ty.width() << nu];
            for (j, &(a, b, c)) in rows.iter().enumerate() {
                z[at(0, j)] = a;
                z[at(1, j)] = b;
                z[at(2, j)] = c;
            }
            SlotWitness::Element(z)
        }
    }

    /// The builder reproduces `circuit_wiring.rs`'s hand-built element chain
    /// EXACTLY — same wiring classes, same counts, same `Circuit::digest`.
    ///
    /// This is the validation the design called for: if the builder is right,
    /// nothing moves.
    #[test]
    fn builder_reproduces_the_hand_built_element_chain() {
        const EL_A: usize = 0;
        const EL_B: usize = 1;
        const EL_C: usize = 2;
        const PUB: usize = 3;
        let (nu, kappa, n) = (12usize, 3usize, 20usize);

        let mut state = 0xC4A1_0001u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = state;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(hi, state)
        };
        let seed = next();
        let a: Vec<F128> = (0..n).map(|_| next()).collect();

        // ---- built ----
        let mut b = CircuitBuilder::new(nu);
        let mult = b.slot(MultGate::new(kappa));
        let seed_w = b.public_value(seed);
        let a_w: Vec<Wire> = a.iter().map(|&x| b.public_value(x)).collect();
        let mut acc = seed_w;
        for &aw in &a_w {
            acc = b.gate(mult, &[aw, acc])[0];
        }
        b.publish(acc);
        let built = b.finish().expect("builder produces a valid circuit");

        // ---- hand-built, verbatim from circuit_wiring.rs ----
        let ty = MultGate::new(kappa);
        let registry = Registry::new(vec![ty.table()], nu);
        let mut hand = vec![vec![Cell::new(PUB, 0), Cell::new(EL_B, 0)]];
        for i in 0..n {
            hand.push(vec![Cell::new(PUB, 1 + i), Cell::new(EL_A, i)]);
        }
        for i in 0..n - 1 {
            hand.push(vec![Cell::new(EL_C, i), Cell::new(EL_B, i + 1)]);
        }
        hand.push(vec![Cell::new(EL_C, n - 1), Cell::new(PUB, 1 + n)]);
        let hand_circuit = Circuit::new(&registry, vec![n], n + 2, hand).expect("valid");

        assert_eq!(
            built.shape.counts,
            vec![n],
            "counts fall out of construction"
        );
        assert_eq!(built.witness.public.len(), n + 2);
        assert_eq!(
            built.shape.circuit.digest(),
            hand_circuit.digest(),
            "builder produced a DIFFERENT statement than the hand-built circuit"
        );

        // And the witness matches the hand-written chain generator.
        let at = |c: usize, j: usize| (c << nu) + j;
        let mut want = vec![F128::ZERO; ty.ty.width() << nu];
        let mut acc_v = seed;
        for (j, &aj) in a.iter().enumerate() {
            want[at(0, j)] = aj;
            want[at(1, j)] = acc_v;
            want[at(2, j)] = aj * acc_v;
            acc_v = aj * acc_v;
        }
        assert_eq!(built.witness.witnesses[0], SlotWitness::Element(want));
        assert!(
            ty.ty.satisfies(
                match &built.witness.witnesses[0] {
                    SlotWitness::Element(z) => z,
                    other => panic!("element slot produced {other:?}"),
                },
                nu,
                n
            ),
            "built witness must satisfy the relation"
        );
    }

    /// `MultGate` with advice: the hint rides into the row but touches no
    /// wire — exercises the plan's hint-ordinal bookkeeping.
    struct HintedMultGate {
        ty: Arc<ElementTableType>,
    }

    impl HintedMultGate {
        fn new(kappa: usize) -> Self {
            let mut b = ElementTableBuilder::new(kappa);
            b.free_wire(0).free_wire(1).mult(2, 0, 1);
            Self {
                ty: Arc::new(b.build().expect("mult block is valid")),
            }
        }
    }

    impl GateType for HintedMultGate {
        type Row = (F128, F128, F128, F128);
        type Hint = F128;

        fn table(&self) -> TableType {
            TableType::element(self.ty.clone()).with_io_schema(vec![
                IoWord::input(0),
                IoWord::input(1),
                IoWord::output(2),
            ])
        }

        fn eval(&self, inputs: &[F128], hint: &F128, outputs: &mut Vec<F128>) -> Self::Row {
            let (a, b) = (inputs[0], inputs[1]);
            let c = a * b;
            outputs.push(c);
            (a, b, c, *hint)
        }

        fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
            let at = |c: usize, j: usize| (c << nu) + j;
            let mut z = vec![F128::ZERO; self.ty.width() << nu];
            for (j, &(a, b, c, _)) in rows.iter().enumerate() {
                z[at(0, j)] = a;
                z[at(1, j)] = b;
                z[at(2, j)] = c;
            }
            SlotWitness::Element(z)
        }
    }

    /// The fill plan against the walk, on a shape that hits every plan path:
    /// alternating slots (length-1 batches), hinted gates, connected
    /// duplicate inputs (the `input_checks` arm), and a gate output connected
    /// to an already-supplied input (the `FILL_CHECK` arm — the Fiat–Shamir
    /// forward reference). Identical `CircuitWitness`, field for field.
    #[test]
    fn fill_plan_matches_the_walk() {
        let (nu, kappa, n) = (6usize, 3usize, 12usize);
        let mut state = 0xF177_C4A1_0007_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = state;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(hi, state)
        };
        let seed = next();
        let a: Vec<F128> = (0..n).map(|_| next()).collect();
        let hints_v: Vec<F128> = (0..n / 2).map(|_| next()).collect();

        let mut b = ShapeBuilder::new(nu);
        let mult = b.slot(MultGate::new(kappa));
        let hmult = b.slot(HintedMultGate::new(kappa));

        let mut vals: Vec<F128> = Vec::new();
        let input = |b: &mut ShapeBuilder, v: F128, vals: &mut Vec<F128>| {
            vals.push(v);
            b.public_input()
        };
        let seed_w = input(&mut b, seed, &mut vals);
        // Two connected inputs supplied the same value: the duplicate arm.
        let dup0 = input(&mut b, a[0], &mut vals);
        let dup1 = input(&mut b, a[0], &mut vals);
        b.connect(dup0, dup1);
        // Alternate slots so no batch coalesces past its neighbor.
        let mut acc = seed_w;
        let mut acc_v = seed;
        for (i, &ai) in a.iter().enumerate() {
            let aw = if i == 0 {
                dup1
            } else {
                input(&mut b, ai, &mut vals)
            };
            acc = if i % 2 == 0 {
                b.gate(mult, &[aw, acc])[0]
            } else {
                b.gate_hinted(hmult, &[aw, acc])[0]
            };
            acc_v = ai * acc_v;
        }
        // The forward reference: the final product is ALSO supplied as an
        // input, and the producing gate's output is checked against it.
        vals.push(acc_v);
        let fwd = b.input();
        b.connect(acc, fwd);
        b.publish(acc);

        let shape = b.finish().expect("the shape builds");
        let hint_refs: Vec<&(dyn Any + Sync)> =
            hints_v.iter().map(|h| h as &(dyn Any + Sync)).collect();
        let plan = shape.fill_plan();

        let walk = shape.run(&vals, &hint_refs);
        let fill = shape.run_filled(&plan, &vals, &hint_refs);

        assert_eq!(walk.public, fill.public, "public segment");
        assert_eq!(walk.witnesses, fill.witnesses, "slot witnesses");
        assert_eq!(
            walk.rows::<MultGate>(mult),
            fill.rows::<MultGate>(mult),
            "mult rows"
        );
        assert_eq!(
            walk.rows::<HintedMultGate>(hmult),
            fill.rows::<HintedMultGate>(hmult),
            "hinted mult rows"
        );
        assert_eq!(walk.public.last(), Some(&acc_v), "the chain closed");

        // The deferred path: rows and publics only, packing after the fact
        // — value-identical to the eager run — and `take_rows_of` moves a
        // slot's rows out by ROW type.
        let mut deferred = shape.run_filled_deferred(&plan, &vals, &hint_refs);
        assert!(
            deferred
                .witnesses
                .iter()
                .all(|w| *w == SlotWitness::DeferredToRows),
            "deferred run packs nothing"
        );
        assert_eq!(deferred.public, walk.public, "deferred publics");
        shape.pack_witnesses(&mut deferred);
        assert_eq!(deferred.witnesses, walk.witnesses, "packed after the fact");
        let taken = deferred.take_rows_of::<(F128, F128, F128)>(mult);
        assert_eq!(
            taken.as_slice(),
            walk.rows::<MultGate>(mult),
            "taken rows are the slot's rows"
        );
    }

    /// Islands under the plan: two parallel mult chains off a shared prefix
    /// product, a forward-referenced check INSIDE an island (gathered, then
    /// asserted), and a suffix joining both islands' ends — identical to the
    /// walk's parallel island mode, without its value-state clones.
    #[test]
    fn fill_plan_matches_the_walk_across_islands() {
        let (nu, kappa, n) = (6usize, 3usize, 8usize);
        let mut state = 0x15_1A_4D_5EEDu64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = state;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(hi, state)
        };
        let (a_v, b_v) = (next(), next());
        let xs: Vec<F128> = (0..2 * n).map(|_| next()).collect();

        let mut b = ShapeBuilder::new(nu);
        let mult = b.slot(MultGate::new(kappa));
        let mut vals: Vec<F128> = Vec::new();
        let mut chains: Vec<F128> = Vec::new();

        vals.push(a_v);
        let a = b.public_input();
        vals.push(b_v);
        let bw = b.public_input();
        let p = b.gate(mult, &[a, bw])[0];
        let p_v = a_v * b_v;

        let mut accs = Vec::new();
        for isl in 0..2 {
            let start = b.begin_island();
            let mut acc = p;
            let mut acc_v = p_v;
            for &x in &xs[isl * n..(isl + 1) * n] {
                vals.push(x);
                let xw = b.public_input();
                acc = b.gate(mult, &[xw, acc])[0];
                acc_v = x * acc_v;
            }
            b.end_island(start);
            accs.push(acc);
            chains.push(acc_v);
        }
        // The in-island forward reference: island 1's end is ALSO supplied
        // as an input (a pre-island value), so its producing gate's output
        // is a check against a gathered cell.
        vals.push(chains[1]);
        let fwd = b.input();
        b.connect(accs[1], fwd);
        let joined = b.gate(mult, &[accs[0], accs[1]])[0];
        b.publish(joined);

        let shape = b.finish().expect("the island shape builds");
        let plan = shape.fill_plan();
        let walk = shape.run(&vals, &[]);
        let fill = shape.run_filled(&plan, &vals, &[]);

        assert_eq!(walk.public, fill.public, "public segment");
        assert_eq!(walk.witnesses, fill.witnesses, "slot witnesses");
        assert_eq!(
            walk.rows::<MultGate>(mult),
            fill.rows::<MultGate>(mult),
            "mult rows in prefix + island + suffix order"
        );
        assert_eq!(
            walk.public.last(),
            Some(&(chains[0] * chains[1])),
            "the join closed"
        );
    }

    /// The independence contract moves to compile time: an island consuming
    /// another island's output fails `fill_plan`, where the walk would only
    /// fail once run.
    #[test]
    #[should_panic(expected = "an island reads a wire another island writes")]
    fn fill_plan_rejects_a_cross_island_read() {
        let (nu, kappa) = (6usize, 3usize);
        let mut b = ShapeBuilder::new(nu);
        let mult = b.slot(MultGate::new(kappa));
        let a = b.public_input();
        let s0 = b.begin_island();
        let x = b.gate(mult, &[a, a])[0];
        b.end_island(s0);
        let s1 = b.begin_island();
        let y = b.gate(mult, &[x, a])[0];
        b.end_island(s1);
        b.publish(y);
        let shape = b.finish().expect("the shape builds");
        let _ = shape.fill_plan();
    }
}
