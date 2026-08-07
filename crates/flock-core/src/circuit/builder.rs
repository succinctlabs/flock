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

use std::any::type_name;
use std::any::{Any, TypeId};
use std::cmp::Reverse;
use std::mem::take;

use crate::field::F128;
use crate::schedule::{IoDirection, Registry, TableType};

use super::{Cell, Circuit, CircuitError};

/// A value in the circuit, and the cells that must hold it.
///
/// A wire IS an equivalence class under construction: binding it as a gate
/// input appends that gate's input cell to the class, and the builder hands
/// the finished classes to [`Circuit::new`].
///
/// Wires are usable before their producer is emitted — a value can be consumed
/// by a gate declared earlier in the program than the one that defines it,
/// because a class is just a set. The Fiat–Shamir chain needs this: a squeezed
/// challenge is re-absorbed into the transcript that produced it. In the online
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
    /// order; returns the `Out` words in schema order, plus the row record.
    /// `hint` is this instance's advice — see [`Hint`](GateType::Hint).
    fn eval(&self, inputs: &[F128], hint: &Self::Hint) -> (Vec<F128>, Self::Row);

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
trait SlotBuild: Any {
    /// MOVE the table out. Called once, by `finish`, on its way into the
    /// registry — which then owns it. Cloning here instead cost 2 deep copies
    /// of every table's matrices; BLAKE3's are ~21M nonzeros, so that was
    /// ~300 ms of pure memcpy per circuit.
    fn take_table(&mut self) -> TableType;
    fn n_in(&self) -> usize;
    fn n_out(&self) -> usize;
    /// A fresh, empty `Vec<G::Row>` for one online run.
    fn new_rows(&self) -> Box<dyn Any>;
    /// Evaluate one gate, appending its row.
    fn push(&self, rows: &mut dyn Any, inputs: &[F128], hint: &dyn Any) -> Vec<F128>;
    fn witness(&self, rows: &dyn Any, nu: usize) -> SlotWitness;
}

struct GateSlot<G: GateType> {
    gate: G,
    /// `None` after `finish` has moved it into the registry.
    table: Option<TableType>,
    n_in: usize,
    n_out: usize,
}

impl<G: GateType + 'static> SlotBuild for GateSlot<G>
where
    G::Row: 'static,
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
    fn new_rows(&self) -> Box<dyn Any> {
        Box::new(Vec::<G::Row>::new())
    }
    fn push(&self, rows: &mut dyn Any, inputs: &[F128], hint: &dyn Any) -> Vec<F128> {
        let rows = rows
            .downcast_mut::<Vec<G::Row>>()
            .expect("row store belongs to another slot");
        let hint = hint.downcast_ref::<G::Hint>().unwrap_or_else(|| {
            panic!(
                "gate expects a hint of type {}; use gate_hinted and supply one",
                type_name::<G::Hint>()
            )
        });
        let (outputs, row) = self.gate.eval(inputs, hint);
        assert_eq!(
            outputs.len(),
            self.n_out,
            "gate returned {} outputs, schema declares {}",
            outputs.len(),
            self.n_out
        );
        rows.push(row);
        outputs
    }
    fn witness(&self, rows: &dyn Any, nu: usize) -> SlotWitness {
        self.gate.witness(
            rows.downcast_ref::<Vec<G::Row>>()
                .expect("row store belongs to another slot"),
            nu,
        )
    }
}

/// One recorded gate instantiation. Wire indices, not values — this is the
/// value-independent half.
struct Step {
    slot: usize,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    hinted: bool,
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
    inputs: Vec<Wire>,
    steps: Vec<Step>,
    rows_per_slot: Vec<usize>,
    n_hints: usize,
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
            inputs: Vec::new(),
            steps: Vec::new(),
            rows_per_slot: Vec::new(),
            n_hints: 0,
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
        G: GateType + 'static,
        G::Row: 'static,
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
        let circuit = Circuit::new(&registry, counts.clone(), pubs.len(), wires)?;
        Ok(CircuitShape {
            registry,
            circuit,
            counts,
            nu: self.nu,
            order,
            registry_slot,
            slots: self.slots,
            slot_types: self.slot_types,
            steps: self.steps,
            n_wires,
            root_of,
            inputs: input_roots,
            publics: pubs,
            n_hints: self.n_hints,
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
    slots: Vec<Box<dyn SlotBuild>>,
    slot_types: Vec<TypeId>,
    steps: Vec<Step>,
    n_wires: usize,
    root_of: Vec<usize>,
    /// Class root per declared input, in declaration order.
    inputs: Vec<usize>,
    /// Class root per published cell, in publication order.
    publics: Vec<usize>,
    n_hints: usize,
}

impl CircuitShape {
    /// Where a declared slot landed in the registry. `Registry::new` sorts
    /// class-major, area-descending, so declaration order is not slot order.
    pub fn registry_slot(&self, s: SlotId) -> usize {
        self.registry_slot[s.0]
    }

    /// How many values [`run`](Self::run) expects.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// How many hints [`run`](Self::run) expects.
    pub fn num_hints(&self) -> usize {
        self.n_hints
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
    pub fn run(&self, inputs: &[F128], hints: &[&dyn Any]) -> CircuitWitness {
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

        let mut rows: Vec<Box<dyn Any>> = self.slots.iter().map(|s| s.new_rows()).collect();
        let unit = ();
        let mut next_hint = 0usize;
        for step in &self.steps {
            let vals: Vec<F128> = step
                .inputs
                .iter()
                .map(|&w| {
                    let r = self.root_of[w];
                    assert!(
                        set[r],
                        "gate input has no value yet: a gate was instantiated before \
                         the gate producing one of its inputs"
                    );
                    values[r]
                })
                .collect();
            let hint: &dyn Any = if step.hinted {
                let h = hints[next_hint];
                next_hint += 1;
                h
            } else {
                &unit
            };
            let outs = self.slots[step.slot].push(rows[step.slot].as_mut(), &vals, hint);
            for (&w, v) in step.outputs.iter().zip(outs) {
                let r = self.root_of[w];
                if set[r] {
                    assert_eq!(
                        values[r], v,
                        "a connected wire disagrees with the gate output that produces it"
                    );
                } else {
                    values[r] = v;
                    set[r] = true;
                }
            }
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
}

/// **The online phase's output**: one proof's worth of witness.
pub struct CircuitWitness {
    /// The public segment, in publication order.
    pub public: Vec<F128>,
    /// Per-slot witnesses, in REGISTRY order.
    pub witnesses: Vec<SlotWitness>,
    /// Per-slot rows, in DECLARED order.
    rows: Vec<Box<dyn Any>>,
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
    hints: Vec<Box<dyn Any>>,
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
        G: GateType + 'static,
        G::Row: 'static,
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
    pub fn gate_with_hint<H: Any>(&mut self, slot: SlotId, inputs: &[Wire], hint: H) -> Vec<Wire> {
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
        let hints: Vec<&dyn Any> = self.hints.iter().map(|b| b.as_ref()).collect();
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

        fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
            let (a, b) = (inputs[0], inputs[1]);
            let c = a * b;
            (vec![c], (a, b, c))
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
}
