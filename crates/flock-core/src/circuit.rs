//! Circuits over a multi-table registry: the cell space, the wiring
//! (copy-constraint) argument, and public IO.
//!
//! `docs/circuit-wiring-design.tex` §5 (the circuit model), §6 (the wiring
//! argument) and §7 (v1 = verifier-known σ) are normative; this module is the
//! implementation. What one circuit proof attests, on top of the union's
//! per-row relations:
//!
//! - every gate row satisfies its table's relation — the existing boolean /
//!   element PIOPs, untouched;
//! - the circuit's **wiring equalities** hold (Plonk's "copy constraints": the
//!   requirement that designated cells hold EQUAL values — not R1CS
//!   constraints, nothing in the constraint system implements them);
//! - designated cells equal the **public words** of the statement.
//!
//! ## The cell space
//!
//! A *cell* is one wireable committed word of one gate: `p = (ι, j)` with the
//! row `j` in the LOW `ν` bits and the *cell-slot* `ι` in the high `c` bits, so
//! `μ = ν + c` and `cells = {0,1}^μ`. Cell-slots enumerate in a FIXED order —
//! registry slot order, within a slot the type's [`IoWord`] schema order, then
//! the public slots, then padding to `2^c` — and that order is digest-visible
//! ([`Circuit::digest`]).
//!
//! A gate cell maps to a committed word by bit-concatenation:
//!
//! ```text
//! word(ι, j) = [ slot_prefix(t) | word_col(ι) | j ]      (LSB-first: j low)
//!            = (o_t >> 7) + (word_col << ν) + j
//! ```
//!
//! — the union's own BatchMajor addressing ([`crate::union::UnionInstance::
//! slot_word_range`]), which is why a gather term is an ordinary packed-direct
//! claim at a point that is random in the row coordinates and Boolean in every
//! high coordinate. `cell_addressing_matches_the_union` pins this against the
//! union machinery for every cell-slot of a mixed registry.
//!
//! The second index space is the point: running the permutation over union
//! addresses would pay a factor per TRACE word (the SHA-256 slot alone is
//! `2^18` words of round intermediates no wire touches, plus the dead
//! inter-class space); the cell space pays per SCHEMA word.
//!
//! ## Why a permutation proves the copy constraints
//!
//! Wires are equivalence classes of cells; σ rotates each class cyclically and
//! fixes everything else. The multiset identity
//! `{(w_x, x)} = {(w_x, σ(x))}` holds iff `w` is constant
//! on σ's orbits — i.e. iff every wire's cells agree. [`crate::product_gkr`]
//! proves exactly that identity as a grand product, at `f = g = w`, with no
//! committed oracle. Direction never enters: σ neither knows nor cares which
//! cell of a class is the producer.
//!
//! ## What the caller owes (the PIOP contract)
//!
//! [`prove_wiring`] / [`verify_wiring`] are a PIOP fragment, sound only if:
//!
//! 1. the statement — registry, counts, commitment root, **circuit digest and
//!    public words** — is bound into the challenger BEFORE the call
//!    ([`crate::union::UnionInstance::bind_statement_circuit`]); `α, β` are
//!    squeezed at the GKR's entry;
//! 2. BOTH surfaced evaluations are bound to the committed witness: the
//!    verifier checks the gather values recombine to `f_eval` AND that
//!    `f_eval == g_eval`. Binding only one leaves the other product's input
//!    vector prover-chosen — a real forgery vector, tamper-tested in
//!    `flock-prover/tests/circuit_wiring.rs`;
//! 3. verification uses the σ-AWARE verifier ([`product_gkr::
//!    verify_batched_with_sigma`]), never the trusting variant;
//! 4. the returned gather claims reach the PCS opening, which is what binds
//!    them to the commitment (and observes their values before `γ`);
//! 5. **dead rows hold zero io words** in the committed witness: every row
//!    `j ≥ count(ty)` of every gate slot the wiring or the bus gathers reads
//!    `0` in each io word. A gather claim is the evaluation of the WHOLE
//!    committed column, dead rows included, while the verifier rebuilds the
//!    leaves as if those words were zero.
//!    - *What rests on it.* The bus's grinding: a dead row with words `w_k`
//!      is a β-free leaf `1 + Σ_k γ_k·w_k` that cannot pair with a live one,
//!      but it adds a factor to the products, so the Schwartz–Zippel degree
//!      becomes the covered rows (up to ports × `2^ν`) instead of the live
//!      entries the fingerprint grind is sized to
//!      ([`product_gkr::BatchedGrinding::fingerprint_target_for`]). The
//!      wiring's grand product: a dead leaf is `w + 1`, so a dead word of `1`
//!      would zero both products. The product verifier rejects a zero
//!      product ([`product_gkr::ProductGkrError::ZeroProduct`]), which takes
//!      the grand products (wiring and bus) off the invariant; any other dead
//!      word scales both wiring products alike. The bus's degree bound still
//!      needs it.
//!    - *Where it is enforced.* Under the default height-`n_t` stacking the
//!      dead rows are not committed: the opening reads them as zeros. Under
//!      [`crate::union::UnionInstance::set_full_height_commit`] (boolean
//!      slots only) they are committed and constrained by the relation
//!      alone: the lincheck's const-pin count claim holds the pin at `0`
//!      there, and every io word's rows read `(A z)·pin = x` (an input's
//!      `x·pin = x`), which forces `x = 0`. A slot that commits dummy rows
//!      therefore keeps those rows in the zerocheck
//!      ([`crate::union::UnionInstance::zeroed_word_cols`] returns nothing
//!      for it); `full_height_dummy_row_input_word_is_rejected`
//!      (`tests/union_lincheck.rs`) commits one. The honest prover commits
//!      dummy rows as zeros whatever its buffers hold
//!      ([`crate::union::UnionInstance::compact_witness`]). A public
//!      column's claim ([`Circuit::with_public_columns_opened`]) sums its
//!      live rows only, so its opening forces zero on its dead rows by
//!      itself.

use std::{env::var, sync::OnceLock, time::Instant};

use blake3::hash;
use product_gkr::{
    BatchedGrinding, LiveMask, ProductGkrBatchedProof, ProductGkrError,
    prove_batched_with_grinding, s_id_basis, verify_batched_with_grinding,
    verify_batched_with_sigma_and_grinding,
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
};
use serde::{Deserialize, Serialize};

use crate::{
    challenger::Challenger,
    circuit::bus::{Bus, BusColumn, BusProof, BusStructure},
    field::F128,
    matrix_fold::{FoldMatrix, MatrixClaim, Weight, bilinear},
    pcs::{DirectEqInd, PackedDirectClaim},
    product_gkr,
    schedule::{IoDirection, IoWord, Registry},
    scratch::{give_f128, take_f128},
    union::UnionInstance,
    zerocheck::univariate_skip::build_eq,
};
pub mod boolean;
pub mod builder;
pub mod bus;

/// Domain label of the circuit digest — versioned, since the digest covers the
/// whole circuit encoding (cell-slot enumeration, counts, wiring, public
/// layout).
const CIRCUIT_LABEL: &[u8] = b"flock-circuit-v1";

// ---------------------------------------------------------------------------
// The cell space
// ---------------------------------------------------------------------------

/// What a cell-slot holds. The enumeration order of [`CellSpace::slots`] is
/// gate slots (registry order × schema order), then public slots, then
/// [`CellSlot::Pad`] out to `2^c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellSlot {
    /// One schema word of one registry type: `ty` is the registry SLOT index.
    Gate { ty: usize, word: IoWord },
    /// `2^ν` consecutive public words (`s` is the public slot's ordinal).
    Public { s: usize },
    /// Enumeration padding out to `2^c`. Never live, never wired, `w = 0`,
    /// σ-fixed.
    Pad,
}

/// Per-gate-slot geometry, derived once from the registry: everything the
/// prover's gather fold and the claim point need, with no registry lookup at
/// use time. (What the slot IS — type, word-column, direction — lives in the
/// parallel [`CellSlot::Gate`] entry.)
#[derive(Clone, Debug)]
struct GateGeom {
    /// Union WORD index of (this cell-slot, row 0): `(o_t >> 7) + (col << ν)`.
    /// Row `j`'s word is `word_base + j` — the rows are the low bits.
    word_base: usize,
    /// The claim point's frozen high coordinates, LSB-first: `bits(word_col)`
    /// then the slot prefix's bits. Length `(M − 7) − ν`.
    high_bits: Vec<bool>,
}

/// One cell: a cell-slot and a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    pub slot: usize,
    pub row: usize,
}

impl Cell {
    pub const fn new(slot: usize, row: usize) -> Self {
        Self { slot, row }
    }
}

/// The enumerated cell space of a registry plus a public segment: the index
/// space `{0,1}^μ` the wiring argument speaks, and the map from gate cells to
/// committed words.
#[derive(Clone, Debug)]
pub struct CellSpace {
    nu: usize,
    c: usize,
    /// Length `2^c`, in enumeration order.
    slots: Vec<CellSlot>,
    /// Parallel to the gate prefix of `slots`.
    gates: Vec<GateGeom>,
    num_public_slots: usize,
    /// `M − 7`: the length of a packed-direct claim point over the union.
    m_words: usize,
}

impl CellSpace {
    /// Enumerate the cell space of `registry` with `num_public` public WORDS.
    ///
    /// Public words are laid out `2^ν` to a slot: public word `p` is cell
    /// `(num_gate_slots + (p >> ν), p & (2^ν − 1))`. The trailing rows of the
    /// last public slot — and every [`CellSlot::Pad`] slot — are dummy cells.
    pub fn new(registry: &Registry, num_public: usize) -> Self {
        let nu = registry.nu();
        let m_words = registry.m_total() - 7;
        let mut slots = Vec::new();
        let mut gates = Vec::new();
        for (t, (ty, slot)) in registry.types().iter().zip(registry.slots()).enumerate() {
            for &word in &ty.io_schema {
                // `[slot_prefix | word_col | row]` — the union's BatchMajor
                // word index, with the row bits low.
                let word_base = (slot.offset >> 7) + (word.word_col << nu);
                let col_bits = ty.k_log - 7;
                let mut high_bits = Vec::with_capacity(m_words - nu);
                for i in 0..col_bits {
                    high_bits.push((word.word_col >> i) & 1 == 1);
                }
                for i in 0..slot.prefix_bits {
                    high_bits.push((slot.prefix >> i) & 1 == 1);
                }
                debug_assert_eq!(high_bits.len(), m_words - nu);
                gates.push(GateGeom {
                    word_base,
                    high_bits,
                });
                slots.push(CellSlot::Gate { ty: t, word });
            }
        }
        let num_public_slots = num_public.div_ceil(1usize << nu);
        for s in 0..num_public_slots {
            slots.push(CellSlot::Public { s });
        }
        let c = slots.len().max(1).next_power_of_two().trailing_zeros() as usize;
        slots.resize(1usize << c, CellSlot::Pad);
        Self {
            nu,
            c,
            slots,
            gates,
            num_public_slots,
            m_words,
        }
    }

    /// `ν` — the row variables, shared with the union (uniform capacity).
    pub fn nu(&self) -> usize {
        self.nu
    }

    /// Per cell-slot LIVE row counts: gate slots live up to their type's
    /// declared count, public slots up to the public segment's tail, pads
    /// never. This is the wiring grand product's [`product_gkr::LiveMask`]
    /// — dead cells become identity leaves and the prover skips them.
    pub fn live_counts(&self, counts: &[usize], num_public: usize) -> Vec<usize> {
        let rows = 1usize << self.nu;
        self.slots
            .iter()
            .map(|s| match *s {
                CellSlot::Gate { ty, .. } => counts[ty].min(rows),
                CellSlot::Public { s } => rows.min(num_public.saturating_sub(s << self.nu)),
                CellSlot::Pad => 0,
            })
            .collect()
    }
    /// `c` — the cell-slot variables.
    pub fn c_bits(&self) -> usize {
        self.c
    }
    /// `μ = ν + c`: the wiring argument runs over `2^μ` cells.
    pub fn mu(&self) -> usize {
        self.nu + self.c
    }
    /// The cell-slots, in enumeration order (length `2^c`).
    pub fn slots(&self) -> &[CellSlot] {
        &self.slots
    }
    /// Number of GATE cell-slots — the prefix of [`Self::slots`], and the
    /// number of gather claims.
    pub fn num_gate_slots(&self) -> usize {
        self.gates.len()
    }
    pub fn num_public_slots(&self) -> usize {
        self.num_public_slots
    }

    /// Cell index of `(slot, row)`: `[ι | j]`, rows low.
    pub fn cell_index(&self, cell: Cell) -> usize {
        (cell.slot << self.nu) | cell.row
    }

    /// The committed word index gate cell-slot `iota`'s row `j` reads.
    pub fn gate_word_addr(&self, iota: usize, row: usize) -> usize {
        self.gates[iota].word_base + row
    }

    /// The packed-direct claim point of gate cell-slot `iota` at row point
    /// `rho_row`: `[ρ_row ‖ bits(word_col) ‖ bits(slot_prefix)]`, LSB-first,
    /// length `M − 7`. Every coordinate past `ρ_row` is Boolean, which is what
    /// makes the eq tensor [`DirectEqInd::Sparse`].
    pub fn gate_claim_point(&self, iota: usize, rho_row: &[F128]) -> Vec<F128> {
        assert_eq!(rho_row.len(), self.nu, "row point must be ν coordinates");
        let mut point = Vec::with_capacity(self.m_words);
        point.extend_from_slice(rho_row);
        point.extend(
            self.gates[iota]
                .high_bits
                .iter()
                .map(|&b| if b { F128::ONE } else { F128::ZERO }),
        );
        point
    }
}

// ---------------------------------------------------------------------------
// The circuit
// ---------------------------------------------------------------------------

/// Why a circuit is invalid. Every one of these is a construction-time hard
/// error: an invalid circuit has no proof, and several of them (dummy cells,
/// repeated cells) would silently break σ's permutation property.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CircuitError {
    /// `counts` has the wrong length, or a count exceeds the row capacity.
    Counts {
        got: usize,
        want: usize,
    },
    CountOverCapacity {
        ty: usize,
        n: usize,
    },
    /// A wire names a cell-slot or row outside the cell space.
    UnknownCell(Cell),
    /// A wire names a DUMMY cell: a pad slot, a gate row past that type's
    /// count, or an unused row of the last public slot. Dummy cells hold zero
    /// and must be σ-fixed.
    DummyCell(Cell),
    /// A cell appears in two classes, or twice in one.
    RepeatedCell(Cell),
    EmptyClass,
    /// The gate dataflow (producer → consumer) has a cycle.
    Cyclic,
    /// σ is not a permutation. Unreachable given disjointness; checked anyway,
    /// because completeness (and the verifier's `s_σ`) rests on it.
    NotAPermutation,
    /// The tuple bus does not fit the circuit ([`bus::Bus`]).
    InvalidBus(&'static str),
}

/// A circuit over a registry: per-type gate counts, a public segment, and a
/// wiring.
///
/// Validated at construction (see [`CircuitError`]): every wired cell exists
/// and is live, classes are disjoint, the dataflow is acyclic, and σ is a
/// permutation. Multiple producers per class are allowed — the permutation
/// forces every cell of a class equal, so a multi-producer class IS the
/// circuit's assert_eq between computed values, and witness generation
/// asserts the producers agree. Semantics are satisfiability — that a
/// satisfying assignment computes "the" output is a property of the circuit
/// (acyclicity + producer agreement), checked by whoever endorses the
/// digest, exactly as in Plonk.
///
/// The wiring is stored CANONICALLY (cells ascending within a class, classes by
/// their least cell), so the digest is a function of the wire PARTITION and σ
/// is a function of the digest — prover and verifier cannot disagree about the
/// rotation.
#[derive(Clone, Debug)]
pub struct Circuit {
    cells: CellSpace,
    /// Immutable gate/type schedule used to construct this circuit. The
    /// digest already binds it; retaining it lets root discharges evaluate
    /// auxiliary circuit-static tables without a separate trusted argument.
    registry: Registry,
    registry_digest: [u8; 32],
    counts: Vec<usize>,
    num_public: usize,
    /// Public words whose values are part of the circuit definition rather
    /// than supplied by the application statement. `None` denotes an ordinary
    /// public input; `Some(v)` requires that public entry to equal `v`.
    ///
    /// Fixed publics are digest-bound. This lets circuits use shared constant
    /// wires without trusting the proof producer to choose their values.
    fixed_public: Vec<Option<F128>>,
    wires: Vec<Vec<usize>>,
    sigma: Vec<usize>,
    /// Content-addressed dataflow beside the wiring ([`bus`]).
    bus: Option<Bus>,
    /// Per gate cell-slot: no wire touches it, so it is left out of the
    /// wiring's grand product ([`Self::with_unwired_pruned`]). Empty when
    /// pruning is off.
    pruned: Vec<bool>,
    /// Per gate cell-slot: every live cell is wired to a public word, so the
    /// opening checks the column directly ([`Self::with_public_columns_opened`]).
    /// `Some(p)` holds the public word index of each live row. Empty when off.
    public_columns: Vec<Option<Vec<u32>>>,
    /// Each cell-slot's position in the wiring's grand product
    /// ([`Self::with_live_slots_packed`]). Empty: the enumeration order.
    gkr_slot: Vec<usize>,
    digest_cache: OnceLock<[u8; 32]>,
}

impl Circuit {
    /// Build and validate a circuit. `counts` is one gate count per REGISTRY
    /// type in slot order (types with an empty schema still declare theirs —
    /// the counts determine the union's counts); `num_public` is the number of
    /// public words; `wires` is the list of wire classes.
    pub fn new(
        registry: &Registry,
        counts: Vec<usize>,
        num_public: usize,
        wires: Vec<Vec<Cell>>,
    ) -> Result<Self, CircuitError> {
        Self::new_with_fixed_public(registry, counts, vec![None; num_public], wires)
    }

    /// Build a circuit with an optional fixed value for each public word.
    /// Fixed values are included in the circuit digest and checked by both the
    /// circuit prover and verifier entry points.
    pub fn new_with_fixed_public(
        registry: &Registry,
        counts: Vec<usize>,
        fixed_public: Vec<Option<F128>>,
        wires: Vec<Vec<Cell>>,
    ) -> Result<Self, CircuitError> {
        let num_public = fixed_public.len();
        if counts.len() != registry.num_types() {
            return Err(CircuitError::Counts {
                got: counts.len(),
                want: registry.num_types(),
            });
        }
        let nu = registry.nu();
        for (t, &n) in counts.iter().enumerate() {
            if n > 1usize << nu {
                return Err(CircuitError::CountOverCapacity { ty: t, n });
            }
        }
        let cells = CellSpace::new(registry, num_public);
        let n_cells = 1usize << cells.mu();

        // ---- Liveness + disjointness, canonicalizing as we go.
        let mut owner = vec![u32::MAX; n_cells];
        let mut classes: Vec<Vec<usize>> = Vec::with_capacity(wires.len());
        for (ci, class) in wires.iter().enumerate() {
            if class.is_empty() {
                return Err(CircuitError::EmptyClass);
            }
            let mut idxs = Vec::with_capacity(class.len());
            for &cell in class {
                if cell.slot >= cells.slots().len() || cell.row >= 1usize << nu {
                    return Err(CircuitError::UnknownCell(cell));
                }
                let live = match cells.slots()[cell.slot] {
                    CellSlot::Gate { ty, .. } => cell.row < counts[ty],
                    CellSlot::Public { s } => (s << nu) + cell.row < num_public,
                    CellSlot::Pad => false,
                };
                if !live {
                    return Err(CircuitError::DummyCell(cell));
                }
                let idx = cells.cell_index(cell);
                if owner[idx] != u32::MAX {
                    return Err(CircuitError::RepeatedCell(cell));
                }
                owner[idx] = ci as u32;
                idxs.push(idx);
            }
            idxs.sort_unstable();
            classes.push(idxs);
        }
        classes.sort_unstable_by_key(|c| c[0]);
        // The canonical order moved the class indices; rebuild `owner` against
        // it (the dataflow pass below reads it).
        for (ci, class) in classes.iter().enumerate() {
            for &idx in class {
                owner[idx] = ci as u32;
            }
        }

        // ---- Producers per class. MULTIPLE producers are allowed: the
        // permutation argument forces every cell of a class equal, so a
        // class holding two gate outputs IS the circuit's assert_eq between
        // computed values (the zero-delta pattern without a published
        // delta), and witness generation asserts the second producer agrees
        // ([`builder::CircuitShape::run`]). The dataflow check below draws
        // edges from EVERY producer to every consumer.
        let mut producers: Vec<Vec<usize>> = vec![Vec::new(); classes.len()];
        for (ci, class) in classes.iter().enumerate() {
            for &idx in class {
                if let CellSlot::Gate { word, .. } = cells.slots()[idx >> nu]
                    && word.dir == IoDirection::Out
                {
                    producers[ci].push(idx);
                }
            }
        }

        // ---- Acyclic dataflow over the live gates. A gate is (type, row); an
        // edge runs from the gate producing a class to every gate consuming it.
        // Kahn's algorithm over gate ids `gate_base[ty] + row`.
        let mut gate_base = Vec::with_capacity(counts.len());
        let mut total_gates = 0usize;
        for &n in &counts {
            gate_base.push(total_gates);
            total_gates += n;
        }
        let gate_of = |idx: usize| -> Option<usize> {
            let (iota, row) = (idx >> nu, idx & ((1usize << nu) - 1));
            match cells.slots()[iota] {
                CellSlot::Gate { ty, .. } => Some(gate_base[ty] + row),
                _ => None,
            }
        };
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); total_gates];
        let mut indeg = vec![0usize; total_gates];
        for (ci, class) in classes.iter().enumerate() {
            for &prod_idx in &producers[ci] {
                let from = gate_of(prod_idx).expect("a producer cell is a gate cell");
                for &idx in class {
                    let iota = idx >> nu;
                    let CellSlot::Gate { word, .. } = cells.slots()[iota] else {
                        continue;
                    };
                    if word.dir != IoDirection::In {
                        continue;
                    }
                    let to = gate_of(idx).expect("gate cell");
                    succ[from].push(to);
                    indeg[to] += 1;
                }
            }
        }
        let mut queue: Vec<usize> = (0..total_gates).filter(|&g| indeg[g] == 0).collect();
        let mut seen = 0usize;
        while let Some(g) = queue.pop() {
            seen += 1;
            for &h in &succ[g] {
                indeg[h] -= 1;
                if indeg[h] == 0 {
                    queue.push(h);
                }
            }
        }
        if seen != total_gates {
            return Err(CircuitError::Cyclic);
        }

        // ---- σ: rotate each class cyclically, fix everything else.
        let mut sigma: Vec<usize> = (0..n_cells).collect();
        for class in &classes {
            for (k, &idx) in class.iter().enumerate() {
                sigma[idx] = class[(k + 1) % class.len()];
            }
        }
        let mut hit = vec![false; n_cells];
        for &s in &sigma {
            if hit[s] {
                return Err(CircuitError::NotAPermutation);
            }
            hit[s] = true;
        }

        let c = Self {
            cells,
            registry: registry.clone(),
            registry_digest: registry.digest(),
            counts,
            num_public,
            fixed_public,
            wires: classes,
            sigma,
            bus: None,
            pruned: Vec::new(),
            public_columns: Vec::new(),
            gkr_slot: Vec::new(),
            digest_cache: OnceLock::new(),
        };
        // Warm the digest here, at statement-construction time: it is a pure
        // function of the shape, and leaving it lazy put the first (only)
        // computation inside the PROVE's bind phase for any caller that
        // builds a fresh circuit — measured 7.6 ms per proof at the
        // recursion node before this.
        c.digest();
        Ok(c)
    }

    pub fn cells(&self) -> &CellSpace {
        &self.cells
    }
    /// Per-registry-type gate counts, in slot order. These DETERMINE the
    /// union's declared counts — [`Self::check_instance`] enforces equality.
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// The wiring grand product's live mask — statement-derived (counts +
    /// public length are part of the circuit encoding), shared by prover,
    /// verifier, the in-circuit transcription's checker, and the sigma
    /// discharge, so the mask cannot drift between them.
    pub fn live_mask(&self) -> LiveMask {
        let mut counts = self.cells.live_counts(&self.counts, self.num_public);
        for (count, &pruned) in counts.iter_mut().zip(&self.pruned) {
            if pruned {
                *count = 0;
            }
        }
        for (count, column) in counts.iter_mut().zip(&self.public_columns) {
            if column.is_some() {
                *count = 0;
            }
        }
        if !self.gkr_slot.is_empty() {
            let mut packed = vec![0; counts.len()];
            for (iota, &q) in self.gkr_slot.iter().enumerate() {
                packed[q] = counts[iota];
            }
            counts = packed;
        }
        LiveMask {
            nu: self.cells.nu(),
            counts,
        }
    }

    /// Leave every gate cell-slot that no wire touches out of the wiring
    /// argument. Such a slot's cells are σ-fixed, so each contributes the
    /// same factor `w + α·x + β` to both grand products: dropping them keeps
    /// the identity, and they need no gather claim. Worth it when a circuit
    /// has whole io words that only the bus reads, or none. Digest-bound.
    pub fn with_unwired_pruned(mut self) -> Self {
        assert!(self.gkr_slot.is_empty(), "pack the live slots last");
        let nu = self.cells.nu();
        let mut wired = vec![false; self.cells.num_gate_slots()];
        for &idx in self.wires.iter().flatten() {
            if let Some(w) = wired.get_mut(idx >> nu) {
                *w = true;
            }
        }
        self.pruned = wired.iter().map(|&w| !w).collect();
        self.digest_cache = OnceLock::new();
        self.digest();
        self
    }

    /// Take every gate cell-slot whose live cells are all wired to public
    /// words out of the wiring's grand product. The verifier knows such a
    /// column, so it keeps its gather claim, but the verifier computes the
    /// value from the public words instead of reading it from the proof: the
    /// opening checks the column directly. Its cells leave their classes (σ
    /// rotates what remains of each) and the grand product skips them. Worth
    /// it when gates take constants or per-statement public words in whole
    /// io words. Digest-bound.
    ///
    /// Sound for the same reason the gather is: the claim's row point comes
    /// from the grand product, after the commitment, so one evaluation pins
    /// the whole committed column (the public words on live rows, zero on
    /// dummy rows) up to `ν/|F|`. What remains of each class still ties its
    /// other cells to the public word.
    pub fn with_public_columns_opened(mut self) -> Self {
        assert!(self.gkr_slot.is_empty(), "pack the live slots last");
        let (nu, ng) = (self.cells.nu(), self.cells.num_gate_slots());
        let rows = 1usize << nu;
        // The public word each wired cell's class holds, if any. Public
        // slots follow the gate slots, so a class's first public cell is its
        // first index past the gate cells.
        let mut public_of = vec![u32::MAX; 1usize << self.cells.mu()];
        for class in &self.wires {
            let Some(&p) = class.iter().find(|&&x| x >> nu >= ng) else {
                continue;
            };
            let CellSlot::Public { s } = self.cells.slots()[p >> nu] else {
                unreachable!("wired cells past the gate slots are public cells")
            };
            let word = ((s << nu) | (p & (rows - 1))) as u32;
            for &x in class {
                public_of[x] = word;
            }
        }
        let mut columns = vec![None; ng];
        for iota in self.gathered_slots() {
            let CellSlot::Gate { ty, .. } = self.cells.slots()[iota] else {
                unreachable!("the first num_gate_slots cell-slots are gate slots")
            };
            let live = self.counts[ty].min(rows);
            let words = &public_of[iota << nu..(iota << nu) + live];
            if live > 0 && words.iter().all(|&p| p != u32::MAX) {
                columns[iota] = Some(words.to_vec());
            }
        }
        if columns.iter().all(Option::is_none) {
            return self;
        }
        let out = |x: usize| columns.get(x >> nu).is_some_and(Option::is_some);
        let mut sigma: Vec<usize> = (0..1usize << self.cells.mu()).collect();
        for class in &self.wires {
            let kept: Vec<usize> = class.iter().copied().filter(|&x| !out(x)).collect();
            for (k, &x) in kept.iter().enumerate() {
                sigma[x] = kept[(k + 1) % kept.len()];
            }
        }
        self.sigma = sigma;
        self.public_columns = columns;
        self.digest_cache = OnceLock::new();
        self.digest();
        self
    }

    /// Lay the grand product's cell-slots out so that live slots meet early
    /// in its product tree. The tree merges slot `q` with `q + 2^(c−1)`
    /// first, then with `q + 2^(c−2)`, and so on, and a merged layer is as
    /// long as its longer input: with pruned or public columns scattered
    /// among live ones, every merge paired a live slot with a dead one and
    /// the bottom layers stayed about as long as the leaves. The slots,
    /// sorted by live count, go to bit-reversed positions, so neighbours in
    /// that order merge first. The gather and the public columns are per
    /// cell-slot and do not move; σ and the live mask are in the grand
    /// product's positions. Call it after pruning and opening public
    /// columns. Digest-bound.
    pub fn with_live_slots_packed(mut self) -> Self {
        assert!(self.gkr_slot.is_empty(), "the slots are already packed");
        let (nu, c) = (self.cells.nu(), self.cells.c_bits());
        if c == 0 {
            return self;
        }
        let counts = self.live_mask().counts;
        let mut order: Vec<usize> = (0..counts.len()).collect();
        order.sort_by_key(|&iota| std::cmp::Reverse(counts[iota]));
        let mut gkr_slot = vec![0usize; counts.len()];
        for (k, &iota) in order.iter().enumerate() {
            gkr_slot[iota] = k.reverse_bits() >> (usize::BITS as usize - c);
        }
        if gkr_slot.iter().enumerate().all(|(iota, &q)| iota == q) {
            return self;
        }
        let row_mask = (1usize << nu) - 1;
        let at = |x: usize| (gkr_slot[x >> nu] << nu) | (x & row_mask);
        let mut sigma = vec![0usize; self.sigma.len()];
        for (x, &s) in self.sigma.iter().enumerate() {
            sigma[at(x)] = at(s);
        }
        self.sigma = sigma;
        self.gkr_slot = gkr_slot;
        self.digest_cache = OnceLock::new();
        self.digest();
        self
    }

    /// Cell-slot `iota`'s position in the grand product
    /// ([`Self::with_live_slots_packed`]).
    pub fn gkr_slot(&self, iota: usize) -> usize {
        self.gkr_slot.get(iota).copied().unwrap_or(iota)
    }

    /// Whether [`Self::with_live_slots_packed`] moved any cell-slot.
    pub fn live_slots_packed(&self) -> bool {
        !self.gkr_slot.is_empty()
    }

    /// The gate cell-slots the wiring's grand product gathers, ascending:
    /// every one unless [`Self::with_unwired_pruned`] dropped it or
    /// [`Self::with_public_columns_opened`] took it out.
    pub fn gathered_slots(&self) -> Vec<usize> {
        (0..self.cells.num_gate_slots())
            .filter(|&iota| !self.pruned.get(iota).copied().unwrap_or(false))
            .filter(|&iota| self.public_columns.get(iota).is_none_or(Option::is_none))
            .collect()
    }

    /// The gate cell-slots [`Self::with_public_columns_opened`] took out of
    /// the grand product, ascending, each with the public word index of every
    /// live row.
    pub fn public_columns(&self) -> impl Iterator<Item = (usize, &[u32])> {
        self.public_columns
            .iter()
            .enumerate()
            .filter_map(|(iota, words)| words.as_deref().map(|w| (iota, w)))
    }
    /// Number of public words.
    pub fn num_public(&self) -> usize {
        self.num_public
    }
    /// Public word `i`'s digest-bound value, when the circuit fixes it.
    pub fn fixed_public(&self, i: usize) -> Option<F128> {
        self.fixed_public.get(i).copied().flatten()
    }
    /// The digest-bound gate/type schedule.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
    /// Check the public segment's shape and every digest-bound constant.
    pub fn check_public(&self, public: &[F128]) -> bool {
        public.len() == self.num_public
            && self
                .fixed_public
                .iter()
                .zip(public)
                .all(|(fixed, value)| fixed.is_none_or(|want| want == *value))
    }
    /// The wire classes, canonical: cell indices ascending within a class,
    /// classes ordered by their least cell.
    pub fn wires(&self) -> &[Vec<usize>] {
        &self.wires
    }
    /// σ over the cell space, length `2^μ`. The verifier evaluates `ŝ_σ(ρ)`
    /// from this in v1 ([`product_gkr::verify_batched_with_sigma`]).
    pub fn sigma(&self) -> &[usize] {
        &self.sigma
    }

    /// Attach a tuple bus ([`bus`]): proven beside the wiring and bound into
    /// the digest.
    pub fn with_bus(mut self, bus: Bus) -> Result<Self, CircuitError> {
        bus.validate(&self)?;
        self.bus = Some(bus);
        self.digest_cache = OnceLock::new();
        self.digest();
        Ok(self)
    }

    /// The circuit's tuple bus, if any.
    pub fn bus(&self) -> Option<&Bus> {
        self.bus.as_ref()
    }

    /// The circuit digest — the statement's circuit half.
    ///
    /// Absorption order (format version 1 for circuits without fixed publics,
    /// format version 2 otherwise): the domain label
    /// [`CIRCUIT_LABEL`], a version byte, the REGISTRY digest, `ν` and `c`,
    /// the schema-derived cell-slot enumeration (gate-slot count, then each
    /// gate slot's registry type, word-column and direction; then the public
    /// slot count), the gate counts, the public-word count, and the canonical
    /// wire encoding (class count, then per class its length and its cell
    /// indices). Every field is fixed-width or length-prefixed, so the
    /// encoding is injective.
    pub fn digest(&self) -> [u8; 32] {
        *self.digest_cache.get_or_init(|| {
            // The payload is serialized into ONE buffer and hashed in a
            // single update: the wire encoding is hundreds of thousands of
            // 8-byte fields, and per-field `Hasher::update` calls made this
            // ~7.6 ms at the recursion node — measured inside the PROVE's
            // bind phase, because the first call used to happen there (the
            // cache is now warmed by `Circuit::new`). Byte-identical to the
            // per-field updates it replaces.
            let n_cells: usize = self.wires.iter().map(|c| c.len()).sum();
            let mut buf: Vec<u8> = Vec::with_capacity(
                256 + 9 * self.cells.num_gate_slots()
                    + 12 * self.counts.len()
                    + 4 * self.wires.len()
                    + 8 * n_cells,
            );
            buf.extend_from_slice(CIRCUIT_LABEL);
            let has_fixed_public = self.fixed_public.iter().any(Option::is_some);
            buf.push(if has_fixed_public { 2u8 } else { 1u8 });
            buf.extend_from_slice(&self.registry_digest);
            buf.extend_from_slice(&(self.cells.nu() as u32).to_le_bytes());
            buf.extend_from_slice(&(self.cells.c_bits() as u32).to_le_bytes());
            buf.extend_from_slice(&(self.cells.num_gate_slots() as u32).to_le_bytes());
            for slot in &self.cells.slots()[..self.cells.num_gate_slots()] {
                let CellSlot::Gate { ty, word } = slot else {
                    unreachable!("the gate prefix holds gate slots")
                };
                buf.extend_from_slice(&(*ty as u32).to_le_bytes());
                buf.extend_from_slice(&(word.word_col as u32).to_le_bytes());
                buf.push(matches!(word.dir, IoDirection::Out) as u8);
            }
            buf.extend_from_slice(&(self.cells.num_public_slots() as u32).to_le_bytes());
            buf.extend_from_slice(&(self.counts.len() as u32).to_le_bytes());
            for &n in &self.counts {
                buf.extend_from_slice(&(n as u64).to_le_bytes());
            }
            buf.extend_from_slice(&(self.num_public as u64).to_le_bytes());
            if has_fixed_public {
                for fixed in &self.fixed_public {
                    match fixed {
                        None => buf.push(0),
                        Some(value) => {
                            buf.push(1);
                            buf.extend_from_slice(&value.lo.to_le_bytes());
                            buf.extend_from_slice(&value.hi.to_le_bytes());
                        }
                    }
                }
            }
            buf.extend_from_slice(&(self.wires.len() as u32).to_le_bytes());
            for class in &self.wires {
                buf.extend_from_slice(&(class.len() as u32).to_le_bytes());
                for &idx in class {
                    buf.extend_from_slice(&(idx as u64).to_le_bytes());
                }
            }
            // Pruning, public columns, the slot order and a bus are appended
            // after the wiring, so circuits with none of them keep their
            // digests.
            if !self.pruned.is_empty() {
                buf.extend_from_slice(b"pruned-unwired");
                buf.extend(self.pruned.iter().map(|&p| p as u8));
            }
            // Each public column's words follow from the wiring above.
            if !self.public_columns.is_empty() {
                buf.extend_from_slice(b"public-columns");
                buf.extend(self.public_columns.iter().map(|c| c.is_some() as u8));
            }
            if !self.gkr_slot.is_empty() {
                buf.extend_from_slice(b"gkr-slot-order");
                for &q in &self.gkr_slot {
                    buf.extend_from_slice(&(q as u32).to_le_bytes());
                }
            }
            if let Some(bus) = &self.bus {
                bus.absorb_into(&mut buf);
            }
            *hash(&buf).as_bytes()
        })
    }

    /// The circuit and the union instance must describe the SAME statement:
    /// same registry, and the union's declared counts are the circuit's gate
    /// counts. Called by both the prove and the verify entry.
    pub fn check_instance(&self, union: &UnionInstance<'_>) -> bool {
        self.registry_digest == union.registry().digest()
            && self.counts == union.counts()
            && self.cells.m_words == union.m_total() - 7
    }

    /// `Σ_j eq(ρ_row, j) · public[(s << ν) + j]` — the MLE of public slot `s`'s
    /// word vector, zero on its unused rows. `O(2^ν)` per slot, so
    /// `O(#public)` over the segment.
    pub fn public_slot_eval(&self, s: usize, eq_row: &[F128], public: &[F128]) -> F128 {
        let nu = self.cells.nu();
        let base = s << nu;
        let hi = ((base + (1usize << nu)).min(public.len())).saturating_sub(base);
        let mut acc = F128::ZERO;
        for j in 0..hi {
            acc += eq_row[j] * public[base + j];
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// The wiring argument
// ---------------------------------------------------------------------------

/// The wiring half of a circuit proof: the product-GKR transcript over the cell
/// space, and the gather values (one per gathered gate cell-slot, in
/// enumeration order; [`Circuit::gathered_slots`]).
/// The gather claims' POINTS are transcript-derived, so only the values ride.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WiringProof {
    pub gkr: ProductGkrBatchedProof,
    pub gather: Vec<F128>,
    /// The tuple bus's argument, when the circuit has a bus.
    pub bus: Option<BusProof>,
}

/// Why a wiring proof was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WiringError {
    /// Wrong number of gather values for this circuit's cell space.
    MalformedProof,
    /// The product-GKR rejected (products differ, layer check, input check).
    Gkr(ProductGkrError),
    /// The gather values do not recombine to `ŵ(ρ)` — the gather
    /// factorization's check, which is what binds the GKR's `f`-side input to
    /// the committed witness and the public words.
    Recombination,
    /// `f_eval ≠ g_eval`: the two grand products ran on different input
    /// vectors. Binding only the `f` side would leave `g` prover-chosen.
    GEvalMismatch,
    /// The bus's grand product rejected.
    BusGkr(ProductGkrError),
    /// The gather values do not rebuild the bus products' leaf claims.
    BusRecombination,
    /// A deferred verify ([`verify_wiring_deferred`]) of a circuit whose
    /// public column reads a public word that is not fixed: the sigma
    /// assertion's structure table holds fixed words only.
    UnfixedPublicColumn,
}

/// Prove the circuit's wiring equalities against the committed witness.
///
/// `packed` is the PADDED union packed buffer — the same buffer the commitment
/// was built from (dummy rows are zero there by the union's witness contract,
/// which is what makes the dummy cells' `w = 0` honest). `public` is the
/// statement's public words, already bound into `ch`.
///
/// Returns the GKR transcript plus one [`PackedDirectClaim`] per gate
/// cell-slot; the caller must route those into the PCS opening (which observes
/// their values before `γ`).
pub fn prove_wiring<C: Challenger>(
    circuit: &Circuit,
    packed: &[F128],
    public: &[F128],
    ch: &mut C,
) -> (WiringProof, Vec<PackedDirectClaim>) {
    prove_wiring_with_grinding(circuit, packed, public, BatchedGrinding::disabled(), ch)
}

/// [`prove_wiring`] with explicit Product-GKR Fiat--Shamir grinding.
pub fn prove_wiring_with_grinding<C: Challenger>(
    circuit: &Circuit,
    packed: &[F128],
    public: &[F128],
    grinding: BatchedGrinding,
    ch: &mut C,
) -> (WiringProof, Vec<PackedDirectClaim>) {
    assert_eq!(
        public.len(),
        circuit.num_public(),
        "public words must match the circuit's public segment"
    );
    let cells = circuit.cells();
    let (nu, mu) = (cells.nu(), cells.mu());
    let rows = 1usize << nu;
    // Phase trace, `WIRING_TRACE=1` — the module's counterpart of `PCS_TRACE` /
    // `GKR_TRACE`, so the wiring overhead can be attributed (build w vs the
    // grand product vs the gather folds) without instrumenting the caller.
    let trace = var("WIRING_TRACE").is_ok();
    let t = Instant::now();

    // ---- w over the cell space. Gate cells read the committed buffer; public
    // cells take the statement words; dummy cells are zero (pooled buffers come
    // back dirty, so every slot is written).
    // LIVE PREFIXES ONLY: the grouped GKR's leaves and the live gather
    // never read a dead cell, so the pooled buffer's dead regions stay
    // untouched — they are SEMANTICALLY zero (the committed data's dead
    // rows are honest zeros; the buffer's are simply never read). This
    // makes the w materialization live-proportional.
    let mask = circuit.live_mask();
    // The grand product's slot `q` holds cell-slot `owner[q]`
    // ([`Circuit::with_live_slots_packed`]).
    let mut owner: Vec<usize> = (0..mask.counts.len()).collect();
    for iota in 0..owner.len() {
        owner[circuit.gkr_slot(iota)] = iota;
    }
    let mut w = take_f128(1usize << mu);
    w.par_chunks_mut(rows).enumerate().for_each(|(q, dst)| {
        let (iota, live) = (owner[q], mask.counts[q]);
        match cells.slots()[iota] {
            CellSlot::Gate { .. } => {
                let base = cells.gate_word_addr(iota, 0);
                dst[..live].copy_from_slice(&packed[base..base + live]);
            }
            CellSlot::Public { s } => {
                let base = s << nu;
                dst[..live].copy_from_slice(&public[base..base + live]);
            }
            CellSlot::Pad => {}
        }
    });

    if trace {
        eprintln!(
            "  [wiring] build w (live cells of 2^{mu}): {:7.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- One grand-product permutation check at f = g = w.
    let t = Instant::now();
    let (gkr, claim) =
        prove_batched_with_grinding(&w, &w, circuit.sigma(), Some(&mask), grinding, ch);
    give_f128(w);
    if trace {
        eprintln!(
            "  [wiring] product GKR (μ = {mu}): {:7.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- The gather: one eq-weighted row fold per gate cell-slot, O(2^ν)
    // each, producing packed-direct claims.
    let t = Instant::now();
    let eq_row = build_eq(&claim.rho[..nu]);
    let gather_claim = |iota: usize| -> PackedDirectClaim {
        let base = cells.gate_word_addr(iota, 0);
        let CellSlot::Gate { ty, .. } = cells.slots()[iota] else {
            unreachable!("the first num_gate_slots cell-slots are gate slots")
        };
        // Fold the LIVE prefix only: dummy rows are honest zeros in the
        // padded buffer (the element closure contract; the boolean padding
        // spec), so this is value-identical — and it makes the claim
        // independent of whatever a pooled buffer's dead tail holds,
        // instead of relying on it being clean.
        let live = circuit.counts()[ty].min(rows);
        let mut v = F128::ZERO;
        for (j, &e) in eq_row.iter().take(live).enumerate() {
            v += e * packed[base + j];
        }
        let point = cells.gate_claim_point(iota, &claim.rho[..nu]);
        PackedDirectClaim {
            value: v,
            // DEFERRED: the merged open (the only transport) never reads
            // `eq_ind` — it derives its identity-fold weights from
            // `point`/`value` alone — so the `2^nu`-entry tensor per gate
            // slot is never built. A claim
            // that DID need a materialized tensor would trip the combine's
            // "EqPoint claims are only supported alone" assert rather than
            // silently dropping the contribution.
            eq_ind: DirectEqInd::EqPoint(point.clone()),
            point,
        }
    };
    let mut claims: Vec<PackedDirectClaim> = circuit
        .gathered_slots()
        .into_iter()
        .map(gather_claim)
        .collect();
    let gather: Vec<F128> = claims.iter().map(|c| c.value).collect();
    // The public columns' claims follow, the same folds; the verifier
    // computes their values from the public words, so they stay out of the
    // proof.
    claims.extend(circuit.public_columns().map(|(iota, _)| gather_claim(iota)));
    if trace {
        eprintln!(
            "  [wiring] gather ({} claims, {} of them public columns, × 2^{nu}): {:7.2} ms",
            claims.len(),
            claims.len() - gather.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- The tuple bus, on the same transcript after the wiring GKR. Its
    // gather claims follow the wiring's into the opening.
    let bus = circuit.bus().map(|bus| {
        let t = Instant::now();
        let (proof, bus_claims) = bus::prove_bus(circuit, bus, packed, grinding, ch);
        if trace {
            eprintln!(
                "  [wiring] tuple bus ({} ports, {} claims): {:7.2} ms",
                bus.ports.len(),
                bus_claims.len(),
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        claims.extend(bus_claims);
        proof
    });

    (WiringProof { gkr, gather, bus }, claims)
}

/// Replay the wiring argument. Returns the gather claims as `(point, value)`
/// pairs for the PCS opening's packed-direct intake — in the same order the
/// prover emitted them.
///
/// Runs the σ-AWARE GKR verifier (v1: the verifier holds the circuit, so it
/// evaluates `ŝ_σ(ρ)` itself in `O(2^μ)` rather than trusting the proof), then
/// the two binding checks of the module contract.
/// The wiring GKR's deferred sigma evaluation (sigma v2 route B — design
/// doc §sigma): the claim `s_sigma_hat(rho) = value`, destined for the
/// accumulator's sigma family and the root discharge. `rho` is the GKR
/// endpoint over the `mu = nu + c` cell coordinates, row-low.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigmaAssertion {
    pub rho: Vec<F128>,
    pub nu: usize,
    /// Log2 of the shared circuit-structure table's base column domain.
    pub base_bits: usize,
    /// `MLE(live * s_id)(rho)`, used by the left Product-GKR input check.
    pub masked_id_value: F128,
    /// `MLE(live)(rho)`, used by both Product-GKR input checks.
    pub live_value: F128,
    /// `MLE(live * s_sigma)(rho)`, used by the right input check.
    pub value: F128,
    /// Boolean const-pin prefix evaluations. Each entry selects the Boolean
    /// registry slot at `type_index` and evaluates its live-row indicator at
    /// `point` (the zerocheck outer row point).
    pub boolean_pins: Vec<(usize, Vec<F128>, F128)>,
    /// Element affine-constant evaluations at the element constraint point:
    /// `(point, a_const_eval, b_const_eval)`.
    pub element_constants: Option<(Vec<F128>, F128, F128)>,
    /// The tuple bus's static columns at its row point
    /// ([`bus::Bus::structure_columns`]), when the circuit has a bus.
    pub bus: Option<BusStructure>,
    /// The public columns' values at the grand product's row point
    /// ([`Circuit::public_columns`], in that order), when the circuit opens
    /// any: each a function of fixed public words only, so a static column
    /// of the table (plane 7).
    #[serde(default)]
    pub public_columns: Option<BusStructure>,
}

impl SigmaAssertion {
    const MASKED_ID_PLANE: usize = 0;
    const LIVE_PLANE: usize = 1;
    const SIGMA_PLANE: usize = 2;
    const ELEMENT_A_PLANE: usize = 3;
    const ELEMENT_B_PLANE: usize = 4;
    const BOOLEAN_PIN_PLANE: usize = 5;
    const BUS_PLANE: usize = 6;
    const PUBLIC_COLUMN_PLANE: usize = 7;

    fn plane_claim(&self, plane: usize, value: F128) -> MatrixClaim {
        let selector = [
            F128::new((plane & 1) as u64, 0),
            F128::new(((plane >> 1) & 1) as u64, 0),
            F128::new(((plane >> 2) & 1) as u64, 0),
        ];
        let mut col_point = self.rho[self.nu..].to_vec();
        col_point.resize(self.base_bits, F128::ZERO);
        col_point.extend_from_slice(&selector);
        MatrixClaim {
            row: Weight::eq(self.rho[..self.nu].to_vec()),
            col: Weight::eq(col_point),
            value,
        }
    }

    /// The historical sigma claim, now selecting plane 2 of the circuit-
    /// structure table. Kept as a convenience for callers that need only the
    /// sigma member; aggregation must use [`Self::claims`] so the Product-GKR
    /// helper evaluations are bound as well.
    pub fn claim(&self) -> MatrixClaim {
        self.plane_claim(Self::SIGMA_PLANE, self.value)
    }

    /// All digest-keyed circuit-structure evaluations, in canonical order:
    /// Product-GKR's `(live*s_id, live, live*s_sigma)`, then Boolean
    /// constant-pin prefixes, then the element affine `A`/`B` constants,
    /// then the bus's static columns, then the public columns.
    pub fn claims(&self) -> Vec<MatrixClaim> {
        let mut claims = vec![
            self.plane_claim(Self::MASKED_ID_PLANE, self.masked_id_value),
            self.plane_claim(Self::LIVE_PLANE, self.live_value),
            self.claim(),
        ];
        for (type_index, point, value) in &self.boolean_pins {
            let base_bits = self.matrix_base_bits();
            let mut col = (0..base_bits)
                .map(|j| F128::new(((type_index >> j) & 1) as u64, 0))
                .collect::<Vec<_>>();
            col.extend_from_slice(&[F128::ONE, F128::ZERO, F128::ONE]); // plane 5
            claims.push(MatrixClaim {
                row: Weight::eq(point.clone()),
                col: Weight::eq(col),
                value: *value,
            });
        }
        if let Some((point, a, b)) = &self.element_constants {
            let base_bits = self.matrix_base_bits();
            let mut base = point.clone();
            base.resize(base_bits, F128::ZERO);
            for (plane, value) in [(Self::ELEMENT_A_PLANE, *a), (Self::ELEMENT_B_PLANE, *b)] {
                let mut col = base.clone();
                col.extend((0..3).map(|j| F128::new(((plane >> j) & 1) as u64, 0)));
                claims.push(MatrixClaim {
                    // The affine vectors are row-independent, so a fixed
                    // zero row point gives a canonical claim identity.
                    row: Weight::eq(vec![F128::ZERO; self.nu]),
                    col: Weight::eq(col),
                    value,
                });
            }
        }
        if let Some(bus) = &self.bus {
            let base_bits = self.matrix_base_bits();
            for (c, &value) in bus.values.iter().enumerate() {
                let mut col: Vec<F128> = (0..base_bits)
                    .map(|j| F128::new(((c >> j) & 1) as u64, 0))
                    .collect();
                col.extend((0..3).map(|j| F128::new(((Self::BUS_PLANE >> j) & 1) as u64, 0)));
                claims.push(MatrixClaim {
                    row: Weight::eq(bus.rho_row.clone()),
                    col: Weight::eq(col),
                    value,
                });
            }
        }
        if let Some(pc) = &self.public_columns {
            let base_bits = self.matrix_base_bits();
            for (c, &value) in pc.values.iter().enumerate() {
                let mut col: Vec<F128> = (0..base_bits)
                    .map(|j| F128::new(((c >> j) & 1) as u64, 0))
                    .collect();
                col.extend(
                    (0..3).map(|j| F128::new(((Self::PUBLIC_COLUMN_PLANE >> j) & 1) as u64, 0)),
                );
                claims.push(MatrixClaim {
                    row: Weight::eq(pc.rho_row.clone()),
                    col: Weight::eq(col),
                    value,
                });
            }
        }
        claims
    }

    /// The words that determine every claim of [`Self::claims`] beyond the
    /// circuit's static facts (`nu`, `base_bits`, the pinned types, the
    /// column counts): `rho`, the pins' shared point, the element point, the
    /// bus's and the public columns' row points (each when present), then
    /// every claim's value in claims order. Shared binding absorbs these
    /// once instead of the claims ([`crate::aggregate`]).
    pub fn shared_surface(&self) -> Vec<F128> {
        let mut out = self.rho.clone();
        if let Some((_, point, _)) = self.boolean_pins.first() {
            assert!(
                self.boolean_pins.iter().all(|(_, p, _)| p == point),
                "the boolean pins share one point"
            );
            out.extend_from_slice(point);
        }
        if let Some((point, _, _)) = &self.element_constants {
            out.extend_from_slice(point);
        }
        if let Some(bus) = &self.bus {
            out.extend_from_slice(&bus.rho_row);
        }
        if let Some(pc) = &self.public_columns {
            out.extend_from_slice(&pc.rho_row);
        }
        out.extend(self.claims().iter().map(|c| c.value));
        out
    }

    fn matrix_base_bits(&self) -> usize {
        // Filled/shape-checked against the circuit in aggregation. The base
        // cell-space width is always at least the active auxiliary domains.
        self.base_bits
    }

    /// The digest-keyed static table used by the accumulator. Its selected
    /// planes contain `live*s_id`, `live`, `live*s_sigma`, the element affine
    /// constants, and Boolean constant-pin prefixes; the remaining planes are
    /// zero. One table therefore binds every circuit-static verifier value
    /// without putting child-specific data into the recursive gate registry.
    pub fn matrix(circuit: &Circuit) -> CircuitStructureMatrix<'_> {
        CircuitStructureMatrix::new(circuit)
    }

    /// The root discharge: the claimed evaluation against the real table —
    /// `O(2^mu)`, paid once at the root, never per node.
    pub fn check(&self, circuit: &Circuit) -> bool {
        let m = Self::matrix(circuit);
        self.claims()
            .iter()
            .all(|c| bilinear(&c.row, &c.col, &m) == c.value)
    }
}

/// A lazy eight-plane table of circuit-static verifier data.
///
/// Columns are `(cell_slot, plane)` and rows are the cell-space row. The
/// table deliberately implements only matrix marginals: materializing every
/// plane would multiply the already-large sigma root table's memory, while a
/// fold/root discharge needs just these two streaming passes.
/// Rows per parallel task in [`CircuitStructureMatrix`]'s SIGMA-plane row
/// marginal.
const SIGMA_ROW_CHUNK: usize = 1 << 10;

pub struct CircuitStructureMatrix<'a> {
    circuit: &'a Circuit,
    mask: LiveMask,
    base_cols: usize,
    bus_cols: Vec<BusColumn>,
    /// Per public column, the public word of each live row.
    public_cols: Vec<&'a [u32]>,
}

impl<'a> CircuitStructureMatrix<'a> {
    fn new(circuit: &'a Circuit) -> Self {
        let cell_cols = 1usize << (circuit.cells().mu() - circuit.cells().nu());
        let element_cols = if circuit.registry().num_element() == 0 {
            1
        } else {
            1usize << (circuit.registry().m_elem() - circuit.registry().nu() - 7)
        };
        let bus_cols = circuit
            .bus()
            .map_or_else(Vec::new, |bus| bus.structure_columns());
        let base_cols = cell_cols
            .max(element_cols)
            .max(circuit.registry().num_boolean().next_power_of_two())
            .max(bus_cols.len().next_power_of_two());
        let public_cols: Vec<&[u32]> = circuit.public_columns().map(|(_, w)| w).collect();
        let base_cols = base_cols.max(public_cols.len().next_power_of_two());
        Self {
            circuit,
            mask: circuit.live_mask(),
            base_cols,
            bus_cols,
            public_cols,
        }
    }

    #[inline]
    fn base_cols(&self) -> usize {
        self.base_cols
    }

    #[inline]
    fn entry(&self, row: usize, col: usize) -> F128 {
        let base_cols = self.base_cols();
        let plane = col / base_cols;
        let slot = col % base_cols;
        match plane {
            SigmaAssertion::MASKED_ID_PLANE
            | SigmaAssertion::LIVE_PLANE
            | SigmaAssertion::SIGMA_PLANE => {
                let cell_cols = 1usize << (self.circuit.cells().mu() - self.circuit.cells().nu());
                if slot >= cell_cols {
                    return F128::ZERO;
                }
                let x = (slot << self.circuit.cells().nu()) | row;
                if !self.mask.is_live(x) {
                    return F128::ZERO;
                }
                match plane {
                    SigmaAssertion::MASKED_ID_PLANE => F128::new(x as u64, 0),
                    SigmaAssertion::LIVE_PLANE => F128::ONE,
                    _ => F128::new(self.circuit.sigma()[x] as u64, 0),
                }
            }
            SigmaAssertion::ELEMENT_A_PLANE | SigmaAssertion::ELEMENT_B_PLANE => {
                let registry = self.circuit.registry();
                let nb = registry.num_boolean();
                let base_word = registry.element_base() >> 7;
                for (ty, layout) in registry.types()[nb..].iter().zip(&registry.slots()[nb..]) {
                    let et = ty.element_type().expect("element registry suffix");
                    let kappa = ty.k_log - 7;
                    let offset = ((layout.offset >> 7) - base_word) >> registry.nu();
                    if (offset..offset + (1usize << kappa)).contains(&slot) {
                        let y = slot - offset;
                        return if plane == SigmaAssertion::ELEMENT_A_PLANE {
                            et.a_const()[y]
                        } else {
                            et.b_const()[y]
                        };
                    }
                }
                F128::ZERO
            }
            SigmaAssertion::BOOLEAN_PIN_PLANE => {
                let registry = self.circuit.registry();
                if slot >= registry.num_boolean() {
                    return F128::ZERO;
                }
                let ty = &registry.boolean_types()[slot];
                if ty.const_pin.is_some() && row < self.circuit.counts()[slot] {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            }
            SigmaAssertion::BUS_PLANE => self
                .bus_cols
                .get(slot)
                .map_or(F128::ZERO, |c| c.entry(self.circuit, row)),
            SigmaAssertion::PUBLIC_COLUMN_PLANE => self
                .public_cols
                .get(slot)
                .and_then(|words| words.get(row))
                .map_or(F128::ZERO, |&p| {
                    self.circuit
                        .fixed_public(p as usize)
                        .expect("a deferred public column reads fixed words")
                }),
            _ => F128::ZERO,
        }
    }
}

impl FoldMatrix for CircuitStructureMatrix<'_> {
    /// `out[r] = Σ_c w[c] · entry(r, c)`. Only the SIGMA plane needs a
    /// per-cell walk. Live rows are a prefix of each slot, so every other
    /// plane's row-r term is a sum over the slots still live at `r`:
    ///
    /// * LIVE and the boolean pin plane add `w[c]` while `r < count`;
    /// * MASKED_ID adds `w[c]·enc(slot<<ν | r) = w[c]·enc(slot<<ν) +
    ///   w[c]·enc(r)` (`enc` places bits, so it is GF(2)-linear and the two
    ///   fields are disjoint);
    /// * the element planes do not depend on the row.
    ///
    /// Each slot's terms land in a bucket at its count, and a suffix sum
    /// over the buckets gives every row's total. Addition in GF(2^128) is
    /// exact, so this equals the plain double loop over [`Self::entry`].
    fn row_marginal(&self, w: &[F128], n_rows: usize) -> Vec<F128> {
        assert_eq!(n_rows, self.n_rows());
        assert_eq!(w.len(), self.n_cols());
        let nu = self.circuit.cells().nu();
        let base_cols = self.base_cols();
        let cell_cols = 1usize << (self.circuit.cells().mu() - nu);
        let col = |plane: usize, slot: usize| w[plane * base_cols + slot];
        let element_const = [
            SigmaAssertion::ELEMENT_A_PLANE,
            SigmaAssertion::ELEMENT_B_PLANE,
        ]
        .into_iter()
        .flat_map(|plane| plane * base_cols..(plane + 1) * base_cols)
        .filter(|&c| !w[c].is_zero())
        .fold(F128::ZERO, |acc, c| acc + w[c] * self.entry(0, c));
        // Bucket k holds the terms valid for rows r < k.
        let mut flat = vec![F128::ZERO; n_rows + 1];
        let mut by_r = vec![F128::ZERO; n_rows + 1];
        let mut sigma_slots: Vec<(usize, F128)> = Vec::new();
        for slot in 0..cell_cols.min(base_cols) {
            let cnt = self.mask.counts[slot];
            let w_id = col(SigmaAssertion::MASKED_ID_PLANE, slot);
            flat[cnt] +=
                w_id * F128::new((slot << nu) as u64, 0) + col(SigmaAssertion::LIVE_PLANE, slot);
            by_r[cnt] += w_id;
            let w_sig = col(SigmaAssertion::SIGMA_PLANE, slot);
            if !w_sig.is_zero() {
                sigma_slots.push((slot, w_sig));
            }
        }
        let registry = self.circuit.registry();
        for slot in 0..registry.num_boolean().min(base_cols) {
            if registry.boolean_types()[slot].const_pin.is_some() {
                let cnt = self.circuit.counts()[slot].min(n_rows);
                flat[cnt] += col(SigmaAssertion::BOOLEAN_PIN_PLANE, slot);
            }
        }
        let mut out = vec![F128::ZERO; n_rows];
        let (mut acc_flat, mut acc_r) = (F128::ZERO, F128::ZERO);
        for r in (0..n_rows).rev() {
            acc_flat += flat[r + 1];
            acc_r += by_r[r + 1];
            out[r] = element_const + acc_flat + acc_r * F128::new(r as u64, 0);
        }
        // The SIGMA plane, row chunks in parallel: inside a chunk each
        // slot's σ entries are read contiguously.
        let sigma = self.circuit.sigma();
        out.par_chunks_mut(SIGMA_ROW_CHUNK)
            .enumerate()
            .for_each(|(k, dst)| {
                let r0 = k * SIGMA_ROW_CHUNK;
                for &(slot, w_sig) in &sigma_slots {
                    let end = self.mask.counts[slot].min(r0 + dst.len());
                    let base = slot << nu;
                    for r in r0..end {
                        dst[r - r0] += w_sig * F128::new(sigma[base + r] as u64, 0);
                    }
                }
            });
        out
    }

    /// `out[c] = Σ_r w[r] · entry(r, c)`, by plane. Live rows are a prefix
    /// of each slot, so with prefix sums `P[k] = Σ_{r<k} w[r]` and
    /// `Q[k] = Σ_{r<k} w[r]·enc(r)`:
    ///
    /// * LIVE is `P[count]`, and the boolean pin plane `P[rows]`;
    /// * MASKED_ID is `enc(slot<<ν)·P[count] + Q[count]` (`enc` is
    ///   GF(2)-linear and the two bit fields are disjoint);
    /// * the element planes are `P[2^ν]` times one entry;
    /// * SIGMA walks its live cells, the only plane that must.
    ///
    /// Columns run in parallel. Addition in GF(2^128) is exact, so this
    /// equals the plain double loop over [`Self::entry`].
    fn col_marginal(&self, w: &[F128], n_cols: usize) -> Vec<F128> {
        assert_eq!(w.len(), self.n_rows());
        assert_eq!(n_cols, self.n_cols());
        let nu = self.circuit.cells().nu();
        let base_cols = self.base_cols();
        let cell_cols = 1usize << (self.circuit.cells().mu() - nu);
        let mut p = Vec::with_capacity(w.len() + 1);
        let mut q = Vec::with_capacity(w.len() + 1);
        let (mut sp, mut sq) = (F128::ZERO, F128::ZERO);
        p.push(sp);
        q.push(sq);
        for (r, &wr) in w.iter().enumerate() {
            sp += wr;
            sq += wr * F128::new(r as u64, 0);
            p.push(sp);
            q.push(sq);
        }
        let registry = self.circuit.registry();
        let sigma = self.circuit.sigma();
        (0..n_cols)
            .into_par_iter()
            .map(|c| {
                let slot = c % base_cols;
                match c / base_cols {
                    SigmaAssertion::MASKED_ID_PLANE
                    | SigmaAssertion::LIVE_PLANE
                    | SigmaAssertion::SIGMA_PLANE
                        if slot >= cell_cols =>
                    {
                        F128::ZERO
                    }
                    SigmaAssertion::MASKED_ID_PLANE => {
                        let cnt = self.mask.counts[slot];
                        F128::new((slot << nu) as u64, 0) * p[cnt] + q[cnt]
                    }
                    SigmaAssertion::LIVE_PLANE => p[self.mask.counts[slot]],
                    SigmaAssertion::SIGMA_PLANE => {
                        let base = slot << nu;
                        w[..self.mask.counts[slot]]
                            .iter()
                            .zip(&sigma[base..])
                            .fold(F128::ZERO, |acc, (&wr, &sx)| {
                                acc + wr * F128::new(sx as u64, 0)
                            })
                    }
                    SigmaAssertion::ELEMENT_A_PLANE | SigmaAssertion::ELEMENT_B_PLANE => {
                        p[w.len()] * self.entry(0, c)
                    }
                    SigmaAssertion::BOOLEAN_PIN_PLANE => {
                        if slot < registry.num_boolean()
                            && registry.boolean_types()[slot].const_pin.is_some()
                        {
                            p[self.circuit.counts()[slot].min(w.len())]
                        } else {
                            F128::ZERO
                        }
                    }
                    _ => F128::ZERO,
                }
            })
            .collect()
    }

    fn n_rows(&self) -> usize {
        1usize << self.circuit.cells().nu()
    }

    fn n_cols(&self) -> usize {
        8 * self.base_cols()
    }
}

/// [`verify_wiring`] with the sigma evaluation DEFERRED (sigma v2 route B):
/// the batched GKR verifies TRUSTING the proof's claimed `s_sigma_eval`,
/// and the claim exits as a [`SigmaAssertion`] for the accumulator — a
/// lying value either breaks the GKR's own input check or fails the root
/// discharge. Everything else — the recombination, the f/g binding, the
/// gather claims — is checked identically to v1.
pub fn verify_wiring_deferred<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    proof: &WiringProof,
    ch: &mut C,
) -> Result<(Vec<(Vec<F128>, F128)>, SigmaAssertion), WiringError> {
    verify_wiring_deferred_with_grinding(circuit, public, proof, BatchedGrinding::disabled(), ch)
}

/// [`verify_wiring_deferred`] with explicit Product-GKR grinding.
pub fn verify_wiring_deferred_with_grinding<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    proof: &WiringProof,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<(Vec<(Vec<F128>, F128)>, SigmaAssertion), WiringError> {
    verify_wiring_core(circuit, public, proof, true, grinding, ch)
}

pub fn verify_wiring<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    proof: &WiringProof,
    ch: &mut C,
) -> Result<Vec<(Vec<F128>, F128)>, WiringError> {
    verify_wiring_with_grinding(circuit, public, proof, BatchedGrinding::disabled(), ch)
}

/// [`verify_wiring`] with explicit Product-GKR grinding.
pub fn verify_wiring_with_grinding<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    proof: &WiringProof,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<Vec<(Vec<F128>, F128)>, WiringError> {
    verify_wiring_core(circuit, public, proof, false, grinding, ch).map(|(g, _)| g)
}

fn verify_wiring_core<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    proof: &WiringProof,
    defer_sigma: bool,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<(Vec<(Vec<F128>, F128)>, SigmaAssertion), WiringError> {
    if public.len() != circuit.num_public() {
        return Err(WiringError::MalformedProof);
    }
    let cells = circuit.cells();
    let (nu, mu) = (cells.nu(), cells.mu());
    let gathered = circuit.gathered_slots();
    if proof.gather.len() != gathered.len() {
        return Err(WiringError::MalformedProof);
    }
    // A deferred verify hands the public columns' values to the structure
    // table, which knows only FIXED public words. A circuit whose public
    // column reads a free public word cannot be deferred: a rejection, not a
    // panic, since a recursion layer takes the circuit from its key.
    if defer_sigma
        && !circuit.public_columns().all(|(_, words)| {
            words
                .iter()
                .all(|&p| circuit.fixed_public(p as usize).is_some())
        })
    {
        return Err(WiringError::UnfixedPublicColumn);
    }

    let mask = circuit.live_mask();
    let claim = if defer_sigma {
        verify_batched_with_grinding(mu, &proof.gkr, Some(&mask), grinding, ch)
    } else {
        verify_batched_with_sigma_and_grinding(
            mu,
            &proof.gkr,
            circuit.sigma(),
            Some(&mask),
            grinding,
            ch,
        )
    }
    .map_err(WiringError::Gkr)?;

    // ---- Recombination:
    // ŵ(ρ) = Σ_{gate ι} eq(ρ_ι, ι)·v_ι + Σ_{public ι} eq(ρ_ι, ι)·v̂_ι(ρ_row).
    // Dummy cell-slots contribute nothing (w = 0 there).
    let eq_row = build_eq(&claim.rho[..nu]);
    let eq_slot = build_eq(&claim.rho[nu..]);
    let mut acc = F128::ZERO;
    for (&iota, &v) in gathered.iter().zip(&proof.gather) {
        acc += eq_slot[circuit.gkr_slot(iota)] * v;
    }
    for (iota, slot) in cells.slots().iter().enumerate() {
        match *slot {
            // Gathered above; a pruned slot is dead in the mask.
            CellSlot::Gate { .. } => {}
            CellSlot::Public { s } => {
                acc +=
                    eq_slot[circuit.gkr_slot(iota)] * circuit.public_slot_eval(s, &eq_row, public)
            }
            CellSlot::Pad => {}
        }
    }
    if acc != claim.f_eval {
        return Err(WiringError::Recombination);
    }
    // BOTH surfaced evals must be bound: `f_eval` by the recombination above,
    // `g_eval` by this equality. Drop it and the RHS product's input vector is
    // prover-chosen.
    if claim.f_eval != claim.g_eval {
        return Err(WiringError::GEvalMismatch);
    }

    let assertion = SigmaAssertion {
        rho: claim.rho.clone(),
        nu,
        base_bits: CircuitStructureMatrix::new(circuit)
            .base_cols()
            .trailing_zeros() as usize,
        masked_id_value: mask.masked_id_eval(&s_id_basis(mu), &claim.rho),
        live_value: mask.live_eval(&claim.rho),
        value: claim.s_sigma_eval,
        boolean_pins: Vec::new(),
        element_constants: None,
        bus: None,
        public_columns: None,
    };
    let mut gather: Vec<(Vec<F128>, F128)> = gathered
        .iter()
        .zip(&proof.gather)
        .map(|(&iota, &v)| (cells.gate_claim_point(iota, &claim.rho[..nu]), v))
        .collect();
    // The public columns' claims, in the prover's order, valued from the
    // public words: the opening then checks each column against them.
    let public_claims: Vec<(Vec<F128>, F128)> = circuit
        .public_columns()
        .map(|(iota, words)| {
            let v = words
                .iter()
                .zip(&eq_row)
                .fold(F128::ZERO, |acc, (&p, &e)| acc + e * public[p as usize]);
            (cells.gate_claim_point(iota, &claim.rho[..nu]), v)
        })
        .collect();
    let public_columns = (!public_claims.is_empty()).then(|| BusStructure {
        rho_row: claim.rho[..nu].to_vec(),
        values: public_claims.iter().map(|&(_, v)| v).collect(),
    });
    gather.extend(public_claims);
    let mut assertion = assertion;
    assertion.public_columns = public_columns;
    match (circuit.bus(), &proof.bus) {
        (Some(bus), Some(bus_proof)) => {
            let (claims, structure) = bus::verify_bus(circuit, bus, bus_proof, grinding, ch)?;
            gather.extend(claims);
            assertion.bus = Some(structure);
        }
        (None, None) => {}
        _ => return Err(WiringError::MalformedProof),
    }
    Ok((gather, assertion))
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, sync::Arc};

    use crate::{
        challenger::FsChallenger,
        circuit::{
            BatchedGrinding, Cell, CellSlot, CellSpace, Circuit, CircuitError, F128, IoWord,
            ProductGkrError, Registry, SigmaAssertion, UnionInstance, WiringError, bilinear,
            build_eq, bus, prove_wiring, prove_wiring_with_grinding, verify_wiring,
            verify_wiring_deferred, verify_wiring_with_grinding,
        },
        element_r1cs::ElementTableBuilder,
        matrix_fold::FoldMatrix,
        r1cs::SparseBinaryMatrix,
        schedule::{TableClass, TableType},
        test_rng::Rng,
    };

    fn stub() -> SparseBinaryMatrix {
        SparseBinaryMatrix::new(0, 0, Vec::new())
    }

    /// A boolean type of the given width with the given schema — only its
    /// geometry matters here.
    fn bool_ty(k_log: usize, useful_bits: usize, schema: Vec<IoWord>) -> TableType {
        TableType {
            k_log,
            useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: None,
            class: TableClass::Boolean,
            io_schema: schema,
            zeroed_word_cols: Vec::new(),
        }
    }

    /// A `mult` element type: columns 0,1 free wires in, column 2 = 0·1 out.
    fn mult_ty(kappa: usize, schema: Vec<IoWord>) -> TableType {
        let mut b = ElementTableBuilder::new(kappa);
        b.free_wire(0).free_wire(1).mult(2, 0, 1);
        TableType::element(Arc::new(b.build().expect("mult block is valid"))).with_io_schema(schema)
    }

    /// A mixed registry with schemas on both classes: one wide boolean type,
    /// two element types of different widths (so the element slots carry real
    /// region prefixes).
    fn mixed_registry(nu: usize) -> Registry {
        Registry::new(
            vec![
                bool_ty(
                    12,
                    3000,
                    vec![IoWord::input(0), IoWord::input(5), IoWord::output(2)],
                ),
                mult_ty(
                    3,
                    vec![IoWord::input(0), IoWord::input(1), IoWord::output(2)],
                ),
                mult_ty(2, vec![IoWord::input(1), IoWord::output(2)]),
            ],
            nu,
        )
    }

    /// **THE ADDRESSING PIN.** For every gate cell-slot of a mixed registry and
    /// every row, the cell → committed-word map must agree with the union's own
    /// addressing: the word lies in that slot's word range, at the BatchMajor
    /// offset `(col << ν) + j`, and — for element slots — at the region offset
    /// the element PIOP uses. The claim point's coordinates, read as bits at a
    /// Boolean row point, must spell the same word index.
    #[test]
    fn cell_addressing_matches_the_union() {
        let nu = 4;
        let reg = mixed_registry(nu);
        let counts = vec![1usize << nu; reg.num_types()];
        let union = UnionInstance::new(&reg, counts);
        let cells = CellSpace::new(&reg, 0);
        let elem_base_word = reg.element_base() >> 7;
        let layouts = union.element_slot_layout();

        assert_eq!(cells.num_gate_slots(), 3 + 3 + 2);
        for (iota, slot) in cells.slots()[..cells.num_gate_slots()].iter().enumerate() {
            let CellSlot::Gate { ty, word } = *slot else {
                unreachable!()
            };
            let range = union.slot_word_range(ty);
            for row in 0..1usize << nu {
                let addr = cells.gate_word_addr(iota, row);
                // The union's own BatchMajor word index.
                assert_eq!(addr, range.start + (word.word_col << nu) + row);
                assert!(range.contains(&addr));
                // Element slots: the same word, addressed through the element
                // region the way the element PIOP addresses it.
                if reg.types()[ty].is_element() {
                    let l = layouts[ty - reg.num_boolean()];
                    assert_eq!(
                        addr,
                        elem_base_word + l.region_word_offset + (word.word_col << nu) + row
                    );
                }
                // The claim point IS the word address, bit for bit.
                let rho_row: Vec<F128> = (0..nu)
                    .map(|i| {
                        if (row >> i) & 1 == 1 {
                            F128::ONE
                        } else {
                            F128::ZERO
                        }
                    })
                    .collect();
                let point = cells.gate_claim_point(iota, &rho_row);
                assert_eq!(point.len(), union.m_total() - 7);
                let mut spelled = 0usize;
                for (i, &co) in point.iter().enumerate() {
                    assert!(co == F128::ZERO || co == F128::ONE, "Boolean point");
                    if co == F128::ONE {
                        spelled |= 1 << i;
                    }
                }
                assert_eq!(spelled, addr, "claim point must spell the word address");
            }
        }
    }

    /// The cell space's shape: gate slots in registry × schema order, then the
    /// public slots, then padding to `2^c`.
    #[test]
    fn cell_space_enumeration_order() {
        let nu = 3;
        let reg = mixed_registry(nu);
        // 17 public words at ν = 3 → 3 public slots (8 + 8 + 1).
        let cells = CellSpace::new(&reg, 17);
        assert_eq!(cells.num_gate_slots(), 8);
        assert_eq!(cells.num_public_slots(), 3);
        assert_eq!(cells.c_bits(), 4); // 11 slots → 16
        assert_eq!(cells.mu(), nu + 4);
        assert!(matches!(cells.slots()[0], CellSlot::Gate { ty: 0, .. }));
        assert!(matches!(cells.slots()[8], CellSlot::Public { s: 0 }));
        assert!(matches!(cells.slots()[10], CellSlot::Public { s: 2 }));
        assert!(cells.slots()[11..].iter().all(|s| *s == CellSlot::Pad));
    }

    fn small_registry(nu: usize) -> Registry {
        Registry::new(
            vec![mult_ty(
                2,
                vec![IoWord::input(0), IoWord::input(1), IoWord::output(2)],
            )],
            nu,
        )
    }

    /// Validation rejects each malformed wiring, and accepts the good one.
    #[test]
    fn circuit_validation() {
        let nu = 3;
        let reg = small_registry(nu);
        let n = 4usize; // 4 live gate rows
        let mk = |wires: Vec<Vec<Cell>>| Circuit::new(&reg, vec![n], 5, wires).err();

        // Gate cell-slots: 0 = a(in), 1 = b(in), 2 = c(out); slot 3 = public.
        assert_eq!(mk(vec![vec![Cell::new(2, 0), Cell::new(0, 1)]]), None);
        assert_eq!(
            mk(vec![vec![Cell::new(9, 0), Cell::new(0, 1)]]),
            Some(CircuitError::UnknownCell(Cell::new(9, 0)))
        );
        assert_eq!(
            mk(vec![vec![Cell::new(2, 0), Cell::new(0, n)]]),
            Some(CircuitError::DummyCell(Cell::new(0, n))),
            "a gate row past the count is a dummy cell"
        );
        assert_eq!(
            mk(vec![vec![Cell::new(2, 0), Cell::new(3, 5)]]),
            Some(CircuitError::DummyCell(Cell::new(3, 5))),
            "an unused public row is a dummy cell"
        );
        assert_eq!(
            mk(vec![vec![Cell::new(2, 0), Cell::new(2, 0)]]),
            Some(CircuitError::RepeatedCell(Cell::new(2, 0)))
        );
        assert_eq!(
            mk(vec![
                vec![Cell::new(2, 0), Cell::new(0, 1)],
                vec![Cell::new(2, 0), Cell::new(1, 1)],
            ]),
            Some(CircuitError::RepeatedCell(Cell::new(2, 0))),
            "classes must be disjoint"
        );
        assert_eq!(mk(vec![Vec::new()]), Some(CircuitError::EmptyClass));
        // Two producers in one class: ALLOWED — the class asserts the two
        // outputs equal (the connect-as-assert_eq pattern); witgen checks
        // the values agree.
        assert_eq!(mk(vec![vec![Cell::new(2, 0), Cell::new(2, 1)]]), None);
        // Self-loop: gate 0's output feeds gate 0's own input.
        assert_eq!(
            mk(vec![vec![Cell::new(2, 0), Cell::new(0, 0)]]),
            Some(CircuitError::Cyclic)
        );
        // Two-gate cycle.
        assert_eq!(
            mk(vec![
                vec![Cell::new(2, 0), Cell::new(0, 1)],
                vec![Cell::new(2, 1), Cell::new(0, 0)],
            ]),
            Some(CircuitError::Cyclic)
        );
        // A public cell is NOT a producer: a gate output wired to a public
        // output cell is the root case of the driving workload.
        assert_eq!(mk(vec![vec![Cell::new(2, 0), Cell::new(3, 0)]]), None);
        // A singleton class is a σ-fixed no-op, not an error.
        let single = Circuit::new(&reg, vec![n], 5, vec![vec![Cell::new(2, 0)]]).unwrap();
        assert_eq!(single.wires().len(), 1);
        assert_eq!(
            single.sigma(),
            (0..single.sigma().len()).collect::<Vec<_>>()
        );
    }

    /// σ's orbits ARE the wire classes, and σ is a permutation that fixes every
    /// unwired cell.
    #[test]
    fn sigma_orbits_are_the_wire_classes() {
        let nu = 3;
        let reg = small_registry(nu);
        let circuit = Circuit::new(
            &reg,
            vec![4],
            5,
            vec![
                vec![Cell::new(2, 0), Cell::new(0, 1), Cell::new(1, 2)],
                vec![Cell::new(3, 0), Cell::new(0, 0), Cell::new(0, 2)],
            ],
        )
        .expect("valid");
        let sigma = circuit.sigma();
        let cells = circuit.cells();
        assert_eq!(sigma.len(), 1 << cells.mu());
        let wired: BTreeSet<usize> = circuit.wires().iter().flatten().copied().collect();
        for (x, &sx) in sigma.iter().enumerate() {
            if !wired.contains(&x) {
                assert_eq!(sx, x, "unwired cells are fixed points");
            }
        }
        // Walk each orbit: it must be exactly its class.
        for class in circuit.wires() {
            let mut orbit = vec![class[0]];
            let mut x = sigma[class[0]];
            while x != class[0] {
                orbit.push(x);
                x = sigma[x];
            }
            orbit.sort_unstable();
            assert_eq!(&orbit, class);
        }
    }

    /// The wiring is canonical: the same PARTITION given in any order is the
    /// same circuit — same digest, same σ. And every component of the circuit
    /// moves the digest.
    #[test]
    fn digest_is_canonical_and_binding() {
        let nu = 3;
        let reg = small_registry(nu);
        let a = Circuit::new(
            &reg,
            vec![4],
            5,
            vec![
                vec![Cell::new(2, 0), Cell::new(0, 1)],
                vec![Cell::new(2, 1), Cell::new(0, 2)],
            ],
        )
        .unwrap();
        let b = Circuit::new(
            &reg,
            vec![4],
            5,
            vec![
                vec![Cell::new(0, 2), Cell::new(2, 1)],
                vec![Cell::new(0, 1), Cell::new(2, 0)],
            ],
        )
        .unwrap();
        assert_eq!(a.digest(), b.digest(), "class/cell order must not matter");
        assert_eq!(a.sigma(), b.sigma());

        let cases: Vec<(&str, Circuit)> = vec![
            (
                "counts",
                Circuit::new(
                    &reg,
                    vec![3],
                    5,
                    vec![
                        vec![Cell::new(2, 0), Cell::new(0, 1)],
                        vec![Cell::new(2, 1), Cell::new(0, 2)],
                    ],
                )
                .unwrap(),
            ),
            (
                "public count",
                Circuit::new(
                    &reg,
                    vec![4],
                    4,
                    vec![
                        vec![Cell::new(2, 0), Cell::new(0, 1)],
                        vec![Cell::new(2, 1), Cell::new(0, 2)],
                    ],
                )
                .unwrap(),
            ),
            (
                "one wire moved",
                Circuit::new(
                    &reg,
                    vec![4],
                    5,
                    vec![
                        vec![Cell::new(2, 0), Cell::new(1, 1)],
                        vec![Cell::new(2, 1), Cell::new(0, 2)],
                    ],
                )
                .unwrap(),
            ),
            (
                "one class dropped",
                Circuit::new(
                    &reg,
                    vec![4],
                    5,
                    vec![vec![Cell::new(2, 0), Cell::new(0, 1)]],
                )
                .unwrap(),
            ),
        ];
        for (what, c) in cases {
            assert_ne!(a.digest(), c.digest(), "digest insensitive to {what}");
        }
        // The registry binds too: same shape, different schema order.
        let other_reg = Registry::new(
            vec![mult_ty(
                2,
                vec![IoWord::input(1), IoWord::input(0), IoWord::output(2)],
            )],
            nu,
        );
        let d = Circuit::new(
            &other_reg,
            vec![4],
            5,
            vec![
                vec![Cell::new(2, 0), Cell::new(0, 1)],
                vec![Cell::new(2, 1), Cell::new(0, 2)],
            ],
        )
        .unwrap();
        assert_ne!(a.digest(), d.digest(), "digest insensitive to the registry");
    }

    // ---- the wiring argument, against a synthetic "committed" buffer -------

    /// A padded union buffer for `reg` whose gate cells satisfy `wires`, built
    /// by writing a random value per wire class. Dummy rows stay zero.
    fn buffer_for(reg: &Registry, circuit: &Circuit, rng: &mut Rng) -> (Vec<F128>, Vec<F128>) {
        let cells = circuit.cells();
        let nu = cells.nu();
        let mut packed = vec![F128::ZERO; 1usize << (reg.m_total() - 7)];
        let mut public = vec![F128::ZERO; circuit.num_public()];
        // Fill every live gate cell with a random value first…
        for iota in 0..cells.num_gate_slots() {
            let CellSlot::Gate { ty, .. } = cells.slots()[iota] else {
                unreachable!()
            };
            for row in 0..circuit.counts()[ty] {
                packed[cells.gate_word_addr(iota, row)] = rng.f128();
            }
        }
        for p in public.iter_mut() {
            *p = rng.f128();
        }
        // …then force each class to one value.
        for class in circuit.wires() {
            let v = rng.f128();
            for &idx in class {
                let (iota, row) = (idx >> nu, idx & ((1 << nu) - 1));
                match cells.slots()[iota] {
                    CellSlot::Gate { .. } => packed[cells.gate_word_addr(iota, row)] = v,
                    CellSlot::Public { s } => public[(s << nu) + row] = v,
                    CellSlot::Pad => unreachable!("validated"),
                }
            }
        }
        (packed, public)
    }

    fn roundtrip(circuit: &Circuit, packed: &[F128], public: &[F128]) -> Result<(), WiringError> {
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, claims) = prove_wiring(circuit, packed, public, &mut ch);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let out = verify_wiring(circuit, public, &proof, &mut ch)?;
        assert_eq!(out.len(), claims.len());
        for (c, (point, value)) in claims.iter().zip(&out) {
            assert_eq!(&c.point, point);
            assert_eq!(c.value, *value);
        }
        Ok(())
    }

    /// The wiring argument accepts a σ-invariant `w` and rejects a witness with
    /// one wire equality broken — over a mixed registry, so both classes' cells
    /// take part.
    #[test]
    fn wiring_roundtrip_and_broken_equality() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let counts = vec![5usize, 4, 6];
        let circuit = Circuit::new(
            &reg,
            counts,
            9,
            vec![
                // boolean out → boolean in (a chain), and a fan-out onto both
                // element types plus a public cell.
                vec![Cell::new(2, 0), Cell::new(0, 1), Cell::new(3, 2)],
                vec![
                    Cell::new(2, 1),
                    Cell::new(4, 3),
                    Cell::new(6, 0),
                    Cell::new(8, 4),
                ],
                vec![Cell::new(5, 1), Cell::new(1, 2)],
            ],
        )
        .expect("valid circuit");
        let mut rng = Rng::new(0xC1C0_0001);
        let (packed, public) = buffer_for(&reg, &circuit, &mut rng);
        roundtrip(&circuit, &packed, &public).expect("honest wiring verifies");

        // Break one wire equality: the products no longer match.
        let cells = circuit.cells();
        let mut bad = packed.clone();
        bad[cells.gate_word_addr(0, 1)] += F128::ONE;
        assert_eq!(
            roundtrip(&circuit, &bad, &public),
            Err(WiringError::Gkr(ProductGkrError::ProductMismatch))
        );
        // Break a WIRED public word instead (public slot 0, row 4 — the cell
        // `Cell::new(8, 4)` above). An unwired public word is unconstrained by
        // the wiring, so it must be a wired one.
        let mut bad_public = public.clone();
        bad_public[4] += F128::ONE;
        assert_eq!(
            roundtrip(&circuit, &packed, &bad_public),
            Err(WiringError::Gkr(ProductGkrError::ProductMismatch))
        );
    }

    /// "What the caller owes", item 5: a committed dead io word of `1`
    /// makes its masked leaf `w + 1 = 0` on both sides. A forger who runs
    /// the GKR over the leaves the verifier reconstructs and sends the
    /// gather values the opening checks (whole committed columns, dead rows
    /// included) passes every check but the products' — here with a copy
    /// constraint broken. The verifier rejects the zero product. The same
    /// forger with a dead word other than `1` and the wiring intact
    /// verifies: such a word scales both products alike.
    #[test]
    fn a_dead_io_word_of_one_is_rejected() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let circuit = Circuit::new(
            &reg,
            vec![5usize, 4, 6],
            9,
            vec![
                vec![Cell::new(2, 0), Cell::new(0, 1), Cell::new(3, 2)],
                vec![
                    Cell::new(2, 1),
                    Cell::new(4, 3),
                    Cell::new(6, 0),
                    Cell::new(8, 4),
                ],
                vec![Cell::new(5, 1), Cell::new(1, 2)],
            ],
        )
        .expect("valid circuit");
        let mut rng = Rng::new(0xC1C0_0DEA);
        let (packed, public) = buffer_for(&reg, &circuit, &mut rng);
        roundtrip(&circuit, &packed, &public).expect("honest wiring verifies");
        let cells = circuit.cells();
        let (mu, rows) = (cells.mu(), 1usize << nu);
        let forge = |packed: &[F128]| {
            let mut w = vec![F128::ZERO; 1 << mu];
            for (iota, slot) in cells.slots().iter().enumerate() {
                for row in 0..rows {
                    w[(iota << nu) + row] = match *slot {
                        CellSlot::Gate { .. } => packed[cells.gate_word_addr(iota, row)],
                        CellSlot::Public { s } => {
                            public.get((s << nu) + row).copied().unwrap_or(F128::ZERO)
                        }
                        CellSlot::Pad => F128::ZERO,
                    };
                }
            }
            let mut ch = FsChallenger::new(b"circuit-unit");
            let (gkr, rho) = crate::product_gkr::forge_masked_batched_for_tests(
                &w,
                circuit.sigma(),
                &circuit.live_mask(),
                &mut ch,
            );
            let eq_row = build_eq(&rho[..nu]);
            let gather = circuit
                .gathered_slots()
                .into_iter()
                .map(|iota| {
                    (0..rows).fold(F128::ZERO, |acc, row| {
                        acc + eq_row[row] * packed[cells.gate_word_addr(iota, row)]
                    })
                })
                .collect();
            let proof = super::WiringProof {
                gkr,
                gather,
                bus: None,
            };
            let mut ch = FsChallenger::new(b"circuit-unit");
            verify_wiring(&circuit, &public, &proof, &mut ch).map(|_| ())
        };
        // Cell-slot 0 is a wired boolean input word with 5 live rows of 8.
        let dead = cells.gate_word_addr(0, 6);
        let mut junk = packed.clone();
        junk[dead] = F128::new(5, 0);
        forge(&junk).expect("a dead word other than 1 verifies");

        let mut bad = packed.clone();
        bad[cells.gate_word_addr(0, 1)] += F128::ONE;
        assert_eq!(
            forge(&bad),
            Err(WiringError::Gkr(ProductGkrError::ProductMismatch))
        );
        bad[dead] = F128::ONE;
        assert_eq!(
            forge(&bad),
            Err(WiringError::Gkr(ProductGkrError::ZeroProduct)),
            "a dead word of 1 zeroes both products"
        );
    }

    /// Sigma v2 route B at the wiring level: the deferred verify agrees
    /// with v1 gather-for-gather on an honest proof, its assertion
    /// discharges against the real table (and via the MatrixClaim path —
    /// the accumulator's form), a tampered assertion value fails the
    /// discharge, and a naively tampered claimed `s_sigma_eval` is caught
    /// by the GKR's own input check.
    #[test]
    fn wiring_sigma_deferral() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let counts = vec![5usize, 4, 6];
        let circuit = Circuit::new(
            &reg,
            counts,
            9,
            vec![
                vec![Cell::new(2, 0), Cell::new(0, 1), Cell::new(3, 2)],
                vec![
                    Cell::new(2, 1),
                    Cell::new(4, 3),
                    Cell::new(6, 0),
                    Cell::new(8, 4),
                ],
                vec![Cell::new(5, 1), Cell::new(1, 2)],
            ],
        )
        .expect("valid circuit");
        let mut rng = Rng::new(0x516A_B001);
        let (packed, public) = buffer_for(&reg, &circuit, &mut rng);
        let mut ch = FsChallenger::new(b"circuit-sigma-b");
        let (proof, _) = prove_wiring(&circuit, &packed, &public, &mut ch);

        let mut ch = FsChallenger::new(b"circuit-sigma-b");
        let v1 = verify_wiring(&circuit, &public, &proof, &mut ch).expect("v1 accepts");
        let mut ch = FsChallenger::new(b"circuit-sigma-b");
        let (gathers, assertion) =
            verify_wiring_deferred(&circuit, &public, &proof, &mut ch).expect("deferred accepts");
        assert_eq!(v1, gathers, "deferred and v1 agree on the gather claims");
        assert!(assertion.check(&circuit), "the sigma assertion discharges");
        let mc = assertion.claim();
        let m = SigmaAssertion::matrix(&circuit);
        assert_eq!(
            bilinear(&mc.row, &mc.col, &m),
            mc.value,
            "the MatrixClaim form discharges — the accumulator's path"
        );
        // The plane-wise marginals equal the plain double loop over `entry`.
        let (n_rows, n_cols) = (m.n_rows(), m.n_cols());
        let w_row: Vec<F128> = (0..n_rows).map(|_| rng.f128()).collect();
        let w_col: Vec<F128> = (0..n_cols).map(|_| rng.f128()).collect();
        let naive_col: Vec<F128> = (0..n_cols)
            .map(|c| (0..n_rows).fold(F128::ZERO, |a, r| a + w_row[r] * m.entry(r, c)))
            .collect();
        let naive_row: Vec<F128> = (0..n_rows)
            .map(|r| (0..n_cols).fold(F128::ZERO, |a, c| a + w_col[c] * m.entry(r, c)))
            .collect();
        assert_eq!(m.col_marginal(&w_row, n_cols), naive_col, "col marginal");
        assert_eq!(m.row_marginal(&w_col, n_rows), naive_row, "row marginal");
        // The same oracle with the boolean type PINNED, so the pin plane's
        // closed form is exercised too.
        {
            let mut types = reg.types().to_vec();
            types
                .iter_mut()
                .find(|t| !t.is_element())
                .expect("a boolean type")
                .const_pin = Some(0);
            let pinned = Registry::new(types, nu);
            let pc = Circuit::new(
                &pinned,
                circuit.counts().to_vec(),
                9,
                vec![
                    vec![Cell::new(2, 0), Cell::new(0, 1), Cell::new(3, 2)],
                    vec![
                        Cell::new(2, 1),
                        Cell::new(4, 3),
                        Cell::new(6, 0),
                        Cell::new(8, 4),
                    ],
                    vec![Cell::new(5, 1), Cell::new(1, 2)],
                ],
            )
            .expect("valid pinned circuit");
            let pm = SigmaAssertion::matrix(&pc);
            let pin_col = SigmaAssertion::BOOLEAN_PIN_PLANE * pm.base_cols();
            assert!(
                (0..n_rows).any(|r| !pm.entry(r, pin_col).is_zero()),
                "the pinned type puts entries on the pin plane"
            );
            let naive_col: Vec<F128> = (0..pm.n_cols())
                .map(|c| (0..n_rows).fold(F128::ZERO, |a, r| a + w_row[r] * pm.entry(r, c)))
                .collect();
            let naive_row: Vec<F128> = (0..n_rows)
                .map(|r| (0..pm.n_cols()).fold(F128::ZERO, |a, c| a + w_col[c] * pm.entry(r, c)))
                .collect();
            assert_eq!(
                pm.col_marginal(&w_row, pm.n_cols()),
                naive_col,
                "pinned col marginal"
            );
            assert_eq!(
                pm.row_marginal(&w_col, n_rows),
                naive_row,
                "pinned row marginal"
            );
        }

        let mut bad = assertion.clone();
        bad.value += F128::ONE;
        assert!(
            !bad.check(&circuit),
            "a tampered assertion fails the discharge"
        );

        let mut bad_proof = proof.clone();
        bad_proof.gkr.s_sigma_eval += F128::ONE;
        let mut ch = FsChallenger::new(b"circuit-sigma-b");
        assert!(
            verify_wiring_deferred(&circuit, &public, &bad_proof, &mut ch).is_err(),
            "a naively tampered s_sigma_eval breaks the GKR input check"
        );
    }

    /// The gather values are exactly the row-folds the recombination expects,
    /// and a tampered one is caught by the recombination check — which is the
    /// half of the binding that ties `f_eval` to the committed witness.
    #[test]
    fn gather_binds_the_f_side() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let circuit = Circuit::new(
            &reg,
            vec![5, 4, 6],
            9,
            vec![vec![Cell::new(2, 0), Cell::new(4, 1), Cell::new(8, 0)]],
        )
        .unwrap();
        let mut rng = Rng::new(0xC1C0_0002);
        let (packed, public) = buffer_for(&reg, &circuit, &mut rng);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&circuit, &packed, &public, &mut ch);
        for i in 0..proof.gather.len() {
            let mut bad = proof.clone();
            bad.gather[i] += F128::ONE;
            let mut ch = FsChallenger::new(b"circuit-unit");
            assert_eq!(
                verify_wiring(&circuit, &public, &bad, &mut ch),
                Err(WiringError::Recombination),
                "gather value {i}"
            );
        }
        // A wrong-length gather list is malformed, not a panic.
        let mut short = proof.clone();
        short.gather.pop();
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert_eq!(
            verify_wiring(&circuit, &public, &short, &mut ch),
            Err(WiringError::MalformedProof)
        );
    }

    /// A tampered GKR transcript is rejected, and so is a proof replayed
    /// against a DIFFERENT circuit (σ is verifier-known, so a different wiring
    /// changes `ŝ_σ(ρ)`).
    #[test]
    fn transcript_and_sigma_tamper() {
        let nu = 3;
        let reg = small_registry(nu);
        let wires = vec![
            vec![Cell::new(2, 0), Cell::new(0, 1)],
            vec![Cell::new(2, 1), Cell::new(1, 2)],
        ];
        let circuit = Circuit::new(&reg, vec![4], 4, wires).unwrap();
        let mut rng = Rng::new(0xC1C0_0003);
        let (packed, public) = buffer_for(&reg, &circuit, &mut rng);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&circuit, &packed, &public, &mut ch);

        let mut bad = proof.clone();
        bad.gkr.top_lhs += F128::ONE;
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert!(verify_wiring(&circuit, &public, &bad, &mut ch).is_err());
        let mut bad = proof.clone();
        bad.gkr.layers[1].vl0 += F128::ONE;
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert!(verify_wiring(&circuit, &public, &bad, &mut ch).is_err());

        // Same buffer, different wiring: σ moves, so the input check fails.
        let other = Circuit::new(
            &reg,
            vec![4],
            4,
            vec![
                vec![Cell::new(2, 0), Cell::new(1, 1)],
                vec![Cell::new(2, 1), Cell::new(1, 2)],
            ],
        )
        .unwrap();
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert!(verify_wiring(&other, &public, &proof, &mut ch).is_err());
    }

    // ---- the tuple bus ------------------------------------------------------

    /// A width-3 bus over the mixed registry: types 0 and 1 send
    /// `(value, key, 7)`, type 2 receives its eight rows' `(value, key, 7)`,
    /// the 7 spelled as a constant on one side and a reduced row tag plus a
    /// constant on the other.
    fn bus_circuit(reg: &Registry, recv_constant: F128) -> Circuit {
        use bus::{Bus, BusDirection, BusPort, BusTerm, RowTag};
        let seven = F128::new(7, 0);
        let zero_tag = RowTag {
            offset: 3,
            div: 1,
            modulo: 1,
        };
        let port = |ty: usize, dir, words: [usize; 2], last: BusTerm| BusPort {
            ty,
            dir,
            terms: vec![BusTerm::word(words[0]), BusTerm::word(words[1]), last],
        };
        let bus = Bus {
            width: 3,
            ports: vec![
                port(
                    0,
                    BusDirection::Send,
                    [0, 1],
                    BusTerm::row(RowTag {
                        offset: 0,
                        div: 100,
                        modulo: 0,
                    })
                    .plus(seven),
                ),
                port(
                    1,
                    BusDirection::Send,
                    [3, 4],
                    BusTerm::row(zero_tag).plus(seven),
                ),
                port(
                    2,
                    BusDirection::Recv,
                    [6, 7],
                    BusTerm::row(zero_tag).plus(recv_constant),
                ),
            ],
        };
        Circuit::new(&reg.clone(), vec![4, 4, 8], 0, Vec::new())
            .unwrap()
            .with_bus(bus)
            .expect("a valid bus")
    }

    /// The type-2 rows are the eight sent tuples in a shuffled order.
    fn bus_buffer(circuit: &Circuit, reg: &Registry, rng: &mut Rng) -> Vec<F128> {
        let cells = circuit.cells();
        let mut packed = vec![F128::ZERO; 1usize << (reg.m_total() - 7)];
        let mut sent = Vec::new();
        for (ty, words) in [(0usize, [0usize, 1]), (1, [3, 4])] {
            for row in 0..circuit.counts()[ty] {
                let t = [rng.f128(), rng.f128()];
                for (w, v) in words.iter().zip(t) {
                    packed[cells.gate_word_addr(*w, row)] = v;
                }
                sent.push(t);
            }
        }
        for (row, j) in [5usize, 2, 7, 0, 3, 6, 1, 4].into_iter().enumerate() {
            packed[cells.gate_word_addr(6, row)] = sent[j][0];
            packed[cells.gate_word_addr(7, row)] = sent[j][1];
        }
        packed
    }

    /// The bus accepts a balanced witness in any order and rejects a changed
    /// tuple, a different static coordinate, and tampered gather values —
    /// with and without grinding.
    #[test]
    fn bus_roundtrip_balance_and_tamper() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let circuit = bus_circuit(&reg, F128::new(7, 0));
        let mut rng = Rng::new(0xB05_0001);
        let packed = bus_buffer(&circuit, &reg, &mut rng);
        roundtrip(&circuit, &packed, &[]).expect("a balanced bus verifies");

        let cells = circuit.cells();
        let mut bad = packed.clone();
        bad[cells.gate_word_addr(7, 3)] += F128::ONE;
        assert_eq!(
            roundtrip(&circuit, &bad, &[]),
            Err(WiringError::BusGkr(ProductGkrError::ProductMismatch))
        );
        let other = bus_circuit(&reg, F128::new(6, 0));
        assert_ne!(other.digest(), circuit.digest(), "the bus binds the digest");
        assert_eq!(
            roundtrip(&other, &packed, &[]),
            Err(WiringError::BusGkr(ProductGkrError::ProductMismatch))
        );

        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&circuit, &packed, &[], &mut ch);
        let n = proof.bus.as_ref().unwrap().gather.len();
        assert_eq!(n, 6);
        for i in 0..n {
            let mut bad = proof.clone();
            bad.bus.as_mut().unwrap().gather[i] += F128::ONE;
            let mut ch = FsChallenger::new(b"circuit-unit");
            assert_eq!(
                verify_wiring(&circuit, &[], &bad, &mut ch),
                Err(WiringError::BusRecombination),
                "bus gather {i}"
            );
        }
        let mut missing = proof.clone();
        missing.bus = None;
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert_eq!(
            verify_wiring(&circuit, &[], &missing, &mut ch),
            Err(WiringError::MalformedProof)
        );

        let grinding = BatchedGrinding::per_challenge_128();
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, claims) = prove_wiring_with_grinding(&circuit, &packed, &[], grinding, &mut ch);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let out = verify_wiring_with_grinding(&circuit, &[], &proof, grinding, &mut ch)
            .expect("a ground bus verifies");
        assert_eq!(out.len(), claims.len());
    }

    /// The deferred verify hands the bus's static columns to the sigma
    /// assertion: they discharge against the circuit-structure table, and a
    /// wrong column value does not.
    #[test]
    fn bus_structure_defers_to_the_sigma_assertion() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let circuit = bus_circuit(&reg, F128::new(7, 0));
        let mut rng = Rng::new(0xB05_0003);
        let packed = bus_buffer(&circuit, &reg, &mut rng);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&circuit, &packed, &[], &mut ch);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (_, assertion) =
            verify_wiring_deferred(&circuit, &[], &proof, &mut ch).expect("deferred verify");
        let bus = assertion.bus.as_ref().expect("the bus columns travel");
        let cols = circuit.bus().unwrap().structure_columns();
        assert_eq!(bus.values.len(), cols.len());
        assert!(cols.iter().any(|c| matches!(c, bus::BusColumn::Tag { .. })));
        assert!(assertion.check(&circuit), "the bus columns discharge");
        for i in 0..cols.len() {
            let mut bad = assertion.clone();
            bad.bus.as_mut().unwrap().values[i] += F128::ONE;
            assert!(!bad.check(&circuit), "bus column {i}");
        }
    }

    /// Every bus gather claim is the MLE of its committed word column at the
    /// claim point.
    #[test]
    fn bus_gather_claims_are_column_evaluations() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let circuit = bus_circuit(&reg, F128::new(7, 0));
        let mut rng = Rng::new(0xB05_0002);
        let packed = bus_buffer(&circuit, &reg, &mut rng);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, claims) = prove_wiring(&circuit, &packed, &[], &mut ch);
        let wiring_claims = proof.gather.len();
        for c in &claims[wiring_claims..] {
            let eq = build_eq(&c.point);
            let mle = eq
                .iter()
                .zip(&packed)
                .fold(F128::ZERO, |acc, (&e, &w)| acc + e * w);
            assert_eq!(mle, c.value);
        }
    }

    /// Pruning drops the unwired cell-slots from the wiring argument and its
    /// gather; the equalities it keeps are still enforced, the digest moves,
    /// and a bus-only circuit (nothing wired) still proves.
    #[test]
    fn pruned_wiring_roundtrip() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let full = Circuit::new(
            &reg,
            vec![5, 4, 6],
            9,
            vec![
                vec![Cell::new(2, 0), Cell::new(0, 1), Cell::new(3, 2)],
                vec![Cell::new(5, 1), Cell::new(1, 2)],
            ],
        )
        .unwrap();
        let pruned = full.clone().with_unwired_pruned();
        assert_ne!(full.digest(), pruned.digest());
        assert_eq!(pruned.gathered_slots(), vec![0, 1, 2, 3, 5]);
        let mut rng = Rng::new(0xC1C0_0004);
        let (packed, public) = buffer_for(&reg, &pruned, &mut rng);
        roundtrip(&pruned, &packed, &public).expect("pruned wiring verifies");
        let mut bad = packed.clone();
        bad[pruned.cells().gate_word_addr(1, 2)] += F128::ONE;
        assert_eq!(
            roundtrip(&pruned, &bad, &public),
            Err(WiringError::Gkr(ProductGkrError::ProductMismatch))
        );
        // A full proof does not verify against the pruned circuit.
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&full, &packed, &public, &mut ch);
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert!(verify_wiring(&pruned, &public, &proof, &mut ch).is_err());

        let bus_only = bus_circuit(&reg, F128::new(7, 0)).with_unwired_pruned();
        assert!(bus_only.gathered_slots().is_empty());
        let packed = bus_buffer(&bus_only, &reg, &mut rng);
        roundtrip(&bus_only, &packed, &[]).expect("a bus-only circuit verifies");
    }

    /// A column whose live cells are all wired to public words leaves the
    /// grand product. Its claim is valued from the public words, so a wrong
    /// cell in it, or a wrong public word only it reads, moves the verifier's
    /// value off the committed column's (which the opening rejects), while
    /// the rest of each class stays tied to its public word.
    #[test]
    fn public_columns_leave_the_grand_product() {
        let nu = 3;
        let reg = mixed_registry(nu);
        // Cell-slot 0 is the boolean type's first input (5 live rows); 8 is
        // the public slot.
        let pub_cell = |p: usize| Cell::new(8, p);
        let full = Circuit::new(
            &reg,
            vec![5, 4, 6],
            9,
            vec![
                vec![pub_cell(0), Cell::new(0, 0), Cell::new(0, 1)],
                // Also read by a column that stays in the grand product.
                vec![pub_cell(1), Cell::new(0, 2), Cell::new(4, 3)],
                vec![pub_cell(2), Cell::new(0, 3)],
                vec![pub_cell(3), Cell::new(0, 4)],
                vec![Cell::new(2, 0), Cell::new(1, 1)],
            ],
        )
        .unwrap();
        let opened = full.clone().with_public_columns_opened();
        assert_ne!(full.digest(), opened.digest());
        let columns: Vec<(usize, Vec<u32>)> = opened
            .public_columns()
            .map(|(i, w)| (i, w.to_vec()))
            .collect();
        assert_eq!(columns, vec![(0, vec![0, 0, 1, 2, 3])]);
        assert_eq!(opened.gathered_slots(), (1..8).collect::<Vec<_>>());
        assert_eq!(opened.live_mask().counts[0], 0);

        let mut rng = Rng::new(0xC1C0_0005);
        let (packed, public) = buffer_for(&reg, &opened, &mut rng);
        roundtrip(&opened, &packed, &public).expect("honest wiring verifies");
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, claims) = prove_wiring(&opened, &packed, &public, &mut ch);
        assert_eq!(proof.gather.len(), 7);
        assert_eq!(claims.len(), 8);
        // The public column's claim is the committed column's MLE.
        let last = claims.last().unwrap();
        let mle = build_eq(&last.point)
            .iter()
            .zip(&packed)
            .fold(F128::ZERO, |acc, (&e, &w)| acc + e * w);
        assert_eq!(mle, last.value);

        // The public column's value as the verifier computes it, for a buffer
        // and public words of the same transcript (the grand product still
        // accepts both tampers below).
        let column_value = |packed: &[F128], public: &[F128]| {
            let mut ch = FsChallenger::new(b"circuit-unit");
            let (proof, claims) = prove_wiring(&opened, packed, public, &mut ch);
            let mut ch = FsChallenger::new(b"circuit-unit");
            let out = verify_wiring(&opened, public, &proof, &mut ch).expect("grand product");
            (claims.last().unwrap().value, out.last().unwrap().1)
        };
        let (prover, verifier) = column_value(&packed, &public);
        assert_eq!(prover, verifier);
        let mut bad = packed.clone();
        bad[opened.cells().gate_word_addr(0, 3)] += F128::ONE;
        let (prover, verifier) = column_value(&bad, &public);
        assert_ne!(prover, verifier, "a wrong public-column cell");
        let mut bad_public = public.clone();
        bad_public[2] += F128::ONE;
        let (prover, verifier) = column_value(&packed, &bad_public);
        assert_ne!(
            prover, verifier,
            "a wrong public word only the column reads"
        );

        // The other cell of public word 1's class is still wired to it.
        let mut bad = packed.clone();
        bad[opened.cells().gate_word_addr(4, 3)] += F128::ONE;
        assert_eq!(
            roundtrip(&opened, &bad, &public),
            Err(WiringError::Gkr(ProductGkrError::ProductMismatch))
        );
        let mut bad_public = public.clone();
        bad_public[1] += F128::ONE;
        assert_eq!(
            roundtrip(&opened, &packed, &bad_public),
            Err(WiringError::Gkr(ProductGkrError::ProductMismatch))
        );

        // A column with an unwired live cell, or one wired to a gate, stays.
        let partial = Circuit::new(
            &reg,
            vec![5, 4, 6],
            9,
            vec![
                vec![pub_cell(0), Cell::new(0, 0), Cell::new(0, 1)],
                vec![Cell::new(2, 0), Cell::new(1, 1)],
            ],
        )
        .unwrap()
        .with_public_columns_opened();
        assert_eq!(partial.public_columns().count(), 0);
        assert_eq!(partial.digest(), {
            Circuit::new(
                &reg,
                vec![5, 4, 6],
                9,
                vec![
                    vec![pub_cell(0), Cell::new(0, 0), Cell::new(0, 1)],
                    vec![Cell::new(2, 0), Cell::new(1, 1)],
                ],
            )
            .unwrap()
            .digest()
        });
    }

    /// Packing moves the grand product's cell-slots so live ones merge first:
    /// the live mask and σ move with them (σ stays a permutation that maps
    /// live cells to live cells), the wiring still verifies and still
    /// rejects a broken equality, the gather claims are the same columns'
    /// evaluations, and the digest moves.
    #[test]
    fn packed_slots_roundtrip() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let wires = vec![
            vec![Cell::new(2, 0), Cell::new(0, 1), Cell::new(8, 2)],
            vec![Cell::new(5, 1), Cell::new(7, 2)],
            vec![Cell::new(8, 0), Cell::new(6, 3)],
        ];
        let base = Circuit::new(&reg, vec![5, 4, 6], 9, wires)
            .unwrap()
            .with_unwired_pruned();
        let packed = base.clone().with_live_slots_packed();
        assert!(packed.live_slots_packed());
        assert_ne!(base.digest(), packed.digest());

        let (before, after) = (base.live_mask().counts, packed.live_mask().counts);
        let mut seen = vec![false; before.len()];
        for (iota, &count) in before.iter().enumerate() {
            let q = packed.gkr_slot(iota);
            assert!(!seen[q], "the slot order is a permutation");
            seen[q] = true;
            assert_eq!(after[q], count);
        }
        // The first merge pairs q with q + half: live with live, here.
        let half = after.len() / 2;
        let merged = |c: &[usize]| (0..half).map(|q| c[q].max(c[q + half])).sum::<usize>();
        assert!(merged(&after) < merged(&before));
        let mask = packed.live_mask();
        let sigma = packed.sigma();
        let mut hit = vec![false; sigma.len()];
        for (x, &s) in sigma.iter().enumerate() {
            assert!(!hit[s]);
            hit[s] = true;
            assert_eq!(mask.is_live(x), mask.is_live(s), "σ keeps liveness");
        }

        let mut rng = Rng::new(0xC1C0_0007);
        let (buf, public) = buffer_for(&reg, &packed, &mut rng);
        roundtrip(&packed, &buf, &public).expect("packed wiring verifies");
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (_, claims) = prove_wiring(&packed, &buf, &public, &mut ch);
        for c in &claims {
            let mle = build_eq(&c.point)
                .iter()
                .zip(&buf)
                .fold(F128::ZERO, |acc, (&e, &w)| acc + e * w);
            assert_eq!(mle, c.value);
        }
        for (cell, err) in [
            (Cell::new(0, 1), "a gate cell"),
            (Cell::new(7, 2), "another"),
        ] {
            let mut bad = buf.clone();
            bad[packed.cells().gate_word_addr(cell.slot, cell.row)] += F128::ONE;
            assert_eq!(
                roundtrip(&packed, &bad, &public),
                Err(WiringError::Gkr(ProductGkrError::ProductMismatch)),
                "{err}"
            );
        }
        let mut bad_public = public.clone();
        bad_public[0] += F128::ONE;
        assert_eq!(
            roundtrip(&packed, &buf, &bad_public),
            Err(WiringError::Gkr(ProductGkrError::ProductMismatch))
        );
        // An unpacked proof does not verify against the packed circuit.
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&base, &buf, &public, &mut ch);
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert!(verify_wiring(&packed, &public, &proof, &mut ch).is_err());
    }

    /// The deferred verify hands the public columns' values to the sigma
    /// assertion (plane 7, over the circuit's FIXED public words) and takes
    /// packed slots: the assertion discharges, and a moved public-column
    /// value does not.
    #[test]
    fn deferred_wiring_defers_public_columns() {
        let nu = 3;
        let reg = mixed_registry(nu);
        let wires = vec![vec![Cell::new(8, 0), Cell::new(0, 0)]];
        let plain = Circuit::new(&reg, vec![1, 4, 6], 1, wires.clone()).unwrap();
        let mut rng = Rng::new(0xC1C0_0006);
        let (packed, public) = buffer_for(&reg, &plain, &mut rng);
        let fixed = public.iter().map(|&p| Some(p)).collect();
        let circuit = Circuit::new_with_fixed_public(&reg, vec![1, 4, 6], fixed, wires)
            .unwrap()
            .with_public_columns_opened()
            .with_live_slots_packed();
        assert!(circuit.public_columns().next().is_some());
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&circuit, &packed, &public, &mut ch);
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (_, assertion) =
            verify_wiring_deferred(&circuit, &public, &proof, &mut ch).expect("deferred verify");
        assert!(assertion.public_columns.is_some(), "the columns travel");
        assert!(assertion.check(&circuit), "the deferred claims discharge");
        let mut bad = assertion.clone();
        bad.public_columns.as_mut().unwrap().values[0] += F128::ONE;
        assert!(!bad.check(&circuit), "a moved public-column value");

        // The same column over a FREE public word cannot be deferred: the
        // deferred verify rejects it (it used to panic), and the plain
        // verify still takes it.
        let free = plain.with_public_columns_opened().with_live_slots_packed();
        assert!(free.public_columns().next().is_some());
        let mut ch = FsChallenger::new(b"circuit-unit");
        let (proof, _) = prove_wiring(&free, &packed, &public, &mut ch);
        let mut ch = FsChallenger::new(b"circuit-unit");
        assert_eq!(
            verify_wiring_deferred(&free, &public, &proof, &mut ch).err(),
            Some(WiringError::UnfixedPublicColumn)
        );
        let mut ch = FsChallenger::new(b"circuit-unit");
        verify_wiring(&free, &public, &proof, &mut ch).expect("the plain verify");
    }
}
