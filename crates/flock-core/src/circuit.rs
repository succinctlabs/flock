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
//! fixes everything else. By tag rigidity (design doc, Lemma "Tag rigidity")
//! the multiset identity `{(w_x, x)} = {(w_x, σ(x))}` holds iff `w` is constant
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
//!    them to the commitment (and observes their values before `γ`).

pub mod builder;

use crate::matrix_fold::DenseMatrix;
use crate::matrix_fold::MatrixClaim;
use crate::matrix_fold::Weight;
use crate::matrix_fold::bilinear;
use crate::scratch::give_f128;
use crate::scratch::take_f128;
#[cfg(test)]
use std::collections::BTreeSet;
use std::env::var;
use std::sync::OnceLock;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::pcs::{DirectEqInd, PackedDirectClaim};
use crate::product_gkr;
use crate::schedule::{IoDirection, IoWord, Registry};
use crate::union::UnionInstance;
use crate::zerocheck::univariate_skip::build_eq;

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
    /// Two `out`-direction cells in one wire class — the single-producer rule.
    MultipleProducers {
        class: usize,
    },
    /// The gate dataflow (producer → consumer) has a cycle.
    Cyclic,
    /// σ is not a permutation. Unreachable given disjointness; checked anyway,
    /// because completeness (and the verifier's `s_σ`) rests on it.
    NotAPermutation,
}

/// A circuit over a registry: per-type gate counts, a public segment, and a
/// wiring.
///
/// Validated at construction (see [`CircuitError`]): every wired cell exists
/// and is live, classes are disjoint, at most one producer per class, the
/// dataflow is acyclic, and σ is a permutation. Semantics are satisfiability —
/// that a satisfying assignment computes "the" output is a property of the
/// circuit (single producer + acyclicity), checked by whoever endorses the
/// digest, exactly as in Plonk.
///
/// The wiring is stored CANONICALLY (cells ascending within a class, classes by
/// their least cell), so the digest is a function of the wire PARTITION and σ
/// is a function of the digest — prover and verifier cannot disagree about the
/// rotation.
#[derive(Clone, Debug)]
pub struct Circuit {
    cells: CellSpace,
    registry_digest: [u8; 32],
    counts: Vec<usize>,
    num_public: usize,
    wires: Vec<Vec<usize>>,
    sigma: Vec<usize>,
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

        // ---- Single producer per class.
        let mut producer: Vec<Option<usize>> = vec![None; classes.len()];
        for (ci, class) in classes.iter().enumerate() {
            for &idx in class {
                if let CellSlot::Gate { word, .. } = cells.slots()[idx >> nu]
                    && word.dir == IoDirection::Out
                {
                    if producer[ci].is_some() {
                        return Err(CircuitError::MultipleProducers { class: ci });
                    }
                    producer[ci] = Some(idx);
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
            let Some(prod_idx) = producer[ci] else {
                continue;
            };
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

        Ok(Self {
            cells,
            registry_digest: registry.digest(),
            counts,
            num_public,
            wires: classes,
            sigma,
            digest_cache: OnceLock::new(),
        })
    }

    pub fn cells(&self) -> &CellSpace {
        &self.cells
    }
    /// Per-registry-type gate counts, in slot order. These DETERMINE the
    /// union's declared counts — [`Self::check_instance`] enforces equality.
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }
    /// Number of public words.
    pub fn num_public(&self) -> usize {
        self.num_public
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

    /// The circuit digest — the statement's circuit half.
    ///
    /// Absorption order (format version 1): the domain label
    /// [`CIRCUIT_LABEL`], a version byte, the REGISTRY digest, `ν` and `c`,
    /// the schema-derived cell-slot enumeration (gate-slot count, then each
    /// gate slot's registry type, word-column and direction; then the public
    /// slot count), the gate counts, the public-word count, and the canonical
    /// wire encoding (class count, then per class its length and its cell
    /// indices). Every field is fixed-width or length-prefixed, so the
    /// encoding is injective.
    pub fn digest(&self) -> [u8; 32] {
        *self.digest_cache.get_or_init(|| {
            let mut h = blake3::Hasher::new();
            h.update(CIRCUIT_LABEL);
            h.update(&[1u8]);
            h.update(&self.registry_digest);
            h.update(&(self.cells.nu() as u32).to_le_bytes());
            h.update(&(self.cells.c_bits() as u32).to_le_bytes());
            h.update(&(self.cells.num_gate_slots() as u32).to_le_bytes());
            for slot in &self.cells.slots()[..self.cells.num_gate_slots()] {
                let CellSlot::Gate { ty, word } = slot else {
                    unreachable!("the gate prefix holds gate slots")
                };
                h.update(&(*ty as u32).to_le_bytes());
                h.update(&(word.word_col as u32).to_le_bytes());
                h.update(&[matches!(word.dir, IoDirection::Out) as u8]);
            }
            h.update(&(self.cells.num_public_slots() as u32).to_le_bytes());
            h.update(&(self.counts.len() as u32).to_le_bytes());
            for &n in &self.counts {
                h.update(&(n as u64).to_le_bytes());
            }
            h.update(&(self.num_public as u64).to_le_bytes());
            h.update(&(self.wires.len() as u32).to_le_bytes());
            for class in &self.wires {
                h.update(&(class.len() as u32).to_le_bytes());
                for &idx in class {
                    h.update(&(idx as u64).to_le_bytes());
                }
            }
            *h.finalize().as_bytes()
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
    fn public_slot_eval(&self, s: usize, eq_row: &[F128], public: &[F128]) -> F128 {
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
/// space, and the gather values (one per GATE cell-slot, in enumeration order).
/// The gather claims' POINTS are transcript-derived, so only the values ride.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WiringProof {
    pub gkr: product_gkr::ProductGkrBatchedProof,
    pub gather: Vec<F128>,
}

/// Why a wiring proof was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WiringError {
    /// Wrong number of gather values for this circuit's cell space.
    MalformedProof,
    /// The product-GKR rejected (products differ, layer check, input check).
    Gkr(product_gkr::VerifyError),
    /// The gather values do not recombine to `ŵ(ρ)` — the gather
    /// factorization's check, which is what binds the GKR's `f`-side input to
    /// the committed witness and the public words.
    Recombination,
    /// `f_eval ≠ g_eval`: the two grand products ran on different input
    /// vectors. Binding only the `f` side would leave `g` prover-chosen.
    GEvalMismatch,
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
    let mut w = take_f128(1usize << mu);
    for (iota, slot) in cells.slots().iter().enumerate() {
        let dst = &mut w[iota << nu..(iota + 1) << nu];
        match *slot {
            CellSlot::Gate { .. } => {
                let base = cells.gate_word_addr(iota, 0);
                dst.copy_from_slice(&packed[base..base + rows]);
            }
            CellSlot::Public { s } => {
                let base = s << nu;
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = public.get(base + j).copied().unwrap_or(F128::ZERO);
                }
            }
            CellSlot::Pad => dst.fill(F128::ZERO),
        }
    }

    if trace {
        eprintln!(
            "  [wiring] build w (2^{mu} cells): {:7.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- One grand-product permutation check at f = g = w.
    let t = Instant::now();
    let (gkr, claim) = product_gkr::prove_batched(&w, &w, circuit.sigma(), ch);
    give_f128(w);
    if trace {
        eprintln!(
            "  [wiring] product GKR (μ = {mu}): {:7.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- The gather: one eq-weighted row fold per gate cell-slot, O(2^ν)
    // each, landing on the packed-direct claim shape (design doc, Lemma
    // "Gather factorization").
    let t = Instant::now();
    let eq_row = build_eq(&claim.rho[..nu]);
    let mut gather = Vec::with_capacity(cells.num_gate_slots());
    let mut claims = Vec::with_capacity(cells.num_gate_slots());
    for iota in 0..cells.num_gate_slots() {
        let base = cells.gate_word_addr(iota, 0);
        let mut v = F128::ZERO;
        for (j, &e) in eq_row.iter().enumerate() {
            v += e * packed[base + j];
        }
        let point = cells.gate_claim_point(iota, &claim.rho[..nu]);
        gather.push(v);
        claims.push(PackedDirectClaim {
            value: v,
            // DEFERRED: the merged open (the only transport) never reads
            // `eq_ind` — it derives its identity-fold weights from
            // `point`/`value` alone — so the `2^nu`-entry tensor per gate
            // slot (~32 MiB at MVP-6's ~120 slots) is never built. A claim
            // that DID need a materialized tensor would trip the combine's
            // "EqPoint claims are only supported alone" assert rather than
            // silently dropping the contribution.
            eq_ind: DirectEqInd::EqPoint(point.clone()),
            point,
        });
    }
    if trace {
        eprintln!(
            "  [wiring] gather ({} claims × 2^{nu}): {:7.2} ms",
            claims.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (WiringProof { gkr, gather }, claims)
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
    pub value: F128,
}

impl SigmaAssertion {
    /// The accumulator's form: a `MatrixClaim` on the sigma table reshaped
    /// `2^nu × 2^c` (`M[r, c] = s_sig[(c << nu) + r]` — the cell space's
    /// `(col << nu) | row` convention, so the point splits row-low).
    pub fn claim(&self) -> MatrixClaim {
        MatrixClaim {
            row: Weight::eq(self.rho[..self.nu].to_vec()),
            col: Weight::eq(self.rho[self.nu..].to_vec()),
            value: self.value,
        }
    }

    /// The sigma table in the accumulator's matrix shape, from the SAME
    /// encoding the verifier evaluates ([`product_gkr::build_s_sigma_vec`]).
    pub fn matrix(circuit: &Circuit) -> DenseMatrix {
        let cells = circuit.cells();
        DenseMatrix {
            vals: product_gkr::build_s_sigma_vec(cells.mu(), circuit.sigma()),
            n_rows_log: cells.nu(),
        }
    }

    /// The root discharge: the claimed evaluation against the real table —
    /// `O(2^mu)`, paid once at the root, never per node.
    pub fn check(&self, circuit: &Circuit) -> bool {
        let c = self.claim();
        bilinear(&c.row, &c.col, &Self::matrix(circuit)) == self.value
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
    verify_wiring_core(circuit, public, proof, true, ch)
}

pub fn verify_wiring<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    proof: &WiringProof,
    ch: &mut C,
) -> Result<Vec<(Vec<F128>, F128)>, WiringError> {
    verify_wiring_core(circuit, public, proof, false, ch).map(|(g, _)| g)
}

fn verify_wiring_core<C: Challenger>(
    circuit: &Circuit,
    public: &[F128],
    proof: &WiringProof,
    defer_sigma: bool,
    ch: &mut C,
) -> Result<(Vec<(Vec<F128>, F128)>, SigmaAssertion), WiringError> {
    if public.len() != circuit.num_public() {
        return Err(WiringError::MalformedProof);
    }
    let cells = circuit.cells();
    let (nu, mu) = (cells.nu(), cells.mu());
    if proof.gather.len() != cells.num_gate_slots() {
        return Err(WiringError::MalformedProof);
    }

    let claim = if defer_sigma {
        product_gkr::verify_batched(mu, &proof.gkr, ch)
    } else {
        product_gkr::verify_batched_with_sigma(mu, &proof.gkr, circuit.sigma(), ch)
    }
    .map_err(WiringError::Gkr)?;

    // ---- Recombination (design doc, Lemma "Gather factorization"):
    // ŵ(ρ) = Σ_{gate ι} eq(ρ_ι, ι)·v_ι + Σ_{public ι} eq(ρ_ι, ι)·v̂_ι(ρ_row).
    // Dummy cell-slots contribute nothing (w = 0 there).
    let eq_row = build_eq(&claim.rho[..nu]);
    let eq_slot = build_eq(&claim.rho[nu..]);
    let mut acc = F128::ZERO;
    for (iota, slot) in cells.slots().iter().enumerate() {
        match *slot {
            CellSlot::Gate { .. } => acc += eq_slot[iota] * proof.gather[iota],
            CellSlot::Public { s } => {
                acc += eq_slot[iota] * circuit.public_slot_eval(s, &eq_row, public)
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
        value: claim.s_sigma_eval,
    };
    Ok((
        (0..cells.num_gate_slots())
            .map(|iota| {
                (
                    cells.gate_claim_point(iota, &claim.rho[..nu]),
                    proof.gather[iota],
                )
            })
            .collect(),
        assertion,
    ))
}

/// The wire classes a circuit declares, as cells — the brute-force oracle's
/// view (and a convenience for test circuits). Cell indices back to `(slot,
/// row)`.
pub fn wire_cells(circuit: &Circuit) -> Vec<Vec<Cell>> {
    let nu = circuit.cells().nu();
    circuit
        .wires()
        .iter()
        .map(|class| {
            class
                .iter()
                .map(|&idx| Cell::new(idx >> nu, idx & ((1usize << nu) - 1)))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::element_r1cs::ElementTableBuilder;
    use crate::r1cs::SparseBinaryMatrix;
    use crate::schedule::{TableClass, TableType};
    use std::sync::Arc;

    /// SplitMix64, the repo's test RNG.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128::new(self.next_u64(), self.next_u64())
        }
    }

    fn stub() -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            num_rows: 0,
            num_cols: 0,
            rows: Vec::new(),
        }
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
        assert_eq!(
            mk(vec![vec![Cell::new(2, 0), Cell::new(2, 1)]]),
            Some(CircuitError::MultipleProducers { class: 0 }),
            "two producers in one class"
        );
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
            Err(WiringError::Gkr(product_gkr::VerifyError::ProductMismatch))
        );
        // Break a WIRED public word instead (public slot 0, row 4 — the cell
        // `Cell::new(8, 4)` above). An unwired public word is unconstrained by
        // the wiring, so it must be a wired one.
        let mut bad_public = public.clone();
        bad_public[4] += F128::ONE;
        assert_eq!(
            roundtrip(&circuit, &packed, &bad_public),
            Err(WiringError::Gkr(product_gkr::VerifyError::ProductMismatch))
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
}
