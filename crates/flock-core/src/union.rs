//! The union instance — multi-table Phase 2 (M1 plumbing, M3 binding).
//!
//! [`UnionInstance`] wraps a [`Registry`] + counts pair (the static slot
//! layout of `schedule.rs` plus the per-proof declared counts) and derives
//! everything the prove/verify paths need from the union address space,
//! replacing what [`BlockR1cs`] provides for a single table today: the
//! count-derived run-list [`PaddingSpec`], the union jagged-grid heights
//! (count-dependent height-`n_t` stacking since M5 — the committed size is
//! count-proportional, floored at the smallest embedded Ligerito config),
//! the layout-aware claim points, the union witness assembly, and — since
//! M3 — the multi-table statement binding ([`Self::bind_statement`], label
//! `flock-mixed-v1`).
//!
//! Under the uniform-capacity convention the union's BatchMajor address
//! split is `[7 in-word | nu batch | M−7−nu chunk]` — structurally a single
//! BatchMajor instance with `k_log = M − nu` (design doc §"The union
//! instance"): every slot shares the row coordinates `[7, 7+nu)`, and a
//! slot's chunk bits together with its frozen prefix form the union
//! chunk-column index. The claim-point helpers below are therefore the
//! `BlockR1cs` BatchMajor formulas evaluated over the union address space;
//! for a one-type registry (one slot at offset 0, `M = m`) they agree with
//! the `BlockR1cs` versions coordinate for coordinate — the union of one
//! slot *is* today's instance.
//!
//! The prove/verify entries (`flock_prover::prover::
//! prove_fast_ligerito_jagged_union` / [`crate::verifier::
//! verify_ligerito_jagged_union`]) accept any registry under the
//! `flock-mixed-v1` binding. The M1/M2 single-type harness binding
//! ([`Self::bind_statement_single_type`], transcript-identical to the
//! direct jagged path) is retained for the byte-identity differential
//! tests — the regression anchor for the plumbing — behind the dedicated
//! `*_harness` entries; it is not a protocol mode.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::QuirkyPoint;
use crate::pcs::Commitment;
use crate::r1cs::{BlockR1cs, WitnessLayout};
use crate::schedule::{Instance, Registry, TableType};
use crate::zerocheck::{K_SKIP, PaddingSpec};

/// Floor of the committed dense-stack size, as a bit-variable count: the
/// smallest embedded Ligerito security config is `m22` (`2^15` packed
/// words), so [`UnionInstance::committed_words`] never shrinks below it —
/// see the config-floor note there.
pub const MIN_DENSE_M: usize = 22;

/// A registry instance viewed as ONE union address space of `2^M` points —
/// the object the union prove/verify paths consume. Thin layer over
/// [`Instance`]: the counts live there; this type adds the derived
/// prove-path bookkeeping (heights, claim points, witness assembly).
#[derive(Clone, Debug)]
pub struct UnionInstance<'r> {
    instance: Instance<'r>,
}

impl<'r> UnionInstance<'r> {
    /// `counts[t]` is the declared invocation count of the registry's type
    /// `t`, in slot order (see [`Instance::new`]).
    pub fn new(registry: &'r Registry, counts: Vec<usize>) -> Self {
        Self::from_instance(Instance::new(registry, counts))
    }

    pub fn from_instance(instance: Instance<'r>) -> Self {
        Self { instance }
    }

    pub fn instance(&self) -> &Instance<'r> {
        &self.instance
    }

    pub fn registry(&self) -> &'r Registry {
        self.instance.registry()
    }

    pub fn counts(&self) -> &[usize] {
        self.instance.counts()
    }

    /// Union variable count `M`: the address space is `{0,1}^M`. The
    /// sumchecks run `M` rounds; registry-static, count-independent.
    pub fn m_total(&self) -> usize {
        self.registry().m_total()
    }

    /// Row/batch variable count `nu` — the uniform capacity convention makes
    /// this the `n_log` of the union viewed as one BatchMajor instance.
    pub fn n_log(&self) -> usize {
        self.registry().nu()
    }

    /// Packed length of the union buffer in 128-bit words = `2^(M−7)`.
    pub fn packed_len(&self) -> usize {
        1usize << (self.m_total() - 7)
    }

    /// Union chunk-column variable count `M − 7 − nu`; the jagged grid has
    /// `2^col_log` columns.
    pub fn col_log(&self) -> usize {
        self.m_total() - 7 - self.n_log()
    }

    /// The count-derived run-list padding over the union BatchMajor buffer —
    /// delegates to [`Instance::padding_spec`].
    pub fn padding_spec(&self) -> PaddingSpec {
        self.instance.padding_spec()
    }

    /// Per-chunk-column heights (in packed words) of the union jagged grid,
    /// for the jagged opening path (`pcs::open_batch_jagged_ligerito`):
    /// `2^col_log` entries in union column order. Slot `t` occupies columns
    /// `[o_t >> (7+nu), o_t >> (7+nu) + 2^{k_log_t−7})` (alignment makes the
    /// offset exact). Shared by the prover and verifier wiring — any
    /// divergence is a transcript break, so both derive it here.
    ///
    /// **Height-`n_t` stacking (M5):** every USED chunk-column — the
    /// leading `ceil(useful_bits_t/128)` columns of slot `t` — has height
    /// `n_t`, the DECLARED count (an arbitrary integer in `[0, 2^nu]`), so
    /// the committed area is count-proportional. Dummy rows `[n_t, 2^nu)`
    /// are dropped from the committed stack along with the useless
    /// chunk-columns and the inter-slot/trailing gaps (all height 0). Every
    /// dropped word of the padded virtual buffer is identically zero — the
    /// partial witness drivers zero dummy rows, useless columns and gaps
    /// are zero by construction ([`Self::compact_witness`] debug-asserts
    /// this) — which is what keeps the fused-opening identity
    /// `⟨q, W_ρ⟩ = f̂(ρ)` intact. The `col_prefix_sums` derived from these
    /// heights ARE the compaction map: `unrank ≡` [`Self::compact_witness`].
    ///
    /// COUNT-DEPENDENT (per proof, unlike M4's registry-static capacity
    /// heights): both sides derive the heights and their `col_prefix_sums`
    /// from the public counts, so a wrong declared count diverges here (and
    /// in the jagged assist's layout evaluator) in addition to the
    /// transcript binding and the lincheck's const-pin target. A one-slot
    /// registry at FULL utilization (`n_t = 2^nu`) reproduces
    /// [`BlockR1cs::jagged_heights`] exactly (the M1 byte-identity anchor).
    pub fn jagged_heights(&self) -> Vec<u64> {
        let nu = self.n_log();
        let mut heights = vec![0u64; 1usize << self.col_log()];
        let registry = self.registry();
        for ((ty, slot), &n_t) in registry
            .types()
            .iter()
            .zip(registry.slots())
            .zip(self.counts())
        {
            let col_offset = slot.offset >> (7 + nu);
            for h in &mut heights[col_offset..col_offset + self.used_cols(ty)] {
                *h = n_t as u64;
            }
        }
        heights
    }

    /// Used chunk-columns of a type: the leading `ceil(useful_bits/128)`
    /// columns carry data; the rest are dropped from the committed stack.
    fn used_cols(&self, ty: &TableType) -> usize {
        ty.useful_bits.div_ceil(128).min(1usize << (ty.k_log - 7))
    }

    // -----------------------------------------------------------------------
    // The dense-stack commit (M4/M5): only the declared rows of the used
    // chunk-columns are committed, stacked contiguously — count-dependent.
    // -----------------------------------------------------------------------

    /// Words of the un-padded dense stack: `Σ_t n_t · used_cols_t` — the
    /// jagged area (= `Σ` [`Self::jagged_heights`]), count-proportional
    /// under height-`n_t` stacking.
    pub fn dense_words(&self) -> usize {
        self.registry()
            .types()
            .iter()
            .zip(self.counts())
            .map(|(ty, &n_t)| self.used_cols(ty) * n_t)
            .sum()
    }

    /// Committed length of the dense stack `q` in packed words:
    /// [`Self::dense_words`] rounded up to a power of two (Ligerito commits
    /// power-of-two messages; the pad tail is zero), then clamped to
    /// `[2^(MIN_DENSE_M − 7), packed_len]`.
    ///
    /// **The config floor:** Ligerito security configs are derived and
    /// embedded per committed size, and the smallest shipped config is
    /// `m22` (`2^15` packed words) — see
    /// [`crate::pcs::ligerito::embedded_security_config`]. Low-count
    /// instances therefore never commit below `2^15` words; the sub-floor
    /// tail is zero padding. The floor is additionally capped at the
    /// union's own padded size so the committed (dense) domain never
    /// exceeds the virtual domain — reachable only for sub-floor address
    /// spaces (`M < MIN_DENSE_M`, test-scale registries), where the
    /// committed length is simply the padded length.
    pub fn committed_words(&self) -> usize {
        let floor = 1usize << (MIN_DENSE_M - 7);
        self.dense_words()
            .next_power_of_two()
            .max(floor)
            .min(self.packed_len())
    }

    /// Bit-variable count of the committed polynomial:
    /// `log2(committed_words) + 7`. This — not [`Self::m_total`] — sizes the
    /// `PcsParams` / Ligerito config of the union commit; the PIOP and the
    /// virtual-opening sumcheck keep running over the `M`-variable padded
    /// address space.
    pub fn dense_m(&self) -> usize {
        self.committed_words().trailing_zeros() as usize + 7
    }

    /// Whether the compaction map is the identity: every slot is at FULL
    /// utilization (`n_t = 2^nu`, so no dummy row is truncated away), every
    /// used chunk-column's stacked offset equals its padded offset (no
    /// dropped column precedes any used column), and the committed length
    /// equals the padded length. True for the M1/M2 byte-identity anchors —
    /// single-slot registries at full utilization whose used columns exceed
    /// half the padded space (BLAKE3: 121 of 128; SHA-256: 246 of 256) —
    /// where `q` IS the padded buffer.
    pub fn compaction_is_identity(&self) -> bool {
        let nu = self.n_log();
        let mut cursor = 0usize; // stacked word offset
        for ((ty, slot), &n_t) in self
            .registry()
            .types()
            .iter()
            .zip(self.registry().slots())
            .zip(self.counts())
        {
            if cursor != slot.offset >> 7 || n_t != 1usize << nu {
                return false;
            }
            cursor += self.used_cols(ty) << nu;
        }
        self.committed_words() == self.packed_len()
    }

    /// Assemble the committed dense stack `q` from the padded union buffer:
    /// per slot in order, the DECLARED `n_t`-row prefix of each of its used
    /// chunk-columns, stacked contiguously, zero-padded to
    /// [`Self::committed_words`]. Dummy rows `[n_t, 2^nu)`, useless
    /// chunk-columns, and the inter-slot/trailing gaps are dropped. This is
    /// exactly the map the `col_prefix_sums`/`unrank` of
    /// [`Self::jagged_heights`] induces.
    ///
    /// Debug builds assert the soundness invariant of the height-`n_t`
    /// transport: every DROPPED word of the padded buffer is zero, so the
    /// fused-opening identity `⟨q, W_ρ⟩ = f̂(ρ)` holds (the deleted terms of
    /// `f̂(ρ)` were all zero). Honest witnesses satisfy this by
    /// construction — the partial batch-major drivers zero dummy rows (pin
    /// included), and useless columns/gaps are never written.
    pub fn compact_witness(&self, z_padded: &[F128]) -> Vec<F128> {
        assert_eq!(z_padded.len(), self.packed_len(), "padded buffer length");
        debug_assert!(
            self.dropped_words_are_zero(z_padded),
            "padded buffer must be zero on every dropped word \
             (dummy rows, useless columns, gaps)"
        );
        let nu = self.n_log();
        let mut q = vec![F128::ZERO; self.committed_words()];
        let mut cursor = 0usize;
        for ((ty, slot), &n_t) in self
            .registry()
            .types()
            .iter()
            .zip(self.registry().slots())
            .zip(self.counts())
        {
            let start = slot.offset >> 7;
            for c in 0..self.used_cols(ty) {
                let col = start + (c << nu);
                q[cursor..cursor + n_t].copy_from_slice(&z_padded[col..col + n_t]);
                cursor += n_t;
            }
        }
        debug_assert_eq!(cursor, self.dense_words());
        q
    }

    /// Whether every word of the padded buffer that [`Self::compact_witness`]
    /// drops is zero — the height-`n_t` transport's soundness invariant
    /// (debug-asserted there).
    fn dropped_words_are_zero(&self, z_padded: &[F128]) -> bool {
        let nu = self.n_log();
        let mut kept = vec![false; z_padded.len()];
        for ((ty, slot), &n_t) in self
            .registry()
            .types()
            .iter()
            .zip(self.registry().slots())
            .zip(self.counts())
        {
            let start = slot.offset >> 7;
            for c in 0..self.used_cols(ty) {
                let col = start + (c << nu);
                kept[col..col + n_t].iter_mut().for_each(|k| *k = true);
            }
        }
        z_padded
            .iter()
            .zip(&kept)
            .all(|(w, &k)| k || *w == F128::ZERO)
    }

    // -----------------------------------------------------------------------
    // Layout-aware claim points — the union counterparts of the BlockR1cs
    // BatchMajor bookkeeping (`x_ab_from_mlv` / `ab_claim_point` /
    // `c_claim_point`). The union address order is `[6 skip | dim6 | nu batch
    // | col_log chunk]`, so the formulas are the BatchMajor ones with
    // `(m, n_log) = (M, nu)`; they depend on no per-slot data, which is what
    // makes them multi-slot-ready as-is (the row coordinates are shared by
    // every slot under uniform capacity). Shared by prover and verifier —
    // any divergence is a transcript break, so both call these.
    // -----------------------------------------------------------------------

    /// Lincheck's **semantic** quirky point from the zerocheck claim: split
    /// the address-ordered `mlv` challenges (length `M − 6`) into
    /// `x_inner_rest = [dim6, chunk…]` and `x_outer = batch`. Union analog of
    /// [`BlockR1cs::x_ab_from_mlv`] (BatchMajor).
    pub fn x_ab_from_mlv(&self, z_skip: F128, mlv: &[F128]) -> QuirkyPoint {
        let nu = self.n_log();
        assert_eq!(mlv.len(), self.m_total() - K_SKIP);
        let mut x_inner_rest = Vec::with_capacity(1 + self.col_log());
        x_inner_rest.push(mlv[0]);
        x_inner_rest.extend_from_slice(&mlv[1 + nu..]);
        QuirkyPoint {
            z_skip,
            x_inner_rest,
            x_outer: mlv[1..1 + nu].to_vec(),
        }
    }

    /// Address-ordered `ZClaim` point for the AB claim after lincheck
    /// replaces the inner coordinates with `(r_inner_skip, r_inner_rest)`.
    /// Union analog of [`BlockR1cs::ab_claim_point`] (BatchMajor): the
    /// address-ordered suffix is `[dim6 | batch | chunk]`.
    pub fn ab_claim_point(
        &self,
        r_inner_skip: F128,
        r_inner_rest: &[F128],
        x_outer: &[F128],
    ) -> QuirkyPoint {
        assert_eq!(x_outer.len(), self.n_log());
        assert_eq!(r_inner_rest.len(), 1 + self.col_log());
        let mut suffix = Vec::with_capacity(x_outer.len() + r_inner_rest.len() - 1);
        suffix.extend_from_slice(x_outer);
        suffix.extend_from_slice(&r_inner_rest[1..]);
        QuirkyPoint {
            z_skip: r_inner_skip,
            x_inner_rest: vec![r_inner_rest[0]],
            x_outer: suffix,
        }
    }

    /// Address-ordered `ZClaim` point for the C claim from the zerocheck's
    /// `r_rest` (already address-ordered). Union analog of
    /// [`BlockR1cs::c_claim_point`] (BatchMajor).
    pub fn c_claim_point(&self, z_skip: F128, r_rest: &[F128]) -> QuirkyPoint {
        assert_eq!(r_rest.len(), self.m_total() - K_SKIP);
        QuirkyPoint {
            z_skip,
            x_inner_rest: vec![r_rest[0]],
            x_outer: r_rest[1..].to_vec(),
        }
    }

    // -----------------------------------------------------------------------
    // Statement binding: the flock-mixed-v1 protocol binding, plus the M1/M2
    // single-type harness binding (differential tests only).
    // -----------------------------------------------------------------------

    /// The multi-table statement binding (design doc §"Statement digest and
    /// transcript"): absorb, before any challenge is squeezed and in this
    /// order, the `flock-mixed-v1` domain label, the registry digest
    /// ([`Registry::digest`]), the counts vector (one u64 LE per type, in
    /// slot order, as a single byte string — its length is additionally
    /// bound through the digest's type count), and the commitment root. The
    /// counts are the only per-proof statement data; everything else is
    /// registry-static.
    ///
    /// Domain-separated from the single-table binding
    /// ([`crate::proof::bind_statement`]: `flock-r1cs-v0` + the `BlockR1cs`
    /// statement digest), so a mixed proof can never be replayed as a
    /// single-table proof or vice versa. This is also why a SINGLE-TYPE
    /// instance proved under this binding is deliberately **not**
    /// byte-identical to the direct jagged path — that byte-identity is the
    /// harness binding's job ([`Self::bind_statement_single_type`]).
    pub fn bind_statement<Ch: Challenger>(&self, challenger: &mut Ch, commitment: &Commitment) {
        challenger.observe_label(b"flock-mixed-v1");
        challenger.observe_bytes(&self.registry().digest());
        let mut counts_le = Vec::with_capacity(8 * self.counts().len());
        for &n_t in self.counts() {
            counts_le.extend_from_slice(&(n_t as u64).to_le_bytes());
        }
        challenger.observe_bytes(&counts_le);
        challenger.observe_bytes(&commitment.root);
    }

    /// M1/M2 **harness** guard (differential tests only): the registry has
    /// exactly one type and `slot_r1cs` is that type's single-table
    /// [`BlockR1cs`] view (same variable count, width, useful bits, const
    /// pin, BatchMajor layout, `k_skip = 6`). Returns the type. The
    /// `*_harness` prove/verify entries call this before doing anything
    /// transcript-visible; the protocol entries (`flock-mixed-v1` binding)
    /// do not. (The base matrices are not compared — they are bound by
    /// [`Self::bind_statement_single_type`] through the `BlockR1cs`
    /// statement digest.)
    pub fn expect_single_type_slot(&self, slot_r1cs: &BlockR1cs) -> &'r TableType {
        let registry = self.registry();
        assert_eq!(
            registry.num_types(),
            1,
            "the single-type harness binding is single-type only; \
             multi-type registries go through the flock-mixed-v1 binding"
        );
        let ty = &registry.types()[0];
        assert_eq!(
            slot_r1cs.layout,
            WitnessLayout::BatchMajor,
            "the union path requires the BatchMajor witness layout"
        );
        assert_eq!(slot_r1cs.m, self.m_total(), "slot BlockR1cs m != union M");
        assert_eq!(slot_r1cs.k_log, ty.k_log, "slot BlockR1cs k_log mismatch");
        assert_eq!(slot_r1cs.k_skip, K_SKIP, "BatchMajor requires k_skip = 6");
        assert_eq!(
            slot_r1cs.useful_bits, ty.useful_bits,
            "slot BlockR1cs useful_bits mismatch"
        );
        assert_eq!(
            slot_r1cs.const_pin, ty.const_pin,
            "slot BlockR1cs const_pin mismatch"
        );
        ty
    }

    /// M1/M2 **harness** transcript binding (differential tests only): bind
    /// exactly today's single-table statement — [`crate::proof::
    /// bind_statement`] over the slot's [`BlockR1cs`] statement digest + the
    /// commitment root — keeping the transcript byte-identical to the
    /// existing jagged path. Single-type registries only.
    ///
    /// Since M3 the protocol binding is [`Self::bind_statement`]
    /// (`flock-mixed-v1`); this one is kept solely so the M1/M2
    /// byte-identity differential tests (`flock-prover`'s
    /// `tests/union_roundtrip.rs`) remain a live regression anchor for the
    /// union plumbing. It is not a protocol mode and does not appear in any
    /// wire format.
    pub fn bind_statement_single_type<Ch: Challenger>(
        &self,
        challenger: &mut Ch,
        slot_r1cs: &BlockR1cs,
        commitment: &Commitment,
    ) {
        assert_eq!(
            self.registry().num_types(),
            1,
            "the harness binding is the single-table statement digest; \
             multi-type registries go through the flock-mixed-v1 binding"
        );
        crate::proof::bind_statement(challenger, slot_r1cs, commitment);
    }

    /// Assemble the union witness from per-slot packed buffers: place each
    /// slot's `(z, a, b)` at its aligned word offset `o_t >> 7` in
    /// union-sized buffers (dummy regions and the gap stay zero). One bundle
    /// per registry type, in slot order.
    ///
    /// A single-slot registry (whose slot spans the whole address space) is
    /// a zero-copy passthrough — the returned buffers ARE the slot's,
    /// unmoved, so M1 costs nothing over the single-table path.
    pub fn assemble_witness(
        &self,
        mut slot_witnesses: Vec<SlotWitness>,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
        let registry = self.registry();
        assert_eq!(
            slot_witnesses.len(),
            registry.num_types(),
            "need one witness bundle per registry type"
        );
        for (slot, w) in registry.slots().iter().zip(&slot_witnesses) {
            let words = 1usize << (slot.m_slot - 7);
            assert_eq!(w.z_packed.len(), words, "slot z_packed length mismatch");
            assert_eq!(w.a_packed.len(), words, "slot a_packed length mismatch");
            assert_eq!(w.b_packed.len(), words, "slot b_packed length mismatch");
        }

        // Single slot spanning the whole space: pass the buffers through.
        if registry.num_types() == 1 && registry.slots()[0].m_slot == self.m_total() {
            let w = slot_witnesses.pop().expect("asserted one bundle above");
            return (w.z_packed, w.a_packed, w.b_packed);
        }
        self.scatter_witnesses(slot_witnesses)
    }

    /// General placement path: zero-initialized union buffers with each
    /// slot's data copied at its word offset.
    ///
    /// Support-proportional copies (M6): a slot at partial utilization only
    /// copies the declared `n_t`-word prefix of each used chunk-column — the
    /// witness contract (see `prove_fast_ligerito_jagged_union`) makes the
    /// slot buffers zero everywhere else (dummy rows, useless columns), so
    /// the zero-initialized union buffers already hold those words'
    /// values and the result is byte-identical to the full-slot copy.
    /// Full-utilization slots keep the one-memcpy path.
    fn scatter_witnesses(
        &self,
        slot_witnesses: Vec<SlotWitness>,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
        let nu = self.n_log();
        let len = self.packed_len();
        let mut z = vec![F128::ZERO; len];
        let mut a = vec![F128::ZERO; len];
        let mut b = vec![F128::ZERO; len];
        for (((ty, slot), w), &n_t) in self
            .registry()
            .types()
            .iter()
            .zip(self.registry().slots())
            .zip(&slot_witnesses)
            .zip(self.counts())
        {
            let start = slot.offset >> 7;
            let words = 1usize << (slot.m_slot - 7);
            if n_t == 1usize << nu {
                z[start..start + words].copy_from_slice(&w.z_packed);
                a[start..start + words].copy_from_slice(&w.a_packed);
                b[start..start + words].copy_from_slice(&w.b_packed);
            } else {
                debug_assert!(
                    slot_buffer_zero_off_support(w, self.used_cols(ty), nu, n_t),
                    "slot buffers must be zero on dummy rows and useless columns \
                     (the union witness contract)"
                );
                for c in 0..self.used_cols(ty) {
                    let col = c << nu;
                    z[start + col..start + col + n_t].copy_from_slice(&w.z_packed[col..col + n_t]);
                    a[start + col..start + col + n_t].copy_from_slice(&w.a_packed[col..col + n_t]);
                    b[start + col..start + col + n_t].copy_from_slice(&w.b_packed[col..col + n_t]);
                }
            }
        }
        (z, a, b)
    }
}

/// Whether a slot's `(z, a, b)` buffers are zero outside the declared
/// `n_t`-row prefixes of the used chunk-columns — the union witness
/// contract [`UnionInstance::scatter_witnesses`] debug-asserts before its
/// support-proportional copies.
fn slot_buffer_zero_off_support(w: &SlotWitness, used_cols: usize, nu: usize, n_t: usize) -> bool {
    let mut kept = vec![false; w.z_packed.len()];
    for c in 0..used_cols {
        kept[(c << nu)..(c << nu) + n_t].fill(true);
    }
    [&w.z_packed, &w.a_packed, &w.b_packed].iter().all(|buf| {
        buf.iter()
            .zip(&kept)
            .all(|(word, &k)| k || *word == F128::ZERO)
    })
}

/// One slot's packed witness buffers, exactly as the existing batch-major
/// drivers produce them (`generate_witness_batch_major`'s `(z, a, b, _)`):
/// `z`, `a = A·z`, `b = B·z`, each `2^{m_t−7}` packed words in the slot's
/// BatchMajor layout. The lincheck stripe stays outside — it is consumed
/// per-slot by the lincheck, never assembled into union buffers.
#[derive(Clone, Debug, Default)]
pub struct SlotWitness {
    pub z_packed: Vec<F128>,
    pub a_packed: Vec<F128>,
    pub b_packed: Vec<F128>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::SparseBinaryMatrix;

    /// Empty matrix stub — nothing here applies the matrices (same practice
    /// as the schedule.rs layout tests).
    fn stub() -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            num_rows: 0,
            num_cols: 0,
            rows: Vec::new(),
        }
    }

    fn ty(k_log: usize, useful_bits: usize) -> TableType {
        TableType {
            k_log,
            useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: None,
        }
    }

    /// Today's single-table instance for the same geometry, BatchMajor.
    fn block_r1cs(k_log: usize, useful_bits: usize, nu: usize) -> BlockR1cs {
        BlockR1cs {
            m: nu + k_log,
            k_log,
            k_skip: K_SKIP,
            useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            layout: WitnessLayout::BatchMajor,
            const_pin: None,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }

    /// SplitMix64 PRNG, deterministic.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn next_f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.next_f128()).collect()
        }
    }

    /// A single-slot union at full utilization declares the same jagged grid
    /// as today's `BlockR1cs::jagged_heights` — on the BLAKE3 and SHA-256
    /// shapes (121 of 128 / 246 of 256 useful chunk-columns).
    #[test]
    fn single_slot_heights_match_block_r1cs_at_full_utilization() {
        for &(k_log, useful_bits, nu) in &[(14usize, 15_409usize, 3usize), (15, 31_401, 2)] {
            let reg = Registry::new(vec![ty(k_log, useful_bits)], nu);
            let union = UnionInstance::new(&reg, vec![1 << nu]);
            let r1cs = block_r1cs(k_log, useful_bits, nu);
            assert_eq!(
                union.jagged_heights(),
                r1cs.jagged_heights(),
                "heights diverged (k_log={k_log})"
            );
            assert_eq!(union.n_log(), r1cs.n_log());
            assert_eq!(union.m_total(), r1cs.m);
        }
    }

    /// The union claim-point helpers reproduce the BlockR1cs BatchMajor
    /// versions on random inputs — the union of one slot at offset 0 is
    /// today's instance verbatim.
    #[test]
    fn single_slot_claim_points_match_block_r1cs() {
        let (k_log, useful_bits, nu) = (14usize, 15_409usize, 3usize);
        let reg = Registry::new(vec![ty(k_log, useful_bits)], nu);
        let union = UnionInstance::new(&reg, vec![1 << nu]);
        let r1cs = block_r1cs(k_log, useful_bits, nu);
        let m = r1cs.m;
        let mut rng = Rng::new(0x0C1A_11A5);

        for _ in 0..16 {
            let z_skip = rng.next_f128();
            let mlv = rng.f128_vec(m - K_SKIP);
            let x_ab_union = union.x_ab_from_mlv(z_skip, &mlv);
            let x_ab_r1cs = r1cs.x_ab_from_mlv(z_skip, &mlv);
            assert_eq!(x_ab_union, x_ab_r1cs, "x_ab_from_mlv diverged");

            let r_inner_skip = rng.next_f128();
            let r_inner_rest = rng.f128_vec(k_log - K_SKIP);
            assert_eq!(
                union.ab_claim_point(r_inner_skip, &r_inner_rest, &x_ab_union.x_outer),
                r1cs.ab_claim_point(r_inner_skip, &r_inner_rest, &x_ab_r1cs.x_outer),
                "ab_claim_point diverged"
            );

            let r_rest = rng.f128_vec(m - K_SKIP);
            assert_eq!(
                union.c_claim_point(z_skip, &r_rest),
                r1cs.c_claim_point(z_skip, &r_rest),
                "c_claim_point diverged"
            );
        }
    }

    /// The union padding spec delegates to `Instance::padding_spec` and, at
    /// full utilization, classifies exactly the bits `BlockR1cs::padding_spec`
    /// does (run encodings differ — multi-run vs one giant block — the
    /// classification must not; the schedule.rs Phase 0 tests prove the
    /// multi-run encoding drives the zerocheck kernels byte-identically).
    #[test]
    fn single_slot_padding_spec_classifies_like_block_r1cs() {
        let (k_log, useful_bits, nu) = (14usize, 15_409usize, 3usize);
        let reg = Registry::new(vec![ty(k_log, useful_bits)], nu);
        let union = UnionInstance::new(&reg, vec![1 << nu]);
        let r1cs = block_r1cs(k_log, useful_bits, nu);
        assert_eq!(
            union.padding_spec(),
            union.instance().padding_spec(),
            "padding_spec must delegate to Instance"
        );
        assert_eq!(
            union.padding_spec().useful_intervals(),
            r1cs.padding_spec().useful_intervals(),
            "count-derived spec must classify the same bits useful as today's"
        );
    }

    /// Multi-slot heights against hand-computed values: two synthetic types
    /// (κ = 10/9, ν = 3 → M = 14, 16 union columns). Height-`n_t` stacking
    /// (M5) semantics: used columns at the DECLARED count `n_t` (arbitrary
    /// integer), useless columns and the gap dropped (height 0) — the
    /// heights are count-dependent, per proof.
    #[test]
    fn multi_slot_heights_hand_computed() {
        // Type A: 8 chunk-columns, ceil(700/128) = 6 used; type B: 4
        // chunk-columns at column offset 8192 >> (7+3) = 8, ceil(300/128) = 3
        // used. Columns 12..16 are the gap past the last slot.
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        #[rustfmt::skip]
        let cases: [(Vec<usize>, Vec<u64>); 3] = [
            // Partial, non-power-of-two counts.
            (vec![5, 3], vec![
                5, 5, 5, 5, 5, 5, 0, 0, // slot A: 6 used at n_A = 5, 2 dropped
                3, 3, 3, 0,             // slot B: 3 used at n_B = 3, 1 dropped
                0, 0, 0, 0,             // gap: dropped
            ]),
            // Full utilization: capacity heights — M4's grid, exactly.
            (vec![8, 8], vec![
                8, 8, 8, 8, 8, 8, 0, 0,
                8, 8, 8, 0,
                0, 0, 0, 0,
            ]),
            // A zero count drops the slot's columns entirely.
            (vec![0, 8], vec![
                0, 0, 0, 0, 0, 0, 0, 0,
                8, 8, 8, 0,
                0, 0, 0, 0,
            ]),
        ];
        for (counts, expected) in cases {
            let union = UnionInstance::new(&reg, counts.clone());
            assert_eq!(union.m_total(), 14);
            assert_eq!(union.col_log(), 4);
            assert_eq!(union.jagged_heights(), expected, "counts {counts:?}");
            // The heights' area IS dense_words (unrank ≡ compaction map).
            assert_eq!(
                union.jagged_heights().iter().sum::<u64>(),
                union.dense_words() as u64,
                "counts {counts:?}"
            );
        }
    }

    /// Dense-stack size arithmetic under height-`n_t` stacking (M5): the
    /// count-proportional dense area, the config floor's clamp on sub-floor
    /// address spaces, and the shapes where nothing changes — single-slot
    /// full utilization (the M1/M2 byte-identity anchors) and full-
    /// utilization mixes of ≥94%-column-dense types (BLAKE3 121/128,
    /// SHA-256 246/256), which round straight back to the padded size.
    #[test]
    fn dense_stack_sizes_and_area_saving() {
        // Synthetic column-sparse pair: A uses 4 of 8 columns, B 3 of 4.
        // Dense 7·8 = 56 words at full utilization; the padded space (128
        // words) sits far below the m22 config floor, so the committed
        // length clamps to the padded length.
        let reg = Registry::new(vec![ty(10, 512), ty(9, 300)], 3);
        let union = UnionInstance::new(&reg, vec![8, 8]);
        assert_eq!(union.dense_words(), 56);
        assert_eq!(union.packed_len(), 128);
        assert_eq!(
            union.committed_words(),
            union.packed_len(),
            "sub-floor address spaces commit the padded length"
        );
        assert_eq!(union.dense_m(), union.m_total());
        assert!(!union.compaction_is_identity());
        // Count-proportional dense area, monotone in the counts.
        let dense = |counts: Vec<usize>| UnionInstance::new(&reg, counts).dense_words();
        assert_eq!(dense(vec![5, 3]), 4 * 5 + 3 * 3);
        assert_eq!(dense(vec![0, 8]), 3 * 8);
        assert!(dense(vec![5, 3]) < dense(vec![8, 8]));
        // The heights' area IS dense_words (unrank ≡ compaction map).
        assert_eq!(
            union.jagged_heights().iter().sum::<u64>(),
            union.dense_words() as u64
        );

        // Single-slot BLAKE3/SHA-256 shapes at full utilization: dense
        // rounds back to padded (used columns > half), and the compaction
        // map is the identity — the M1/M2 byte-identity precondition.
        // Partial counts break the identity (dummy rows get dropped).
        for &(k_log, useful_bits, nu) in &[(14usize, 15_409usize, 3usize), (15, 31_401, 2)] {
            let reg = Registry::new(vec![ty(k_log, useful_bits)], nu);
            let union = UnionInstance::new(&reg, vec![1 << nu]);
            assert_eq!(union.committed_words(), union.packed_len());
            assert_eq!(union.dense_m(), union.m_total());
            assert!(union.compaction_is_identity());
            let partial = UnionInstance::new(&reg, vec![(1 << nu) - 1]);
            assert!(!partial.compaction_is_identity());
        }

        // Mixed BLAKE3+SHA-256 (the M3/M4 registry shape, scaled) at FULL
        // utilization: 367 of 512 columns used → committed == padded in
        // words, but the compaction is NOT the identity (SHA-256 drops 10
        // columns before BLAKE3's slot, which stacks at column 246 instead
        // of 256).
        let reg = Registry::new(vec![ty(14, 15_409), ty(15, 31_401)], 3);
        let union = UnionInstance::new(&reg, vec![8, 8]);
        assert_eq!(union.dense_words(), (246 + 121) << 3);
        assert_eq!(union.committed_words(), union.packed_len());
        assert!(!union.compaction_is_identity());
    }

    /// THE M5 area gate at the sizing level, on the real BLAKE3+SHA-256
    /// column shapes at ν = 7 (the `blake3+sha2@nu7` tier geometry, M = 23):
    /// committed words scale with the counts — halving at counts (32, 32)
    /// against M4's capacity-height 2^16 — bottoming out at the m22 config
    /// floor, and monotone (componentwise higher counts commit ≥ words).
    #[test]
    fn committed_words_scale_with_counts_and_floor() {
        // Registry sorts κ descending: slot 0 is the SHA-256 shape
        // (246/256 used columns), slot 1 the BLAKE3 shape (121/128).
        let reg = Registry::new(vec![ty(14, 15_409), ty(15, 31_401)], 7);
        let u = |counts: [usize; 2]| UnionInstance::new(&reg, counts.to_vec());
        assert_eq!(u([0, 0]).m_total(), 23);
        assert_eq!(u([0, 0]).packed_len(), 1 << 16);

        // The gate: counts (32, 32) → dense 32·(246+121) = 11 744 words →
        // committed 2^15 (the config floor; next_pow2 alone would say 2^14)
        // — HALF of M4's capacity-height 2^16.
        let partial = u([32, 32]);
        assert_eq!(partial.dense_words(), 11_744);
        assert_eq!(partial.committed_words(), 1 << (MIN_DENSE_M - 7));
        assert_eq!(partial.dense_m(), MIN_DENSE_M);
        assert_eq!(
            partial.committed_words() * 2,
            partial.packed_len(),
            "counts (32, 32) must commit half of M4's capacity-height size"
        );

        // Monotone across count vectors (incl. non-powers-of-two), from the
        // floor up to the full-utilization padded size.
        let ladder: [[usize; 2]; 4] = [[8, 8], [32, 32], [50, 37], [128, 128]];
        let mut prev = (0usize, 0usize);
        for counts in ladder {
            let union = u(counts);
            let cur = (union.dense_words(), union.committed_words());
            assert!(
                cur.0 >= prev.0 && cur.1 >= prev.1,
                "committed area must be monotone in the counts ({counts:?})"
            );
            prev = cur;
        }
        assert_eq!(u([8, 8]).committed_words(), 1 << 15, "floor binds");
        assert_eq!(u([50, 37]).dense_words(), 50 * 246 + 37 * 121);
        assert_eq!(u([50, 37]).committed_words(), 1 << 15);
        assert_eq!(u([128, 128]).committed_words(), 1 << 16, "full = M4 size");
    }

    /// `compact_witness` against a hand-built map: the declared `n_t`-row
    /// prefix of each used column lands at its stacked offset, dummy rows,
    /// dropped columns and gaps vanish, the pad tail is zero, and a
    /// single-slot full-utilization identity registry round-trips the
    /// buffer unchanged.
    #[test]
    fn compact_witness_matches_map() {
        let reg = Registry::new(vec![ty(10, 512), ty(9, 300)], 3);
        let union = UnionInstance::new(&reg, vec![5, 3]);
        // Padded buffer: declared word i of used column c holds (c, i)
        // tags; dropped words (dummy rows, useless columns, the gap) stay
        // zero — the honest-witness invariant compact_witness asserts.
        let mut z = vec![F128::ZERO; union.packed_len()];
        for (cols, n_t) in [(0..4usize, 5usize), (8..11, 3)] {
            for c in cols {
                for i in 0..n_t {
                    z[(c << 3) + i] = F128 {
                        lo: i as u64,
                        hi: c as u64,
                    };
                }
            }
        }
        let q = union.compact_witness(&z);
        // Sub-floor address space: committed clamps to the padded length.
        assert_eq!(q.len(), 128);
        assert_eq!(union.dense_words(), 4 * 5 + 3 * 3);
        // Slot A used columns 0..4 stack their 5-word prefixes at 0..20;
        // slot B used columns 8..11 (padded) their 3-word prefixes at
        // 20..29; tail 29..128 zero.
        let mut cursor = 0usize;
        for (padded_col, n_t) in (0..4).map(|c| (c, 5)).chain((8..11).map(|c| (c, 3))) {
            for i in 0..n_t {
                assert_eq!(
                    q[cursor],
                    F128 {
                        lo: i as u64,
                        hi: padded_col as u64
                    },
                    "padded column {padded_col} word {i}"
                );
                cursor += 1;
            }
        }
        assert_eq!(cursor, union.dense_words());
        assert!(q[cursor..].iter().all(|w| *w == F128::ZERO), "pad tail");
        // unrank ≡ compaction: every dense index maps back to the padded
        // word it was copied from.
        let params = crate::pcs::jagged::JaggedParams::from_heights(
            &union.jagged_heights(),
            union.n_log(),
            union.dense_m() - 7,
        );
        for e in 0..union.dense_words() as u64 {
            let (row, col) = params.unrank(e);
            assert_eq!(q[e as usize], z[(col << 3) + row], "unrank at {e}");
        }

        // Identity registry (full utilization): q is byte-identical to the
        // padded buffer.
        let reg1 = Registry::new(vec![ty(10, 700)], 3);
        let union1 = UnionInstance::new(&reg1, vec![8]);
        assert!(union1.compaction_is_identity());
        let mut rng = Rng::new(0xDE_45E);
        let z1 = rng.f128_vec(union1.packed_len());
        // Honest useless columns are zero; emulate by zeroing them so the
        // identity claim is about real buffers.
        let mut z1 = z1;
        for w in &mut z1[(6usize << 3)..] {
            *w = F128::ZERO;
        }
        assert_eq!(union1.compact_witness(&z1), z1);
    }

    /// Single-slot witness assembly is a zero-copy passthrough: the returned
    /// buffers are the slot's own allocations, unmoved.
    #[test]
    fn single_slot_assembly_is_passthrough() {
        let reg = Registry::new(vec![ty(10, 700)], 3);
        let union = UnionInstance::new(&reg, vec![5]);
        let words = union.packed_len();
        assert_eq!(words, 1 << (13 - 7));
        let mut rng = Rng::new(0xA55E_B1E5);
        let w = SlotWitness {
            z_packed: rng.f128_vec(words),
            a_packed: rng.f128_vec(words),
            b_packed: rng.f128_vec(words),
        };
        let ptrs = (
            w.z_packed.as_ptr(),
            w.a_packed.as_ptr(),
            w.b_packed.as_ptr(),
        );
        let (z, a, b) = union.assemble_witness(vec![w]);
        assert_eq!(
            (z.as_ptr(), a.as_ptr(), b.as_ptr()),
            ptrs,
            "single-slot assembly must not copy"
        );
    }

    /// Multi-slot witness assembly places each slot's words at its aligned
    /// word offset `o_t >> 7`, leaving the gap zero — at FULL utilization
    /// (the one-memcpy path), with marks everywhere in the slot buffers.
    #[test]
    fn multi_slot_assembly_places_slots_at_offsets() {
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let union = UnionInstance::new(&reg, vec![8, 8]);
        // Slot A: 2^(13-7) = 64 words at word offset 0; slot B: 32 words at
        // word offset 8192 >> 7 = 64; union: 2^(14-7) = 128 words.
        assert_eq!(union.packed_len(), 128);
        let mark = |tag: u64, n: usize| -> Vec<F128> {
            (0..n)
                .map(|i| F128 {
                    lo: i as u64,
                    hi: tag,
                })
                .collect()
        };
        let slot_a = SlotWitness {
            z_packed: mark(0xA0, 64),
            a_packed: mark(0xA1, 64),
            b_packed: mark(0xA2, 64),
        };
        let slot_b = SlotWitness {
            z_packed: mark(0xB0, 32),
            a_packed: mark(0xB1, 32),
            b_packed: mark(0xB2, 32),
        };
        let (z, a, b) = union.assemble_witness(vec![slot_a, slot_b]);
        for (buf, tag_a, tag_b) in [(&z, 0xA0, 0xB0), (&a, 0xA1, 0xB1), (&b, 0xA2, 0xB2)] {
            assert_eq!(buf.len(), 128);
            assert_eq!(buf[..64], mark(tag_a, 64)[..], "slot A misplaced");
            assert_eq!(buf[64..96], mark(tag_b, 32)[..], "slot B misplaced");
            assert!(
                buf[96..].iter().all(|x| *x == F128::ZERO),
                "gap must stay zero"
            );
        }
    }

    /// Partial-count assembly (the M6 support-proportional copy path): slot
    /// buffers honoring the witness contract — nonzero only on the declared
    /// `n_t`-row prefixes of the used chunk-columns — are placed
    /// byte-identically to a full-slot copy: support words land at their
    /// aligned offsets, and dummy rows, useless columns, and the gap are
    /// zero.
    #[test]
    fn multi_slot_assembly_partial_counts_places_support() {
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let union = UnionInstance::new(&reg, vec![5, 3]);
        assert_eq!(union.packed_len(), 128);
        // Used columns: A = ceil(700/128) = 6 of 8; B = ceil(300/128) = 3 of
        // 4. Support-marked slot buffer: word i of used column c gets a
        // (tag, c, i) mark for rows i < n_t, zero elsewhere.
        let mark = |tag: u64, c: usize, i: usize| F128 {
            lo: ((c as u64) << 32) | i as u64,
            hi: tag,
        };
        let support_buf = |tag: u64, words: usize, used_cols: usize, n_t: usize| -> Vec<F128> {
            let mut v = vec![F128::ZERO; words];
            for c in 0..used_cols {
                for i in 0..n_t {
                    v[(c << 3) + i] = mark(tag, c, i);
                }
            }
            v
        };
        let slot_a = SlotWitness {
            z_packed: support_buf(0xA0, 64, 6, 5),
            a_packed: support_buf(0xA1, 64, 6, 5),
            b_packed: support_buf(0xA2, 64, 6, 5),
        };
        let slot_b = SlotWitness {
            z_packed: support_buf(0xB0, 32, 3, 3),
            a_packed: support_buf(0xB1, 32, 3, 3),
            b_packed: support_buf(0xB2, 32, 3, 3),
        };
        let expected = |tag_a: u64, tag_b: u64| -> Vec<F128> {
            let mut v = vec![F128::ZERO; 128];
            v[..64].copy_from_slice(&support_buf(tag_a, 64, 6, 5));
            v[64..96].copy_from_slice(&support_buf(tag_b, 32, 3, 3));
            v
        };
        let (z, a, b) = union.assemble_witness(vec![slot_a, slot_b]);
        for (buf, tag_a, tag_b) in [(&z, 0xA0, 0xB0), (&a, 0xA1, 0xB1), (&b, 0xA2, 0xB2)] {
            assert_eq!(buf[..], expected(tag_a, tag_b)[..], "tag {tag_a:#x}");
        }
    }

    /// The harness guard rejects multi-type registries — multi-type
    /// instances go through the flock-mixed-v1 binding, never the harness.
    #[test]
    #[should_panic(expected = "single-type only")]
    fn expect_single_type_slot_rejects_multi_type() {
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let union = UnionInstance::new(&reg, vec![5, 3]);
        let r1cs = block_r1cs(10, 700, 3);
        let _ = union.expect_single_type_slot(&r1cs);
    }

    /// The `flock-mixed-v1` binding is deterministic and sensitive to every
    /// bound component — registry digest, counts (value AND slot order), and
    /// commitment root: divergence anywhere yields a different first
    /// challenge, which is what makes the statement non-substitutable.
    #[test]
    fn bind_statement_sensitivity() {
        use crate::challenger::FsChallenger;
        use crate::pcs::PcsParams;

        let commitment = |root_byte: u8| Commitment {
            root: [root_byte; 32],
            params: PcsParams {
                m: 14,
                log_inv_rate: 1,
                log_batch_size: 6,
                profile: Default::default(),
            },
        };
        let sample = |union: &UnionInstance<'_>, root: u8| {
            let mut ch = FsChallenger::new(b"flock-test-v0");
            union.bind_statement(&mut ch, &commitment(root));
            ch.sample_f128()
        };

        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let base = sample(&UnionInstance::new(&reg, vec![5, 3]), 0xAA);
        assert_eq!(
            base,
            sample(&UnionInstance::new(&reg, vec![5, 3]), 0xAA),
            "binding must be deterministic"
        );
        assert_ne!(
            base,
            sample(&UnionInstance::new(&reg, vec![3, 5]), 0xAA),
            "count order must bind"
        );
        assert_ne!(
            base,
            sample(&UnionInstance::new(&reg, vec![5, 4]), 0xAA),
            "count value must bind"
        );
        assert_ne!(
            base,
            sample(&UnionInstance::new(&reg, vec![5, 3]), 0xAB),
            "commitment root must bind"
        );
        // A registry tamper invisible to every other verifier-side quantity
        // (useful_bits +1 within the same chunk-column) still moves the
        // digest, hence the binding.
        let reg2 = Registry::new(vec![ty(10, 701), ty(9, 300)], 3);
        assert_ne!(
            base,
            sample(&UnionInstance::new(&reg2, vec![5, 3]), 0xAA),
            "registry digest must bind"
        );
    }
}
