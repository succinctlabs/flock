//! Multi-table union instances.
//!
//! [`UnionInstance`] wraps a [`Registry`] + counts pair (the static slot
//! layout of `schedule.rs` plus the per-proof declared counts) and derives
//! everything the prove and verify paths need from the union address space:
//! count-derived run-list [`PaddingSpec`], the union jagged-grid heights
//! (count-dependent height-`n_t` stacking; the committed size is
//! count-proportional, floored at the smallest embedded Ligerito config),
//! the layout-aware claim points, the union witness assembly, and
//! the multi-table statement binding ([`Self::bind_statement`], label
//! `flock-mixed-v1`).
//!
//! Under the uniform-capacity convention the union's BatchMajor address
//! split is `[7 in-word | nu batch | M−7−nu chunk]` — structurally a single
//! BatchMajor instance with `k_log = M − nu`. Every slot shares the row
//! coordinates `[7, 7+nu)`, and a
//! slot's chunk bits together with its frozen prefix form the union
//! chunk-column index. The claim-point helpers below are therefore the
//! `BlockR1cs` BatchMajor formulas evaluated over the union address space;
//! for a one-type registry (one slot at offset 0, `M = m`) they agree with
//! the `BlockR1cs` versions coordinate for coordinate.
//!
//! The prove/verify entries (`flock_prover::prover::
//! prove_fast_ligerito_union` / [`crate::verifier::
//! verify_ligerito_union`], and their mixed-class variants)
//! accept any registry under the `flock-mixed-v1` binding.

use core::{mem::take, ops::Range};
use std::iter::repeat_n;

use rayon::{
    join,
    prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut},
};

#[cfg(test)]
use crate::schedule::TableClass;
use crate::{
    challenger::Challenger,
    field::F128,
    lincheck::{QuirkyPoint, SkipPoint},
    merkle::{HashKind, hash_leaf, hash_pair},
    pcs::{Commitment, dense_lanes},
    schedule::{Instance, Registry, TableType},
    scratch::{give_zeroed_f128, take_f128, take_zeroed_f128},
    zerocheck::{K_SKIP, PaddingSpec},
};

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
    /// Optional dense floor `m*`: [`Self::committed_words`] commits at least
    /// `2^(m*−7)` words regardless of content. See
    /// [`Self::set_dense_floor`].
    dense_floor_m: Option<usize>,
}

impl<'r> UnionInstance<'r> {
    /// `counts[t]` is the declared invocation count of the registry's type
    /// `t`, in slot order (see [`Instance::new`]).
    pub fn new(registry: &'r Registry, counts: Vec<usize>) -> Self {
        Self::from_instance(Instance::new(registry, counts))
    }

    pub fn from_instance(instance: Instance<'r>) -> Self {
        Self {
            instance,
            dense_floor_m: None,
        }
    }

    /// Pin the committed size from below: commit `max(next_pow2(content),
    /// 2^(m−7))` words. The envelope capability — shapes whose CONTENT
    /// differs commit (and therefore open, query and absorb) at ONE size, so
    /// a verifier of their proofs has one geometry. The floor only extends
    /// the zero tail the pow2 rounding already commits (see
    /// [`Self::commit_lanes`]: the tail is whole zero lanes, never encoded);
    /// content above the floor commits exactly as without one.
    ///
    /// The floor is STATEMENT data, like the counts: prover and verifier
    /// must construct their instances with the same value or the config
    /// lookup diverges loudly. Panics if the floor exceeds the padded
    /// virtual domain (the committed domain may never outgrow the address
    /// space) — raise `nu` first.
    pub fn set_dense_floor(&mut self, m: usize) {
        assert!(
            m >= MIN_DENSE_M,
            "the config floor already commits 2^{MIN_DENSE_M}"
        );
        assert!(
            1usize << (m - 7) <= self.packed_len(),
            "dense floor m={m} exceeds the padded virtual domain (m_total {}); \
             raise nu before flooring",
            self.m_total()
        );
        self.dense_floor_m = Some(m);
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

    // -----------------------------------------------------------------------
    // Class regions. The two classes occupy DISJOINT aligned subcubes
    // (`Registry::new`), and each class's PIOP runs over its own region only —
    // the boolean zerocheck aliases `c = z`, so on the element region the
    // honest summand `ab − c` is `0·0 − z ≠ 0` and a shared domain could not
    // be proven honestly (see the module-level "Disjoint PIOPs" note).
    // -----------------------------------------------------------------------

    /// `M_bool`: the boolean PIOP's variable count. Its domain is the PREFIX
    /// subcube `[0, 2^{M_bool})` of the union address space, so it reads the
    /// leading `2^{M_bool−7}` words of the padded buffers and its claim points
    /// gain `M − M_bool` frozen-zero high coordinates on the way out
    /// ([`Self::ab_claim_point`] / [`Self::c_claim_point`] append them).
    /// Equals [`Self::m_total`] for a boolean-only registry.
    pub fn m_bool(&self) -> usize {
        self.registry().m_bool()
    }

    /// Packed length of the boolean region in 128-bit words, `2^(M_bool−7)`;
    /// `0` when there are no boolean types (the region is empty — `M_bool = 0`,
    /// so the naive shift would underflow).
    pub fn boolean_packed_len(&self) -> usize {
        if self.num_boolean() == 0 {
            0
        } else {
            1usize << (self.m_bool() - 7)
        }
    }

    /// Boolean-region chunk-column variable count `M_bool − 7 − nu` — the
    /// boolean lincheck's column domain past the shared row block. `0` when
    /// there are no boolean types (no boolean PIOP runs).
    pub fn boolean_col_log(&self) -> usize {
        if self.num_boolean() == 0 {
            0
        } else {
            self.m_bool() - 7 - self.n_log()
        }
    }

    /// Number of boolean types (a prefix of the slots) and of element types
    /// (the suffix).
    pub fn num_boolean(&self) -> usize {
        self.registry().num_boolean()
    }
    pub fn num_element(&self) -> usize {
        self.registry().num_element()
    }

    /// Whether the registry has any large-field type — i.e. whether the
    /// element PIOP runs at all. `false` restores every boolean-only path
    /// exactly.
    pub fn has_element(&self) -> bool {
        self.num_element() > 0
    }

    /// `M_elem`: the element region is the aligned subcube
    /// `[element_base, element_base + 2^{M_elem})`, so the element PIOP runs
    /// over `M_elem − 7` WORD variables (elements are words; there is no
    /// in-word structure to fold).
    pub fn m_elem(&self) -> usize {
        self.registry().m_elem()
    }

    /// Word range of the element region inside the padded union buffers.
    /// Empty when there are no element types.
    pub fn element_word_range(&self) -> Range<usize> {
        if !self.has_element() {
            return 0..0;
        }
        let start = self.registry().element_base() >> 7;
        start..start + (1usize << (self.m_elem() - 7))
    }

    /// The element region's frozen high WORD coordinates, LSB-first: the
    /// Boolean pattern the top `M − M_elem` word variables take for every
    /// address in the region (`element_base >> M_elem`). Element claim points
    /// are `region point ‖ these`.
    pub fn element_prefix_coords(&self) -> Vec<F128> {
        if !self.has_element() {
            return Vec::new();
        }
        let bits = self.m_total() - self.m_elem();
        let prefix = self.registry().element_base() >> self.m_elem();
        (0..bits)
            .map(|i| {
                if (prefix >> i) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            })
            .collect()
    }

    /// Per-element-slot geometry, in slot order: the slot's word offset
    /// RELATIVE to the element region, its `kappa`, and its declared count.
    /// The element PIOP addresses the region, not the union, so these are the
    /// offsets it uses.
    pub fn element_slot_layout(&self) -> Vec<ElementSlotLayout> {
        let base_word = self.registry().element_base() >> 7;
        let nb = self.num_boolean();
        self.registry().types()[nb..]
            .iter()
            .zip(&self.registry().slots()[nb..])
            .zip(&self.counts()[nb..])
            .map(|((ty, slot), &n_t)| ElementSlotLayout {
                region_word_offset: (slot.offset >> 7) - base_word,
                kappa: ty.k_log - 7,
                n_t,
            })
            .collect()
    }

    /// The count-derived run-list padding over the union BatchMajor buffer —
    /// delegates to [`Instance::padding_spec`]. Covers BOTH classes: it
    /// describes the padded buffer's zero structure for the transport
    /// (ring-switch folds, the virtual-opening sumcheck's live intervals),
    /// which spans the whole address space regardless of class.
    pub fn padding_spec(&self) -> PaddingSpec {
        self.instance.padding_spec()
    }

    /// The boolean region's own run-list — the padding spec the boolean
    /// zerocheck consumes over its `M_bool`-variable domain. Identical to
    /// [`Self::padding_spec`] for a boolean-only registry.
    pub fn boolean_padding_spec(&self) -> PaddingSpec {
        self.instance.boolean_padding_spec()
    }

    /// Per-chunk-column heights (in packed words) of the union jagged grid,
    /// for the opening path (`pcs::open_batch_merged`):
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
        // The instance floor ([`Self::set_dense_floor`]) composes with the
        // config floor the same way; its setter already asserted it fits the
        // padded domain, so the clamp below cannot silently shave it.
        let floor = floor.max(self.dense_floor_m.map_or(0, |m| 1usize << (m - 7)));
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

    /// The integer-lane count for this instance's commit, or `None` when the
    /// dense stack fills every lane (the power-of-two case — keep today's
    /// commit, byte-identically).
    ///
    /// [`Self::committed_words`] rounds the dense stack UP to a power of two,
    /// and that rounding tax is committed: the zero tail is RS-encoded and
    /// Merkle-hashed like real data (28% of the stack for a full-utilization
    /// SHA-256 + BLAKE3 mix at `M = 30`). Under the high-bit-lane labelling —
    /// lane `l` owns the contiguous block `q[l·2^log_dim .. (l+1)·2^log_dim)`
    /// — that tail is WHOLE zero lanes, so committing only the first
    /// `t = ceil(dense_words / 2^log_dim)` of them drops the encode + hash of
    /// the rest. See [`crate::pcs::commit_lane_major`] for the layout and
    /// `pcs::open_batch_merged` for how the opening follows (the
    /// relabelling is a rotation of the index variables, so the jagged
    /// weight/assist machinery is untouched — only its evaluation point
    /// rotates).
    pub fn commit_lanes(&self, log_batch_size: usize) -> Option<usize> {
        let log_dim = self.dense_m() - 7 - log_batch_size;
        let t = dense_lanes(self.dense_words(), log_batch_size, log_dim);
        (t < 1usize << log_batch_size).then_some(t)
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
    ///
    /// The gather is parallel over the used chunk-columns (disjoint
    /// destination runs of `n_t` words, disjoint sources), which is what the
    /// copy costs at scale: it is pure memory traffic, ~127 MB at `M = 30`.
    pub fn compact_witness(&self, z_padded: &[F128]) -> Vec<F128> {
        debug_assert!(
            self.dropped_words_are_zero(z_padded),
            "padded buffer must be zero on every dropped word \
             (dummy rows, useless columns, gaps)"
        );
        self.compact_witness_unchecked(z_padded)
    }

    /// [`Self::compact_witness`] without the dropped-words-are-zero debug
    /// assertion — for the `PooledDirty` witness mode, where dropped words
    /// are dirty by design and provably never read (the gather below reads
    /// declared rows only).
    pub fn compact_witness_unchecked(&self, z_padded: &[F128]) -> Vec<F128> {
        assert_eq!(z_padded.len(), self.packed_len(), "padded buffer length");
        let nu = self.n_log();
        // POOLED, not `vec![F128::ZERO; …]`. A fresh 134 MB zeroed Vec at
        // M = 30 costs ~3.0 ms to allocate and write versus ~0.6 ms for an
        // already-resident buffer — it pays a real memset AND a soft page
        // fault per page on first touch, which was ~2.4 ms of this gather's
        // ~3.1 ms. The loop below writes `[0, dense_words)` exactly, so only
        // the power-of-two pad tail needs zeroing.
        let dense = self.dense_words();
        let mut q = take_f128(self.committed_words());
        q[dense..]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
        // Hand each slot its (contiguous, disjoint) destination run, then
        // fill that run's per-column chunks in parallel.
        let mut rest: &mut [F128] = &mut q;
        let mut cursor = 0usize;
        for ((ty, slot), &n_t) in self
            .registry()
            .types()
            .iter()
            .zip(self.registry().slots())
            .zip(self.counts())
        {
            let used_cols = self.used_cols(ty);
            let (dst, tail) = rest.split_at_mut(used_cols * n_t);
            rest = tail;
            cursor += used_cols * n_t;
            if n_t == 0 {
                continue; // an empty slot contributes no chunk (and `par_chunks_mut(0)` panics)
            }
            let start = slot.offset >> 7;
            dst.par_chunks_mut(n_t).enumerate().for_each(|(c, out)| {
                let col = start + (c << nu);
                out.copy_from_slice(&z_padded[col..col + n_t]);
            });
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
    //
    // Since the element class, the boolean PIOP runs over the `M_bool`-variable
    // PREFIX subcube, so these speak `M_bool` on the input side and append the
    // `M − M_bool` frozen-ZERO high address coordinates on the output side —
    // the boolean region is `[0, 2^{M_bool})`, so those bits are 0 at every
    // boolean address. For a boolean-only registry `M_bool = M` and nothing is
    // appended, which is why the existing anchors are byte-identical.
    // -----------------------------------------------------------------------

    /// Lincheck's **semantic** quirky point from the boolean zerocheck claim:
    /// split the address-ordered `mlv` challenges (length `M_bool − 6`) into
    /// `x_inner_rest = [dim6, chunk…]` and `x_outer = batch`. Union analog of
    /// [`BlockR1cs::x_ab_from_mlv`] (BatchMajor). Stays in BOOLEAN-REGION
    /// coordinates — the boolean lincheck's column domain is `M_bool − nu`.
    pub fn x_ab_from_mlv(&self, z_skip: SkipPoint, mlv: &[F128]) -> QuirkyPoint {
        let nu = self.n_log();
        assert_eq!(mlv.len(), self.m_bool() - K_SKIP);
        let mut x_inner_rest = Vec::with_capacity(1 + self.boolean_col_log());
        x_inner_rest.push(mlv[0]);
        x_inner_rest.extend_from_slice(&mlv[1 + nu..]);
        QuirkyPoint {
            z_skip,
            x_inner_rest,
            x_outer: mlv[1..1 + nu].to_vec(),
        }
    }

    /// The `M − M_bool` frozen-zero high address coordinates that lift a
    /// boolean-region point into the union address space.
    fn boolean_frozen_high(&self) -> usize {
        self.m_total() - self.m_bool()
    }

    /// Address-ordered `ZClaim` point for the AB claim after lincheck
    /// replaces the inner coordinates with `(r_inner_skip, r_inner_rest)`.
    /// Union analog of [`BlockR1cs::ab_claim_point`] (BatchMajor): the
    /// address-ordered suffix is `[dim6 | batch | chunk | frozen zeros]`.
    pub fn ab_claim_point(
        &self,
        r_inner_skip: SkipPoint,
        r_inner_rest: &[F128],
        x_outer: &[F128],
    ) -> QuirkyPoint {
        assert_eq!(x_outer.len(), self.n_log());
        assert_eq!(r_inner_rest.len(), 1 + self.boolean_col_log());
        let frozen = self.boolean_frozen_high();
        let mut suffix = Vec::with_capacity(x_outer.len() + r_inner_rest.len() - 1 + frozen);
        suffix.extend_from_slice(x_outer);
        suffix.extend_from_slice(&r_inner_rest[1..]);
        suffix.extend(repeat_n(F128::ZERO, frozen));
        QuirkyPoint {
            z_skip: r_inner_skip,
            x_inner_rest: vec![r_inner_rest[0]],
            x_outer: suffix,
        }
    }

    /// Address-ordered `ZClaim` point for the C claim from the boolean
    /// zerocheck's `r_rest` (already address-ordered, length `M_bool − 6`),
    /// lifted with the frozen-zero high coordinates. Union analog of
    /// [`BlockR1cs::c_claim_point`] (BatchMajor).
    pub fn c_claim_point(&self, z_skip: SkipPoint, r_rest: &[F128]) -> QuirkyPoint {
        assert_eq!(r_rest.len(), self.m_bool() - K_SKIP);
        let frozen = self.boolean_frozen_high();
        let mut x_outer = Vec::with_capacity(r_rest.len() - 1 + frozen);
        x_outer.extend_from_slice(&r_rest[1..]);
        x_outer.extend(repeat_n(F128::ZERO, frozen));
        QuirkyPoint {
            z_skip,
            x_inner_rest: vec![r_rest[0]],
            x_outer,
        }
    }

    // -----------------------------------------------------------------------
    // Statement binding: the flock-mixed-v1 protocol binding, plus the M1/M2
    // single-type harness binding (differential tests only).
    // -----------------------------------------------------------------------

    /// The multi-table statement binding. Before any challenge, absorb in this
    /// order, the `flock-mixed-v1` domain label, the registry digest
    /// ([`Registry::digest`]), the counts vector (one u64 LE per type, in
    /// slot order, as a single byte string — its length is additionally
    /// bound through the digest's type count), and the commitment CAP. The
    /// counts are the only per-proof statement data; everything else is
    /// registry-static.
    ///
    /// Domain-separated from the single-table binding
    /// ([`crate::proof::bind_statement`]: `flock-r1cs-v0` + the `BlockR1cs`
    /// statement digest), so a mixed proof can never be replayed as a
    /// single-table proof or vice versa.
    pub fn bind_statement<Ch: Challenger>(&self, challenger: &mut Ch, commitment: &Commitment) {
        challenger.observe_label(b"flock-mixed-v1");
        challenger.observe_bytes(&self.registry().digest());
        let mut counts_le = Vec::with_capacity(8 * self.counts().len());
        for &n_t in self.counts() {
            counts_le.extend_from_slice(&(n_t as u64).to_le_bytes());
        }
        challenger.observe_bytes(&counts_le);
        challenger.observe_bytes(commitment.cap.as_flattened());
    }

    /// [`Self::bind_statement`] plus the CIRCUIT half of the statement: the
    /// circuit digest ([`crate::circuit::Circuit::digest`], which covers the
    /// gate counts, the IO schemas through the registry digest, the wiring and
    /// the public layout) and a 32-byte COMMITMENT to the public words
    /// ([`publics_digest`]) — v2 absorbs the digest, not the words, so a
    /// node's transcript (and hence its circuit shape) no longer scales with
    /// its public segment. A recursion parent re-derives the digest
    /// in-circuit from witness wires and connects it to the absorbed words;
    /// the publics' length is fixed by the circuit digest (the shape's
    /// public layout), so the digest binds content alone.
    ///
    /// Absorbed before any challenge, and in particular before the wiring
    /// GKR — which squeezes `α, β` at entry, so the multiset statement must
    /// already be fixed before `circuit::prove_wiring` starts.
    ///
    /// **Append-only:** the existing four observations are unchanged and the
    /// circuit payload follows under its own versioned label, so a non-circuit
    /// proof's transcript is byte-identical to today's.
    pub fn bind_statement_circuit<Ch: Challenger>(
        &self,
        challenger: &mut Ch,
        commitment: &Commitment,
        circuit_digest: &[u8; 32],
        public: &[F128],
    ) {
        self.bind_statement(challenger, commitment);
        challenger.observe_label(b"flock-circuit-stmt-v2");
        challenger.observe_bytes(circuit_digest);
        challenger.observe_bytes(&publics_digest(public));
    }

    // -----------------------------------------------------------------------
    // In-place witness generation (no scatter). A slot's BatchMajor word
    // index `(c << nu) + row` plus `o_t >> 7` IS its union word index — the
    // uniform-capacity convention makes `nu` the slot's own `n_log`, so a
    // slot's local layout is literally a contiguous, aligned sub-block of
    // the union buffer (which is why `scatter_witnesses`' full-utilization
    // path is a single memcpy). Handing each witness driver that sub-block
    // as its destination therefore produces the SAME padded buffer with no
    // copy at all — 3 × 134 MB of memory traffic saved at `M = 30`.
    // -----------------------------------------------------------------------

    /// Word range of slot `t`'s aligned block in the padded union buffers.
    pub fn slot_word_range(&self, t: usize) -> Range<usize> {
        let slot = &self.registry().slots()[t];
        let start = slot.offset >> 7;
        start..start + (1usize << (slot.m_slot - 7))
    }

    /// The three padded union witness buffers, ready for in-place generation:
    /// pooled (already-resident) allocations whose inter-slot and trailing
    /// GAPS are zeroed. The slot blocks are left dirty on purpose — every
    /// witness driver fully writes its block (the producers cover the useful
    /// chunk-column prefix, the useless-column tail and any dummy rows are
    /// written as zeros), so after [`Self::slot_dests`] + generation the
    /// buffers hold exactly what [`Self::assemble_witness`] would have
    /// scattered.
    /// `padding_unread`: the caller certifies that every downstream
    /// consumer of these buffers is support-gated (the merged pipeline —
    /// zerocheck run-lists, count-proportional lincheck, declared-only
    /// compaction, precomputed-`s_hat_v` ring switch), so padding may stay
    /// dirty — pooled resident buffers with NO zeroing at any utilization.
    /// Otherwise: pooled + zeroed at high utilization; all-zero from the
    /// ZERO pool (or fresh lazy-zero on a pool miss) when padding dominates
    /// (capacity ≥ 2× the declared dense area).
    pub fn take_witness_buffers(
        &self,
        padding_unread: bool,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, WitnessBufMode) {
        let len = self.packed_len();
        if padding_unread {
            return (
                take_f128(len),
                take_f128(len),
                take_f128(len),
                WitnessBufMode::PooledDirty,
            );
        }
        if self.dense_words() * 2 <= len {
            return (
                take_zeroed_f128(len),
                take_zeroed_f128(len),
                take_zeroed_f128(len),
                WitnessBufMode::FreshZeroed,
            );
        }
        let mut bufs = [take_f128(len), take_f128(len), take_f128(len)];
        for buf in &mut bufs {
            self.zero_gaps(buf);
        }
        let [z, a, b] = bufs;
        (z, a, b, WitnessBufMode::PooledZeroed)
    }

    /// Return a [`WitnessBufMode::FreshZeroed`] witness buffer to the zero
    /// pool, re-zeroing only the slot block areas. That dirty accounting is
    /// STRUCTURAL, not promised: [`Self::slot_dests`] carves exactly the slot
    /// blocks out of the buffer (borrow-checked slices), so no driver can
    /// have written a gap or the trailing tail — those stay zero, and their
    /// lazy pages stay untouched. Prebuilt sources that write whole blocks
    /// (padding zeros included) are covered by the same ranges; tightening to
    /// live words only is the recorded v2 (needs the prebuilt copy sources
    /// switched to in-place first).
    pub fn give_back_witness_buffer(&self, buf: Vec<F128>) {
        debug_assert_eq!(buf.len(), self.packed_len(), "padded buffer length");
        let dirty: Vec<Range<usize>> = (0..self.registry().num_types())
            .map(|t| self.slot_word_range(t))
            .collect();
        give_zeroed_f128(buf, &dirty);
    }

    /// A fully zeroed padded union buffer from the scratch pool — resident
    /// pages, parallel memset, no allocation tax.
    fn take_zeroed_buffer(&self) -> Vec<F128> {
        let mut buf = take_f128(self.packed_len());
        buf.par_chunks_mut(1 << 16).for_each(|c| c.fill(F128::ZERO));
        buf
    }

    /// Zero the words of a padded union buffer that no slot owns (the
    /// inter-slot and trailing gaps) — everything else is written by the
    /// drivers. A registry whose slots tile the address space exactly has
    /// no gaps and this does nothing.
    fn zero_gaps(&self, buf: &mut [F128]) {
        debug_assert_eq!(buf.len(), self.packed_len());
        let mut cursor = 0usize;
        for t in 0..self.registry().num_types() {
            let range = self.slot_word_range(t);
            buf[cursor..range.start]
                .par_chunks_mut(1 << 16)
                .for_each(|c| c.fill(F128::ZERO));
            cursor = range.end;
        }
        buf[cursor..]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
    }

    /// Split the three padded union buffers into one [`SlotWitnessDest`] per
    /// slot, in slot order — the destinations the witness drivers write.
    /// Slot blocks are disjoint and offset-ascending, so the views carve the
    /// buffers without overlapping; the gaps between them are skipped (left
    /// as [`Self::take_witness_buffers`] zeroed them).
    pub fn slot_dests<'d>(
        &self,
        z: &'d mut [F128],
        a: &'d mut [F128],
        b: &'d mut [F128],
        elide_padding_writes: bool,
    ) -> Vec<SlotWitnessDest<'d>> {
        for buf in [&*z, &*a, &*b] {
            assert_eq!(buf.len(), self.packed_len(), "padded buffer length");
        }
        /// Carve `words` off `rest` after skipping `skip`, keeping the
        /// caller's lifetime (`mem::take` hands the borrow over wholesale).
        fn carve<'d>(rest: &mut &'d mut [F128], skip: usize, words: usize) -> &'d mut [F128] {
            let (head, tail) = take(rest).split_at_mut(skip + words);
            *rest = tail;
            &mut head[skip..]
        }
        let (mut zr, mut ar, mut br) = (z, a, b);
        let mut cursor = 0usize;
        let mut dests = Vec::with_capacity(self.registry().num_types());
        for t in 0..self.registry().num_types() {
            let range = self.slot_word_range(t);
            let (skip, words) = (range.start - cursor, range.end - range.start);
            dests.push(SlotWitnessDest {
                z: carve(&mut zr, skip, words),
                a: carve(&mut ar, skip, words),
                b: carve(&mut br, skip, words),
                elide_padding_writes,
            });
            cursor = range.end;
        }
        dests
    }

    /// Assemble the union witness from per-slot packed buffers: place each
    /// slot's `(z, a, b)` at its aligned word offset `o_t >> 7` in
    /// union-sized buffers (dummy regions and the gap stay zero). One bundle
    /// per registry type, in slot order.
    ///
    /// Prefer [`Self::take_witness_buffers`] + [`Self::slot_dests`] on the
    /// hot path: generating in place skips this copy entirely.
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
    /// witness contract (see `prove_fast_ligerito_union`) makes the
    /// slot buffers zero everywhere else (dummy rows, useless columns), so
    /// the zero-initialized union buffers already hold those words'
    /// values and the result is byte-identical to the full-slot copy.
    /// Full-utilization slots keep the one-memcpy path.
    ///
    /// The three buffers are scattered concurrently and each slot's copy is
    /// itself chunk-parallel — like [`Self::compact_witness`] this is pure
    /// memory traffic (3 × 134 MB at `M = 30`), so it scales with cores
    /// until it hits memory bandwidth.
    fn scatter_witnesses(
        &self,
        slot_witnesses: Vec<SlotWitness>,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
        let nu = self.n_log();
        // Pooled + parallel zero-fill rather than `vec![F128::ZERO; len]` —
        // see `compact_witness` for the measured allocation tax. Partial-
        // utilization slots only copy their declared prefixes, so unlike
        // `take_witness_buffers` these must be zeroed in full, not just on
        // the gaps.
        let (mut z, mut a, mut b) = (
            self.take_zeroed_buffer(),
            self.take_zeroed_buffer(),
            self.take_zeroed_buffer(),
        );
        let ws = &slot_witnesses;
        // The witness contract, checked once per slot for all three buffers
        // (the per-buffer scatters below rely on it for their partial copies).
        debug_assert!(
            self.registry()
                .types()
                .iter()
                .zip(ws)
                .zip(self.counts())
                .all(|((ty, w), &n_t)| n_t == 1usize << nu
                    || slot_buffer_zero_off_support(w, self.used_cols(ty), nu, n_t)),
            "slot buffers must be zero on dummy rows and useless columns \
             (the union witness contract)"
        );
        join(
            || self.scatter_one(&mut z, ws, |w| &w.z_packed),
            || {
                join(
                    || self.scatter_one(&mut a, ws, |w| &w.a_packed),
                    || self.scatter_one(&mut b, ws, |w| &w.b_packed),
                )
            },
        );
        (z, a, b)
    }

    /// One buffer's worth of [`Self::scatter_witnesses`]: place every slot's
    /// `pick`ed source at its aligned word offset in `dst`.
    fn scatter_one(
        &self,
        dst: &mut [F128],
        slot_witnesses: &[SlotWitness],
        pick: impl Fn(&SlotWitness) -> &[F128] + Sync,
    ) {
        let nu = self.n_log();
        for (((ty, slot), w), &n_t) in self
            .registry()
            .types()
            .iter()
            .zip(self.registry().slots())
            .zip(slot_witnesses)
            .zip(self.counts())
        {
            let start = slot.offset >> 7;
            let words = 1usize << (slot.m_slot - 7);
            let src = pick(w);
            let out = &mut dst[start..start + words];
            if n_t == 1usize << nu {
                // Whole-slot memcpy, split into cache-sized chunks.
                const CHUNK: usize = 1 << 12; // 64 KiB of F128
                out.par_chunks_mut(CHUNK)
                    .zip(src.par_chunks(CHUNK))
                    .for_each(|(o, s)| o.copy_from_slice(s));
            } else {
                if n_t == 0 {
                    continue;
                }
                // One chunk per chunk-column; only the declared `n_t`-row
                // prefix is copied (the rest is zero on both sides).
                out.par_chunks_mut(1usize << nu)
                    .take(self.used_cols(ty))
                    .enumerate()
                    .for_each(|(c, o)| {
                        let col = c << nu;
                        o[..n_t].copy_from_slice(&src[col..col + n_t]);
                    });
            }
        }
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

/// One element slot's geometry inside the element REGION (not the union):
/// where its aligned block starts in region-word coordinates, how wide its
/// rows are, and how many are declared. Produced by
/// [`UnionInstance::element_slot_layout`]; the element PIOP addresses the
/// region, so these are the only offsets it needs.
///
/// The region-word index of (column `c`, row `j`) in this slot is
/// `region_word_offset + (c << nu) + j`, and — since `region_word_offset` is a
/// multiple of `2^{nu + kappa}` — that is also
/// `(q << (nu + kappa)) | (c << nu) | j` for the slot's region prefix
/// `q = region_word_offset >> (nu + kappa)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementSlotLayout {
    pub region_word_offset: usize,
    pub kappa: usize,
    pub n_t: usize,
}

impl ElementSlotLayout {
    /// The slot's region prefix `q` — the value the region's top
    /// `(M_elem − 7) − nu − kappa` word coordinates are frozen to inside this
    /// slot.
    pub fn region_prefix(&self, nu: usize) -> usize {
        self.region_word_offset >> (nu + self.kappa)
    }

    /// The slot's column-domain block in the region column domain (the region
    /// word index with the `nu` row bits dropped): `[q << kappa, (q+1) << kappa)`.
    pub fn column_offset(&self, nu: usize) -> usize {
        self.region_word_offset >> nu
    }
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

/// How [`UnionInstance::take_witness_buffers`] sourced the padded buffers — it
/// decides whether the drivers may elide zero-valued writes and whether
/// the buffers belong back in the scratch pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WitnessBufMode {
    /// Pooled buffers, gaps zeroed here, drivers zero their padding —
    /// the classic contract: every dropped word IS zero.
    PooledZeroed,
    /// All-zero buffers without a memset: the ZERO pool when a same-length
    /// buffer is pooled ([`crate::scratch::take_zeroed_f128`]), else fresh
    /// `alloc_zeroed` lazy zero pages. Drivers elide all zero-valued
    /// writes. Returned via [`UnionInstance::give_back_witness_buffer`],
    /// which re-zeros the slot block areas — NOT to the dirty scratch pool
    /// (a fresh multi-GiB `alloc_zeroed` per prove is nearly free early in
    /// a process but memsets recycled arena memory for real once the
    /// process has churned; a long-running prover is exactly that shape).
    FreshZeroed,
    /// Pooled buffers, nothing zeroed: the caller guarantees every
    /// consumer is support-gated (dummy rows, gaps, and padding are
    /// NEVER read), so their contents are unobservable. Drivers elide
    /// all zero-valued writes; buffers return to the pool resident.
    PooledDirty,
}

/// The 32-byte commitment to a circuit's public segment that
/// [`UnionInstance::bind_statement_circuit`] absorbs (v2): the words'
/// 16-byte-LE byte string, split into 1 KiB chunks, each hashed as a
/// BLAKE3 chunk leaf ([`crate::merkle::hash_leaf`] — one chunk, counter 0)
/// and LEFT-FOLDED through [`crate::merkle::hash_pair`]:
/// `cv = h(...h(h(L0, L1), L2)..., Ln)`. A chain, not a tree — nothing is
/// ever opened, both sides recompute in full — and both pieces are exactly
/// the compressions a circuit's BLAKE3 gate rows replay (the collapsed
/// pins), so a recursion parent re-derives the digest from witness wires
/// with no new gate types. The last chunk hashes at its TRUE byte length
/// (publics are whole words, so a partial BLOCK at most — the gate's free
/// `b` input); the length itself is bound by the circuit digest's public
/// layout, absorbed alongside.
pub fn publics_digest(public: &[F128]) -> [u8; 32] {
    // An empty segment digests to the zero string: hash_leaf rejects empty
    // input (a BLAKE3 empty message must be root-flagged), the length is
    // statement-bound through the circuit digest, and equating it with a
    // nonempty segment's digest would need a BLAKE3 preimage of zero.
    if public.is_empty() {
        return [0u8; 32];
    }
    let mut bytes = Vec::with_capacity(16 * public.len());
    for w in public {
        bytes.extend_from_slice(&w.lo.to_le_bytes());
        bytes.extend_from_slice(&w.hi.to_le_bytes());
    }
    let mut chunks = bytes.chunks(1024);
    let mut cv = hash_leaf(chunks.next().expect("nonempty"), HashKind::Blake3);
    for chunk in chunks {
        cv = hash_pair(&cv, &hash_leaf(chunk, HashKind::Blake3), HashKind::Blake3);
    }
    cv
}

/// A per-slot destination view into the padded union witness buffers: the
/// slot's aligned `2^{m_t−7}`-word block of each of `z`, `a`, `b`, handed
/// out by [`UnionInstance::slot_dests`]. A witness driver writes these in
/// place instead of allocating its own [`SlotWitness`] buffers, which makes
/// the union assembly copy-free (see the module comment there).
///
/// **Contract:** the driver must write EVERY word of all three views —
/// declared rows from the producers, dummy rows and useless chunk-columns as
/// zeros. The views come from the recycled scratch pool and start out
/// holding stale data, exactly as the drivers' own pooled buffers do.
pub struct SlotWitnessDest<'d> {
    pub z: &'d mut [F128],
    pub a: &'d mut [F128],
    pub b: &'d mut [F128],
    /// The driver may ELIDE every zero-valued write (dummy groups, the
    /// padding suffix, dummy stripe regions): the destination is either
    /// already zero (`FreshZeroed`) or its dropped words are never read
    /// (`PooledDirty`). See [`UnionInstance::take_witness_buffers`].
    pub elide_padding_writes: bool,
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, OnceLock};

    use crate::{
        challenger::FsChallenger,
        element_r1cs::ElementTableBuilder,
        pcs::{PcsParams, jagged::JaggedParams},
        r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
        test_rng::Rng,
        union::{
            Challenger, Commitment, ElementSlotLayout, F128, K_SKIP, MIN_DENSE_M, Registry,
            SkipPoint, SlotWitness, TableClass, TableType, UnionInstance,
        },
        zerocheck::PaddingRun,
    };
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
            class: TableClass::Boolean,
            io_schema: Vec::new(),
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
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
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
            let z_skip = rng.f128();
            let mlv = rng.f128_vec(m - K_SKIP);
            let x_ab_union = union.x_ab_from_mlv(SkipPoint::Phi8(z_skip), &mlv);
            let x_ab_r1cs = r1cs.x_ab_from_mlv(SkipPoint::Phi8(z_skip), &mlv);
            assert_eq!(x_ab_union, x_ab_r1cs, "x_ab_from_mlv diverged");

            let r_inner_skip = rng.f128();
            let r_inner_rest = rng.f128_vec(k_log - K_SKIP);
            assert_eq!(
                union.ab_claim_point(
                    SkipPoint::Phi8(r_inner_skip),
                    &r_inner_rest,
                    &x_ab_union.x_outer
                ),
                r1cs.ab_claim_point(
                    SkipPoint::Phi8(r_inner_skip),
                    &r_inner_rest,
                    &x_ab_r1cs.x_outer
                ),
                "ab_claim_point diverged"
            );

            let r_rest = rng.f128_vec(m - K_SKIP);
            assert_eq!(
                union.c_claim_point(SkipPoint::Phi8(z_skip), &r_rest),
                r1cs.c_claim_point(SkipPoint::Phi8(z_skip), &r_rest),
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
        let params =
            JaggedParams::from_heights(&union.jagged_heights(), union.n_log(), union.dense_m() - 7);
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

    /// In-place generation reproduces the scatter EXACTLY: gaps zeroed, each
    /// slot's destination view carved at its aligned block. Buffers start
    /// POISONED (as the recycled scratch pool hands them out), so any word
    /// neither zeroed as a gap nor written by a "driver" would show up.
    #[test]
    fn slot_dests_reproduce_the_scatter() {
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let union = UnionInstance::new(&reg, vec![8, 8]);
        assert_eq!(union.packed_len(), 128);
        // Slot A: words 0..64; slot B: 64..96; gap: 96..128.
        assert_eq!(union.slot_word_range(0), 0..64);
        assert_eq!(union.slot_word_range(1), 64..96);

        let mut rng = Rng::new(0x51_07_DE_57);
        let slot = |n: usize, rng: &mut Rng| SlotWitness {
            z_packed: rng.f128_vec(n),
            a_packed: rng.f128_vec(n),
            b_packed: rng.f128_vec(n),
        };
        let (slot_a, slot_b) = (slot(64, &mut rng), slot(32, &mut rng));

        let (rz, ra, rb) = union.assemble_witness(vec![slot_a.clone(), slot_b.clone()]);

        let poison = F128 {
            lo: 0xDEAD_BEEF_DEAD_BEEF,
            hi: 0xDEAD_BEEF_DEAD_BEEF,
        };
        let (mut z, mut a, mut b) = (vec![poison; 128], vec![poison; 128], vec![poison; 128]);
        for buf in [&mut z, &mut a, &mut b] {
            union.zero_gaps(buf);
        }
        for (d, w) in union
            .slot_dests(&mut z, &mut a, &mut b, false)
            .into_iter()
            .zip([&slot_a, &slot_b])
        {
            // What an in-place driver does: fully write its own block.
            d.z.copy_from_slice(&w.z_packed);
            d.a.copy_from_slice(&w.a_packed);
            d.b.copy_from_slice(&w.b_packed);
        }
        assert_eq!((z, a, b), (rz, ra, rb), "in-place must match the scatter");
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

    /// The `flock-mixed-v1` binding is deterministic and sensitive to every
    /// bound component — registry digest, counts (value AND slot order), and
    /// commitment cap: divergence anywhere yields a different first
    /// challenge, which is what makes the statement non-substitutable.
    #[test]
    fn bind_statement_sensitivity() {
        let commitment = |root_byte: u8| Commitment {
            cap: vec![[root_byte; 32]],
            params: PcsParams {
                m: 14,
                log_inv_rate: 1,
                log_batch_size: 6,
                profile: Default::default(),
                num_lanes: None,
                merkle_hash: Default::default(),
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
            "commitment cap must bind"
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

    // ---- element slots through the union bookkeeping -----------------------

    /// An element type of shape `(kappa, k)`: `k` free-wire columns, the rest
    /// self-pinned zero padding. Only the shape matters here.
    fn elem_ty(kappa: usize, k: usize) -> TableType {
        let mut b = ElementTableBuilder::new(kappa);
        for y in 0..k {
            b.free_wire(y);
        }
        TableType::element(Arc::new(b.build().expect("free wires are valid")))
    }

    /// A boolean-only registry's `boolean_padding_spec` IS its `padding_spec`
    /// (the class gap and the element runs are exactly what it drops), and the
    /// boolean claim-point helpers append nothing — the byte-identity
    /// precondition for every existing anchor.
    #[test]
    fn boolean_only_regions_are_the_whole_space() {
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let union = UnionInstance::new(&reg, vec![5, 3]);
        assert_eq!(union.m_bool(), union.m_total());
        assert!(!union.has_element());
        assert_eq!(union.num_element(), 0);
        assert_eq!(union.boolean_packed_len(), union.packed_len());
        assert_eq!(union.boolean_col_log(), union.col_log());
        assert_eq!(union.boolean_padding_spec(), union.padding_spec());
        assert_eq!(union.element_word_range(), 0..0);
        assert!(union.element_prefix_coords().is_empty());
        assert!(union.element_slot_layout().is_empty());
    }

    /// Mirror of the above for an ELEMENT-ONLY registry: `M_bool = 0`, so the
    /// boolean-region accessors must return 0 rather than underflow on
    /// `M_bool − 7` (a release build masks the shift and hides it; debug
    /// panics). No boolean PIOP runs, so 0 is also the honest answer.
    #[test]
    fn element_only_boolean_accessors_do_not_underflow() {
        let reg = Registry::new(vec![elem_ty(3, 5), elem_ty(2, 3)], 4);
        let union = UnionInstance::new(&reg, vec![7, 5]);
        assert_eq!(union.num_boolean(), 0);
        assert_eq!(union.m_bool(), 0);
        assert_eq!(union.boolean_packed_len(), 0);
        assert_eq!(union.boolean_col_log(), 0);
        assert!(union.boolean_padding_spec().runs().is_empty());
        assert!(union.has_element());
    }

    /// Mixed registry: heights, dense words, region ranges and the padding
    /// run-list, hand-computed. Element slots need no special case — their
    /// `k_log = kappa + 7` makes `used_cols` count element columns.
    #[test]
    fn mixed_class_bookkeeping_hand_computed() {
        // Boolean: k_log 10 (8 word-cols, 6 used) and 9 (4 word-cols, 3 used).
        // Element: kappa 3, k 5 → k_log 10, 8 word-cols, 5 used.
        // nu = 3 → boolean areas 2^13 + 2^12 = 0x3000, M_bool = 14;
        // element area 2^13, M_elem = 13, base 2^14, M = 15.
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300), elem_ty(3, 5)], 3);
        let union = UnionInstance::new(&reg, vec![5, 3, 7]);
        assert_eq!(
            (union.m_total(), union.m_bool(), union.m_elem()),
            (15, 14, 13)
        );
        assert_eq!(union.col_log(), 15 - 7 - 3); // 32 union word-columns
        assert_eq!(union.boolean_col_log(), 14 - 7 - 3); // 16 boolean columns
        assert_eq!(union.packed_len(), 1 << 8);
        assert_eq!(union.boolean_packed_len(), 1 << 7);

        // Element region: [2^14, 2^14 + 2^13) in bits = [128, 192) in words.
        assert_eq!(union.element_word_range(), 128..192);
        // Its frozen high WORD coords: M − M_elem = 2 bits of value
        // element_base >> M_elem = 2^14 >> 13 = 0b10.
        assert_eq!(
            union.element_prefix_coords(),
            vec![F128::ZERO, F128::ONE],
            "prefix 0b10, LSB-first"
        );
        // One element slot filling the region, at region offset 0.
        assert_eq!(
            union.element_slot_layout(),
            vec![ElementSlotLayout {
                region_word_offset: 0,
                kappa: 3,
                n_t: 7
            }]
        );

        // Heights: 32 columns. Slot A cols 0..8 (6 used @5), slot B cols 8..12
        // (3 used @3), gap 12..16, element cols 16..24 (5 used @7), 24..32 gap.
        #[rustfmt::skip]
        let expected: Vec<u64> = vec![
            5, 5, 5, 5, 5, 5, 0, 0,
            3, 3, 3, 0,
            0, 0, 0, 0,             // boolean subcube tail (the class gap)
            7, 7, 7, 7, 7, 0, 0, 0, // element slot: 5 of 8 columns used
            0, 0, 0, 0, 0, 0, 0, 0, // element region tail
        ];
        assert_eq!(union.jagged_heights(), expected);
        assert_eq!(union.dense_words(), 6 * 5 + 3 * 3 + 5 * 7);
        assert_eq!(
            union.jagged_heights().iter().sum::<u64>(),
            union.dense_words() as u64,
            "unrank ≡ compaction map"
        );
        assert!(
            !union.compaction_is_identity(),
            "a class gap breaks identity"
        );

        // The padding run-list: an explicit zero run for the class gap, and
        // the element slot's two runs exactly like a boolean slot's.
        let runs = union.padding_spec();
        let cols = |n_blocks, useful| PaddingRun {
            k_log: 7 + 3,
            useful_bits_per_block: useful,
            n_blocks,
        };
        assert_eq!(
            runs.runs(),
            &[
                cols(6, 5 << 7),
                cols(2, 0),
                cols(3, 3 << 7),
                cols(1, 0),
                cols(4, 0),      // class gap [0x3000, 0x4000)
                cols(5, 7 << 7), // element: 5 used columns at n = 7
                cols(3, 0),
            ]
        );
        assert_eq!(runs.covered_bits(), (1 << 14) + (1 << 13));

        // The boolean spec stops at the boolean slots — no class gap, no
        // element runs — and covers the boolean extent only.
        let bool_runs = union.boolean_padding_spec();
        assert_eq!(
            bool_runs.runs(),
            &[cols(6, 5 << 7), cols(2, 0), cols(3, 3 << 7), cols(1, 0)]
        );
        assert_eq!(bool_runs.covered_bits(), (1 << 13) + (1 << 12));
        // Every useful interval of the boolean spec lies in the boolean region.
        for (_, e) in bool_runs.useful_intervals() {
            assert!(e <= 1usize << union.m_bool());
        }
        // …and the full spec's element intervals lie in the element region.
        let elem_lo = 1usize << 14;
        assert!(
            runs.useful_intervals()
                .iter()
                .all(|&(s, e)| e <= 1usize << 14 || s >= elem_lo)
        );
    }

    /// Two element slots of different widths pack area-descending inside the
    /// region, each at its own aligned offset, and their region prefixes are
    /// the subcube patterns the element lincheck freezes.
    #[test]
    fn two_element_slots_share_the_region() {
        // Element kappa 3 (k_log 10, area 2^13) + kappa 2 (k_log 9, area 2^12)
        // at nu = 3, no boolean types: M_elem = 14, base 0, M = 14.
        let reg = Registry::new(vec![elem_ty(2, 3), elem_ty(3, 6)], 3);
        let union = UnionInstance::new(&reg, vec![4, 6]);
        assert_eq!(
            (union.m_total(), union.m_bool(), union.m_elem()),
            (14, 0, 14)
        );
        assert_eq!(union.element_word_range(), 0..(1 << 7));
        assert!(
            union.element_prefix_coords().is_empty(),
            "the region IS the address space — nothing to freeze"
        );
        // Class-major + area-descending: kappa 3 first.
        let layout = union.element_slot_layout();
        assert_eq!(
            layout,
            vec![
                ElementSlotLayout {
                    region_word_offset: 0,
                    kappa: 3,
                    n_t: 4
                },
                ElementSlotLayout {
                    region_word_offset: 64,
                    kappa: 2,
                    n_t: 6
                },
            ]
        );
        let nu = union.n_log();
        // Region prefixes: slot 0 spans [0, 2^6) words (nu+kappa = 6) → q = 0;
        // slot 1 spans [2^6, 2^6 + 2^5) → q = 2 over its own 2 prefix bits.
        assert_eq!(layout[0].region_prefix(nu), 0);
        assert_eq!(layout[1].region_prefix(nu), 2);
        assert_eq!(layout[0].column_offset(nu), 0);
        assert_eq!(layout[1].column_offset(nu), 8);
        // Counts are in SLOT order: slot 0 (kappa 3, 6 used cols) at n = 4,
        // slot 1 (kappa 2, 3 used cols) at n = 6.
        #[rustfmt::skip]
        assert_eq!(union.jagged_heights(), vec![
            4, 4, 4, 4, 4, 4, 0, 0,
            6, 6, 6, 0,
            0, 0, 0, 0,
        ]);
        assert_eq!(union.dense_words(), 6 * 4 + 3 * 6);
        // Boolean side is empty: no runs, no columns.
        assert!(union.boolean_padding_spec().runs().is_empty());
    }

    /// `compact_witness` over a mixed registry: element words compact exactly
    /// like boolean ones (the commitment does not care what a word means), and
    /// the class gap vanishes along with the dummy rows and useless columns.
    #[test]
    fn compact_witness_carries_element_words() {
        let reg = Registry::new(vec![ty(10, 512), elem_ty(3, 5)], 3);
        let union = UnionInstance::new(&reg, vec![5, 7]);
        // Boolean: 8 cols, 4 used; element: 8 cols, 5 used.
        let mut z = vec![F128::ZERO; union.packed_len()];
        let base = union.element_word_range().start;
        for c in 0..4usize {
            for i in 0..5usize {
                z[(c << 3) + i] = F128 {
                    lo: i as u64,
                    hi: c as u64,
                };
            }
        }
        for c in 0..5usize {
            for i in 0..7usize {
                z[base + (c << 3) + i] = F128 {
                    lo: i as u64,
                    hi: 0x100 + c as u64,
                };
            }
        }
        let q = union.compact_witness(&z);
        assert_eq!(union.dense_words(), 4 * 5 + 5 * 7);
        let mut cursor = 0usize;
        for (tag, cols, n_t) in [(0u64, 4usize, 5usize), (0x100, 5, 7)] {
            for c in 0..cols {
                for i in 0..n_t {
                    assert_eq!(
                        q[cursor],
                        F128 {
                            lo: i as u64,
                            hi: tag + c as u64
                        },
                        "tag {tag:#x} col {c} word {i}"
                    );
                    cursor += 1;
                }
            }
        }
        assert_eq!(cursor, union.dense_words());
        assert!(q[cursor..].iter().all(|w| *w == F128::ZERO), "pad tail");

        // unrank ≡ compaction across the class boundary too.
        let params =
            JaggedParams::from_heights(&union.jagged_heights(), union.n_log(), union.dense_m() - 7);
        for e in 0..union.dense_words() as u64 {
            let (row, col) = params.unrank(e);
            assert_eq!(q[e as usize], z[(col << 3) + row], "unrank at {e}");
        }
    }

    /// `slot_dests` carves the element slot's block too, and `zero_gaps`
    /// zeroes the class gap — so an element driver writing its own block in
    /// place leaves the region's gaps honestly zero, which the element
    /// zerocheck relies on (it sums over the whole region).
    #[test]
    fn slot_dests_cover_the_element_slot_and_zero_the_class_gap() {
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300), elem_ty(3, 5)], 3);
        let union = UnionInstance::new(&reg, vec![8, 8, 8]);
        assert_eq!(union.packed_len(), 1 << 8);
        assert_eq!(union.slot_word_range(0), 0..64);
        assert_eq!(union.slot_word_range(1), 64..96);
        assert_eq!(union.slot_word_range(2), 128..192);

        let poison = F128 { lo: !0, hi: !0 };
        let (mut z, mut a, mut b) = (
            vec![poison; 1 << 8],
            vec![poison; 1 << 8],
            vec![poison; 1 << 8],
        );
        for buf in [&mut z, &mut a, &mut b] {
            union.zero_gaps(buf);
        }
        // Gaps: [96, 128) (class gap) and [192, 256) (region tail).
        for buf in [&z, &a, &b] {
            assert!(buf[96..128].iter().all(|w| *w == F128::ZERO), "class gap");
            assert!(buf[192..].iter().all(|w| *w == F128::ZERO), "region tail");
        }
        let dests = union.slot_dests(&mut z, &mut a, &mut b, false);
        assert_eq!(dests.len(), 3);
        assert_eq!(dests[2].z.len(), 64, "element slot block is 2^(nu+kappa)");
        // A driver writing every word of its block leaves no poison behind.
        for d in dests {
            d.z.fill(F128::ZERO);
            d.a.fill(F128::ZERO);
            d.b.fill(F128::ZERO);
        }
        for buf in [&z, &a, &b] {
            assert!(buf.iter().all(|w| *w == F128::ZERO));
        }
    }

    /// Boolean claim points gain exactly `M − M_bool` frozen-ZERO high
    /// coordinates and are otherwise the boolean-region formulas — so the
    /// resulting union point evaluates the witness on the boolean region only.
    #[test]
    fn boolean_claim_points_freeze_the_high_coords() {
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300), elem_ty(3, 5)], 3);
        let union = UnionInstance::new(&reg, vec![5, 3, 7]);
        let (m, m_bool) = (union.m_total(), union.m_bool());
        assert_eq!((m, m_bool), (15, 14));
        let frozen = m - m_bool;
        let mut rng = Rng::new(0xE1E_C7);

        let mlv = rng.f128_vec(m_bool - K_SKIP);
        let x_ab = union.x_ab_from_mlv(SkipPoint::Phi8(rng.f128()), &mlv);
        assert_eq!(x_ab.x_outer.len(), union.n_log());
        assert_eq!(
            x_ab.x_inner_rest.len(),
            1 + union.boolean_col_log(),
            "the semantic point stays in boolean-region coordinates"
        );

        let r_inner_rest = rng.f128_vec(1 + union.boolean_col_log());
        let ab = union.ab_claim_point(SkipPoint::Phi8(rng.f128()), &r_inner_rest, &x_ab.x_outer);
        let full = ab.x_inner_rest.len() + ab.x_outer.len();
        assert_eq!(full, m - K_SKIP, "the point must address the UNION space");
        assert!(
            ab.x_outer[ab.x_outer.len() - frozen..]
                .iter()
                .all(|c| *c == F128::ZERO),
            "high coords frozen to zero"
        );
        assert_eq!(
            &ab.x_outer[..ab.x_outer.len() - frozen],
            &[x_ab.x_outer.as_slice(), &r_inner_rest[1..]].concat()[..],
            "the boolean part is unchanged"
        );

        let r_rest = rng.f128_vec(m_bool - K_SKIP);
        let c = union.c_claim_point(SkipPoint::Phi8(rng.f128()), &r_rest);
        assert_eq!(c.x_inner_rest.len() + c.x_outer.len(), m - K_SKIP);
        assert_eq!(&c.x_outer[..r_rest.len() - 1], &r_rest[1..]);
        assert!(
            c.x_outer[r_rest.len() - 1..]
                .iter()
                .all(|x| *x == F128::ZERO)
        );
    }
}
