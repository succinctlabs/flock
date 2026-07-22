//! Multi-table registry and slot schedule — Phase 0 scaffolding.
//!
//! Pure data types for the multi-table design (`docs/multi-table-design.tex`):
//! a [`Registry`] fixes, once, an ordered list of [`TableType`]s (one per hash
//! type) and ONE uniform row capacity `2^nu`; from those it derives the static
//! slot layout — each type's capacity-sized slot placed in one address space
//! of `2^M` points, aligned so every slot is a subcube selected by freezing a
//! prefix of the top address bits (design doc, Lemma "Alignment"). An
//! [`Instance`] adds the per-proof declared counts `n_t` and derives the
//! run-list [`PaddingSpec`] the zerocheck kernels consume.
//!
//! Nothing here is wired into the prover/verifier paths yet: Phase 0 lands
//! the types and their arithmetic; Phase 2 makes the union PIOP consume them.

use crate::r1cs::SparseBinaryMatrix;
use crate::zerocheck::{PaddingRun, PaddingSpec};

/// One table type: the base block of a single hash relation — exactly what
/// [`crate::r1cs::BlockR1cs`] stores per block (a one-type registry is
/// today's struct, minus the replication count).
///
/// The matrices are `2^k_log × 2^k_log` sparse boolean in circuit form
/// (`C_0 = I`); like `BlockR1cs`, walker-based encoders (Keccak) may carry
/// empty stubs here and supply their own `LincheckCircuit`.
#[derive(Clone, Debug)]
pub struct TableType {
    /// log2 of the base-block side `k = 2^k_log` — the design doc's `κ_t`.
    pub k_log: usize,
    /// Useful bits per block: columns `[0, useful_bits)` carry real trace
    /// data; columns `[useful_bits, 2^k_log)` are zero padding.
    pub useful_bits: usize,
    pub a_0: SparseBinaryMatrix,
    pub b_0: SparseBinaryMatrix,
    pub c_0: SparseBinaryMatrix,
    /// Column of a constant-one wire to pin to 1 across all blocks, or
    /// `None` (see `BlockR1cs::const_pin`).
    pub const_pin: Option<usize>,
}

/// Static layout of one type's slot in the union address space. Computed
/// once at registry construction from the capacity areas — the per-proof
/// counts never move anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// Slot variable count `m_t = nu + k_log_t`; the slot's capacity area
    /// is `2^m_t`.
    pub m_slot: usize,
    /// Slot offset `o_t` in the union address space — a multiple of `2^m_t`
    /// (the alignment invariant).
    pub offset: usize,
    /// Slot prefix `p_t = o_t >> m_t`: the value the top `M − m_t` address
    /// bits are frozen to for every address in the slot.
    pub prefix: usize,
    /// Prefix length `M − m_t` in bits (larger slots have shorter prefixes).
    pub prefix_bits: usize,
}

impl Slot {
    /// Capacity area `s_t = 2^m_t` in bits.
    pub fn area(&self) -> usize {
        1usize << self.m_slot
    }
}

/// The type registry: an ordered list of [`TableType`]s plus ONE uniform row
/// capacity `2^nu` (the design's uniform-capacity convention — every slot
/// shares the same row coordinates). Construction sorts the types by
/// non-increasing capacity area `2^{nu + k_log_t}` — under uniform capacity,
/// simply by `k_log` descending (stable, so equal-width types keep their
/// given order) — which is what guarantees the alignment invariant.
#[derive(Clone, Debug)]
pub struct Registry {
    types: Vec<TableType>,
    nu: usize,
    slots: Vec<Slot>,
    m_total: usize,
    /// Lazily computed [`Self::digest`]. Unlike `BlockR1cs` (public fields,
    /// manual `Clone` resetting the cache), every field here is private and
    /// immutable after construction, so the cache can never go stale and the
    /// derived `Clone` may carry it.
    digest_cache: std::sync::OnceLock<[u8; 32]>,
}

impl Registry {
    /// Build a registry from table types and the uniform log2 row capacity
    /// `nu` (any `nu ≥ 0`; the type is unsigned). Requires `k_log ≥ 7` for
    /// every type (BatchMajor 128-bit chunking; all current hash encoders
    /// have `k_log ≥ 14`).
    ///
    /// Computes each slot's area, offset `o_t`, and prefix
    /// `p_t = o_t >> m_t`, and the total variable count
    /// `M = log2(Σ_t 2^{m_t} rounded up to a power of two)`; asserts the
    /// alignment invariant `o_t ≡ 0 (mod 2^{m_t})`.
    pub fn new(mut types: Vec<TableType>, nu: usize) -> Self {
        assert!(!types.is_empty(), "registry needs at least one table type");
        for ty in &types {
            assert!(
                ty.k_log >= 7,
                "BatchMajor chunking requires k_log >= 7, got {}",
                ty.k_log
            );
            assert!(
                ty.useful_bits <= 1usize << ty.k_log,
                "useful_bits {} exceeds block size 2^{}",
                ty.useful_bits,
                ty.k_log
            );
        }
        // Non-increasing capacity area = k_log descending (uniform capacity).
        types.sort_by_key(|ty| std::cmp::Reverse(ty.k_log));

        let mut offset = 0usize;
        let mut partial: Vec<(usize, usize)> = Vec::with_capacity(types.len()); // (m_slot, offset)
        for ty in &types {
            let m_slot = nu + ty.k_log;
            // Guaranteed by the descending-area sort (each earlier area is a
            // multiple of 2^m_slot); asserted because everything downstream
            // (prefix freezing, subcube disjointness) rests on it.
            assert!(
                offset.is_multiple_of(1usize << m_slot),
                "slot offset {offset} not aligned to 2^{m_slot}"
            );
            partial.push((m_slot, offset));
            offset += 1usize << m_slot;
        }
        let m_total = offset.next_power_of_two().trailing_zeros() as usize;
        let slots = partial
            .into_iter()
            .map(|(m_slot, offset)| Slot {
                m_slot,
                offset,
                prefix: offset >> m_slot,
                prefix_bits: m_total - m_slot,
            })
            .collect();
        Self {
            types,
            nu,
            slots,
            m_total,
            digest_cache: std::sync::OnceLock::new(),
        }
    }

    /// BLAKE3 digest of the registry — the multi-table statement binding for
    /// the Fiat-Shamir transcript (design doc, "Statement, transcript, wire
    /// format"). Stable across runs; two registries agree iff they absorb
    /// the same bytes below.
    ///
    /// Normative absorption order (format version 1):
    /// 1. domain label `b"flock-registry-v1"` — intentionally
    ///    domain-separated from the single-table `b"flock-r1cs-stmt-v1"` of
    ///    [`crate::r1cs::BlockR1cs::statement_digest`], so a registry digest
    ///    can never collide with a single-table statement digest;
    /// 2. format-version byte `1u8`;
    /// 3. `nu` as u32 LE;
    /// 4. type count `T` as u32 LE;
    /// 5. per type, IN SLOT ORDER (the registry's sorted order): `k_log`
    ///    (u32 LE), `useful_bits` (u64 LE), `const_pin` as
    ///    `(present: u8, value: u64 LE)` — `(0, 0)` for `None`, `(1, col)`
    ///    for `Some(col)` — then the base matrices `a_0`, `b_0`, `c_0`, each
    ///    absorbed by the same length-prefixed routine `statement_digest`
    ///    uses (`crate::r1cs::absorb_matrix`).
    ///
    /// Lazily cached in `digest_cache`; first call materializes it,
    /// subsequent calls are essentially free.
    pub fn digest(&self) -> [u8; 32] {
        *self.digest_cache.get_or_init(|| {
            let mut h = blake3::Hasher::new();
            h.update(b"flock-registry-v1");
            h.update(&[1u8]);
            h.update(&(self.nu as u32).to_le_bytes());
            h.update(&(self.types.len() as u32).to_le_bytes());
            for ty in &self.types {
                h.update(&(ty.k_log as u32).to_le_bytes());
                h.update(&(ty.useful_bits as u64).to_le_bytes());
                let (present, value) = match ty.const_pin {
                    Some(col) => (1u8, col as u64),
                    None => (0u8, 0u64),
                };
                h.update(&[present]);
                h.update(&value.to_le_bytes());
                crate::r1cs::absorb_matrix(&mut h, &ty.a_0);
                crate::r1cs::absorb_matrix(&mut h, &ty.b_0);
                crate::r1cs::absorb_matrix(&mut h, &ty.c_0);
            }
            *h.finalize().as_bytes()
        })
    }

    /// The types, in slot order (non-increasing capacity area).
    pub fn types(&self) -> &[TableType] {
        &self.types
    }

    /// The per-slot layouts, parallel to [`Self::types`].
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// Uniform log2 row capacity: every slot holds up to `2^nu` invocations.
    pub fn nu(&self) -> usize {
        self.nu
    }

    /// Union variable count `M`: the address space is `{0,1}^M`. Registry-
    /// static — round counts and verifier control flow depend only on this,
    /// never on the per-proof counts.
    pub fn m_total(&self) -> usize {
        self.m_total
    }

    pub fn num_types(&self) -> usize {
        self.types.len()
    }
}

/// A proof instance over a registry: the public declared counts `n_t` —
/// arbitrary integers `0 ≤ n_t ≤ 2^nu`, chosen at prove time. Rows
/// `[n_t, 2^nu)` of slot `t` are dummy rows, identically zero.
#[derive(Clone, Debug)]
pub struct Instance<'r> {
    registry: &'r Registry,
    counts: Vec<usize>,
}

impl<'r> Instance<'r> {
    /// `counts[t]` is the declared invocation count of the registry's type
    /// `t`, in slot order.
    pub fn new(registry: &'r Registry, counts: Vec<usize>) -> Self {
        assert_eq!(
            counts.len(),
            registry.num_types(),
            "need one count per registry type"
        );
        for (t, &n) in counts.iter().enumerate() {
            assert!(
                n <= 1usize << registry.nu(),
                "count n_{t} = {n} exceeds row capacity 2^{}",
                registry.nu()
            );
        }
        Self { registry, counts }
    }

    pub fn registry(&self) -> &'r Registry {
        self.registry
    }

    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// The count-derived run-list [`PaddingSpec`] over the union BatchMajor
    /// buffer — the generalization of `BlockR1cs::padding_spec` to the slot
    /// schedule (design doc §5.2).
    ///
    /// Within slot `t` the BatchMajor address split is
    /// `[7 in-word | nu row | k_log_t − 7 chunk]`: the slot is `2^{k_log_t−7}`
    /// chunk-columns of `2^{7+nu}` bits, each holding one 128-bit word per
    /// invocation, words contiguous across invocations. The mapping to runs:
    ///
    /// - **Blocks are chunk-columns** (`k_log = 7 + nu`), so a run's
    ///   per-block useful prefix expresses the declared-row prefix of every
    ///   chunk-column at once.
    /// - **Dummy rows** `[n_t, 2^nu)` are each chunk-column's zero tail:
    ///   useful prefix `= 128·n_t` bits.
    /// - **Useless columns** `[useful_bits_t, 2^k_log_t)` round to whole
    ///   chunk-columns (the first `ceil(useful_bits_t/128)` carry data —
    ///   the same chunk-granular rounding as the BatchMajor
    ///   `BlockR1cs::padding_spec`); the rest are an explicit zero run
    ///   (`useful = 0`), NOT an implicit gap, because later slots must keep
    ///   their static offsets.
    /// - **The trailing gap** `[Σ_t 2^{nu+k_log_t}, 2^M)` after the last
    ///   slot is the run-list's implicit all-zero gap.
    pub fn padding_spec(&self) -> PaddingSpec {
        let nu = self.registry.nu();
        let mut runs = Vec::with_capacity(2 * self.counts.len());
        for (ty, &n_t) in self.registry.types().iter().zip(&self.counts) {
            let n_cols = 1usize << (ty.k_log - 7);
            let useful_cols = ty.useful_bits.div_ceil(128).min(n_cols);
            // Declared data: chunk-columns with the declared-row prefix.
            runs.push(PaddingRun {
                k_log: 7 + nu,
                useful_bits_per_block: n_t << 7,
                n_blocks: useful_cols,
            });
            // Useless chunk-columns: address space with no data (explicit).
            runs.push(PaddingRun {
                k_log: 7 + nu,
                useful_bits_per_block: 0,
                n_blocks: n_cols - useful_cols,
            });
        }
        PaddingSpec::from_runs(runs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs::{BlockR1cs, WitnessLayout};

    /// Empty matrix stub — layout tests never apply the matrices, mirroring
    /// the walker-based encoders' stub practice.
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
    }

    /// Offset/prefix/alignment arithmetic on the doc's 3-type shape
    /// (κ = 16/15/14, ν = 10), fed in shuffled order to exercise the sort.
    #[test]
    fn three_type_layout_arithmetic() {
        let reg = Registry::new(vec![ty(14, 15_409), ty(16, 42_560), ty(15, 31_401)], 10);

        // Sorted by capacity area descending = κ descending.
        let k_logs: Vec<usize> = reg.types().iter().map(|t| t.k_log).collect();
        assert_eq!(k_logs, vec![16, 15, 14]);

        // Areas 2^26 + 2^25 + 2^24 = 0x7000000 → M = 27.
        assert_eq!(reg.m_total(), 27);
        assert_eq!(
            reg.slots(),
            &[
                Slot {
                    m_slot: 26,
                    offset: 0,
                    prefix: 0b0,
                    prefix_bits: 1
                },
                Slot {
                    m_slot: 25,
                    offset: 1 << 26,
                    prefix: 0b10,
                    prefix_bits: 2
                },
                Slot {
                    m_slot: 24,
                    offset: (1 << 26) + (1 << 25),
                    prefix: 0b110,
                    prefix_bits: 3
                },
            ]
        );
        // Alignment invariant, spelled out.
        for slot in reg.slots() {
            assert!(slot.offset.is_multiple_of(slot.area()));
            assert_eq!(slot.prefix << slot.m_slot, slot.offset);
            assert_eq!(slot.prefix_bits, reg.m_total() - slot.m_slot);
        }
    }

    /// A single-type registry reproduces the geometry of today's BlockR1cs:
    /// same variable count, and — at full utilization — the same padding
    /// semantics as the BatchMajor `BlockR1cs::padding_spec` (the run
    /// encodings differ; the useful-bit classification must not).
    #[test]
    fn single_type_registry_matches_block_r1cs_geometry() {
        let (k_log, useful_bits, nu) = (14usize, 15_409usize, 3usize);
        let reg = Registry::new(vec![ty(k_log, useful_bits)], nu);
        let r1cs = BlockR1cs {
            m: nu + k_log,
            k_log,
            k_skip: 6,
            useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            layout: WitnessLayout::BatchMajor,
            const_pin: None,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };

        assert_eq!(reg.m_total(), r1cs.m);
        assert_eq!(reg.num_types(), 1);
        let slot = reg.slots()[0];
        assert_eq!(slot.m_slot, r1cs.m);
        assert_eq!((slot.offset, slot.prefix, slot.prefix_bits), (0, 0, 0));

        // Full utilization: the declared counts fill the capacity.
        let inst = Instance::new(&reg, vec![1 << nu]);
        assert_eq!(
            inst.padding_spec().useful_intervals(),
            r1cs.padding_spec().useful_intervals(),
            "count-derived spec must classify the same bits useful as today's"
        );
    }

    /// Counts → run-list derivation, including the n_t = 0 and n_t = 2^nu
    /// edge cases and the implicit trailing gap.
    #[test]
    fn counts_to_run_list_derivation() {
        // κ = 10/9, ν = 3: slot areas 2^13 + 2^12 = 12288, M = 14.
        // Type A: 8 chunk-columns, 6 useful (ceil(700/128)); type B: 4
        // chunk-columns, 3 useful (ceil(300/128)).
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        assert_eq!(reg.m_total(), 14);

        // Mid-range count + full capacity.
        let inst = Instance::new(&reg, vec![5, 8]);
        let spec = inst.padding_spec();
        assert_eq!(
            spec.runs(),
            &[
                // Slot A: 6 data columns with 5 of 8 rows declared, 2 useless.
                PaddingRun {
                    k_log: 10,
                    useful_bits_per_block: 5 * 128,
                    n_blocks: 6
                },
                PaddingRun {
                    k_log: 10,
                    useful_bits_per_block: 0,
                    n_blocks: 2
                },
                // Slot B at full utilization: dense data columns, 1 useless.
                PaddingRun {
                    k_log: 10,
                    useful_bits_per_block: 1024,
                    n_blocks: 3
                },
                PaddingRun {
                    k_log: 10,
                    useful_bits_per_block: 0,
                    n_blocks: 1
                },
            ]
        );
        // Runs end at the last slot's end; [12288, 2^14) is the implicit gap.
        assert_eq!(spec.covered_bits(), 12288);
        // Slot B's dense columns start at its offset and merge.
        assert_eq!(reg.slots()[1].offset, 8192);
        assert!(spec.useful_intervals().contains(&(8192, 8192 + 3 * 1024)));

        // n_t = 0: the slot still occupies its address space, all zero.
        let empty = Instance::new(&reg, vec![0, 0]);
        let spec = empty.padding_spec();
        assert_eq!(spec.covered_bits(), 12288);
        assert_eq!(spec.useful_intervals(), Vec::<(usize, usize)>::new());
        assert!(spec.runs().iter().all(|r| r.useful_bits_per_block == 0));
    }

    /// End-to-end: a schedule-derived multi-run spec drives the zerocheck
    /// prover through the general kernel paths and produces the same proof
    /// as the dense prover on an honestly padded union witness.
    #[test]
    fn instance_padding_spec_proves_like_dense() {
        use crate::challenger::{Challenger, FsChallenger};
        use crate::zerocheck::univariate_skip::pack_bits;
        use crate::zerocheck::{prove_packed, prove_packed_padded};

        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let m = reg.m_total();
        let inst = Instance::new(&reg, vec![5, 3]);
        let spec = inst.padding_spec();
        assert!(spec.as_single_run().is_none(), "must exercise multi-run");

        // Random bits on the useful intervals, zero elsewhere; c = a AND b.
        let mut rng = Rng::new(0x5C4E_D01E);
        let mut useful = vec![false; 1 << m];
        for (s, e) in spec.useful_intervals() {
            useful[s..e].fill(true);
        }
        let a: Vec<bool> = useful
            .iter()
            .map(|u| *u && rng.next_u64() & 1 == 1)
            .collect();
        let b: Vec<bool> = useful
            .iter()
            .map(|u| *u && rng.next_u64() & 1 == 1)
            .collect();
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = (pack_bits(&a), pack_bits(&b), pack_bits(&c));

        let mut ch_dense = FsChallenger::new(b"flock-test-v0");
        let (proof_dense, claim_dense) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_dense);
        let mut ch_padded = FsChallenger::new(b"flock-test-v0");
        let (proof_padded, claim_padded) =
            prove_packed_padded(&a_p, &b_p, &c_p, m, &spec, &mut ch_padded);

        assert_eq!(proof_dense, proof_padded, "proof mismatch");
        assert_eq!(claim_dense, claim_padded, "claim mismatch");
        assert_eq!(
            ch_dense.sample_f128(),
            ch_padded.sample_f128(),
            "post-proof transcript state diverged"
        );
    }

    #[test]
    #[should_panic(expected = "k_log >= 7")]
    fn registry_rejects_narrow_type() {
        let _ = Registry::new(vec![ty(6, 64)], 4);
    }

    #[test]
    #[should_panic(expected = "at least one table type")]
    fn registry_rejects_empty_type_list() {
        let _ = Registry::new(Vec::new(), 4);
    }

    #[test]
    #[should_panic(expected = "exceeds row capacity")]
    fn instance_rejects_count_over_capacity() {
        let reg = Registry::new(vec![ty(9, 300)], 3);
        let _ = Instance::new(&reg, vec![9]);
    }

    // The registry digest's `b"flock-registry-v1"` label intentionally
    // domain-separates it from `BlockR1cs::statement_digest`'s
    // `b"flock-r1cs-stmt-v1"`: a registry digest can never collide with a
    // single-table statement digest, even for a one-type registry whose
    // parameters and matrices match a `BlockR1cs` exactly.

    /// Sparse matrix with the given rows (shape and contents are absorbed
    /// as-is; the digest does not validate dimensions against `k_log`, same
    /// as the walker-encoder stub convention).
    fn matrix(rows: Vec<Vec<usize>>) -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            num_rows: rows.len(),
            num_cols: 512,
            rows,
        }
    }

    /// Digest is stable across calls (cache), across identically constructed
    /// registries, and across clones.
    #[test]
    fn registry_digest_deterministic() {
        let mk = || {
            Registry::new(
                vec![
                    ty(10, 700),
                    TableType {
                        k_log: 9,
                        useful_bits: 300,
                        a_0: matrix(vec![vec![0, 3], vec![7]]),
                        b_0: stub(),
                        c_0: stub(),
                        const_pin: Some(2),
                    },
                ],
                3,
            )
        };
        let a = mk();
        let d = a.digest();
        assert_eq!(d, a.digest(), "digest must be stable across calls");
        assert_eq!(
            d,
            mk().digest(),
            "identically constructed registries must agree"
        );
        assert_eq!(d, a.clone().digest(), "clone must carry the same digest");
    }

    /// Every absorbed component moves the digest: nu, a single matrix
    /// entry, useful_bits, and const_pin (including Some(0) vs None — the
    /// present byte). Type-order sensitivity is not testable at the
    /// constructor boundary: `Registry::new` sorts, so two constructions
    /// differing only in input order are the SAME registry and must (and
    /// do, per `registry_digest_deterministic`) agree.
    #[test]
    fn registry_digest_sensitivity() {
        let mk = |nu, useful_bits, a_rows: Vec<Vec<usize>>, const_pin| {
            Registry::new(
                vec![
                    ty(10, 700),
                    TableType {
                        k_log: 9,
                        useful_bits,
                        a_0: matrix(a_rows),
                        b_0: stub(),
                        c_0: stub(),
                        const_pin,
                    },
                ],
                nu,
            )
        };
        let d = mk(3, 300, vec![vec![0, 3], vec![7]], None).digest();
        let cases = [
            ("nu", mk(4, 300, vec![vec![0, 3], vec![7]], None)),
            (
                "single matrix entry",
                mk(3, 300, vec![vec![0, 4], vec![7]], None),
            ),
            ("useful_bits", mk(3, 301, vec![vec![0, 3], vec![7]], None)),
            (
                "const_pin Some(0) vs None",
                mk(3, 300, vec![vec![0, 3], vec![7]], Some(0)),
            ),
        ];
        for (what, reg) in cases {
            assert_ne!(d, reg.digest(), "digest insensitive to {what}");
        }
    }
}
