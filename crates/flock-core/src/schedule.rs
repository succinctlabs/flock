//! Multi-table registry and slot schedule.
//!
//! A [`Registry`] fixes table types and a uniform row capacity `2^nu`.
//! Each slot is an aligned subcube in one address space of `2^M` points.
//! An [`Instance`] adds row counts and derives the zerocheck [`PaddingSpec`].
//!
//! Types carry a [`TableClass`]: `Boolean` (the bit-level hash relations) or
//! `LargeField` (the element relation of [`crate::element_r1cs`]). The
//! scheduler is class-blind — an element type presents `k_log = kappa + 7`
//! and `useful_bits = k·128`, so all the slot arithmetic below is shared —
//! The class-major layout gives each class a disjoint aligned subcube. Each
//! class PIOP runs only over its region (see
//! [`Registry::new`]).

use std::sync::Arc;

use crate::element_r1cs::ElementTableType;
use crate::r1cs::SparseBinaryMatrix;
use crate::zerocheck::{PaddingRun, PaddingSpec};

/// Largest `k_log` a registry accepts. Far above any real table (`M` would
/// already be astronomical), and load-bearing for the injectivity of
/// [`Registry::digest`]: it keeps the four leading bytes of a type's
/// absorbed header (`k_log` as u32 LE) strictly below the four leading bytes
/// of [`ELEMENT_CLASS_LABEL`], so no boolean type's encoding can be mistaken
/// for an element-class suffix. See the digest's absorption order.
pub const MAX_K_LOG: usize = 40;

/// Domain label for large-field metadata in [`Registry::digest`]. Boolean
/// registries do not absorb this label.
const ELEMENT_CLASS_LABEL: &[u8] = b"flock-element-class-v0";

/// Domain label for non-empty IO schemas in [`Registry::digest`]. Registries
/// without schemas do not absorb this label.
const IO_SCHEMA_LABEL: &[u8] = b"flock-io-schema-v0";

/// Which side of a gate an [`IoWord`] is. **Metadata for circuit validation
/// only** — the wiring argument is direction-blind (a permutation neither
/// knows nor cares which cell of a wire class is the producer); the circuit
/// layer uses it for the dataflow order that acyclicity is checked against,
/// and witness generation uses it to assert that a class's several producers
/// agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoDirection {
    In,
    Out,
}

impl IoDirection {
    /// Absorption byte for [`Registry::digest`].
    fn code(self) -> u8 {
        match self {
            IoDirection::In => 0,
            IoDirection::Out => 1,
        }
    }
}

/// One entry of a table type's **IO schema**: a 128-bit word-column of the
/// type's row that circuits may wire, plus its direction.
///
/// The unit is a committed WORD, not a bit — the aligned IO regions
/// (`region_log = 8` slots for SHA-256/BLAKE3,
/// one element column for a large-field type) make a wire value exactly one
/// committed word. A word outside the schema is internal. The relation must
/// pin each internal word that cannot remain free.
///
/// `word_col` indexes the type's own row: word `w` of slot `t`'s row `j` is
/// union word `(o_t >> 7) + (w << nu) + j` — see
/// [`crate::union::UnionInstance::slot_word_range`] and
/// [`crate::circuit::CellSpace::gate_word_addr`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoWord {
    /// Word-column index within the type's row, `< ceil(useful_bits / 128)`.
    pub word_col: usize,
    pub dir: IoDirection,
}

impl IoWord {
    pub const fn input(word_col: usize) -> Self {
        Self {
            word_col,
            dir: IoDirection::In,
        }
    }
    pub const fn output(word_col: usize) -> Self {
        Self {
            word_col,
            dir: IoDirection::Out,
        }
    }
}

/// What a table type's witness words MEAN — and therefore which PIOP proves
/// the slot. Both classes use the same `k_log`, `useful_bits`, and [`Slot`]
/// arithmetic. The class selects the prover.
#[derive(Clone, Debug, Default)]
pub enum TableClass {
    /// GF(2) bit-level tables — the hash relations. The payload is
    /// [`TableType`]'s own `a_0`/`b_0`/`c_0`/`const_pin` fields.
    #[default]
    Boolean,
    /// Large-field (element) tables: one F128 element per committed word,
    /// the relation `(A_0 z + a)(B_0 z + b) = z` of [`crate::element_r1cs`].
    /// The boolean matrix fields are empty stubs for these types (nothing
    /// reads them; [`TableType::element`] is the only constructor).
    ///
    /// `Arc` because [`ElementTableType`] caches its digest in a `OnceLock`
    /// and is therefore not `Clone`, while [`TableType`] is.
    LargeField(Arc<ElementTableType>),
}

/// Base block for one boolean or large-field relation.
///
/// For a boolean type the matrices are `2^k_log × 2^k_log` sparse boolean in
/// circuit form (`C_0 = I`); like `BlockR1cs`, walker-based encoders
/// (the tower's stub gates) may carry empty stubs here and supply their own
/// `LincheckCircuit`.
#[derive(Clone, Debug)]
pub struct TableType {
    /// log2 of the base-block side `k = 2^k_log`.
    /// For an element type this is `kappa + 7`: one row is `2^kappa` words
    /// of 128 bits, and the 7 in-word bits are the element's basis
    /// coordinates, so the slot bookkeeping below applies unchanged.
    pub k_log: usize,
    /// Useful bits per block: columns `[0, useful_bits)` carry real trace
    /// data; columns `[useful_bits, 2^k_log)` are zero padding. For an
    /// element type this is `k · 128` — the real element columns — so
    /// `used_cols` counts element columns.
    pub useful_bits: usize,
    pub a_0: SparseBinaryMatrix,
    pub b_0: SparseBinaryMatrix,
    pub c_0: SparseBinaryMatrix,
    /// Column of a constant-one wire to pin to 1 across all blocks, or
    /// `None` (see `BlockR1cs::const_pin`). Always `None` for element types:
    /// the element relation pins constants through `a_const`/`b_const`.
    pub const_pin: Option<usize>,
    /// Boolean or large-field. Defaults to [`TableClass::Boolean`], so a
    /// struct literal that predates the class tag keeps its meaning.
    pub class: TableClass,
    /// The type's **IO schema**: the ordered list of wireable word-columns
    /// (see [`IoWord`]). EMPTY for a type no circuit wires — the default,
    /// and the state in which this type absorbs no schema bytes into
    /// [`Registry::digest`]. The order is the type's contribution to the
    /// cell-slot enumeration ([`crate::circuit::CellSpace`]), so it is
    /// digest-visible and must not be permuted casually.
    pub io_schema: Vec<IoWord>,
}

impl TableType {
    /// One-type view of an existing single-table [`crate::r1cs::BlockR1cs`]:
    /// the base block, width, useful bits, and const pin — everything except
    /// the replication count, which becomes the registry's uniform capacity
    /// `nu` (pass the r1cs's `n_log()` to [`Registry::new`] to reproduce
    /// today's geometry exactly).
    pub fn from_block_r1cs(r1cs: &crate::r1cs::BlockR1cs) -> Self {
        Self {
            k_log: r1cs.k_log,
            useful_bits: r1cs.useful_bits,
            a_0: r1cs.a_0.clone(),
            b_0: r1cs.b_0.clone(),
            c_0: r1cs.c_0.clone(),
            const_pin: r1cs.const_pin,
            class: TableClass::Boolean,
            io_schema: Vec::new(),
        }
    }

    /// A large-field (element) table type, presented to the scheduler as a
    /// `k_log = kappa + 7` block with `useful_bits = k · 128` (see the field
    /// docs). The boolean matrix fields are empty stubs — the real payload
    /// rides in [`TableClass::LargeField`].
    pub fn element(ty: Arc<ElementTableType>) -> Self {
        let stub = || SparseBinaryMatrix {
            num_rows: 0,
            num_cols: 0,
            rows: Vec::new(),
        };
        Self {
            k_log: ty.kappa() + 7,
            useful_bits: ty.k() * 128,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: None,
            class: TableClass::element(ty),
            io_schema: Vec::new(),
        }
    }

    /// This type with the given IO schema attached — the only way a circuit
    /// can name any of its words. Word-columns are validated against the
    /// type's real width by [`Registry::new`].
    pub fn with_io_schema(mut self, io_schema: Vec<IoWord>) -> Self {
        self.io_schema = io_schema;
        self
    }

    /// Number of 128-bit word-columns a row of this type really uses:
    /// `ceil(useful_bits / 128)`. The IO schema's `word_col` bound.
    pub fn used_word_cols(&self) -> usize {
        self.useful_bits.div_ceil(128)
    }

    /// The element payload, or `None` for a boolean type.
    pub fn element_type(&self) -> Option<&ElementTableType> {
        match &self.class {
            TableClass::Boolean => None,
            TableClass::LargeField(ty) => Some(ty),
        }
    }

    /// Whether this type is proven by the large-field PIOP.
    pub fn is_element(&self) -> bool {
        matches!(self.class, TableClass::LargeField(_))
    }
}

impl TableClass {
    /// [`TableClass::LargeField`] from an owned element type.
    pub fn element(ty: Arc<ElementTableType>) -> Self {
        Self::LargeField(ty)
    }
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
    /// Number of boolean types — a PREFIX of `types`/`slots` by the
    /// class-major sort. `types[num_boolean..]` are the element types.
    num_boolean: usize,
    /// `M_bool`: the boolean region is the prefix subcube `[0, 2^M_bool)`.
    /// `0` when there are no boolean types (the region is empty).
    m_bool: usize,
    /// `M_elem`: the element region is the aligned subcube
    /// `[element_base, element_base + 2^M_elem)`. `0` when there are no
    /// element types.
    m_elem: usize,
    /// Start of the element region — a multiple of `2^M_elem`, hence a
    /// subcube base whose top `M − M_elem` address bits are the fixed
    /// pattern `element_base >> M_elem`. Meaningless when `m_elem == 0`.
    element_base: usize,
    /// Lazily computed [`Self::digest`]. Unlike `BlockR1cs` (public fields,
    /// manual `Clone` resetting the cache), every field here is private and
    /// immutable after construction, so the cache can never go stale and the
    /// derived `Clone` may carry it.
    digest_cache: std::sync::OnceLock<[u8; 32]>,
}

impl Registry {
    /// Build a registry from table types and the uniform log2 row capacity
    /// `nu` (any `nu ≥ 0`; the type is unsigned). Requires
    /// `7 ≤ k_log ≤ MAX_K_LOG` for every type (BatchMajor 128-bit chunking;
    /// all current hash encoders have `k_log ≥ 14`).
    ///
    /// Computes each slot's area, offset `o_t`, and prefix
    /// `p_t = o_t >> m_t`, and the total variable count
    /// `M = log2(Σ_t 2^{m_t} rounded up to a power of two)`; asserts the
    /// alignment invariant `o_t ≡ 0 (mod 2^{m_t})`.
    ///
    /// ## Class-major layout
    ///
    /// Types sort **class-major**: boolean types first (area-descending, as
    /// before, packing from offset 0), then element types (area-descending).
    /// The two classes then occupy DISJOINT ALIGNED SUBCUBES, which is what
    /// lets each class's PIOP run over its own region only (the union
    /// zerocheck aliases `c = z`, so a shared domain with the element region
    /// "marked dead" would make the honest global sum non-zero — see
    /// [`crate::union`]):
    ///
    /// - the **boolean region** is the prefix subcube `[0, 2^{M_bool})`,
    ///   where `2^{M_bool}` is the smallest power of two covering the
    ///   boolean slots' total extent;
    /// - the **element region** starts at `element_base`, the boolean
    ///   region's end rounded up to a multiple of the element region's own
    ///   size `2^{M_elem}` — so `element_base = 2^{max(M_bool, M_elem)}`
    ///   when both classes are present, and `0` when there are no boolean
    ///   types. Its top `M − M_elem` address bits are a fixed Boolean
    ///   pattern.
    ///
    /// A boolean-only registry is unaffected in every respect (the element
    /// arithmetic collapses and `M = M_bool`), which is what keeps the
    /// existing byte-identity anchors intact.
    pub fn new(mut types: Vec<TableType>, nu: usize) -> Self {
        assert!(!types.is_empty(), "registry needs at least one table type");
        for ty in &types {
            assert!(
                ty.k_log >= 7,
                "BatchMajor chunking requires k_log >= 7, got {}",
                ty.k_log
            );
            assert!(
                ty.k_log <= MAX_K_LOG,
                "k_log {} exceeds MAX_K_LOG = {MAX_K_LOG}",
                ty.k_log
            );
            assert!(
                ty.useful_bits <= 1usize << ty.k_log,
                "useful_bits {} exceeds block size 2^{}",
                ty.useful_bits,
                ty.k_log
            );
            if let Some(el) = ty.element_type() {
                assert_eq!(ty.k_log, el.kappa() + 7, "element k_log must be kappa + 7");
                assert_eq!(
                    ty.useful_bits,
                    el.k() * 128,
                    "element useful_bits must be k · 128"
                );
                assert!(
                    ty.const_pin.is_none(),
                    "element types have no const pin (constants ride in a_const/b_const)"
                );
            }
            // IO schema: every entry must name a real word-column of this
            // type's row, and no word may appear twice (one cell per word —
            // the cell-slot enumeration and hence σ's index space depend on
            // it).
            let used_cols = ty.used_word_cols();
            let mut seen = std::collections::BTreeSet::new();
            for w in &ty.io_schema {
                assert!(
                    w.word_col < used_cols,
                    "IO schema word-column {} is outside the type's {used_cols} used columns",
                    w.word_col
                );
                assert!(
                    seen.insert(w.word_col),
                    "IO schema names word-column {} twice",
                    w.word_col
                );
            }
        }
        // Class-major, then non-increasing capacity area = k_log descending
        // (uniform capacity). Stable, so equal-width types keep their given
        // order — and the boolean types stay a prefix of the list.
        types.sort_by_key(|ty| (ty.is_element(), std::cmp::Reverse(ty.k_log)));
        let num_boolean = types.iter().filter(|ty| !ty.is_element()).count();

        // Pack each class from its own base, area-descending. Boolean starts
        // at 0; the element base needs the boolean extent first.
        let extent =
            |tys: &[TableType]| -> usize { tys.iter().map(|ty| 1usize << (nu + ty.k_log)).sum() };
        let bool_extent = extent(&types[..num_boolean]);
        let elem_extent = extent(&types[num_boolean..]);
        let pow2_log = |n: usize| n.next_power_of_two().trailing_zeros() as usize;
        let m_bool = if bool_extent == 0 {
            0
        } else {
            pow2_log(bool_extent)
        };
        let m_elem = if elem_extent == 0 {
            0
        } else {
            pow2_log(elem_extent)
        };
        // The boolean region is the prefix SUBCUBE, so the element region
        // starts past `2^M_bool` (not past the tighter `bool_extent`), rounded
        // up to its own alignment. Both are powers of two, so this is
        // `2^max(M_bool, M_elem)` — and `0` with no boolean types.
        // With no element types the base is meaningless (0); with no boolean
        // types the element region starts at 0 and IS the prefix subcube, so
        // an element-only registry wastes no address space.
        let element_base = if elem_extent == 0 || bool_extent == 0 {
            0
        } else {
            1usize << m_bool.max(m_elem)
        };

        let mut partial: Vec<(usize, usize)> = Vec::with_capacity(types.len()); // (m_slot, offset)
        let mut offset = 0usize;
        for (t, ty) in types.iter().enumerate() {
            if t == num_boolean {
                offset = element_base; // cross into the element region
            }
            let m_slot = nu + ty.k_log;
            // Guaranteed by the descending-area sort within each class (each
            // earlier area is a multiple of 2^m_slot, and `element_base` is a
            // multiple of 2^M_elem ≥ 2^m_slot); asserted because everything
            // downstream (prefix freezing, subcube disjointness) rests on it.
            assert!(
                offset.is_multiple_of(1usize << m_slot),
                "slot offset {offset} not aligned to 2^{m_slot}"
            );
            partial.push((m_slot, offset));
            offset += 1usize << m_slot;
        }
        let m_total = pow2_log(offset);
        let slots: Vec<Slot> = partial
            .into_iter()
            .map(|(m_slot, offset)| Slot {
                m_slot,
                offset,
                prefix: offset >> m_slot,
                prefix_bits: m_total - m_slot,
            })
            .collect();

        // The two region invariants the disjoint PIOPs rest on, spelled out.
        if num_boolean > 0 {
            assert!(
                m_bool <= m_total,
                "boolean region exceeds the address space"
            );
            let last = &slots[num_boolean - 1];
            assert!(
                last.offset + last.area() <= 1usize << m_bool,
                "boolean slots must fit the prefix subcube [0, 2^M_bool)"
            );
        }
        if num_boolean < types.len() {
            assert!(
                m_elem <= m_total,
                "element region exceeds the address space"
            );
            assert!(
                element_base.is_multiple_of(1usize << m_elem),
                "element region base {element_base} not aligned to 2^{m_elem}"
            );
            assert!(
                num_boolean == 0 || element_base >= 1usize << m_bool,
                "element region must start past the boolean prefix subcube"
            );
            assert!(
                element_base + (1usize << m_elem) <= 1usize << m_total,
                "element region subcube exceeds the address space"
            );
        }

        Self {
            types,
            nu,
            slots,
            m_total,
            num_boolean,
            m_bool,
            m_elem,
            element_base,
            digest_cache: std::sync::OnceLock::new(),
        }
    }

    /// BLAKE3 statement binding for the Fiat-Shamir transcript. Two registries
    /// agree when they absorb the same bytes below.
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
    ///    uses (`crate::r1cs::absorb_matrix`);
    /// 6. **for element types only**, appended after that type's stub
    ///    matrices: the label [`ELEMENT_CLASS_LABEL`] followed by the
    ///    element base block's own digest
    ///    ([`ElementTableType::digest`], which covers `kappa`, `k`, `A_0`,
    ///    `B_0`, `a_const`, `b_const`). A **boolean-only registry therefore
    ///    absorbs exactly the bytes it absorbed before the element class
    ///    existed** — the byte-identity bar for every pinned fixture;
    /// 7. **for types with a non-empty IO schema only**, appended after
    ///    that (so after the element payload when both are present): the
    ///    label [`IO_SCHEMA_LABEL`], the entry count as u32 LE, then per
    ///    [`IoWord`] in schema order its `word_col` (u32 LE) and direction
    ///    byte. A registry whose types carry no schema — every registry
    ///    that predates the wiring layer — absorbs nothing here, the same
    ///    byte-identity bar.
    ///
    /// The conditional suffixes keep the encoding injective: a type's boolean
    /// part is self-delimiting (fixed 21-byte header + three length-prefixed
    /// matrices), and after it the next four bytes either begin one of the two
    /// labels (`"floc"`, i.e. `0x636f6c66` read as u32 LE) or the next type's
    /// `k_log`, which [`MAX_K_LOG`] bounds far below that — so a left-to-right
    /// parse can never confuse a suffix with the next type. The two labels are
    /// distinct and neither is a prefix of the other, they appear in a fixed
    /// order, and the schema payload is length-prefixed, so the suffixes cannot
    /// be confused with each other either.
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
                // Element payload appends ONLY when present — see above.
                if let Some(el) = ty.element_type() {
                    h.update(ELEMENT_CLASS_LABEL);
                    h.update(&el.digest());
                }
                // IO schema appends ONLY when non-empty — see above.
                if !ty.io_schema.is_empty() {
                    h.update(IO_SCHEMA_LABEL);
                    h.update(&(ty.io_schema.len() as u32).to_le_bytes());
                    for w in &ty.io_schema {
                        h.update(&(w.word_col as u32).to_le_bytes());
                        h.update(&[w.dir.code()]);
                    }
                }
            }
            *h.finalize().as_bytes()
        })
    }

    /// Number of boolean types — a PREFIX of [`Self::types`] /
    /// [`Self::slots`], by the class-major sort.
    pub fn num_boolean(&self) -> usize {
        self.num_boolean
    }

    /// Number of element types — the SUFFIX `types()[num_boolean()..]`.
    pub fn num_element(&self) -> usize {
        self.types.len() - self.num_boolean
    }

    /// The boolean types, in slot order.
    pub fn boolean_types(&self) -> &[TableType] {
        &self.types[..self.num_boolean]
    }

    /// The element types, in slot order.
    pub fn element_types(&self) -> &[TableType] {
        &self.types[self.num_boolean..]
    }

    /// `M_bool` — the boolean region is the prefix subcube `[0, 2^{M_bool})`
    /// and the boolean PIOP runs over exactly that many variables. `0` when
    /// there are no boolean types. Equals [`Self::m_total`] for a
    /// boolean-only registry (which is why that case is untouched).
    pub fn m_bool(&self) -> usize {
        self.m_bool
    }

    /// `M_elem` — the element region is the aligned subcube
    /// `[element_base, element_base + 2^{M_elem})`. `0` with no element types.
    pub fn m_elem(&self) -> usize {
        self.m_elem
    }

    /// Base address of the element region: a multiple of `2^{M_elem}`, so the
    /// region's top `M − M_elem` address bits are frozen to the Boolean
    /// pattern `element_base >> M_elem`. Meaningless when
    /// [`Self::num_element`] is 0.
    pub fn element_base(&self) -> usize {
        self.element_base
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
    /// buffer. This applies `BlockR1cs::padding_spec` to the slot schedule.
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
    /// - **Inter-slot gaps** — since the class-major layout, the boolean
    ///   region's subcube tail plus the run-up to `element_base` — are an
    ///   explicit zero run, for the same reason useless columns are: later
    ///   slots must keep their static offsets. Every gap is a whole number
    ///   of chunk-columns (`element_base` is a power of two `≥ 2^{7+nu}` and
    ///   every slot area is a multiple of `2^{7+nu}`).
    /// - **The trailing gap** after the last slot is the run-list's implicit
    ///   all-zero gap.
    ///
    /// Element slots need no special case: their `k_log = kappa + 7` makes
    /// `n_cols = 2^kappa` word-columns and `useful_cols = k`, so the same
    /// two runs describe "k used element columns at the declared row count,
    /// the rest zero padding".
    pub fn padding_spec(&self) -> PaddingSpec {
        let nu = self.registry.nu();
        let col_bits = 7 + nu;
        let mut runs = Vec::with_capacity(3 * self.counts.len());
        let mut cursor = 0usize;
        for ((ty, slot), &n_t) in self
            .registry
            .types()
            .iter()
            .zip(self.registry.slots())
            .zip(&self.counts)
        {
            // Explicit zero run for any gap before this slot.
            debug_assert!(slot.offset >= cursor);
            let gap = slot.offset - cursor;
            debug_assert!(
                gap.is_multiple_of(1usize << col_bits),
                "gap {gap} not column-aligned"
            );
            runs.push(PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: 0,
                n_blocks: gap >> col_bits,
            });
            let n_cols = 1usize << (ty.k_log - 7);
            let useful_cols = ty.useful_bits.div_ceil(128).min(n_cols);
            // Declared data: chunk-columns with the declared-row prefix.
            runs.push(PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: n_t << 7,
                n_blocks: useful_cols,
            });
            // Useless chunk-columns: address space with no data (explicit).
            runs.push(PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: 0,
                n_blocks: n_cols - useful_cols,
            });
            cursor = slot.offset + slot.area();
        }
        PaddingSpec::from_runs(runs)
    }

    /// The boolean-class prefix of [`Self::padding_spec`] — the run-list over
    /// the boolean region `[0, 2^{M_bool})` alone, which is the domain the
    /// boolean zerocheck runs on. Identical to [`Self::padding_spec`] for a
    /// boolean-only registry (the element runs and the class gap are exactly
    /// what it drops), so the boolean PIOP is untouched there.
    pub fn boolean_padding_spec(&self) -> PaddingSpec {
        let nu = self.registry.nu();
        let col_bits = 7 + nu;
        let nb = self.registry.num_boolean();
        let mut runs = Vec::with_capacity(3 * nb);
        let mut cursor = 0usize;
        for ((ty, slot), &n_t) in self.registry.types()[..nb]
            .iter()
            .zip(&self.registry.slots()[..nb])
            .zip(&self.counts[..nb])
        {
            let gap = slot.offset - cursor;
            runs.push(PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: 0,
                n_blocks: gap >> col_bits,
            });
            let n_cols = 1usize << (ty.k_log - 7);
            let useful_cols = ty.useful_bits.div_ceil(128).min(n_cols);
            runs.push(PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: n_t << 7,
                n_blocks: useful_cols,
            });
            runs.push(PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: 0,
                n_blocks: n_cols - useful_cols,
            });
            cursor = slot.offset + slot.area();
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
            class: TableClass::Boolean,
            io_schema: Vec::new(),
        }
    }

    /// An element type of the given shape: `kappa` column bits, all `2^kappa`
    /// columns free wires (so `k = 2^kappa` and every column is used). Only
    /// the shape matters to the schedule.
    pub(crate) fn elem_ty(kappa: usize) -> TableType {
        use crate::element_r1cs::ElementTableBuilder;
        let mut b = ElementTableBuilder::new(kappa);
        for y in 0..1usize << kappa {
            b.free_wire(y);
        }
        TableType::element(Arc::new(b.build().expect("free-wire block is valid")))
    }

    use crate::test_rng::Rng;

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
                        class: TableClass::Boolean,
                        io_schema: Vec::new(),
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
                        class: TableClass::Boolean,
                        io_schema: Vec::new(),
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

    // ---- element class: digest byte-identity + class-major layout ----------

    /// A boolean-only registry does not absorb large-field metadata. The fixed
    /// digest detects changes on both sides of the comparison.
    #[test]
    fn boolean_only_digest_is_byte_identical() {
        // Same shapes as `registry_digest_deterministic` / the layout tests.
        let plain = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        assert_eq!(
            hex(&plain.digest()),
            "8010ecf651ca43eadfe415eac6a081c8f9022dd077070da0f22cea19297699ea",
            "boolean-only digest moved — the element class must append nothing"
        );
        let with_pin = Registry::new(
            vec![
                ty(10, 700),
                TableType {
                    k_log: 9,
                    useful_bits: 300,
                    a_0: matrix(vec![vec![0, 3], vec![7]]),
                    b_0: stub(),
                    c_0: stub(),
                    const_pin: Some(2),
                    class: TableClass::Boolean,
                    io_schema: Vec::new(),
                },
            ],
            3,
        );
        assert_eq!(
            hex(&with_pin.digest()),
            "2662f8c311c8cdf715ed636bc46a5de16daf06e7516a2e299e6f80132ad9e9c2",
            "boolean-only digest (with matrices + pin) moved"
        );
    }

    fn hex(d: &[u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    // ---- IO schemas: the same append-only bar, and full digest coverage ----

    /// **THE BYTE-IDENTITY BAR for IO schemas.** A registry whose types carry
    /// no schema — every registry that predates the wiring layer — must digest
    /// exactly as before, against the same pinned constants as
    /// [`boolean_only_digest_is_byte_identical`]. An explicitly EMPTY schema is
    /// the same registry as no schema at all: the payload appends only when
    /// non-empty.
    #[test]
    fn schemaless_digest_is_byte_identical() {
        let plain = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        let empty = Registry::new(
            vec![
                ty(10, 700).with_io_schema(Vec::new()),
                ty(9, 300).with_io_schema(Vec::new()),
            ],
            3,
        );
        assert_eq!(hex(&plain.digest()), hex(&empty.digest()));
        assert_eq!(
            hex(&plain.digest()),
            "8010ecf651ca43eadfe415eac6a081c8f9022dd077070da0f22cea19297699ea",
            "an empty IO schema must absorb nothing"
        );
    }

    /// Every part of a schema moves the digest: its presence, the entry count,
    /// a word-column, a direction, and the ORDER of the entries (which is the
    /// type's contribution to the cell-slot enumeration, so it must bind).
    #[test]
    fn io_schema_binds_the_digest() {
        let mk = |schema: Vec<IoWord>| {
            Registry::new(vec![ty(10, 700).with_io_schema(schema), ty(9, 300)], 3).digest()
        };
        let base = mk(vec![IoWord::input(0), IoWord::output(2)]);
        let cases = [
            ("presence", mk(Vec::new())),
            ("entry count", mk(vec![IoWord::input(0)])),
            ("word column", mk(vec![IoWord::input(1), IoWord::output(2)])),
            ("direction", mk(vec![IoWord::output(0), IoWord::output(2)])),
            ("order", mk(vec![IoWord::output(2), IoWord::input(0)])),
        ];
        for (what, d) in cases {
            assert_ne!(base, d, "digest insensitive to {what}");
        }
        // …and which TYPE carries it: the same schema on the other type.
        let other = Registry::new(
            vec![
                ty(10, 700),
                ty(9, 300).with_io_schema(vec![IoWord::input(0), IoWord::output(2)]),
            ],
            3,
        );
        assert_ne!(
            base,
            other.digest(),
            "digest insensitive to the schema's type"
        );
    }

    /// The schema's word-columns are validated against the type's real width,
    /// and no word may be named twice.
    #[test]
    #[should_panic(expected = "outside the type's")]
    fn io_schema_word_column_out_of_range() {
        // useful_bits = 300 → ceil(300/128) = 3 used word-columns.
        Registry::new(vec![ty(9, 300).with_io_schema(vec![IoWord::input(3)])], 3);
    }

    #[test]
    #[should_panic(expected = "twice")]
    fn io_schema_duplicate_word_column() {
        Registry::new(
            vec![ty(9, 300).with_io_schema(vec![IoWord::input(1), IoWord::output(1)])],
            3,
        );
    }

    /// An element type moves the digest, and every component of its base
    /// block is covered (kappa/k through the header, matrices and constants
    /// through [`ElementTableType::digest`]).
    #[test]
    fn element_payload_binds_the_digest() {
        use crate::element_r1cs::ElementTableBuilder;
        use crate::field::F128;

        let bool_only = Registry::new(vec![ty(10, 700)], 3);
        let mixed = Registry::new(vec![ty(10, 700), elem_ty(3)], 3);
        assert_ne!(
            bool_only.digest(),
            mixed.digest(),
            "adding an element type must move the digest"
        );

        // Two element types with the SAME (k_log, useful_bits) header but
        // different base blocks: only the appended element digest separates
        // them, which is exactly what the suffix is for.
        let mk = |scale: u64| {
            let mut b = ElementTableBuilder::new(2);
            b.free_wire(0).free_wire(1).free_wire(2);
            b.linear(3, &[(0, F128::new(scale, 0))]);
            Registry::new(
                vec![
                    ty(10, 700),
                    TableType::element(Arc::new(b.build().expect("valid"))),
                ],
                3,
            )
        };
        assert_ne!(
            mk(5).digest(),
            mk(7).digest(),
            "element matrix coefficients must bind"
        );
        assert_eq!(mk(5).digest(), mk(5).digest(), "deterministic");
    }

    /// Class-major sort: boolean slots are a prefix, element slots the
    /// suffix, each area-descending — even when the element type is WIDER
    /// than a boolean one (so a pure `k_log` sort would interleave them).
    #[test]
    fn class_major_sort_puts_booleans_first() {
        // Element κ = 7 → k_log = 14; boolean k_logs 12 and 10. A pure
        // area-descending sort would put the element type first.
        let reg = Registry::new(vec![elem_ty(7), ty(10, 700), ty(12, 4000)], 2);
        let shape: Vec<(usize, bool)> = reg
            .types()
            .iter()
            .map(|t| (t.k_log, t.is_element()))
            .collect();
        assert_eq!(shape, vec![(12, false), (10, false), (14, true)]);
        assert_eq!(reg.num_boolean(), 2);
        assert_eq!(reg.num_element(), 1);
        assert_eq!(reg.boolean_types().len(), 2);
        assert!(reg.element_types()[0].is_element());
    }

    /// Region arithmetic, hand-computed on the three shapes that matter:
    /// boolean-only (unchanged from today), element-only (the element region
    /// IS the whole space, no wasted half), and mixed (disjoint aligned
    /// subcubes with the boolean region a prefix subcube).
    #[test]
    fn class_regions_are_disjoint_aligned_subcubes() {
        // (a) Boolean-only: M = M_bool, no element region — today's geometry.
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300)], 3);
        assert_eq!((reg.m_total(), reg.m_bool(), reg.m_elem()), (14, 14, 0));
        assert_eq!(reg.num_element(), 0);

        // (b) Element-only: κ = 3 → k_log 10, ν = 3 → area 2^13, M_elem = 13,
        // base 0, M = 13. The element region is the whole prefix subcube.
        let reg = Registry::new(vec![elem_ty(3)], 3);
        assert_eq!((reg.m_total(), reg.m_bool(), reg.m_elem()), (13, 0, 13));
        assert_eq!(reg.element_base(), 0);
        assert_eq!(reg.slots()[0].offset, 0);

        // (c) Mixed, boolean-dominant: boolean areas 2^13 + 2^12 = 0x3000 →
        // M_bool = 14; element area 2^13 → M_elem = 13. Base rounds the
        // boolean SUBCUBE end (2^14) up to a multiple of 2^13 = 2^14, so
        // there is a virtual gap [0x3000, 0x4000) and M = 15.
        let reg = Registry::new(vec![ty(10, 700), ty(9, 300), elem_ty(3)], 3);
        assert_eq!((reg.m_total(), reg.m_bool(), reg.m_elem()), (15, 14, 13));
        assert_eq!(reg.element_base(), 1 << 14);
        assert_eq!(
            reg.slots(),
            &[
                Slot {
                    m_slot: 13,
                    offset: 0,
                    prefix: 0b00,
                    prefix_bits: 2
                },
                Slot {
                    m_slot: 12,
                    offset: 1 << 13,
                    prefix: 0b010,
                    prefix_bits: 3
                },
                Slot {
                    m_slot: 13,
                    offset: 1 << 14,
                    prefix: 0b10,
                    prefix_bits: 2
                },
            ]
        );

        // (d) Mixed, element-dominant: boolean area 2^10 → M_bool = 10;
        // element areas 2^13 + 2^12 → M_elem = 14. The base is the boolean
        // subcube end rounded up to 2^14, so M = 15 and the element region
        // is [2^14, 2^15).
        let reg = Registry::new(vec![ty(7, 128), elem_ty(3), elem_ty(2)], 3);
        assert_eq!((reg.m_total(), reg.m_bool(), reg.m_elem()), (15, 10, 14));
        assert_eq!(reg.element_base(), 1 << 14);
        assert_eq!(reg.slots()[1].offset, 1 << 14);
        assert_eq!(reg.slots()[2].offset, (1 << 14) + (1 << 13));

        // The invariants, spelled out for every mixed shape above.
        for reg in [
            Registry::new(vec![ty(10, 700), ty(9, 300), elem_ty(3)], 3),
            Registry::new(vec![ty(7, 128), elem_ty(3), elem_ty(2)], 3),
            Registry::new(vec![elem_ty(3), elem_ty(3)], 4),
        ] {
            let nb = reg.num_boolean();
            // Boolean region is the prefix subcube [0, 2^M_bool).
            for slot in &reg.slots()[..nb] {
                assert!(slot.offset + slot.area() <= 1usize << reg.m_bool());
            }
            // Element region is an aligned subcube disjoint from it.
            let base = reg.element_base();
            let top = base + (1usize << reg.m_elem());
            assert!(nb == 0 || base >= 1usize << reg.m_bool(), "regions overlap");
            assert!(top <= 1usize << reg.m_total());
            assert!(base.is_multiple_of(1usize << reg.m_elem()));
            for slot in &reg.slots()[nb..] {
                assert!(slot.offset >= base && slot.offset + slot.area() <= top);
                assert!(slot.offset.is_multiple_of(slot.area()));
            }
        }
    }

    /// An element type's presented shape: `k_log = kappa + 7`,
    /// `useful_bits = k · 128`, no const pin — the three facts the slot
    /// bookkeeping (`used_cols`, heights, `padding_spec`) reads.
    #[test]
    fn element_type_presents_word_geometry() {
        use crate::element_r1cs::ElementTableBuilder;
        let mut b = ElementTableBuilder::new(3); // width 8
        b.free_wire(0).free_wire(1).mult(2, 0, 1); // k = 3 real columns
        let el = Arc::new(b.build().unwrap());
        let ty = TableType::element(el);
        assert_eq!(ty.k_log, 3 + 7);
        assert_eq!(ty.useful_bits, 3 * 128);
        assert_eq!(ty.const_pin, None);
        assert!(ty.is_element());
        assert_eq!(ty.element_type().map(|e| e.kappa()), Some(3));
        // 8 word-columns in the slot, 3 of them used.
        assert_eq!(1usize << (ty.k_log - 7), 8);
        assert_eq!(ty.useful_bits.div_ceil(128), 3);
    }

    #[test]
    #[should_panic(expected = "MAX_K_LOG")]
    fn registry_rejects_absurd_k_log() {
        let _ = Registry::new(vec![ty(MAX_K_LOG + 1, 128)], 1);
    }
}
