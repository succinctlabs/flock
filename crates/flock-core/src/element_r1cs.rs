//! Large-field (element-level) R1CS: a table class whose witness entries are
//! **F128 elements** — one field element per variable, one committed word per
//! variable — instead of GF(2) bits.
//!
//! The point of the class is arithmetic density. In the bit-level tables a
//! single F128 multiplication costs ~2187 constraints; here it costs one. The
//! price is that every variable occupies a full 128-bit word, so element tables
//! pay for the wires they use rather than for the bits they touch (see
//! `docs/local/recursion-verifier-map.md` §4.2).
//!
//! ## The relation
//!
//! A table type is a **base block** over F128 = GF(2^128) (GHASH modulus,
//! char 2 — so subtraction *is* addition). With `k` witness columns per row
//! padded to `2^kappa`, sparse matrices `A_0, B_0 ∈ F128^{2^kappa × 2^kappa}`
//! and affine constant vectors `a_const, b_const ∈ F128^{2^kappa}` (part of the
//! statement, not the witness), every row `j` and column `y` must satisfy
//!
//! ```text
//! (A_0[y]·z_j + a_const[y]) · (B_0[y]·z_j + b_const[y]) = z_j[y]
//! ```
//!
//! **C is the identity**: the constraint domain *is* the column domain, one
//! constraint per column. That is what lets the zerocheck's C-claim go straight
//! out as a witness evaluation claim with no lincheck term.
//!
//! The type constructor enforces `a_const[y] · b_const[y] = 0` for every `y`
//! (disjoint supports). That is what makes an all-zero row satisfying —
//! `(0 + a_const)(0 + b_const) = 0 = z_y` — so dummy/padding rows are
//! definitionally satisfying, zero-contributing, and consistent with the jagged
//! "dropped words are zero" convention.
//!
//! ## Witness layout
//!
//! BatchMajor at **word** level with rows in the LOW bits: the committed word
//! index of (column `c`, row `j`) is `(c << n_log) + j`. There is no in-word
//! packing structure — the element index *is* the packed-word index — so the
//! committed polynomial has `m_words = kappa + n_log` variables. The rows-low
//! convention is load-bearing for the future wiring layer.
//!
//! Because the full system is `I_{2^n_log} ⊗ A_0` (block diagonal per row), the
//! MLEs factor as `M̂((x_row,x_con),(y_row,y_col)) = eq(x_row,y_row)·M̂_0(x_con,y_col)`,
//! and the constant vectors — uniform across rows — collapse by partition of
//! unity: `â_const(r_row, r_con) = â_const_base(r_con)`, with no row and no
//! count dependence.
//!
//! ## Protocol
//!
//! Spartan-style, all in the large field:
//!
//! 1. [`zerocheck`] — a plain eq-weighted degree-3 sumcheck over
//!    `n_log + kappa` variables proving
//!    `Σ_x eq(τ,x)·((Az+a_const)(x)·(Bz+b_const)(x) + z(x)) = 0`. No univariate
//!    skip, no packing, no φ8. Outputs `ea`, `eb` (which Phase 2 reduces) and
//!    `ec = ẑ(r)` (a witness claim already).
//! 2. [`lincheck`] — one degree-2 sumcheck batching `ea`/`eb` into a single
//!    witness claim `ẑ(r')`.
//! 3. [`prove`] / [`verify`] — commit, bind the statement, run both phases, and
//!    open `ec` and `ẑ(r')` as **packed-direct** claims. No ring-switching
//!    anywhere: the witness words already are field elements, so there is no
//!    bit-MLE ↔ packed-MLE bridge to cross.
//!
//! Fiat–Shamir order: commit → bind statement → τ → zerocheck rounds → α →
//! lincheck rounds → γ-batched opening.

pub mod lincheck;
pub mod union;
pub mod zerocheck;

use std::sync::OnceLock;

use crate::challenger::Challenger;
use crate::field::F128;
use crate::merkle::Hash;
use crate::pcs::ligerito::{ProverConfig, VerifierConfig};
use crate::pcs::ring_switch::build_eq_parallel;
use crate::pcs::{
    self, Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
    commit,
};
use crate::zerocheck::PaddingSpec;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Statement-binding domain label. Absorbed before any challenge is squeezed.
const DOMAIN: &[u8] = b"flock-element-r1cs-v0";

// ---------------------------------------------------------------------------
// Sparse F128 matrix
// ---------------------------------------------------------------------------

/// Sparse matrix over F128. `rows[i]` lists the `(col, coeff)` entries of row
/// `i`; coefficients are non-zero and column indices within a row are distinct
/// (both enforced by [`SparseF128Matrix::from_rows`], so the canonical form is
/// unique up to the order of a row's entries).
///
/// Row `i` is read as the linear form `M[i]·v = Σ_(c,w) w·v[c]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseF128Matrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub rows: Vec<Vec<(usize, F128)>>,
}

impl SparseF128Matrix {
    /// The all-zero `num_rows × num_cols` matrix.
    pub fn zeros(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            rows: vec![Vec::new(); num_rows],
        }
    }

    /// Validating constructor. `which` names the matrix in any error.
    pub fn from_rows(
        which: &'static str,
        num_cols: usize,
        rows: Vec<Vec<(usize, F128)>>,
    ) -> Result<Self, TypeError> {
        let m = Self {
            num_rows: rows.len(),
            num_cols,
            rows,
        };
        m.validate(which)?;
        Ok(m)
    }

    fn validate(&self, which: &'static str) -> Result<(), TypeError> {
        for (row, entries) in self.rows.iter().enumerate() {
            for (i, &(col, coeff)) in entries.iter().enumerate() {
                if col >= self.num_cols {
                    return Err(TypeError::ColumnOutOfRange { which, row, col });
                }
                if coeff.is_zero() {
                    return Err(TypeError::ZeroCoefficient { which, row, col });
                }
                if entries[..i].iter().any(|&(c, _)| c == col) {
                    return Err(TypeError::DuplicateColumn { which, row, col });
                }
            }
        }
        Ok(())
    }

    /// `M[row] · v` for a length-`num_cols` slice.
    pub fn row_dot(&self, row: usize, v: &[F128]) -> F128 {
        debug_assert_eq!(v.len(), self.num_cols);
        let mut acc = F128::ZERO;
        for &(col, coeff) in &self.rows[row] {
            acc += coeff * v[col];
        }
        acc
    }

    /// Absorb the matrix into a statement digest, length-prefixed per row so no
    /// two distinct matrices share an encoding. Mirrors
    /// `crate::r1cs::absorb_matrix` with F128 coefficients appended.
    fn absorb(&self, h: &mut blake3::Hasher) {
        h.update(&(self.num_rows as u64).to_le_bytes());
        h.update(&(self.num_cols as u64).to_le_bytes());
        for row in &self.rows {
            h.update(&(row.len() as u64).to_le_bytes());
            for &(col, coeff) in row {
                h.update(&(col as u64).to_le_bytes());
                h.update(&coeff.lo.to_le_bytes());
                h.update(&coeff.hi.to_le_bytes());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Table type
// ---------------------------------------------------------------------------

/// Why an [`ElementTableType`] could not be constructed. Every variant is a
/// *statement* defect — caught once at construction, never at proving time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeError {
    /// A matrix is not `2^kappa × 2^kappa`.
    MatrixShape {
        which: &'static str,
        num_rows: usize,
        num_cols: usize,
        expected: usize,
    },
    /// A constant vector is not length `2^kappa`.
    ConstLen {
        which: &'static str,
        got: usize,
        expected: usize,
    },
    /// `k` real columns exceeds the padded width `2^kappa`.
    TooManyColumns { k: usize, kappa: usize },
    /// A matrix row references a column outside `[0, 2^kappa)`.
    ColumnOutOfRange {
        which: &'static str,
        row: usize,
        col: usize,
    },
    /// A matrix row lists the same column twice.
    DuplicateColumn {
        which: &'static str,
        row: usize,
        col: usize,
    },
    /// An explicitly-stored zero coefficient — the canonical sparse form keeps
    /// non-zeros only, so this would make the type digest ambiguous.
    ZeroCoefficient {
        which: &'static str,
        row: usize,
        col: usize,
    },
    /// The validity rule `a_const[y] · b_const[y] = 0` is violated at `y`.
    /// Without disjoint supports an all-zero row does NOT satisfy the relation,
    /// so dummy/padding rows would be unsatisfiable and the jagged
    /// "dropped words are zero" convention would break.
    ConstantsOverlap { y: usize },
    /// A padding column `y ≥ k` does not carry the all-zero constraint row
    /// (`A_0[y] = B_0[y] = 0`, `a_const[y] = b_const[y] = 0`) that pins
    /// `z_y = 0`. Self-enforcing zero padding is the declared convention for
    /// the columns past the `k` real ones.
    PaddingRowNotZero { y: usize },
}

/// A large-field table type: the base block of the relation in the module docs,
/// plus the padded width and the count `k` of real columns.
///
/// Fields are private so the construction-time invariants (shape, disjoint
/// constant supports, zero padding rows) hold for the whole lifetime of the
/// value — the prover and verifier both rely on them.
#[derive(Debug)]
pub struct ElementTableType {
    kappa: usize,
    k: usize,
    a_0: SparseF128Matrix,
    b_0: SparseF128Matrix,
    a_const: Vec<F128>,
    b_const: Vec<F128>,
    digest_cache: OnceLock<[u8; 32]>,
}

impl ElementTableType {
    /// Validating constructor. `k` is the number of real columns; columns
    /// `[k, 2^kappa)` must carry all-zero rows (self-enforcing zero padding).
    pub fn new(
        kappa: usize,
        k: usize,
        a_0: SparseF128Matrix,
        b_0: SparseF128Matrix,
        a_const: Vec<F128>,
        b_const: Vec<F128>,
    ) -> Result<Self, TypeError> {
        let width = 1usize << kappa;
        for (which, m) in [("a_0", &a_0), ("b_0", &b_0)] {
            if m.num_rows != width || m.num_cols != width {
                return Err(TypeError::MatrixShape {
                    which,
                    num_rows: m.num_rows,
                    num_cols: m.num_cols,
                    expected: width,
                });
            }
            m.validate(which)?;
        }
        for (which, v) in [("a_const", &a_const), ("b_const", &b_const)] {
            if v.len() != width {
                return Err(TypeError::ConstLen {
                    which,
                    got: v.len(),
                    expected: width,
                });
            }
        }
        if k > width {
            return Err(TypeError::TooManyColumns { k, kappa });
        }
        // The validity rule. Checked over the FULL padded width, not just the
        // real columns: the zerocheck sums over every column of every row.
        for y in 0..width {
            if !(a_const[y] * b_const[y]).is_zero() {
                return Err(TypeError::ConstantsOverlap { y });
            }
        }
        for y in k..width {
            if !a_0.rows[y].is_empty()
                || !b_0.rows[y].is_empty()
                || !a_const[y].is_zero()
                || !b_const[y].is_zero()
            {
                return Err(TypeError::PaddingRowNotZero { y });
            }
        }
        Ok(Self {
            kappa,
            k,
            a_0,
            b_0,
            a_const,
            b_const,
            digest_cache: OnceLock::new(),
        })
    }

    /// log2 of the padded column count.
    pub fn kappa(&self) -> usize {
        self.kappa
    }
    /// Padded column count `2^kappa` — the width of one row's witness.
    pub fn width(&self) -> usize {
        1usize << self.kappa
    }
    /// Number of real (non-padding) columns.
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn a_0(&self) -> &SparseF128Matrix {
        &self.a_0
    }
    pub fn b_0(&self) -> &SparseF128Matrix {
        &self.b_0
    }
    /// The affine constant vector the spec calls `a0`.
    pub fn a_const(&self) -> &[F128] {
        &self.a_const
    }
    /// The affine constant vector the spec calls `b0`.
    pub fn b_const(&self) -> &[F128] {
        &self.b_const
    }

    /// Statement digest over the whole base block.
    ///
    /// Absorbs, in order: the domain tag `b"flock-element-type-v0"` (distinct
    /// from the bit-level `b"flock-registry-v1"` / `b"flock-r1cs-stmt-v1"`, so
    /// an element digest can never collide with a boolean one), a format
    /// version byte, `kappa` and `k` as u32 LE, the two matrices via
    /// [`SparseF128Matrix::absorb`], then the two constant vectors
    /// length-prefixed. Lazily cached.
    pub fn digest(&self) -> [u8; 32] {
        *self.digest_cache.get_or_init(|| {
            let mut h = blake3::Hasher::new();
            h.update(b"flock-element-type-v0");
            h.update(&[0u8]);
            h.update(&(self.kappa as u32).to_le_bytes());
            h.update(&(self.k as u32).to_le_bytes());
            self.a_0.absorb(&mut h);
            self.b_0.absorb(&mut h);
            for v in [&self.a_const, &self.b_const] {
                h.update(&(v.len() as u64).to_le_bytes());
                for e in v {
                    h.update(&e.lo.to_le_bytes());
                    h.update(&e.hi.to_le_bytes());
                }
            }
            *h.finalize().as_bytes()
        })
    }

    /// `Az` and `Bz` for a BatchMajor witness (`z[(c << n_log) + j]`), by sparse
    /// gather: one pass per stored matrix entry per row, i.e. `O(nnz · 2^n_log)`
    /// with no matrix application on any hot path. For a mult gate the outputs
    /// are literally the operand values.
    ///
    /// Output layout matches the input: `az[(y << n_log) + j] = A_0[y]·z_j`.
    pub fn apply(&self, z: &[F128], n_log: usize) -> (Vec<F128>, Vec<F128>) {
        assert_eq!(z.len(), self.width() << n_log, "witness length");
        // `gather_into` seeds every slot before accumulating, so uninitialized
        // is fine here — no memset tax.
        let mut az = crate::alloc_uninit_vec::<F128>(z.len());
        let mut bz = crate::alloc_uninit_vec::<F128>(z.len());
        gather_into(&self.a_0, z, n_log, None, None, &mut az);
        gather_into(&self.b_0, z, n_log, None, None, &mut bz);
        (az, bz)
    }

    /// `pa = A_0·z + a_const` and `pb = B_0·z + b_const` written into
    /// caller-supplied buffers — the two tables the zerocheck consumes, in one
    /// pass each and with no allocation. Same sparse gather as [`Self::apply`]
    /// (one shared kernel, so the two cannot drift), with the row-uniform
    /// constant seeded into the accumulator instead of broadcast afterwards;
    /// F128 addition is XOR, so the result is bit-identical to
    /// `apply` + [`broadcast_add`].
    ///
    /// This is the in-place path the union's element witness assembly uses:
    /// the destinations are slices of the padded union `a`/`b` buffers.
    ///
    /// `live = Some(n)` writes only rows `[0, n)` per column — the relation is
    /// row-diagonal, so the live prefix is exact — and leaves the dead rows
    /// UNWRITTEN (they hold buffer background, not the constants the full pass
    /// writes). Sound only when nothing reads them: the region zerocheck's
    /// sparse path substitutes dead halves from `RowSupport::{a,b}_dead`
    /// analytically. Callers gate on `element_r1cs::union::dead_rows_unread`.
    pub fn affine_products_into(
        &self,
        z: &[F128],
        n_log: usize,
        live: Option<usize>,
        pa: &mut [F128],
        pb: &mut [F128],
    ) {
        assert_eq!(z.len(), self.width() << n_log, "witness length");
        assert_eq!(pa.len(), z.len(), "pa length");
        assert_eq!(pb.len(), z.len(), "pb length");
        gather_into(&self.a_0, z, n_log, Some(&self.a_const), live, pa);
        gather_into(&self.b_0, z, n_log, Some(&self.b_const), live, pb);
    }

    /// Brute-force check that every row `j < n` satisfies the relation and that
    /// rows `[n, 2^n_log)` are honestly zero. Reference for the tests, and the
    /// contract [`prove`] assumes of its caller.
    pub fn satisfies(&self, z: &[F128], n_log: usize, n: usize) -> bool {
        let width = self.width();
        let rows = 1usize << n_log;
        if z.len() != width << n_log || n > rows {
            return false;
        }
        let row_of = |j: usize| -> Vec<F128> { (0..width).map(|c| z[(c << n_log) + j]).collect() };
        for j in 0..rows {
            let zj = row_of(j);
            if j >= n && zj.iter().any(|e| !e.is_zero()) {
                return false;
            }
            for y in 0..width {
                let lhs = (self.a_0.row_dot(y, &zj) + self.a_const[y])
                    * (self.b_0.row_dot(y, &zj) + self.b_const[y]);
                if lhs != zj[y] {
                    return false;
                }
            }
        }
        true
    }
}

/// The shared sparse-gather kernel: `out[(y << n_log) + j] = M[y]·z_j + c[y]`,
/// one pass per stored matrix entry per row (`O(nnz · 2^n_log)`, no matrix
/// application on any hot path). `c = None` means no constant. Parallel over
/// the output's `2^n_log`-word column chunks, which are disjoint.
/// `live = Some(n)` writes rows `[0, n)` only — exact per row (the relation is
/// row-diagonal), dead rows left unwritten (see [`ElementTableType::
/// affine_products_into`] for when that is sound).
fn gather_into(
    m: &SparseF128Matrix,
    z: &[F128],
    n_log: usize,
    c: Option<&[F128]>,
    live: Option<usize>,
    out: &mut [F128],
) {
    let rows = 1usize << n_log;
    let n = live.map_or(rows, |l| l.min(rows));
    debug_assert_eq!(out.len(), m.num_rows << n_log);
    out.par_chunks_mut(rows).enumerate().for_each(|(y, dst)| {
        let seed = c.map_or(F128::ZERO, |c| c[y]);
        let dst = &mut dst[..n];
        dst.fill(seed);
        for &(col, coeff) in &m.rows[y] {
            let src = &z[col << n_log..(col << n_log) + n];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += coeff * *s;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Row-encoding builder
// ---------------------------------------------------------------------------

/// Incremental builder for the standard row encodings. Starts from the all-zero
/// block of width `2^kappa` (every column self-pinned to zero) and lets each
/// gate claim its output column.
///
/// `k` — the real-column count — is `1 + max(column touched)`, so the untouched
/// tail keeps its all-zero rows and is therefore *padding* in the
/// [`ElementTableType::new`] sense.
///
/// This is deliberately a handful of test gates, not a gate library.
#[derive(Clone, Debug)]
pub struct ElementTableBuilder {
    kappa: usize,
    a_rows: Vec<Vec<(usize, F128)>>,
    b_rows: Vec<Vec<(usize, F128)>>,
    a_const: Vec<F128>,
    b_const: Vec<F128>,
    k: usize,
}

impl ElementTableBuilder {
    pub fn new(kappa: usize) -> Self {
        let width = 1usize << kappa;
        Self {
            kappa,
            a_rows: vec![Vec::new(); width],
            b_rows: vec![Vec::new(); width],
            a_const: vec![F128::ZERO; width],
            b_const: vec![F128::ZERO; width],
            k: 0,
        }
    }

    fn touch(&mut self, y: usize) {
        assert!(y < 1usize << self.kappa, "column {y} exceeds 2^kappa");
        self.k = self.k.max(y + 1);
        self.a_rows[y].clear();
        self.b_rows[y].clear();
        self.a_const[y] = F128::ZERO;
        self.b_const[y] = F128::ZERO;
    }

    /// Multiplication `z_out = z_a · z_b`: `A_0[out] = e_a`, `B_0[out] = e_b`,
    /// both constants zero.
    pub fn mult(&mut self, out: usize, a: usize, b: usize) -> &mut Self {
        self.touch(out);
        self.a_rows[out] = vec![(a, F128::ONE)];
        self.b_rows[out] = vec![(b, F128::ONE)];
        self
    }

    /// Free wire — an input constrained only by future wiring. The tautology row
    /// `(z_y)(1) = z_y`: `A_0[y] = e_y`, `B_0[y] = 0`, `b_const[y] = 1`.
    pub fn free_wire(&mut self, y: usize) -> &mut Self {
        self.touch(y);
        self.a_rows[y] = vec![(y, F128::ONE)];
        self.b_const[y] = F128::ONE;
        self
    }

    /// Linear constraint pinning a linear combination to a wire:
    /// `(Σ w·z_c)(1) = z_y`. `terms` must have distinct, non-zero-weighted
    /// columns.
    pub fn linear(&mut self, y: usize, terms: &[(usize, F128)]) -> &mut Self {
        self.touch(y);
        self.a_rows[y] = terms.to_vec();
        self.b_const[y] = F128::ONE;
        self
    }

    /// Multiply-accumulate `z_out = z_a·z_b + z_addend`, spelled as the two rows
    /// the relation shape allows: a [`Self::mult`] into `tmp`, then a
    /// [`Self::linear`] summing `tmp` and `addend` into `out`. (One row cannot
    /// do it: the right-hand side of a row is exactly one column.)
    /// `z[out] = (Σ aᵢ·z[..]) · (Σ bⱼ·z[..])` — a product of two **linear
    /// combinations**, in one constraint.
    ///
    /// `A_0` and `B_0` are matrix *rows*, so a sum on either side of a product
    /// is free: it rides the row of the multiplication that consumes it rather
    /// than costing a column of its own. That matters because in this class an
    /// addition is not free — every committed column is the output of exactly
    /// one row, `linear` ones included — so `(a+b)·c` written as an add then a
    /// mult costs two constraints where this costs one.
    ///
    /// [`Self::mult`] is the special case of one term per side.
    pub fn mult_lin(&mut self, out: usize, a: &[(usize, F128)], b: &[(usize, F128)]) -> &mut Self {
        self.touch(out);
        self.a_rows[out] = a.to_vec();
        self.b_rows[out] = b.to_vec();
        self
    }

    pub fn mult_acc(
        &mut self,
        out: usize,
        a: usize,
        b: usize,
        addend: usize,
        tmp: usize,
    ) -> &mut Self {
        self.mult(tmp, a, b);
        self.linear(out, &[(tmp, F128::ONE), (addend, F128::ONE)])
    }

    /// Finish, validating every invariant of [`ElementTableType::new`].
    pub fn build(self) -> Result<ElementTableType, TypeError> {
        let width = 1usize << self.kappa;
        ElementTableType::new(
            self.kappa,
            self.k,
            SparseF128Matrix::from_rows("a_0", width, self.a_rows)?,
            SparseF128Matrix::from_rows("b_0", width, self.b_rows)?,
            self.a_const,
            self.b_const,
        )
    }
}

// ---------------------------------------------------------------------------
// Statement
// ---------------------------------------------------------------------------

/// The public statement of one element table: the type, the row capacity
/// `2^n_log`, and the declared count `n ≤ 2^n_log` of real rows.
#[derive(Clone, Copy, Debug)]
pub struct ElementStatement<'a> {
    pub ty: &'a ElementTableType,
    pub n_log: usize,
    /// Declared count of real rows. Transcript-bound (so a proof does not
    /// transfer between counts), but note the scope boundary: this milestone's
    /// PIOP proves the *relation* over the whole padded row domain and does not
    /// independently prove that rows `[n, 2^n_log)` are **zero**. That costs
    /// nothing standalone — the relation holds on every row, so a proof at count
    /// `n` is equally a proof at full capacity — and it becomes structural in
    /// the union integration, where the committed height is count-derived and
    /// the dummy rows are not committed at all. Pinned by
    /// `e2e_tests::satisfying_dummy_row_is_not_detected`.
    pub n: usize,
}

impl ElementStatement<'_> {
    /// Committed word-variable count `m_words = kappa + n_log`.
    pub fn m_words(&self) -> usize {
        self.ty.kappa() + self.n_log
    }

    /// Total committed words `2^m_words`.
    pub fn n_words(&self) -> usize {
        1usize << self.m_words()
    }

    /// Absorb label, type digest, capacity, count and commitment cap — the
    /// whole statement — BEFORE any challenge is squeezed. Prover and verifier
    /// call this at the same transcript position.
    fn bind<C: Challenger>(&self, cap: &[Hash], ch: &mut C) {
        ch.observe_label(DOMAIN);
        ch.observe_bytes(&self.ty.digest());
        ch.observe_bytes(&(self.n_log as u64).to_le_bytes());
        ch.observe_bytes(&(self.n as u64).to_le_bytes());
        ch.observe_bytes(cap.as_flattened());
    }
}

// ---------------------------------------------------------------------------
// Commitment parameters
// ---------------------------------------------------------------------------

/// RS inverse rate (log2) and interleaving batch size (log2) for the element
/// witness commitment. Both backends' L0 commit and Ligerito's `default_config`
/// must agree on these, so they live in one place.
const PCS_LOG_INV_RATE: usize = 1;
const PCS_LOG_BATCH_SIZE: usize = 1;

/// PCS parameters for a witness of `m_words` F128 words.
///
/// `PcsParams::m` is the **bit**-level variable count, so it is
/// `m_words + LOG_PACKING`; for an element table that offset is pure
/// bookkeeping (there is no real in-word packing — the element index *is* the
/// packed-word index), and it makes `log_msg_len() == m_words` so the
/// packed-direct opening points have length `m_words`. Deterministic in
/// `m_words`, so the verifier rebuilds these from the statement and the proof
/// carries only the root.
fn pcs_params(m_words: usize, grinding: Grinding) -> PcsParams {
    PcsParams {
        m: m_words + LOG_PACKING,
        log_inv_rate: PCS_LOG_INV_RATE,
        log_batch_size: PCS_LOG_BATCH_SIZE,
        profile: if grinding.enabled {
            pcs::ligerito::LigeritoProfile::Secure
        } else {
            Default::default()
        },
        num_lanes: None,
        merkle_hash: Default::default(),
    }
}

/// Smallest `m_words` Ligerito's recursion can open at the parameters above
/// (the L0 block must be at least `udr_queries(1) = 243` wide). Queries are
/// sampled with replacement, so that is a ladder-shape convention rather than
/// a soundness requirement — but it is the one `default_config` enforces.
/// Below it [`prove`] cannot run; the PIOP phases themselves have no such
/// floor.
pub const MIN_M_WORDS: usize = 8;

fn ligerito_prover_config(m_words: usize) -> ProverConfig {
    pcs::ligerito::default_config(m_words, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito config for the element witness; requires m_words >= MIN_M_WORDS")
}

fn ligerito_verifier_config(m_words: usize) -> VerifierConfig {
    pcs::ligerito::default_verifier_config(m_words, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito verifier config for the element witness; requires m_words >= MIN_M_WORDS")
}

// ---------------------------------------------------------------------------
// End-to-end proof
// ---------------------------------------------------------------------------

/// Fiat--Shamir grinding policy for the element/dense PIOP.
///
/// The initial equality point is a polynomial of total degree at most the
/// number of word variables. Each Convention-A zerocheck and product-lincheck
/// round is degree two, while the lincheck batching scalar is linear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grinding {
    enabled: bool,
}

impl Grinding {
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }

    pub const fn per_challenge_128() -> Self {
        Self { enabled: true }
    }

    /// `ceil(log2(m_words + 1))`, enough that
    /// `m_words / (2^bits |F|) < 2^-128` for `m_words >= 1`.
    pub fn initial_bits(self, m_words: usize) -> Option<u32> {
        self.enabled
            .then_some(usize::BITS - m_words.leading_zeros())
    }

    pub fn round_bits(self) -> Option<u32> {
        self.enabled.then_some(2)
    }

    pub fn alpha_bits(self) -> Option<u32> {
        self.enabled.then_some(1)
    }

    pub fn zerocheck_nonce_count(self, m_words: usize) -> usize {
        usize::from(self.initial_bits(m_words).is_some())
            + usize::from(self.round_bits().is_some()) * m_words
    }

    pub fn lincheck_nonce_count(self, rounds: usize) -> usize {
        usize::from(self.alpha_bits().is_some()) + usize::from(self.round_bits().is_some()) * rounds
    }
}

/// A standalone single-table element proof: the witness commitment root, the two
/// PIOP phases, and one batched opening of both output claims.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementProof {
    /// Merkle cap layer of the committed witness words (the commitment).
    pub cap: Vec<Hash>,
    pub zerocheck: zerocheck::Proof,
    pub lincheck: lincheck::Proof,
    /// Packed-direct opening of `ec = ẑ(r)` and `ẑ(r')` — the mixed open with
    /// zero ring-switched claims.
    pub open: pcs::BatchOpeningProofLigerito,
}

/// Why an element proof was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElementR1csError {
    /// `m_words` is below Ligerito's feasibility floor ([`MIN_M_WORDS`]).
    TooSmall {
        m_words: usize,
        min: usize,
    },
    /// The declared count exceeds the row capacity.
    CountExceedsCapacity {
        n: usize,
        n_log: usize,
    },
    Zerocheck(zerocheck::ElementZerocheckError),
    Lincheck(lincheck::ElementLincheckError),
    /// The packed-direct opening rejected.
    Open(pcs::PcsError),
}

/// The claims a verified element proof leaves behind: the two witness
/// evaluation points and values, both already discharged by the opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementClaim {
    /// Zerocheck point `r = (r_row, r_con)`, LSB-first (rows low).
    pub r: Vec<F128>,
    /// `ẑ(r)` — the C-claim, direct because C is the identity.
    pub ec: F128,
    /// Lincheck point `r'`, LSB-first.
    pub r_prime: Vec<F128>,
    /// `ẑ(r')`.
    pub z_eval: F128,
}

/// Prove that `z` satisfies `stmt`.
///
/// `z` is the BatchMajor witness (`z[(c << n_log) + j]`, length `2^m_words`)
/// with rows `[n, 2^n_log)` all zero. Committed at full height — dummy rows are
/// zero, so this is honest; count-proportional/jagged heights are the union
/// integration's job.
pub fn prove<C: Challenger>(
    stmt: &ElementStatement<'_>,
    z: &[F128],
    ch: &mut C,
) -> (ElementProof, ElementClaim) {
    prove_with_grinding(stmt, z, Grinding::disabled(), ch)
}

/// [`prove`] with an explicit Fiat--Shamir grinding policy for both element
/// PIOP phases and the packed-direct opening transport. The normal
/// mixed-table prover selects this from
/// [`PcsParams::element_grinding`]; this entry point gives standalone users
/// the same protection without changing the legacy transcript by default.
pub fn prove_with_grinding<C: Challenger>(
    stmt: &ElementStatement<'_>,
    z: &[F128],
    grinding: Grinding,
    ch: &mut C,
) -> (ElementProof, ElementClaim) {
    let m_words = stmt.m_words();
    assert!(
        m_words >= MIN_M_WORDS,
        "element prove needs m_words >= {MIN_M_WORDS} (got {m_words})"
    );
    assert!(stmt.n <= 1usize << stmt.n_log, "count exceeds capacity");
    assert_eq!(z.len(), stmt.n_words(), "witness length");

    // ---- 1. Commit the witness words, then bind the whole statement. ----
    let params = pcs_params(m_words, grinding);
    let (commitment, pdata) = commit(z, &params);
    stmt.bind(&commitment.cap, ch);

    // ---- 2. Phase 1: element zerocheck. ----
    //
    // `Az`/`Bz` by sparse gather, then the affine constants folded in — the
    // zerocheck works directly on `(Az + a_const)` and `(Bz + b_const)`.
    let (mut pa, mut pb) = stmt.ty.apply(z, stmt.n_log);
    broadcast_add(&mut pa, stmt.ty.a_const(), stmt.n_log);
    broadcast_add(&mut pb, stmt.ty.b_const(), stmt.n_log);
    let (zc_proof, zc_claim) = zerocheck::prove_with_grinding(pa, pb, z, m_words, grinding, ch);

    // ---- 3. Phase 2: batched lincheck. ----
    //
    // The verifier's own correction: `ea`/`eb` are claims on `Az + a_const`, and
    // the constants' MLEs collapse to the base-block evaluation at `r_con` with
    // no row dependence, so subtracting them (char 2: adding) leaves the pure
    // `Âz(r)` / `B̂z(r)` claims the lincheck reduces.
    let (va, vb) = strip_constants(stmt.ty, &zc_claim);
    let (lc_proof, lc_claim) =
        lincheck::prove_with_grinding(stmt.ty, z, stmt.n_log, &zc_claim.r, va, vb, grinding, ch);

    // ---- 4. Open both witness claims, packed-direct, no ring-switch. ----
    let claims = packed_direct_claims(&zc_claim.r, zc_claim.ec, &lc_claim.r_prime, lc_claim.z_eval);
    let open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
        z.to_vec(),
        &pdata,
        &commitment,
        &[],
        &[],
        &claims,
        &PaddingSpec::dense(params.m),
        &ligerito_prover_config(m_words),
        params.opening_grinding(),
        ch,
    );

    let proof = ElementProof {
        cap: commitment.cap,
        zerocheck: zc_proof,
        lincheck: lc_proof,
        open,
    };
    let claim = ElementClaim {
        r: zc_claim.r,
        ec: zc_claim.ec,
        r_prime: lc_claim.r_prime,
        z_eval: lc_claim.z_eval,
    };
    (proof, claim)
}

/// Verify an element proof against `stmt`. Walks the challenger in lockstep with
/// [`prove`].
pub fn verify<C: Challenger>(
    stmt: &ElementStatement<'_>,
    proof: &ElementProof,
    ch: &mut C,
) -> Result<ElementClaim, ElementR1csError> {
    verify_with_grinding(stmt, proof, Grinding::disabled(), ch)
}

/// [`verify`] with an explicit Fiat--Shamir grinding policy.  The verifier
/// checks every nonce before the challenge it protects is sampled.
pub fn verify_with_grinding<C: Challenger>(
    stmt: &ElementStatement<'_>,
    proof: &ElementProof,
    grinding: Grinding,
    ch: &mut C,
) -> Result<ElementClaim, ElementR1csError> {
    let m_words = stmt.m_words();
    if m_words < MIN_M_WORDS {
        return Err(ElementR1csError::TooSmall {
            m_words,
            min: MIN_M_WORDS,
        });
    }
    if stmt.n > 1usize << stmt.n_log {
        return Err(ElementR1csError::CountExceedsCapacity {
            n: stmt.n,
            n_log: stmt.n_log,
        });
    }

    // Rebuild the commitment from the proof's cap + statement-derived params,
    // and bind at the prover's transcript position.
    let commitment = Commitment {
        cap: proof.cap.clone(),
        params: pcs_params(m_words, grinding),
    };
    stmt.bind(&commitment.cap, ch);

    let zc_claim = zerocheck::verify_with_grinding(m_words, &proof.zerocheck, grinding, ch)
        .map_err(ElementR1csError::Zerocheck)?;
    let (va, vb) = strip_constants(stmt.ty, &zc_claim);
    let lc_claim = lincheck::verify_with_grinding(
        stmt.ty,
        stmt.n_log,
        &zc_claim.r,
        va,
        vb,
        &proof.lincheck,
        grinding,
        ch,
    )
    .map_err(ElementR1csError::Lincheck)?;

    let points = [zc_claim.r.as_slice(), lc_claim.r_prime.as_slice()];
    let values = [zc_claim.ec, lc_claim.z_eval];
    let refs: Vec<PackedDirectClaimRef<'_>> = points
        .iter()
        .zip(values)
        .map(|(point, value)| PackedDirectClaimRef { point, value })
        .collect();
    pcs::verify_opening_batch_ligerito_mixed_with_grinding(
        &commitment,
        &[],
        &[],
        &[],
        &refs,
        &proof.open,
        &ligerito_verifier_config(m_words),
        commitment.params.opening_grinding(),
        ch,
    )
    .map_err(ElementR1csError::Open)?;

    Ok(ElementClaim {
        r: zc_claim.r,
        ec: zc_claim.ec,
        r_prime: lc_claim.r_prime,
        z_eval: lc_claim.z_eval,
    })
}

/// The two packed-direct claims, in the fixed order `[ec at r, ẑ(r')]`. Both
/// points are fully random (the challenger never hands out an exactly-zero
/// coordinate in practice), so the dense `eq_ind` is the right representation.
fn packed_direct_claims(
    r: &[F128],
    ec: F128,
    r_prime: &[F128],
    z_eval: F128,
) -> Vec<PackedDirectClaim> {
    [(r, ec), (r_prime, z_eval)]
        .into_iter()
        .map(|(point, value)| PackedDirectClaim {
            point: point.to_vec(),
            value,
            eq_ind: DirectEqInd::Dense(build_eq_parallel(point)),
        })
        .collect()
}

/// `v[(y << n_log) + j] += c[y]` — broadcast the row-uniform constant vector
/// across every row.
fn broadcast_add(v: &mut [F128], c: &[F128], n_log: usize) {
    let rows = 1usize << n_log;
    debug_assert_eq!(v.len(), c.len() << n_log);
    v.par_chunks_mut(rows)
        .zip(c.par_iter())
        .for_each(|(dst, &cy)| {
            if !cy.is_zero() {
                for d in dst {
                    *d += cy;
                }
            }
        });
}

/// Turn the zerocheck's `(ea, eb)` — claims on `(Az + a_const)`, `(Bz + b_const)`
/// at `r` — into the pure `Âz(r)`, `B̂z(r)` claims the lincheck reduces, by
/// subtracting the constants' closed-form MLEs.
///
/// `x ↦ a_const[x_con]` is uniform in `x_row`, so its MLE is
/// `â_const_base(r_con)` — no row dependence, no count dependence (partition of
/// unity over the row block). `O(2^kappa)` for the verifier.
fn strip_constants(ty: &ElementTableType, zc: &zerocheck::Claim) -> (F128, F128) {
    let r_con = &zc.r[zc.r.len() - ty.kappa()..];
    let eq_con = crate::zerocheck::univariate_skip::build_eq(r_con);
    let dot = |c: &[F128]| -> F128 {
        eq_con
            .iter()
            .zip(c)
            .fold(F128::ZERO, |acc, (e, v)| acc + *e * *v)
    };
    (zc.ea + dot(ty.a_const()), zc.eb + dot(ty.b_const()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::test_rng::Rng;

    /// The canonical test gate: `kappa = 2`, columns `0,1` free wires (the
    /// operands), column `2` their product, column `3` padding.
    pub(crate) fn mult_gate(kappa: usize) -> ElementTableType {
        let mut b = ElementTableBuilder::new(kappa);
        b.free_wire(0).free_wire(1).mult(2, 0, 1);
        b.build().expect("mult gate is valid")
    }

    /// A mixed table exercising every row encoding: free wires, a mult, a
    /// linear pin, a mult-acc (which itself is mult + linear), and padding.
    pub(crate) fn mixed_gate(rng: &mut Rng) -> ElementTableType {
        // kappa = 3 → width 8. Columns: 0,1,2 free; 3 = z0·z1; 4 = w3·z0 + z2
        // via tmp column 5; 6 = a·z0 + b·z1 (linear); 7 padding.
        let mut b = ElementTableBuilder::new(3);
        let (wa, wb) = (rng.nonzero(), rng.nonzero());
        b.free_wire(0)
            .free_wire(1)
            .free_wire(2)
            .mult(3, 0, 1)
            .mult_acc(4, 3, 0, 2, 5)
            .linear(6, &[(0, wa), (1, wb)]);
        b.build().expect("mixed gate is valid")
    }

    /// Fill a satisfying witness for [`mult_gate`]: `n` real rows with random
    /// operands, the rest zero.
    pub(crate) fn mult_witness(
        ty: &ElementTableType,
        n_log: usize,
        n: usize,
        rng: &mut Rng,
    ) -> Vec<F128> {
        let rows = 1usize << n_log;
        let mut z = vec![F128::ZERO; ty.width() << n_log];
        for j in 0..n {
            let (a, b) = (rng.f128(), rng.f128());
            z[j] = a;
            z[rows + j] = b;
            z[2 * rows + j] = a * b;
        }
        z
    }

    /// Fill a satisfying witness for [`mixed_gate`].
    pub(crate) fn mixed_witness(
        ty: &ElementTableType,
        n_log: usize,
        n: usize,
        rng: &mut Rng,
    ) -> Vec<F128> {
        let at = |c: usize, j: usize| (c << n_log) + j;
        let mut z = vec![F128::ZERO; ty.width() << n_log];
        let (wa, wb) = (ty.a_0().rows[6][0].1, ty.a_0().rows[6][1].1);
        for j in 0..n {
            let (z0, z1, z2) = (rng.f128(), rng.f128(), rng.f128());
            z[at(0, j)] = z0;
            z[at(1, j)] = z1;
            z[at(2, j)] = z2;
            z[at(3, j)] = z0 * z1;
            z[at(5, j)] = z[at(3, j)] * z0;
            z[at(4, j)] = z[at(5, j)] + z2;
            z[at(6, j)] = wa * z0 + wb * z1;
        }
        z
    }

    // ---- type construction -------------------------------------------------

    #[test]
    fn builder_rows_are_satisfiable() {
        let mult = mult_gate(2);
        assert!(
            mult.satisfies(&mult_witness(&mult, 4, 9, &mut Rng::new(2)), 4, 9),
            "mult gate"
        );
        let mixed = mixed_gate(&mut Rng::new(1));
        assert!(
            mixed.satisfies(&mixed_witness(&mixed, 4, 9, &mut Rng::new(3)), 4, 9),
            "mixed gate (free wires, mult, mult-acc, linear, padding)"
        );
    }

    #[test]
    fn free_wire_is_a_tautology() {
        // A free wire constrains nothing: any value satisfies it.
        let ty = mult_gate(2);
        let mut rng = Rng::new(11);
        let mut z = mult_witness(&ty, 4, 16, &mut rng);
        // Perturbing an operand AND its product stays satisfying.
        let (a, b) = (rng.f128(), rng.f128());
        z[0] = a;
        z[16] = b;
        z[32] = a * b;
        assert!(ty.satisfies(&z, 4, 16));
    }

    #[test]
    fn padding_columns_are_pinned_to_zero() {
        let ty = mult_gate(2);
        let mut z = mult_witness(&ty, 4, 16, &mut Rng::new(12));
        // Column 3 is padding: its row is all-zero, forcing z_3 = 0.
        z[3 * 16] = F128::ONE;
        assert!(!ty.satisfies(&z, 4, 16), "padding column must be pinned");
    }

    #[test]
    fn dummy_rows_satisfy_by_the_validity_rule() {
        // Every row past the count is all-zero, and the disjoint-support rule
        // makes that satisfying for EVERY row encoding — free wires included
        // (`(0)(1) = 0`).
        let ty = mult_gate(2);
        for n in [0usize, 1, 7, 16] {
            let z = mult_witness(&ty, 4, n, &mut Rng::new(100 + n as u64));
            assert!(ty.satisfies(&z, 4, n), "n={n}");
        }
    }

    #[test]
    fn overlapping_constants_are_rejected_at_construction() {
        let width = 4usize;
        let a_const = {
            let mut v = vec![F128::ZERO; width];
            v[0] = F128::ONE;
            v
        };
        let b_const = {
            let mut v = vec![F128::ZERO; width];
            v[0] = F128::new(3, 0);
            v
        };
        let err = ElementTableType::new(
            2,
            1,
            SparseF128Matrix::zeros(width, width),
            SparseF128Matrix::zeros(width, width),
            a_const,
            b_const,
        )
        .expect_err("a0 ⊙ b0 ≠ 0 must be rejected");
        assert_eq!(err, TypeError::ConstantsOverlap { y: 0 });
    }

    #[test]
    fn disjoint_constants_are_accepted() {
        // The free-wire encoding has a_const = 0, b_const = 1 — disjoint.
        let width = 4usize;
        let mut b_const = vec![F128::ZERO; width];
        b_const[0] = F128::ONE;
        assert!(
            ElementTableType::new(
                2,
                1,
                SparseF128Matrix::zeros(width, width),
                SparseF128Matrix::zeros(width, width),
                vec![F128::ZERO; width],
                b_const,
            )
            .is_ok()
        );
    }

    #[test]
    fn shape_and_sparsity_errors_are_rejected() {
        let width = 4usize;
        let zeros = || SparseF128Matrix::zeros(width, width);
        let cz = || vec![F128::ZERO; width];

        // Wrong matrix shape.
        assert!(matches!(
            ElementTableType::new(2, 1, SparseF128Matrix::zeros(3, 4), zeros(), cz(), cz()),
            Err(TypeError::MatrixShape { .. })
        ));
        // Wrong constant length.
        assert!(matches!(
            ElementTableType::new(2, 1, zeros(), zeros(), vec![F128::ZERO; 3], cz()),
            Err(TypeError::ConstLen { .. })
        ));
        // k > width.
        assert!(matches!(
            ElementTableType::new(2, 5, zeros(), zeros(), cz(), cz()),
            Err(TypeError::TooManyColumns { .. })
        ));
        // Column out of range.
        let mut m = zeros();
        m.rows[0].push((width, F128::ONE));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::ColumnOutOfRange { .. })
        ));
        // Explicit zero coefficient.
        let mut m = zeros();
        m.rows[0].push((0, F128::ZERO));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::ZeroCoefficient { .. })
        ));
        // Duplicate column.
        let mut m = zeros();
        m.rows[0].push((1, F128::ONE));
        m.rows[0].push((1, F128::ONE));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::DuplicateColumn { .. })
        ));
        // Padding row must be all-zero.
        let mut m = zeros();
        m.rows[3].push((0, F128::ONE));
        assert!(matches!(
            ElementTableType::new(2, 1, m, zeros(), cz(), cz()),
            Err(TypeError::PaddingRowNotZero { y: 3 })
        ));
    }

    #[test]
    fn digest_is_deterministic_and_sensitive() {
        let a = mult_gate(2);
        let b = mult_gate(2);
        assert_eq!(a.digest(), b.digest(), "same type, same digest");
        assert_eq!(a.digest(), a.digest(), "cached digest is stable");

        // Any change to the block moves the digest.
        let mut wider = ElementTableBuilder::new(3);
        wider.free_wire(0).free_wire(1).mult(2, 0, 1);
        assert_ne!(a.digest(), wider.build().unwrap().digest(), "kappa");

        let mut swapped = ElementTableBuilder::new(2);
        swapped.free_wire(0).free_wire(1).mult(2, 1, 0);
        // Operand order is a real difference in A_0 vs B_0.
        assert_ne!(a.digest(), swapped.build().unwrap().digest(), "operands");

        let mut scaled = ElementTableBuilder::new(2);
        scaled
            .free_wire(0)
            .free_wire(1)
            .linear(2, &[(0, F128::new(7, 0))]);
        assert_ne!(a.digest(), scaled.build().unwrap().digest(), "coefficients");
    }

    // ---- Az / Bz -----------------------------------------------------------

    #[test]
    fn apply_matches_per_row_matrix_product() {
        let mut rng = Rng::new(77);
        let ty = mixed_gate(&mut rng);
        let n_log = 5usize;
        let rows = 1usize << n_log;
        let z: Vec<F128> = (0..ty.width() << n_log).map(|_| rng.f128()).collect();
        let (az, bz) = ty.apply(&z, n_log);
        for j in 0..rows {
            let zj: Vec<F128> = (0..ty.width()).map(|c| z[(c << n_log) + j]).collect();
            for y in 0..ty.width() {
                assert_eq!(
                    az[(y << n_log) + j],
                    ty.a_0().row_dot(y, &zj),
                    "az y={y} j={j}"
                );
                assert_eq!(
                    bz[(y << n_log) + j],
                    ty.b_0().row_dot(y, &zj),
                    "bz y={y} j={j}"
                );
            }
        }
    }

    #[test]
    fn broadcast_add_is_row_uniform() {
        let n_log = 3usize;
        let c = vec![F128::new(5, 0), F128::ZERO, F128::new(9, 1), F128::ONE];
        let mut v = vec![F128::ZERO; c.len() << n_log];
        broadcast_add(&mut v, &c, n_log);
        for (y, &cy) in c.iter().enumerate() {
            for j in 0..1usize << n_log {
                assert_eq!(v[(y << n_log) + j], cy);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end tests: commit → bind → zerocheck → lincheck → packed-direct open.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod e2e_tests {
    use super::tests::{mixed_gate, mixed_witness, mult_gate, mult_witness};
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::test_rng::Rng;

    const TRANSCRIPT: &[u8] = b"flock-element-e2e";

    fn run_prove(stmt: &ElementStatement<'_>, z: &[F128]) -> (ElementProof, ElementClaim) {
        let mut ch = FsChallenger::new(TRANSCRIPT);
        prove(stmt, z, &mut ch)
    }

    fn run_verify(
        stmt: &ElementStatement<'_>,
        proof: &ElementProof,
    ) -> Result<ElementClaim, ElementR1csError> {
        let mut ch = FsChallenger::new(TRANSCRIPT);
        verify(stmt, proof, &mut ch)
    }

    /// The standalone API exposes the same secure PIOP schedule as the union
    /// profile path.  This pins nonce placement (before the protected squeeze),
    /// exact proof shape, and verifier rejection of a missing witness.
    #[test]
    fn grinded_prove_verify_roundtrip_and_rejects_missing_nonce() {
        let mut rng = Rng::new(0x128_E1E);
        let (n_log, n) = (6usize, 37usize); // κ=2, so m_words=8.
        let ty = mult_gate(2);
        let z = mult_witness(&ty, n_log, n, &mut rng);
        let stmt = ElementStatement { ty: &ty, n_log, n };
        let grinding = Grinding::per_challenge_128();

        let mut ch_p = FsChallenger::new(TRANSCRIPT);
        let (proof, claim_p) = prove_with_grinding(&stmt, &z, grinding, &mut ch_p);
        assert_eq!(
            proof.zerocheck.grinding_nonces.len(),
            grinding.zerocheck_nonce_count(stmt.m_words())
        );
        assert_eq!(
            proof.lincheck.grinding_nonces.len(),
            grinding.lincheck_nonce_count(ty.kappa())
        );
        assert_eq!(
            proof.open.batching_nonces.len(),
            1,
            "the two packed-direct opening claims require one batching PoW"
        );

        let mut ch_v = FsChallenger::new(TRANSCRIPT);
        assert_eq!(
            verify_with_grinding(&stmt, &proof, grinding, &mut ch_v).expect("honest proof"),
            claim_p
        );

        let mut missing_zc = proof.clone();
        missing_zc.zerocheck.grinding_nonces.pop();
        let mut ch_v = FsChallenger::new(TRANSCRIPT);
        assert!(matches!(
            verify_with_grinding(&stmt, &missing_zc, grinding, &mut ch_v),
            Err(ElementR1csError::Zerocheck(
                zerocheck::ElementZerocheckError::BadGrindingNonceCount { .. }
            ))
        ));

        let mut missing_open = proof.clone();
        missing_open.open.batching_nonces.pop();
        let mut ch_v = FsChallenger::new(TRANSCRIPT);
        assert!(matches!(
            verify_with_grinding(&stmt, &missing_open, grinding, &mut ch_v),
            Err(ElementR1csError::Open(_))
        ));

        // Pick a different nonce that fails the *initial* PoW predicate.
        // (The 4-bit initial difficulty at m_words=8 makes the scan tiny.)
        let original = proof.zerocheck.grinding_nonces[0];
        let mut rejected_bad_nonce = false;
        for delta in 1..=64u64 {
            let mut bad = proof.clone();
            bad.zerocheck.grinding_nonces[0] = original.wrapping_add(delta);
            let mut ch_v = FsChallenger::new(TRANSCRIPT);
            if matches!(
                verify_with_grinding(&stmt, &bad, grinding, &mut ch_v),
                Err(ElementR1csError::Zerocheck(
                    zerocheck::ElementZerocheckError::InvalidGrindingNonce { which: "initial" }
                ))
            ) {
                rejected_bad_nonce = true;
                break;
            }
        }
        assert!(
            rejected_bad_nonce,
            "a changed initial PoW nonce must reject"
        );

        let mut missing_lc = proof;
        missing_lc.lincheck.grinding_nonces.pop();
        let mut ch_v = FsChallenger::new(TRANSCRIPT);
        assert!(matches!(
            verify_with_grinding(&stmt, &missing_lc, grinding, &mut ch_v),
            Err(ElementR1csError::Lincheck(
                lincheck::ElementLincheckError::BadGrindingNonceCount { .. }
            ))
        ));
    }

    /// Round-trip a mult-gate table at several `(n_log, kappa, n)` shapes —
    /// `m_words` at the Ligerito floor and above it, non-power-of-two counts,
    /// full utilization, and the empty table. The claims `verify` returns must
    /// match the prover's byte for byte.
    #[test]
    fn prove_verify_roundtrip_mult_gate() {
        let mut rng = Rng::new(20260729);
        // (n_log, n): kappa = 2, so m_words = n_log + 2 ≥ MIN_M_WORDS ⇒ n_log ≥ 6.
        for (n_log, n) in [
            (6usize, 64usize), // m_words = 8, the floor, full utilization
            (6, 37),           // non-power-of-two count
            (6, 1),            // one real row
            (6, 0),            // empty table: all-zero witness
            (7, 100),          // non-power-of-two, wider
            (8, 256),          // full utilization one level up
        ] {
            let ty = mult_gate(2);
            let z = mult_witness(&ty, n_log, n, &mut rng);
            assert!(ty.satisfies(&z, n_log, n), "n_log={n_log} n={n}");
            let stmt = ElementStatement { ty: &ty, n_log, n };

            let (proof, claim_p) = run_prove(&stmt, &z);
            let claim_v = run_verify(&stmt, &proof)
                .unwrap_or_else(|e| panic!("verify rejected n_log={n_log} n={n}: {e:?}"));
            assert_eq!(claim_p, claim_v, "n_log={n_log} n={n}");
            // The two output claims are the witness MLE at two distinct points.
            assert_eq!(claim_v.r.len(), stmt.m_words());
            assert_eq!(claim_v.r_prime.len(), stmt.m_words());
            assert_ne!(claim_v.r, claim_v.r_prime, "claim points must differ");
            // The lincheck collapses the row half, so it costs `kappa` rounds —
            // not `m_words` — and the two points share their row coordinates.
            assert_eq!(proof.lincheck.rounds.len(), ty.kappa(), "n_log={n_log}");
            assert_eq!(&claim_v.r[..n_log], &claim_v.r_prime[..n_log]);
        }
    }

    /// The mixed table — free wires, a mult, a mult-acc, a linear pin and a
    /// padding column — round-trips at `kappa = 3`.
    #[test]
    fn prove_verify_roundtrip_mixed_gate() {
        let mut rng = Rng::new(4242);
        let ty = mixed_gate(&mut rng);
        for (n_log, n) in [(5usize, 32usize), (5, 19), (6, 40)] {
            let z = mixed_witness(&ty, n_log, n, &mut rng);
            assert!(ty.satisfies(&z, n_log, n));
            let stmt = ElementStatement { ty: &ty, n_log, n };
            let (proof, claim_p) = run_prove(&stmt, &z);
            let claim_v = run_verify(&stmt, &proof)
                .unwrap_or_else(|e| panic!("verify rejected n_log={n_log} n={n}: {e:?}"));
            assert_eq!(claim_p, claim_v);
        }
    }

    /// A witness violating ONE constraint in ONE row must be rejected, even
    /// though the prover follows the honest algorithm on it.
    #[test]
    fn violated_constraint_is_rejected() {
        let mut rng = Rng::new(31415);
        let (n_log, n) = (6usize, 40usize);
        let ty = mult_gate(2);
        let rows = 1usize << n_log;
        for bad_row in [0usize, 17, 39] {
            let mut z = mult_witness(&ty, n_log, n, &mut rng);
            // Break the product column of one real row.
            z[2 * rows + bad_row] += F128::ONE;
            assert!(!ty.satisfies(&z, n_log, n));
            let stmt = ElementStatement { ty: &ty, n_log, n };
            let (proof, _) = run_prove(&stmt, &z);
            assert_eq!(
                run_verify(&stmt, &proof),
                Err(ElementR1csError::Zerocheck(
                    zerocheck::ElementZerocheckError::SumcheckFinalFailed
                )),
                "row {bad_row}"
            );
        }
    }

    /// A dirty dummy row (non-zero past the declared count) is a violation too:
    /// the padding rows sit inside the zerocheck's sum, they are not skipped.
    #[test]
    fn relation_violating_dummy_row_is_rejected() {
        let mut rng = Rng::new(2718);
        let (n_log, n) = (6usize, 40usize);
        let ty = mult_gate(2);
        let mut z = mult_witness(&ty, n_log, n, &mut rng);
        // Row 50 is past the count. Setting its product column alone makes
        // `(0)(0) = 1` — a relation violation, caught like any other.
        z[2 * (1usize << n_log) + 50] = F128::ONE;
        let stmt = ElementStatement { ty: &ty, n_log, n };
        let (proof, _) = run_prove(&stmt, &z);
        assert!(run_verify(&stmt, &proof).is_err());
    }

    /// SCOPE BOUNDARY (documented, not a defect). The PIOP proves the relation
    /// over the whole padded row domain; it does NOT independently prove the
    /// other half of the statement, that rows `[n, 2^n_log)` are *zero*. A row
    /// past the count that is itself relation-satisfying therefore verifies
    /// under the smaller declared count.
    ///
    /// This costs nothing here — the relation holds on every row, so a proof at
    /// count `n` is equally a proof at full capacity — and it disappears in the
    /// union integration, where the committed height is count-derived and the
    /// dummy rows do not exist in the commitment at all ("dropped words are
    /// zero" becomes structural). Pinned as a test so the boundary is visible
    /// rather than latent; see the `n`-field docs on [`ElementStatement`].
    #[test]
    fn satisfying_dummy_row_is_not_detected() {
        let mut rng = Rng::new(2719);
        let (n_log, n) = (6usize, 40usize);
        let ty = mult_gate(2);
        let rows = 1usize << n_log;
        let mut z = mult_witness(&ty, n_log, n, &mut rng);
        // Fill row 50 (past the count) with a *satisfying* mult triple.
        let (a, b) = (rng.f128(), rng.f128());
        z[50] = a;
        z[rows + 50] = b;
        z[2 * rows + 50] = a * b;
        let stmt = ElementStatement { ty: &ty, n_log, n };
        let (proof, _) = run_prove(&stmt, &z);
        assert!(
            run_verify(&stmt, &proof).is_ok(),
            "the relation holds on every row, so this proof is accepted — \
             zero-ness of dummy rows is the union integration's job"
        );
    }

    /// Correct round messages but wrong final claim values. Both `ec` and
    /// `ẑ(r')` must be pinned — `ec` by the zerocheck's own final check, and
    /// `ẑ(r')` by the lincheck's residual check.
    #[test]
    fn wrong_claim_values_are_rejected() {
        let mut rng = Rng::new(161803);
        let (n_log, n) = (6usize, 40usize);
        let ty = mult_gate(2);
        let z = mult_witness(&ty, n_log, n, &mut rng);
        let stmt = ElementStatement { ty: &ty, n_log, n };
        let (proof, _) = run_prove(&stmt, &z);
        assert!(run_verify(&stmt, &proof).is_ok(), "honest proof");

        let mut bad_ec = proof.clone();
        bad_ec.zerocheck.ec += F128::ONE;
        assert_eq!(
            run_verify(&stmt, &bad_ec),
            Err(ElementR1csError::Zerocheck(
                zerocheck::ElementZerocheckError::SumcheckFinalFailed
            )),
            "wrong ec"
        );

        let mut bad_z = proof.clone();
        bad_z.lincheck.z_eval += F128::ONE;
        assert_eq!(
            run_verify(&stmt, &bad_z),
            Err(ElementR1csError::Lincheck(
                lincheck::ElementLincheckError::SumcheckFinalFailed
            )),
            "wrong z_eval"
        );

        // `ea`/`eb` feed the lincheck target, so tampering with either breaks
        // the reduction rather than the zerocheck.
        type Tamper = fn(&mut ElementProof);
        for (name, mutate) in [
            ("ea", (|p| p.zerocheck.ea += F128::ONE) as Tamper),
            ("eb", |p| p.zerocheck.eb += F128::ONE),
        ] {
            let mut bad = proof.clone();
            mutate(&mut bad);
            assert!(run_verify(&stmt, &bad).is_err(), "wrong {name}");
        }

        // A different root claims a different witness entirely.
        let mut bad_root = proof.clone();
        bad_root.cap[0][0] ^= 1;
        assert!(run_verify(&stmt, &bad_root).is_err(), "wrong root");
    }

    /// Statement mismatch: the whole statement is bound before any challenge, so
    /// verifying under a different count or a different table type diverges at
    /// the first challenge and rejects.
    #[test]
    fn statement_mismatch_is_rejected() {
        let mut rng = Rng::new(577215);
        let (n_log, n) = (6usize, 40usize);
        let ty = mult_gate(2);
        let z = mult_witness(&ty, n_log, n, &mut rng);
        let stmt = ElementStatement { ty: &ty, n_log, n };
        let (proof, _) = run_prove(&stmt, &z);
        assert!(run_verify(&stmt, &proof).is_ok(), "honest statement");

        // Wrong count — same witness, same capacity, different declared `n`.
        for bad_n in [n - 1, n + 1, 1usize << n_log] {
            let bad = ElementStatement {
                ty: &ty,
                n_log,
                n: bad_n,
            };
            assert!(run_verify(&bad, &proof).is_err(), "count {bad_n}");
        }

        // Wrong type digest — an operand swap in the mult row. The witness still
        // satisfies nothing about it; the transcript diverges regardless.
        let mut swapped = ElementTableBuilder::new(2);
        swapped.free_wire(0).free_wire(1).mult(2, 1, 0);
        let other = swapped.build().unwrap();
        assert_ne!(ty.digest(), other.digest());
        let bad = ElementStatement {
            ty: &other,
            n_log,
            n,
        };
        assert!(run_verify(&bad, &proof).is_err(), "wrong type digest");

        // Wrong capacity: a different `n_log` changes m_words, so the rebuilt
        // commitment params and every round count are wrong.
        let bad = ElementStatement {
            ty: &ty,
            n_log: n_log + 1,
            n,
        };
        assert!(run_verify(&bad, &proof).is_err(), "wrong capacity");

        // A count above the capacity is rejected on shape alone.
        let bad = ElementStatement {
            ty: &ty,
            n_log,
            n: (1usize << n_log) + 1,
        };
        assert!(matches!(
            run_verify(&bad, &proof),
            Err(ElementR1csError::CountExceedsCapacity { .. })
        ));
    }

    /// Truncated and bit-flipped proof bytes. Serialization is the repo's
    /// `bincode`; a mutation must either fail to deserialize or fail to verify —
    /// never verify.
    #[test]
    fn mutated_proof_bytes_are_rejected() {
        let mut rng = Rng::new(1414213);
        let (n_log, n) = (6usize, 40usize);
        let ty = mult_gate(2);
        let z = mult_witness(&ty, n_log, n, &mut rng);
        let stmt = ElementStatement { ty: &ty, n_log, n };
        let (proof, _) = run_prove(&stmt, &z);
        let bytes = bincode::serialize(&proof).expect("serialize");
        // The honest bytes round-trip and verify.
        let decoded: ElementProof = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(decoded, proof);
        assert!(run_verify(&stmt, &decoded).is_ok());

        // Truncation at several depths.
        for frac in [1usize, 2, 4, 8] {
            let cut = bytes.len() - bytes.len() / frac;
            match bincode::deserialize::<ElementProof>(&bytes[..cut]) {
                Err(_) => {}
                Ok(p) => assert!(
                    run_verify(&stmt, &p).is_err(),
                    "truncated to {cut} bytes verified"
                ),
            }
        }

        // Bit flips spread across the payload — the roots, the round messages,
        // the claim values and the Ligerito opening all live in here.
        let n_flips = 24usize;
        for i in 0..n_flips {
            let pos = i * (bytes.len() / n_flips);
            let mut bad = bytes.clone();
            bad[pos] ^= 1 << (i % 8);
            match bincode::deserialize::<ElementProof>(&bad) {
                Err(_) => {}
                Ok(p) => assert!(
                    run_verify(&stmt, &p).is_err(),
                    "bit flip at byte {pos} verified"
                ),
            }
        }
    }

    /// A proof is not transferable to a different transcript.
    #[test]
    fn proof_is_bound_to_its_transcript() {
        let mut rng = Rng::new(66);
        let (n_log, n) = (6usize, 40usize);
        let ty = mult_gate(2);
        let z = mult_witness(&ty, n_log, n, &mut rng);
        let stmt = ElementStatement { ty: &ty, n_log, n };
        let (proof, _) = run_prove(&stmt, &z);
        let mut other = FsChallenger::new(b"a-different-domain");
        assert!(verify(&stmt, &proof, &mut other).is_err());
    }

    /// Smoke measurement (NOT a benchmark): PIOP-only prove time at
    /// `n_log = 16, kappa = 2` — the two sumcheck phases, without the commit or
    /// the opening, which is what the milestone's cost target is about. Prints
    /// under `--nocapture`; the assertion is on correctness, not on the clock.
    #[test]
    fn piop_smoke_at_n_log_16() {
        let (n_log, kappa) = (16usize, 2usize);
        let ty = mult_gate(kappa);
        let n = (1usize << n_log) - 3; // non-trivial, near-full utilization
        let z = mult_witness(&ty, n_log, n, &mut Rng::new(7));

        let t_wit = std::time::Instant::now();
        let (mut pa, mut pb) = ty.apply(&z, n_log);
        broadcast_add(&mut pa, ty.a_const(), n_log);
        broadcast_add(&mut pb, ty.b_const(), n_log);
        let wit_ms = t_wit.elapsed().as_secs_f64() * 1e3;

        let mut ch = FsChallenger::new(b"element-smoke");
        let t_zc = std::time::Instant::now();
        let (_zc_proof, zc_claim) = zerocheck::prove(pa, pb, &z, n_log + kappa, &mut ch);
        let zc_ms = t_zc.elapsed().as_secs_f64() * 1e3;

        let (va, vb) = strip_constants(&ty, &zc_claim);
        let t_lc = std::time::Instant::now();
        let (_lc_proof, lc_claim) = lincheck::prove(&ty, &z, n_log, &zc_claim.r, va, vb, &mut ch);
        let lc_ms = t_lc.elapsed().as_secs_f64() * 1e3;

        eprintln!(
            "[element-smoke] n_log={n_log} kappa={kappa} m_words={} \
             (Az,Bz gather {wit_ms:.2} ms) zerocheck {zc_ms:.2} ms + lincheck {lc_ms:.2} ms \
             = PIOP {:.2} ms",
            n_log + kappa,
            zc_ms + lc_ms
        );

        // Correctness of what we just timed.
        assert_eq!(zc_claim.r.len(), n_log + kappa);
        assert_eq!(lc_claim.r_prime.len(), n_log + kappa);
        assert!(!lc_claim.z_eval.is_zero());
    }
}
