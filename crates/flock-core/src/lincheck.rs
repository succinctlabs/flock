//! Lincheck PIOP for **block-diagonal** R1CS over GF(2).
//!
//! Reduces three MLE evaluation claims (`â(x)=v`, `b̂(x')=v'`, `ĉ(x'')=v''`)
//! plus the linear constraints (`a = Az`, `b = Bz`, `c = Cz`) to three MLE
//! evaluation claims on `z`, all sharing a fresh random inner coord.
//!
//! ## Matrix structure (the assumption we exploit)
//!
//! `A = I_{2^n_log} ⊗ A_0` (block-diagonal with `A_0` repeated `2^n_log`
//! times along the diagonal). Same for B, C. Storage is `O(k²)` for the
//! small base matrices, not `O(N²)`.
//!
//! With the row/col index decomposed as `(i_inner, i_outer)` with `k_log`
//! inner bits and `n_log` outer bits (`m = k_log + n_log`), the bilinear MLE
//! factors:
//!
//!   `Â(i, x)  =  Â_0(i_inner, x_inner) · eq(i_outer, x_outer)`
//!
//! So for the claim `v = â(x) = Σ_i z(i) · Â(i, x)` the outer summation
//! collapses by the eq-MLE identity:
//!
//!   `v  =  Σ_{i_inner}  Â_0(i_inner, x_inner) · ẑ(i_inner, x_outer)`
//!
//! — a sum over only `2^k_log` terms, with `ẑ(·, x_outer)` being the
//! partial fold of `z` at the outer half of the claim point.
//!
//! ## Protocol shape (circuit R1CS: C = I, A & B share a claim point)
//!
//! For R1CS coming from circuits, `C = I` (identity), so `c = Cz = z` and
//! the zerocheck's c-claim `ĉ(point_c) = v_c` IS a direct `z`-claim
//! `ẑ(point_c) = v_c` — handled by the PCS without going through lincheck.
//! Likewise the zerocheck's `â` and `b̂` claims live at the **same** point
//! `(z, ρ-values)`, so lincheck only needs to fold `z` **once** at that
//! shared point.
//!
//! 1. **Prover sends** one length-`k = 2^k_log` F128 vector
//!    `z_vec[i_inner] = ẑ(i_inner, x_ab.x_outer)`.
//! 2. **Verifier checks** *two* consistency equations against the same
//!    `z_vec`:
//!    ```text
//!    Σ_{i_inner}  Â_0_quirky(z_skip, x_inner_rest, i_inner) · z_vec[i_inner]  ==  v_a
//!    Σ_{i_inner}  B̂_0_quirky(z_skip, x_inner_rest, i_inner) · z_vec[i_inner]  ==  v_b
//!    ```
//! 3. **Verifier samples** quirky `(r_inner_skip, r_inner_rest)` after
//!    observing `z_vec`.
//! 4. **Verifier derives** one z-claim at the shared output point:
//!    ```text
//!    w = ẑ((r_inner_skip, r_inner_rest), x_ab.x_outer)
//!      = Σ_{i_inner} quirky_eq(r_inner_skip, r_inner_rest, i_inner) · z_vec[i_inner]
//!    ```
//!
//! The lincheck output is one `(point, value)` z-claim; combined with the
//! c-claim handed in directly by the caller, the PCS sees **two** z-openings.
//!
//! ## Soundness
//!
//! - The two scalar checks tie `z_vec` to `v_a` and `v_b` from the upstream
//!   layer — without them a malicious prover could send any vector.
//! - The post-vector random `(r_inner_skip, r_inner_rest)` plus Schwartz-Zippel
//!   ensures that if `z_vec_claimed` differs from the true partial fold of `z`,
//!   the derived `w` differs from the true `ẑ((r_inner_skip, r_inner_rest), x_outer)`
//!   with probability `≈ 1 − 2⁻¹²⁸`. The PCS opening catches that downstream.
//!
//! ## Quirky (univariate-skip) claim points
//!
//! To compose with the **zerocheck's univariate skip** for the first `k_skip`
//! variables, claim points use the [`QuirkyPoint`] representation:
//!
//!   `x = (z_skip ∈ F_{2^128},  x_inner_rest ∈ F_{2^128}^{k_log − k_skip},  x_outer ∈ F_{2^128}^{n_log})`
//!
//! - `z_skip` is the univariate-skip challenge; it represents all `k_skip`
//!   skip variables collapsed via the polynomial extension with Lagrange
//!   basis on `φ_8(0), …, φ_8(2^{k_skip} − 1)`.
//! - The remaining `k_log − k_skip` inner coords plus the `n_log` outer
//!   coords are standard multilinear.
//!
//! When evaluating the bilinear matrix MLE at a quirky claim point, the
//! eq factor for the inner row index becomes the **outer product of**:
//! `L_{i_skip}(z_skip) · eq(x_inner_rest, i_inner_rest)`, where `L_*` are
//! Lagrange weights at `z_skip` for the `k_skip` skip dims (see
//! [`build_quirky_eq_table`]).
//!
//! The prover's partial fold `ẑ(·, x_outer)` is unchanged — it only depends
//! on `x_outer` (still pure multilinear). The verifier-side eq tables and
//! the final-sample reduction are the only changes.
//!
//! ## Conventions
//!
//! - **Point ordering inside `QuirkyPoint`.** `x_inner_rest[0..k_log − k_skip]`
//!   bind to inner variables `i_inner_rest[0..k_log − k_skip]`. `x_outer[0..n_log]`
//!   to outer vars.
//! - **Eq table layout.** `eq_table[i]` where `i = Σ b_j · 2^j` is
//!   `Π_j eq(point[j], b_j) = Π_j (1 + point[j] + b_j)`.
//! - **`z_packed` byte layout (specific to lincheck — enables column-scan
//!   lookup tables without an explicit transpose).** Writing `i_outer = 8·byte_idx + r`
//!   with `r ∈ {0,..,7}` and `byte_idx ∈ {0,..,n_outer/8 − 1}`, the bit
//!   `z[i_inner, i_outer]` lives at:
//!     - **byte position** `byte_idx · k + i_inner`,
//!     - **bit-within-byte** `r`.
//!
//!   Equivalently: `z_packed` is organized in `n_outer/8` *stripes* of `k`
//!   contiguous bytes each. Stripe `byte_idx` covers all `i_inner ∈ {0,..,k}`
//!   for the same outer batch `i_outer ∈ {8·byte_idx, …, 8·byte_idx + 7}`.
//!   Each byte holds 8 outer bits for one i_inner.
//!
//!   In bit-position terms, the bit-index decomposes as:
//!   ```text
//!   LSB:  3 bits = r           (= low 3 bits of i_outer, = bit-within-byte)
//!         k_log bits = i_inner
//!   MSB:  (n_log − 3) bits = byte_idx (= upper bits of i_outer)
//!   ```
//!
//!   This layout makes the partial-fold column scan sequential — for each
//!   `byte_idx`, all `k` per-i_inner bytes are at consecutive byte positions,
//!   so we build a 256-entry sum table for the 8 outer values once per
//!   `byte_idx` and apply it across all `i_inner` with one lookup + one XOR
//!   per byte.

use crate::challenger::Challenger;
use crate::field::{F128, F256Unreduced};
use crate::r1cs::SparseBinaryMatrix;
use crate::zerocheck::multilinear::lagrange_weights_naive;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LincheckCircuit: the per-block linear structure lincheck consumes
// ---------------------------------------------------------------------------
//
// Lincheck's hot path computes a single length-`k = 2^k_log` vector
//
//   `comb_vec[c] = α · ξ_A(c) + ξ_B(c)`
//
// where `ξ_M(c) = Σ_r eq_inner[r] · M[r, c]` is the eq-weighted column
// marginal of base matrix `M ∈ {A_0, B_0}`. Today's `sparse_row_fold_alpha_batched`
// computes it by scattering `eq_inner[r]` to every column in row r's nonzero
// set — cost ∝ NNZ.
//
// For circuit-shaped R1CS (Keccak, BLAKE3, SHA-256) the same `comb_vec` can be
// produced by walking the constraint graph in round order — same operations
// the witness gen already does, just with eq-weights instead of bit values.
// Per-hash impls can also avoid materializing matrices entirely (relevant for
// encodings where intermediate state slots are dropped and substitution would
// otherwise blow up A/B density).
//
// `LincheckCircuit` is the seam: `lincheck::prove`/`verify` take
// `&dyn LincheckCircuit` instead of a pair of matrices. The default impl
// `SparseMatrixCircuit` wraps the existing fused sparse kernel so callers
// that haven't ported get identical behavior.

/// Per-block linear structure consumed by lincheck. Implementations produce
/// the α-batched column marginal `comb_vec[c] = α · ξ_A(c) + ξ_B(c)` either
/// by sparse-matrix iteration (default) or by walking the circuit directly.
pub trait LincheckCircuit: Sync {
    /// Number of columns in the per-block matrices A_0, B_0 (= k = 2^k_log).
    fn n_cols(&self) -> usize;

    /// Compute `comb_vec[c] = α · (eq^T · A_0)[c] + (eq^T · B_0)[c]` over
    /// `c ∈ [0, n_cols())`. `eq_inner.len() == n_cols()`.
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128>;

    /// Column index of a constant-one wire to pin, or `None` if the circuit has
    /// no such wire. When `Some(col)`, lincheck folds one extra `β`-term into the
    /// comb so the sumcheck also proves that the committed constant column is the
    /// all-ones vector (whose MLE is the constant `1`), closing the all-zero
    /// witness soundness gap. This REQUIRES the witness to set that wire to `1`
    /// in *every* batched instance — padding included. See
    /// `docs/const-wire-pin.md`. Default `None` keeps the transcript unchanged
    /// for circuits without a constant wire.
    fn const_pin_col(&self) -> Option<usize> {
        None
    }
}

/// Default `LincheckCircuit` over a pair of sparse binary matrices. Delegates
/// to the existing fused row-fold kernel. Callers that haven't migrated to a
/// per-hash circuit walker use this wrapper.
pub struct SparseMatrixCircuit<'a> {
    pub a_0: &'a SparseBinaryMatrix,
    pub b_0: &'a SparseBinaryMatrix,
    /// Constant-wire pin column (see [`LincheckCircuit::const_pin_col`]).
    const_pin: Option<usize>,
}

impl<'a> SparseMatrixCircuit<'a> {
    pub fn new(a_0: &'a SparseBinaryMatrix, b_0: &'a SparseBinaryMatrix) -> Self {
        debug_assert_eq!(a_0.num_rows, b_0.num_rows);
        debug_assert_eq!(a_0.num_cols, b_0.num_cols);
        Self {
            a_0,
            b_0,
            const_pin: None,
        }
    }

    /// Set the constant-wire pin column (see `docs/const-wire-pin.md`).
    pub fn with_const_pin(mut self, const_pin: Option<usize>) -> Self {
        self.const_pin = const_pin;
        self
    }
}

impl<'a> LincheckCircuit for SparseMatrixCircuit<'a> {
    fn n_cols(&self) -> usize {
        self.a_0.num_cols
    }
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        sparse_row_fold_alpha_batched(alpha, self.a_0, self.b_0, eq_inner)
    }
    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }
}

/// Column-major (CSC) `LincheckCircuit`: `(A_0, B_0)` transposed once into
/// flat `col_ptr`/`row_idx` arrays. `fold_alpha_batched` becomes a gather —
/// each column reads its own row list and sums `eq_inner[r]`, so columns are
/// independent (parallel with no per-thread accumulator copies and no write
/// scatter) and the α-mul amortizes to one per column:
///
///   `comb[c] = α · Σ_{r ∈ colA(c)} eq_inner[r] + Σ_{r ∈ colB(c)} eq_inner[r]`
///
/// On the SHA-256 hybrid matrices (k = 2^15, ~1.3M nonzeros) this measures
/// ~7× faster than the row-scatter fold and ~100× faster than the symbolic
/// per-hash walkers; on BLAKE3 (~21M nonzeros) ~1.7× faster than row-scatter.
/// Construction costs one pass over the nonzeros (~4 ms / ~40 ms for the
/// above) — do it once at setup, e.g. via
/// [`crate::r1cs::BlockR1cs::csc_lincheck_circuit`].
#[derive(Clone)]
pub struct CscCircuit {
    n_cols: usize,
    a_col_ptr: Vec<u32>,
    a_rows: Vec<u32>,
    b_col_ptr: Vec<u32>,
    b_rows: Vec<u32>,
    /// Constant-wire pin column (see [`LincheckCircuit::const_pin_col`]).
    const_pin: Option<usize>,
}

/// Flatten one sparse matrix into CSC arrays: rows with a 1 in column `c` are
/// `rows_flat[col_ptr[c] as usize .. col_ptr[c+1] as usize]`.
fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    assert!(m.num_rows <= u32::MAX as usize);
    assert!(m.num_cols <= u32::MAX as usize);
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

// Compact Debug — the row arrays run to millions of entries.
impl std::fmt::Debug for CscCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CscCircuit")
            .field("n_cols", &self.n_cols)
            .field("nnz_a", &self.a_rows.len())
            .field("nnz_b", &self.b_rows.len())
            .finish()
    }
}

impl CscCircuit {
    pub fn from_matrices(a_0: &SparseBinaryMatrix, b_0: &SparseBinaryMatrix) -> Self {
        assert_eq!(a_0.num_rows, b_0.num_rows);
        assert_eq!(a_0.num_cols, b_0.num_cols);
        let (a_col_ptr, a_rows) = csc_from_rows(a_0);
        let (b_col_ptr, b_rows) = csc_from_rows(b_0);
        Self {
            n_cols: a_0.num_cols,
            a_col_ptr,
            a_rows,
            b_col_ptr,
            b_rows,
            const_pin: None,
        }
    }

    /// Set the constant-wire pin column (see `docs/const-wire-pin.md`).
    pub fn with_const_pin(mut self, const_pin: Option<usize>) -> Self {
        self.const_pin = const_pin;
        self
    }
}

impl LincheckCircuit for CscCircuit {
    fn n_cols(&self) -> usize {
        self.n_cols
    }
    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(eq_inner.len(), self.n_cols);
        let one_col = |c: usize| {
            let mut sa = F128::ZERO;
            for &r in &self.a_rows[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize] {
                sa += eq_inner[r as usize];
            }
            let mut sb = F128::ZERO;
            for &r in &self.b_rows[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize] {
                sb += eq_inner[r as usize];
            }
            alpha * sa + sb
        };
        if self.n_cols < SUMCHECK_PAR_THRESHOLD {
            return (0..self.n_cols).map(one_col).collect();
        }
        let mut out = vec![F128::ZERO; self.n_cols];
        out.par_iter_mut()
            .enumerate()
            .for_each(|(c, slot)| *slot = one_col(c));
        out
    }
}

/// Row-compacted CSC circuit: remaps row indices to a dense `[0, n_unique_rows)`
/// range during construction so that `fold_alpha_batched_compact` can use a
/// compact eq_inner table that fits in L1 cache. For MHOT route (2838 unique
/// rows out of 32768), the compact table is ~44 KiB instead of ~512 KiB.
///
/// Use [`build_quirky_eq_table_sparse`] to build the compact eq_inner, then
/// pass it to [`CompactCscCircuit::fold_alpha_batched_compact`].
///
/// Also implements [`LincheckCircuit`] so it can be used as a drop-in
/// replacement in the standard lincheck prover (accepting full-length
/// eq_inner). The compact path is available via the dedicated method.
#[derive(Clone)]
pub struct CompactCscCircuit {
    n_cols: usize,
    n_orig_rows: usize,
    a_col_ptr: Vec<u32>,
    a_rows_compact: Vec<u32>,
    b_col_ptr: Vec<u32>,
    b_rows_compact: Vec<u32>,
    /// Sorted original row indices that have at least one nonzero in A or B.
    pub referenced_rows: Vec<u32>,
    const_pin: Option<usize>,
}

impl std::fmt::Debug for CompactCscCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactCscCircuit")
            .field("n_cols", &self.n_cols)
            .field("n_orig_rows", &self.n_orig_rows)
            .field("n_unique_rows", &self.referenced_rows.len())
            .field("nnz_a", &self.a_rows_compact.len())
            .field("nnz_b", &self.b_rows_compact.len())
            .finish()
    }
}

impl CompactCscCircuit {
    pub fn from_matrices(a_0: &SparseBinaryMatrix, b_0: &SparseBinaryMatrix) -> Self {
        assert_eq!(a_0.num_rows, b_0.num_rows);
        assert_eq!(a_0.num_cols, b_0.num_cols);

        let (a_col_ptr, a_rows_orig) = csc_from_rows(a_0);
        let (b_col_ptr, b_rows_orig) = csc_from_rows(b_0);

        let mut row_set = std::collections::BTreeSet::new();
        for &r in &a_rows_orig {
            row_set.insert(r);
        }
        for &r in &b_rows_orig {
            row_set.insert(r);
        }
        let referenced_rows: Vec<u32> = row_set.iter().copied().collect();

        let mut orig_to_compact = vec![u32::MAX; a_0.num_rows];
        for (compact_idx, &orig) in referenced_rows.iter().enumerate() {
            orig_to_compact[orig as usize] = compact_idx as u32;
        }

        let a_rows_compact: Vec<u32> = a_rows_orig
            .iter()
            .map(|&r| orig_to_compact[r as usize])
            .collect();
        let b_rows_compact: Vec<u32> = b_rows_orig
            .iter()
            .map(|&r| orig_to_compact[r as usize])
            .collect();

        Self {
            n_cols: a_0.num_cols,
            n_orig_rows: a_0.num_rows,
            a_col_ptr,
            a_rows_compact,
            b_col_ptr,
            b_rows_compact,
            referenced_rows,
            const_pin: None,
        }
    }

    pub fn with_const_pin(mut self, const_pin: Option<usize>) -> Self {
        self.const_pin = const_pin;
        self
    }

    /// Number of unique rows referenced by any nonzero in A or B.
    pub fn n_unique_rows(&self) -> usize {
        self.referenced_rows.len()
    }

    /// Fold using a compact eq_inner table (length = `n_unique_rows()`).
    /// The compact table must be indexed by the remapped row indices.
    /// Returns a full-length `n_cols` comb_vec.
    pub fn fold_alpha_batched_compact(
        &self,
        alpha: F128,
        eq_inner_compact: &[F128],
    ) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(
            eq_inner_compact.len(),
            self.referenced_rows.len(),
            "compact eq_inner length must match n_unique_rows"
        );
        let one_col = |c: usize| {
            let mut sa = F128::ZERO;
            for &r in &self.a_rows_compact
                [self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize]
            {
                sa += eq_inner_compact[r as usize];
            }
            let mut sb = F128::ZERO;
            for &r in &self.b_rows_compact
                [self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize]
            {
                sb += eq_inner_compact[r as usize];
            }
            alpha * sa + sb
        };
        if self.n_cols < SUMCHECK_PAR_THRESHOLD {
            return (0..self.n_cols).map(one_col).collect();
        }
        let mut out = vec![F128::ZERO; self.n_cols];
        out.par_iter_mut()
            .enumerate()
            .for_each(|(c, slot)| *slot = one_col(c));
        out
    }
}

impl LincheckCircuit for CompactCscCircuit {
    fn n_cols(&self) -> usize {
        self.n_cols
    }
    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }
    fn fold_alpha_batched(&self, alpha: F128, eq_inner_full: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(eq_inner_full.len(), self.n_cols);
        let one_col = |c: usize| {
            let mut sa = F128::ZERO;
            for &r in &self.a_rows_compact
                [self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize]
            {
                // Gather from the full table using the ORIGINAL row index
                // (referenced_rows maps compact -> original).
                sa += eq_inner_full[self.referenced_rows[r as usize] as usize];
            }
            let mut sb = F128::ZERO;
            for &r in &self.b_rows_compact
                [self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize]
            {
                sb += eq_inner_full[self.referenced_rows[r as usize] as usize];
            }
            alpha * sa + sb
        };
        if self.n_cols < SUMCHECK_PAR_THRESHOLD {
            return (0..self.n_cols).map(one_col).collect();
        }
        let mut out = vec![F128::ZERO; self.n_cols];
        out.par_iter_mut()
            .enumerate()
            .for_each(|(c, slot)| *slot = one_col(c));
        out
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A "quirky" claim point: one univariate-skip coord (`z_skip`) representing
/// the first `k_skip` variables via the polynomial extension with the φ_8 basis,
/// followed by multilinear coords for the rest of inner and for outer.
///
/// Total "elements" = `1 + (k_log − k_skip) + n_log`, which is the shape the
/// zerocheck's extract_c output uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyPoint {
    /// Univariate-skip challenge ∈ F₁₂₈. Binds all `k_skip` skip variables.
    pub z_skip: F128,
    /// Multilinear coords for the inner dims *after* the skip block. Length
    /// `k_log − k_skip`.
    pub x_inner_rest: Vec<F128>,
    /// Multilinear coords for the outer dims. Length `n_log = m − k_log`.
    pub x_outer: Vec<F128>,
}

/// Lincheck prover message: a partial product-sumcheck that proves the two
/// scalar consistency equations against `z` partially folded at the shared
/// outer half `x_ab.x_outer`, without sending the full length-`2^k_log`
/// `z_vec`. Sumcheck binds the high `k_log − k_skip` multilinear inner dims;
/// the low `k_skip` (φ8 univariate-skip) dims are handled by sending
/// `z_partial` (the post-sumcheck length-`2^k_skip` collapsed vector) and
/// applying a fresh-`z_skip` φ8 Lagrange combination at verify time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LincheckProof {
    /// Per-round messages `(q(1), q(∞))` of the `k_log − k_skip`-round
    /// product-sumcheck. `q(0)` is recovered from the running claim
    /// (`q(0) = T_r + q(1)` in char 2). Standard multilinear binding.
    pub rounds: Vec<(F128, F128)>,
    /// The length-`2^k_skip` collapse of the prover's `z_vec` over the
    /// sumcheck-bound `r_rest` dims. Folded against φ8 Lagrange weights at a
    /// fresh `z_skip` to yield the output claim's value.
    pub z_partial: Vec<F128>,
}

/// Lincheck output: one MLE evaluation claim on `z`, at the quirky inner
/// point `(r_inner_skip, r_inner_rest)` combined with `x_ab.x_outer`
/// (publicly known to the caller).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LincheckClaim {
    /// Univariate-skip post-vector random sample.
    pub r_inner_skip: F128,
    /// Multilinear post-vector random sample, length `k_log − k_skip`.
    pub r_inner_rest: Vec<F128>,
    /// `ẑ((r_inner_skip, r_inner_rest), x_ab.x_outer)` — the single
    /// `z`-claim derived from the A and B consistency checks.
    pub w: F128,
}

/// Reasons the verifier may reject.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// One of the proof vectors has the wrong length (expected `2^k_log`).
    BadVectorLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },
    /// One of the input quirky points has wrong `x_inner_rest` length
    /// (expected `k_log − k_skip`).
    BadInnerRestLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },
    /// One of the input quirky points has wrong `x_outer` length
    /// (expected `n_log = m − k_log`).
    BadOuterLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },
    /// One of the base matrices isn't `2^k_log × 2^k_log`.
    BadMatrixShape {
        which: &'static str,
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },
    /// `k_skip` exceeds `k_log` (the matrix inner dimension).
    KSkipExceedsKLog { k_skip: usize, k_log: usize },
    /// The scalar consistency check failed for one of (A, B, C).
    /// Detected: `Σ_{i_inner} M̂_0_quirky(z_skip, x_inner_rest, i_inner) · z_x_vec[i_inner] ≠ v`.
    ConsistencyFailed { which: &'static str },
}

// ---------------------------------------------------------------------------
// Core kernels
// ---------------------------------------------------------------------------

/// Build the eq-MLE table at `point ∈ F^d`. Returns a length-`2^d` vector
/// where `output[i] = Π_j (1 + point[j] + bit_j(i)) = Π_j eq(point[j], bit_j(i))`.
///
/// Standard "doubling-in-half" construction: `O(2^d)` F128 muls, no
/// inversions. Indexing is LSB-first — `bit_j(i)` is the `j`-th LSB of `i`.
pub fn build_eq_table(point: &[F128]) -> Vec<F128> {
    let d = point.len();
    let mut out: Vec<F128> = Vec::with_capacity(1usize << d);
    out.push(F128::ONE);
    for j in 0..d {
        let r_j = point[j];
        let one_plus_r_j = F128::ONE + r_j;
        let len = 1usize << j;
        out.resize(2 * len, F128::ZERO);
        // For each existing entry i ∈ [0, len), produce two children:
        //   out[i]       *= (1 + r_j)     ← new bit_j = 0
        //   out[i + len]  = out[i] * r_j  ← new bit_j = 1
        // Forward iteration is safe: the [i] and [i+len] slots are disjoint.
        for i in 0..len {
            let v = out[i];
            out[i + len] = v * r_j;
            out[i] = v * one_plus_r_j;
        }
    }
    out
}

/// Fold a sparse boolean matrix's rows against an eq table at the row
/// coords. Computes the **transposed** matrix-vector product:
///
///   `output[col] = Σ_{row: M[row, col] = 1} eq_table[row]`
///
/// This is the row-MLE `M̂_0(x_inner, ·)` evaluated at all boolean column
/// indices — the length-`k` vector the verifier needs for the consistency
/// check. Cost: `nnz(M)` F128 adds.
/// Below this matrix row count, the sequential path beats rayon dispatch
/// overhead. Tuned for `k = 2^14` (BLAKE3) — small matrices stay scalar,
/// big ones parallelize.
const SPARSE_ROW_FOLD_PAR_THRESHOLD: usize = 1usize << 12;

pub fn sparse_row_fold(matrix: &SparseBinaryMatrix, eq_table: &[F128]) -> Vec<F128> {
    assert_eq!(
        eq_table.len(),
        matrix.num_rows,
        "eq_table length must match matrix row count"
    );
    let n_cols = matrix.num_cols;
    if matrix.rows.len() < SPARSE_ROW_FOLD_PAR_THRESHOLD {
        let mut out = vec![F128::ZERO; n_cols];
        for (row_idx, row) in matrix.rows.iter().enumerate() {
            let e = eq_table[row_idx];
            for &col in row {
                out[col] += e;
            }
        }
        out
    } else {
        // Scatter-reduce: per-thread accumulator, XOR-merge at the end. Each
        // thread allocates a length-n_cols buffer (~256 KB at k=16384) — fine
        // vs the witness-scale buffers already in flight.
        use rayon::prelude::*;
        matrix
            .rows
            .par_iter()
            .enumerate()
            .fold(
                || vec![F128::ZERO; n_cols],
                |mut acc, (row_idx, row)| {
                    let e = eq_table[row_idx];
                    for &col in row {
                        acc[col] += e;
                    }
                    acc
                },
            )
            .reduce(
                || vec![F128::ZERO; n_cols],
                |mut a, b| {
                    for i in 0..n_cols {
                        a[i] += b[i];
                    }
                    a
                },
            )
    }
}

/// Partial fold of `z` at the outer half of a claim point — single-matrix,
/// **scalar reference**. Uses the lincheck `z_packed` stripe layout
/// (see module docs).
///
///   `output[i_inner] = Σ_{i_outer ∈ {0,1}^n_log}  z[i_inner, i_outer] · eq_outer[i_outer]`
///
/// Equivalently, `output[i_inner] = ẑ(i_inner_as_F128, x_outer)` for boolean
/// `i_inner`. Used as the cross-check oracle for the production
/// `partial_fold_packed_z_triple`.
pub fn partial_fold_packed_z(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 3, "need n_outer ≥ 8 for byte stripes");
    let n_stripes = n_outer / 8;

    let mut out = vec![F128::ZERO; k];
    for byte_idx in 0..n_stripes {
        let stripe = &z_packed[byte_idx * k..(byte_idx + 1) * k];
        for (i_inner, &byte) in stripe.iter().enumerate() {
            if byte == 0 {
                continue;
            }
            let mut bits = byte;
            while bits != 0 {
                let r = bits.trailing_zeros() as usize;
                let i_outer = 8 * byte_idx + r;
                out[i_inner] += eq_outer[i_outer];
                bits &= bits - 1;
            }
        }
    }
    out
}

/// **Optimized single-matrix partial fold.** Same shape as
/// [`partial_fold_packed_z`] but uses 256-entry **sum-table lookups** and is
/// parallelized via rayon. The hot inner kernel does just **1 byte load +
/// 1 table lookup + 1 XOR** per `(byte_idx, i_inner)` pair.
///
/// At m=29 multi-thread this is ~3× faster than the naive scalar
/// `partial_fold_packed_z` (which we keep as the cross-check reference).
///
/// Iteration:
/// 1. For each `byte_idx ∈ 0..n_outer/8`, build a 256-entry F128 table
///    where `table[b] = Σ_{r: bit r set in b} eq_outer[8·byte_idx + r]`.
///    Cost: 255 F128 XORs (doubling construction).
/// 2. Sweep the `k`-byte stripe at `z_packed[byte_idx·k .. (byte_idx+1)·k]`.
///    For each `i_inner`, do `out[i_inner] ^= table[z_byte]`.
///
/// Parallel: each worker owns a contiguous range of stripes and a private
/// length-`k` accumulator; results XOR-reduced.
pub fn partial_fold_packed_z_fast(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let k = 1usize << k_log;
    partial_fold_packed_z_fast_padded(z_packed, m, k_log, k, eq_outer)
}

/// Padding-aware variant of [`partial_fold_packed_z_fast`]. Skips rows
/// `i_inner ∈ [useful_bits, k)` — those rows hold zero in every block of an
/// honestly padded witness, so the fold over the outer dim is zero. Output
/// is byte-identical to the dense path on such witnesses.
pub fn partial_fold_packed_z_fast_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 3, "need n_outer ≥ 8 for byte stripes");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;

    let stripes_per_chunk = (n_stripes / 256).max(1);
    let bytes_per_chunk = stripes_per_chunk * k;

    // fold(): one length-k accumulator per WORKER rather than per chunk —
    // at large k the per-chunk accumulators of map().reduce() dominate MT
    // time with allocation + tree-reduce XOR traffic (keccak3: k = 2^17
    // means 2 MB per chunk across ~128 chunks).
    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut acc, (chunk_idx, chunk_bytes)| {
                let stripe_start = chunk_idx * stripes_per_chunk;
                let mut table = vec![F128::ZERO; 256];
                for (rel_stripe, stripe) in chunk_bytes.chunks(k).enumerate() {
                    let byte_idx = stripe_start + rel_stripe;
                    build_sum_table(&eq_outer[8 * byte_idx..8 * byte_idx + 8], &mut table);
                    for (i_inner, &z_byte) in stripe[..useful_bits].iter().enumerate() {
                        acc[i_inner] += table[z_byte as usize];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Stripes swept per accumulator touch in the NEON tiled partial fold.
/// Larger ⇒ the length-`k` accumulator is re-streamed fewer times
/// (`n_stripes / NEON_TILE_T`), but the per-tile sum tables grow
/// `NEON_TILE_T × 4 KB` and must stay L1-resident.
const NEON_TILE_T: usize = 8;

/// Single-matrix partial fold with **tiled + NEON-register accumulators**.
/// Keeps `BLOCK_K = 8` accumulators in NEON registers across a `NEON_TILE_T`
/// stripe sweep — no per-byte accumulator LD/ST. Hand-rolled aarch64
/// intrinsics force the F128 XOR to a single `EOR.16B` and pin the 8 accs
/// in Q registers.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_single(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let k = 1usize << k_log;
    partial_fold_packed_z_neon_single_padded(z_packed, m, k_log, k, eq_outer)
}

/// Padding-aware variant of [`partial_fold_packed_z_neon_single`]. Rounds
/// `useful_bits` up to a multiple of `BLOCK_K = 8` and processes only the
/// covered blocks; the trailing blocks (entirely padding) stay zero in the
/// accumulator. Any partially-useful boundary block is processed in full —
/// its padding bytes are zero, table[0] = 0, so they contribute nothing.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_single_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;
    let n_blocks_full = k / BLOCK_K;
    // Cover only the blocks that touch useful bits. The boundary block
    // contains padding bytes which are 0 — table[0] = 0 → they contribute
    // nothing to the per-block XOR chain.
    let n_blocks = useful_bits.div_ceil(BLOCK_K).min(n_blocks_full);

    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut out, (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;
                // TILE_T × 256 F128 = 32 KB tables. L1 resident.
                let mut tables = vec![F128::ZERO; TILE_T * 256];

                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    let tile_bytes_ptr = unsafe { chunk_bytes.as_ptr().add(tile_rel * TILE_T * k) };

                    for t in 0..TILE_T {
                        let byte_idx = stripe_base + t;
                        let eq_off = 8 * byte_idx;
                        build_sum_table(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut tables[t * 256..(t + 1) * 256],
                        );
                    }

                    let tables_ptr = tables.as_ptr() as *const u8;

                    for block_idx in 0..n_blocks {
                        let bs = block_idx * BLOCK_K;
                        unsafe {
                            process_block_neon_single(
                                tile_bytes_ptr,
                                k,
                                bs,
                                tables_ptr,
                                out.as_mut_ptr().add(bs),
                            );
                        }
                    }
                }
                // Suppress unused variable warning when not aarch64
                let _ = unsafe { vdupq_n_u8(0) };
                out
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Single-matrix NEON inner kernel — sweep TILE_T=8 stripes of a stripe-tile
/// for one BLOCK_K=8 block of i_inner positions, keeping all 8 accumulators
/// in NEON Q-registers.
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `TILE_T * k` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of mutable storage.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block_neon_single(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use std::arch::aarch64::*;
    const TILE_T: usize = NEON_TILE_T;

    let o = out_ptr as *mut u8;

    let mut a0 = vld1q_u8(o);
    let mut a1 = vld1q_u8(o.add(16));
    let mut a2 = vld1q_u8(o.add(32));
    let mut a3 = vld1q_u8(o.add(48));
    let mut a4 = vld1q_u8(o.add(64));
    let mut a5 = vld1q_u8(o.add(80));
    let mut a6 = vld1q_u8(o.add(96));
    let mut a7 = vld1q_u8(o.add(112));

    for t in 0..TILE_T {
        let stripe_ptr = tile_bytes_ptr.add(t * k + bs);
        let ta = tables_ptr.add(t * 256 * 16);

        let i0 = *stripe_ptr as usize;
        let i1 = *stripe_ptr.add(1) as usize;
        let i2 = *stripe_ptr.add(2) as usize;
        let i3 = *stripe_ptr.add(3) as usize;
        let i4 = *stripe_ptr.add(4) as usize;
        let i5 = *stripe_ptr.add(5) as usize;
        let i6 = *stripe_ptr.add(6) as usize;
        let i7 = *stripe_ptr.add(7) as usize;

        a0 = veorq_u8(a0, vld1q_u8(ta.add(i0 * 16)));
        a1 = veorq_u8(a1, vld1q_u8(ta.add(i1 * 16)));
        a2 = veorq_u8(a2, vld1q_u8(ta.add(i2 * 16)));
        a3 = veorq_u8(a3, vld1q_u8(ta.add(i3 * 16)));
        a4 = veorq_u8(a4, vld1q_u8(ta.add(i4 * 16)));
        a5 = veorq_u8(a5, vld1q_u8(ta.add(i5 * 16)));
        a6 = veorq_u8(a6, vld1q_u8(ta.add(i6 * 16)));
        a7 = veorq_u8(a7, vld1q_u8(ta.add(i7 * 16)));
    }

    vst1q_u8(o, a0);
    vst1q_u8(o.add(16), a1);
    vst1q_u8(o.add(32), a2);
    vst1q_u8(o.add(48), a3);
    vst1q_u8(o.add(64), a4);
    vst1q_u8(o.add(80), a5);
    vst1q_u8(o.add(96), a6);
    vst1q_u8(o.add(112), a7);
}

/// **i_inner-partitioned** NEON partial fold. Same result as
/// [`partial_fold_packed_z_neon_single_padded`] but parallelizes over the
/// **output** (`i_inner`) instead of over z stripes.
///
/// Why: the stripe-parallel kernel gives every worker its own full length-`k`
/// accumulator (2 MB at k = 2¹⁷). With P workers that's `P · 2 MB` of live
/// accumulators — past ~3 workers it exceeds L2, so each worker's accumulator
/// spills and gets re-streamed from **main memory** once per stripe-tile
/// (≈ `n_tiles · 2·k` F128 of memory traffic). Measured: scaling saturates at
/// ~5× on 10 cores (memory-bound), not ~10×.
///
/// Here the workers own **disjoint** slices of a single shared `out`, so the
/// total live accumulator is just `k` F128 = 2 MB — it stays L2-resident, never
/// re-streamed from memory, and there is **no final reduction**. Main-memory
/// traffic drops to one pass over z plus one write of `out`. Each worker still
/// uses the register-tiled inner kernel (8 accumulators across `TILE_T`
/// stripes); it just rebuilds the per-tile sum tables for its own slice (a few
/// % of redundant table-build XORs, far cheaper than the memory re-streaming).
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_iblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;

    // Only i_inner < useful_bits can be nonzero (padded rows fold to 0). Round
    // up to BLOCK_K; the boundary block's padding bytes are 0 ⇒ table[0] = 0 ⇒
    // contribute nothing. Rows [useful, k) stay zero from the vec init.
    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);

    let mut out = vec![F128::ZERO; k];
    if useful == 0 {
        return out;
    }

    // Partition the useful i_inner range across workers. Each chunk independently
    // rebuilds the per-tile sum tables, so chunk count drives redundant table
    // work — work that does NOT scale with cores and dominates the residual at
    // m=30 (≈3.3 ms/core at 3 chunks/worker). On the homogeneous pinned P-core
    // pool, 1 chunk/worker is perfectly balanced (par_chunks_mut → exactly `p`
    // equal chunks) and cuts that residual ~3×: partial-fold MT 6.2 → 4.5 ms,
    // no ST change. Oversubscribe (3/worker) only when the pool is larger than
    // the P-core count — i.e. likely includes slower E-cores — so rayon can
    // steal from a straggler. Each chunk is a BLOCK_K multiple.
    let p = rayon::current_num_threads().max(1);
    let chunks_per_worker = if p <= crate::perf_core_count_cached() {
        1
    } else {
        3
    };
    let i_chunk = (useful / (p * chunks_per_worker))
        .max(BLOCK_K)
        .next_multiple_of(BLOCK_K);

    out[..useful]
        .par_chunks_mut(i_chunk)
        .enumerate()
        .for_each(|(ci, out_slice)| {
            let i_base = ci * i_chunk;
            let n_block = out_slice.len() / BLOCK_K;
            // TILE_T × 256 F128 = 32 KB tables, L1-resident, rebuilt per tile.
            let mut tables = vec![F128::ZERO; TILE_T * 256];
            for tile in 0..n_tiles {
                let stripe_base = tile * TILE_T;
                for t in 0..TILE_T {
                    let eq_off = 8 * (stripe_base + t);
                    build_sum_table(
                        &eq_outer[eq_off..eq_off + 8],
                        &mut tables[t * 256..(t + 1) * 256],
                    );
                }
                let tables_ptr = tables.as_ptr() as *const u8;
                // Base of this (tile, i_base): process_block reads
                // z_base[t·k + bs] = z[(stripe_base+t)·k + i_base + bs].
                let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k + i_base) };
                for b in 0..n_block {
                    let i = b * BLOCK_K;
                    unsafe {
                        process_block_neon_single(
                            z_base,
                            k,
                            i,
                            tables_ptr,
                            out_slice.as_mut_ptr().add(i),
                        );
                    }
                }
            }
        });
    out
}

/// Dispatch helper: pick the fastest single-matrix partial fold available
/// for the given (m, k_log). Threads `useful_bits` through so the kernel
/// can skip blocks past the useful region of each block (byte-identical to
/// the dense path on honestly-padded witnesses).
fn partial_fold_packed_z_best(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    if n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
        #[cfg(target_arch = "aarch64")]
        {
            // i_inner-partitioned: keeps the length-k accumulator L2-resident so
            // the fold scales with cores instead of saturating on memory
            // bandwidth (~1.8× the stripe-parallel kernel at m=30, ~8.8× scaling
            // on 10 P-cores vs ~4.8×). See `partial_fold_packed_z_neon_iblock_padded`.
            partial_fold_packed_z_neon_iblock_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            partial_fold_packed_z_fast_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
    } else {
        partial_fold_packed_z_fast_padded(z_packed, m, k_log, useful_bits, eq_outer)
    }
}

/// Quick test for "can we use the tiled fast path?". Tile uses `TILE_T`
/// stripes; we need `n_stripes` divisible by TILE_T and enough outer dim.
fn n_log_ok_for_tile(m: usize, k_log: usize, tile_t: usize) -> bool {
    let n_log = m - k_log;
    if n_log < 3 + (tile_t.trailing_zeros() as usize) {
        return false;
    }
    let n_stripes = 1usize << (n_log - 3);
    n_stripes.is_multiple_of(tile_t)
}

/// Build a 256-entry sum table over 8 F128 values:
///   `table[b] = Σ_{r: bit r of b is set}  eq8[r]`
///
/// Doubling construction (255 XORs): for each new bit position `i ∈ 0..8`,
/// extend the table by XORing `eq8[i]` into each existing entry. This
/// avoids the naive 8·256 = 2048 operations.
#[inline]
fn build_sum_table(eq8: &[F128], table: &mut [F128]) {
    debug_assert_eq!(eq8.len(), 8);
    debug_assert_eq!(table.len(), 256);
    table[0] = F128::ZERO;
    for i in 0..8 {
        let e = eq8[i];
        let len = 1usize << i;
        for j in 0..len {
            table[len + j] = table[j] + e;
        }
    }
}

/// Pack a logical Boolean witness vector into the lincheck `z_packed`
/// stripe layout. The input `z_logical` is indexed linearly with
/// `z_logical[i_inner + i_outer · k]` = z's value at `(i_inner, i_outer)`.
/// The output `z_packed[byte_idx · k + i_inner]` holds 8 outer bits
/// `z[i_inner, 8·byte_idx + r]` for `r ∈ 0..8`, with bit `r` within the byte.
///
/// See the module-level docs for the full bit-position decomposition.
pub fn pack_z_lincheck(z_logical: &[bool], m: usize, k_log: usize) -> Vec<u8> {
    let k = 1usize << k_log;
    let n_total = 1usize << m;
    assert_eq!(z_logical.len(), n_total);
    let n_outer = n_total / k;
    assert_eq!(n_outer % 8, 0, "need n_outer ≥ 8 for byte stripes");
    let n_stripes = n_outer / 8;

    // Uninit alloc — every byte is written exactly once in the loop below.
    let mut z_packed: Vec<u8> = crate::alloc_uninit_vec(n_total / 8);
    for byte_idx in 0..n_stripes {
        for i_inner in 0..k {
            let mut byte = 0u8;
            for r in 0..8 {
                let i_outer = 8 * byte_idx + r;
                let logical_idx = i_inner + i_outer * k;
                if z_logical[logical_idx] {
                    byte |= 1u8 << r;
                }
            }
            z_packed[byte_idx * k + i_inner] = byte;
        }
    }
    z_packed
}

/// Same output as [`pack_z_lincheck`] but reads bits from an F_{2^128}-packed
/// witness (polynomial basis: bit `i` of logical = bit `i % 128` of
/// `z_packed_f128[i / 128]`).
pub fn pack_z_lincheck_from_packed(
    z_packed_f128: &[crate::field::F128],
    m: usize,
    k_log: usize,
) -> Vec<u8> {
    use rayon::prelude::*;
    let k = 1usize << k_log;
    let n_total = 1usize << m;
    assert_eq!(z_packed_f128.len(), n_total / 128);
    let n_outer = n_total / k;
    assert_eq!(n_outer % 8, 0, "need n_outer ≥ 8 for byte stripes");

    // Uninit alloc — the par_chunks_mut loop below writes every byte of
    // every k-byte stripe exactly once. Saves ~10 ms of sequential
    // zero-fill at m=29 (64 MB byte buffer) on the main thread.
    let mut z_packed: Vec<u8> = crate::alloc_uninit_vec(n_total / 8);
    // Each stripe (byte_idx) writes a disjoint k-byte chunk — process them in
    // parallel. Inside one stripe, k independent output bytes.
    z_packed
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(byte_idx, chunk)| {
            for i_inner in 0..k {
                let mut byte = 0u8;
                for r in 0..8 {
                    let i_outer = 8 * byte_idx + r;
                    let logical_idx = i_inner + i_outer * k;
                    let f128_idx = logical_idx / 128;
                    let local_bit = logical_idx % 128;
                    let bit = if local_bit < 64 {
                        (z_packed_f128[f128_idx].lo >> local_bit) & 1 == 1
                    } else {
                        (z_packed_f128[f128_idx].hi >> (local_bit - 64)) & 1 == 1
                    };
                    if bit {
                        byte |= 1u8 << r;
                    }
                }
                chunk[i_inner] = byte;
            }
        });
    z_packed
}

/// Build the **quirky eq table** for a claim point on the inner half:
///
///   `out[i_skip + i_inner_rest · 2^k_skip]
///     = L_{i_skip}(z_skip)  ·  eq(x_inner_rest, i_inner_rest)`
///
/// where `L_{i_skip}` are Lagrange weights at `z_skip` for the φ_8 basis
/// over `{0, …, 2^k_skip − 1}`. Length: `2^k_log`.
///
/// Encoding: the skip dim occupies the **low** `k_skip` bits of the table
/// index (matches z_packed's stripe layout / zerocheck's LSB-first
/// univariate-skip variable ordering). The `k_log − k_skip` multilinear
/// inner-rest dims occupy the next bits.
///
/// Cost: 64 (Lagrange) + 32 (eq) + 2048 outer products ≈ tiny.
pub fn build_quirky_eq_table(z_skip: F128, x_inner_rest: &[F128], k_skip: usize) -> Vec<F128> {
    let ell_skip = 1usize << k_skip;
    let ell_rest = 1usize << x_inner_rest.len();
    let lambda_skip = lagrange_weights_naive(k_skip, z_skip);
    let eq_rest = build_eq_table(x_inner_rest);
    let total = ell_skip * ell_rest;
    let mut out = Vec::with_capacity(total);
    // Layout: index = i_skip + i_inner_rest · 2^k_skip  ⇒  i_skip is low bits.
    for &er in &eq_rest {
        for &ls in &lambda_skip {
            out.push(ls * er);
        }
    }
    debug_assert_eq!(out.len(), total);
    out
}

/// Build a **sparse** quirky eq table containing only the entries at the given
/// `row_indices`. Output has `row_indices.len()` entries, where `out[i]` is the
/// value that `build_quirky_eq_table` would produce at index `row_indices[i]`.
///
/// Each entry is computed directly as `lambda_skip[r % ell_skip] * eq_rest[r >> k_skip]`.
/// The `eq_rest` table (length `2^(k_log - k_skip)`) is built in full via the
/// standard doubling construction; only the outer-product gather is sparse.
///
/// For MHOT route (2838 referenced rows, k_log=15, k_skip=6): builds
/// 512 eq_rest + 64 lagrange + 2838 muls ≈ 3414 F128 ops
/// vs dense: 512 + 64 + 32768 = 33344 ops.
pub fn build_quirky_eq_table_sparse(
    z_skip: F128,
    x_inner_rest: &[F128],
    k_skip: usize,
    row_indices: &[u32],
) -> Vec<F128> {
    let ell_skip = 1usize << k_skip;
    let lambda_skip = lagrange_weights_naive(k_skip, z_skip);
    let eq_rest = build_eq_table(x_inner_rest);
    let mut out = Vec::with_capacity(row_indices.len());
    for &r in row_indices {
        let r = r as usize;
        let i_skip = r & (ell_skip - 1);
        let i_rest = r >> k_skip;
        out.push(lambda_skip[i_skip] * eq_rest[i_rest]);
    }
    out
}

/// Dot product of two equal-length F128 slices.
fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len());
    let mut acc = F128::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += *x * *y;
    }
    acc
}

/// Length above which the inner product / element-wise kernels split via
/// rayon. Below it, sequential beats dispatch overhead.
const SUMCHECK_PAR_THRESHOLD: usize = 1usize << 12;

/// Fused `sparse_row_fold(A) + α-batch + sparse_row_fold(B)`: produces the
/// `comb_vec[c] = α · (A^T·eq)[c] + (B^T·eq)[c]` in a single pass, halving the
/// allocations and reduction phases vs. two separate sparse_row_folds + an
/// α-batch step. Both matrices must be `k × k` and `eq_table.len() == k`.
fn sparse_row_fold_alpha_batched(
    alpha: F128,
    a_0: &SparseBinaryMatrix,
    b_0: &SparseBinaryMatrix,
    eq_table: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;
    let n_cols = a_0.num_cols;
    debug_assert_eq!(b_0.num_cols, n_cols);
    debug_assert_eq!(eq_table.len(), a_0.num_rows);
    debug_assert_eq!(eq_table.len(), b_0.num_rows);

    let total_rows = a_0.num_rows + b_0.num_rows;
    if total_rows < SPARSE_ROW_FOLD_PAR_THRESHOLD {
        // Scalar fused path.
        let mut out = vec![F128::ZERO; n_cols];
        for (r, row) in a_0.rows.iter().enumerate() {
            let e = alpha * eq_table[r];
            for &c in row {
                out[c] += e;
            }
        }
        for (r, row) in b_0.rows.iter().enumerate() {
            let e = eq_table[r];
            for &c in row {
                out[c] += e;
            }
        }
        return out;
    }

    // Parallel fused path with a BOUNDED number of accumulators. These base
    // matrices are dense (e.g. BLAKE3: ~21M nonzeros over 16384 rows), so the
    // fold is ~21M F128 adds. The natural `par_iter().fold()` form spawns a
    // fresh length-`n_cols` (256 KB) accumulator per work-steal split and then
    // tree-reduces all of them — O(n_cols × num_splits) of pure overhead that
    // doesn't shrink with useful work, which capped scaling at ~1.5×. Here we
    // split the *rows* into a fixed number of contiguous chunks (rows are
    // evenly sized, so this load-balances), give each chunk one private
    // accumulator, then reduce. Overhead is O(n_cols × num_chunks) with
    // num_chunks ≈ 4× the thread count — negligible vs. the 21M-add body.
    let n_rows = a_0.num_rows;
    let p = rayon::current_num_threads().max(1);
    // ~4 chunks per worker for work-stealing balance, ≥256 rows each to keep
    // accumulator alloc/reduce overhead amortized.
    let chunk_rows = (n_rows.div_ceil(p * 4)).max(256);
    let n_chunks = n_rows.div_ceil(chunk_rows);

    let partials: Vec<Vec<F128>> = (0..n_chunks)
        .into_par_iter()
        .map(|ci| {
            let lo = ci * chunk_rows;
            let hi = ((ci + 1) * chunk_rows).min(n_rows);
            let mut acc = vec![F128::ZERO; n_cols];
            for r in lo..hi {
                let ea = alpha * eq_table[r];
                let eb = eq_table[r];
                for &c in &a_0.rows[r] {
                    acc[c] += ea;
                }
                for &c in &b_0.rows[r] {
                    acc[c] += eb;
                }
            }
            acc
        })
        .collect();

    let mut out = vec![F128::ZERO; n_cols];
    for acc in &partials {
        for i in 0..n_cols {
            out[i] += acc[i];
        }
    }
    out
}

/// One round of product-sumcheck on `(c, z)`: compute `(q(1), q(∞))` =
/// `(Σ c_hi·z_hi, Σ (c_hi+c_lo)·(z_hi+z_lo))` over the top-bit split. The
/// `len()` of `c` and `z` is even; `half = len/2`.
fn sumcheck_round_eval_par(c: &[F128], z: &[F128]) -> (F128, F128) {
    use rayon::prelude::*;
    let half = c.len() / 2;
    debug_assert_eq!(z.len(), c.len());
    let (clo, chi) = c.split_at(half);
    let (zlo, zhi) = z.split_at(half);
    if half < SUMCHECK_PAR_THRESHOLD {
        let mut e1 = F128::ZERO;
        let mut einf = F128::ZERO;
        for i in 0..half {
            e1 += chi[i] * zhi[i];
            einf += (chi[i] + clo[i]) * (zhi[i] + zlo[i]);
        }
        return (e1, einf);
    }
    (0..half)
        .into_par_iter()
        .map(|i| {
            let e1_i = chi[i] * zhi[i];
            let einf_i = (chi[i] + clo[i]) * (zhi[i] + zlo[i]);
            (e1_i, einf_i)
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// Bind the top remaining variable of `v` at challenge `r`: `v[i] ← v[i] +
/// r·(v[i+half] + v[i])` for `i ∈ [0, half)`, then truncate to `half`. In-place.
fn sumcheck_bind_top_in_place_par(v: &mut Vec<F128>, r: F128) {
    use rayon::prelude::*;
    let half = v.len() / 2;
    if half < SUMCHECK_PAR_THRESHOLD {
        for i in 0..half {
            v[i] = v[i] + r * (v[i + half] + v[i]);
        }
    } else {
        let (lo, hi) = v.split_at_mut(half);
        let hi = &hi[..half];
        lo.par_iter_mut()
            .zip(hi.par_iter())
            .for_each(|(lo_i, &hi_i)| {
                *lo_i = *lo_i + r * (hi_i + *lo_i);
            });
    }
    v.truncate(half);
}

/// **Fused fold + next-round evaluation.** Binds the top variable of *both*
/// `comb` and `z` at `r` (in place, each length halves) AND returns the next
/// product-sumcheck round's message `(q(1), q(∞))` over the just-bound tables —
/// all in a single pass over the data.
///
/// Why it fuses: round `t`'s message must be sent before `r_t` is sampled, so
/// eval(t) and bind(t) can't share a pass. But binding at `r_t` produces
/// exactly the table eval(t+1) reads, and `r_t` is known by then. The bound
/// values `new[i]` and `new[i+half2]` are precisely the `lo`/`hi` halves the
/// next round's eval pairs up, so we form each product the moment both bound
/// values exist. This replaces eval + two binds (3 passes) with 1.
///
/// Operates on quarters of each array (`half2 = len/4`). For `i ∈ 0..half2`:
/// ```text
///   lo' = q0[i] + r·(q2[i] + q0[i])   (= new[i],        next round's lo)
///   hi' = q1[i] + r·(q3[i] + q1[i])   (= new[i+half2],  next round's hi)
///   q0[i] ← lo';  q1[i] ← hi'
///   e1   += hi'·zhi';   einf += (hi'+lo')·(zhi'+zlo')
/// ```
/// In-place is safe: each `i` reads its 4 quarter-entries before writing the 2
/// low-half slots, and writes across distinct `i` are disjoint. Requires
/// `comb.len() == z.len()`, a power of two ≥ 4 (so the bound length ≥ 2 has a
/// well-defined next round — the caller guarantees this by only fusing when a
/// later round exists). The returned message is bit-identical to
/// `sumcheck_round_eval_par` run on the bound tables.
fn sumcheck_bind_both_and_eval_next(
    comb: &mut Vec<F128>,
    z: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    use rayon::prelude::*;
    let len = comb.len();
    debug_assert_eq!(z.len(), len);
    let half = len / 2;
    let half2 = half / 2;
    debug_assert!(half2 >= 1, "fused step needs a well-defined next round");

    // q0,q1 = low half (written); q2,q3 = high half (read-only).
    let (c_lo, c_hi) = comb.split_at_mut(half);
    let (cq0, cq1) = c_lo.split_at_mut(half2);
    let (cq2, cq3) = c_hi.split_at(half2);
    let (z_lo, z_hi) = z.split_at_mut(half);
    let (zq0, zq1) = z_lo.split_at_mut(half2);
    let (zq2, zq3) = z_hi.split_at(half2);

    let (e1, einf) = if half2 < SUMCHECK_PAR_THRESHOLD {
        let mut e1 = F128::ZERO;
        let mut einf = F128::ZERO;
        for i in 0..half2 {
            let lo = cq0[i] + r * (cq2[i] + cq0[i]);
            let hi = cq1[i] + r * (cq3[i] + cq1[i]);
            let zlo = zq0[i] + r * (zq2[i] + zq0[i]);
            let zhi = zq1[i] + r * (zq3[i] + zq1[i]);
            cq0[i] = lo;
            cq1[i] = hi;
            zq0[i] = zlo;
            zq1[i] = zhi;
            e1 += hi * zhi;
            einf += (hi + lo) * (zhi + zlo);
        }
        (e1, einf)
    } else {
        cq0.par_iter_mut()
            .zip(cq1.par_iter_mut())
            .zip(cq2.par_iter())
            .zip(cq3.par_iter())
            .zip(zq0.par_iter_mut())
            .zip(zq1.par_iter_mut())
            .zip(zq2.par_iter())
            .zip(zq3.par_iter())
            .map(|(((((((c0, c1), c2), c3), z0), z1), z2), z3)| {
                let lo = *c0 + r * (*c2 + *c0);
                let hi = *c1 + r * (*c3 + *c1);
                let zlo = *z0 + r * (*z2 + *z0);
                let zhi = *z1 + r * (*z3 + *z1);
                *c0 = lo;
                *c1 = hi;
                *z0 = zlo;
                *z1 = zhi;
                (hi * zhi, (hi + lo) * (zhi + zlo))
            })
            .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1))
    };

    comb.truncate(half);
    z.truncate(half);
    (e1, einf)
}

// ---------------------------------------------------------------------------
// Deferred-reduction variants: accumulate F256Unreduced products, reduce once.
// ---------------------------------------------------------------------------

/// Deferred-reduction variant of [`sumcheck_round_eval_par`]. Accumulates
/// unreduced 256-bit products via XOR, then reduces once at the end. Saves
/// `(half − 1)` ghash_reduce calls per accumulator (2 accumulators total).
fn sumcheck_round_eval_par_deferred(c: &[F128], z: &[F128]) -> (F128, F128) {
    use rayon::prelude::*;
    let half = c.len() / 2;
    debug_assert_eq!(z.len(), c.len());
    let (clo, chi) = c.split_at(half);
    let (zlo, zhi) = z.split_at(half);
    if half < SUMCHECK_PAR_THRESHOLD {
        let mut e1_acc = F256Unreduced::ZERO;
        let mut einf_acc = F256Unreduced::ZERO;
        for i in 0..half {
            e1_acc ^= chi[i].mul_unreduced(zhi[i]);
            einf_acc ^= (chi[i] + clo[i]).mul_unreduced(zhi[i] + zlo[i]);
        }
        return (e1_acc.reduce(), einf_acc.reduce());
    }
    (0..half)
        .into_par_iter()
        .map(|i| {
            let e1_i = chi[i].mul_unreduced(zhi[i]);
            let einf_i = (chi[i] + clo[i]).mul_unreduced(zhi[i] + zlo[i]);
            (e1_i, einf_i)
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |a, b| (a.0 ^ b.0, a.1 ^ b.1),
        )
        .into_reduced()
}

/// Deferred-reduction variant of [`sumcheck_bind_both_and_eval_next`].
/// The bind step (per-element writes) still uses eager F128 multiply because
/// each result is stored, not accumulated. Only the two accumulation sums
/// `e1` and `einf` benefit from deferred reduction.
fn sumcheck_bind_both_and_eval_next_deferred(
    comb: &mut Vec<F128>,
    z: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    use rayon::prelude::*;
    let len = comb.len();
    debug_assert_eq!(z.len(), len);
    let half = len / 2;
    let half2 = half / 2;
    debug_assert!(half2 >= 1, "fused step needs a well-defined next round");

    let (c_lo, c_hi) = comb.split_at_mut(half);
    let (cq0, cq1) = c_lo.split_at_mut(half2);
    let (cq2, cq3) = c_hi.split_at(half2);
    let (z_lo, z_hi) = z.split_at_mut(half);
    let (zq0, zq1) = z_lo.split_at_mut(half2);
    let (zq2, zq3) = z_hi.split_at(half2);

    let (e1, einf) = if half2 < SUMCHECK_PAR_THRESHOLD {
        let mut e1_acc = F256Unreduced::ZERO;
        let mut einf_acc = F256Unreduced::ZERO;
        for i in 0..half2 {
            let lo = cq0[i] + r * (cq2[i] + cq0[i]);
            let hi = cq1[i] + r * (cq3[i] + cq1[i]);
            let zlo = zq0[i] + r * (zq2[i] + zq0[i]);
            let zhi = zq1[i] + r * (zq3[i] + zq1[i]);
            cq0[i] = lo;
            cq1[i] = hi;
            zq0[i] = zlo;
            zq1[i] = zhi;
            e1_acc ^= hi.mul_unreduced(zhi);
            einf_acc ^= (hi + lo).mul_unreduced(zhi + zlo);
        }
        (e1_acc.reduce(), einf_acc.reduce())
    } else {
        cq0.par_iter_mut()
            .zip(cq1.par_iter_mut())
            .zip(cq2.par_iter())
            .zip(cq3.par_iter())
            .zip(zq0.par_iter_mut())
            .zip(zq1.par_iter_mut())
            .zip(zq2.par_iter())
            .zip(zq3.par_iter())
            .map(|(((((((c0, c1), c2), c3), z0), z1), z2), z3)| {
                let lo = *c0 + r * (*c2 + *c0);
                let hi = *c1 + r * (*c3 + *c1);
                let zlo = *z0 + r * (*z2 + *z0);
                let zhi = *z1 + r * (*z3 + *z1);
                *c0 = lo;
                *c1 = hi;
                *z0 = zlo;
                *z1 = zhi;
                (hi.mul_unreduced(zhi), (hi + lo).mul_unreduced(zhi + zlo))
            })
            .reduce(
                || (F256Unreduced::ZERO, F256Unreduced::ZERO),
                |a, b| (a.0 ^ b.0, a.1 ^ b.1),
            )
            .into_reduced()
    };

    comb.truncate(half);
    z.truncate(half);
    (e1, einf)
}

trait IntoReduced {
    fn into_reduced(self) -> (F128, F128);
}

impl IntoReduced for (F256Unreduced, F256Unreduced) {
    #[inline]
    fn into_reduced(self) -> (F128, F128) {
        (self.0.reduce(), self.1.reduce())
    }
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/// Prove the lincheck statement for the block-diagonal R1CS instance
/// `A = I_{2^n_log} ⊗ a_0`, `B = I ⊗ b_0`, `C = I ⊗ c_0`.
///
/// Preconditions:
/// - `m ≥ k_log`, `m = k_log + n_log` (caller's responsibility).
/// - `a_0, b_0, c_0` are each `k × k` where `k = 2^k_log`.
/// - `x.len() == x_prime.len() == x_pprime.len() == m`.
/// - `z_packed.len() == 2^m / 8`.
///
/// Returns `(LincheckProof, LincheckClaim)`. The claim's `r_inner` is
/// sampled from the challenger after the proof vectors are observed.
pub fn prove<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim) {
    prove_padded(
        z_packed,
        m,
        k_log,
        k_skip,
        1usize << k_log,
        circuit,
        x_ab,
        challenger,
    )
}

/// Padding-aware variant of [`prove`]. `useful_bits ≤ 2^k_log` declares how
/// many rows of each block carry real witness data; rows
/// `[useful_bits, 2^k_log)` are honest zero padding. The partial-fold over
/// the outer dimension skips work for those padding rows — byte-identical
/// proof on a witness with zero-padded blocks.
pub fn prove_padded<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim) {
    let (proof, claim, _) = prove_padded_inner(
        z_packed,
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        false,
        challenger,
    );
    (proof, claim)
}

/// Variant of [`prove_padded`] that also returns the **pre-sumcheck** z_vec
/// (`output[i_inner] = ẑ(i_inner, x_ab.x_outer)`, length `2^k_log`). The
/// downstream PCS reuses this vector to compute the AB-claim's ring-switch
/// `s_hat_v` via [`crate::pcs::ring_switch::s_hat_v_from_z_vec`], skipping a
/// `fold_1b_rows` pass at open time.
///
/// Pays one extra `2^k_log` F128 clone (~2 MB at k_log=17) before the
/// sumcheck loop; callers that don't need the reuse should keep using
/// [`prove_padded`] to avoid that clone.
pub fn prove_padded_capture_z_vec<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    let (proof, claim, captured) = prove_padded_inner(
        z_packed,
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        true,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce z_vec"),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_padded_inner<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    capture_z_vec: bool,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Option<Vec<F128>>) {
    let k = 1usize << k_log;
    let n_log = m - k_log;
    assert!(m >= k_log);
    assert!(k_skip <= k_log, "k_skip must be ≤ k_log");
    assert!(useful_bits <= k, "useful_bits ({useful_bits}) > k ({k})");
    let inner_rest_len = k_log - k_skip;
    assert_eq!(circuit.n_cols(), k);
    assert_eq!(x_ab.x_inner_rest.len(), inner_rest_len);
    assert_eq!(x_ab.x_outer.len(), n_log);

    challenger.observe_label(b"flock-lincheck-v0");
    let trace = std::env::var("LINCHECK_TRACE").is_ok();

    // 1. Sample α (matches verifier's order). Used to batch the two scalar
    //    consistency checks v_a, v_b into a single sumcheck.
    let alpha = challenger.sample_f128();

    // 2. Build the α-batched comb_vec via the circuit's per-block fold. For
    //    the sparse-matrix default this is the fused single-pass row-fold;
    //    per-hash circuit walkers compute the same `comb_vec` directly from
    //    the constraint graph.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "build_quirky_eq",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut comb_vec = circuit.fold_alpha_batched(alpha, &eq_inner);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "fold_alpha_batched",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 2b. Constant-wire pin. Fold β·eq(j*, ·) into the comb so the same sumcheck
    //     also proves z_vec[j*] = 1 (the all-ones constant column). Since j* is a
    //     boolean index, eq(j*, ·) is the one-hot vector and this is a single
    //     entry update. β is sampled after α; the verifier mirrors both. See
    //     docs/const-wire-pin.md.
    if let Some(col) = circuit.const_pin_col() {
        let beta = challenger.sample_f128();
        comb_vec[col] += beta;
    }

    // 3. Partial fold of z at the shared outer half (length-k F128 vector).
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let eq_x_outer = build_eq_table(&x_ab.x_outer);
    let mut z_vec = partial_fold_packed_z_best(z_packed, m, k_log, useful_bits, &eq_x_outer);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "partial_fold_z",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // 3b. Optional capture: clone the pre-sumcheck z_vec for downstream reuse
    //     (PCS open's AB-claim s_hat_v skipping fold_1b_rows). Only pay the
    //     clone when explicitly requested.
    let captured_z_vec: Option<Vec<F128>> = if capture_z_vec {
        Some(z_vec.clone())
    } else {
        None
    };
    let t_sumcheck_start = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // 5. Standard multilinear product-sumcheck over the high `inner_rest_len`
    //    bits of `i`. Each round binds the TOP remaining bit (mirrors
    //    chain::prove_chain_shift). After `inner_rest_len` rounds, both
    //    tables collapse to length `2^k_skip`. Per-round work is parallel via
    //    rayon when the residual table is large enough.
    let mut rounds = Vec::with_capacity(inner_rest_len);
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    if inner_rest_len > 0 {
        // Round 0's message is the only standalone evaluation pass; every later
        // round's message falls out of binding the previous round (fold +
        // next-eval fused into one pass — see `sumcheck_bind_both_and_eval_next`).
        let (mut e1, mut einf) = sumcheck_round_eval_par_deferred(&comb_vec, &z_vec);
        for t in 0..inner_rest_len {
            challenger.observe_f128(e1);
            challenger.observe_f128(einf);
            let r = challenger.sample_f128();
            rounds.push((e1, einf));
            r_rounds.push(r);
            if t + 1 < inner_rest_len {
                // Fused: bind both tables at r AND compute round (t+1)'s message.
                let (ne1, neinf) = sumcheck_bind_both_and_eval_next_deferred(&mut comb_vec, &mut z_vec, r);
                e1 = ne1;
                einf = neinf;
            } else {
                // Final round: just fold; z_vec collapses to z_partial.
                sumcheck_bind_top_in_place_par(&mut comb_vec, r);
                sumcheck_bind_top_in_place_par(&mut z_vec, r);
            }
        }
    }
    if let Some(t) = t_sumcheck_start {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "sumcheck (all rounds)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 6. Send `z_partial` (the post-sumcheck collapsed z_vec). Length 2^k_skip.
    let z_partial = z_vec.clone();
    challenger.observe_f128_slice(&z_partial);

    // 7. Sample fresh z_skip AFTER observing z_partial — gives Schwartz-Zippel
    //    soundness on the φ8 (univariate-skip) dim.
    let r_inner_skip = challenger.sample_f128();

    // 8. Output claim's value: φ8 Lagrange combination of z_partial at z_skip.
    //    Equals ẑ_φ8(z_skip, r_rest, x_outer) when z_partial is honest; the
    //    PCS catches mismatches downstream.
    let lambda = lagrange_weights_naive(k_skip, r_inner_skip);
    let w = inner_product(&lambda, &z_partial);

    // 9. Convert sumcheck challenges to LSB-first `x_inner_rest` order. The
    //    loop binds the TOP bit each round, so r_rounds[0] bound bit
    //    (inner_rest_len − 1) of the i_rest part (= bit (k_log − 1) of i).
    //    LSB-first: x_inner_rest[j] binds bit (k_skip + j) of i — i.e.,
    //    r_inner_rest[j] = r_rounds[inner_rest_len − 1 − j].
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();

    let proof = LincheckProof { rounds, z_partial };
    let claim = LincheckClaim {
        r_inner_skip,
        r_inner_rest,
        w,
    };
    (proof, claim, captured_z_vec)
}

/// Verify a lincheck proof. Walks the challenger in lockstep with `prove`,
/// performs the three scalar consistency checks against `v, v', v''`, and
/// derives the three output z claims.
pub fn verify<Ch: Challenger>(
    m: usize,
    k_log: usize,
    k_skip: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    challenger: &mut Ch,
) -> Result<LincheckClaim, VerifyError> {
    let k = 1usize << k_log;
    let n_log = m - k_log;

    if k_skip > k_log {
        return Err(VerifyError::KSkipExceedsKLog { k_skip, k_log });
    }
    let inner_rest_len = k_log - k_skip;
    let n_skip = 1usize << k_skip;

    if x_ab.x_inner_rest.len() != inner_rest_len {
        return Err(VerifyError::BadInnerRestLength {
            which: "x_ab",
            expected: inner_rest_len,
            got: x_ab.x_inner_rest.len(),
        });
    }
    if x_ab.x_outer.len() != n_log {
        return Err(VerifyError::BadOuterLength {
            which: "x_ab",
            expected: n_log,
            got: x_ab.x_outer.len(),
        });
    }
    if circuit.n_cols() != k {
        return Err(VerifyError::BadMatrixShape {
            which: "circuit",
            expected: k,
            got_rows: k,
            got_cols: circuit.n_cols(),
        });
    }
    if proof.rounds.len() != inner_rest_len {
        return Err(VerifyError::BadVectorLength {
            which: "rounds",
            expected: inner_rest_len,
            got: proof.rounds.len(),
        });
    }
    if proof.z_partial.len() != n_skip {
        return Err(VerifyError::BadVectorLength {
            which: "z_partial",
            expected: n_skip,
            got: proof.z_partial.len(),
        });
    }

    challenger.observe_label(b"flock-lincheck-v0");

    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    // 1. Sample α (matches prover's order).
    let alpha = challenger.sample_f128();

    // 2. Build α-batched comb_vec via the circuit's per-block fold (same call
    //    the prover made — sparse default delegates to the fused row-fold;
    //    per-hash impls walk the constraint graph directly).
    let t = std::time::Instant::now();
    let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
    if trace {
        eprintln!(
            "        [lcv] build_quirky_eq_table (2^{k_log}): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }
    let t = std::time::Instant::now();
    let mut comb_vec = circuit.fold_alpha_batched(alpha, &eq_inner);
    if trace {
        eprintln!(
            "        [lcv] circuit.fold_alpha_batched: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // 3. Replay the multilinear product-sumcheck (inner_rest_len rounds),
    //    folding comb_vec in lockstep so we end up with the "comb_partial"
    //    vector of length 2^k_skip. Parallel fold for the early (large) rounds.
    let t = std::time::Instant::now();
    // Constant-wire pin (mirror of prove): β sampled after α, comb gains +β at
    // the constant column, and the initial target gains +β·1 — the honest
    // all-ones constant column folds to 1. See docs/const-wire-pin.md.
    let mut target = alpha * v_a + v_b;
    if let Some(col) = circuit.const_pin_col() {
        let beta = challenger.sample_f128();
        comb_vec[col] += beta;
        target += beta;
    }
    let mut running = target;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    for &(e1, einf) in &proof.rounds {
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = challenger.sample_f128();
        // q(0) = claim + q(1) in char 2; q(X) = einf·X² + c1·X + e0.
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
        // Fold comb_vec at the same r (mirrors prover's fold).
        sumcheck_bind_top_in_place_par(&mut comb_vec, r);
        r_rounds.push(r);
    }
    debug_assert_eq!(comb_vec.len(), n_skip);
    if trace {
        eprintln!(
            "        [lcv] sumcheck replay + comb_vec fold ({} rounds): {}",
            inner_rest_len,
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // 4. Observe z_partial AFTER the sumcheck rounds (matches prover order).
    challenger.observe_f128_slice(&proof.z_partial);

    // 5. Final sumcheck consistency: Σ comb_partial[i_skip] · z_partial[i_skip]
    //    must equal the running claim. Ties z_partial to the upstream v_a, v_b.
    //    Small (length 2^k_skip = 64); sequential.
    let final_sum = inner_product(&comb_vec, &proof.z_partial);
    if running != final_sum {
        return Err(VerifyError::ConsistencyFailed {
            which: "sumcheck-final",
        });
    }

    // 6. Sample fresh z_skip AFTER z_partial — gives SZ on the φ8 dim.
    let r_inner_skip = challenger.sample_f128();

    // 7. Derive output claim value via φ8 Lagrange on z_partial at z_skip.
    //    Equals ẑ_φ8(z_skip, r_rest, x_outer) when z_partial is honest;
    //    PCS catches mismatches downstream.
    let t = std::time::Instant::now();
    let lambda = lagrange_weights_naive(k_skip, r_inner_skip);
    let w = inner_product(&lambda, &proof.z_partial);
    if trace {
        eprintln!(
            "        [lcv] final consistency + lagrange_weights_naive: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // 8. Convert sumcheck challenges to LSB-first x_inner_rest order
    //    (same convention as prover).
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();

    Ok(LincheckClaim {
        r_inner_skip,
        r_inner_rest,
        w,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    /// Naive MLE evaluation: `f̂(point) = Σ_i eq(point, i) · f[i]` where i ∈
    /// {0,1}^d and f[i] is given as a bool slice.
    fn mle_eval_bool(f: &[bool], point: &[F128]) -> F128 {
        let d = point.len();
        assert_eq!(f.len(), 1 << d);
        let eq = build_eq_table(point);
        let mut acc = F128::ZERO;
        for (i, &b) in f.iter().enumerate() {
            if b {
                acc += eq[i];
            }
        }
        acc
    }

    /// Sample a random `QuirkyPoint` for testing: z_skip ∈ F₁₂₈,
    /// x_inner_rest of length `k_log − k_skip`, x_outer of length `n_log`.
    fn random_quirky_point(m: usize, k_log: usize, k_skip: usize, rng: &mut Rng) -> QuirkyPoint {
        QuirkyPoint {
            z_skip: rng.f128(),
            x_inner_rest: rng.f128_vec(k_log - k_skip),
            x_outer: rng.f128_vec(m - k_log),
        }
    }

    /// "Quirky MLE evaluation" of a Boolean vector `f` at a quirky point.
    ///
    /// `ã(z_skip, x_inner_rest, x_outer) = Σ_i  f[i] · L_{i_skip}(z_skip)
    ///                                          · eq(x_inner_rest, i_inner_rest)
    ///                                          · eq(x_outer, i_outer)`
    ///
    /// where `i = i_skip + 2^k_skip · i_inner_rest + 2^k_log · i_outer` (matches
    /// the linear-LSB indexing of `f`).
    fn mle_eval_bool_quirky(
        f: &[bool],
        m: usize,
        k_log: usize,
        k_skip: usize,
        point: &QuirkyPoint,
    ) -> F128 {
        let k_skip_dim = 1usize << k_skip;
        let inner_rest_len = k_log - k_skip;
        let inner_rest_dim = 1usize << inner_rest_len;
        let k = 1usize << k_log;
        let n_outer = 1usize << (m - k_log);
        assert_eq!(f.len(), 1 << m);

        let lambda = crate::zerocheck::multilinear::lagrange_weights_naive(k_skip, point.z_skip);
        let eq_rest = build_eq_table(&point.x_inner_rest);
        let eq_outer = build_eq_table(&point.x_outer);
        debug_assert_eq!(lambda.len(), k_skip_dim);
        debug_assert_eq!(eq_rest.len(), inner_rest_dim);
        debug_assert_eq!(eq_outer.len(), n_outer);

        let mut acc = F128::ZERO;
        for i in 0..(1 << m) {
            if !f[i] {
                continue;
            }
            let i_skip = i & (k_skip_dim - 1);
            let i_inner_rest = (i >> k_skip) & (inner_rest_dim - 1);
            let i_outer = i / k;
            acc += lambda[i_skip] * eq_rest[i_inner_rest] * eq_outer[i_outer];
        }
        acc
    }

    /// Naive sparse matrix · bool-vector product: `out[i] = ⊕_{j: M[i,j]=1} z[j]`.
    fn matrix_vector_product(m: &SparseBinaryMatrix, z: &[bool]) -> Vec<bool> {
        assert_eq!(z.len(), m.num_cols);
        m.rows
            .iter()
            .map(|row| {
                let mut acc = false;
                for &col in row {
                    acc ^= z[col];
                }
                acc
            })
            .collect()
    }

    /// Build a block-diagonal full witness vector from a base matrix and the
    /// outer dimension: full[i_inner + i_outer · k] for the i_outer-th block.
    /// Used to construct `a = (I_{2^n_log} ⊗ A_0) · z` directly for tests.
    fn apply_block_diag(m_0: &SparseBinaryMatrix, z: &[bool], k_log: usize) -> Vec<bool> {
        let k = 1usize << k_log;
        assert_eq!(m_0.num_rows, k);
        assert_eq!(m_0.num_cols, k);
        assert_eq!(z.len() % k, 0);
        let n_outer = z.len() / k;
        let mut out = vec![false; z.len()];
        for i_outer in 0..n_outer {
            let z_block = &z[i_outer * k..(i_outer + 1) * k];
            let a_block = matrix_vector_product(m_0, z_block);
            out[i_outer * k..(i_outer + 1) * k].copy_from_slice(&a_block);
        }
        out
    }

    /// Build a sparse boolean matrix with `nnz` random nonzero entries among
    /// `k × k` slots. Used for tests.
    fn random_sparse_matrix(k: usize, nnz: usize, rng: &mut Rng) -> SparseBinaryMatrix {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut seen = std::collections::HashSet::new();
        let mut count = 0;
        while count < nnz {
            let r = (rng.next_u64() as usize) % k;
            let c = (rng.next_u64() as usize) % k;
            if seen.insert((r, c)) {
                rows[r].push(c);
                count += 1;
            }
        }
        for row in &mut rows {
            row.sort();
        }
        SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows,
        }
    }

    // ---- Unit tests for the kernels ----

    /// `build_eq_table` produces eq(point, i) for all boolean i.
    #[test]
    fn eq_table_matches_direct_formula() {
        for &d in &[1usize, 2, 3, 5, 8] {
            let mut rng = Rng::new(11 + d as u64);
            let point = rng.f128_vec(d);
            let table = build_eq_table(&point);
            assert_eq!(table.len(), 1 << d);
            for i in 0..(1 << d) {
                let mut expected = F128::ONE;
                for j in 0..d {
                    let bit = ((i >> j) & 1) as u64;
                    // eq(r, bit) = (1 + r) if bit = 0 else r
                    let factor = if bit == 0 {
                        F128::ONE + point[j]
                    } else {
                        point[j]
                    };
                    expected *= factor;
                }
                assert_eq!(table[i], expected, "mismatch at d={d}, i={i}");
            }
        }
    }

    /// `sparse_row_fold` matches a brute-force dense implementation.
    #[test]
    fn sparse_row_fold_matches_dense() {
        let mut rng = Rng::new(22);
        let k = 16;
        let nnz = 40;
        let matrix = random_sparse_matrix(k, nnz, &mut rng);
        let eq_table: Vec<F128> = rng.f128_vec(k);

        let got = sparse_row_fold(&matrix, &eq_table);

        // Brute force: for each col j, sum eq[i] over rows i where M[i,j] = 1.
        let mut expected = vec![F128::ZERO; k];
        for (i, row) in matrix.rows.iter().enumerate() {
            for &j in row {
                expected[j] += eq_table[i];
            }
        }
        assert_eq!(got, expected);
    }

    /// `partial_fold_packed_z` matches the direct sum.
    #[test]
    fn partial_fold_matches_direct() {
        for &(m, k_log) in &[(10usize, 3), (12, 4), (14, 5), (16, 8)] {
            let mut rng = Rng::new(33 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let outer_point = rng.f128_vec(n_log);
            let eq_outer = build_eq_table(&outer_point);

            let got = partial_fold_packed_z(&z_packed, m, k_log, &eq_outer);

            let k = 1usize << k_log;
            assert_eq!(got.len(), k);
            for i_inner in 0..k {
                let mut acc = F128::ZERO;
                for i_outer in 0..(1usize << n_log) {
                    let i = i_inner + i_outer * k;
                    if z[i] {
                        acc += eq_outer[i_outer];
                    }
                }
                assert_eq!(got[i_inner], acc, "mismatch at m={m}, i_inner={i_inner}");
            }
        }
    }

    /// `partial_fold_packed_z_fast` (parallel lookup-table) matches the scalar
    /// reference `partial_fold_packed_z`.
    #[test]
    fn partial_fold_fast_matches_serial() {
        for &(m, k_log) in &[(10usize, 3), (12, 4), (14, 5), (16, 8), (18, 10)] {
            let mut rng = Rng::new(800 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);

            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let fast = partial_fold_packed_z_fast(&z_packed, m, k_log, &eq);
            assert_eq!(serial, fast, "at m={m}, k_log={k_log}");
        }
    }

    /// NEON single-matrix kernel matches the scalar reference.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn partial_fold_neon_single_matches_serial() {
        for &(m, k_log) in &[(14usize, 4), (14, 5), (16, 5), (16, 8), (18, 10)] {
            if !n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
                continue;
            }
            let mut rng = Rng::new(7000 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);

            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let neon = partial_fold_packed_z_neon_single(&z_packed, m, k_log, &eq);
            assert_eq!(serial, neon, "at m={m}, k_log={k_log}");
            let iblock =
                partial_fold_packed_z_neon_iblock_padded(&z_packed, m, k_log, 1usize << k_log, &eq);
            assert_eq!(serial, iblock, "iblock at m={m}, k_log={k_log}");
        }
    }

    /// **Padding skip is byte-identical to the dense partial fold.** On a
    /// witness with honest zeros at rows `[useful_bits, 2^k_log)` of every
    /// block, the padded kernels (fast + NEON single) must produce the
    /// exact same `z_vec` as the dense kernels — and the dense scalar
    /// reference is the ground truth.
    ///
    /// Covers the three hash padding shapes plus a non-byte-aligned
    /// `useful_bits` to exercise the NEON's boundary block (rounded up to
    /// `BLOCK_K = 8`).
    #[test]
    fn partial_fold_padded_matches_dense() {
        // (m, k_log, useful_bits)
        let cases: &[(usize, usize, usize)] = &[
            // BLAKE3 (k_log=14, useful=15409 — boundary not byte-aligned).
            (17, 14, 15_409),
            // SHA-2  (k_log=15, useful=31401 — boundary not byte-aligned).
            (18, 15, 31_401),
            // Keccak (k_log=16, useful=42560 — exact byte boundary).
            (19, 16, 42_560),
        ];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xBADD_BEEF_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let n_log = m - k_log;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << n_log;

            // Random witness with bits [useful_bits, block_size) of every block
            // zeroed — mirrors the hash-module layout.
            let mut z = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    z[blk * block_size + j] = false;
                }
            }
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let outer_point = rng.f128_vec(n_log);
            let eq_outer = build_eq_table(&outer_point);

            let dense_fast = partial_fold_packed_z_fast(&z_packed, m, k_log, &eq_outer);
            let padded_fast =
                partial_fold_packed_z_fast_padded(&z_packed, m, k_log, useful_bits, &eq_outer);
            assert_eq!(
                dense_fast, padded_fast,
                "fast: m={m}, k_log={k_log}, useful={useful_bits}"
            );

            #[cfg(target_arch = "aarch64")]
            if n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
                let dense_neon = partial_fold_packed_z_neon_single(&z_packed, m, k_log, &eq_outer);
                let padded_neon = partial_fold_packed_z_neon_single_padded(
                    &z_packed,
                    m,
                    k_log,
                    useful_bits,
                    &eq_outer,
                );
                assert_eq!(
                    dense_neon, padded_neon,
                    "neon: m={m}, k_log={k_log}, useful={useful_bits}"
                );
                // i_inner-partitioned kernel: dense and padded must both match.
                let dense_iblock = partial_fold_packed_z_neon_iblock_padded(
                    &z_packed,
                    m,
                    k_log,
                    1usize << k_log,
                    &eq_outer,
                );
                let padded_iblock = partial_fold_packed_z_neon_iblock_padded(
                    &z_packed,
                    m,
                    k_log,
                    useful_bits,
                    &eq_outer,
                );
                assert_eq!(
                    dense_neon, dense_iblock,
                    "iblock dense: m={m}, k_log={k_log}, useful={useful_bits}"
                );
                assert_eq!(
                    dense_neon, padded_iblock,
                    "iblock padded: m={m}, k_log={k_log}, useful={useful_bits}"
                );
            }
        }
    }

    /// `partial_fold_packed_z(eq_outer) ↦ ẑ(·, x_outer)` matches direct MLE
    /// evaluation of z at `(i_inner, x_outer)` for boolean i_inner.
    #[test]
    fn partial_fold_is_mle_at_outer_point() {
        let m = 14;
        let k_log = 5;
        let k = 1 << k_log;
        let mut rng = Rng::new(44);
        let z = rng.bits(1 << m);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_outer = rng.f128_vec(m - k_log);
        let eq_outer = build_eq_table(&x_outer);

        let z_partial = partial_fold_packed_z(&z_packed, m, k_log, &eq_outer);

        // For each boolean i_inner ∈ {0,1}^k_log, the partial fold should
        // equal ẑ(i_inner, x_outer).
        for i_inner in 0..k {
            // Construct the m-dim point: first k_log coords from i_inner (boolean lifted),
            // then m-k_log coords from x_outer.
            let mut point = Vec::with_capacity(m);
            for j in 0..k_log {
                point.push(if (i_inner >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            point.extend_from_slice(&x_outer);
            let z_eval = mle_eval_bool(&z, &point);
            assert_eq!(z_partial[i_inner], z_eval, "i_inner={i_inner}");
        }
    }

    // ---- End-to-end prove/verify roundtrip on honest data ----

    /// Build a small honest instance: random sparse A_0/B_0/C_0, random z;
    /// compute a, b, c via apply_block_diag; pick three points; compute true
    /// MLE evals as v, v', v''. Roundtrip prove/verify, check claim matches
    /// what the verifier would re-derive from the (now-known-honest) z.
    #[test]
    fn prove_verify_roundtrip_honest() {
        // Exercise a range of k_skip values:
        //   k_skip = 0 (no skip)     — reduces to multilinear lincheck
        //   k_skip = k_log (max)     — only univariate inner
        //   k_skip < k_log (typical) — protocol-realistic case
        for &(m, k_log, k_skip) in &[
            (10usize, 4, 0),
            (10, 4, 2),
            (10, 4, 4),
            (12, 5, 3),
            (14, 7, 6),
            (14, 7, 0),
        ] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(55 + (m * 100 + k_log * 10 + k_skip) as u64);

            // Random sparse base matrices A_0, B_0 (no C since C = I in our use case).
            let nnz_per_mat = k * 2;
            let a_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);

            // Random witness z, then a = A·z, b = B·z.
            let z = rng.bits(1 << m);
            let a = apply_block_diag(&a_0, &z, k_log);
            let b = apply_block_diag(&b_0, &z, k_log);
            let z_packed = pack_z_lincheck(&z, m, k_log);

            // **One shared quirky point** (since zerocheck gives a, b claims at
            // the same point).
            let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);

            // True quirky-MLE eval claims at the shared point.
            let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
            let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

            // Prove and verify with matched challengers.
            let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
            let mut ch_p = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let claim_v = verify(
                m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &proof, &mut ch_v,
            )
            .unwrap_or_else(|e| {
                panic!("verify rejected honest proof at m={m},k_log={k_log},k_skip={k_skip}: {e:?}")
            });

            assert_eq!(
                claim_p, claim_v,
                "claim mismatch at m={m}, k_log={k_log}, k_skip={k_skip}"
            );

            // The single `w` value must match the true z quirky evaluation
            // at ((r_inner_skip, r_inner_rest), x_ab.x_outer).
            let pt = QuirkyPoint {
                z_skip: claim_v.r_inner_skip,
                x_inner_rest: claim_v.r_inner_rest.clone(),
                x_outer: x_ab.x_outer.clone(),
            };
            assert_eq!(
                claim_v.w,
                mle_eval_bool_quirky(&z, m, k_log, k_skip, &pt),
                "w wrong at m={m}, k_log={k_log}, k_skip={k_skip}"
            );
        }
    }

    /// Verify must reject byte-mutated proofs. Mutation positions are picked
    /// where the corresponding matrix row-vector entry is **nonzero** —
    /// otherwise the inner-product delta vanishes and the mutation is
    /// undetectable (a property of the random sparse matrix, not a verifier
    /// bug). The verifier's consistency check is sound for *any* mutation in
    /// a nonzero-weighted slot.
    #[test]
    fn verify_rejects_mutations() {
        let m = 12;
        let k_log = 4;
        let k_skip = 2;
        let k = 1 << k_log;
        let mut rng = Rng::new(66);
        let a_0 = random_sparse_matrix(k, k * 5, &mut rng);
        let b_0 = random_sparse_matrix(k, k * 5, &mut rng);
        let z = rng.bits(1 << m);
        let a = apply_block_diag(&a_0, &z, k_log);
        let b = apply_block_diag(&b_0, &z, k_log);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
        let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
        let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

        let _seed: u64 = 0xFEEDFACE;
        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

        // Pick a mutation position where BOTH row vectors are nonzero so the
        // mutation guarantees both checks would diverge.
        let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
        let row_a = sparse_row_fold(&a_0, &eq_inner);
        let row_b = sparse_row_fold(&b_0, &eq_inner);
        let idx = (0..k)
            .find(|&i| row_a[i] != F128::ZERO || row_b[i] != F128::ZERO)
            .expect("no row-vector slot is nonzero in either A or B — test degenerate");

        // Mutations now target `z_partial` (the post-sumcheck length-2^k_skip
        // vector). Bit-flipping any entry must cause the sumcheck-final check
        // to fail (running_claim ≠ Σ comb_partial · z_partial).
        let n_skip = 1usize << k_skip;
        let skip_idx = idx % n_skip;
        let mutations: Vec<(String, Box<dyn Fn(&LincheckProof) -> LincheckProof>)> = vec![
            (
                format!("z_partial[{skip_idx}].lo bit-flip"),
                Box::new(move |p| {
                    let mut q = p.clone();
                    q.z_partial[skip_idx].lo ^= 1;
                    q
                }),
            ),
            (
                format!("z_partial[{skip_idx}].hi bit-flip"),
                Box::new(move |p| {
                    let mut q = p.clone();
                    q.z_partial[skip_idx].hi ^= 1;
                    q
                }),
            ),
        ];
        for (label, mutate) in mutations {
            let bad = mutate(&proof);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &bad, &mut ch);
            assert!(
                matches!(res, Err(VerifyError::ConsistencyFailed { .. })),
                "verify did not reject {label}: got {res:?}"
            );
        }
    }

    /// Verify must reject shape errors.
    #[test]
    fn verify_rejects_shape_errors() {
        let m = 10;
        let k_log = 3;
        let k_skip = 1;
        let k = 1 << k_log;
        let mut rng = Rng::new(77);
        let a_0 = random_sparse_matrix(k, k, &mut rng);
        let b_0 = random_sparse_matrix(k, k, &mut rng);
        let z = rng.bits(1 << m);
        let a = apply_block_diag(&a_0, &z, k_log);
        let b = apply_block_diag(&b_0, &z, k_log);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
        let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
        let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

        // Truncate z_partial.
        let mut bad = proof.clone();
        bad.z_partial.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &bad, &mut ch),
            Err(VerifyError::BadVectorLength { .. })
        ));

        // Wrong x_inner_rest length.
        let mut ch = FsChallenger::new(b"flock-test-v0");
        let bad_x_ab = QuirkyPoint {
            z_skip: x_ab.z_skip,
            x_inner_rest: x_ab.x_inner_rest[..x_ab.x_inner_rest.len() - 1].to_vec(),
            x_outer: x_ab.x_outer.clone(),
        };
        assert!(matches!(
            verify(
                m, k_log, k_skip, &circuit, &bad_x_ab, v_a, v_b, &proof, &mut ch
            ),
            Err(VerifyError::BadInnerRestLength { .. })
        ));

        // k_skip > k_log.
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(
                m,
                k_log,
                k_log + 1,
                &circuit,
                &x_ab,
                v_a,
                v_b,
                &proof,
                &mut ch,
            ),
            Err(VerifyError::KSkipExceedsKLog { .. })
        ));
    }

    // ---- Deferred-reduction correctness ----

    /// The deferred-reduction `sumcheck_round_eval_par_deferred` must produce
    /// bit-identical output to the eager `sumcheck_round_eval_par` for all
    /// table sizes (both sequential and parallel branches).
    #[test]
    fn deferred_round_eval_matches_eager() {
        for &half_log in &[2usize, 4, 8, 13] {
            let half = 1usize << half_log;
            let len = half * 2;
            let mut rng = Rng::new(9000 + half_log as u64);
            let c = rng.f128_vec(len);
            let z = rng.f128_vec(len);

            let (e1_eager, einf_eager) = sumcheck_round_eval_par(&c, &z);
            let (e1_def, einf_def) = sumcheck_round_eval_par_deferred(&c, &z);

            assert_eq!(
                e1_eager, e1_def,
                "e1 mismatch at half_log={half_log}"
            );
            assert_eq!(
                einf_eager, einf_def,
                "einf mismatch at half_log={half_log}"
            );
        }
    }

    /// The deferred-reduction `sumcheck_bind_both_and_eval_next_deferred` must
    /// produce bit-identical output AND identical side-effects (table mutations)
    /// to the eager variant.
    #[test]
    fn deferred_bind_eval_next_matches_eager() {
        for &half2_log in &[1usize, 3, 6, 13] {
            let half2 = 1usize << half2_log;
            let len = half2 * 4;
            let mut rng = Rng::new(9100 + half2_log as u64);
            let comb_orig = rng.f128_vec(len);
            let z_orig = rng.f128_vec(len);
            let r = rng.f128();

            let mut comb_eager = comb_orig.clone();
            let mut z_eager = z_orig.clone();
            let (e1_eager, einf_eager) =
                sumcheck_bind_both_and_eval_next(&mut comb_eager, &mut z_eager, r);

            let mut comb_def = comb_orig.clone();
            let mut z_def = z_orig.clone();
            let (e1_def, einf_def) =
                sumcheck_bind_both_and_eval_next_deferred(&mut comb_def, &mut z_def, r);

            assert_eq!(e1_eager, e1_def, "e1 mismatch at half2_log={half2_log}");
            assert_eq!(
                einf_eager, einf_def,
                "einf mismatch at half2_log={half2_log}"
            );
            assert_eq!(
                comb_eager, comb_def,
                "comb side-effect mismatch at half2_log={half2_log}"
            );
            assert_eq!(
                z_eager, z_def,
                "z side-effect mismatch at half2_log={half2_log}"
            );
        }
    }

    /// Full lincheck prove/verify roundtrip using the deferred-reduction path
    /// (which is now the default in `prove_padded_inner`). This re-runs the
    /// same cases as `prove_verify_roundtrip_honest` to confirm transcript
    /// compatibility.
    #[test]
    fn deferred_reduction_lincheck_matches_eager() {
        for &(m, k_log, k_skip) in &[
            (10usize, 4, 0),
            (10, 4, 2),
            (10, 4, 4),
            (12, 5, 3),
            (14, 7, 6),
            (14, 7, 0),
        ] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(55 + (m * 100 + k_log * 10 + k_skip) as u64);

            let nnz_per_mat = k * 2;
            let a_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);

            let z = rng.bits(1 << m);
            let a = apply_block_diag(&a_0, &z, k_log);
            let b = apply_block_diag(&b_0, &z, k_log);
            let z_packed = pack_z_lincheck(&z, m, k_log);

            let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
            let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
            let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

            let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
            let mut ch_p = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) =
                prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let claim_v = verify(
                m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &proof, &mut ch_v,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "deferred verify rejected honest proof at m={m},k_log={k_log},k_skip={k_skip}: {e:?}"
                )
            });

            assert_eq!(
                claim_p, claim_v,
                "claim mismatch at m={m}, k_log={k_log}, k_skip={k_skip}"
            );

            let pt = QuirkyPoint {
                z_skip: claim_v.r_inner_skip,
                x_inner_rest: claim_v.r_inner_rest.clone(),
                x_outer: x_ab.x_outer.clone(),
            };
            assert_eq!(
                claim_v.w,
                mle_eval_bool_quirky(&z, m, k_log, k_skip, &pt),
                "w wrong at m={m}, k_log={k_log}, k_skip={k_skip}"
            );
        }
    }

    // ---- Sparse eq_inner correctness ----

    #[test]
    fn sparse_eq_inner_matches_dense() {
        for &(k_log, k_skip) in &[
            (4usize, 0),
            (4, 2),
            (6, 3),
            (8, 4),
            (10, 6),
            (15, 6),
        ] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(7700 + k_log as u64 * 100 + k_skip as u64);
            let x_inner_rest = rng.f128_vec(k_log - k_skip);
            let z_skip = rng.f128();

            let dense = build_quirky_eq_table(z_skip, &x_inner_rest, k_skip);
            assert_eq!(dense.len(), k);

            let useful = k / 4;
            let nnz_per_mat = useful * 2;
            let a_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);

            let compact = CompactCscCircuit::from_matrices(&a_0, &b_0);
            let sparse = build_quirky_eq_table_sparse(
                z_skip,
                &x_inner_rest,
                k_skip,
                &compact.referenced_rows,
            );
            assert_eq!(sparse.len(), compact.n_unique_rows());

            for (compact_idx, &orig_row) in compact.referenced_rows.iter().enumerate() {
                assert_eq!(
                    sparse[compact_idx], dense[orig_row as usize],
                    "mismatch at orig_row={orig_row}, k_log={k_log}, k_skip={k_skip}"
                );
            }
        }
    }

    #[test]
    fn compact_csc_fold_matches_dense_csc() {
        for &(k_log, k_skip) in &[
            (4usize, 0),
            (4, 2),
            (6, 3),
            (8, 4),
            (10, 6),
        ] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(8800 + k_log as u64 * 100 + k_skip as u64);
            let x_inner_rest = rng.f128_vec(k_log - k_skip);
            let z_skip = rng.f128();
            let alpha = rng.f128();

            let useful = k / 4;
            let nnz_per_mat = useful * 2;
            let a_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);

            let dense_csc = CscCircuit::from_matrices(&a_0, &b_0);
            let compact = CompactCscCircuit::from_matrices(&a_0, &b_0);

            let eq_inner_dense = build_quirky_eq_table(z_skip, &x_inner_rest, k_skip);
            let eq_inner_sparse = build_quirky_eq_table_sparse(
                z_skip,
                &x_inner_rest,
                k_skip,
                &compact.referenced_rows,
            );

            let comb_dense = dense_csc.fold_alpha_batched(alpha, &eq_inner_dense);
            let comb_compact = compact.fold_alpha_batched_compact(alpha, &eq_inner_sparse);

            assert_eq!(
                comb_dense, comb_compact,
                "fold mismatch at k_log={k_log}, k_skip={k_skip}"
            );
        }
    }

    #[test]
    fn compact_csc_trait_fold_matches_dense() {
        for &(k_log, k_skip) in &[(4usize, 2), (6, 3), (8, 4)] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(9900 + k_log as u64 * 100 + k_skip as u64);
            let x_inner_rest = rng.f128_vec(k_log - k_skip);
            let z_skip = rng.f128();
            let alpha = rng.f128();

            let useful = k / 4;
            let nnz_per_mat = useful * 2;
            let a_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);

            let dense_csc = CscCircuit::from_matrices(&a_0, &b_0);
            let compact = CompactCscCircuit::from_matrices(&a_0, &b_0);

            let eq_inner = build_quirky_eq_table(z_skip, &x_inner_rest, k_skip);

            let comb_dense = dense_csc.fold_alpha_batched(alpha, &eq_inner);
            let comb_trait = compact.fold_alpha_batched(alpha, &eq_inner);

            assert_eq!(
                comb_dense, comb_trait,
                "trait-fold mismatch at k_log={k_log}, k_skip={k_skip}"
            );
        }
    }

    #[test]
    fn compact_csc_prove_verify_roundtrip() {
        for &(m, k_log, k_skip) in &[
            (10usize, 4, 2),
            (12, 5, 3),
            (14, 7, 6),
        ] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(10100 + (m * 100 + k_log * 10 + k_skip) as u64);

            let useful = k / 3;
            let nnz_per_mat = useful * 2;
            let a_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);

            let z = rng.bits(1 << m);
            let a = apply_block_diag(&a_0, &z, k_log);
            let b = apply_block_diag(&b_0, &z, k_log);
            let z_packed = pack_z_lincheck(&z, m, k_log);

            let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
            let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
            let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

            let compact = CompactCscCircuit::from_matrices(&a_0, &b_0);
            let mut ch_p = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) =
                prove(&z_packed, m, k_log, k_skip, &compact, &x_ab, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let claim_v = verify(
                m, k_log, k_skip, &compact, &x_ab, v_a, v_b, &proof, &mut ch_v,
            )
            .unwrap_or_else(|e| {
                panic!("compact verify rejected at m={m},k_log={k_log},k_skip={k_skip}: {e:?}")
            });

            assert_eq!(
                claim_p, claim_v,
                "claim mismatch at m={m}, k_log={k_log}, k_skip={k_skip}"
            );
        }
    }

    #[test]
    fn sparse_eq_inner_speedup_measurement() {
        let k_log = 15usize;
        let k_skip = 6usize;
        let k = 1usize << k_log;
        let useful = 3094;
        let mut rng = Rng::new(5555);
        let x_inner_rest = rng.f128_vec(k_log - k_skip);
        let z_skip = rng.f128();
        let alpha = rng.f128();

        let nnz_per_mat = useful * 3;
        let a_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);
        let b_0 = random_sparse_matrix_in_range(k, useful, nnz_per_mat, &mut rng);

        let dense_csc = CscCircuit::from_matrices(&a_0, &b_0);
        let compact = CompactCscCircuit::from_matrices(&a_0, &b_0);

        eprintln!(
            "K={k}, useful_rows={useful}, n_unique_rows={}, a_nnz={}, b_nnz={}",
            compact.n_unique_rows(),
            compact.a_rows_compact.len(),
            compact.b_rows_compact.len(),
        );

        let n_iter = 50;

        let mut dense_times = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let t = std::time::Instant::now();
            let eq = build_quirky_eq_table(z_skip, &x_inner_rest, k_skip);
            let _comb = dense_csc.fold_alpha_batched(alpha, &eq);
            dense_times.push(t.elapsed());
        }

        let mut sparse_times = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let t = std::time::Instant::now();
            let eq = build_quirky_eq_table_sparse(
                z_skip,
                &x_inner_rest,
                k_skip,
                &compact.referenced_rows,
            );
            let _comb = compact.fold_alpha_batched_compact(alpha, &eq);
            sparse_times.push(t.elapsed());
        }

        dense_times.sort();
        sparse_times.sort();

        let dense_median = dense_times[n_iter / 2].as_secs_f64() * 1e3;
        let sparse_median = sparse_times[n_iter / 2].as_secs_f64() * 1e3;
        let speedup = dense_median / sparse_median;

        eprintln!(
            "  dense  (build_quirky_eq + fold): median {dense_median:.3} ms"
        );
        eprintln!(
            "  sparse (build_sparse + fold_compact): median {sparse_median:.3} ms"
        );
        eprintln!("  speedup: {speedup:.2}x");
        eprintln!(
            "  savings: {:.3} ms ({:.1}%)",
            dense_median - sparse_median,
            (1.0 - sparse_median / dense_median) * 100.0
        );
    }

    fn random_sparse_matrix_in_range(
        k: usize,
        useful_rows: usize,
        nnz: usize,
        rng: &mut Rng,
    ) -> SparseBinaryMatrix {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut seen = std::collections::HashSet::new();
        let mut count = 0;
        while count < nnz {
            let r = (rng.next_u64() as usize) % useful_rows;
            let c = (rng.next_u64() as usize) % k;
            if seen.insert((r, c)) {
                rows[r].push(c);
                count += 1;
            }
        }
        for row in &mut rows {
            row.sort();
        }
        SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows,
        }
    }
}
