//! Block-diagonal R1CS over GF(2).
//!
//! The standard R1CS is `(A·z) ⊙ (B·z) ⊕ (C·z) = 0`. We fix `C = I` (the
//! circuit-R1CS shape `(A·z) ⊙ (B·z) = z`), so the c-claim emitted by
//! zerocheck is already a `z`-claim — no transformation needed downstream.
//!
//! We further specialize to **block-diagonal `A` and `B`**:
//!   `A = I_{2^n_log} ⊗ A_0`, etc. The base matrices are `k × k` sparse
//! boolean (`k = 2^k_log`). `C_0 = I_k` is implicit (we still carry the
//! materialized `c_0` matrix for utilities like `satisfies`).

/// Sparse boolean matrix. `rows[i]` lists the column indices where the entry is 1.
#[derive(Clone, Debug)]
pub struct SparseBinaryMatrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub rows: Vec<Vec<usize>>,
}

/// Memory/variable layout of the committed witness (address bit `i` of the
/// packed buffer = MLE variable `i`).
///
/// - **RowMajor** (legacy): `addr = [k_log inner bits | n_log batch bits]` —
///   each instance is one contiguous `2^k_log`-bit block.
/// - **BatchMajor**: `addr = [7 in-word bits | n_log batch | k_log−7 chunk]`
///   — column-major at 128-bit chunk granularity. The sumcheck binds the
///   batch dims right after the univariate skip + one in-word round (the
///   fold-log_n-first order required for jagged multi-table composition),
///   the batch dims live over the ring-switch suffix (packed words), and
///   per-block zero padding coalesces into one contiguous buffer suffix.
///
/// Convention for **`ZClaim` points under BatchMajor**: `x_inner_rest`
/// holds only the address-dim-6 coordinate and `x_outer` holds
/// `[batch…, chunk…]`, so the PCS-side concatenation
/// `x_inner_rest ++ x_outer` yields the address-ordered suffix for the
/// committed polynomial. (Lincheck's *semantic* `QuirkyPoint` is unchanged
/// in both layouts.) Requires `k_log ≥ 7` and `k_skip = 6`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WitnessLayout {
    #[default]
    RowMajor,
    BatchMajor,
}

/// Block-diagonal R1CS instance.
///
/// Total witness length: `N = 2^m = 2^k_log · 2^n_log`.
/// Base matrices `A_0`, `B_0`, `C_0` are each `k × k` with `k = 2^k_log`.
///
/// `k_skip` is the zerocheck's univariate-skip dimension (`k_skip ≤ k_log`).
/// It defines how the m-dim claim point is laid out in the protocol: one
/// univariate F128 coord binds the LSB `k_skip` bits, `k_log − k_skip`
/// multilinear F128 coords bind the next inner bits, and `n_log` multilinear
/// F128 coords bind the outer bits.
#[derive(Debug)]
pub struct BlockR1cs {
    pub m: usize,
    pub k_log: usize,
    pub k_skip: usize,
    /// Useful bits per block: rows `[0, useful_bits)` of each block carry real
    /// witness data; rows `[useful_bits, 2^k_log)` are zero padding (and have
    /// empty rows in `a_0/b_0`). Default `1 << k_log` (no padding). The prover
    /// can use this to skip URM work on chunks that fall entirely in padding.
    pub useful_bits: usize,
    pub a_0: SparseBinaryMatrix,
    pub b_0: SparseBinaryMatrix,
    pub c_0: SparseBinaryMatrix,
    /// Memory/variable layout of the committed witness (see [`WitnessLayout`]).
    /// Bound into [`Self::statement_digest`]; the prover and verifier derive
    /// their claim-point assembly from it.
    pub layout: WitnessLayout,
    /// Column of a constant-one wire to pin to 1 across all blocks, or `None`.
    /// Drives the lincheck constant-wire pin for matrix-based encoders whose
    /// circuit is built from these matrices (BLAKE3, SHA-2 via
    /// [`Self::csc_lincheck_circuit`]). Walker-based encoders (Keccak) set this
    /// `None` and carry the pin on their own `LincheckCircuit`. See
    /// `docs/const-wire-pin.md`.
    pub const_pin: Option<usize>,
    /// Lazily-cached BLAKE3 digest of the R1CS instance — see
    /// [`Self::statement_digest`]. Computed on first access; reused thereafter
    /// (the matrices are public fields, so callers that mutate them after the
    /// cache is populated will see a stale digest — don't do that).
    #[doc(hidden)]
    pub digest_cache: std::sync::OnceLock<[u8; 32]>,
    /// Lazily-cached CSC transpose of `(a_0, b_0)` for lincheck's
    /// `fold_alpha_batched` — see [`Self::csc_lincheck_circuit`]. Same
    /// stale-cache caveat as [`Self::digest_cache`].
    #[doc(hidden)]
    pub csc_cache: std::sync::OnceLock<crate::lincheck::CscCircuit>,
}

// Manual Clone — std::sync::OnceLock doesn't derive Clone, and a fresh cache
// after cloning is the right behavior (recomputes lazily on first use).
impl Clone for BlockR1cs {
    fn clone(&self) -> Self {
        Self {
            m: self.m,
            k_log: self.k_log,
            k_skip: self.k_skip,
            useful_bits: self.useful_bits,
            a_0: self.a_0.clone(),
            b_0: self.b_0.clone(),
            c_0: self.c_0.clone(),
            layout: self.layout,
            const_pin: self.const_pin,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }
}

impl BlockR1cs {
    /// Number of A_0-blocks tiled diagonally = 2^n_log.
    pub fn n_outer(&self) -> usize {
        1usize << self.n_log()
    }
    /// Outer-dimension log count = m − k_log.
    pub fn n_log(&self) -> usize {
        self.m - self.k_log
    }
    /// Inner dimension = 2^k_log = base-matrix side.
    pub fn k(&self) -> usize {
        1usize << self.k_log
    }
    /// Total witness length = 2^m.
    pub fn n(&self) -> usize {
        1usize << self.m
    }

    /// Whether `c_0` is exactly the identity matrix — the circuit-R1CS
    /// convention (`C = I` ⇒ `c = C·z = z`). The generic provers use this to
    /// alias `c` to `z` instead of running a full block-diagonal apply.
    pub fn c0_is_identity(&self) -> bool {
        self.c_0.num_rows == self.k()
            && self.c_0.num_cols == self.k()
            && self
                .c_0
                .rows
                .iter()
                .enumerate()
                .all(|(i, row)| row.as_slice() == [i])
    }

    /// Default `LincheckCircuit` wrapping this R1CS's sparse matrices.
    /// Per-hash setups that supply a custom circuit walker bypass this.
    pub fn sparse_lincheck_circuit(&self) -> crate::lincheck::SparseMatrixCircuit<'_> {
        crate::lincheck::SparseMatrixCircuit::new(&self.a_0, &self.b_0)
            .with_const_pin(self.const_pin)
    }

    /// CSC-transposed `LincheckCircuit` over this R1CS's sparse matrices —
    /// the fastest `fold_alpha_batched` when `a_0`/`b_0` are materialized
    /// (gather per column instead of scatter per row). Built lazily on first
    /// access and cached; call once at setup to keep the build cost (one pass
    /// over the nonzeros) out of the prove path. NOT meaningful for setups
    /// whose `BlockR1cs` carries empty matrix stubs (e.g. Keccak) — those
    /// must keep their circuit walkers.
    pub fn csc_lincheck_circuit(&self) -> &crate::lincheck::CscCircuit {
        self.csc_cache.get_or_init(|| {
            crate::lincheck::CscCircuit::from_matrices(&self.a_0, &self.b_0)
                .with_const_pin(self.const_pin)
        })
    }

    /// Apply `A = I_{2^n_log} ⊗ A_0` to a Boolean witness `z`. Returns
    /// `a = A · z` ∈ GF(2)^N (length 2^m).
    pub fn apply_a(&self, z: &[bool]) -> Vec<bool> {
        apply_block_diag(&self.a_0, z, self.k_log)
    }

    /// Apply `B = I_{2^n_log} ⊗ B_0` to `z`.
    pub fn apply_b(&self, z: &[bool]) -> Vec<bool> {
        apply_block_diag(&self.b_0, z, self.k_log)
    }

    /// Apply `C = I_{2^n_log} ⊗ C_0` to `z`.
    pub fn apply_c(&self, z: &[bool]) -> Vec<bool> {
        apply_block_diag(&self.c_0, z, self.k_log)
    }

    /// Check whether `(A·z) ⊙ (B·z) = C·z` (over GF(2), Hadamard product).
    pub fn satisfies(&self, z: &[bool]) -> bool {
        assert_eq!(z.len(), self.n());
        let a = self.apply_a(z);
        let b = self.apply_b(z);
        let c = self.apply_c(z);
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .all(|((ai, bi), ci)| (*ai & *bi) == *ci)
    }

    // -----------------------------------------------------------------------
    // Packed variants: operate on F_{2^128}-packed witnesses (polynomial-basis
    // bit layout: bit r of z_packed[i] = logical bit i·128 + r). This is the
    // canonical witness form throughout the protocol.
    // -----------------------------------------------------------------------

    /// Packed `a = A · z` ∈ GF(2)^N. Output is F_{2^128}-packed (length 2^(m-7)).
    pub fn apply_a_packed(&self, z_packed: &[crate::field::F128]) -> Vec<crate::field::F128> {
        apply_block_diag_packed(&self.a_0, z_packed, self.m, self.k_log)
    }

    /// Packed `b = B · z`.
    pub fn apply_b_packed(&self, z_packed: &[crate::field::F128]) -> Vec<crate::field::F128> {
        apply_block_diag_packed(&self.b_0, z_packed, self.m, self.k_log)
    }

    /// Packed `c = C · z`.
    pub fn apply_c_packed(&self, z_packed: &[crate::field::F128]) -> Vec<crate::field::F128> {
        apply_block_diag_packed(&self.c_0, z_packed, self.m, self.k_log)
    }

    // -----------------------------------------------------------------------
    // Layout-aware protocol bookkeeping — the single source of truth for how
    // the zerocheck's address-ordered challenges map to lincheck's semantic
    // quirky point and to the PCS claims' address-ordered points. Shared by
    // `flock_prover::prover::prove_fast_core` and `verifier::verify_core`
    // (any divergence between the two is a transcript break, so both call
    // these).
    // -----------------------------------------------------------------------

    /// Witness padding descriptor for URM / PCS work-skipping under this
    /// layout. Both are single-run [`crate::zerocheck::PaddingSpec`]s tiling
    /// the whole domain (the kernels' fast path). RowMajor: per-block useful
    /// prefix (padding interleaved at each block's tail). BatchMajor: the
    /// padding chunk-columns coalesce into ONE contiguous buffer suffix,
    /// expressed as a single giant block (`k_log = m`) with a useful prefix.
    pub fn padding_spec(&self) -> crate::zerocheck::PaddingSpec {
        match self.layout {
            WitnessLayout::RowMajor => crate::zerocheck::PaddingSpec::uniform(
                self.k_log,
                self.useful_bits,
                1usize << self.n_log(),
            ),
            WitnessLayout::BatchMajor => crate::zerocheck::PaddingSpec::uniform(
                self.m,
                self.useful_bits.div_ceil(128) << (7 + self.n_log()),
                1,
            ),
        }
    }

    /// Lincheck's **semantic** quirky point from the zerocheck claim: split
    /// the address-ordered `mlv_challenges` into (inner-rest, outer=batch)
    /// coordinates. RowMajor address order is `[inner-rest | batch]`;
    /// BatchMajor is `[dim6 | batch | chunk]` with the inner-rest coords
    /// being `[dim6, chunk…]`.
    pub fn x_ab_from_mlv(
        &self,
        z_skip: crate::field::F128,
        mlv: &[crate::field::F128],
    ) -> crate::lincheck::QuirkyPoint {
        let inner_rest_len = self.k_log - self.k_skip;
        assert_eq!(mlv.len(), self.m - self.k_skip);
        match self.layout {
            WitnessLayout::RowMajor => crate::lincheck::QuirkyPoint {
                z_skip,
                x_inner_rest: mlv[..inner_rest_len].to_vec(),
                x_outer: mlv[inner_rest_len..].to_vec(),
            },
            WitnessLayout::BatchMajor => {
                assert!(
                    self.k_log >= 7 && self.k_skip == 6,
                    "BatchMajor needs k_log >= 7, k_skip = 6"
                );
                let n_log = self.n_log();
                let mut x_inner_rest = Vec::with_capacity(inner_rest_len);
                x_inner_rest.push(mlv[0]);
                x_inner_rest.extend_from_slice(&mlv[1 + n_log..]);
                crate::lincheck::QuirkyPoint {
                    z_skip,
                    x_inner_rest,
                    x_outer: mlv[1..1 + n_log].to_vec(),
                }
            }
        }
    }

    /// Address-ordered `ZClaim` point for the AB claim after lincheck
    /// replaces the inner coordinates with `(r_inner_skip, r_inner_rest)`.
    /// See [`WitnessLayout`] for the BatchMajor point convention.
    pub fn ab_claim_point(
        &self,
        r_inner_skip: crate::field::F128,
        r_inner_rest: &[crate::field::F128],
        x_outer: &[crate::field::F128],
    ) -> crate::lincheck::QuirkyPoint {
        match self.layout {
            WitnessLayout::RowMajor => crate::lincheck::QuirkyPoint {
                z_skip: r_inner_skip,
                x_inner_rest: r_inner_rest.to_vec(),
                x_outer: x_outer.to_vec(),
            },
            WitnessLayout::BatchMajor => {
                let mut suffix = Vec::with_capacity(x_outer.len() + r_inner_rest.len() - 1);
                suffix.extend_from_slice(x_outer);
                suffix.extend_from_slice(&r_inner_rest[1..]);
                crate::lincheck::QuirkyPoint {
                    z_skip: r_inner_skip,
                    x_inner_rest: vec![r_inner_rest[0]],
                    x_outer: suffix,
                }
            }
        }
    }

    /// Address-ordered `ZClaim` point for the C claim from the zerocheck's
    /// `r_rest` (which is address-ordered in both layouts).
    pub fn c_claim_point(
        &self,
        z_skip: crate::field::F128,
        r_rest: &[crate::field::F128],
    ) -> crate::lincheck::QuirkyPoint {
        let inner_rest_len = self.k_log - self.k_skip;
        match self.layout {
            WitnessLayout::RowMajor => crate::lincheck::QuirkyPoint {
                z_skip,
                x_inner_rest: r_rest[..inner_rest_len].to_vec(),
                x_outer: r_rest[inner_rest_len..].to_vec(),
            },
            WitnessLayout::BatchMajor => crate::lincheck::QuirkyPoint {
                z_skip,
                x_inner_rest: vec![r_rest[0]],
                x_outer: r_rest[1..].to_vec(),
            },
        }
    }

    /// BLAKE3 hash of the R1CS instance (parameters + sparse matrices).
    /// Stable across runs; used to bind the Fiat-Shamir transcript to the
    /// statement being proved.
    ///
    /// Lazily cached in [`Self::digest_cache`]; first call materializes it
    /// (matters for dense matrices — BLAKE3's R1CS has ~21M nonzeros, ~250 ms
    /// to hash), subsequent calls are essentially free.
    pub fn statement_digest(&self) -> [u8; 32] {
        *self.digest_cache.get_or_init(|| {
            let mut h = blake3::Hasher::new();
            h.update(b"flock-r1cs-stmt-v1");
            h.update(&(self.m as u64).to_le_bytes());
            h.update(&(self.k_log as u64).to_le_bytes());
            h.update(&(self.k_skip as u64).to_le_bytes());
            // The layout determines which polynomial a given witness commits
            // to — it is part of the statement.
            h.update(&[match self.layout {
                WitnessLayout::RowMajor => 0u8,
                WitnessLayout::BatchMajor => 1u8,
            }]);
            absorb_matrix(&mut h, &self.a_0);
            absorb_matrix(&mut h, &self.b_0);
            absorb_matrix(&mut h, &self.c_0);
            *h.finalize().as_bytes()
        })
    }

    /// Check the R1CS constraint `(A·z) ⊙ (B·z) = C·z` over GF(2) on a packed
    /// witness. Per-element check is `a & b == c` bitwise.
    pub fn satisfies_packed(&self, z_packed: &[crate::field::F128]) -> bool {
        use crate::field::F128;
        assert_eq!(z_packed.len(), 1usize << (self.m - 7));
        let a = self.apply_a_packed(z_packed);
        let b = self.apply_b_packed(z_packed);
        let c = self.apply_c_packed(z_packed);
        a.iter().zip(b.iter()).zip(c.iter()).all(|((ai, bi), ci)| {
            let ab = F128 {
                lo: ai.lo & bi.lo,
                hi: ai.hi & bi.hi,
            };
            ab == *ci
        })
    }
}

/// Length-prefixed absorption of a sparse matrix into a BLAKE3 hasher.
/// `(num_rows, num_cols, [(row_len, col_indices...) for each row])`, all
/// little-endian u64, so two matrices with different shapes/contents always
/// produce different states.
fn absorb_matrix(h: &mut blake3::Hasher, m: &SparseBinaryMatrix) {
    h.update(&(m.num_rows as u64).to_le_bytes());
    h.update(&(m.num_cols as u64).to_le_bytes());
    for row in &m.rows {
        h.update(&(row.len() as u64).to_le_bytes());
        for &col in row {
            h.update(&(col as u64).to_le_bytes());
        }
    }
}

/// Block-diagonal `(I_{2^n_log} ⊗ M_0) · z` over GF(2).
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

/// Block-diagonal `(I_{2^n_log} ⊗ M_0) · z_packed` over GF(2), packed form.
///
/// Computes `out_packed[i_packed] = (A · z)[i_packed * 128 .. i_packed * 128 + 128]`
/// for an F_{2^128}-packed witness `z_packed` of length `2^(m-7)`.
///
/// Block size `k = 2^k_log`; total witness has `n_outer = 2^(m - k_log)` blocks.
/// Each block is processed independently (block-diagonal). The implementation
/// is bit-parallel-within-a-row but iterates rows of the sparse matrix.
///
/// Cost dominator: for each block, `Σ_r |M_0.row(r)|` bit-lookups + one set per
/// nonzero output bit. For typical sparse `M_0` with avg row size `s`, total
/// = `n_outer · k · s` bit ops.
pub fn apply_block_diag_packed(
    m_0: &SparseBinaryMatrix,
    z_packed: &[crate::field::F128],
    m: usize,
    k_log: usize,
) -> Vec<crate::field::F128> {
    use crate::field::F128;
    use rayon::prelude::*;

    let k = 1usize << k_log;
    assert_eq!(m_0.num_rows, k);
    assert_eq!(m_0.num_cols, k);
    let n_packed = 1usize << (m - 7);
    assert_eq!(z_packed.len(), n_packed);
    let n_outer = 1usize << (m - k_log);

    let mut out = vec![F128::ZERO; n_packed];

    if k_log >= 7 {
        // Fast path: flatten the matrix to CSR once (one pass over the
        // Vec<Vec> rows — amortized over the n_outer block applications),
        // then process the blocks in parallel STRIPs of 8. For each strip,
        // M_0's nonzeros are streamed ONCE and each (row, col)'s index
        // decode is shared by all 8 instances, instead of re-walking the
        // whole matrix (and chasing one Vec per row) for every block.
        // The strip working set is 8 blocks of z + out (8 KB at k = 2^15),
        // L1-resident; the CSR arrays stream sequentially.
        let (row_ptr, cols) = flatten_csr(m_0);
        let f128_per_block = k / 128;
        // Strip width: 64 (u64 column bits, ~5x the 8-wide kernel) when
        // there are enough 64-block strips to keep every rayon worker busy;
        // otherwise 8 (more, smaller tasks). Single-threaded runs always
        // qualify for 64.
        let strip = if n_outer / 64 >= rayon::current_num_threads().max(1) {
            64
        } else {
            APPLY_STRIP
        };
        out.par_chunks_mut(strip * f128_per_block)
            .zip(z_packed.par_chunks(strip * f128_per_block))
            .for_each(|(out_strip, z_strip)| {
                let n_blocks = z_strip.len() / f128_per_block;
                if n_blocks == 64 {
                    apply_strip64_csr(&row_ptr, &cols, z_strip, out_strip, f128_per_block);
                } else if n_blocks == APPLY_STRIP {
                    apply_strip_csr(&row_ptr, &cols, z_strip, out_strip, f128_per_block);
                } else {
                    // Tail strip: per-block.
                    for (ob, zb) in out_strip
                        .chunks_mut(f128_per_block)
                        .zip(z_strip.chunks(f128_per_block))
                    {
                        apply_one_block_csr(&row_ptr, &cols, zb, ob);
                    }
                }
            });
    } else {
        // Slow path (k_log < 7): blocks straddle F128 elements. Used only by
        // small tests; production R1CS always has k_log ≥ 7.
        for i_outer in 0..n_outer {
            let block_start_bit = i_outer * k;
            for r in 0..k {
                let mut acc = false;
                for &j in m_0.rows[r].iter() {
                    let global_in_bit = block_start_bit + j;
                    if get_bit_packed(z_packed, global_in_bit) {
                        acc = !acc;
                    }
                }
                if acc {
                    set_bit_packed(&mut out, block_start_bit + r);
                }
            }
        }
    }
    out
}

/// Number of blocks processed per matrix pass in the packed fast path.
/// 8 keeps the per-strip z/out working set L1-resident (8 KB at k = 2^15)
/// while cutting the matrix-stream and index-decode cost 8x vs per-block.
const APPLY_STRIP: usize = 8;

/// Flatten `m.rows` (Vec<Vec<usize>>) into CSR arrays: row `r`'s column
/// indices are `cols[row_ptr[r] as usize .. row_ptr[r+1] as usize]`. One
/// sequential pass; removes the per-row Vec indirection from the hot loops.
fn flatten_csr(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    assert!(m.num_cols <= u32::MAX as usize);
    let nnz: usize = m.rows.iter().map(|r| r.len()).sum();
    let mut row_ptr = Vec::with_capacity(m.num_rows + 1);
    let mut cols = Vec::with_capacity(nnz);
    row_ptr.push(0u32);
    for row in &m.rows {
        for &c in row {
            cols.push(c as u32);
        }
        row_ptr.push(cols.len() as u32);
    }
    (row_ptr, cols)
}

/// View a block of F128s as u128 words (F128 is repr(C, align(16)) with two
/// little-endian u64s — bit `b` of the u128 is logical bit `b` of the block).
#[inline]
fn as_u128s(block: &[crate::field::F128]) -> &[u128] {
    // SAFETY: F128 has u128's size and alignment on all supported targets;
    // the lo/hi little-endian layout matches the u128 bit order.
    unsafe { std::slice::from_raw_parts(block.as_ptr() as *const u128, block.len()) }
}

/// Apply `M_0` (CSR form) to APPLY_STRIP = 8 consecutive blocks at once.
///
/// Phase 1 transposes the strip's witness to column-major bytes:
/// `colbits[j]` holds column j's bit for all 8 blocks (bit s = block s).
/// Phase 2 streams the matrix ONCE for the whole strip; each nonzero is a
/// single byte load + XOR (`row_bits ^= colbits[j]`), so the word/shift
/// decode and the load both amortize 8x vs the per-block kernel — and each
/// witness bit is gathered once instead of once per row it appears in
/// (avg nnz/k ≈ 29 times for the sha2 A_0).
///
/// Working set: colbits = k bytes (32 KB at k = 2^15) + the strip's z/out —
/// L1-resident on Apple P-cores (128 KB L1d).
fn apply_strip_csr(
    row_ptr: &[u32],
    cols: &[u32],
    z_strip: &[crate::field::F128],
    out_strip: &mut [crate::field::F128],
    f128_per_block: usize,
) {
    use crate::bits::transpose_8_u64s_to_64_bytes;

    debug_assert_eq!(z_strip.len(), APPLY_STRIP * f128_per_block);
    debug_assert_eq!(out_strip.len(), APPLY_STRIP * f128_per_block);
    let k = f128_per_block * 128;
    let u64_per_block = k / 64;
    // SAFETY: F128 is repr(C, align(16)) = two little-endian u64s; viewing the
    // strip as u64 words preserves bit order within each block.
    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_strip.as_ptr() as *const u64, z_strip.len() * 2) };

    // Phase 1: bit-transpose the 8 blocks to column-major bytes.
    let mut colbits = vec![0u8; k];
    for w in 0..u64_per_block {
        let lanes: [u64; 8] = std::array::from_fn(|s| z_u64[s * u64_per_block + w]);
        transpose_8_u64s_to_64_bytes(&lanes, &mut colbits[w * 64..w * 64 + 64]);
    }

    // Phase 2: one matrix pass for all 8 blocks; 1 byte-lookup per nonzero.
    for out_idx in 0..f128_per_block {
        let mut acc = [0u128; APPLY_STRIP];
        for offset in 0..128 {
            let r = out_idx * 128 + offset;
            let lo = row_ptr[r] as usize;
            let hi = row_ptr[r + 1] as usize;
            // Bit s of `row_bits` = output bit r of block s.
            let mut row_bits: u8 = 0;
            for &j in &cols[lo..hi] {
                row_bits ^= colbits[j as usize];
            }
            if row_bits != 0 {
                for (s, a) in acc.iter_mut().enumerate() {
                    *a |= (((row_bits >> s) & 1) as u128) << offset;
                }
            }
        }
        for (s, a) in acc.iter().enumerate() {
            if *a != 0 {
                let slot = &mut out_strip[s * f128_per_block + out_idx];
                slot.lo |= *a as u64;
                slot.hi |= (*a >> 64) as u64;
            }
        }
    }
}

/// 64×64 bit-matrix transpose, LSB-first bit numbering: output word t bit s
/// = input word s bit t. Hacker's Delight 7-3 delta-swap network, mirrored
/// for LSB-first (high half of a[k] swaps with low half of a[k+j]).
#[inline]
fn transpose_64x64(a: &mut [u64; 64]) {
    let mut j: usize = 32;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k: usize = 0;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// 64-block strip kernel: like [`apply_strip_csr`] but with u64 column bits,
/// so one matrix pass serves 64 instances and each nonzero costs one u64
/// load + XOR. Both the input AND the output are bit-transposed per strip
/// (the rows×blocks intermediate stays column-major); the two transposes
/// cost O(k) words each vs O(nnz) lookups — negligible for nnz/k ≳ 8.
/// ~5x the 8-wide kernel on the sha2 matrices (more single-threaded, where
/// it is purely compute-bound).
fn apply_strip64_csr(
    row_ptr: &[u32],
    cols: &[u32],
    z_strip: &[crate::field::F128],
    out_strip: &mut [crate::field::F128],
    f128_per_block: usize,
) {
    const S: usize = 64;
    debug_assert_eq!(z_strip.len(), S * f128_per_block);
    let k = f128_per_block * 128;
    let u64_per_block = k / 64;
    // SAFETY: F128 is repr(C, align(16)) = two little-endian u64s; u64 views
    // preserve bit order within each block.
    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_strip.as_ptr() as *const u64, z_strip.len() * 2) };
    let out_u64: &mut [u64] = unsafe {
        std::slice::from_raw_parts_mut(out_strip.as_mut_ptr() as *mut u64, out_strip.len() * 2)
    };

    // Transpose in: colbits[j] = column j's bit across the 64 blocks.
    let mut colbits = vec![0u64; k];
    let mut lanes = [0u64; 64];
    for w in 0..u64_per_block {
        for (s, lane) in lanes.iter_mut().enumerate() {
            *lane = z_u64[s * u64_per_block + w];
        }
        transpose_64x64(&mut lanes);
        colbits[w * 64..w * 64 + 64].copy_from_slice(&lanes);
    }

    // One matrix pass: rowbits[r] = XOR of colbits over row r's nonzeros.
    let mut rowbits = vec![0u64; k];
    for (r, rb) in rowbits.iter_mut().enumerate() {
        let lo = row_ptr[r] as usize;
        let hi = row_ptr[r + 1] as usize;
        let mut x = 0u64;
        for &j in &cols[lo..hi] {
            x ^= colbits[j as usize];
        }
        *rb = x;
    }

    // Transpose out: back to block-major.
    for w in 0..u64_per_block {
        lanes.copy_from_slice(&rowbits[w * 64..w * 64 + 64]);
        transpose_64x64(&mut lanes);
        for (s, lane) in lanes.iter().enumerate() {
            out_u64[s * u64_per_block + w] = *lane;
        }
    }
}

/// Single-block CSR kernel (tail strips whose block count < APPLY_STRIP).
fn apply_one_block_csr(
    row_ptr: &[u32],
    cols: &[u32],
    z_block: &[crate::field::F128],
    out_block: &mut [crate::field::F128],
) {
    let z = as_u128s(z_block);
    let f128_per_block = z_block.len();
    for out_idx in 0..f128_per_block {
        let mut acc: u128 = 0;
        for offset in 0..128 {
            let r = out_idx * 128 + offset;
            let lo = row_ptr[r] as usize;
            let hi = row_ptr[r + 1] as usize;
            let mut row_acc: u128 = 0;
            for &j in &cols[lo..hi] {
                row_acc ^= (z[(j >> 7) as usize] >> (j & 127)) & 1;
            }
            acc |= row_acc << offset;
        }
        if acc != 0 {
            out_block[out_idx].lo |= acc as u64;
            out_block[out_idx].hi |= (acc >> 64) as u64;
        }
    }
}

#[inline]
fn get_bit_packed(z_packed: &[crate::field::F128], global_bit: usize) -> bool {
    let i_packed = global_bit / 128;
    let local = global_bit % 128;
    if local < 64 {
        (z_packed[i_packed].lo >> local) & 1 == 1
    } else {
        (z_packed[i_packed].hi >> (local - 64)) & 1 == 1
    }
}

#[inline]
fn set_bit_packed(z_packed: &mut [crate::field::F128], global_bit: usize) {
    let i_packed = global_bit / 128;
    let local = global_bit % 128;
    if local < 64 {
        z_packed[i_packed].lo |= 1u64 << local;
    } else {
        z_packed[i_packed].hi |= 1u64 << (local - 64);
    }
}

/// `out[i] = ⊕_{j: M[i, j] = 1} z[j]` (over GF(2)).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity base matrix: `A_0 = I_k`. Each row has exactly one nonzero at
    /// the diagonal.
    fn identity(k: usize) -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: (0..k).map(|i| vec![i]).collect(),
        }
    }

    /// Packed apply_a matches bool apply_a.
    #[test]
    fn packed_matches_bool_apply() {
        use crate::pcs::pack_witness;

        // Test at several k_log values: k_log < 7, k_log = 7, k_log > 7.
        // (13,7)/(15,8) give n_outer = 64/128 — exercises the 64-wide strip
        // kernel when enough rayon workers' worth of strips exist.
        for &(m, k_log) in &[(10usize, 3), (10, 6), (10, 7), (12, 8), (13, 7), (15, 8)] {
            // Build a random sparse matrix.
            let k = 1usize << k_log;
            let mut rng = TestRng::new(0xABCD + m as u64 + k_log as u64 * 37);
            let mat = SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,
                // Each row picks a few random columns.
                rows: (0..k)
                    .map(|_| {
                        let n_nonzero = 2 + (rng.next_u64() % 4) as usize;
                        (0..n_nonzero)
                            .map(|_| (rng.next_u64() as usize) % k)
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            };

            // Random witness.
            let z: Vec<bool> = (0..(1 << m)).map(|_| rng.next_u64() & 1 == 1).collect();
            let z_packed = pack_witness(&z, m);

            // Bool reference.
            let a_bool = apply_block_diag(&mat, &z, k_log);
            let a_packed = apply_block_diag_packed(&mat, &z_packed, m, k_log);

            // Compare bit-by-bit.
            for i in 0..(1 << m) {
                let expected = a_bool[i];
                let got = get_bit_packed(&a_packed, i);
                assert_eq!(got, expected, "mismatch at i={i} (m={m}, k_log={k_log})");
            }
        }
    }

    struct TestRng(u64);
    impl TestRng {
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

    #[test]
    fn identity_matrices_accept_any_witness() {
        // A_0 = B_0 = C_0 = I_k ⇒ a = z, b = z, c = z. Boolean idempotence
        // makes the circuit-R1CS constraint trivially satisfied for any z.
        let k_log = 3;
        let m = 6;
        let r1cs = BlockR1cs {
            m,
            k_log,
            k_skip: 2,
            useful_bits: 1 << k_log,
            a_0: identity(1 << k_log),
            b_0: identity(1 << k_log),
            c_0: identity(1 << k_log),
            layout: WitnessLayout::RowMajor,
            const_pin: None,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        for seed in 0..4 {
            let z: Vec<bool> = (0..(1 << m)).map(|i| ((i ^ seed) & 1) == 1).collect();
            assert!(r1cs.satisfies(&z), "seed={seed}");
        }
    }

    #[test]
    fn zero_matrices_require_zero_witness() {
        // A_0 = B_0 = 0, C_0 = I ⇒ a·b = 0 ⇒ z = 0.
        let k_log = 3;
        let m = 6;
        let zero = SparseBinaryMatrix {
            num_rows: 1 << k_log,
            num_cols: 1 << k_log,
            rows: vec![Vec::new(); 1 << k_log],
        };
        let r1cs = BlockR1cs {
            m,
            k_log,
            k_skip: 2,
            useful_bits: 1 << k_log,
            a_0: zero.clone(),
            b_0: zero,
            c_0: identity(1 << k_log),
            layout: WitnessLayout::RowMajor,
            const_pin: None,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        let z_zero = vec![false; 1 << m];
        assert!(r1cs.satisfies(&z_zero));
        let mut z_nonzero = vec![false; 1 << m];
        z_nonzero[5] = true;
        assert!(!r1cs.satisfies(&z_nonzero));
    }
}
