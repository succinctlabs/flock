//! E0 — the L1′ layout shim.
//!
//! L1′ (see ../witness-layout-plan.md §2): chunk = 128 bits, batch dims as the
//! low word-index dims:
//!
//! ```text
//!   addr_l1 = [ 7 in-word bits | n_log batch bits | k_log-7 chunk-index bits ]
//!   addr_row = [ k_log in-block bits | n_log batch bits ]        (current layout)
//! ```
//!
//! Everything here is a pure re-labeling of MLE variables: `perm_bit` maps a
//! row-major address bit to its L1′ position, and `relabel_point` moves a
//! claim point's coordinates the same way, so
//! `MLE_row(z, x) == MLE_l1(permute(z), relabel(x))` — tested in
//! `tests/e0_oracle.rs`.

use flock_core::field::F128;

/// log2 of the chunk width. Chunk = one F128 word = 2^7 bits (= `pcs::LOG_PACKING`).
pub const C_LOG: usize = 7;
/// Chunk width in bits.
pub const CHUNK_BITS: usize = 1 << C_LOG;

/// The L1′ layout descriptor for a single (k_log, n_log) table.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// log2 of the per-instance block size in bits.
    pub k_log: usize,
    /// log2 of the batch (instance) count.
    pub n_log: usize,
}

impl Layout {
    pub fn new(k_log: usize, n_log: usize) -> Self {
        assert!(k_log >= C_LOG, "chunked layout needs k_log >= {C_LOG}");
        Self { k_log, n_log }
    }

    /// Total MLE variables.
    pub fn m(&self) -> usize {
        self.k_log + self.n_log
    }

    /// Row-major (current) address of in-block bit `j` of instance `o`.
    #[inline]
    pub fn addr_row(&self, o: usize, j: usize) -> usize {
        debug_assert!(o < 1 << self.n_log && j < 1 << self.k_log);
        (o << self.k_log) | j
    }

    /// L1′ address of in-block bit `j` of instance `o`.
    #[inline]
    pub fn addr_l1(&self, o: usize, j: usize) -> usize {
        debug_assert!(o < 1 << self.n_log && j < 1 << self.k_log);
        (j & (CHUNK_BITS - 1)) | (o << C_LOG) | ((j >> C_LOG) << (C_LOG + self.n_log))
    }

    /// Where row-major address bit `i` lands in the L1′ address.
    ///
    /// - in-word bits (i < 7) stay put;
    /// - chunk-index bits (7 ≤ i < k_log) shift up past the batch bits;
    /// - batch bits (k_log ≤ i < m) drop down to sit right above the word.
    #[inline]
    pub fn perm_bit(&self, i: usize) -> usize {
        debug_assert!(i < self.m());
        if i < C_LOG {
            i
        } else if i < self.k_log {
            i + self.n_log
        } else {
            C_LOG + (i - self.k_log)
        }
    }

    /// Relabel a claim point: variable `i` of the row-major MLE becomes
    /// variable `perm_bit(i)` of the L1′ MLE.
    pub fn relabel_point(&self, x: &[F128]) -> Vec<F128> {
        assert_eq!(x.len(), self.m());
        let mut y = vec![F128::ZERO; self.m()];
        for (i, &xi) in x.iter().enumerate() {
            y[self.perm_bit(i)] = xi;
        }
        y
    }

    /// Permute a bit-level witness row-major → L1′. Reference implementation
    /// for tests only (production producers emit L1′ directly — never build
    /// row-major and permute).
    pub fn permute_bits_row_to_l1(&self, z: &[bool]) -> Vec<bool> {
        assert_eq!(z.len(), 1usize << self.m());
        let mut out = vec![false; z.len()];
        for o in 0..1usize << self.n_log {
            for j in 0..1usize << self.k_log {
                out[self.addr_l1(o, j)] = z[self.addr_row(o, j)];
            }
        }
        out
    }

    /// Permute an F128-packed witness row-major → L1′: a plain transpose of
    /// the (instance × chunk-index) word matrix. Reference/test helper.
    ///
    /// Row-major word index: `(o << (k_log-7)) | c`; L1′: `(c << n_log) | o`.
    pub fn permute_words_row_to_l1(&self, zw: &[F128]) -> Vec<F128> {
        let chunks_per_block = 1usize << (self.k_log - C_LOG);
        assert_eq!(zw.len(), chunks_per_block << self.n_log);
        let mut out = vec![F128::ZERO; zw.len()];
        for o in 0..1usize << self.n_log {
            for c in 0..chunks_per_block {
                out[(c << self.n_log) | o] = zw[(o << (self.k_log - C_LOG)) | c];
            }
        }
        out
    }

    /// Same word transpose on raw u64 buffers (one word = 2 consecutive u64s).
    /// Used by the E1 harness to cross-validate the staged producers against
    /// the row-major baseline.
    pub fn permute_words_u64_row_to_l1(&self, zw: &[u64]) -> Vec<u64> {
        let chunks_per_block = 1usize << (self.k_log - C_LOG);
        assert_eq!(zw.len(), (chunks_per_block << self.n_log) * 2);
        let mut out = vec![0u64; zw.len()];
        for o in 0..1usize << self.n_log {
            for c in 0..chunks_per_block {
                let src = ((o << (self.k_log - C_LOG)) | c) * 2;
                let dst = ((c << self.n_log) | o) * 2;
                out[dst] = zw[src];
                out[dst + 1] = zw[src + 1];
            }
        }
        out
    }
}
