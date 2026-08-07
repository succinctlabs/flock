//! **Merkle glue**: the two small boolean tables that let a Merkle opening be
//! expressed as *wiring over the shipped BLAKE3 table*, instead of as a
//! composite that embeds BLAKE3's constraints once per tree shape.
//!
//! ## Why
//!
//! The lincheck sweeps every table type's CSC once per slot, so its cost
//! scales with the NUMBER of boolean table types — not with rows, and not
//! with trace size. Each [`MerkleTreeLayout`](super::merkle_r1cs::MerkleTreeLayout)
//! shape is its own type and each one's walker stores its own copy of
//! BLAKE3's base (21.03M nonzeros), so the four levels of the m=26 Fast
//! ladder cost 4 × 21M on top of the FS chain's own 21M — 105.1M swept, ~91 ms
//! of a 174 ms prove (`circuit_merkle::mvp5_all_levels_query_phase`).
//!
//! Expressed as wiring, every compression is a row of ONE BLAKE3 table and the
//! sweep is ~21M regardless of how many tree shapes there are.
//!
//! ## What has to move out of the composite
//!
//! Two things the composite did *inside* a row, which have nowhere to live
//! once each compression is its own row:
//!
//! - **[`SwapTable`]** — the conditional swap. A Merkle step hashes
//!   `(left, right)` with the running digest on one side or the other
//!   depending on the position bit; BLAKE3's message words are free inputs, so
//!   something must compute `left‖right` from `(prev, sibling, bit)`.
//! - **[`BitSpreadTable`]** — and this one exists purely because **a table's
//!   relation is uniform across its rows**. The composite could read
//!   `index_bit(l)` — a different column per level — because all levels shared
//!   one row. Split apart, every swap row reads the *same* column, so it can
//!   only ever see bit 0 of its bit-word. Each level therefore needs its own
//!   word carrying that level's bit in position 0, and this table relocates
//!   them out of the one challenge word the transcript produced.
//!
//! Both are tiny (~3.8k and ~2.1k nonzeros against BLAKE3's 21M), which is the
//! whole point: they add table types whose sweep cost rounds to nothing.
//!
//! ## What deliberately does NOT move
//!
//! The **sibling** stays free witness and is not in [`SwapTable`]'s IO schema.
//! No other gate reads it, and the relation binds it anyway — it feeds the
//! swap, whose output feeds the compression, whose output chains to the root.
//! Same treatment the composite gives it, same reason.

use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};
use flock_core::schedule::IoWord;
use flock_core::zerocheck::K_SKIP;
use std::sync::OnceLock;

use super::common::identity;
use super::merkle_r1cs::SLOT_WORDS;

/// Bits in a digest.
const SLOT_BITS: usize = 32 * SLOT_WORDS;

/// Bit `j` of a digest.
#[inline]
fn digest_bit(d: &[u32; SLOT_WORDS], j: usize) -> bool {
    (d[j / 32] >> (j % 32)) & 1 == 1
}

/// The 128 bools at `[base, base+128)` as one `F128`.
#[inline]
fn pack_word(bits: &[bool], base: usize) -> F128 {
    let mut lo = 0u64;
    let mut hi = 0u64;
    for t in 0..64 {
        if bits[base + t] {
            lo |= 1u64 << t;
        }
        if bits[base + 64 + t] {
            hi |= 1u64 << t;
        }
    }
    F128 { lo, hi }
}

/// Scatter per-row `(z, a, b)` bool vectors into the union's BatchMajor
/// buffers plus the lincheck stripe. Same contract as
/// `MerkleTreeLayout::scatter_zab_batch_major`; duplicated rather than shared
/// because that one is a private method keyed to its own `k`.
fn scatter_zab(
    per_row: &[[Vec<bool>; 3]],
    k: usize,
    useful_bits: usize,
    nu: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    use rayon::prelude::*;
    let n_total = 1usize << nu;
    assert!(
        n_total.is_multiple_of(8),
        "the lincheck stripe needs 2^nu ≥ 8 (nu ≥ 3)"
    );
    let words_per_block = k / 128;
    let total = n_total * words_per_block;

    let mut z = vec![F128::ZERO; total];
    let mut a = vec![F128::ZERO; total];
    let mut b = vec![F128::ZERO; total];
    for (i, [pz, pa, pb]) in per_row.iter().enumerate() {
        for w in 0..words_per_block {
            let addr = (w << nu) + i;
            z[addr] = pack_word(pz, w * 128);
            a[addr] = pack_word(pa, w * 128);
            b[addr] = pack_word(pb, w * 128);
        }
    }

    let mut stripe = vec![0u8; (n_total / 8) * k];
    stripe.par_chunks_mut(k).enumerate().for_each(|(g, chunk)| {
        for r in 0..8 {
            let row = 8 * g + r;
            if row >= per_row.len() {
                continue;
            }
            for c in 0..useful_bits {
                if per_row[row][0][c] {
                    chunk[c] |= 1u8 << r;
                }
            }
        }
    });
    (z, a, b, stripe)
}

// ---------------------------------------------------------------------------
// The conditional swap
// ---------------------------------------------------------------------------

/// One Merkle level's conditional swap, as one row.
///
/// ```text
///   t_j     = (1 + b) · (prev_j ⊕ sib_j)      j ∈ 0..256
///   left_j  = sib_j  ⊕ t_j
///   right_j = prev_j ⊕ t_j
/// ```
///
/// `b = 0` puts the running digest LEFT (`left = prev`, `right = sib`), which
/// is what `flock_core::merkle` means by an even node index — the same
/// polarity `57aeb48` gave the composite, so the table's bit and the tree's
/// position are the same number and a Fiat–Shamir challenge wires straight in.
///
/// Column layout (`k_log = 11`, `k = 2048`, everything word-aligned so it can
/// be wired):
///
/// ```text
///   0    .. 128    bit-word   (the relation reads column 0; the rest ride free)
///   128  .. 384    prev
///   384  .. 640    sibling    — free witness, NOT in the IO schema
///   640  .. 896    t
///   896  .. 1152   left
///   1152 .. 1408   right
///   1408           the constant-one column
/// ```
pub struct SwapTable;

/// Cell-slot indices into [`SwapTable::io_schema`].
pub const SWAP_IO_BIT: usize = 0;
pub const SWAP_IO_PREV0: usize = 1;
pub const SWAP_IO_LEFT0: usize = 3;
pub const SWAP_IO_RIGHT0: usize = 5;

impl SwapTable {
    pub const K_LOG: usize = 11;
    pub const BIT: usize = 0;
    pub const PREV: usize = 128;
    pub const SIB: usize = Self::PREV + SLOT_BITS;
    pub const T: usize = Self::SIB + SLOT_BITS;
    pub const LEFT: usize = Self::T + SLOT_BITS;
    pub const RIGHT: usize = Self::LEFT + SLOT_BITS;
    pub const CONST: usize = Self::RIGHT + SLOT_BITS;
    pub const USEFUL_BITS: usize = Self::CONST + 1;

    pub fn k() -> usize {
        1usize << Self::K_LOG
    }

    /// Inputs: the bit-word and `prev`. Outputs: `left` and `right`. The
    /// sibling is absent on purpose — see the module docs.
    pub fn io_schema() -> Vec<IoWord> {
        let w = |bit: usize| {
            debug_assert_eq!(bit % 128, 0);
            bit / 128
        };
        vec![
            IoWord::input(w(Self::BIT)),
            IoWord::input(w(Self::PREV)),
            IoWord::input(w(Self::PREV) + 1),
            IoWord::output(w(Self::LEFT)),
            IoWord::output(w(Self::LEFT) + 1),
            IoWord::output(w(Self::RIGHT)),
            IoWord::output(w(Self::RIGHT) + 1),
        ]
    }

    pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = Self::k();
        let gc = Self::CONST;
        let mut a: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut b: Vec<Vec<usize>> = vec![Vec::new(); k];

        // Free columns: `z·1 = z`, satisfied by any bit.
        let free = |a: &mut Vec<Vec<usize>>, b: &mut Vec<Vec<usize>>, r: usize| {
            a[r] = vec![r];
            b[r] = vec![gc];
        };
        for j in 0..128 {
            free(&mut a, &mut b, Self::BIT + j);
        }
        for j in 0..SLOT_BITS {
            free(&mut a, &mut b, Self::PREV + j);
            free(&mut a, &mut b, Self::SIB + j);
        }

        for j in 0..SLOT_BITS {
            // The only AND per bit. `A` is `1 + b` — the complement, so that
            // `b` means what the TREE means by it.
            a[Self::T + j] = vec![gc, Self::BIT];
            b[Self::T + j] = vec![Self::PREV + j, Self::SIB + j];
            a[Self::LEFT + j] = vec![Self::SIB + j, Self::T + j];
            b[Self::LEFT + j] = vec![gc];
            a[Self::RIGHT + j] = vec![Self::PREV + j, Self::T + j];
            b[Self::RIGHT + j] = vec![gc];
        }
        a[gc] = vec![gc];
        b[gc] = vec![gc];

        let m = |rows: Vec<Vec<usize>>| SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows,
        };
        (m(a), m(b))
    }

    pub fn build_block_r1cs(n_log: usize) -> BlockR1cs {
        let (a_0, b_0) = Self::build_matrices();
        BlockR1cs {
            m: n_log + Self::K_LOG,
            k_log: Self::K_LOG,
            k_skip: K_SKIP,
            useful_bits: Self::USEFUL_BITS,
            a_0,
            b_0,
            c_0: identity(Self::k()),
            layout: WitnessLayout::BatchMajor,
            const_pin: Some(Self::CONST),
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
        }
    }

    /// The `(z, a, b)` row-witness for one swap.
    pub fn build_witness(input: &SwapInput) -> [Vec<bool>; 3] {
        let k = Self::k();
        let (mut z, mut a, mut b) = (vec![false; k], vec![false; k], vec![false; k]);
        let free = |z: &mut Vec<bool>, a: &mut Vec<bool>, b: &mut Vec<bool>, r, v| {
            z[r] = v;
            a[r] = v;
            b[r] = true;
        };
        for j in 0..128 {
            free(
                &mut z,
                &mut a,
                &mut b,
                Self::BIT + j,
                (input.bit_word >> j) & 1 == 1,
            );
        }
        for j in 0..SLOT_BITS {
            free(
                &mut z,
                &mut a,
                &mut b,
                Self::PREV + j,
                digest_bit(&input.prev, j),
            );
            free(
                &mut z,
                &mut a,
                &mut b,
                Self::SIB + j,
                digest_bit(&input.sib, j),
            );
        }

        let bit = input.bit_word & 1 == 1;
        for j in 0..SLOT_BITS {
            let xor = digest_bit(&input.prev, j) ^ digest_bit(&input.sib, j);
            let t = !bit && xor;
            z[Self::T + j] = t;
            a[Self::T + j] = !bit;
            b[Self::T + j] = xor;

            let l = digest_bit(&input.sib, j) ^ t;
            z[Self::LEFT + j] = l;
            a[Self::LEFT + j] = l;
            b[Self::LEFT + j] = true;

            let r = digest_bit(&input.prev, j) ^ t;
            z[Self::RIGHT + j] = r;
            a[Self::RIGHT + j] = r;
            b[Self::RIGHT + j] = true;
        }
        z[Self::CONST] = true;
        a[Self::CONST] = true;
        b[Self::CONST] = true;
        [z, a, b]
    }

    /// The pair this swap feeds to the compression, natively.
    pub fn outputs(input: &SwapInput) -> ([u32; SLOT_WORDS], [u32; SLOT_WORDS]) {
        if input.bit_word & 1 == 1 {
            (input.sib, input.prev)
        } else {
            (input.prev, input.sib)
        }
    }

    pub fn generate_witness_batch_major(
        rows: &[SwapInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;
        let per: Vec<[Vec<bool>; 3]> = rows.par_iter().map(Self::build_witness).collect();
        scatter_zab(&per, Self::k(), Self::USEFUL_BITS, nu)
    }
}

/// One swap's inputs. `bit_word` is a whole 128-bit word because it is wired;
/// only bit 0 is read.
#[derive(Clone, Debug)]
pub struct SwapInput {
    pub bit_word: u128,
    pub prev: [u32; SLOT_WORDS],
    pub sib: [u32; SLOT_WORDS],
}

// ---------------------------------------------------------------------------
// The bit spread
// ---------------------------------------------------------------------------

/// Relocate the bits of one index word into `depth` separate words, each
/// carrying its bit in position 0.
///
/// Needed only because a table's relation is uniform across rows — see the
/// module docs. It proves nothing about bit-ness: the input is already a
/// boolean table's committed word, so its bits are bits. It only moves them.
///
/// ```text
///   0                  .. 128            the index word
///   128 + 128·l        .. +128           output l  (bit 0 = index bit l, rest 0)
///   128·(depth+1)                        the constant-one column
/// ```
pub struct BitSpreadTable {
    pub depth: usize,
}

impl BitSpreadTable {
    pub fn new(depth: usize) -> Self {
        assert!(depth >= 1 && depth <= 127, "depth {depth} out of range");
        Self { depth }
    }

    pub fn k_log(&self) -> usize {
        (128 * (self.depth + 1) + 1)
            .next_power_of_two()
            .trailing_zeros() as usize
    }

    pub fn k(&self) -> usize {
        1usize << self.k_log()
    }

    pub fn out(&self, l: usize) -> usize {
        debug_assert!(l < self.depth);
        128 * (l + 1)
    }

    pub fn const_pos(&self) -> usize {
        128 * (self.depth + 1)
    }

    pub fn useful_bits(&self) -> usize {
        self.const_pos() + 1
    }

    /// Input: the index word. Outputs: one single-bit word per level.
    pub fn io_schema(&self) -> Vec<IoWord> {
        let mut s = vec![IoWord::input(0)];
        s.extend((0..self.depth).map(|l| IoWord::output(self.out(l) / 128)));
        s
    }

    pub fn build_matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = self.k();
        let gc = self.const_pos();
        let mut a: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut b: Vec<Vec<usize>> = vec![Vec::new(); k];

        for j in 0..128 {
            a[j] = vec![j];
            b[j] = vec![gc];
        }
        for l in 0..self.depth {
            // Bit 0 of output `l` IS index bit `l`.
            a[self.out(l)] = vec![l];
            b[self.out(l)] = vec![gc];
            // The rest are pinned to zero: `A = []`, `B = [const]`, so the row
            // reads `0 · 1 = z`. NOT an empty B — the row-witness `b` bit must
            // be 1 for these to match the emitted `b` (the same convention
            // `merkle_r1cs`'s override 3 uses).
            for j in 1..128 {
                b[self.out(l) + j] = vec![gc];
            }
        }
        a[gc] = vec![gc];
        b[gc] = vec![gc];

        let m = |rows: Vec<Vec<usize>>| SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows,
        };
        (m(a), m(b))
    }

    pub fn build_block_r1cs(&self, n_log: usize) -> BlockR1cs {
        let (a_0, b_0) = self.build_matrices();
        BlockR1cs {
            m: n_log + self.k_log(),
            k_log: self.k_log(),
            k_skip: K_SKIP,
            useful_bits: self.useful_bits(),
            a_0,
            b_0,
            c_0: identity(self.k()),
            layout: WitnessLayout::BatchMajor,
            const_pin: Some(self.const_pos()),
            digest_cache: OnceLock::new(),
            csc_cache: OnceLock::new(),
        }
    }

    pub fn build_witness(&self, index_word: u128) -> [Vec<bool>; 3] {
        let k = self.k();
        let (mut z, mut a, mut b) = (vec![false; k], vec![false; k], vec![false; k]);
        for j in 0..128 {
            let v = (index_word >> j) & 1 == 1;
            z[j] = v;
            a[j] = v;
            b[j] = true;
        }
        for l in 0..self.depth {
            let v = (index_word >> l) & 1 == 1;
            z[self.out(l)] = v;
            a[self.out(l)] = v;
            b[self.out(l)] = true;
            for j in 1..128 {
                b[self.out(l) + j] = true;
            }
        }
        let gc = self.const_pos();
        z[gc] = true;
        a[gc] = true;
        b[gc] = true;
        [z, a, b]
    }

    pub fn generate_witness_batch_major(
        &self,
        rows: &[u128],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;
        let per: Vec<[Vec<bool>; 3]> = rows.par_iter().map(|&i| self.build_witness(i)).collect();
        scatter_zab(&per, self.k(), self.useful_bits(), nu)
    }
}
