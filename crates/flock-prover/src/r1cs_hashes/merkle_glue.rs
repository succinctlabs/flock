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
//! BLAKE3's base (21.03M nonzeros). Four levels therefore sweep about 105M
//! nonzeros with the FS chain.
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

use std::sync::OnceLock;

use flock_core::{
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    schedule::IoWord,
    union::SlotWitnessDest,
    zerocheck::K_SKIP,
};
use flock_field::F128;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};

use crate::r1cs_hashes::{
    common::{BM_V, BmRow, drive_witness_batch_major_partial_into, identity},
    merkle_r1cs::SLOT_WORDS,
};

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
    let half = |at: usize| {
        let mut out = 0u64;
        for (byte, chunk) in bits[at..at + 64].as_chunks::<8>().0.iter().enumerate() {
            // Eight 0/1 bytes -> one byte, lane t to bit t: the multiplier
            // routes byte t's low bit to bit 56 + t with no carries between
            // the partial products (their bit positions are all distinct).
            let x = u64::from_le_bytes(chunk.map(u8::from));
            out |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * byte);
        }
        out
    };
    F128 {
        lo: half(base),
        hi: half(base + 64),
    }
}

/// Write each live row's packed words at its BatchMajor addresses, one
/// chunk-column (`2^nu` contiguous words) per task: column `w` of row `i` is
/// word `(w << nu) + i`, so the columns are disjoint slices.
fn pack_columns(
    per_row: &[[Vec<bool>; 3]],
    words_per_block: usize,
    nu: usize,
    z: &mut [F128],
    a: &mut [F128],
    b: &mut [F128],
) {
    let col = 1usize << nu;
    z.par_chunks_mut(col)
        .zip(a.par_chunks_mut(col))
        .zip(b.par_chunks_mut(col))
        .take(words_per_block)
        .enumerate()
        .for_each(|(w, ((zc, ac), bc))| {
            for (i, [pz, pa, pb]) in per_row.iter().enumerate() {
                zc[i] = pack_word(pz, w * 128);
                ac[i] = pack_word(pa, w * 128);
                bc[i] = pack_word(pb, w * 128);
            }
        });
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
    pack_columns(per_row, words_per_block, nu, &mut z, &mut a, &mut b);

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
// The packed glue path
// ---------------------------------------------------------------------------

/// One lane of a batch-major group's staging rows (`rz[u][lane]`, …): a
/// packed glue builder writes its row's `(z, a, b)` here as block-`u64`s —
/// bit `j` of the row at bit `j % 64` of `u64` `j / 64` — instead of three
/// `Vec<bool>`s. The rows start zero, so fields are OR-ed in.
struct LaneRows<'r> {
    rz: &'r mut [BmRow],
    ra: &'r mut [BmRow],
    rb: &'r mut [BmRow],
    lane: usize,
}

impl LaneRows<'_> {
    /// Bits `[at, at + width)` of `z`, `a`, `b` (`width <= 64`; the field
    /// may straddle two `u64`s).
    #[inline]
    fn put(&mut self, at: usize, width: usize, z: u64, a: u64, b: u64) {
        debug_assert!((1..=64).contains(&width));
        let mask = u64::MAX >> (64 - width);
        let (z, a, b) = (z & mask, a & mask, b & mask);
        let (u, s, l) = (at / 64, at % 64, self.lane);
        self.rz[u][l] |= z << s;
        self.ra[u][l] |= a << s;
        self.rb[u][l] |= b << s;
        if s + width > 64 {
            self.rz[u + 1][l] |= z >> (64 - s);
            self.ra[u + 1][l] |= a >> (64 - s);
            self.rb[u + 1][l] |= b >> (64 - s);
        }
    }

    /// The whole 128-bit word `w`.
    #[inline]
    fn word(&mut self, w: usize, z: u128, a: u128, b: u128) {
        self.put(128 * w, 64, z as u64, a as u64, b as u64);
        self.put(
            128 * w + 64,
            64,
            (z >> 64) as u64,
            (a >> 64) as u64,
            (b >> 64) as u64,
        );
    }
}

/// A glue table's in-place witness through the shared partial batch-major
/// driver (non-temporal flush, 8×8 stripe transpose), its rows built packed
/// by `build`. Same words as the `Vec<bool>` path; the padding contract is
/// the driver's (the BLAKE3 and SHA-2 tables').
fn generate_packed_into<S: Sync>(
    rows: &[S],
    k_log: usize,
    useful_bits: usize,
    dst: SlotWitnessDest<'_>,
    build: impl Fn(&S, &mut LaneRows<'_>) + Sync + Send,
) -> Vec<u8> {
    let words_per_block = (1usize << k_log) / 128;
    let n_blocks_log = (dst.z.len() / words_per_block).trailing_zeros() as usize;
    let row_u64 = 2 * useful_bits.div_ceil(128);
    drive_witness_batch_major_partial_into(
        rows,
        n_blocks_log,
        k_log,
        useful_bits,
        dst,
        |group, rz, ra, rb| {
            for r in [&mut *rz, &mut *ra, &mut *rb] {
                r[..row_u64].fill([0; BM_V]);
            }
            for (lane, input) in group.iter().enumerate() {
                build(
                    input,
                    &mut LaneRows {
                        rz: &mut *rz,
                        ra: &mut *ra,
                        rb: &mut *rb,
                        lane,
                    },
                );
            }
        },
    )
}

/// Bits `0, 8, …, 56` of `v` as an 8-bit value.
#[inline]
fn byte_low_bits(v: u64) -> u64 {
    (v & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56
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
/// is what `flock_merkle` means by an even node index — the same
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

        let m = |rows: Vec<Vec<usize>>| SparseBinaryMatrix::new(k, k, rows);
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
        let per: Vec<[Vec<bool>; 3]> = rows.par_iter().map(Self::build_witness).collect();
        scatter_zab(&per, Self::k(), Self::USEFUL_BITS, nu)
    }

    /// [`Self::generate_witness_batch_major`] writing into a union slot's
    /// destination block — the copy-free union assembly path. Every column
    /// group is word-aligned, so a row is a dozen word operations
    /// ([`generate_packed_into`]), never per-bit bools: 7,294 rows went
    /// 28.5 → 0.35 ms at one thread.
    pub fn generate_witness_batch_major_into(
        rows: &[SwapInput],
        dst: SlotWitnessDest<'_>,
    ) -> Vec<u8> {
        const ONES: u64 = u64::MAX;
        generate_packed_into(rows, Self::K_LOG, Self::USEFUL_BITS, dst, |input, r| {
            let lanes = |d: &[u32; SLOT_WORDS]| -> [u64; SLOT_BITS / 64] {
                std::array::from_fn(|i| d[2 * i] as u64 | (d[2 * i + 1] as u64) << 32)
            };
            let (prev, sib) = (lanes(&input.prev), lanes(&input.sib));
            let not_bit = if input.bit_word & 1 == 1 { 0 } else { ONES };
            let bw = input.bit_word;
            r.word(Self::BIT / 128, bw, bw, u128::MAX);
            for (i, (&p, &sb)) in prev.iter().zip(&sib).enumerate() {
                let t = not_bit & (p ^ sb);
                r.put(Self::PREV + 64 * i, 64, p, p, ONES);
                r.put(Self::SIB + 64 * i, 64, sb, sb, ONES);
                r.put(Self::T + 64 * i, 64, t, not_bit, p ^ sb);
                r.put(Self::LEFT + 64 * i, 64, sb ^ t, sb ^ t, ONES);
                r.put(Self::RIGHT + 64 * i, 64, p ^ t, p ^ t, ONES);
            }
            r.put(Self::CONST, 1, 1, 1, 1);
        })
    }
}

/// One swap's inputs. `bit_word` is a whole 128-bit word because it is wired;
/// only bit 0 is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwapInput {
    pub bit_word: u128,
    pub prev: [u32; SLOT_WORDS],
    pub sib: [u32; SLOT_WORDS],
}

// ---------------------------------------------------------------------------
// The bit spread
// ---------------------------------------------------------------------------

/// Relocate the bits of one word into `depth` separate words, each carrying
/// its bit in position 0, and optionally require selected input bits to be
/// zero.
///
/// Needed only because a table's relation is uniform across rows — see the
/// module docs. It proves nothing about bit-ness: the input is already a
/// boolean table's committed word, so its bits are bits. It only moves them.
///
/// ```text
///   0                  .. 128            the input word
///   128 + 128·l        .. +128           output l  (bit 0 = index bit l, rest 0)
///   128·(depth+1)      .. +128           zero mask
///   128·(depth+2)      .. +128           check word (wired input, zero)
///   128·(depth+3)      .. +128           position mask
///   128·(depth+4)      .. +128           position prefix
///   128·(depth+5)      .. +128           masked position
///   128·(depth+6)      .. +128           position
///   128·(depth+7)                        the constant-one column
/// ```
///
/// In addition to the relocation constraints, every input bit satisfies
///
/// ```text
///   input_j · zero_mask_j = 0.
/// ```
///
/// Merkle index spreading uses the all-zero mask, so its historical relation
/// is unchanged.  The recursive verifier reuses the SAME small table to check
/// the leading-zero mask on a BLAKE3 PoW digest and to pin a nonce's unused
/// high 64 bits to zero.  Keeping this in the existing table avoids adding a
/// new boolean table (and hence another full lincheck family) solely for a
/// handful of grinding rows.
pub struct BitSpreadTable {
    pub depth: usize,
}

impl BitSpreadTable {
    pub fn new(depth: usize) -> Self {
        assert!((1..=127).contains(&depth), "depth {depth} out of range");
        Self { depth }
    }

    pub fn k_log(&self) -> usize {
        (128 * (self.depth + 7) + 1)
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
        self.position_pos() + 128
    }

    pub fn mask_pos(&self) -> usize {
        128 * (self.depth + 1)
    }

    pub fn check_pos(&self) -> usize {
        self.mask_pos() + 128
    }

    pub fn position_mask_pos(&self) -> usize {
        self.check_pos() + 128
    }

    pub fn position_prefix_pos(&self) -> usize {
        self.position_mask_pos() + 128
    }

    pub fn masked_position_pos(&self) -> usize {
        self.position_prefix_pos() + 128
    }

    pub fn position_pos(&self) -> usize {
        self.masked_position_pos() + 128
    }

    pub fn useful_bits(&self) -> usize {
        self.const_pos() + 1
    }

    /// Inputs: the word, its zero mask, a zero check word, and the mask/prefix
    /// defining `position = (word & position_mask) XOR position_prefix`.
    /// Outputs: one single-bit word per level and the derived position word.
    /// Wiring the check word as an input is
    /// essential: under the circuit's `C = I` convention the R1CS equations
    /// define it as `word & mask`; the surrounding circuit supplies zero,
    /// turning that definition into a selected-zero assertion.
    pub fn io_schema(&self) -> Vec<IoWord> {
        let mut s = vec![
            IoWord::input(0),
            IoWord::input(self.mask_pos() / 128),
            IoWord::input(self.check_pos() / 128),
            IoWord::input(self.position_mask_pos() / 128),
            IoWord::input(self.position_prefix_pos() / 128),
        ];
        s.extend((0..self.depth).map(|l| IoWord::output(self.out(l) / 128)));
        s.push(IoWord::output(self.position_pos() / 128));
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

            // The mask is an ordinary wired input, so first give its bit the
            // usual free assignment equation mask_j * 1 = mask_j.
            a[self.mask_pos() + j] = vec![self.mask_pos() + j];
            b[self.mask_pos() + j] = vec![gc];

            // The optimized boolean prover uses the C=I convention. Put the
            // selected-zero predicate in a dedicated input word:
            // input_j * mask_j = check_j. The circuit wires that whole word
            // to zero, so a malicious witness cannot set check_j to the
            // product and satisfy the relation.
            a[self.check_pos() + j] = vec![j];
            b[self.check_pos() + j] = vec![self.mask_pos() + j];

            // The position mask and prefix are ordinary wired inputs.
            for p in [self.position_mask_pos() + j, self.position_prefix_pos() + j] {
                a[p] = vec![p];
                b[p] = vec![gc];
            }
            // masked_j = word_j * position_mask_j.
            a[self.masked_position_pos() + j] = vec![j];
            b[self.masked_position_pos() + j] = vec![self.position_mask_pos() + j];
            // position_j = masked_j + prefix_j.
            a[self.position_pos() + j] = vec![
                self.masked_position_pos() + j,
                self.position_prefix_pos() + j,
            ];
            b[self.position_pos() + j] = vec![gc];
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

        let m = |rows: Vec<Vec<usize>>| SparseBinaryMatrix::new(k, k, rows);
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

    pub fn build_witness(&self, input_word: u128) -> [Vec<bool>; 3] {
        self.build_masked_witness(BitSpreadInput {
            word: input_word,
            zero_mask: 0,
            position_mask: 0,
            position_prefix: 0,
        })
    }

    /// The row witness including the optional zero-mask constraint.
    pub fn build_masked_witness(&self, input: BitSpreadInput) -> [Vec<bool>; 3] {
        let k = self.k();
        let (mut z, mut a, mut b) = (vec![false; k], vec![false; k], vec![false; k]);
        for j in 0..128 {
            let v = (input.word >> j) & 1 == 1;
            z[j] = v;
            a[j] = v;
            b[j] = true;

            let mask = (input.zero_mask >> j) & 1 == 1;
            z[self.mask_pos() + j] = mask;
            a[self.mask_pos() + j] = mask;
            b[self.mask_pos() + j] = true;

            // z/check stays zero.  A/B carry the product inputs so an
            // overlapping bit makes A*B != z exactly where zerocheck reads.
            a[self.check_pos() + j] = v;
            b[self.check_pos() + j] = mask;

            let position_mask = (input.position_mask >> j) & 1 == 1;
            z[self.position_mask_pos() + j] = position_mask;
            a[self.position_mask_pos() + j] = position_mask;
            b[self.position_mask_pos() + j] = true;

            let prefix = (input.position_prefix >> j) & 1 == 1;
            z[self.position_prefix_pos() + j] = prefix;
            a[self.position_prefix_pos() + j] = prefix;
            b[self.position_prefix_pos() + j] = true;

            let masked = v && position_mask;
            z[self.masked_position_pos() + j] = masked;
            a[self.masked_position_pos() + j] = v;
            b[self.masked_position_pos() + j] = position_mask;

            let position = masked ^ prefix;
            z[self.position_pos() + j] = position;
            a[self.position_pos() + j] = position;
            b[self.position_pos() + j] = true;
        }
        for l in 0..self.depth {
            let v = (input.word >> l) & 1 == 1;
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
        rows: &[BitSpreadInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let per: Vec<[Vec<bool>; 3]> = rows
            .par_iter()
            .map(|&i| self.build_masked_witness(i))
            .collect();
        scatter_zab(&per, self.k(), self.useful_bits(), nu)
    }

    /// [`Self::generate_witness_batch_major`] writing into a union slot's
    /// destination block — the copy-free union assembly path.
    pub fn generate_witness_batch_major_into(
        &self,
        rows: &[BitSpreadInput],
        dst: SlotWitnessDest<'_>,
    ) -> Vec<u8> {
        const ONES: u128 = u128::MAX;
        generate_packed_into(rows, self.k_log(), self.useful_bits(), dst, |i, r| {
            let w = i.word;
            r.word(0, w, w, ONES);
            for l in 0..self.depth {
                let v = (w >> l) & 1;
                r.word(self.out(l) / 128, v, v, ONES);
            }
            r.word(self.mask_pos() / 128, i.zero_mask, i.zero_mask, ONES);
            // z/check stays zero; A/B carry the product inputs.
            r.word(self.check_pos() / 128, 0, w, i.zero_mask);
            let (pm, pp) = (i.position_mask, i.position_prefix);
            r.word(self.position_mask_pos() / 128, pm, pm, ONES);
            r.word(self.position_prefix_pos() / 128, pp, pp, ONES);
            let masked = w & pm;
            r.word(self.masked_position_pos() / 128, masked, w, pm);
            r.word(self.position_pos() / 128, masked ^ pp, masked ^ pp, ONES);
            r.put(self.const_pos(), 1, 1, 1, 1);
        })
    }
}

/// One row of [`BitSpreadTable`].  `zero_mask` is statement data: every set
/// bit requires the corresponding bit of `word` to be zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitSpreadInput {
    pub word: u128,
    pub zero_mask: u128,
    /// Bits retained from `word` in the derived position output.
    pub position_mask: u128,
    /// Fixed high-bit stratum inserted into the derived position output.
    pub position_prefix: u128,
}

// ---------------------------------------------------------------------------
// The fused PoW mask row
// ---------------------------------------------------------------------------

/// One 512-bit row carrying BOTH selected-zero relations of a fused grinding
/// check: the predicate word's leading-`lambda` prefix and the nonce word's
/// high 64 bits.  This is the whole recursive cost of one PoW site beyond
/// the challenge squeeze it already shares.
///
/// ```text
///   0   .. 128    predicate word            (wired input 0)
///   128 .. 256    nonce word                (wired input 1)
///   256 .. 320    mask, low 64 bits         (wired input 2, low half)
///   320 .. 384    nonce bits 64..128        (input 2's HIGH half, repurposed)
///   384 .. 511    check_j = pred_j · mask_j (wired input, zero)
///   511           the constant-one column
/// ```
///
/// The trick that makes it fit one 512-bit row: a prefix mask for
/// `lambda <= 64` lives entirely in the low half (serialized "leading bits"
/// are the FIRST bytes), so the mask constant's high half is zero BY
/// CONSTRUCTION — and the relation writes the nonce's high bits into exactly
/// those cells (`z[320+t] = nonce[64+t]`).  The input word's wire binding
/// then forces them to equal the statement constant's zero high half: the
/// nonce-width check costs no cells of its own.  A zero-bit site passes the
/// nonce as BOTH input words with the all-ones low mask, pinning the whole
/// canonical-zero nonce (low half through the prefix cells, high half
/// through the repurposed cells) — no extra grinding knob.
///
/// `lambda <= 64` is asserted at emission; the embedded Ligerito profiles
/// currently reach 23 bits, while the non-Ligerito circuit policies remain
/// below that bound.
pub struct PowMaskTable;

/// One row of [`PowMaskTable`].  `mask` is statement data confined to the
/// low 64 bits; every set bit requires that bit of `pred` to be zero, and
/// the relation itself requires `nonce`'s high 64 bits to be zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PowMaskInput {
    pub pred: u128,
    pub nonce: u128,
    pub mask: u128,
}

// ---------------------------------------------------------------------------
// Family-H 8x8 transpose tiles
// ---------------------------------------------------------------------------

/// One dynamically placed 8x8 tile of the 128x128 bit-matrix transpose used
/// by the ring-switch verifier.  The selector's low nibble chooses one input
/// byte and its high nibble chooses the destination byte.  Eight input rows
/// become eight output columns; every output word is zero outside the selected
/// destination byte.
///
/// Keeping the selector wired is essential: without it a witness could choose
/// which tile it exposes.  The 8 inputs + selector + 8 outputs make exactly 17
/// IO words, the remaining cell-slot headroom of the m32 recursion envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyTransposeTileInput {
    pub rows: [F128; 8],
    pub selector: u8,
}

pub struct FamilyTransposeTileTable;

impl FamilyTransposeTileTable {
    pub const K_LOG: usize = 12;
    const INPUT: usize = 0;
    const SELECTOR: usize = 8 * 128;
    const NOT_SELECTOR: usize = Self::SELECTOR + 128;
    const SRC_ONE_HOT: usize = Self::NOT_SELECTOR + 8;
    const DST_ONE_HOT: usize = Self::SRC_ONE_HOT + 16 * 3;
    const SELECTED_PRODUCTS: usize = Self::DST_ONE_HOT + 16 * 3;
    const CONST: usize = Self::SELECTED_PRODUCTS + 8 * 8 * 16;
    const OUTPUT: usize = 18 * 128;
    pub const USEFUL_BITS: usize = Self::OUTPUT + 8 * 128;

    fn k() -> usize {
        1usize << Self::K_LOG
    }

    fn bit(x: F128, j: usize) -> bool {
        if j < 64 {
            (x.lo >> j) & 1 == 1
        } else {
            (x.hi >> (j - 64)) & 1 == 1
        }
    }

    fn selector_literal(bit: usize, value: bool) -> usize {
        if value {
            Self::SELECTOR + bit
        } else {
            Self::NOT_SELECTOR + bit
        }
    }

    fn hot_base(source: bool, value: usize) -> usize {
        let base = if source {
            Self::SRC_ONE_HOT
        } else {
            Self::DST_ONE_HOT
        };
        base + 3 * value
    }

    fn product(r: usize, c: usize, source_byte: usize) -> usize {
        Self::SELECTED_PRODUCTS + ((r * 8 + c) * 16 + source_byte)
    }

    pub fn io_schema() -> Vec<IoWord> {
        let mut schema: Vec<IoWord> = (0..8).map(IoWord::input).collect();
        schema.push(IoWord::input(Self::SELECTOR / 128));
        schema.extend((0..8).map(|w| IoWord::output(Self::OUTPUT / 128 + w)));
        schema
    }

    pub fn outputs(input: &FamilyTransposeTileInput) -> [F128; 8] {
        let source_byte = (input.selector & 0x0f) as usize;
        let destination_byte = (input.selector >> 4) as usize;
        let mut out = [F128::ZERO; 8];
        for c in 0..8 {
            for r in 0..8 {
                if Self::bit(input.rows[r], 8 * source_byte + c) {
                    let at = 8 * destination_byte + r;
                    if at < 64 {
                        out[c].lo |= 1u64 << at;
                    } else {
                        out[c].hi |= 1u64 << (at - 64);
                    }
                }
            }
        }
        out
    }

    pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = Self::k();
        let gc = Self::CONST;
        let mut a = vec![Vec::new(); k];
        let mut b = vec![Vec::new(); k];
        let free = |a: &mut Vec<Vec<usize>>, b: &mut Vec<Vec<usize>>, at: usize| {
            a[at] = vec![at];
            b[at] = vec![gc];
        };

        for at in Self::INPUT..Self::SELECTOR + 128 {
            free(&mut a, &mut b, at);
        }
        for q in 0..8 {
            let at = Self::NOT_SELECTOR + q;
            a[at] = vec![gc, Self::SELECTOR + q];
            b[at] = vec![gc];
        }

        for source in [true, false] {
            let selector_base = if source { 0 } else { 4 };
            for value in 0..16 {
                let base = Self::hot_base(source, value);
                let lit =
                    |q: usize| Self::selector_literal(selector_base + q, ((value >> q) & 1) == 1);
                a[base] = vec![lit(0)];
                b[base] = vec![lit(1)];
                a[base + 1] = vec![lit(2)];
                b[base + 1] = vec![lit(3)];
                a[base + 2] = vec![base];
                b[base + 2] = vec![base + 1];
            }
        }

        for r in 0..8 {
            for c in 0..8 {
                for source_byte in 0..16 {
                    let at = Self::product(r, c, source_byte);
                    a[at] = vec![Self::INPUT + 128 * r + 8 * source_byte + c];
                    b[at] = vec![Self::hot_base(true, source_byte) + 2];
                }
            }
        }

        for c in 0..8 {
            for destination_byte in 0..16 {
                for r in 0..8 {
                    let at = Self::OUTPUT + 128 * c + 8 * destination_byte + r;
                    a[at] = (0..16).map(|s| Self::product(r, c, s)).collect();
                    b[at] = vec![Self::hot_base(false, destination_byte) + 2];
                }
            }
        }
        a[gc] = vec![gc];
        b[gc] = vec![gc];

        let m = |rows: Vec<Vec<usize>>| SparseBinaryMatrix::new(k, k, rows);
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

    pub fn build_witness(input: &FamilyTransposeTileInput) -> [Vec<bool>; 3] {
        let k = Self::k();
        let (mut z, mut a, mut b) = (vec![false; k], vec![false; k], vec![false; k]);
        let free = |z: &mut [bool], a: &mut [bool], b: &mut [bool], at: usize, v: bool| {
            z[at] = v;
            a[at] = v;
            b[at] = true;
        };

        for r in 0..8 {
            for j in 0..128 {
                free(
                    &mut z,
                    &mut a,
                    &mut b,
                    Self::INPUT + 128 * r + j,
                    Self::bit(input.rows[r], j),
                );
            }
        }
        for j in 0..128 {
            free(
                &mut z,
                &mut a,
                &mut b,
                Self::SELECTOR + j,
                j < 8 && ((input.selector >> j) & 1) == 1,
            );
        }
        for q in 0..8 {
            let s = ((input.selector >> q) & 1) == 1;
            let at = Self::NOT_SELECTOR + q;
            z[at] = !s;
            a[at] = !s;
            b[at] = true;
        }

        for source in [true, false] {
            let selector_base = if source { 0 } else { 4 };
            for value in 0..16 {
                let base = Self::hot_base(source, value);
                let lit = |q: usize| {
                    ((input.selector >> (selector_base + q)) & 1) as usize == ((value >> q) & 1)
                };
                z[base] = lit(0) && lit(1);
                a[base] = lit(0);
                b[base] = lit(1);
                z[base + 1] = lit(2) && lit(3);
                a[base + 1] = lit(2);
                b[base + 1] = lit(3);
                z[base + 2] = z[base] && z[base + 1];
                a[base + 2] = z[base];
                b[base + 2] = z[base + 1];
            }
        }

        let source_byte = (input.selector & 0x0f) as usize;
        let destination_byte = (input.selector >> 4) as usize;
        for r in 0..8 {
            for c in 0..8 {
                let mut selected = false;
                for s in 0..16 {
                    let at = Self::product(r, c, s);
                    let x = Self::bit(input.rows[r], 8 * s + c);
                    let hot = s == source_byte;
                    z[at] = x && hot;
                    a[at] = x;
                    b[at] = hot;
                    selected ^= z[at];
                }
                for d in 0..16 {
                    let at = Self::OUTPUT + 128 * c + 8 * d + r;
                    z[at] = selected && d == destination_byte;
                    a[at] = selected;
                    b[at] = d == destination_byte;
                }
            }
        }
        z[Self::CONST] = true;
        a[Self::CONST] = true;
        b[Self::CONST] = true;
        [z, a, b]
    }

    pub fn generate_witness_batch_major(
        rows: &[FamilyTransposeTileInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let per: Vec<[Vec<bool>; 3]> = rows.par_iter().map(Self::build_witness).collect();
        scatter_zab(&per, Self::k(), Self::USEFUL_BITS, nu)
    }

    pub fn generate_witness_batch_major_into(
        rows: &[FamilyTransposeTileInput],
        dst: SlotWitnessDest<'_>,
    ) -> Vec<u8> {
        const ONES: u128 = u128::MAX;
        generate_packed_into(rows, Self::K_LOG, Self::USEFUL_BITS, dst, |input, r| {
            let word = |x: F128| x.lo as u128 | (x.hi as u128) << 64;
            for (w, &row) in input.rows.iter().enumerate() {
                r.word(Self::INPUT / 128 + w, word(row), word(row), ONES);
            }
            let sel = input.selector;
            r.word(Self::SELECTOR / 128, sel as u128, sel as u128, ONES);
            r.put(Self::NOT_SELECTOR, 8, !sel as u64, !sel as u64, 0xff);
            for source in [true, false] {
                let nibble = if source { sel & 0x0f } else { sel >> 4 } as usize;
                for value in 0..16 {
                    let lit = |q: usize| (nibble >> q) & 1 == (value >> q) & 1;
                    let (z0, z1) = (lit(0) && lit(1), lit(2) && lit(3));
                    let bits = |x: [bool; 3]| x[0] as u64 | (x[1] as u64) << 1 | (x[2] as u64) << 2;
                    r.put(
                        Self::hot_base(source, value),
                        3,
                        bits([z0, z1, z0 && z1]),
                        bits([lit(0), lit(2), z0]),
                        bits([lit(1), lit(3), z1]),
                    );
                }
            }
            // Products `x_{r,c,s} · [s = source]` over the 16 source bytes;
            // the selected bit lands at row `r` of every destination byte.
            let (source, destination) = ((sel & 0x0f) as usize, (sel >> 4) as usize);
            let hot = 1u64 << source;
            let mut out_a = [0u128; 8];
            for (row_i, &row) in input.rows.iter().enumerate() {
                for (c, out) in out_a.iter_mut().enumerate() {
                    let x = byte_low_bits(row.lo >> c) | byte_low_bits(row.hi >> c) << 8;
                    r.put(Self::product(row_i, c, 0), 16, x & hot, x, hot);
                    if x & hot != 0 {
                        *out |= 0x0101_0101_0101_0101_0101_0101_0101_0101u128 << row_i;
                    }
                }
            }
            let out_b = 0xffu128 << (8 * destination);
            for (c, &a) in out_a.iter().enumerate() {
                r.word(Self::OUTPUT / 128 + c, a & out_b, a, out_b);
            }
            r.put(Self::CONST, 1, 1, 1, 1);
        })
    }
}

impl PowMaskTable {
    pub fn k_log(&self) -> usize {
        9
    }

    pub fn k(&self) -> usize {
        512
    }

    pub fn const_pos(&self) -> usize {
        511
    }

    pub fn useful_bits(&self) -> usize {
        512
    }

    /// Inputs: predicate, nonce, mask, and the final check word. The caller
    /// wires that word to the constant whose low 127 bits are zero and whose
    /// last bit is the table's constant-one column. Under `C = I`, binding
    /// this input is what forces every prefix product to zero.
    pub fn io_schema(&self) -> Vec<IoWord> {
        vec![
            IoWord::input(0),
            IoWord::input(1),
            IoWord::input(2),
            IoWord::input(3),
        ]
    }

    pub fn build_matrices(&self) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
        let k = self.k();
        let gc = self.const_pos();
        let mut a: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut b: Vec<Vec<usize>> = vec![Vec::new(); k];

        // Free assignment for the two full input words and the mask's low
        // half: bit_j * 1 = bit_j.
        for j in 0..128 {
            a[j] = vec![j];
            b[j] = vec![gc];
            a[128 + j] = vec![128 + j];
            b[128 + j] = vec![gc];
        }
        for j in 0..64 {
            a[256 + j] = vec![256 + j];
            b[256 + j] = vec![gc];
            // Input 2's high half IS the nonce-width check: the cell must
            // equal nonce bit 64+j, and the wire binding pins the word to
            // the statement's mask constant, whose high half is zero.
            a[320 + j] = vec![192 + j];
            b[320 + j] = vec![gc];
        }
        // The prefix checks: check_j = pred_j * mask_j. The whole check word
        // is a circuit input wired to 0..0,1, which forces these products to
        // zero. For j >= 64 the "mask" reference lands on the repurposed
        // nonce cells (honest zero), so those checks are vacuous — masks are
        // low-half by the lambda <= 64 contract.
        for j in 0..127 {
            a[384 + j] = vec![j];
            b[384 + j] = vec![256 + j];
        }
        a[gc] = vec![gc];
        b[gc] = vec![gc];

        let m = |rows: Vec<Vec<usize>>| SparseBinaryMatrix::new(k, k, rows);
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

    pub fn build_witness(&self, input: PowMaskInput) -> [Vec<bool>; 3] {
        let k = self.k();
        let (mut z, mut a, mut b) = (vec![false; k], vec![false; k], vec![false; k]);
        for j in 0..128 {
            let p = (input.pred >> j) & 1 == 1;
            z[j] = p;
            a[j] = p;
            b[j] = true;
            let n = (input.nonce >> j) & 1 == 1;
            z[128 + j] = n;
            a[128 + j] = n;
            b[128 + j] = true;
        }
        for j in 0..64 {
            let m = (input.mask >> j) & 1 == 1;
            z[256 + j] = m;
            a[256 + j] = m;
            b[256 + j] = true;
            let nh = (input.nonce >> (64 + j)) & 1 == 1;
            z[320 + j] = nh;
            a[320 + j] = nh;
            b[320 + j] = true;
        }
        for j in 0..127 {
            // z/check stays zero.  A/B carry the product inputs so an
            // overlapping bit makes A*B != z exactly where zerocheck reads.
            a[384 + j] = (input.pred >> j) & 1 == 1;
            b[384 + j] = z[256 + j];
        }
        let gc = self.const_pos();
        z[gc] = true;
        a[gc] = true;
        b[gc] = true;
        [z, a, b]
    }

    pub fn generate_witness_batch_major(
        &self,
        rows: &[PowMaskInput],
        nu: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let per: Vec<[Vec<bool>; 3]> = rows.par_iter().map(|&i| self.build_witness(i)).collect();
        scatter_zab(&per, self.k(), self.useful_bits(), nu)
    }

    /// [`Self::generate_witness_batch_major`] writing into a union slot's
    /// destination block — the copy-free union assembly path.
    pub fn generate_witness_batch_major_into(
        &self,
        rows: &[PowMaskInput],
        dst: SlotWitnessDest<'_>,
    ) -> Vec<u8> {
        const ONES: u64 = u64::MAX;
        generate_packed_into(rows, self.k_log(), self.useful_bits(), dst, |i, r| {
            r.word(0, i.pred, i.pred, u128::MAX);
            r.word(1, i.nonce, i.nonce, u128::MAX);
            let (mask, nonce_hi) = (i.mask as u64, (i.nonce >> 64) as u64);
            r.put(256, 64, mask, mask, ONES);
            r.put(320, 64, nonce_hi, nonce_hi, ONES);
            // z/check stays zero; A/B carry the product inputs
            // `pred_j · z[256 + j]`, j < 127.
            r.put(384, 64, 0, i.pred as u64, mask);
            r.put(448, 63, 0, (i.pred >> 64) as u64, nonce_hi);
            r.put(self.const_pos(), 1, 1, 1, 1);
        })
    }
}

#[cfg(test)]
mod pack_tests {
    use super::{F128, pack_word};

    #[test]
    fn pack_word_matches_the_bit_loop() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..64 {
            let bits: Vec<bool> = (0..256)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state & 1 == 1
                })
                .collect();
            for base in [0, 64, 128] {
                let (mut lo, mut hi) = (0u64, 0u64);
                for t in 0..64 {
                    lo |= u64::from(bits[base + t]) << t;
                    hi |= u64::from(bits[base + 64 + t]) << t;
                }
                assert_eq!(pack_word(&bits, base), F128 { lo, hi });
            }
        }
    }
}

#[cfg(test)]
mod family_h_tests {
    use std::array::from_fn;

    use flock_core::pcs::ring_switch::tensor_algebra_transpose;

    use crate::r1cs_hashes::merkle_glue::{
        F128, FamilyTransposeTileInput, FamilyTransposeTileTable,
    };

    #[test]
    fn transpose_tiles_assemble_exactly_and_are_constrained() {
        let rows: [F128; 128] = from_fn(|i| {
            F128::new(
                (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                (!(i as u64)).rotate_left((i % 64) as u32),
            )
        });
        let mut assembled = [F128::ZERO; 128];
        let r1cs = FamilyTransposeTileTable::build_block_r1cs(0);
        for destination_byte in 0..16 {
            let tile_rows: [F128; 8] = rows[8 * destination_byte..8 * destination_byte + 8]
                .try_into()
                .unwrap();
            for source_byte in 0..16 {
                let input = FamilyTransposeTileInput {
                    rows: tile_rows,
                    selector: (source_byte | (destination_byte << 4)) as u8,
                };
                let out = FamilyTransposeTileTable::outputs(&input);
                for c in 0..8 {
                    assembled[8 * source_byte + c] += out[c];
                }
                let witness = FamilyTransposeTileTable::build_witness(&input);
                assert!(r1cs.satisfies(&witness[0]), "honest tile must satisfy");
                let mut bad = witness[0].clone();
                bad[FamilyTransposeTileTable::OUTPUT] ^= true;
                assert!(!r1cs.satisfies(&bad), "a changed output bit must fail");
                let mut bad_selector = witness[0].clone();
                bad_selector[FamilyTransposeTileTable::SELECTOR] ^= true;
                assert!(
                    !r1cs.satisfies(&bad_selector),
                    "the tile selector is part of the constrained input"
                );
            }
        }
        assert_eq!(assembled.to_vec(), tensor_algebra_transpose(&rows),);
    }
}

#[cfg(test)]
mod packed_glue_tests {
    use flock_core::union::SlotWitnessDest;
    use flock_field::F128;

    use crate::r1cs_hashes::merkle_glue::{
        BitSpreadInput, BitSpreadTable, FamilyTransposeTileInput, FamilyTransposeTileTable,
        PowMaskInput, PowMaskTable, SwapInput, SwapTable,
    };

    fn mix(i: u64, salt: u64) -> u128 {
        let x = (i ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (x as u128) << 64 | x.rotate_left(29) as u128
    }

    /// The packed in-place generator writes exactly the canonical
    /// `Vec<bool>` path's `z`, `a`, `b` and lincheck stripe into a zero,
    /// non-eliding destination, at full, partial and empty counts.
    fn check(
        what: &str,
        nu: usize,
        canonical: (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
        generate: impl Fn(SlotWitnessDest<'_>) -> Vec<u8>,
    ) {
        let n = canonical.0.len();
        let (mut z, mut a, mut b) = (
            vec![F128::ZERO; n],
            vec![F128::ZERO; n],
            vec![F128::ZERO; n],
        );
        let stripe = generate(SlotWitnessDest {
            z: &mut z,
            a: &mut a,
            b: &mut b,
            elide_padding_writes: false,
            dead_padding_unread: false,
        });
        assert!(z == canonical.0, "{what}: z (nu {nu})");
        assert!(a == canonical.1, "{what}: a (nu {nu})");
        assert!(b == canonical.2, "{what}: b (nu {nu})");
        assert!(stripe == canonical.3, "{what}: stripe (nu {nu})");
    }

    #[test]
    fn packed_glue_witnesses_match_the_bool_path() {
        let nu = 6;
        let spread = BitSpreadTable::new(24);
        for count in [0u64, 1, 7, 8, 9, 37, 64] {
            let swap: Vec<SwapInput> = (0..count)
                .map(|i| SwapInput {
                    bit_word: mix(i, 1),
                    prev: std::array::from_fn(|j| mix(i, 2 + j as u64) as u32),
                    sib: std::array::from_fn(|j| mix(i, 20 + j as u64) as u32),
                })
                .collect();
            check(
                "swap",
                nu,
                SwapTable::generate_witness_batch_major(&swap, nu),
                |dst| SwapTable::generate_witness_batch_major_into(&swap, dst),
            );
            let spread_rows: Vec<BitSpreadInput> = (0..count)
                .map(|i| BitSpreadInput {
                    word: mix(i, 3),
                    zero_mask: mix(i, 4) & mix(i, 5),
                    position_mask: mix(i, 6),
                    position_prefix: mix(i, 7),
                })
                .collect();
            check(
                "spread",
                nu,
                spread.generate_witness_batch_major(&spread_rows, nu),
                |dst| spread.generate_witness_batch_major_into(&spread_rows, dst),
            );
            let pow: Vec<PowMaskInput> = (0..count)
                .map(|i| PowMaskInput {
                    pred: mix(i, 8),
                    nonce: mix(i, 9),
                    mask: mix(i, 10),
                })
                .collect();
            check(
                "pow mask",
                nu,
                PowMaskTable.generate_witness_batch_major(&pow, nu),
                |dst| PowMaskTable.generate_witness_batch_major_into(&pow, dst),
            );
            let tiles: Vec<FamilyTransposeTileInput> = (0..count)
                .map(|i| FamilyTransposeTileInput {
                    rows: std::array::from_fn(|r| {
                        let x = mix(i, 11 + r as u64);
                        F128::new(x as u64, (x >> 64) as u64)
                    }),
                    selector: mix(i, 30) as u8,
                })
                .collect();
            check(
                "family tile",
                nu,
                FamilyTransposeTileTable::generate_witness_batch_major(&tiles, nu),
                |dst| FamilyTransposeTileTable::generate_witness_batch_major_into(&tiles, dst),
            );
        }
    }
}
