//! Monolithic BLAKE3 compression-function R1CS — one R1CS instance per
//! `compress(cv, m, counter, block_len, flags) → state[16]` call. Encodes
//! the 16-word state init, all 7 rounds (8 G's per round + the message
//! permutation), and the final output XORs in one big sparse system.
//!
//! ## Encoding choice — "Option F" (carry-only cascade + fused/constant adders)
//!
//! BLAKE3 has no AND-based Ch/Maj; the only nonlinear constraints are the
//! aux products of 32-bit ADDs. Option E encoded 6 two-operand ADDs per G
//! (6 × 31 = 186 rows). Option F ports the zk.golf BLAKE3-compression
//! record's two refinements (zk.golf submission `86e6227e`, Lean-verified,
//! same C = I canonical row shape as ours):
//!
//! - **Fused 3-operand adds**: each `a + b + m` pair of chained ADDs
//!   becomes one carry-save adder — 31 majority products
//!   `w_i = (a_i⊕m_i)(b_i⊕m_i)` plus 30 ripple products of the partial sum
//!   `p = a⊕b⊕m` against the shifted majority word `bw = (w⊕m) << 1`
//!   (whose bit 0 is structurally zero, so bit 0's product row vanishes).
//!   61 rows instead of 62, twice per G: **184 rows per generic G**.
//! - **Constant-operand `c + d₁` in round 1's column G's**: their `c` lane
//!   holds `IV[0..4]` — compile-time constants — so the carry into bit 1,
//!   `k₀·y₀`, is affine (no row), and for the two even IV words the carry
//!   into bit 2 is affine as well. 30 rows for G0/G1 (IV odd), 29 for
//!   G2/G3 (IV even): blocks of 183 / 182 bits.
//!
//! Total: 52·184 + 2·183 + 2·182 = **10,298 ANDs** per compression (was
//! 10,416). Everything else is Option E unchanged — we materialize **only
//! the irreducible slots**, the ADD aux products and the I/O regions:
//!
//! - **No sum-bit slots**. Each ADD's 32 sum bits expand into lin_funcs at
//!   the use site (2-op: `s[i] = X[i] ⊕ Y[i] ⊕ ⊕_{j<i} aux[j]`; fused:
//!   `s[i] = a_i⊕b_i⊕m_i ⊕ w_{i-1}⊕m_{i-1} ⊕ ⊕_{1≤l<i} v_l`).
//! - **No lin-id slots for ANY lane.** All 16 state lanes cascade: every
//!   read inlines the chain of aux-product references from prior G's that
//!   touched the lane. (Option D materialized per-G `b_new`/`d_new` lin-id
//!   slots — 3,584 bits/compression — to break half the cascades, fearing
//!   density blowup. Measured 2026-08-05: char-2 xor_dedup keeps the
//!   cascade LINEAR — full drop is 48.3M nnz vs Option D's 21.0M (2.3×),
//!   max row ~5.6k terms, while the row narrows 121 → 93 committed
//!   word-cols (−23%). The CSC fold prices nnz at ~1 ms per 21M/prove, so
//!   the area win dominates. Measured by the since-deleted
//!   `tests/b3_width_audit.rs` probe, bloat ledger §E.)
//!
//! Trade-off: the matrix template is dense (48.3M nnz), so template build
//! and any O(nnz) pass cost more — but those are per-shape/cacheable
//! (`CscCircuit` build) or deferred to the folded matrix-claim discharge.
//! Committed area is what every per-proof O(N) pass scales with, and it
//! shrinks 23%. Picks favor `prove_fast` over `prove`.
//!
//! ## Witness layout per compression block (`k_log = 14`, `k = 16,384`)
//!
//! ```text
//!   z[0     ..    256)         = cv[0..8]   (input chaining value)
//!   z[256   ..    512)         = out_lo[0..8] = state[0..8] ^ state[8..16]
//!   z[512   ..  1,024)         = m[0..16]   (16 × 32-bit words)
//!   z[1,024 ..  1,152)         = counter_lo | counter_hi | block_len | flags
//!   z[1,152 ..  1,408)         = out_hi[0..8] = state[8..16] ^ cv[0..8]
//!   z[1,408 .. 11,706)         = 56 G blocks; 184 bits each, except the
//!                                round-1 column G's (183/183/182/182)
//!   z[11,706]                  = 1                    (constant)
//!   z[11,707 .. 16,384)        = padding (forced to 0 by empty rows)
//! ```
//!
//! Per G block layout (offsets relative to `G_BASE[g]`; `c1 = g_c1_rows(g)`
//! is 31 generically, 30/30/29/29 for the round-1 column G's):
//! ```text
//!   [0      .. 31)       maj products for FADD1  = a + b + mx        (→ a_1)
//!   [31     .. 61)       rip products for FADD1
//!   [61     .. 61+c1)    aux products for ADD_C1 = c + d_1           (→ c_1)
//!   [61+c1  .. 92+c1)    maj products for FADD2  = a_1 + b_1 + my    (→ a_2)
//!   [92+c1  .. 122+c1)   rip products for FADD2
//!   [122+c1 .. 153+c1)   aux products for ADD_C2 = c_1 + d_2         (→ c_2)
//! ```
//!
//! `a_1`, `c_1`, `a_2 (a_new)`, `c_2 (c_new)`, `d_1`, `b_1`, `d_2`,
//! `b_new`, `d_new` are NEVER materialized as slots — they're lin_funcs
//! evaluated at row-build time and threaded forward in the state cascade.
//!
//! ## Constraint shape (`C = I`)
//!
//! Every z-slot is the output of one R1CS row:
//!
//! | Row kind            | A_row            | B_row           | Output       |
//! |---------------------|------------------|-----------------|--------------|
//! | Constant `z[0]`     | `[0]`            | `[0]`           | `z[0]·z[0]`  |
//! | Input slot          | `[slot]`         | `[Z_CONST]`     | `z[slot]·1`  |
//! | out_lo/out_hi slot  | lin_func         | `[Z_CONST]`     | lin_func·1   |
//! | ADD aux product     | lin_func_L       | lin_func_R      | (L)·(R)      |
//! | Padding             | `[]`             | `[]`            | `0·0`        |
//!
//! ## What this enforces
//!
//! - The 56 G-functions execute correctly: each 2-op ADD's aux witness is
//!   constrained to `(X[i] ⊕ cin[i]) · (Y[i] ⊕ cin[i])`, so the sum bits
//!   `X[i] ⊕ Y[i] ⊕ cin[i]` are the correct 32-bit sum modulo 2³²; each
//!   fused ADD's maj/rip products pin the carry-save identity
//!   `x + y + m = p + 2·maj (mod 2³²)` the same way.
//! - `out_lo[w] = state[w] ^ state[w+8]` and `out_hi[w] = state[w+8] ^ cv[w]`
//!   (BLAKE3 finalization).
//!
//! ## What this does NOT enforce
//!
//! - **Public-input pinning**: `cv`, `m`, `counter_*`, `block_len`, `flags`
//!   are "free" witness bits. PCS-level openings at fixed indices will
//!   eventually pin them to claimed public inputs.

use super::common::{
    BitRecord, add_carry_parts, fused_add3_parts, or_bit_at, or_u32_at_bit, xor_dedup,
};
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::R1csClaim;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};
use flock_core::verifier;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Block dim: one BLAKE3 compression occupies `2^K_LOG = 16,384` z slots.
pub const K_LOG: usize = 14;
/// `k = 2^K_LOG`.
pub const K: usize = 1 << K_LOG;
/// Univariate-skip dim — must match [`flock_core::zerocheck::K_SKIP`].
pub const K_SKIP: usize = 6;

/// Number of BLAKE3 rounds.
pub const N_ROUNDS: usize = 7;
/// Number of G calls per round (4 column + 4 diagonal).
pub const N_G_PER_ROUND: usize = 8;
/// Total G calls per compression.
pub const N_G: usize = N_ROUNDS * N_G_PER_ROUND;
/// Bits per BLAKE3 word.
pub const WORD_BITS: usize = 32;

/// Aux-product bits per generic 32-bit ADD (bit 0..30; bit 31 is the
/// discarded mod-2³² carry-out and isn't allocated).
pub const CARRY_BITS_PER_ADD: usize = WORD_BITS - 1; // 31
/// Ripple (layer-2) products per fused 3-operand ADD — bit 0's product is
/// structurally zero (the shifted majority word has a zero low bit), bit 31
/// is the discarded carry-out.
pub const RIPPLE_BITS_PER_FADD: usize = WORD_BITS - 2; // 30
/// Bits per fused 3-operand ADD: 31 majority + 30 ripple products.
pub const FADD_BITS: usize = CARRY_BITS_PER_ADD + RIPPLE_BITS_PER_FADD; // 61
/// Bits of a generic G block: two fused ADDs + two 2-op ADDs.
pub const G_STRIDE: usize = 2 * FADD_BITS + 2 * CARRY_BITS_PER_ADD; // 184

/// Materialized product rows of G `g`'s `c + d_1` ADD. Round 1's column G's
/// (g = 0..4) read `c = IV[g]` — a compile-time constant — so the carry into
/// bit 1 (`k_0·y_0`) is affine and bit 0 contributes no row; for the two
/// even IV words the carry into bit 2 (`k_1·y_1`, with `k_1 = 1`) is affine
/// too. Every other G pays the full 31.
pub const fn g_c1_rows(g: usize) -> usize {
    match g {
        0 | 1 => CARRY_BITS_PER_ADD - 1, // IV[0], IV[1] odd → 30
        2 | 3 => CARRY_BITS_PER_ADD - 2, // IV[2], IV[3] even, bit 1 set → 29
        _ => CARRY_BITS_PER_ADD,
    }
}

/// Bits of G `g`'s block: FADD1 (61) + ADD_C1 + FADD2 (61) + ADD_C2 (31).
pub const fn g_block_bits(g: usize) -> usize {
    2 * FADD_BITS + CARRY_BITS_PER_ADD + g_c1_rows(g)
}

/// BLAKE3 initial hash values (identical to SHA-256 IV).
pub const BLAKE3_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// BLAKE3 message permutation applied between rounds.
pub const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Lanes touched by G index `g` within a round: `[a, b, c, d]`.
/// First 4 are column G's, last 4 are diagonal G's.
pub const G_LANES: [[usize; 4]; N_G_PER_ROUND] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

/// Message-index pairs `(mx, my)` consumed by G index `g` within a round,
/// indexing into the (already-permuted) per-round message buffer.
pub const G_MSG_IDX: [[usize; 2]; N_G_PER_ROUND] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15],
];

// ---------------------------------------------------------------------------
// Layout positions (bit indices into the per-block z slice of length K)
// ---------------------------------------------------------------------------

// **I/O-aligned layout**: every wireable region sits on a 128-bit word
// boundary, so a region word is exactly one committed F128 word (the
// circuit/wiring layer's requirement; the chain shift argument's 256-bit
// slot alignment for cv/out_lo is the special case that started this).
// The first 11 words are the full I/O prefix:
//   words 0-1  cv       [0, 256)      input chaining value
//   words 2-3  out_lo   [256, 512)    output cv (state[0..8]^state[8..16])
//   words 4-7  m        [512, 1024)   the 16 message words
//   word  8    params   [1024, 1152)  t_lo | t_hi | blen | flags
//   words 9-10 out_hi   [1152, 1408)  high output half (extended output)
// The G-blocks (internal, never wired) follow unaligned, and the constant
// pin sits at the very END so it displaces no region — the same move
// SHA-256's layout made (`sha2.rs`: pin after the aligned slots). All bit
// placement goes through the `*_bit` accessors below.
pub const SLOT_BITS: usize = 256; // 2^8, one 256-bit chaining value
pub const CV_BASE: usize = 0; // input region, slot 0: [0, 256)
pub const OUT_LO_BASE: usize = SLOT_BITS; // output region, slot 1: [256, 512)
pub const M_BASE: usize = 2 * SLOT_BITS; // 512, words 4-7
pub const T_LO_BASE: usize = M_BASE + 16 * WORD_BITS; // 1024
pub const T_HI_BASE: usize = T_LO_BASE + WORD_BITS; // 1056
pub const BLEN_BASE: usize = T_HI_BASE + WORD_BITS; // 1088
pub const FLAGS_BASE: usize = BLEN_BASE + WORD_BITS; // 1120
pub const OUT_HI_BASE: usize = FLAGS_BASE + WORD_BITS; // 1152, words 9-10
pub const GS_BASE: usize = OUT_HI_BASE + 8 * WORD_BITS; // 1408

/// `G_BASE[g]` is the first bit of G `g`'s block; `G_BASE[N_G]` is the end
/// of the G region. Strides vary: 184 generically, 183/183/182/182 for the
/// four round-1 column G's (see [`g_block_bits`]).
pub const G_BASE: [usize; N_G + 1] = {
    let mut t = [0usize; N_G + 1];
    let mut acc = GS_BASE;
    let mut g = 0;
    while g < N_G {
        t[g] = acc;
        acc += g_block_bits(g);
        g += 1;
    }
    t[N_G] = acc;
    t
};

pub const Z_CONST_POS: usize = G_BASE[N_G]; // 11,706
pub const USEFUL_BITS: usize = Z_CONST_POS + 1; // 11,707

// ---------------------------------------------------------------------------
// Wiring IO schema
// ---------------------------------------------------------------------------

/// Cell-slot indices into [`io_schema`] — the schema's order IS the
/// enumeration order, so these are the `ι` a circuit wires against.
pub const IO_CV0: usize = 0;
pub const IO_M0: usize = 2;
pub const IO_PARAMS: usize = 6;
pub const IO_OUT_LO0: usize = 7;
pub const IO_OUT_HI0: usize = 9;

/// The wireable words of one compression, in 128-bit words of the row.
///
/// **Everything a caller supplies must be here.** `cv`, `m`, and the packed
/// `(counter, block_len, flags)` word are free witness — the relation does not
/// pin them — so a word left out of the schema could be chosen by the prover.
/// That is why the params word is an *input* rather than a constant: a circuit
/// wires it to a public cell, which is what pins the chunk index and the
/// `CHUNK_START`/`CHUNK_END`/`ROOT` flags per row position.
///
/// Both output halves are exposed. `out_lo` is the chaining value the next
/// block consumes; `out_hi` is only meaningful for root/XOF compressions, but
/// a schema word that goes unwired is σ-fixed and costs nothing, whereas an
/// omitted one would be unconstrained.
///
/// The 128-bit alignment this depends on landed in `f95dfbb`
/// ([`CV_BASE`] = 0, [`OUT_LO_BASE`] = 256, [`M_BASE`] = 512).
pub fn io_schema() -> Vec<flock_core::schedule::IoWord> {
    use flock_core::schedule::IoWord;
    let w = |bit_base: usize| bit_base / 128;
    vec![
        IoWord::input(w(CV_BASE)),          // cv[0..4]
        IoWord::input(w(CV_BASE) + 1),      // cv[4..8]
        IoWord::input(w(M_BASE)),           // m[0..4]
        IoWord::input(w(M_BASE) + 1),       // m[4..8]
        IoWord::input(w(M_BASE) + 2),       // m[8..12]
        IoWord::input(w(M_BASE) + 3),       // m[12..16]
        IoWord::input(w(T_LO_BASE)),        // counter_lo|counter_hi|block_len|flags
        IoWord::output(w(OUT_LO_BASE)),     // out_lo[0..4]
        IoWord::output(w(OUT_LO_BASE) + 1), // out_lo[4..8]
        IoWord::output(w(OUT_HI_BASE)),     // out_hi[0..4]
        IoWord::output(w(OUT_HI_BASE) + 1), // out_hi[4..8]
    ]
}

// Within-G bit offsets (relative to `G_BASE[g]`). The two fused ADDs carry
// a 31-bit majority group and a 30-bit ripple group each; ADD_C1's width
// varies per G (`g_c1_rows`), shifting the second half.
const OFF_MAJ1: usize = 0;
const OFF_RIP1: usize = CARRY_BITS_PER_ADD; // 31
const OFF_C1: usize = FADD_BITS; // 61
const fn off_maj2(g: usize) -> usize {
    OFF_C1 + g_c1_rows(g)
}
const fn off_rip2(g: usize) -> usize {
    off_maj2(g) + CARRY_BITS_PER_ADD
}
const fn off_c2(g: usize) -> usize {
    off_rip2(g) + RIPPLE_BITS_PER_FADD
}
// Generic-G (g ≥ 4, c1 = 31) offsets, const for the packed writer's
// BitRecord pushes.
const OFF_MAJ2G: usize = OFF_C1 + CARRY_BITS_PER_ADD; // 92
const OFF_RIP2G: usize = OFF_MAJ2G + CARRY_BITS_PER_ADD; // 123
const OFF_C2G: usize = OFF_RIP2G + RIPPLE_BITS_PER_FADD; // 153

#[inline]
fn cv_bit(w: usize, b: usize) -> usize {
    debug_assert!(w < 8 && b < WORD_BITS);
    CV_BASE + WORD_BITS * w + b
}
#[inline]
fn m_bit(i: usize, b: usize) -> usize {
    debug_assert!(i < 16 && b < WORD_BITS);
    M_BASE + WORD_BITS * i + b
}
#[inline]
fn g_bit(g: usize, off: usize) -> usize {
    debug_assert!(g < N_G && off < g_block_bits(g));
    G_BASE[g] + off
}
#[inline]
fn out_lo_bit(w: usize, b: usize) -> usize {
    debug_assert!(w < 8 && b < WORD_BITS);
    OUT_LO_BASE + WORD_BITS * w + b
}
#[inline]
fn out_hi_bit(w: usize, b: usize) -> usize {
    debug_assert!(w < 8 && b < WORD_BITS);
    OUT_HI_BASE + WORD_BITS * w + b
}

// ---------------------------------------------------------------------------
// Reference BLAKE3 compression — the witness oracle. Cross-checked against
// the `blake3` crate in tests.
// ---------------------------------------------------------------------------

/// BLAKE3 compression function. Returns the full 16-word output state
/// (post-finalization XOR). For chaining, the new CV is `out[0..8]`.
#[inline]
pub fn blake3_compress(
    cv: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    // Canonical copy lives in flock-core (the sponge-chained challenger
    // builds on the same primitive this table proves) — one implementation.
    flock_core::hash::blake3_compress(cv, block_words, counter, block_len, flags)
}

/// `per_round_msg_idx()[r][g] = (mx_idx, my_idx)` for round `r`, G index `g`
/// — i.e., `PERM^r [G_MSG_IDX[g]]`.
fn per_round_msg_idx() -> [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] {
    let mut perm = [0usize; 16];
    for i in 0..16 {
        perm[i] = i;
    }
    let mut out = [[[0usize; 2]; N_G_PER_ROUND]; N_ROUNDS];
    for r in 0..N_ROUNDS {
        for g in 0..N_G_PER_ROUND {
            out[r][g][0] = perm[G_MSG_IDX[g][0]];
            out[r][g][1] = perm[G_MSG_IDX[g][1]];
        }
        let mut next = [0usize; 16];
        for i in 0..16 {
            next[i] = perm[MSG_PERMUTATION[i]];
        }
        perm = next;
    }
    out
}

// ---------------------------------------------------------------------------
// Lin_func cascade — per-bit lists of slot indices XOR'd to evaluate one bit.
//
// In Option D, sum bits aren't materialized as slots; instead, the "value" of
// any intermediate bit is a `LinBits[i] = Vec<usize>` whose XOR equals that
// bit. The G-builder threads these lin_funcs forward through the state, so
// each lane's value at any point in the protocol is represented as a `Word`.
// ---------------------------------------------------------------------------

/// A 32-bit symbolic word. `bits[i]` is a list of slot indices whose XOR
/// equals bit `i` of the word.
#[derive(Clone)]
struct Word {
    bits: [Vec<usize>; WORD_BITS],
}

impl Word {
    fn zero() -> Self {
        Self {
            bits: std::array::from_fn(|_| Vec::new()),
        }
    }
    /// Construct from a 32-bit witness or lin-id slot whose 32 bits live at
    /// `[base + 0, base + 1, …, base + 31]`.
    fn from_slot_base(base: usize) -> Self {
        Self {
            bits: std::array::from_fn(|i| vec![base + i]),
        }
    }
    /// Construct from a 32-bit constant — bit `i` is `[Z_CONST]` if set,
    /// `[]` otherwise.
    fn from_const(val: u32) -> Self {
        Self {
            bits: std::array::from_fn(|i| {
                if (val >> i) & 1 == 1 {
                    vec![Z_CONST_POS]
                } else {
                    Vec::new()
                }
            }),
        }
    }
    /// Bitwise XOR, no dedup. Caller calls `dedup()` after a chain if it
    /// wants canonical rows.
    fn xor(&self, other: &Word) -> Word {
        let mut out = self.clone();
        for i in 0..WORD_BITS {
            out.bits[i].extend(&other.bits[i]);
        }
        out
    }
    /// `rotr(n)` — pure index permutation; doesn't touch slot lists.
    fn rotr(&self, n: usize) -> Word {
        Word {
            bits: std::array::from_fn(|i| self.bits[(i + n) % WORD_BITS].clone()),
        }
    }
    /// Sort + cancel duplicates per bit.
    fn dedup(mut self) -> Word {
        for i in 0..WORD_BITS {
            self.bits[i] = xor_dedup(std::mem::take(&mut self.bits[i]));
        }
        self
    }
    /// "Sum bit" lin_func of an ADD `x + y` whose carry_aux slots live at
    /// `[carry_base, carry_base + 31)`.
    ///
    ///   sum[i] = x[i] ⊕ y[i] ⊕ ⊕_{j<i} carry_aux[j]
    fn add_sum(x: &Word, y: &Word, carry_base: usize) -> Word {
        let mut out = Word::zero();
        for i in 0..WORD_BITS {
            let mut v = x.bits[i].clone();
            v.extend(&y.bits[i]);
            for j in 0..i {
                v.push(carry_base + j);
            }
            out.bits[i] = v;
        }
        out.dedup()
    }
    /// "Sum bit" lin_func of a fused 3-operand ADD `x + y + m` whose maj
    /// products live at `[maj_base, maj_base + 31)` and ripple products at
    /// `[rip_base, rip_base + 30)`:
    ///
    ///   sum[i] = x[i] ⊕ y[i] ⊕ m[i] ⊕ (w[i-1] ⊕ m[i-1]) ⊕ ⊕_{1≤l<i} v[l]
    ///
    /// (partial sum ⊕ shifted majority ⊕ layer-2 carry; the `i = 0` terms
    /// beyond the partial sum vanish).
    fn fused_add_sum(x: &Word, y: &Word, m: &Word, maj_base: usize, rip_base: usize) -> Word {
        let mut out = Word::zero();
        for i in 0..WORD_BITS {
            let mut v = x.bits[i].clone();
            v.extend(&y.bits[i]);
            v.extend(&m.bits[i]);
            if i >= 1 {
                v.push(maj_base + i - 1);
                v.extend(&m.bits[i - 1]);
                for l in 1..i.min(RIPPLE_BITS_PER_FADD + 1) {
                    v.push(rip_base + l - 1);
                }
            }
            out.bits[i] = v;
        }
        out.dedup()
    }
    /// "Sum bit" lin_func of a constant-operand ADD `k + y` whose product
    /// rows for bits `t..31` (t = 31 − n_rows) live at `[base, base+n_rows)`.
    /// The carry into bit `t` is the affine seed `y[t-1]` (bit `t−1` of `k`
    /// is its lowest set bit); carries below that are zero.
    fn const_add_sum(k: u32, y: &Word, base: usize, n_rows: usize) -> Word {
        let t = CARRY_BITS_PER_ADD - n_rows;
        let mut out = Word::zero();
        for i in 0..WORD_BITS {
            let mut v = y.bits[i].clone();
            if (k >> i) & 1 == 1 {
                v.push(Z_CONST_POS);
            }
            if i >= t {
                // carry(i) = seed ⊕ products for bits t..i-1
                v.extend(&y.bits[t - 1]);
                for j in t..i {
                    v.push(base + j - t);
                }
            }
            out.bits[i] = v;
        }
        out.dedup()
    }
}

// ---------------------------------------------------------------------------
// Per-ADD: write the 31 carry_aux rows and return the sum-bit `Word`.
//
//   carry_aux[i] = (X[i] ⊕ cin[i]) · (Y[i] ⊕ cin[i])   (R1CS AND row)
//   sum[i]       = X[i] ⊕ Y[i] ⊕ cin[i]                (no slot, lin_func)
//
// where cin[i] = ⊕_{j<i} carry_aux[j].
// ---------------------------------------------------------------------------

fn write_add_carry_rows(
    a_rows: &mut [Vec<usize>],
    b_rows: &mut [Vec<usize>],
    x: &Word,
    y: &Word,
    carry_base: usize,
) -> Word {
    for i in 0..CARRY_BITS_PER_ADD {
        let mut a = x.bits[i].clone();
        for j in 0..i {
            a.push(carry_base + j);
        }
        let mut b = y.bits[i].clone();
        for j in 0..i {
            b.push(carry_base + j);
        }
        a_rows[carry_base + i] = xor_dedup(a);
        b_rows[carry_base + i] = xor_dedup(b);
    }
    Word::add_sum(x, y, carry_base)
}

// ---------------------------------------------------------------------------
// Fused 3-operand ADD `x + y + m`: 31 majority rows + 30 ripple rows.
//
//   maj row i (i = 0..30):   w[i] = (x[i] ⊕ m[i]) · (y[i] ⊕ m[i])
//   rip row j (j = 1..30):   v[j] = (p[j] ⊕ g[j]) · (bw[j] ⊕ g[j])
//
// where p = x ⊕ y ⊕ m, bw[j] = w[j-1] ⊕ m[j-1] (the shifted majority), and
// g[j] = ⊕_{1≤l<j} v[l] (layer-2 carries; g[1] = 0 since bw[0] = 0). `m` is
// the single-slot message word — it rides both sides of the maj rows, which
// is why it is the shared operand and not one of the cascaded lanes.
// ---------------------------------------------------------------------------

fn write_fused_add_rows(
    a_rows: &mut [Vec<usize>],
    b_rows: &mut [Vec<usize>],
    x: &Word,
    y: &Word,
    m: &Word,
    maj_base: usize,
    rip_base: usize,
) -> Word {
    for i in 0..CARRY_BITS_PER_ADD {
        let mut a = x.bits[i].clone();
        a.extend(&m.bits[i]);
        let mut b = y.bits[i].clone();
        b.extend(&m.bits[i]);
        a_rows[maj_base + i] = xor_dedup(a);
        b_rows[maj_base + i] = xor_dedup(b);
    }
    for j in 1..=RIPPLE_BITS_PER_FADD {
        // A side: p[j] ⊕ g[j]
        let mut a = x.bits[j].clone();
        a.extend(&y.bits[j]);
        a.extend(&m.bits[j]);
        for l in 1..j {
            a.push(rip_base + l - 1);
        }
        // B side: bw[j] ⊕ g[j]
        let mut b = vec![maj_base + j - 1];
        b.extend(&m.bits[j - 1]);
        for l in 1..j {
            b.push(rip_base + l - 1);
        }
        a_rows[rip_base + j - 1] = xor_dedup(a);
        b_rows[rip_base + j - 1] = xor_dedup(b);
    }
    Word::fused_add_sum(x, y, m, maj_base, rip_base)
}

// ---------------------------------------------------------------------------
// Constant-operand ADD `k + y` (k a compile-time constant): product rows for
// bits t..30 only (t = 31 − n_rows; the lowest set bit of k is bit t−1).
//
//   row for bit i (i = t..30) at base + (i − t):
//     prod[i] = (k[i] ⊕ c[i]) · (y[i] ⊕ c[i])
//
// where c[i] = y[t-1] ⊕ ⊕_{t≤j<i} prod[j] — the affine seed plus the
// materialized prefix. Bits below t contribute no row: their carries are
// affine (0 below the seed, y[t-1] at t).
// ---------------------------------------------------------------------------

fn write_const_add_rows(
    a_rows: &mut [Vec<usize>],
    b_rows: &mut [Vec<usize>],
    k: u32,
    y: &Word,
    base: usize,
    n_rows: usize,
) -> Word {
    let t = CARRY_BITS_PER_ADD - n_rows;
    debug_assert!(t >= 1 && k.trailing_zeros() as usize == t - 1);
    for i in t..WORD_BITS - 1 {
        // carry(i) = seed ⊕ products for bits t..i-1
        let mut carry = y.bits[t - 1].clone();
        for j in t..i {
            carry.push(base + j - t);
        }
        let mut a = carry.clone();
        if (k >> i) & 1 == 1 {
            a.push(Z_CONST_POS);
        }
        let mut b = y.bits[i].clone();
        b.extend(&carry);
        a_rows[base + i - t] = xor_dedup(a);
        b_rows[base + i - t] = xor_dedup(b);
    }
    Word::const_add_sum(k, y, base, n_rows)
}

// ---------------------------------------------------------------------------
// Initial lane sources at the start of compression.
// ---------------------------------------------------------------------------

fn initial_lane_words() -> [Word; 16] {
    let mut s: [Word; 16] = std::array::from_fn(|_| Word::zero());
    for w in 0..8 {
        s[w] = Word::from_slot_base(cv_bit(w, 0));
    }
    for i in 0..4 {
        s[8 + i] = Word::from_const(BLAKE3_IV[i]);
    }
    s[12] = Word::from_slot_base(T_LO_BASE);
    s[13] = Word::from_slot_base(T_HI_BASE);
    s[14] = Word::from_slot_base(BLEN_BASE);
    s[15] = Word::from_slot_base(FLAGS_BASE);
    s
}

// ---------------------------------------------------------------------------
// Matrix builder
// ---------------------------------------------------------------------------

/// Build the per-block base matrices `(A_0, B_0)`. `C_0 = I_k` (circuit-shape
/// R1CS — every z slot is the output of its row).
pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
    let mut b_rows: Vec<Vec<usize>> = vec![Vec::new(); K];

    // Constant z[0]: z[0]·z[0] = z[0]. Trivially satisfied for any boolean.
    a_rows[Z_CONST_POS] = vec![Z_CONST_POS];
    b_rows[Z_CONST_POS] = vec![Z_CONST_POS];

    // Input rows for cv, m, counter_lo, counter_hi, block_len, flags.
    let mut input_emit = |base: usize, len: usize| {
        for j in 0..len {
            let s = base + j;
            a_rows[s] = vec![s];
            b_rows[s] = vec![Z_CONST_POS];
        }
    };
    input_emit(CV_BASE, 8 * WORD_BITS);
    input_emit(M_BASE, 16 * WORD_BITS);
    input_emit(T_LO_BASE, WORD_BITS);
    input_emit(T_HI_BASE, WORD_BITS);
    input_emit(BLEN_BASE, WORD_BITS);
    input_emit(FLAGS_BASE, WORD_BITS);

    let msg_idx = per_round_msg_idx();
    let mut state: [Word; 16] = initial_lane_words();

    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_idx, my_idx] = msg_idx[r][g_in_round];

            // Snapshot inputs before any state mutation. Cloning is cheap
            // (lane Words point at the same slot lists — we never alias).
            let a = state[la].clone();
            let b = state[lb].clone();
            let c = state[lc].clone();
            let d = state[ld].clone();
            let mx = Word::from_slot_base(m_bit(mx_idx, 0));
            let my = Word::from_slot_base(m_bit(my_idx, 0));

            // a_1 = a + b + mx   (fused 3-operand ADD)
            let a_1 = write_fused_add_rows(
                &mut a_rows,
                &mut b_rows,
                &a,
                &b,
                &mx,
                g_bit(g, OFF_MAJ1),
                g_bit(g, OFF_RIP1),
            );
            // d_1 = rotr16(d ^ a_1)
            let d_1 = d.xor(&a_1).dedup().rotr(16);
            // c_1 = c + d_1 — a constant-operand ADD for round 1's column
            // G's, whose c lane still holds IV[g].
            let c_1 = if g < 4 {
                write_const_add_rows(
                    &mut a_rows,
                    &mut b_rows,
                    BLAKE3_IV[g],
                    &d_1,
                    g_bit(g, OFF_C1),
                    g_c1_rows(g),
                )
            } else {
                write_add_carry_rows(&mut a_rows, &mut b_rows, &c, &d_1, g_bit(g, OFF_C1))
            };
            // b_1 = rotr12(b ^ c_1)
            let b_1 = b.xor(&c_1).dedup().rotr(12);
            // a_2 = a_1 + b_1 + my   (fused; = a_new — cascades)
            let a_2 = write_fused_add_rows(
                &mut a_rows,
                &mut b_rows,
                &a_1,
                &b_1,
                &my,
                g_bit(g, off_maj2(g)),
                g_bit(g, off_rip2(g)),
            );
            // d_2 = rotr8(d_1 ^ a_2)
            let d_2 = d_1.xor(&a_2).dedup().rotr(8);
            // c_2 = c_1 + d_2    (= c_new — cascades)
            let c_2 =
                write_add_carry_rows(&mut a_rows, &mut b_rows, &c_1, &d_2, g_bit(g, off_c2(g)));
            // b_new = rotr7(b_1 ^ c_2), d_new = d_2 — not materialized;
            // all four lanes cascade (Option E).
            let b_new_word = b_1.xor(&c_2).dedup().rotr(7);
            state[la] = a_2;
            state[lb] = b_new_word;
            state[lc] = c_2;
            state[ld] = d_2;
        }
    }

    // Finalization XORs.
    //   out_lo[w] = state[w] ^ state[w+8]
    //   out_hi[w] = state[w+8] ^ cv[w]
    for w in 0..8 {
        let lo = state[w].xor(&state[w + 8]).dedup();
        for i in 0..WORD_BITS {
            let s = out_lo_bit(w, i);
            a_rows[s] = lo.bits[i].clone();
            b_rows[s] = vec![Z_CONST_POS];
        }
        let cv_w = Word::from_slot_base(cv_bit(w, 0));
        let hi = state[w + 8].xor(&cv_w).dedup();
        for i in 0..WORD_BITS {
            let s = out_hi_bit(w, i);
            a_rows[s] = hi.bits[i].clone();
            b_rows[s] = vec![Z_CONST_POS];
        }
    }

    // Padding rows [USEFUL_BITS..K): A = B = []. Constraint 0·0 = z[i]
    // forces z[i] = 0 for all padding bits.

    let to_mat = |rows| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_mat(a_rows), to_mat(b_rows))
}

/// Build a [`BlockR1cs`] batching `2^n_blocks_log` independent BLAKE3
/// compressions. `n_blocks_log ≥ 3` is required (lincheck needs `n_outer ≥ 8`).
pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices();
    super::common::build_block_r1cs_with_matrices(
        n_blocks_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        a_0,
        b_0,
        // Constant-wire pin (docs/const-wire-pin.md): forces z[Z_CONST_POS] = 1
        // in every block. Requires padding blocks filled with valid compressions.
        Some(Z_CONST_POS),
    )
}

// ---------------------------------------------------------------------------
// Lincheck circuit walker — mirrors `build_matrices`. Same structure as
// `blake3::Blake3LincheckCircuit` but uses this module's I/O-aligned slot
// positions (cv_bit/m_bit/etc.).
// ---------------------------------------------------------------------------

#[inline]
fn scatter_add_carry_rows(
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
    x: &Word,
    y: &Word,
    carry_base: usize,
) -> Word {
    for i in 0..CARRY_BITS_PER_ADD {
        let row = carry_base + i;
        let e = eq_inner[row];
        let ea = alpha * e;
        for &slot in x.bits[i].iter() {
            comb[slot] += ea;
        }
        for j in 0..i {
            comb[carry_base + j] += ea;
        }
        for &slot in y.bits[i].iter() {
            comb[slot] += e;
        }
        for j in 0..i {
            comb[carry_base + j] += e;
        }
    }
    Word::add_sum(x, y, carry_base)
}

/// Scatter mirror of [`write_fused_add_rows`]. Duplicate slot adds cancel
/// in char 2, matching the matrix builder's `xor_dedup`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn scatter_fused_add_rows(
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
    x: &Word,
    y: &Word,
    m: &Word,
    maj_base: usize,
    rip_base: usize,
) -> Word {
    for i in 0..CARRY_BITS_PER_ADD {
        let e = eq_inner[maj_base + i];
        let ea = alpha * e;
        for &slot in x.bits[i].iter() {
            comb[slot] += ea;
        }
        for &slot in y.bits[i].iter() {
            comb[slot] += e;
        }
        for &slot in m.bits[i].iter() {
            comb[slot] += ea;
            comb[slot] += e;
        }
    }
    for j in 1..=RIPPLE_BITS_PER_FADD {
        let e = eq_inner[rip_base + j - 1];
        let ea = alpha * e;
        for &slot in x.bits[j].iter() {
            comb[slot] += ea;
        }
        for &slot in y.bits[j].iter() {
            comb[slot] += ea;
        }
        for &slot in m.bits[j].iter() {
            comb[slot] += ea;
        }
        comb[maj_base + j - 1] += e;
        for &slot in m.bits[j - 1].iter() {
            comb[slot] += e;
        }
        for l in 1..j {
            comb[rip_base + l - 1] += ea;
            comb[rip_base + l - 1] += e;
        }
    }
    Word::fused_add_sum(x, y, m, maj_base, rip_base)
}

/// Scatter mirror of [`write_const_add_rows`].
#[inline]
fn scatter_const_add_rows(
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
    k: u32,
    y: &Word,
    base: usize,
    n_rows: usize,
) -> Word {
    let t = CARRY_BITS_PER_ADD - n_rows;
    for i in t..WORD_BITS - 1 {
        let e = eq_inner[base + i - t];
        let ea = alpha * e;
        // carry(i) = seed ⊕ products t..i-1, on both sides
        for &slot in y.bits[t - 1].iter() {
            comb[slot] += ea;
            comb[slot] += e;
        }
        for j in t..i {
            comb[base + j - t] += ea;
            comb[base + j - t] += e;
        }
        if (k >> i) & 1 == 1 {
            comb[Z_CONST_POS] += ea;
        }
        for &slot in y.bits[i].iter() {
            comb[slot] += e;
        }
    }
    Word::const_add_sum(k, y, base, n_rows)
}

#[inline]
fn scatter_lin_id_row(
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
    row: usize,
    word_bits_i: &[usize],
) {
    let e = eq_inner[row];
    let ea = alpha * e;
    for &slot in word_bits_i.iter() {
        comb[slot] += ea;
    }
    comb[Z_CONST_POS] += e;
}

pub struct Blake3LincheckCircuit;

impl flock_core::lincheck::LincheckCircuit for Blake3LincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    // Without this override the trait default (`None`) silently drops the
    // constant-wire pin the R1CS declares, reopening the all-zero-witness
    // gap for any caller pairing this walker with a pinned setup — the
    // keccak/merkle walkers all override; this one had inherited the default.
    fn const_pin_col(&self) -> Option<usize> {
        Some(Z_CONST_POS)
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
        let mut comb = vec![F128::ZERO; K];

        // Const row.
        let e0 = eq_inner[Z_CONST_POS];
        comb[Z_CONST_POS] += alpha * e0;
        comb[Z_CONST_POS] += e0;

        // Input self-loops for cv, m, counter, blen, flags.
        let input_emit = |comb: &mut [F128], base: usize, len: usize| {
            for j in 0..len {
                let s = base + j;
                let e = eq_inner[s];
                comb[s] += alpha * e;
                comb[Z_CONST_POS] += e;
            }
        };
        input_emit(&mut comb, CV_BASE, 8 * WORD_BITS);
        input_emit(&mut comb, M_BASE, 16 * WORD_BITS);
        input_emit(&mut comb, T_LO_BASE, WORD_BITS);
        input_emit(&mut comb, T_HI_BASE, WORD_BITS);
        input_emit(&mut comb, BLEN_BASE, WORD_BITS);
        input_emit(&mut comb, FLAGS_BASE, WORD_BITS);

        let msg_idx = per_round_msg_idx();
        let mut state: [Word; 16] = initial_lane_words();

        for r in 0..N_ROUNDS {
            for g_in_round in 0..N_G_PER_ROUND {
                let g = r * N_G_PER_ROUND + g_in_round;
                let [la, lb, lc, ld] = G_LANES[g_in_round];
                let [mx_idx, my_idx] = msg_idx[r][g_in_round];

                let a = state[la].clone();
                let b = state[lb].clone();
                let c = state[lc].clone();
                let d = state[ld].clone();
                let mx = Word::from_slot_base(m_bit(mx_idx, 0));
                let my = Word::from_slot_base(m_bit(my_idx, 0));

                let a_1 = scatter_fused_add_rows(
                    &mut comb,
                    alpha,
                    eq_inner,
                    &a,
                    &b,
                    &mx,
                    g_bit(g, OFF_MAJ1),
                    g_bit(g, OFF_RIP1),
                );
                let d_1 = d.xor(&a_1).dedup().rotr(16);
                let c_1 = if g < 4 {
                    scatter_const_add_rows(
                        &mut comb,
                        alpha,
                        eq_inner,
                        BLAKE3_IV[g],
                        &d_1,
                        g_bit(g, OFF_C1),
                        g_c1_rows(g),
                    )
                } else {
                    scatter_add_carry_rows(&mut comb, alpha, eq_inner, &c, &d_1, g_bit(g, OFF_C1))
                };
                let b_1 = b.xor(&c_1).dedup().rotr(12);
                let a_2 = scatter_fused_add_rows(
                    &mut comb,
                    alpha,
                    eq_inner,
                    &a_1,
                    &b_1,
                    &my,
                    g_bit(g, off_maj2(g)),
                    g_bit(g, off_rip2(g)),
                );
                let d_2 = d_1.xor(&a_2).dedup().rotr(8);
                let c_2 = scatter_add_carry_rows(
                    &mut comb,
                    alpha,
                    eq_inner,
                    &c_1,
                    &d_2,
                    g_bit(g, off_c2(g)),
                );

                let b_new_word = b_1.xor(&c_2).dedup().rotr(7);
                state[la] = a_2;
                state[lb] = b_new_word;
                state[lc] = c_2;
                state[ld] = d_2;
            }
        }

        for w in 0..8 {
            let lo = state[w].xor(&state[w + 8]).dedup();
            for i in 0..WORD_BITS {
                let s = out_lo_bit(w, i);
                scatter_lin_id_row(&mut comb, alpha, eq_inner, s, &lo.bits[i]);
            }
            let cv_w = Word::from_slot_base(cv_bit(w, 0));
            let hi = state[w + 8].xor(&cv_w).dedup();
            for i in 0..WORD_BITS {
                let s = out_hi_bit(w, i);
                scatter_lin_id_row(&mut comb, alpha, eq_inner, s, &hi.bits[i]);
            }
        }

        comb
    }
}

// ---------------------------------------------------------------------------
// Witness generation (boolean)
// ---------------------------------------------------------------------------

/// Compute one 32-bit ADD, writing 31 carry_aux bits into `z` at `carry_base`.
/// Returns `x.wrapping_add(y)` (sum bits are NOT materialized in this
/// encoding — see module docs).
fn add_with_witness_carry_only(x: u32, y: u32, z: &mut [bool], carry_base: usize) -> u32 {
    let mut cin: u32 = 0;
    for i in 0..WORD_BITS {
        if i < CARRY_BITS_PER_ADD {
            let xi = (x >> i) & 1;
            let yi = (y >> i) & 1;
            let ci = (cin >> i) & 1;
            let carry_aux = (xi ^ ci) & (yi ^ ci);
            z[carry_base + i] = carry_aux == 1;
            let real_carry = carry_aux ^ ci;
            cin |= real_carry << (i + 1);
        }
    }
    x.wrapping_add(y)
}

/// Fused 3-operand ADD `x + y + m`, writing 31 maj products at `maj_base`
/// and 30 ripple products at `rip_base`. Returns the mod-2³² sum.
fn fused_add_with_witness(
    x: u32,
    y: u32,
    m: u32,
    z: &mut [bool],
    maj_base: usize,
    rip_base: usize,
) -> u32 {
    let (sum, maj, rip) = fused_add3_parts(x, y, m);
    for i in 0..CARRY_BITS_PER_ADD {
        z[maj_base + i] = (maj[2] >> i) & 1 == 1;
    }
    for j in 0..RIPPLE_BITS_PER_FADD {
        z[rip_base + j] = (rip[2] >> j) & 1 == 1;
    }
    sum
}

/// Constant-operand ADD `k + y`, writing only the products for bits
/// `t..30` (t = 31 − n_rows). Returns the mod-2³² sum.
fn const_add_with_witness(k: u32, y: u32, z: &mut [bool], base: usize, n_rows: usize) -> u32 {
    let (sum, _left, _right, carry) = add_carry_parts(k, y);
    let t = CARRY_BITS_PER_ADD - n_rows;
    for i in 0..n_rows {
        z[base + i] = (carry >> (t + i)) & 1 == 1;
    }
    sum
}

#[inline]
fn write_word(z: &mut [bool], base: usize, val: u32) {
    for i in 0..WORD_BITS {
        z[base + i] = ((val >> i) & 1) == 1;
    }
}

/// Build the witness block for ONE compression. Length = `K`.
pub fn build_block_witness(
    cv: &[u32; 8],
    m: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> Vec<bool> {
    let mut z = vec![false; K];
    z[Z_CONST_POS] = true;
    // Inputs.
    for w in 0..8 {
        write_word(&mut z, cv_bit(w, 0), cv[w]);
    }
    for i in 0..16 {
        write_word(&mut z, m_bit(i, 0), m[i]);
    }
    let counter_lo = counter as u32;
    let counter_hi = (counter >> 32) as u32;
    write_word(&mut z, T_LO_BASE, counter_lo);
    write_word(&mut z, T_HI_BASE, counter_hi);
    write_word(&mut z, BLEN_BASE, block_len);
    write_word(&mut z, FLAGS_BASE, flags);

    // Internal state evolution (matches the matrix builder's symbolic
    // cascade by construction).
    let mut state: [u32; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];
    let msg_idx = per_round_msg_idx();

    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_i, my_i] = msg_idx[r][g_in_round];
            let mx = m[mx_i];
            let my = m[my_i];

            let a = state[la];
            let b = state[lb];
            let c = state[lc];
            let d = state[ld];

            let a_1 =
                fused_add_with_witness(a, b, mx, &mut z, g_bit(g, OFF_MAJ1), g_bit(g, OFF_RIP1));
            let d_1 = (d ^ a_1).rotate_right(16);
            let c_1 = if g < 4 {
                debug_assert_eq!(c, BLAKE3_IV[g]);
                const_add_with_witness(BLAKE3_IV[g], d_1, &mut z, g_bit(g, OFF_C1), g_c1_rows(g))
            } else {
                add_with_witness_carry_only(c, d_1, &mut z, g_bit(g, OFF_C1))
            };
            let b_1 = (b ^ c_1).rotate_right(12);
            let a_2 = fused_add_with_witness(
                a_1,
                b_1,
                my,
                &mut z,
                g_bit(g, off_maj2(g)),
                g_bit(g, off_rip2(g)),
            );
            let d_2 = (d_1 ^ a_2).rotate_right(8);
            let c_2 = add_with_witness_carry_only(c_1, d_2, &mut z, g_bit(g, off_c2(g)));
            let b_new = (b_1 ^ c_2).rotate_right(7);
            let d_new = d_2;

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    for w in 0..8 {
        let lo = state[w] ^ state[w + 8];
        let hi = state[w + 8] ^ cv[w];
        write_word(&mut z, out_lo_bit(w, 0), lo);
        write_word(&mut z, out_hi_bit(w, 0), hi);
    }
    z
}

/// Minimum `n_blocks_log` needed to prove `n_blocks` BLAKE3 compressions,
/// subject to the lincheck floor of `n_blocks_log ≥ 3` (`n_outer ≥ 8`).
pub fn min_n_blocks_log(n_blocks: usize) -> usize {
    assert!(n_blocks >= 1, "n_blocks must be ≥ 1");
    let n = n_blocks.max(8);
    n.next_power_of_two().trailing_zeros() as usize
}

/// One BLAKE3 compression input: `(cv, m, counter, block_len, flags)`.
pub type Compression = ([u32; 8], [u32; 16], u64, u32, u32);

/// Generate the boolean witness vector for `blocks.len()` independent BLAKE3
/// compressions, padded to `2^n_blocks_log` slots. Padding blocks are
/// all-zero (trivially satisfy the R1CS). Parallel across instances via rayon.
pub fn generate_witness(blocks: &[Compression], n_blocks_log: usize) -> Vec<bool> {
    use rayon::prelude::*;
    let n_total = 1usize << n_blocks_log;
    let n_blocks = blocks.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} compressions > 2^{n_blocks_log} = {n_total} slots"
    );
    let mut z = vec![false; n_total * K];
    z.par_chunks_mut(K)
        .take(n_blocks)
        .zip(blocks.par_iter())
        .for_each(|(chunk, (cv, m, t, b, d))| {
            let block = build_block_witness(cv, m, *t, *b, *d);
            chunk.copy_from_slice(&block);
        });
    z
}

// ---------------------------------------------------------------------------
// Fast witness generation with (a, b, c) — emits the R1CS row-witnesses
// directly from the BLAKE3 computation, in F_{2^128}-packed form. Skips the
// `apply_block_diag_packed` pass downstream.
//
// Row-witness semantics (matching `build_matrices`):
// - Constant z[0]:       (z, a, b, c) = (1, 1, 1, 1).
// - Input slot:          (z, a, b, c) = (val, val, 1, val).
// - Lin-id slot:         (z, a, b, c) = (lin_val, lin_val, 1, lin_val).
// - Carry_aux row i:     (z, a, b, c) = (carry_aux, X⊕cin, Y⊕cin, carry_aux).
// - Padding row:         all zero (already zero on entry).
// ---------------------------------------------------------------------------

// Per-row (z, a, b) word derivation: `add_carry_parts` / `fused_add3_parts`
// (common.rs) return each ADD's per-bit row values — `z = A·B` products,
// `a = A`-side, `b = B`-side — pre-masked to their group widths. **c is not
// written**: since `C = I` in this R1CS, `c == z` byte-for-byte, so callers
// use `z_packed` directly as the c-side input to zerocheck.
//
// Record-relative positions are the within-G offsets (`OFF_*`): the whole
// G block (≤ 184 bits) is one `BitRecord<3>`. Generic G's use the const
// offsets so shifts fold at compile time; round 1's four column G's have a
// narrower ADD_C1 group and take the runtime `push_at` path.

/// Write a 32-bit lin-id (or input) slot: (z, a) = val, b = all-ones.
/// **c is not written** — same `c == z` aliasing trick as above.
#[inline]
fn write_lin_word_ab_packed(bit_off: usize, val: u32, z: &mut [u64], a: &mut [u64], b: &mut [u64]) {
    or_u32_at_bit(z, bit_off, val);
    or_u32_at_bit(a, bit_off, val);
    or_u32_at_bit(b, bit_off, 0xFFFF_FFFF);
}

/// Build the (z, a, b) blocks for ONE compression instance, into u64 views
/// of the F128-packed per-block storage. Buffers must be zero on entry.
///
/// **No c buffer.** Since `C = I` (this is the circuit-shape R1CS), `c == z`
/// byte-for-byte; callers use `z_packed` directly as the c-side input to
/// zerocheck.
// `pub(crate)` so the composite Merkle encoder (`super::merkle_r1cs`) can
// embed one compression's row-witness at a column offset instead of
// re-deriving it — the row kinds above are exactly what its overridden rows
// demand (see that module's `HashSpec::node_witness_ab`).
pub(crate) fn build_block_witness_ab_packed_into(
    cv: &[u32; 8],
    m: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    const U64_PER_BLOCK: usize = K / 64;
    debug_assert_eq!(z.len(), U64_PER_BLOCK);
    debug_assert_eq!(a.len(), U64_PER_BLOCK);
    debug_assert_eq!(b.len(), U64_PER_BLOCK);

    // Constant z[0] = 1; a/b also 1 (z[0]·z[0] = z[0]).
    or_bit_at(z, Z_CONST_POS);
    or_bit_at(a, Z_CONST_POS);
    or_bit_at(b, Z_CONST_POS);

    // Input rows.
    let counter_lo = counter as u32;
    let counter_hi = (counter >> 32) as u32;
    for w in 0..8 {
        write_lin_word_ab_packed(cv_bit(w, 0), cv[w], z, a, b);
    }
    for i in 0..16 {
        write_lin_word_ab_packed(m_bit(i, 0), m[i], z, a, b);
    }
    write_lin_word_ab_packed(T_LO_BASE, counter_lo, z, a, b);
    write_lin_word_ab_packed(T_HI_BASE, counter_hi, z, a, b);
    write_lin_word_ab_packed(BLEN_BASE, block_len, z, a, b);
    write_lin_word_ab_packed(FLAGS_BASE, flags, z, a, b);

    // BLAKE3 state evolution.
    let mut state: [u32; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];
    let msg_idx = per_round_msg_idx();
    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_i, my_i] = msg_idx[r][g_in_round];
            let mx = m[mx_i];
            let my = m[my_i];

            let a_val = state[la];
            let b_val = state[lb];
            let c_val = state[lc];
            let d_val = state[ld];

            let mut rz = BitRecord::<3>::new();
            let mut ra = BitRecord::<3>::new();
            let mut rb = BitRecord::<3>::new();

            // Const-position triple push (z, a, b) — generic G's.
            macro_rules! push3c {
                ($pos:ident, $zv:expr, $av:expr, $bv:expr) => {{
                    rz.push::<$pos>($zv);
                    ra.push::<$pos>($av);
                    rb.push::<$pos>($bv);
                }};
            }
            // Runtime-position triple push — round 1's column G's.
            macro_rules! push3 {
                ($pos:expr, $zv:expr, $av:expr, $bv:expr) => {{
                    let pos = $pos;
                    rz.push_at(pos, $zv);
                    ra.push_at(pos, $av);
                    rb.push_at(pos, $bv);
                }};
            }

            let (a_1, maj1, rip1) = fused_add3_parts(a_val, b_val, mx);
            push3c!(OFF_MAJ1, maj1[2], maj1[0], maj1[1]);
            push3c!(OFF_RIP1, rip1[2], rip1[0], rip1[1]);
            let d_1 = (d_val ^ a_1).rotate_right(16);

            let c_1 = if g < 4 {
                let (sum, left, right, carry) = add_carry_parts(BLAKE3_IV[g], d_1);
                let n = g_c1_rows(g);
                let t = CARRY_BITS_PER_ADD - n;
                let mask = (1u32 << n) - 1;
                push3!(
                    OFF_C1,
                    (carry >> t) & mask,
                    (left >> t) & mask,
                    (right >> t) & mask
                );
                sum
            } else {
                let (sum, left, right, carry) = add_carry_parts(c_val, d_1);
                push3c!(OFF_C1, carry, left, right);
                sum
            };
            let b_1 = (b_val ^ c_1).rotate_right(12);

            let (a_2, maj2, rip2) = fused_add3_parts(a_1, b_1, my);
            let d_2 = (d_1 ^ a_2).rotate_right(8);
            let (c_2, left2, right2, carry2) = add_carry_parts(c_1, d_2);
            if g < 4 {
                push3!(off_maj2(g), maj2[2], maj2[0], maj2[1]);
                push3!(off_rip2(g), rip2[2], rip2[0], rip2[1]);
                push3!(off_c2(g), carry2, left2, right2);
            } else {
                push3c!(OFF_MAJ2G, maj2[2], maj2[0], maj2[1]);
                push3c!(OFF_RIP2G, rip2[2], rip2[0], rip2[1]);
                push3c!(OFF_C2G, carry2, left2, right2);
            }
            let b_new = (b_1 ^ c_2).rotate_right(7);
            let d_new = d_2;

            let g_base = G_BASE[g];
            rz.flush(z, g_base);
            ra.flush(a, g_base);
            rb.flush(b, g_base);

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    // Finalization XOR rows.
    for w in 0..8 {
        let lo = state[w] ^ state[w + 8];
        let hi = state[w + 8] ^ cv[w];
        write_lin_word_ab_packed(out_lo_bit(w, 0), lo, z, a, b);
        write_lin_word_ab_packed(out_hi_bit(w, 0), hi, z, a, b);
    }
}

/// **The fast path.** Produces `(z, a, b)` directly as F_{2^128}-packed
/// vectors — no bool intermediates, no `pack_witness` step, no
/// `apply_block_diag_packed`. Parallel across compression instances via rayon.
///
/// **No c buffer** — since `C = I` (circuit-shape R1CS), `c == z`
/// byte-for-byte; callers wrap `z_packed` as the c-side input to zerocheck.
pub fn generate_witness_with_ab_packed(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
) {
    use flock_core::field::F128;
    use rayon::prelude::*;
    let n_total = 1usize << n_blocks_log;
    let n_blocks = blocks.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} compressions > 2^{n_blocks_log} = {n_total} slots"
    );

    const F128_PER_BLOCK: usize = K / 128;
    let total_f128 = n_total * F128_PER_BLOCK;
    let mut z = vec![F128::ZERO; total_f128];
    let mut a = vec![F128::ZERO; total_f128];
    let mut b = vec![F128::ZERO; total_f128];

    // Constant-wire pin (docs/const-wire-pin.md): padding slots get a valid
    // compression of the all-zero input (constant = 1), matching
    // [`generate_witness_with_ab_packed_and_lincheck`].
    let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

    z.par_chunks_mut(F128_PER_BLOCK)
        .zip(a.par_chunks_mut(F128_PER_BLOCK))
        .zip(b.par_chunks_mut(F128_PER_BLOCK))
        .enumerate()
        .for_each(|(idx, ((z_c, a_c), b_c))| {
            let (cv, m, t, bl, fl) = if idx < n_blocks {
                &blocks[idx]
            } else {
                &padding
            };
            // SAFETY: F128 is repr(C, align(16)) with LE u64 halves — same
            // byte layout as a u64 pair.
            let z_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(z_c.as_mut_ptr() as *mut u64, z_c.len() * 2)
            };
            let a_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(a_c.as_mut_ptr() as *mut u64, a_c.len() * 2)
            };
            let b_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(b_c.as_mut_ptr() as *mut u64, b_c.len() * 2)
            };
            build_block_witness_ab_packed_into(cv, m, *t, *bl, *fl, z_u64, a_u64, b_u64);
        });

    (z, a, b)
}

/// Like [`generate_witness_with_ab_packed`] but also emits the lincheck
/// byte-stripe layout in the same parallel pass. Replaces the separate
/// `pack_z_lincheck_from_packed` call entirely.
///
/// Returns `(z, a, b, z_lincheck)`; **no c buffer** (c == z byte-for-byte).
///
/// `z_lincheck` has length `n_total · K / 8`, indexed as
/// `z_lincheck[byte_idx · K + i_inner]`, with bit `r` of that byte equal to
/// `z[i_inner, 8·byte_idx + r]`.
///
/// Parallelism granularity: 8 compressions per task; each task writes its 8
/// commit chunks then bit-transposes the just-written z u64s into its
/// lincheck stripe while they are still hot in L1.
pub fn generate_witness_with_ab_packed_and_lincheck(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    // Constant-wire pin (docs/const-wire-pin.md): fill padding blocks with a
    // valid compression (of the all-zero input) so the constant cell is 1 in
    // every block. (The chain forbids padding, so this only affects the
    // standalone batch setup.)
    let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
    super::common::drive_witness_packed_and_lincheck(
        blocks,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        |block: &Compression, z_u64, a_u64, b_u64| {
            let (cv, m, t, bl, fl) = block;
            build_block_witness_ab_packed_into(cv, m, *t, *bl, *fl, z_u64, a_u64, b_u64);
        },
    )
}

// ---------------------------------------------------------------------------
// Convenience API: Blake3Setup
// ---------------------------------------------------------------------------

/// Bundles the monolithic BLAKE3 compression R1CS, its single-slot union
/// registry, and dense/integer-lane PCS params for `n_blocks` compressions.
/// Proving goes through the UNION commit — dense stack + integer lanes;
/// the padded-commit prove path was retired 2026-08-14 with the rest of
/// the legacy standalone machinery.
#[derive(Debug)]
pub struct Blake3Setup {
    pub n_blocks: usize,
    pub r1cs: BlockR1cs,
    pub registry: crate::schedule::Registry,
    pub pcs_params: PcsParams,
}

impl Blake3Setup {
    /// Fast-path witness generation dispatched on the r1cs's witness layout.
    /// Its only caller is [`Self::prove_fast_ag`], so it carries the same
    /// aarch64 gate (the x86_64 lint leg runs with `-D warnings`).
    #[cfg(target_arch = "aarch64")]
    fn generate_witness_ab(
        &self,
        blocks: &[Compression],
    ) -> (
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<u8>,
    ) {
        match self.r1cs.layout {
            flock_core::r1cs::WitnessLayout::RowMajor => {
                generate_witness_with_ab_packed_and_lincheck(blocks, self.n_blocks_log())
            }
            flock_core::r1cs::WitnessLayout::BatchMajor => {
                generate_witness_batch_major(blocks, self.n_blocks_log())
            }
        }
    }

    pub fn new(n_blocks: usize) -> Self {
        Self::with_log_inv_rate(n_blocks, 1)
    }

    /// Build a setup with a custom PCS `log_inv_rate`.
    pub fn with_log_inv_rate(n_blocks: usize, log_inv_rate: usize) -> Self {
        // Rate keys the legacy profiles: 1 -> Fast, 2 -> Slim.
        let profile = match log_inv_rate {
            1 => flock_core::pcs::ligerito::LigeritoProfile::Fast,
            2 => flock_core::pcs::ligerito::LigeritoProfile::Slim,
            _ => flock_core::pcs::ligerito::LigeritoProfile::Fast, // other rates default to Fast
        };
        Self::with_profile_and_rate(n_blocks, profile, log_inv_rate)
    }

    /// Build a setup for a named Ligerito profile (fast/slim/secure);
    /// the PCS rate follows the profile.
    pub fn with_profile(
        n_blocks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        Self::with_profile_and_rate(n_blocks, profile, profile.log_inv_rate())
    }

    fn with_profile_and_rate(
        n_blocks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
        log_inv_rate: usize,
    ) -> Self {
        assert!(n_blocks >= 1, "n_blocks must be ≥ 1");
        let n_log = min_n_blocks_log(n_blocks);
        let mut r1cs = build_block_r1cs(n_log);
        r1cs.layout = flock_core::r1cs::WitnessLayout::BatchMajor;
        // Warm the CSC fold circuit here so its one-time build stays out of
        // the first prove/verify, and pre-fault the prove-cycle scratch
        // buffers (see scratch::prewarm_prover).
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let registry = crate::schedule::Registry::new(
            vec![crate::schedule::TableType::from_block_r1cs(&r1cs)],
            n_log,
        );
        // Warm the registry digest too: `bind_statement` absorbs it before
        // any challenge, and materializing it BLAKE3-hashes every type's
        // sparse A/B/C matrices (~21M nonzeros here), which measured ~0.8 s
        // inside the FIRST prove's statement binding. It is cached in a
        // `OnceLock` and is a pure function of the registry, so warming it
        // here only moves when the cache is filled — no transcript effect.
        let _ = registry.digest();
        // Dense/integer-lane commit params: the union commits the compacted
        // stack (used chunk-columns × declared count) at its dense_m, with
        // only the active lanes encoded and hashed.
        let pcs_params = {
            let union = flock_core::union::UnionInstance::new(&registry, vec![n_blocks]);
            let m = union.dense_m();
            let batch = flock_core::pcs::ligerito::embedded_initial_k_or_default(m, profile);
            PcsParams {
                m,
                log_inv_rate,
                log_batch_size: batch,
                profile,
                num_lanes: union.commit_lanes(batch),
                merkle_hash: Default::default(),
            }
        };
        Self {
            n_blocks,
            r1cs,
            registry,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn n_block_slots(&self) -> usize {
        1usize << self.n_blocks_log()
    }

    /// Prove `n_blocks` compressions over the single-slot UNION commit
    /// (dense stack + integer lanes; `PCS_TRACE=1` prints the per-phase
    /// breakdown). Counts below capacity leave zero dummy rows — no
    /// padding compressions needed.
    pub fn prove_fast<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofMergedLigerito,
        Commitment,
        R1csClaim,
    ) {
        assert_eq!(blocks.len(), self.n_blocks);
        let union = flock_core::union::UnionInstance::new(&self.registry, vec![self.n_blocks]);
        let slot = crate::prover::UnionSlotProverInput::new(
            generate_witness_batch_major_partial(blocks, self.n_blocks_log()),
            self.r1cs.csc_lincheck_circuit(),
        );
        crate::prover::prove_fast_ligerito_union(&union, &self.pcs_params, vec![slot], challenger)
    }

    pub fn verify<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &flock_core::proof::R1csProofMergedLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, verifier::VerifyError> {
        let union = flock_core::union::UnionInstance::new(&self.registry, vec![self.n_blocks]);
        let circuit = self.r1cs.csc_lincheck_circuit();
        let circs: [&dyn flock_core::lincheck::LincheckCircuit; 1] = [circuit];
        verifier::verify_ligerito_union(
            &union,
            &circs,
            commitment,
            proof,
            &self.pcs_params,
            challenger,
        )
    }

    /// [`Self::prove_fast`] with the **AG-skip** boolean zerocheck — the SAME
    /// single-slot union commit, lincheck, and merged opening; only round 1
    /// of the zerocheck differs. aarch64-only (NEON round-1 kernel). Verify
    /// with [`Self::verify_union_ag`].
    #[cfg(target_arch = "aarch64")]
    pub fn prove_fast_union_ag<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofMergedLigeritoAg,
        Commitment,
        R1csClaim,
    ) {
        assert_eq!(blocks.len(), self.n_blocks);
        let union = flock_core::union::UnionInstance::new(&self.registry, vec![self.n_blocks]);
        let slot = crate::prover::UnionSlotProverInput::new(
            generate_witness_batch_major_partial(blocks, self.n_blocks_log()),
            self.r1cs.csc_lincheck_circuit(),
        );
        crate::prover::prove_fast_ligerito_union_ag(
            &union,
            &self.pcs_params,
            vec![slot],
            challenger,
        )
    }

    /// Verify a [`Self::prove_fast_union_ag`] proof. (Unlike the prove side,
    /// this runs on every target.)
    pub fn verify_union_ag<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &flock_core::proof::R1csProofMergedLigeritoAg,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, verifier::VerifyError> {
        let union = flock_core::union::UnionInstance::new(&self.registry, vec![self.n_blocks]);
        let circuit = self.r1cs.csc_lincheck_circuit();
        let circs: [&dyn flock_core::lincheck::LincheckCircuit; 1] = [circuit];
        verifier::verify_ligerito_union_ag(
            &union,
            &circs,
            commitment,
            proof,
            &self.pcs_params,
            challenger,
        )
    }

    /// The AG-skip prover runs the DIRECT (dense pow2-lane) commit — the
    /// standard-pack shape its zerocheck and claims are wired for — while
    /// `prove_fast` moved to the single-slot UNION commit (dense stack +
    /// integer lanes). These params are the direct shape at the r1cs's own
    /// `m`; the commitment carries them, so [`Self::verify_ag`] follows.
    fn direct_pcs_params(&self) -> PcsParams {
        PcsParams {
            m: self.r1cs.m,
            log_inv_rate: self.pcs_params.log_inv_rate,
            log_batch_size: flock_core::pcs::ligerito::embedded_initial_k_or_default(
                self.r1cs.m,
                self.pcs_params.profile,
            ),
            profile: self.pcs_params.profile,
            num_lanes: None,
            merkle_hash: self.pcs_params.merkle_hash,
        }
    }

    /// AG-skip mirror of [`Self::prove_fast`]: round 1 of the zerocheck runs
    /// on the genus-95 AG multiplication code instead of the RS additive-NTT
    /// skip; everything else (witness gen, commit, lincheck, ring-switch open)
    /// is shared. Witness generation dispatches on the r1cs's witness layout,
    /// so both row-major and batch-major setups work. aarch64-only (NEON
    /// round-1 kernel).
    #[cfg(target_arch = "aarch64")]
    pub fn prove_fast_ag<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigeritoAg,
        Commitment,
        R1csClaim,
    ) {
        assert_eq!(blocks.len(), self.n_blocks);
        let pcs_params = self.direct_pcs_params();
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&pcs_params, || {
                self.generate_witness_ab(blocks)
            });
        let lc_circuit = self.r1cs.csc_lincheck_circuit();
        crate::prover::prove_fast_ligerito_ag_from_witness(
            &self.r1cs,
            &pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            lc_circuit,
            codeword,
            challenger,
        )
    }

    /// AG-skip mirror of [`Self::verify`].
    pub fn verify_ag<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &flock_core::proof::R1csProofLigeritoAg,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, verifier::VerifyError> {
        let lc_circuit = self.r1cs.csc_lincheck_circuit();
        verifier::verify_ligerito_ag(
            &self.r1cs,
            commitment,
            proof,
            lc_circuit,
            &self.direct_pcs_params(),
            challenger,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Batch-major witness producer (WitnessLayout::BatchMajor).
//
// V = 8 compressions in lockstep ([u32; 8] lanes); witness fields OR'd
// V-wide into an L1-resident interleaved row buffer (already batch-major
// order), NT-flushed per useful 128-bit chunk by the shared driver. See
// `common::drive_witness_batch_major`.
// ---------------------------------------------------------------------------

use super::common::{BM_V, BmRow, add_carry_parts_v, fused_add3_parts_v, or_bit_row, or_u32_row};

#[inline(always)]
fn bm_xor_rotr(x: &[u32; BM_V], y: &[u32; BM_V], r: u32) -> [u32; BM_V] {
    std::array::from_fn(|j| (x[j] ^ y[j]).rotate_right(r))
}

struct BmRows<'a> {
    z: &'a mut [BmRow],
    a: &'a mut [BmRow],
    b: &'a mut [BmRow],
}

#[inline(always)]
fn bm_write_lin(rows: &mut BmRows<'_>, bit: usize, vals: &[u32; BM_V]) {
    or_u32_row(rows.z, bit, vals);
    or_u32_row(rows.a, bit, vals);
    or_u32_row(rows.b, bit, &[0xFFFF_FFFF; BM_V]);
}

#[inline(always)]
fn bm_add_inline(
    rows: &mut BmRows<'_>,
    x: &[u32; BM_V],
    y: &[u32; BM_V],
    carry_bit: usize,
) -> [u32; BM_V] {
    let (sum, left, right, carry) = add_carry_parts_v(x, y);
    or_u32_row(rows.z, carry_bit, &carry);
    or_u32_row(rows.a, carry_bit, &left);
    or_u32_row(rows.b, carry_bit, &right);
    sum
}

#[inline(always)]
fn bm_fused_add_inline(
    rows: &mut BmRows<'_>,
    x: &[u32; BM_V],
    y: &[u32; BM_V],
    m: &[u32; BM_V],
    maj_bit: usize,
    rip_bit: usize,
) -> [u32; BM_V] {
    let (sum, maj, rip) = fused_add3_parts_v(x, y, m);
    or_u32_row(rows.z, maj_bit, &maj[2]);
    or_u32_row(rows.a, maj_bit, &maj[0]);
    or_u32_row(rows.b, maj_bit, &maj[1]);
    or_u32_row(rows.z, rip_bit, &rip[2]);
    or_u32_row(rows.a, rip_bit, &rip[0]);
    or_u32_row(rows.b, rip_bit, &rip[1]);
    sum
}

#[inline(always)]
fn bm_const_add_inline(
    rows: &mut BmRows<'_>,
    k: u32,
    y: &[u32; BM_V],
    bit: usize,
    n_rows: usize,
) -> [u32; BM_V] {
    let t = CARRY_BITS_PER_ADD - n_rows;
    let mask = (1u32 << n_rows) - 1;
    let mut sum = [0u32; BM_V];
    let mut zc = [0u32; BM_V];
    let mut ac = [0u32; BM_V];
    let mut bc = [0u32; BM_V];
    for j in 0..BM_V {
        let (s, left, right, carry) = add_carry_parts(k, y[j]);
        sum[j] = s;
        zc[j] = (carry >> t) & mask;
        ac[j] = (left >> t) & mask;
        bc[j] = (right >> t) & mask;
    }
    or_u32_row(rows.z, bit, &zc);
    or_u32_row(rows.a, bit, &ac);
    or_u32_row(rows.b, bit, &bc);
    sum
}

/// Build one V = 8 group of compressions into interleaved rows. Mirrors
/// [`build_block_witness_ab_packed_into`] field-for-field (byte-equality is
/// pinned by the lockstep test below).
/// `pub(crate)` so the composite Merkle block can reuse it per level: a
/// level's subcube IS this base block, aligned to a `2^k_log` boundary (hence
/// a whole number of `u64` rows), so `merkle_r1cs` calls this on the level's
/// row window and then writes only the swap-gadget and global columns itself.
pub(crate) fn build_group_batch_major(
    inputs: [&Compression; BM_V],
    rz: &mut [BmRow],
    ra: &mut [BmRow],
    rb: &mut [BmRow],
) {
    let mut rows = BmRows {
        z: rz,
        a: ra,
        b: rb,
    };
    let cv: [[u32; BM_V]; 8] = std::array::from_fn(|w| std::array::from_fn(|j| inputs[j].0[w]));
    let m: [[u32; BM_V]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| inputs[j].1[i]));
    let counter_lo: [u32; BM_V] = std::array::from_fn(|j| inputs[j].2 as u32);
    let counter_hi: [u32; BM_V] = std::array::from_fn(|j| (inputs[j].2 >> 32) as u32);
    let block_len: [u32; BM_V] = std::array::from_fn(|j| inputs[j].3);
    let flags: [u32; BM_V] = std::array::from_fn(|j| inputs[j].4);

    or_bit_row(rows.z, Z_CONST_POS);
    or_bit_row(rows.a, Z_CONST_POS);
    or_bit_row(rows.b, Z_CONST_POS);

    for w in 0..8 {
        bm_write_lin(&mut rows, cv_bit(w, 0), &cv[w]);
    }
    for i in 0..16 {
        bm_write_lin(&mut rows, m_bit(i, 0), &m[i]);
    }
    bm_write_lin(&mut rows, T_LO_BASE, &counter_lo);
    bm_write_lin(&mut rows, T_HI_BASE, &counter_hi);
    bm_write_lin(&mut rows, BLEN_BASE, &block_len);
    bm_write_lin(&mut rows, FLAGS_BASE, &flags);

    let mut state: [[u32; BM_V]; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        [BLAKE3_IV[0]; BM_V],
        [BLAKE3_IV[1]; BM_V],
        [BLAKE3_IV[2]; BM_V],
        [BLAKE3_IV[3]; BM_V],
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];
    let msg_idx = per_round_msg_idx();
    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_i, my_i] = msg_idx[r][g_in_round];
            let mx = m[mx_i];
            let my = m[my_i];

            let a_val = state[la];
            let b_val = state[lb];
            let c_val = state[lc];
            let d_val = state[ld];

            let a_1 = bm_fused_add_inline(
                &mut rows,
                &a_val,
                &b_val,
                &mx,
                g_bit(g, OFF_MAJ1),
                g_bit(g, OFF_RIP1),
            );
            let d_1 = bm_xor_rotr(&d_val, &a_1, 16);
            let c_1 = if g < 4 {
                bm_const_add_inline(
                    &mut rows,
                    BLAKE3_IV[g],
                    &d_1,
                    g_bit(g, OFF_C1),
                    g_c1_rows(g),
                )
            } else {
                bm_add_inline(&mut rows, &c_val, &d_1, g_bit(g, OFF_C1))
            };
            let b_1 = bm_xor_rotr(&b_val, &c_1, 12);
            let a_2 = bm_fused_add_inline(
                &mut rows,
                &a_1,
                &b_1,
                &my,
                g_bit(g, off_maj2(g)),
                g_bit(g, off_rip2(g)),
            );
            let d_2 = bm_xor_rotr(&d_1, &a_2, 8);
            let c_2 = bm_add_inline(&mut rows, &c_1, &d_2, g_bit(g, off_c2(g)));
            let b_new = bm_xor_rotr(&b_1, &c_2, 7);
            let d_new = d_2;

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    for w in 0..8 {
        let lo: [u32; BM_V] = std::array::from_fn(|j| state[w][j] ^ state[w + 8][j]);
        let hi: [u32; BM_V] = std::array::from_fn(|j| state[w + 8][j] ^ cv[w][j]);
        bm_write_lin(&mut rows, out_lo_bit(w, 0), &lo);
        bm_write_lin(&mut rows, out_hi_bit(w, 0), &hi);
    }
}

/// Batch-major counterpart of [`generate_witness_with_ab_packed_and_lincheck`]
/// — `(z, a, b, z_lincheck)` with z/a/b in the batch-major layout. Padding
/// slots run a compression of the all-zero input (constant wire = 1).
pub fn generate_witness_batch_major(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
    super::common::drive_witness_batch_major(
        blocks,
        &padding,
        n_blocks_log,
        K_LOG,
        USEFUL_BITS,
        build_group_batch_major,
    )
}

/// Partial-count batch-major witness for the union's dynamic invocation
/// counts (M4): rows `[blocks.len(), 2^n_blocks_log)` are left
/// **identically zero** (z, a, b, and stripe — constant wire included; the
/// union lincheck's count-derived const-pin target requires zero dummies,
/// not padding compressions). `blocks.len()` may be any value up to the
/// capacity, not necessarily a power of two.
pub fn generate_witness_batch_major_partial(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    super::common::drive_witness_batch_major_partial(
        blocks,
        n_blocks_log,
        K_LOG,
        USEFUL_BITS,
        build_group_batch_major,
    )
}

/// [`generate_witness_batch_major`] writing into a union slot's destination
/// block instead of fresh buffers — the copy-free union assembly path (see
/// [`flock_core::union::SlotWitnessDest`]). Returns the lincheck stripe.
pub fn generate_witness_batch_major_into(
    blocks: &[Compression],
    n_blocks_log: usize,
    dst: flock_core::union::SlotWitnessDest<'_>,
) -> Vec<u8> {
    let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
    super::common::drive_witness_batch_major_into(
        blocks,
        &padding,
        n_blocks_log,
        K_LOG,
        USEFUL_BITS,
        dst,
        build_group_batch_major,
    )
}

/// [`generate_witness_batch_major_partial`] writing into a union slot's
/// destination block — the copy-free union assembly path for dynamic counts.
pub fn generate_witness_batch_major_partial_into(
    blocks: &[Compression],
    n_blocks_log: usize,
    dst: flock_core::union::SlotWitnessDest<'_>,
) -> Vec<u8> {
    super::common::drive_witness_batch_major_partial_into(
        blocks,
        n_blocks_log,
        K_LOG,
        USEFUL_BITS,
        dst,
        build_group_batch_major,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The walker's constant-wire pin must equal the pin the R1CS itself
    /// declares — a walker inheriting the trait's `None` default silently
    /// drops the pin and reopens the all-zero-witness gap.
    #[test]
    fn walker_const_pin_matches_r1cs() {
        use flock_core::lincheck::LincheckCircuit as _;
        let r1cs = build_block_r1cs(3);
        assert_eq!(Blake3LincheckCircuit.const_pin_col(), r1cs.const_pin);
        assert_eq!(r1cs.const_pin, Some(Z_CONST_POS));
    }

    /// The IO schema must cover **every free-witness input region**: a word
    /// the relation does not pin and the schema does not expose is a value the
    /// prover picks. It must also be word-aligned and non-overlapping, and its
    /// named cell-slot constants must match its order.
    #[test]
    fn io_schema_covers_every_unpinned_input() {
        use flock_core::schedule::IoDirection;
        let schema = io_schema();

        // Named indices agree with the enumeration order.
        assert_eq!(schema[IO_CV0].word_col, CV_BASE / 128);
        assert_eq!(schema[IO_M0].word_col, M_BASE / 128);
        assert_eq!(schema[IO_PARAMS].word_col, T_LO_BASE / 128);
        assert_eq!(schema[IO_OUT_LO0].word_col, OUT_LO_BASE / 128);
        assert_eq!(schema[IO_OUT_HI0].word_col, OUT_HI_BASE / 128);

        // Distinct words, and each region is 128-bit aligned.
        let mut cols: Vec<usize> = schema.iter().map(|w| w.word_col).collect();
        let n = cols.len();
        cols.sort_unstable();
        cols.dedup();
        assert_eq!(cols.len(), n, "schema repeats a word column");
        for base in [CV_BASE, OUT_LO_BASE, M_BASE, T_LO_BASE, OUT_HI_BASE] {
            assert_eq!(base % 128, 0, "region at bit {base} is not word-aligned");
        }

        // Every input bit of the block is exposed: cv, m and the packed params
        // word. `GS_BASE` onward is internal (round intermediates, carries) and
        // is pinned by the relation, so it is correctly absent.
        let covered: std::collections::HashSet<usize> = schema
            .iter()
            .filter(|w| w.dir == IoDirection::In)
            .map(|w| w.word_col)
            .collect();
        for bit in (CV_BASE..CV_BASE + SLOT_BITS).step_by(128) {
            assert!(covered.contains(&(bit / 128)), "cv bit {bit} unexposed");
        }
        for bit in (M_BASE..M_BASE + 16 * WORD_BITS).step_by(128) {
            assert!(covered.contains(&(bit / 128)), "m bit {bit} unexposed");
        }
        assert!(
            covered.contains(&(T_LO_BASE / 128)),
            "counter/block_len/flags word unexposed — the prover could choose \
             the chunk index and the CHUNK_START/CHUNK_END/ROOT flags"
        );
        // And the params word covers exactly those four u32s.
        assert_eq!(OUT_HI_BASE - T_LO_BASE, 128);
    }

    use flock_core::test_rng::Rng;

    /// BLAKE3 chunk flags (subset).
    const CHUNK_START: u32 = 1 << 0;
    const CHUNK_END: u32 = 1 << 1;
    const ROOT: u32 = 1 << 3;

    /// Batch-major witness equality vs the row-major driver (word-transpose
    /// + identical stripe), incl. padding slots via a non-power-of-two count.
    #[test]
    fn batch_major_witness_matches_row_major_transposed() {
        for (n_inputs, n_log) in [(8usize, 3usize), (11, 4)] {
            let mut rng = Rng::new(0xBA7C_B3 + n_log as u64);
            let inputs: Vec<Compression> = (0..n_inputs)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                    (cv, m, counter, 64u32, 11u32)
                })
                .collect();

            let (z_r, a_r, b_r, stripe_r) =
                generate_witness_with_ab_packed_and_lincheck(&inputs, n_log);
            let (z_b, a_b, b_b, stripe_b) = generate_witness_batch_major(&inputs, n_log);

            assert_eq!(stripe_b, stripe_r, "stripe diverged (n_log={n_log})");

            let chunks_per_block = K / 128;
            let transpose = |row: &[flock_core::field::F128]| {
                let mut out = vec![flock_core::field::F128::ZERO; row.len()];
                for o in 0..1usize << n_log {
                    for c in 0..chunks_per_block {
                        out[(c << n_log) + o] = row[o * chunks_per_block + c];
                    }
                }
                out
            };
            assert_eq!(z_b, transpose(&z_r), "z diverged (n_log={n_log})");
            assert_eq!(a_b, transpose(&a_r), "a diverged (n_log={n_log})");
            assert_eq!(b_b, transpose(&b_r), "b diverged (n_log={n_log})");
        }
    }

    /// The partial-count driver (M4 dynamic counts): declared rows match the
    /// full driver word for word, dummy rows `[n, 2^n_log)` are identically
    /// zero in z/a/b, and the stripe equals the canonical `pack_z_lincheck`
    /// of the (zero-dummy) witness. Covers a partial final 8-group, an exact
    /// group boundary, fully-dummy trailing groups, and the empty count.
    #[test]
    fn batch_major_partial_zeroes_dummy_rows() {
        use flock_core::field::F128;
        use flock_core::lincheck::pack_z_lincheck_from_packed;

        let n_log = 4usize;
        let n_total = 1usize << n_log;
        let m = K_LOG + n_log;
        let mut rng = Rng::new(0xBA7C_9427);
        let inputs: Vec<Compression> = (0..n_total)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let msg: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                (cv, msg, counter, 64u32, 11u32)
            })
            .collect();
        let (z_f, a_f, b_f, _) = generate_witness_batch_major(&inputs, n_log);

        let chunks_per_block = K / 128;
        for n in [11usize, 8, 16, 0] {
            let (z_p, a_p, b_p, stripe_p) =
                generate_witness_batch_major_partial(&inputs[..n], n_log);
            for ((pb, fb), what) in [(&z_p, &z_f), (&a_p, &a_f), (&b_p, &b_f)]
                .into_iter()
                .zip(["z", "a", "b"])
            {
                for c in 0..chunks_per_block {
                    for o in 0..n_total {
                        let w = (c << n_log) + o;
                        if o < n {
                            assert_eq!(pb[w], fb[w], "{what} declared word (n={n}, c={c}, o={o})");
                        } else {
                            assert_eq!(
                                pb[w],
                                F128::ZERO,
                                "{what} dummy word must be zero (n={n}, c={c}, o={o})"
                            );
                        }
                    }
                }
            }
            // Stripe: canonical pack_z_lincheck of the zero-dummy witness
            // (batch-major → row-major word transpose first).
            let mut z_row = vec![F128::ZERO; z_p.len()];
            for o in 0..n_total {
                for c in 0..chunks_per_block {
                    z_row[o * chunks_per_block + c] = z_p[(c << n_log) + o];
                }
            }
            assert_eq!(
                stripe_p,
                pack_z_lincheck_from_packed(&z_row, m, K_LOG),
                "stripe diverged (n={n})"
            );
        }
    }

    /// Batch-major end-to-end Ligerito roundtrip + tamper rejection.
    #[test]
    #[ignore]
    fn batch_major_prove_fast_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let setup = Blake3Setup::new(256);
        let mut rng = Rng::new(0xBA7C_F013);
        let inputs: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                (cv, m, counter, 64u32, 11u32)
            })
            .collect();

        let mut ch_p = FsChallenger::new(b"flock-lig-batch-major-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-lig-batch-major-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("batch-major verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let mut bad = proof.clone();
        bad.zerocheck.final_a_eval.lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-lig-batch-major-v0");
        assert!(
            setup.verify(&commitment, &bad, &mut ch).is_err(),
            "tampered batch-major proof accepted"
        );
    }

    /// AG-skip e2e: prove_fast_ag → verify_ag roundtrip + tamper rejection,
    /// through the full commit → AG zerocheck → lincheck → ring-switch
    /// Ligerito open pipeline.
    #[cfg(target_arch = "aarch64")]
    #[test]
    // Default-run: this is the guard for the direct-shape params class of
    // stranding (the AG/timed entry points commit the standard-pack witness,
    // so a union-shaped `pcs_params` panics them all).
    fn prove_fast_ligerito_ag_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = Blake3Setup::new(256);
        let mut rng = Rng::new(0xb1a_3a9_211e);
        let blocks: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, 0u64, 64u32, 11u32)
            })
            .collect();
        let mut ch_p = FsChallenger::new(b"flock-blake3-ag-v0");
        let (proof, commitment, claim_p) = setup.prove_fast_ag(&blocks, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-blake3-ag-v0");
        let claim_v = setup
            .verify_ag(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("AG verify rejected honest blake3 proof: {e:?}"));
        assert_eq!(claim_p, claim_v, "AG verifier claim != prover claim");

        // Tampering an AG round-1 message must reject.
        let mut bad = proof.clone();
        bad.ag.round1_ab[0] += flock_core::field::F128::ONE;
        let mut ch_b = FsChallenger::new(b"flock-blake3-ag-v0");
        assert!(
            setup.verify_ag(&commitment, &bad, &mut ch_b).is_err(),
            "must reject a tampered AG proof"
        );
    }

    /// Batch-major AG roundtrip: the AG claim points are built through the
    /// layout-aware [`BlockR1cs`] constructors, so the batch-major witness
    /// layout works unchanged (mirror of `batch_major_prove_fast_roundtrip`).
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore] // Heavy — run with `cargo test batch_major_prove_fast_ag_roundtrip -- --ignored`
    fn batch_major_prove_fast_ag_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let setup = Blake3Setup::new(256);
        let mut rng = Rng::new(0xBA7C_F013);
        let inputs: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                (cv, m, counter, 64u32, 11u32)
            })
            .collect();

        let mut ch_p = FsChallenger::new(b"flock-ag-batch-major-v0");
        let (proof, commitment, claim_p) = setup.prove_fast_ag(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-ag-batch-major-v0");
        let claim_v = setup
            .verify_ag(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("batch-major AG verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let mut bad = proof.clone();
        bad.ag.final_a_eval.lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-ag-batch-major-v0");
        assert!(
            setup.verify_ag(&commitment, &bad, &mut ch).is_err(),
            "tampered batch-major AG proof accepted"
        );
    }

    /// UNION-AG e2e: prove_fast_union_ag → verify_union_ag roundtrip +
    /// tamper rejection — the AG zerocheck inside the single-slot union
    /// transport (dense stack + integer lanes, union lincheck, merged
    /// opening on `SkipPoint::Ag` claim points), under the profile's full
    /// grinding schedule.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore] // Heavy — run with `cargo test prove_fast_union_ag_roundtrip -- --ignored`
    fn prove_fast_union_ag_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = Blake3Setup::new(256);
        let mut rng = Rng::new(0xA9_0110_4A6);
        let blocks: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                (cv, m, counter, 64u32, 11u32)
            })
            .collect();
        let mut ch_p = FsChallenger::new(b"flock-union-ag-v0");
        let (proof, commitment, claim_p) = setup.prove_fast_union_ag(&blocks, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-union-ag-v0");
        let claim_v = setup
            .verify_union_ag(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("union-AG verify rejected honest proof: {e:?}"));
        assert_eq!(claim_p, claim_v, "union-AG verifier claim != prover claim");

        // Tampering an AG round-1 message must reject.
        let mut bad = proof.clone();
        bad.boolean.ag.round1_ab[0] += flock_core::field::F128::ONE;
        let mut ch = FsChallenger::new(b"flock-union-ag-v0");
        assert!(
            setup.verify_union_ag(&commitment, &bad, &mut ch).is_err(),
            "must reject a tampered union-AG round-1 message"
        );

        // A nonce vector off the grinding schedule must reject (count check).
        let mut bad = proof.clone();
        bad.boolean.ag.grinding_nonces.push(0);
        let mut ch = FsChallenger::new(b"flock-union-ag-v0");
        assert!(
            setup.verify_union_ag(&commitment, &bad, &mut ch).is_err(),
            "must reject a proof with an off-schedule nonce count"
        );

        // If the schedule grinds, a corrupted nonce must reject too.
        if !proof.boolean.ag.grinding_nonces.is_empty() {
            let mut bad = proof.clone();
            bad.boolean.ag.grinding_nonces[0] ^= 1;
            let mut ch = FsChallenger::new(b"flock-union-ag-v0");
            assert!(
                setup.verify_union_ag(&commitment, &bad, &mut ch).is_err(),
                "must reject a corrupted grinding nonce"
            );
        }

        // The FUSED r1 nonce: any change must reject (bad PoW, bad point, or
        // a diverged r1 failing the c-eval bind).
        let mut bad = proof.clone();
        bad.boolean.ag.r1_nonce = bad.boolean.ag.r1_nonce.wrapping_add(1);
        let mut ch = FsChallenger::new(b"flock-union-ag-v0");
        assert!(
            setup.verify_union_ag(&commitment, &bad, &mut ch).is_err(),
            "must reject a tampered fused r1 nonce"
        );

        // The lincheck's FUSED AG skip nonce (the last lincheck nonce).
        let mut bad = proof.clone();
        *bad.boolean
            .lincheck
            .grinding_nonces
            .last_mut()
            .expect("the AG arm carries a fused skip nonce") ^= 1;
        let mut ch = FsChallenger::new(b"flock-union-ag-v0");
        assert!(
            setup.verify_union_ag(&commitment, &bad, &mut ch).is_err(),
            "must reject a tampered fused lincheck skip nonce"
        );
    }

    /// UNION-AG at PARTIAL UTILIZATION (200/256 declared rows): the
    /// count-derived multi-run spec drives the AG run-list arms — round-1
    /// full/partial/dead segments and the gated fold — through the real
    /// transport, with the witness buffers eligible for the dirty pool
    /// (the PooledDirty election no longer excludes the AG flavor). The
    /// prove runs TWICE: the second draw takes the buffers the first
    /// returned DIRTY, and byte-identical proofs pin the arms'
    /// read-exactness in situ (the kernel-level legs live in
    /// `ag_skip::padded_arms_match_dense_and_ignore_dirty_padding`).
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore] // Heavy — run with `-- --ignored`.
    fn prove_fast_union_ag_partial_utilization_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = Blake3Setup::new(200);
        let mut rng = Rng::new(0xA9_0110_4A7);
        let blocks: Vec<Compression> = (0..200)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                (cv, m, counter, 64u32, 11u32)
            })
            .collect();
        let mut ch_p = FsChallenger::new(b"flock-union-ag-part");
        let (proof, commitment, claim_p) = setup.prove_fast_union_ag(&blocks, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-union-ag-part");
        let claim_v = setup
            .verify_union_ag(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("partial-utilization union-AG rejected: {e:?}"));
        assert_eq!(claim_p, claim_v, "verifier claim != prover claim");

        let mut ch_p2 = FsChallenger::new(b"flock-union-ag-part");
        let (proof2, _, _) = setup.prove_fast_union_ag(&blocks, &mut ch_p2);
        assert_eq!(proof, proof2, "re-prove over pooled-dirty buffers");
    }

    #[test]
    fn layout_constants() {
        // I/O-aligned layout: cv in slot 0, out_lo in slot 1 (both 256-bit),
        // and EVERY wireable region on a 128-bit word boundary (m, params,
        // out_hi), const pin at the end.
        assert_eq!(CV_BASE, 0);
        assert_eq!(OUT_LO_BASE, 256);
        assert_eq!(M_BASE, 512);
        assert_eq!(T_LO_BASE, 1024);
        assert_eq!(OUT_HI_BASE, 1152);
        assert_eq!(GS_BASE, 1408);
        assert_eq!(G_STRIDE, 184);
        assert_eq!(N_G, 56);
        // Option F G-region: 52 generic G's at 184 bits, round 1's column
        // G's at 183/183/182/182 (constant-c adders) = 10,298 AND rows.
        assert_eq!(G_BASE[0], GS_BASE);
        assert_eq!(g_block_bits(0), 183);
        assert_eq!(g_block_bits(1), 183);
        assert_eq!(g_block_bits(2), 182);
        assert_eq!(g_block_bits(3), 182);
        assert_eq!(g_block_bits(4), G_STRIDE);
        assert_eq!(G_BASE[N_G] - GS_BASE, 10_298);
        assert_eq!(Z_CONST_POS, 11_706);
        assert_eq!(USEFUL_BITS, 11_707);
        assert!(USEFUL_BITS <= K);
        assert_eq!(CV_BASE % SLOT_BITS, 0);
        assert_eq!(OUT_LO_BASE % SLOT_BITS, 0);
        // Word alignment of every wireable region (the wiring layer's
        // gather lemma freezes whole 128-bit words).
        for base in [CV_BASE, OUT_LO_BASE, M_BASE, T_LO_BASE, OUT_HI_BASE] {
            assert_eq!(base % 128, 0);
        }
        assert_eq!(Z_CONST_POS, USEFUL_BITS - 1);
    }

    /// Reference compression matches the `blake3` crate for empty input
    /// (a single root-block, single-chunk, ROOT-flagged compression).
    #[test]
    fn compress_matches_blake3_crate_empty() {
        let state = blake3_compress(
            &BLAKE3_IV,
            &[0u32; 16],
            0,
            0,
            CHUNK_START | CHUNK_END | ROOT,
        );
        let mut got = [0u8; 32];
        for w in 0..8 {
            got[w * 4..w * 4 + 4].copy_from_slice(&state[w].to_le_bytes());
        }
        let expected = *::blake3::hash(b"").as_bytes();
        assert_eq!(got, expected);
    }

    /// Reference compression matches the `blake3` crate for a full 64-byte
    /// input (single block + single chunk + root).
    #[test]
    fn compress_matches_blake3_crate_64_bytes() {
        let mut rng = Rng::new(0xDEAD_BEEF);
        let mut bytes = [0u8; 64];
        for byte in bytes.iter_mut() {
            *byte = (rng.next_u32() & 0xFF) as u8;
        }
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let state = blake3_compress(&BLAKE3_IV, &m, 0, 64, CHUNK_START | CHUNK_END | ROOT);
        let mut got = [0u8; 32];
        for w in 0..8 {
            got[w * 4..w * 4 + 4].copy_from_slice(&state[w].to_le_bytes());
        }
        let expected = *::blake3::hash(&bytes).as_bytes();
        assert_eq!(got, expected);
    }

    /// Witness's out_lo / out_hi slots equal the BLAKE3 finalization XORs.
    #[test]
    fn witness_encodes_correct_output() {
        let mut rng = Rng::new(0x1234_5678);
        let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
        let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
        let block_len = 64;
        let flags = CHUNK_START | CHUNK_END | ROOT;
        let z = build_block_witness(&cv, &m, counter, block_len, flags);
        let expected = blake3_compress(&cv, &m, counter, block_len, flags);
        for w in 0..8 {
            let mut got = 0u32;
            for b in 0..WORD_BITS {
                if z[out_lo_bit(w, b)] {
                    got |= 1 << b;
                }
            }
            assert_eq!(got, expected[w], "out_lo[{w}] mismatch");
            let mut got_hi = 0u32;
            for b in 0..WORD_BITS {
                if z[out_hi_bit(w, b)] {
                    got_hi |= 1 << b;
                }
            }
            assert_eq!(got_hi, expected[w + 8], "out_hi[{w}] mismatch");
        }
    }

    #[test]
    fn honest_witness_satisfies_r1cs() {
        let mut rng = Rng::new(0xCAFE_F00D);
        for &n_blocks in &[1usize, 3, 8] {
            let n_log = min_n_blocks_log(n_blocks).max(3);
            let r1cs = build_block_r1cs(n_log);
            let blocks: Vec<Compression> = (0..n_blocks)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
                })
                .collect();
            let z = generate_witness(&blocks, n_log);
            assert_eq!(z.len(), r1cs.n());
            assert!(
                r1cs.satisfies(&z),
                "witness for {n_blocks} compressions fails R1CS"
            );
        }
    }

    #[test]
    fn mutated_witness_fails() {
        let mut rng = Rng::new(0xBEEF_F00D);
        let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
        let r1cs = build_block_r1cs(3);
        let blocks = vec![(cv, m, 0u64, 64u32, 11u32)];
        let mut z = generate_witness(&blocks, 3);
        assert!(r1cs.satisfies(&z));
        // Flip a maj-product bit inside G #10's second fused ADD.
        z[g_bit(10, off_maj2(10) + 5)] ^= true;
        assert!(
            !r1cs.satisfies(&z),
            "tampered carry bit should violate R1CS"
        );
    }

    /// `generate_witness_with_ab_packed` agrees with the matrix-vector
    /// products `apply_a_packed(z)` and `apply_b_packed(z)`. Also asserts
    /// `apply_c_packed(z) == z` (C = I), validating the aliasing assumption
    /// used by prove_fast.
    #[test]
    fn generate_witness_with_ab_packed_matches_apply() {
        for &n_blocks in &[1usize, 4, 8] {
            let n_log = min_n_blocks_log(n_blocks).max(3);
            let r1cs = build_block_r1cs(n_log);
            let mut rng = Rng::new(0xABCD_5A55 + n_blocks as u64);
            let blocks: Vec<Compression> = (0..n_blocks)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
                })
                .collect();

            let (z, a, b) = generate_witness_with_ab_packed(&blocks, n_log);
            let a_ref = r1cs.apply_a_packed(&z);
            let b_ref = r1cs.apply_b_packed(&z);
            let c_ref = r1cs.apply_c_packed(&z);
            assert_eq!(a, a_ref, "a mismatch at n_blocks={n_blocks}");
            assert_eq!(b, b_ref, "b mismatch at n_blocks={n_blocks}");
            // C = I, so c == z. prove_fast relies on this for the c-aliasing.
            assert_eq!(c_ref, z, "C is not identity at n_blocks={n_blocks}");
            assert!(r1cs.satisfies_packed(&z));
        }
    }

    /// The fused generator produces (z, a, b) byte-identical to
    /// `generate_witness_with_ab_packed` AND a lincheck stripe byte-identical
    /// `Blake3LincheckCircuit` walker matches the sparse fold byte-for-byte
    /// at random α + random eq_inner.
    #[test]
    fn lincheck_circuit_matches_sparse() {
        use flock_core::lincheck::{LincheckCircuit, SparseMatrixCircuit};

        let mut rng = Rng::new(0xB1A_E3_CCA1);
        let (a_0, b_0) = build_matrices();
        let sparse = SparseMatrixCircuit::new(&a_0, &b_0);
        let walker = Blake3LincheckCircuit;
        assert_eq!(sparse.n_cols(), walker.n_cols());

        let n_cols = walker.n_cols();
        let alpha = F128 {
            lo: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
            hi: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
        };
        let eq_inner: Vec<F128> = (0..n_cols)
            .map(|_| F128 {
                lo: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
                hi: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
            })
            .collect();

        let expected = sparse.fold_alpha_batched(alpha, &eq_inner);
        let got = walker.fold_alpha_batched(alpha, &eq_inner);
        for c in 0..n_cols {
            assert_eq!(expected[c], got[c], "comb mismatch at col {c}");
        }

        // CSC gather (what prove_fast/verify actually use) matches too.
        let csc = flock_core::lincheck::CscCircuit::from_matrices(&a_0, &b_0);
        let got_csc = csc.fold_alpha_batched(alpha, &eq_inner);
        assert_eq!(expected, got_csc, "CSC fold mismatch");
    }

    /// to `pack_z_lincheck_from_packed(z)`.
    #[test]
    fn fused_lincheck_matches_separate() {
        use flock_core::lincheck::pack_z_lincheck_from_packed;
        for &n_blocks in &[1usize, 4, 8, 13] {
            let n_log = min_n_blocks_log(n_blocks).max(3);
            let r1cs = build_block_r1cs(n_log);
            let mut rng = Rng::new(0xABCD_EF00 + n_blocks as u64);
            let blocks: Vec<Compression> = (0..n_blocks)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
                })
                .collect();

            let (z1, a1, b1) = generate_witness_with_ab_packed(&blocks, n_log);
            let lincheck_ref = pack_z_lincheck_from_packed(&z1, r1cs.m, r1cs.k_log);
            let (z2, a2, b2, lincheck_new) =
                generate_witness_with_ab_packed_and_lincheck(&blocks, n_log);
            assert_eq!(z1, z2, "z mismatch at n_blocks={n_blocks}");
            assert_eq!(a1, a2, "a mismatch at n_blocks={n_blocks}");
            assert_eq!(b1, b2, "b mismatch at n_blocks={n_blocks}");
            assert_eq!(
                lincheck_ref, lincheck_new,
                "lincheck stripe mismatch at n_blocks={n_blocks}"
            );
        }
    }

    /// Full prove→verify round-trip through the Ligerito PCS for EACH named
    /// profile (fast = JohnsonOod 100-bit, slim = JohnsonOod 100-bit + query
    /// grinding, secure = UDR 120-bit). 256 blocks → m=22, the smallest
    /// embedded config. Drives OOD binding + fold grinding through the real
    /// R1CS / ring-switch / recursive-sumcheck pipeline end to end.
    #[test]
    fn prove_verify_ligerito_all_profiles() {
        use flock_core::challenger::FsChallenger;
        use flock_core::pcs::ligerito::LigeritoProfile;
        let blocks: Vec<Compression> = {
            let mut rng = Rng::new(0x9A11_0F11);
            (0..256)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, 0u64, 64u32, 11u32)
                })
                .collect()
        };
        for profile in [
            LigeritoProfile::Fast,
            LigeritoProfile::Slim,
            LigeritoProfile::Secure,
        ] {
            let setup = Blake3Setup::with_profile(256, profile);
            let mut ch_p = FsChallenger::new(b"flock-blake3-prof");
            let (proof, commitment, claim_p) = setup.prove_fast(&blocks, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"flock-blake3-prof");
            let claim_v = setup
                .verify(&commitment, &proof, &mut ch_v)
                .unwrap_or_else(|e| {
                    panic!(
                        "ligerito verify rejected for profile {}: {e:?}",
                        profile.as_str()
                    )
                });
            assert_eq!(
                claim_p,
                claim_v,
                "claim mismatch for profile {}",
                profile.as_str()
            );
        }
    }

    /// Ligerito-backend prove_fast roundtrip. Needs ≥ 256 blocks (m=22) for
    /// the default Ligerito config at log_batch_size=6.
    #[test]
    #[ignore]
    fn prove_fast_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = Blake3Setup::new(256);
        let mut rng = Rng::new(0xb1a_3211e);
        let blocks: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, 0u64, 64u32, 11u32)
            })
            .collect();
        let mut ch_p = FsChallenger::new(b"flock-blake3-lig-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&blocks, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-blake3-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    /// Constant-wire pin (docs/const-wire-pin.md). `new(250)` is a partial
    /// count: `prove_fast`'s batch-major partial witness leaves the dummy
    /// rows identically zero (no padding compressions) and the honest proof
    /// verifies; the all-zero witness must be rejected by the pin. (For BLAKE3 the pin lives on the R1CS-built CSC circuit, not
    /// the walker.)
    #[test]
    #[ignore] // Heavier — Ligerito needs m=22; run with `cargo test const_pin_all_zero_rejected -- --ignored`
    fn const_pin_all_zero_rejected() {
        use flock_core::challenger::FsChallenger;

        let n = 250; // 6 padding blocks at n_block_slots = 256 (m = 22)
        let setup = Blake3Setup::new(n);

        // (1) Honest proof with filled padding verifies.
        let mut rng = Rng::new(0x5EED_B1A3);
        let blocks: Vec<Compression> = (0..n)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, rng.next_u32() as u64, 64u32, 11u32)
            })
            .collect();
        let mut ch_p = FsChallenger::new(b"honest");
        let (proof, commitment, claim_p) = setup.prove_fast(&blocks, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"honest");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("honest padded proof rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        // (2) All-zero witness must be rejected by the pin (union path:
        // the count-derived const-pin target).
        let zeros: Vec<Compression> = vec![([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32); n];
        let (mut z, mut a, mut b, mut zlc) =
            generate_witness_batch_major_partial(&zeros, setup.n_blocks_log());
        z.iter_mut()
            .for_each(|v| *v = flock_core::field::F128::ZERO);
        a.iter_mut()
            .for_each(|v| *v = flock_core::field::F128::ZERO);
        b.iter_mut()
            .for_each(|v| *v = flock_core::field::F128::ZERO);
        zlc.iter_mut().for_each(|v| *v = 0);
        let union = flock_core::union::UnionInstance::new(&setup.registry, vec![n]);
        let slot = crate::prover::UnionSlotProverInput::new(
            (z, a, b, zlc),
            setup.r1cs.csc_lincheck_circuit(),
        );
        let mut ch_p = FsChallenger::new(b"poc");
        let (proof, commitment, _) = crate::prover::prove_fast_ligerito_union(
            &union,
            &setup.pcs_params,
            vec![slot],
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(b"poc");
        let res = setup.verify(&commitment, &proof, &mut ch_v);
        assert!(
            matches!(res, Err(flock_core::verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }

    #[test]
    fn setup_sizes_correctly() {
        for &(n_blocks, expected_n_log) in
            &[(1usize, 3), (8, 3), (9, 4), (16, 4), (17, 5), (1000, 10)]
        {
            let setup = Blake3Setup::new(n_blocks);
            assert_eq!(setup.n_blocks_log(), expected_n_log, "n_blocks={n_blocks}");
            assert_eq!(setup.m(), K_LOG + expected_n_log);
            assert!(setup.n_block_slots() >= n_blocks);
        }
    }
}
