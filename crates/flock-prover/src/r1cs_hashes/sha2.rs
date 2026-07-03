//! **SHA-256** compression-function R1CS-over-GF(2), **I/O-aligned layout for
//! the hash chain** (forked from [`super::sha2`]). Identical R1CS semantics;
//! only the input chaining value `H_in` and the output chaining value `H_out`
//! move to aligned 256-bit slots (slots 0 and 1), so the chain shift argument
//! folds them via a single tensor opening. `H_in` and `H_out` are exactly 256
//! bits each, so the slots have NO interior padding. `K_LOG = 15` is unchanged.
//!
//! ## Slot layout (single instance, chain-aligned)
//!
//! ```text
//! z[0..256]         H_in        — 8 words × 32 bits  (slot 0, byte 0)
//! z[256..512]       H_out       — 8 words × 32 bits  (slot 1, byte 32)
//! z[512]            Z_CONST (= 1)
//! z[513..1025]      M_in        — 16 words × 32 bits
//! z[1025..3073]     ch_and      — 64 rounds × 32 bits (AND outputs)
//! z[3073..5121]     maj_and     — 64 rounds × 32 bits (AND outputs)
//! z[5121..19009]    round carry-aux — 64 rounds × 7 adds × 31 carries
//! z[19009..20545]   W[t]        — 48 schedule final sums (sched_2)
//! z[20545..25009]   sched carries — 48 × 3 × 31
//! z[25009..27057]   T1[r]       — 64 round final T1 sums
//! z[27057..29105]   E_NEW[r]    — 64 round new-e sums
//! z[29105..31153]   A_NEW[r]    — 64 round new-a sums
//! z[31153..31401]   output carries — 8 × 31
//! z[31401..32768]   padding (forced to 0)
//! ```
//!
//! All bit placement goes through the `*_bit` accessors below — flipping the
//! base offsets is the only change required for the R1CS construction.
//!
//! ## Inlined adders
//!
//! Per 32-bit add, only the 31 `carry_aux` slots are allocated; the 32 sum
//! bits are symbolic XOR expressions inlined into the next consumer's row.
//! This keeps the witness compact (~31,401 useful rows).
//!
//! ## Sum slots that *are* materialized
//!
//! - `W[t]` for `t ∈ 16..64` — referenced once each by `T1_3`, but the
//!   schedule chain is 3 deep and `W[t]` itself depends on prior `W`'s
//!   (cascades for `t ≥ 32`). Slotting breaks the cascade.
//! - `T1[r]` — referenced twice (E_NEW and A_NEW), so slotting saves
//!   duplicate inlining.
//! - `E_NEW[r]`, `A_NEW[r]` — feed downstream rounds (4 uses each via
//!   register shift); without slots the state would cascade end-to-end and
//!   each Ch / Maj AND row would blow up to thousands of terms.
//! - `H_out[w]` — the public output of the compression.

use super::common::{BitRecord, add_carry_parts, or_bit_at, or_u32_at_bit};
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

// ───────────────────────────────────────────────────────────────────────────
// Compile-time slot layout
// ───────────────────────────────────────────────────────────────────────────

/// Inner-dimension log: `K = 2^15 = 32,768` rows per block.
pub const K_LOG: usize = 15;
pub const K: usize = 1 << K_LOG;
/// Univariate-skip width.
pub const K_SKIP: usize = 6;

pub const N_ROUNDS: usize = 64;
pub const N_SCHED: usize = 48;
pub const WORD_BITS: usize = 32;
pub const H_WORDS: usize = 8;
pub const M_WORDS: usize = 16;
pub const N_OUT_WORDS: usize = 8;
pub const ADDS_PER_ROUND: usize = 7;
pub const ADDS_PER_SCHED: usize = 3;
pub const CARRIES_PER_ADD: usize = WORD_BITS - 1; // 31

/// SHA-256 IV (FIPS 180-4 §5.3.3).
pub const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
/// SHA-256 round constants (FIPS 180-4 §4.2.2).
pub const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// **I/O-aligned layout** for the hash chain (forked from `sha2`): the input
// chaining value `H_in` lives in aligned slot 0 and the output chaining value
// `H_out` in aligned slot 1 — each a clean 256-bit (`2^8`) window, so the
// chain shift argument folds them via a single tensor opening. H_in/H_out are
// *exactly* 256 bits, so the slots have NO interior padding. Everything else
// (const, M, intermediates, output carries) packs after the two slots. The
// re-layout is purely a change of these base offsets — all bit placement goes
// through the `*_bit` accessors below.
pub const SLOT_BITS: usize = 256; // 2^8, one 256-bit chaining value
pub const H_BASE: usize = 0; // input region, slot 0: [0, 256)
pub const H_OUT_BASE: usize = SLOT_BITS; // output region, slot 1: [256, 512)
// Note: M (the 512-bit message block) lives at bits 512..1024 — directly
// after H_OUT, with no Z_CONST gap in the middle. This gives a clean 4-slot
// region of 1024 bits at the start of each block (slot 0 = H, slot 1 = H_OUT,
// slot 2 = M_lo, slot 3 = M_hi), so the Merkle-path protocol's
// `MerkleLayout` can address `(H_in, H_out, M_left, M_right)` by single-bit
// slot selectors. The Z_CONST constant-1 bit moved to the end of useful_bits
// (after the OUT_CARRY block), where it sits in a 1-bit gap that doesn't
// disturb the slot alignment.
pub const M_BASE: usize = 2 * SLOT_BITS; // 512
pub const CH_AND_BASE: usize = M_BASE + M_WORDS * WORD_BITS; // 1,024
pub const MAJ_AND_BASE: usize = CH_AND_BASE + N_ROUNDS * WORD_BITS; // 3,072
pub const ROUND_CARRY_BASE: usize = MAJ_AND_BASE + N_ROUNDS * WORD_BITS; // 5,120
pub const W_BASE: usize = ROUND_CARRY_BASE + N_ROUNDS * ADDS_PER_ROUND * CARRIES_PER_ADD; // 19,008
pub const SCHED_CARRY_BASE: usize = W_BASE + N_SCHED * WORD_BITS; // 20,544
pub const T1_BASE: usize = SCHED_CARRY_BASE + N_SCHED * ADDS_PER_SCHED * CARRIES_PER_ADD; // 25,008
pub const E_NEW_BASE: usize = T1_BASE + N_ROUNDS * WORD_BITS; // 27,056
pub const A_NEW_BASE: usize = E_NEW_BASE + N_ROUNDS * WORD_BITS; // 29,104
pub const OUT_CARRY_BASE: usize = A_NEW_BASE + N_ROUNDS * WORD_BITS; // 31,152
pub const Z_CONST_POS: usize = OUT_CARRY_BASE + N_OUT_WORDS * CARRIES_PER_ADD; // 31,400
pub const USEFUL_BITS: usize = Z_CONST_POS + 1; // 31,401

// Slot accessors.
#[inline]
pub fn h_bit(w: usize, b: usize) -> usize {
    H_BASE + WORD_BITS * w + b
}
#[inline]
pub fn m_bit(i: usize, b: usize) -> usize {
    M_BASE + WORD_BITS * i + b
}
#[inline]
pub fn ch_and_bit(r: usize, b: usize) -> usize {
    CH_AND_BASE + WORD_BITS * r + b
}
#[inline]
pub fn maj_and_bit(r: usize, b: usize) -> usize {
    MAJ_AND_BASE + WORD_BITS * r + b
}
#[inline]
pub fn round_carry_bit(r: usize, add: usize, b: usize) -> usize {
    ROUND_CARRY_BASE + r * ADDS_PER_ROUND * CARRIES_PER_ADD + add * CARRIES_PER_ADD + b
}
#[inline]
pub fn w_bit(t: usize, b: usize) -> usize {
    debug_assert!(t < N_SCHED + 16);
    if t < 16 {
        m_bit(t, b)
    } else {
        W_BASE + (t - 16) * WORD_BITS + b
    }
}
#[inline]
pub fn sched_carry_bit(t: usize, add: usize, b: usize) -> usize {
    debug_assert!((16..16 + N_SCHED).contains(&t));
    SCHED_CARRY_BASE + (t - 16) * ADDS_PER_SCHED * CARRIES_PER_ADD + add * CARRIES_PER_ADD + b
}
#[inline]
pub fn t1_bit(r: usize, b: usize) -> usize {
    T1_BASE + WORD_BITS * r + b
}
#[inline]
pub fn e_new_bit(r: usize, b: usize) -> usize {
    E_NEW_BASE + WORD_BITS * r + b
}
#[inline]
pub fn a_new_bit(r: usize, b: usize) -> usize {
    A_NEW_BASE + WORD_BITS * r + b
}
#[inline]
pub fn out_carry_bit(w: usize, b: usize) -> usize {
    OUT_CARRY_BASE + w * CARRIES_PER_ADD + b
}
#[inline]
pub fn h_out_bit(w: usize, b: usize) -> usize {
    H_OUT_BASE + w * WORD_BITS + b
}

// ───────────────────────────────────────────────────────────────────────────
// Symbolic XOR-support builder
// ───────────────────────────────────────────────────────────────────────────

/// Sorted-deduplicated XOR support — a row of `A` or `B` is one such Vec.
type Sup = Vec<usize>;
/// 32 per-bit supports = one 32-bit "word" in the symbolic computation.
type Word = Vec<Sup>;

fn zero_word() -> Word {
    (0..WORD_BITS).map(|_| Sup::new()).collect()
}

fn wire_word<F: Fn(usize) -> usize>(slot: F) -> Word {
    (0..WORD_BITS).map(|b| vec![slot(b)]).collect()
}

/// Symmetric difference of two sorted Vecs.
fn xor_sup(a: &Sup, b: &Sup) -> Sup {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn xor3(a: &Sup, b: &Sup, c: &Sup) -> Sup {
    xor_sup(&xor_sup(a, b), c)
}

fn xor_words(x: &Word, y: &Word) -> Word {
    (0..WORD_BITS).map(|i| xor_sup(&x[i], &y[i])).collect()
}

fn rotr(w: &Word, n: usize) -> Word {
    (0..WORD_BITS)
        .map(|i| w[(i + n) % WORD_BITS].clone())
        .collect()
}

fn shr(w: &Word, n: usize) -> Word {
    (0..WORD_BITS)
        .map(|i| {
            if i + n < WORD_BITS {
                w[i + n].clone()
            } else {
                Sup::new()
            }
        })
        .collect()
}

fn rot_xor3(w: &Word, r1: usize, r2: usize, r3: usize) -> Word {
    let a = rotr(w, r1);
    let b = rotr(w, r2);
    let c = rotr(w, r3);
    (0..WORD_BITS).map(|i| xor3(&a[i], &b[i], &c[i])).collect()
}

fn sigma_xor(w: &Word, r1: usize, r2: usize, sh: usize) -> Word {
    let a = rotr(w, r1);
    let b = rotr(w, r2);
    let s = shr(w, sh);
    (0..WORD_BITS).map(|i| xor3(&a[i], &b[i], &s[i])).collect()
}

#[inline]
fn sigma_0(w: &Word) -> Word {
    sigma_xor(w, 7, 18, 3)
}
#[inline]
fn sigma_1(w: &Word) -> Word {
    sigma_xor(w, 17, 19, 10)
}
#[inline]
fn big_sigma_0(w: &Word) -> Word {
    rot_xor3(w, 2, 13, 22)
}
#[inline]
fn big_sigma_1(w: &Word) -> Word {
    rot_xor3(w, 6, 11, 25)
}

/// 32-bit modular add `x + y`. Allocates 31 carry-aux AND rows via
/// `carry_slot(i)`; the carry chain is `cin[i+1] = cin[i] ⊕ carry_aux[i]`.
/// Returns the symbolic 32-bit sum (per-bit XOR support).
fn add32_inline<F: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F,
    a_rows: &mut [Sup],
    b_rows: &mut [Sup],
) -> Word {
    let mut sum = zero_word();
    let mut cin: Sup = Sup::new();
    for i in 0..WORD_BITS {
        sum[i] = xor3(&x[i], &y[i], &cin);
        if i < CARRIES_PER_ADD {
            let slot = carry_slot(i);
            a_rows[slot] = xor_sup(&x[i], &cin);
            b_rows[slot] = xor_sup(&y[i], &cin);
            cin = xor_sup(&cin, &vec![slot]);
        }
    }
    sum
}

/// Materialize a symbolic word at fresh slots: emit 32 rows
/// `(linear support) · z[Z_CONST] = z[slot]`, return a slot-word.
fn materialize<F: Fn(usize) -> usize>(
    raw: &Word,
    slot_fn: F,
    a_rows: &mut [Sup],
    b_rows: &mut [Sup],
) -> Word {
    let mut out = zero_word();
    for b in 0..WORD_BITS {
        let s = slot_fn(b);
        a_rows[s] = raw[b].clone();
        b_rows[s] = vec![Z_CONST_POS];
        out[b] = vec![s];
    }
    out
}

fn add32_alloc<F1: Fn(usize) -> usize, F2: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F1,
    sum_slot: F2,
    a_rows: &mut [Sup],
    b_rows: &mut [Sup],
) -> Word {
    let raw = add32_inline(x, y, carry_slot, a_rows, b_rows);
    materialize(&raw, sum_slot, a_rows, b_rows)
}

// ───────────────────────────────────────────────────────────────────────────
// Public matrix builder
// ───────────────────────────────────────────────────────────────────────────

/// Build `(A_0, B_0)` for one block of the hybrid SHA-256 R1CS. `C_0 = I`
/// (circuit shape); use [`build_block_r1cs`] to wrap these into a
/// [`BlockR1cs`].
pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Sup> = vec![Sup::new(); K];
    let mut b_rows: Vec<Sup> = vec![Sup::new(); K];

    // Z_CONST tautology: z[0]·z[0] = z[0] (boolean-pin).
    a_rows[Z_CONST_POS] = vec![Z_CONST_POS];
    b_rows[Z_CONST_POS] = vec![Z_CONST_POS];

    // H_in, M_in: free-witness rows.
    for w in 0..H_WORDS {
        for b in 0..WORD_BITS {
            let s = h_bit(w, b);
            a_rows[s] = vec![s];
            b_rows[s] = vec![Z_CONST_POS];
        }
    }
    for i in 0..M_WORDS {
        for b in 0..WORD_BITS {
            let s = m_bit(i, b);
            a_rows[s] = vec![s];
            b_rows[s] = vec![Z_CONST_POS];
        }
    }

    let h_in: Vec<Word> = (0..H_WORDS).map(|w| wire_word(|b| h_bit(w, b))).collect();
    let mut w_arr: Vec<Word> = (0..M_WORDS).map(|i| wire_word(|b| m_bit(i, b))).collect();

    // Message schedule (W[16..64]). Inline sched_0, sched_1; allocate W[t] = sched_2.
    for t in 16..(16 + N_SCHED) {
        let s1 = sigma_1(&w_arr[t - 2]);
        let s0 = sigma_0(&w_arr[t - 15]);
        let w_m7 = w_arr[t - 7].clone();
        let w_m16 = w_arr[t - 16].clone();
        let sched_0 = add32_inline(
            &s1,
            &w_m7,
            |i| sched_carry_bit(t, 0, i),
            &mut a_rows,
            &mut b_rows,
        );
        let sched_1 = add32_inline(
            &sched_0,
            &s0,
            |i| sched_carry_bit(t, 1, i),
            &mut a_rows,
            &mut b_rows,
        );
        let w_t = add32_alloc(
            &sched_1,
            &w_m16,
            |i| sched_carry_bit(t, 2, i),
            |b| w_bit(t, b),
            &mut a_rows,
            &mut b_rows,
        );
        w_arr.push(w_t);
    }

    // Working state (a, b, c, d, e, f, g, h).
    let mut state: [Word; 8] = [
        h_in[0].clone(),
        h_in[1].clone(),
        h_in[2].clone(),
        h_in[3].clone(),
        h_in[4].clone(),
        h_in[5].clone(),
        h_in[6].clone(),
        h_in[7].clone(),
    ];

    for r in 0..N_ROUNDS {
        let a = state[0].clone();
        let bb = state[1].clone();
        let c = state[2].clone();
        let d = state[3].clone();
        let e = state[4].clone();
        let f = state[5].clone();
        let g = state[6].clone();
        let h_var = state[7].clone();

        // ch_and[r][bit] = e[bit] · (f[bit] ⊕ g[bit])
        let mut ch_and = zero_word();
        for bit in 0..WORD_BITS {
            let s = ch_and_bit(r, bit);
            a_rows[s] = e[bit].clone();
            b_rows[s] = xor_sup(&f[bit], &g[bit]);
            ch_and[bit] = vec![s];
        }
        // maj_and[r][bit] = (a[bit] ⊕ b[bit]) · (a[bit] ⊕ c[bit])
        let mut maj_and = zero_word();
        for bit in 0..WORD_BITS {
            let s = maj_and_bit(r, bit);
            a_rows[s] = xor_sup(&a[bit], &bb[bit]);
            b_rows[s] = xor_sup(&a[bit], &c[bit]);
            maj_and[bit] = vec![s];
        }
        let ch_out = xor_words(&ch_and, &g); // Ch = e·(f⊕g) ⊕ g
        let maj_out = xor_words(&maj_and, &a); // Maj = (a⊕b)·(a⊕c) ⊕ a

        // T1 chain: inline T1_0..T1_2, allocate T1.
        let t1_0 = add32_inline(
            &h_var,
            &big_sigma_1(&e),
            |i| round_carry_bit(r, 0, i),
            &mut a_rows,
            &mut b_rows,
        );
        let t1_1 = add32_inline(
            &t1_0,
            &ch_out,
            |i| round_carry_bit(r, 1, i),
            &mut a_rows,
            &mut b_rows,
        );
        let k_word: Word = (0..WORD_BITS)
            .map(|i| {
                if (SHA256_K[r] >> i) & 1 == 1 {
                    vec![Z_CONST_POS]
                } else {
                    Sup::new()
                }
            })
            .collect();
        let t1_2 = add32_inline(
            &t1_1,
            &k_word,
            |i| round_carry_bit(r, 2, i),
            &mut a_rows,
            &mut b_rows,
        );
        let t1 = add32_alloc(
            &t1_2,
            &w_arr[r],
            |i| round_carry_bit(r, 3, i),
            |b| t1_bit(r, b),
            &mut a_rows,
            &mut b_rows,
        );
        // T2 inlined; E_NEW, A_NEW allocated.
        let t2 = add32_inline(
            &big_sigma_0(&a),
            &maj_out,
            |i| round_carry_bit(r, 4, i),
            &mut a_rows,
            &mut b_rows,
        );
        let e_new = add32_alloc(
            &d,
            &t1,
            |i| round_carry_bit(r, 5, i),
            |b| e_new_bit(r, b),
            &mut a_rows,
            &mut b_rows,
        );
        let a_new = add32_alloc(
            &t1,
            &t2,
            |i| round_carry_bit(r, 6, i),
            |b| a_new_bit(r, b),
            &mut a_rows,
            &mut b_rows,
        );

        // Register shift: (a', b', c', d', e', f', g', h') = (A_NEW, a, b, c, E_NEW, e, f, g)
        state = [a_new, a, bb, c, e_new, e, f, g];
    }

    // Output feed-forward: H_out[w] = state[w] + H_in[w].
    for w in 0..N_OUT_WORDS {
        let _ = add32_alloc(
            &state[w],
            &h_in[w],
            |i| out_carry_bit(w, i),
            |b| h_out_bit(w, b),
            &mut a_rows,
            &mut b_rows,
        );
    }

    let to_mat = |rows| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_mat(a_rows), to_mat(b_rows))
}

// ───────────────────────────────────────────────────────────────────────────
// Witness generator
// ───────────────────────────────────────────────────────────────────────────

fn write_word(z: &mut [bool], base: usize, v: u32) {
    for b in 0..WORD_BITS {
        z[base + b] = (v >> b) & 1 == 1;
    }
}

/// 32-bit add with carry-aux output. `cin[i+1] = cin[i] ⊕ carry_aux[i]`.
fn add32_w(x: u32, y: u32, carry_base: usize, z: &mut [bool]) -> u32 {
    let mut cin: bool = false;
    for i in 0..CARRIES_PER_ADD {
        let xi = ((x >> i) & 1) == 1;
        let yi = ((y >> i) & 1) == 1;
        let aux = (xi ^ cin) && (yi ^ cin);
        z[carry_base + i] = aux;
        cin ^= aux;
    }
    x.wrapping_add(y)
}

/// Build the per-block boolean witness for one SHA-256 compression
/// `f(h_in, m) → H_out`. Length = `K = 2^15`. Slot positions [USEFUL_BITS, K)
/// are zero-padded.
pub fn build_block_witness(h_in: &[u32; 8], m: &[u32; 16]) -> Vec<bool> {
    let mut z = vec![false; K];
    z[Z_CONST_POS] = true;

    for w in 0..H_WORDS {
        write_word(&mut z, h_bit(w, 0), h_in[w]);
    }
    for i in 0..M_WORDS {
        write_word(&mut z, m_bit(i, 0), m[i]);
    }

    // Schedule W[16..64].
    let mut w_arr = [0u32; 64];
    w_arr[..16].copy_from_slice(m);
    for t in 16..64 {
        let s0 =
            w_arr[t - 15].rotate_right(7) ^ w_arr[t - 15].rotate_right(18) ^ (w_arr[t - 15] >> 3);
        let s1 =
            w_arr[t - 2].rotate_right(17) ^ w_arr[t - 2].rotate_right(19) ^ (w_arr[t - 2] >> 10);
        let sched_0 = add32_w(s1, w_arr[t - 7], sched_carry_bit(t, 0, 0), &mut z);
        let sched_1 = add32_w(sched_0, s0, sched_carry_bit(t, 1, 0), &mut z);
        let w_t = add32_w(sched_1, w_arr[t - 16], sched_carry_bit(t, 2, 0), &mut z);
        write_word(&mut z, w_bit(t, 0), w_t);
        w_arr[t] = w_t;
    }

    // Rounds.
    let mut state = *h_in;
    for r in 0..N_ROUNDS {
        let (a, b, c, d, e, f, g, h_var) = (
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        );
        let ch_and = e & (f ^ g);
        write_word(&mut z, ch_and_bit(r, 0), ch_and);
        let maj_and = (a ^ b) & (a ^ c);
        write_word(&mut z, maj_and_bit(r, 0), maj_and);

        let ch_out = ch_and ^ g;
        let maj_out = maj_and ^ a;
        let s1e = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let s0a = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);

        let t1_0 = add32_w(h_var, s1e, round_carry_bit(r, 0, 0), &mut z);
        let t1_1 = add32_w(t1_0, ch_out, round_carry_bit(r, 1, 0), &mut z);
        let t1_2 = add32_w(t1_1, SHA256_K[r], round_carry_bit(r, 2, 0), &mut z);
        let t1 = add32_w(t1_2, w_arr[r], round_carry_bit(r, 3, 0), &mut z);
        write_word(&mut z, t1_bit(r, 0), t1);

        let t2 = add32_w(s0a, maj_out, round_carry_bit(r, 4, 0), &mut z);
        let e_new = add32_w(d, t1, round_carry_bit(r, 5, 0), &mut z);
        write_word(&mut z, e_new_bit(r, 0), e_new);
        let a_new = add32_w(t1, t2, round_carry_bit(r, 6, 0), &mut z);
        write_word(&mut z, a_new_bit(r, 0), a_new);

        state = [a_new, a, b, c, e_new, e, f, g];
    }

    // Output feed-forward.
    for w in 0..N_OUT_WORDS {
        let h_out = add32_w(state[w], h_in[w], out_carry_bit(w, 0), &mut z);
        write_word(&mut z, h_out_bit(w, 0), h_out);
    }
    z
}

/// Read the 8-word post-compression hash out of a single block of witness.
pub fn read_h_out(z: &[bool]) -> [u32; 8] {
    std::array::from_fn(|w| {
        (0..WORD_BITS).fold(0u32, |acc, b| acc | ((z[h_out_bit(w, b)] as u32) << b))
    })
}

// ───────────────────────────────────────────────────────────────────────────
// BlockR1cs constructor
// ───────────────────────────────────────────────────────────────────────────

/// Build a [`BlockR1cs`] for `2^n_blocks_log` SHA-256 compressions batched
/// block-diagonally (one compression per block). `n_blocks_log ≥ 3` is the
/// lincheck floor.
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

// ───────────────────────────────────────────────────────────────────────────
// Lincheck circuit walker — mirrors `build_matrices`. Same structure as
// `sha2::Sha2LincheckCircuit` but uses this module's I/O-aligned slot
// positions.
// ───────────────────────────────────────────────────────────────────────────

fn scatter_add32_inline<F: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F,
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
) -> Word {
    let mut sum = zero_word();
    let mut cin: Sup = Sup::new();
    for i in 0..WORD_BITS {
        sum[i] = xor3(&x[i], &y[i], &cin);
        if i < CARRIES_PER_ADD {
            let slot = carry_slot(i);
            let row = slot;
            let e = eq_inner[row];
            let ea = alpha * e;
            for &c in xor_sup(&x[i], &cin).iter() {
                comb[c] += ea;
            }
            for &c in xor_sup(&y[i], &cin).iter() {
                comb[c] += e;
            }
            cin = xor_sup(&cin, &vec![slot]);
        }
    }
    sum
}

fn scatter_materialize<F: Fn(usize) -> usize>(
    raw: &Word,
    slot_fn: F,
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
) -> Word {
    let mut out = zero_word();
    for b in 0..WORD_BITS {
        let s = slot_fn(b);
        let e = eq_inner[s];
        let ea = alpha * e;
        for &c in raw[b].iter() {
            comb[c] += ea;
        }
        comb[Z_CONST_POS] += e;
        out[b] = vec![s];
    }
    out
}

fn scatter_add32_alloc<F1: Fn(usize) -> usize, F2: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F1,
    sum_slot: F2,
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
) -> Word {
    let raw = scatter_add32_inline(x, y, carry_slot, comb, alpha, eq_inner);
    scatter_materialize(&raw, sum_slot, comb, alpha, eq_inner)
}

pub struct Sha2LincheckCircuit;

impl flock_core::lincheck::LincheckCircuit for Sha2LincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
        let mut comb = vec![F128::ZERO; K];

        let e0 = eq_inner[Z_CONST_POS];
        comb[Z_CONST_POS] += alpha * e0;
        comb[Z_CONST_POS] += e0;

        for w in 0..H_WORDS {
            for b in 0..WORD_BITS {
                let s = h_bit(w, b);
                let e = eq_inner[s];
                comb[s] += alpha * e;
                comb[Z_CONST_POS] += e;
            }
        }
        for i in 0..M_WORDS {
            for b in 0..WORD_BITS {
                let s = m_bit(i, b);
                let e = eq_inner[s];
                comb[s] += alpha * e;
                comb[Z_CONST_POS] += e;
            }
        }

        let h_in: Vec<Word> = (0..H_WORDS).map(|w| wire_word(|b| h_bit(w, b))).collect();
        let mut w_arr: Vec<Word> = (0..M_WORDS).map(|i| wire_word(|b| m_bit(i, b))).collect();

        for t in 16..(16 + N_SCHED) {
            let s1 = sigma_1(&w_arr[t - 2]);
            let s0 = sigma_0(&w_arr[t - 15]);
            let w_m7 = w_arr[t - 7].clone();
            let w_m16 = w_arr[t - 16].clone();
            let sched_0 = scatter_add32_inline(
                &s1,
                &w_m7,
                |i| sched_carry_bit(t, 0, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let sched_1 = scatter_add32_inline(
                &sched_0,
                &s0,
                |i| sched_carry_bit(t, 1, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let w_t = scatter_add32_alloc(
                &sched_1,
                &w_m16,
                |i| sched_carry_bit(t, 2, i),
                |b| w_bit(t, b),
                &mut comb,
                alpha,
                eq_inner,
            );
            w_arr.push(w_t);
        }

        let mut state: [Word; 8] = [
            h_in[0].clone(),
            h_in[1].clone(),
            h_in[2].clone(),
            h_in[3].clone(),
            h_in[4].clone(),
            h_in[5].clone(),
            h_in[6].clone(),
            h_in[7].clone(),
        ];

        for r in 0..N_ROUNDS {
            let a = state[0].clone();
            let bb = state[1].clone();
            let c = state[2].clone();
            let d = state[3].clone();
            let e = state[4].clone();
            let f = state[5].clone();
            let g = state[6].clone();
            let h_var = state[7].clone();

            let mut ch_and = zero_word();
            for bit in 0..WORD_BITS {
                let s = ch_and_bit(r, bit);
                let eq = eq_inner[s];
                let ea = alpha * eq;
                for &c2 in e[bit].iter() {
                    comb[c2] += ea;
                }
                for &c2 in xor_sup(&f[bit], &g[bit]).iter() {
                    comb[c2] += eq;
                }
                ch_and[bit] = vec![s];
            }
            let mut maj_and = zero_word();
            for bit in 0..WORD_BITS {
                let s = maj_and_bit(r, bit);
                let eq = eq_inner[s];
                let ea = alpha * eq;
                for &c2 in xor_sup(&a[bit], &bb[bit]).iter() {
                    comb[c2] += ea;
                }
                for &c2 in xor_sup(&a[bit], &c[bit]).iter() {
                    comb[c2] += eq;
                }
                maj_and[bit] = vec![s];
            }
            let ch_out = xor_words(&ch_and, &g);
            let maj_out = xor_words(&maj_and, &a);

            let t1_0 = scatter_add32_inline(
                &h_var,
                &big_sigma_1(&e),
                |i| round_carry_bit(r, 0, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let t1_1 = scatter_add32_inline(
                &t1_0,
                &ch_out,
                |i| round_carry_bit(r, 1, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let k_word: Word = (0..WORD_BITS)
                .map(|i| {
                    if (SHA256_K[r] >> i) & 1 == 1 {
                        vec![Z_CONST_POS]
                    } else {
                        Sup::new()
                    }
                })
                .collect();
            let t1_2 = scatter_add32_inline(
                &t1_1,
                &k_word,
                |i| round_carry_bit(r, 2, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let t1 = scatter_add32_alloc(
                &t1_2,
                &w_arr[r],
                |i| round_carry_bit(r, 3, i),
                |b| t1_bit(r, b),
                &mut comb,
                alpha,
                eq_inner,
            );
            let t2 = scatter_add32_inline(
                &big_sigma_0(&a),
                &maj_out,
                |i| round_carry_bit(r, 4, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let e_new = scatter_add32_alloc(
                &d,
                &t1,
                |i| round_carry_bit(r, 5, i),
                |b| e_new_bit(r, b),
                &mut comb,
                alpha,
                eq_inner,
            );
            let a_new = scatter_add32_alloc(
                &t1,
                &t2,
                |i| round_carry_bit(r, 6, i),
                |b| a_new_bit(r, b),
                &mut comb,
                alpha,
                eq_inner,
            );

            state = [a_new, a, bb, c, e_new, e, f, g];
        }

        for w in 0..N_OUT_WORDS {
            let _ = scatter_add32_alloc(
                &state[w],
                &h_in[w],
                |i| out_carry_bit(w, i),
                |b| h_out_bit(w, b),
                &mut comb,
                alpha,
                eq_inner,
            );
        }

        comb
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Fast-path: fused (z, a, b, z_lincheck) packed witness builder.
//
// Each adder writes its carry-aux rows always; the sum row is written only
// for *slotted* adders (W[t], T1, E_NEW, A_NEW, H_out).
//
// Witness-value insight: at a carry-aux slot, `a` and `b` are the *scalar
// evaluations* of the row's linear A/B supports — `(x[i] ⊕ cin[i])` is the
// same bit value regardless of how many slots the A-row carries.
// ───────────────────────────────────────────────────────────────────────────

// ───────────────────────────────────────────────────────────────────────────
// SHA-256 reference helpers (used by witness gen).
// ───────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
#[inline]
pub(crate) fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
#[inline]
pub(crate) fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
#[inline]
pub(crate) fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// 32-bit add `x + y`. Writes 31 carry-aux rows at `carry_base..+31` with
/// `(z, a, b) = (aux, left, right)` where `aux = left & right`,
/// `left = (x ⊕ cin) & 0x7FFFFFFF`, `right = (y ⊕ cin) & 0x7FFFFFFF`. Top
/// carry bit is masked so the unallocated 32nd slot isn't touched.
///
/// **No c buffer.** C = I, so c == z byte-for-byte; callers wrap z_packed
/// as the c-side input to zerocheck.
#[inline(always)]
fn add_inline_ab(
    x: u32,
    y: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
    carry_base: usize,
) -> u32 {
    let sum_word: u32 = x.wrapping_add(y);
    let cin: u32 = sum_word ^ x ^ y;
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let left = (x ^ cin) & MASK_LO31;
    let right = (y ^ cin) & MASK_LO31;
    let carry_aux = left & right;
    or_u32_at_bit(z, carry_base, carry_aux);
    or_u32_at_bit(a, carry_base, left);
    or_u32_at_bit(b, carry_base, right);
    sum_word
}

/// 32-bit add that ALSO materializes the sum bits at `sum_base..+32` with
/// `(z, a, b) = (sum, sum, 1)`. c == z by aliasing.
#[inline(always)]
fn add_alloc_ab(
    x: u32,
    y: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
    sum_base: usize,
    carry_base: usize,
) -> u32 {
    let sum = add_inline_ab(x, y, z, a, b, carry_base);
    or_u32_at_bit(z, sum_base, sum);
    or_u32_at_bit(a, sum_base, sum);
    or_u32_at_bit(b, sum_base, 0xFFFF_FFFF);
    sum
}

/// Build the (z, a, b) packed buffers for ONE SHA-256 compression into the
/// u64 views (one block worth: `K / 64` u64s each). Buffers must be zero on
/// entry. **No c buffer** (c == z byte-for-byte since C = I).
fn build_block_ab_packed_into(
    h_in: &[u32; 8],
    m: &[u32; 16],
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    const U64_PER_BLOCK: usize = K / 64;
    debug_assert_eq!(z.len(), U64_PER_BLOCK);
    debug_assert_eq!(a.len(), U64_PER_BLOCK);
    debug_assert_eq!(b.len(), U64_PER_BLOCK);

    // Z_CONST: (z, a, b) = (1, 1, 1).
    or_bit_at(z, Z_CONST_POS);
    or_bit_at(a, Z_CONST_POS);
    or_bit_at(b, Z_CONST_POS);

    // H_in, M: free-witness tautologies → (z, a, b) = (v, v, 1).
    for w in 0..H_WORDS {
        let off = h_bit(w, 0);
        let v = h_in[w];
        or_u32_at_bit(z, off, v);
        or_u32_at_bit(a, off, v);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
    }
    for i in 0..M_WORDS {
        let off = m_bit(i, 0);
        let v = m[i];
        or_u32_at_bit(z, off, v);
        or_u32_at_bit(a, off, v);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
    }

    // Message schedule. sched_0, sched_1 inlined; W[t] = sched_2 allocated.
    // The 3 × 31-bit sched carries per t are contiguous (93 bits at stride
    // 93) — composed in a register record and flushed once per buffer (see
    // [`BitRecord`]).
    let mut w_sched = [0u32; 64];
    w_sched[..16].copy_from_slice(m);
    const SC0: usize = 0;
    const SC1: usize = CARRIES_PER_ADD;
    const SC2: usize = 2 * CARRIES_PER_ADD;
    for t in 16..64 {
        let mut rz = BitRecord::<2>::new();
        let mut ra = BitRecord::<2>::new();
        let mut rb = BitRecord::<2>::new();

        macro_rules! add_into {
            ($pos:ident, $x:expr, $y:expr) => {{
                let (sum, left, right, carry) = add_carry_parts($x, $y);
                rz.push::<$pos>(carry);
                ra.push::<$pos>(left);
                rb.push::<$pos>(right);
                sum
            }};
        }

        let s_0 = add_into!(SC0, small_sigma1(w_sched[t - 2]), w_sched[t - 7]);
        let s_1 = add_into!(SC1, s_0, small_sigma0(w_sched[t - 15]));
        let w_t = add_into!(SC2, s_1, w_sched[t - 16]);

        let sched_base = sched_carry_bit(t, 0, 0);
        rz.flush(z, sched_base);
        ra.flush(a, sched_base);
        rb.flush(b, sched_base);

        // W[t] sum row: (z, a, b) = (w_t, w_t, 1).
        let off = w_bit(t, 0);
        or_u32_at_bit(z, off, w_t);
        or_u32_at_bit(a, off, w_t);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        w_sched[t] = w_t;
    }

    // 64 rounds.
    let [
        mut aa,
        mut bb,
        mut cc,
        mut dd,
        mut ee,
        mut ff,
        mut gg,
        mut hh,
    ] = *h_in;
    for r in 0..N_ROUNDS {
        // ch_and AND row: (z, a, b) = (ch, e, f⊕g); c == z = ch.
        let f_xor_g = ff ^ gg;
        let ch_and_v = ee & f_xor_g;
        let off = ch_and_bit(r, 0);
        or_u32_at_bit(z, off, ch_and_v);
        or_u32_at_bit(a, off, ee);
        or_u32_at_bit(b, off, f_xor_g);
        let ch_out = ch_and_v ^ gg;

        // maj_and AND row.
        let b_xor_a = bb ^ aa;
        let c_xor_a = cc ^ aa;
        let maj_and_v = b_xor_a & c_xor_a;
        let off = maj_and_bit(r, 0);
        or_u32_at_bit(z, off, maj_and_v);
        or_u32_at_bit(a, off, b_xor_a);
        or_u32_at_bit(b, off, c_xor_a);
        let maj_out = maj_and_v ^ aa;

        // The 7 × 31-bit round carries are contiguous (217 bits at stride
        // 217) — composed in a register record and flushed once per buffer.
        const RC0: usize = 0;
        const RC1: usize = CARRIES_PER_ADD;
        const RC2: usize = 2 * CARRIES_PER_ADD;
        const RC3: usize = 3 * CARRIES_PER_ADD;
        const RC4: usize = 4 * CARRIES_PER_ADD;
        const RC5: usize = 5 * CARRIES_PER_ADD;
        const RC6: usize = 6 * CARRIES_PER_ADD;
        let mut rz = BitRecord::<4>::new();
        let mut ra = BitRecord::<4>::new();
        let mut rb = BitRecord::<4>::new();

        macro_rules! add_into {
            ($pos:ident, $x:expr, $y:expr) => {{
                let (sum, left, right, carry) = add_carry_parts($x, $y);
                rz.push::<$pos>(carry);
                ra.push::<$pos>(left);
                rb.push::<$pos>(right);
                sum
            }};
        }

        // T1 chain: T1_0..T1_2 inlined, T1 (= T1_3) allocated.
        let t1_0 = add_into!(RC0, hh, big_sigma1(ee));
        let t1_1 = add_into!(RC1, t1_0, ch_out);
        let t1_2 = add_into!(RC2, t1_1, SHA256_K[r]);
        let t1 = add_into!(RC3, t1_2, w_sched[r]);
        let off = t1_bit(r, 0);
        or_u32_at_bit(z, off, t1);
        or_u32_at_bit(a, off, t1);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        // T2 inlined.
        let t2 = add_into!(RC4, big_sigma0(aa), maj_out);
        // E_NEW, A_NEW allocated.
        let e_new = add_into!(RC5, dd, t1);
        let off = e_new_bit(r, 0);
        or_u32_at_bit(z, off, e_new);
        or_u32_at_bit(a, off, e_new);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
        let a_new = add_into!(RC6, t1, t2);
        let off = a_new_bit(r, 0);
        or_u32_at_bit(z, off, a_new);
        or_u32_at_bit(a, off, a_new);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        let round_base = round_carry_bit(r, 0, 0);
        rz.flush(z, round_base);
        ra.flush(a, round_base);
        rb.flush(b, round_base);

        // Register shift.
        hh = gg;
        gg = ff;
        ff = ee;
        ee = e_new;
        dd = cc;
        cc = bb;
        bb = aa;
        aa = a_new;
    }

    // Output feed-forward.
    let final_state = [aa, bb, cc, dd, ee, ff, gg, hh];
    for w in 0..N_OUT_WORDS {
        add_alloc_ab(
            final_state[w],
            h_in[w],
            z,
            a,
            b,
            h_out_bit(w, 0),
            out_carry_bit(w, 0),
        );
    }
}

/// Like [`generate_witness`] but produces F128-packed `(z, a, b, c)` AND the
/// lincheck byte-stripe in one fused parallel pass. Replaces
/// `pack_witness` + `apply_{a,b,c}_packed` + `pack_z_lincheck_from_packed`.
///
/// 8 k-blocks per parallel task (matching the lincheck stripe granularity).
pub fn generate_witness_with_ab_packed_and_lincheck(
    compressions: &[([u32; 8], [u32; 16])],
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
    let padding: ([u32; 8], [u32; 16]) = ([0u32; 8], [0u32; 16]);
    super::common::drive_witness_packed_and_lincheck(
        compressions,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        |comp: &([u32; 8], [u32; 16]), z_u64, a_u64, b_u64| {
            let (h_in, m) = comp;
            build_block_ab_packed_into(h_in, m, z_u64, a_u64, b_u64);
        },
    )
}

// ───────────────────────────────────────────────────────────────────────────
// Multi-block witness gen + Setup
// ───────────────────────────────────────────────────────────────────────────

/// Minimum `n_blocks_log` to fit `n_compressions` (one compression per
/// k-block), subject to the lincheck floor of `n_blocks_log ≥ 3`.
pub fn min_n_blocks_log(n_compressions: usize) -> usize {
    assert!(n_compressions >= 1);
    let n = n_compressions.max(8);
    n.next_power_of_two().trailing_zeros() as usize
}

/// Build the boolean witness across `2^n_blocks_log` blocks, one compression
/// per block. Parallelized via rayon.
pub fn generate_witness(compressions: &[([u32; 8], [u32; 16])], n_blocks_log: usize) -> Vec<bool> {
    use rayon::prelude::*;
    let n_total_blocks = 1usize << n_blocks_log;
    assert!(compressions.len() <= n_total_blocks);

    let mut z = vec![false; n_total_blocks * K];
    z.par_chunks_mut(K)
        .enumerate()
        .for_each(|(block_idx, chunk)| {
            if block_idx >= compressions.len() {
                return; // padding k-block (all zeros)
            }
            let (h_in, m) = &compressions[block_idx];
            let block_witness = build_block_witness(h_in, m);
            chunk.copy_from_slice(&block_witness);
        });
    z
}

#[derive(Clone, Debug)]
pub struct Sha256HybridSetup {
    pub n_compressions: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: flock_core::pcs::PcsParams,
}

impl Sha256HybridSetup {
    pub fn new(n_compressions: usize) -> Self {
        Self::with_log_inv_rate(n_compressions, 1)
    }

    /// [`Self::new`] with the **batch-major** witness layout (see
    /// [`flock_core::r1cs::WitnessLayout`]). The generic matrix provers and
    /// chain/Merkle wrappers still require row-major.
    pub fn new_batch_major(n_compressions: usize) -> Self {
        let mut s = Self::new(n_compressions);
        s.r1cs.layout = flock_core::r1cs::WitnessLayout::BatchMajor;
        s
    }

    /// Fast-path witness generation dispatched on the r1cs's witness layout.
    fn generate_witness_ab(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
    ) -> (
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<u8>,
    ) {
        match self.r1cs.layout {
            flock_core::r1cs::WitnessLayout::RowMajor => {
                generate_witness_with_ab_packed_and_lincheck(compressions, self.n_blocks_log())
            }
            flock_core::r1cs::WitnessLayout::BatchMajor => {
                generate_witness_batch_major(compressions, self.n_blocks_log())
            }
        }
    }

    pub fn with_log_inv_rate(n_compressions: usize, log_inv_rate: usize) -> Self {
        // Rate keys the legacy profiles: 1 -> Fast, 2 -> Slim.
        let profile = match log_inv_rate {
            1 => flock_core::pcs::ligerito::LigeritoProfile::Fast,
            2 => flock_core::pcs::ligerito::LigeritoProfile::Slim,
            _ => flock_core::pcs::ligerito::LigeritoProfile::Fast, // BaseFold-only rates
        };
        Self::with_profile_and_rate(n_compressions, profile, log_inv_rate)
    }

    /// Build a setup for a named Ligerito profile (fast/slim/secure);
    /// the PCS rate follows the profile.
    pub fn with_profile(
        n_compressions: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        Self::with_profile_and_rate(n_compressions, profile, profile.log_inv_rate())
    }

    fn with_profile_and_rate(
        n_compressions: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
        log_inv_rate: usize,
    ) -> Self {
        assert!(n_compressions >= 1, "n_compressions must be ≥ 1");
        let n_log = min_n_blocks_log(n_compressions);
        let r1cs = build_block_r1cs(n_log);
        // Warm the CSC fold circuit so its one-time build stays out of the
        // first prove/verify, and pre-fault the prove-cycle scratch buffers
        // so even the first prove performs no page faults.
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let pcs_params = flock_core::pcs::PcsParams {
            m: r1cs.m,
            log_inv_rate,
            log_batch_size: 6,
            profile,
        };
        Self {
            n_compressions,
            r1cs,
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

    pub fn generate_witness(&self, compressions: &[([u32; 8], [u32; 16])]) -> Vec<bool> {
        assert_eq!(compressions.len(), self.n_compressions);
        generate_witness(compressions, self.n_blocks_log())
    }

    /// Slow-path prover: builds the boolean witness, packs it, and calls the
    /// generic [`crate::prover::prove`] (which materializes `a, b, c` via
    /// `apply_*_packed`). Use this for correctness verification.
    ///
    /// A `prove_fast` that fuses (z, a, b, c, z_lincheck) construction
    /// directly is a follow-up port.
    pub fn prove<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProof,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        let z_packed = self.generate_witness_packed(compressions);
        crate::prover::prove(&self.r1cs, &z_packed, &self.pcs_params, challenger)
    }

    /// Packed witness trace for the generic (matrix-driven) provers — the
    /// per-circuit code they need. Implemented by reusing the fused builder
    /// (its a/b outputs are discarded): no separate packed-trace writer to
    /// maintain, and ~5× cheaper than the bool trace → `pack_witness` path.
    pub fn generate_witness_packed(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
    ) -> Vec<flock_core::field::F128> {
        let (z_packed, _a, _b, _stripe) =
            self.generate_witness_ab(compressions);
        z_packed
    }

    /// Generic (matrix-driven) prover on the **Ligerito** backend — the
    /// counterpart of [`Self::prove`] (BaseFold). Same witness path
    /// (bool trace → pack); produces a proof byte-identical to
    /// [`Self::prove_fast`] and verifiable with [`Self::verify`].
    pub fn prove_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        let z_packed = self.generate_witness_packed(compressions);
        crate::prover::prove_ligerito(&self.r1cs, z_packed, &self.pcs_params, challenger)
    }

    /// Fast prover: skips `pack_witness`, `apply_{a,b,c}_packed`, and
    /// `pack_z_lincheck_from_packed` by emitting `(z, a, b, c, z_lincheck)`
    /// directly via [`generate_witness_with_abc_packed_and_lincheck`].
    pub fn prove_fast_basefold<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProof,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            self.generate_witness_ab(compressions);
        crate::prover::prove_fast_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    pub fn verify_basefold<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProof,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::VerifyError> {
        flock_core::verifier::verify(
            &self.r1cs,
            commitment,
            proof,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Ligerito-backend mirror of [`Self::prove_fast`]. Requires m ≥ ~21.
    pub fn prove_fast<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        // Pre-fault the commit codeword buffer on a background (E-core) thread
        // while witness generation runs on the perf cores; gated so
        // RAYON_NUM_THREADS=1 stays truly serial (no extra thread).
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                self.generate_witness_ab(compressions)
            });
        crate::prover::prove_fast_ligerito_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            codeword,
            challenger,
        )
    }

    /// [`Self::prove_fast`] with a per-phase timing breakdown of the real
    /// Ligerito prover (witness gen + commit + zerocheck + lincheck + recursive
    /// open). Benchmark-only.
    pub fn prove_fast_timed<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        let t0 = std::time::Instant::now();
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            self.generate_witness_ab(compressions);
        let witness_s = t0.elapsed().as_secs_f64();
        let lc_circuit = self.r1cs.csc_lincheck_circuit();
        let (proof, commitment, claim, mut timings) = crate::prover::prove_fast_ligerito_timed(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            lc_circuit,
            None,
            challenger,
        );
        timings.witness_s = witness_s;
        (proof, commitment, claim, timings)
    }

    pub fn verify<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::VerifyError> {
        flock_core::verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Hash chain: SHA-256 geometry + thin wrappers over the generic chain core.
// ───────────────────────────────────────────────────────────────────────────

pub use super::chain_common::{ChainFold, ChainProof, ChainVerifyError};

/// One SHA-256 compression input: `(H_in, M)` — the 8-word input chaining
/// value plus the 16-word message block. Mirrors the [`Sha256HybridSetup`]
/// witness-gen tuple type.
pub type Compression = ([u32; 8], [u32; 16]);

/// SHA-256's I/O-region geometry for the generic chain core. The input
/// chaining value `H_in` sits in aligned slot 0 (byte 0), the output chaining
/// value `H_out` in slot 1 (byte 32); each region is exactly 256 bits in a
/// 256-bit (`region_log = 8`) slot — no interior padding. Within a slot the
/// layout is word-contiguous (8 × 32-bit words), and since the low
/// `K_SKIP = 6` physical bits are the φ8 z-skip block, the fold weight matches
/// the generic `phys_weights[p] = λ[p & 63]·eq(r_rest, p >> 6)`.
pub const CHAIN_LAYOUT: super::chain_common::ChainLayout = super::chain_common::ChainLayout {
    k_log: K_LOG,
    k_skip: K_SKIP,
    region_log: 8,                   // SLOT_BITS = 2^8 = 256
    region_bits: 256,                // 8 words × 32 bits, fills the slot exactly
    input_byte_off: H_BASE / 8,      // 0
    output_byte_off: H_OUT_BASE / 8, // 32
};

/// Convert a public 256-bit chaining value (8 × u32 words, LE bit order within
/// each word) to the region's **physical** within-slot bool layout. The region
/// is word-contiguous: physical bit `32·w + b` holds bit `b` of word `w`.
pub fn cv_to_phys_bits(cv: &[u32; 8]) -> Vec<bool> {
    let mut phys = vec![false; 256];
    for w in 0..8 {
        for b in 0..WORD_BITS {
            phys[WORD_BITS * w + b] = (cv[w] >> b) & 1 == 1;
        }
    }
    phys
}

/// Reference SHA-256 compression. Returns the 8-word output chaining value
/// `H_out = H_in + state` where `state = compress256(M)` is the post-rounds
/// register state. Public so the chain caller can build honest chains via
/// `blocks[i+1].0 = sha256_compress(blocks[i])`.
pub fn sha256_compress(h_in: &[u32; 8], m: &[u32; 16]) -> [u32; 8] {
    let mut w = [0u32; 64];
    w[..16].copy_from_slice(m);
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = s1
            .wrapping_add(w[t - 7])
            .wrapping_add(s0)
            .wrapping_add(w[t - 16]);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = [
        h_in[0], h_in[1], h_in[2], h_in[3], h_in[4], h_in[5], h_in[6], h_in[7],
    ];
    for r in 0..N_ROUNDS {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[r])
            .wrapping_add(w[r]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    [
        h_in[0].wrapping_add(a),
        h_in[1].wrapping_add(b),
        h_in[2].wrapping_add(c),
        h_in[3].wrapping_add(d),
        h_in[4].wrapping_add(e),
        h_in[5].wrapping_add(f),
        h_in[6].wrapping_add(g),
        h_in[7].wrapping_add(h),
    ]
}

impl Sha256HybridSetup {
    /// Prove that the committed compressions form a sequential chaining-value
    /// chain: for each instance `i`, the output CV (`H_out`) equals the input
    /// CV (`H_in`) of instance `i+1`, with public endpoints `cv_0` (first
    /// input) and `cv_last` (last output).
    ///
    /// The prover is **given the full sequence** of `Compression`s (one per
    /// instance) so trace-gen is parallel; for an honest chain the caller sets
    /// `blocks[i+1].0 = sha256_compress(&blocks[i].0, &blocks[i].1)`.
    pub fn prove_chain_basefold<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[Compression],
        challenger: &mut Ch,
    ) -> (ChainProof, flock_core::pcs::Commitment) {
        assert_eq!(compressions.len(), self.n_compressions);
        // The chain shift sumcheck enforces the relation across ALL witness
        // slots, including padding. If n_compressions < n_block_slots, padding
        // blocks (all-zero) break the chain at the boundary and the proof
        // cannot verify with the user's intended endpoints. Require an exact
        // fit (n_compressions a power of 2 ≥ 8, the lincheck floor).
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_chain requires n_compressions to exactly fill n_block_slots \
             (no padding); got n_compressions={}, n_block_slots={}. Use a \
             power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        let n_log = self.n_blocks_log();
        let (z_packed, a_packed, b_packed, z_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(compressions, n_log);
        super::chain_common::prove_chain_generic(
            &self.r1cs,
            &self.pcs_params,
            &CHAIN_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Ligerito-backend chain prove.
    pub fn prove_chain<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[Compression],
        challenger: &mut Ch,
    ) -> (
        super::chain_common::ChainProofLigerito,
        flock_core::pcs::Commitment,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(self.n_compressions, self.n_block_slots());
        let n_log = self.n_blocks_log();
        let (z_packed, a_packed, b_packed, z_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(compressions, n_log);
        super::chain_common::prove_chain_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &CHAIN_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    pub fn verify_chain<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &super::chain_common::ChainProofLigerito,
        cv_0: &[u32; 8],
        cv_last: &[u32; 8],
        challenger: &mut Ch,
    ) -> Result<(), ChainVerifyError> {
        assert_eq!(self.n_compressions, self.n_block_slots());
        let n_log = self.n_blocks_log();
        let cv0_phys = cv_to_phys_bits(cv_0);
        let cvlast_phys = cv_to_phys_bits(cv_last);
        super::chain_common::verify_chain_ligerito_generic(
            &self.r1cs,
            &CHAIN_LAYOUT,
            commitment,
            proof,
            n_log,
            &cv0_phys,
            &cvlast_phys,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }

    /// Verify a [`ChainProof`] against public endpoints `cv_0` (first input CV)
    /// and `cv_last` (last output CV).
    pub fn verify_chain_basefold<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &ChainProof,
        cv_0: &[u32; 8],
        cv_last: &[u32; 8],
        challenger: &mut Ch,
    ) -> Result<(), ChainVerifyError> {
        // Mirror `prove_chain`'s requirement: chain proof must cover exactly
        // one compression per witness slot (no padding) to be meaningful.
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "verify_chain requires n_compressions to exactly fill n_block_slots \
             (no padding); got n_compressions={}, n_block_slots={}. Use a \
             power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        let n_log = self.n_blocks_log();
        let cv0_phys = cv_to_phys_bits(cv_0);
        let cvlast_phys = cv_to_phys_bits(cv_last);
        super::chain_common::verify_chain_generic(
            &self.r1cs,
            &CHAIN_LAYOUT,
            commitment,
            proof,
            n_log,
            &cv0_phys,
            &cvlast_phys,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Merkle path: SHA-256 geometry + thin wrappers over the generic Merkle core.
// ───────────────────────────────────────────────────────────────────────────

pub use super::merkle_path_common::{
    MerklePathProof, MerklePathProofLigerito, MerklePathVerifyError,
};

/// SHA-256's 4-slot geometry for the Merkle-path protocol. The block starts
/// with four 256-bit slots in order:
/// - slot 0 (bytes 0..32)    = `H` (the IV — committed but unconstrained by
///                              the Merkle protocol)
/// - slot 1 (bytes 32..64)   = `H_out` (= z_i, the per-hash output) → `Z`
/// - slot 2 (bytes 64..96)   = `M[0..8]` (left 8 words of the message) → `X_L`
/// - slot 3 (bytes 96..128)  = `M[8..16]` (right 8 words of the message) → `X_R`
///
/// The slot offsets are byte-aligned (32 byte each) and contiguous because of
/// the layout adjustment that moved `Z_CONST_POS` to the end of the witness.
pub const MERKLE_LAYOUT: super::merkle_path_common::MerkleLayout =
    super::merkle_path_common::MerkleLayout {
        k_log: K_LOG,
        k_skip: K_SKIP,
        region_log: 8,         // 2^8 = 256-bit slots
        region_bits: 256,      // each slot fully used (no padding within slot)
        slot_base_byte_off: 0, // 4-slot region starts at byte 0
        z_slot: 1,
        x_l_slot: 2,
        x_r_slot: 3,
        // other_slot = 0 (the IV / H_in)
    };

/// Convert a public 256-bit hash value (8 × u32 words, LE bit order within
/// each word) to physical within-slot bool order. Same shape as
/// `cv_to_phys_bits`; reused for the Merkle-path leaf and root.
pub fn hash_to_phys_bits(h: &[u32; 8]) -> Vec<bool> {
    cv_to_phys_bits(h)
}

impl Sha256HybridSetup {
    /// Prove a Merkle path of length `K = 2^n_log` SHA-256 hashes. The prover
    /// is given the full per-hash `Compression` sequence (each hash's
    /// `(H_in, M)`) so trace-gen is parallel. Honest data must satisfy: for
    /// each `i = 1..K-1`, the `b[i]`-selected half of `M[i]` equals
    /// `compressions[i-1].1[H_out]` (i.e. the previous compression's
    /// digest, byte-equivalent).
    pub fn prove_merkle_path<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[Compression],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> (MerklePathProof, flock_core::pcs::Commitment) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_merkle_path requires n_compressions to exactly fill \
             n_block_slots (no padding); got n_compressions={}, \
             n_block_slots={}. Use a power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let (z_packed, a_packed, b_packed, z_lincheck) =
            self.generate_witness_ab(compressions);
        super::merkle_path_common::prove_merkle_path_generic(
            &self.r1cs,
            &self.pcs_params,
            &MERKLE_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Verify a [`MerklePathProof`] against public `leaf` and `root` (each as
    /// 8 × u32 words) and the public bit vector `b`.
    pub fn verify_merkle_path<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &MerklePathProof,
        leaf: &[u32; 8],
        root: &[u32; 8],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> Result<(), MerklePathVerifyError> {
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "verify_merkle_path requires n_compressions to exactly fill \
             n_block_slots (no padding)",
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let n_log = self.n_blocks_log();
        let leaf_phys = hash_to_phys_bits(leaf);
        let root_phys = hash_to_phys_bits(root);
        super::merkle_path_common::verify_merkle_path_generic(
            &self.r1cs,
            &MERKLE_LAYOUT,
            commitment,
            proof,
            n_log,
            &leaf_phys,
            &root_phys,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Prove `P = 2^path_log` independent SHA-256 Merkle paths into a single
    /// shared root. `compressions` is the concatenation of all path
    /// compressions in path-id order — path `i_p` occupies rows
    /// `[i_p · L, (i_p + 1) · L)` where `L = n_block_slots / P`. `b_bits` is
    /// likewise the per-path bit vectors concatenated in path-id order; the
    /// first bit of each path is unused by the protocol (the leaf goes into
    /// the `in_L` slot of each path's first hash by convention).
    pub fn prove_merkle_paths<Ch: flock_core::challenger::Challenger>(
        &self,
        path_log: usize,
        compressions: &[Compression],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> (MerklePathProof, flock_core::pcs::Commitment) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_merkle_paths requires n_compressions to exactly fill \
             n_block_slots (no padding); got n_compressions={}, \
             n_block_slots={}. Use a power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        assert!(
            path_log <= self.n_blocks_log(),
            "path_log {} > n_blocks_log {}",
            path_log,
            self.n_blocks_log(),
        );
        let (z_packed, a_packed, b_packed, z_lincheck) =
            self.generate_witness_ab(compressions);
        super::merkle_path_common::prove_merkle_paths_generic(
            &self.r1cs,
            &self.pcs_params,
            &MERKLE_LAYOUT,
            path_log,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Verify a multi-path [`MerklePathProof`] against `P = 2^path_log` public
    /// leaves and a single shared `root` (each 8 × u32 words).
    pub fn verify_merkle_paths<Ch: flock_core::challenger::Challenger>(
        &self,
        path_log: usize,
        commitment: &flock_core::pcs::Commitment,
        proof: &MerklePathProof,
        leaves: &[[u32; 8]],
        root: &[u32; 8],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> Result<(), MerklePathVerifyError> {
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "verify_merkle_paths requires n_compressions to exactly fill \
             n_block_slots (no padding)",
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let n_paths = 1usize << path_log;
        assert_eq!(leaves.len(), n_paths, "leaves must have length 2^path_log");
        let n_log = self.n_blocks_log();
        let leaves_phys: Vec<Vec<bool>> = leaves.iter().map(hash_to_phys_bits).collect();
        let leaves_phys_refs: Vec<&[bool]> = leaves_phys.iter().map(|v| v.as_slice()).collect();
        let root_phys = hash_to_phys_bits(root);
        super::merkle_path_common::verify_merkle_paths_generic(
            &self.r1cs,
            &MERKLE_LAYOUT,
            path_log,
            commitment,
            proof,
            n_log,
            &leaves_phys_refs,
            &root_phys,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Ligerito-backend mirror of [`Self::prove_merkle_path`]. Same protocol;
    /// the final PCS open routes through Ligerito (smaller proof). Requires a
    /// registered Ligerito security config for this `m` (m ≥ 22).
    pub fn prove_merkle_path_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[Compression],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> (MerklePathProofLigerito, flock_core::pcs::Commitment) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_merkle_path_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding); got n_compressions={}, \
             n_block_slots={}. Use a power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let (z_packed, a_packed, b_packed, z_lincheck) =
            self.generate_witness_ab(compressions);
        super::merkle_path_common::prove_merkle_paths_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &MERKLE_LAYOUT,
            0, // path_log: single-path
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Ligerito-backend mirror of [`Self::verify_merkle_path`].
    pub fn verify_merkle_path_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &MerklePathProofLigerito,
        leaf: &[u32; 8],
        root: &[u32; 8],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> Result<(), MerklePathVerifyError> {
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "verify_merkle_path_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding)",
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let n_log = self.n_blocks_log();
        let leaf_phys = hash_to_phys_bits(leaf);
        let root_phys = hash_to_phys_bits(root);
        super::merkle_path_common::verify_merkle_paths_ligerito_generic(
            &self.r1cs,
            &MERKLE_LAYOUT,
            0, // path_log: single-path
            commitment,
            proof,
            n_log,
            &[leaf_phys.as_slice()],
            &root_phys,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }

    /// Ligerito-backend mirror of [`Self::prove_merkle_paths`].
    pub fn prove_merkle_paths_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        path_log: usize,
        compressions: &[Compression],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> (MerklePathProofLigerito, flock_core::pcs::Commitment) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_merkle_paths_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding); got n_compressions={}, \
             n_block_slots={}. Use a power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        assert!(
            path_log <= self.n_blocks_log(),
            "path_log {} > n_blocks_log {}",
            path_log,
            self.n_blocks_log(),
        );
        let (z_packed, a_packed, b_packed, z_lincheck) =
            self.generate_witness_ab(compressions);
        super::merkle_path_common::prove_merkle_paths_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &MERKLE_LAYOUT,
            path_log,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    /// Ligerito-backend mirror of [`Self::verify_merkle_paths`].
    pub fn verify_merkle_paths_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        path_log: usize,
        commitment: &flock_core::pcs::Commitment,
        proof: &MerklePathProofLigerito,
        leaves: &[[u32; 8]],
        root: &[u32; 8],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> Result<(), MerklePathVerifyError> {
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "verify_merkle_paths_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding)",
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let n_paths = 1usize << path_log;
        assert_eq!(leaves.len(), n_paths, "leaves must have length 2^path_log");
        let n_log = self.n_blocks_log();
        let leaves_phys: Vec<Vec<bool>> = leaves.iter().map(hash_to_phys_bits).collect();
        let leaves_phys_refs: Vec<&[bool]> = leaves_phys.iter().map(|v| v.as_slice()).collect();
        let root_phys = hash_to_phys_bits(root);
        super::merkle_path_common::verify_merkle_paths_ligerito_generic(
            &self.r1cs,
            &MERKLE_LAYOUT,
            path_log,
            commitment,
            proof,
            n_log,
            &leaves_phys_refs,
            &root_phys,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────


// ---------------------------------------------------------------------------
// Batch-major witness producer (WitnessLayout::BatchMajor).
//
// V = 8 compressions in lockstep ([u32; 8] lanes — the adds/sigmas
// auto-vectorize); witness fields are OR'd V-wide into an L1-resident
// interleaved row buffer (already batch-major order) and NT-flushed per
// useful 128-bit chunk by the shared driver. See
// `common::drive_witness_batch_major`.
// ---------------------------------------------------------------------------

use super::common::{BM_V, BmRow, add_carry_parts_v, or_bit_row, or_u32_row};

#[inline(always)]
fn map_v(x: &[u32; BM_V], f: impl Fn(u32) -> u32) -> [u32; BM_V] {
    std::array::from_fn(|j| f(x[j]))
}
#[inline(always)]
fn xor_v(x: &[u32; BM_V], y: &[u32; BM_V]) -> [u32; BM_V] {
    std::array::from_fn(|j| x[j] ^ y[j])
}
#[inline(always)]
fn and_v(x: &[u32; BM_V], y: &[u32; BM_V]) -> [u32; BM_V] {
    std::array::from_fn(|j| x[j] & y[j])
}

struct BmRows<'a> {
    z: &'a mut [BmRow],
    a: &'a mut [BmRow],
    b: &'a mut [BmRow],
}

/// z = a = v, b = all-ones (free-witness tautology rows).
#[inline(always)]
fn bm_write_lin(rows: &mut BmRows<'_>, bit: usize, vals: &[u32; BM_V]) {
    or_u32_row(rows.z, bit, vals);
    or_u32_row(rows.a, bit, vals);
    or_u32_row(rows.b, bit, &[0xFFFF_FFFF; BM_V]);
}

/// Inline add: carry rows only.
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

/// Build one V = 8 group of compressions into interleaved rows. Mirrors
/// [`build_block_ab_packed_into`] field-for-field (the lockstep test below
/// pins byte-equality against the row-major driver).
fn build_group_batch_major(
    inputs: [&([u32; 8], [u32; 16]); BM_V],
    rz: &mut [BmRow],
    ra: &mut [BmRow],
    rb: &mut [BmRow],
) {
    let mut rows = BmRows { z: rz, a: ra, b: rb };
    let h_in: [[u32; BM_V]; 8] = std::array::from_fn(|w| std::array::from_fn(|j| inputs[j].0[w]));
    let m: [[u32; BM_V]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| inputs[j].1[i]));

    or_bit_row(rows.z, Z_CONST_POS);
    or_bit_row(rows.a, Z_CONST_POS);
    or_bit_row(rows.b, Z_CONST_POS);

    for w in 0..H_WORDS {
        bm_write_lin(&mut rows, h_bit(w, 0), &h_in[w]);
    }
    for i in 0..M_WORDS {
        bm_write_lin(&mut rows, m_bit(i, 0), &m[i]);
    }

    // Message schedule.
    let mut w_sched: Vec<[u32; BM_V]> = Vec::with_capacity(64);
    w_sched.extend_from_slice(&m);
    for t in 16..64 {
        let s_0 = bm_add_inline(
            &mut rows,
            &map_v(&w_sched[t - 2], small_sigma1),
            &w_sched[t - 7],
            sched_carry_bit(t, 0, 0),
        );
        let s_1 = bm_add_inline(
            &mut rows,
            &s_0,
            &map_v(&w_sched[t - 15], small_sigma0),
            sched_carry_bit(t, 1, 0),
        );
        let w_t = bm_add_inline(&mut rows, &s_1, &w_sched[t - 16], sched_carry_bit(t, 2, 0));
        bm_write_lin(&mut rows, w_bit(t, 0), &w_t);
        w_sched.push(w_t);
    }

    // 64 rounds.
    let mut aa = h_in[0];
    let mut bb = h_in[1];
    let mut cc = h_in[2];
    let mut dd = h_in[3];
    let mut ee = h_in[4];
    let mut ff = h_in[5];
    let mut gg = h_in[6];
    let mut hh = h_in[7];
    for r in 0..N_ROUNDS {
        let f_xor_g = xor_v(&ff, &gg);
        let ch_and_v = and_v(&ee, &f_xor_g);
        or_u32_row(rows.z, ch_and_bit(r, 0), &ch_and_v);
        or_u32_row(rows.a, ch_and_bit(r, 0), &ee);
        or_u32_row(rows.b, ch_and_bit(r, 0), &f_xor_g);
        let ch_out = xor_v(&ch_and_v, &gg);

        let b_xor_a = xor_v(&bb, &aa);
        let c_xor_a = xor_v(&cc, &aa);
        let maj_and_v = and_v(&b_xor_a, &c_xor_a);
        or_u32_row(rows.z, maj_and_bit(r, 0), &maj_and_v);
        or_u32_row(rows.a, maj_and_bit(r, 0), &b_xor_a);
        or_u32_row(rows.b, maj_and_bit(r, 0), &c_xor_a);
        let maj_out = xor_v(&maj_and_v, &aa);

        let k_r = [SHA256_K[r]; BM_V];
        let t1_0 = bm_add_inline(&mut rows, &hh, &map_v(&ee, big_sigma1), round_carry_bit(r, 0, 0));
        let t1_1 = bm_add_inline(&mut rows, &t1_0, &ch_out, round_carry_bit(r, 1, 0));
        let t1_2 = bm_add_inline(&mut rows, &t1_1, &k_r, round_carry_bit(r, 2, 0));
        let t1 = bm_add_inline(&mut rows, &t1_2, &w_sched[r], round_carry_bit(r, 3, 0));
        bm_write_lin(&mut rows, t1_bit(r, 0), &t1);

        let t2 = bm_add_inline(&mut rows, &map_v(&aa, big_sigma0), &maj_out, round_carry_bit(r, 4, 0));
        let e_new = bm_add_inline(&mut rows, &dd, &t1, round_carry_bit(r, 5, 0));
        bm_write_lin(&mut rows, e_new_bit(r, 0), &e_new);
        let a_new = bm_add_inline(&mut rows, &t1, &t2, round_carry_bit(r, 6, 0));
        bm_write_lin(&mut rows, a_new_bit(r, 0), &a_new);

        hh = gg;
        gg = ff;
        ff = ee;
        ee = e_new;
        dd = cc;
        cc = bb;
        bb = aa;
        aa = a_new;
    }

    // Output feed-forward.
    let final_state = [aa, bb, cc, dd, ee, ff, gg, hh];
    for w in 0..N_OUT_WORDS {
        let sum = bm_add_inline(&mut rows, &final_state[w], &h_in[w], out_carry_bit(w, 0));
        bm_write_lin(&mut rows, h_out_bit(w, 0), &sum);
    }
}

/// Batch-major counterpart of [`generate_witness_with_ab_packed_and_lincheck`]
/// — `(z, a, b, z_lincheck)` with z/a/b in the batch-major layout. Padding
/// slots run a compression of the all-zero input (constant wire = 1).
pub fn generate_witness_batch_major(
    compressions: &[([u32; 8], [u32; 16])],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    let padding: ([u32; 8], [u32; 16]) = ([0u32; 8], [0u32; 16]);
    super::common::drive_witness_batch_major(
        compressions,
        &padding,
        n_blocks_log,
        K_LOG,
        USEFUL_BITS,
        |group, rz, ra, rb| build_group_batch_major(group, rz, ra, rb),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 PRNG, deterministic.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (z ^ (z >> 31)) as u32
        }
        fn next_block(&mut self) -> [u32; 16] {
            std::array::from_fn(|_| self.next_u32())
        }
    }

    /// Batch-major witness equality vs the row-major driver (word-transpose
    /// + identical stripe), incl. padding slots via a non-power-of-two count.
    #[test]
    fn batch_major_witness_matches_row_major_transposed() {
        for (n_inputs, n_log) in [(8usize, 3usize), (11, 4)] {
            let mut rng = Rng::new(0xBA7C_5A + n_log as u64);
            let inputs: Vec<([u32; 8], [u32; 16])> = (0..n_inputs)
                .map(|_| (std::array::from_fn(|_| rng.next_u32()), rng.next_block()))
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

    /// Batch-major end-to-end roundtrip (BaseFold) + tamper rejection.
    #[test]
    fn batch_major_prove_fast_basefold_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let setup = Sha256HybridSetup::new_batch_major(8);
        let mut rng = Rng::new(0xBA7C_F012);
        let inputs: Vec<([u32; 8], [u32; 16])> = (0..8)
            .map(|_| (std::array::from_fn(|_| rng.next_u32()), rng.next_block()))
            .collect();

        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, commitment, claim_p) = setup.prove_fast_basefold(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-test-v0");
        let claim_v = setup
            .verify_basefold(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("batch-major verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let mut bad = proof.clone();
        bad.zerocheck.final_a_eval.lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(
            setup.verify_basefold(&commitment, &bad, &mut ch).is_err(),
            "tampered batch-major proof accepted"
        );
    }

    /// Row-by-row R1CS check `(A·z) ⊙ (B·z) = (C·z) = z`.
    fn satisfies_singleblock(
        a: &SparseBinaryMatrix,
        b: &SparseBinaryMatrix,
        z: &[bool],
    ) -> Result<(), usize> {
        for i in 0..a.rows.len() {
            let av = a.rows[i].iter().fold(false, |acc, &s| acc ^ z[s]);
            let bv = b.rows[i].iter().fold(false, |acc, &s| acc ^ z[s]);
            if (av && bv) != z[i] {
                return Err(i);
            }
        }
        Ok(())
    }

    #[test]
    fn useful_bits_matches_constants() {
        // Merkle-aligned: H, H_out, M_lo, M_hi occupy the first four 256-bit
        // slots (= one 4-slot region of 1024 bits) for clean Merkle-path
        // protocol addressing. Z_CONST_POS moved to bit 31,400 so it doesn't
        // interrupt the slot alignment.
        assert_eq!(H_BASE, 0);
        assert_eq!(H_OUT_BASE, 256);
        assert_eq!(M_BASE, 512);
        assert_eq!(CH_AND_BASE, 1024);
        assert_eq!(MAJ_AND_BASE, 3072);
        assert_eq!(ROUND_CARRY_BASE, 5120);
        assert_eq!(W_BASE, 19008);
        assert_eq!(SCHED_CARRY_BASE, 20544);
        assert_eq!(T1_BASE, 25008);
        assert_eq!(E_NEW_BASE, 27056);
        assert_eq!(A_NEW_BASE, 29104);
        assert_eq!(OUT_CARRY_BASE, 31152);
        assert_eq!(Z_CONST_POS, 31400);
        assert_eq!(USEFUL_BITS, 31401);
        assert!(USEFUL_BITS <= K);
    }

    #[test]
    fn block_witness_satisfies_matrix_and_matches_reference() {
        let (a, b) = build_matrices();
        let mut rng = Rng::new(0xC0FFEE_5A55);
        let cases: [([u32; 8], [u32; 16]); 4] = [
            (SHA256_IV, [0u32; 16]),
            (
                SHA256_IV,
                [
                    0x6162_6380,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0x0000_0018,
                ],
            ),
            (SHA256_IV, rng.next_block()),
            (std::array::from_fn(|_| rng.next_u32()), rng.next_block()),
        ];
        for (h_in, m) in cases {
            let z = build_block_witness(&h_in, &m);
            assert!(
                satisfies_singleblock(&a, &b, &z).is_ok(),
                "R1CS not satisfied for h_in={:08x?}, m[0]={:08x}",
                h_in,
                m[0]
            );
            assert_eq!(
                read_h_out(&z),
                sha256_compress(&h_in, &m),
                "H_out mismatch for h_in={:08x?}, m[0]={:08x}",
                h_in,
                m[0]
            );
        }
    }

    /// `Sha2LincheckCircuit` walker matches sparse fold byte-for-byte.
    #[test]
    fn lincheck_circuit_matches_sparse() {
        use flock_core::lincheck::{LincheckCircuit, SparseMatrixCircuit};

        let mut rng = Rng::new(0x5_4A2_CCA1);
        let (a_0, b_0) = build_matrices();
        let sparse = SparseMatrixCircuit::new(&a_0, &b_0);
        let walker = Sha2LincheckCircuit;
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

    /// Ligerito-backend prove_fast roundtrip. Needs ≥ 128 compressions (m=22).
    #[test]
    #[ignore]
    fn prove_fast_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let mut rng = Rng::new(0x5_a2_211e);
        let n = 128;
        let compressions: Vec<([u32; 8], [u32; 16])> =
            (0..n).map(|_| (SHA256_IV, rng.next_block())).collect();
        let setup = Sha256HybridSetup::new(n);
        let mut ch_p = FsChallenger::new(b"flock-sha2-lig-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&compressions, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-sha2-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    fn prove_fast_basefold_matches_prove() {
        use flock_core::challenger::FsChallenger;
        let mut rng = Rng::new(0xFACE_FEED);
        let n = 8;
        let compressions: Vec<([u32; 8], [u32; 16])> =
            (0..n).map(|_| (SHA256_IV, rng.next_block())).collect();
        let setup = Sha256HybridSetup::new(n);

        let mut ch_slow = FsChallenger::new(b"flock-test-v0");
        let (proof_slow, commit_slow, claim_slow) = setup.prove(&compressions, &mut ch_slow);
        let mut ch_fast = FsChallenger::new(b"flock-test-v0");
        let (proof_fast, commit_fast, claim_fast) =
            setup.prove_fast_basefold(&compressions, &mut ch_fast);

        // Same transcript ⇒ identical commitment root, claim, transcript.
        assert_eq!(commit_slow.root, commit_fast.root, "commitments must match");
        assert_eq!(claim_slow, claim_fast, "claims must match");
        assert_eq!(proof_slow.lincheck.rounds, proof_fast.lincheck.rounds);
        assert_eq!(proof_slow.lincheck.z_partial, proof_fast.lincheck.z_partial);

        // Verify the fast proof end-to-end.
        let mut v = FsChallenger::new(b"flock-test-v0");
        let v_claim = setup
            .verify_basefold(&commit_fast, &proof_fast, &mut v)
            .expect("verify ok");
        assert_eq!(v_claim, claim_fast);
    }

    /// Generic (matrix-driven) Ligerito prove produces a byte-identical
    /// proof to the specialized `prove_fast` — pins that the generic path
    /// (bool trace → pack → apply → prove) and the fused path agree.
    /// 128 compressions = m = 22, the smallest default-Ligerito-legal size.
    #[test]
    fn prove_ligerito_generic_matches_prove_fast() {
        use flock_core::challenger::FsChallenger;
        let n = 128;
        let setup = Sha256HybridSetup::new(n);
        let mut rng = Rng::new(0x5A2_63112);
        let comps: Vec<Compression> = (0..n)
            .map(|_| {
                (
                    std::array::from_fn(|_| rng.next_u32()),
                    std::array::from_fn(|_| rng.next_u32()),
                )
            })
            .collect();
        let mut ch_f = FsChallenger::new(b"flock-sha2-gvf");
        let (proof_f, commit_f, claim_f) = setup.prove_fast(&comps, &mut ch_f);
        let mut ch_g = FsChallenger::new(b"flock-sha2-gvf");
        let (proof_g, commit_g, claim_g) = setup.prove_ligerito(&comps, &mut ch_g);
        assert_eq!(commit_f.root, commit_g.root);
        assert_eq!(claim_f, claim_g);
        assert_eq!(
            bincode::serialize(&proof_f).unwrap(),
            bincode::serialize(&proof_g).unwrap(),
            "generic and fused Ligerito proofs must be byte-identical"
        );
        // And it verifies through the standard (Ligerito) verifier.
        let mut ch_v = FsChallenger::new(b"flock-sha2-gvf");
        let claim_v = setup
            .verify(&commit_g, &proof_g, &mut ch_v)
            .expect("verify ok");
        assert_eq!(claim_v, claim_g);
    }

    #[test]
    fn prove_verify_roundtrip_small() {
        use flock_core::challenger::FsChallenger;
        let mut rng = Rng::new(0xBEEF_CAFE);
        let n = 8;
        let compressions: Vec<([u32; 8], [u32; 16])> =
            (0..n).map(|_| (SHA256_IV, rng.next_block())).collect();
        let setup = Sha256HybridSetup::new(n);

        let mut p_ch = FsChallenger::new(b"flock-test-v0");
        let (proof, commit, claim) = setup.prove(&compressions, &mut p_ch);

        let mut v_ch = FsChallenger::new(b"flock-test-v0");
        let v_claim = setup
            .verify_basefold(&commit, &proof, &mut v_ch)
            .expect("verify ok");
        assert_eq!(v_claim, claim);
    }

    /// Constant-wire pin (docs/const-wire-pin.md). `new(5)` has padding blocks
    /// (filled with a valid all-zero-input compression, constant = 1) so the
    /// honest proof verifies; the all-zero witness must be rejected by the pin.
    /// (For SHA-2 the pin lives on the R1CS-built CSC circuit, not the walker.)
    #[test]
    fn const_pin_all_zero_rejected() {
        use flock_core::challenger::FsChallenger;

        let n = 5; // 3 padding blocks
        let setup = Sha256HybridSetup::new(n);

        // (1) Honest proof with filled padding verifies.
        let mut rng = Rng::new(0x5EED_50A2);
        let compressions: Vec<([u32; 8], [u32; 16])> =
            (0..n).map(|_| (SHA256_IV, rng.next_block())).collect();
        let mut ch_p = FsChallenger::new(b"honest");
        let (proof, commit, claim_p) = setup.prove(&compressions, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"honest");
        let claim_v = setup
            .verify_basefold(&commit, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("honest padded proof rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        // (2) All-zero witness must be rejected by the pin.
        let zeros: Vec<([u32; 8], [u32; 16])> = vec![([0u32; 8], [0u32; 16]); n];
        let (mut z, mut a, mut b, mut zlc) =
            generate_witness_with_ab_packed_and_lincheck(&zeros, setup.n_blocks_log());
        z.iter_mut().for_each(|v| *v = flock_core::field::F128::ZERO);
        a.iter_mut().for_each(|v| *v = flock_core::field::F128::ZERO);
        b.iter_mut().for_each(|v| *v = flock_core::field::F128::ZERO);
        zlc.iter_mut().for_each(|v| *v = 0);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let mut ch_p = FsChallenger::new(b"poc");
        let (proof, commit, _) = crate::prover::prove_fast_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            zlc,
            circuit,
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(b"poc");
        let res = setup.verify_basefold(&commit, &proof, &mut ch_v);
        assert!(
            matches!(res, Err(flock_core::verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }

    #[test]
    fn block_r1cs_satisfies_for_one_block() {
        // Smallest valid: n_blocks_log = 3 → 8 outer blocks, 7 of which are empty padding.
        let r1cs = build_block_r1cs(3);
        let n_blocks = 1 << 3;
        let z_block = build_block_witness(&SHA256_IV, &[0u32; 16]);
        // Tile: real block in slot 0, zeros elsewhere.
        let mut z = vec![false; n_blocks * K];
        z[..K].copy_from_slice(&z_block);
        // The remaining (n_blocks - 1) blocks are all-zero, which trivially
        // satisfies the R1CS — all AND rows become 0·0 = 0, all "free witness"
        // tautologies hold for 0, padding rows are 0.
        // BUT z[0] = 1 only in block 0; in other blocks z[0]=0, which breaks
        // the K-row's z[0]·z[0] = z[0] when z[0]=0 trivially (0·0=0 ✓).
        // The H/M free-witness rows are fine at 0 as well.
        // The carry rows are 0·0 = 0 ✓.
        // Sum rows constrain z[slot] = XOR of zeros = 0 ✓.
        assert!(r1cs.satisfies(&z));
    }

    // -----------------------------------------------------------------------
    // Hash-chain end-to-end tests: honest chain, prove → verify roundtrip,
    // and verifier mutation rejection. Mirrors the blake3_chain / keccak_chain
    // suite.
    // -----------------------------------------------------------------------

    /// Build an honest SHA-256 chain of `n` compressions: each block's H_in
    /// equals the previous block's H_out (= `sha256_compress` of the previous).
    /// Returns `(blocks, cv_0, cv_last)`.
    fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
        let mut rng = Rng::new(seed);
        let mut cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let cv0 = cv;
        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let m = rng.next_block();
            blocks.push((cv, m));
            cv = sha256_compress(&cv, &m);
        }
        (blocks, cv0, cv)
    }

    /// Ligerito-backend chain roundtrip.
    #[test]
    #[ignore]
    fn prove_chain_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        // n=128 → m=22 with K_LOG=15.
        let setup = Sha256HybridSetup::new(128);
        let (blocks, cv_0, cv_last) = honest_chain(setup.n_compressions, 0x511_3E_C0DE);
        let mut ch = FsChallenger::new(b"sha2-chain-lig");
        let (proof, commitment) = setup.prove_chain(&blocks, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-chain-lig");
        setup
            .verify_chain(&commitment, &proof, &cv_0, &cv_last, &mut chv)
            .expect("ligerito chain must verify");
    }

    #[test]
    fn prove_chain_basefold_roundtrip_small() {
        use flock_core::challenger::FsChallenger;
        // n_compressions=8 → m=18 (K_LOG=15 + n_log=3). Small but exercises
        // the full chain protocol (legacy BaseFold path).
        let setup = Sha256HybridSetup::new(8);
        let (blocks, cv_0, cv_last) = honest_chain(setup.n_compressions, 0xC0FFEE);
        let mut ch = FsChallenger::new(b"sha2-chain-test");
        let (proof, commitment) = setup.prove_chain_basefold(&blocks, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-chain-test");
        setup
            .verify_chain_basefold(&commitment, &proof, &cv_0, &cv_last, &mut chv)
            .expect("honest chain must verify");
    }

    #[test]
    fn verify_chain_basefold_rejects_wrong_endpoint() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(8);
        let (blocks, cv_0, cv_last) = honest_chain(setup.n_compressions, 0xBADBADBAD);
        let mut ch = FsChallenger::new(b"sha2-chain-test");
        let (proof, commitment) = setup.prove_chain_basefold(&blocks, &mut ch);

        let mut mutated_last = cv_last;
        mutated_last[0] ^= 1;
        let mut chv = FsChallenger::new(b"sha2-chain-test");
        let res = setup.verify_chain_basefold(&commitment, &proof, &cv_0, &mutated_last, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong cv_last");

        let mut mutated_0 = cv_0;
        mutated_0[7] ^= 1 << 31;
        let mut chv = FsChallenger::new(b"sha2-chain-test");
        let res = setup.verify_chain_basefold(&commitment, &proof, &mutated_0, &cv_last, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong cv_0");
    }

    /// `prove_chain_basefold` must require `n_compressions == n_block_slots`.
    #[test]
    #[should_panic(expected = "prove_chain requires n_compressions to exactly fill n_block_slots")]
    fn prove_chain_basefold_panics_on_padding() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(5);
        assert_eq!(setup.n_compressions, 5);
        assert_eq!(setup.n_block_slots(), 8);
        let (blocks, _, _) = honest_chain(setup.n_compressions, 0xDEADBEEF);
        let mut ch = FsChallenger::new(b"sha2-chain-test");
        let _ = setup.prove_chain_basefold(&blocks, &mut ch);
    }

    // -----------------------------------------------------------------------
    // Merkle-path end-to-end tests.
    // -----------------------------------------------------------------------

    /// Build an honest Merkle path of `n` SHA-256 compressions:
    /// - block 0: M = (leaf, sibling_0). z_0 = sha256_compress(IV, M).
    /// - block i ≥ 1: depending on `b_bits[i]`, M = (z_{i-1}, sibling_i) when
    ///   `b=0`, or (sibling_i, z_{i-1}) when `b=1`. z_i = sha256_compress(IV, M).
    /// All blocks use the public SHA-256 IV as `H_in`.
    /// Returns `(blocks, leaf, root, b_bits)`.
    fn honest_merkle_path(
        n: usize,
        seed: u64,
    ) -> (Vec<Compression>, [u32; 8], [u32; 8], Vec<bool>) {
        let mut rng = Rng::new(seed);
        let leaf: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let mut b_bits = vec![false; n];
        for bit in b_bits.iter_mut().skip(1) {
            *bit = rng.next_u32() & 1 == 1;
        }
        let mut blocks = Vec::with_capacity(n);
        let mut current = leaf;
        for i in 0..n {
            let sibling: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = if !b_bits[i] {
                // selected = left half of M = current; unselected = sibling
                let mut m = [0u32; 16];
                m[..8].copy_from_slice(&current);
                m[8..].copy_from_slice(&sibling);
                m
            } else {
                // selected = right half of M = current
                let mut m = [0u32; 16];
                m[..8].copy_from_slice(&sibling);
                m[8..].copy_from_slice(&current);
                m
            };
            blocks.push((SHA256_IV, m));
            current = sha256_compress(&SHA256_IV, &m);
        }
        let root = current;
        (blocks, leaf, root, b_bits)
    }

    #[test]
    fn prove_merkle_path_roundtrip_small() {
        use flock_core::challenger::FsChallenger;
        // n=8 compressions → K=2^3, smallest valid Merkle path size.
        let setup = Sha256HybridSetup::new(8);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0xC0FFEE_BEEF);
        let mut ch = FsChallenger::new(b"sha2-merkle-test");
        let (proof, commitment) = setup.prove_merkle_path(&blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-test");
        setup
            .verify_merkle_path(&commitment, &proof, &leaf, &root, &b, &mut chv)
            .expect("honest merkle path must verify");
    }

    #[test]
    fn verify_merkle_path_rejects_wrong_leaf() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(8);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0xDEAD_BEEF);
        let mut ch = FsChallenger::new(b"sha2-merkle-test");
        let (proof, commitment) = setup.prove_merkle_path(&blocks, &b, &mut ch);

        let mut bad_leaf = leaf;
        bad_leaf[0] ^= 1;
        let mut chv = FsChallenger::new(b"sha2-merkle-test");
        let res = setup.verify_merkle_path(&commitment, &proof, &bad_leaf, &root, &b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong leaf");
    }

    #[test]
    fn verify_merkle_path_rejects_wrong_root() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(8);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0xBAD_0123);
        let mut ch = FsChallenger::new(b"sha2-merkle-test");
        let (proof, commitment) = setup.prove_merkle_path(&blocks, &b, &mut ch);

        let mut bad_root = root;
        bad_root[7] ^= 1 << 31;
        let mut chv = FsChallenger::new(b"sha2-merkle-test");
        let res = setup.verify_merkle_path(&commitment, &proof, &leaf, &bad_root, &b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong root");
    }

    // -----------------------------------------------------------------------
    // Multi-path Merkle tests.
    // -----------------------------------------------------------------------

    /// Build `n_paths = 2^path_log` Merkle paths into a single shared root by
    /// replicating the same honest path `n_paths` times. Per-path length
    /// `L = total_compressions / n_paths`. Returns `(compressions, leaves,
    /// root, b_bits)` where `compressions` is path-id-major concatenation,
    /// `leaves` has length `P`, and `b_bits` is the concatenated bit vector
    /// of length `total_compressions`.
    fn honest_merkle_paths_identical(
        total: usize,
        path_log: usize,
        seed: u64,
    ) -> (Vec<Compression>, Vec<[u32; 8]>, [u32; 8], Vec<bool>) {
        let n_paths = 1usize << path_log;
        let l = total / n_paths;
        assert_eq!(n_paths * l, total, "total must factor as n_paths · L");

        let (path_blocks, leaf, root, path_b) = honest_merkle_path(l, seed);
        let mut compressions = Vec::with_capacity(total);
        let mut b_bits = Vec::with_capacity(total);
        for _ in 0..n_paths {
            compressions.extend_from_slice(&path_blocks);
            b_bits.extend_from_slice(&path_b);
        }
        let leaves = vec![leaf; n_paths];
        (compressions, leaves, root, b_bits)
    }

    #[test]
    fn multi_path_log0_matches_single_path() {
        use flock_core::challenger::FsChallenger;
        // path_log = 0 should be byte-identical to single-path.
        let setup = Sha256HybridSetup::new(8);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0xC0FFEE);

        // Single-path proof.
        let mut ch_single = FsChallenger::new(b"sha2-merkle-equiv");
        let (proof_single, commit_single) = setup.prove_merkle_path(&blocks, &b, &mut ch_single);

        // Multi-path with path_log=0.
        let mut ch_multi = FsChallenger::new(b"sha2-merkle-equiv");
        let (proof_multi, commit_multi) = setup.prove_merkle_paths(0, &blocks, &b, &mut ch_multi);

        let cb_single = bincode::serialize(&commit_single).unwrap();
        let cb_multi = bincode::serialize(&commit_multi).unwrap();
        assert_eq!(cb_single, cb_multi, "commitments must match");
        let bytes_single = bincode::serialize(&proof_single).unwrap();
        let bytes_multi = bincode::serialize(&proof_multi).unwrap();
        assert_eq!(
            bytes_single, bytes_multi,
            "path_log=0 must serialize identically to single-path"
        );

        // Both verify against the leaves array of length 1.
        let mut chv = FsChallenger::new(b"sha2-merkle-equiv");
        setup
            .verify_merkle_paths(0, &commit_multi, &proof_multi, &[leaf], &root, &b, &mut chv)
            .expect("multi-path with path_log=0 must verify");
    }

    #[test]
    fn prove_merkle_paths_roundtrip_small() {
        use flock_core::challenger::FsChallenger;
        // path_log = 1 → P = 2 paths of length 8 each = 16 total compressions.
        let setup = Sha256HybridSetup::new(16);
        let (blocks, leaves, root, b) =
            honest_merkle_paths_identical(setup.n_compressions, 1, 0xBABE);
        let mut ch = FsChallenger::new(b"sha2-merkle-paths-test");
        let (proof, commitment) = setup.prove_merkle_paths(1, &blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-test");
        setup
            .verify_merkle_paths(1, &commitment, &proof, &leaves, &root, &b, &mut chv)
            .expect("honest 2-path proof must verify");
    }

    #[test]
    fn prove_merkle_paths_roundtrip_p4() {
        use flock_core::challenger::FsChallenger;
        // path_log = 2 → P = 4 paths of length 8 each = 32 total.
        let setup = Sha256HybridSetup::new(32);
        let (blocks, leaves, root, b) =
            honest_merkle_paths_identical(setup.n_compressions, 2, 0xF00D);
        let mut ch = FsChallenger::new(b"sha2-merkle-paths-p4");
        let (proof, commitment) = setup.prove_merkle_paths(2, &blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-p4");
        setup
            .verify_merkle_paths(2, &commitment, &proof, &leaves, &root, &b, &mut chv)
            .expect("honest 4-path proof must verify");
    }

    #[test]
    fn prove_merkle_path_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        // n=128 → m=22 with K_LOG=15 (smallest m with a Ligerito config).
        let setup = Sha256HybridSetup::new(128);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0x5EED_F00D);
        let mut ch = FsChallenger::new(b"sha2-merkle-lig");
        let (proof, commitment) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-lig");
        setup
            .verify_merkle_path_ligerito(&commitment, &proof, &leaf, &root, &b, &mut chv)
            .expect("ligerito merkle path must verify");
    }

    #[test]
    fn prove_merkle_paths_ligerito_roundtrip_p2() {
        use flock_core::challenger::FsChallenger;
        // path_log=1 → P=2 paths of length 64 = 128 total → m=22.
        let setup = Sha256HybridSetup::new(128);
        let (blocks, leaves, root, b) =
            honest_merkle_paths_identical(setup.n_compressions, 1, 0xC0DE_BABE);
        let mut ch = FsChallenger::new(b"sha2-merkle-paths-lig");
        let (proof, commitment) = setup.prove_merkle_paths_ligerito(1, &blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-lig");
        setup
            .verify_merkle_paths_ligerito(1, &commitment, &proof, &leaves, &root, &b, &mut chv)
            .expect("ligerito 2-path proof must verify");
    }

    #[test]
    fn verify_merkle_paths_rejects_wrong_leaf() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(16);
        let (blocks, leaves, root, b) =
            honest_merkle_paths_identical(setup.n_compressions, 1, 0xDEAD);
        let mut ch = FsChallenger::new(b"sha2-merkle-paths-rej");
        let (proof, commitment) = setup.prove_merkle_paths(1, &blocks, &b, &mut ch);

        let mut bad_leaves = leaves.clone();
        bad_leaves[0][0] ^= 1;
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-rej");
        let res =
            setup.verify_merkle_paths(1, &commitment, &proof, &bad_leaves, &root, &b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong leaf in path 0");
    }

    #[test]
    fn verify_merkle_paths_rejects_wrong_root() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(16);
        let (blocks, leaves, root, b) =
            honest_merkle_paths_identical(setup.n_compressions, 1, 0x12345);
        let mut ch = FsChallenger::new(b"sha2-merkle-paths-rej");
        let (proof, commitment) = setup.prove_merkle_paths(1, &blocks, &b, &mut ch);

        let mut bad_root = root;
        bad_root[7] ^= 1 << 31;
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-rej");
        let res =
            setup.verify_merkle_paths(1, &commitment, &proof, &leaves, &bad_root, &b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong shared root");
    }

    #[test]
    fn verify_merkle_path_rejects_wrong_bit() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(8);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0xF00D);
        let mut ch = FsChallenger::new(b"sha2-merkle-test");
        let (proof, commitment) = setup.prove_merkle_path(&blocks, &b, &mut ch);

        // Flip one non-leading bit (B(0) := 0 by convention regardless of
        // b[0], so flipping b[0] wouldn't actually change the protocol value;
        // flip b[1] instead, which is a real chain constraint).
        let mut bad_b = b.clone();
        bad_b[1] = !bad_b[1];
        let mut chv = FsChallenger::new(b"sha2-merkle-test");
        let res = setup.verify_merkle_path(&commitment, &proof, &leaf, &root, &bad_b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong bit vector");
    }

    #[test]
    fn cv_to_phys_bits_roundtrips() {
        // Round-trip a fixed CV through bool-pack and assert the per-word bits
        // are recovered (sanity check on the within-slot layout convention).
        let cv: [u32; 8] = [
            0x01234567, 0x89ABCDEF, 0xDEADBEEF, 0xFEEDC0DE, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0,
            0x0F1E2D3C,
        ];
        let phys = cv_to_phys_bits(&cv);
        assert_eq!(phys.len(), 256);
        for w in 0..8 {
            let mut recovered = 0u32;
            for b in 0..WORD_BITS {
                if phys[WORD_BITS * w + b] {
                    recovered |= 1 << b;
                }
            }
            assert_eq!(recovered, cv[w], "word {w} mismatch");
        }
    }
}
