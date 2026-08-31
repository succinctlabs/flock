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

use super::common::{
    BitRecord, add_carry_parts, const_add_parts, fused_add3_parts, fused_add4_parts, or_bit_at,
    or_u32_at_bit,
};
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
pub const CARRIES_PER_ADD: usize = WORD_BITS - 1; // 31
/// Ripple products of a fused carry-save adder (bit 0's product row
/// vanishes against the shifted majority word's zero low bit).
pub const RIPPLE_BITS: usize = WORD_BITS - 2; // 30

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
// slot 2 = M_lo, slot 3 = M_hi); the since-retired Merkle-path shift
// protocol addressed `(H_in, H_out, M_left, M_right)` by single-bit slot
// selectors, and the geometry is pinned by fixtures, so it stays. The
// Z_CONST constant-1 bit moved to the end of useful_bits
// (after the OUT_CARRY block), where it sits in a 1-bit gap that doesn't
// disturb the slot alignment.
pub const M_BASE: usize = 2 * SLOT_BITS; // 512
pub const CH_AND_BASE: usize = M_BASE + M_WORDS * WORD_BITS; // 1,024
pub const MAJ_AND_BASE: usize = CH_AND_BASE + N_ROUNDS * WORD_BITS; // 3,072
pub const ROUND_CARRY_BASE: usize = MAJ_AND_BASE + N_ROUNDS * WORD_BITS; // 5,120

// **"Option F for SHA-256"** (the zk.golf `gf2-sha256-compress-canonical`
// record's systematic techniques, ported 2026-08-14): the round's five
// T1-chain additions become one constant add plus one fused 4-operand
// carry-save tree, `a_new` fuses `T2` into a 3-operand tree, `T1` is never
// materialized (its two consumers inline it), and each schedule step is a
// fused 4-operand tree. Per round:
//
//   hk    = h + K[r]            constant add — `31 − t` aux products, where
//                               `t = trailing_zeros(K) + 1` (the carry into
//                               bit t is the affine seed)
//   T1    = hk + Σ1(e) + Ch + W[r]   fused 4-op: 31 + 31 + 30 = 92 products
//   a_new = T1 + Σ0(a) + Maj        fused 3-op: 31 + 30 = 61 products
//   e_new = d + T1                  ripple add: 31 products
//
// Round r's aux block (variable stride, `184 + (31 − t_r)` bits):
//   [0..31)   maj1 (hk, Σ1 | shared Ch)     [92..123)  a_new maj (T1, Σ0 | Maj)
//   [31..62)  maj2 (p1, b1 | shared W)      [123..153) a_new ripple
//   [62..92)  T1 ripple                     [153..184) e_new carries
//                                           [184..184+31−t) hk = h + K aux
/// `t` of round r's constant add: `trailing_zeros(K[r]) + 1`.
pub const fn k_seed_t(r: usize) -> usize {
    SHA256_K[r].trailing_zeros() as usize + 1
}
/// Aux products of round r's `h + K[r]` constant add.
pub const fn k_add_rows(r: usize) -> usize {
    CARRIES_PER_ADD - k_seed_t(r)
}
/// Bits of round r's aux block.
pub const fn round_add_bits(r: usize) -> usize {
    RC_ADDK + k_add_rows(r)
}
// Within-round aux offsets (relative to `ROUND_BASE[r]`).
pub const RC_MAJ1: usize = 0;
pub const RC_MAJ2: usize = CARRIES_PER_ADD; // 31
pub const RC_RIP: usize = 2 * CARRIES_PER_ADD; // 62
pub const RC_AMAJ: usize = RC_RIP + RIPPLE_BITS; // 92
pub const RC_ARIP: usize = RC_AMAJ + CARRIES_PER_ADD; // 123
pub const RC_ENEW: usize = RC_ARIP + RIPPLE_BITS; // 153
pub const RC_ADDK: usize = RC_ENEW + CARRIES_PER_ADD; // 184
/// `ROUND_BASE[r]` = first bit of round r's aux block; `ROUND_BASE[64]` ends
/// the region.
pub const ROUND_BASE: [usize; N_ROUNDS + 1] = {
    let mut t = [0usize; N_ROUNDS + 1];
    let mut acc = ROUND_CARRY_BASE;
    let mut r = 0;
    while r < N_ROUNDS {
        t[r] = acc;
        acc += round_add_bits(r);
        r += 1;
    }
    t[N_ROUNDS] = acc;
    t
};

// Schedule step: W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16] as one
// fused 4-op tree — 92 aux bits per step.
pub const SCHED_ADD_BITS: usize = 2 * CARRIES_PER_ADD + RIPPLE_BITS; // 92
pub const SC_MAJ1: usize = 0;
pub const SC_MAJ2: usize = CARRIES_PER_ADD; // 31
pub const SC_RIP: usize = 2 * CARRIES_PER_ADD; // 62

// W is not materialized. E_NEW and A_NEW are materialized every other round
// to limit expression growth.
pub const EA_PERIOD: usize = 2;
/// Rounds whose E_NEW/A_NEW are materialized (r % EA_PERIOD == 1).
pub const N_EA_SLOTS: usize = N_ROUNDS / EA_PERIOD; // 32
pub const SCHED_CARRY_BASE: usize = ROUND_BASE[N_ROUNDS]; // 18,757
pub const E_NEW_BASE: usize = SCHED_CARRY_BASE + N_SCHED * SCHED_ADD_BITS; // 23,173
pub const A_NEW_BASE: usize = E_NEW_BASE + N_EA_SLOTS * WORD_BITS; // 24,197
pub const OUT_CARRY_BASE: usize = A_NEW_BASE + N_EA_SLOTS * WORD_BITS; // 25,221
pub const Z_CONST_POS: usize = OUT_CARRY_BASE + N_OUT_WORDS * CARRIES_PER_ADD; // 25,469
pub const USEFUL_BITS: usize = Z_CONST_POS + 1; // 25,470

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
pub fn round_bit(r: usize, off: usize) -> usize {
    debug_assert!(r < N_ROUNDS && off < round_add_bits(r));
    ROUND_BASE[r] + off
}
#[inline]
pub fn sched_bit(t: usize, off: usize) -> usize {
    debug_assert!((16..16 + N_SCHED).contains(&t) && off < SCHED_ADD_BITS);
    SCHED_CARRY_BASE + (t - 16) * SCHED_ADD_BITS + off
}
#[inline]
pub fn e_new_bit(r: usize, b: usize) -> usize {
    debug_assert!(r % EA_PERIOD == EA_PERIOD - 1);
    E_NEW_BASE + WORD_BITS * (r / EA_PERIOD) + b
}
#[inline]
pub fn a_new_bit(r: usize, b: usize) -> usize {
    debug_assert!(r % EA_PERIOD == EA_PERIOD - 1);
    A_NEW_BASE + WORD_BITS * (r / EA_PERIOD) + b
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

// ───────────────────────────────────────────────────────────────────────────
// Row sink — ONE symbolic walk serves both the matrix builder and the
// lincheck scatter walker, so the two can never drift apart.
// ───────────────────────────────────────────────────────────────────────────

trait RowSink {
    /// Emit row `slot` with `A = a`, `B = b` (products and copy rows alike).
    fn row(&mut self, slot: usize, a: &Sup, b: &Sup);
}

/// Writes the sparse `(A_0, B_0)` rows.
struct MatSink<'m> {
    a_rows: &'m mut [Sup],
    b_rows: &'m mut [Sup],
}
impl RowSink for MatSink<'_> {
    fn row(&mut self, slot: usize, a: &Sup, b: &Sup) {
        self.a_rows[slot] = a.clone();
        self.b_rows[slot] = b.clone();
    }
}

/// Scatters `alpha·eq[row]` (A side) and `eq[row]` (B side) into the
/// lincheck combination — duplicate slot adds cancel in char 2, matching
/// the matrix side's XOR support.
struct FoldSink<'m> {
    comb: &'m mut [F128],
    alpha: F128,
    eq_inner: &'m [F128],
}
impl RowSink for FoldSink<'_> {
    fn row(&mut self, slot: usize, a: &Sup, b: &Sup) {
        let e = self.eq_inner[slot];
        let ea = self.alpha * e;
        for &c in a {
            self.comb[c] += ea;
        }
        for &c in b {
            self.comb[c] += e;
        }
    }
}

/// 32-bit modular add `x + y`. Allocates 31 aux AND rows via
/// `carry_slot(i)`; the carry chain is `cin[i+1] = cin[i] ⊕ aux[i]`.
/// Returns the symbolic 32-bit sum (per-bit XOR support).
fn add32_inline<S: RowSink, F: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F,
    sink: &mut S,
) -> Word {
    let mut sum = zero_word();
    let mut cin: Sup = Sup::new();
    for i in 0..WORD_BITS {
        sum[i] = xor3(&x[i], &y[i], &cin);
        if i < CARRIES_PER_ADD {
            let slot = carry_slot(i);
            sink.row(slot, &xor_sup(&x[i], &cin), &xor_sup(&y[i], &cin));
            cin = xor_sup(&cin, &vec![slot]);
        }
    }
    sum
}

/// Constant-operand add `k + y` (`k` a compile-time constant, here a round
/// constant): aux products only for bits `t..30` at `base..`, where
/// `t = trailing_zeros(k) + 1` — the carries below `k`'s lowest set bit are
/// zero and the carry into bit `t` is the affine seed `y_{t-1}`.
fn add_const_inline<S: RowSink>(k: u32, y: &Word, base: usize, sink: &mut S) -> Word {
    let t = k.trailing_zeros() as usize + 1;
    let k_sup = |i: usize| -> Sup {
        if (k >> i) & 1 == 1 {
            vec![Z_CONST_POS]
        } else {
            Sup::new()
        }
    };
    // carry(i) for i >= t: the seed y_{t-1} plus the materialized prefix.
    let mut sum = zero_word();
    let mut carry: Sup = Sup::new();
    for i in 0..WORD_BITS {
        if i == t {
            carry = y[t - 1].clone();
        }
        sum[i] = xor3(&k_sup(i), &y[i], &carry);
        if (t..CARRIES_PER_ADD).contains(&i) {
            let slot = base + (i - t);
            sink.row(slot, &xor_sup(&k_sup(i), &carry), &xor_sup(&y[i], &carry));
            carry = xor_sup(&carry, &vec![slot]);
        }
    }
    sum
}

/// One carry-save layer over `(x, y | shared z)`: 31 majority-product rows
/// at `maj_base..`. Returns the partial sum `p = x⊕y⊕z` and the shifted
/// majority word `b` (`b_0 = 0`, `b_{i+1} = slot_i ⊕ z_i`).
fn csa_layer<S: RowSink>(
    x: &Word,
    y: &Word,
    z: &Word,
    maj_base: usize,
    sink: &mut S,
) -> (Word, Word) {
    let mut p = zero_word();
    let mut bw = zero_word();
    for i in 0..WORD_BITS {
        p[i] = xor3(&x[i], &y[i], &z[i]);
        if i < CARRIES_PER_ADD {
            let slot = maj_base + i;
            sink.row(slot, &xor_sup(&x[i], &z[i]), &xor_sup(&y[i], &z[i]));
            if i + 1 < WORD_BITS {
                bw[i + 1] = xor_sup(&vec![slot], &z[i]);
            }
        }
    }
    (p, bw)
}

/// The fused ripple of a carry-save pair `(p, b)` with `b_0 = 0`: 30
/// product rows at `rip_base..` (bit 0's product vanishes), sum inlined.
fn csa_ripple<S: RowSink>(p: &Word, bw: &Word, rip_base: usize, sink: &mut S) -> Word {
    let mut sum = zero_word();
    let mut g: Sup = Sup::new();
    for i in 0..WORD_BITS {
        sum[i] = xor3(&p[i], &bw[i], &g);
        if (1..=RIPPLE_BITS).contains(&i) {
            let slot = rip_base + (i - 1);
            sink.row(slot, &xor_sup(&p[i], &g), &xor_sup(&bw[i], &g));
            g = xor_sup(&g, &vec![slot]);
        }
    }
    sum
}

/// Fused 3-operand add `x + y + z` (z the shared/compact operand):
/// 31 + 30 = 61 rows.
fn fused_add3_inline<S: RowSink>(
    x: &Word,
    y: &Word,
    z: &Word,
    maj_base: usize,
    rip_base: usize,
    sink: &mut S,
) -> Word {
    let (p, bw) = csa_layer(x, y, z, maj_base, sink);
    csa_ripple(&p, &bw, rip_base, sink)
}

/// Fused 4-operand add `x + y + z + w` (z shared in layer 1, w in layer 2):
/// 31 + 31 + 30 = 92 rows.
#[allow(clippy::too_many_arguments)]
fn fused_add4_inline<S: RowSink>(
    x: &Word,
    y: &Word,
    z: &Word,
    w: &Word,
    maj1_base: usize,
    maj2_base: usize,
    rip_base: usize,
    sink: &mut S,
) -> Word {
    let (p1, b1) = csa_layer(x, y, z, maj1_base, sink);
    let (p2, b2) = csa_layer(&p1, &b1, w, maj2_base, sink);
    csa_ripple(&p2, &b2, rip_base, sink)
}

/// Materialize a symbolic word at fresh slots: emit 32 rows
/// `(linear support) · z[Z_CONST] = z[slot]`, return a slot-word.
fn materialize<S: RowSink, F: Fn(usize) -> usize>(raw: &Word, slot_fn: F, sink: &mut S) -> Word {
    let mut out = zero_word();
    for b in 0..WORD_BITS {
        let s = slot_fn(b);
        sink.row(s, &raw[b], &vec![Z_CONST_POS]);
        out[b] = vec![s];
    }
    out
}

fn add32_alloc<S: RowSink, F1: Fn(usize) -> usize, F2: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F1,
    sum_slot: F2,
    sink: &mut S,
) -> Word {
    let raw = add32_inline(x, y, carry_slot, sink);
    materialize(&raw, sum_slot, sink)
}

// ───────────────────────────────────────────────────────────────────────────
// Public matrix builder
// ───────────────────────────────────────────────────────────────────────────

/// The complete symbolic walk: every row of one compression, emitted into
/// `sink`. Serves the matrix builder and the lincheck scatterer alike.
/// Whether the walk materializes the periodic E_NEW/A_NEW cascade
/// breakers (production: yes, at [`EA_PERIOD`]). The density probe's
/// full-inlining variant sets `ea: false` — measured 184M template nnz,
/// which is why the zk.golf record's shape is NOT the production one.
#[derive(Copy, Clone)]
struct MatCfg {
    ea: bool,
}
const MAT_PROD: MatCfg = MatCfg { ea: true };

fn walk_rows<S: RowSink>(sink: &mut S) {
    walk_rows_cfg(sink, MAT_PROD)
}

fn walk_rows_cfg<S: RowSink>(sink: &mut S, mat: MatCfg) {
    // Z_CONST tautology: z[ZC]·z[ZC] = z[ZC] (boolean-pin).
    sink.row(Z_CONST_POS, &vec![Z_CONST_POS], &vec![Z_CONST_POS]);

    // H_in, M_in: free-witness rows.
    for w in 0..H_WORDS {
        for b in 0..WORD_BITS {
            let s = h_bit(w, b);
            sink.row(s, &vec![s], &vec![Z_CONST_POS]);
        }
    }
    for i in 0..M_WORDS {
        for b in 0..WORD_BITS {
            let s = m_bit(i, b);
            sink.row(s, &vec![s], &vec![Z_CONST_POS]);
        }
    }

    let h_in: Vec<Word> = (0..H_WORDS).map(|w| wire_word(|b| h_bit(w, b))).collect();
    let mut w_arr: Vec<Word> = (0..M_WORDS).map(|i| wire_word(|b| m_bit(i, b))).collect();

    // Message schedule (W[16..64]): one fused 4-operand tree per step
    // (σ1(W[t-2]) + σ0(W[t-15]) + W[t-7] + W[t-16], the plain-slot words
    // riding the shared sides); W[t] materialized (the schedule cascades
    // for t ≥ 32 without it).
    for t in 16..(16 + N_SCHED) {
        let s1 = sigma_1(&w_arr[t - 2]);
        let s0 = sigma_0(&w_arr[t - 15]);
        let w_m7 = w_arr[t - 7].clone();
        let w_m16 = w_arr[t - 16].clone();
        let raw = fused_add4_inline(
            &s1,
            &s0,
            &w_m7,
            &w_m16,
            sched_bit(t, SC_MAJ1),
            sched_bit(t, SC_MAJ2),
            sched_bit(t, SC_RIP),
            sink,
        );
        w_arr.push(raw);
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
            sink.row(s, &e[bit], &xor_sup(&f[bit], &g[bit]));
            ch_and[bit] = vec![s];
        }
        // maj_and[r][bit] = (a[bit] ⊕ b[bit]) · (a[bit] ⊕ c[bit])
        let mut maj_and = zero_word();
        for bit in 0..WORD_BITS {
            let s = maj_and_bit(r, bit);
            sink.row(s, &xor_sup(&a[bit], &bb[bit]), &xor_sup(&a[bit], &c[bit]));
            maj_and[bit] = vec![s];
        }
        let ch_out = xor_words(&ch_and, &g); // Ch = e·(f⊕g) ⊕ g
        let maj_out = xor_words(&maj_and, &a); // Maj = (a⊕b)·(a⊕c) ⊕ a

        // hk = h + K[r]: the round constant folded in as a constant add.
        let hk = add_const_inline(SHA256_K[r], &h_var, round_bit(r, RC_ADDK), sink);
        // T1 = hk + Σ1(e) + Ch + W[r]: one fused 4-operand tree. T1 is NOT
        // materialized — its two consumers inline it (both feed rows whose
        // other operands are materialized state, so the expansion is
        // bounded; no cross-round cascade).
        let t1 = fused_add4_inline(
            &hk,
            &big_sigma_1(&e),
            &ch_out,
            &w_arr[r],
            round_bit(r, RC_MAJ1),
            round_bit(r, RC_MAJ2),
            round_bit(r, RC_RIP),
            sink,
        );
        // a_new = T1 + Σ0(a) + Maj: T2 dissolves into a fused 3-op tree.
        let a_raw = fused_add3_inline(
            &t1,
            &big_sigma_0(&a),
            &maj_out,
            round_bit(r, RC_AMAJ),
            round_bit(r, RC_ARIP),
            sink,
        );
        let mat_this_round = mat.ea && r % EA_PERIOD == EA_PERIOD - 1;
        let a_new = if mat_this_round {
            materialize(&a_raw, |b| a_new_bit(r, b), sink)
        } else {
            a_raw
        };
        // e_new = d + T1.
        let e_raw = add32_inline(&d, &t1, |i| round_bit(r, RC_ENEW + i), sink);
        let e_new = if mat_this_round {
            materialize(&e_raw, |b| e_new_bit(r, b), sink)
        } else {
            e_raw
        };

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
            sink,
        );
    }
}

/// Build `(A_0, B_0)` for one block of the hybrid SHA-256 R1CS. `C_0 = I`
/// (circuit shape); use [`build_block_r1cs`] to wrap these into a
/// [`BlockR1cs`].
pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Sup> = vec![Sup::new(); K];
    let mut b_rows: Vec<Sup> = vec![Sup::new(); K];
    walk_rows(&mut MatSink {
        a_rows: &mut a_rows,
        b_rows: &mut b_rows,
    });
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

    // Bit-writer for a parts value whose row block starts at `base`.
    fn write_bits(z: &mut [bool], base: usize, v: u32, n: usize) {
        for i in 0..n {
            z[base + i] = (v >> i) & 1 == 1;
        }
    }

    // Schedule W[16..64]: fused 4-op trees.
    let mut w_arr = [0u32; 64];
    w_arr[..16].copy_from_slice(m);
    for t in 16..64 {
        let s0 = small_sigma0(w_arr[t - 15]);
        let s1 = small_sigma1(w_arr[t - 2]);
        let (w_t, m1, m2, rp) = fused_add4_parts(s1, s0, w_arr[t - 7], w_arr[t - 16]);
        write_bits(&mut z, sched_bit(t, SC_MAJ1), m1[2], CARRIES_PER_ADD);
        write_bits(&mut z, sched_bit(t, SC_MAJ2), m2[2], CARRIES_PER_ADD);
        write_bits(&mut z, sched_bit(t, SC_RIP), rp[2], RIPPLE_BITS);
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
        let s1e = big_sigma1(e);
        let s0a = big_sigma0(a);

        // hk = h + K[r] (constant add), then T1 as one fused 4-op tree.
        let (hk, _kl, _kr, kp) = const_add_parts(SHA256_K[r], h_var);
        write_bits(&mut z, round_bit(r, RC_ADDK), kp, k_add_rows(r));
        let (t1, m1, m2, rp) = fused_add4_parts(hk, s1e, ch_out, w_arr[r]);
        write_bits(&mut z, round_bit(r, RC_MAJ1), m1[2], CARRIES_PER_ADD);
        write_bits(&mut z, round_bit(r, RC_MAJ2), m2[2], CARRIES_PER_ADD);
        write_bits(&mut z, round_bit(r, RC_RIP), rp[2], RIPPLE_BITS);
        // a_new = T1 + Σ0(a) + Maj (fused 3-op); e_new = d + T1.
        let (a_new, am, ar) = fused_add3_parts(t1, s0a, maj_out);
        write_bits(&mut z, round_bit(r, RC_AMAJ), am[2], CARRIES_PER_ADD);
        write_bits(&mut z, round_bit(r, RC_ARIP), ar[2], RIPPLE_BITS);
        let e_new = add32_w(d, t1, round_bit(r, RC_ENEW), &mut z);
        if r % EA_PERIOD == EA_PERIOD - 1 {
            write_word(&mut z, a_new_bit(r, 0), a_new);
            write_word(&mut z, e_new_bit(r, 0), e_new);
        }

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
// Lincheck circuit walker — the SAME `walk_rows` the matrix builder runs,
// scattered into the combination via `FoldSink`.
// ───────────────────────────────────────────────────────────────────────────

pub struct Sha2LincheckCircuit;

impl flock_core::lincheck::LincheckCircuit for Sha2LincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    // See Blake3LincheckCircuit::const_pin_col — same latent trait-default
    // gap, same fix.
    fn const_pin_col(&self) -> Option<usize> {
        Some(Z_CONST_POS)
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
        let mut comb = vec![F128::ZERO; K];
        walk_rows(&mut FoldSink {
            comb: &mut comb,
            alpha,
            eq_inner,
        });
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
    for t in 16..64 {
        let mut rz = BitRecord::<2>::new();
        let mut ra = BitRecord::<2>::new();
        let mut rb = BitRecord::<2>::new();

        macro_rules! push3 {
            ($pos:expr, $tr:expr) => {{
                let tr = $tr;
                rz.push::<{ $pos }>(tr[2]);
                ra.push::<{ $pos }>(tr[0]);
                rb.push::<{ $pos }>(tr[1]);
            }};
        }

        let (w_t, m1, m2, rp) = fused_add4_parts(
            small_sigma1(w_sched[t - 2]),
            small_sigma0(w_sched[t - 15]),
            w_sched[t - 7],
            w_sched[t - 16],
        );
        push3!(SC_MAJ1, m1);
        push3!(SC_MAJ2, m2);
        push3!(SC_RIP, rp);

        let sched_base = sched_bit(t, 0);
        rz.flush(z, sched_base);
        ra.flush(a, sched_base);
        rb.flush(b, sched_base);

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

        // The round's aux block (≤ 215 bits at a variable stride) is
        // composed in a register record and flushed once per buffer. The
        // fixed-offset prefix takes const-position pushes; the trailing
        // variable-width `h + K` aux takes a runtime push.
        let mut rz = BitRecord::<4>::new();
        let mut ra = BitRecord::<4>::new();
        let mut rb = BitRecord::<4>::new();

        macro_rules! push3 {
            ($pos:expr, $tr:expr) => {{
                let tr = $tr;
                rz.push::<{ $pos }>(tr[2]);
                ra.push::<{ $pos }>(tr[0]);
                rb.push::<{ $pos }>(tr[1]);
            }};
        }

        // hk = h + K[r] (constant add), then T1 as one fused 4-op tree —
        // T1 is NOT materialized.
        let (hk, kl, kr, kp) = const_add_parts(SHA256_K[r], hh);
        rz.push_at(RC_ADDK, kp);
        ra.push_at(RC_ADDK, kl);
        rb.push_at(RC_ADDK, kr);
        let (t1, m1, m2, rp) = fused_add4_parts(hk, big_sigma1(ee), ch_out, w_sched[r]);
        push3!(RC_MAJ1, m1);
        push3!(RC_MAJ2, m2);
        push3!(RC_RIP, rp);

        // a_new = T1 + Σ0(a) + Maj (fused 3-op), materialized.
        let (a_new, am, ar2) = fused_add3_parts(t1, big_sigma0(aa), maj_out);
        push3!(RC_AMAJ, am);
        push3!(RC_ARIP, ar2);
        let (e_new, el, er, ep) = add_carry_parts(dd, t1);
        rz.push::<{ RC_ENEW }>(ep);
        ra.push::<{ RC_ENEW }>(el);
        rb.push::<{ RC_ENEW }>(er);
        // E_NEW/A_NEW materialize only on the periodic cascade-breaker rounds.
        if r % EA_PERIOD == EA_PERIOD - 1 {
            let off = a_new_bit(r, 0);
            or_u32_at_bit(z, off, a_new);
            or_u32_at_bit(a, off, a_new);
            or_u32_at_bit(b, off, 0xFFFF_FFFF);
            let off = e_new_bit(r, 0);
            or_u32_at_bit(z, off, e_new);
            or_u32_at_bit(a, off, e_new);
            or_u32_at_bit(b, off, 0xFFFF_FFFF);
        }

        let round_base = round_bit(r, 0);
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

/// Like the retired `generate_witness` but produces F128-packed `(z, a, b, c)` AND the
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

/// The monolithic SHA-256 R1CS + its single-slot union registry. Batch
/// proving ([`Self::prove_fast`]) goes through the UNION commit (dense
/// stack + integer lanes; `pcs_params` are the union params).
#[derive(Debug)]
pub struct Sha256HybridSetup {
    pub n_compressions: usize,
    pub r1cs: BlockR1cs,
    pub registry: crate::schedule::Registry,
    pub pcs_params: flock_core::pcs::PcsParams,
}

impl Sha256HybridSetup {
    pub fn new(n_compressions: usize) -> Self {
        Self::with_log_inv_rate(n_compressions, 1)
    }

    pub fn with_log_inv_rate(n_compressions: usize, log_inv_rate: usize) -> Self {
        // Rate keys the legacy profiles: 1 -> Fast, 2 -> Slim.
        let profile = match log_inv_rate {
            1 => flock_core::pcs::ligerito::LigeritoProfile::Fast,
            2 => flock_core::pcs::ligerito::LigeritoProfile::Slim,
            _ => flock_core::pcs::ligerito::LigeritoProfile::Fast, // other rates default to Fast
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
        let registry = crate::schedule::Registry::new(
            vec![crate::schedule::TableType::from_block_r1cs(&r1cs)],
            n_log,
        );
        let pcs_params = {
            let union = flock_core::union::UnionInstance::new(&registry, vec![n_compressions]);
            let m = union.dense_m();
            let batch = flock_core::pcs::ligerito::embedded_initial_k_or_default(m, profile);
            flock_core::pcs::PcsParams {
                m,
                log_inv_rate,
                log_batch_size: batch,
                profile,
                num_lanes: union.commit_lanes(batch),
                merkle_hash: Default::default(),
            }
        };
        Self {
            n_compressions,
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

    /// Prove `n_compressions` over the single-slot UNION commit (dense
    /// stack + integer lanes; `PCS_TRACE=1` prints the per-phase
    /// breakdown). Counts below capacity leave zero dummy rows.
    pub fn prove_fast<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofMergedLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        let union =
            flock_core::union::UnionInstance::new(&self.registry, vec![self.n_compressions]);
        let slot = crate::prover::UnionSlotProverInput::new(
            generate_witness_batch_major_partial(compressions, self.n_blocks_log()),
            self.r1cs.csc_lincheck_circuit(),
        );
        crate::prover::prove_fast_ligerito_union(&union, &self.pcs_params, vec![slot], challenger)
    }

    pub fn verify<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProofMergedLigerito,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::FlockVerifyError> {
        let union =
            flock_core::union::UnionInstance::new(&self.registry, vec![self.n_compressions]);
        let circuit = self.r1cs.csc_lincheck_circuit();
        let circs: [&dyn flock_core::lincheck::LincheckCircuit; 1] = [circuit];
        flock_core::verifier::verify_ligerito_union(
            &union,
            &circs,
            commitment,
            proof,
            &self.pcs_params,
            challenger,
        )
    }
}

/// One SHA-256 compression input: `(H_in, M)` — the 8-word input chaining
/// value plus the 16-word message block. Mirrors the [`Sha256HybridSetup`]
/// witness-gen tuple type.
pub type Compression = ([u32; 8], [u32; 16]);

// ───────────────────────────────────────────────────────────────────────────
// Reference helpers.
// ───────────────────────────────────────────────────────────────────────────

/// Reference SHA-256 compression. Returns the 8-word output chaining value
/// `H_out = H_in + state` where `state = compress256(M)` is the post-rounds
/// register state.
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

/// Convert a public 256-bit hash value (8 × u32 words, LE bit order within
/// each word) to physical within-slot bool order — the region is
/// word-contiguous, physical bit `32·w + b` holds bit `b` of word `w`.
pub fn hash_to_phys_bits(h: &[u32; 8]) -> Vec<bool> {
    let mut phys = vec![false; 256];
    for w in 0..8 {
        for b in 0..WORD_BITS {
            phys[WORD_BITS * w + b] = (h[w] >> b) & 1 == 1;
        }
    }
    phys
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
#[inline(always)]
fn bm_triple(rows: &mut BmRows<'_>, bit: usize, l: &[u32; BM_V], r: &[u32; BM_V], p: &[u32; BM_V]) {
    or_u32_row(rows.z, bit, p);
    or_u32_row(rows.a, bit, l);
    or_u32_row(rows.b, bit, r);
}

/// V-wide fused 4-operand add: maj1/maj2/ripple triples at the given bits.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn bm_fused_add4(
    rows: &mut BmRows<'_>,
    x: &[u32; BM_V],
    y: &[u32; BM_V],
    z: &[u32; BM_V],
    w: &[u32; BM_V],
    maj1_bit: usize,
    maj2_bit: usize,
    rip_bit: usize,
) -> [u32; BM_V] {
    let mut sum = [0u32; BM_V];
    let mut m1 = [[0u32; BM_V]; 3];
    let mut m2 = [[0u32; BM_V]; 3];
    let mut rp = [[0u32; BM_V]; 3];
    for j in 0..BM_V {
        let (s, a1, a2, a3) = fused_add4_parts(x[j], y[j], z[j], w[j]);
        sum[j] = s;
        for t in 0..3 {
            m1[t][j] = a1[t];
            m2[t][j] = a2[t];
            rp[t][j] = a3[t];
        }
    }
    bm_triple(rows, maj1_bit, &m1[0], &m1[1], &m1[2]);
    bm_triple(rows, maj2_bit, &m2[0], &m2[1], &m2[2]);
    bm_triple(rows, rip_bit, &rp[0], &rp[1], &rp[2]);
    sum
}

/// V-wide fused 3-operand add.
#[inline(always)]
fn bm_fused_add3(
    rows: &mut BmRows<'_>,
    x: &[u32; BM_V],
    y: &[u32; BM_V],
    z: &[u32; BM_V],
    maj_bit: usize,
    rip_bit: usize,
) -> [u32; BM_V] {
    let mut sum = [0u32; BM_V];
    let mut mj = [[0u32; BM_V]; 3];
    let mut rp = [[0u32; BM_V]; 3];
    for j in 0..BM_V {
        let (s, a1, a2) = fused_add3_parts(x[j], y[j], z[j]);
        sum[j] = s;
        for t in 0..3 {
            mj[t][j] = a1[t];
            rp[t][j] = a2[t];
        }
    }
    bm_triple(rows, maj_bit, &mj[0], &mj[1], &mj[2]);
    bm_triple(rows, rip_bit, &rp[0], &rp[1], &rp[2]);
    sum
}

/// V-wide constant add `k + y`.
#[inline(always)]
fn bm_const_add(rows: &mut BmRows<'_>, k: u32, y: &[u32; BM_V], bit: usize) -> [u32; BM_V] {
    let mut sum = [0u32; BM_V];
    let mut tr = [[0u32; BM_V]; 3];
    for j in 0..BM_V {
        let (s, l, r, p) = const_add_parts(k, y[j]);
        sum[j] = s;
        tr[0][j] = l;
        tr[1][j] = r;
        tr[2][j] = p;
    }
    bm_triple(rows, bit, &tr[0], &tr[1], &tr[2]);
    sum
}

fn build_group_batch_major(
    inputs: [&([u32; 8], [u32; 16]); BM_V],
    rz: &mut [BmRow],
    ra: &mut [BmRow],
    rb: &mut [BmRow],
) {
    let mut rows = BmRows {
        z: rz,
        a: ra,
        b: rb,
    };
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
        let w_t = bm_fused_add4(
            &mut rows,
            &map_v(&w_sched[t - 2], small_sigma1),
            &map_v(&w_sched[t - 15], small_sigma0),
            &w_sched[t - 7],
            &w_sched[t - 16],
            sched_bit(t, SC_MAJ1),
            sched_bit(t, SC_MAJ2),
            sched_bit(t, SC_RIP),
        );
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

        let hk = bm_const_add(&mut rows, SHA256_K[r], &hh, round_bit(r, RC_ADDK));
        let t1 = bm_fused_add4(
            &mut rows,
            &hk,
            &map_v(&ee, big_sigma1),
            &ch_out,
            &w_sched[r],
            round_bit(r, RC_MAJ1),
            round_bit(r, RC_MAJ2),
            round_bit(r, RC_RIP),
        );
        let a_new = bm_fused_add3(
            &mut rows,
            &t1,
            &map_v(&aa, big_sigma0),
            &maj_out,
            round_bit(r, RC_AMAJ),
            round_bit(r, RC_ARIP),
        );
        let e_new = bm_add_inline(&mut rows, &dd, &t1, round_bit(r, RC_ENEW));
        if r % EA_PERIOD == EA_PERIOD - 1 {
            bm_write_lin(&mut rows, a_new_bit(r, 0), &a_new);
            bm_write_lin(&mut rows, e_new_bit(r, 0), &e_new);
        }

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
        build_group_batch_major,
    )
}

/// Partial-count batch-major witness for the union's dynamic invocation
/// counts (M4): rows `[compressions.len(), 2^n_blocks_log)` are left
/// **identically zero** (z, a, b, and stripe — constant wire included; the
/// union lincheck's count-derived const-pin target requires zero dummies,
/// not padding compressions). `compressions.len()` may be any value up to
/// the capacity, not necessarily a power of two.
pub fn generate_witness_batch_major_partial(
    compressions: &[([u32; 8], [u32; 16])],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    super::common::drive_witness_batch_major_partial(
        compressions,
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
    compressions: &[([u32; 8], [u32; 16])],
    n_blocks_log: usize,
    dst: flock_core::union::SlotWitnessDest<'_>,
) -> Vec<u8> {
    let padding: ([u32; 8], [u32; 16]) = ([0u32; 8], [0u32; 16]);
    super::common::drive_witness_batch_major_into(
        compressions,
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
    compressions: &[([u32; 8], [u32; 16])],
    n_blocks_log: usize,
    dst: flock_core::union::SlotWitnessDest<'_>,
) -> Vec<u8> {
    super::common::drive_witness_batch_major_partial_into(
        compressions,
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
    /// declares — see the blake3 twin of this test.
    #[test]
    fn walker_const_pin_matches_r1cs() {
        use flock_core::lincheck::LincheckCircuit as _;
        let r1cs = build_block_r1cs(3);
        assert_eq!(Sha2LincheckCircuit.const_pin_col(), r1cs.const_pin);
        assert_eq!(r1cs.const_pin, Some(Z_CONST_POS));
    }

    use flock_core::test_rng::Rng;

    /// Site-specific draws kept verbatim from this file's former local `Rng`.
    trait RngExt {
        fn next_block(&mut self) -> [u32; 16];
    }
    impl RngExt for Rng {
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
        let mut rng = Rng::new(0xBA7C_5427);
        let inputs: Vec<([u32; 8], [u32; 16])> = (0..n_total)
            .map(|_| (std::array::from_fn(|_| rng.next_u32()), rng.next_block()))
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

        let setup = Sha256HybridSetup::new(128);
        let mut rng = Rng::new(0xBA7C_F012);
        let inputs: Vec<([u32; 8], [u32; 16])> = (0..128)
            .map(|_| (std::array::from_fn(|_| rng.next_u32()), rng.next_block()))
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
        // Option F rounds: 184 + (31 − t_r) aux bits each, Σ t_r = 123 over
        // the 64 round constants → 13,637 round-aux bits (was 7 × 31 × 64 =
        // 13,888), T1 no longer materialized, schedule steps at 92 (was 93).
        assert_eq!(
            ROUND_BASE[N_ROUNDS] - ROUND_CARRY_BASE,
            (0..N_ROUNDS).map(round_add_bits).sum::<usize>()
        );
        // W is never materialized; E_NEW/A_NEW only every EA_PERIOD-th
        // round (32 slots each).
        assert_eq!(SCHED_CARRY_BASE, 18_757);
        assert_eq!(E_NEW_BASE, 23_173);
        assert_eq!(A_NEW_BASE, 24_197);
        assert_eq!(OUT_CARRY_BASE, 25_221);
        assert_eq!(Z_CONST_POS, 25_469);
        assert_eq!(USEFUL_BITS, 25_470);
        assert!(USEFUL_BITS <= K);
    }

    /// Density audit: template nnz + max row width of the REAL matrices —
    /// the price of Option F's T1 inlining is bounded row growth in T1's
    /// two consumers (run with `-- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn sha2_density_profile() {
        let (a, b) = build_matrices();
        let nnz = |m: &SparseBinaryMatrix| -> usize { m.rows.iter().map(|r| r.len()).sum() };
        let (na, nb) = (nnz(&a), nnz(&b));
        let max_row = (0..K)
            .map(|s| a.rows[s].len() + b.rows[s].len())
            .max()
            .unwrap();
        eprintln!(
            "sha2 template: A {na} + B {nb} = {} nnz over {USEFUL_BITS} useful rows \
             (avg {:.1}/row, max A+B row {max_row})",
            na + nb,
            (na + nb) as f64 / USEFUL_BITS as f64,
        );
        assert!(na + nb > 0);
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

    /// Constant-wire pin (docs/const-wire-pin.md). `new(120)` is a partial
    /// count: `prove_fast`'s batch-major partial witness leaves the dummy
    /// rows identically zero (no padding compressions) and the honest proof
    /// verifies; the all-zero witness must be rejected by the pin. (For SHA-2 the pin lives on the R1CS-built CSC circuit, not
    /// the walker.)
    #[test]
    fn const_pin_all_zero_rejected() {
        use flock_core::challenger::FsChallenger;

        let n = 120; // 8 padding blocks at n_block_slots = 128 (m = 22)
        let setup = Sha256HybridSetup::new(n);

        // (1) Honest proof with filled padding verifies.
        let mut rng = Rng::new(0x5EED_50A2);
        let compressions: Vec<([u32; 8], [u32; 16])> =
            (0..n).map(|_| (SHA256_IV, rng.next_block())).collect();
        let mut ch_p = FsChallenger::new(b"honest");
        let (proof, commit, claim_p) = setup.prove_fast(&compressions, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"honest");
        let claim_v = setup
            .verify(&commit, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("honest padded proof rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        // (2) All-zero witness must be rejected by the pin (union path:
        // the count-derived const-pin target).
        let zeros: Vec<([u32; 8], [u32; 16])> = vec![([0u32; 8], [0u32; 16]); n];
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
        let (proof, commit, _) = crate::prover::prove_fast_ligerito_union(
            &union,
            &setup.pcs_params,
            vec![slot],
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(b"poc");
        let res = setup.verify(&commit, &proof, &mut ch_v);
        assert!(
            matches!(
                res,
                Err(flock_core::verifier::FlockVerifyError::Lincheck(_))
            ),
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
    // and verifier mutation rejection. Mirrors the blake3_chain suite.
    // -----------------------------------------------------------------------

    #[test]
    fn cv_to_phys_bits_roundtrips() {
        // Round-trip a fixed CV through bool-pack and assert the per-word bits
        // are recovered (sanity check on the within-slot layout convention).
        let cv: [u32; 8] = [
            0x01234567, 0x89ABCDEF, 0xDEADBEEF, 0xFEEDC0DE, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0,
            0x0F1E2D3C,
        ];
        let phys = hash_to_phys_bits(&cv);
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
