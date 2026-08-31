//! AG-skip zerocheck PIOP (M3) — the algebraic-geometry analog of the
//! Reed–Solomon univariate skip in [`super::univariate_skip_optimized`], built
//! on the genus-95 product-code kernel ([`crate::genus95_curve_code::round1`])
//! and the arbitrary-point evaluator ([`crate::genus95_curve_code`]).
//!
//! Assembled incrementally. So far it provides the two primitives unique to the
//! AG path:
//!   - the geometric-progression *friendly challenges* over γ for all `N_INNER`
//!     inner dims, and the constant `D = ∏(1+γ^{2^j})` they introduce, and
//!   - the grinding-style derivation of the post-round-1 evaluation point `r₁`:
//!     the PROVER rejection-samples by scanning nonces (each seeding a
//!     one-attempt SHA-256 DRBG from the transcript squeeze) and ships the
//!     first working nonce in the proof; the VERIFIER re-derives the point with
//!     a single attempt ([`sample_r1_prover`] / [`replay_r1_verifier`]). The
//!     DRBG reuses the transcript's own hash (the
//!     [`crate::challenger::FsChallenger`] is SHA-256), so no second
//!     cryptographic primitive enters the soundness argument.

use rayon::prelude::*;

use super::multilinear::{
    LiveLayout, expand_to_dense, fold_and_compute_round_pair_into, fold_and_round_pair_sparse_into,
    fold_in_place_pair, fold1_lookahead_into, fold2_lookahead_into, lookahead_msg_first,
    lookahead_msg_second, round_pair_naive,
};
use crate::challenger::Challenger;
use crate::field::{F128, F256Unreduced, mul_by_x};
use crate::genus95_curve_code::{
    EvaluationPoint, base_evaluation_functional, product_evaluation_functional,
};

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// A/B toggle: when set, the mlv tail runs the classic one-round-per-pass loop
/// (friendly-Horner + general fused kernels) instead of sumcheck LOOKAHEAD
/// ([`super::multilinear::fold2_lookahead_into`]) — one pass per TWO rounds,
/// deriving the second message from the bivariate Q. The proof is bit-identical
/// either way (see `lookahead_matches_classic`); lookahead cuts tail traffic
/// by ~44%.
pub static LOOKAHEAD_DISABLE: AtomicBool = AtomicBool::new(false);

/// AG-code message-dimension exponent: `2^K_SKIP = 64` base-code coordinates.
pub const K_SKIP: usize = 6;
/// Friendly inner dimensions folded by the kernel's within-block `γ^b`
/// reinterpret (matches the RS path's `N_INNER`, but all over `γ`).
pub const N_INNER: usize = 7;

/// `γ^b ∈ F128` (the GHASH generator to the `b`-th power). For `b < 128` this is
/// the element with bit `b` set; computed via `mul_by_x` for clarity/reduction.
pub fn gamma_pow(b: usize) -> F128 {
    let mut g = F128::ONE;
    for _ in 0..b {
        g = mul_by_x(g);
    }
    g
}

/// The `N_INNER` geometric-progression friendly challenges over `γ`:
/// `r_j = γ^{2^j} / (1 + γ^{2^j})`, chosen so that
/// `eq(r_inner, b) = γ^{int(b)} / D` with `D = ∏_j (1 + γ^{2^j})`.
/// The verifier pins `r[K_SKIP .. K_SKIP+N_INNER]` to these constants.
pub fn friendly_challenges() -> [F128; N_INNER] {
    let mut out = [F128::ZERO; N_INNER];
    for j in 0..N_INNER {
        let g = gamma_pow(1 << j);
        out[j] = g * (F128::ONE + g).inv();
    }
    out
}

/// `D = ∏_{j=0}^{N_INNER-1} (1 + γ^{2^j})` — the normalizing constant the
/// kernel's raw `γ^b` reinterpret omits (`kernel_output = D · true_message`).
pub fn d_const() -> F128 {
    let mut d = F128::ONE;
    for j in 0..N_INNER {
        d *= F128::ONE + gamma_pow(1 << j);
    }
    d
}

/// `D⁻¹`: the per-coordinate rescale applied to the kernel output to recover the
/// true eq-weighted round-1 message (the AG analog of restoring the RS `C_s`).
pub fn d_inv() -> F128 {
    d_const().inv()
}

// ---------------------------------------------------------------------------
// Round 1: prover message + verifier evaluation at r₁.
// ---------------------------------------------------------------------------

/// The round-1 wire message in canonical (`D⁻¹`-scaled) form, evaluator basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round1Message {
    /// `P^{ab}` fresh coords (158): kernel fresh slot `s` ↦ evaluator product
    /// coord `64 + s`. (Garbage kernel slots 158/159 dropped.)
    pub ab_fresh: Vec<F128>,
    /// The folded c message `w̄ = Σ_x eq(r_rest, x)·c(·, x)` (64). For an honest
    /// witness this equals `P^{ab}`'s value (order0) section (systematic
    /// vanishing: `P^{ab} − P^{c} = 0` on the value coords), so the verifier
    /// reuses it to reconstruct `P^{ab}`'s value rather than receiving it.
    pub c_msg: Vec<F128>,
}

/// Prover round 1: run the kernel on the packed witness, rescale by `D⁻¹`, and
/// split into the canonical `(P^{ab}` fresh, `w̄)`. `eq` holds one outer weight
/// per 1024-byte block.
#[cfg(target_arch = "aarch64")]
pub fn prove_round1(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> Round1Message {
    let (res_ab, wbar) =
        crate::genus95_curve_code::round1::round1_slp_packed(a_packed, b_packed, c_packed, eq);
    let di = d_inv();
    Round1Message {
        ab_fresh: (0..158).map(|s| di * res_ab[s]).collect(),
        c_msg: (0..64).map(|i| di * wbar[i]).collect(),
    }
}

/// Verifier: `P^{ab}(r₁) = ⟨E(r₁), [w̄ | P^{ab}_fresh]⟩` — a 222-term F128 inner
/// product over the product-code evaluation functional. The value (order0)
/// section is reconstructed as `w̄` via systematic vanishing; the fresh coords
/// map by the identity bridge (kernel slot `s` ↦ coord `64+s`).
pub fn eval_ab_at(msg: &Round1Message, point: &EvaluationPoint) -> F128 {
    let pf = product_evaluation_functional(point).expect("denominator nonzero at r1");
    let mut acc = F128::ZERO;
    for i in 0..64 {
        acc += pf[i] * msg.c_msg[i];
    }
    for s in 0..158 {
        acc += pf[64 + s] * msg.ab_fresh[s];
    }
    acc
}

/// Verifier: `ĉ(r₁, r_rest) = ⟨base(r₁), w̄⟩` via the 64-coord base functional.
pub fn eval_c_at(msg: &Round1Message, point: &EvaluationPoint) -> F128 {
    let bf = base_evaluation_functional(point).expect("denominator nonzero at r1");
    let mut acc = F128::ZERO;
    for i in 0..64 {
        acc += bf[i] * msg.c_msg[i];
    }
    acc
}

/// 8×256 byte-dot tables for the base functional `w` (64 coords): `table[pos][byte]`
/// = Σ of `w[pos*8+bit]` over the set bits of `byte`. A message's `â(r₁,·)` is then
/// `Σ_pos table[pos][byte_pos]` — 8 lookups + 7 adds instead of ~32 set-bit adds.
pub(super) fn build_w_tables(w: &[F128]) -> Vec<[F128; 256]> {
    assert_eq!(w.len(), 64, "base functional has 64 coords");
    let mut table = vec![[F128::ZERO; 256]; 8];
    for pos in 0..8 {
        for bit in 0..8 {
            let coord = w[pos * 8 + bit];
            let bm = 1usize << bit;
            for v in 0..bm {
                table[pos][bm | v] = table[pos][v] + coord;
            }
        }
    }
    table
}

/// `â(r₁, rest=r) = ⟨w, message_r⟩` via the byte-dot tables. Unchecked: the
/// caller guarantees `r*8+7 < packed.len()` and `table.len() == 8`; the byte
/// indices are in range because they come from `u8` extractions.
///
/// The 8 index bytes are fetched as ONE unaligned `u64` load and extracted in
/// registers (shifts), not 8 separate byte loads: in the fused fold the load
/// ports are contended by the message mults and the `a_mlv` stores, and the
/// 28-fewer-loads-per-pair relief measures +12% (m=28) / +19% (m=30) on the
/// whole phase (paired A/B; lookups-only microbench shows only ~4% — the win
/// is contention relief, not raw lookup speed). Tree-associated adds shorten
/// the XOR chain from depth 7 to 3. Output bit-identical.
#[inline]
pub(super) fn byte_dot(packed: &[u8], r: usize, table: &[[F128; 256]]) -> F128 {
    unsafe {
        let v = (packed.as_ptr().add(r * 8) as *const u64).read_unaligned();
        byte_dot_u64(v, table)
    }
}

/// [`byte_dot`] on a pre-loaded (possibly pre-combined) 64-bit message. Lets
/// the tensor fold dot `m₀⊕m₁` without a memory round-trip.
#[inline]
pub(super) fn byte_dot_u64(v: u64, table: &[[F128; 256]]) -> F128 {
    unsafe {
        let t0 = *table.get_unchecked(0).get_unchecked((v & 0xFF) as usize);
        let t1 = *table
            .get_unchecked(1)
            .get_unchecked(((v >> 8) & 0xFF) as usize);
        let t2 = *table
            .get_unchecked(2)
            .get_unchecked(((v >> 16) & 0xFF) as usize);
        let t3 = *table
            .get_unchecked(3)
            .get_unchecked(((v >> 24) & 0xFF) as usize);
        let t4 = *table
            .get_unchecked(4)
            .get_unchecked(((v >> 32) & 0xFF) as usize);
        let t5 = *table
            .get_unchecked(5)
            .get_unchecked(((v >> 40) & 0xFF) as usize);
        let t6 = *table
            .get_unchecked(6)
            .get_unchecked(((v >> 48) & 0xFF) as usize);
        let t7 = *table.get_unchecked(7).get_unchecked((v >> 56) as usize);
        ((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7))
    }
}

/// Fold the witness's skip dimension at `r₁`, producing `â(r₁, rest)` for every
/// rest position — the AG analog of the RS `UniSkipFoldTable(z)` fold. Witness
/// order is skip = low 6 bits, rest = high, so rest position `r`'s 64-bit message
/// is the 8 bytes at offset `r*8`. Parallel byte-dot.
pub fn fold_witness_at_r1(packed: &[u8], w: &[F128]) -> Vec<F128> {
    assert_eq!(
        packed.len() % 8,
        0,
        "witness must be a whole number of 64-bit messages"
    );
    let table = build_w_tables(w);
    (0..packed.len() / 8)
        .into_par_iter()
        .map(|r| byte_dot(packed, r, &table))
        .collect()
}

/// One wide-Horner step: `acc << 2` (256-bit) then XOR the reduced product `p`
/// into the low 128 bits. Multiplying by `γ² = x²` is a pure 2-bit shift, so the
/// geometric inner sum `Σ (γ²)^i pᵢ` accumulates with no field multiplies and no
/// per-step reduction — the caller reduces the 256-bit total once. (A log-depth
/// XOR tree was tried and is slower: the chain is already hidden by across-block
/// parallelism, and the tree's 64-wide working set adds L1 traffic.)
#[inline]
pub(super) fn shl2_xor(acc: F256Unreduced, p: F128) -> F256Unreduced {
    F256Unreduced {
        r0: (acc.r0 << 2) ^ p.lo,
        r1: ((acc.r1 << 2) | (acc.r0 >> 62)) ^ p.hi,
        r2: (acc.r2 << 2) | (acc.r1 >> 62),
        r3: (acc.r3 << 2) | (acc.r2 >> 62),
    }
}

/// Fused fold + first multilinear message — the AG analog of RS's
/// `uni_skip_fold_and_round_pair`, parallel over the `2^(m-13)` outer blocks.
/// Each block folds its 128 messages at `r₁` (byte-dot) AND accumulates round 0's
/// `(G(1), G(∞))`.
///
/// The first round binds inner-bit-0; the *remaining* friendly dims
/// (`r_rest[1..N_INNER]`, the 64 inner positions per block) are `γ²`-geometric, so
/// their eq-weighted inner sum is a **Horner in `γ²`** — shift-reduce, no field
/// multiplies for the weighting. Only the **outer** eq is general
/// (`build_eq(r_rest[N_INNER..])`, `2^(m-13)` ≪ the full `2^(m-7)` table).
///
/// Requires `r_rest[1..N_INNER]` to be the geometric friendly constants
/// (`friendly_challenges()[1..]`); the caller always sets `r_rest = friendly ‖ outer`.
pub fn fold_and_first_round(
    a_packed: &[u8],
    b_packed: &[u8],
    w: &[F128],
    r_rest: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a_packed.len(), b_packed.len());
    debug_assert_eq!(
        &r_rest[1..N_INNER],
        &friendly_challenges()[1..N_INNER],
        "fold_and_first_round assumes γ-geometric friendly dims"
    );
    let n = a_packed.len() / 8;
    let table = build_w_tables(w);
    let eq_outer = super::univariate_skip::build_eq(&r_rest[N_INNER..]);
    let n_outer = eq_outer.len();
    assert_eq!(n, n_outer * 128, "each outer block is 128 (2^7) messages");
    // D₁ = ∏_{j=1}^{6}(1+γ^{2^j}); the eq over the remaining friendly dims is
    // γ^{2·int(inner')}/D₁, so the Horner (which omits 1/D₁) is rescaled by D₁⁻¹.
    let mut d1 = F128::ONE;
    for j in 1..N_INNER {
        d1 *= F128::ONE + gamma_pow(1usize << j);
    }
    let d1_inv = d1.inv();

    // Uninit alloc: the parallel loop below writes every slot of a_mlv/b_mlv, so
    // `vec![ZERO; n]`'s serial zero-fill (512 MB at m=30) is pure waste and caps
    // the multi-thread speedup (Amdahl) — the same reason RS's fold uses this.
    // Write-before-read contract upheld: every chunk's 128 slots are written.
    let mut a_mlv = crate::scratch::take_f128(n);
    let mut b_mlv = crate::scratch::take_f128(n);
    let (g1, g_inf) = a_mlv
        .par_chunks_mut(128)
        .zip(b_mlv.par_chunks_mut(128))
        .zip(eq_outer.par_iter())
        .enumerate()
        .map(|(outer, ((am, bm), &eo))| {
            let (s1, s_inf) = fold_block_at(a_packed, b_packed, outer * 128, &table, am, bm);
            let e = eo * d1_inv;
            (e * s1.reduce(), e * s_inf.reduce())
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (r, s)| (p + r, q + s));
    (a_mlv, b_mlv, g1, g_inf)
}

/// One outer block of the fused fold + first-round accumulation — one fused
/// pass over the 64 pairs (reverse, for the γ²-Horner): fold the four rows
/// at `r₁` via the byte-dot `table`, write them to `am`/`bm`, AND
/// accumulate the first message from those just-folded values — no
/// read-back of the folded array (the two-pass version re-read 2 KB/block).
/// γ² = x², so each Σ weight is a pure 2-bit shift: a *wide* Horner on a
/// 256-bit accumulator (no per-step reduction — max degree 2·63+127 = 253 <
/// 256), reduced once per block by the caller. `msg_base` is the block's
/// first 64-bit-message index in `a_src`/`b_src` — a cleansed single-block
/// scratch buffer passes 0.
#[inline]
fn fold_block_at(
    a_src: &[u8],
    b_src: &[u8],
    msg_base: usize,
    table: &[[F128; 256]],
    am: &mut [F128],
    bm: &mut [F128],
) -> (F256Unreduced, F256Unreduced) {
    let mut s1 = F256Unreduced::ZERO;
    let mut s_inf = F256Unreduced::ZERO;
    for inner in (0..64).rev() {
        let (i0, i1) = (2 * inner, 2 * inner + 1);
        let a0 = byte_dot(a_src, msg_base + i0, table);
        let a1 = byte_dot(a_src, msg_base + i1, table);
        let b0 = byte_dot(b_src, msg_base + i0, table);
        let b1 = byte_dot(b_src, msg_base + i1, table);
        am[i0] = a0;
        am[i1] = a1;
        bm[i0] = b0;
        bm[i1] = b1;
        s1 = shl2_xor(s1, a1 * b1);
        s_inf = shl2_xor(s_inf, (a0 + a1) * (b0 + b1));
    }
    (s1, s_inf)
}

/// [`fold_and_first_round`] under a witness run-list ([`BlockCoverage`] at
/// the 8192-bit code-block grid, one entry per outer block): DEAD blocks
/// write zero folds and contribute nothing (their honest bits are all
/// zero), FULL blocks run the in-place fused pass, and PARTIAL blocks are
/// cleansed into zeroed scratch first — no declared-dead bit is ever read,
/// so `PooledDirty` witnesses are legal. Value-identical to the dense fold
/// on an honestly zero-padded witness.
pub fn fold_and_first_round_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    w: &[F128],
    r_rest: &[F128],
    coverage: &[super::BlockCoverage],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a_packed.len(), b_packed.len());
    debug_assert_eq!(
        &r_rest[1..N_INNER],
        &friendly_challenges()[1..N_INNER],
        "fold_and_first_round assumes γ-geometric friendly dims"
    );
    let n = a_packed.len() / 8;
    let table = build_w_tables(w);
    let eq_outer = super::univariate_skip::build_eq(&r_rest[N_INNER..]);
    let n_outer = eq_outer.len();
    assert_eq!(n, n_outer * 128, "each outer block is 128 (2^7) messages");
    assert_eq!(
        coverage.len(),
        n_outer,
        "one coverage entry per outer block"
    );
    let mut d1 = F128::ONE;
    for j in 1..N_INNER {
        d1 *= F128::ONE + gamma_pow(1usize << j);
    }
    let d1_inv = d1.inv();

    let mut a_mlv = crate::scratch::take_f128(n);
    let mut b_mlv = crate::scratch::take_f128(n);
    let (g1, g_inf) = a_mlv
        .par_chunks_mut(128)
        .zip(b_mlv.par_chunks_mut(128))
        .zip(eq_outer.par_iter())
        .enumerate()
        .map(|(outer, ((am, bm), &eo))| match &coverage[outer] {
            super::BlockCoverage::Dead => {
                am.fill(F128::ZERO);
                bm.fill(F128::ZERO);
                (F128::ZERO, F128::ZERO)
            }
            super::BlockCoverage::Full => {
                let (s1, s_inf) = fold_block_at(a_packed, b_packed, outer * 128, &table, am, bm);
                let e = eo * d1_inv;
                (e * s1.reduce(), e * s_inf.reduce())
            }
            super::BlockCoverage::Partial(ranges) => {
                let mut a_buf = [0u8; 1024];
                let mut b_buf = [0u8; 1024];
                super::cleanse_block(a_packed, outer * 1024, ranges, &mut a_buf);
                super::cleanse_block(b_packed, outer * 1024, ranges, &mut b_buf);
                let (s1, s_inf) = fold_block_at(&a_buf, &b_buf, 0, &table, am, bm);
                let e = eo * d1_inv;
                (e * s1.reduce(), e * s_inf.reduce())
            }
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (r, s)| (p + r, q + s));
    (a_mlv, b_mlv, g1, g_inf)
}

/// [`fold_and_first_round_padded`] with LIVE-SPAN output — the AG analog of
/// RS's `uni_skip_fold_and_round_pair_runs_sparse`: dead blocks get no
/// storage at all, so the fold's cost AND footprint are count-derived. The
/// returned [`LiveLayout`] maps the compacted buffers back to the padded
/// `2^(m−6)` domain: the k-th live block's 128 slots sit at offset `128·k`.
/// Intervals are 128-aligned by construction — pair-aligned for every tail
/// round. Message and live values are byte-identical to the dense fold on
/// an honestly zero-padded witness; a Partial block stores its full 128
/// slots (the cleansed fold writes honest zeros past its ranges).
pub fn fold_and_first_round_sparse(
    a_packed: &[u8],
    b_packed: &[u8],
    w: &[F128],
    r_rest: &[F128],
    coverage: &[super::BlockCoverage],
) -> (Vec<F128>, Vec<F128>, F128, F128, LiveLayout) {
    assert_eq!(a_packed.len(), b_packed.len());
    debug_assert_eq!(
        &r_rest[1..N_INNER],
        &friendly_challenges()[1..N_INNER],
        "fold_and_first_round assumes γ-geometric friendly dims"
    );
    let table = build_w_tables(w);
    let eq_outer = super::univariate_skip::build_eq(&r_rest[N_INNER..]);
    let n_outer = eq_outer.len();
    assert_eq!(a_packed.len() / 8, n_outer * 128, "128 messages per block");
    assert_eq!(
        coverage.len(),
        n_outer,
        "one coverage entry per outer block"
    );
    let mut d1 = F128::ONE;
    for j in 1..N_INNER {
        d1 *= F128::ONE + gamma_pow(1usize << j);
    }
    let d1_inv = d1.inv();

    // The live block list + its 128-aligned mlv-slot intervals.
    let live: Vec<usize> = (0..n_outer)
        .filter(|&o| !matches!(coverage[o], super::BlockCoverage::Dead))
        .collect();
    debug_assert!(!live.is_empty(), "a witness with no live block");
    let mut intervals: Vec<(usize, usize)> = Vec::new();
    for &o in &live {
        match intervals.last_mut() {
            Some((_, e)) if *e == 128 * o => *e = 128 * (o + 1),
            _ => intervals.push((128 * o, 128 * (o + 1))),
        }
    }
    let store = LiveLayout::new(intervals);
    let n_live = 128 * live.len();
    let mut a_mlv = crate::scratch::take_f128(n_live);
    let mut b_mlv = crate::scratch::take_f128(n_live);
    let (g1, g_inf) = a_mlv
        .par_chunks_mut(128)
        .zip(b_mlv.par_chunks_mut(128))
        .zip(live.par_iter())
        .map(|((am, bm), &outer)| {
            let (s1, s_inf) = match &coverage[outer] {
                super::BlockCoverage::Dead => unreachable!("the live list has no dead entry"),
                super::BlockCoverage::Full => {
                    fold_block_at(a_packed, b_packed, outer * 128, &table, am, bm)
                }
                super::BlockCoverage::Partial(ranges) => {
                    let mut a_buf = [0u8; 1024];
                    let mut b_buf = [0u8; 1024];
                    super::cleanse_block(a_packed, outer * 1024, ranges, &mut a_buf);
                    super::cleanse_block(b_packed, outer * 1024, ranges, &mut b_buf);
                    fold_block_at(&a_buf, &b_buf, 0, &table, am, bm)
                }
            };
            let e = eq_outer[outer] * d1_inv;
            (e * s1.reduce(), e * s_inf.reduce())
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (r, s)| (p + r, q + s));
    a_mlv.truncate(n_live);
    b_mlv.truncate(n_live);
    (a_mlv, b_mlv, g1, g_inf, store)
}

/// One wide-Horner step with a compile-time shift `S` (the friendly base
/// `γ^{2^{i+1}} = x^S`): `acc << S` (256-bit) then XOR the reduced product `p`
/// into the low 128 bits. Generalizes [`shl2_xor`] (S=2, round 0) to rounds
/// 1..5, where the *remaining* friendly dims are `(γ^{2^{i+1}})`-geometric.
/// `S ∈ {4,8,16,32,64}` — the `S == 64` case is split out because `u64 << 64`
/// is UB (and the const generic makes the branch compile away).
#[inline]
pub(super) fn shl_xor<const S: u32>(acc: F256Unreduced, p: F128) -> F256Unreduced {
    if S == 64 {
        F256Unreduced {
            r0: p.lo,
            r1: acc.r0 ^ p.hi,
            r2: acc.r1,
            r3: acc.r2,
        }
    } else {
        let inv = 64 - S;
        F256Unreduced {
            r0: (acc.r0 << S) ^ p.lo,
            r1: ((acc.r1 << S) | (acc.r0 >> inv)) ^ p.hi,
            r2: (acc.r2 << S) | (acc.r1 >> inv),
            r3: (acc.r3 << S) | (acc.r2 >> inv),
        }
    }
}

/// `norm_i = (∏_{j=i+1}^{6}(1+γ^{2^j}))⁻¹` — the round-`i` analog of
/// `fold_and_first_round`'s `d1_inv` (which is `norm_0`). The friendly eq over
/// the message's remaining dims (`vars i+1..6`) is `norm_i · (γ^{2^{i+1}})^p`,
/// so the Horner (which omits `norm_i`) is rescaled by it.
fn friendly_norm(i: usize) -> F128 {
    let mut d = F128::ONE;
    for j in (i + 1)..N_INNER {
        d *= F128::ONE + gamma_pow(1usize << j);
    }
    d.inv()
}

/// Fused fold-by-`rho` + friendly-Horner round message — the rounds-1..5 analog
/// of [`fold_and_first_round`], and a drop-in (bit-identical output) replacement
/// for the general [`super::multilinear::fold_and_compute_round_pair_into`] on
/// those rounds.
///
/// After binding friendly bits `0..i` and folding bit `i-1` by `rho`, the round-`i`
/// message's free index splits as `[friendly i+1..6 (low, lo_size = 2^{6-i}) |
/// outer (high, 2^{m-13})]`. The friendly lo dims are `(γ^{2^{i+1}})`-geometric,
/// so their eq-weighted inner sum is a **Horner in `x^SHIFT`** (`SHIFT = 2^{i+1}`,
/// a pure shift — no per-term PMULL), where the general kernel pays 2 `eq_lo`
/// PMULLs per term. The outer hi dims keep the general split-eq multiply
/// (`eq_outer_lo`/`eq_outer_hi`, constant across rounds 1..5). `c_inv = norm_i`
/// (see [`friendly_norm`]) rescales the Horner.
///
/// Folds `a→a_out`, `b→b_out` by `rho`; returns the bare `(G(1), G(∞))`
/// (Convention A). Parallel over the outer-hi dim.
#[allow(clippy::too_many_arguments)]
fn fold_and_friendly_round_pair_into<const SHIFT: u32>(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho: F128,
    lo_size: usize,
    eq_outer_lo: &[F128],
    eq_outer_hi: &[F128],
    c_inv: F128,
) -> (F128, F128) {
    let half = a.len() / 2;
    debug_assert_eq!(a_out.len(), half);
    debug_assert_eq!(b_out.len(), half);
    let n_ol = eq_outer_lo.len();
    let n_oh = eq_outer_hi.len();
    let block_out = 2 * lo_size; // a_out entries per friendly block (incl. bound var)
    let block_in = 2 * block_out; // a entries per friendly block (pre-fold)
    debug_assert_eq!(block_out * n_ol * n_oh, half);
    let chunk_out = n_ol * block_out;
    let chunk_in = n_ol * block_in;

    let (sum1, sum_inf) = a_out
        .par_chunks_mut(chunk_out)
        .zip(b_out.par_chunks_mut(chunk_out))
        .enumerate()
        .map(|(oh, (a_oc, b_oc))| {
            let a_ic = &a[oh * chunk_in..(oh + 1) * chunk_in];
            let b_ic = &b[oh * chunk_in..(oh + 1) * chunk_in];
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            for ol in 0..n_ol {
                let ib = ol * block_in;
                let ob = ol * block_out;
                let mut s1 = F256Unreduced::ZERO;
                let mut sinf = F256Unreduced::ZERO;
                // Reverse for the Horner: the first-processed pair carries the
                // highest power of ω = x^SHIFT.
                for p in (0..lo_size).rev() {
                    let i0 = ib + 4 * p;
                    // LSB fold by rho (binds the previous round's variable).
                    let a0 = a_ic[i0] + rho * (a_ic[i0 + 1] + a_ic[i0]);
                    let a1 = a_ic[i0 + 2] + rho * (a_ic[i0 + 3] + a_ic[i0 + 2]);
                    let b0 = b_ic[i0] + rho * (b_ic[i0 + 1] + b_ic[i0]);
                    let b1 = b_ic[i0 + 2] + rho * (b_ic[i0 + 3] + b_ic[i0 + 2]);
                    let o = ob + 2 * p;
                    a_oc[o] = a0;
                    a_oc[o + 1] = a1;
                    b_oc[o] = b0;
                    b_oc[o + 1] = b1;
                    s1 = shl_xor::<SHIFT>(s1, a1 * b1);
                    sinf = shl_xor::<SHIFT>(sinf, (a0 + a1) * (b0 + b1));
                }
                let el = eq_outer_lo[ol];
                p1_acc ^= el.mul_unreduced(s1.reduce());
                pinf_acc ^= el.mul_unreduced(sinf.reduce());
            }
            let eh = eq_outer_hi[oh] * c_inv;
            (eh * p1_acc.reduce(), eh * pinf_acc.reduce())
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(x, y), (u, v)| (x + u, y + v));
    (sum1, sum_inf)
}

// ---------------------------------------------------------------------------
// Rounds 2..m: the multilinear sumcheck over the rest variables (AB only — c
// dropped out in round 1). Reuses `multilinear::{round_pair_naive,
// fold_in_place_pair}`; the round structure mirrors the RS path exactly.
// ---------------------------------------------------------------------------

/// Prover's multilinear sumcheck on the folded rows `â(r₁,·)`, `b̂(r₁,·)`. Binds
/// each rest variable (low-bit first) to the supplied challenge `rhos[i]`,
/// emitting `(G(1), G(∞))` per round (Convention A — no eq factor on the wire).
/// `r_rest` are the eq weights (friendly inner, then outer). Returns the
/// per-round messages and the final evals `(â(r₁,ρ), b̂(r₁,ρ))`.
pub fn prove_multilinear(
    mut a_mlv: Vec<F128>,
    mut b_mlv: Vec<F128>,
    r_rest: &[F128],
    rhos: &[F128],
) -> (Vec<(F128, F128)>, F128, F128) {
    let n_mlv = r_rest.len();
    assert_eq!(rhos.len(), n_mlv);
    assert_eq!(a_mlv.len(), 1usize << n_mlv);
    let mut msgs = Vec::with_capacity(n_mlv);

    // Round 0: message from the full rows (no fold yet) — the biggest round, so
    // compute it with a parallel reduce (Convention A: bare G(1), since the
    // bound var's eq factor r[0]=ONE). `eq` weights the remaining vars.
    {
        use rayon::prelude::*;
        let half = a_mlv.len() / 2;
        let eq = super::univariate_skip::build_eq(&r_rest[1..]);
        let (g1, g_inf) = (0..half)
            .into_par_iter()
            .map(|x| {
                let (a0, a1) = (a_mlv[2 * x], a_mlv[2 * x + 1]);
                let (b0, b1) = (b_mlv[2 * x], b_mlv[2 * x + 1]);
                let e = eq[x];
                (e * a1 * b1, e * (a0 + a1) * (b0 + b1))
            })
            .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (r, s)| (p + r, q + s));
        msgs.push((g1, g_inf));
    }

    // Rounds 1..n_mlv: fuse the fold by ρ_{i-1} with this round's message. Use
    // the parallel fused path while ≥10 vars remain; below that, naive. Ping-pong
    // scratch avoids a fresh alloc+free per round.
    let n_in = a_mlv.len();
    let (mut a_nxt, mut b_nxt) = if n_in >= 1024 {
        // Uninit pooled buffers (NOT vec![ZERO; n_in/2]): the fused fold writes
        // every slot of a_nxt/b_nxt[..half] before reading (write-before-read),
        // so a zero-fill is wasted — and it's a *serial* 256 MB memset before the
        // parallel loop (Amdahl), the same trap `a_mlv`/`b_mlv` avoid via the pool.
        (
            crate::scratch::take_f128(n_in / 2),
            crate::scratch::take_f128(n_in / 2),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    for i in 1..n_mlv {
        let rho_prev = rhos[i - 1];
        let log_before = a_mlv.len().trailing_zeros() as usize;
        let mut r_next = vec![F128::ONE; log_before - 1];
        r_next[1..].copy_from_slice(&r_rest[i + 1..]);
        let pair = if log_before >= 10 {
            let half = a_mlv.len() / 2;
            let pair = fold_and_compute_round_pair_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                rho_prev,
                &r_next,
            );
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            pair
        } else {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            round_pair_naive(&a_mlv, &b_mlv, &r_next)
        };
        msgs.push(pair);
    }

    // Final binding by the last challenge.
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rhos[n_mlv - 1]);
    (msgs, a_mlv[0], b_mlv[0])
}

/// Verifier's multilinear consistency: propagate the running inner claim from the
/// round-1 value `claim_ab` through every round, reconstructing `G(0)` from the
/// eq weight and interpolating at `ρ`. Returns the final running claim, which the
/// caller checks equals `a_eval · b_eval`.
pub fn verify_multilinear(
    claim_ab: F128,
    r_rest: &[F128],
    msgs: &[(F128, F128)],
    rhos: &[F128],
) -> F128 {
    let mut c = claim_ab;
    for i in 0..r_rest.len() {
        let (g1, g_inf) = msgs[i];
        let r_eq = r_rest[i];
        // c = (1+r_eq)·G(0) + r_eq·G(1)  ⇒  G(0) = (c + r_eq·G(1))·(1+r_eq)⁻¹.
        let g0 = (c + r_eq * g1) * (F128::ONE + r_eq).inv();
        let rho = rhos[i];
        // G(ρ) = G(0)(1+ρ) + G(1)ρ + G(∞)ρ(ρ+1).
        c = g0 * (F128::ONE + rho) + g1 * rho + g_inf * rho * (rho + F128::ONE);
    }
    c
}

// ---------------------------------------------------------------------------
// Fiat–Shamir driver: prove / verify against a SHA-256 `Challenger`.
// ---------------------------------------------------------------------------

/// A fixed, valid point of `Y`, the fallback for the (≈`2^-289`) chance that
/// Zero-count bound behind the r₁ grinding budget: a false round-1 message
/// pair survives at r₁ only if a nonzero PRODUCT-code word (a function in
/// `L(2D)`, at most `deg 2D = 316` zeros — the ab check) or a nonzero
/// BASE-code word (`L(D)`, at most `deg D = 158` zeros — the c check)
/// vanishes there. `316 + 158 = 474` bad points over the ~`2^128` valid
/// cover points; `bits_for(474) = 9` total bits required.
pub const R1_ZERO_BOUND: usize = 474;

/// Provable grinding contribution of the rejection sampler itself:
/// `log2(BASE_Y_DEGREE · 2^3) = log2(32) = 5` bits. The sampler weights
/// every reachable cover point at exactly `1/(2^128 · 32)` (slot flattening
/// over the degree-4 base fiber, three all-or-nothing Artin–Schreier levels
/// with uniform choice bits), and Hasse–Weil pins the genus-95 cover's point
/// count to `2^128 · (1 ± 2^-56)` — so each valid draw provably costs
/// ≥ ~32 attempts. A protocol constant (the sampler's shape), NOT an
/// empirical acceptance estimate; guarded by `credit_constants_are_pinned`.
pub const AG_SAMPLING_CREDIT_BITS: u32 = 5;

/// Explicit PoW bits on the FUSED r₁ nonce under a strict grinding
/// schedule: ALL of `bits_for(R1_ZERO_BOUND) = 9` — the sampler's
/// [`AG_SAMPLING_CREDIT_BITS`] = 5 no longer discount r₁'s explicit bits.
/// WHY (phase D, docs/ag-recursion-plan.md): the recursion circuit binds
/// the r₁ decode with RELAXED canonicity — it enforces `x` from the
/// nonce-seed's XOF and fiber membership for `(y, z₁..z₃)`, but not WHICH
/// fiber point (the ≤ 32-point fiber = exactly the sampler's 5 flattening
/// bits). A circuit-side prover may therefore choose among the fiber
/// points, so those 5 bits are repaid explicitly; the total stays
/// `bits_for(474)`, the split moves. The sampler's credit still stands
/// where the decode is canonical end-to-end (the lincheck fresh skip).
pub const R1_POW_BITS: u32 = 9;

/// Prover-side scan budget for the fused r₁ nonce: success per nonce is
/// `2^-R1_POW_BITS / 32 = 2^-14`, so exhausting `2^24` nonces has
/// probability `(1 − 2^-14)^(2^24) ≈ 2^-1477` — a completeness error we
/// accept (panic). Fits the 4-byte nonce encoding with room.
pub const R1_FUSED_ATTEMPT_BUDGET: u32 = 1 << 24;

/// rejection sampling exhausts its attempt budget. Baked from one offline sample.
///
/// No longer used by the AG-skip `r₁` derivation (the nonce-grind path panics
/// on exhaustion instead — an accepted fallback claim would let a cheating
/// prover pin `r₁` to this public constant). Still backs
/// [`crate::lincheck::SkipPoint::sample_fresh`], where BOTH sides replay the
/// same deterministic loop, so a prover cannot claim failure unilaterally:
/// forcing it costs `~1/P_fail ≈ 2^916` (acceptance is 1/32 per attempt, so
/// `P_fail = (1 − 1/32)^20000`), far above the `2^128` target. (Keep linked
/// to the sampler's attempt cap; lowering the cap raises `P_fail`.)
pub(crate) fn fallback_point() -> EvaluationPoint {
    EvaluationPoint {
        x: F128 {
            lo: 17376575154980521683,
            hi: 16956381729448392221,
        },
        y: F128 {
            lo: 3703262331298828801,
            hi: 354800370152310762,
        },
        z1: F128 {
            lo: 12509146339372511806,
            hi: 14754238706087384589,
        },
        z2: F128 {
            lo: 7848999207301165946,
            hi: 6313416638638821140,
        },
        z3: F128 {
            lo: 4302402807812267211,
            hi: 11246714171241606505,
        },
    }
}

/// Squeeze the 32-byte SHA-256 seed for `r₁` from the transcript (two `F128`
/// samples). Shared by the prover's nonce grind and the verifier's replay.
fn r1_seed<C: Challenger>(challenger: &mut C) -> [u8; 32] {
    challenger.observe_label(b"flock-ag-skip-r1-point");
    let s0 = challenger.sample_f128();
    let s1 = challenger.sample_f128();
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&s0.lo.to_le_bytes());
    seed[8..16].copy_from_slice(&s0.hi.to_le_bytes());
    seed[16..24].copy_from_slice(&s1.lo.to_le_bytes());
    seed[24..32].copy_from_slice(&s1.hi.to_le_bytes());
    seed
}

/// Bind the chosen `r₁` nonce into the transcript. Must happen before any
/// later challenge is sampled — ξ̂ and the tail ρ's depend on `r₁` through it.
fn observe_r1_nonce<C: Challenger>(challenger: &mut C, nonce: u32) {
    challenger.observe_label(b"flock-ag-skip-r1-nonce");
    challenger.observe_bytes(&nonce.to_le_bytes());
}

/// Prover: derive `r₁` by GRINDING a nonce, instead of the verifier replaying
/// the whole rejection-sampling loop. Each nonce seeds an independent
/// one-attempt DRBG `SHA256(seed ‖ nonce)`
/// ([`crate::genus95_curve_code::evaluation_point_from_nonce`]); the first
/// nonce whose attempt lands a valid point is observed into the transcript and
/// shipped in the proof, so the verifier re-derives `r₁` with a SINGLE attempt
/// rather than the expected ~32 (worst-case 20 000) replayed ones.
///
/// Per-attempt acceptance is 1/32 (measured 3.141% over 10⁶ attempts;
/// structurally `1/(BASE_Y_DEGREE · 2^3)`), so exhausting all
/// [`SAMPLE_ATTEMPT_BUDGET`](crate::genus95_curve_code::SAMPLE_ATTEMPT_BUDGET)
/// nonces has probability ~2⁻⁹²¹ — a completeness error we accept (panic)
/// instead of a verifier-side fallback point: an unconditionally accepted
/// fallback claim would let a cheating prover fix `r₁` to a public constant.
///
/// UNGRINDED path only (the direct route / disabled schedules, which make no
/// 128-bit claim): the verifier's single attempt lets a cheating prover pick
/// among the ~630 expected valid nonces in the budget — ≤ log₂(20 000) ≈ 14.3
/// bits of freedom. Strict schedules use [`sample_r1_prover_pow`] instead.
pub(super) fn sample_r1_prover<C: Challenger>(challenger: &mut C) -> (EvaluationPoint, u32) {
    let seed = r1_seed(challenger);
    let kind = challenger.hash_kind();
    for nonce in 0..crate::genus95_curve_code::SAMPLE_ATTEMPT_BUDGET {
        if let Some(point) =
            crate::genus95_curve_code::evaluation_point_from_nonce(&seed, nonce, kind)
        {
            observe_r1_nonce(challenger, nonce);
            return (point, nonce);
        }
    }
    unreachable!("r1 nonce grind exhausted its budget (probability ~2^-921)")
}

/// [`sample_r1_prover`] under a strict grinding schedule: the FUSED nonce —
/// `H(seed ‖ nonce)` must clear `pow_bits` of PoW AND decode to a valid
/// cover point, both criteria on the same hash. Every candidate the prover
/// (or an attacker) evaluates re-enters the PoW, so there is no free choice
/// among valid nonces; with the sampler's provable
/// [`AG_SAMPLING_CREDIT_BITS`] = 5 bits on top, `r₁` carries
/// `pow_bits + 5` grinding bits total against a CANONICAL decode — the
/// strict schedule sets `pow_bits =` [`R1_POW_BITS`] = 9 so the budget
/// holds even against the recursion circuit's relaxed-canonicity decode,
/// which returns the fiber's 5 bits to the prover. Expected prover cost is
/// `2^pow_bits · 32` hash calls plus `2^pow_bits` point attempts (~1 ms at
/// 9 bits); the verifier stays ONE-SHOT.
pub(super) fn sample_r1_prover_pow<C: Challenger>(
    challenger: &mut C,
    pow_bits: u32,
) -> (EvaluationPoint, u32) {
    let seed = r1_seed(challenger);
    let kind = challenger.hash_kind();
    for nonce in 0..R1_FUSED_ATTEMPT_BUDGET {
        if let Some(point) =
            crate::genus95_curve_code::evaluation_point_from_nonce_pow(&seed, nonce, kind, pow_bits)
        {
            observe_r1_nonce(challenger, nonce);
            return (point, nonce);
        }
    }
    unreachable!("fused r1 nonce grind exhausted its budget (see R1_FUSED_ATTEMPT_BUDGET)")
}

/// Verifier: re-derive `r₁` from the proof's nonce — range-check it, run the
/// single attempt, and reject unless it lands a valid point. The verifier does
/// NOT check the nonce is the prover's minimal one, so a cheating prover may
/// pick any of the ~630 expected valid nonces in the budget: ≤ log₂(20 000) ≈
/// 14.3 bits of freedom over `r₁`. UNGRINDED path only — strict schedules
/// verify the fused nonce via [`replay_r1_verifier_pow`], which has no such
/// freedom (every candidate carries its own PoW).
pub(super) fn replay_r1_verifier<C: Challenger>(
    challenger: &mut C,
    nonce: u32,
) -> Result<EvaluationPoint, AgVerifyError> {
    let seed = r1_seed(challenger);
    if nonce >= crate::genus95_curve_code::SAMPLE_ATTEMPT_BUDGET {
        return Err(AgVerifyError::BadR1Nonce { nonce });
    }
    observe_r1_nonce(challenger, nonce);
    crate::genus95_curve_code::evaluation_point_from_nonce(&seed, nonce, challenger.hash_kind())
        .ok_or(AgVerifyError::BadR1Nonce { nonce })
}

/// Verifier mirror of [`sample_r1_prover_pow`]: ONE hash + ONE point attempt,
/// rejecting unless the fused nonce clears the PoW target AND lands a valid
/// point. Constant-shape — no rejection replay, no data-dependent loop.
pub(super) fn replay_r1_verifier_pow<C: Challenger>(
    challenger: &mut C,
    nonce: u32,
    pow_bits: u32,
) -> Result<EvaluationPoint, AgVerifyError> {
    let seed = r1_seed(challenger);
    if nonce >= R1_FUSED_ATTEMPT_BUDGET {
        return Err(AgVerifyError::BadR1Nonce { nonce });
    }
    observe_r1_nonce(challenger, nonce);
    crate::genus95_curve_code::evaluation_point_from_nonce_pow(
        &seed,
        nonce,
        challenger.hash_kind(),
        pow_bits,
    )
    .ok_or(AgVerifyError::BadR1Nonce { nonce })
}

/// All round messages the AG-skip prover sends, in order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgProof {
    /// Round 1: `P^{ab}` fresh coords (158).
    pub round1_ab: Vec<F128>,
    /// Round 1: the folded c message `w̄` (64).
    pub round1_c: Vec<F128>,
    /// Grinding nonce for `r₁`: the verifier re-derives the point from
    /// `SHA256(seed ‖ nonce)` in one attempt (see [`replay_r1_verifier`]).
    pub r1_nonce: u32,
    /// Multilinear rounds: `(G(1), G(∞))` each; length `m − K_SKIP`.
    pub multilinear_rounds: Vec<(F128, F128)>,
    pub final_a_eval: F128,
    pub final_b_eval: F128,
    pub final_c_eval: F128,
    /// PoW nonces in transcript order: the initial outer-eq point, then one
    /// per multilinear round. Empty under [`ZerocheckGrinding::disabled`]
    /// (the direct-route entries and every pre-grinding proof). `r₁` keeps
    /// its own rejection-sampling nonce (`r1_nonce`) regardless.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// Evaluation claims the AG-skip zerocheck reduces to (no PCS). `a_eval`, `b_eval`
/// are at the AG point `r₁` with rest coords `mlv_challenges`; `c_eval` is at `r₁`
/// with rest coords `r_rest` (a *different* point — c dropped out in round 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgClaim {
    pub r1: EvaluationPoint,
    pub mlv_challenges: Vec<F128>,
    pub r_rest: Vec<F128>,
    pub a_eval: F128,
    pub b_eval: F128,
    pub c_eval: F128,
}

/// Verifier rejection reasons.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgVerifyError {
    LogNTooSmall {
        log_n: usize,
    },
    BadRound1Length {
        which: &'static str,
        expected: usize,
        got: usize,
    },
    BadMultilinearRoundsLength {
        expected: usize,
        got: usize,
    },
    /// The claimed `r₁` nonce is out of budget or its attempt rejects.
    BadR1Nonce {
        nonce: u32,
    },
    CEvalMismatch,
    SumcheckFinalFailed,
    /// The nonce vector does not match the configured grinding schedule.
    BadGrindingNonceCount {
        expected: usize,
        got: usize,
    },
    /// A nonce fails the PoW at the FS position sampling its challenge.
    InvalidGrindingNonce,
}

/// Challenger adapter for the multilinear tail under per-round grinding:
/// every `sample_f128` in the tail IS a round challenge, so the adapter
/// grinds `bits` before each and records the nonce — the tail's kernels stay
/// untouched. Framing-sensitive ops delegate verbatim.
struct RoundGrindProver<'a, C: Challenger> {
    inner: &'a mut C,
    bits: u32,
    nonces: &'a mut Vec<u64>,
}

impl<C: Challenger> Challenger for RoundGrindProver<'_, C> {
    fn supports_fused_pow_squeeze(&self) -> bool {
        self.inner.supports_fused_pow_squeeze()
    }
    fn hash_kind(&self) -> crate::merkle::HashKind {
        self.inner.hash_kind()
    }
    fn observe_label(&mut self, label: &[u8]) {
        self.inner.observe_label(label)
    }
    fn observe_f128(&mut self, value: F128) {
        self.inner.observe_f128(value)
    }
    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.inner.observe_f128_slice(values)
    }
    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.inner.observe_bytes(bytes)
    }
    fn sample_f128(&mut self) -> F128 {
        let (nonce, r) = self.inner.grind_pow_and_sample_f128(self.bits);
        self.nonces.push(nonce);
        r
    }
    fn sample_f128_vec(&mut self, _n: usize) -> Vec<F128> {
        // Fail loudly rather than silently skip the per-round grind: a tail
        // change that vector-squeezes must extend the adapter first.
        unreachable!("the grinding tail adapter never vector-squeezes")
    }
    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.inner.grind_pow(bits)
    }
    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.inner.verify_pow(nonce, bits)
    }
    fn fork_from_seed(&self, _seed: [F128; 2], _label: &'static [u8]) -> Self {
        unreachable!("the grinding tail adapter is never forked")
    }
}

/// Nonce count [`AgProof::grinding_nonces`] must carry for `m` under a
/// schedule: the initial outer-eq point + one per multilinear round. (`r₁`'s
/// nonce — fused PoW+sampling under a strict schedule, plain sampling
/// otherwise — is the separate [`AgProof::r1_nonce`] field.)
pub fn grinding_nonce_count(grinding: super::ZerocheckGrinding, m: usize) -> usize {
    usize::from(grinding.initial_bits(m).is_some())
        + usize::from(grinding.multilinear_round_bits().is_some()) * (m - K_SKIP)
}

/// Prove `a(y)·b(y) = c(y)` for all `y ∈ {0,1}^m` via the AG-skip zerocheck.
/// `a/b/c_packed` are LSB-first bit-packed (length `2^m/8`); requires `m ≥ 13`.
#[cfg(target_arch = "aarch64")]
pub fn prove<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    challenger: &mut C,
) -> (AgProof, AgClaim) {
    assert!(m >= K_SKIP + N_INNER, "m >= 13 required");
    let expected = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected);
    assert_eq!(b_packed.len(), expected);
    assert_eq!(c_packed.len(), expected);

    challenger.observe_label(b"flock-ag-skip-v1");
    let r_outer = challenger.sample_f128_vec(m - K_SKIP - N_INNER);
    let eq = crate::zerocheck::univariate_skip::build_eq(&r_outer);

    let msg = prove_round1(a_packed, b_packed, c_packed, &eq);
    prove_from_round1(
        a_packed,
        b_packed,
        msg,
        &r_outer,
        super::ZerocheckGrinding::disabled(),
        Vec::new(),
        None,
        challenger,
    )
}

/// [`prove`] that ALSO returns the c-claim's `s_hat_v_c` — the length-128
/// ring-switch fold the PCS open would otherwise recompute via `fold_1b_rows`
/// on `z_packed` at the c-claim's suffix. Captured as a near-free byproduct of
/// round 1's c-scan (the two-bank [`round1::round1_slp_packed_banks`]), so the
/// open skips the c witness scan — the AG analog of the RS path's
/// `prove_packed_padded_capture_s_hat_v_c`. The returned `AgProof`/`AgClaim` are
/// byte-identical to [`prove`]'s (`bank0 + bank1 == wbar`).
///
/// `s_hat_v_c` is in canonical (`fold_1b_rows`) form: layout
/// `s_hat_v_c[skip + b·64]` for `skip ∈ [0,64)`, `b ∈ {0,1}` the 7th packing
/// bit, with the friendly-bit-0 `eq` factor stripped (re-applied by
/// `build_claim_weights_from_skip` via `eq(x_outer[0])`).
#[cfg(target_arch = "aarch64")]
pub fn prove_capture_s_hat_v_c<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    challenger: &mut C,
) -> (AgProof, AgClaim, Vec<F128>) {
    assert!(m >= K_SKIP + N_INNER, "m >= 13 required");
    let expected = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected);
    assert_eq!(b_packed.len(), expected);
    assert_eq!(c_packed.len(), expected);

    challenger.observe_label(b"flock-ag-skip-v1");
    let r_outer = challenger.sample_f128_vec(m - K_SKIP - N_INNER);
    let eq = crate::zerocheck::univariate_skip::build_eq(&r_outer);

    let (msg, s_hat_v_c) = prove_round1_banks(a_packed, b_packed, c_packed, &eq);
    let (proof, claim) = prove_from_round1(
        a_packed,
        b_packed,
        msg,
        &r_outer,
        super::ZerocheckGrinding::disabled(),
        Vec::new(),
        None,
        challenger,
    );
    (proof, claim, s_hat_v_c)
}

/// [`prove_capture_s_hat_v_c`] under a Fiat--Shamir grinding schedule — the
/// UNION route's strict-profile entry. Grinds the initial outer-eq point
/// (`initial_bits(m)`) and every multilinear round
/// (`multilinear_round_bits`), mirroring the RS zerocheck's schedule; `r₁`
/// switches to the FUSED nonce ([`sample_r1_prover_pow`]):
/// [`R1_POW_BITS`] = 9 explicit PoW bits = `bits_for(474)` for the
/// product+base code bad set ([`R1_ZERO_BOUND`]) — all explicit, so the
/// budget survives the recursion circuit's relaxed-canonicity decode —
/// with a ONE-SHOT verifier.
///
/// PADDING CONTRACT: this entry is run-list READ-EXACT — see the owning
/// statement on [`crate::proof::BooleanPiopProofAg`]. Round 1 and the fold
/// consult the `padding` spec's block coverage; declared-dead bits are
/// never read, whatever the pooled buffers hold.
#[cfg(target_arch = "aarch64")]
pub fn prove_capture_s_hat_v_c_with_grinding<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &super::PaddingSpec,
    grinding: super::ZerocheckGrinding,
    challenger: &mut C,
) -> (AgProof, AgClaim, Vec<F128>) {
    assert!(m >= K_SKIP + N_INNER, "m >= 13 required");
    let expected = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected);
    assert_eq!(b_packed.len(), expected);
    assert_eq!(c_packed.len(), expected);
    // The run-list at the kernels' 8192-bit code-block grid — round 1 and
    // the fold share it. Read-exact (Dead skipped, Partial cleansed), so
    // the honest-zero witness-mode forcing this entry used to need is gone:
    // declared-dead bits are never read, whatever they hold.
    let zc_timing = std::env::var_os("FLOCK_ZC_TIMING").is_some();
    let t0 = std::time::Instant::now();
    let coverage = padding.block_coverage(K_SKIP + N_INNER, 1usize << (m - K_SKIP - N_INNER));

    challenger.observe_label(b"flock-ag-skip-v1");
    let mut grinding_nonces = Vec::with_capacity(grinding_nonce_count(grinding, m));
    let r_outer = match grinding.initial_bits(m) {
        Some(bits) => {
            let (nonce, v) = challenger.grind_pow_and_sample_f128_vec(bits, m - K_SKIP - N_INNER);
            grinding_nonces.push(nonce);
            v
        }
        None => challenger.sample_f128_vec(m - K_SKIP - N_INNER),
    };
    let eq = crate::zerocheck::univariate_skip::build_eq(&r_outer);
    let t_r1 = std::time::Instant::now();
    let (msg, s_hat_v_c) = prove_round1_banks_padded(a_packed, b_packed, c_packed, &eq, &coverage);
    if zc_timing {
        let (mut full, mut part, mut dead) = (0usize, 0usize, 0usize);
        for c in &coverage {
            match c {
                super::BlockCoverage::Full => full += 1,
                super::BlockCoverage::Partial(_) => part += 1,
                super::BlockCoverage::Dead => dead += 1,
            }
        }
        eprintln!(
            "[ag-zc-timing] m={m} setup+eq {:.2} ms | round1 {:.2} ms (blocks: {full} full / {part} partial / {dead} dead)",
            (t_r1 - t0).as_secs_f64() * 1e3,
            t_r1.elapsed().as_secs_f64() * 1e3,
        );
    }
    let t_tail = std::time::Instant::now();
    let (proof, claim) = prove_from_round1(
        a_packed,
        b_packed,
        msg,
        &r_outer,
        grinding,
        grinding_nonces,
        Some(&coverage),
        challenger,
    );
    if zc_timing {
        eprintln!(
            "[ag-zc-timing] m={m} fold+tail {:.2} ms",
            t_tail.elapsed().as_secs_f64() * 1e3
        );
    }
    (proof, claim, s_hat_v_c)
}

/// Round 1 via the two-bank kernel: returns the canonical [`Round1Message`]
/// (`w̄ = bank0 + bank1`, then `D⁻¹`-scaled — bit-identical to [`prove_round1`])
/// AND the length-128 canonical `s_hat_v_c`. See [`prove_capture_s_hat_v_c`].
#[cfg(target_arch = "aarch64")]
fn prove_round1_banks(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> (Round1Message, Vec<F128>) {
    banks_to_message(
        crate::genus95_curve_code::round1::round1_slp_packed_banks_fused(
            a_packed, b_packed, c_packed, eq,
        ),
    )
}

/// [`prove_round1_banks`] under a witness run-list:
/// [`round1_slp_packed_banks_fused_padded`] — ONE parallel pass over
/// the live-block list, Partial blocks cleansed inline, per-element parity
/// with the dense kernel. (A per-run-segment driver over the unfused kernel
/// was MEASURED 2-3x SLOWER than the full dense scan at the envelope's ~450
/// per-column runs — never fan a hot kernel into per-segment par calls.
/// Deleted with its `ROUND1_UNFUSED` toggle 2026-08-27.)
#[cfg(target_arch = "aarch64")]
fn prove_round1_banks_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
    coverage: &[super::BlockCoverage],
) -> (Round1Message, Vec<F128>) {
    banks_to_message(
        crate::genus95_curve_code::round1::round1_slp_packed_banks_fused_padded(
            a_packed, b_packed, c_packed, eq, coverage,
        ),
    )
}

/// The shared banks → wire-message post-processing: `D⁻¹` scaling of the
/// fresh coords and the recombined `w̄`, plus the canonical `s_hat_v_c`
/// capture — `s_hat_v_c[skip + b·64] = bank_b[skip] / D₁`, with `γ⁻¹` for
/// bank 1 (the kernel's odd-i bank carries an extra `γ` from friendly bit
/// 0 = 1). `1/D₁ = (1+γ)·κ` since `D = (1+γ)·D₁` and `κ = D⁻¹ = d_inv()`.
#[cfg(target_arch = "aarch64")]
fn banks_to_message(
    (res_ab, bank0, bank1): ([F128; 160], [F128; 64], [F128; 64]),
) -> (Round1Message, Vec<F128>) {
    let di = d_inv();
    let msg = Round1Message {
        ab_fresh: (0..158).map(|s| di * res_ab[s]).collect(),
        c_msg: (0..64).map(|i| di * (bank0[i] + bank1[i])).collect(),
    };
    let inv_d1 = (F128::ONE + gamma_pow(1)) * di;
    let gamma_inv = gamma_pow(1).inv();
    let n_skip = 1usize << K_SKIP;
    let mut s_hat_v_c = vec![F128::ZERO; 1 << crate::pcs::LOG_PACKING];
    for skip in 0..n_skip {
        s_hat_v_c[skip] = bank0[skip] * inv_d1;
        s_hat_v_c[n_skip + skip] = bank1[skip] * inv_d1 * gamma_inv;
    }
    (msg, s_hat_v_c)
}

/// Shared post-round-1 tail of [`prove`] / [`prove_capture_s_hat_v_c`]: observe
/// the round-1 message, sample `r₁`, fold `a/b` at `r₁`, and run the multilinear
/// sumcheck to the final evals. `msg` is consumed into the returned `AgProof`.
#[cfg(target_arch = "aarch64")]
fn prove_from_round1<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    msg: Round1Message,
    r_outer: &[F128],
    grinding: super::ZerocheckGrinding,
    mut grinding_nonces: Vec<u64>,
    coverage: Option<&[super::BlockCoverage]>,
    challenger: &mut C,
) -> (AgProof, AgClaim) {
    challenger.observe_f128_slice(&msg.ab_fresh);
    challenger.observe_f128_slice(&msg.c_msg);

    let (r1, r1_nonce) = match grinding.ag_r1_bits() {
        Some(bits) => sample_r1_prover_pow(challenger, bits),
        None => sample_r1_prover(challenger),
    };
    let c_eval = eval_c_at(&msg, &r1);

    let bf = base_evaluation_functional(&r1).expect("denominator nonzero at r1");
    let w: Vec<F128> = bf.iter().copied().collect();
    let mut r_rest = friendly_challenges().to_vec();
    r_rest.extend_from_slice(r_outer);

    // Round 0 + the tail. The sparse-support decision mirrors the RS
    // zerocheck's `sparse_from_round2`: when the run-list has dead blocks
    // and the live fraction clears the gate, the fold emits LIVE-SPAN
    // buffers and the tail runs support-proportional rounds; otherwise the
    // padded (or dense) fold feeds the lookahead tail. Byte-identical wire
    // output on every path.
    let sparse = coverage.is_some_and(|cov| {
        let live_blocks = cov
            .iter()
            .filter(|c| !matches!(c, super::BlockCoverage::Dead))
            .count();
        let n_out = cov.len() * 128;
        live_blocks < cov.len()
            && n_out >= 8
            && live_blocks * 128 * super::sparse_tail_gate() <= n_out
    });
    let (a_mlv, b_mlv, g1_0, ginf_0, store) = if sparse {
        let cov = coverage.expect("sparse implies coverage");
        let (a, b, g1, gi, st) = fold_and_first_round_sparse(a_packed, b_packed, &w, &r_rest, cov);
        (a, b, g1, gi, Some(st))
    } else {
        let (a, b, g1, gi) = match coverage {
            Some(cov) => fold_and_first_round_padded(a_packed, b_packed, &w, &r_rest, cov),
            None => fold_and_first_round(a_packed, b_packed, &w, &r_rest),
        };
        (a, b, g1, gi, None)
    };
    let (rounds, rhos, a_eval, b_eval) = match grinding.multilinear_round_bits() {
        Some(bits) => {
            let mut gch = RoundGrindProver {
                inner: challenger,
                bits,
                nonces: &mut grinding_nonces,
            };
            mlv_tail_dispatch(a_mlv, b_mlv, g1_0, ginf_0, store, &r_rest, &mut gch)
        }
        None => mlv_tail_dispatch(a_mlv, b_mlv, g1_0, ginf_0, store, &r_rest, challenger),
    };

    let proof = AgProof {
        round1_ab: msg.ab_fresh,
        round1_c: msg.c_msg,
        r1_nonce,
        multilinear_rounds: rounds,
        final_a_eval: a_eval,
        final_b_eval: b_eval,
        final_c_eval: c_eval,
        grinding_nonces,
    };
    let claim = AgClaim {
        r1,
        mlv_challenges: rhos,
        r_rest,
        a_eval,
        b_eval,
        c_eval,
    };
    (proof, claim)
}

/// FS-interleaved multilinear tail shared by the 64-slot and tensor provers:
/// observes round 0's message `(g1_0, ginf_0)`, then runs the
/// lookahead/friendly/classic rounds to the final binding. `r_rest` must be
/// `friendly ‖ outer`; the wire output is bit-identical across the internal
/// path choices (lookahead on/off, friendly on/off).
pub(super) fn mlv_tail_fs<C: Challenger>(
    a_mlv: Vec<F128>,
    b_mlv: Vec<F128>,
    g1_0: F128,
    ginf_0: F128,
    r_rest: &[F128],
    challenger: &mut C,
) -> (Vec<(F128, F128)>, Vec<F128>, F128, F128) {
    let mut rounds = Vec::with_capacity(r_rest.len());
    let mut rhos = Vec::with_capacity(r_rest.len());
    rounds.push((g1_0, ginf_0));
    challenger.observe_f128(g1_0);
    challenger.observe_f128(ginf_0);
    let rho0 = challenger.sample_f128();
    rhos.push(rho0);
    let (tail_rounds, tail_rhos, a_eval, b_eval) =
        mlv_tail_fs_resume(a_mlv, b_mlv, 1, rho0, None, r_rest, challenger);
    rounds.extend(tail_rounds);
    rhos.extend(tail_rhos);
    (rounds, rhos, a_eval, b_eval)
}

/// One tail, two storages: dispatch on whether the fold handed over
/// live-span buffers (+ their [`LiveLayout`]) or a full dense pair.
fn mlv_tail_dispatch<C: Challenger>(
    a_mlv: Vec<F128>,
    b_mlv: Vec<F128>,
    g1_0: F128,
    ginf_0: F128,
    store: Option<LiveLayout>,
    r_rest: &[F128],
    challenger: &mut C,
) -> (Vec<(F128, F128)>, Vec<F128>, F128, F128) {
    match store {
        Some(st) => mlv_tail_fs_sparse(a_mlv, b_mlv, g1_0, ginf_0, st, r_rest, challenger),
        None => mlv_tail_fs(a_mlv, b_mlv, g1_0, ginf_0, r_rest, challenger),
    }
}

/// [`mlv_tail_fs`] over LIVE-SPAN buffers (the sparse fold's output): the
/// tail runs the support-proportional rounds — RS's
/// [`fold_and_round_pair_sparse_into`], with the friendly constants riding
/// as ordinary `r_next` weights — while the live set clears the gate and
/// the domain keeps the fused threshold, then expands to dense ONCE and
/// resumes the lookahead tail mid-stream. Wire output is bit-identical to
/// the dense tail: every skipped term carries an `a·b` factor of zero.
pub(super) fn mlv_tail_fs_sparse<C: Challenger>(
    mut a_mlv: Vec<F128>,
    mut b_mlv: Vec<F128>,
    g1_0: F128,
    ginf_0: F128,
    mut store: LiveLayout,
    r_rest: &[F128],
    challenger: &mut C,
) -> (Vec<(F128, F128)>, Vec<F128>, F128, F128) {
    let n_mlv = r_rest.len();
    let mut rounds = Vec::with_capacity(n_mlv);
    let mut rhos = Vec::with_capacity(n_mlv);
    rounds.push((g1_0, ginf_0));
    challenger.observe_f128(g1_0);
    challenger.observe_f128(ginf_0);
    let mut rho_prev = challenger.sample_f128();
    rhos.push(rho_prev);

    // The sparse rounds — the RS tail's loop shape: the logical `domain`
    // halves every round regardless of the compacted buffer length, and
    // the gate re-checks per round (interval ends round outward, so the
    // live fraction can cross it mid-tail).
    let mut domain = 1usize << n_mlv;
    let (mut a_nxt, mut b_nxt) = (Vec::new(), Vec::new());
    let mut i = 1usize;
    while i < n_mlv && domain >= 1024 && store.len() * super::sparse_tail_gate() <= domain {
        let log_before = domain.trailing_zeros() as usize;
        let mut r_next = vec![F128::ONE; log_before - 1];
        r_next[1..].copy_from_slice(&r_rest[i + 1..]);
        // Output storage is bounded by the input's: shrinking pairs can
        // only round outward by one slot per interval end.
        let cap = store.len() + 2 * store.intervals().len() + 2;
        if a_nxt.len() < cap {
            crate::scratch::give_f128(a_nxt);
            crate::scratch::give_f128(b_nxt);
            a_nxt = crate::scratch::take_f128(cap);
            b_nxt = crate::scratch::take_f128(cap);
        }
        let (m1, mi, store_out) = fold_and_round_pair_sparse_into(
            &a_mlv,
            &b_mlv,
            &mut a_nxt[..cap],
            &mut b_nxt[..cap],
            rho_prev,
            &r_next,
            &store,
            domain,
        );
        std::mem::swap(&mut a_mlv, &mut a_nxt);
        std::mem::swap(&mut b_mlv, &mut b_nxt);
        a_mlv.truncate(store_out.len());
        b_mlv.truncate(store_out.len());
        store = store_out;
        domain /= 2;
        rounds.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        rho_prev = challenger.sample_f128();
        rhos.push(rho_prev);
        i += 1;
    }

    // Back to global indexing: scatter the live span into a full padded
    // buffer once and resume the lookahead tail mid-stream (the arrays are
    // folded through every sampled challenge except `rho_prev` — resume's
    // own invariant).
    let a_full = expand_to_dense(&a_mlv, &store, domain);
    let b_full = expand_to_dense(&b_mlv, &store, domain);
    crate::scratch::give_f128(a_mlv);
    crate::scratch::give_f128(b_mlv);
    crate::scratch::give_f128(a_nxt);
    crate::scratch::give_f128(b_nxt);
    let (tail_rounds, tail_rhos, a_eval, b_eval) =
        mlv_tail_fs_resume(a_full, b_full, i, rho_prev, None, r_rest, challenger);
    rounds.extend(tail_rounds);
    rhos.extend(tail_rhos);
    (rounds, rhos, a_eval, b_eval)
}

/// Resume the FS tail mid-stream at round `i0`: the arrays are folded through
/// every sampled challenge EXCEPT `rho_prev` (and `pending2`, if set — the
/// lookahead deferred-fold invariant). Lets a caller that derived the early
/// round messages elsewhere (e.g. the swoop's fold-level lookahead, which
/// covers rounds 0–1 during the witness fold) hand off without extra passes.
pub(super) fn mlv_tail_fs_resume<C: Challenger>(
    mut a_mlv: Vec<F128>,
    mut b_mlv: Vec<F128>,
    i0: usize,
    mut rho_prev: F128,
    mut pending2: Option<F128>,
    r_rest: &[F128],
    challenger: &mut C,
) -> (Vec<(F128, F128)>, Vec<F128>, F128, F128) {
    if LOOKAHEAD_DISABLE.load(Ordering::Relaxed) {
        crate::suboptimal_path!(
            "classic per-round tail (LOOKAHEAD_DISABLE set)",
            "lookahead tail (default)"
        );
    }
    let n_mlv = r_rest.len();
    let mut rounds = Vec::with_capacity(n_mlv);
    let mut rhos = Vec::with_capacity(n_mlv);

    // Rounds 1..n_mlv: fused fold(ρ_{i-1}) + this round's message, FS-interleaved
    // (parallel fused path while ≥10 vars remain, else naive). Ping-pong scratch.
    //
    // For the friendly rounds 1..=5 the message still has `6-i` `γ`-geometric
    // inner dims, so we use the friendly-Horner kernel (no per-term `eq_lo`
    // PMULL) with the outer eq pre-split once. Rounds 6+ (no friendly dims left)
    // and the small tail fall back to the general kernel / naive path. The
    // friendly kernel is bit-identical to the general one (see
    // `friendly_round_matches_general`), so the proof is unchanged.
    let split_outer = super::univariate_skip::SplitEqGhash::new(&r_rest[N_INNER..]);
    let n_in = a_mlv.len();
    let (mut a_nxt, mut b_nxt) = if n_in >= 1024 {
        // Uninit pooled buffers (NOT vec![ZERO; n_in/2]): the fused fold writes
        // every slot of a_nxt/b_nxt[..half] before reading (write-before-read),
        // so a zero-fill is wasted — and it's a *serial* 256 MB memset before the
        // parallel loop (Amdahl), the same trap `a_mlv`/`b_mlv` avoid via the pool.
        (
            crate::scratch::take_f128(n_in / 2),
            crate::scratch::take_f128(n_in / 2),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    // LOOKAHEAD state: `pending2`, when set, is a sampled challenge whose fold
    // has been deferred (the lookahead pass folds it together with `rho_prev`,
    // 4→1, on the next pass). Invariant: the arrays are folded through all
    // sampled challenges EXCEPT `rho_prev` (and `pending2` if set).
    let mut i = i0;
    while i < n_mlv {
        let len = a_mlv.len();
        // Lookahead pass: needs another message round after this one and a
        // large enough array for the fused path (same threshold as classic).
        let lookahead = !LOOKAHEAD_DISABLE.load(Ordering::Relaxed) && i + 1 < n_mlv && len >= 1024;
        if lookahead {
            let out_len = if pending2.is_some() { len / 4 } else { len / 2 };
            let (ao, bo) = (&mut a_nxt[..out_len], &mut b_nxt[..out_len]);
            // (A friendly-Horner variant of this pass for the iterations whose eq
            // still spans γ-geometric friendly dims — i=1, i=3 — was bit-identical
            // but measured −1.5% on an M-series Air: the 8×256-bit Horner
            // accumulators spill. Deleted 2026-08-27 with its `LOOKAHEAD_FRIENDLY`
            // toggle; see git history if M4 Max-class DRAM bandwidth makes it
            // worth re-measuring.)
            let q = if let Some(r2) = pending2 {
                fold2_lookahead_into(&a_mlv, &b_mlv, ao, bo, (rho_prev, r2), &r_rest[i + 2..])
            } else {
                fold1_lookahead_into(
                    &a_mlv,
                    &b_mlv,
                    ao,
                    bo,
                    (rho_prev, F128::ZERO),
                    &r_rest[i + 2..],
                )
            };
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(out_len);
            b_mlv.truncate(out_len);
            let (m1a, mia) = lookahead_msg_first(&q, r_rest[i + 1]);
            rounds.push((m1a, mia));
            challenger.observe_f128(m1a);
            challenger.observe_f128(mia);
            let rho_a = challenger.sample_f128();
            rhos.push(rho_a);
            let (m1b, mib) = lookahead_msg_second(&q, rho_a);
            rounds.push((m1b, mib));
            challenger.observe_f128(m1b);
            challenger.observe_f128(mib);
            let rho_b = challenger.sample_f128();
            rhos.push(rho_b);
            rho_prev = rho_a;
            pending2 = Some(rho_b);
            i += 2;
            continue;
        }
        // Leaving lookahead mode: resolve the deferred fold before classic
        // processing (arrays here are ≤ 2·1024 elements — serial fold is free).
        if let Some(r2) = pending2.take() {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            rho_prev = r2;
            continue;
        }
        let log_before = a_mlv.len().trailing_zeros() as usize;
        let (m1, mi) = if log_before >= 10 {
            let half = a_mlv.len() / 2;
            let use_friendly = (1..=5).contains(&i);
            let pair = if use_friendly {
                // Friendly-Horner: lo = friendly dims i+1..6, hi = outer (split once).
                let lo_size = 1usize << (N_INNER - 1 - i);
                let c_inv = friendly_norm(i);
                let (lo, hi) = (&split_outer.lo, &split_outer.hi);
                let (ao, bo) = (&mut a_nxt[..half], &mut b_nxt[..half]);
                match i {
                    1 => fold_and_friendly_round_pair_into::<4>(
                        &a_mlv, &b_mlv, ao, bo, rho_prev, lo_size, lo, hi, c_inv,
                    ),
                    2 => fold_and_friendly_round_pair_into::<8>(
                        &a_mlv, &b_mlv, ao, bo, rho_prev, lo_size, lo, hi, c_inv,
                    ),
                    3 => fold_and_friendly_round_pair_into::<16>(
                        &a_mlv, &b_mlv, ao, bo, rho_prev, lo_size, lo, hi, c_inv,
                    ),
                    4 => fold_and_friendly_round_pair_into::<32>(
                        &a_mlv, &b_mlv, ao, bo, rho_prev, lo_size, lo, hi, c_inv,
                    ),
                    5 => fold_and_friendly_round_pair_into::<64>(
                        &a_mlv, &b_mlv, ao, bo, rho_prev, lo_size, lo, hi, c_inv,
                    ),
                    _ => unreachable!(),
                }
            } else {
                let mut r_next = vec![F128::ONE; log_before - 1];
                r_next[1..].copy_from_slice(&r_rest[i + 1..]);
                fold_and_compute_round_pair_into(
                    &a_mlv,
                    &b_mlv,
                    &mut a_nxt[..half],
                    &mut b_nxt[..half],
                    rho_prev,
                    &r_next,
                )
            };
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            pair
        } else {
            let mut r_next = vec![F128::ONE; log_before - 1];
            r_next[1..].copy_from_slice(&r_rest[i + 1..]);
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            round_pair_naive(&a_mlv, &b_mlv, &r_next)
        };
        rounds.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        rho_prev = challenger.sample_f128();
        rhos.push(rho_prev);
        i += 1;
    }
    // Final binding: one or (after a trailing lookahead pass) two deferred
    // challenges remain.
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
    if let Some(r2) = pending2 {
        fold_in_place_pair(&mut a_mlv, &mut b_mlv, r2);
    }
    let (a_eval, b_eval) = (a_mlv[0], b_mlv[0]);
    challenger.observe_f128(a_eval);
    challenger.observe_f128(b_eval);
    // Recycle the tail buffers (same as the RS tail in `zerocheck.rs`) — the
    // truncated Vecs keep their full capacity, and dropping them instead
    // would munmap + re-fault 100s of MB on the next prove.
    crate::scratch::give_f128(a_mlv);
    crate::scratch::give_f128(b_mlv);
    crate::scratch::give_f128(a_nxt);
    crate::scratch::give_f128(b_nxt);
    (rounds, rhos, a_eval, b_eval)
}

/// Verify an AG-skip zerocheck proof for an instance over `{0,1}^m`. Walks the
/// challenger in lockstep with the prover and checks every round's consistency.
pub fn verify<C: Challenger>(
    m: usize,
    proof: &AgProof,
    challenger: &mut C,
) -> Result<AgClaim, AgVerifyError> {
    verify_with_grinding(m, proof, super::ZerocheckGrinding::disabled(), challenger)
}

/// [`verify`] under a grinding schedule — the mirror of
/// [`prove_capture_s_hat_v_c_with_grinding`]. With a disabled schedule this
/// accepts exactly the old proofs (an empty nonce vector is required).
pub fn verify_with_grinding<C: Challenger>(
    m: usize,
    proof: &AgProof,
    grinding: super::ZerocheckGrinding,
    challenger: &mut C,
) -> Result<AgClaim, AgVerifyError> {
    if m < K_SKIP + N_INNER {
        return Err(AgVerifyError::LogNTooSmall { log_n: m });
    }
    let n_mlv = m - K_SKIP;
    if proof.round1_ab.len() != 158 {
        return Err(AgVerifyError::BadRound1Length {
            which: "ab",
            expected: 158,
            got: proof.round1_ab.len(),
        });
    }
    if proof.round1_c.len() != 64 {
        return Err(AgVerifyError::BadRound1Length {
            which: "c",
            expected: 64,
            got: proof.round1_c.len(),
        });
    }
    if proof.multilinear_rounds.len() != n_mlv {
        return Err(AgVerifyError::BadMultilinearRoundsLength {
            expected: n_mlv,
            got: proof.multilinear_rounds.len(),
        });
    }
    let expected_nonces = grinding_nonce_count(grinding, m);
    if proof.grinding_nonces.len() != expected_nonces {
        return Err(AgVerifyError::BadGrindingNonceCount {
            expected: expected_nonces,
            got: proof.grinding_nonces.len(),
        });
    }
    let mut nonces = proof.grinding_nonces.iter().copied();

    challenger.observe_label(b"flock-ag-skip-v1");
    let r_outer = match grinding.initial_bits(m) {
        Some(bits) => {
            let nonce = nonces.next().ok_or(AgVerifyError::InvalidGrindingNonce)?;
            challenger
                .verify_pow_and_sample_f128_vec(nonce, bits, m - K_SKIP - N_INNER)
                .ok_or(AgVerifyError::InvalidGrindingNonce)?
        }
        None => challenger.sample_f128_vec(m - K_SKIP - N_INNER),
    };
    challenger.observe_f128_slice(&proof.round1_ab);
    challenger.observe_f128_slice(&proof.round1_c);

    let r1 = match grinding.ag_r1_bits() {
        Some(bits) => replay_r1_verifier_pow(challenger, proof.r1_nonce, bits)?,
        None => replay_r1_verifier(challenger, proof.r1_nonce)?,
    };
    let msg = Round1Message {
        ab_fresh: proof.round1_ab.clone(),
        c_msg: proof.round1_c.clone(),
    };
    if eval_c_at(&msg, &r1) != proof.final_c_eval {
        return Err(AgVerifyError::CEvalMismatch);
    }
    let mut c_running = eval_ab_at(&msg, &r1);

    let mut r_rest = friendly_challenges().to_vec();
    r_rest.extend_from_slice(&r_outer);

    let mlv_bits = grinding.multilinear_round_bits();
    let mut rhos = Vec::with_capacity(n_mlv);
    for i in 0..n_mlv {
        let (g1, g_inf) = proof.multilinear_rounds[i];
        let r_eq = r_rest[i];
        let g0 = (c_running + r_eq * g1) * (F128::ONE + r_eq).inv();
        challenger.observe_f128(g1);
        challenger.observe_f128(g_inf);
        let rho = match mlv_bits {
            Some(bits) => {
                let nonce = nonces.next().ok_or(AgVerifyError::InvalidGrindingNonce)?;
                challenger
                    .verify_pow_and_sample_f128(nonce, bits)
                    .ok_or(AgVerifyError::InvalidGrindingNonce)?
            }
            None => challenger.sample_f128(),
        };
        rhos.push(rho);
        c_running = g0 * (F128::ONE + rho) + g1 * rho + g_inf * rho * (rho + F128::ONE);
    }
    challenger.observe_f128(proof.final_a_eval);
    challenger.observe_f128(proof.final_b_eval);

    if c_running != proof.final_a_eval * proof.final_b_eval {
        return Err(AgVerifyError::SumcheckFinalFailed);
    }
    Ok(AgClaim {
        r1,
        mlv_challenges: rhos,
        r_rest,
        a_eval: proof.final_a_eval,
        b_eval: proof.final_b_eval,
        c_eval: proof.final_c_eval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genus95_curve_code::{RngCore, Sha256Rng, sample_random_evaluation_point};
    use flock_hash::HashKind;

    /// The pinned friendly challenges reproduce the geometric eq weight
    /// `eq(r_inner, b) = γ^{int(b)} / D` for every inner index `b ∈ [0, 128)`.
    #[test]
    fn friendly_eq_is_gamma_geometric_over_d() {
        let r = friendly_challenges();
        let di = d_inv();
        for b in 0..(1usize << N_INNER) {
            let mut eq = F128::ONE;
            for j in 0..N_INNER {
                eq *= if (b >> j) & 1 == 1 {
                    r[j]
                } else {
                    F128::ONE + r[j]
                };
            }
            assert_eq!(eq, gamma_pow(b) * di, "b={b}");
        }
    }

    #[test]
    fn d_times_d_inv_is_one() {
        assert_eq!(d_const() * d_inv(), F128::ONE);
    }

    /// The SHA-256 DRBG drives the rejection sampler deterministically: equal
    /// seeds give the same `r₁`, distinct seeds (almost surely) differ.
    #[test]
    fn sha256_rng_seeds_sampler_deterministically() {
        let seed = [0x5Au8; 32];
        let p1 = sample_random_evaluation_point(&mut Sha256Rng::new(seed)).expect("point");
        let p2 = sample_random_evaluation_point(&mut Sha256Rng::new(seed)).expect("point");
        assert_eq!(p1, p2, "same seed must give same r1");
        let p3 = sample_random_evaluation_point(&mut Sha256Rng::new([0xA5u8; 32])).expect("point");
        assert_ne!(p1, p3, "different seed should give a different r1");
    }

    /// Round-1 keystone: the prover's message, evaluated at `r₁` by the verifier
    /// (`⟨E(r₁), [w̄|fresh]⟩` for AB, `⟨base(r₁), w̄⟩` for C), matches a direct
    /// per-column reference `Σ_o eq[o] Σ_b (γ^b/D)·[…]`. Honest witness (`c=a&b`)
    /// so systematic vanishing holds and `w̄` is the true value section.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn round1_eval_matches_direct_reference() {
        use crate::genus95_curve_code::{BaseMessage, evaluate_base_functional};

        let mut rng = Sha256Rng::new([42u8; 32]);
        let n = 2usize;
        let mut am = vec![[0u64; 128]; n];
        let mut bm = vec![[0u64; 128]; n];
        let mut cm = vec![[0u64; 128]; n];
        for o in 0..n {
            for b in 0..128 {
                am[o][b] = rng.next_u64();
                bm[o][b] = rng.next_u64();
                cm[o][b] = am[o][b] & bm[o][b];
            }
        }
        let pack = |ms: &[[u64; 128]]| -> Vec<u8> {
            let mut p = vec![0u8; ms.len() * 1024];
            for (o, blk) in ms.iter().enumerate() {
                for b in 0..128 {
                    p[o * 1024 + b * 8..o * 1024 + b * 8 + 8]
                        .copy_from_slice(&blk[b].to_le_bytes());
                }
            }
            p
        };
        let eq: Vec<F128> = (0..n)
            .map(|_| F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            })
            .collect();

        let msg = prove_round1(&pack(&am), &pack(&bm), &pack(&cm), &eq);
        let r1 = sample_random_evaluation_point(&mut Sha256Rng::new([7u8; 32])).expect("point");

        let v_ab = eval_ab_at(&msg, &r1);
        let v_c = eval_c_at(&msg, &r1);

        let bf =
            crate::genus95_curve_code::base_evaluation_functional(&r1).expect("base functional");
        let di = d_inv();
        let mut d_ab = F128::ZERO;
        let mut d_c = F128::ZERO;
        for o in 0..n {
            for b in 0..128 {
                let w = eq[o] * gamma_pow(b) * di;
                let ea = evaluate_base_functional(&bf, &BaseMessage(am[o][b]));
                let eb = evaluate_base_functional(&bf, &BaseMessage(bm[o][b]));
                let ec = evaluate_base_functional(&bf, &BaseMessage(cm[o][b]));
                d_ab += w * ea * eb;
                d_c += w * ec;
            }
        }
        assert_eq!(v_ab, d_ab, "P^ab(r1) != direct reference");
        assert_eq!(v_c, d_c, "c(r1) != direct reference");
    }

    /// The folded rows feed the multilinear sumcheck: (1) `â(r₁, rest=o*128+b)`
    /// equals the per-column base evaluation, and (2) the eq-weighted sum of the
    /// folded products, `Σ_rest eq(r_rest,rest)·â(r₁,rest)·b̂(r₁,rest)`, equals the
    /// round-1 AB claim `eval_ab_at` — i.e. the multilinear phase starts from the
    /// exact value round 1 produced.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fold_at_r1_consistent_with_round1_claim() {
        use crate::genus95_curve_code::{
            BaseMessage, base_evaluation_functional, evaluate_base_functional,
        };

        let mut rng = Sha256Rng::new([99u8; 32]);
        let n = 2usize;
        let mut am = vec![[0u64; 128]; n];
        let mut bm = vec![[0u64; 128]; n];
        let mut cm = vec![[0u64; 128]; n];
        for o in 0..n {
            for b in 0..128 {
                am[o][b] = rng.next_u64();
                bm[o][b] = rng.next_u64();
                cm[o][b] = am[o][b] & bm[o][b];
            }
        }
        let pack = |ms: &[[u64; 128]]| -> Vec<u8> {
            let mut p = vec![0u8; ms.len() * 1024];
            for (o, blk) in ms.iter().enumerate() {
                for b in 0..128 {
                    p[o * 1024 + b * 8..o * 1024 + b * 8 + 8]
                        .copy_from_slice(&blk[b].to_le_bytes());
                }
            }
            p
        };
        let (a_packed, b_packed, c_packed) = (pack(&am), pack(&bm), pack(&cm));
        let eq: Vec<F128> = (0..n)
            .map(|_| F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            })
            .collect();

        let msg = prove_round1(&a_packed, &b_packed, &c_packed, &eq);
        let r1 = sample_random_evaluation_point(&mut Sha256Rng::new([7u8; 32])).expect("point");
        let bf = base_evaluation_functional(&r1).expect("base functional");
        let w: Vec<F128> = bf.iter().copied().collect();

        let a_mlv = fold_witness_at_r1(&a_packed, &w);
        let b_mlv = fold_witness_at_r1(&b_packed, &w);

        // (1) folded row == per-column base evaluation.
        for o in 0..n {
            for b in 0..128 {
                assert_eq!(
                    a_mlv[o * 128 + b],
                    evaluate_base_functional(&bf, &BaseMessage(am[o][b])),
                    "a_mlv[{o},{b}]"
                );
            }
        }

        // (2) eq-weighted fold sum == round-1 AB claim.
        let di = d_inv();
        let mut s = F128::ZERO;
        for o in 0..n {
            for b in 0..128 {
                let r = o * 128 + b;
                let eq_rest = eq[o] * gamma_pow(b) * di;
                s += eq_rest * a_mlv[r] * b_mlv[r];
            }
        }
        assert_eq!(s, eval_ab_at(&msg, &r1), "fold sum != round-1 AB claim");
    }

    /// The fused `fold_and_first_round` equals doing the two folds and the round-0
    /// message separately.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fold_and_first_round_matches_separate() {
        use crate::genus95_curve_code::base_evaluation_functional;
        use crate::zerocheck::univariate_skip::build_eq;

        let mut rng = Sha256Rng::new([88u8; 32]);
        let m = 14usize;
        let nbytes = (1usize << m) / 8;
        let mut a = vec![0u8; nbytes];
        let mut b = vec![0u8; nbytes];
        for x in a.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        for x in b.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        let r_outer: Vec<F128> = (0..m - K_SKIP - N_INNER)
            .map(|_| F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            })
            .collect();
        let mut r_rest = friendly_challenges().to_vec();
        r_rest.extend_from_slice(&r_outer);
        let _ = build_eq(&r_outer); // (eq weights are derived inside)
        let r1 = sample_random_evaluation_point(&mut Sha256Rng::new([3u8; 32])).expect("point");
        let bf = base_evaluation_functional(&r1).expect("bf");
        let w: Vec<F128> = bf.iter().copied().collect();

        let (a_mlv_f, b_mlv_f, g1, ginf) = fold_and_first_round(&a, &b, &w, &r_rest);
        let a_mlv = fold_witness_at_r1(&a, &w);
        let b_mlv = fold_witness_at_r1(&b, &w);
        assert_eq!(a_mlv_f, a_mlv, "fused a_mlv != separate fold");
        assert_eq!(b_mlv_f, b_mlv, "fused b_mlv != separate fold");

        let log_now = a_mlv.len().trailing_zeros() as usize;
        let mut r_next = vec![F128::ONE; log_now];
        r_next[1..].copy_from_slice(&r_rest[1..]);
        assert_eq!(
            (g1, ginf),
            round_pair_naive(&a_mlv, &b_mlv, &r_next),
            "fused first message != round_pair_naive"
        );
    }

    /// The friendly-Horner round kernel ([`fold_and_friendly_round_pair_into`])
    /// produces a **bit-identical** folded `a_out`/`b_out` and `(G(1), G(∞))` to
    /// the general [`super::super::multilinear::fold_and_compute_round_pair_into`]
    /// for every friendly round `i ∈ 1..=5`. (Same field value, so the two
    /// factorizations — per-term `eq_lo` PMULL vs friendly `γ`-Horner — agree to
    /// the bit; this is what makes the wired `prove` byte-identical.)
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn friendly_round_matches_general() {
        use crate::zerocheck::univariate_skip::SplitEqGhash;

        let mut rng = Sha256Rng::new([0x1f; 32]);
        let o = 9usize; // outer dims (m - 13), big enough to exercise n_ol > 1
        let r_outer: Vec<F128> = (0..o)
            .map(|_| F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            })
            .collect();
        let friendly = friendly_challenges();
        let split = SplitEqGhash::new(&r_outer);

        for i in 1..=5usize {
            let l = 8 - i + o; // array log at the start of round i
            let n = 1usize << l;
            let a: Vec<F128> = (0..n)
                .map(|_| F128 {
                    lo: rng.next_u64(),
                    hi: rng.next_u64(),
                })
                .collect();
            let b: Vec<F128> = (0..n)
                .map(|_| F128 {
                    lo: rng.next_u64(),
                    hi: rng.next_u64(),
                })
                .collect();
            let rho = F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            };
            let half = n / 2;

            // General kernel: r_next = [ONE, friendly[i+1..7], r_outer].
            let mut r_next = vec![F128::ONE; l - 1];
            let mut tail = friendly[i + 1..N_INNER].to_vec();
            tail.extend_from_slice(&r_outer);
            r_next[1..].copy_from_slice(&tail);
            let mut a_gen = crate::alloc_uninit_f128_vec(half);
            let mut b_gen = crate::alloc_uninit_f128_vec(half);
            let msg_gen =
                fold_and_compute_round_pair_into(&a, &b, &mut a_gen, &mut b_gen, rho, &r_next);

            // Friendly-Horner kernel.
            let mut a_fr = crate::alloc_uninit_f128_vec(half);
            let mut b_fr = crate::alloc_uninit_f128_vec(half);
            let lo_size = 1usize << (N_INNER - 1 - i); // 2^{6-i}
            let c_inv = friendly_norm(i);
            let (lo, hi) = (&split.lo, &split.hi);
            let msg_fr = match i {
                1 => fold_and_friendly_round_pair_into::<4>(
                    &a, &b, &mut a_fr, &mut b_fr, rho, lo_size, lo, hi, c_inv,
                ),
                2 => fold_and_friendly_round_pair_into::<8>(
                    &a, &b, &mut a_fr, &mut b_fr, rho, lo_size, lo, hi, c_inv,
                ),
                3 => fold_and_friendly_round_pair_into::<16>(
                    &a, &b, &mut a_fr, &mut b_fr, rho, lo_size, lo, hi, c_inv,
                ),
                4 => fold_and_friendly_round_pair_into::<32>(
                    &a, &b, &mut a_fr, &mut b_fr, rho, lo_size, lo, hi, c_inv,
                ),
                5 => fold_and_friendly_round_pair_into::<64>(
                    &a, &b, &mut a_fr, &mut b_fr, rho, lo_size, lo, hi, c_inv,
                ),
                _ => unreachable!(),
            };

            assert_eq!(a_fr, a_gen, "round {i}: folded a_out mismatch");
            assert_eq!(b_fr, b_gen, "round {i}: folded b_out mismatch");
            assert_eq!(msg_fr, msg_gen, "round {i}: (G(1), G(∞)) mismatch");
        }
    }

    /// End-to-end (no PCS): round 1 → fold → multilinear sumcheck, and the
    /// verifier's running claim telescopes from the round-1 value down to
    /// `a_eval · b_eval` at the final folded point — i.e. the verifier accepts a
    /// correctly-produced proof. `m = 14` (7 inner + 1 outer rest dims).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn full_chain_verifier_accepts() {
        use crate::zerocheck::univariate_skip::build_eq;

        let mut rng = Sha256Rng::new([123u8; 32]);
        let n = 2usize;
        let mut am = vec![[0u64; 128]; n];
        let mut bm = vec![[0u64; 128]; n];
        let mut cm = vec![[0u64; 128]; n];
        for o in 0..n {
            for b in 0..128 {
                am[o][b] = rng.next_u64();
                bm[o][b] = rng.next_u64();
                cm[o][b] = am[o][b] & bm[o][b];
            }
        }
        let pack = |ms: &[[u64; 128]]| -> Vec<u8> {
            let mut p = vec![0u8; ms.len() * 1024];
            for (o, blk) in ms.iter().enumerate() {
                for b in 0..128 {
                    p[o * 1024 + b * 8..o * 1024 + b * 8 + 8]
                        .copy_from_slice(&blk[b].to_le_bytes());
                }
            }
            p
        };
        let (a_packed, b_packed, c_packed) = (pack(&am), pack(&bm), pack(&cm));

        // One outer challenge → eq weights (one per block); r_rest = friendly ‖ outer.
        let r_outer = vec![F128 {
            lo: rng.next_u64(),
            hi: rng.next_u64(),
        }];
        let eq = build_eq(&r_outer);
        assert_eq!(eq.len(), n);

        let msg = prove_round1(&a_packed, &b_packed, &c_packed, &eq);
        let r1 = sample_random_evaluation_point(&mut Sha256Rng::new([7u8; 32])).expect("point");
        let claim_ab = eval_ab_at(&msg, &r1);

        let bf =
            crate::genus95_curve_code::base_evaluation_functional(&r1).expect("base functional");
        let w: Vec<F128> = bf.iter().copied().collect();
        let a_mlv = fold_witness_at_r1(&a_packed, &w);
        let b_mlv = fold_witness_at_r1(&b_packed, &w);

        let mut r_rest = friendly_challenges().to_vec();
        r_rest.extend_from_slice(&r_outer);
        let rhos: Vec<F128> = (0..r_rest.len())
            .map(|_| F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            })
            .collect();

        let (msgs, a_eval, b_eval) = prove_multilinear(a_mlv, b_mlv, &r_rest, &rhos);
        let c = verify_multilinear(claim_ab, &r_rest, &msgs, &rhos);
        assert_eq!(
            c,
            a_eval * b_eval,
            "verifier final claim != a_eval * b_eval"
        );
    }

    fn random_witness(m: usize, seed: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut rng = Sha256Rng::new([seed; 32]);
        let nbytes = (1usize << m) / 8;
        let mut a = vec![0u8; nbytes];
        let mut b = vec![0u8; nbytes];
        for x in a.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        for x in b.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        let c: Vec<u8> = a.iter().zip(&b).map(|(x, y)| x & y).collect(); // honest c = a & b
        (a, b, c)
    }

    /// Full Fiat–Shamir roundtrip: prove and verify with matching SHA-256
    /// challengers; the verifier accepts and reproduces the prover's claim.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prove_verify_roundtrip() {
        use crate::challenger::FsChallenger;
        let m = 20usize; // ≥ 16 so the fused (≥10-var) multilinear path is exercised
        let (a, b, c) = random_witness(m, 55);
        let (proof, claim) = prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-skip-test"));
        let vclaim = verify(m, &proof, &mut FsChallenger::new(b"flock-ag-skip-test"))
            .expect("verifier accepts honest proof");
        assert_eq!(vclaim, claim, "verifier claim != prover claim");
    }

    /// The r₁ nonce grind follows the transcript hash: a BLAKE3-FS transcript
    /// derives r₁ through the BLAKE3 expander (`FsRng::Blake3`) and still
    /// roundtrips — and produces a different r₁/nonce than the SHA-256
    /// transcript (the two proofs genuinely diverge, so both arms are live).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prove_verify_roundtrip_blake3_fs() {
        use crate::challenger::FsChallenger;
        let m = 14usize;
        let (a, b, c) = random_witness(m, 91);
        let mk = |kind| FsChallenger::with_hash(b"flock-ag-skip-b3", kind);
        let (proof_b3, claim_b3) = prove(&a, &b, &c, m, &mut mk(HashKind::Blake3));
        let vclaim = verify(m, &proof_b3, &mut mk(HashKind::Blake3))
            .expect("verifier accepts honest BLAKE3-FS proof");
        assert_eq!(vclaim, claim_b3);

        let (proof_sha, _) = prove(&a, &b, &c, m, &mut mk(HashKind::Sha256));
        assert_ne!(
            proof_b3, proof_sha,
            "BLAKE3-FS and SHA-256-FS transcripts must diverge"
        );
    }

    /// LOOKAHEAD parity: the sumcheck-lookahead tail produces a bit-identical
    /// proof and claim to the classic one-round-per-pass tail (the transcript
    /// is unchanged — lookahead only changes the prover's evaluation strategy).
    /// m = 20 gives 14 mlv rounds: entry fold1 pass, several fold2 passes, the
    /// pending-resolution demotion, and the small-round classic fallback are
    /// all exercised.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn lookahead_matches_classic() {
        use crate::challenger::FsChallenger;
        let m = 20usize;
        let (a, b, c) = random_witness(m, 77);
        LOOKAHEAD_DISABLE.store(false, Ordering::Relaxed);
        let (p_look, c_look) = prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-la-test"));
        LOOKAHEAD_DISABLE.store(true, Ordering::Relaxed);
        let (p_classic, c_classic) =
            prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-la-test"));
        LOOKAHEAD_DISABLE.store(false, Ordering::Relaxed);
        assert_eq!(p_look, p_classic, "lookahead proof != classic proof");
        assert_eq!(c_look, c_classic, "lookahead claim != classic claim");
    }

    /// The verifier rejects a proof with a tampered round message.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn verify_rejects_mutated_proof() {
        use crate::challenger::FsChallenger;
        let m = 14usize;
        let (a, b, c) = random_witness(m, 66);
        let (mut proof, _) = prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-skip-test"));
        proof.multilinear_rounds[0].0 += F128::ONE;
        assert!(
            verify(m, &proof, &mut FsChallenger::new(b"flock-ag-skip-test")).is_err(),
            "must reject a tampered round message"
        );
    }

    /// The verifier rejects a tampered `r₁` grinding nonce: out-of-budget
    /// nonces outright (`BadR1Nonce`), and any shifted in-budget nonce — either
    /// its attempt rejects, or it lands a *different* valid `r₁` and the c-eval
    /// / sumcheck consistency breaks downstream.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn verify_rejects_tampered_r1_nonce() {
        use crate::challenger::FsChallenger;
        let m = 14usize;
        let (a, b, c) = random_witness(m, 67);
        let (proof, _) = prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-skip-test"));

        let mut oob = proof.clone();
        oob.r1_nonce = crate::genus95_curve_code::SAMPLE_ATTEMPT_BUDGET;
        assert_eq!(
            verify(m, &oob, &mut FsChallenger::new(b"flock-ag-skip-test")),
            Err(AgVerifyError::BadR1Nonce {
                nonce: oob.r1_nonce
            }),
            "must reject an out-of-budget nonce"
        );

        for delta in 1..=3u32 {
            let mut tampered = proof.clone();
            tampered.r1_nonce = proof.r1_nonce + delta;
            assert!(
                verify(m, &tampered, &mut FsChallenger::new(b"flock-ag-skip-test")).is_err(),
                "must reject nonce shifted by {delta}"
            );
        }
    }

    /// **Std-pack keystone (#4).** The AG c-claim's point — skip = `r₁` (the AG
    /// `EvaluationPoint`), rest = `r_rest` (friendly ‖ outer) — maps cleanly onto
    /// the *standard* packed witness `[skip 6 | bit6 | …]` prefix. Concretely: the
    /// standard ring-switch claim check with AG **base** weights for the skip
    /// recovers the AG zerocheck's `c_eval`:
    ///
    /// ```text
    /// claim_check( base(r₁) ⊗ eq(r_rest[0]),  fold_1b_rows(pack_witness(z)) ) == c_eval
    /// ```
    ///
    /// This is the direct analog of the RS φ₈ c-ring-switch
    /// (`ring_switch::claim_check_recovers_zhat_skip`), and is what lets the AG
    /// `(ab, c)` claims reuse the existing RS open path unchanged. No extra κ
    /// factor: the `D⁻¹` carry is already baked into `w̄`, and `r_rest[0] =
    /// friendly_challenges()[0]` supplies the matching `eq` factor for bit 6.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ag_c_claim_maps_onto_std_pack() {
        use crate::challenger::FsChallenger;
        use crate::genus95_curve_code::base_evaluation_functional;
        use crate::pcs::pack::pack_witness;
        use crate::pcs::ring_switch::{
            build_claim_weights_from_skip, claim_check, fold_1b_rows_naive,
        };
        use crate::zerocheck::univariate_skip::build_eq;

        for &m in &[13usize, 14, 15] {
            // Random bit-witness z; c = C·z = z (the C = I convention).
            let mut rng = Sha256Rng::new([m as u8; 32]);
            let nbits = 1usize << m;
            let mut z_bits = vec![false; nbits];
            let mut z_bytes = vec![0u8; nbits / 8];
            for j in 0..(nbits / 8) {
                let byte = rng.next_u64() as u8;
                z_bytes[j] = byte;
                for r in 0..8 {
                    z_bits[8 * j + r] = (byte >> r) & 1 == 1;
                }
            }
            // a, b are arbitrary here — only the c-claim is exercised.
            let (a, b) = (z_bytes.clone(), z_bytes.clone());

            let (_proof, claim) = prove(
                &a,
                &b,
                &z_bytes,
                m,
                &mut FsChallenger::new(b"flock-ag-stdpack-keystone"),
            );

            // c-claim point: skip = r₁, full (m − K_SKIP) multilinear = r_rest.
            let x_outer = &claim.r_rest;
            assert_eq!(x_outer.len(), m - K_SKIP);

            // Standard packing + ring-switch s_hat_v over the suffix x_outer[1..].
            let packed = pack_witness(&z_bits, m);
            let suffix_tensor = build_eq(&x_outer[1..]);
            assert_eq!(packed.len(), suffix_tensor.len());
            let s_hat_v = fold_1b_rows_naive(&packed, &suffix_tensor);

            // AG base weights for the skip (64), tensored with eq(x_outer[0]).
            let bf = base_evaluation_functional(&claim.r1).expect("base functional");
            let skip_w: Vec<F128> = (0..(1usize << K_SKIP)).map(|i| bf[i]).collect();
            let weights = build_claim_weights_from_skip(&skip_w, x_outer[0]);

            assert_eq!(
                claim_check(&weights, &s_hat_v),
                claim.c_eval,
                "AG c-claim claim_check != c_eval at m={m}",
            );
        }
    }

    /// The captured `s_hat_v_c` (from [`prove_capture_s_hat_v_c`]) is byte-for-byte
    /// the canonical ring-switch fold `fold_1b_rows(pack_witness(z), eq(r_rest[1..]))`
    /// the PCS open would otherwise recompute — and the `AgProof`/`AgClaim` are
    /// identical to plain [`prove`]'s. This pins the two-bank kernel + the
    /// `1/D₁`, `γ⁻¹` rescaling against the independent `fold_1b_rows` oracle.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn s_hat_v_c_matches_fold_1b_rows() {
        use crate::challenger::FsChallenger;
        use crate::pcs::pack::pack_witness;
        use crate::pcs::ring_switch::fold_1b_rows_naive;
        use crate::zerocheck::univariate_skip::build_eq;

        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Sha256Rng::new([m as u8 ^ 0x3c; 32]);
            let nbits = 1usize << m;
            let mut z_bits = vec![false; nbits];
            let mut z_bytes = vec![0u8; nbits / 8];
            for j in 0..(nbits / 8) {
                let byte = rng.next_u64() as u8;
                z_bytes[j] = byte;
                for r in 0..8 {
                    z_bits[8 * j + r] = (byte >> r) & 1 == 1;
                }
            }
            let (a, b) = (z_bytes.clone(), z_bytes.clone());

            // Plain prove (reference proof) vs capture (proof must be identical).
            let (proof_ref, _claim_ref) = prove(
                &a,
                &b,
                &z_bytes,
                m,
                &mut FsChallenger::new(b"flock-ag-shatvc"),
            );
            let (proof, claim, s_hat_v_c) = prove_capture_s_hat_v_c(
                &a,
                &b,
                &z_bytes,
                m,
                &mut FsChallenger::new(b"flock-ag-shatvc"),
            );
            assert_eq!(proof, proof_ref, "captured AgProof != plain prove at m={m}");

            // Canonical reference: fold_1b_rows over the c-claim suffix r_rest[1..].
            let packed = pack_witness(&z_bits, m);
            let suffix = build_eq(&claim.r_rest[1..]);
            let want = fold_1b_rows_naive(&packed, &suffix);
            assert_eq!(s_hat_v_c, want, "s_hat_v_c != fold_1b_rows at m={m}");
        }
    }

    /// The verifier rejects a witness that violates `a·b = c`: the systematic
    /// vanishing reconstruction injects the wrong value section, so the sumcheck
    /// running claim fails to telescope to `a_eval·b_eval`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn verify_rejects_dishonest_witness() {
        use crate::challenger::FsChallenger;
        let m = 14usize;
        let mut rng = Sha256Rng::new([77u8; 32]);
        let nbytes = (1usize << m) / 8;
        let mut a = vec![0u8; nbytes];
        let mut b = vec![0u8; nbytes];
        let mut c = vec![0u8; nbytes];
        for x in a.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        for x in b.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        for x in c.iter_mut() {
            *x = rng.next_u64() as u8; // random c, ≠ a & b
        }
        let (proof, _) = prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-skip-test"));
        assert!(
            verify(m, &proof, &mut FsChallenger::new(b"flock-ag-skip-test")).is_err(),
            "must reject a witness violating a*b=c"
        );
    }

    /// The fused-nonce grinding credit rests on pinned protocol constants —
    /// this test ties every number in the derivation together so a change to
    /// the sampler's shape or the code parameters cannot silently invalidate
    /// the 5-bit credit or the explicit-bit splits.
    #[test]
    fn credit_constants_are_pinned() {
        const GENUS: usize = 95;
        let bits_for = |n: usize| usize::BITS - n.leading_zeros();
        // Code degrees from the Riemann–Roch dimensions: deg = dim + g − 1.
        let base_deg = crate::genus95_curve_code::BASE_MESSAGE_BITS + GENUS - 1;
        let product_deg = crate::genus95_curve_code::PRODUCT_MESSAGE_BITS + GENUS - 1;
        assert_eq!(base_deg, 158);
        assert_eq!(product_deg, 316);
        assert_eq!(R1_ZERO_BOUND, product_deg + base_deg);
        // The sampler's per-point weight is 1/(2^128 · BASE_Y_DEGREE · 2^3):
        // the slot flattening over the base fiber and the three
        // Artin–Schreier choice bits. log2 of that constant is the credit.
        let sampler_denom = crate::genus95_curve_code::BASE_Y_DEGREE * 8;
        assert_eq!(sampler_denom, 32);
        assert_eq!(AG_SAMPLING_CREDIT_BITS, sampler_denom.trailing_zeros());
        // r₁: ALL bits explicit — the recursion circuit's relaxed-canonicity
        // decode (phase D) hands the fiber's 5 flattening bits back to the
        // prover, so the sampler credit no longer discounts this site.
        assert_eq!(R1_POW_BITS, bits_for(R1_ZERO_BOUND));
        assert_eq!(
            crate::lincheck::AG_LINCHECK_SKIP_POW_BITS + AG_SAMPLING_CREDIT_BITS,
            bits_for(base_deg)
        );
        // The schedule methods expose exactly these constants.
        let g = crate::zerocheck::ZerocheckGrinding::per_challenge_128();
        assert_eq!(g.ag_r1_bits(), Some(R1_POW_BITS));
        assert_eq!(
            crate::zerocheck::ZerocheckGrinding::disabled().ag_r1_bits(),
            None
        );
    }

    /// The fused predicate really gates on BOTH criteria: a nonce whose
    /// attempt lands a valid point still rejects under a PoW target its hash
    /// does not clear, and pow_bits = 0 degenerates to the plain attempt.
    #[test]
    fn fused_nonce_gates_on_pow_and_point() {
        use crate::genus95_curve_code::{
            evaluation_point_from_nonce, evaluation_point_from_nonce_pow,
        };
        let seed = [5u8; 32];
        let n = (0..20_000u32)
            .find(|&n| evaluation_point_from_nonce(&seed, n, HashKind::Sha256).is_some())
            .expect("a valid nonce exists in the budget");
        // pow_bits = 0: identical to the plain attempt.
        assert_eq!(
            evaluation_point_from_nonce_pow(&seed, n, HashKind::Sha256, 0),
            evaluation_point_from_nonce(&seed, n, HashKind::Sha256)
        );
        // A 40-bit target rejects this specific point-valid nonce (its hash
        // clears 40 zero bits with probability 2^-40 — deterministic here).
        assert_eq!(
            evaluation_point_from_nonce_pow(&seed, n, HashKind::Sha256, 40),
            None,
            "the PoW gate must reject a point-valid nonce whose hash misses the target"
        );
        // And a nonce found UNDER the 4-bit target also passes the plain attempt.
        let n4 = (0..(1u32 << 20))
            .find(|&n| evaluation_point_from_nonce_pow(&seed, n, HashKind::Sha256, 4).is_some())
            .expect("a fused nonce exists in the budget");
        assert!(evaluation_point_from_nonce(&seed, n4, HashKind::Sha256).is_some());
    }

    /// The run-list arms are value-identical to the dense kernels on an
    /// honestly zero-padded witness AND read no declared-dead bit: proving
    /// over a witness whose dead regions hold garbage (deliberately
    /// inconsistent garbage — dead `c` bits set where `a·b` is zero) yields
    /// byte-identical round-1 message, fold, `s_hat_v_c`, and full proof —
    /// the exactness `PooledDirty` requires.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn padded_arms_match_dense_and_ignore_dirty_padding() {
        use crate::challenger::FsChallenger;
        use crate::zerocheck::{PaddingRun, PaddingSpec};
        let m = 16usize;
        let spec = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 1 << 13,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 3001,
                n_blocks: 1,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 0,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 14,
                useful_bits_per_block: 9000,
                n_blocks: 1,
            },
        ]);
        // LSB-first byte mask of the useful bits.
        let mut mask = vec![0u8; (1usize << m) / 8];
        for (s, e) in spec.useful_intervals() {
            for i in s..e {
                mask[i / 8] |= 1 << (i % 8);
            }
        }
        let (a0, b0, c0) = random_witness(m, 91);
        let honest = |v: &[u8]| -> Vec<u8> { v.iter().zip(&mask).map(|(x, k)| x & k).collect() };
        let dirty = |v: &[u8], g: u8| -> Vec<u8> {
            v.iter()
                .zip(&mask)
                .map(|(x, k)| (x & k) | (g & !k))
                .collect()
        };
        let (ah, bh, ch) = (honest(&a0), honest(&b0), honest(&c0));
        let (ad, bd, cd) = (dirty(&a0, 0xA5), dirty(&b0, 0x5A), dirty(&c0, 0xFF));

        // Kernel level, round 1: dense-honest vs padded-dirty.
        let cov = spec.block_coverage(K_SKIP + N_INNER, 1usize << (m - K_SKIP - N_INNER));
        let mut rng = Sha256Rng::new([7u8; 32]);
        let r_outer: Vec<F128> = (0..m - K_SKIP - N_INNER)
            .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
            .collect();
        let eq = crate::zerocheck::univariate_skip::build_eq(&r_outer);
        let dense = prove_round1_banks(&ah, &bh, &ch, &eq);
        let padded = prove_round1_banks_padded(&ad, &bd, &cd, &eq, &cov);
        assert_eq!(dense.0, padded.0, "round-1 message");
        assert_eq!(dense.1, padded.1, "s_hat_v_c capture");

        // Kernel level, the fold + first message, at a decoded point.
        let seed = [3u8; 32];
        let pt = (0..u32::MAX)
            .find_map(|n| {
                crate::genus95_curve_code::evaluation_point_from_nonce(&seed, n, HashKind::Sha256)
            })
            .expect("a decodable nonce exists");
        let w: Vec<F128> = base_evaluation_functional(&pt)
            .expect("functional at a sampled point")
            .iter()
            .copied()
            .collect();
        let mut r_rest = friendly_challenges().to_vec();
        r_rest.extend_from_slice(&r_outer);
        let (af, bf, g1, gi) = fold_and_first_round(&ah, &bh, &w, &r_rest);
        let (afp, bfp, g1p, gip) = fold_and_first_round_padded(&ad, &bd, &w, &r_rest, &cov);
        assert_eq!((g1, gi), (g1p, gip), "first multilinear message");
        assert_eq!(af, afp, "folded a");
        assert_eq!(bf, bfp, "folded b");

        // End to end under the strict schedule: the padded-dirty proof is
        // byte-identical to the dense-honest one, and to the padded-honest
        // one, and verifies.
        let g = crate::zerocheck::ZerocheckGrinding::per_challenge_128();
        let mk = || FsChallenger::new(b"flock-ag-padded-test");
        let (p0, cl0, sv0) = prove_capture_s_hat_v_c_with_grinding(
            &ah,
            &bh,
            &ch,
            m,
            &PaddingSpec::dense(m),
            g,
            &mut mk(),
        );
        let (p1, cl1, sv1) =
            prove_capture_s_hat_v_c_with_grinding(&ad, &bd, &cd, m, &spec, g, &mut mk());
        assert_eq!(p0, p1, "proof bytes");
        assert_eq!(cl0, cl1, "claims");
        assert_eq!(sv0, sv1, "s_hat_v_c");
        let (p2, _, sv2) =
            prove_capture_s_hat_v_c_with_grinding(&ah, &bh, &ch, m, &spec, g, &mut mk());
        assert_eq!(p0, p2, "padded-honest proof bytes");
        assert_eq!(sv0, sv2, "padded-honest s_hat_v_c");
        verify_with_grinding(m, &p1, g, &mut mk()).expect("the padded-dirty proof verifies");
    }

    /// The SPARSE tail at deep low utilization (3 of 64 code blocks live):
    /// the live-span fold + four support-proportional rounds + the
    /// expand-and-resume handoff produce a proof byte-identical to the
    /// dense-honest one, under both grinding schedules, over a witness
    /// whose dead regions hold garbage.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sparse_tail_matches_dense_at_low_utilization() {
        use crate::challenger::FsChallenger;
        use crate::zerocheck::{PaddingRun, PaddingSpec};
        let m = 19usize; // 64 blocks; n_mlv = 13, so the gate holds 4 rounds
        let spec = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 1 << 13,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 0,
                n_blocks: 30,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 5000,
                n_blocks: 1,
            },
        ]); // + an implicit 31-block trailing gap
        let mut mask = vec![0u8; (1usize << m) / 8];
        for (s, e) in spec.useful_intervals() {
            for i in s..e {
                mask[i / 8] |= 1 << (i % 8);
            }
        }
        let (a0, b0, c0) = random_witness(m, 55);
        let honest = |v: &[u8]| -> Vec<u8> { v.iter().zip(&mask).map(|(x, k)| x & k).collect() };
        let dirty = |v: &[u8], g: u8| -> Vec<u8> {
            v.iter()
                .zip(&mask)
                .map(|(x, k)| (x & k) | (g & !k))
                .collect()
        };
        let (ah, bh, ch) = (honest(&a0), honest(&b0), honest(&c0));
        let (ad, bd, cd) = (dirty(&a0, 0xC3), dirty(&b0, 0x3C), dirty(&c0, 0xFF));
        for g in [
            crate::zerocheck::ZerocheckGrinding::disabled(),
            crate::zerocheck::ZerocheckGrinding::per_challenge_128(),
        ] {
            let mk = || FsChallenger::new(b"flock-ag-sparse-test");
            let (p0, cl0, sv0) = prove_capture_s_hat_v_c_with_grinding(
                &ah,
                &bh,
                &ch,
                m,
                &PaddingSpec::dense(m),
                g,
                &mut mk(),
            );
            let (p1, cl1, sv1) =
                prove_capture_s_hat_v_c_with_grinding(&ad, &bd, &cd, m, &spec, g, &mut mk());
            assert_eq!(p0, p1, "proof bytes (grinding: {})", g.enabled);
            assert_eq!(cl0, cl1, "claims");
            assert_eq!(sv0, sv1, "s_hat_v_c");
            verify_with_grinding(m, &p1, g, &mut mk()).expect("the sparse-path proof verifies");
        }
    }

    /// Full grinding roundtrip on the fused transcript + fused-nonce tamper
    /// rejection: any change to `r1_nonce` must reject (bad PoW, bad point,
    /// or — if both criteria fluke — a different r₁ failing the c-eval bind).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn grinding_roundtrip_rejects_fused_nonce_tampers() {
        use crate::challenger::FsChallenger;
        let m = 14usize;
        let (a, b, c) = random_witness(m, 77);
        let g = crate::zerocheck::ZerocheckGrinding::per_challenge_128();
        let mk = || FsChallenger::new(b"flock-ag-grind-test");
        let (proof, claim, _) = prove_capture_s_hat_v_c_with_grinding(
            &a,
            &b,
            &c,
            m,
            &crate::zerocheck::PaddingSpec::dense(m),
            g,
            &mut mk(),
        );
        let vclaim =
            verify_with_grinding(m, &proof, g, &mut mk()).expect("honest fused proof verifies");
        assert_eq!(vclaim, claim);

        for delta in [1u32, 2, 3, 17] {
            let mut bad = proof.clone();
            bad.r1_nonce = proof.r1_nonce.wrapping_add(delta);
            assert!(
                verify_with_grinding(m, &bad, g, &mut mk()).is_err(),
                "tampered fused r1 nonce (+{delta}) accepted"
            );
        }
        // The grinding-nonce vector still binds: wrong count rejects.
        let mut bad = proof.clone();
        bad.grinding_nonces.push(0);
        assert!(matches!(
            verify_with_grinding(m, &bad, g, &mut mk()),
            Err(AgVerifyError::BadGrindingNonceCount { .. })
        ));
    }
}
