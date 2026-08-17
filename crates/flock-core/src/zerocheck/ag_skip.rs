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

use super::multilinear::{
    fold_and_compute_round_pair_into, fold_in_place_pair, fold1_lookahead_into,
    fold2_lookahead_into, lookahead_msg_first, lookahead_msg_second, round_pair_naive,
};
use crate::challenger::Challenger;
use crate::field::{F128, F256Unreduced, mul_by_x};
use crate::genus95_curve_code::{
    EvaluationPoint, base_evaluation_functional, product_evaluation_functional,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// Bench-only A/B toggle: when set, [`prove`]'s tail uses the general
/// `fold_and_compute_round_pair_into` for *every* round instead of the
/// friendly-Horner kernel on rounds 1..=5. Lets one process time both paths
/// back-to-back so thermal drift cancels (the friendly win is small vs the
/// cross-process noise floor). Output is bit-identical either way.
pub static DISABLE_FRIENDLY_HORNER: AtomicBool = AtomicBool::new(false);

/// Bench-only A/B toggle: when set, the tail's ping-pong buffers `a_nxt`/`b_nxt`
/// are `vec![F128::ZERO; n_in/2]` (the old serial-zero-filled, non-pooled path)
/// instead of uninit pooled `take_f128`. Lets one process time both back-to-back.
pub static NXT_ZEROFILL: AtomicBool = AtomicBool::new(false);

/// Bench-only A/B toggle: when set, round 1 uses the unfused
/// [`crate::genus95_curve_code::round1::round1_slp_packed_banks`] instead of the
/// fused/lazy-reduction [`round1_slp_packed_banks_fused`]. Lets one process time
/// both back-to-back so thermal drift cancels. Output is bit-identical either way.
pub static ROUND1_UNFUSED: AtomicBool = AtomicBool::new(false);

/// A/B toggle: when set, the mlv tail runs the classic one-round-per-pass loop
/// (friendly-Horner + general fused kernels) instead of sumcheck LOOKAHEAD
/// ([`super::multilinear::fold2_lookahead_into`]) — one pass per TWO rounds,
/// deriving the second message from the bivariate Q. The proof is bit-identical
/// either way (see `lookahead_matches_classic`); lookahead cuts tail traffic
/// by ~44%.
pub static LOOKAHEAD_DISABLE: AtomicBool = AtomicBool::new(false);

/// Opt-in toggle: when set, lookahead iterations i=1,3 use the friendly-Horner
/// Q accumulation ([`lookahead_friendly_pass`]) instead of the general eq
/// multiply. Bit-identical output. OFF by default: on M-series Air it measures
/// −1.5% (the 8×256-bit Horner accumulators spill — 32 GPRs — and the wide
/// shifts on spilled accs cost more than the 8 PMULLs they replace, which hide
/// under memory traffic anyway). Re-evaluate on M4 Max, where higher DRAM
/// bandwidth may expose the mult savings.
pub static LOOKAHEAD_FRIENDLY: AtomicBool = AtomicBool::new(false);

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
    use rayon::prelude::*;
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
    use rayon::prelude::*;
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
            // One fused pass over the 64 pairs (reverse, for the γ²-Horner): fold
            // the four rows at r₁, write them to a_mlv/b_mlv, AND accumulate the
            // first message from those just-folded values — no read-back of the
            // folded array (the two-pass version re-read 2 KB/block). γ² = x², so
            // each Σ weight is a pure 2-bit shift: a *wide* Horner on a 256-bit
            // accumulator (no per-step reduction — max degree 2·63+127 = 253 <
            // 256), reduced once per block.
            let base = outer * 128;
            let mut s1 = F256Unreduced::ZERO;
            let mut s_inf = F256Unreduced::ZERO;
            for inner in (0..64).rev() {
                let (i0, i1) = (2 * inner, 2 * inner + 1);
                let a0 = byte_dot(a_packed, base + i0, &table);
                let a1 = byte_dot(a_packed, base + i1, &table);
                let b0 = byte_dot(b_packed, base + i0, &table);
                let b1 = byte_dot(b_packed, base + i1, &table);
                am[i0] = a0;
                am[i1] = a1;
                bm[i0] = b0;
                bm[i1] = b1;
                s1 = shl2_xor(s1, a1 * b1);
                s_inf = shl2_xor(s_inf, (a0 + a1) * (b0 + b1));
            }
            let e = eo * d1_inv;
            (e * s1.reduce(), e * s_inf.reduce())
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (r, s)| (p + r, q + s));
    (a_mlv, b_mlv, g1, g_inf)
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
    use rayon::prelude::*;
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

/// Friendly-Horner LOOKAHEAD pass — the lookahead analog of
/// [`fold_and_friendly_round_pair_into`], for the two lookahead iterations
/// whose eq still spans friendly dims (i=1: dims 3..6, SHIFT=8; i=3: dims
/// 5..6, SHIFT=32). The 8 Q-sums accumulate via `shl_xor::<SHIFT>` wide-Horner
/// (pure shifts) over the γ-geometric friendly-lo dims instead of 8 `eq_lo`
/// PMULLs per position; the outer eq keeps the general split multiply.
/// `PER_U` = 8 (entry, fold one pending challenge) or 16 (steady, fold two).
/// Output identical to the general lookahead pass (exact field arithmetic).
#[allow(clippy::too_many_arguments)]
fn lookahead_friendly_pass<const SHIFT: u32, const PER_U: usize>(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rhos: (F128, F128),
    lo_size: usize,
    eq_outer_lo: &[F128],
    eq_outer_hi: &[F128],
    c_inv: F128,
) -> crate::zerocheck::multilinear::LookaheadSums {
    use crate::zerocheck::multilinear::{lookahead_finish, lookahead_products};
    use rayon::prelude::*;
    let n_u = a.len() / PER_U;
    debug_assert_eq!(a_out.len(), 4 * n_u);
    let n_ol = eq_outer_lo.len();
    debug_assert_eq!(lo_size * n_ol * eq_outer_hi.len(), n_u);
    // Wide-Horner degree bound: SHIFT·(lo_size−1) + 127 < 256.
    debug_assert!(SHIFT as usize * (lo_size - 1) + 127 < 256);
    let chunk_u = lo_size * n_ol; // u-positions per outer-hi chunk
    let sums = a_out
        .par_chunks_mut(4 * chunk_u)
        .zip(b_out.par_chunks_mut(4 * chunk_u))
        .enumerate()
        .map(|(oh, (ao, bo))| {
            let mut chunk_acc = [F256Unreduced::ZERO; 8];
            for ol in 0..n_ol {
                let mut horner = [F256Unreduced::ZERO; 8];
                // Reverse, so the first-processed position carries the highest
                // power of ω = x^SHIFT.
                for p in (0..lo_size).rev() {
                    let u = (oh * n_ol + ol) * lo_size + p;
                    let mut ga = [F128::ZERO; 4];
                    let mut gb = [F128::ZERO; 4];
                    for v in 0..4usize {
                        let base = u * PER_U + v * (PER_U / 4);
                        let (fa, fb) = if PER_U == 8 {
                            (
                                a[base] + rhos.0 * (a[base] + a[base + 1]),
                                b[base] + rhos.0 * (b[base] + b[base + 1]),
                            )
                        } else {
                            let xa0 = a[base] + rhos.0 * (a[base] + a[base + 1]);
                            let xa1 = a[base + 2] + rhos.0 * (a[base + 2] + a[base + 3]);
                            let xb0 = b[base] + rhos.0 * (b[base] + b[base + 1]);
                            let xb1 = b[base + 2] + rhos.0 * (b[base + 2] + b[base + 3]);
                            (xa0 + rhos.1 * (xa0 + xa1), xb0 + rhos.1 * (xb0 + xb1))
                        };
                        ga[v] = fa;
                        gb[v] = fb;
                        let o = 4 * (ol * lo_size + p) + v;
                        ao[o] = fa;
                        bo[o] = fb;
                    }
                    let prods = lookahead_products(&ga, &gb);
                    for k in 0..8 {
                        horner[k] = shl_xor_generic::<SHIFT>(horner[k], prods[k]);
                    }
                }
                let el = eq_outer_lo[ol];
                for k in 0..8 {
                    chunk_acc[k] ^= el.mul_unreduced(horner[k].reduce());
                }
            }
            let eh = eq_outer_hi[oh] * c_inv;
            let mut out = [F128::ZERO; 8];
            for k in 0..8 {
                out[k] = eh * chunk_acc[k].reduce();
            }
            out
        })
        .reduce(
            || [F128::ZERO; 8],
            |mut p, q| {
                for k in 0..8 {
                    p[k] += q[k];
                }
                p
            },
        );
    lookahead_finish(sums)
}

/// [`shl_xor`] with the shift as a const generic (the friendly bases here are
/// `x^8` and `x^32`; both < 64 so the plain path suffices).
#[inline]
fn shl_xor_generic<const S: u32>(acc: F256Unreduced, p: F128) -> F256Unreduced {
    let inv = 64 - S;
    F256Unreduced {
        r0: (acc.r0 << S) ^ p.lo,
        r1: ((acc.r1 << S) | (acc.r0 >> inv)) ^ p.hi,
        r2: (acc.r2 << S) | (acc.r1 >> inv),
        r3: (acc.r3 << S) | (acc.r2 >> inv),
    }
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
/// rejection sampling exhausts its attempt budget. Baked from one offline sample.
///
/// No longer used by the AG-skip `r₁` derivation (the nonce-grind path panics
/// on exhaustion instead — an accepted fallback claim would let a cheating
/// prover pin `r₁` to this public constant). Still backs
/// [`crate::lincheck::SkipPoint::sample_fresh`], where BOTH sides replay the
/// same deterministic loop, so a prover cannot claim failure unilaterally:
/// forcing it costs `~1/P_fail ≈ 2^289`, far above the `2^128` target. (Keep
/// linked to the sampler's attempt cap; lowering the cap raises `P_fail`.)
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
/// rather than the expected ~100 (worst-case 20 000) replayed ones.
///
/// Per-attempt acceptance is ~1%, so exhausting all
/// [`SAMPLE_ATTEMPT_BUDGET`](crate::genus95_curve_code::SAMPLE_ATTEMPT_BUDGET)
/// nonces has probability ~2⁻²⁸⁹ — a completeness error we accept (panic)
/// instead of a verifier-side fallback point: an unconditionally accepted
/// fallback claim would let a cheating prover fix `r₁` to a public constant.
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
    unreachable!("r1 nonce grind exhausted its budget (probability ~2^-289)")
}

/// Verifier: re-derive `r₁` from the proof's nonce — range-check it, run the
/// single attempt, and reject unless it lands a valid point. The verifier does
/// NOT check the nonce is the prover's minimal one, so a cheating prover may
/// pick any of the ~200 expected valid nonces in the budget: ≤ log₂(20 000) ≈
/// 14.3 bits of grinding over `r₁`, charged to the soundness budget exactly
/// like a PoW grind.
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
    prove_from_round1(a_packed, b_packed, msg, &r_outer, challenger)
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
    let (proof, claim) = prove_from_round1(a_packed, b_packed, msg, &r_outer, challenger);
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
    let (res_ab, bank0, bank1) = if ROUND1_UNFUSED.load(Ordering::Relaxed) {
        crate::genus95_curve_code::round1::round1_slp_packed_banks(a_packed, b_packed, c_packed, eq)
    } else {
        crate::genus95_curve_code::round1::round1_slp_packed_banks_fused(
            a_packed, b_packed, c_packed, eq,
        )
    };
    let di = d_inv();
    let msg = Round1Message {
        ab_fresh: (0..158).map(|s| di * res_ab[s]).collect(),
        c_msg: (0..64).map(|i| di * (bank0[i] + bank1[i])).collect(),
    };
    // Canonical s_hat_v_c[skip + b·64] = bank_b[skip] / D₁, with γ⁻¹ for bank 1
    // (the kernel's odd-i bank carries an extra γ from friendly bit 0 = 1).
    // 1/D₁ = (1+γ)·κ since D = (1+γ)·D₁ and κ = D⁻¹ = d_inv().
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
    challenger: &mut C,
) -> (AgProof, AgClaim) {
    challenger.observe_f128_slice(&msg.ab_fresh);
    challenger.observe_f128_slice(&msg.c_msg);

    let (r1, r1_nonce) = sample_r1_prover(challenger);
    let c_eval = eval_c_at(&msg, &r1);

    let bf = base_evaluation_functional(&r1).expect("denominator nonzero at r1");
    let w: Vec<F128> = bf.iter().copied().collect();
    let mut r_rest = friendly_challenges().to_vec();
    r_rest.extend_from_slice(r_outer);

    // Round 0: fused fold of a,b at r1 + the first (full-size) multilinear message.
    let (a_mlv, b_mlv, g1_0, ginf_0) = fold_and_first_round(a_packed, b_packed, &w, &r_rest);
    let (rounds, rhos, a_eval, b_eval) =
        mlv_tail_fs(a_mlv, b_mlv, g1_0, ginf_0, &r_rest, challenger);

    let proof = AgProof {
        round1_ab: msg.ab_fresh,
        round1_c: msg.c_msg,
        r1_nonce,
        multilinear_rounds: rounds,
        final_a_eval: a_eval,
        final_b_eval: b_eval,
        final_c_eval: c_eval,
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
    if NXT_ZEROFILL.load(Ordering::Relaxed) {
        crate::suboptimal_path!(
            "zero-filled tail scratch (NXT_ZEROFILL set)",
            "pooled uninit scratch (default)"
        );
    }
    if DISABLE_FRIENDLY_HORNER.load(Ordering::Relaxed) {
        crate::suboptimal_path!(
            "general kernel on friendly rounds (DISABLE_FRIENDLY_HORNER set)",
            "friendly-Horner kernel (default)"
        );
    }
    if LOOKAHEAD_FRIENDLY.load(Ordering::Relaxed) {
        crate::suboptimal_path!(
            "friendly-Horner lookahead (LOOKAHEAD_FRIENDLY set; measured −1.5% on Air)",
            "general lookahead (default)"
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
        // (`NXT_ZEROFILL` restores the old vec![ZERO] path for a within-process A/B.)
        if NXT_ZEROFILL.load(Ordering::Relaxed) {
            (vec![F128::ZERO; n_in / 2], vec![F128::ZERO; n_in / 2])
        } else {
            (
                crate::scratch::take_f128(n_in / 2),
                crate::scratch::take_f128(n_in / 2),
            )
        }
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
            // Friendly-Horner lookahead for the iterations whose eq still spans
            // γ-geometric friendly dims: i=1 (dims 3..6) and i=3 (dims 5..6).
            // Opt-in (see LOOKAHEAD_FRIENDLY): measured slower on Air.
            let use_friendly = LOOKAHEAD_FRIENDLY.load(Ordering::Relaxed);
            let q = if let Some(r2) = pending2 {
                if use_friendly && i == 3 {
                    lookahead_friendly_pass::<32, 16>(
                        &a_mlv,
                        &b_mlv,
                        ao,
                        bo,
                        (rho_prev, r2),
                        4,
                        &split_outer.lo,
                        &split_outer.hi,
                        friendly_norm(4),
                    )
                } else {
                    fold2_lookahead_into(&a_mlv, &b_mlv, ao, bo, (rho_prev, r2), &r_rest[i + 2..])
                }
            } else if use_friendly && i == 1 {
                lookahead_friendly_pass::<8, 8>(
                    &a_mlv,
                    &b_mlv,
                    ao,
                    bo,
                    (rho_prev, F128::ZERO),
                    16,
                    &split_outer.lo,
                    &split_outer.hi,
                    friendly_norm(2),
                )
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
            let use_friendly =
                (1..=5).contains(&i) && !DISABLE_FRIENDLY_HORNER.load(Ordering::Relaxed);
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

    challenger.observe_label(b"flock-ag-skip-v1");
    let r_outer = challenger.sample_f128_vec(m - K_SKIP - N_INNER);
    challenger.observe_f128_slice(&proof.round1_ab);
    challenger.observe_f128_slice(&proof.round1_c);

    let r1 = replay_r1_verifier(challenger, proof.r1_nonce)?;
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

    let mut rhos = Vec::with_capacity(n_mlv);
    for i in 0..n_mlv {
        let (g1, g_inf) = proof.multilinear_rounds[i];
        let r_eq = r_rest[i];
        let g0 = (c_running + r_eq * g1) * (F128::ONE + r_eq).inv();
        challenger.observe_f128(g1);
        challenger.observe_f128(g_inf);
        let rho = challenger.sample_f128();
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
        use crate::hash::HashKind;
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
        LOOKAHEAD_FRIENDLY.store(true, Ordering::Relaxed);
        let (p_friend, c_friend) =
            prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-la-test"));
        LOOKAHEAD_FRIENDLY.store(false, Ordering::Relaxed);
        LOOKAHEAD_DISABLE.store(true, Ordering::Relaxed);
        let (p_classic, c_classic) =
            prove(&a, &b, &c, m, &mut FsChallenger::new(b"flock-ag-la-test"));
        LOOKAHEAD_DISABLE.store(false, Ordering::Relaxed);
        assert_eq!(p_look, p_classic, "lookahead proof != classic proof");
        assert_eq!(c_look, c_classic, "lookahead claim != classic claim");
        assert_eq!(
            p_friend, p_classic,
            "friendly-lookahead proof != classic proof"
        );
        assert_eq!(
            c_friend, c_classic,
            "friendly-lookahead claim != classic claim"
        );
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
}
