//! Multilinear sumcheck — rounds 2..(m − k_skip + 1) of the zerocheck protocol.
//!
//! After the round-1 URM and the verifier's univariate-skip fold-point `z`, the
//! protocol enters a standard multilinear sumcheck over `n = m − k_skip` variables.
//! For the **extract_c** variant, only AB participate (C was pinned down at round
//! 1 as `res_C_lifted`), so the polynomial we sumcheck is
//!
//!   `Σ_x eq(r_rest, x) · a_mlv(x) · b_mlv(x)`
//!
//! with claim `P^{AB}(z)` from round 1. Each subsequent round sends `(P_r(1),
//! P_r(∞))` via the Karatsuba ∞-trick.
//!
//! This module begins with the **naive reference** (separately compute the
//! Lagrange-weighted fold, then a direct sum for the round-2 message). The
//! optimized fused-fold-plus-round-2 implementation (`uni_skip_fold_and_compute
//! _round_pair_ghash` in the C++) will be added next and cross-checked against
//! these naive functions.
//!
//! **Index convention** (matches the C++ extract_c pipeline's `sumcheck_round_pair`
//! and the NEON `fold_in_place_pair`): the **low bit** of the multilinear index
//! is bound first. So `a_mlv[2k]` is the X=0 value and `a_mlv[2k+1]` is the X=1
//! value, paired by the round message and the fold.
//!
//! For `mlv_challenges = [r_0, …, r_{n-1}]` (one per round) built so `build_eq`
//! places `r_i` at bit i, **round r=2 uses `mlv_challenges[0]`** for the
//! variable being bound, with eq over `mlv_challenges[1..]` for the remaining
//! variables. Subsequent rounds peel off `mlv_challenges[1]`, etc.
//!
//! **Round message format** (matches the C++): returns `(r_now · G(1), G(∞))`
//! where `r_now` is the challenge for the variable being bound *this* round.
//! The protocol polynomial sent is `Π(X) = eq(r_now, X) · G(X)` of degree 3;
//! at X=1 it equals `r_now · G(1)`, and the leading coefficient is `G(∞)`.
//! Verifier reconstructs `G(0)` from the running claim via
//! `current_claim = (1+r_now)·G(0) + r_now·G(1)`.

use crate::alloc_uninit_f128_vec;
use crate::bits::lowest_one;
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
)))]
use crate::field::f128_slice::fold_pairs;
use crate::scratch::take_f128;
use crate::zerocheck::PaddingRun;
use flock_multilinear::eq_eval as multilinear_eq_eval;
use flock_multilinear::fold_low;
use rayon::current_num_threads;
use rayon::prelude::*;
use std::array::from_fn;
use std::mem::take;
use std::ops::Range;
use std::sync::OnceLock;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use crate::field::gf2_128::x86_64::{WideGhashX4, f128x4_loadu, f128x4_set, ghash_mul_x4};
use crate::field::{F128, F256Unreduced, PHI_8_TABLE};
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::univariate_skip::{SplitEqGhash, build_eq, pack_bits};

#[cfg(target_arch = "aarch64")]
use kernels::aarch64::fold_one_row_neon_unchecked_8;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use kernels::x86_64::{fold_and_message_x86_avx512, fold_round2_pair_x86_unchecked_8};
mod kernels;

/// Returns `(pair_in_block_mask, useful_pairs_inclusive)` for the round-2
/// fused-fold kernel, from a **single-run** padding spec (the multi-run case
/// takes [`uni_skip_fold_and_round_pair_runs`] instead). A pair (post-URM
/// chunks `2k`, `2k+1`) is fully inside padding iff
/// `(k & pair_in_block_mask) >= useful_pairs_inclusive` — those pairs
/// contribute zero to both the message and the folded output (which is
/// already zero-initialized), so the kernel can `continue` past them.
///
/// `useful_pairs_inclusive` is the index AFTER the last pair that has any
/// useful chunk. The boundary "mixed" pair (one useful + one padding chunk,
/// when `useful_bits` is odd in chunk units) is INSIDE the useful range and
/// processed normally — its padding side has value 0 so the message
/// contribution is naturally correct.
fn round2_pair_skip(run: &PaddingRun, k_skip: usize) -> (usize, usize) {
    if run.k_log <= k_skip + 1 {
        return (0, usize::MAX);
    }
    let pairs_per_block = 1usize << (run.k_log - k_skip - 1);
    let chunk_bits = 1usize << k_skip;
    let useful_pairs = run.useful_bits_per_block.div_ceil(2 * chunk_bits);
    if useful_pairs >= pairs_per_block {
        return (0, usize::MAX);
    }
    (pairs_per_block - 1, useful_pairs)
}

// ---------------------------------------------------------------------------
// Lagrange weights for the univariate-skip fold at z.
// ---------------------------------------------------------------------------

/// Lagrange weights `L_i(z)` for `i ∈ 0..2^k_skip` at the fold point `z`.
///
/// `L_i(z) = ∏_{j ≠ i} (z + φ_8(j)) / (φ_8(i) + φ_8(j))` — the standard Lagrange
/// formula, with the nodes being the F_8 elements `0..2^k_skip` embedded into
/// F_{2^128} via `φ_8`. Subtraction is XOR in characteristic 2.
///
/// O(2^{2·k_skip}) field multiplies — one-time cost.
/// `Π_{v ∈ V, v ≠ 0} v` for `V = φ_8({0..2^dim−1})`.
///
/// This single constant is **every** Lagrange denominator on `V` *and on any
/// coset of it*: for a node set `N = a + V` and `i ∈ N`, as `j` ranges over
/// `N \ {i}` the difference `node_i + node_j` ranges over `V \ {0}`, so
/// `Π_{j≠i}(node_i + node_j)` is independent of both `i` and `a`. (It is the
/// formal derivative of `V`'s linearized vanishing polynomial — see
/// `phi8_node_sets_have_linearized_vanishing_polynomials`.)
///
/// Cached because in the recursion circuit it is a **public constant** costing
/// nothing; recomputing it inside the traced verifier would add `2^dim − 1`
/// constraints for a value that never varies.
/// `(den, den⁻¹)`. The **inverse is cached too**: it is a constant, and
/// inverting it per call cost one `inv` (255 native muls) every time for a
/// value that never varies. In-circuit it is a public constant, so this takes
/// it from one constraint to none.
/// `(den, den^{-1})` for the dimension-`dim` φ8 node subspace — the shared
/// Lagrange denominator of [`lagrange_weights_on_coset`]'s closed form.
/// `pub`: the recursion circuit's in-circuit lagrange lows bake `den^{-1}`
/// as a statement constant and must name the SAME value the native weights
/// use (verifier-exported references over formulas-written-twice).
pub fn subspace_denominator_pair(dim: usize) -> (F128, F128) {
    static CACHE: OnceLock<[(F128, F128); 9]> = OnceLock::new();
    let table = CACHE.get_or_init(|| {
        from_fn(|d| {
            let den = (1..(1usize << d)).fold(F128::ONE, |acc, i| acc * PHI_8_TABLE[i]);
            (den, den.inv())
        })
    });
    assert!(dim < 9, "dim {dim} exceeds PHI_8_TABLE's 2^8 nodes");
    table[dim]
}

/// `Z_N(z) · den⁻¹` — the part of every weight on `nodes` that does not depend
/// on which node. `None` when `z` is itself a node, where `Z_N(z) = 0` and the
/// weights degenerate to that node's indicator.
///
/// Split out from [`lagrange_weights_on_coset`] so a caller that needs only
/// *some* of the weights does not pay an inversion for the rest —
/// [`interpolate_at_z_combined`] needs the Λ half of a 2·ell-node set and was
/// computing, then discarding, the other 64.
fn coset_weight_scale(nodes: &[F128], dim: usize, z: F128) -> Option<F128> {
    debug_assert_eq!(nodes.len(), 1usize << dim);
    let z_n = nodes.iter().fold(F128::ONE, |acc, &s| acc * (z + s));
    if z_n.is_zero() {
        return None;
    }
    Some(z_n * subspace_denominator_pair(dim).1)
}

/// Lagrange weights on `nodes`, an F₂-subspace of dimension `dim` or a coset
/// of one, evaluated at `z`.
///
/// Uses the closed form
///
/// ```text
///     L_i(z) = Z_N(z) / ((z + node_i) · den),     Z_N(X) = Π_{s∈N}(X + s)
/// ```
///
/// with `den` the shared constant from [`subspace_denominator`]. That is
/// `O(|N|)` where the textbook product form is `O(|N|²)`.
///
/// Measured (`benches/verifier_mul_count.rs`, counting one constraint per
/// multiplication and one per inversion — the recursion circuit's cost model,
/// not the native one, where an inversion is ~255 muls):
///
/// | routine | before | after |
/// |---|---|---|
/// | `lagrange_weights_naive` | 8,192 | 194 |
/// | `interpolate_at_z_on_lambda` | 8,256 | 258 |
/// | `interpolate_at_z_combined` | 16,448 | 450 |
///
/// End to end that takes a BLAKE3 boolean verify from 183,965 to 119,721
/// constraints — **35% off the verifier's whole arithmetic** — because these
/// are called from a dozen sites (the lincheck, `pcs.rs`, `ring_switch.rs`
/// per claim), not just the zerocheck's round 1.
///
/// Natively the difference is sub-millisecond either way, which is why the
/// textbook form survived this long.
fn lagrange_weights_on_coset(nodes: &[F128], dim: usize, z: F128) -> Vec<F128> {
    match coset_weight_scale(nodes, dim, z) {
        Some(scale) => nodes.iter().map(|&s| scale * (z + s).inv()).collect(),
        // `z` landed exactly on a node, where the closed form divides by zero.
        // The weights are then that node's indicator. On a Fiat–Shamir
        // challenge this has probability ≈ 2^-121; it is handled exactly
        // anyway because natively it costs one branch. The circuit backend
        // omits the branch and carries the negligible term in its soundness
        // accounting instead — a fixed-topology circuit cannot afford it.
        None => nodes
            .iter()
            .map(|&s| if s == z { F128::ONE } else { F128::ZERO })
            .collect(),
    }
}

pub fn lagrange_weights_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(ell <= 256, "k_skip > 8 would exceed PHI_8_TABLE");
    lagrange_weights_on_coset(&PHI_8_TABLE[..ell], k_skip, z)
}

/// Lagrange weights `L_i^Λ(z)` for `i ∈ 0..2^k_skip` at the fold point `z`,
/// where the nodes are the **extension domain** `Λ = {2^k_skip, …, 2^(k_skip+1) − 1}`
/// embedded via `φ_8` (offset by `2^k_skip` from the S-domain nodes).
///
/// Used to interpolate the extract_c round-1 output `round1_c` (which carries
/// the polynomial `P^C` as its 2^k_skip evaluations on Λ) at the URM challenge `z`.
pub fn lagrange_weights_lambda_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    // Λ is the coset `φ_8(2^k_skip) + V`, so it shares V's denominator.
    lagrange_weights_on_coset(&PHI_8_TABLE[ell..2 * ell], k_skip, z)
}

/// Interpolate a degree-`< 2^k_skip` polynomial at z, given its `2^k_skip`
/// evaluations on Λ. Returns `Σ_i L_i^Λ(z) · values[i]`.
///
/// In the extract_c protocol the prover ships `round1_c` (the `P^C` polynomial
/// in Λ-form) and the verifier (or higher-level prover) needs `P^C(z) = ĉ(z, r_rest)`.
/// That value is *the c-claim* at the bound point `(z, r_rest)`.
pub fn interpolate_at_z_on_lambda(values: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values.len(), ell);
    let weights = lagrange_weights_lambda_naive(k_skip, z);
    let mut acc = F128::ZERO;
    for i in 0..ell {
        acc += weights[i] * values[i];
    }
    acc
}

/// Interpolate a degree-`< 2·2^k_skip` polynomial at z, given its `2^k_skip`
/// evaluations on Λ and the assumption that it equals **zero on S**.
///
/// This is the verifier's round-1 reconstruction trick: for an honest prover
/// the combined polynomial `P = P^{AB} + P^C` satisfies `P(λ) = 0` for every
/// `λ ∈ S` (the zerocheck identity at S). Together with the `2^k_skip`
/// evaluations on Λ that the prover sends, that's `2·2^k_skip` evaluations —
/// enough to interpolate the degree-`< 2·2^k_skip` polynomial uniquely.
///
/// Cost: `2·ell × (2·ell − 1)` F128 muls + `ell` inversions for the Lagrange
/// weights. At ell=64 that's ~16K muls + 64 inversions. Sub-millisecond
/// one-time cost in the verifier.
pub fn interpolate_at_z_combined(values_on_lambda: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values_on_lambda.len(), ell);
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    // The node set is Λ ∪ S = `φ_8({0..2^{k_skip+1}−1})`, itself an
    // F₂-subspace of dimension `k_skip + 1`; only the Λ half carries values,
    // the S half being zero by the zerocheck assumption. So the whole
    // `O(ell²)` double loop collapses to the same closed form — and only the
    // Λ-half weights are ever used, so only those are computed. (Materializing
    // all `2·ell` and indexing the top half cost 64 inversions per call for
    // values immediately discarded.)
    let nodes = &PHI_8_TABLE[..2 * ell];
    match coset_weight_scale(nodes, k_skip + 1, z) {
        Some(scale) => {
            let mut acc = F128::ZERO;
            for i in 0..ell {
                acc += scale * (z + nodes[ell + i]).inv() * values_on_lambda[i];
            }
            acc
        }
        // `z` is one of the nodes: on Λ the interpolant is that node's value,
        // on S it is zero (the zerocheck assumption this reconstruction rests on).
        None => nodes[ell..]
            .iter()
            .position(|&s| s == z)
            .map_or(F128::ZERO, |i| values_on_lambda[i]),
    }
}

/// Evaluate the multilinear eq polynomial at a point: `eq(r, x) = Π_i (1 + r_i + x_i)`
/// for `r, x ∈ F_{2^128}^n` (char-2 simplification of `(1-r)(1-x) + r·x`).
pub fn eq_eval(r: &[F128], x: &[F128]) -> F128 {
    multilinear_eq_eval(r, x, F128::ONE)
}

// ---------------------------------------------------------------------------
// Fold a Boolean witness at z.
// ---------------------------------------------------------------------------

/// Evaluate the univariate-skip polynomial at the fold point `z`, given the
/// precomputed Lagrange `weights`. Returns the multilinear extension table
/// `a_mlv` of length `2^(m − k_skip)` over F_{2^128}.
///
///   `a_mlv[x_rest] = Σ_s a(s, x_rest) · L_s(z)`
///
/// `a(s, x_rest)` is the witness bit at index `x_rest * 2^k_skip + s` (low
/// bits = skip variable, high bits = rest variables).
pub fn fold_at_z_naive(witness: &[bool], m: usize, k_skip: usize, weights: &[F128]) -> Vec<F128> {
    assert!(k_skip <= m);
    let ell = 1usize << k_skip;
    let n_rest = 1usize << (m - k_skip);
    assert_eq!(witness.len(), 1usize << m);
    assert_eq!(weights.len(), ell);

    let mut folded = vec![F128::ZERO; n_rest];
    for x_rest in 0..n_rest {
        let base = x_rest * ell;
        let mut acc = F128::ZERO;
        for s in 0..ell {
            if witness[base + s] {
                acc += weights[s];
            }
        }
        folded[x_rest] = acc;
    }
    folded
}

// ---------------------------------------------------------------------------
// Naive round-2 prover message (AB-pair multilinear sumcheck).
// ---------------------------------------------------------------------------

/// Round-2 (and any subsequent round) prover message for the AB-pair
/// multilinear sumcheck.
///
/// Inputs:
/// - `a_mlv`, `b_mlv`: F128 vectors of length `2^n` for some `n ≥ 1`.
/// - `r`: full eq challenges, length `n`. `r[0]` is the challenge for the
///   variable being bound *this* round; `r[1..]` is for the remaining `n − 1`
///   variables.
///
/// Output: `(r[0] · G(1), G(∞))` for the round polynomial `G(X) = Σ_{x'} eq(r[1..], x')
/// · a_mlv(X, x') · b_mlv(X, x')`, where `a_mlv(0, x') = a_mlv[2x']` and
/// `a_mlv(1, x') = a_mlv[2x' + 1]` (low bit bound).
///
/// The `r[0]` prefactor matches the C++ `sumcheck_round_pair` convention: the
/// quantity sent on the wire is `Π(1) = eq(r[0], 1) · G(1) = r[0] · G(1)`,
/// where `Π(X) = eq(r[0], X) · G(X)` is the actual round polynomial.
pub fn round_pair_naive(a_mlv: &[F128], b_mlv: &[F128], r: &[F128]) -> (F128, F128) {
    let n = a_mlv.len();
    assert_eq!(b_mlv.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r.len(), log_n);

    let eq_remaining = build_eq(&r[1..]);
    assert_eq!(eq_remaining.len(), half);

    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for x_prime in 0..half {
        let a0 = a_mlv[2 * x_prime];
        let a1 = a_mlv[2 * x_prime + 1];
        let b0 = b_mlv[2 * x_prime];
        let b1 = b_mlv[2 * x_prime + 1];
        let eq_x = eq_remaining[x_prime];
        g_one += eq_x * a1 * b1;
        // Char-2: (a_1 − a_0)(b_1 − b_0) = (a_0 + a_1)(b_0 + b_1).
        g_inf += eq_x * (a0 + a1) * (b0 + b1);
    }
    (r[0] * g_one, g_inf)
}

// ---------------------------------------------------------------------------
// Naive fused (fold at z + round-2 message) for AB-pair.
// ---------------------------------------------------------------------------

/// Naive fold (at the univariate-skip challenge `z`) of `a` and `b`, plus the
/// round-2 prover message on the resulting multilinear polynomials.
///
/// `mlv_challenges` is of length `m − k_skip` — one challenge per multilinear
/// round. `mlv_challenges[0]` is for the variable bound in round 2 (this
/// round's message uses it as the `r_now` multiplier); `mlv_challenges[1..]`
/// is for subsequent rounds (eq table).
///
/// This is the *unfused* reference: it computes the fold and the round-2
/// message in two separate passes. The optimized version (next) will do both
/// in one pass through the witness.
///
/// Returns `(a_mlv, b_mlv, mlv_challenges[0] · G(1), G(∞))`.
pub fn uni_skip_fold_and_round_pair_naive(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert!(
        m > k_skip,
        "need at least one multilinear variable past the skip"
    );
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let weights = lagrange_weights_naive(k_skip, z);
    let a_mlv = fold_at_z_naive(a, m, k_skip, &weights);
    let b_mlv = fold_at_z_naive(b, m, k_skip, &weights);
    let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, mlv_challenges);
    (a_mlv, b_mlv, msg_1, msg_inf)
}

// ---------------------------------------------------------------------------
// Optimized fused fold + round-2 message.
// ---------------------------------------------------------------------------

/// Precomputed fold table for the univariate-skip fold at a fixed `z`.
///
/// Storage: `n_chunks × 256` F128 entries (32 KB at `k_skip=6`). For each
/// byte-chunk `j ∈ 0..n_chunks` and byte value `v ∈ 0..256`:
///
///   `data[j * 256 + v] = Σ_{b : bit b of v set} weights[8j + b]`
///
/// where `weights = lagrange_weights_naive(k_skip, z)`. Built incrementally by
/// XOR-composition over the set bits of `v` (one XOR per non-power-of-2 entry).
///
/// Per-row fold then becomes one table lookup + XOR per byte (n_chunks lookups
/// total instead of `ell` Lagrange multiplications).
#[derive(Clone, Debug)]
pub struct UniSkipFoldTable {
    pub n_chunks: usize,
    pub data: Vec<F128>,
}

impl UniSkipFoldTable {
    pub fn new(k_skip: usize, z: F128) -> Self {
        let ell = 1usize << k_skip;
        assert_eq!(ell % 8, 0, "k_skip must be ≥ 3 (need ell divisible by 8)");
        let n_chunks = ell / 8;
        let weights = lagrange_weights_naive(k_skip, z);

        let mut data = vec![F128::ZERO; n_chunks * 256];
        for j in 0..n_chunks {
            let basis = &weights[8 * j..8 * j + 8];
            // v = 0: zero (already initialized).
            for b in 0..8 {
                data[j * 256 + (1 << b)] = basis[b];
            }
            // Non-powers-of-2: composed by XOR of (v ^ lo_bit) and lo_bit entries.
            for v in 3usize..256 {
                if (v & (v - 1)) == 0 {
                    continue; // skip powers of 2 (already written)
                }
                let lo_bit = lowest_one(v);
                let parent = v ^ lo_bit;
                data[j * 256 + v] = data[j * 256 + parent] + data[j * 256 + lo_bit];
            }
        }
        Self { n_chunks, data }
    }

    /// Scalar one-row fold: `Σ_j table[j][bytes[j]]`. Ports the NEON
    /// `uni_skip_fold_one_output_ghash` in scalar form.
    #[inline]
    pub fn fold_one_row(&self, bytes: &[u8]) -> F128 {
        assert_eq!(bytes.len(), self.n_chunks);
        let mut acc = F128::ZERO;
        for j in 0..self.n_chunks {
            acc += self.data[j * 256 + bytes[j] as usize];
        }
        acc
    }
}

/// Optimized fused fold (at the URM challenge `z`, baked into `table`) plus
/// round-2 prover message. **Packed input** (LSB-first bit packing). **Parallel
/// by default** via rayon — the outer x_hi loop is distributed across workers,
/// each writing to a disjoint chunk of `a_folded`/`b_folded` via `par_chunks_mut`
/// and accumulating its own `(sum1_contrib, sum_inf_contrib)`. The final
/// reduce sums the per-worker contributions (commutative + associative F128
/// XOR/multiply).
///
/// Algorithm (per worker, one x_hi):
/// 1. For each `(x0, x1) = (2k, 2k+1)` pair (k within this x_hi's range),
///    fold the four rows `a[x0], b[x0], a[x1], b[x1]` via the table.
/// 2. Accumulate `eq_lo · a1·b1` and `eq_lo · (a0+a1)·(b0+b1)` with deferred
///    256-bit reduction, reduced once at the end of the worker's x_lo loop.
/// 3. Outer fold by `eq.hi[x_hi]` into the worker's `(sum1_contrib, sum_inf_contrib)`.
///
/// Returns `(a_folded, b_folded, mlv_challenges[0] · G(1), G(∞))` — same
/// convention as `uni_skip_fold_and_round_pair_naive`.
///
/// To run single-threaded for debugging, set `RAYON_NUM_THREADS=1`.
///
/// `k_skip = 6` is currently hardcoded (the protocol headline).
pub fn uni_skip_fold_and_round_pair_optimized_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    uni_skip_fold_and_round_pair_optimized_packed_padded(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        &PaddingSpec::dense(m),
    )
}

/// Padding-aware variant of [`uni_skip_fold_and_round_pair_optimized_packed`].
/// Skips pairs whose post-URM chunk indices both fall in the per-block zero
/// padding: the fold output is already zero-initialized and the message
/// contribution would be zero, so we can `continue` past those pairs.
pub fn uni_skip_fold_and_round_pair_optimized_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    match padding.as_single_run() {
        // Single run: the block structure is periodic, so a pair is dead iff
        // its WITHIN-BLOCK index is past the useful prefix.
        Some(run) => {
            let (mask, useful_pairs_inclusive) = round2_pair_skip(&run, k_skip);
            fold_and_round_pair_kernel(
                a_packed,
                b_packed,
                m,
                k_skip,
                table,
                mlv_challenges,
                move |pair| (pair & mask) >= useful_pairs_inclusive,
                true,
                None,
            )
        }
        // Multi-run (the multi-table slot schedule): no periodic pattern, so
        // the predicate comes from a precomputed per-pair table instead.
        None => uni_skip_fold_and_round_pair_runs(
            a_packed,
            b_packed,
            m,
            k_skip,
            table,
            mlv_challenges,
            padding,
            true,
        ),
    }
}

/// Cut a canonical live-interval list into rayon tasks of roughly `target`
/// LIVE elements each — long intervals split, short ones coalesced across the
/// dead gaps between them. Returns the re-cut piece list together with the
/// index ranges into it that each task owns.
///
/// This is what keeps the sparse kernels' task count proportional to the
/// declared support rather than to the padded domain. Both failure modes are
/// real and were measured on the capacity sweep (counts fixed, `ν` 14 → 18):
/// one task per interval gives 367 near-empty tasks per round once the slot
/// schedule fragments (the coalescing fixes that), while a uniform cut of the
/// output domain scans 16x more pairs than are live (the splitting bound
/// keeps a task's *live* work uniform instead).
///
/// A task's pieces are contiguous in index but not in address: its output
/// range spans the gaps it skips, and it simply never writes them, which is
/// exactly the sparse contract (dead output is left untouched).
pub(crate) fn balanced_interval_tasks(
    live: &[(usize, usize)],
    target: usize,
) -> (Vec<(usize, usize)>, Vec<Range<usize>>) {
    debug_assert!(target > 0);
    let mut pieces: Vec<(usize, usize)> = Vec::with_capacity(live.len());
    for &(s, e) in live {
        let mut c = s;
        while c < e {
            let next = (c + target).min(e);
            pieces.push((c, next));
            c = next;
        }
    }
    let mut tasks: Vec<Range<usize>> = Vec::new();
    let (mut start, mut acc) = (0usize, 0usize);
    for (i, &(s, e)) in pieces.iter().enumerate() {
        acc += e - s;
        if acc >= target {
            tasks.push(start..i + 1);
            (start, acc) = (i + 1, 0);
        }
    }
    if start < pieces.len() {
        tasks.push(start..pieces.len());
    }
    (pieces, tasks)
}

/// Zero a dead pair's four output slots. The fold buffers are allocated
/// uninit, so every slot not folded into must be written — unless the caller
/// reads only the useful intervals (`write_dead = false`).
#[inline(always)]
fn kill_pair(a: &mut [F128], b: &mut [F128], x0l: usize, x1l: usize, write_dead: bool) {
    if write_dead {
        for idx in [x0l, x1l] {
            a[idx] = F128::ZERO;
            b[idx] = F128::ZERO;
        }
    }
}

/// Round-2's inner loop: fold the pairs `[ps, pe)` — which must all share one
/// `x_hi` block, so one hoisted `eq_hi` factor covers the run — and return the
/// run's UNREDUCED message accumulators.
///
/// Pair `p` reads post-URM rows `2p`, `2p+1` of the packed witness and writes
/// its two folded values to `a_out[2·(p − out_base)]` and the slot after it;
/// `out_base` is the pair index the caller's output slice starts at. Dead
/// pairs (`is_dead`) fold to zero and are skipped, zero-filled only when
/// `write_dead`.
///
/// Both round-2 dispatches — the dense one over output chunks and the
/// support-proportional one over live intervals — call THIS body, so the
/// arch-specific row folds exist once. That is deliberate: the run-list path
/// was once a second copy of this loop and silently missed the NEON/AVX-512
/// folds for as long as it had no production caller (~4.6 ms at m = 30).
#[allow(clippy::too_many_arguments)]
#[inline]
fn fold_pair_run<D>(
    a_packed: &[u8],
    b_packed: &[u8],
    table: &UniSkipFoldTable,
    eq_lo: &[F128],
    lo_size: usize,
    is_dead: &D,
    write_dead: bool,
    ps: usize,
    pe: usize,
    out_base: usize,
    a_out: &mut [F128],
    b_out: &mut [F128],
) -> (F256Unreduced, F256Unreduced)
where
    D: Fn(usize) -> bool + Sync,
{
    let lo_mask = lo_size - 1;
    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let table_ptr = table.data.as_ptr() as *const u8;
        let a_pkt_ptr = a_packed.as_ptr();
        let b_pkt_ptr = b_packed.as_ptr();

        for pair in ps..pe {
            let x0l = 2 * (pair - out_base);
            let x1l = x0l + 1;
            if is_dead(pair) {
                // Padding hole: write zero (a_folded/b_folded were alloc'd
                // uninit, so we have to write every slot we don't fold into).
                kill_pair(a_out, b_out, x0l, x1l, write_dead);
                continue;
            }
            let x0g = 2 * pair;
            let x1g = x0g + 1;

            let a0 = fold_one_row_neon_unchecked_8(table_ptr, a_pkt_ptr.add(x0g * 8));
            let b0 = fold_one_row_neon_unchecked_8(table_ptr, b_pkt_ptr.add(x0g * 8));
            let a1 = fold_one_row_neon_unchecked_8(table_ptr, a_pkt_ptr.add(x1g * 8));
            let b1 = fold_one_row_neon_unchecked_8(table_ptr, b_pkt_ptr.add(x1g * 8));

            a_out[x0l] = a0;
            a_out[x1l] = a1;
            b_out[x0l] = b0;
            b_out[x1l] = b1;

            let eq_l = eq_lo[pair & lo_mask];
            let g1 = a1 * b1;
            p1_acc ^= eq_l.mul_unreduced(g1);
            let g_inf = (a0 + a1) * (b0 + b1);
            pinf_acc ^= eq_l.mul_unreduced(g_inf);
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        let table_ptr = table.data.as_ptr();
        let a_pkt_ptr = a_packed.as_ptr();
        let b_pkt_ptr = b_packed.as_ptr();
        let mut p1_wide = WideGhashX4::zero();
        let mut pinf_wide = WideGhashX4::zero();
        let mut pair = ps;

        while pair + 4 <= pe {
            let mut a0 = [F128::ZERO; 4];
            let mut a1 = [F128::ZERO; 4];
            let mut b0 = [F128::ZERO; 4];
            let mut b1 = [F128::ZERO; 4];

            for lane in 0..4 {
                let p = pair + lane;
                let x0l = 2 * (p - out_base);
                let x1l = x0l + 1;
                if is_dead(p) {
                    kill_pair(a_out, b_out, x0l, x1l, write_dead);
                    continue;
                }

                let x0g = 2 * p;
                let x1g = x0g + 1;
                let folded = fold_round2_pair_x86_unchecked_8(
                    table_ptr,
                    a_pkt_ptr.add(x0g * 8),
                    a_pkt_ptr.add(x1g * 8),
                    b_pkt_ptr.add(x0g * 8),
                    b_pkt_ptr.add(x1g * 8),
                );
                [a0[lane], a1[lane], b0[lane], b1[lane]] = folded;
                a_out[x0l] = a0[lane];
                a_out[x1l] = a1[lane];
                b_out[x0l] = b0[lane];
                b_out[x1l] = b1[lane];
            }

            let a1x4 = f128x4_loadu(a1.as_ptr());
            let b1x4 = f128x4_loadu(b1.as_ptr());
            let a_sum_x4 = f128x4_set(a0[0] + a1[0], a0[1] + a1[1], a0[2] + a1[2], a0[3] + a1[3]);
            let b_sum_x4 = f128x4_set(b0[0] + b1[0], b0[1] + b1[1], b0[2] + b1[2], b0[3] + b1[3]);
            let g1x4 = ghash_mul_x4(a1x4, b1x4);
            let g_inf_x4 = ghash_mul_x4(a_sum_x4, b_sum_x4);
            // The run never crosses an x_hi boundary, so these 4 eq weights
            // are in bounds whenever 4 pairs remain.
            let eqx4 = f128x4_loadu(eq_lo[pair & lo_mask..].as_ptr());
            p1_wide.mul_acc(eqx4, g1x4);
            pinf_wide.mul_acc(eqx4, g_inf_x4);
            pair += 4;
        }

        // Small instances (and short live runs) can leave a 1- to 3-pair tail.
        while pair < pe {
            let x0l = 2 * (pair - out_base);
            let x1l = x0l + 1;
            if is_dead(pair) {
                kill_pair(a_out, b_out, x0l, x1l, write_dead);
                pair += 1;
                continue;
            }

            let x0g = 2 * pair;
            let x1g = x0g + 1;
            let [a0, a1, b0, b1] = fold_round2_pair_x86_unchecked_8(
                table_ptr,
                a_pkt_ptr.add(x0g * 8),
                a_pkt_ptr.add(x1g * 8),
                b_pkt_ptr.add(x0g * 8),
                b_pkt_ptr.add(x1g * 8),
            );
            a_out[x0l] = a0;
            a_out[x1l] = a1;
            b_out[x0l] = b0;
            b_out[x1l] = b1;
            let eq_l = eq_lo[pair & lo_mask];
            p1_acc ^= eq_l.mul_unreduced(a1 * b1);
            pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
            pair += 1;
        }

        p1_acc ^= p1_wide.fold();
        pinf_acc ^= pinf_wide.fold();
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )
    )))]
    {
        let n_chunks = table.n_chunks;
        for pair in ps..pe {
            let x0l = 2 * (pair - out_base);
            let x1l = x0l + 1;
            if is_dead(pair) {
                // See aarch64 branch above for why this zero write is needed.
                kill_pair(a_out, b_out, x0l, x1l, write_dead);
                continue;
            }
            let x0g = 2 * pair;
            let x1g = x0g + 1;
            let a0 = table.fold_one_row(&a_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
            let b0 = table.fold_one_row(&b_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
            let a1 = table.fold_one_row(&a_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
            let b1 = table.fold_one_row(&b_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
            a_out[x0l] = a0;
            a_out[x1l] = a1;
            b_out[x0l] = b0;
            b_out[x1l] = b1;
            let eq_l = eq_lo[pair & lo_mask];
            let g1 = a1 * b1;
            p1_acc ^= eq_l.mul_unreduced(g1);
            let g_inf = (a0 + a1) * (b0 + b1);
            pinf_acc ^= eq_l.mul_unreduced(g_inf);
        }
    }

    (p1_acc, pinf_acc)
}

/// The round-2 fused fold + message kernel, shared by BOTH padding regimes.
///
/// `is_dead(pair)` decides, for a GLOBAL post-URM pair index, whether that
/// pair's window is entirely padding — the only thing the single-run and
/// run-list paths ever disagreed about. Everything else (the per-row fold, the
/// eq weighting, the parallel-over-`x_hi` structure, the accumulator
/// reduction) is common, so the arch-specific SIMD row folds live here ONCE
/// and both regimes get them.
///
/// They used to be two copies, and the run-list copy never received the
/// NEON / AVX-512 row folds — it ran `UniSkipFoldTable::fold_one_row` scalar.
/// At `M = 30` that cost every multi-run zerocheck ~4.6 ms in this round
/// (11.2 vs 6.6 ms) for bit-identical output, which is most of the multi-table
/// zerocheck's overhead over the single-table path.
///
/// `write_dead`: when true, dead pairs are zero-filled so the whole output is
/// valid; when false (the M6 sparse-tail prover) they are left untouched and
/// the caller reads only the useful intervals.
fn fold_and_round_pair_kernel<D>(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    is_dead: D,
    write_dead: bool,
    live_pairs: Option<&[(usize, usize)]>,
) -> (Vec<F128>, Vec<F128>, F128, F128)
where
    D: Fn(usize) -> bool + Sync,
{
    assert_eq!(
        k_skip, 6,
        "optimized fold-and-round_pair variant is k_skip=6 only"
    );
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);
    assert!(
        live_pairs.is_none() || !write_dead,
        "chunk skipping leaves dead chunks unwritten — write_dead callers \
         must not pass live_pairs"
    );

    // Uninit alloc — the parallel loop below writes every slot (dense path)
    // or explicitly writes F128::ZERO at padding holes (padded path).
    // Saves ~22 ms of sequential zero-fill at m=29 (256 MB total) that would
    // otherwise cap the parallel speedup of this phase at ~2.5× on 8 cores.
    //
    // LIVE-SPAN OUTPUT on the sparse dispatch: the buffer holds 2 slots per
    // LIVE pair instead of the full padded `n_out`. The live pair count is
    // count-derived and so identical at every capacity (6.0M at the m30 load),
    // which is what takes this phase — and the scratch pool it shares with the
    // open — off the capacity axis. Dead pairs are not stored at all rather
    // than stored-and-skipped.
    let out_len = match live_pairs {
        Some(iv) => 2 * iv.iter().map(|&(s, e)| e - s).sum::<usize>(),
        None => n_out,
    };
    let mut a_folded: Vec<F128> = take_f128(out_len);
    let mut b_folded: Vec<F128> = take_f128(out_len);

    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;

    // Support-proportional dispatch (M6 run-list sparse round 2): tasks come
    // from the live pair-interval list, so both the task count and the work
    // per task follow the DECLARED SUPPORT. The previous chunk-level skip
    // still walked every pair of every partially-live chunk: at the m30 load
    // that is 6.1M pair tests at ν = 14 but 97.5M at ν = 18 (91.5M of them
    // dead) against an unchanged 6.0M live pairs, plus a `2^(m−k_skip−1)`-bool
    // liveness table (8 MB → 128 MB) to allocate, fill and stream. Both are
    // gone here: a task walks only live pairs, and `is_dead` is never
    // consulted (the interval list IS the live set).
    //
    // Value-identical to the dense dispatch: the message is a sum of terms
    // that each carry an `a·b` factor of zero off the support, and regrouping
    // the XOR accumulation is exact (`reduce` is XOR-linear, so reducing a
    // hi-run in pieces and adding equals adding then reducing).
    if let Some(iv) = live_pairs {
        // Task size: at most ~2^16 live pairs (the live work the dense
        // dispatch gives one chunk at the anchor shape, so per-task fold
        // cost stays in the measured-good regime), but never so coarse that
        // the pool starves — a small support otherwise cuts into fewer
        // tasks than threads and the round runs on one or two cores. Any
        // task partition is byte-identical (the note above: reduction is
        // XOR-linear, regrouping is exact), so the target only moves work,
        // never values.
        const LIVE_PAIRS_PER_TASK: usize = 1 << 16;
        let live_total: usize = iv.iter().map(|&(s, e)| e - s).sum();
        let threads = current_num_threads().max(1);
        let target = (live_total / (4 * threads)).clamp(1 << 10, LIVE_PAIRS_PER_TASK);
        let (pieces, tasks) = balanced_interval_tasks(iv, target);

        // Each task owns the COMPACTED output span of its pieces — 2 slots per
        // live pair, disjoint and in address order, so the buffers carve by
        // successive `split_at_mut` with no gap arithmetic (the previous
        // global carve had to skip `2*s - off` dead slots between tasks).
        let mut work: Vec<(&[(usize, usize)], &mut [F128], &mut [F128])> =
            Vec::with_capacity(tasks.len());
        {
            let (mut a_rem, mut b_rem): (&mut [F128], &mut [F128]) =
                (&mut a_folded[..], &mut b_folded[..]);
            for t in &tasks {
                let span: usize = pieces[t.clone()].iter().map(|&(s, e)| 2 * (e - s)).sum();
                let (a_task, rest) = take(&mut a_rem).split_at_mut(span);
                a_rem = rest;
                let (b_task, rest) = take(&mut b_rem).split_at_mut(span);
                b_rem = rest;
                work.push((&pieces[t.clone()], a_task, b_task));
            }
        }

        let (sum1, sum_inf) = work
            .into_par_iter()
            .map(|(task_pieces, a_task, b_task)| {
                let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
                // Slots written so far in this task. `fold_pair_run` reads at
                // the GLOBAL `2*pair` but writes at `2*(pair - out_base)`, so
                // biasing `out_base` per piece compacts the output without the
                // kernel body knowing — keeping one fold body shared with the
                // dense dispatch and all three arch variants (bd0f222).
                let mut local = 0usize;
                for &(ps, pe) in task_pieces {
                    let out_base = ps - local / 2;
                    // Split at hi-block boundaries: one `eq_hi` factor per run,
                    // hoisted out of the inner loop exactly as in the dense
                    // dispatch.
                    let mut t = ps;
                    while t < pe {
                        let run_end = (((t >> eq.n_lo) + 1) << eq.n_lo).min(pe);
                        let (p1, pinf) = fold_pair_run(
                            a_packed, b_packed, table, eq_lo, lo_size, &is_dead, write_dead, t,
                            run_end, out_base, a_task, b_task,
                        );
                        let eq_h = eq_hi[t >> eq.n_lo];
                        s1 += eq_h * p1.reduce();
                        s_inf += eq_h * pinf.reduce();
                        t = run_end;
                    }
                    local += 2 * (pe - ps);
                }
                (s1, s_inf)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
            );

        return (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf);
    }

    // Dense dispatch: each worker writes one disjoint chunk of
    // a_folded/b_folded and returns its (sum1, sum_inf) contribution.
    // Reduce by F128 XOR.
    let (sum1, sum_inf) = a_folded
        .par_chunks_mut(chunk_size)
        .zip(b_folded.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_chunk, b_chunk))| {
            let pair_idx_base = x_hi * lo_size;
            let (p1_acc, pinf_acc) = fold_pair_run(
                a_packed,
                b_packed,
                table,
                eq_lo,
                lo_size,
                &is_dead,
                write_dead,
                pair_idx_base,
                pair_idx_base + lo_size,
                pair_idx_base,
                a_chunk,
                b_chunk,
            );

            let p1 = p1_acc.reduce();
            let pinf = pinf_acc.reduce();
            let eq_h = eq_hi[x_hi];
            (eq_h * p1, eq_h * pinf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
        );

    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}

/// General run-list path for
/// [`uni_skip_fold_and_round_pair_optimized_packed_padded`]: handles
/// multi-run [`PaddingSpec`]s (the multi-table slot schedule). Same
/// parallel-over-x_hi structure and output convention as
/// the optimized kernel, but with the portable scalar per-row fold and a
/// precomputed per-pair skip table instead of the periodic mask/threshold
/// predicate. A pair (post-URM chunks `2k`, `2k+1`) covers witness bits
/// `[k·2^(k_skip+1), (k+1)·2^(k_skip+1))`; pairs whose window contains no
/// useful bits fold to zero and are skipped. Output is byte-identical to the
/// dense path when the padding/gap bits are honestly zero.
///
/// `write_dead`: when true (the public wrapper's contract), dead pairs are
/// written `F128::ZERO` so the whole output is valid; when false (the M6
/// sparse-tail prover, [`uni_skip_fold_and_round_pair_runs_sparse`]), dead
/// pairs are left untouched — the caller promises to only read the useful
/// intervals (zero-substituting the rest), which is what the sparse tail
/// does.
fn uni_skip_fold_and_round_pair_runs(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
    write_dead: bool,
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert!(
        padding.covered_bits() <= 1usize << m,
        "PaddingSpec covers {} bits but the domain has only 2^{m}",
        padding.covered_bits()
    );

    debug_assert!(write_dead, "the sparse path builds its own live-pair list");

    // Dense mode: every output slot must be written, so the dispatch stays
    // over the whole domain and the predicate comes from a precomputed
    // per-pair table.
    let pair_bits = 1usize << (k_skip + 1);
    let n_out = 1usize << (m - k_skip);
    let mut pair_useful = vec![false; n_out / 2];
    for (start, end) in padding.useful_intervals() {
        let (ps, pe) = (start / pair_bits, (end - 1) / pair_bits + 1);
        pair_useful[ps..pe].fill(true);
    }

    fold_and_round_pair_kernel(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        |pair| !pair_useful[pair],
        write_dead,
        None,
    )
}

/// [`uni_skip_fold_and_round_pair_optimized_packed_padded`]'s multi-run path
/// WITHOUT the dead-pair zero writes (M6): for sparse-support instances whose
/// tail runs the support-proportional folds from the first round, the dead
/// regions of the output are never read (the sparse tail zero-substitutes
/// them, and zeroes them explicitly on any switch back to the dense
/// kernels), so the `2·2^(m−k_skip)`-word zero fill can be skipped. Message
/// and useful-interval values are byte-identical to the public wrapper.
/// Multi-run specs only.
/// Returns the fold outputs in LIVE-SPAN (compacted) storage together with the
/// [`LiveLayout`] that maps them back to the padded domain: slot `2*rank(k)`
/// and `2*rank(k)+1` hold live pair `k`'s two outputs. Dead pairs occupy no
/// storage, so the buffers are count-derived — the same size at every capacity
/// — instead of `2^(m-k_skip)`.
pub fn uni_skip_fold_and_round_pair_runs_sparse(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, F128, F128, LiveLayout) {
    assert!(
        padding.as_single_run().is_none(),
        "the sparse round-2 variant is for multi-run specs only"
    );
    // The canonical live-pair interval list IS the skip predicate — the kernel
    // visits live pairs only, so no per-pair table is built and `is_dead` is
    // never consulted. A pair (post-URM chunks `2k`, `2k+1`) covers witness
    // bits `[k·2^(k_skip+1), (k+1)·2^(k_skip+1))`, which is exactly
    // `useful_block_intervals` at block size `2^(k_skip+1)` (merged, so
    // intervals that share a boundary pair are never processed twice).
    let pair_intervals = padding.useful_block_intervals(k_skip + 1);
    // Each live pair contributes the two adjacent output positions 2k, 2k+1,
    // so the stored set is the pair list doubled — pair-aligned by
    // construction, which is what keeps the NEXT round's fold pairs
    // well-defined under compaction.
    let store = LiveLayout::new(
        pair_intervals
            .iter()
            .map(|&(s, e)| (2 * s, 2 * e))
            .collect(),
    );
    let (a, b, msg1, msg_inf) = fold_and_round_pair_kernel(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        |_| false,
        false,
        Some(&pair_intervals),
    );
    debug_assert_eq!(a.len(), store.len(), "compacted output length");
    (a, b, msg1, msg_inf, store)
}

// ---------------------------------------------------------------------------
// Subsequent multilinear rounds (3..(m−k_skip+1)): fold + next message.
// ---------------------------------------------------------------------------

/// In-place fold of a single multilinear polynomial table at `challenge`.
/// Pairs `(a[2x], a[2x+1])` collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])`.
/// After the call, `a.len()` is halved.
pub fn fold_in_place_single(a: &mut Vec<F128>, challenge: F128) {
    fold_low(a, challenge);
}

/// In-place fold of a pair `(a, b)` of multilinear polynomial tables at
/// `challenge`. Binds the lowest bit of the index: pairs `(a[2x], a[2x+1])`
/// collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])` (and same for b).
/// After the call, `a.len()` and `b.len()` are halved.
///
/// Used at the tail of the multilinear-round sequence where the polynomial is
/// small enough that parallel/fusion overhead outweighs benefit.
pub fn fold_in_place_pair(a: &mut Vec<F128>, b: &mut Vec<F128>, challenge: F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        let b0 = b[2 * x];
        let b1 = b[2 * x + 1];
        a[x] = a0 + challenge * (a1 + a0);
        b[x] = b0 + challenge * (b1 + b0);
    }
    a.truncate(half);
    b.truncate(half);
}

/// Fused: bind one variable at `r_fold` AND compute the *next* round's prover
/// message. Returns the new (folded) `a, b` vectors (half the input size) and
/// `(r_next[0] · G(1), G(∞))` for the next round.
///
/// Parallelized via rayon: each worker reads one disjoint 4·lo_size chunk of
/// the input and writes the corresponding 2·lo_size chunk of the output.
///
/// Requires `a.len() = b.len() ≥ 8` so the post-fold polynomial has at least
/// one bit of x_lo (lo_size ≥ 2). Smaller polynomials should use the
/// unfused `fold_in_place_pair + round_pair_naive` pair.
pub fn fold_and_compute_round_pair_optimized(
    a: &[F128],
    b: &[F128],
    r_fold: F128,
    r_next: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    let half = a.len() / 2;
    // Uninit alloc — `_into` writes every slot of a_new/b_new.
    let mut a_new = alloc_uninit_f128_vec(half);
    let mut b_new = alloc_uninit_f128_vec(half);
    let (m1, mi) = fold_and_compute_round_pair_into(a, b, &mut a_new, &mut b_new, r_fold, r_next);
    (a_new, b_new, m1, mi)
}

/// Buffer-reusing variant of [`fold_and_compute_round_pair_optimized`]: writes
/// the folded `a`/`b` into the caller-provided `a_out`/`b_out` (each length
/// `a.len() / 2`) instead of allocating. Returns `(r_next[0] · G(1), G(∞))`.
///
/// Lets the multilinear-sumcheck tail ping-pong between two persistent scratch
/// buffers, so the ~22 decreasing-size buffers are allocated/freed once rather
/// than per round. The per-round `munmap` of the old buffer (64 MB at m=29)
/// runs single-threaded and otherwise caps the tail's parallel speedup.
pub fn fold_and_compute_round_pair_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
) -> (F128, F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 8);
    let half = n / 2;
    assert_eq!(a_out.len(), half);
    assert_eq!(b_out.len(), half);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 1);

    let eq = SplitEqGhash::new(&r_next[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "fold_and_compute requires lo_size ≥ 2");
    // Total non-bound multilinear vars is log_n - 1; eq covers log_n - 2 of those.
    assert_eq!(lo_size * hi_size * 2, half);

    let chunk_in = 4 * lo_size; // read chunk per worker
    let chunk_out = 2 * lo_size; // write chunk per worker
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    let (sum1, sum_inf) = a_out
        .par_chunks_mut(chunk_out)
        .zip(b_out.par_chunks_mut(chunk_out))
        .enumerate()
        .map(|(x_hi, (a_out, b_out))| {
            let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: chunk geometry supplies two inputs per output and two
            // outputs per eq_lo value; features are guaranteed by the cfg.
            let (p1, pinf) =
                unsafe { fold_and_message_x86_avx512(a_in, b_in, a_out, b_out, r_fold, eq_lo) };

            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            let (p1, pinf) = {
                // Fold a_in→a_out and b_in→b_out at r_fold. The field layer
                // selects the architecture kernel; this loop only consumes
                // the resulting values to build the message.
                fold_pairs(a_in, 0, a_out, r_fold);
                fold_pairs(b_in, 0, b_out, r_fold);

                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                // x86: 4-wide deferred-reduction accumulators for the unrolled loop;
                // the 2-wide tail still uses the scalar `*_acc` above, folded in
                // before the final reduce.
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                ))]
                // SAFETY: vpclmulqdq+avx512f guaranteed by the cfg gate.
                let (mut p1_wide, mut pinf_wide) =
                    unsafe { (WideGhashX4::zero(), WideGhashX4::zero()) };

                // Unroll 4 x_lo's per iteration when lo_size % 4 == 0 (the common
                // case for the fused path; falls back to 2-wide for lo_size==2 at
                // the smallest fused round). 16 independent r_fold muls and 8
                // independent msg muls in flight gives the M4 OoO engine and
                // 2/cy PMULL throughput maximum ILP.
                assert!(lo_size & 1 == 0, "lo_size must be even");
                let mut x_lo = 0;
                if lo_size.is_multiple_of(4) {
                    while x_lo + 4 <= lo_size {
                        let x_lo_a = x_lo;
                        // Read the just-folded pairs: (a0,a1) = (a_out[2·x_lo], a_out[2·x_lo+1]).
                        let o = 2 * x_lo;
                        let a0_a = a_out[o];
                        let a1_a = a_out[o + 1];
                        let b0_a = b_out[o];
                        let b1_a = b_out[o + 1];
                        let a0_b = a_out[o + 2];
                        let a1_b = a_out[o + 3];
                        let b0_b = b_out[o + 2];
                        let b1_b = b_out[o + 3];
                        let a0_c = a_out[o + 4];
                        let a1_c = a_out[o + 5];
                        let b0_c = b_out[o + 4];
                        let b1_c = b_out[o + 5];
                        let a0_d = a_out[o + 6];
                        let a1_d = a_out[o + 7];
                        let b0_d = b_out[o + 6];
                        let b1_d = b_out[o + 7];

                        // 8 reduced msg muls (g1 = a1·b1, g_inf = (a0+a1)(b0+b1)).
                        let g1_a = a1_a * b1_a;
                        let g1_b = a1_b * b1_b;
                        let g1_c = a1_c * b1_c;
                        let g1_d = a1_d * b1_d;
                        let g_inf_a = (a0_a + a1_a) * (b0_a + b1_a);
                        let g_inf_b = (a0_b + a1_b) * (b0_b + b1_b);
                        let g_inf_c = (a0_c + a1_c) * (b0_c + b1_c);
                        let g_inf_d = (a0_d + a1_d) * (b0_d + b1_d);
                        // Deferred-reduction accumulate: on x86 widen all 8 products
                        // 4 lanes at a time (eq_lo[x_lo_a..x_lo_a+4] is contiguous),
                        // reduced once after the loop; else scalar mul_unreduced.
                        #[cfg(all(
                            target_arch = "x86_64",
                            target_feature = "avx512f",
                            target_feature = "vpclmulqdq"
                        ))]
                        // SAFETY: vpclmulqdq+avx512f guaranteed by the cfg gate; the
                        // four eq values eq_lo[x_lo_a..x_lo_a+4] are in bounds (the
                        // 4-wide loop runs only while x_lo + 4 <= lo_size == eq_lo.len()).
                        unsafe {
                            let eq4 = f128x4_loadu(eq_lo[x_lo_a..].as_ptr());
                            p1_wide.mul_acc(eq4, f128x4_set(g1_a, g1_b, g1_c, g1_d));
                            pinf_wide.mul_acc(eq4, f128x4_set(g_inf_a, g_inf_b, g_inf_c, g_inf_d));
                        }
                        #[cfg(not(all(
                            target_arch = "x86_64",
                            target_feature = "avx512f",
                            target_feature = "vpclmulqdq"
                        )))]
                        {
                            let eq_l_a = eq_lo[x_lo_a];
                            let eq_l_b = eq_lo[x_lo_a + 1];
                            let eq_l_c = eq_lo[x_lo_a + 2];
                            let eq_l_d = eq_lo[x_lo_a + 3];
                            p1_acc ^= eq_l_a.mul_unreduced(g1_a);
                            p1_acc ^= eq_l_b.mul_unreduced(g1_b);
                            p1_acc ^= eq_l_c.mul_unreduced(g1_c);
                            p1_acc ^= eq_l_d.mul_unreduced(g1_d);
                            pinf_acc ^= eq_l_a.mul_unreduced(g_inf_a);
                            pinf_acc ^= eq_l_b.mul_unreduced(g_inf_b);
                            pinf_acc ^= eq_l_c.mul_unreduced(g_inf_c);
                            pinf_acc ^= eq_l_d.mul_unreduced(g_inf_d);
                        }

                        x_lo += 4;
                    }
                }
                // 2-wide tail (handles lo_size == 2 case and any remainder when
                // 4-wide loop is skipped or doesn't cover everything).
                while x_lo + 2 <= lo_size {
                    let x_lo_a = x_lo;
                    let x_lo_b = x_lo + 1;
                    let o = 2 * x_lo;
                    let a0_a = a_out[o];
                    let a1_a = a_out[o + 1];
                    let b0_a = b_out[o];
                    let b1_a = b_out[o + 1];
                    let a0_b = a_out[o + 2];
                    let a1_b = a_out[o + 3];
                    let b0_b = b_out[o + 2];
                    let b1_b = b_out[o + 3];

                    let eq_l_a = eq_lo[x_lo_a];
                    let eq_l_b = eq_lo[x_lo_b];
                    let g1_a = a1_a * b1_a;
                    let g1_b = a1_b * b1_b;
                    let g_inf_a = (a0_a + a1_a) * (b0_a + b1_a);
                    let g_inf_b = (a0_b + a1_b) * (b0_b + b1_b);
                    p1_acc ^= eq_l_a.mul_unreduced(g1_a);
                    p1_acc ^= eq_l_b.mul_unreduced(g1_b);
                    pinf_acc ^= eq_l_a.mul_unreduced(g_inf_a);
                    pinf_acc ^= eq_l_b.mul_unreduced(g_inf_b);

                    x_lo += 2;
                }

                // Merge the 4-wide deferred accumulators with the scalar tail, then
                // reduce once (reduction is F2-linear, so this equals the scalar
                // Σ mul_unreduced then reduce).
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                ))]
                // SAFETY: vpclmulqdq+avx512f+sse4.1 guaranteed by the cfg gate.
                unsafe {
                    p1_acc ^= p1_wide.fold();
                    pinf_acc ^= pinf_wide.fold();
                }
                let p1 = p1_acc.reduce();
                let pinf = pinf_acc.reduce();
                (p1, pinf)
            };
            let eq_h = eq_hi[x_hi];
            (eq_h * p1, eq_h * pinf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
        );

    (r_next[0] * sum1, sum_inf)
}

/// Halve a sorted, disjoint interval list under one binding round: index `x`
/// of the folded (half-size) domain is live iff `{2x, 2x+1}` intersects a
/// live input interval — `[s, e)` maps to `[s/2, (e+1)/2)` — with touching
/// output intervals merged so the list stays canonical.
/// Compacted storage for a sparse multilinear table: the buffer holds only
/// the live positions, in global order, instead of the full padded domain.
///
/// WHY. Under a count-derived multi-run spec the live support is a fixed
/// ~6.0M pairs at every capacity, but the fold buffers are sized by the
/// PADDED domain — 2^24 words at nu = 14 and 2^28 at nu = 18 for the same
/// live work. The sparse kernels already skip the dead positions, so the
/// arithmetic is capacity-free; what is not free is scattering that fixed
/// live set over a 16x wider span. MEASURED (m30, steady state): zerocheck
/// round 2 + tail cost +8.6 ms at nu = 18 over nu = 14 on identical live
/// volume, and the open pays another ~6 ms because the capacity-sized
/// buffers crowd the scratch pool.
///
/// The mapping is an interval-rank: global `y` in `intervals[i]` is stored at
/// `offs[i] + (y - intervals[i].0)`; anything outside is honestly ZERO and is
/// zero-substituted on read (never stored). Reads within a kernel task ascend,
/// so the interval index is a monotone cursor, not a search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveLayout {
    /// Canonical global intervals: sorted, disjoint, non-touching.
    intervals: Vec<(usize, usize)>,
    /// `offs[i]` = number of live positions before `intervals[i]`.
    offs: Vec<usize>,
    /// Total stored positions = compacted buffer length.
    len: usize,
}

impl LiveLayout {
    /// Build from a canonical interval list (sorted, disjoint, non-touching —
    /// exactly what [`shrink_intervals`] and `useful_block_intervals` return).
    pub fn new(intervals: Vec<(usize, usize)>) -> Self {
        debug_assert!(
            intervals.windows(2).all(|w| w[0].1 < w[1].0),
            "canonical list required (sorted, disjoint, non-touching)"
        );
        debug_assert!(intervals.iter().all(|&(s, e)| s < e), "non-empty pieces");
        let mut offs = Vec::with_capacity(intervals.len());
        let mut acc = 0usize;
        for &(s, e) in &intervals {
            offs.push(acc);
            acc += e - s;
        }
        Self {
            intervals,
            offs,
            len: acc,
        }
    }

    /// Compacted buffer length.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn intervals(&self) -> &[(usize, usize)] {
        &self.intervals
    }

    /// Compacted offset of `intervals()[i]`'s first position.
    pub fn offset_of(&self, i: usize) -> usize {
        self.offs[i]
    }

    /// Seed a monotone cursor for reads starting at global `y`.
    pub fn seek(&self, y: usize) -> usize {
        self.intervals.partition_point(|&(_, e)| e <= y)
    }

    /// Compacted slot of global `y`, or `None` when `y` is dead (value ZERO).
    /// `cur` must be a cursor from [`Self::seek`] advanced only by ASCENDING
    /// `y` — the kernels read in address order, so this stays O(1) amortized.
    #[inline]
    pub fn rank(&self, cur: &mut usize, y: usize) -> Option<usize> {
        while *cur < self.intervals.len() && self.intervals[*cur].1 <= y {
            *cur += 1;
        }
        let (s, _) = *self.intervals.get(*cur)?;
        (s <= y).then(|| self.offs[*cur] + (y - s))
    }

    /// True when the layout stores a contiguous prefix of the domain, i.e.
    /// compaction is the identity and the dense kernels can run unchanged.
    pub fn is_dense(&self, domain: usize) -> bool {
        matches!(self.intervals.as_slice(), [(0, e)] if *e == domain)
    }
}

pub fn shrink_intervals(live: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(live.len());
    for &(s, e) in live {
        let (s2, e2) = (s >> 1, e.div_ceil(2));
        match out.last_mut() {
            Some((_, prev_e)) if *prev_e >= s2 => *prev_e = (*prev_e).max(e2),
            _ => out.push((s2, e2)),
        }
    }
    out
}

/// Support-proportional variant of [`fold_and_compute_round_pair_into`] for
/// sparse-support instances (the multi-table count-derived run lists): bind
/// one variable at `r_fold` and compute the next round's message, touching
/// only the pairs that cover `live_in` — the canonical (sorted, disjoint,
/// merged) interval list outside of which the input tables are **zero** on an
/// honest witness. Cost `O(live + intervals)` instead of `O(n)`.
///
/// Contract differences from the dense kernel:
/// - Input positions outside `live_in` are never read (they may hold scratch
///   garbage); their honest value — zero — is substituted, which is what
///   makes the output byte-identical to the dense fold.
/// - Only the pair-cover of the folded live set is written; everything else
///   in `a_out`/`b_out` is left untouched. Callers switching back to a dense
///   kernel bridge through [`expand_to_dense`], which scatters only the live
///   span into a fresh zeroed buffer.
///
/// Returns `(r_next[0] · G(1), G(∞), live_out)` where `live_out` is the
/// folded domain's live list — the PAIR-ALIGNED COVER of
/// [`shrink_intervals`] of `live_in` (a superset when a live interval's
/// image ends on an odd position; the extra covered slots are zero on an
/// honest witness).
/// The message equals the dense kernel's value exactly: the skipped terms
/// each carry an `a·b` factor of zero, and field ops are exact, so dropping
/// them cannot change the sum.
pub fn fold_and_round_pair_sparse_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
    store_in: &LiveLayout,
    domain: usize,
) -> (F128, F128, LiveLayout) {
    assert!(domain.is_power_of_two() && domain >= 8);
    assert_eq!(a.len(), store_in.len());
    assert_eq!(b.len(), store_in.len());
    let n = domain;
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 1);
    let live_in = store_in.intervals();
    debug_assert!(live_in.last().is_none_or(|&(_, e)| e <= n));

    // eq split over the message's pair domain (size half/2), exactly as the
    // dense kernel: eq(k) = lo[k & mask] · hi[k >> n_lo].
    let eq = SplitEqGhash::new(&r_next[1..]);
    let lo_mask = (1usize << eq.n_lo) - 1;

    let live_out = shrink_intervals(live_in);
    let pair_cover = shrink_intervals(&live_out);
    // Stored set of the OUTPUT: 2 slots per covered pair. A pair may straddle
    // a live/dead boundary (`pair_cover` is not pair-aligned in general); the
    // dead half folds from zero-substituted reads to an honest ZERO, so
    // storing it costs one slot and keeps the layout pair-aligned for the next
    // round. Superset of `live_out`, and every extra slot holds a true zero.
    let store_out = LiveLayout::new(pair_cover.iter().map(|&(s, e)| (2 * s, 2 * e)).collect());
    assert!(a_out.len() >= store_out.len() && b_out.len() >= store_out.len());

    // The pair cover split into tasks of roughly equal LIVE work, each with
    // its own disjoint output slices — parallel like the dense kernel,
    // instead of one scalar walk (measured ~3x per element at low
    // utilization: no cores, no hoisted eq_hi, one reduced multiply per
    // term). Tasks are cut from the interval list, so their count follows the
    // live support and not the domain: a fragmented slot schedule (367
    // intervals at 6.25% utilization, against 2 at full) would otherwise
    // spawn one near-empty task per interval per round — 6.6K tasks over the
    // tail against 1.5K for the same live volume when unfragmented.
    // Within a task the per-hi-run products accumulate UNREDUCED and reduce
    // once, and eq_hi multiplies the run total — pure reassociation of exact
    // field algebra (reduction commutes with XOR), so the message stays
    // byte-identical to the scalar loop and to the dense kernel.
    //
    // Task size: at most 2^12 covered pairs, but thread-aware below that —
    // the tail's live set halves every round, and a fixed absolute target
    // starves the pool (fewer tasks than threads) rounds before the work
    // itself is small. Regrouping is exact (above), so the target only
    // moves work between tasks, never values.
    const CHUNK: usize = 1 << 12;
    let live_total: usize = pair_cover.iter().map(|&(s, e)| e - s).sum();
    let threads = current_num_threads().max(1);
    let target = (live_total / (4 * threads)).clamp(1 << 8, CHUNK);
    let (pieces, tasks) = balanced_interval_tasks(&pair_cover, target);
    let mut work: Vec<(&[(usize, usize)], &mut [F128], &mut [F128])> =
        Vec::with_capacity(tasks.len());
    {
        let mut a_rem: &mut [F128] = &mut a_out[..store_out.len()];
        let mut b_rem: &mut [F128] = &mut b_out[..store_out.len()];
        for t in &tasks {
            let span: usize = pieces[t.clone()].iter().map(|&(s, e)| 2 * (e - s)).sum();
            let (a_task, rest) = take(&mut a_rem).split_at_mut(span);
            a_rem = rest;
            let (b_task, rest) = take(&mut b_rem).split_at_mut(span);
            b_rem = rest;
            work.push((&pieces[t.clone()], a_task, b_task));
        }
    }

    let (sum1, sum_inf) = work
        .into_par_iter()
        .map(|(task_pieces, a_task, b_task)| {
            let ps = task_pieces[0].0;
            // Zero-substituting source reads THROUGH the compaction map: a
            // task-local cursor walks the stored intervals monotonically as
            // the read position `y` ascends (reads within a task ascend),
            // seeded by binary search. A dead `y` is not stored at all and
            // reads as its honest value, ZERO.
            let mut cur = store_in.seek(4 * ps);
            let read2 = |cur: &mut usize, y: usize| -> (F128, F128) {
                match store_in.rank(cur, y) {
                    Some(i) => (a[i], b[i]),
                    None => (F128::ZERO, F128::ZERO),
                }
            };
            let mut s1 = F128::ZERO;
            let mut s_inf = F128::ZERO;
            // A task may cover several live pieces; the dead gaps between them
            // occupy no output storage, so each piece's slots follow the
            // previous piece's directly.
            let mut local = 0usize;
            for &(piece_s, piece_e) in task_pieces {
                let piece_base = local;
                let mut t = piece_s;
                while t < piece_e {
                    let run_end = (((t >> eq.n_lo) + 1) << eq.n_lo).min(piece_e);
                    let mut p1 = F256Unreduced::ZERO;
                    let mut p_inf = F256Unreduced::ZERO;
                    // Advance the cursor exactly as the first `rank` of this
                    // run would; when the run's whole read span then sits in
                    // ONE stored interval — the common case away from
                    // live/dead boundaries — the reads are contiguous in the
                    // compacted buffer, so index directly instead of paying
                    // four cursor gathers per pair. Same positions read in
                    // the same order (and no dead position is in the span,
                    // so the fallback's zero-substitution never fires here):
                    // byte-identical by construction.
                    let (y0, y1) = (4 * t, 4 * run_end);
                    while cur < store_in.intervals().len() && store_in.intervals()[cur].1 <= y0 {
                        cur += 1;
                    }
                    let contig = store_in
                        .intervals()
                        .get(cur)
                        .is_some_and(|&(s, e)| s <= y0 && y1 <= e);
                    if contig {
                        let base = store_in.offset_of(cur) + (y0 - store_in.intervals()[cur].0);
                        for tt in t..run_end {
                            let i = base + 4 * (tt - t);
                            let (a00, b00) = (a[i], b[i]);
                            let (a01, b01) = (a[i + 1], b[i + 1]);
                            let (a10, b10) = (a[i + 2], b[i + 2]);
                            let (a11, b11) = (a[i + 3], b[i + 3]);
                            let a0 = a00 + r_fold * (a01 + a00);
                            let a1 = a10 + r_fold * (a11 + a10);
                            let b0 = b00 + r_fold * (b01 + b00);
                            let b1 = b10 + r_fold * (b11 + b10);
                            let o = piece_base + 2 * (tt - piece_s);
                            a_task[o] = a0;
                            a_task[o + 1] = a1;
                            b_task[o] = b0;
                            b_task[o + 1] = b1;
                            let eq_l = eq.lo[tt & lo_mask];
                            p1 ^= eq_l.mul_unreduced(a1 * b1);
                            p_inf ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                        }
                    } else {
                        for tt in t..run_end {
                            let y = 4 * tt;
                            let (a00, b00) = read2(&mut cur, y);
                            let (a01, b01) = read2(&mut cur, y + 1);
                            let (a10, b10) = read2(&mut cur, y + 2);
                            let (a11, b11) = read2(&mut cur, y + 3);
                            let a0 = a00 + r_fold * (a01 + a00);
                            let a1 = a10 + r_fold * (a11 + a10);
                            let b0 = b00 + r_fold * (b01 + b00);
                            let b1 = b10 + r_fold * (b11 + b10);
                            let o = piece_base + 2 * (tt - piece_s);
                            a_task[o] = a0;
                            a_task[o + 1] = a1;
                            b_task[o] = b0;
                            b_task[o + 1] = b1;
                            let eq_l = eq.lo[tt & lo_mask];
                            p1 ^= eq_l.mul_unreduced(a1 * b1);
                            p_inf ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                        }
                    }
                    let eq_h = eq.hi[t >> eq.n_lo];
                    s1 += eq_h * p1.reduce();
                    s_inf += eq_h * p_inf.reduce();
                    t = run_end;
                }
                local += 2 * (piece_e - piece_s);
            }
            (s1, s_inf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x1, xi), (y1, yi)| (x1 + y1, xi + yi),
        );
    (r_next[0] * sum1, sum_inf, store_out)
}

/// Scatter a live-span buffer back to the full padded domain, zeroing the
/// dead positions — the bridge back to the dense kernels, which index by
/// global position. Needed only when the tail leaves the sparse path
/// (`SPARSE_TAIL_GATE > 1`, or the domain drops below the fused threshold).
pub fn expand_to_dense(compact: &[F128], store: &LiveLayout, domain: usize) -> Vec<F128> {
    debug_assert_eq!(compact.len(), store.len());
    let mut out = take_f128(domain);
    out.fill(F128::ZERO);
    for (i, &(s, e)) in store.intervals().iter().enumerate() {
        let off = store.offset_of(i);
        out[s..e].copy_from_slice(&compact[off..off + (e - s)]);
    }
    out
}

/// Serial reference — identical I/O contract to
/// [`uni_skip_fold_and_round_pair_optimized_packed`], no rayon. Kept under
/// `#[cfg(test)]` as the cross-check oracle for the parallel version.
#[cfg(test)]
fn uni_skip_fold_and_round_pair_optimized_packed_serial(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let mut a_folded = vec![F128::ZERO; n_out];
    let mut b_folded = vec![F128::ZERO; n_out];
    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let mut sum1 = F128::ZERO;
    let mut sum_inf = F128::ZERO;
    for x_hi in 0..hi_size {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;
        let k_base = x_hi << eq.n_lo;
        for x_lo in 0..lo_size {
            let k = k_base | x_lo;
            let x0 = 2 * k;
            let x1 = x0 + 1;
            let a0 = table.fold_one_row(&a_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let b0 = table.fold_one_row(&b_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let a1 = table.fold_one_row(&a_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            let b1 = table.fold_one_row(&b_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            a_folded[x0] = a0;
            b_folded[x0] = b0;
            a_folded[x1] = a1;
            b_folded[x1] = b1;
            let eq_l = eq.lo[x_lo];
            let g1 = a1 * b1;
            p1_acc ^= eq_l.mul_unreduced(g1);
            let g_inf = (a0 + a1) * (b0 + b1);
            pinf_acc ^= eq_l.mul_unreduced(g_inf);
        }
        let p1 = p1_acc.reduce();
        let pinf = pinf_acc.reduce();
        sum1 += eq.hi[x_hi] * p1;
        sum_inf += eq.hi[x_hi] * pinf;
    }
    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}

// ---------------------------------------------------------------------------
// Sumcheck LOOKAHEAD (Rothblum): one pass accumulates the BIVARIATE
//
//   Q(X,Y) = Σ_u eq(rest,u) · â(X,Y,u) · b̂(X,Y,u)
//
// covering the next TWO rounds. The first round's message is the Y∈{0,1}
// eq-weighted combination of Q's rows; once its challenge ρ arrives, the
// second round's message is Q(ρ,·) — both O(1) from 8 accumulated sums, no
// pass over the data. Each pass then folds TWO already-bound variables (4→1),
// so per two rounds the traffic drops from (read 2n + write n) + (read n +
// write n/2) to (read 2n + write n/2) — −44% in the tail's fully memory-bound
// regime, where the extra multiplies hide (measured: fold-only == full).
//
// TRANSCRIPT-INVARIANT: exact field arithmetic means the derived messages
// equal the classically computed ones bit-for-bit (see the parity test); only
// the prover's evaluation strategy changes.
//
// The 8 sums (X kept in coefficient form — ρ is unknown at accumulation time;
// the Y=0 column only ever feeds the first message at X∈{1,∞}, so it needs
// just those two projections rather than 3 coefficients):
//   s10   = Σ eq·A(1,0)B(1,0)                (col Y=0 @ X=1)
//   sinf0 = Σ eq·ΔₓA(0)·ΔₓB(0)               (col Y=0, X-leading)
//   s1    = X-coeffs of col Y=1  (accumulated as c0, value@X=1, c2)
//   sinf  = X-coeffs of col Y=∞  (the Y-slope product column, same trick)
// ---------------------------------------------------------------------------

/// The reduced lookahead sums for one pass.
#[derive(Clone, Copy, Debug)]
pub struct LookaheadSums {
    pub s10: F128,
    pub sinf0: F128,
    pub s1: [F128; 3],
    pub sinf: [F128; 3],
}

/// First-message derivation (Convention A, the AG tail's `r_next[0] = ONE`):
/// eq-weighted combination of the Y∈{0,1} rows, `r_y` = eq coord of Y.
#[inline]
pub fn lookahead_msg_first(q: &LookaheadSums, r_y: F128) -> (F128, F128) {
    let om = F128::ONE + r_y; // 1−r_y in char 2
    let col1_at1 = q.s1[0] + q.s1[1] + q.s1[2];
    (om * q.s10 + r_y * col1_at1, om * q.sinf0 + r_y * q.s1[2])
}

/// Second-message derivation after the first variable binds to `rho`:
/// evaluate the two column polynomials at `rho`. Zero passes.
#[inline]
pub fn lookahead_msg_second(q: &LookaheadSums, rho: F128) -> (F128, F128) {
    let r2 = rho * rho;
    (
        q.s1[0] + rho * q.s1[1] + r2 * q.s1[2],
        q.sinf[0] + rho * q.sinf[1] + r2 * q.sinf[2],
    )
}

/// The 8 per-position Q products from the 4 folded values per witness
/// (index = x + 2y): [s10, sinf0, c0, t11, c2, d0, dt, d2] — see the module
/// comment for which sum each feeds.
#[inline(always)]
pub(crate) fn lookahead_products(ga: &[F128; 4], gb: &[F128; 4]) -> [F128; 8] {
    let (ga00, ga10, ga01, ga11) = (ga[0], ga[1], ga[2], ga[3]);
    let (gb00, gb10, gb01, gb11) = (gb[0], gb[1], gb[2], gb[3]);
    let sxa0 = ga00 + ga10;
    let sxb0 = gb00 + gb10;
    let sxa1 = ga01 + ga11;
    let sxb1 = gb01 + gb11;
    let dca = ga00 + ga01;
    let dcb = gb00 + gb01;
    let dsa = sxa0 + sxa1;
    let dsb = sxb0 + sxb1;
    [
        ga10 * gb10,               // s10
        sxa0 * sxb0,               // sinf0
        ga01 * gb01,               // col1 c0
        ga11 * gb11,               // col1 @ X=1
        sxa1 * sxb1,               // col1 c2
        dca * dcb,                 // col∞ c0
        (dca + dsa) * (dcb + dsb), // col∞ @ X=1
        dsa * dsb,                 // col∞ c2
    ]
}

/// Per-position Q contribution, eq-weighted into the 8 unreduced accumulators.
#[inline(always)]
fn lookahead_accum(ga: &[F128; 4], gb: &[F128; 4], eq: F128, acc: &mut [F256Unreduced; 8]) {
    let p = lookahead_products(ga, gb);
    for k in 0..8 {
        acc[k] ^= eq.mul_unreduced(p[k]);
    }
}

pub(crate) fn lookahead_finish(s: [F128; 8]) -> LookaheadSums {
    let [s10, sinf0, c0, t11, c2, d0, dt, d2] = s;
    LookaheadSums {
        s10,
        sinf0,
        s1: [c0, t11 + c0 + c2, c2],
        sinf: [d0, dt + d0 + d2, d2],
    }
}

macro_rules! lookahead_pass {
    ($name:ident, $per_u:expr, $fold:expr, $doc:literal) => {
        #[doc = $doc]
        /// Writes the folded arrays into `a_out`/`b_out` and returns the 8
        /// lookahead sums over the next two variables. `r_y` is the eq coord of
        /// the SECOND lookahead variable; `rest` the coords after it
        /// (`rest.len() == log2(out_len/4)`). Parallel over the eq-hi chunks.
        pub fn $name(
            a: &[F128],
            b: &[F128],
            a_out: &mut [F128],
            b_out: &mut [F128],
            rhos: (F128, F128),
            rest: &[F128],
        ) -> LookaheadSums {
            const PER_U: usize = $per_u;
            let n_u = a.len() / PER_U;
            assert_eq!(a.len(), b.len());
            assert_eq!(a.len() % PER_U, 0);
            assert_eq!(a_out.len(), 4 * n_u);
            assert_eq!(b_out.len(), 4 * n_u);
            assert_eq!(
                1usize << rest.len(),
                n_u,
                "rest coords must cover log2(len/{PER_U})"
            );
            let n_lo = rest.len() / 2;
            let eq_lo = build_eq(&rest[..n_lo]);
            let eq_hi = build_eq(&rest[n_lo..]);
            let lo_size = eq_lo.len();
            let fold = $fold;
            let sums = a_out
                .par_chunks_mut(4 * lo_size)
                .zip(b_out.par_chunks_mut(4 * lo_size))
                .enumerate()
                .map(|(u_hi, (ao, bo))| {
                    let mut acc = [F256Unreduced::ZERO; 8];
                    let base_u = u_hi * lo_size;
                    for u_lo in 0..lo_size {
                        let u = base_u + u_lo;
                        let mut ga = [F128::ZERO; 4];
                        let mut gb = [F128::ZERO; 4];
                        for v in 0..4usize {
                            let base = u * PER_U + v * (PER_U / 4);
                            ga[v] = fold(&a[base..], rhos);
                            gb[v] = fold(&b[base..], rhos);
                            ao[4 * u_lo + v] = ga[v];
                            bo[4 * u_lo + v] = gb[v];
                        }
                        lookahead_accum(&ga, &gb, eq_lo[u_lo], &mut acc);
                    }
                    let eh = eq_hi[u_hi];
                    let mut out = [F128::ZERO; 8];
                    for k in 0..8 {
                        out[k] = eh * acc[k].reduce();
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
    };
}

lookahead_pass!(
    fold1_lookahead_into,
    8,
    |e: &[F128], r: (F128, F128)| e[0] + r.0 * (e[0] + e[1]),
    "Entry lookahead pass: fold ONE pending variable (`rhos.0`; `rhos.1` unused), n → n/2."
);
lookahead_pass!(
    fold2_lookahead_into,
    16,
    |e: &[F128], r: (F128, F128)| {
        let x0 = e[0] + r.0 * (e[0] + e[1]);
        let x1 = e[2] + r.0 * (e[2] + e[3]);
        x0 + r.1 * (x0 + x1)
    },
    "Steady-state lookahead pass: fold TWO pending variables (4→1), n → n/4."
);

/// `&[bool]` convenience wrapper around
/// [`uni_skip_fold_and_round_pair_optimized_packed`]. Packs internally, builds
/// the fold table from `z`.
pub fn uni_skip_fold_and_round_pair_optimized(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    let a_packed = pack_bits(a);
    let b_packed = pack_bits(b);
    let table = UniSkipFoldTable::new(k_skip, z);
    uni_skip_fold_and_round_pair_optimized_packed(
        &a_packed,
        &b_packed,
        m,
        k_skip,
        &table,
        mlv_challenges,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::F8;
    use crate::field::PHI_8_TABLE;
    use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
    use crate::test_rng::Rng;
    use crate::zerocheck::univariate_skip_optimized::{
        c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed,
        small_challenges_ghash,
    };
    /// The closed-form Lagrange weights agree with the textbook `O(ell²)`
    /// product, at every `k_skip` the protocol can use, on random points AND
    /// on the degenerate points the closed form has to special-case (`z`
    /// exactly on a node, where it would otherwise divide by zero).
    ///
    /// The textbook form is reproduced here rather than kept in the module: it
    /// is the reference this is differential-tested against, and having it live
    /// only in the test makes it impossible to call by accident.
    #[test]
    fn closed_form_lagrange_matches_the_textbook_product() {
        fn textbook(nodes: &[F128], z: F128) -> Vec<F128> {
            (0..nodes.len())
                .map(|i| {
                    let mut num = F128::ONE;
                    let mut den = F128::ONE;
                    for j in 0..nodes.len() {
                        if j != i {
                            num *= z + nodes[j];
                            den *= nodes[i] + nodes[j];
                        }
                    }
                    num * den.inv()
                })
                .collect()
        }

        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = state;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(hi, state)
        };

        for k_skip in 1..=6usize {
            let ell = 1usize << k_skip;
            // S-domain, Λ-domain (a coset), and the combined domain.
            let domains: [(&[F128], usize); 3] = [
                (&PHI_8_TABLE[..ell], k_skip),
                (&PHI_8_TABLE[ell..2 * ell], k_skip),
                (&PHI_8_TABLE[..2 * ell], k_skip + 1),
            ];
            for (nodes, dim) in domains {
                for _ in 0..8 {
                    let z = next();
                    assert_eq!(
                        lagrange_weights_on_coset(nodes, dim, z),
                        textbook(nodes, z),
                        "closed form disagrees at k_skip={k_skip}, dim={dim}"
                    );
                }
                // Every node, the branch the closed form special-cases.
                for &node in nodes {
                    assert_eq!(
                        lagrange_weights_on_coset(nodes, dim, node),
                        textbook(nodes, node),
                        "closed form disagrees ON a node at dim={dim}"
                    );
                }
            }

            // And the public entry points, end to end.
            let values: Vec<F128> = (0..ell).map(|_| next()).collect();
            for _ in 0..4 {
                let z = next();
                let want: F128 = textbook(&PHI_8_TABLE[ell..2 * ell], z)
                    .iter()
                    .zip(&values)
                    .fold(F128::ZERO, |a, (w, v)| a + *w * *v);
                assert_eq!(interpolate_at_z_on_lambda(&values, k_skip, z), want);

                let want_combined: F128 = textbook(&PHI_8_TABLE[..2 * ell], z)[ell..]
                    .iter()
                    .zip(&values)
                    .fold(F128::ZERO, |a, (w, v)| a + *w * *v);
                assert_eq!(interpolate_at_z_combined(&values, k_skip, z), want_combined);
            }
        }
    }

    /// Every Lagrange denominator on a subspace or coset is the SAME constant
    /// — the fact the closed form is built on. Checked directly so a change to
    /// `PHI_8_TABLE`'s ordering (which would break the subspace structure
    /// without breaking anything else) fails here.
    #[test]
    fn subspace_denominators_are_uniform_across_nodes_and_cosets() {
        for dim in 1..=6usize {
            let n = 1usize << dim;
            let expected = subspace_denominator_pair(dim).0;
            for (label, nodes) in [
                ("subspace", &PHI_8_TABLE[..n]),
                ("coset", &PHI_8_TABLE[n..2 * n]),
            ] {
                for i in 0..n {
                    let den = (0..n)
                        .filter(|&j| j != i)
                        .fold(F128::ONE, |acc, j| acc * (nodes[i] + nodes[j]));
                    assert_eq!(
                        den, expected,
                        "{label} denominator varies at dim={dim}, i={i}"
                    );
                }
            }
        }
    }

    /// The interpolation node sets are **F₂-subspaces**, so their vanishing
    /// polynomials are *linearized* (additive).
    ///
    /// `φ_8` is a field homomorphism, hence additive, and `{0..2^m−1}` is an
    /// F₂-subspace of GF(2^8) under XOR — so `V = φ_8({0..2^m−1})` is an
    /// F₂-subspace of F_{2^128} and `Z_V(X) = Π_{s∈V}(X+s)` satisfies
    /// `Z_V(a+b) = Z_V(a) + Z_V(b)`.
    ///
    /// This is a *cost* property, and a large one. The Lagrange weights
    /// [`lagrange_weights_naive`] computes as `Π_{j≠i}(z+φ(j))` are really
    /// `Z_V(z) / (z + φ(i))` — one additive map plus an inverse, instead of
    /// the naive `O(ell²)` product. At `ell = 64` that is ~4k muls collapsing
    /// to ~64, and [`interpolate_at_z_combined`]'s ~16k muls collapsing to
    /// ~128. Natively the naive form is sub-millisecond and not worth
    /// changing; **in-circuit the difference is decisive**, and an additive
    /// map over F_{2^128} is F₂-linear, i.e. free in the boolean class.
    ///
    /// Pinned here because the recursion circuit's cost model depends on it,
    /// and because it is a property of `PHI_8_TABLE`'s contents rather than of
    /// any code that could be reviewed for it.
    #[test]
    fn phi8_node_sets_have_linearized_vanishing_polynomials() {
        // Z_V(x) = Π_{s ∈ V} (x + s) over the first 2^m table entries.
        let z_v = |m: usize, x: F128| -> F128 {
            (0..(1usize << m)).fold(F128::ONE, |acc, i| acc * (x + PHI_8_TABLE[i]))
        };

        for m in 1..=6usize {
            // Additivity on a spread of points, including the subspace itself.
            let probes = [
                F128::new(0, 0),
                F128::new(1, 0),
                F128::new(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
                F128::new(0xDEAD_BEEF_CAFE_F00D, 0x0BAD_C0DE_1234_5678),
                PHI_8_TABLE[1],
                PHI_8_TABLE[(1usize << m) - 1],
            ];
            for a in probes {
                for b in probes {
                    assert_eq!(
                        z_v(m, a + b),
                        z_v(m, a) + z_v(m, b),
                        "Z_V not additive at m={m}"
                    );
                }
            }
            // And it vanishes exactly on the subspace.
            for i in 0..(1usize << m) {
                assert_eq!(z_v(m, PHI_8_TABLE[i]), F128::ZERO, "Z_V(s) != 0 at m={m}");
            }
        }

        // The payoff, stated as an identity: the naive Lagrange weight equals
        // the closed form `Z_V(z) / ((z + φ(i)) · Z_V'(φ(i)))`, where the
        // derivative of a linearized polynomial is the constant `c_0` — here
        // recovered as `Z_V(x)/x` in the limit, i.e. the coefficient obtained
        // from any single evaluation off the subspace divided out. We check
        // the ratio form directly, which is what a circuit would evaluate.
        let m = 3usize;
        let z = F128::new(0x9E37_79B9_7F4A_7C15, 0xBF58_476D_1CE4_E5B9);
        let naive = lagrange_weights_naive(m, z);
        let zv_z = z_v(m, z);
        for (i, &w) in naive.iter().enumerate() {
            // den_i = Π_{j≠i}(φ(i)+φ(j)), the constant part.
            let mut den = F128::ONE;
            for j in 0..(1usize << m) {
                if j != i {
                    den *= PHI_8_TABLE[i] + PHI_8_TABLE[j];
                }
            }
            let closed = zv_z * ((z + PHI_8_TABLE[i]) * den).inv();
            assert_eq!(w, closed, "closed-form Lagrange weight disagrees at i={i}");
        }
    }

    /// `LiveLayout` is an order-preserving bijection between the live global
    /// positions and `0..len`, and reports every dead position as absent.
    /// Checked against a brute-force rank over the whole domain, including
    /// the fragmented and boundary-touching shapes the multi-run specs
    /// actually produce.
    #[test]
    fn live_layout_ranks_match_brute_force() {
        let cases: Vec<(Vec<(usize, usize)>, usize)> = vec![
            (vec![], 16),
            (vec![(0, 32)], 32),
            (vec![(0, 10), (16, 26), (32, 42)], 64),
            (vec![(3, 4)], 8),
            (vec![(0, 1), (2, 3), (4, 5), (6, 7)], 8),
            (vec![(5, 9), (11, 12), (20, 40)], 48),
        ];
        for (iv, domain) in cases {
            let layout = LiveLayout::new(iv.clone());
            // Brute-force: the compacted index is the count of live positions
            // strictly before y, and None when y itself is dead.
            let live_at = |y: usize| iv.iter().any(|&(s, e)| s <= y && y < e);
            let mut expect_len = 0usize;
            let mut cur = layout.seek(0);
            for y in 0..domain {
                let got = layout.rank(&mut cur, y);
                if live_at(y) {
                    assert_eq!(got, Some(expect_len), "rank at {y} for {iv:?}");
                    expect_len += 1;
                } else {
                    assert_eq!(got, None, "dead {y} must not be stored, {iv:?}");
                }
            }
            assert_eq!(layout.len(), expect_len, "len for {iv:?}");
            assert_eq!(layout.is_empty(), expect_len == 0);
            // A cursor seeded mid-domain agrees with one walked from zero.
            for probe in [0, domain / 3, domain / 2, domain.saturating_sub(1)] {
                let mut c = layout.seek(probe);
                let mut c0 = layout.seek(0);
                let mut walk = None;
                for y in 0..=probe {
                    walk = layout.rank(&mut c0, y);
                }
                assert_eq!(layout.rank(&mut c, probe), walk, "seek({probe}) {iv:?}");
            }
        }
    }

    /// `is_dense` recognises exactly the layouts where compaction is the
    /// identity — the gate for keeping the dense kernels on the anchor shape.
    #[test]
    fn live_layout_is_dense_only_for_full_cover() {
        assert!(LiveLayout::new(vec![(0, 32)]).is_dense(32));
        assert!(!LiveLayout::new(vec![(0, 16)]).is_dense(32));
        assert!(!LiveLayout::new(vec![(1, 32)]).is_dense(32));
        assert!(!LiveLayout::new(vec![(0, 8), (16, 32)]).is_dense(32));
        assert!(!LiveLayout::new(vec![]).is_dense(32));
    }

    /// The task cut is a faithful partition of the live set: pieces
    /// concatenate back to the input intervals (no element added, dropped or
    /// visited twice — a duplicate would double-count that pair into the
    /// message), tasks tile the piece list in order, and no task exceeds the
    /// target while only the last may fall short of it.
    #[test]
    fn balanced_interval_tasks_partitions_the_live_set() {
        let target = 8usize;
        let cases: Vec<Vec<(usize, usize)>> = vec![
            vec![],
            vec![(0, 1)],
            vec![(0, 100)],                                // one long interval: split
            vec![(3, 4), (9, 10), (11, 12), (40, 41)],     // fragments: coalesced
            vec![(0, 8), (8, 16)],                         // exactly on target
            (0..40).map(|i| (5 * i, 5 * i + 2)).collect(), // many tiny fragments
            vec![(1, 2), (4, 30), (31, 33), (60, 100)],    // mixed
        ];
        for live in cases {
            let (pieces, tasks) = balanced_interval_tasks(&live, target);

            let flat: Vec<usize> = pieces.iter().flat_map(|&(s, e)| s..e).collect();
            let expect: Vec<usize> = live.iter().flat_map(|&(s, e)| s..e).collect();
            assert_eq!(
                flat, expect,
                "pieces must re-flatten to the live set: {live:?}"
            );

            // Tasks tile the pieces contiguously, in order, without overlap.
            let mut next = 0usize;
            for t in &tasks {
                assert_eq!(t.start, next, "task ranges must be contiguous: {live:?}");
                assert!(t.end > t.start, "empty task: {live:?}");
                next = t.end;
            }
            assert_eq!(next, pieces.len(), "tasks must cover every piece: {live:?}");

            for (i, t) in tasks.iter().enumerate() {
                let work: usize = pieces[t.clone()].iter().map(|&(s, e)| e - s).sum();
                assert!(
                    work >= target || i == tasks.len() - 1,
                    "only the last task may be under target: {live:?}"
                );
                // Grouping stops the moment the target is met, and every
                // piece is at most `target` long, so a task is bounded by 2x.
                assert!(work < 2 * target, "task work {work} unbounded: {live:?}");
            }
        }
    }

    // ----------------------------------------------------------------------
    // Lagrange weights — algebraic properties.
    // ----------------------------------------------------------------------

    /// `Σ_i L_i(z) = 1` for all z. The polynomial `1` interpolates to constant
    /// `1` at every node, so its evaluation at z is `Σ_i 1·L_i(z) = Σ_i L_i(z)`.
    #[test]
    fn lagrange_weights_sum_to_one() {
        let mut rng = Rng::new(1);
        for &k_skip in &[1usize, 2, 3, 4, 5, 6] {
            for _ in 0..4 {
                let z = rng.f128();
                let weights = lagrange_weights_naive(k_skip, z);
                let sum: F128 = weights.iter().copied().fold(F128::ZERO, |a, b| a + b);
                assert_eq!(sum, F128::ONE, "Σ L_i ≠ 1 at k_skip={k_skip}");
            }
        }
    }

    /// `L_i(s_j) = δ_{ij}` — Kronecker delta. At a node, exactly one weight is 1.
    #[test]
    fn lagrange_at_node_is_indicator() {
        for k_skip in [2usize, 3, 4, 5] {
            let ell = 1usize << k_skip;
            for i in 0..ell {
                let z = PHI_8_TABLE[i];
                let weights = lagrange_weights_naive(k_skip, z);
                for j in 0..ell {
                    let expected = if j == i { F128::ONE } else { F128::ZERO };
                    assert_eq!(weights[j], expected, "k_skip={k_skip}, z=node{i}, j={j}");
                }
            }
        }
    }

    // ----------------------------------------------------------------------
    // Fold — algebraic properties.
    // ----------------------------------------------------------------------

    /// At a node `z = φ_8(i)`, fold reduces to the witness restricted to s=i:
    /// `a_mlv[x_rest] = a[x_rest · 2^k_skip + i]` (lifted to F_128).
    #[test]
    fn fold_at_node_recovers_witness_slice() {
        let m = 8;
        let k_skip = 3;
        let ell = 1usize << k_skip;
        let n_rest = 1usize << (m - k_skip);
        let mut rng = Rng::new(7);
        let a = rng.bits(1 << m);
        for i in 0..ell {
            let z = PHI_8_TABLE[i];
            let weights = lagrange_weights_naive(k_skip, z);
            let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
            for x_rest in 0..n_rest {
                let expected = if a[x_rest * ell + i] {
                    F128::ONE
                } else {
                    F128::ZERO
                };
                assert_eq!(
                    a_mlv[x_rest], expected,
                    "fold at node {i} mismatch at x_rest={x_rest}"
                );
            }
        }
    }

    /// Fold is linear in the input witness: fold(a ⊕ a') = fold(a) + fold(a').
    /// (XOR-linearity is the defining property of the multilinear extension.)
    #[test]
    fn fold_is_xor_linear() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(11);
        let a = rng.bits(1 << m);
        let aprime = rng.bits(1 << m);
        let a_xor: Vec<bool> = a.iter().zip(&aprime).map(|(x, y)| x ^ y).collect();
        let z = rng.f128();
        let weights = lagrange_weights_naive(k_skip, z);

        let fa = fold_at_z_naive(&a, m, k_skip, &weights);
        let fap = fold_at_z_naive(&aprime, m, k_skip, &weights);
        let fxor = fold_at_z_naive(&a_xor, m, k_skip, &weights);
        for i in 0..fa.len() {
            assert_eq!(fa[i] + fap[i], fxor[i], "linearity broken at i={i}");
        }
    }

    // ----------------------------------------------------------------------
    // Round-2 message — properties + cross-checks.
    // ----------------------------------------------------------------------

    /// All-zero witness ⇒ a_mlv = b_mlv = 0 ⇒ G(1) = G(∞) = 0, so the message
    /// elements (r[0]·G(1), G(∞)) are also both zero.
    #[test]
    fn zero_witness_gives_zero_round_message() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(20);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let zeros = vec![false; 1 << m];
        let (a_mlv, b_mlv, msg_1, msg_inf) =
            uni_skip_fold_and_round_pair_naive(&zeros, &zeros, m, k_skip, z, &mlv_challenges);
        assert!(a_mlv.iter().all(|v| v.is_zero()));
        assert!(b_mlv.iter().all(|v| v.is_zero()));
        assert_eq!(msg_1, F128::ZERO);
        assert_eq!(msg_inf, F128::ZERO);
    }

    #[test]
    fn deterministic() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(33);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let o1 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let o2 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        assert_eq!(o1, o2);
    }

    /// Round-pair message is symmetric in a, b: swapping a↔b gives the same
    /// message. `a · b = b · a` is built-in, and the `r[0]` multiplier doesn't
    /// distinguish AB.
    #[test]
    fn round_pair_symmetric_in_ab() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(40);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let (_, _, m1_ab, minf_ab) =
            uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let (_, _, m1_ba, minf_ba) =
            uni_skip_fold_and_round_pair_naive(&b, &a, m, k_skip, z, &mlv_challenges);
        assert_eq!(m1_ab, m1_ba);
        assert_eq!(minf_ab, minf_ba);
    }

    // ----------------------------------------------------------------------
    // Optimized fused — UniSkipFoldTable + fold_one_row, then naive cross-check.
    // ----------------------------------------------------------------------

    /// NEON `fold_one_row_neon_unchecked_8` matches scalar `fold_one_row`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fold_one_row_neon_matches_scalar() {
        let k_skip = 6;
        let mut rng = Rng::new(70);
        let z = rng.f128();
        let table = UniSkipFoldTable::new(k_skip, z);

        for _ in 0..256 {
            let mut bytes = [0u8; 8];
            for byte in bytes.iter_mut() {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let scalar = table.fold_one_row(&bytes);
            // SAFETY: on aarch64; bytes has 8 entries; table has 8 chunks.
            let neon = unsafe {
                fold_one_row_neon_unchecked_8(table.data.as_ptr() as *const u8, bytes.as_ptr())
            };
            assert_eq!(scalar, neon, "fold mismatch bytes={bytes:02x?}");
        }
    }

    /// Four-row x86 lookup fold matches four independent scalar folds.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold_round2_pair_x86_matches_scalar() {
        let mut rng = Rng::new(71);
        let table = UniSkipFoldTable::new(6, rng.f128());

        for _ in 0..256 {
            let mut rows = [[0u8; 8]; 4];
            for row in &mut rows {
                for byte in row {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
            }
            let expected = rows.map(|row| table.fold_one_row(&row));
            // SAFETY: each row has 8 bytes and the table has 8 × 256 entries.
            let actual = unsafe {
                fold_round2_pair_x86_unchecked_8(
                    table.data.as_ptr(),
                    rows[0].as_ptr(),
                    rows[1].as_ptr(),
                    rows[2].as_ptr(),
                    rows[3].as_ptr(),
                )
            };
            assert_eq!(actual, expected);
        }
    }

    /// `fold_in_place_pair` correctness: post-fold a[x] = a[2x] + X·(a[2x+1]+a[2x]).
    #[test]
    fn fold_in_place_pair_matches_formula() {
        let mut rng = Rng::new(300);
        for &log_n in &[1usize, 2, 3, 4, 6] {
            let n = 1usize << log_n;
            let a_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let challenge = rng.f128();

            let mut a = a_orig.clone();
            let mut b = b_orig.clone();
            fold_in_place_pair(&mut a, &mut b, challenge);

            assert_eq!(a.len(), n / 2);
            assert_eq!(b.len(), n / 2);
            for x in 0..(n / 2) {
                let a0 = a_orig[2 * x];
                let a1 = a_orig[2 * x + 1];
                let b0 = b_orig[2 * x];
                let b1 = b_orig[2 * x + 1];
                assert_eq!(a[x], a0 + challenge * (a1 + a0), "log_n={log_n}, x={x}");
                assert_eq!(b[x], b0 + challenge * (b1 + b0), "log_n={log_n}, x={x}");
            }
        }
    }

    /// **The c-claim identity**: `C_s · interpolate(round1_c, k_skip, z)` equals
    /// `ĉ(z, r_rest)` computed by direct folding (Lagrange at z, then bind each
    /// `r_rest` value). This is the math identity that lets the extract_c
    /// prover skip per-round c tracking entirely.
    #[test]
    fn c_eval_from_round1_c_matches_direct_fold() {
        const K_SKIP: usize = 6;
        const N_INNER: usize = 7;

        for &m in &[14usize, 15, 16] {
            let mut rng = Rng::new(500 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);

            // Build r with protocol-fixed constants in the middle 7 dims,
            // matching how `prove` constructs it.
            let mut r = vec![F128::ZERO; m];
            for slot in r[..K_SKIP].iter_mut() {
                *slot = rng.f128();
            }
            for (i, v) in small_challenges_ghash().iter().enumerate() {
                r[K_SKIP + i] = *v;
            }
            for (i, v) in medium_challenges_ghash().iter().enumerate() {
                r[K_SKIP + 3 + i] = *v;
            }
            for slot in r[K_SKIP + N_INNER..].iter_mut() {
                *slot = rng.f128();
            }
            let z = rng.f128();

            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let c_packed = pack_bits(&c);

            let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
            let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let (_round1_ab, round1_c) = round1_shift_reduce_extract_c_packed(
                &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table,
            );

            // Path A: interpolate round1_c at z, scale by C_s.
            let c_eval_via_interpolation =
                c_s_f128() * interpolate_at_z_on_lambda(&round1_c, K_SKIP, z);

            // Path B: direct fold of c at z (Lagrange) then bind each
            // r_rest = r[K_SKIP..m] element with fold_in_place_single.
            let weights = lagrange_weights_naive(K_SKIP, z);
            let mut c_mlv = fold_at_z_naive(&c, m, K_SKIP, &weights);
            for &r_val in &r[K_SKIP..] {
                fold_in_place_single(&mut c_mlv, r_val);
            }
            assert_eq!(c_mlv.len(), 1);
            let c_eval_via_fold = c_mlv[0];

            assert_eq!(
                c_eval_via_interpolation, c_eval_via_fold,
                "c-claim identity broken at m={m}"
            );
        }
    }

    /// **The big cross-check**: fused `fold_and_compute_round_pair_optimized`
    /// produces the same output as the unfused sequence
    /// `fold_in_place_pair` → `round_pair_naive`.
    #[test]
    fn fused_round_matches_unfused() {
        let mut rng = Rng::new(310);
        // fold_and_compute requires lo_size ≥ 2 in SplitEqGhash. eq is over
        // r_next[1..] (size log_n − 2); with MAX_N_HI = 7, n_lo ≥ 1 needs
        // eq size ≥ 8 ⇒ log_n ≥ 10. Smaller cases use the unfused path.
        for &log_n in &[10usize, 11, 12] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let r_fold = rng.f128();
            let r_next = rng.f128_vec(log_n - 1);

            // Fused path.
            let (a_fused, b_fused, m1_fused, minf_fused) =
                fold_and_compute_round_pair_optimized(&a, &b, r_fold, &r_next);

            // Unfused path: clone, in-place fold, naive message.
            let mut a_unf = a.clone();
            let mut b_unf = b.clone();
            fold_in_place_pair(&mut a_unf, &mut b_unf, r_fold);
            let (m1_unf, minf_unf) = round_pair_naive(&a_unf, &b_unf, &r_next);

            assert_eq!(a_fused, a_unf, "a mismatch at log_n={log_n}");
            assert_eq!(b_fused, b_unf, "b mismatch at log_n={log_n}");
            assert_eq!(m1_fused, m1_unf, "msg_1 mismatch at log_n={log_n}");
            assert_eq!(minf_fused, minf_unf, "msg_inf mismatch at log_n={log_n}");
        }
    }

    /// Parallel `uni_skip_fold_and_round_pair_optimized_packed` produces
    /// byte-identical output to the serial version. F128 XOR + multiply sum
    /// is commutative + associative, so worker scheduling order doesn't
    /// affect the result.
    #[test]
    fn parallel_matches_serial() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(200 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let table = UniSkipFoldTable::new(k_skip, z);

            let par = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );
            let ser = uni_skip_fold_and_round_pair_optimized_packed_serial(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );

            assert_eq!(par.0, ser.0, "a_mlv mismatch at m={m}");
            assert_eq!(par.1, ser.1, "b_mlv mismatch at m={m}");
            assert_eq!(par.2, ser.2, "msg_1 mismatch at m={m}");
            assert_eq!(par.3, ser.3, "msg_inf mismatch at m={m}");
        }
    }

    /// **Padding skip is byte-identical to the dense round-2 kernel.** Builds
    /// witnesses with bits `[useful_bits, 2^k_log)` of every block honestly
    /// zero, then asserts the `_padded` kernel produces the same
    /// `(a_mlv, b_mlv, msg_1, msg_inf)` as the dense path.
    ///
    /// Covers all three hash padding shapes: BLAKE3 (k_log=14, useful=15409),
    /// SHA-2 (k_log=15, useful=31401), Keccak (k_log=16, useful=42560).
    #[test]
    fn uni_skip_fold_round_pair_padded_matches_dense() {
        const K_SKIP: usize = 6;
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xFADE_F00D_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << (m - k_log);

            // Random witness, then zero bits [useful_bits, block_size) of each
            // block in both a and b (matches honestly-padded hash R1CS).
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    a[blk * block_size + j] = false;
                    b[blk * block_size + j] = false;
                }
            }
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);

            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - K_SKIP);
            let table = UniSkipFoldTable::new(K_SKIP, z);
            let padding = PaddingSpec::uniform(k_log, useful_bits, n_blocks);

            let dense = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
            );
            let padded = uni_skip_fold_and_round_pair_optimized_packed_padded(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
                &padding,
            );
            assert_eq!(
                dense.0, padded.0,
                "a_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.1, padded.1,
                "b_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.2, padded.2,
                "msg_1: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.3, padded.3,
                "msg_inf: m={m}, k_log={k_log}, useful={useful_bits}"
            );
        }
    }

    /// `fold_one_row` via the table equals direct-Lagrange fold.
    #[test]
    fn fold_table_one_row_matches_direct_lagrange() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(60);
        let z = rng.f128();
        let a = rng.bits(1 << m);
        let weights = lagrange_weights_naive(k_skip, z);
        let table = UniSkipFoldTable::new(k_skip, z);
        let a_packed = pack_bits(&a);

        let n_chunks = 1usize << (k_skip / 8);
        let _ = n_chunks; // ell/8 = (1<<k_skip)/8
        let n_chunks = table.n_chunks;

        for x_rest in 0..(1usize << (m - k_skip)) {
            let direct = {
                let mut acc = F128::ZERO;
                for s in 0..(1usize << k_skip) {
                    if a[x_rest * (1usize << k_skip) + s] {
                        acc += weights[s];
                    }
                }
                acc
            };
            let via_table =
                table.fold_one_row(&a_packed[x_rest * n_chunks..(x_rest + 1) * n_chunks]);
            assert_eq!(via_table, direct, "x_rest={x_rest}");
        }
    }

    /// **The full cross-check**: optimized fused output matches naive
    /// byte-for-byte at the headline `k_skip = 6` (and other small m). Same eq
    /// weights, same z, same r — so a_mlv, b_mlv, and the two message values
    /// must all agree exactly.
    #[test]
    fn optimized_matches_naive() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(100 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);

            let (a_n, b_n, m1_n, minf_n) =
                uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
            let (a_o, b_o, m1_o, minf_o) =
                uni_skip_fold_and_round_pair_optimized(&a, &b, m, k_skip, z, &mlv_challenges);

            assert_eq!(a_n, a_o, "a_mlv mismatch at m={m}");
            assert_eq!(b_n, b_o, "b_mlv mismatch at m={m}");
            assert_eq!(m1_n, m1_o, "msg_1 mismatch at m={m}");
            assert_eq!(minf_n, minf_o, "msg_inf mismatch at m={m}");
        }
    }

    /// Strong cross-check: compute G(0), G(1), G(∞) by direct sum (using the
    /// LSB-first index convention `a_mlv(0, x') = a[2x']`, `a_mlv(1, x') = a[2x'+1]`),
    /// then verify that G interpolated through those three values agrees with
    /// the direct multilinear evaluation at a fresh random X — confirming G
    /// genuinely has degree ≤ 2.
    ///
    /// Also verifies `round_pair_naive` returns `(r[0] · G(1), G(∞))`.
    #[test]
    fn round_pair_message_has_degree_two() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(55);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let r = rng.f128_vec(m - k_skip);

        let weights = lagrange_weights_naive(k_skip, z);
        let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
        let b_mlv = fold_at_z_naive(&b, m, k_skip, &weights);

        let n = a_mlv.len();
        let half = n / 2;
        let eq_remaining = build_eq(&r[1..]);

        // G(0), G(1), G(∞) by direct definition.
        let mut g0 = F128::ZERO;
        let mut g1 = F128::ZERO;
        let mut g_inf = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let eq_x = eq_remaining[x_prime];
            g0 += eq_x * a0 * b0;
            g1 += eq_x * a1 * b1;
            g_inf += eq_x * (a0 + a1) * (b0 + b1);
        }

        // round_pair_naive returns (r[0] · g1, g_inf).
        let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, &r);
        assert_eq!(msg_1, r[0] * g1);
        assert_eq!(msg_inf, g_inf);

        // Degree-2 check: G(X) reconstructed through (G(0), G(1), G(∞)) must
        // agree with the direct multilinear evaluation at a fresh point X.
        // Char-2 interpolation: G(X) = G(0) + X·(G(0)+G(1)) + X·(X+1)·G(∞).
        let x = rng.f128();
        let g_via_poly = g0 + x * (g0 + g1) + x * (x + F128::ONE) * g_inf;
        let mut g_via_sum = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let a_x = a0 + x * (a0 + a1);
            let b_x = b0 + x * (b0 + b1);
            g_via_sum += eq_remaining[x_prime] * a_x * b_x;
        }
        assert_eq!(g_via_poly, g_via_sum);
    }
}
