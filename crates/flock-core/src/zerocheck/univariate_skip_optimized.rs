//! Round-1 prover message — fully optimized (shift_reduce + extract_c, scalar).
//!
//! Scalar Rust implementation (no NEON). Three layered optimizations on top of
//! the [`super::round1_extract_c`] scaffold:
//!
//! 1. **Geometric small-eq + shift_reduce inner** (3 inner-most rest-dims).
//!    Protocol fixes the three small challenges to
//!    `r[k_skip..k_skip+3] = φ_8([0xF7, 0x53, 0xB5])`, which makes
//!    `eq_small[K] = C_s · α^K` (geometric in α, the AES root in GHASH).
//!    The shift_reduce trick computes
//!    `Σ_K eq_small[K] · φ_8(y_K)  =  C_s · φ_8(reduce(Σ_K y_K << K))`,
//!    replacing 8 F128 mults per lane with 8 u16 XOR-shifts + one F_8
//!    reduction.
//!
//! 2. **Geometric medium-eq + convert table** (4 next rest-dims).
//!    Protocol fixes the four medium challenges to
//!    `β_i = γ^{2^{i-1}} / (1 + γ^{2^{i-1}})`, which makes
//!    `eq_med[b] = γ^b / D` for `D = ∏(1+γ^{2^{i-1}})`.
//!    Precomputed table `convert[b][v] = γ^b · φ_8(v)` (64 KB) reduces the
//!    per-lane medium-eq sum from 16 F128 mults to 16 lookups + 16 XORs.
//!
//! 3. **D⁻¹ absorbed into eq_lo.**
//!    Pre-scale `eq_lo[i] ← eq_lo[i] · D⁻¹` once before the loop; this cancels
//!    the `1/D` from the medium-eq factorization, leaving only the `C_s`
//!    factor in the relative output scaling.
//!
//! Net output relationship vs the naive / structural versions:
//!   `C_s · (res_AB[i] + res_C_lifted[i])  ==  naive_p_ab[i] + naive_p_c[i]`
//! with `C_s = φ_8(0x1C)`.
//!
//! This variant is hardcoded for `k_skip = 6` (ell=64, n_chunks=8, N_INNER=7).

use std::sync::OnceLock;

use crate::field::{F8, F128, PHI_8_TABLE, mul_by_x, phi8};
use crate::ntt::InvNttTableByteSingleGf8;

use super::PaddingSpec;
use super::univariate_skip::{SplitEqGhash, ntt_extend_f128_vec_ghash, pack_bits};

mod kernels;

#[cfg(all(test, target_arch = "aarch64"))]
use kernels::aarch64::{
    bit_transpose_64bytes_neon, shift_reduce_inner_ab_fused_neon, shift_reduce_inner_ab_neon,
};
#[cfg(all(test, target_arch = "aarch64"))]
use kernels::bit_transpose_64bytes_scalar;
#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )
))]
use kernels::shift_reduce_inner_ab_scalar;
#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
use kernels::x86_64::shift_reduce_inner_ab_x86_avx512;
#[cfg(all(test, target_arch = "x86_64", target_feature = "gfni"))]
use kernels::x86_64::shift_reduce_inner_ab_x86_sse;

// ---------------------------------------------------------------------------
// Protocol constants — fixed by the optimization design.
// ---------------------------------------------------------------------------

/// Number of variables folded in round 1 for the shift_reduce variant.
pub const K_SKIP: usize = 6;
const ELL: usize = 64;
const N_CHUNKS: usize = 8;
/// Total inner-most dims absorbed by the optimization: 3 small + 4 medium.
const N_INNER: usize = 7;
const N_MEDIUM: usize = 4;

/// The three small-eq challenges (as F_8 values, then embedded via φ_8).
/// Choosing these specific values is what makes `eq_small[K] = C_s · α^K`.
///
/// **Soundness dependency.** These three constants — together with the
/// four medium constants returned by [`medium_challenges_ghash`] — must be
/// **F₂-linearly independent** in F₁₂₈. Zerocheck soundness relies on this
/// (a witness aligned with the friendly subspace would otherwise let the
/// prover cancel the URM message), and so does Ligerito's L0 list-collapse
/// argument (the SZ bound `(m−7)/|F|` for MLE collisions at `r` requires
/// the seven friendly coords to span a 7-dim F₂-subspace). Asserted by
/// `tests::friendly_challenges_f2_independent`.
pub const SMALL_CHAL_F8: [u8; 3] = [0xF7, 0x53, 0xB5];

/// `C_s` as an F_8 value. Verified empirically by the C++ project.
pub const C_S_F8: u8 = 0x1C;

/// The constant `C_s = φ_8(0x1C) ∈ F_{2^128}` — the relative scaling factor
/// between this optimized output and the naive output.
pub fn c_s_f128() -> F128 {
    phi8(F8(C_S_F8))
}

/// The three F_128 small challenges (embeddings of [`SMALL_CHAL_F8`]) — caller
/// must place these at `r[k_skip..k_skip+3]` for the naive cross-check to
/// produce a result related to the optimized output by exactly `C_s`.
pub fn small_challenges_ghash() -> [F128; 3] {
    [
        phi8(F8(SMALL_CHAL_F8[0])),
        phi8(F8(SMALL_CHAL_F8[1])),
        phi8(F8(SMALL_CHAL_F8[2])),
    ]
}

/// The four F_128 medium challenges `β_i = γ^{2^{i-1}} / (1 + γ^{2^{i-1}})`.
/// Caller must place these at `r[k_skip+3..k_skip+7]` for the naive
/// cross-check.
pub fn medium_challenges_ghash() -> [F128; 4] {
    let g1 = F128 {
        lo: 1u64 << 1,
        hi: 0,
    }; // γ^1
    let g2 = F128 {
        lo: 1u64 << 2,
        hi: 0,
    }; // γ^2
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    }; // γ^4
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    }; // γ^8
    [
        g1 * (F128::ONE + g1).inv(),
        g2 * (F128::ONE + g2).inv(),
        g4 * (F128::ONE + g4).inv(),
        g8 * (F128::ONE + g8).inv(),
    ]
}

/// `C_2 = (1+r_2)(1+r_3)` where `r_2 = φ_8(0x53)` (= `α^2/(1+α^2)`),
/// `r_3 = φ_8(0xB5)` (= `α^4/(1+α^4)`). This is the residual small-eq
/// constant after the first small friendly bit (`b_3[0]`, indexed by
/// `r[k_skip] = φ_8(α)`) has been pulled out for the s_hat_v_c bank split:
///
/// ```text
/// eq([r[k_skip+1], r[k_skip+2]], (b_3[1], b_3[2])) = C_2 · α^{2 b_3[1] + 4 b_3[2]}
/// ```
///
/// Used in [`round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`] to
/// post-scale the raw bank values into canonical `s_hat_v_c` (which
/// `ring_switch::fold_1b_rows` would produce against suffix `r[k_skip+1..m]`).
pub fn c_2_small_f128() -> F128 {
    let r_2 = phi8(F8(SMALL_CHAL_F8[1]));
    let r_3 = phi8(F8(SMALL_CHAL_F8[2]));
    (F128::ONE + r_2) * (F128::ONE + r_3)
}

/// `α⁻¹` in F_128, as a subfield-embedded F_8 element. Used to strip the
/// extra `α` factor from `s_hat_v_c`'s bank 1 (the K-odd lattice's raw
/// contribution is `α · α^{2 b_3[1] + 4 b_3[2]}`; canonical wants just
/// `α^{2 b_3[1] + 4 b_3[2]}`).
pub fn alpha_inv_f128() -> F128 {
    // α in F_8 = byte 0x02 (the polynomial generator). Its inverse is α^254;
    // F8::inv computes it via the standard extended Euclidean / power table.
    phi8(F8(0x02).inv())
}

/// `D = (1+γ)(1+γ^2)(1+γ^4)(1+γ^8)`; `D⁻¹` cancels the medium-eq normalization.
fn compute_d_inv() -> F128 {
    let g1 = F128 {
        lo: 1u64 << 1,
        hi: 0,
    };
    let g2 = F128 {
        lo: 1u64 << 2,
        hi: 0,
    };
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    };
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    };
    ((F128::ONE + g1) * (F128::ONE + g2) * (F128::ONE + g4) * (F128::ONE + g8)).inv()
}

static D_INV_CACHE: OnceLock<F128> = OnceLock::new();
fn d_inv() -> F128 {
    *D_INV_CACHE.get_or_init(compute_d_inv)
}

// ---------------------------------------------------------------------------
// Convert table: γ^b · φ_8(v) for b ∈ [0, 16), v ∈ [0, 256).
// 16 × 256 × 16 bytes = 64 KB. Computed once, cached via OnceLock.
// ---------------------------------------------------------------------------

const CONVERT_TABLE_SIZE: usize = 16 * 256;

static CONVERT_TABLE_CACHE: OnceLock<Vec<F128>> = OnceLock::new();

fn build_convert_table() -> Vec<F128> {
    let mut gamma_pow = [F128::ZERO; 16];
    gamma_pow[0] = F128::ONE;
    for b in 1..16 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }
    let mut table = vec![F128::ZERO; CONVERT_TABLE_SIZE];
    for b in 0..16 {
        let g_b = gamma_pow[b];
        for v in 0..256 {
            table[b * 256 + v] = g_b * PHI_8_TABLE[v];
        }
    }
    table
}

fn convert_table() -> &'static [F128] {
    CONVERT_TABLE_CACHE.get_or_init(build_convert_table)
}

#[inline]
pub fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    kernels::bit_transpose_64bytes(input, output);
}

// ---------------------------------------------------------------------------
// Shift_reduce inner kernel (AB only — extract_c handles C separately).
//
// For one medium-position b_med and the 8 small-positions K ∈ 0..8:
//   1. Look up NTT-extended A,B at chunk `chunk_byte_base + (b_med*8 + K)*8`.
//   2. y_K[lane] = ntt_a[lane] · ntt_b[lane]  (in F_8).
//   3. acc[lane] ^= (y_K[lane] as u16) << K   (no reduction yet).
// At the end, reduce each acc[lane] back to a u8 in F_8.
//
// Output `out[lane]` is the F_8 representative of Σ_K x^K · y_K[lane] mod p.
// ---------------------------------------------------------------------------

fn shift_reduce_inner_ab(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    kernels::shift_reduce_inner_ab(
        a_packed,
        b_packed,
        inv_table,
        chunk_byte_base,
        b_med,
        out,
        a_col,
        b_col,
    );
}

// ---------------------------------------------------------------------------
// Main optimized round-1 prover message.
// ---------------------------------------------------------------------------

/// Compute the round-1 prover message via the full shift_reduce + extract_c
/// optimization, in scalar Rust.
///
/// Output relative to [`super::round1_naive`]:
///   `C_s · (res_AB[i] + res_C_lifted[i]) = naive_p_ab[i] + naive_p_c[i]`
///
/// Preconditions:
/// - `k_skip == K_SKIP` (= 6)
/// - `m >= k_skip + N_INNER` (= 13)
/// - `r.len() == m`. `r[k_skip..k_skip+7]` must hold the protocol-fixed small
///   + medium constants (see [`small_challenges_ghash`] /
///   [`medium_challenges_ghash`]) for the naive cross-check to line up. Only
///   `r[k_skip+7..m]` is used internally.
/// - `inv_table.k == k_skip`.
pub fn round1_shift_reduce_extract_c(
    a: &[bool],
    b: &[bool],
    c: &[bool],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert_eq!(c.len(), 1usize << m);
    let a_packed = pack_bits(a);
    let b_packed = pack_bits(b);
    let c_packed = pack_bits(c);
    round1_shift_reduce_extract_c_packed(&a_packed, &b_packed, &c_packed, m, k_skip, r, inv_table)
}

// Per-worker scratch + local accumulator. ~6 KB total, stack-allocated.
struct WorkerState {
    partial_ab: [F128; ELL],
    partial_c: [F128; ELL],
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    chunk_c_bytes: [[u8; 64]; 1 << N_MEDIUM],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s: [F128; ELL],
}

impl WorkerState {
    fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            partial_c: [F128::ZERO; ELL],
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            chunk_c_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s: [F128::ZERO; ELL],
        }
    }
}

/// Process one outer x_hi value: middle-loop over x_outer_lo (reset `partial_ab/c`,
/// run shift_reduce_inner + bit_transpose + convert+apply), then outer fold by
/// `eq_hi_val` into `state.local_res_ab/c_s`.
///
/// Called per-x_hi by both the parallel public function and the serial test oracle.
///
/// `within_outer_mask` and `b_med_counts` together encode the per-block padding
/// pattern (see [`PaddingSpec`]). For each x_outer, `within_hash_outer =
/// x_outer & within_outer_mask` is the position of its 8192-bit window within
/// a block, and `b_med_counts[within_hash_outer]` tells the kernel how many
/// of the 16 b_med 512-bit sub-windows are worth processing — the rest fall
/// entirely in zero padding and are skipped. Pass `within_outer_mask = 0` and
/// `b_med_counts = &[1 << N_MEDIUM]` to disable skipping.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    state: &mut WorkerState,
) {
    state.partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
    state.partial_c.iter_mut().for_each(|p| *p = F128::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;

        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        // Two paths: when n_b_med == 16 (the full case — true for every
        // x_outer_lo on the dense path, and for most of them on the padded
        // path too), use compile-time loop bounds so the SIMD XOR chain
        // unrolls. The slow path handles the rare boundary window where
        // n_b_med < 16.
        if n_b_med == (1 << N_MEDIUM) {
            for b_med in 0..(1 << N_MEDIUM) {
                shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            kernels::accumulate_convert(
                &state.chunk_ab_bytes,
                &state.chunk_c_bytes,
                1 << N_MEDIUM,
                convert,
                eq_lo_val,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
        } else {
            // Partial path: n_b_med ∈ (0, 1 << N_MEDIUM). At most one
            // within_hash_outer value per [`PaddingSpec`] lands here (the
            // window straddling the useful/padding boundary), so the tighter
            // loop wins despite losing the SIMD chain unroll.
            for b_med in 0..n_b_med {
                shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            kernels::accumulate_convert(
                &state.chunk_ab_bytes,
                &state.chunk_c_bytes,
                n_b_med,
                convert,
                eq_lo_val,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
        }
    }

    // Outer fold by eq_hi.
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        state.local_res_c_s[lane] += eq_hi_val * state.partial_c[lane];
    }
}

// ---------------------------------------------------------------------------
// Fusion: two-bank C accumulator that produces s_hat_v_c alongside round 1.
//
// The only structural change from `process_one_x_hi` is in the C-side inner
// loop: instead of one `cf_c` accumulator collapsing all 3 small bits, we
// keep `b_3[0]` (= bit `k_skip` of the witness, = `b_7` in ring-switch's
// packed-prefix index) as a routing dim. Two `cf_c` banks: bank 0 takes
// the K-even contributions (`v_c & 0x55`), bank 1 takes K-odd (`v_c & 0xAA`).
// By F_2-linearity of φ_8, `PHI_8(v) == PHI_8(v & 0x55) + PHI_8(v & 0xAA)`,
// so summing the two banks reconstructs the original `cf_c` → wire `res_c_s`.
//
// Per chunk-lane-b_med, this costs +1 `vld1q_u8` + +1 `veorq_u8`. Everything
// else (shift_reduce_inner_ab, bit_transpose, partial_ab/c fold, eq_hi
// outer fold) is unchanged.
// ---------------------------------------------------------------------------

/// Per-worker scratch + local accumulator for the two-bank C variant.
/// Identical to [`WorkerState`] except `partial_c` and `local_res_c_s` are
/// split into bank 0 / bank 1.
struct WorkerStateWithSHatV {
    partial_ab: [F128; ELL],
    partial_c_0: [F128; ELL],
    partial_c_1: [F128; ELL],
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    chunk_c_bytes: [[u8; 64]; 1 << N_MEDIUM],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s_0: [F128; ELL],
    local_res_c_s_1: [F128; ELL],
}

impl WorkerStateWithSHatV {
    fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            partial_c_0: [F128::ZERO; ELL],
            partial_c_1: [F128::ZERO; ELL],
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            chunk_c_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s_0: [F128::ZERO; ELL],
            local_res_c_s_1: [F128::ZERO; ELL],
        }
    }
}

/// Two-bank C variant of [`process_one_x_hi`]. AB-side and witness traffic
/// unchanged; the only modification is the C-side inner loop now maintains
/// `cf_c_0` and `cf_c_1` via masked convert-table lookups.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_s_hat_v(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    state: &mut WorkerStateWithSHatV,
    compute_c: bool,
    ab_pre: Option<&Round1AbPre>,
) {
    state.partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
    state.partial_c_0.iter_mut().for_each(|p| *p = F128::ZERO);
    state.partial_c_1.iter_mut().for_each(|p| *p = F128::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        if n_b_med == (1 << N_MEDIUM) {
            for b_med in 0..(1 << N_MEDIUM) {
                if ab_pre.is_none() {
                    shift_reduce_inner_ab(
                        a_packed,
                        b_packed,
                        inv_table,
                        chunk_byte_base,
                        b_med,
                        &mut state.chunk_ab_bytes[b_med],
                        &mut state.a_col,
                        &mut state.b_col,
                    );
                }
                if compute_c {
                    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                    let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                        .try_into()
                        .expect("64 c-bytes per medium position");
                    bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
                }
            }

            let ab_src = match ab_pre {
                Some(pre) => pre.outer(x_outer),
                None => &state.chunk_ab_bytes,
            };
            if compute_c {
                kernels::accumulate_convert_with_s_hat_v(
                    ab_src,
                    &state.chunk_c_bytes,
                    1 << N_MEDIUM,
                    convert,
                    eq_lo_val,
                    &mut state.partial_ab,
                    &mut state.partial_c_0,
                    &mut state.partial_c_1,
                );
            } else {
                kernels::accumulate_convert_ab_only(
                    ab_src,
                    1 << N_MEDIUM,
                    convert,
                    eq_lo_val,
                    &mut state.partial_ab,
                );
            }
        } else {
            for b_med in 0..n_b_med {
                if ab_pre.is_none() {
                    shift_reduce_inner_ab(
                        a_packed,
                        b_packed,
                        inv_table,
                        chunk_byte_base,
                        b_med,
                        &mut state.chunk_ab_bytes[b_med],
                        &mut state.a_col,
                        &mut state.b_col,
                    );
                }
                if compute_c {
                    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                    let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                        .try_into()
                        .expect("64 c-bytes per medium position");
                    bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
                }
            }

            let ab_src = match ab_pre {
                Some(pre) => pre.outer(x_outer),
                None => &state.chunk_ab_bytes,
            };
            if compute_c {
                kernels::accumulate_convert_with_s_hat_v(
                    ab_src,
                    &state.chunk_c_bytes,
                    n_b_med,
                    convert,
                    eq_lo_val,
                    &mut state.partial_ab,
                    &mut state.partial_c_0,
                    &mut state.partial_c_1,
                );
            } else {
                kernels::accumulate_convert_ab_only(
                    ab_src,
                    n_b_med,
                    convert,
                    eq_lo_val,
                    &mut state.partial_ab,
                );
            }
        }
    }

    // Outer fold by eq_hi (per bank).
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        state.local_res_c_s_0[lane] += eq_hi_val * state.partial_c_0[lane];
        state.local_res_c_s_1[lane] += eq_hi_val * state.partial_c_1[lane];
    }
}

/// The standard k_skip=6 inverse-NTT table, built once per process.
///
/// The AB precompute and the round-1 drain must read the same table; caching
/// also avoids rebuilding it on every prove.
pub fn cached_inv_table_k6() -> &'static InvNttTableByteSingleGf8 {
    static T: OnceLock<InvNttTableByteSingleGf8> = OnceLock::new();
    T.get_or_init(|| {
        let ntt_s = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    })
}

/// One `shift_reduce_inner_ab` output block.
const AB_PRE_CHUNK: usize = 64;
/// All `b_med` blocks for one `x_outer`.
const AB_PRE_PER_OUTER: usize = (1 << N_MEDIUM) * AB_PRE_CHUNK;

/// The challenge-independent half of round 1, materialized.
///
/// `shift_reduce_inner_ab` reads only the witness and the inverse-NTT table —
/// the verifier's challenge enters round 1 solely through `eq_lo_scaled` and
/// the convert table, both owned by the drain. So the AB transform can be
/// computed before the commitment root exists, and the caller can overlap it
/// with the commit (`rayon::join` in the prover's fast paths).
///
/// Layout is `[x_outer][b_med][64]`, exactly the shape the drain reads —
/// consuming it costs a borrow, not a copy. Size is `2^(m-13)` KiB
/// (512 MiB at m = 32).
pub struct Round1AbPre {
    /// Pool-recycled backing storage, viewed as bytes. Slots for skipped
    /// `b_med` rows are UNINITIALIZED/stale: every consumer bounds its reads
    /// by the same `n_b_med` (derived from the same `PaddingSpec`), so they
    /// are never read — zero-filling 2^(m-13) KiB per prove costs more than
    /// the whole overlap saves.
    storage: Vec<F128>,
    n_outer: usize,
}

impl Round1AbPre {
    /// All `b_med` blocks for one `x_outer`.
    #[inline]
    fn outer(&self, x_outer: usize) -> &[[u8; AB_PRE_CHUNK]; 1 << N_MEDIUM] {
        debug_assert!(x_outer < self.n_outer);
        let off = x_outer * AB_PRE_PER_OUTER;
        // SAFETY: the storage holds `n_outer * AB_PRE_PER_OUTER` bytes and u8
        // has no alignment requirement; in-bounds by the debug_assert.
        unsafe { &*((self.storage.as_ptr() as *const u8).add(off) as *const [[u8; AB_PRE_CHUNK]; 1 << N_MEDIUM]) }
    }

    pub fn len_bytes(&self) -> usize {
        self.storage.len() * core::mem::size_of::<F128>()
    }

    /// Recycle the backing storage (call after round 1 consumed the buffer).
    pub fn recycle(self) {
        crate::scratch::give_f128(self.storage);
    }
}

/// Run the challenge-independent AB transform over the whole witness.
///
/// Mirrors the prep half of [`process_one_x_hi_with_s_hat_v`] exactly,
/// including the `b_med` padding skip, so the drain sees identical bytes.
pub fn precompute_round1_ab(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> Round1AbPre {
    use rayon::prelude::*;

    assert_eq!(k_skip, K_SKIP, "precompute is k_skip=6 only");
    assert!(m >= k_skip + N_INNER);
    let n_outer = 1usize << (m - k_skip - N_INNER);
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding, m);

    let total = n_outer * AB_PRE_PER_OUTER;
    debug_assert_eq!(total % core::mem::size_of::<F128>(), 0);
    let mut storage = crate::scratch::take_f128(total / core::mem::size_of::<F128>());
    // SAFETY: F128 is plain bytes; the byte view covers exactly the buffer.
    // Contents start uninitialized/stale — see the field docs for why the
    // skipped-b_med holes are sound.
    let bytes: &mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(storage.as_mut_ptr() as *mut u8, total) };

    bytes
        .par_chunks_mut(AB_PRE_PER_OUTER)
        .enumerate()
        .for_each(|(x_outer, out)| {
            let within_hash_outer = x_outer & within_outer_mask;
            let n_b_med = b_med_counts[within_hash_outer] as usize;
            if n_b_med == 0 {
                return;
            }
            let chunk_byte_base = (x_outer << N_INNER) * N_CHUNKS;
            let mut a_col = [F8::ZERO; ELL];
            let mut b_col = [F8::ZERO; ELL];
            for b_med in 0..n_b_med {
                let dst: &mut [u8; AB_PRE_CHUNK] = (&mut out
                    [b_med * AB_PRE_CHUNK..(b_med + 1) * AB_PRE_CHUNK])
                    .try_into()
                    .expect("AB_PRE_CHUNK bytes");
                kernels::shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    dst,
                    &mut a_col,
                    &mut b_col,
                );
            }
        });

    Round1AbPre { storage, n_outer }
}

/// Build the `b_med_counts` table from a [`PaddingSpec`] for use by
/// [`process_one_x_hi`].
///
/// Returns `(within_outer_mask, b_med_counts)`:
///   - `within_outer_mask` masks `x_outer` to the bits identifying the
///     window (within-block bits on the single-run fast path; all outer bits
///     on the general run-list path).
///   - `b_med_counts[w]` is how many of the 16 b_med 512-bit sub-windows of
///     window `w` we should process. Entries past the useful prefix are 0
///     (full skip) — kernels just `continue` past those x_outer_lo iterations.
fn build_b_med_counts(padding: &PaddingSpec, m: usize) -> (usize, Vec<u8>) {
    const STRIDE: usize = 1 << (K_SKIP + N_INNER); // 8192 bits per within-window
    const B_MED_WINDOW: usize = 1 << (K_SKIP + 3); // 512 bits per b_med
    const N_B_MED_MAX: usize = 1 << N_MEDIUM;

    // Single-run fast path: the block structure is periodic, so one count per
    // within-block window suffices (byte-identical to the pre-run-list code;
    // the trailing gap, if any, is classified periodically — sound because
    // gap bits are honestly zero, like all padding).
    if let Some(run) = padding.as_single_run() {
        // For k_log < K_SKIP + N_INNER (= 13) the within-window granularity is
        // coarser than the block itself — skipping at this granularity would be
        // incorrect, so we fall back to "no skip". All hash modules use
        // k_log ∈ {14, 15, 16}.
        if run.k_log < K_SKIP + N_INNER {
            return (0, vec![N_B_MED_MAX as u8]);
        }
        let within_outer_bits = run.k_log - K_SKIP - N_INNER;
        let within_outer_count = 1usize << within_outer_bits;
        let within_outer_mask = within_outer_count - 1;
        let useful = run.useful_bits_per_block;
        let counts: Vec<u8> = (0..within_outer_count)
            .map(|w| {
                let block_start = w * STRIDE;
                if block_start >= useful {
                    0u8
                } else {
                    let bits_left = useful - block_start;
                    let processed = bits_left.div_ceil(B_MED_WINDOW);
                    processed.min(N_B_MED_MAX) as u8
                }
            })
            .collect();
        return (within_outer_mask, counts);
    }

    // General run-list path (the multi-table slot schedule — the union
    // prove path reaches it): one count per window over the whole domain, computed
    // from the useful intervals; the mask covers all outer bits. A window's
    // count reaches up to its highest useful bit — all-padding sub-windows
    // below that are processed anyway (contributing zero), which keeps the
    // per-window prefix contract of `process_one_x_hi`.
    assert!(
        padding.covered_bits() <= 1usize << m,
        "PaddingSpec covers {} bits but the domain has only 2^{m}",
        padding.covered_bits()
    );
    let n_windows = 1usize << (m - K_SKIP - N_INNER);
    let mut counts = vec![0u8; n_windows];
    for (start, end) in padding.useful_intervals() {
        for (w, count) in counts
            .iter_mut()
            .enumerate()
            .take((end - 1) / STRIDE + 1)
            .skip(start / STRIDE)
        {
            let covered = end.min((w + 1) * STRIDE) - w * STRIDE;
            *count = (*count).max(covered.div_ceil(B_MED_WINDOW).min(N_B_MED_MAX) as u8);
        }
    }
    (n_windows - 1, counts)
}

/// Packed-input variant of [`round1_shift_reduce_extract_c`]. **Parallel by
/// default** via rayon — the outer x_hi loop is distributed across workers,
/// each with its own scratch + local accumulator. Reduction is a per-lane
/// F128 XOR across workers (commutative + associative).
///
/// To run single-threaded for debugging, set `RAYON_NUM_THREADS=1`.
pub fn round1_shift_reduce_extract_c_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    round1_shift_reduce_extract_c_packed_padded(
        a_packed,
        b_packed,
        c_packed,
        m,
        k_skip,
        r,
        inv_table,
        &PaddingSpec::dense(m),
    )
}

/// Padding-aware variant of [`round1_shift_reduce_extract_c_packed`]. Skips
/// 512-bit b_med sub-windows that fall entirely in the zero padding of every
/// witness block per `padding`. Output is byte-identical to the dense path
/// when the padding bits are honestly zero.
pub fn round1_shift_reduce_extract_c_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;

    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be ≥ k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let eq_hi = &eq.hi;

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding, m);

    // Parallel fold: each worker accumulates a subset of x_hi values into its
    // own WorkerState. Reduce step combines the per-worker `local_res_*` by
    // per-lane F128 XOR.
    let (res_ab, res_c_s) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerState::new, |mut state, x_hi| {
            let eq_hi_val = eq_hi[x_hi];
            process_one_x_hi(
                x_hi,
                big_lo_size,
                n_lo_and_inner,
                within_outer_mask,
                &b_med_counts,
                a_packed,
                b_packed,
                c_packed,
                inv_table,
                &eq_lo_scaled,
                eq_hi_val,
                convert,
                &mut state,
            );
            state
        })
        .map(|s| (s.local_res_ab, s.local_res_c_s))
        .reduce(
            || ([F128::ZERO; ELL], [F128::ZERO; ELL]),
            |(mut ab1, mut c1), (ab2, c2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                    c1[i] += c2[i];
                }
                (ab1, c1)
            },
        );

    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    (res_ab.to_vec(), res_c_lifted)
}

/// Same as [`round1_shift_reduce_extract_c_packed_padded`] but **also returns
/// `s_hat_v_c`** — the length-128 vector ring-switch would otherwise produce
/// via `fold_1b_rows` for the c-claim's PCS opening at suffix `r[k_skip+1..m]`.
///
/// The wire output `(res_ab, res_c_lifted)` is byte-identical to
/// [`round1_shift_reduce_extract_c_packed_padded`] — same eq weights, same
/// `C_s` drop convention. `s_hat_v_c` is returned in **canonical form**
/// (matches `fold_1b_rows`), with the residual `C_2` and `α⁻¹` scaling
/// applied internally so the caller can feed it straight into
/// `pcs::ring_switch::prove_batched_padded_with_precomputed`.
///
/// Cost vs the original: per chunk-lane-`b_med`, +1 `vld1q_u8` + +1 `veorq_u8`
/// (the bank-split convert lookup). bit_transpose, shift_reduce, eq folds
/// are unchanged. See module-level docs for the F_2-linearity argument that
/// makes `s_hat_v_c[(λ, 0)] + s_hat_v_c[(λ, 1)] · α == res_c_s_opt[λ]`.
/// Round-1 C banks computed as a multilinear fold of the lincheck stripe.
///
/// The C side of round 1 is linear in the witness (C = I in the fast BLAKE3
/// setup, so ĉ = ẑ), which means its two per-lane banks are multilinear
/// evaluations and never needed the bit-transpose + convert-table machinery
/// the quadratic AB side needs. Fold the lincheck stripe -- a buffer the
/// prover already builds -- over the non-kept dims at the round-1 eq point:
///
///   - dims `k_log..m` (outer): [`crate::lincheck::partial_fold_packed_z_fast_padded`]
///     with `eq_outer = build_eq(r[k_log..])`, the same O(witness) pass shape
///     lincheck itself uses;
///   - dims `7..k_log`: plain multilinear folds at `r[dim]` (this covers the
///     top window bit(s), the four pinned medium dims -- whose true eq weights
///     ARE `γ^b/D`, the identity the convert table hardcodes -- and small dims
///     7..8);
///   - dim 6 is kept: it is the bank parity;
///   - dims 0..6 are kept: they are the ELL = 64 λ lanes.
///
/// Bank constants from the geometric small-eq identity `eq₃(K) = C_s·α^K`:
/// folding dims 7..8 gives `Y_p = (C_s / eq(r₆, p)) · bank_p`, so
/// `bank_p = eq(r₆, p) · C_s⁻¹ · Y_p` with `C_s = eq₃(0) = (1+r₆)(1+r₇)(1+r₈)`
/// computed directly from `r` (no reliance on the pinned-value constants).
///
/// Returns `(bank0, bank1)` bit-identical to the drain's
/// `(res_c_s_0, res_c_s_1)` -- asserted by
/// `stripe_c_banks_match_drain_banks`.
pub fn round1_c_banks_from_stripe(
    z_stripe: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    r: &[F128],
) -> ([F128; ELL], [F128; ELL]) {
    assert_eq!(r.len(), m);
    assert!(k_log >= K_SKIP + 1 + 2, "need the parity dim + small dims in-block");
    assert!(m - k_log >= 3, "stripe fold needs n_outer >= 8");

    // 1. The one O(witness) pass: fold the outer dims.
    let eq_outer = super::univariate_skip::build_eq(&r[k_log..]);
    // The tiled NEON dispatcher, not the portable fallback: at k_log = 14 the
    // portable kernel's length-k accumulator is 256 KiB -- twice M1's L1D --
    // so every accumulate becomes an L2 round trip. The tiled kernels keep
    // BLOCK_K = 8 accumulators in registers across a stripe sweep, which is
    // why lincheck's own fold runs this shape at roughly half the cost.
    let mut v = crate::lincheck::partial_fold_packed_z_best(
        z_stripe, m, k_log, useful_bits, &eq_outer,
    );

    // 2. Fold dims k_log-1 down to 7 at their r values. All remaining data is
    //    tiny (<= 2^k_log F128s, halving each round).
    let mut len = 1usize << k_log;
    for dim in (7..k_log).rev() {
        let rj = r[dim];
        len >>= 1;
        for i in 0..len {
            let f0 = v[i];
            let f1 = v[i + len];
            v[i] = f0 + rj * (f0 + f1);
        }
    }
    debug_assert_eq!(len, 2 * ELL);

    // 3. Undo the fold's eq factors down to the drain's bank convention.
    let r6 = r[K_SKIP];
    let c_s = (F128::ONE + r6) * (F128::ONE + r[K_SKIP + 1]) * (F128::ONE + r[K_SKIP + 2]);
    let c_s_inv = c_s.inv();
    let k0 = (F128::ONE + r6) * c_s_inv;
    let k1 = r6 * c_s_inv;
    let mut bank0 = [F128::ZERO; ELL];
    let mut bank1 = [F128::ZERO; ELL];
    for lane in 0..ELL {
        bank0[lane] = k0 * v[lane];
        bank1[lane] = k1 * v[ELL + lane];
    }
    (bank0, bank1)
}

/// Like [`round1_c_banks_from_stripe`], but also returns the **banked**
/// `s_hat_v_c` for the direct (basis-free) PCS opening: the same middle fold
/// stopped `c_banks` word dims early, so the C claim's suffix coords
/// `r[7..7+c_banks]` stay unfolded as the bank index. Bank `e`, slice `b`:
///
///   banked[e][b] = w_b · v2[b + 128·e],  w_b = the same per-slice diagonal
///   (C_2·k0 for b < 64, C_2·α⁻¹·k1 for b ≥ 64) that maps the drain banks to
///   canonical `s_hat_v_c` — so `Σ_e eq(r[7..7+c_banks], e)·banked[e]` equals
///   the flat `s_hat_v_c` exactly (multilinearity of the fold).
///
/// Zero extra O(witness) work: the outer partial fold is shared; the banked
/// capture costs one extra `2^(7+c_banks)`-sized intermediate.
pub fn round1_c_banks_from_stripe_with_banked(
    z_stripe: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    r: &[F128],
    c_banks: usize,
) -> ([F128; ELL], [F128; ELL], Vec<Vec<F128>>) {
    assert_eq!(r.len(), m);
    assert!(k_log >= K_SKIP + 1 + 2, "need the parity dim + small dims in-block");
    assert!(m - k_log >= 3, "stripe fold needs n_outer >= 8");
    const LOG2_PACKED: usize = 7; // in-word bit dims (F_{2^128} packing)
    assert!(
        LOG2_PACKED + c_banks <= k_log,
        "banked capture needs the kept word dims in-block (7 + c ≤ k_log)"
    );

    // 1. The one O(witness) pass: fold the outer dims (shared with the
    //    bank-only variant).
    let eq_outer = super::univariate_skip::build_eq(&r[k_log..]);
    let mut v = crate::lincheck::partial_fold_packed_z_best(
        z_stripe, m, k_log, useful_bits, &eq_outer,
    );

    // 2a. Fold the middle dims down to 7 + c_banks (kept: 7 in-word dims +
    //     the c_banks lowest suffix word dims).
    let mut len = 1usize << k_log;
    for dim in ((LOG2_PACKED + c_banks)..k_log).rev() {
        let rj = r[dim];
        len >>= 1;
        for i in 0..len {
            let f0 = v[i];
            let f1 = v[i + len];
            v[i] = f0 + rj * (f0 + f1);
        }
    }
    let v2 = &v[..1 << (LOG2_PACKED + c_banks)];

    // 2b. Canonical per-slice diagonal (identical to the flat derivation).
    let r6 = r[K_SKIP];
    let c_s = (F128::ONE + r6) * (F128::ONE + r[K_SKIP + 1]) * (F128::ONE + r[K_SKIP + 2]);
    let c_s_inv = c_s.inv();
    let k0 = (F128::ONE + r6) * c_s_inv;
    let k1 = r6 * c_s_inv;
    let _ = r; // (small-eq constants are protocol-pinned, not r-derived)
    let c_2 = c_2_small_f128();
    let c_2_alpha_inv = c_2 * alpha_inv_f128();
    let w0 = c_2 * k0;
    let w1 = c_2_alpha_inv * k1;
    let n_banks = 1usize << c_banks;
    let mut banked = vec![vec![F128::ZERO; 2 * ELL]; n_banks];
    for (e, bank) in banked.iter_mut().enumerate() {
        let base = e << LOG2_PACKED;
        for lane in 0..ELL {
            bank[lane] = w0 * v2[base + lane];
            bank[ELL + lane] = w1 * v2[base + ELL + lane];
        }
    }

    // 3. Continue the fold over the kept word dims for the flat banks
    //    (bit-identical to the bank-only variant: same fold, same order).
    let mut len = 1usize << (LOG2_PACKED + c_banks);
    for dim in (LOG2_PACKED..(LOG2_PACKED + c_banks)).rev() {
        let rj = r[dim];
        len >>= 1;
        for i in 0..len {
            let f0 = v[i];
            let f1 = v[i + len];
            v[i] = f0 + rj * (f0 + f1);
        }
    }
    debug_assert_eq!(len, 2 * ELL);
    let mut bank0 = [F128::ZERO; ELL];
    let mut bank1 = [F128::ZERO; ELL];
    for lane in 0..ELL {
        bank0[lane] = k0 * v[lane];
        bank1[lane] = k1 * v[ELL + lane];
    }
    (bank0, bank1, banked)
}

/// The lincheck stripe handed to the stripe-C round-1 entry.
#[derive(Clone, Copy)]
pub struct StripeC<'a> {
    pub stripe: &'a [u8],
    pub k_log: usize,
    pub useful_bits: usize,
    /// When `Some(c)`, the stripe fold also captures the BANKED `s_hat_v_c`
    /// (the C claim's direct-open sufficient statistic, `2^c` banks) at no
    /// extra O(witness) cost — see
    /// [`round1_c_banks_from_stripe_with_banked`].
    pub banked_c: Option<usize>,
}

pub fn round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
    let (ab, c, s, _banked) = round1_with_s_hat_v_impl(
        a_packed, b_packed, c_packed, m, k_skip, r, inv_table, padding, None, None,
    );
    (ab, c, s)
}

/// Round 1 with the C side computed by [`round1_c_banks_from_stripe`] instead
/// of the transpose + convert-table drain. Requires `c_packed` to be the
/// witness itself (C = I), since the stripe is a repacking of z. The AB side
/// is unchanged; the drain runs AB-only and the C transpose is skipped.
#[allow(clippy::too_many_arguments)]
pub fn round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_stripe_c(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
    stripe_c: StripeC<'_>,
    ab_pre: Option<&Round1AbPre>,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Option<Vec<Vec<F128>>>) {
    round1_with_s_hat_v_impl(
        a_packed, b_packed, c_packed, m, k_skip, r, inv_table, padding, Some(stripe_c), ab_pre,
    )
}

#[allow(clippy::too_many_arguments)]
fn round1_with_s_hat_v_impl(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
    stripe_c: Option<StripeC<'_>>,
    ab_pre: Option<&Round1AbPre>,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Option<Vec<Vec<F128>>>) {

    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be ≥ k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let eq_hi = &eq.hi;

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding, m);

    let r1_trace = std::env::var("FLOCK_ZC_TIMING").is_ok();

    // Hetero drain: the gather/PMULL-bound x_hi chunks pull from one shared
    // queue drained by the rayon pool AND the utility-QoS E-core helpers
    // (the whole-prove epool pattern; the E-cores hold at most the chunk
    // they're on, so the merge never gates on them for long). Values are
    // identical under any work distribution: per-worker partials are F128
    // sums, order-free.
    let (res_ab, res_c_s_0, res_c_s_1) = crate::run_hetero_chunks_stateful(
        hi_size,
        WorkerStateWithSHatV::new,
        |state, x_hi| {
            let eq_hi_val = eq_hi[x_hi];
            process_one_x_hi_with_s_hat_v(
                x_hi,
                big_lo_size,
                n_lo_and_inner,
                within_outer_mask,
                &b_med_counts,
                a_packed,
                b_packed,
                c_packed,
                inv_table,
                &eq_lo_scaled,
                eq_hi_val,
                convert,
                state,
                stripe_c.is_none(),
                ab_pre,
            );
        },
    )
    .into_iter()
    .map(|s| (s.local_res_ab, s.local_res_c_s_0, s.local_res_c_s_1))
    .fold(
        ([F128::ZERO; ELL], [F128::ZERO; ELL], [F128::ZERO; ELL]),
        |(mut ab1, mut c0_1, mut c1_1), (ab2, c0_2, c1_2)| {
            for i in 0..ELL {
                ab1[i] += ab2[i];
                c0_1[i] += c0_2[i];
                c1_1[i] += c1_2[i];
            }
            (ab1, c0_1, c1_1)
        },
    );

    // With a stripe, the C banks come from the multilinear fold; the workers
    // above ran AB-only and left their C accumulators zero.
    let mut banked_s_hat_v_c: Option<Vec<Vec<F128>>> = None;
    let (res_c_s_0, res_c_s_1) = match stripe_c {
        Some(sc) => {
            let t_fold = std::time::Instant::now();
            let banks = match sc.banked_c {
                Some(c_banks) => {
                    let (b0, b1, banked) = round1_c_banks_from_stripe_with_banked(
                        sc.stripe,
                        m,
                        sc.k_log,
                        sc.useful_bits,
                        r,
                        c_banks,
                    );
                    banked_s_hat_v_c = Some(banked);
                    (b0, b1)
                }
                None => round1_c_banks_from_stripe(sc.stripe, m, sc.k_log, sc.useful_bits, r),
            };
            if r1_trace {
                eprintln!(
                    "[zc-r1] stripe fold: {:.2} ms",
                    t_fold.elapsed().as_secs_f64() * 1e3
                );
            }
            banks
        }
        None => (res_c_s_0, res_c_s_1),
    };

    // Wire output: bank_0 + bank_1 reconstructs the original `res_c_s` (by
    // F_2-linearity of φ_8 over the masked-byte sum).
    let mut res_c_s_combined = [F128::ZERO; ELL];
    for i in 0..ELL {
        res_c_s_combined[i] = res_c_s_0[i] + res_c_s_1[i];
    }
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s_combined, inv_table);

    // s_hat_v_c canonical form: apply residual C_2 (small-eq constant for
    // r[k_skip+1..k_skip+3]) and α⁻¹ (strips bank 1's extra α factor).
    let c_2 = c_2_small_f128();
    let alpha_inv = alpha_inv_f128();
    let c_2_alpha_inv = c_2 * alpha_inv;
    let mut s_hat_v_c = vec![F128::ZERO; 2 * ELL];
    for lane in 0..ELL {
        s_hat_v_c[lane] = c_2 * res_c_s_0[lane];
        s_hat_v_c[ELL + lane] = c_2_alpha_inv * res_c_s_1[lane];
    }

    (res_ab.to_vec(), res_c_lifted, s_hat_v_c, banked_s_hat_v_c)
}

/// Serial reference — same I/O as [`round1_shift_reduce_extract_c_packed`],
/// no rayon. Kept under `#[cfg(test)]` as the cross-check oracle for the
/// parallel version: future "optimizations" to the parallel path must still
/// produce identical output to this straight-line loop.
#[cfg(test)]
fn round1_shift_reduce_extract_c_packed_serial(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(k_skip, K_SKIP);
    assert!(m >= k_skip + N_INNER);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();

    let (within_outer_mask, b_med_counts) = build_b_med_counts(&PaddingSpec::dense(m), m);

    let mut state = WorkerState::new();
    for x_hi in 0..hi_size {
        process_one_x_hi(
            x_hi,
            big_lo_size,
            n_lo_and_inner,
            within_outer_mask,
            &b_med_counts,
            a_packed,
            b_packed,
            c_packed,
            inv_table,
            &eq_lo_scaled,
            eq.hi[x_hi],
            convert,
            &mut state,
        );
    }

    let res_c_lifted = ntt_extend_f128_vec_ghash(&state.local_res_c_s, inv_table);
    (state.local_res_ab.to_vec(), res_c_lifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::AdditiveNttGf8;
    use crate::zerocheck::univariate_skip::round1_naive;

    /// **Soundness assumption.** Zerocheck and the Ligerito PCS opening at
    /// L0 both depend on the seven "friendly" constants — three small
    /// (`φ_8(SMALL_CHAL_F8[k])`, k ∈ 0..3) and four medium
    /// (`γ^{2^i}/(1+γ^{2^i})`, i ∈ 0..4) — being **F₂-linearly independent**
    /// in F₁₂₈.
    ///
    /// Zerocheck needs this so that the prover's URM message can't be
    /// trivially canceled by a malicious witness aligned with the friendly
    /// subspace. Ligerito's L0 list-collapse argument (which leans on the
    /// zerocheck `(r, v)` claim as an OOD-equivalent) also depends on it
    /// — see the soundness writeup. If any subset of these seven values is
    /// F₂-dependent, the SZ bound `(m−7)/|F|` for collisions between
    /// distinct candidate codewords' MLEs at `r` no longer holds, and a
    /// cheating prover could engineer their witness so two candidates'
    /// MLEs agree at the friendly point with probability 1.
    ///
    /// The check: form the 7×128 binary matrix whose rows are the bit
    /// representations of the seven constants, Gauss-eliminate over F₂,
    /// assert rank = 7.
    #[test]
    fn friendly_challenges_f2_independent() {
        // Pack each F₁₂₈ element into a u128 (lo, hi → 128 bits).
        let mut basis: Vec<u128> = small_challenges_ghash()
            .iter()
            .chain(medium_challenges_ghash().iter())
            .map(|f| ((f.hi as u128) << 64) | (f.lo as u128))
            .collect();
        assert_eq!(
            basis.len(),
            7,
            "expected 3 small + 4 medium friendly values"
        );

        // Row-reduce over F₂. For each column from MSB to LSB, find a row
        // with that bit set (a pivot), swap it into place, and XOR it into
        // every other row to clear that column. Final rank = number of
        // pivots placed.
        let mut rank = 0usize;
        for col in (0..128).rev() {
            let mask = 1u128 << col;
            let pivot = (rank..basis.len()).find(|&i| basis[i] & mask != 0);
            if let Some(p) = pivot {
                basis.swap(rank, p);
                for i in 0..basis.len() {
                    if i != rank && basis[i] & mask != 0 {
                        basis[i] ^= basis[rank];
                    }
                }
                rank += 1;
            }
        }
        assert_eq!(
            rank, 7,
            "friendly challenges must be F₂-linearly independent in F₁₂₈; \
             zerocheck and Ligerito L0 soundness depend on it"
        );
    }

    use crate::test_rng::Rng;

    /// Build the full `r` vector with the protocol-fixed constants in the
    /// small/medium slots. Only `r[k_skip + N_INNER..]` is the actual
    /// randomness fed to the optimized URM.
    fn build_protocol_r(m: usize, outer: &[F128]) -> Vec<F128> {
        assert_eq!(outer.len(), m - K_SKIP - N_INNER);
        let mut r = vec![F128::ZERO; m];
        // r[0..K_SKIP]: not used by either function — can be anything.
        for (i, &small) in small_challenges_ghash().iter().enumerate() {
            r[K_SKIP + i] = small;
        }
        for (i, &med) in medium_challenges_ghash().iter().enumerate() {
            r[K_SKIP + 3 + i] = med;
        }
        for (i, &x) in outer.iter().enumerate() {
            r[K_SKIP + N_INNER + i] = x;
        }
        r
    }

    fn make_inv_table() -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    #[test]
    fn output_shape() {
        let m = 14;
        let mut rng = Rng::new(1);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let table = make_inv_table();

        let (ab, c_l) = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        assert_eq!(ab.len(), ELL);
        assert_eq!(c_l.len(), ELL);
    }

    #[test]
    fn deterministic() {
        let m = 14;
        let mut rng = Rng::new(2);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let table = make_inv_table();

        let out1 = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        let out2 = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        assert_eq!(out1, out2);
    }

    /// **The defining cross-check**: `C_s · (opt_AB + opt_C) == naive_AB + naive_C`,
    /// element-wise on Λ. Verifies all three optimization layers compose
    /// correctly — geometric small eq, geometric medium eq, and the D⁻¹
    /// pre-scaling.
    #[test]
    fn matches_naive_with_c_s_factor() {
        let c_s = c_s_f128();
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(100 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();

            let (naive_ab, naive_c) = round1_naive(&a, &b, &c, m, K_SKIP, &r);
            let (opt_ab, opt_c) = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);

            // Combined: C_s · (opt_AB + opt_C) == naive_AB + naive_C
            for i in 0..ELL {
                let lhs = naive_ab[i] + naive_c[i];
                let rhs = c_s * (opt_ab[i] + opt_c[i]);
                assert_eq!(
                    lhs, rhs,
                    "combined mismatch at m={m}, i={i}:\n  naive={lhs:?}\n  C_s·opt={rhs:?}"
                );
            }

            // Stronger: the AB and C pieces match independently (the AB-only
            // shift_reduce and the C bit_transpose both drop the same C_s).
            for i in 0..ELL {
                assert_eq!(naive_ab[i], c_s * opt_ab[i], "AB mismatch at i={i}");
                assert_eq!(naive_c[i], c_s * opt_c[i], "C mismatch at i={i}");
            }
        }
    }

    /// The structured-b shortcut paths (all-ones 8-K block → a-only kernel;
    /// single-live-K0 block → one dual transform) against the naive oracle.
    /// Random data never exercises them, so craft b: first quarter all-ones,
    /// second quarter zero except each block's first 64 bits, rest random.
    #[test]
    fn matches_naive_with_structured_b_shortcuts() {
        let c_s = c_s_f128();
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(4200 + m as u64);
            let a = rng.bits(1 << m);
            let mut b = rng.bits(1 << m);
            let n = 1usize << m;
            for (i, slot) in b.iter_mut().enumerate() {
                if i < n / 4 {
                    *slot = true; // all-ones blocks
                } else if i < n / 2 {
                    // single-K0 blocks: only the first 64 bits of each
                    // 512-bit (8-K-word) block survive.
                    if i % 512 >= 64 {
                        *slot = false;
                    }
                }
            }
            let c = rng.bits(1 << m);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();

            let (naive_ab, naive_c) = round1_naive(&a, &b, &c, m, K_SKIP, &r);
            let (opt_ab, opt_c) = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
            for i in 0..ELL {
                assert_eq!(naive_ab[i], c_s * opt_ab[i], "AB mismatch at m={m}, i={i}");
                assert_eq!(naive_c[i], c_s * opt_c[i], "C mismatch at m={m}, i={i}");
            }
        }
    }

    #[test]
    fn small_and_medium_challenges_sanity() {
        // Reach into the constants and verify their structural identities.
        // Medium: β_i · (1 + γ^{2^{i-1}}) == γ^{2^{i-1}}.
        let med = medium_challenges_ghash();
        let powers = [1u64 << 1, 1u64 << 2, 1u64 << 4, 1u64 << 8];
        for (i, &p) in powers.iter().enumerate() {
            let g = F128 { lo: p, hi: 0 };
            assert_eq!(med[i] * (F128::ONE + g), g, "β_{i} identity");
        }

        // D · D_inv == 1.
        let d_inv_val = d_inv();
        let g1 = F128 {
            lo: 1u64 << 1,
            hi: 0,
        };
        let g2 = F128 {
            lo: 1u64 << 2,
            hi: 0,
        };
        let g4 = F128 {
            lo: 1u64 << 4,
            hi: 0,
        };
        let g8 = F128 {
            lo: 1u64 << 8,
            hi: 0,
        };
        let d = (F128::ONE + g1) * (F128::ONE + g2) * (F128::ONE + g4) * (F128::ONE + g8);
        assert_eq!(d * d_inv_val, F128::ONE);
    }

    /// Full-entry equivalence: the stripe-C round 1 must match the classic
    /// entry on all three outputs, dense and padded.
    #[test]
    fn stripe_c_entry_matches_classic() {
        use crate::lincheck::pack_z_lincheck;
        use crate::zerocheck::univariate_skip::pack_bits;

        let k_log = 14usize;
        for &(m, useful) in &[(17usize, 1usize << 14), (18, 1 << 14), (18, 3 << 12)] {
            let mut rng = Rng::new(0x57217E5 + m as u64 + useful as u64);
            let a_bits = rng.bits(1 << m);
            let b_bits = rng.bits(1 << m);
            let mut z_bits = rng.bits(1 << m);
            // Honest padding: zero the rows >= useful in every block.
            for (i, bit) in z_bits.iter_mut().enumerate() {
                if (i & ((1 << k_log) - 1)) >= useful {
                    *bit = false;
                }
            }
            let a_p = pack_bits(&a_bits);
            let b_p = pack_bits(&b_bits);
            let c_p = pack_bits(&z_bits);
            let stripe = pack_z_lincheck(&z_bits, m, k_log);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let padding = PaddingSpec::dense(m);

            let classic = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
            );
            let striped = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_stripe_c(
                &a_p,
                &b_p,
                &c_p,
                m,
                K_SKIP,
                &r,
                &table,
                &padding,
                StripeC {
                    stripe: &stripe,
                    k_log,
                    useful_bits: useful,
                    banked_c: None,
                },
                None,
            );
            assert_eq!(classic.0, striped.0, "res_ab at m={m} useful={useful}");
            assert_eq!(classic.1, striped.1, "res_c_lifted at m={m} useful={useful}");
            assert_eq!(classic.2, striped.2, "s_hat_v_c at m={m} useful={useful}");

            // The hoisted AB precompute must be bit-identical to inline prep.
            let ab_pre = precompute_round1_ab(&a_p, &b_p, m, K_SKIP, &table, &padding);
            let pre = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_stripe_c(
                &a_p,
                &b_p,
                &c_p,
                m,
                K_SKIP,
                &r,
                &table,
                &padding,
                StripeC {
                    stripe: &stripe,
                    k_log,
                    useful_bits: useful,
                    banked_c: None,
                },
                Some(&ab_pre),
            );
            assert_eq!(classic.0, pre.0, "ab_pre res_ab at m={m} useful={useful}");
            assert_eq!(classic.1, pre.1, "ab_pre res_c at m={m} useful={useful}");
            assert_eq!(classic.2, pre.2, "ab_pre s_hat_v at m={m} useful={useful}");
        }
    }

    /// The stripe-fold C path must reproduce the drain's banks bit-for-bit.
    /// Banks are recovered from the s_hat_v_c wire output via the module's
    /// own constants, and the lifted C message is cross-checked too.
    /// The banked stripe capture: (a) its flat banks are bit-identical to
    /// the bank-only variant's; (b) `Σ_e eq(r[7..7+c], e)·banked[e]`
    /// reconstructs the flat canonical `s_hat_v_c` exactly.
    #[test]
    fn stripe_banked_s_hat_v_c_matches_flat() {
        use crate::lincheck::pack_z_lincheck;
        use crate::zerocheck::univariate_skip::{build_eq, pack_bits};

        for &(m, c) in &[(17usize, 3usize), (18, 5), (18, 6)] {
            let k_log = 14usize;
            let mut rng = Rng::new(0xBA2CED + (m * 31 + c) as u64);
            let a_bits = rng.bits(1 << m);
            let b_bits = rng.bits(1 << m);
            let z_bits = rng.bits(1 << m);
            let a_p = pack_bits(&a_bits);
            let b_p = pack_bits(&b_bits);
            let c_p = pack_bits(&z_bits);
            let stripe = pack_z_lincheck(&z_bits, m, k_log);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let padding = PaddingSpec::dense(m);

            let (_ab, _c_lifted, s_hat_v_c) =
                round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                    &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
                );
            let (b0_ref, b1_ref) =
                round1_c_banks_from_stripe(&stripe, m, k_log, 1 << k_log, &r);
            let (b0, b1, banked) = round1_c_banks_from_stripe_with_banked(
                &stripe,
                m,
                k_log,
                1 << k_log,
                &r,
                c,
            );
            assert_eq!(b0, b0_ref, "flat bank0 at (m={m}, c={c})");
            assert_eq!(b1, b1_ref, "flat bank1 at (m={m}, c={c})");

            let lo_eq = build_eq(&r[7..7 + c]);
            let mut recon = vec![F128::ZERO; 2 * ELL];
            for (e, bank) in banked.iter().enumerate() {
                for (b, &v) in bank.iter().enumerate() {
                    recon[b] += lo_eq[e] * v;
                }
            }
            assert_eq!(recon, s_hat_v_c, "banked reconstruction at (m={m}, c={c})");
        }
    }

    fn stripe_c_banks_match_drain_banks() {
        use crate::lincheck::pack_z_lincheck;
        use crate::zerocheck::univariate_skip::pack_bits;

        for &m in &[17usize, 18] {
            let k_log = 14usize;
            let mut rng = Rng::new(0x57217EC + m as u64);
            let a_bits = rng.bits(1 << m);
            let b_bits = rng.bits(1 << m);
            let z_bits = rng.bits(1 << m);
            let a_p = pack_bits(&a_bits);
            let b_p = pack_bits(&b_bits);
            let c_p = pack_bits(&z_bits);
            let stripe = pack_z_lincheck(&z_bits, m, k_log);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let padding = PaddingSpec::dense(m);

            let (_ab, c_lifted, s_hat_v_c) =
                round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                    &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
                );
            let (bank0, bank1) =
                round1_c_banks_from_stripe(&stripe, m, k_log, 1 << k_log, &r);

            let c_2 = c_2_small_f128();
            let alpha_inv = alpha_inv_f128();
            for lane in 0..ELL {
                assert_eq!(
                    c_2 * bank0[lane],
                    s_hat_v_c[lane],
                    "bank0 lane {lane} at m={m}"
                );
                assert_eq!(
                    c_2 * alpha_inv * bank1[lane],
                    s_hat_v_c[ELL + lane],
                    "bank1 lane {lane} at m={m}"
                );
            }
            let mut comb = [F128::ZERO; ELL];
            for lane in 0..ELL {
                comb[lane] = bank0[lane] + bank1[lane];
            }
            let lifted = ntt_extend_f128_vec_ghash(&comb, &table);
            assert_eq!(lifted, c_lifted, "lifted C message at m={m}");
        }
    }

    #[test]
    fn parallel_matches_serial() {
        use crate::zerocheck::univariate_skip::pack_bits;

        // At small m the parallel overhead dominates, but the *output* must
        // still match the serial version bit-for-bit. F128 XOR-sum reduction
        // is commutative + associative, so any thread-scheduling order yields
        // the same result.
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xCAFE_F00D + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let a_p = pack_bits(&a);
            let b_p = pack_bits(&b);
            let c_p = pack_bits(&c);

            let (par_ab, par_c) =
                round1_shift_reduce_extract_c_packed(&a_p, &b_p, &c_p, m, K_SKIP, &r, &table);
            let (ser_ab, ser_c) = round1_shift_reduce_extract_c_packed_serial(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table,
            );

            assert_eq!(par_ab, ser_ab, "parallel AB ≠ serial AB at m={m}");
            assert_eq!(par_c, ser_c, "parallel C ≠ serial C at m={m}");
        }
    }

    /// **Padding skip is byte-identical to the dense path.** On a witness
    /// where bits `[useful_bits, 2^k_log)` of every block are honestly zero,
    /// the padded URM must produce the exact same `(round1_ab, round1_c)`
    /// vectors as the dense URM — every chunk we skip would have contributed
    /// a literal zero to the dense sum (the convert table maps φ_8(0) = 0).
    ///
    /// Covers the three hash padding shapes:
    ///   - BLAKE3: k_log=14, useful=15409 → b_med_counts ≈ [16, 15]
    ///   - SHA-2:  k_log=15, useful=31401 → b_med_counts ≈ [16, 16, 16, 14]
    ///   - Keccak: k_log=16, useful=42560 → b_med_counts = [16, 16, 16, 16, 16, 4, 0, 0]
    ///     (this is the only shape that exercises the full-skip case.)
    #[test]
    fn padded_matches_dense_with_zero_padding() {
        use crate::zerocheck::PaddingSpec;
        use crate::zerocheck::univariate_skip::pack_bits;

        // (k_log, useful_bits, n_blocks_log) — pick n_blocks_log so
        // m = k_log + n_blocks_log is small enough to keep the test fast
        // while still exercising the kernel's parallel + boundary paths.
        let cases = [
            (14usize, 15_409usize, 0usize), // BLAKE3, m=14
            (15, 31_401, 0),                // SHA-2,  m=15
            (16, 42_560, 0),                // Keccak, m=16
            (16, 42_560, 3),                // Keccak, m=19 (multiple hashes)
        ];

        for (k_log, useful_bits, n_blocks_log) in cases {
            let m = k_log + n_blocks_log;
            assert!(m >= K_SKIP + N_INNER);

            let mut rng = Rng::new(0xBEEF_DEAD_u64.wrapping_add((k_log * 31 + m) as u64));
            let n_blocks = 1usize << n_blocks_log;
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;

            // Random witness, but force bits [useful_bits, 2^k_log) of every
            // block to zero (mirrors the hash-module witness layout).
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    let idx = blk * block_size + j;
                    a[idx] = false;
                    b[idx] = false;
                    c[idx] = false;
                }
            }

            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let a_p = pack_bits(&a);
            let b_p = pack_bits(&b);
            let c_p = pack_bits(&c);

            let (dense_ab, dense_c) =
                round1_shift_reduce_extract_c_packed(&a_p, &b_p, &c_p, m, K_SKIP, &r, &table);
            let padding = PaddingSpec::uniform(k_log, useful_bits, n_blocks);
            let (padded_ab, padded_c) = round1_shift_reduce_extract_c_packed_padded(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
            );

            assert_eq!(
                dense_ab, padded_ab,
                "AB mismatch: k_log={k_log}, useful={useful_bits}, m={m}"
            );
            assert_eq!(
                dense_c, padded_c,
                "C mismatch: k_log={k_log}, useful={useful_bits}, m={m}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_bit_transpose_matches_scalar() {
        let mut rng = Rng::new(0xB17_BB17);
        for _ in 0..64 {
            let mut input = [0u8; 64];
            for byte in input.iter_mut() {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_neon = [0u8; 64];
            bit_transpose_64bytes_scalar(&input, &mut out_scalar);
            // SAFETY: on aarch64.
            unsafe { bit_transpose_64bytes_neon(&input, &mut out_neon) };
            assert_eq!(out_scalar, out_neon, "bit_transpose disagreement");
        }
    }

    #[cfg(target_arch = "aarch64")]
    /// The structurally-zero-b fast path in `fused_apply_one_k` must be exact.
    /// Random witnesses essentially never contain an all-zero 8-byte b K-row,
    /// so the general oracle above does not reach it; craft the cases directly.
    /// Also covers all-ones and mixed rows so the guard cannot pass wrongly.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_inner_matches_scalar_on_pinned_b_rows() {
        let mut rng = Rng::new(0x5EED_0B);
        let m = 14;
        let table = make_inv_table();
        let a_packed = super::super::univariate_skip::pack_bits(&rng.bits(1 << m));

        // Patterns applied to every 8-byte K-row of the b window under test.
        let patterns: [(&str, fn(usize) -> u8); 4] = [
            ("all-zero", |_| 0x00),
            ("all-ones", |_| 0xff),
            ("zero-then-ones", |i| if i < 32 { 0x00 } else { 0xff }),
            ("one-nonzero-byte", |i| if i == 3 { 0x01 } else { 0x00 }),
        ];

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];
        for (name, f) in patterns {
            let mut b_packed = vec![0u8; a_packed.len()];
            for (i, byte) in b_packed.iter_mut().enumerate() {
                *byte = f(i % 64);
            }
            for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7)] {
                let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
                if needed > a_packed.len() {
                    continue;
                }
                let mut out_scalar = [0u8; 64];
                let mut out_fused = [0u8; 64];
                shift_reduce_inner_ab_scalar(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_scalar,
                    &mut a_col,
                    &mut b_col,
                );
                shift_reduce_inner_ab_fused_neon(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_fused,
                );
                assert_eq!(
                    out_fused, out_scalar,
                    "pattern={name} base={chunk_byte_base} b_med={b_med}"
                );
            }
        }
    }

    #[test]
    fn neon_fused_inner_matches_scalar_inner() {
        // The new register-fused NEON kernel — verify against the same scalar
        // oracle as the intermediate one.
        let mut rng = Rng::new(0xF050D);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_fused = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_fused_neon(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_fused,
            );
            assert_eq!(
                out_scalar, out_fused,
                "fused-neon disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
    #[test]
    fn x86_gfni_sse_inner_matches_scalar_inner() {
        // The SSE/GFNI fallback must remain byte-identical to the scalar oracle.
        let mut rng = Rng::new(0xF050D);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_x86 = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            // SAFETY: gated on gfni target feature.
            unsafe {
                shift_reduce_inner_ab_x86_sse(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_x86,
                    &mut a_col,
                    &mut b_col,
                );
            }
            assert_eq!(
                out_scalar, out_x86,
                "gfni disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn x86_gfni_avx512_inner_matches_scalar_inner() {
        let mut rng = Rng::new(0xA5_512);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);
        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_avx512 = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            // SAFETY: test is compiled only when all kernel features are active.
            unsafe {
                shift_reduce_inner_ab_x86_avx512(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_avx512,
                );
            }
            assert_eq!(
                out_scalar, out_avx512,
                "avx512/gfni disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_inner_matches_scalar_inner() {
        // Pin down the NEON kernel directly: same inputs, same output bytes.
        let mut rng = Rng::new(0x5EED);
        let m = 14;
        let table = make_inv_table();
        let n_chunks = 1 << (K_SKIP / 8); // unused; just sanity
        let _ = n_chunks;
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        // A few representative (chunk_byte_base, b_med) values.
        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            // Guard: don't read past the witness.
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_neon = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_neon(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_neon,
                &mut a_col,
                &mut b_col,
            );
            assert_eq!(
                out_scalar, out_neon,
                "scalar/neon inner disagree at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[test]
    fn convert_table_structure() {
        // convert[b][v] == γ^b · φ_8(v); check at a handful of (b, v).
        let t = convert_table();
        let mut g_pow = F128::ONE;
        for b in 0..16 {
            for &v in &[0u8, 1, 0x57, 0xFF] {
                let expected = g_pow * PHI_8_TABLE[v as usize];
                assert_eq!(t[b * 256 + v as usize], expected, "b={b}, v={v}");
            }
            g_pow = mul_by_x(g_pow);
        }
    }

    /// The two-bank fusion variant produces `(res_ab, res_c_lifted)` that
    /// matches the existing optimized output, AND a `s_hat_v_c` that matches
    /// the scalar-oracle's canonical form.
    #[test]
    fn fusion_matches_existing_and_scalar_oracle() {
        use crate::zerocheck::univariate_skip::round1_extract_c_packed_with_s_hat_v;

        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xF00D_u64.wrapping_add(m as u64));
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let mut r = vec![F128::ZERO; m];
            // Friendly inner constants must match the optimization's
            // expectations: 3 small + 4 medium ghash.
            for i in 0..3 {
                r[K_SKIP + i] = phi8(F8(SMALL_CHAL_F8[i]));
            }
            let medium = crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash();
            for i in 0..4 {
                r[K_SKIP + 3 + i] = medium[i];
            }
            for i in 0..K_SKIP {
                r[i] = rng.f128();
            }
            for i in (K_SKIP + N_INNER)..m {
                r[i] = rng.f128();
            }

            let inv_table = {
                let ntt_s = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8::ZERO);
                let ntt_l = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
                InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
            };

            // Reference 1: existing optimized output (no s_hat_v).
            let (ref_ab, ref_c) = round1_shift_reduce_extract_c_packed_padded(
                &a,
                &b,
                &c,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &PaddingSpec::dense(m),
            );

            // Reference 2: scalar oracle (canonical s_hat_v_c).
            let (_, _, oracle_s_hat_v) =
                round1_extract_c_packed_with_s_hat_v(&a, &b, &c, m, K_SKIP, &r, &inv_table);

            // System under test.
            let (got_ab, got_c, got_s_hat_v) =
                round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                    &a,
                    &b,
                    &c,
                    m,
                    K_SKIP,
                    &r,
                    &inv_table,
                    &PaddingSpec::dense(m),
                );

            assert_eq!(got_ab, ref_ab, "res_ab mismatch at m={m}");
            assert_eq!(got_c, ref_c, "res_c_lifted mismatch at m={m}");
            assert_eq!(got_s_hat_v.len(), 2 * ELL, "s_hat_v length at m={m}");
            assert_eq!(
                got_s_hat_v, oracle_s_hat_v,
                "s_hat_v_c mismatch vs scalar oracle at m={m}"
            );
        }
    }
}
