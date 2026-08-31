//! Merkle-path shift sumcheck.
//!
//! Generalizes the retired hash-chain shift sumcheck (`chain.rs`, deleted
//! 2026-08-14) with a per-row **bit
//! selector**: each non-leaf hash has two inputs `(x_i, y_i)`, and a public
//! bit `b_i` chooses which input is the previous hash's output `z_{i-1}`.
//! The unselected input (the "sibling") is committed in the witness but
//! protocol-unconstrained — soundness comes from preimage resistance: the
//! prover must still hash up to the public root.
//!
//! It implements the Merkle-path variation of the chain shift sumcheck over
//! row-folded scalars `(X_L(i), X_R(i), Z(i))` and the bit MLE `B(i)`.
//!
//! ## Sumcheck shape and bind order
//!
//! Bind order: `y_{n-1}, y_{n-2}, ..., y_0, ss, sd` — n y-rounds first,
//! then 2 slot-rounds. This keeps per-slot `g` tables independent through
//! the y rounds and merges them at the end via the slot-selector folds.
//!
//! - Round 0..n-1 (y-rounds): degree 3 in the bound y-bit. The summand
//!   factors as `T_eq · g_Z + T_shift_α · (1+T_b) · g_XL + T_shift_α · T_b · g_XR`
//!   (cross-slot terms vanish at boolean `(ss, sd)`); the product
//!   `T_shift_α · T_b` is what raises the degree to 3.
//! - Round n, n+1 (ss-, sd-rounds): degree 2. Standard product sumcheck on
//!   4-element `W`/`g` tables.
//!
//! Per-round messages send three field evaluations `(q(1), q(ω), q(ω+1))`
//! where `ω = X` (the polynomial `X` in `F_2[X]/p(X)`, distinct from 0 and
//! 1). Together with `q(0) = C - q(1)` from the running claim, four
//! evaluations interpolate a degree-3 polynomial via Lagrange over
//! `{0, 1, ω, ω+1}`. Degree-2 rounds use the same message shape; the
//! extra evaluation just constrains the interpolation to the lower-degree
//! polynomial.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::zerocheck::multilinear::eq_eval;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which `(sel_slot, side) ∈ {0,1}^2` cube position holds each of the three
/// referenced regions. The fourth slot is the "other" position — holds whatever
/// the per-hash R1CS puts there (e.g. SHA-2's IV); the protocol's weight is
/// zero there at boolean evaluation, so its contents are invisible to the
/// sumcheck.
///
/// Each slot is encoded LSB-first: `slot_index = sel_slot | (side << 1)`.
#[derive(Clone, Copy, Debug)]
pub struct SlotLayout {
    /// Slot index `0..4` holding `Z` (the per-hash output).
    pub z_slot: u8,
    /// Slot index holding `X_L` (left input — selected when `b_i = 0`).
    pub x_l_slot: u8,
    /// Slot index holding `X_R` (right input — selected when `b_i = 1`).
    pub x_r_slot: u8,
}

impl SlotLayout {
    pub fn validate(&self) {
        assert!(
            self.z_slot < 4 && self.x_l_slot < 4 && self.x_r_slot < 4,
            "slot out of range"
        );
        assert_ne!(self.z_slot, self.x_l_slot);
        assert_ne!(self.z_slot, self.x_r_slot);
        assert_ne!(self.x_l_slot, self.x_r_slot);
    }

    /// The "other" slot — the one not assigned to Z, X_L, or X_R.
    pub fn other_slot(&self) -> u8 {
        (0..4)
            .find(|&s| s != self.z_slot && s != self.x_l_slot && s != self.x_r_slot)
            .expect("must exist since 3 slots are assigned out of 4")
    }
}

/// Per-round sumcheck message: three field evaluations of the round
/// polynomial at the points `{1, ω, ω+1}`. The fourth interpolation point
/// `0` is recovered from the running claim via `q(0) = C - q(1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundMsg {
    pub q_1: F128,
    pub q_omega: F128,
    pub q_omega_plus_1: F128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerklePathShiftProof {
    pub rounds: Vec<RoundMsg>, // length n + 2
    pub g_at_point: F128,
    /// PoW witnesses in transcript order: initial `(tau, alpha)`, then one
    /// for each verifier-interpolated round.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// Grinding schedule for packed-position and Merkle-path shift challenges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MerklePathGrinding {
    pub packed_position_bits: u32,
    pub initial_bits: u32,
    /// The verifier accepts a degree-three interpolation in every round.
    pub round_bits: u32,
}

impl MerklePathGrinding {
    pub const fn disabled() -> Self {
        Self {
            packed_position_bits: 0,
            initial_bits: 0,
            round_bits: 0,
        }
    }

    pub const fn per_challenge_128() -> Self {
        Self {
            packed_position_bits: 1,
            initial_bits: 1,
            round_bits: 2,
        }
    }

    pub fn for_profile(profile: flock_core::pcs::ligerito::LigeritoProfile) -> Self {
        match profile {
            flock_core::pcs::ligerito::LigeritoProfile::Fast
            | flock_core::pcs::ligerito::LigeritoProfile::Slim
            | flock_core::pcs::ligerito::LigeritoProfile::Secure => Self::per_challenge_128(),
            flock_core::pcs::ligerito::LigeritoProfile::Fast100
            | flock_core::pcs::ligerito::LigeritoProfile::Slim100 => Self::disabled(),
        }
    }

    pub fn packed_position_bits_for(self, dimension: usize) -> u32 {
        if self.packed_position_bits == 0 || dimension == 0 {
            0
        } else {
            self.packed_position_bits
                .max(flock_core::challenger::grinding_bits_for_degree(dimension))
        }
    }

    fn initial_bits_for(self, n: usize, path_log: usize) -> u32 {
        if self.initial_bits == 0 {
            0
        } else {
            self.initial_bits
                .max(flock_core::challenger::grinding_bits_for_degree(
                    n.max(path_log + 1),
                ))
        }
    }

    fn nonce_count(self, n: usize, path_log: usize) -> usize {
        usize::from(self.initial_bits_for(n, path_log) != 0)
            + (n + 2) * usize::from(self.round_bits != 0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerklePathClaims {
    pub instance_point: Vec<F128>, // (r_{y_0}, ..., r_{y_{n-1}})
    pub sel_slot: F128,
    pub side: F128,
    pub value: F128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MerklePathError {
    MalformedProof,
    SumcheckFinal,
    InvalidGrinding,
}

// ---------------------------------------------------------------------------
// Auxiliary evaluation point
// ---------------------------------------------------------------------------

/// `ω` — the polynomial `X` in `F_2[X]/p(X)`. Distinct from 0 and 1, so
/// `{0, 1, ω, ω+1}` are four distinct field points usable for degree-3
/// Lagrange interpolation.
const OMEGA: F128 = F128 { lo: 2, hi: 0 };

#[inline]
fn omega_plus_1() -> F128 {
    OMEGA + F128::ONE
}

// ---------------------------------------------------------------------------
// Lagrange interpolation over {0, 1, ω, ω+1}
// ---------------------------------------------------------------------------

#[inline]
fn lagrange_eval_degree3(
    q_0: F128,
    q_1: F128,
    q_omega: F128,
    q_omega_plus_1: F128,
    r: F128,
) -> F128 {
    let one = F128::ONE;
    let omega = OMEGA;
    let opo = omega_plus_1();

    // Each L_p(r) = Π_{p' ≠ p} (r - p') / (p - p'). In char 2 subtraction == addition,
    // and all four denominators collapse to the same constant:
    //   d_0 = 1·ω·(ω+1), d_1 = 1·(1+ω)·ω, d_ω = ω·(ω+1)·1, d_{ω+1} = (ω+1)·ω·1,
    // each equal to ω·(ω+1). Factor out one precomputed inverse instead of four
    // per-call Fermat inversions; field ops are exact so the value is unchanged.
    let n0 = (r + one) * (r + omega) * (r + opo);
    let n1 = r * (r + omega) * (r + opo);
    let n2 = r * (r + one) * (r + opo);
    let n3 = r * (r + one) * (r + omega);

    (q_0 * n0 + q_1 * n1 + q_omega * n2 + q_omega_plus_1 * n3) * lagrange_inv_denom()
}

/// (ω·(ω+1))⁻¹ — the shared Lagrange denominator over {0, 1, ω, ω+1} in char 2.
#[inline]
fn lagrange_inv_denom() -> F128 {
    use std::sync::OnceLock;
    static INV: OnceLock<F128> = OnceLock::new();
    *INV.get_or_init(|| (OMEGA * omega_plus_1()).inv())
}

// ---------------------------------------------------------------------------
// Slot indicator MLE
// ---------------------------------------------------------------------------

/// δ_S(ss, sd) for slot `s = ss_target | (sd_target << 1)`. Each is a product
/// of two char-2 multilinear basis polynomials.
#[inline]
fn slot_indicator(target_slot: u8, ss: F128, sd: F128) -> F128 {
    let ss_target = target_slot & 1;
    let sd_target = (target_slot >> 1) & 1;
    let ss_part = if ss_target == 1 { ss } else { F128::ONE + ss };
    let sd_part = if sd_target == 1 { sd } else { F128::ONE + sd };
    ss_part * sd_part
}

// ---------------------------------------------------------------------------
// Shift MLE — the successor-relation MLE `shift(a, b)`: for boolean `y`,
// `shift(τ, y) = eq(τ, y − 1)`. Inlined from the retired hash-chain shift
// argument (`chain.rs`, deleted 2026-08-14); the Merkle-path protocol is
// its only remaining consumer.
// ---------------------------------------------------------------------------

fn shift_mle(a: &[F128], b: &[F128]) -> F128 {
    let n = a.len();
    assert_eq!(b.len(), n, "shift_mle: arity mismatch");

    // pre[j] = Π_{l<j} a_l·(1 + b_l)
    let mut pre = vec![F128::ONE; n + 1];
    for j in 0..n {
        pre[j + 1] = pre[j] * (a[j] * (F128::ONE + b[j]));
    }
    // eqsuf[j] = Π_{l=j}^{n-1} eq(a_l, b_l)
    let mut eqsuf = vec![F128::ONE; n + 1];
    for j in (0..n).rev() {
        let eq_l = F128::ONE + a[j] + b[j];
        eqsuf[j] = eqsuf[j + 1] * eq_l;
    }

    let mut acc = F128::ZERO;
    for j in 0..n {
        let mid = (F128::ONE + a[j]) * b[j]; // bit j flips 0 → 1
        acc += pre[j] * mid * eqsuf[j + 1]; // eqsuf[j+1] = Π_{l>j} eq
    }
    acc
}

// ---------------------------------------------------------------------------
// Bit MLE evaluator (naive O(K))
// ---------------------------------------------------------------------------

fn eval_bit_mle(b_bits: &[bool], r: &[F128]) -> F128 {
    let eq_r = build_eq_table(r);
    let mut acc = F128::ZERO;
    for (y, &bit) in b_bits.iter().enumerate() {
        if bit {
            acc += eq_r[y];
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove the merkle-path shift relation. Inputs:
/// - `path_log` — number of high-order row-index bits that select a *path*
///   id. The total row index `i ∈ {0..N}` decomposes as
///   `i = i_p · 2^pos_log + i_q` with `i_p ∈ {0..P=2^path_log}`,
///   `i_q ∈ {0..L=2^pos_log}`, and `pos_log = n - path_log`. Set `path_log=0`
///   for the single-path protocol (`P = 1`, the whole cube is one path).
/// - `x_l_vals[i] = X_L(i)`, `x_r_vals[i] = X_R(i)`, `z_vals[i] = Z(i)` —
///   instance-indexed scalars after the per-row bit-fold. Length `2^n` each.
/// - `iv_vals[i]` — what's at the "other" slot. Protocol weight is zero there
///   at boolean (ss, sd), but the values participate in the multilinear
///   extension over (ss, sd) and must be provided.
/// - `b_bits` — concatenated public bit vector of length `N`. Bit `b[i_p · L]`
///   (i.e. the first bit of each path) is unused by the protocol — set
///   `B(i_p · L) := 0` by convention; the other bits select which half of each
///   row's input is the within-path chain link.
/// - `layout` — which slot indices hold Z, X_L, X_R.
/// - `grinding` — the Fiat--Shamir PoW schedule: [`MerklePathGrinding::for_profile`]
///   in production, [`MerklePathGrinding::disabled`] for none.
#[allow(clippy::too_many_arguments)]
pub fn prove_merkle_path_shift_with_grinding<Ch: Challenger>(
    path_log: usize,
    x_l_vals: &[F128],
    x_r_vals: &[F128],
    z_vals: &[F128],
    iv_vals: &[F128],
    b_bits: &[bool],
    layout: SlotLayout,
    grinding: MerklePathGrinding,
    challenger: &mut Ch,
) -> (MerklePathShiftProof, MerklePathClaims) {
    layout.validate();
    let n_total = z_vals.len();
    assert!(n_total.is_power_of_two(), "N must be a power of two");
    assert_eq!(x_l_vals.len(), n_total);
    assert_eq!(x_r_vals.len(), n_total);
    assert_eq!(iv_vals.len(), n_total);
    assert_eq!(b_bits.len(), n_total);
    let n = n_total.trailing_zeros() as usize;
    assert!(path_log <= n, "path_log must be ≤ n");
    let pos_log = n - path_log;
    let n_paths = 1usize << path_log;
    let n_pos = 1usize << pos_log;

    let mut grinding_nonces = Vec::with_capacity(grinding.nonce_count(n, path_log));
    let initial_bits = grinding.initial_bits_for(n, path_log);
    // τ, α (transcript-driven, mirrored by verifier).
    let tau = if initial_bits != 0 {
        let (nonce, tau) = challenger.grind_pow_and_sample_f128_vec(initial_bits, n);
        grinding_nonces.push(nonce);
        tau
    } else {
        challenger.sample_f128_vec(n)
    };
    let alpha = challenger.sample_f128();
    // LSB-first split: τ = (τ_q ‖ τ_p) where bits 0..pos_log are the position
    // (within-path) coordinates and bits pos_log..n are the path-id coordinates.
    let tau_q = &tau[..pos_log];
    let tau_p = &tau[pos_log..n];

    let eq_tau = build_eq_table(&tau); // eq(τ, y) for boolean y — size N
    let eq_tau_p = build_eq_table(tau_p); // eq(τ_p, i_p) — size P
    let eq_tau_q = build_eq_table(tau_q); // eq(τ_q, i_q) — size L

    // Factor table: T_shift_α[y=(i_p, i_q)] :=
    //   eq(τ_p, i_p) · shift(τ_q, i_q)  +  α · eq(τ_p, i_p) · δ(i_q = 0)
    //   = eq(τ_p, i_p) · ( shift_q(i_q) + α · δ(i_q = 0) )
    // where shift_q(0) = 0 and shift_q(i_q) = eq(τ_q, i_q - 1) for i_q ≥ 1.
    // For path_log=0: eq_tau_p[0] = 1 and this reduces to
    //   shift(τ, y) + α · δ(y = 0) — the single-path formula.
    let mut t_shift_alpha = vec![F128::ZERO; n_total];
    for i_p in 0..n_paths {
        let weight_p = eq_tau_p[i_p];
        let row_base = i_p << pos_log;
        for i_q in 0..n_pos {
            let shift_iq = if i_q == 0 {
                F128::ZERO
            } else {
                eq_tau_q[i_q - 1]
            };
            let leaf_iq = if i_q == 0 { alpha } else { F128::ZERO };
            t_shift_alpha[row_base | i_q] = weight_p * (shift_iq + leaf_iq);
        }
    }
    let mut t_eq = eq_tau.clone();
    let mut t_b: Vec<F128> = b_bits
        .iter()
        .map(|&b| if b { F128::ONE } else { F128::ZERO })
        .collect();
    // First row of every path has B := 0 by convention (the path's leaf goes
    // into the in_L slot of its first hash). For path_log=0 this is just
    // B(0) := 0 — the single-path convention.
    for i_p in 0..n_paths {
        t_b[i_p << pos_log] = F128::ZERO;
    }

    // Per-slot g tables (length 2^n each). Folded independently through y rounds.
    let mut g_z = z_vals.to_vec();
    let mut g_xl = x_l_vals.to_vec();
    let mut g_xr = x_r_vals.to_vec();
    let mut g_other = iv_vals.to_vec();

    // ------------------------------------------------------------------
    // n y-rounds: bind y_{n-1}, y_{n-2}, ..., y_0 (MSB-first)
    // ------------------------------------------------------------------
    //
    // Each round's polynomial in the bound y-bit:
    //   q(X) = Σ_{y_rem boolean} [
    //       T_eq[X,y_rem] · g_Z[X,y_rem]
    //     + T_shift_α[X,y_rem] · (1 + T_b[X,y_rem]) · g_XL[X,y_rem]
    //     + T_shift_α[X,y_rem] · T_b[X,y_rem] · g_XR[X,y_rem]
    //   ]
    // where T[X,y_rem] := T_lo[y_rem] + X · (T_hi[y_rem] + T_lo[y_rem]).

    let mut rounds: Vec<RoundMsg> = Vec::with_capacity(n + 2);
    let mut r_pts: Vec<F128> = Vec::with_capacity(n + 2);
    let eval_pts = [F128::ONE, OMEGA, omega_plus_1()];

    for _round_idx in 0..n {
        let half = t_b.len() / 2;
        let mut sums = [F128::ZERO; 3];

        for (e, &xx) in eval_pts.iter().enumerate() {
            let mut acc = F128::ZERO;
            for i in 0..half {
                let ta = t_shift_alpha[i] + xx * (t_shift_alpha[i + half] + t_shift_alpha[i]);
                let te = t_eq[i] + xx * (t_eq[i + half] + t_eq[i]);
                let tb = t_b[i] + xx * (t_b[i + half] + t_b[i]);
                let one_plus_tb = F128::ONE + tb;

                let gz = g_z[i] + xx * (g_z[i + half] + g_z[i]);
                let gxl = g_xl[i] + xx * (g_xl[i + half] + g_xl[i]);
                let gxr = g_xr[i] + xx * (g_xr[i + half] + g_xr[i]);

                acc += te * gz + ta * one_plus_tb * gxl + ta * tb * gxr;
            }
            sums[e] = acc;
        }

        let msg = RoundMsg {
            q_1: sums[0],
            q_omega: sums[1],
            q_omega_plus_1: sums[2],
        };
        challenger.observe_f128(msg.q_1);
        challenger.observe_f128(msg.q_omega);
        challenger.observe_f128(msg.q_omega_plus_1);
        let r = if grinding.round_bits != 0 {
            let (nonce, r) = challenger.grind_pow_and_sample_f128(grinding.round_bits);
            grinding_nonces.push(nonce);
            r
        } else {
            challenger.sample_f128()
        };
        rounds.push(msg);
        r_pts.push(r);

        // Fold all 4-or-more tables by r.
        for i in 0..half {
            t_shift_alpha[i] = t_shift_alpha[i] + r * (t_shift_alpha[i + half] + t_shift_alpha[i]);
            t_eq[i] = t_eq[i] + r * (t_eq[i + half] + t_eq[i]);
            t_b[i] = t_b[i] + r * (t_b[i + half] + t_b[i]);
            g_z[i] = g_z[i] + r * (g_z[i + half] + g_z[i]);
            g_xl[i] = g_xl[i] + r * (g_xl[i + half] + g_xl[i]);
            g_xr[i] = g_xr[i] + r * (g_xr[i + half] + g_xr[i]);
            g_other[i] = g_other[i] + r * (g_other[i + half] + g_other[i]);
        }
        t_shift_alpha.truncate(half);
        t_eq.truncate(half);
        t_b.truncate(half);
        g_z.truncate(half);
        g_xl.truncate(half);
        g_xr.truncate(half);
        g_other.truncate(half);
    }

    // After n y-rounds, each table is a single scalar at τ_y.
    let ta = t_shift_alpha[0];
    let te = t_eq[0];
    let tb = t_b[0];
    let one_plus_tb = F128::ONE + tb;

    // ------------------------------------------------------------------
    // 2 slot rounds: build 4-element W, g over (ss, sd) and run a standard
    // degree-2 product sumcheck. Bind sd first (high bit), then ss.
    // ------------------------------------------------------------------
    //
    // Slot table layout (LSB-first cube index = ss | (sd << 1)):
    //   index 0 = (ss=0, sd=0), index 1 = (ss=1, sd=0),
    //   index 2 = (ss=0, sd=1), index 3 = (ss=1, sd=1)
    //
    // W is non-zero at three slot positions (Z, X_L, X_R) and zero at "other".

    let mut w_table = [F128::ZERO; 4];
    let mut g_table = [F128::ZERO; 4];
    let z_slot = layout.z_slot as usize;
    let xl_slot = layout.x_l_slot as usize;
    let xr_slot = layout.x_r_slot as usize;
    let other_slot = layout.other_slot() as usize;
    w_table[z_slot] = te;
    w_table[xl_slot] = ta * one_plus_tb;
    w_table[xr_slot] = ta * tb;
    // w_table[other_slot] = 0 (default).
    g_table[z_slot] = g_z[0];
    g_table[xl_slot] = g_xl[0];
    g_table[xr_slot] = g_xr[0];
    g_table[other_slot] = g_other[0];

    let mut w_vec = w_table.to_vec();
    let mut g_vec = g_table.to_vec();

    // sd round (high bit), then ss round.
    for _round_idx in 0..2 {
        let half = w_vec.len() / 2;
        let mut sums = [F128::ZERO; 3];
        for (e, &xx) in eval_pts.iter().enumerate() {
            let mut acc = F128::ZERO;
            for i in 0..half {
                let w_at_xx = w_vec[i] + xx * (w_vec[i + half] + w_vec[i]);
                let g_at_xx = g_vec[i] + xx * (g_vec[i + half] + g_vec[i]);
                acc += w_at_xx * g_at_xx;
            }
            sums[e] = acc;
        }
        let msg = RoundMsg {
            q_1: sums[0],
            q_omega: sums[1],
            q_omega_plus_1: sums[2],
        };
        challenger.observe_f128(msg.q_1);
        challenger.observe_f128(msg.q_omega);
        challenger.observe_f128(msg.q_omega_plus_1);
        let r = if grinding.round_bits != 0 {
            let (nonce, r) = challenger.grind_pow_and_sample_f128(grinding.round_bits);
            grinding_nonces.push(nonce);
            r
        } else {
            challenger.sample_f128()
        };
        rounds.push(msg);
        r_pts.push(r);
        for i in 0..half {
            w_vec[i] = w_vec[i] + r * (w_vec[i + half] + w_vec[i]);
            g_vec[i] = g_vec[i] + r * (g_vec[i + half] + g_vec[i]);
        }
        w_vec.truncate(half);
        g_vec.truncate(half);
    }

    let g_at_point = g_vec[0];

    // r_pts ordering: y_{n-1}, y_{n-2}, ..., y_0, sd, ss (MSB-first within each group).
    // Build the LSB-first claim point.
    //   instance_point[j] = y_j coord = r bound at position (n - 1 - j) within the
    //                       y-block, i.e. r_pts[n - 1 - j].
    //   sd = r_pts[n]
    //   ss = r_pts[n + 1]
    let mut instance_point = vec![F128::ZERO; n];
    for j in 0..n {
        instance_point[j] = r_pts[n - 1 - j];
    }
    let side = r_pts[n];
    let sel_slot = r_pts[n + 1];

    (
        MerklePathShiftProof {
            rounds,
            g_at_point,
            grinding_nonces,
        },
        MerklePathClaims {
            instance_point,
            sel_slot,
            side,
            value: g_at_point,
        },
    )
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify the merkle-path shift proof. `leaf_evals[i_p]` is the public scalar
/// `leaf_{i_p}(r)` (the r-fold of path `i_p`'s leaf bit-vector); `root_r` is
/// the single shared `root(r)` scalar. For `path_log=0`, `leaf_evals` must be
/// length 1 (the single-path leaf).
/// `grinding` must match the prover's schedule.
#[allow(clippy::too_many_arguments)]
pub fn verify_merkle_path_shift_with_grinding<Ch: Challenger>(
    path_log: usize,
    proof: &MerklePathShiftProof,
    leaf_evals: &[F128],
    root_r: F128,
    b_bits: &[bool],
    n: usize,
    layout: SlotLayout,
    grinding: MerklePathGrinding,
    challenger: &mut Ch,
) -> Result<MerklePathClaims, MerklePathError> {
    layout.validate();
    let d = n + 2;
    if proof.rounds.len() != d {
        return Err(MerklePathError::MalformedProof);
    }
    if proof.grinding_nonces.len() != grinding.nonce_count(n, path_log) {
        return Err(MerklePathError::InvalidGrinding);
    }
    let mut nonce_idx = 0usize;
    assert!(path_log <= n, "path_log must be ≤ n");
    let pos_log = n - path_log;
    let n_paths = 1usize << path_log;
    assert_eq!(b_bits.len(), 1usize << n);
    assert_eq!(
        leaf_evals.len(),
        n_paths,
        "leaf_evals must have length 2^path_log"
    );

    // Resample τ, α (mirror prover).
    let initial_bits = grinding.initial_bits_for(n, path_log);
    let tau = if initial_bits != 0 {
        let tau = challenger
            .verify_pow_and_sample_f128_vec(proof.grinding_nonces[nonce_idx], initial_bits, n)
            .ok_or(MerklePathError::InvalidGrinding)?;
        nonce_idx += 1;
        tau
    } else {
        challenger.sample_f128_vec(n)
    };
    let alpha = challenger.sample_f128();
    let tau_q = &tau[..pos_log];
    let tau_p = &tau[pos_log..n];

    // Initial public claim
    //   C = eq(τ_q, 1^{pos_log}) · root(r)
    //     + α · Σ_{i_p} eq(τ_p, i_p) · leaf_{i_p}(r)
    // (The root term has no τ_p dependence because the per-path boundary
    //  contributions sum out via Σ_{i_p} eq(τ_p, i_p) = 1.)
    let eq_tau_q_ones = tau_q.iter().copied().fold(F128::ONE, |acc, t| acc * t);
    let eq_tau_p_table = build_eq_table(tau_p);
    let combined_leaf: F128 = eq_tau_p_table
        .iter()
        .zip(leaf_evals.iter())
        .map(|(&w, &v)| w * v)
        .fold(F128::ZERO, |a, b| a + b);
    let mut claim = eq_tau_q_ones * root_r + alpha * combined_leaf;

    // Replay sumcheck rounds.
    let mut r_pts: Vec<F128> = Vec::with_capacity(d);
    for msg in &proof.rounds {
        challenger.observe_f128(msg.q_1);
        challenger.observe_f128(msg.q_omega);
        challenger.observe_f128(msg.q_omega_plus_1);
        let r = if grinding.round_bits != 0 {
            let r = challenger
                .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], grinding.round_bits)
                .ok_or(MerklePathError::InvalidGrinding)?;
            nonce_idx += 1;
            r
        } else {
            challenger.sample_f128()
        };
        // q(0) = claim + q(1) (char 2 == subtraction).
        let q_0 = claim + msg.q_1;
        // Evaluate q(r) via Lagrange over {0, 1, ω, ω+1}.
        claim = lagrange_eval_degree3(q_0, msg.q_1, msg.q_omega, msg.q_omega_plus_1, r);
        r_pts.push(r);
    }

    // Reconstruct the random point — same convention as the prover.
    let mut instance_point = vec![F128::ZERO; n];
    for j in 0..n {
        instance_point[j] = r_pts[n - 1 - j];
    }
    let side = r_pts[n];
    let sel_slot = r_pts[n + 1];

    // Compute W(τ_y, sel_slot, side):
    //   shift_wp(τ, τ_y) = eq(τ_p, τ_y_p) · shift(τ_q, τ_y_q)
    //   leaf_α(τ_y)      = α · eq(τ_p, τ_y_p) · eq(τ_y_q, 0^{pos_log})
    //   T_shift_α(τ_y)   = shift_wp + leaf_α
    //                    = eq(τ_p, τ_y_p) · ( shift(τ_q, τ_y_q) + α · eq(τ_y_q, 0^{pos_log}) )
    //   T_eq(τ_y)        = eq(τ, τ_y)
    //   T_b(τ_y)         = B(τ_y) — bit-MLE at the random point
    //
    //   W = δ_Z(ss, sd) · T_eq + δ_XL(ss, sd) · T_shift_α · (1 + T_b)
    //                          + δ_XR(ss, sd) · T_shift_α · T_b
    // (Single-path collapses to the previous formula: τ_p is empty so
    // eq(τ_p, τ_y_p) = 1, and the leaf factor reduces to α · eq(τ_y, 0^n).)
    let tau_y_q = &instance_point[..pos_log];
    let tau_y_p = &instance_point[pos_log..n];
    let eq_taup_tauyp = eq_eval(tau_p, tau_y_p);
    let shift_q = shift_mle(tau_q, tau_y_q);
    // eq(τ_y_q, 0^{pos_log}) = Π_j (1 + τ_y_q,j)
    let eq_tauyq_zero = tau_y_q
        .iter()
        .copied()
        .fold(F128::ONE, |acc, t| acc * (F128::ONE + t));
    let t_shift_alpha = eq_taup_tauyp * (shift_q + alpha * eq_tauyq_zero);
    let t_eq = eq_eval(&tau, &instance_point);
    // B(τ_y) — naive O(N). Apply the per-path B-convention (first row of every
    // path forced to 0) to mirror the prover.
    let mut b_local = b_bits.to_vec();
    for i_p in 0..n_paths {
        b_local[i_p << pos_log] = false;
    }
    let t_b = eval_bit_mle(&b_local, &instance_point);
    let one_plus_t_b = F128::ONE + t_b;

    let w_z_contrib = slot_indicator(layout.z_slot, sel_slot, side) * t_eq;
    let w_xl_contrib =
        slot_indicator(layout.x_l_slot, sel_slot, side) * t_shift_alpha * one_plus_t_b;
    let w_xr_contrib = slot_indicator(layout.x_r_slot, sel_slot, side) * t_shift_alpha * t_b;
    let w_final = w_z_contrib + w_xl_contrib + w_xr_contrib;

    if claim != w_final * proof.g_at_point {
        return Err(MerklePathError::SumcheckFinal);
    }
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());

    Ok(MerklePathClaims {
        instance_point,
        sel_slot,
        side,
        value: proof.g_at_point,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

    /// No-grinding wrapper for the tests below (production always goes
    /// through `_with_grinding`).
    #[allow(clippy::too_many_arguments)]
    fn prove_merkle_path_shift<Ch: Challenger>(
        path_log: usize,
        x_l_vals: &[F128],
        x_r_vals: &[F128],
        z_vals: &[F128],
        iv_vals: &[F128],
        b_bits: &[bool],
        layout: SlotLayout,
        challenger: &mut Ch,
    ) -> (MerklePathShiftProof, MerklePathClaims) {
        prove_merkle_path_shift_with_grinding(
            path_log,
            x_l_vals,
            x_r_vals,
            z_vals,
            iv_vals,
            b_bits,
            layout,
            MerklePathGrinding::disabled(),
            challenger,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_merkle_path_shift<Ch: Challenger>(
        path_log: usize,
        proof: &MerklePathShiftProof,
        leaf_evals: &[F128],
        root_r: F128,
        b_bits: &[bool],
        n: usize,
        layout: SlotLayout,
        challenger: &mut Ch,
    ) -> Result<MerklePathClaims, MerklePathError> {
        verify_merkle_path_shift_with_grinding(
            path_log,
            proof,
            leaf_evals,
            root_r,
            b_bits,
            n,
            layout,
            MerklePathGrinding::disabled(),
            challenger,
        )
    }

    #[test]
    fn fast_slim_and_secure_enable_merkle_path_grinding() {
        use flock_core::pcs::ligerito::LigeritoProfile;

        for profile in [
            LigeritoProfile::Fast,
            LigeritoProfile::Slim,
            LigeritoProfile::Secure,
        ] {
            assert_eq!(
                MerklePathGrinding::for_profile(profile),
                MerklePathGrinding::per_challenge_128()
            );
        }
        for profile in [LigeritoProfile::Fast100, LigeritoProfile::Slim100] {
            assert_eq!(
                MerklePathGrinding::for_profile(profile),
                MerklePathGrinding::disabled()
            );
        }
    }

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn bit(&mut self) -> bool {
            self.next_u64() & 1 == 1
        }
    }

    /// Honest scenario (Framing A):
    /// - K hashes at i = 0..K-1.
    /// - Sel(0) = leaf (= X_L(0) with B(0) := 0 convention).
    /// - For i = 1..K-1: Sel(i) = Z(i-1).
    /// - Z(K-1) = root.
    fn build_honest_scenario(
        n: usize,
        seed: u64,
    ) -> (
        Vec<F128>, // x_l
        Vec<F128>, // x_r
        Vec<F128>, // z
        Vec<F128>, // iv (arbitrary)
        Vec<bool>, // b
        F128,      // leaf = Sel(0) = X_L(0)
        F128,      // root = Z(K-1)
    ) {
        let mut rng = Rng::new(seed);
        let k = 1usize << n;
        // Random per-hash outputs (we're testing the sumcheck math; R1CS would
        // enforce z_i = h(x_i, y_i) but here we just need the scalars).
        let z_vals: Vec<F128> = (0..k).map(|_| rng.f128()).collect();
        // Bit vector: b_0 is unused (B(0) := 0); b_1..b_{K-1} are random.
        let mut b_bits = vec![false; k];
        for i in 1..k {
            b_bits[i] = rng.bit();
        }
        let mut x_l = vec![F128::ZERO; k];
        let mut x_r = vec![F128::ZERO; k];
        // Hash 0: Sel(0) = leaf. With B(0) = 0, Sel(0) = X_L(0). So X_L(0) = leaf.
        let leaf = rng.f128();
        x_l[0] = leaf;
        x_r[0] = rng.f128(); // sibling — arbitrary
        // Hashes 1..K-1: Sel(i) = Z(i-1).
        for i in 1..k {
            if !b_bits[i] {
                x_l[i] = z_vals[i - 1]; // selected
                x_r[i] = rng.f128(); // sibling
            } else {
                x_r[i] = z_vals[i - 1]; // selected
                x_l[i] = rng.f128(); // sibling
            }
        }
        let iv: Vec<F128> = (0..k).map(|_| rng.f128()).collect();
        let root = z_vals[k - 1];
        (x_l, x_r, z_vals, iv, b_bits, leaf, root)
    }

    fn canonical_layout() -> SlotLayout {
        SlotLayout {
            z_slot: 0,
            x_l_slot: 1,
            x_r_slot: 2,
        }
    }

    #[test]
    fn lagrange_inv_denom_is_inverse() {
        assert_eq!(lagrange_inv_denom() * (OMEGA * omega_plus_1()), F128::ONE);
    }

    #[test]
    fn lagrange_constant_inverse_matches_reference() {
        // Reference: the pre-optimization form with four per-term Fermat inversions.
        fn reference(q_0: F128, q_1: F128, q_omega: F128, q_omega_plus_1: F128, r: F128) -> F128 {
            let one = F128::ONE;
            let omega = OMEGA;
            let opo = omega_plus_1();
            let n0 = (r + one) * (r + omega) * (r + opo);
            let l0 = n0 * (omega * opo).inv();
            let n1 = r * (r + omega) * (r + opo);
            let l1 = n1 * ((one + omega) * omega).inv();
            let n2 = r * (r + one) * (r + opo);
            let l2 = n2 * (omega * opo).inv();
            let n3 = r * (r + one) * (r + omega);
            let l3 = n3 * (opo * omega).inv();
            q_0 * l0 + q_1 * l1 + q_omega * l2 + q_omega_plus_1 * l3
        }
        let mut rng = Rng::new(0x1A65EED);
        for _ in 0..1000 {
            let (a, b, c, d, r) = (rng.f128(), rng.f128(), rng.f128(), rng.f128(), rng.f128());
            assert_eq!(
                lagrange_eval_degree3(a, b, c, d, r),
                reference(a, b, c, d, r)
            );
        }
        // Boundary points: r ∈ {0, 1, ω, ω+1} must reproduce the four inputs.
        let (a, b, c, d) = (rng.f128(), rng.f128(), rng.f128(), rng.f128());
        assert_eq!(lagrange_eval_degree3(a, b, c, d, F128::ZERO), a);
        assert_eq!(lagrange_eval_degree3(a, b, c, d, F128::ONE), b);
        assert_eq!(lagrange_eval_degree3(a, b, c, d, OMEGA), c);
        assert_eq!(lagrange_eval_degree3(a, b, c, d, omega_plus_1()), d);
    }

    #[test]
    fn honest_roundtrip_accepts() {
        for &n in &[3usize, 4, 5] {
            let (x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xC0FFEE + n as u64);
            let layout = canonical_layout();

            let mut ch_p = FsChallenger::new(b"merkle-test-v0");
            let (proof, claims_p) =
                prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"merkle-test-v0");
            let claims_v =
                verify_merkle_path_shift(0, &proof, &[leaf], root, &b, n, layout, &mut ch_v)
                    .unwrap_or_else(|e| panic!("verify rejected honest proof at n={n}: {e:?}"));

            assert_eq!(claims_v, claims_p, "claims must match at n={n}");
        }
    }

    #[test]
    fn grinding_roundtrip_and_rejects_bad_nonce_shape() {
        let n = 4;
        let path_log = 0;
        let (x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0x1280_4D50);
        let layout = canonical_layout();
        let policy = MerklePathGrinding::per_challenge_128();
        let mut chp = FsChallenger::new(b"merkle-path-grinding-test");
        let (proof, claims_p) = prove_merkle_path_shift_with_grinding(
            path_log, &x_l, &x_r, &z, &iv, &b, layout, policy, &mut chp,
        );
        let leaves = vec![leaf; 1 << path_log];
        let mut chv = FsChallenger::new(b"merkle-path-grinding-test");
        let claims_v = verify_merkle_path_shift_with_grinding(
            path_log, &proof, &leaves, root, &b, n, layout, policy, &mut chv,
        )
        .expect("grinded Merkle-path shift verifies");
        assert_eq!(claims_p, claims_v);

        let mut missing = proof;
        missing.grinding_nonces.pop();
        let mut chv = FsChallenger::new(b"merkle-path-grinding-test");
        assert_eq!(
            verify_merkle_path_shift_with_grinding(
                path_log, &missing, &leaves, root, &b, n, layout, policy, &mut chv,
            ),
            Err(MerklePathError::InvalidGrinding)
        );
    }

    #[test]
    fn broken_chain_rejects() {
        let n = 4;
        let (mut x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xBAD);
        // Corrupt a linked X_L (b=0 case).
        for i in 1..(1 << n) {
            if !b[i] {
                x_l[i] += F128::ONE;
                break;
            }
        }
        let layout = canonical_layout();
        let mut ch_p = FsChallenger::new(b"merkle-test-v0");
        let (proof, _) = prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"merkle-test-v0");
        let res = verify_merkle_path_shift(0, &proof, &[leaf], root, &b, n, layout, &mut ch_v);
        assert!(matches!(res, Err(MerklePathError::SumcheckFinal)));
    }

    #[test]
    fn wrong_leaf_rejects() {
        let n = 4;
        let (x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xC0DE);
        let layout = canonical_layout();
        let mut ch_p = FsChallenger::new(b"merkle-test-v0");
        let (proof, _) = prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);
        // Verify with a tampered leaf.
        let mut bad_leaf = leaf;
        bad_leaf.lo ^= 1;
        let mut ch_v = FsChallenger::new(b"merkle-test-v0");
        let res = verify_merkle_path_shift(0, &proof, &[bad_leaf], root, &b, n, layout, &mut ch_v);
        assert!(matches!(res, Err(MerklePathError::SumcheckFinal)));
    }

    #[test]
    fn wrong_root_rejects() {
        let n = 4;
        let (x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xDEAD);
        let layout = canonical_layout();
        let mut ch_p = FsChallenger::new(b"merkle-test-v0");
        let (proof, _) = prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);
        let mut bad_root = root;
        bad_root.lo ^= 1;
        let mut ch_v = FsChallenger::new(b"merkle-test-v0");
        let res = verify_merkle_path_shift(0, &proof, &[leaf], bad_root, &b, n, layout, &mut ch_v);
        assert!(matches!(res, Err(MerklePathError::SumcheckFinal)));
    }
}
