//! Keccak-f[1600] permutation primitives + compact per-Keccak R1CS encoder.
//!
//! ## Two layers
//!
//! 1. **Keccak-f primitives**: reference bit-level implementation and the
//!    fast u64-lane version following FIPS 202. Shared with the rest of the
//!    crate (used as the oracle in tests, the witness generator's forward
//!    simulator, and as a baseline for `keccak_native_chain`).
//!
//! 2. **Monolithic R1CS encoder** at **K_LOG = 16**: one R1CS instance per
//!    full Keccak, all 24 rounds folded inline. Drops every intermediate
//!    state slot — only `state_0` (input) and `state_24` (output) plus
//!    `t_0..t_23` (the 24 χ AND-outputs) are materialized in the witness.
//!    State_r for r ∈ {1..23} is *implicit* via the substitution
//!    `state_r = φ^r·state_0 ⊕ Σ_{i<r} φ^{r-1-i}·t_i ⊕ RC_r` (with
//!    φ = θ∘ρ∘π). The substituted A/B matrices would be ~700× denser than
//!    a fully-materialized baseline, but we never materialize them — the
//!    [`KeccakLincheckCircuit`] walker computes the lincheck column marginal
//!    by a backward transpose recurrence
//!    `K_r = φᵀ(K_{r+1}) ⊕ χ_r` density-independent in ~1M F128 ops.
//!
//! `state_24` is kept at an I/O-aligned slot (2048-bit window starting at
//! bit 2048) so the generic chain shift sumcheck can compare consecutive
//! instances' `state_24[i] == state_0[i+1]` without any layout change. It
//! also enforces `state_24 = keccak_f(state_0)` end-to-end via a 1,600-row
//! pin block (`L_24(state_0, t_<24)[j] = z[state_24 col j]`).
//!
//! ## Witness layout per block (k = 2^16 = 65,536 slots, 1024 u64 lanes)
//!
//! ```text
//!   bit 0     .. 1600   state_0   (input region; aligned slot 0, pad 1600..2048 zero)
//!   bit 2048  .. 3648   state_24  (output region; aligned slot 1, pad 3648..4096 zero)
//!   bit 4096            constant z = 1 (bit 0 of u64[64])
//!   bit 4160  .. 42560  t_0 .. t_23  (24 vecs, tight, 1600 stride)
//!   bit 42560 .. K      zero tail
//! ```
//!
//! Useful bits = `4160 + 24·1600 = 42,560` (padding ≈ 35%).
//!
//! `state_0` and `state_24` live in 2048-bit aligned slots (`region_log = 11`)
//! so the generic chain shift sumcheck operates on them unchanged.
//!
//! ## What the R1CS enforces
//!
//! - Row 0 (const): `z[0]·z[0] = z[0]`.
//! - state_0 input self-loops (1,600 rows): `z[row]·z[0] = z[row]`.
//! - **state_24 pin rows (1,600 rows)**: `L_24(state_0, t_<24)[j] · z[0]
//!   = z[z_pos_state24(j)]`, where `L_24 = φ^24·state_0 ⊕ Σ_{i<24} φ^{23-i}·t_i
//!   ⊕ RC_24`. After substitution the A-side row has hundreds of dense taps
//!   on (state_0 + t_<24) columns — never materialized; circuit walker handles.
//! - 24 × 1,600 t-AND rows: `t_r[x,y,z] = (¬φ(state_r)[(x+1)%5,y,z]) ·
//!   φ(state_r)[(x+2)%5,y,z]`, with `state_r` implicit via L_r.
//! - Padding: empty A, B.
//!
//! ## Matrices
//!
//! [`build_block_r1cs`] returns `BlockR1cs` with **empty** `a_0`, `b_0`
//! stubs and `c_0 = I_K`. The empty matrices are never consulted on the
//! `prove_fast`/`verify` path — the constraint definition lives in
//! [`KeccakLincheckCircuit`], and witness gen emits `a`, `b` values
//! directly from the running keccak_f simulation. `r1cs.satisfies(z)` and
//! the slow `prove` path will report "everything satisfied vacuously" for
//! this encoding — only use `prove_fast`/`prove_chain`.

use crate::r1cs_hashes::chain_common::{ChainLayout, ChainProof, ChainVerifyError};
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProof};
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier;

// ===========================================================================
// Keccak-f[1600] primitives
// ===========================================================================

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Keccak-f state size in bits.
pub const STATE_BITS: usize = 1600;

/// Number of u64 lanes per Keccak state vector.
pub const N_LANES: usize = 25;
/// Bits per u64 lane.
pub const LANE_BITS: usize = 64;
/// Bits per state vector. Equals `STATE_BITS = N_LANES · LANE_BITS`.
pub const STATE_SIZE_BITS: usize = N_LANES * LANE_BITS;
/// Number of materialized t vectors per Keccak (t_0 … t_23).
pub const N_T: usize = 24;
/// Number of Keccak-f rounds. Matches `ROUND_CONSTANTS.len()`.
pub const N_ROUNDS: usize = 24;
/// Univariate-skip dim (must match `zerocheck::K_SKIP`).
pub const K_SKIP: usize = 6;

/// ρ rotation offsets `r[x][y]` (FIPS 202 Table 2).
const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// Keccak-f round constants for the ι step (24 rounds).
pub const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// State-bit linear index: `state_idx(x, y, z) = x + 5y + 25z`, range [0, 1600).
#[inline]
pub fn state_idx(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < 5 && y < 5 && z < 64);
    x + 5 * y + 25 * z
}

// ---------------------------------------------------------------------------
// Linear preimage: which state_in bits XOR to give B[x,y,z]?
// ---------------------------------------------------------------------------

/// Returns the state_in bit indices whose XOR equals `B[x, y, z]`, where
/// `B = (π ∘ ρ ∘ θ)(state_in)`. There are always 11 distinct indices.
pub fn theta_rho_pi_preimage(x: usize, y: usize, z: usize) -> [usize; 11] {
    // π: B[x, y, z] = A_ρ[(x + 3y) mod 5, x, z]
    // ρ: A_ρ[a, b, c] = A_θ[a, b, (c − r[a][b]) mod 64]
    // θ: A_θ[a, b, c] = A[a, b, c]
    //                  ⊕ ⊕_{y'} A[(a−1) mod 5, y', c]
    //                  ⊕ ⊕_{y'} A[(a+1) mod 5, y', (c−1) mod 64]
    let a = (x + 3 * y) % 5;
    let b = x;
    let r = (RHO_OFFSETS[a][b] as usize) % 64;
    let c = (z + 64 - r) % 64;
    let c_prev = (c + 63) % 64;
    let a_minus = (a + 4) % 5;
    let a_plus = (a + 1) % 5;

    let mut bits = [0usize; 11];
    bits[0] = state_idx(a, b, c);
    for yp in 0..5 {
        bits[1 + yp] = state_idx(a_minus, yp, c);
    }
    for yp in 0..5 {
        bits[6 + yp] = state_idx(a_plus, yp, c_prev);
    }
    bits
}

// ---------------------------------------------------------------------------
// State types
// ---------------------------------------------------------------------------

/// Keccak-f state as 1600 GF(2) bits.
pub type State = [bool; STATE_BITS];

/// 25 u64 lanes = 1600 bits. Lane index = `x + 5y`; bit z within a lane is at
/// u64 bit position `z`.
pub type Lanes = [u64; 25];

#[inline]
fn lane_idx(x: usize, y: usize) -> usize {
    debug_assert!(x < 5 && y < 5);
    x + 5 * y
}

/// Convert `[bool; 1600]` (state_idx layout) → `[u64; 25]` lanes.
pub fn state_to_lanes(s: &State) -> Lanes {
    let mut lanes = [0u64; 25];
    for z in 0..64 {
        for y in 0..5 {
            for x in 0..5 {
                if s[state_idx(x, y, z)] {
                    lanes[lane_idx(x, y)] |= 1u64 << z;
                }
            }
        }
    }
    lanes
}

/// Convert `[u64; 25]` lanes → `[bool; 1600]` (state_idx layout).
pub fn lanes_to_state(lanes: &Lanes) -> State {
    let mut s = [false; STATE_BITS];
    for y in 0..5 {
        for x in 0..5 {
            let lane = lanes[lane_idx(x, y)];
            for z in 0..64 {
                s[state_idx(x, y, z)] = (lane >> z) & 1 == 1;
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Fast u64-lane Keccak — 50-100× faster than bit-by-bit reference.
// ---------------------------------------------------------------------------

/// θ on u64 lanes.
#[inline]
pub fn theta_lanes(s: &mut Lanes) {
    let c = [
        s[lane_idx(0, 0)]
            ^ s[lane_idx(0, 1)]
            ^ s[lane_idx(0, 2)]
            ^ s[lane_idx(0, 3)]
            ^ s[lane_idx(0, 4)],
        s[lane_idx(1, 0)]
            ^ s[lane_idx(1, 1)]
            ^ s[lane_idx(1, 2)]
            ^ s[lane_idx(1, 3)]
            ^ s[lane_idx(1, 4)],
        s[lane_idx(2, 0)]
            ^ s[lane_idx(2, 1)]
            ^ s[lane_idx(2, 2)]
            ^ s[lane_idx(2, 3)]
            ^ s[lane_idx(2, 4)],
        s[lane_idx(3, 0)]
            ^ s[lane_idx(3, 1)]
            ^ s[lane_idx(3, 2)]
            ^ s[lane_idx(3, 3)]
            ^ s[lane_idx(3, 4)],
        s[lane_idx(4, 0)]
            ^ s[lane_idx(4, 1)]
            ^ s[lane_idx(4, 2)]
            ^ s[lane_idx(4, 3)]
            ^ s[lane_idx(4, 4)],
    ];
    let d = [
        c[4] ^ c[1].rotate_left(1),
        c[0] ^ c[2].rotate_left(1),
        c[1] ^ c[3].rotate_left(1),
        c[2] ^ c[4].rotate_left(1),
        c[3] ^ c[0].rotate_left(1),
    ];
    for y in 0..5 {
        for x in 0..5 {
            s[lane_idx(x, y)] ^= d[x];
        }
    }
}

/// Combined ρ ∘ π on u64 lanes.
#[inline]
pub fn rho_pi_lanes(s_in: &Lanes) -> Lanes {
    let mut s_out = [0u64; 25];
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = RHO_OFFSETS[a][b] % 64;
            s_out[lane_idx(x, y)] = s_in[lane_idx(a, b)].rotate_left(r);
        }
    }
    s_out
}

/// χ on u64 lanes.
#[inline]
pub fn chi_lanes(s_in: &Lanes) -> Lanes {
    let mut s_out = [0u64; 25];
    for y in 0..5 {
        let a0 = s_in[lane_idx(0, y)];
        let a1 = s_in[lane_idx(1, y)];
        let a2 = s_in[lane_idx(2, y)];
        let a3 = s_in[lane_idx(3, y)];
        let a4 = s_in[lane_idx(4, y)];
        s_out[lane_idx(0, y)] = a0 ^ ((!a1) & a2);
        s_out[lane_idx(1, y)] = a1 ^ ((!a2) & a3);
        s_out[lane_idx(2, y)] = a2 ^ ((!a3) & a4);
        s_out[lane_idx(3, y)] = a3 ^ ((!a4) & a0);
        s_out[lane_idx(4, y)] = a4 ^ ((!a0) & a1);
    }
    s_out
}

/// ι on u64 lanes (in-place).
#[inline]
pub fn iota_lanes(s: &mut Lanes, round_idx: usize) {
    s[lane_idx(0, 0)] ^= ROUND_CONSTANTS[round_idx];
}

/// One Keccak-f round on u64 lanes (in-place).
#[inline]
pub fn keccak_round_lanes(s: &mut Lanes, round_idx: usize) {
    theta_lanes(s);
    *s = rho_pi_lanes(s);
    *s = chi_lanes(s);
    iota_lanes(s, round_idx);
}

/// Compute `t = χ((π∘ρ∘θ)(state_in))`'s "AND output" component as u64 lanes:
/// `t_lane(x, y) = (¬B_lane(x+1, y)) ∧ B_lane(x+2, y)` where
/// `B = (π∘ρ∘θ)(state_in)`.
#[inline]
pub fn compute_round_t_lanes(state_in: &Lanes) -> Lanes {
    let mut b = *state_in;
    theta_lanes(&mut b);
    let b = rho_pi_lanes(&b);
    let mut t = [0u64; 25];
    for y in 0..5 {
        let b1 = b[lane_idx(1, y)];
        let b2 = b[lane_idx(2, y)];
        let b3 = b[lane_idx(3, y)];
        let b4 = b[lane_idx(4, y)];
        let b0 = b[lane_idx(0, y)];
        t[lane_idx(0, y)] = (!b1) & b2;
        t[lane_idx(1, y)] = (!b2) & b3;
        t[lane_idx(2, y)] = (!b3) & b4;
        t[lane_idx(3, y)] = (!b4) & b0;
        t[lane_idx(4, y)] = (!b0) & b1;
    }
    t
}

// ---------------------------------------------------------------------------
// Legacy bool-array Keccak primitives (kept as the test oracle / reference).
// ---------------------------------------------------------------------------

/// θ step (in-place). Linear over GF(2).
pub fn theta(s: &mut State) {
    let mut c = [[false; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            let mut acc = false;
            for y in 0..5 {
                acc ^= s[state_idx(x, y, z)];
            }
            c[x][z] = acc;
        }
    }
    let mut d = [[false; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            d[x][z] = c[(x + 4) % 5][z] ^ c[(x + 1) % 5][(z + 63) % 64];
        }
    }
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                s[state_idx(x, y, z)] ^= d[x][z];
            }
        }
    }
}

/// Combined ρ ∘ π step.
pub fn rho_pi(s_in: &State) -> State {
    let mut s_out = [false; STATE_BITS];
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                let a = (x + 3 * y) % 5;
                let b = x;
                let r = (RHO_OFFSETS[a][b] as usize) % 64;
                let c = (z + 64 - r) % 64;
                s_out[state_idx(x, y, z)] = s_in[state_idx(a, b, c)];
            }
        }
    }
    s_out
}

/// χ step. The only nonlinear step.
pub fn chi(s_in: &State) -> State {
    let mut s_out = [false; STATE_BITS];
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                let a = s_in[state_idx(x, y, z)];
                let b = s_in[state_idx((x + 1) % 5, y, z)];
                let c = s_in[state_idx((x + 2) % 5, y, z)];
                s_out[state_idx(x, y, z)] = a ^ ((!b) & c);
            }
        }
    }
    s_out
}

/// ι step (in-place). XORs the round constant into lane (0, 0).
pub fn iota(s: &mut State, round_idx: usize) {
    let rc = ROUND_CONSTANTS[round_idx];
    for z in 0..64 {
        if (rc >> z) & 1 != 0 {
            s[state_idx(0, 0, z)] ^= true;
        }
    }
}

/// One Keccak-f round. Fast path uses u64 lanes; bool↔lane conversions at the
/// boundary.
pub fn keccak_round(s: &mut State, round_idx: usize) {
    let mut lanes = state_to_lanes(s);
    keccak_round_lanes(&mut lanes, round_idx);
    *s = lanes_to_state(&lanes);
}

/// Full Keccak-f[1600] (24 rounds).
pub fn keccak_f(s: &mut State) {
    for r in 0..24 {
        keccak_round(s, r);
    }
}

// ===========================================================================
// Compact monolithic R1CS encoder at K_LOG = 16 (I/O-aligned)
// ===========================================================================

// ---------------------------------------------------------------------------
// Constants and layout
// ---------------------------------------------------------------------------

pub const K_LOG: usize = 16;
pub const K: usize = 1 << K_LOG;

/// 2^11 = 2048-bit aligned region for state_0 and state_24.
pub const SLOT_BITS: usize = 2048;

/// state_0 base (slot 0, bit 0).
pub const STATE0_BIT_BASE: usize = 0;
/// state_24 base (slot 1, bit 2048).
pub const STATE24_BIT_BASE: usize = SLOT_BITS;
/// Constant slot — bit 0 of u64[64].
pub const Z_CONST: usize = 2 * SLOT_BITS; // 4096
/// First tightly-packed t_r begins one lane past the const.
pub const T_PACKED_BIT_BASE: usize = Z_CONST + LANE_BITS; // 4160

pub const USEFUL_BITS: usize = T_PACKED_BIT_BASE + N_T * STATE_SIZE_BITS; // 42,560

const _: () = {
    assert!(USEFUL_BITS <= K);
};

pub const U64_PER_BLOCK: usize = K / 64; // 1024

/// `u64`-lane index for state_r where r ∈ {0, 24}.
#[inline]
fn state_u64_base(r: usize) -> usize {
    match r {
        0 => STATE0_BIT_BASE / LANE_BITS,
        24 => STATE24_BIT_BASE / LANE_BITS,
        _ => panic!("keccak_chain only materializes state_0 and state_24, got r={r}"),
    }
}

/// `u64`-lane index for t_r (r ∈ 0..N_T).
#[inline]
fn t_u64_base(r: usize) -> usize {
    debug_assert!(r < N_T);
    (T_PACKED_BIT_BASE / LANE_BITS) + r * N_LANES
}

/// `u64`-lane index for the constant.
const Z_CONST_U64: usize = Z_CONST / LANE_BITS; // 64

/// Within-sub-vector offset for logical bit `j` (lane-contiguous).
#[inline]
fn within_lane_contiguous(j: usize) -> usize {
    let lane_xy = j % N_LANES;
    let z_in_lane = j / N_LANES;
    LANE_BITS * lane_xy + z_in_lane
}

/// Position of bit `j` of state_r (r ∈ {0, 24}) in the per-block witness.
#[inline]
pub fn z_pos_state(r: usize, j: usize) -> usize {
    debug_assert!(j < STATE_BITS);
    match r {
        0 => STATE0_BIT_BASE + within_lane_contiguous(j),
        24 => STATE24_BIT_BASE + within_lane_contiguous(j),
        _ => panic!("keccak_chain only materializes state_0 and state_24, got r={r}"),
    }
}

/// Position of bit `j` of `t_r` in the per-block witness.
#[inline]
pub fn z_pos_t(r: usize, j: usize) -> usize {
    debug_assert!(r < N_T);
    debug_assert!(j < STATE_BITS);
    T_PACKED_BIT_BASE + r * STATE_SIZE_BITS + within_lane_contiguous(j)
}

// ---------------------------------------------------------------------------
// R1CS shell — uses shared helpers in `super::common`.
// ---------------------------------------------------------------------------

/// Minimum `n_keccaks_log` needed to prove `n_keccaks` Keccak permutations,
/// subject to the lincheck floor (≥ 3).
pub fn min_n_keccaks_log(n_keccaks: usize) -> usize {
    assert!(n_keccaks >= 1, "n_keccaks must be ≥ 1");
    let n = n_keccaks.max(8);
    let bits = n.next_power_of_two().trailing_zeros() as usize;
    bits.max(3)
}

/// Apply φᵀ to a length-`STATE_BITS` F128 buffer: `out[s_in] = Σ_{s_out :
/// s_in ∈ preim(s_out)} in[s_out]`.
pub(crate) fn apply_phi_t(v: &[F128]) -> Vec<F128> {
    debug_assert_eq!(v.len(), STATE_BITS);
    let mut out = vec![F128::ZERO; STATE_BITS];
    for s_out in 0..STATE_BITS {
        let z_p = s_out / N_LANES;
        let xy = s_out % N_LANES;
        let y_p = xy / 5;
        let x_p = xy % 5;
        let preim = theta_rho_pi_preimage(x_p, y_p, z_p);
        let val = v[s_out];
        for &s_in in preim.iter() {
            out[s_in] += val;
        }
    }
    out
}

/// Forward-apply φ on a `bool` state — tracks the round-constant accumulator
/// `RC_r` (an F_2 state). `out[s_out] = XOR_{s_in ∈ preim(s_out)} in[s_in]`.
pub(crate) fn apply_phi_bool(v: &[bool; STATE_BITS]) -> [bool; STATE_BITS] {
    let mut out = [false; STATE_BITS];
    for s_out in 0..STATE_BITS {
        let z_p = s_out / N_LANES;
        let xy = s_out % N_LANES;
        let y_p = xy / 5;
        let x_p = xy % 5;
        let preim = theta_rho_pi_preimage(x_p, y_p, z_p);
        let mut bit = false;
        for &s_in in preim.iter() {
            bit ^= v[s_in];
        }
        out[s_out] = bit;
    }
    out
}

pub fn build_block_r1cs(n_keccaks_log: usize) -> BlockR1cs {
    super::common::build_block_r1cs_empty_stub(n_keccaks_log, K_LOG, K_SKIP, USEFUL_BITS)
}

// ---------------------------------------------------------------------------
// Witness generation
// ---------------------------------------------------------------------------

fn build_chain_witness_ab_packed_into(
    initial: &State,
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    debug_assert_eq!(z_u64.len(), U64_PER_BLOCK);
    debug_assert_eq!(a_u64.len(), U64_PER_BLOCK);
    debug_assert_eq!(b_u64.len(), U64_PER_BLOCK);

    // Constant z[Z_CONST] = 1, a = b = 1.
    z_u64[Z_CONST_U64] = 1;
    a_u64[Z_CONST_U64] = 1;
    b_u64[Z_CONST_U64] = 1;

    // state_0 input self-loops.
    let mut state_lanes: Lanes = state_to_lanes(initial);
    let s0_base = state_u64_base(0);
    for lane_idx in 0..N_LANES {
        let pos = s0_base + lane_idx;
        let v = state_lanes[lane_idx];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }

    // 24 rounds: forward-simulate, write t_r AND-row values.
    for r in 0..N_ROUNDS {
        let mut b_state: Lanes = state_lanes;
        theta_lanes(&mut b_state);
        let b_state: Lanes = rho_pi_lanes(&b_state);

        let mut t_lanes: Lanes = [0u64; 25];
        for y in 0..5 {
            let b0 = b_state[5 * y];
            let b1 = b_state[1 + 5 * y];
            let b2 = b_state[2 + 5 * y];
            let b3 = b_state[3 + 5 * y];
            let b4 = b_state[4 + 5 * y];
            t_lanes[5 * y] = (!b1) & b2;
            t_lanes[1 + 5 * y] = (!b2) & b3;
            t_lanes[2 + 5 * y] = (!b3) & b4;
            t_lanes[3 + 5 * y] = (!b4) & b0;
            t_lanes[4 + 5 * y] = (!b0) & b1;
        }

        let mut next: Lanes = [0u64; 25];
        for i in 0..25 {
            next[i] = b_state[i] ^ t_lanes[i];
        }
        iota_lanes(&mut next, r);

        // t_r AND-row values.
        let t_base = t_u64_base(r);
        for y in 0..5 {
            for x in 0..5 {
                let lane_idx = x + 5 * y;
                let pos = t_base + lane_idx;
                z_u64[pos] = t_lanes[lane_idx];
                a_u64[pos] = !b_state[(x + 1) % 5 + 5 * y];
                b_u64[pos] = b_state[(x + 2) % 5 + 5 * y];
            }
        }

        state_lanes = next;
    }

    // After 24 rounds state_lanes = state_24 = keccak_f(initial). Write the
    // state_24 pin rows: z = a = state_24, b = 1 (B side is [Z_CONST]).
    let s24_base = state_u64_base(24);
    for lane_idx in 0..N_LANES {
        let pos = s24_base + lane_idx;
        let v = state_lanes[lane_idx];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }
    // Trailing padding stays zero.
}

pub fn generate_witness_with_ab_packed_and_lincheck(
    initial_states: &[State],
    n_keccaks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    // Constant-wire pin (docs/const-wire-pin.md): fill padding blocks with a
    // valid keccak_f(0) computation so the constant cell is 1 in every block.
    // (The chain forbids padding — `prove_chain` asserts no padding — so this is
    // a no-op there and only affects the standalone batch setup.)
    let padding: State = [false; STATE_BITS];
    super::common::drive_witness_packed_and_lincheck(
        initial_states,
        Some(&padding),
        n_keccaks_log,
        K_LOG,
        build_chain_witness_ab_packed_into,
    )
}

// ---------------------------------------------------------------------------
// Lincheck circuit walker — extends [`keccak::KeccakLincheckCircuit`] by one
// transpose-recurrence round to absorb the state_24 pin rows.
//
//   K^A_24 = vec_pin (where vec_pin[j] = eq_inner[z_pos_state(24, j)])
//   K^A_r  = φᵀ(K^A_{r+1}) ⊕ χ_{r,A}   for r ∈ 23..0
//
// Scattering: K^A_24 → t_23 col (pin row's t_23 coefficient = φ^0 · K^A_24);
// K^A_r → t_{r-1} col for r ∈ {23..1}; K^A_0 → state_0 col. The state_24 col
// itself receives no A or B contribution (pin row writes to it via C = I).
//
// B side: pin's B = [Z_CONST] only. K^B_24 = 0; recurrence collapses to
// `keccak`'s B-side recurrence.
//
// Z_CONST: receives `α·dot(vec_pin, RC_24) + sum_eq_pin` extra (vs `keccak`).
// ---------------------------------------------------------------------------

pub struct KeccakLincheckCircuit;

impl LincheckCircuit for KeccakLincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    /// Pin the constant cell to 1 across all blocks. Requires the witness to fill
    /// padding blocks with valid keccak computations (constant = 1) — see
    /// [`generate_witness_with_ab_packed_and_lincheck`] and docs/const-wire-pin.md.
    fn const_pin_col(&self) -> Option<usize> {
        Some(Z_CONST)
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
        let mut comb = vec![F128::ZERO; K];

        // ---- Row 0 (const): A = [Z_CONST], B = [Z_CONST].
        let e0 = eq_inner[Z_CONST];
        comb[Z_CONST] += alpha * e0;
        comb[Z_CONST] += e0;

        // ---- state_0 input self-loops: A = [row], B = [Z_CONST].
        for j in 0..STATE_BITS {
            let row = z_pos_state(0, j);
            let e = eq_inner[row];
            comb[row] += alpha * e;
            comb[Z_CONST] += e;
        }

        // ---- state_24 pin rows: A = L_24[j], B = [Z_CONST].
        let mut vec_pin: Vec<F128> = vec![F128::ZERO; STATE_BITS];
        let mut sum_eq_pin = F128::ZERO;
        for j in 0..STATE_BITS {
            let row = z_pos_state(24, j);
            let e = eq_inner[row];
            vec_pin[j] = e;
            sum_eq_pin += e;
        }
        comb[Z_CONST] += sum_eq_pin; // B-side from pin's B = [Z_CONST]

        // ---- t-AND rows: build per-round χ marginals on state_r positions.
        let mut chi_a: Vec<Vec<F128>> = (0..N_T).map(|_| vec![F128::ZERO; STATE_BITS]).collect();
        let mut chi_b: Vec<Vec<F128>> = (0..N_T).map(|_| vec![F128::ZERO; STATE_BITS]).collect();
        let mut sum_eq_t = F128::ZERO;

        for r in 0..N_T {
            for zpos in 0..64 {
                for y in 0..5 {
                    for x in 0..5 {
                        let row = z_pos_t(r, state_idx(x, y, zpos));
                        let e = eq_inner[row];
                        sum_eq_t += e;
                        for &s in theta_rho_pi_preimage((x + 1) % 5, y, zpos).iter() {
                            chi_a[r][s] += e;
                        }
                        for &s in theta_rho_pi_preimage((x + 2) % 5, y, zpos).iter() {
                            chi_b[r][s] += e;
                        }
                    }
                }
            }
        }
        comb[Z_CONST] += alpha * sum_eq_t;

        // ---- Round-constant accumulation. After loop rc = RC_24.
        let mut rc = [false; STATE_BITS];
        let mut rc_a = F128::ZERO;
        let mut rc_b = F128::ZERO;
        for r in 0..N_T {
            for s in 0..STATE_BITS {
                if rc[s] {
                    rc_a += chi_a[r][s];
                    rc_b += chi_b[r][s];
                }
            }
            rc = apply_phi_bool(&rc);
            for zpos in 0..64 {
                if (ROUND_CONSTANTS[r] >> zpos) & 1 == 1 {
                    let s = state_idx(0, 0, zpos);
                    rc[s] ^= true;
                }
            }
        }
        let mut rc_pin = F128::ZERO;
        for s in 0..STATE_BITS {
            if rc[s] {
                rc_pin += vec_pin[s];
            }
        }
        comb[Z_CONST] += alpha * rc_a;
        comb[Z_CONST] += rc_b;
        comb[Z_CONST] += alpha * rc_pin;

        // ---- Transpose recurrence, A side. Starts at K^A_24 = vec_pin.
        // Step 1: scatter K^A_24 → t_23 col.
        let t23_base = t_u64_base(N_T - 1);
        for s in 0..STATE_BITS {
            let pos = (t23_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
            comb[pos] += alpha * vec_pin[s];
        }
        // Step 2: K^A_23 = φᵀ(K^A_24) ⊕ χ_{23,A}.
        let mut k_a = apply_phi_t(&vec_pin);
        for s in 0..STATE_BITS {
            k_a[s] += chi_a[N_T - 1][s];
        }
        // Step 3: standard recurrence for r ∈ {23..1}.
        for r in (1..N_T).rev() {
            let t_base = t_u64_base(r - 1);
            for s in 0..STATE_BITS {
                let pos = (t_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
                comb[pos] += alpha * k_a[s];
            }
            let mut new_k = apply_phi_t(&k_a);
            for s in 0..STATE_BITS {
                new_k[s] += chi_a[r - 1][s];
            }
            k_a = new_k;
        }
        let s0_base = state_u64_base(0);
        for s in 0..STATE_BITS {
            let pos = (s0_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
            comb[pos] += alpha * k_a[s];
        }

        // ---- Transpose recurrence, B side. K^B_24 = 0, so K^B_23 = χ_{23,B}
        // — same starting point as `keccak`. Pin row doesn't contribute to
        // t_23 col on B side.
        let mut k_b = chi_b[N_T - 1].clone();
        for r in (1..N_T).rev() {
            let t_base = t_u64_base(r - 1);
            for s in 0..STATE_BITS {
                let pos = (t_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
                comb[pos] += k_b[s];
            }
            let mut new_k = apply_phi_t(&k_b);
            for s in 0..STATE_BITS {
                new_k[s] += chi_b[r - 1][s];
            }
            k_b = new_k;
        }
        for s in 0..STATE_BITS {
            let pos = (s0_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
            comb[pos] += k_b[s];
        }

        comb
    }
}

// ---------------------------------------------------------------------------
// CHAIN_LAYOUT
// ---------------------------------------------------------------------------

/// I/O geometry for the generic chain core: `state_0` in aligned slot 0,
/// `state_24` in slot 1, each in a 2048-bit (`region_log = 11`) window
/// holding a 1600-bit Keccak state.
pub const CHAIN_LAYOUT: ChainLayout = ChainLayout {
    k_log: K_LOG,
    k_skip: K_SKIP,
    region_log: 11,
    region_bits: STATE_BITS,
    input_byte_off: STATE0_BIT_BASE / 8,   // 0
    output_byte_off: STATE24_BIT_BASE / 8, // 256
};

/// Convert a public state (logical FIPS bit order) to the region's physical
/// within-slot bit order. Used for the public-endpoint fold in verify_chain.
pub fn state_to_phys_bits(x: &State) -> Vec<bool> {
    let mut phys = vec![false; STATE_BITS];
    for j in 0..STATE_BITS {
        phys[within_lane_contiguous(j)] = x[j];
    }
    phys
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct KeccakSetup {
    pub n_keccaks: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: PcsParams,
}

impl KeccakSetup {
    pub fn new(n_keccaks: usize) -> Self {
        Self::with_log_inv_rate(n_keccaks, 1)
    }

    pub fn with_log_inv_rate(n_keccaks: usize, log_inv_rate: usize) -> Self {
        // Rate keys the legacy profiles: 1 -> Fast, 2 -> Slim.
        let profile = match log_inv_rate {
            1 => flock_core::pcs::ligerito::LigeritoProfile::Fast,
            2 => flock_core::pcs::ligerito::LigeritoProfile::Slim,
            _ => flock_core::pcs::ligerito::LigeritoProfile::Fast, // BaseFold-only rates
        };
        Self::with_profile_and_rate(n_keccaks, profile, log_inv_rate)
    }

    /// Build a setup for a named Ligerito profile (fast/slim/secure);
    /// the PCS rate follows the profile.
    pub fn with_profile(
        n_keccaks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        Self::with_profile_and_rate(n_keccaks, profile, profile.log_inv_rate())
    }

    fn with_profile_and_rate(
        n_keccaks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
        log_inv_rate: usize,
    ) -> Self {
        assert!(n_keccaks >= 1);
        let n_log = min_n_keccaks_log(n_keccaks);
        let r1cs = build_block_r1cs(n_log);
        // Pre-fault the prove-cycle scratch buffers — see scratch::prewarm_prover.
        flock_core::scratch::prewarm_prover(r1cs.m);
        let pcs_params = PcsParams {
            m: r1cs.m,
            log_inv_rate,
            log_batch_size: 6,
            profile,
        };
        Self {
            n_keccaks,
            r1cs,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_keccaks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn n_keccak_slots(&self) -> usize {
        1usize << self.n_keccaks_log()
    }

    /// **Legacy BaseFold-backend** prove. Use `prove_fast` (Ligerito) for
    /// production. Kept for m < ~21 and for backend timing comparisons.
    pub fn prove_fast_basefold<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (R1csProof, Commitment, R1csClaim) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_keccaks_log());
        crate::prover::prove_fast_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            &KeccakLincheckCircuit,
            challenger,
        )
    }

    /// **Legacy BaseFold-backend** verify. Pairs with `prove_fast_basefold`.
    pub fn verify_basefold<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &R1csProof,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, verifier::VerifyError> {
        verifier::verify(
            &self.r1cs,
            commitment,
            proof,
            &KeccakLincheckCircuit,
            challenger,
        )
    }

    /// Build a base R1CS proof (without the chain shift sumcheck) using the
    /// **Ligerito** backend. Smaller proof, slightly slower prover.
    /// Requires `m ≥ ~21`; for tiny instances use `prove_fast_basefold`.
    pub fn prove_fast<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_keccaks_log())
            });
        crate::prover::prove_fast_ligerito_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            &KeccakLincheckCircuit,
            codeword,
            challenger,
        )
    }

    /// Verify a [`Self::prove_fast`] (Ligerito) proof.
    pub fn verify<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, verifier::VerifyError> {
        verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            &KeccakLincheckCircuit,
            &self.pcs_params,
            challenger,
        )
    }

    /// Legacy BaseFold-backend chain prove.
    pub fn prove_chain_basefold<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (ChainProof, Commitment) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        assert_eq!(
            self.n_keccaks,
            self.n_keccak_slots(),
            "prove_chain requires n_keccaks to exactly fill n_keccak_slots \
             (no padding); got n_keccaks={}, n_keccak_slots={}. Use a \
             power-of-2 ≥ 8.",
            self.n_keccaks,
            self.n_keccak_slots(),
        );
        let n_log = self.n_keccaks_log();
        let (z_packed, a_packed, b_packed, z_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(initial_states, n_log);
        crate::r1cs_hashes::chain_common::prove_chain_generic(
            &self.r1cs,
            &self.pcs_params,
            &CHAIN_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            &KeccakLincheckCircuit,
            challenger,
        )
    }

    /// Prove that `initial_states` form a sequential chain: for the committed
    /// witness, `state_24` of instance `i` equals `state_0` of instance `i+1`,
    /// with public endpoints `x_0 = state_0[0]` and `x_last = state_24[N-1]`.
    /// **Ligerito backend.** Requires m ≥ ~21.
    pub fn prove_chain<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (
        crate::r1cs_hashes::chain_common::ChainProofLigerito,
        Commitment,
    ) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        assert_eq!(self.n_keccaks, self.n_keccak_slots());
        let n_log = self.n_keccaks_log();
        let (z_packed, a_packed, b_packed, z_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(initial_states, n_log);
        crate::r1cs_hashes::chain_common::prove_chain_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &CHAIN_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            &KeccakLincheckCircuit,
            challenger,
        )
    }

    /// Verify a [`Self::prove_chain`] (Ligerito) proof.
    pub fn verify_chain<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &crate::r1cs_hashes::chain_common::ChainProofLigerito,
        x_0: &State,
        x_last: &State,
        challenger: &mut Ch,
    ) -> Result<(), ChainVerifyError> {
        assert_eq!(self.n_keccaks, self.n_keccak_slots());
        let n_log = self.n_keccaks_log();
        let x0_phys = state_to_phys_bits(x_0);
        let xlast_phys = state_to_phys_bits(x_last);
        crate::r1cs_hashes::chain_common::verify_chain_ligerito_generic(
            &self.r1cs,
            &CHAIN_LAYOUT,
            commitment,
            proof,
            n_log,
            &x0_phys,
            &xlast_phys,
            &KeccakLincheckCircuit,
            &self.pcs_params,
            challenger,
        )
    }

    /// Legacy BaseFold-backend chain verify.
    pub fn verify_chain_basefold<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &ChainProof,
        x_0: &State,
        x_last: &State,
        challenger: &mut Ch,
    ) -> Result<(), ChainVerifyError> {
        assert_eq!(
            self.n_keccaks,
            self.n_keccak_slots(),
            "verify_chain requires n_keccaks to exactly fill n_keccak_slots \
             (no padding); got n_keccaks={}, n_keccak_slots={}. Use a \
             power-of-2 ≥ 8.",
            self.n_keccaks,
            self.n_keccak_slots(),
        );
        let n_log = self.n_keccaks_log();
        let x0_phys = state_to_phys_bits(x_0);
        let xlast_phys = state_to_phys_bits(x_last);
        crate::r1cs_hashes::chain_common::verify_chain_generic(
            &self.r1cs,
            &CHAIN_LAYOUT,
            commitment,
            proof,
            n_log,
            &x0_phys,
            &xlast_phys,
            &KeccakLincheckCircuit,
            challenger,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 PRNG.
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
    }

    fn random_state(rng: &mut Rng) -> State {
        let mut s = [false; STATE_BITS];
        let mut i = 0;
        while i < STATE_BITS {
            let w = rng.next_u64();
            for b in 0..64 {
                if i + b < STATE_BITS {
                    s[i + b] = (w >> b) & 1 == 1;
                }
            }
            i += 64;
        }
        s
    }

    // -------- Reference (u64-lane) Keccak-f, used as an independent oracle.

    fn keccak_f_u64(a: &mut [[u64; 5]; 5]) {
        for round in 0..24 {
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4];
            }
            let mut d = [0u64; 5];
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }
            for x in 0..5 {
                for y in 0..5 {
                    a[x][y] ^= d[x];
                }
            }
            let mut b = [[0u64; 5]; 5];
            for x in 0..5 {
                for y in 0..5 {
                    b[y][(2 * x + 3 * y) % 5] = a[x][y].rotate_left(RHO_OFFSETS[x][y]);
                }
            }
            for x in 0..5 {
                for y in 0..5 {
                    a[x][y] = b[x][y] ^ ((!b[(x + 1) % 5][y]) & b[(x + 2) % 5][y]);
                }
            }
            a[0][0] ^= ROUND_CONSTANTS[round];
        }
    }

    fn bit_state_to_lanes(s: &State) -> [[u64; 5]; 5] {
        let mut out = [[0u64; 5]; 5];
        for x in 0..5 {
            for y in 0..5 {
                let mut lane = 0u64;
                for z in 0..64 {
                    if s[state_idx(x, y, z)] {
                        lane |= 1u64 << z;
                    }
                }
                out[x][y] = lane;
            }
        }
        out
    }

    fn lanes_to_bit_state(lanes: &[[u64; 5]; 5]) -> State {
        let mut s = [false; STATE_BITS];
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    s[state_idx(x, y, z)] = (lanes[x][y] >> z) & 1 == 1;
                }
            }
        }
        s
    }

    // -------- Primitive tests.

    #[test]
    fn bit_keccak_f_matches_u64_lane() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..3 {
            let s = random_state(&mut rng);
            let mut s_bit = s;
            keccak_f(&mut s_bit);

            let mut lanes = bit_state_to_lanes(&s);
            keccak_f_u64(&mut lanes);
            let s_lane = lanes_to_bit_state(&lanes);

            assert_eq!(
                s_bit
                    .iter()
                    .zip(s_lane.iter())
                    .filter(|(a, b)| a != b)
                    .count(),
                0,
                "bit-level Keccak-f disagrees with u64-lane reference"
            );
        }
    }

    /// XOR'ing the named state_in bits equals the actual `(π ∘ ρ ∘ θ)(state_in)`
    /// bit at the target position.
    #[test]
    fn theta_rho_pi_preimage_matches_pipeline() {
        let mut rng = Rng::new(42);
        let state_in = random_state(&mut rng);
        let mut step = state_in;
        theta(&mut step);
        let b = rho_pi(&step);

        for z in 0..64 {
            for y in 0..5 {
                for x in 0..5 {
                    let preimage = theta_rho_pi_preimage(x, y, z);
                    let xor: bool = preimage.iter().fold(false, |acc, &i| acc ^ state_in[i]);
                    assert_eq!(
                        xor,
                        b[state_idx(x, y, z)],
                        "preimage mismatch at (x,y,z)=({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// All 11 preimage indices for any (x, y, z) are distinct.
    #[test]
    fn theta_rho_pi_preimage_has_no_duplicates() {
        for z in 0..64 {
            for y in 0..5 {
                for x in 0..5 {
                    let p = theta_rho_pi_preimage(x, y, z);
                    let mut sorted = p;
                    sorted.sort();
                    for w in sorted.windows(2) {
                        assert_ne!(
                            w[0], w[1],
                            "duplicate preimage bit at (x,y,z)=({x},{y},{z}): {sorted:?}"
                        );
                    }
                }
            }
        }
    }

    // -------- R1CS tests (I/O-aligned layout with state_24 pin).

    #[test]
    fn layout_constants_consistent() {
        assert_eq!(K, 65_536);
        assert_eq!(USEFUL_BITS, 4160 + 24 * 1600);
        assert_eq!(USEFUL_BITS, 42_560);
        assert!(USEFUL_BITS < K);
        assert_eq!(STATE0_BIT_BASE, 0);
        assert_eq!(STATE24_BIT_BASE, 2048);
        assert_eq!(Z_CONST, 4096);
        assert_eq!(T_PACKED_BIT_BASE, 4160);
        assert_eq!(U64_PER_BLOCK, 1024);
        assert_eq!(state_u64_base(0), 0);
        assert_eq!(state_u64_base(24), 32);
        assert_eq!(t_u64_base(0), 65);
        assert_eq!(t_u64_base(23), 65 + 23 * N_LANES);
    }

    #[test]
    fn witness_layout_round_trip() {
        let mut rng = Rng::new(0xCAB1E_FA17);
        let initial = random_state(&mut rng);

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_chain_witness_ab_packed_into(&initial, &mut z_u64, &mut a_u64, &mut b_u64);

        assert_eq!(z_u64[Z_CONST_U64] & 1, 1);

        let mut s0_lanes: Lanes = [0u64; 25];
        let s0_base = state_u64_base(0);
        for lane_idx in 0..N_LANES {
            s0_lanes[lane_idx] = z_u64[s0_base + lane_idx];
        }
        assert_eq!(lanes_to_state(&s0_lanes), initial);

        let mut s24_lanes: Lanes = [0u64; 25];
        let s24_base = state_u64_base(24);
        for lane_idx in 0..N_LANES {
            s24_lanes[lane_idx] = z_u64[s24_base + lane_idx];
        }
        let s24_recovered = lanes_to_state(&s24_lanes);
        let mut native = initial;
        keccak_f(&mut native);
        assert_eq!(s24_recovered, native, "state_24 != keccak_f(initial)");

        for pos in 0..U64_PER_BLOCK {
            assert_eq!(
                a_u64[pos] & b_u64[pos],
                z_u64[pos],
                "a·b ≠ z at u64 pos {pos}"
            );
        }
    }

    /// `comb · z = α · v_a + v_b` for random α and eq_inner.
    #[test]
    fn fold_matches_witness_consistency() {
        let mut rng = Rng::new(0xFA7E_C001_BABE);
        let initial = random_state(&mut rng);

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_chain_witness_ab_packed_into(&initial, &mut z_u64, &mut a_u64, &mut b_u64);

        let alpha = F128 {
            lo: rng.next_u64(),
            hi: rng.next_u64(),
        };
        let eq_inner: Vec<F128> = (0..K)
            .map(|_| F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            })
            .collect();

        let mut v_a = F128::ZERO;
        let mut v_b = F128::ZERO;
        for i in 0..K {
            let u = i / 64;
            let bit = i % 64;
            if (a_u64[u] >> bit) & 1 == 1 {
                v_a += eq_inner[i];
            }
            if (b_u64[u] >> bit) & 1 == 1 {
                v_b += eq_inner[i];
            }
        }
        let expected = alpha * v_a + v_b;

        let walker = KeccakLincheckCircuit;
        let comb = walker.fold_alpha_batched(alpha, &eq_inner);

        let mut got = F128::ZERO;
        for c in 0..K {
            let u = c / 64;
            let bit = c % 64;
            if (z_u64[u] >> bit) & 1 == 1 {
                got += comb[c];
            }
        }

        assert_eq!(got, expected, "fold consistency: comb·z ≠ α·v_a + v_b");
    }

    #[test]
    fn setup_sizes_correctly() {
        for &n in &[1usize, 8, 4096] {
            let setup = KeccakSetup::new(n);
            let expected_log = min_n_keccaks_log(n);
            assert_eq!(setup.n_keccaks_log(), expected_log);
            assert_eq!(setup.m(), K_LOG + expected_log);
            assert_eq!(setup.n_keccak_slots(), 1 << expected_log);
        }
    }

    /// End-to-end `prove_fast` → `verify` roundtrip (Ligerito default).
    /// K=64 (m=22) is the smallest size for which Ligerito's default config
    /// is feasible at log_batch_size=6.
    #[test]
    #[ignore]
    fn prove_fast_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = KeccakSetup::new(64);
        let mut rng = Rng::new(0x21111_2170);
        let inputs: Vec<State> = (0..64).map(|_| random_state(&mut rng)).collect();
        let mut ch_p = FsChallenger::new(b"flock-lig-keccak-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-lig-keccak-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("prove_fast: verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    /// Legacy BaseFold `prove_fast_basefold` → `verify_basefold` roundtrip at K=8.
    #[test]
    fn prove_fast_basefold_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let setup = KeccakSetup::new(8);
        let mut rng = Rng::new(0xFAB1E_F011);
        let inputs: Vec<State> = (0..8).map(|_| random_state(&mut rng)).collect();

        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, commitment, claim_p) = setup.prove_fast_basefold(&inputs, &mut ch_p);

        let mut ch_v = FsChallenger::new(b"flock-test-v0");
        let claim_v = setup
            .verify_basefold(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("prove_fast_basefold: verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    /// Constant-wire pin (docs/const-wire-pin.md). A non-power-of-two count gives
    /// padding blocks (filled with keccak_f(0), constant = 1) so the honest proof
    /// still verifies; the all-zero witness — encoding the FALSE keccak_f(0) = 0 —
    /// must be rejected by the pin.
    #[test]
    fn const_pin_all_zero_rejected() {
        use flock_core::challenger::FsChallenger;

        let n = 5; // < 8 slots ⇒ 3 padding blocks, exercises padding fill
        let setup = KeccakSetup::new(n);

        // (1) Honest proof with filled padding verifies.
        let mut rng = Rng::new(0x5EED_C0DE);
        let inputs: Vec<State> = (0..n).map(|_| random_state(&mut rng)).collect();
        let mut ch_p = FsChallenger::new(b"honest");
        let (proof, commitment, claim_p) = setup.prove_fast_basefold(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"honest");
        let claim_v = setup
            .verify_basefold(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("honest padded proof rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        // (2) All-zero witness must be rejected by the pin.
        let zeros: Vec<State> = vec![[false; STATE_BITS]; n];
        let (mut z, mut a, mut b, mut zlc) =
            generate_witness_with_ab_packed_and_lincheck(&zeros, setup.n_keccaks_log());
        z.iter_mut().for_each(|v| *v = F128::ZERO);
        a.iter_mut().for_each(|v| *v = F128::ZERO);
        b.iter_mut().for_each(|v| *v = F128::ZERO);
        zlc.iter_mut().for_each(|v| *v = 0);
        let mut ch_p = FsChallenger::new(b"poc");
        let (proof, commitment, _) = crate::prover::prove_fast_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            zlc,
            &KeccakLincheckCircuit,
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(b"poc");
        let res = setup.verify_basefold(&commitment, &proof, &mut ch_v);
        assert!(
            matches!(res, Err(flock_core::verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }

    /// Chain prove → verify roundtrip (Ligerito default). K=64 → m=22.
    #[test]
    #[ignore]
    fn prove_chain_roundtrip() {
        use flock_core::challenger::RandomChallenger;
        let setup = KeccakSetup::new(64);
        let n_inst = setup.n_keccak_slots();
        let mut rng = Rng::new(0xC0DE_5170);
        let x0 = random_state(&mut rng);
        let mut inputs = Vec::with_capacity(n_inst);
        let mut cur = x0;
        for _ in 0..n_inst {
            inputs.push(cur);
            keccak_f(&mut cur);
        }
        let x_last = cur;
        let mut chp = RandomChallenger::new(0xCA1);
        let (proof, comm) = setup.prove_chain(&inputs, &mut chp);
        let mut chv = RandomChallenger::new(0xCA1);
        setup
            .verify_chain(&comm, &proof, &x0, &x_last, &mut chv)
            .expect("chain must verify");
    }

    /// Comparison: proof size BaseFold vs Ligerito at m=22 (K=64 keccaks).
    /// Run with `cargo test ligerito_vs_basefold_size -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn ligerito_vs_basefold_size() {
        use flock_core::challenger::FsChallenger;
        let setup = KeccakSetup::new(64);
        let mut rng = Rng::new(0x517E_5717);
        let inputs: Vec<State> = (0..64).map(|_| random_state(&mut rng)).collect();

        let mut ch_p = FsChallenger::new(b"size-cmp");
        let (bf_proof, bf_commitment, _) = setup.prove_fast_basefold(&inputs, &mut ch_p);
        let bf_bundle = crate::proof_io::R1csProofBundle {
            commitment: bf_commitment,
            proof: bf_proof,
        };
        let bf_bytes = bf_bundle.to_bytes().len();

        let mut ch_p2 = FsChallenger::new(b"size-cmp");
        let (lig_proof, lig_commitment, _) = setup.prove_fast(&inputs, &mut ch_p2);
        let lig_bundle = crate::proof_io::R1csProofBundleLigerito {
            commitment: lig_commitment,
            proof: lig_proof,
        };
        let lig_bytes = lig_bundle.to_bytes().len();

        eprintln!("Keccak m=22 (K=64) proof size:");
        eprintln!(
            "  BaseFold:  {} bytes  ({:.1} KB)",
            bf_bytes,
            bf_bytes as f64 / 1024.0
        );
        eprintln!(
            "  Ligerito:  {} bytes  ({:.1} KB)",
            lig_bytes,
            lig_bytes as f64 / 1024.0
        );
        eprintln!(
            "  Ratio:     {:.2}× smaller",
            bf_bytes as f64 / lig_bytes as f64
        );
    }

    /// Legacy BaseFold chain roundtrip at K=8.
    #[test]
    fn prove_chain_basefold_roundtrip() {
        use flock_core::challenger::RandomChallenger;

        let setup = KeccakSetup::new(8);
        let n_inst = setup.n_keccak_slots();
        let mut rng = Rng::new(0xC0DE_C001);
        let x0 = random_state(&mut rng);
        let mut inputs = Vec::with_capacity(n_inst);
        let mut cur = x0;
        for _ in 0..n_inst {
            inputs.push(cur);
            keccak_f(&mut cur);
        }
        let x_last = cur;

        let mut chp = RandomChallenger::new(7);
        let (proof, comm) = setup.prove_chain_basefold(&inputs, &mut chp);

        let mut chv = RandomChallenger::new(7);
        setup
            .verify_chain_basefold(&comm, &proof, &x0, &x_last, &mut chv)
            .expect("honest chain must verify");
    }
}
