//! E1 workload — per-instance keccak witness builder.
//!
//! Faithful reimplementation of the private
//! `flock_prover::r1cs_hashes::keccak::build_chain_witness_ab_packed_into`
//! using that module's public primitives and layout constants. Kept in
//! lockstep by the `matches_public_driver` test (compares against the public
//! `generate_witness_with_ab_packed_and_lincheck`).

use flock_prover::r1cs_hashes::keccak::{
    K, LANE_BITS, Lanes, N_LANES, N_ROUNDS, N_T, STATE24_BIT_BASE, STATE0_BIT_BASE, State,
    T_PACKED_BIT_BASE, Z_CONST, iota_lanes, rho_pi_lanes, state_to_lanes, theta_lanes,
};

pub const U64_PER_BLOCK: usize = K / 64; // 1024
const Z_CONST_U64: usize = Z_CONST / LANE_BITS;

#[inline]
fn state_u64_base(r: usize) -> usize {
    match r {
        0 => STATE0_BIT_BASE / LANE_BITS,
        24 => STATE24_BIT_BASE / LANE_BITS,
        _ => unreachable!(),
    }
}

#[inline]
fn t_u64_base(r: usize) -> usize {
    (T_PACKED_BIT_BASE / LANE_BITS) + r * N_LANES
}

/// Fill one instance's `(z, a, b)` u64 buffers (each `U64_PER_BLOCK` long,
/// pre-zeroed) from the initial state. Mirrors keccak.rs's private builder.
pub fn build_block_witness(initial: &State, z_u64: &mut [u64], a_u64: &mut [u64], b_u64: &mut [u64]) {
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

    // state_24 pin rows: z = a = state_24, b = 1.
    let s24_base = state_u64_base(24);
    for lane_idx in 0..N_LANES {
        let pos = s24_base + lane_idx;
        let v = state_lanes[lane_idx];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }
    let _ = N_T;
}

/// Deterministic pseudo-random keccak state (splitmix64 over the 25 lanes).
pub fn random_state(seed: u64) -> State {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let lanes: Lanes = std::array::from_fn(|_| next());
    let mut st: State = [false; 1600];
    for (i, bit) in st.iter_mut().enumerate() {
        let lane = i % 25;
        let z_in_lane = i / 25;
        // state_to_lanes packs state_idx(x,y,z) = x + 5y + 25z with bit z of
        // lane (x,y); invert that mapping.
        *bit = (lanes[lane] >> z_in_lane) & 1 == 1;
    }
    st
}
