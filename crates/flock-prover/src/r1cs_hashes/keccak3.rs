//! 3-wide Keccak-f[1600] R1CS: packs **three independent** Keccak
//! permutations per block at **K_LOG = 17**, for tighter PCS utilization
//! (useful = 127,552 / 131,072 ≈ 97.3%, vs the single-keccak encoder's
//! 42,560 / 65,536 ≈ 65%).
//!
//! All Keccak-f primitives and the φ = θ∘ρ∘π transpose recurrence are reused
//! verbatim from [`super::keccak`]; only the per-block layout, witness
//! generation, and lincheck walker are widened to three sub-keccaks. The
//! three sub-keccaks share one constant slot but are otherwise three disjoint
//! copies of the single-keccak constraint set — there is no chaining between
//! them.
//!
//! ## Witness layout per block (k = 2^17 = 131,072 slots, 2048 u64 lanes)
//!
//! ```text
//!   bit 0      .. 1600    k0.state_0   (slot 0, pad 1600..2048 zero)
//!   bit 2048   .. 3648    k0.state_24  (slot 1)
//!   bit 4096   .. 5696    k1.state_0   (slot 2)
//!   bit 6144   .. 7744    k1.state_24  (slot 3)
//!   bit 8192   .. 9792    k2.state_0   (slot 4)
//!   bit 10240  .. 11840   k2.state_24  (slot 5)
//!   bit 12288             constant z = 1  (bit 0 of u64[192])
//!   bit 12352  .. 127552  t[i,r] for i∈0..3, r∈0..24 (tight, 1600 stride,
//!                         enumerated as i·24 + r)
//!   bit 127552 .. K       zero tail
//! ```
//!
//! Each `state_0`/`state_24` lives in a 2048-bit (`region_log = 11`) aligned
//! slot, mirroring the single encoder, so the 1600-bit state occupies 25 of
//! each slot's 32 u64 lanes.
//!
//! ## What the R1CS enforces (per sub-keccak i, shared C = I)
//!
//! - Row 0 (const, shared): `z[Z_CONST]·z[Z_CONST] = z[Z_CONST]`.
//! - `state_0[i]` input self-loops (1,600 rows): `z[row]·z[Z_CONST] = z[row]`.
//! - `state_24[i]` pin rows (1,600 rows): `L_24(state_0[i], t[i,<24])[j]
//!   · z[Z_CONST] = z[state_24[i] col j]`.
//! - 24 × 1,600 t-AND rows: `t[i,r] = (¬φ(state_r))[(x+1)%5] · φ(state_r)[(x+2)%5]`
//!   with `state_r` implicit via the φ substitution.
//! - Padding: empty A, B.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProof};
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier;

use super::keccak::{
    LANE_BITS, Lanes, N_LANES, N_ROUNDS, N_T, ROUND_CONSTANTS, STATE_BITS, STATE_SIZE_BITS, State,
    apply_phi_bool, apply_phi_t, iota_lanes, rho_pi_lanes, state_idx, state_to_lanes, theta_lanes,
    theta_rho_pi_preimage,
};

// ---------------------------------------------------------------------------
// Constants and layout
// ---------------------------------------------------------------------------

pub const K_LOG: usize = 17;
pub const K: usize = 1 << K_LOG; // 131,072

/// Univariate-skip dim (must match `zerocheck::K_SKIP`).
pub const K_SKIP: usize = 6;

/// Sub-keccaks packed per block.
pub const N_SUB: usize = 3;

/// 2^11 = 2048-bit aligned region for each state_0 / state_24.
pub const SLOT_BITS: usize = 2048;

/// Bit base of `state_0` for sub-keccak `i` (slot `2i`).
#[inline]
fn state0_bit_base(i: usize) -> usize {
    (2 * i) * SLOT_BITS
}
/// Bit base of `state_24` for sub-keccak `i` (slot `2i+1`).
#[inline]
fn state24_bit_base(i: usize) -> usize {
    (2 * i + 1) * SLOT_BITS
}

/// Constant slot — one lane past the 6 state slots.
pub const Z_CONST: usize = 2 * N_SUB * SLOT_BITS; // 12,288
const Z_CONST_U64: usize = Z_CONST / LANE_BITS; // 192
/// First tightly-packed t begins one lane past the const.
pub const T_PACKED_BIT_BASE: usize = Z_CONST + LANE_BITS; // 12,352

pub const USEFUL_BITS: usize = T_PACKED_BIT_BASE + N_SUB * N_T * STATE_SIZE_BITS; // 127,552

pub const U64_PER_BLOCK: usize = K / 64; // 2048

const _: () = {
    assert!(USEFUL_BITS <= K);
};

/// Within-sub-vector offset for logical bit `j` (lane-contiguous), identical
/// to the single encoder's mapping: `64·(j%25) + j/25`.
#[inline]
fn within_lane_contiguous(j: usize) -> usize {
    let lane_xy = j % N_LANES;
    let z_in_lane = j / N_LANES;
    LANE_BITS * lane_xy + z_in_lane
}

/// `u64`-lane base for state_r (r ∈ {0, 24}) of sub-keccak `i`.
#[inline]
fn state_u64_base(i: usize, r: usize) -> usize {
    match r {
        0 => state0_bit_base(i) / LANE_BITS,
        24 => state24_bit_base(i) / LANE_BITS,
        _ => panic!("only state_0 and state_24 are materialized, got r={r}"),
    }
}

/// Position of bit `j` of state_r (r ∈ {0, 24}) of sub-keccak `i`.
#[inline]
pub fn z_pos_state(i: usize, r: usize, j: usize) -> usize {
    debug_assert!(j < STATE_BITS);
    let base = match r {
        0 => state0_bit_base(i),
        24 => state24_bit_base(i),
        _ => panic!("only state_0 and state_24 are materialized, got r={r}"),
    };
    base + within_lane_contiguous(j)
}

/// `u64`-lane base for `t[i,r]`.
#[inline]
fn t_u64_base(i: usize, r: usize) -> usize {
    debug_assert!(i < N_SUB && r < N_T);
    (T_PACKED_BIT_BASE / LANE_BITS) + (i * N_T + r) * N_LANES
}

/// Position of bit `j` of `t[i,r]`.
#[inline]
pub fn z_pos_t(i: usize, r: usize, j: usize) -> usize {
    debug_assert!(i < N_SUB && r < N_T && j < STATE_BITS);
    T_PACKED_BIT_BASE + (i * N_T + r) * STATE_SIZE_BITS + within_lane_contiguous(j)
}

/// Minimum `n_blocks_log` (= outer dim) needed to prove `n_keccaks`
/// permutations packed 3-per-block, subject to the lincheck floor (≥ 3).
pub fn min_n_blocks_log(n_keccaks: usize) -> usize {
    assert!(n_keccaks >= 1, "n_keccaks must be ≥ 1");
    let n_blocks = n_keccaks.div_ceil(N_SUB);
    let n = n_blocks.max(8);
    (n.next_power_of_two().trailing_zeros() as usize).max(3)
}

pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    super::common::build_block_r1cs_empty_stub(n_blocks_log, K_LOG, K_SKIP, USEFUL_BITS)
}

// ---------------------------------------------------------------------------
// Witness generation
// ---------------------------------------------------------------------------

/// Fill the (z, a, b) lanes for one sub-keccak `i` from `initial`. The shared
/// constant lane is written once by [`build_block_witness_into`].
fn fill_subkeccak(
    i: usize,
    initial: &State,
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    // state_0 input self-loops.
    let mut state_lanes: Lanes = state_to_lanes(initial);
    let s0_base = state_u64_base(i, 0);
    for lane in 0..N_LANES {
        let pos = s0_base + lane;
        let v = state_lanes[lane];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }

    // 24 rounds: forward-simulate, write t[i,r] AND-row values.
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
        for k in 0..25 {
            next[k] = b_state[k] ^ t_lanes[k];
        }
        iota_lanes(&mut next, r);

        let t_base = t_u64_base(i, r);
        for y in 0..5 {
            for x in 0..5 {
                let lane = x + 5 * y;
                let pos = t_base + lane;
                z_u64[pos] = t_lanes[lane];
                a_u64[pos] = !b_state[(x + 1) % 5 + 5 * y];
                b_u64[pos] = b_state[(x + 2) % 5 + 5 * y];
            }
        }

        state_lanes = next;
    }

    // state_24 pin rows: z = a = state_24, b = 1 (B side is [Z_CONST]).
    let s24_base = state_u64_base(i, 24);
    for lane in 0..N_LANES {
        let pos = s24_base + lane;
        let v = state_lanes[lane];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }
}

fn build_block_witness_into(
    triple: &[State; N_SUB],
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    debug_assert_eq!(z_u64.len(), U64_PER_BLOCK);
    debug_assert_eq!(a_u64.len(), U64_PER_BLOCK);
    debug_assert_eq!(b_u64.len(), U64_PER_BLOCK);

    // Shared constant z[Z_CONST] = 1, a = b = 1.
    z_u64[Z_CONST_U64] = 1;
    a_u64[Z_CONST_U64] = 1;
    b_u64[Z_CONST_U64] = 1;

    for i in 0..N_SUB {
        fill_subkeccak(i, &triple[i], z_u64, a_u64, b_u64);
    }
    // Trailing padding stays zero.
}

/// Group a flat list of keccak inputs into 3-wide blocks (padding the final
/// block's missing sub-keccaks with the all-zero input state, whose witness
/// is a valid keccak computation) and drive the parallel packed witness build.
pub fn generate_witness_with_ab_packed_and_lincheck(
    initial_states: &[State],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    let n_blocks = initial_states.len().div_ceil(N_SUB);
    let zero: State = [false; STATE_BITS];
    let triples: Vec<[State; N_SUB]> = (0..n_blocks)
        .map(|blk| {
            let mut t = [zero; N_SUB];
            for (i, slot) in t.iter_mut().enumerate() {
                let idx = blk * N_SUB + i;
                if idx < initial_states.len() {
                    *slot = initial_states[idx];
                }
            }
            t
        })
        .collect();

    // Constant-wire pin (docs/const-wire-pin.md): every block — padding included
    // — must carry a valid keccak computation so its shared constant cell is 1.
    // A padding block is the all-zero input triple, whose witness is keccak_f(0)
    // (a genuine, satisfying computation), exactly as the partial last block
    // already pads its missing sub-keccaks.
    let padding: [State; N_SUB] = [zero; N_SUB];

    super::common::drive_witness_packed_and_lincheck(
        &triples,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        build_block_witness_into,
    )
}

// ---------------------------------------------------------------------------
// Lincheck circuit walker — three disjoint copies of the single-keccak
// transpose recurrence, accumulating into a shared `comb` and shared
// `Z_CONST` column. See [`super::keccak::KeccakLincheckCircuit`] for the
// per-sub-keccak derivation.
// ---------------------------------------------------------------------------

/// Accumulate sub-keccak `i`'s contribution to `comb` (everything except the
/// shared const self-loop, which [`KeccakLincheckCircuit::fold_alpha_batched`]
/// adds once).
fn accumulate_subkeccak(i: usize, alpha: F128, eq_inner: &[F128], comb: &mut [F128]) {
    // ---- state_0 input self-loops: A = [row], B = [Z_CONST].
    for j in 0..STATE_BITS {
        let row = z_pos_state(i, 0, j);
        let e = eq_inner[row];
        comb[row] += alpha * e;
        comb[Z_CONST] += e;
    }

    // ---- state_24 pin rows: A = L_24[j], B = [Z_CONST].
    let mut vec_pin: Vec<F128> = vec![F128::ZERO; STATE_BITS];
    let mut sum_eq_pin = F128::ZERO;
    for j in 0..STATE_BITS {
        let row = z_pos_state(i, 24, j);
        let e = eq_inner[row];
        vec_pin[j] = e;
        sum_eq_pin += e;
    }
    comb[Z_CONST] += sum_eq_pin; // B-side from pin's B = [Z_CONST]

    // ---- t-AND rows: per-round χ marginals on state_r positions.
    // Rounds are independent (each writes only chi_*[r]); F128 addition is
    // XOR (exactly associative/commutative), so the parallel reduction is
    // bit-identical to the serial loop.
    use rayon::prelude::*;
    let chi: Vec<(Vec<F128>, Vec<F128>, F128)> = (0..N_T)
        .into_par_iter()
        .map(|r| {
            let mut ca = vec![F128::ZERO; STATE_BITS];
            let mut cb = vec![F128::ZERO; STATE_BITS];
            let mut se = F128::ZERO;
            for zpos in 0..64 {
                for y in 0..5 {
                    for x in 0..5 {
                        let row = z_pos_t(i, r, state_idx(x, y, zpos));
                        let e = eq_inner[row];
                        se += e;
                        for &s in theta_rho_pi_preimage((x + 1) % 5, y, zpos).iter() {
                            ca[s] += e;
                        }
                        for &s in theta_rho_pi_preimage((x + 2) % 5, y, zpos).iter() {
                            cb[s] += e;
                        }
                    }
                }
            }
            (ca, cb, se)
        })
        .collect();
    let mut chi_a: Vec<Vec<F128>> = Vec::with_capacity(N_T);
    let mut chi_b: Vec<Vec<F128>> = Vec::with_capacity(N_T);
    let mut sum_eq_t = F128::ZERO;
    for (ca, cb, se) in chi {
        chi_a.push(ca);
        chi_b.push(cb);
        sum_eq_t += se;
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
    let t23_base = t_u64_base(i, N_T - 1);
    for s in 0..STATE_BITS {
        let pos = (t23_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
        comb[pos] += alpha * vec_pin[s];
    }
    let mut k_a = apply_phi_t(&vec_pin);
    for s in 0..STATE_BITS {
        k_a[s] += chi_a[N_T - 1][s];
    }
    for r in (1..N_T).rev() {
        let t_base = t_u64_base(i, r - 1);
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
    let s0_base = state_u64_base(i, 0);
    for s in 0..STATE_BITS {
        let pos = (s0_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
        comb[pos] += alpha * k_a[s];
    }

    // ---- Transpose recurrence, B side. K^B_24 = 0.
    let mut k_b = chi_b[N_T - 1].clone();
    for r in (1..N_T).rev() {
        let t_base = t_u64_base(i, r - 1);
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
}

pub struct KeccakLincheckCircuit;

impl LincheckCircuit for KeccakLincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    /// Pin the shared constant cell to 1 across all blocks. Requires the witness
    /// to fill padding blocks with valid keccak computations (constant = 1) — see
    /// [`generate_witness_with_ab_packed_and_lincheck`] and docs/const-wire-pin.md.
    fn const_pin_col(&self) -> Option<usize> {
        Some(Z_CONST)
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");

        // The three sub-keccaks are independent (disjoint column regions
        // apart from comb[Z_CONST], which accumulates) — run each into a
        // private comb and merge. F128 addition is XOR, so the regrouping
        // is bit-identical to the serial accumulation.
        let mut combs: Vec<Vec<F128>> = (0..N_SUB)
            .into_par_iter()
            .map(|i| {
                let mut comb = vec![F128::ZERO; K];
                accumulate_subkeccak(i, alpha, eq_inner, &mut comb);
                comb
            })
            .collect();

        let mut comb = combs.swap_remove(0);
        for other in &combs {
            comb.par_chunks_mut(1 << 13)
                .zip(other.par_chunks(1 << 13))
                .for_each(|(dst, src)| {
                    for (x, y) in dst.iter_mut().zip(src.iter()) {
                        *x += *y;
                    }
                });
        }

        // ---- Row 0 (const, shared): A = [Z_CONST], B = [Z_CONST].
        let e0 = eq_inner[Z_CONST];
        comb[Z_CONST] += alpha * e0;
        comb[Z_CONST] += e0;

        comb
    }
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
        let n_blocks_log = min_n_blocks_log(n_keccaks);
        let r1cs = build_block_r1cs(n_blocks_log);
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
    /// Outer (block) dimension log count = m − k_log.
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn n_block_slots(&self) -> usize {
        1usize << self.n_blocks_log()
    }

    pub fn prove_fast_basefold<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (R1csProof, Commitment, R1csClaim) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_blocks_log());
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

    /// Ligerito-backend prove. Requires m ≥ ~21.
    pub fn prove_fast<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_blocks_log())
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

    /// [`Self::prove_fast`] with a per-phase timing breakdown of the real
    /// Ligerito prover (witness gen + commit + zerocheck + lincheck + recursive
    /// open). Benchmark-only.
    pub fn prove_fast_timed<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        Commitment,
        R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let t0 = std::time::Instant::now();
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_blocks_log());
        let witness_s = t0.elapsed().as_secs_f64();
        let (proof, commitment, claim, mut timings) = crate::prover::prove_fast_ligerito_timed(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            &KeccakLincheckCircuit,
            None,
            challenger,
        );
        timings.witness_s = witness_s;
        (proof, commitment, claim, timings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::keccak::{keccak_f, lanes_to_state};

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

    #[test]
    fn layout_constants_consistent() {
        assert_eq!(K, 131_072);
        assert_eq!(USEFUL_BITS, 12_352 + 3 * 24 * 1600);
        assert_eq!(USEFUL_BITS, 127_552);
        assert!(USEFUL_BITS <= K);
        assert_eq!(Z_CONST, 12_288);
        assert_eq!(Z_CONST_U64, 192);
        assert_eq!(T_PACKED_BIT_BASE, 12_352);
        assert_eq!(U64_PER_BLOCK, 2048);
        // Sub-keccak state slots are disjoint and below the const lane.
        assert_eq!(state_u64_base(0, 0), 0);
        assert_eq!(state_u64_base(0, 24), 32);
        assert_eq!(state_u64_base(2, 24), 160);
        assert!(state_u64_base(2, 24) + N_LANES <= Z_CONST_U64);
        assert_eq!(t_u64_base(0, 0), 193);
        assert!(t_u64_base(N_SUB - 1, N_T - 1) + N_LANES <= U64_PER_BLOCK);
    }

    #[test]
    fn witness_layout_round_trip() {
        let mut rng = Rng::new(0xCAB1E_FA17);
        let triple: [State; N_SUB] = [
            random_state(&mut rng),
            random_state(&mut rng),
            random_state(&mut rng),
        ];

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_block_witness_into(&triple, &mut z_u64, &mut a_u64, &mut b_u64);

        assert_eq!(z_u64[Z_CONST_U64] & 1, 1);

        for i in 0..N_SUB {
            let mut s0_lanes: Lanes = [0u64; 25];
            let s0_base = state_u64_base(i, 0);
            for lane in 0..N_LANES {
                s0_lanes[lane] = z_u64[s0_base + lane];
            }
            assert_eq!(lanes_to_state(&s0_lanes), triple[i], "sub {i} state_0");

            let mut s24_lanes: Lanes = [0u64; 25];
            let s24_base = state_u64_base(i, 24);
            for lane in 0..N_LANES {
                s24_lanes[lane] = z_u64[s24_base + lane];
            }
            let mut native = triple[i];
            keccak_f(&mut native);
            assert_eq!(
                lanes_to_state(&s24_lanes),
                native,
                "sub {i} state_24 != keccak_f(state_0)"
            );
        }

        // a·b = z everywhere (the R1CS product holds bit-for-bit).
        for pos in 0..U64_PER_BLOCK {
            assert_eq!(
                a_u64[pos] & b_u64[pos],
                z_u64[pos],
                "a·b ≠ z at u64 pos {pos}"
            );
        }
    }

    /// The lincheck identity: `comb · z = α · (eq·a) + (eq·b)` for random α
    /// and eq_inner. This is the load-bearing correctness check on the walker.
    #[test]
    fn fold_matches_witness_consistency() {
        let mut rng = Rng::new(0xFA7E_C001_BABE);
        let triple: [State; N_SUB] = [
            random_state(&mut rng),
            random_state(&mut rng),
            random_state(&mut rng),
        ];

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_block_witness_into(&triple, &mut z_u64, &mut a_u64, &mut b_u64);

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

        let comb = KeccakLincheckCircuit.fold_alpha_batched(alpha, &eq_inner);

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
        // 25 keccaks → ceil(25/3)=9 blocks → pow2 16 → n_blocks_log=4.
        for &(n, blk_log) in &[(1usize, 3), (24, 3), (25, 4), (4096, 11)] {
            let setup = KeccakSetup::new(n);
            assert_eq!(setup.n_blocks_log(), blk_log, "n={n}");
            assert_eq!(setup.m(), K_LOG + blk_log, "n={n}");
        }
    }

    /// Head-to-head packing comparison vs the single-keccak encoder at
    /// `n = 6144 = 3·2048`, a sweet spot where the single encoder rounds up to
    /// m=29 (8192 slots) but 3-wide lands exactly on m=28 (2048 blocks) — a 2×
    /// smaller commitment. Run with:
    ///   cargo test --release -p flock keccak3::tests::pack_win -- --ignored --nocapture
    #[test]
    #[ignore = "timing comparison; run manually with --nocapture"]
    fn pack_win() {
        use crate::r1cs_hashes::keccak as k1;
        use flock_core::challenger::FsChallenger;
        use std::time::Instant;

        let n = 6144usize;
        let mut rng = Rng::new(0xC0FFEE_BEEF);
        let inputs: Vec<State> = (0..n).map(|_| random_state(&mut rng)).collect();

        let s1 = k1::KeccakSetup::new(n);
        let s3 = KeccakSetup::new(n);
        println!(
            "n={n}: single m={} (2^{} bits)  |  3-wide m={} (2^{} bits)",
            s1.m(),
            s1.m(),
            s3.m(),
            s3.m()
        );

        let mut t = Instant::now();
        let mut ch = FsChallenger::new(b"pack-bench");
        let (p1, _, _) = s1.prove_fast(&inputs, &mut ch);
        let single = t.elapsed().as_secs_f64();
        std::hint::black_box(&p1);

        t = Instant::now();
        let mut ch = FsChallenger::new(b"pack-bench");
        let (p3, _, _) = s3.prove_fast(&inputs, &mut ch);
        let wide = t.elapsed().as_secs_f64();
        std::hint::black_box(&p3);

        println!(
            "prove_fast: single={:.1} ms  3-wide={:.1} ms  ({:.2}x)",
            single * 1e3,
            wide * 1e3,
            single / wide
        );
    }

    /// End-to-end `prove_fast` → `verify` roundtrip, with a non-multiple-of-3
    /// count to exercise the trailing-triple zero-state padding.
    /// Ligerito-backend prove_fast roundtrip.
    #[test]
    #[ignore]
    fn prove_fast_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let n_keccaks = 49; // n_blocks_log = 5 → m = 22 with K_LOG=17
        let mut rng = Rng::new(0x21111_2170);
        let inputs: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();
        let setup = KeccakSetup::new(n_keccaks);
        let mut ch_p = FsChallenger::new(b"k3-lig-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"k3-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    #[ignore]
    fn prove_fast_timed_roundtrip() {
        // prove_fast_timed mirrors prove_fast with phase timers added, so its
        // proof must verify and yield the same claim, and every phase must
        // record a positive duration.
        use flock_core::challenger::FsChallenger;
        let n_keccaks = 49; // m = 22 (smallest Ligerito target)
        let mut rng = Rng::new(0x21111_2171);
        let inputs: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();
        let setup = KeccakSetup::new(n_keccaks);
        let mut ch_p = FsChallenger::new(b"k3-lig-v0");
        let (proof, commitment, claim_p, t) = setup.prove_fast_timed(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"k3-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("timed ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
        assert!(t.witness_s > 0.0 && t.commit_s > 0.0 && t.zerocheck_s > 0.0);
        assert!(t.lincheck_s > 0.0 && t.open_s > 0.0);
    }

    #[test]
    fn prove_fast_roundtrip() {
        use flock_core::challenger::FsChallenger;

        // n_keccaks = 49 → n_blocks_log = 5 → m = 22 (smallest Ligerito target).
        // Non-power-of-2 exercises the trailing-triple zero-state padding.
        let n_keccaks = 49;
        let mut rng = Rng::new(0xFAB1E_F011);
        let inputs: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();

        let setup = KeccakSetup::new(n_keccaks);
        let mut ch_p = FsChallenger::new(b"keccak3-test-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);

        let mut ch_v = FsChallenger::new(b"keccak3-test-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("prove_fast: verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    /// Regression for the all-zero soundness break (docs/const-wire-pin.md): the
    /// all-zero witness encodes the FALSE transition keccak_f([0;1600]) = [0;1600]
    /// (zero is not a fixed point). Before the constant-wire pin, the honest
    /// prover produced a proof that `KeccakSetup::verify` ACCEPTED. The pin
    /// (z[Z_CONST] = 1, folded into lincheck) must now reject it.
    #[test]
    fn all_zero_witness_rejected() {
        use crate::r1cs_hashes::keccak::keccak_f;
        use flock_core::challenger::FsChallenger;

        let n_keccaks = 49; // m = 22 (m22_fast)
        let setup = KeccakSetup::new(n_keccaks);

        // Zero is not a fixed point of keccak-f — so accepting this is unsound.
        let mut s0 = [false; STATE_BITS];
        keccak_f(&mut s0);
        assert!(s0.iter().any(|&b| b), "keccak_f(0) must be nonzero");

        // Correctly-shaped buffers, then zero EVERYTHING (incl. the constant
        // lane) to craft the all-zero witness the attacker would commit.
        let inputs: Vec<State> = vec![[false; STATE_BITS]; n_keccaks];
        let (mut z, mut a, mut b, mut zlc) =
            generate_witness_with_ab_packed_and_lincheck(&inputs, setup.n_blocks_log());
        z.iter_mut().for_each(|v| *v = F128::ZERO);
        a.iter_mut().for_each(|v| *v = F128::ZERO);
        b.iter_mut().for_each(|v| *v = F128::ZERO);
        zlc.iter_mut().for_each(|v| *v = 0);

        let mut ch_p = FsChallenger::new(b"poc");
        let (proof, commitment, _claim) = crate::prover::prove_fast_ligerito_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            zlc,
            &KeccakLincheckCircuit,
            None,
            &mut ch_p,
        );

        let mut ch_v = FsChallenger::new(b"poc");
        let res = setup.verify(&commitment, &proof, &mut ch_v);
        assert!(
            matches!(res, Err(verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }
}
