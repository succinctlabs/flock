//! **MiniSHAKE** (Keccak-p[800, 12]) per-permutation R1CS encoder — a
//! benchmark port of the retired Keccak-f[1600] walker encoder (removed in
//! 0f0d632) to the 800-bit, 12-round permutation of ePrint 2026/2363.
//!
//! MiniSHAKE128 = sponge over Keccak-p[800, 12] with capacity 256, so the
//! rate is 68 bytes: a 64-byte message (one 2-to-1 compression) plus its
//! domain/padding bytes fits ONE permutation call. This module proves batches
//! of that permutation; [`minishake128_block`] builds the absorbed state.
//!
//! ## Encoding
//!
//! One R1CS instance per permutation, all 12 rounds folded inline. Only
//! `state_0` (input), `state_12` (output) and the 12 χ AND-output vectors
//! `t_0..t_11` are materialized. With `φ = π∘ρ∘θ` and ι folded into a
//! constant, `state_{r+1} = φ(state_r) ⊕ t_r ⊕ RC_r`, so every `state_r` is a
//! linear function of `(state_0, t_<r, 1)`. The substituted A/B rows are
//! dense but never materialized: [`MiniShakeLincheckCircuit`] computes the
//! lincheck column marginal by the backward transpose recurrence
//! `K_r = φᵀ(K_{r+1}) ⊕ χ_r`.
//!
//! ## Witness layout per block (k = 2^14 = 16,384 bits)
//!
//! ```text
//!   bit 0     .. 800    state_0   (aligned 1024-bit slot 0; 800..1024 zero)
//!   bit 1024  .. 1824   state_12  (aligned 1024-bit slot 1; 1824..2048 zero)
//!   bit 2048            constant 1
//!   bit 2112  .. 11712  t_0 .. t_11 (800-bit stride, tight)
//!   bit 11712 .. K      zero tail
//! ```
//!
//! A state vector is lane-major: bit `z` of lane `L = x + 5y` is at offset
//! `32·L + z`, so a vector is the 25 lanes' `u32`s back to back.
//!
//! ## Rows (C = I)
//!
//! - const row: `1·1 = 1`.
//! - state_0 rows: `s·1 = s`.
//! - t rows: `t_r[x,y,z] = ¬B_r[x+1,y,z] · B_r[x+2,y,z]`, `B_r = φ(state_r)`.
//! - state_12 pin rows: `L_12(state_0, t_<12, 1)[j] · 1 = state_12[j]`.
//!
//! The table proves the permutation on an arbitrary `state_0`; it does not
//! constrain the capacity or padding of the absorbed block.

use flock_core::{
    lincheck::LincheckCircuit,
    pcs::{
        Commitment, PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    proof::{R1csClaim, R1csProofMergedLigerito},
    r1cs::{BlockR1cs, WitnessLayout},
    union::{SlotWitnessDest, UnionInstance},
    verifier::{FlockVerifyError, verify_ligerito_union},
};
use flock_field::F128;
use flock_transcript::challenger::Challenger;

use crate::{
    prover::{UnionSlotProverInput, prove_fast_ligerito_union},
    r1cs_hashes::common::{
        BM_V, BmRow, build_block_r1cs_with_matrices, drive_witness_batch_major_partial,
        drive_witness_batch_major_partial_into, empty_matrix, or_bit_row, or_u32_row,
    },
    schedule::{Registry, TableType},
};

// ===========================================================================
// Keccak-p[800, 12]
// ===========================================================================

pub const N_LANES: usize = 25;
pub const LANE_BITS: usize = 32;
pub const STATE_BITS: usize = N_LANES * LANE_BITS; // 800
pub const N_ROUNDS: usize = 12;
/// Univariate-skip width.
pub const K_SKIP: usize = 6;

/// ρ offsets `r[x][y]` (FIPS 202 Table 2); taken mod 32 for 32-bit lanes.
const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// ι constants of Keccak-p[800, 12]: rounds 10..21 of Keccak-f[800], i.e.
/// the low 32 bits of FIPS 202's RC[10..22].
pub const ROUND_CONSTANTS: [u32; N_ROUNDS] = [
    0x8000_8009,
    0x8000_000A,
    0x8000_808B,
    0x0000_008B,
    0x0000_8089,
    0x0000_8003,
    0x0000_8002,
    0x0000_0080,
    0x0000_800A,
    0x8000_000A,
    0x8000_8081,
    0x0000_8080,
];

/// 25 lanes, lane index `x + 5y`.
pub type Lanes = [u32; N_LANES];

/// State-bit index `32·(x + 5y) + z` (lane-major).
#[inline]
pub fn state_idx(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < 5 && y < 5 && z < LANE_BITS);
    LANE_BITS * (x + 5 * y) + z
}

#[inline]
fn rho(x: usize, y: usize) -> u32 {
    RHO_OFFSETS[x][y] % LANE_BITS as u32
}

/// The 11 `state_in` bits whose XOR is `B[x, y, z]`, `B = φ(state_in)`.
pub fn theta_rho_pi_preimage(x: usize, y: usize, z: usize) -> [usize; 11] {
    // π: B[x, y] = A_ρ[(x + 3y) mod 5, x]
    // ρ: A_ρ[a, b, c] = A_θ[a, b, (c − r[a][b]) mod 32]
    // θ: A_θ[a, b, c] = A[a, b, c] ⊕ ⊕_y' A[a−1, y', c] ⊕ ⊕_y' A[a+1, y', c−1]
    let a = (x + 3 * y) % 5;
    let b = x;
    let c = (z + LANE_BITS - rho(a, b) as usize) % LANE_BITS;
    let c_prev = (c + LANE_BITS - 1) % LANE_BITS;
    let mut bits = [0usize; 11];
    bits[0] = state_idx(a, b, c);
    for yp in 0..5 {
        bits[1 + yp] = state_idx((a + 4) % 5, yp, c);
        bits[6 + yp] = state_idx((a + 1) % 5, yp, c_prev);
    }
    bits
}

/// `φ = π∘ρ∘θ` on lanes (any lane-wise `u32` container via the closure-free
/// array form used by the V-wide witness generator).
#[inline(always)]
fn phi_v<const V: usize>(s: &[[u32; V]; N_LANES]) -> [[u32; V]; N_LANES] {
    let mut c = [[0u32; V]; 5];
    for x in 0..5 {
        for y in 0..5 {
            for j in 0..V {
                c[x][j] ^= s[x + 5 * y][j];
            }
        }
    }
    let mut d = [[0u32; V]; 5];
    for x in 0..5 {
        for j in 0..V {
            d[x][j] = c[(x + 4) % 5][j] ^ c[(x + 1) % 5][j].rotate_left(1);
        }
    }
    let mut out = [[0u32; V]; N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            let (a, b) = ((x + 3 * y) % 5, x);
            let r = rho(a, b);
            for j in 0..V {
                out[x + 5 * y][j] = (s[a + 5 * b][j] ^ d[a][j]).rotate_left(r);
            }
        }
    }
    out
}

/// χ AND outputs `t[x,y] = ¬B[x+1,y] ∧ B[x+2,y]`.
#[inline(always)]
fn chi_t_v<const V: usize>(bs: &[[u32; V]; N_LANES]) -> [[u32; V]; N_LANES] {
    let mut t = [[0u32; V]; N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            for j in 0..V {
                t[x + 5 * y][j] = !bs[(x + 1) % 5 + 5 * y][j] & bs[(x + 2) % 5 + 5 * y][j];
            }
        }
    }
    t
}

/// One round: `state ← φ(state) ⊕ t ⊕ RC_r`. Returns `(B_r, t_r)`.
#[inline(always)]
#[allow(clippy::type_complexity)]
fn round_v<const V: usize>(
    s: &mut [[u32; V]; N_LANES],
    r: usize,
) -> ([[u32; V]; N_LANES], [[u32; V]; N_LANES]) {
    let bs = phi_v(s);
    let t = chi_t_v(&bs);
    for i in 0..N_LANES {
        for j in 0..V {
            s[i][j] = bs[i][j] ^ t[i][j];
        }
    }
    for j in 0..V {
        s[0][j] ^= ROUND_CONSTANTS[r];
    }
    (bs, t)
}

/// Keccak-p[800, 12] in place.
pub fn keccak_p800_12(lanes: &mut Lanes) {
    let mut s: [[u32; 1]; N_LANES] = std::array::from_fn(|i| [lanes[i]]);
    for r in 0..N_ROUNDS {
        round_v(&mut s, r);
    }
    for i in 0..N_LANES {
        lanes[i] = s[i][0];
    }
}

/// MiniSHAKE128 rate in bytes (`100 − 256/8`).
pub const MINISHAKE128_RATE: usize = 68;

/// The absorbed state of MiniSHAKE128 on a 64-byte message with domain byte
/// `d` (one block: `M ‖ d ‖ 0x00 ‖ 0x00 ‖ (0x00 ⊕ 0x80)`, capacity zero).
/// Permuting it once gives the output; the digest is its first 32 bytes.
pub fn minishake128_block(msg: &[u8; 64], d: u8) -> Lanes {
    assert!(
        (0x01..=0x7F).contains(&d),
        "domain byte must be in 0x01..=0x7F"
    );
    let mut bytes = [0u8; 100];
    bytes[..64].copy_from_slice(msg);
    bytes[64] = d;
    bytes[MINISHAKE128_RATE - 1] ^= 0x80;
    std::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}

/// MiniSHAKE[c](M, D), `out_len` output bytes (paper §1, generic reference).
pub fn minishake(c_bits: usize, msg: &[u8], d: u8, out_len: usize) -> Vec<u8> {
    assert!(c_bits.is_multiple_of(8) && c_bits <= 512);
    assert!((0x01..=0x7F).contains(&d));
    let rate = 100 - c_bits / 8;
    let mut m = msg.to_vec();
    m.push(d);
    while !m.len().is_multiple_of(rate) {
        m.push(0);
    }
    *m.last_mut().unwrap() ^= 0x80;
    let mut st = [0u8; 100];
    let permute = |st: &mut [u8; 100]| {
        let mut l: Lanes =
            std::array::from_fn(|i| u32::from_le_bytes(st[4 * i..4 * i + 4].try_into().unwrap()));
        keccak_p800_12(&mut l);
        for i in 0..N_LANES {
            st[4 * i..4 * i + 4].copy_from_slice(&l[i].to_le_bytes());
        }
    };
    for block in m.chunks(rate) {
        for (s, b) in st.iter_mut().zip(block) {
            *s ^= b;
        }
        permute(&mut st);
    }
    let mut out = Vec::with_capacity(out_len);
    loop {
        let take = rate.min(out_len - out.len());
        out.extend_from_slice(&st[..take]);
        if out.len() == out_len {
            return out;
        }
        permute(&mut st);
    }
}

// ===========================================================================
// Layout
// ===========================================================================

pub const K_LOG: usize = 14;
pub const K: usize = 1 << K_LOG;
pub const SLOT_BITS: usize = 1024;
pub const STATE0_BASE: usize = 0;
pub const STATE_OUT_BASE: usize = SLOT_BITS;
pub const Z_CONST: usize = 2 * SLOT_BITS; // 2048
pub const T_BASE: usize = Z_CONST + 64; // 2112
pub const USEFUL_BITS: usize = T_BASE + N_ROUNDS * STATE_BITS; // 11,712
const _: () = assert!(USEFUL_BITS <= K);

#[inline]
pub fn z_pos_state0(s: usize) -> usize {
    STATE0_BASE + s
}
#[inline]
pub fn z_pos_state_out(s: usize) -> usize {
    STATE_OUT_BASE + s
}
#[inline]
pub fn z_pos_t(r: usize, s: usize) -> usize {
    T_BASE + r * STATE_BITS + s
}

/// Smallest `n_log` (≥ 3) holding `n` permutations.
pub fn min_n_perms_log(n: usize) -> usize {
    assert!(n >= 1);
    n.max(8).next_power_of_two().trailing_zeros() as usize
}

/// `out[s_in] = Σ_{s_out : s_in ∈ preim(s_out)} v[s_out]` (φᵀ).
fn apply_phi_t(v: &[F128]) -> Vec<F128> {
    let mut out = vec![F128::ZERO; STATE_BITS];
    for (s_out, &val) in v.iter().enumerate() {
        let (lane, z) = (s_out / LANE_BITS, s_out % LANE_BITS);
        for s_in in theta_rho_pi_preimage(lane % 5, lane / 5, z) {
            out[s_in] += val;
        }
    }
    out
}

/// φ on a bit vector (tracks the round-constant accumulator).
fn apply_phi_bool(v: &[bool; STATE_BITS]) -> [bool; STATE_BITS] {
    std::array::from_fn(|s_out| {
        let (lane, z) = (s_out / LANE_BITS, s_out % LANE_BITS);
        theta_rho_pi_preimage(lane % 5, lane / 5, z)
            .iter()
            .fold(false, |acc, &s| acc ^ v[s])
    })
}

pub fn build_block_r1cs(n_log: usize) -> BlockR1cs {
    let mut r1cs = build_block_r1cs_with_matrices(
        n_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        empty_matrix(K),
        empty_matrix(K),
        Some(Z_CONST),
    );
    r1cs.layout = WitnessLayout::BatchMajor;
    r1cs
}

// ===========================================================================
// Witness generation (batch-major, V = 8 instances in lockstep)
// ===========================================================================

type VLanes = [[u32; BM_V]; N_LANES];

fn emit_vec(rows: &mut [BmRow], base: usize, v: &VLanes) {
    for (l, vals) in v.iter().enumerate() {
        or_u32_row(rows, base + LANE_BITS * l, vals);
    }
}

fn build_group(group: [&Lanes; BM_V], rz: &mut [BmRow], ra: &mut [BmRow], rb: &mut [BmRow]) {
    let ones: VLanes = [[u32::MAX; BM_V]; N_LANES];
    for rows in [&mut *rz, &mut *ra, &mut *rb] {
        or_bit_row(rows, Z_CONST);
    }
    let mut s: VLanes = std::array::from_fn(|i| std::array::from_fn(|j| group[j][i]));
    emit_vec(rz, STATE0_BASE, &s);
    emit_vec(ra, STATE0_BASE, &s);
    emit_vec(rb, STATE0_BASE, &ones);
    for r in 0..N_ROUNDS {
        let (bs, t) = round_v(&mut s, r);
        let base = T_BASE + r * STATE_BITS;
        let not_b1: VLanes = std::array::from_fn(|i| {
            let (x, y) = (i % 5, i / 5);
            bs[(x + 1) % 5 + 5 * y].map(|v| !v)
        });
        let b2: VLanes = std::array::from_fn(|i| {
            let (x, y) = (i % 5, i / 5);
            bs[(x + 2) % 5 + 5 * y]
        });
        emit_vec(rz, base, &t);
        emit_vec(ra, base, &not_b1);
        emit_vec(rb, base, &b2);
    }
    emit_vec(rz, STATE_OUT_BASE, &s);
    emit_vec(ra, STATE_OUT_BASE, &s);
    emit_vec(rb, STATE_OUT_BASE, &ones);
}

/// `(z, a, b, stripe)` for `inputs` (dummy rows past `inputs.len()` zero).
pub fn generate_witness_batch_major_partial(
    inputs: &[Lanes],
    n_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    drive_witness_batch_major_partial(inputs, n_log, K_LOG, USEFUL_BITS, build_group)
}

/// In-place counterpart of [`generate_witness_batch_major_partial`].
pub fn generate_witness_batch_major_partial_into(
    inputs: &[Lanes],
    n_log: usize,
    dst: SlotWitnessDest<'_>,
) -> Vec<u8> {
    drive_witness_batch_major_partial_into(inputs, n_log, K_LOG, USEFUL_BITS, dst, build_group)
}

// ===========================================================================
// Lincheck walker
// ===========================================================================

/// Column marginals of the substituted A/B rows (see the module docs):
///
/// - A side: `K^A_12 = pin weights`; `K^A_r = φᵀ(K^A_{r+1}) ⊕ χ^A_r`;
///   `K^A_r` lands on `t_{r−1}` (and `K^A_0` on `state_0`).
/// - B side: same recurrence from `K^B_12 = 0` (pin rows' B is the constant).
/// - Constant column: the `¬` in every t row's A, the B = 1 of the state
///   rows, and the round-constant accumulator `RC_r` dotted with each
///   round's χ marginals and the pin weights.
pub struct MiniShakeLincheckCircuit;

impl LincheckCircuit for MiniShakeLincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(Z_CONST)
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), K);
        let mut comb = vec![F128::ZERO; K];

        // Const row: A = B = [Z_CONST].
        let e0 = eq_inner[Z_CONST];
        comb[Z_CONST] += alpha * e0 + e0;

        // state_0 rows: A = [row], B = [Z_CONST].
        for s in 0..STATE_BITS {
            let row = z_pos_state0(s);
            comb[row] += alpha * eq_inner[row];
            comb[Z_CONST] += eq_inner[row];
        }

        // Pin rows: A = L_12[s], B = [Z_CONST].
        let vec_pin: Vec<F128> = (0..STATE_BITS)
            .map(|s| eq_inner[z_pos_state_out(s)])
            .collect();
        let sum_pin = vec_pin.iter().fold(F128::ZERO, |acc, &e| acc + e);
        comb[Z_CONST] += sum_pin;

        // χ marginals on state_r coordinates.
        let mut chi_a = vec![vec![F128::ZERO; STATE_BITS]; N_ROUNDS];
        let mut chi_b = vec![vec![F128::ZERO; STATE_BITS]; N_ROUNDS];
        let mut sum_t = F128::ZERO;
        for r in 0..N_ROUNDS {
            for y in 0..5 {
                for x in 0..5 {
                    for z in 0..LANE_BITS {
                        let e = eq_inner[z_pos_t(r, state_idx(x, y, z))];
                        sum_t += e;
                        for s in theta_rho_pi_preimage((x + 1) % 5, y, z) {
                            chi_a[r][s] += e;
                        }
                        for s in theta_rho_pi_preimage((x + 2) % 5, y, z) {
                            chi_b[r][s] += e;
                        }
                    }
                }
            }
        }
        comb[Z_CONST] += alpha * sum_t; // the ¬ in A = 1 ⊕ B_r[x+1]

        // Constant part of state_r: rc_0 = 0, rc_{r+1} = φ(rc_r) ⊕ RC_r.
        let mut rc = [false; STATE_BITS];
        let (mut rc_a, mut rc_b) = (F128::ZERO, F128::ZERO);
        for r in 0..N_ROUNDS {
            for s in 0..STATE_BITS {
                if rc[s] {
                    rc_a += chi_a[r][s];
                    rc_b += chi_b[r][s];
                }
            }
            rc = apply_phi_bool(&rc);
            for z in 0..LANE_BITS {
                if (ROUND_CONSTANTS[r] >> z) & 1 == 1 {
                    rc[state_idx(0, 0, z)] ^= true;
                }
            }
        }
        let rc_pin = (0..STATE_BITS)
            .filter(|&s| rc[s])
            .fold(F128::ZERO, |acc, s| acc + vec_pin[s]);
        comb[Z_CONST] += alpha * (rc_a + rc_pin) + rc_b;

        // Transpose recurrences. state_r = φ(state_{r−1}) ⊕ t_{r−1} ⊕ RC, so
        // the weight K_r on state_r lands on t_{r−1} and K_{r−1} =
        // φᵀ(K_r) ⊕ χ_{r−1}.
        let mut k_a = vec_pin;
        let mut k_b = vec![F128::ZERO; STATE_BITS];
        for r in (1..=N_ROUNDS).rev() {
            for s in 0..STATE_BITS {
                let pos = z_pos_t(r - 1, s);
                comb[pos] += alpha * k_a[s] + k_b[s];
            }
            let mut na = apply_phi_t(&k_a);
            let mut nb = apply_phi_t(&k_b);
            for s in 0..STATE_BITS {
                na[s] += chi_a[r - 1][s];
                nb[s] += chi_b[r - 1][s];
            }
            k_a = na;
            k_b = nb;
        }
        for s in 0..STATE_BITS {
            comb[z_pos_state0(s)] += alpha * k_a[s] + k_b[s];
        }
        comb
    }
}

// ===========================================================================
// Setup
// ===========================================================================

/// Keccak-p[800, 12] R1CS + its single-slot union registry.
#[derive(Debug)]
pub struct MiniShakeSetup {
    pub n_perms: usize,
    pub r1cs: BlockR1cs,
    pub registry: Registry,
    pub pcs_params: PcsParams,
}

impl MiniShakeSetup {
    pub fn new(n_perms: usize) -> Self {
        Self::with_profile(n_perms, LigeritoProfile::Fast)
    }

    pub fn with_profile(n_perms: usize, profile: LigeritoProfile) -> Self {
        let n_log = min_n_perms_log(n_perms);
        let r1cs = build_block_r1cs(n_log);
        flock_core::scratch::prewarm_prover(r1cs.m);
        let registry = Registry::new(vec![TableType::from_block_r1cs(&r1cs)], n_log);
        let _ = registry.digest();
        let pcs_params = {
            let union = UnionInstance::new(&registry, vec![n_perms]);
            let m = union.dense_m();
            let batch = embedded_initial_k_or_default(m, profile);
            PcsParams {
                m,
                log_inv_rate: profile.log_inv_rate(),
                log_batch_size: batch,
                profile,
                num_lanes: union.commit_lanes(batch),
                merkle_hash: Default::default(),
            }
        };
        Self {
            n_perms,
            r1cs,
            registry,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }

    fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, vec![self.n_perms])
    }

    /// Prove `n_perms` permutations of the given input states (RS zerocheck).
    pub fn prove_fast<Ch: Challenger>(
        &self,
        inputs: &[Lanes],
        challenger: &mut Ch,
    ) -> (R1csProofMergedLigerito, Commitment, R1csClaim) {
        assert_eq!(inputs.len(), self.n_perms);
        let nu = self.n_log();
        let slot = UnionSlotProverInput::in_place(
            move |dst| generate_witness_batch_major_partial_into(inputs, nu, dst),
            &MiniShakeLincheckCircuit,
        );
        prove_fast_ligerito_union(&self.union(), &self.pcs_params, vec![slot], challenger)
    }

    /// [`Self::prove_fast`] with the AG-skip boolean zerocheck (aarch64).
    #[cfg(target_arch = "aarch64")]
    pub fn prove_fast_union_ag<Ch: Challenger>(
        &self,
        inputs: &[Lanes],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofMergedLigeritoAg,
        Commitment,
        R1csClaim,
    ) {
        assert_eq!(inputs.len(), self.n_perms);
        let nu = self.n_log();
        let slot = UnionSlotProverInput::in_place(
            move |dst| generate_witness_batch_major_partial_into(inputs, nu, dst),
            &MiniShakeLincheckCircuit,
        );
        crate::prover::prove_fast_ligerito_union_ag(
            &self.union(),
            &self.pcs_params,
            vec![slot],
            challenger,
        )
    }

    pub fn verify<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &R1csProofMergedLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, FlockVerifyError> {
        let circs: [&dyn LincheckCircuit; 1] = [&MiniShakeLincheckCircuit];
        verify_ligerito_union(
            &self.union(),
            &circs,
            commitment,
            proof,
            &self.pcs_params,
            challenger,
        )
    }

    pub fn verify_union_ag<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &flock_core::proof::R1csProofMergedLigeritoAg,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, FlockVerifyError> {
        let circs: [&dyn LincheckCircuit; 1] = [&MiniShakeLincheckCircuit];
        flock_core::verifier::verify_ligerito_union_ag(
            &self.union(),
            &circs,
            commitment,
            proof,
            &self.pcs_params,
            challenger,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::test_rng::Rng;

    fn random_lanes(rng: &mut Rng) -> Lanes {
        std::array::from_fn(|_| rng.next_u32())
    }

    #[test]
    fn permutation_matches_keccak_crate() {
        let mut rng = Rng::new(1);
        for _ in 0..64 {
            let mut ours = random_lanes(&mut rng);
            let mut theirs = ours;
            keccak_p800_12(&mut ours);
            keccak::p800(&mut theirs, N_ROUNDS);
            assert_eq!(ours, theirs);
        }
    }

    #[test]
    fn one_block_state_matches_generic_sponge() {
        let mut rng = Rng::new(2);
        let msg: [u8; 64] = std::array::from_fn(|_| rng.next_u32() as u8);
        let mut s = minishake128_block(&msg, 0x01);
        keccak_p800_12(&mut s);
        let digest: Vec<u8> = s.iter().flat_map(|l| l.to_le_bytes()).take(32).collect();
        assert_eq!(digest, minishake(256, &msg, 0x01, 32));
    }

    #[test]
    fn preimage_matches_phi() {
        let mut rng = Rng::new(3);
        let s = random_lanes(&mut rng);
        let bs = phi_v(&s.map(|v| [v]));
        let bit = |l: &Lanes, i: usize| (l[i / LANE_BITS] >> (i % LANE_BITS)) & 1 == 1;
        for y in 0..5 {
            for x in 0..5 {
                for z in 0..LANE_BITS {
                    let want = (bs[x + 5 * y][0] >> z) & 1 == 1;
                    let got = theta_rho_pi_preimage(x, y, z)
                        .iter()
                        .fold(false, |acc, &i| acc ^ bit(&s, i));
                    assert_eq!(got, want);
                }
            }
        }
    }

    /// Unpack slot `i`'s block from a batch-major buffer
    /// (`addr = [7 in-word | n_log batch | k_log−7 chunk]`).
    fn block_bits(buf: &[F128], n_log: usize, i: usize) -> Vec<bool> {
        let words: &[u64] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u64, buf.len() * 2) };
        (0..K)
            .map(|bit| {
                let chunk = bit / 128;
                let f = (chunk << n_log) + i;
                let w = 2 * f + (bit % 128) / 64;
                (words[w] >> (bit % 64)) & 1 == 1
            })
            .collect()
    }

    /// For every block: rows satisfy `a·b = z`, the output slot holds the
    /// permutation, and the walker's comb vector reproduces
    /// `Σ_row eq[row]·(α·a[row] + b[row])` — i.e. `a = A·z`, `b = B·z`.
    #[test]
    fn witness_and_walker_are_consistent() {
        let n_log = 3;
        let mut rng = Rng::new(4);
        let inputs: Vec<Lanes> = (0..7).map(|_| random_lanes(&mut rng)).collect();
        let (z, a, b, _) = generate_witness_batch_major_partial(&inputs, n_log);
        let alpha = F128::new(rng.next_u64(), rng.next_u64());
        let eq: Vec<F128> = (0..K)
            .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
            .collect();
        let comb = MiniShakeLincheckCircuit.fold_alpha_batched(alpha, &eq);
        for i in 0..8 {
            let (zb, ab, bb) = (
                block_bits(&z, n_log, i),
                block_bits(&a, n_log, i),
                block_bits(&b, n_log, i),
            );
            for row in 0..K {
                assert_eq!(ab[row] & bb[row], zb[row], "block {i} row {row}");
            }
            if let Some(input) = inputs.get(i) {
                let mut want = *input;
                keccak_p800_12(&mut want);
                for s in 0..STATE_BITS {
                    let bit = (want[s / LANE_BITS] >> (s % LANE_BITS)) & 1 == 1;
                    assert_eq!(zb[z_pos_state_out(s)], bit);
                }
            } else {
                assert!(zb.iter().all(|&v| !v), "dummy block must be zero");
            }
            let lhs = (0..K)
                .filter(|&c| zb[c])
                .fold(F128::ZERO, |acc, c| acc + comb[c]);
            let rhs = (0..K).fold(F128::ZERO, |acc, row| {
                let mut v = F128::ZERO;
                if ab[row] {
                    v += alpha;
                }
                if bb[row] {
                    v += F128::ONE;
                }
                acc + eq[row] * v
            });
            assert_eq!(lhs, rhs, "block {i}");
        }
    }

    #[test]
    #[ignore = "m = 22 prove; run with --ignored"]
    fn prove_verify_roundtrip() {
        use crate::challenger::FsChallenger;
        let n = 256;
        let setup = MiniShakeSetup::new(n);
        let mut rng = Rng::new(5);
        let inputs: Vec<Lanes> = (0..n).map(|_| random_lanes(&mut rng)).collect();
        let mut ch = FsChallenger::new(b"minishake-test");
        let (proof, commitment, _) = setup.prove_fast(&inputs, &mut ch);
        let mut ch = FsChallenger::new(b"minishake-test");
        setup
            .verify(&commitment, &proof, &mut ch)
            .expect("verify failed");
    }
}
