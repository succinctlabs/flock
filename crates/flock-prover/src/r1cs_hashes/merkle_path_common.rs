//! Generic Merkle-path glue, analogous to [`super::chain_common`].
//!
//! The Merkle-path protocol (packed-pos fold → shift-with-bit-selector sumcheck
//! → batched PCS open over `[ab, c] ring-switched + [merkle] packed-direct`)
//! is hash-agnostic; only the *geometry* of where the left-input / right-input
//! / output regions live in a witness block varies. This module captures that
//! geometry in [`MerkleLayout`] and provides
//! [`prove_merkle_path_generic`] / [`verify_merkle_path_generic`]. A per-hash
//! module supplies its `MerkleLayout`, a `Hash → physical-bits` converter for
//! the public leaf/root, and thin wrappers.
//!
//! ## Region requirements
//!
//! The Merkle protocol carves out a **4-slot region** at the start of each
//! witness block: 4 consecutive aligned `2^region_log`-bit slots holding
//! (Z, X_L, X_R, other) in some order specified by the layout. The fourth
//! "other" slot holds whatever the per-hash R1CS uses there (e.g. SHA-2's
//! IV) — the protocol's weight is structurally zero at that slot's boolean
//! coords, so its contents are invisible to the sumcheck but participate in
//! the multilinear extension over the slot-selector dimensions.

use crate::merkle_path::{MerklePathShiftProof, SlotLayout};
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::pcs::{
    Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
};
use flock_core::r1cs::BlockR1cs;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// Geometry of one hash's 4-slot region within a witness block.
#[derive(Clone, Copy, Debug)]
pub struct MerkleLayout {
    /// `log2` of the per-block witness size.
    pub k_log: usize,
    /// Univariate-skip dimension (API parity with the zerocheck/PCS `k_skip`).
    pub k_skip: usize,
    /// `log2` of each slot's aligned size.
    pub region_log: usize,
    /// Real bits per slot (≤ `2^region_log`, multiple of 8).
    pub region_bits: usize,
    /// Byte offset of the 4-slot region's start within a block. Slot `s` lives
    /// at bytes `[slot_base_byte_off + s · slot_size_bytes,
    /// slot_base_byte_off + (s+1) · slot_size_bytes)` where
    /// `slot_size_bytes = 2^region_log / 8`.
    pub slot_base_byte_off: usize,
    /// Slot index `0..4` (LSB-first: `s = sel_slot | (side << 1)`) holding Z.
    pub z_slot: u8,
    /// Slot index `0..4` holding X_L.
    pub x_l_slot: u8,
    /// Slot index `0..4` holding X_R.
    pub x_r_slot: u8,
}

impl MerkleLayout {
    /// `τ_pos` length = `region_log − LOG_PACKING`.
    #[inline]
    pub fn tau_pos_len(&self) -> usize {
        self.region_log - LOG_PACKING
    }

    /// Number of zero coords in the merkle claim's point between the side
    /// selector and the instance index. The protocol pins the high slot-bits
    /// of the claim point to 0 (they encode "we're inside the first 4 slots
    /// of the block, not the rest").
    ///
    /// `high_zeros = k_log − region_log − 2` (the `−2` accounts for the
    /// (sel_slot, side) bits, each handling one of the 4-slot region's
    /// selector bits).
    #[inline]
    pub fn high_zeros(&self) -> usize {
        self.k_log - self.region_log - 2
    }

    /// Slot index of the "other" slot — the one not assigned to Z, X_L, or
    /// X_R. Its contents are committed but unconstrained by the protocol.
    pub fn other_slot(&self) -> u8 {
        (0..4)
            .find(|&s| s != self.z_slot && s != self.x_l_slot && s != self.x_r_slot)
            .expect("3 of 4 slots are assigned; the fourth always exists")
    }

    /// Convert this layout's slot assignment to the bare `SlotLayout` used by
    /// the merkle-path sumcheck driver.
    pub fn slot_layout(&self) -> SlotLayout {
        SlotLayout {
            z_slot: self.z_slot,
            x_l_slot: self.x_l_slot,
            x_r_slot: self.x_r_slot,
        }
    }
}

// ---------------------------------------------------------------------------
// Fold parameters
// ---------------------------------------------------------------------------

/// Packed-level fold parameters: `τ_pos` binds the packed-position dimension
/// within each slot. The verifier samples `τ_pos`, then the prover folds each
/// instance's 4 slots down to one `F128` apiece via
/// `Σ_{pos} eq(τ_pos, pos) · ẑ_packed[(inst, slot, pos)]`.
#[derive(Clone, Debug)]
pub struct MerklePathFold {
    pub tau_pos: Vec<F128>,
}

impl MerklePathFold {
    pub fn new(layout: &MerkleLayout, tau_pos: Vec<F128>) -> Self {
        assert_eq!(
            tau_pos.len(),
            layout.tau_pos_len(),
            "τ_pos length must be region_log − LOG_PACKING"
        );
        Self { tau_pos }
    }

    /// Fold a public k-bit endpoint (given as `region_bits` bools in physical
    /// within-slot order) to a single F128 — the τ_pos-MLE of the endpoint
    /// over its slot's packed positions. Mirrors what the prover computes
    /// against the committed witness.
    pub fn fold_public_phys(&self, phys_bits: &[bool]) -> F128 {
        let bits_per_packed = 1usize << LOG_PACKING; // 128
        let n_packed = 1usize << self.tau_pos.len();
        let slot_bits = n_packed * bits_per_packed;
        assert!(
            phys_bits.len() <= slot_bits,
            "fold_public_phys: phys_bits length {} > slot bits {}",
            phys_bits.len(),
            slot_bits,
        );
        let eq_tau = build_eq_table(&self.tau_pos);
        let mut acc = F128::ZERO;
        for pos in 0..n_packed {
            let mut packed = F128::ZERO;
            for b in 0..bits_per_packed {
                let bit_idx = pos * bits_per_packed + b;
                if bit_idx < phys_bits.len() && phys_bits[bit_idx] {
                    if b < 64 {
                        packed.lo |= 1u64 << b;
                    } else {
                        packed.hi |= 1u64 << (b - 64);
                    }
                }
            }
            acc += eq_tau[pos] * packed;
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Per-slot fold from packed witness
// ---------------------------------------------------------------------------

/// For each of the 4 slot positions, compute one F128 per instance: the
/// τ_pos-MLE of that slot's content. Output `result[s][i]` is the τ_pos-fold
/// of slot `s` for instance `i`. Slot index uses LSB-first encoding
/// `s = sel_slot | (side << 1)`, matching the cube convention.
pub fn fold_all_slots(
    layout: &MerkleLayout,
    packed: &[F128],
    fold: &MerklePathFold,
) -> [Vec<F128>; 4] {
    use rayon::prelude::*;

    let bits_per_packed = 1usize << LOG_PACKING;
    let n_packed_per_slot = 1usize << fold.tau_pos.len();
    let block_packed = (1usize << layout.k_log) / bits_per_packed;
    let slot_base_packed = (layout.slot_base_byte_off * 8) / bits_per_packed;
    assert_eq!(
        packed.len() % block_packed,
        0,
        "packed witness length must be a whole number of blocks"
    );
    let n_inst = packed.len() / block_packed;

    let eq_tau = build_eq_table(&fold.tau_pos);

    let fold_one = |base: usize| -> F128 {
        let mut acc = F128::ZERO;
        for pos in 0..n_packed_per_slot {
            acc += eq_tau[pos] * packed[base + pos];
        }
        acc
    };

    // Build each slot's vector in parallel.
    let mut results: [Vec<F128>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (slot_idx, vec_out) in results.iter_mut().enumerate() {
        let slot_offset = slot_base_packed + slot_idx * n_packed_per_slot;
        *vec_out = (0..n_inst)
            .into_par_iter()
            .map(|i| fold_one(i * block_packed + slot_offset))
            .collect();
    }
    results
}

// ---------------------------------------------------------------------------
// Packed-direct claim assembly
// ---------------------------------------------------------------------------

/// Assemble the packed-direct merkle claim from the fold and the shift
/// sumcheck output. Claim point layout (LSB-first over `L = m − LOG_PACKING`
/// coords):
/// ```text
///   [τ_pos ..., sel_slot, side, 0, 0, ..., 0, instance_point ...]
///     ^^^^^    ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^
///     fold     slot selectors    high slot bits   sumcheck output
///     coords   (sel_slot, side)  pinned to 0      instance coord
/// ```
/// The high-slot-bits-zero coords (`high_zeros = k_log − region_log − 2`) keep
/// `eq_ind(point)` sparse with a `2^high_zeros ×` density reduction.
pub fn assemble_merkle_path_claim(
    layout: &MerkleLayout,
    fold: &MerklePathFold,
    claims: &crate::merkle_path::MerklePathClaims,
) -> PackedDirectClaim {
    let high = layout.high_zeros();
    let point_len = fold.tau_pos.len() + 2 + high + claims.instance_point.len();
    let mut point = Vec::with_capacity(point_len);
    point.extend_from_slice(&fold.tau_pos);
    point.push(claims.sel_slot);
    point.push(claims.side);
    point.extend(std::iter::repeat_n(F128::ZERO, high));
    point.extend_from_slice(&claims.instance_point);
    debug_assert_eq!(point.len(), point_len);

    let sparse_eq = flock_core::pcs::ring_switch::build_eq_sparse(&point);
    PackedDirectClaim {
        point,
        value: claims.value,
        eq_ind: DirectEqInd::Sparse(sparse_eq),
    }
}

/// Verifier-side helper: build the claim point identically to
/// [`assemble_merkle_path_claim`] without constructing the sparse eq tensor.
fn build_merkle_claim_point(
    layout: &MerkleLayout,
    fold: &MerklePathFold,
    claims: &crate::merkle_path::MerklePathClaims,
) -> Vec<F128> {
    let high = layout.high_zeros();
    let point_len = fold.tau_pos.len() + 2 + high + claims.instance_point.len();
    let mut point = Vec::with_capacity(point_len);
    point.extend_from_slice(&fold.tau_pos);
    point.push(claims.sel_slot);
    point.push(claims.side);
    point.extend(std::iter::repeat_n(F128::ZERO, high));
    point.extend_from_slice(&claims.instance_point);
    debug_assert_eq!(point.len(), point_len);
    point
}

// ---------------------------------------------------------------------------
// Proof + Error types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerklePathProof {
    pub zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub lincheck: flock_core::lincheck::LincheckProof,
    pub shift: MerklePathShiftProof,
    pub pcs_open: flock_core::pcs::BatchOpeningProof,
}

/// Ligerito-backend mirror of [`MerklePathProof`]. Same protocol upstream;
/// only the final PCS opening differs (Ligerito recursive proof instead of
/// BaseFold + FRI), which is what shrinks the serialized proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerklePathProofLigerito {
    pub zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub lincheck: flock_core::lincheck::LincheckProof,
    pub shift: MerklePathShiftProof,
    pub pcs_open: flock_core::pcs::BatchOpeningProofLigerito,
}

#[derive(Debug)]
pub enum MerklePathVerifyError {
    /// Base R1CS replay failed.
    R1cs(flock_core::verifier::VerifyError),
    /// Merkle-path shift sumcheck check failed.
    Shift(crate::merkle_path::MerklePathError),
    /// The batched PCS opening failed.
    Pcs(flock_core::pcs::VerifyError),
}

// ---------------------------------------------------------------------------
// Generic prover / verifier
// ---------------------------------------------------------------------------

/// Generic Merkle-path prover. Runs core → packed-pos fold → shift sumcheck
/// (with bit selector) → one batched PCS open over `[ab, c] ring-switched
/// + [merkle] packed-direct`.
#[allow(clippy::too_many_arguments)]
pub fn prove_merkle_path_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    layout: &MerkleLayout,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (MerklePathProof, Commitment) {
    let trace = std::env::var("MERKLE_TRACE").is_ok();

    // ---- Core: commit → zerocheck → lincheck.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let core = crate::prover::prove_fast_core(
        r1cs,
        pcs_params,
        z_packed,
        a_packed,
        b_packed,
        z_lincheck,
        lincheck_circuit,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "base_r1cs (zc+lc)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Packed-pos fold: sample τ_pos, compute the 4 slot vectors.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);
    let slot_vals = fold_all_slots(layout, &core.z_packed, &fold);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "fold_slots",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Merkle-path shift sumcheck. Pass the slot vectors in role order.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let x_l_vals = &slot_vals[layout.x_l_slot as usize];
    let x_r_vals = &slot_vals[layout.x_r_slot as usize];
    let z_vals = &slot_vals[layout.z_slot as usize];
    let iv_vals = &slot_vals[layout.other_slot() as usize];
    let (shift, claims) = crate::merkle_path::prove_merkle_path_shift(
        0, // path_log: single-path
        x_l_vals,
        x_r_vals,
        z_vals,
        iv_vals,
        b_bits,
        layout.slot_layout(),
        challenger,
    );
    let merkle_claim = assemble_merkle_path_claim(layout, &fold, &claims);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "shift_sumcheck",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Batched open: [ab, c] ring-switched + [merkle] packed-direct.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let padding = flock_core::zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let ab_x_outer = crate::prover::quirky_x_outer_full(&core.ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&core.c.point);
    let pre_ab: Option<&[flock_core::field::F128]> = core.s_hat_v_ab.as_deref();
    let pre_c: Option<&[flock_core::field::F128]> = Some(core.s_hat_v_c.as_slice());
    let pcs_open = flock_core::pcs::open_batch_mixed_with_precomputed_s_hat_v(
        &core.z_packed,
        &core.prover_data,
        &core.commitment,
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        &[pre_ab, pre_c],
        std::slice::from_ref(&merkle_claim),
        &padding,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "open_batch",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        MerklePathProof {
            zerocheck: core.zc_proof,
            lincheck: core.lc_proof,
            shift,
            pcs_open,
        },
        core.commitment,
    )
}

/// Generic Merkle-path verifier. `leaf_phys` / `root_phys` are the public
/// endpoints in physical within-slot bool order; `b_bits` is the public bit
/// vector (length `2^n_log`); `n_log = m − k_log` is the instance arity.
#[allow(clippy::too_many_arguments)]
pub fn verify_merkle_path_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    layout: &MerkleLayout,
    commitment: &Commitment,
    proof: &MerklePathProof,
    n_log: usize,
    leaf_phys: &[bool],
    root_phys: &[bool],
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(), MerklePathVerifyError> {
    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    // ---- Replay core → (ab, c).
    let t = std::time::Instant::now();
    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )
    .map_err(MerklePathVerifyError::R1cs)?;
    if trace {
        eprintln!(
            "    [vm] verify_core (zc+lc): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Packed-pos fold (matches prover transcript order).
    let t = std::time::Instant::now();
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);

    let leaf_r = fold.fold_public_phys(leaf_phys);
    let root_r = fold.fold_public_phys(root_phys);

    // ---- Verify merkle shift sumcheck.
    let claims = crate::merkle_path::verify_merkle_path_shift(
        0, // path_log: single-path
        &proof.shift,
        &[leaf_r],
        root_r,
        b_bits,
        n_log,
        layout.slot_layout(),
        challenger,
    )
    .map_err(MerklePathVerifyError::Shift)?;
    if trace {
        eprintln!(
            "    [vm] τ_pos + shift sumcheck: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Verify the mixed batched open (2 ring-switched + 1 packed-direct).
    let t = std::time::Instant::now();
    let merkle_point = build_merkle_claim_point(layout, &fold, &claims);
    let ab_x_outer = crate::prover::quirky_x_outer_full(&ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&c.point);
    let pd_ref = PackedDirectClaimRef {
        point: &merkle_point,
        value: claims.value,
    };
    flock_core::pcs::verify_opening_batch_mixed(
        commitment,
        &[ab.value, c.value],
        &[ab.point.z_skip, c.point.z_skip],
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        std::slice::from_ref(&pd_ref),
        &proof.pcs_open,
        challenger,
    )
    .map_err(MerklePathVerifyError::Pcs)?;
    if trace {
        eprintln!(
            "    [vm] PCS verify_opening_batch_mixed (2 rs + 1 pd): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-path prover / verifier
// ---------------------------------------------------------------------------

/// Multi-path generalisation of [`prove_merkle_path_generic`]. Proves `P =
/// 2^path_log` independent Merkle paths of length `L = 2^pos_log` (where
/// `pos_log = n_log − path_log`) against one shared root, over a single PCS
/// commitment. `b_bits` is the concatenated bit vector of length `N = PL`
/// (per-path bit vectors stacked in path-id order; the first bit of each path
/// is treated as 0 by convention regardless of the supplied value).
///
/// `path_log = 0` recovers single-path. The proof shape is identical to the
/// single-path case; only τ and the verifier's claim assembly differ.
#[allow(clippy::too_many_arguments)]
pub fn prove_merkle_paths_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    layout: &MerkleLayout,
    path_log: usize,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (MerklePathProof, Commitment) {
    let trace = std::env::var("MERKLE_TRACE").is_ok();

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let core = crate::prover::prove_fast_core(
        r1cs,
        pcs_params,
        z_packed,
        a_packed,
        b_packed,
        z_lincheck,
        lincheck_circuit,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "base_r1cs (zc+lc)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);
    let slot_vals = fold_all_slots(layout, &core.z_packed, &fold);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "fold_slots",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let x_l_vals = &slot_vals[layout.x_l_slot as usize];
    let x_r_vals = &slot_vals[layout.x_r_slot as usize];
    let z_vals = &slot_vals[layout.z_slot as usize];
    let iv_vals = &slot_vals[layout.other_slot() as usize];
    let (shift, claims) = crate::merkle_path::prove_merkle_path_shift(
        path_log,
        x_l_vals,
        x_r_vals,
        z_vals,
        iv_vals,
        b_bits,
        layout.slot_layout(),
        challenger,
    );
    let merkle_claim = assemble_merkle_path_claim(layout, &fold, &claims);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "shift_sumcheck",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let padding = flock_core::zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let ab_x_outer = crate::prover::quirky_x_outer_full(&core.ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&core.c.point);
    let pre_ab: Option<&[flock_core::field::F128]> = core.s_hat_v_ab.as_deref();
    let pre_c: Option<&[flock_core::field::F128]> = Some(core.s_hat_v_c.as_slice());
    let pcs_open = flock_core::pcs::open_batch_mixed_with_precomputed_s_hat_v(
        &core.z_packed,
        &core.prover_data,
        &core.commitment,
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        &[pre_ab, pre_c],
        std::slice::from_ref(&merkle_claim),
        &padding,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "open_batch",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        MerklePathProof {
            zerocheck: core.zc_proof,
            lincheck: core.lc_proof,
            shift,
            pcs_open,
        },
        core.commitment,
    )
}

/// Multi-path verifier. `leaves_phys[i_p]` is path `i_p`'s leaf bit-vector
/// (length `region_bits`); `root_phys` is the single shared root.
#[allow(clippy::too_many_arguments)]
pub fn verify_merkle_paths_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    layout: &MerkleLayout,
    path_log: usize,
    commitment: &Commitment,
    proof: &MerklePathProof,
    n_log: usize,
    leaves_phys: &[&[bool]],
    root_phys: &[bool],
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(), MerklePathVerifyError> {
    assert!(path_log <= n_log, "path_log must be ≤ n_log");
    let n_paths = 1usize << path_log;
    assert_eq!(
        leaves_phys.len(),
        n_paths,
        "leaves_phys must have length 2^path_log"
    );

    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    let t = std::time::Instant::now();
    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )
    .map_err(MerklePathVerifyError::R1cs)?;
    if trace {
        eprintln!(
            "    [vm] verify_core (zc+lc): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    let t = std::time::Instant::now();
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);

    // Fold each per-path leaf at τ_pos; root only once (it's shared).
    let leaf_evals: Vec<F128> = leaves_phys
        .iter()
        .map(|lp| fold.fold_public_phys(lp))
        .collect();
    let root_r = fold.fold_public_phys(root_phys);

    let claims = crate::merkle_path::verify_merkle_path_shift(
        path_log,
        &proof.shift,
        &leaf_evals,
        root_r,
        b_bits,
        n_log,
        layout.slot_layout(),
        challenger,
    )
    .map_err(MerklePathVerifyError::Shift)?;
    if trace {
        eprintln!(
            "    [vm] τ_pos + shift sumcheck: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    let t = std::time::Instant::now();
    let merkle_point = build_merkle_claim_point(layout, &fold, &claims);
    let ab_x_outer = crate::prover::quirky_x_outer_full(&ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&c.point);
    let pd_ref = PackedDirectClaimRef {
        point: &merkle_point,
        value: claims.value,
    };
    flock_core::pcs::verify_opening_batch_mixed(
        commitment,
        &[ab.value, c.value],
        &[ab.point.z_skip, c.point.z_skip],
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        std::slice::from_ref(&pd_ref),
        &proof.pcs_open,
        challenger,
    )
    .map_err(MerklePathVerifyError::Pcs)?;
    if trace {
        eprintln!(
            "    [vm] PCS verify_opening_batch_mixed (2 rs + 1 pd): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Ligerito-backend prover / verifier
// ---------------------------------------------------------------------------

/// Ligerito-backend mirror of [`prove_merkle_paths_generic`]. Identical
/// protocol upstream (core → packed-pos fold → shift sumcheck); routes the
/// final batched open through Ligerito instead of BaseFold. `path_log = 0`
/// recovers single-path, so this backs both single- and multi-path Ligerito
/// Merkle wrappers.
#[allow(clippy::too_many_arguments)]
pub fn prove_merkle_paths_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    layout: &MerkleLayout,
    path_log: usize,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (MerklePathProofLigerito, Commitment) {
    let trace = std::env::var("MERKLE_TRACE").is_ok();

    let log_n = r1cs.m - LOG_PACKING;
    let lig_config = flock_core::pcs::ligerito::prover_config_for(
        log_n,
        pcs_params.log_batch_size,
        pcs_params.profile,
    )
    .expect("Ligerito config for merkle-path prove; bump m for tiny instances");

    // ---- Core: commit → zerocheck → lincheck.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let core = crate::prover::prove_fast_core(
        r1cs,
        pcs_params,
        z_packed,
        a_packed,
        b_packed,
        z_lincheck,
        lincheck_circuit,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "base_r1cs (zc+lc)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Packed-pos fold.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);
    let slot_vals = fold_all_slots(layout, &core.z_packed, &fold);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "fold_slots",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Merkle-path shift sumcheck.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let x_l_vals = &slot_vals[layout.x_l_slot as usize];
    let x_r_vals = &slot_vals[layout.x_r_slot as usize];
    let z_vals = &slot_vals[layout.z_slot as usize];
    let iv_vals = &slot_vals[layout.other_slot() as usize];
    let (shift, claims) = crate::merkle_path::prove_merkle_path_shift(
        path_log,
        x_l_vals,
        x_r_vals,
        z_vals,
        iv_vals,
        b_bits,
        layout.slot_layout(),
        challenger,
    );
    let merkle_claim = assemble_merkle_path_claim(layout, &fold, &claims);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "shift_sumcheck",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Batched open: [ab, c] ring-switched + [merkle] packed-direct, via Ligerito.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let padding = flock_core::zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let ab_x_outer = crate::prover::quirky_x_outer_full(&core.ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&core.c.point);
    // Destructure core to move z_packed by value into the open (saves a large
    // clone at high m), mirroring `prove_chain_ligerito_generic`.
    let crate::prover::ProveCore {
        zc_proof,
        lc_proof,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
        ..
    } = core;
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = flock_core::pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        &prover_data,
        &commitment,
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        &[pre_ab, pre_c],
        std::slice::from_ref(&merkle_claim),
        &padding,
        &lig_config,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "open_batch",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        MerklePathProofLigerito {
            zerocheck: zc_proof,
            lincheck: lc_proof,
            shift,
            pcs_open,
        },
        commitment,
    )
}

/// Ligerito-backend mirror of [`verify_merkle_paths_generic`]. `path_log = 0`
/// with a single leaf recovers the single-path verifier.
#[allow(clippy::too_many_arguments)]
pub fn verify_merkle_paths_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    layout: &MerkleLayout,
    path_log: usize,
    commitment: &Commitment,
    proof: &MerklePathProofLigerito,
    n_log: usize,
    leaves_phys: &[&[bool]],
    root_phys: &[bool],
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), MerklePathVerifyError> {
    assert!(path_log <= n_log, "path_log must be ≤ n_log");
    let n_paths = 1usize << path_log;
    assert_eq!(
        leaves_phys.len(),
        n_paths,
        "leaves_phys must have length 2^path_log"
    );

    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )
    .map_err(MerklePathVerifyError::R1cs)?;

    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);

    let leaf_evals: Vec<F128> = leaves_phys
        .iter()
        .map(|lp| fold.fold_public_phys(lp))
        .collect();
    let root_r = fold.fold_public_phys(root_phys);

    let claims = crate::merkle_path::verify_merkle_path_shift(
        path_log,
        &proof.shift,
        &leaf_evals,
        root_r,
        b_bits,
        n_log,
        layout.slot_layout(),
        challenger,
    )
    .map_err(MerklePathVerifyError::Shift)?;

    let merkle_point = build_merkle_claim_point(layout, &fold, &claims);
    let ab_x_outer = crate::prover::quirky_x_outer_full(&ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&c.point);
    let pd_ref = PackedDirectClaimRef {
        point: &merkle_point,
        value: claims.value,
    };

    let log_n = r1cs.m - LOG_PACKING;
    let lig_v_config = flock_core::pcs::ligerito::verifier_config_for(
        log_n,
        pcs_params.log_batch_size,
        pcs_params.profile,
    )
    .expect("Ligerito verifier config for merkle-path verify");

    flock_core::pcs::verify_opening_batch_ligerito_mixed(
        commitment,
        &[ab.value, c.value],
        &[ab.point.z_skip, c.point.z_skip],
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        std::slice::from_ref(&pd_ref),
        &proof.pcs_open,
        &lig_v_config,
        challenger,
    )
    .map_err(MerklePathVerifyError::Pcs)?;

    Ok(())
}
