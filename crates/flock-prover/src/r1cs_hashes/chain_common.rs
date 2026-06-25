//! Generic hash-chain glue, shared by the per-hash `*_chain` modules.
//!
//! The chain protocol (packed-pos fold → shift sumcheck → batched PCS open
//! over `[ab, c] ring-switched + [chain] packed-direct`) is hash-agnostic; only
//! the *geometry* of where the input/output regions live in a witness block
//! varies. This module captures that geometry in [`ChainLayout`] and provides
//! the fully generic [`prove_chain_generic`] / [`verify_chain_generic`]. A
//! per-hash module supplies its `ChainLayout`, a `State → physical-bits`
//! converter for the public endpoints, and thin wrappers.
//!
//! ## Region requirements
//!
//! Both the input and output regions must be **byte-contiguous, aligned slots**:
//! each occupies an aligned `2^region_log`-bit window of the block, with the
//! `region_bits` real bits (a multiple of 8) at the low end and zero padding
//! above. The two slots must be consecutive (slot 0 = input, slot 1 = output)
//! so the chain claim's selector is a single bit-flip in the multilinear cube.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::pcs::{
    Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
};
use flock_core::r1cs::BlockR1cs;
use serde::{Deserialize, Serialize};

/// Geometry of one hash's input/output regions within a witness block. All
/// fields are `const`-known per hash.
#[derive(Clone, Copy, Debug)]
pub struct ChainLayout {
    /// `log2` of the per-block witness size (inner variables).
    pub k_log: usize,
    /// Univariate-skip dimension (kept for API parity with the zerocheck/PCS
    /// `k_skip`; the packed-direct chain path does not consume it).
    pub k_skip: usize,
    /// `log2` of the aligned slot size holding each I/O region.
    pub region_log: usize,
    /// Real bits per region (≤ `2^region_log`, multiple of 8).
    pub region_bits: usize,
    /// Byte offset of the input slot within a block.
    pub input_byte_off: usize,
    /// Byte offset of the output slot within a block.
    pub output_byte_off: usize,
}

impl ChainLayout {
    /// Length of the packed-position fold coord `τ_pos` = `region_log − LOG_PACKING`.
    /// One `F128` per packed position within a region's 2^region_log-bit slot.
    #[inline]
    pub fn tau_pos_len(&self) -> usize {
        self.region_log - LOG_PACKING
    }

    /// Number of zero coords in the chain claim's point between the selector
    /// bit and the instance index. Drives the sparseness of `eq_ind(point)`:
    /// each zero gives a 2× density reduction. = `k_log − region_log − 1`.
    #[inline]
    pub fn high_zeros(&self) -> usize {
        self.k_log - self.region_log - 1
    }
}

/// Packed-level fold parameters: `τ_pos` binds the packed-position dimension of
/// each region. The verifier samples `τ_pos`, then the prover folds each
/// instance's input/output region down to one `F128` via
/// `Σ_{pos} eq(τ_pos, pos) · ẑ_packed[(inst, slot, pos)]`.
#[derive(Clone, Debug)]
pub struct ChainFold {
    pub tau_pos: Vec<F128>,
}

impl ChainFold {
    pub fn new(layout: &ChainLayout, tau_pos: Vec<F128>) -> Self {
        assert_eq!(
            tau_pos.len(),
            layout.tau_pos_len(),
            "τ_pos length must be region_log − LOG_PACKING"
        );
        Self { tau_pos }
    }

    /// Fold a public endpoint (given as `region_bits` bools in physical
    /// within-slot order) to a single `F128` — the τ_pos-MLE of the endpoint
    /// over the region's packed positions. Mirrors what the prover computes
    /// against the committed witness.
    ///
    /// Algorithm: pack the bits into `2^τ_pos_len` `F128` elements (padding the
    /// region's `region_bits..slot_bits` tail with zeros to match the witness
    /// layout), then take the inner product with `eq(τ_pos, ·)`.
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

/// Packed-level region fold: produces `(in_vals, out_vals)` where
/// `in_vals[i] = Σ_pos eq(τ_pos, pos) · ẑ_packed[(i, slot=0, pos)]` (state_0
/// of instance i, τ_pos-folded) and analogously for `out_vals`. Parallel over
/// instances.
///
/// Replaces the prior bit-level byte-table fold over `region_bits` per
/// instance; here the per-instance work is just `2^τ_pos_len` F128
/// mul-adds (16 for keccak, 2 for blake3/sha2).
pub fn fold_in_out(
    layout: &ChainLayout,
    packed: &[F128],
    fold: &ChainFold,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;

    let bits_per_packed = 1usize << LOG_PACKING; // 128
    let n_packed_per_region = 1usize << fold.tau_pos.len();
    let block_packed = (1usize << layout.k_log) / bits_per_packed;
    let in_pos_base = (layout.input_byte_off * 8) / bits_per_packed;
    let out_pos_base = (layout.output_byte_off * 8) / bits_per_packed;
    assert_eq!(
        packed.len() % block_packed,
        0,
        "packed witness length must be a whole number of blocks"
    );
    let n_inst = packed.len() / block_packed;

    let eq_tau = build_eq_table(&fold.tau_pos);

    let fold_one = |base: usize| -> F128 {
        let mut acc = F128::ZERO;
        for pos in 0..n_packed_per_region {
            acc += eq_tau[pos] * packed[base + pos];
        }
        acc
    };

    let in_vals: Vec<F128> = (0..n_inst)
        .into_par_iter()
        .map(|i| fold_one(i * block_packed + in_pos_base))
        .collect();
    let out_vals: Vec<F128> = (0..n_inst)
        .into_par_iter()
        .map(|i| fold_one(i * block_packed + out_pos_base))
        .collect();

    (in_vals, out_vals)
}

/// Assemble the packed-direct chain claim from the fold and the shift
/// sumcheck output. The point layout (LSB-first over `L = m − LOG_PACKING`
/// coords):
/// ```text
///   [τ_pos ..., sel0, 0, 0, ..., 0, instance_point ...]
///     ^^^^^   ^^^^   ^^^^^^^^^^^^   ^^^^^^^^^^^^^^
///     fold    in/out  high slot      sumcheck output
///     coords  selector  bits = 0     instance coord
/// ```
/// The high-slot-bits-zero coords (`high_zeros = k_log − region_log − 1`) make
/// `eq_ind(point)` sparse with a 2^high_zeros × density reduction. We use
/// `build_eq_sparse` to skip the zero-coord halvings.
pub fn assemble_chain_claim(
    layout: &ChainLayout,
    fold: &ChainFold,
    claims: &crate::chain::ChainClaims,
) -> PackedDirectClaim {
    let high = layout.high_zeros();
    let point_len = fold.tau_pos.len() + 1 + high + claims.instance_point.len();
    let mut point = Vec::with_capacity(point_len);
    point.extend_from_slice(&fold.tau_pos);
    point.push(claims.sel0);
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

/// Verifier-side helper: build the chain-claim point identically to
/// [`assemble_chain_claim`] but without constructing the sparse eq tensor (the
/// verifier evaluates `eq_eval(point, basefold_challenges)` directly).
fn build_chain_claim_point(
    layout: &ChainLayout,
    fold: &ChainFold,
    claims: &crate::chain::ChainClaims,
) -> Vec<F128> {
    let high = layout.high_zeros();
    let point_len = fold.tau_pos.len() + 1 + high + claims.instance_point.len();
    let mut point = Vec::with_capacity(point_len);
    point.extend_from_slice(&fold.tau_pos);
    point.push(claims.sel0);
    point.extend(std::iter::repeat_n(F128::ZERO, high));
    point.extend_from_slice(&claims.instance_point);
    debug_assert_eq!(point.len(), point_len);
    point
}

/// Proof that `2^n` committed hash instances form a sequential chain
/// `x_{i+1} = h(x_i)` with public endpoints. Composes the base R1CS sub-proofs,
/// the shift-sumcheck sub-proof, and ONE PCS open over
/// `[ab, c] (ring-switched) + [chain] (packed-direct)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainProof {
    pub zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub lincheck: flock_core::lincheck::LincheckProof,
    pub shift: crate::chain::ChainShiftProof,
    pub pcs_open: flock_core::pcs::BatchOpeningProof,
}

/// Ligerito-backend variant of [`ChainProof`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainProofLigerito {
    pub zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub lincheck: flock_core::lincheck::LincheckProof,
    pub shift: crate::chain::ChainShiftProof,
    pub pcs_open: flock_core::pcs::BatchOpeningProofLigerito,
}

/// Errors from chain verification.
#[derive(Debug)]
pub enum ChainVerifyError {
    /// Base R1CS (zerocheck/lincheck) replay failed.
    R1cs(flock_core::verifier::VerifyError),
    /// The shift-sumcheck (glue + endpoints) check failed.
    Shift(crate::chain::ChainError),
    /// The batched PCS opening failed.
    Pcs(flock_core::pcs::VerifyError),
}

/// Generic chain prover. The caller supplies the hash's already-generated packed
/// witness buffers (`z`, `a`, `b`, lincheck stripe) and the layout; this runs
/// core → packed-pos fold → shift sumcheck → one batched open over `[ab, c] +
/// [chain]` where the chain claim is packed-direct.
#[allow(clippy::too_many_arguments)]
pub fn prove_chain_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    layout: &ChainLayout,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (ChainProof, Commitment) {
    let trace = std::env::var("CHAIN_TRACE").is_ok();

    // ---- Core: commit → zerocheck → lincheck → base claims (ab, c).
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
            "[chain] {:<18} {:>8.2} ms",
            "base_r1cs (zc+lc)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Packed-pos fold: sample τ_pos, compute In/Out via τ_pos-MLE of each
    //      instance's input/output region.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = ChainFold::new(layout, tau_pos);
    let (in_vals, out_vals) = fold_in_out(layout, &core.z_packed, &fold);
    if let Some(t) = t {
        eprintln!(
            "[chain] {:<18} {:>8.2} ms",
            "fold_in_out",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Shift sumcheck (samples τ, α internally) → proof + claim pieces.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let (shift, claims) = crate::chain::prove_chain_shift(&in_vals, &out_vals, challenger);
    let chain_claim = assemble_chain_claim(layout, &fold, &claims);
    if let Some(t) = t {
        eprintln!(
            "[chain] {:<18} {:>8.2} ms",
            "shift_sumcheck",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- Single batched open over [ab, c] (ring-switched) + [chain] (packed-direct).
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
        std::slice::from_ref(&chain_claim),
        &padding,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[chain] {:<18} {:>8.2} ms",
            "open_batch",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        ChainProof {
            zerocheck: core.zc_proof,
            lincheck: core.lc_proof,
            shift,
            pcs_open,
        },
        core.commitment,
    )
}

/// Ligerito-backend mirror of [`prove_chain_generic`]. Same protocol upstream;
/// routes the final PCS open through Ligerito + builds the per-hash
/// LigeritoConfig from the PCS params.
#[allow(clippy::too_many_arguments)]
pub fn prove_chain_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    layout: &ChainLayout,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (ChainProofLigerito, Commitment) {
    let log_n = r1cs.m - flock_core::pcs::LOG_PACKING;
    let lig_config = flock_core::pcs::ligerito::prover_config_for(
        log_n,
        pcs_params.log_batch_size,
        pcs_params.profile,
    )
    .expect("Ligerito default config for chain prove; bump m for tiny instances");

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

    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = ChainFold::new(layout, tau_pos);
    let (in_vals, out_vals) = fold_in_out(layout, &core.z_packed, &fold);

    let (shift, claims) = crate::chain::prove_chain_shift(&in_vals, &out_vals, challenger);
    let chain_claim = assemble_chain_claim(layout, &fold, &claims);

    let padding = flock_core::zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let ab_x_outer = crate::prover::quirky_x_outer_full(&core.ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&core.c.point);
    // Destructure core to move z_packed by value into the open (saves a 128 MB
    // clone at m=30 BLAKE3).
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
        std::slice::from_ref(&chain_claim),
        &padding,
        &lig_config,
        challenger,
    );

    (
        ChainProofLigerito {
            zerocheck: zc_proof,
            lincheck: lc_proof,
            shift,
            pcs_open,
        },
        commitment,
    )
}

/// Ligerito-backend mirror of [`verify_chain_generic`].
#[allow(clippy::too_many_arguments)]
pub fn verify_chain_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    layout: &ChainLayout,
    commitment: &Commitment,
    proof: &ChainProofLigerito,
    n_log: usize,
    x0_phys: &[bool],
    xlast_phys: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), ChainVerifyError> {
    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )
    .map_err(ChainVerifyError::R1cs)?;

    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = ChainFold::new(layout, tau_pos);

    let x0_r = fold.fold_public_phys(x0_phys);
    let xlast_r = fold.fold_public_phys(xlast_phys);
    let claims = crate::chain::verify_chain_shift(&proof.shift, x0_r, xlast_r, n_log, challenger)
        .map_err(ChainVerifyError::Shift)?;

    let chain_point = build_chain_claim_point(layout, &fold, &claims);
    let ab_x_outer = crate::prover::quirky_x_outer_full(&ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&c.point);
    let pd_ref = PackedDirectClaimRef {
        point: &chain_point,
        value: claims.value,
    };

    let log_n = r1cs.m - flock_core::pcs::LOG_PACKING;
    let lig_v_config = flock_core::pcs::ligerito::verifier_config_for(
        log_n,
        pcs_params.log_batch_size,
        pcs_params.profile,
    )
    .expect("Ligerito default verifier config for chain verify");

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
    .map_err(ChainVerifyError::Pcs)?;

    Ok(())
}

/// Generic chain verifier. `x0_phys` / `xlast_phys` are the public endpoints as
/// `region_bits` bools in **physical** within-slot order; `n_log` is the
/// instance-index arity (`m − k_log`).
#[allow(clippy::too_many_arguments)]
pub fn verify_chain_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    layout: &ChainLayout,
    commitment: &Commitment,
    proof: &ChainProof,
    n_log: usize,
    x0_phys: &[bool],
    xlast_phys: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> Result<(), ChainVerifyError> {
    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    // ---- Replay base core → (ab, c).
    let t = std::time::Instant::now();
    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )
    .map_err(ChainVerifyError::R1cs)?;
    if trace {
        eprintln!(
            "    [vch] verify_core (zerocheck+lincheck): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Packed-pos fold parameters (matches prover transcript order).
    let t = std::time::Instant::now();
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = ChainFold::new(layout, tau_pos);

    // ---- Verify the shift sumcheck; fold public endpoints with same τ_pos.
    let x0_r = fold.fold_public_phys(x0_phys);
    let xlast_r = fold.fold_public_phys(xlast_phys);
    let claims = crate::chain::verify_chain_shift(&proof.shift, x0_r, xlast_r, n_log, challenger)
        .map_err(ChainVerifyError::Shift)?;
    if trace {
        eprintln!(
            "    [vch] τ_pos fold + chain shift sumcheck: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // ---- Build the packed-direct chain claim point + value, then verify the
    //      mixed batched open. ab/c go through ring-switch; chain goes
    //      packed-direct.
    let t = std::time::Instant::now();
    let chain_point = build_chain_claim_point(layout, &fold, &claims);
    let ab_x_outer = crate::prover::quirky_x_outer_full(&ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&c.point);
    let pd_ref = PackedDirectClaimRef {
        point: &chain_point,
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
    .map_err(ChainVerifyError::Pcs)?;
    if trace {
        eprintln!(
            "    [vch] PCS verify_opening_batch_mixed (2 rs + 1 pd): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    Ok(())
}
