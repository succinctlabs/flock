//! E6 — end-to-end R1CS prove/verify on an L1′-committed witness (keccak,
//! BaseFold backend).
//!
//! Mirrors `flock_prover::prover::prove_fast_from_witness` /
//! `flock_core::verifier::verify`, with the L1′ bookkeeping differences:
//!
//! - the witness is produced directly in L1′ (`keccak_vwide::build_l1_direct`)
//!   and committed as-is;
//! - zerocheck runs on the L1′ buffers with the suffix `PaddingSpec`
//!   (`k_log = m`, useful prefix);
//! - the zerocheck→lincheck point translation splits `mlv_challenges` in
//!   L1′ address order: `[dim6 | batch (n_log) | chunk (k_log−7)]`, so
//!   `x_inner_rest = [mlv[0]] ++ mlv[1+n_log..]` and
//!   `x_outer = mlv[1..1+n_log]`;
//! - PCS claim points are assembled in **address order**:
//!   `x_full_ab = [r_inner_rest[0]] ++ x_outer ++ r_inner_rest[1..]`,
//!   `x_full_c = zc.r_rest` (already address-ordered). The verifier reuses
//!   `flock_core::verifier::verify_claims` by carrying the full
//!   address-ordered vector in `ZClaim.point.x_inner_rest` (its PCS step
//!   just concatenates the two segments).
//!
//! Lincheck itself is untouched: its inputs (the byte-stripe, the circuit
//! walker, the semantic quirky point) are layout-independent.

use crate::keccak_vwide;
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::{self, QuirkyPoint};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::{R1csProof, ZClaim, bind_statement};
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier::verify_claims;
use flock_core::zerocheck::{self, PaddingSpec};
use flock_prover::r1cs_hashes::keccak::{
    K_LOG, KeccakLincheckCircuit, State, USEFUL_BITS, build_block_r1cs,
};

/// L1′ suffix padding: one giant block with a useful chunk-column prefix.
pub fn l1_padding_spec(m: usize, k_log: usize, useful_bits: usize) -> PaddingSpec {
    let n_log = m - k_log;
    PaddingSpec {
        k_log: m,
        useful_bits_per_block: useful_bits.div_ceil(128) << (7 + n_log),
    }
}

fn as_u8(v: &[F128]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn as_u64_mut(v: &mut [F128]) -> &mut [u64] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, v.len() * 2) }
}

/// Split the zerocheck's address-ordered `mlv_challenges` into the semantic
/// quirky point under L1′.
fn x_ab_from_zerocheck(zc: &zerocheck::ZerocheckClaim, k_log: usize, n_log: usize) -> QuirkyPoint {
    let mlv = &zc.mlv_challenges;
    assert_eq!(mlv.len(), 1 + n_log + (k_log - 7));
    let mut x_inner_rest = Vec::with_capacity(k_log - 6);
    x_inner_rest.push(mlv[0]);
    x_inner_rest.extend_from_slice(&mlv[1 + n_log..]);
    QuirkyPoint {
        z_skip: zc.z,
        x_inner_rest,
        x_outer: mlv[1..1 + n_log].to_vec(),
    }
}

/// Address-ordered PCS point for the AB claim.
fn x_full_ab(r_inner_rest: &[F128], x_outer: &[F128]) -> Vec<F128> {
    let mut v = Vec::with_capacity(r_inner_rest.len() + x_outer.len());
    v.push(r_inner_rest[0]);
    v.extend_from_slice(x_outer);
    v.extend_from_slice(&r_inner_rest[1..]);
    v
}

/// `ZClaim` carrying an address-ordered point (x_outer segment empty — the
/// verifier's PCS step concatenates the segments, so this reproduces the
/// L1′ ordering through the unmodified `verify_claims`).
fn flat_claim(z_skip: F128, x_full: Vec<F128>, value: F128) -> ZClaim {
    ZClaim {
        point: QuirkyPoint {
            z_skip,
            x_inner_rest: x_full,
            x_outer: Vec::new(),
        },
        value,
    }
}

pub struct L1Prove {
    pub proof: R1csProof,
    pub commitment: Commitment,
    pub ab: ZClaim,
    pub c: ZClaim,
}

/// End-to-end keccak prove on an L1′-committed witness (BaseFold).
pub fn prove_l1_keccak<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    states: &[State],
    challenger: &mut Ch,
) -> L1Prove {
    let m = r1cs.m;
    let k_log = r1cs.k_log;
    let n_log = m - k_log;
    assert_eq!(k_log, K_LOG);
    let total_f128 = 1usize << (m - 7);
    let timing = std::env::var_os("E6_TIMING").is_some();
    let mut t0 = std::time::Instant::now();
    let phase = |name: &str, t0: &mut std::time::Instant| {
        if timing {
            eprintln!("[e6-l1] {name}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        *t0 = std::time::Instant::now();
    };

    // ---- Witness directly in L1′ (+ lincheck stripe). Buffers come from
    // flock's scratch pool (recycled across proves, like production); the
    // direct producer's contract needs padding words zeroed, so par-zero
    // them (the pool hands back dirty buffers).
    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut stripe = vec![0u8; 1usize << (m - 3)];
    {
        use rayon::prelude::*;
        for buf in [&mut z, &mut a, &mut b] {
            buf.par_chunks_mut(1 << 16).for_each(|c| c.fill(F128::ZERO));
        }
    }
    phase("alloc", &mut t0);
    keccak_vwide::build_l1_direct(
        states,
        n_log,
        Some(&mut stripe),
        as_u64_mut(&mut z),
        as_u64_mut(&mut a),
        as_u64_mut(&mut b),
    );
    phase("witness", &mut t0);

    // ---- Commit + bind.
    let (commitment, prover_data) = pcs::commit(&z, pcs_params);
    bind_statement(challenger, r1cs, &commitment);
    phase("commit", &mut t0);

    // ---- Zerocheck on the L1′ buffers (c aliases z), suffix padding.
    let padding = l1_padding_spec(m, k_log, USEFUL_BITS);
    let (zc_proof, zc_claim) =
        zerocheck::prove_packed_padded(as_u8(&a), as_u8(&b), as_u8(&z), m, &padding, challenger);
    flock_core::scratch::give_f128(a);
    flock_core::scratch::give_f128(b);
    phase("zerocheck", &mut t0);

    // ---- Lincheck (layout-independent; stripe + semantic quirky point).
    let x_ab = x_ab_from_zerocheck(&zc_claim, k_log, n_log);
    let (lc_proof, lc_claim) = lincheck::prove_padded(
        &stripe,
        m,
        k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        &KeccakLincheckCircuit,
        &x_ab,
        challenger,
    );
    drop(stripe);
    phase("lincheck", &mut t0);

    // ---- Address-ordered PCS claims.
    let ab = flat_claim(
        lc_claim.r_inner_skip,
        x_full_ab(&lc_claim.r_inner_rest, &x_ab.x_outer),
        lc_claim.w,
    );
    let c = flat_claim(zc_claim.z, zc_claim.r_rest.clone(), zc_claim.c_eval);

    // ---- Batched PCS open at the two L1′ points. NOTE: no precomputed
    // s_hat_v (production skips two fold_1b_rows passes via the lincheck
    // z_vec + zerocheck two-bank captures; re-deriving those under L1′ is a
    // known follow-up) — the open pays two extra full-witness folds here.
    let x_refs: Vec<&[F128]> = vec![&ab.point.x_inner_rest, &c.point.x_inner_rest];
    let pcs_open = pcs::open_batch_padded(
        &z,
        &prover_data,
        &commitment,
        &x_refs,
        &padding,
        challenger,
    );
    flock_core::scratch::give_f128(z);
    phase("open", &mut t0);

    L1Prove {
        proof: R1csProof {
            zerocheck: zc_proof,
            lincheck: lc_proof,
            pcs_open,
        },
        commitment,
        ab,
        c,
    }
}

#[derive(Debug)]
pub enum L1VerifyError {
    Zerocheck(zerocheck::VerifyError),
    Lincheck(lincheck::VerifyError),
    Pcs(pcs::VerifyError),
}

/// Mirror verifier: replay bind → zerocheck → lincheck with the L1′ point
/// assembly, then check the batched PCS opening.
pub fn verify_l1_keccak<Ch: Challenger>(
    r1cs: &BlockR1cs,
    commitment: &Commitment,
    proof: &R1csProof,
    challenger: &mut Ch,
) -> Result<(ZClaim, ZClaim), L1VerifyError> {
    let m = r1cs.m;
    let k_log = r1cs.k_log;
    let n_log = m - k_log;

    bind_statement(challenger, r1cs, commitment);

    let zc_claim =
        zerocheck::verify(m, &proof.zerocheck, challenger).map_err(L1VerifyError::Zerocheck)?;

    let x_ab = x_ab_from_zerocheck(&zc_claim, k_log, n_log);
    let lc_claim = lincheck::verify(
        m,
        k_log,
        r1cs.k_skip,
        &KeccakLincheckCircuit,
        &x_ab,
        zc_claim.a_eval,
        zc_claim.b_eval,
        &proof.lincheck,
        challenger,
    )
    .map_err(L1VerifyError::Lincheck)?;

    let ab = flat_claim(
        lc_claim.r_inner_skip,
        x_full_ab(&lc_claim.r_inner_rest, &x_ab.x_outer),
        lc_claim.w,
    );
    let c = flat_claim(zc_claim.z, zc_claim.r_rest.clone(), zc_claim.c_eval);

    verify_claims(
        commitment,
        &[ab.clone(), c.clone()],
        &proof.pcs_open,
        challenger,
    )
    .map_err(L1VerifyError::Pcs)?;

    Ok((ab, c))
}

/// Convenience: (r1cs, pcs_params) for a keccak batch of `2^n_log` slots
/// with the BaseFold-compatible params (mirrors `KeccakSetup`).
pub fn keccak_setup(n_log: usize) -> (BlockR1cs, PcsParams) {
    let r1cs = build_block_r1cs(n_log);
    let pcs_params = PcsParams {
        m: r1cs.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    (r1cs, pcs_params)
}
