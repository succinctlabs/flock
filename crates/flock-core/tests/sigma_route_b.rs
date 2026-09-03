//! Sigma v2, route B (circuit-wiring-design.tex §sigma): the wiring GKR's
//! `s_sigma_hat(rho)` evaluation DEFERS into the accumulator as a
//! MatrixClaim on the sigma table reshaped `2^nu × 2^c`, instead of the
//! verifier's O(2^mu) evaluation — the root discharge evaluates once.
//!
//! This pins the native chain end to end: the TRUSTING batched verify
//! (`verify_batched` — legitimate exactly because the claim goes to the
//! accumulator) emits (rho, s_sigma_eval); the claim converts to a
//! MatrixClaim via the reshape convention (`row = eq(rho[..nu])`,
//! `col = eq(rho[nu..])`, `M[r, c] = s_sig[(c << nu) + r]`); `bilinear`
//! discharges it; and two claims from independent proofs fold 2 -> 1
//! through `prove_fold`/`verify_fold` with the folded claim discharging
//! identically — the merge-node operation.

use flock_core::{
    challenger::FsChallenger,
    field::F128,
    matrix_fold::{
        DenseMatrix, FoldMatrix, MatrixClaim, Weight, bilinear, prove_fold, verify_fold,
    },
    product_gkr::{build_s_sigma_vec, prove_batched, verify_batched},
    test_rng::Rng,
};

/// One trusting-verified GKR over a constant witness (every permutation is
/// honest on a constant `f = g`, so sigma is unconstrained and exercises a
/// full random table) -> the sigma claim as a MatrixClaim.
fn sigma_claim(mu: usize, nu: usize, sigma: &[usize], domain: &[u8]) -> MatrixClaim {
    let n = 1usize << mu;
    let w = vec![F128::new(0xD00D, 7); n];
    let mut chp = FsChallenger::new(domain);
    let (proof, _) = prove_batched(&w, &w, sigma, None, &mut chp);
    let mut chv = FsChallenger::new(domain);
    let claim = verify_batched(mu, &proof, None, &mut chv).expect("trusting verify accepts");
    MatrixClaim {
        row: Weight::eq(claim.rho[..nu].to_vec()),
        col: Weight::eq(claim.rho[nu..].to_vec()),
        value: claim.s_sigma_eval,
    }
}

#[test]
fn sigma_claims_fold_and_discharge() {
    let (mu, nu) = (6usize, 4usize);
    let mut rng = Rng::new(0x516A_0001);
    let sigma = rng.permutation(1 << mu);
    let s_sig = build_s_sigma_vec(mu, &sigma);
    let m = DenseMatrix {
        vals: s_sig,
        n_rows_log: nu,
    };

    // The emitted claim discharges directly: the trusting value IS the
    // reshaped bilinear — pinning the reshape convention.
    let c1 = sigma_claim(mu, nu, &sigma, b"sigma-route-b-1");
    assert_eq!(
        bilinear(&c1.row, &c1.col, &m),
        c1.value,
        "the emitted sigma claim discharges against the reshaped table"
    );
    let c2 = sigma_claim(mu, nu, &sigma, b"sigma-route-b-2");
    assert_ne!(
        c1.value, c2.value,
        "independent transcripts, distinct points"
    );

    // The merge-node operation: fold 2 -> 1, verify the fold, discharge the
    // folded claim — the root's single evaluation.
    let claims = [c1, c2];
    let n_cols = 1usize << (mu - nu);
    let combs: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| FoldMatrix::col_marginal(&m, &c.row.materialize(), n_cols))
        .collect();
    let mut chp = FsChallenger::new(b"sigma-route-b-fold");
    let (fproof, folded_p) = prove_fold(&m, &combs, &claims, &mut chp);
    let mut chv = FsChallenger::new(b"sigma-route-b-fold");
    let folded_v = verify_fold(&claims, &fproof, &mut chv).expect("the sigma fold verifies");
    assert_eq!(folded_p, folded_v, "prover and verifier agree on the fold");
    assert_eq!(
        bilinear(&folded_v.row, &folded_v.col, &m),
        folded_v.value,
        "the folded sigma claim discharges — the root evaluation"
    );

    // A wrong claimed value must fail the discharge.
    let mut bad = folded_v;
    bad.value += F128::ONE;
    assert_ne!(
        bilinear(&bad.row, &bad.col, &m),
        bad.value,
        "a tampered sigma claim fails the discharge"
    );
}
