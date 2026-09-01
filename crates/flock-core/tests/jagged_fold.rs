//! The jagged layout table J — the fourth assertion family's native chain
//! (the count win): the assist verifier's count-dependent `W`-values as
//! claims on the layout, the merge-node fold, the root discharge, and the
//! key discipline (a claim discharges only against its own heights).
//!
//! The direct-formula reference here is the per-COLUMN enumeration written
//! independently of `matrix_fold`'s per-run walk; the tie to the production
//! verifier's own statements (`frobenius_statements`) lands with the
//! assertion-export step, mirroring how sigma route B was pinned.

use flock_core::{
    challenger::FsChallenger,
    field::F128,
    matrix_fold::{
        FoldError, JaggedClaim, JaggedRowWeight, JaggedTable, MatrixClaim, discharge_jagged,
        prove_fold_jagged, verify_fold_jagged,
    },
    pcs::jagged::JaggedParams,
};

struct Rng(u64);
impl Rng {
    fn f128(&mut self) -> F128 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let lo = z ^ (z >> 31);
        self.0 = self.0.wrapping_add(0x1234_5678_9ABC_DEF0);
        let mut w = self.0;
        w = (w ^ (w >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        F128::new(lo, w ^ (w >> 32))
    }
    fn f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.f128()).collect()
    }
}

/// A jagged layout from explicit heights — zero-height columns and the
/// zero tail included, exactly the shapes `assist_boundaries` collapses.
fn params_from_heights(heights: &[u64], n: usize, m: usize) -> JaggedParams {
    assert!(heights.len().is_power_of_two());
    let k = heights.len().trailing_zeros() as usize;
    let mut prefix = Vec::with_capacity(heights.len() + 1);
    let mut acc = 0u64;
    prefix.push(0);
    for &h in heights {
        acc += h;
        prefix.push(acc);
    }
    assert!(acc <= 1 << m, "area exceeds the dense bound");
    JaggedParams {
        n,
        k,
        m,
        col_prefix_sums: prefix,
    }
}

/// The direct formula, per COLUMN:
/// `Σ_y w_y · Π_ℓ eq(t_{y-1}[ℓ], ρ_{c,ℓ}) · eq(t_y[ℓ], ρ_{d,ℓ})` with `ρ`
/// interleaved `(c_0, d_0, c_1, d_1, …)`.
fn w_reference(weights: &[F128], prefix: &[u64], m: usize, rho: &[F128]) -> F128 {
    assert_eq!(rho.len(), 2 * (m + 1));
    let mut acc = F128::ZERO;
    for (y, &w) in weights.iter().enumerate() {
        let (t_c, t_next) = (prefix[y], prefix[y + 1]);
        let mut term = w;
        for l in 0..=m {
            let (rc, rd) = (rho[2 * l], rho[2 * l + 1]);
            term *= if (t_c >> l) & 1 == 1 {
                rc
            } else {
                F128::ONE + rc
            };
            term *= if (t_next >> l) & 1 == 1 {
                rd
            } else {
                F128::ONE + rd
            };
        }
        acc += term;
    }
    acc
}

/// Dense eq tensor over `2^k`, built independently of the src doubling.
fn dense_eq(point: &[F128]) -> Vec<F128> {
    let mut out = vec![F128::ONE];
    for &p in point {
        let mut next = vec![F128::ZERO; out.len() * 2];
        for (i, &e) in out.iter().enumerate() {
            next[i] = e * (F128::ONE + p);
            next[i + out.len()] = e * p;
        }
        out = next;
    }
    out
}

/// The realistic mixed shape: live runs, interior zero-height columns, a
/// zero tail. Area 45 in a 2^6 dense bound; 32 layout columns.
const HEIGHTS: [u64; 32] = [
    3, 7, 0, 5, 1, 0, 0, 9, 4, 0, 2, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const N: usize = 4;
const M: usize = 6;

#[test]
fn jagged_claims_match_the_direct_formula() {
    let params = params_from_heights(&HEIGHTS, N, M);
    let table = JaggedTable::from_params(&params);
    let mut rng = Rng(0x1A66_0001);
    let rho = rng.f128_vec(2 * (M + 1));

    // The general statement's shape: row weight eq(z_col).
    let z_col = rng.f128_vec(params.k);
    let eq_claim = JaggedClaim::honest(JaggedRowWeight::eq(z_col.clone()), rho.clone(), &table);
    assert_eq!(
        eq_claim.value,
        w_reference(&dense_eq(&z_col), &params.col_prefix_sums, M, &rho),
        "eq-weight claim matches the per-column enumeration"
    );
    assert!(eq_claim.check_direct(&table));

    // A scalar group's shape: γ powers at scattered constant addresses,
    // including a zero-height column and the tail.
    let gamma = rng.f128();
    let addrs: [u32; 6] = [0, 3, 7, 9, 20, 31];
    let mut coeff = F128::ONE;
    let mut terms = Vec::new();
    let mut dense = vec![F128::ZERO; HEIGHTS.len()];
    for &a in &addrs {
        terms.push((coeff, a));
        dense[a as usize] += coeff;
        coeff *= gamma;
    }
    let combo_claim = JaggedClaim::honest(JaggedRowWeight::Combo(terms), rho.clone(), &table);
    assert_eq!(
        combo_claim.value,
        w_reference(&dense, &params.col_prefix_sums, M, &rho),
        "combo-weight claim matches the per-column enumeration"
    );
    assert!(combo_claim.check_direct(&table));

    let mut bad = combo_claim;
    bad.value += F128::ONE;
    assert!(!bad.check_direct(&table), "a lying W-value fails discharge");
}

/// Fresh claims from one "proof": the element pair's eq shape plus a scalar
/// group's combo shape, both at the proof's single assist point.
fn proof_claims(table: &JaggedTable, rng: &mut Rng) -> Vec<JaggedClaim> {
    let rho = rng.f128_vec(table.n_col_vars());
    let z_col = rng.f128_vec(table.k);
    let gamma = rng.f128();
    let mut coeff = F128::ONE;
    let terms: Vec<(F128, u32)> = [1u32, 4, 8, 11, 15]
        .iter()
        .map(|&a| {
            let t = (coeff, a);
            coeff *= gamma;
            t
        })
        .collect();
    vec![
        JaggedClaim::honest(JaggedRowWeight::eq(z_col), rho.clone(), table),
        JaggedClaim::honest(JaggedRowWeight::Combo(terms), rho, table),
    ]
}

#[test]
fn jagged_fold_verifies_inherits_and_discharges_at_the_root() {
    let params = params_from_heights(&HEIGHTS, N, M);
    let table = JaggedTable::from_params(&params);
    let mut rng = Rng(0x1A66_0002);

    // Two proofs' fresh claims — same table (same heights), distinct points.
    let mut claims = proof_claims(&table, &mut rng);
    claims.extend(proof_claims(&table, &mut rng));

    let mut chp = FsChallenger::new(b"jagged-fold-test");
    let (fproof, folded_p) = prove_fold_jagged(&table, &claims, &mut chp);
    let mut chv = FsChallenger::new(b"jagged-fold-test");
    let folded_v =
        verify_fold_jagged(table.k, &claims, &fproof, &mut chv).expect("the jagged fold verifies");
    assert_eq!(folded_p, folded_v, "prover and verifier agree on the fold");
    assert!(
        discharge_jagged(&folded_v, &table),
        "the folded claim discharges — the root evaluation"
    );

    // Inheritance: the folded output re-enters the next level's fold as a
    // plain-eq claim beside fresh ones — the cross-level shape.
    let mut next = vec![JaggedClaim::from_folded(&folded_v).expect("folded claims are plain eq")];
    next.extend(proof_claims(&table, &mut rng));
    let mut chp = FsChallenger::new(b"jagged-fold-lvl2");
    let (fproof2, folded2) = prove_fold_jagged(&table, &next, &mut chp);
    let mut chv = FsChallenger::new(b"jagged-fold-lvl2");
    let folded2_v = verify_fold_jagged(table.k, &next, &fproof2, &mut chv)
        .expect("the inherited fold verifies");
    assert_eq!(folded2, folded2_v);
    assert!(
        discharge_jagged(&folded2, &table),
        "the inherited fold discharges"
    );

    // A tampered input claim fails the replay against the honest proof.
    let mut lying = claims.clone();
    lying[0].value += F128::ONE;
    let mut chv = FsChallenger::new(b"jagged-fold-test");
    assert!(
        verify_fold_jagged(table.k, &lying, &fproof, &mut chv).is_err(),
        "a tampered input claim fails the fold replay"
    );

    // A tampered folded value fails the discharge.
    let mut bad = folded_v.clone();
    bad.value += F128::ONE;
    assert!(!discharge_jagged(&bad, &table));

    // The key discipline: the same folded claim must NOT discharge against
    // a different layout — claims are about ONE height vector.
    let mut other_heights = HEIGHTS;
    other_heights[3] = 6;
    other_heights[4] = 0;
    let other = JaggedTable::from_params(&params_from_heights(&other_heights, N, M));
    assert!(
        !discharge_jagged(&folded_v, &other),
        "a claim about one height vector fails another's discharge"
    );

    // Malformed shapes are rejected, not folded.
    let mut short = fproof.clone();
    short.row_rounds.pop();
    let mut chv = FsChallenger::new(b"jagged-fold-test");
    assert_eq!(
        verify_fold_jagged(table.k, &claims, &short, &mut chv),
        Err(FoldError::Malformed)
    );
    let mut chv = FsChallenger::new(b"jagged-fold-test");
    assert_eq!(
        verify_fold_jagged(table.k, &[], &fproof, &mut chv),
        Err(FoldError::Malformed)
    );
}

/// The claim value is what the ANCHOR-EXPECT consumes, so the whole family
/// is only sound if a fold cannot silently change the table. Two tables
/// with different heights give different fold outputs for the same input
/// points — the digest key is load-bearing, not decorative.
#[test]
fn distinct_heights_are_distinct_tables() {
    let ta = JaggedTable::from_params(&params_from_heights(&HEIGHTS, N, M));
    let mut other_heights = HEIGHTS;
    other_heights[0] = 2;
    other_heights[1] = 8;
    let tb = JaggedTable::from_params(&params_from_heights(&other_heights, N, M));

    let mut rng = Rng(0x1A66_0003);
    let rho = rng.f128_vec(ta.n_col_vars());
    let z = rng.f128_vec(ta.k);
    let ca = JaggedClaim::honest(JaggedRowWeight::eq(z.clone()), rho.clone(), &ta);
    let cb = JaggedClaim::honest(JaggedRowWeight::eq(z), rho, &tb);
    assert_ne!(
        ca.value, cb.value,
        "the same statement about different heights has different W-values"
    );

    // And a claim built against table A cannot ride a fold over table B.
    // The gate is the VERIFIER, not the discharge: the prover's output is
    // derived from B and the claims' POINTS alone (values enter only through
    // the transcript), so it is an honest B-statement and rightly
    // discharges — but the foreign VALUE is bound into the fold's target,
    // and the replay diverges at the first column round. That replay is
    // exactly what a merge node's circuit runs, so a foreign claim cannot
    // enter an accumulator; if a cheating prover instead forced the replay
    // through, sumcheck soundness lands the lie in the output claim, where
    // the root discharge catches it.
    let claims = vec![ca];
    let mut chp = FsChallenger::new(b"jagged-cross");
    let (fproof, folded) = prove_fold_jagged(&tb, &claims, &mut chp);
    let _: &MatrixClaim = &folded;
    assert!(
        discharge_jagged(&folded, &tb),
        "the honest prover's output is an honest B-statement"
    );
    let mut chv = FsChallenger::new(b"jagged-cross");
    assert_eq!(
        verify_fold_jagged(tb.k, &claims, &fproof, &mut chv),
        Err(FoldError::ConsistencyFailed { which: "col" }),
        "the foreign value diverges the replay — the accumulation gate"
    );
}
