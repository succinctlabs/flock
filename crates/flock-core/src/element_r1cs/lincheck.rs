//! Phase 2 of the element PIOP: the batched lincheck.
//!
//! The zerocheck leaves two claims at `r = (r_row, r_con)` — `Âz(r)` and
//! `B̂z(r)`, once the verifier has stripped the affine constants. The verifier
//! samples `α` and one degree-2 sumcheck reduces both to a single witness claim.
//!
//! ## Collapsing the row half
//!
//! The statement to reduce is
//!
//! ```text
//! Σ_y (Â + α·B̂)(r, y) · ẑ(y)  =  va + α·vb,      y = (y_row, y_col)
//! ```
//!
//! but it does **not** need a sumcheck over `y_row`. Because the full system is
//! `I_{2^n_log} ⊗ A_0`, the MLE splits as
//! `Â((x_row,x_con),(y_row,y_col)) = eq(x_row,y_row)·Â_0(x_con,y_col)`, and the
//! `eq(r_row, y_row)` factor sums the row variables away in closed form:
//!
//! ```text
//! Σ_y Â(r,y)·ẑ(y) = Σ_c Â_0(r_con, c) · zc[c],
//! zc[c] = Σ_j eq(r_row, j) · z[(c << n_log) + j]  =  ẑ(r_row, c)
//! ```
//!
//! That is an **exact algebraic identity**, not a probabilistic reduction, so
//! collapsing it costs nothing in soundness. What is left is a sumcheck over the
//! `kappa` column variables only:
//!
//! ```text
//! Σ_c comb[c] · zc[c] = va + α·vb,
//! comb[c] = Σ_con eq_con[con] · (A_0 + α·B_0)[con, c]
//! ```
//!
//! `comb` is an `O(nnz)` base-block marginal — the same comb shape as the
//! boolean lincheck's, but tiny (a few entries per gate type). `zc` is one
//! partial fold of the witness at `r_row`, exactly what the boolean lincheck's
//! `partial_fold_packed_z` produces for its own outer half.
//!
//! This is the structure the boolean lincheck already has: the output claim
//! **reuses** the incoming point's row coordinates rather than sampling fresh
//! ones, so the claim point is `(r_row, r'_col)` and the verifier's final check
//! is just `(Â_0 + α·B̂_0)(r_con, r'_col) · ẑ(r_row, r'_col)` — no
//! `eq(r_row, r'_row)` factor, because there is no `r'_row`.
//!
//! Cost, versus a sumcheck over all of `y`: `kappa` rounds instead of
//! `kappa + n_log` (so `2·n_log` fewer field elements on the wire), one
//! `O(2^m_words)` fold pass instead of `m_words` of them, and no
//! full-domain weight table to materialize at all.
//!
//! The sumcheck loop itself is the boolean lincheck's calibrated
//! product-sumcheck core, called directly:
//! [`crate::lincheck::sumcheck_round_eval_par`],
//! [`crate::lincheck::sumcheck_bind_both_and_eval_next`] (fold + next-round
//! message in one pass) and [`crate::lincheck::sumcheck_bind_top_in_place_par`].
//! Rounds bind the **top** remaining variable, so the challenge list reversed is
//! the column point LSB-first — matching the rows-low witness layout.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::ElementTableType;
use super::Grinding;
use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::{
    sumcheck_bind_both_and_eval_next, sumcheck_bind_top_in_place_par, sumcheck_round_eval_par,
};
use crate::zerocheck::univariate_skip::build_eq;

/// Domain label of the standalone single-table lincheck. The union's
/// element-region lincheck runs the same sumcheck core under its own label
/// (see [`column_sumcheck_prove`] / [`column_sumcheck_replay`]).
pub const LABEL: &[u8] = b"flock-element-lc-v0";

// ---------------------------------------------------------------------------
// The shared product-sumcheck core. Both element linchecks — the standalone
// one below and the union's element-region one (`super::union`) — reduce the
// SAME shape: a length-`2^rounds` weight vector against a length-`2^rounds`
// row-collapsed witness vector. Factored rather than copy-pasted: this
// codebase has a documented drift bug from duplicated sumcheck loops.
// ---------------------------------------------------------------------------

/// Run the product sumcheck `Σ_c comb[c]·g[c]`, binding the TOP remaining
/// variable each round. Consumes both vectors down to length 1 and returns the
/// per-round `(q(1), q(∞))` messages plus the challenges **in binding order**
/// (top variable first); `g[0]` afterwards is `ĝ(r'_col)`.
///
/// `rounds = log2(comb.len())`. The `rounds == 0` case (a one-column block) is
/// a no-op: the "sumcheck" is the bare claim.
pub(crate) fn column_sumcheck_prove<C: Challenger>(
    comb: &mut Vec<F128>,
    g: &mut Vec<F128>,
    grinding: Grinding,
    grinding_nonces: &mut Vec<u64>,
    ch: &mut C,
) -> (Vec<(F128, F128)>, Vec<F128>) {
    debug_assert_eq!(comb.len(), g.len());
    let rounds = comb.len().trailing_zeros() as usize;
    debug_assert_eq!(comb.len(), 1usize << rounds);
    let mut msgs = Vec::with_capacity(rounds);
    let mut challenges = Vec::with_capacity(rounds);
    if rounds == 0 {
        return (msgs, challenges);
    }
    // Round 0's message is the only standalone pass; every later message
    // falls out of binding the previous round.
    let (mut e1, mut einf) = sumcheck_round_eval_par(comb, g);
    for t in 0..rounds {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let rho = if let Some(bits) = grinding.round_bits() {
            let (nonce, rho) = ch.grind_pow_and_sample_f128(bits);
            grinding_nonces.push(nonce);
            rho
        } else {
            ch.sample_f128()
        };
        msgs.push((e1, einf));
        challenges.push(rho);
        if t + 1 < rounds {
            let (n1, ninf) = sumcheck_bind_both_and_eval_next(comb, g, rho);
            e1 = n1;
            einf = ninf;
        } else {
            sumcheck_bind_top_in_place_par(comb, rho);
            sumcheck_bind_top_in_place_par(g, rho);
        }
    }
    (msgs, challenges)
}

/// Verifier mirror of [`column_sumcheck_prove`]: replay the rounds from
/// `target` and return the residual claim plus the challenges in binding order.
/// `q(0) = running + q(1)` in char 2, then `q(X) = einf·X² + c1·X + q(0)`.
pub(crate) fn column_sumcheck_replay<C: Challenger>(
    target: F128,
    rounds: &[(F128, F128)],
    grinding: Grinding,
    grinding_nonces: &[u64],
    nonce_idx: &mut usize,
    ch: &mut C,
) -> Result<(F128, Vec<F128>), VerifyError> {
    let mut running = target;
    let mut challenges = Vec::with_capacity(rounds.len());
    for &(e1, einf) in rounds {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let rho = if let Some(bits) = grinding.round_bits() {
            let rho = ch
                .verify_pow_and_sample_f128(grinding_nonces[*nonce_idx], bits)
                .ok_or(VerifyError::InvalidGrindingNonce { which: "round" })?;
            *nonce_idx += 1;
            rho
        } else {
            ch.sample_f128()
        };
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * rho * rho + c1 * rho + e0;
        challenges.push(rho);
    }
    Ok((running, challenges))
}

/// Round messages plus the output witness claim value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    /// Per-round `(q(1), q(∞))`, length **`kappa`** — the column variables only;
    /// the row variables are summed away in closed form (see the module docs).
    /// Top variable bound first.
    pub rounds: Vec<(F128, F128)>,
    /// `ẑ(r_row, r'_col)` — the second packed-direct claim.
    pub z_eval: F128,
    /// Per element slot in region order, the UNSCALED bilinear forms
    /// `(⟨eq_con ⊗ eq_col, A_0⟩, ⟨…, B_0⟩)` — the matrix work, split so it
    /// can be accumulated instead of evaluated by the verifier.
    ///
    /// Element combs are small, which is beside the point: arithmetising one
    /// is nnz-PRESERVING (the gadget's matrix is the matrix it evaluates), so
    /// a recursion circuit that pays it inline cannot close its fixed point
    /// regardless of size. Same treatment as the boolean class's
    /// `lincheck::LincheckProof::matrix_evals`.
    pub matrix_evals: Vec<(F128, F128)>,
    /// α batching then one PoW nonce per product-sumcheck round.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// What a verified lincheck leaves for the opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// The claim point `(r_row, r'_col)`, LSB-first (rows low), length
    /// `n_log + kappa`. The low `n_log` coordinates are **inherited** from the
    /// zerocheck's point — only the `kappa` column coordinates are fresh.
    pub r_prime: Vec<F128>,
    pub z_eval: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// Wrong number of round messages (expected `kappa`).
    BadRoundCount {
        expected: usize,
        got: usize,
    },
    /// `r` (the zerocheck point) has the wrong length for this statement.
    BadPointLength {
        expected: usize,
        got: usize,
    },
    BadGrindingNonceCount {
        expected: usize,
        got: usize,
    },
    InvalidGrindingNonce {
        which: &'static str,
    },
    /// The final consistency check
    /// `running == (Â_0 + α·B̂_0)(r_con, r'_col) · z_eval` failed.
    SumcheckFinalFailed,
}

/// Prove the batched lincheck.
///
/// `r` is the zerocheck's claim point (LSB-first, length `kappa + n_log`), and
/// `va`, `vb` are the constant-stripped `Âz(r)`, `B̂z(r)` claims. `z` is the
/// committed witness, which the caller keeps for the opening.
pub fn prove<C: Challenger>(
    ty: &ElementTableType,
    z: &[F128],
    n_log: usize,
    r: &[F128],
    va: F128,
    vb: F128,
    ch: &mut C,
) -> (Proof, Claim) {
    prove_with_grinding(ty, z, n_log, r, va, vb, Grinding::disabled(), ch)
}

/// [`prove`] with an explicit element grinding policy.
pub fn prove_with_grinding<C: Challenger>(
    ty: &ElementTableType,
    z: &[F128],
    n_log: usize,
    r: &[F128],
    va: F128,
    vb: F128,
    grinding: Grinding,
    ch: &mut C,
) -> (Proof, Claim) {
    let kappa = ty.kappa();
    let m_words = kappa + n_log;
    assert_eq!(r.len(), m_words, "zerocheck point length");
    assert_eq!(z.len(), 1usize << m_words, "witness length");

    ch.observe_label(LABEL);
    let mut grinding_nonces = Vec::with_capacity(grinding.lincheck_nonce_count(kappa));
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let (nonce, alpha) = ch.grind_pow_and_sample_f128(bits);
        grinding_nonces.push(nonce);
        alpha
    } else {
        ch.sample_f128()
    };

    // Rows live in the LOW coordinates of the point, columns in the high ones.
    let (r_row, r_con) = r.split_at(n_log);
    let mut comb = comb_vector(ty, alpha, &build_eq(r_con));
    // The row collapse: one pass over the witness, `2^kappa` outputs.
    let mut zc = partial_fold_rows(z, r_row);
    debug_assert_eq!(
        comb.iter()
            .zip(&zc)
            .fold(F128::ZERO, |a, (x, y)| a + *x * *y),
        va + alpha * vb,
        "lincheck target must be the honest weighted inner product"
    );

    // The shared product sumcheck over the column variables, top first.
    let (rounds, r_rounds) =
        column_sumcheck_prove(&mut comb, &mut zc, grinding, &mut grinding_nonces, ch);
    debug_assert_eq!(r_rounds.len(), kappa);
    debug_assert_eq!(zc.len(), 1);
    let z_eval = zc[0];

    // The standalone element lincheck does not accumulate; the union path
    // fills this in (see `union::prove`).
    let proof = Proof {
        rounds,
        z_eval,
        matrix_evals: Vec::new(),
        grinding_nonces,
    };
    let claim = Claim {
        r_prime: claim_point(r_row, r_rounds),
        z_eval,
    };
    (proof, claim)
}

/// Verify a lincheck proof, walking the challenger in lockstep with [`prove`].
pub fn verify<C: Challenger>(
    ty: &ElementTableType,
    n_log: usize,
    r: &[F128],
    va: F128,
    vb: F128,
    proof: &Proof,
    ch: &mut C,
) -> Result<Claim, VerifyError> {
    verify_with_grinding(ty, n_log, r, va, vb, proof, Grinding::disabled(), ch)
}

/// [`verify`] with an explicit element grinding policy.
pub fn verify_with_grinding<C: Challenger>(
    ty: &ElementTableType,
    n_log: usize,
    r: &[F128],
    va: F128,
    vb: F128,
    proof: &Proof,
    grinding: Grinding,
    ch: &mut C,
) -> Result<Claim, VerifyError> {
    let kappa = ty.kappa();
    let m_words = kappa + n_log;
    if r.len() != m_words {
        return Err(VerifyError::BadPointLength {
            expected: m_words,
            got: r.len(),
        });
    }
    if proof.rounds.len() != kappa {
        return Err(VerifyError::BadRoundCount {
            expected: kappa,
            got: proof.rounds.len(),
        });
    }
    if proof.grinding_nonces.len() != grinding.lincheck_nonce_count(kappa) {
        return Err(VerifyError::BadGrindingNonceCount {
            expected: grinding.lincheck_nonce_count(kappa),
            got: proof.grinding_nonces.len(),
        });
    }

    ch.observe_label(LABEL);
    let mut nonce_idx = 0;
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let alpha = ch
            .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], bits)
            .ok_or(VerifyError::InvalidGrindingNonce { which: "alpha" })?;
        nonce_idx += 1;
        alpha
    } else {
        ch.sample_f128()
    };

    // Replay the shared product sumcheck.
    let (running, r_rounds) = column_sumcheck_replay(
        va + alpha * vb,
        &proof.rounds,
        grinding,
        &proof.grinding_nonces,
        &mut nonce_idx,
        ch,
    )?;
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());
    let (r_row, r_con) = r.split_at(n_log);
    let r_prime = claim_point(r_row, r_rounds);

    // Final check: `(Â_0 + α·B̂_0)(r_con, r'_col)` in O(2^kappa + nnz) — the
    // same `comb` marginal the prover built, evaluated against `eq(r'_col)`.
    // There is no `eq(r_row, r'_row)` factor: the row coordinates were never
    // resampled, they are `r_row` itself.
    let comb = comb_vector(ty, alpha, &build_eq(r_con));
    let eq_col = build_eq(&r_prime[n_log..]);
    let base = comb
        .iter()
        .zip(&eq_col)
        .fold(F128::ZERO, |acc, (c, e)| acc + *c * *e);
    if running != base * proof.z_eval {
        return Err(VerifyError::SumcheckFinalFailed);
    }

    Ok(Claim {
        r_prime,
        z_eval: proof.z_eval,
    })
}

/// The output claim point `(r_row, r'_col)`, LSB-first.
///
/// `r_row` is inherited from the zerocheck; `col_rounds` are the sumcheck
/// challenges in **binding** order, which is top-variable-first, so round `t`
/// bound column bit `kappa − 1 − t` and reversing puts them LSB-first.
fn claim_point(r_row: &[F128], col_rounds: Vec<F128>) -> Vec<F128> {
    let mut point = Vec::with_capacity(r_row.len() + col_rounds.len());
    point.extend_from_slice(r_row);
    point.extend(col_rounds.into_iter().rev());
    point
}

/// `comb[c] = Σ_con eq_con[con] · (A_0 + α·B_0)[con, c]` — the eq-weighted
/// column marginal of the base block. `O(nnz(A_0) + nnz(B_0))`; both prover and
/// verifier call this, so there is one definition to disagree with.
fn comb_vector(ty: &ElementTableType, alpha: F128, eq_con: &[F128]) -> Vec<F128> {
    debug_assert_eq!(eq_con.len(), ty.width());
    let mut comb = vec![F128::ZERO; ty.width()];
    for (m, scale) in [(ty.a_0(), F128::ONE), (ty.b_0(), alpha)] {
        for (con, row) in m.rows.iter().enumerate() {
            if row.is_empty() {
                continue;
            }
            let w = scale * eq_con[con];
            for &(c, coeff) in row {
                comb[c] += w * coeff;
            }
        }
    }
    comb
}

/// The row collapse: `zc[c] = Σ_j eq(r_row, j) · z[(c << n_log) + j] = ẑ(r_row, c)`,
/// a length-`2^kappa` vector.
///
/// Rows-low layout means each column's rows are contiguous, so this is one
/// chunked dot product — `2^m_words` multiplications in a single pass, split
/// across columns. This replaces both the full-domain weight table and the
/// `n_log` row rounds of the uncollapsed protocol.
///
/// Takes the row **point**, not a prebuilt eq table, so no caller can pass the
/// wrong one of the two.
///
/// Parallelism is over columns, so it engages `2^kappa` threads — ample for a
/// wide block, but at the `kappa = 2` smoke shape only four. This pass is now
/// essentially the whole cost of the phase (~0.85 ms of a ~3 ms PIOP at
/// `n_log = 16`), so if the phase ever needs tuning, splitting by row block and
/// reducing per-column accumulators is the thing to try. Left simple: the
/// milestone is 3× inside its target.
fn partial_fold_rows(z: &[F128], r_row: &[F128]) -> Vec<F128> {
    let eq_row = crate::pcs::ring_switch::build_eq_parallel(r_row);
    let rows = eq_row.len();
    debug_assert_eq!(z.len() % rows, 0);
    z.par_chunks(rows)
        .map(|col| {
            col.iter()
                .zip(&eq_row)
                .fold(F128::ZERO, |acc, (v, e)| acc + *v * *e)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::element_r1cs::broadcast_add;
    use crate::element_r1cs::tests::{mixed_gate, mixed_witness, mult_gate, mult_witness};
    use crate::test_rng::Rng;
    use crate::zerocheck::multilinear::eq_eval;

    /// Direct MLE evaluation at `point`, binding the low variable first.
    fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
        let mut t = table.to_vec();
        for &p in point {
            crate::zerocheck::multilinear::fold_in_place_single(&mut t, p);
        }
        t[0]
    }

    /// `(Âz(r), B̂z(r))` straight from the matrices — the claims the lincheck is
    /// supposed to reduce, computed without any of its machinery.
    fn true_claims(ty: &ElementTableType, z: &[F128], n_log: usize, r: &[F128]) -> (F128, F128) {
        let (az, bz) = ty.apply(z, n_log);
        (mle_eval(&az, r), mle_eval(&bz, r))
    }

    fn bits(v: usize, n: usize) -> Vec<F128> {
        (0..n)
            .map(|i| {
                if (v >> i) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            })
            .collect()
    }

    /// Brute-force `Σ_y (Â + α·B̂)(r, y)·ẑ(y)` from the *unfactored* definition:
    /// walk every `(x, y)` pair of the block-diagonal system explicitly. This is
    /// the independent check on the row collapse — if the closed-form row sum or
    /// the index convention were wrong, this disagrees.
    fn brute_force_weighted_sum(
        ty: &ElementTableType,
        z: &[F128],
        n_log: usize,
        r: &[F128],
        alpha: F128,
    ) -> F128 {
        let width = ty.width();
        let rows = 1usize << n_log;
        let (r_row, r_con) = r.split_at(n_log);
        let mut acc = F128::ZERO;
        // Σ_x eq(r, x) Σ_y M[x, y] z[y], with M = I ⊗ (A_0 + αB_0):
        // x = (x_row, con), y = (x_row, c) — the identity factor forces the rows
        // to agree, which is exactly what `eq(x_row, y_row)` encodes.
        for x_row in 0..rows {
            let eq_x_row = eq_eval(r_row, &bits(x_row, n_log));
            for con in 0..width {
                let eq_x_con = eq_eval(r_con, &bits(con, ty.kappa()));
                let mut inner = F128::ZERO;
                for (m, scale) in [(ty.a_0(), F128::ONE), (ty.b_0(), alpha)] {
                    for &(c, coeff) in &m.rows[con] {
                        inner += scale * coeff * z[(c << n_log) + x_row];
                    }
                }
                acc += eq_x_row * eq_x_con * inner;
            }
        }
        acc
    }

    /// The collapsed inner product `Σ_c comb[c]·zc[c]` must equal the
    /// brute-force block-diagonal sum, and must equal `va + α·vb` for the true
    /// `Âz(r)`, `B̂z(r)`. This is the load-bearing identity behind dropping the
    /// row rounds — the sumcheck below only ever sees `kappa` variables, so if
    /// the collapse were wrong nothing else would catch it.
    #[test]
    fn row_collapse_matches_brute_force() {
        let mut rng = Rng::new(1234);
        for (ty, kappa) in [(mult_gate(2), 2usize), (mixed_gate(&mut rng), 3)] {
            for n_log in [1usize, 2, 4] {
                let m_words = kappa + n_log;
                // Random z — the identity is about the weights, not satisfaction.
                let z: Vec<F128> = (0..1usize << m_words).map(|_| rng.f128()).collect();
                let r: Vec<F128> = (0..m_words).map(|_| rng.f128()).collect();
                let alpha = rng.f128();

                let (r_row, r_con) = r.split_at(n_log);
                let comb = comb_vector(&ty, alpha, &build_eq(r_con));
                let zc = partial_fold_rows(&z, r_row);
                assert_eq!(zc.len(), ty.width());
                let collapsed = comb
                    .iter()
                    .zip(&zc)
                    .fold(F128::ZERO, |a, (x, y)| a + *x * *y);

                assert_eq!(
                    collapsed,
                    brute_force_weighted_sum(&ty, &z, n_log, &r, alpha),
                    "κ={kappa} n_log={n_log}: collapsed weights vs brute force"
                );
                let (va, vb) = true_claims(&ty, &z, n_log, &r);
                assert_eq!(
                    collapsed,
                    va + alpha * vb,
                    "κ={kappa} n_log={n_log}: target vs Âz(r) + α·B̂z(r)"
                );
            }
        }
    }

    /// The collapsed vector really is the witness restricted to `r_row`:
    /// `zc[c] = ẑ(r_row, c)` for every boolean column `c`. That identity is what
    /// makes the output claim `ẑ(r_row, r'_col)` an evaluation of the *committed*
    /// polynomial, hence openable by the PCS.
    #[test]
    fn partial_fold_is_the_witness_at_r_row() {
        let mut rng = Rng::new(4711);
        for (kappa, n_log) in [(2usize, 3usize), (3, 4), (1, 5)] {
            let width = 1usize << kappa;
            let z: Vec<F128> = (0..width << n_log).map(|_| rng.f128()).collect();
            let r_row: Vec<F128> = (0..n_log).map(|_| rng.f128()).collect();
            let zc = partial_fold_rows(&z, &r_row);
            for c in 0..width {
                let mut point = r_row.clone();
                point.extend(bits(c, kappa));
                assert_eq!(zc[c], mle_eval(&z, &point), "κ={kappa} c={c}");
            }
            // …and therefore its MLE at a random column point is ẑ(r_row, ·).
            let r_col: Vec<F128> = (0..kappa).map(|_| rng.f128()).collect();
            let mut point = r_row.clone();
            point.extend_from_slice(&r_col);
            assert_eq!(mle_eval(&zc, &r_col), mle_eval(&z, &point), "κ={kappa}");
        }
    }

    /// **Differential test** on random instances: the prover's round messages
    /// must be the honest sumcheck of the true weighted inner product. Replaying
    /// the verifier's chain from the *brute-force* target must land on
    /// `ĉomb(r'_col)·ẑ(r_row, r'_col)`, and `z_eval` must be the witness MLE at
    /// the claim point.
    #[test]
    fn round_messages_match_brute_force_on_random_instances() {
        let mut rng = Rng::new(99);
        for (ty, kappa) in [(mult_gate(2), 2usize), (mixed_gate(&mut rng), 3)] {
            for n_log in [1usize, 3, 5] {
                let m_words = kappa + n_log;
                let z: Vec<F128> = (0..1usize << m_words).map(|_| rng.f128()).collect();
                let r: Vec<F128> = (0..m_words).map(|_| rng.f128()).collect();
                let (va, vb) = true_claims(&ty, &z, n_log, &r);

                let mut ch = FsChallenger::new(b"element-lc-diff");
                let (proof, claim) = prove(&ty, &z, n_log, &r, va, vb, &mut ch);

                // The wire cost is now kappa rounds, not kappa + n_log.
                assert_eq!(proof.rounds.len(), kappa, "round count is kappa only");
                // The row coordinates are inherited, not resampled.
                assert_eq!(&claim.r_prime[..n_log], &r[..n_log], "rows inherited");

                // Re-derive α as the prover did.
                let mut ch2 = FsChallenger::new(b"element-lc-diff");
                ch2.observe_label(LABEL);
                let alpha = ch2.sample_f128();

                assert_eq!(
                    claim.z_eval,
                    mle_eval(&z, &claim.r_prime),
                    "κ={kappa} n_log={n_log}: z_eval is ẑ(r_row, r'_col)"
                );

                let mut running = brute_force_weighted_sum(&ty, &z, n_log, &r, alpha);
                // Challenges in binding order are the reverse of the column half.
                let bind_order: Vec<F128> = claim.r_prime[n_log..].iter().rev().copied().collect();
                for (&(e1, einf), &rho) in proof.rounds.iter().zip(&bind_order) {
                    let e0 = running + e1;
                    let c1 = e0 + e1 + einf;
                    running = einf * rho * rho + c1 * rho + e0;
                }
                let comb = comb_vector(&ty, alpha, &build_eq(&r[n_log..]));
                let eq_col = build_eq(&claim.r_prime[n_log..]);
                let base = comb
                    .iter()
                    .zip(&eq_col)
                    .fold(F128::ZERO, |a, (c, e)| a + *c * *e);
                assert_eq!(
                    running,
                    base * claim.z_eval,
                    "κ={kappa} n_log={n_log}: chain from brute-force target"
                );
            }
        }
    }

    /// Prove → verify roundtrip on satisfying witnesses, several shapes.
    #[test]
    fn prove_verify_roundtrip_honest() {
        let mut rng = Rng::new(555);
        for (n_log, n) in [(1usize, 1usize), (3, 5), (4, 16), (6, 41)] {
            let ty = mult_gate(2);
            let z = mult_witness(&ty, n_log, n, &mut rng);
            let r: Vec<F128> = (0..2 + n_log).map(|_| rng.f128()).collect();
            let (va, vb) = true_claims(&ty, &z, n_log, &r);

            let mut ch_p = FsChallenger::new(b"element-lc-rt");
            let (proof, claim_p) = prove(&ty, &z, n_log, &r, va, vb, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"element-lc-rt");
            let claim_v = verify(&ty, n_log, &r, va, vb, &proof, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected n_log={n_log} n={n}: {e:?}"));
            assert_eq!(claim_p, claim_v, "n_log={n_log} n={n}");
        }
    }

    /// The mixed-gate table (free wires, mult, mult-acc, linear pin, padding)
    /// round-trips too, and the honest witness's claims are consistent with the
    /// zerocheck's constant stripping.
    #[test]
    fn prove_verify_roundtrip_mixed_gate() {
        let mut rng = Rng::new(556);
        let ty = mixed_gate(&mut rng);
        let (n_log, n) = (4usize, 13usize);
        let z = mixed_witness(&ty, n_log, n, &mut rng);
        assert!(ty.satisfies(&z, n_log, n));
        // Sanity: the constants really are row-uniform, so `pa − Az` is the
        // broadcast constant vector.
        let (az, _) = ty.apply(&z, n_log);
        let mut pa = az.clone();
        broadcast_add(&mut pa, ty.a_const(), n_log);

        let r: Vec<F128> = (0..ty.kappa() + n_log).map(|_| rng.f128()).collect();
        let (va, vb) = true_claims(&ty, &z, n_log, &r);
        let mut ch_p = FsChallenger::new(b"element-lc-mixed");
        let (proof, claim_p) = prove(&ty, &z, n_log, &r, va, vb, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"element-lc-mixed");
        let claim_v = verify(&ty, n_log, &r, va, vb, &proof, &mut ch_v).expect("verify");
        assert_eq!(claim_p, claim_v);
    }

    /// A single-column table (`kappa = 1`) still has a well-formed one-round
    /// sumcheck — the loop's fused branch is never taken there, so it is the
    /// edge the `t + 1 < kappa` guard exists for.
    #[test]
    fn kappa_one_roundtrips() {
        let mut rng = Rng::new(31337);
        // kappa = 1: one column, a free wire (tautology row).
        let mut b = crate::element_r1cs::ElementTableBuilder::new(1);
        b.free_wire(0);
        let ty = b.build().expect("free wire is valid");
        let n_log = 4usize;
        let z: Vec<F128> = (0..1usize << (1 + n_log)).map(|_| rng.f128()).collect();
        let r: Vec<F128> = (0..1 + n_log).map(|_| rng.f128()).collect();
        let (va, vb) = true_claims(&ty, &z, n_log, &r);

        let mut ch_p = FsChallenger::new(b"element-lc-k1");
        let (proof, claim_p) = prove(&ty, &z, n_log, &r, va, vb, &mut ch_p);
        assert_eq!(proof.rounds.len(), 1);
        let mut ch_v = FsChallenger::new(b"element-lc-k1");
        let claim_v = verify(&ty, n_log, &r, va, vb, &proof, &mut ch_v).expect("verify");
        assert_eq!(claim_p, claim_v);
        assert_eq!(claim_v.z_eval, mle_eval(&z, &claim_v.r_prime));
    }

    /// Tamper matrix: every round message, `z_eval`, and the incoming claims
    /// must all be pinned.
    #[test]
    fn verify_rejects_mutations() {
        let mut rng = Rng::new(31);
        let (n_log, n) = (4usize, 11usize);
        let ty = mult_gate(2);
        let z = mult_witness(&ty, n_log, n, &mut rng);
        let r: Vec<F128> = (0..2 + n_log).map(|_| rng.f128()).collect();
        let (va, vb) = true_claims(&ty, &z, n_log, &r);
        let mut ch_p = FsChallenger::new(b"element-lc-mut");
        let (proof, _) = prove(&ty, &z, n_log, &r, va, vb, &mut ch_p);

        let mut cases: Vec<(String, Proof)> = Vec::new();
        for i in 0..proof.rounds.len() {
            for which in 0..2 {
                let mut bad = proof.clone();
                if which == 0 {
                    bad.rounds[i].0 += F128::ONE;
                } else {
                    bad.rounds[i].1 += F128::ONE;
                }
                cases.push((format!("round {i} msg {which}"), bad));
            }
        }
        let mut bad = proof.clone();
        bad.z_eval += F128::ONE;
        cases.push(("z_eval".to_string(), bad));
        let mut bad = proof.clone();
        bad.rounds.pop();
        cases.push(("truncated rounds".to_string(), bad));

        for (name, bad) in cases {
            let mut ch = FsChallenger::new(b"element-lc-mut");
            assert!(
                verify(&ty, n_log, &r, va, vb, &bad, &mut ch).is_err(),
                "verify accepted mutation: {name}"
            );
        }

        // Wrong incoming claims: the sumcheck target is wrong from round 0.
        for (name, (bva, bvb)) in [("va", (va + F128::ONE, vb)), ("vb", (va, vb + F128::ONE))] {
            let mut ch = FsChallenger::new(b"element-lc-mut");
            assert!(
                verify(&ty, n_log, &r, bva, bvb, &proof, &mut ch).is_err(),
                "verify accepted wrong claim: {name}"
            );
        }
    }
}
