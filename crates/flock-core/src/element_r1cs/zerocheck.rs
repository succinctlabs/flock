//! Phase 1 of the element PIOP: the large-field zerocheck.
//!
//! With `x = [x_row (n_log low bits) | x_con (kappa high bits)]`, the verifier
//! sends `τ` and the prover proves
//!
//! ```text
//! Σ_x eq(τ, x) · ( (Az+a_const)(x) · (Bz+b_const)(x) + z(x) ) = 0
//! ```
//!
//! (char 2, so the relation's `− z` is `+ z`). This is a plain eq-weighted
//! degree-3 sumcheck over `n_log + kappa` rounds — **no univariate skip, no
//! packing, no φ8**. Rounds 2+ of the boolean zerocheck are essentially this
//! protocol, and the round-message/verifier conventions here are deliberately
//! the same so the two verifiers stay structurally parallel:
//!
//! - low bit bound first, so the challenge list *is* the claim point LSB-first;
//! - **Convention A** — the prover sends the bare inner `(G(1), G(∞))` and the
//!   verifier absorbs the current variable's eq factor via the consistency
//!   identity `G_{r-1}(ρ) = (1+τ_r)·G_r(0) + τ_r·G_r(1)`, one inversion per
//!   round (`crate::zerocheck::verify`, zerocheck.rs:835);
//! - the running claim is the inner value, never eq-weighted, so the initial
//!   target is `0` and the final one is `ea·eb + ec`.
//!
//! Output claims at the final point `r = (r_row, r_con)`:
//!
//! - `ea = MLE of (Az+a_const)` and `eb = MLE of (Bz+b_const)` at `r`, which
//!   Phase 2 reduces once the verifier has subtracted the closed-form constants
//!   ([`super::strip_constants`]);
//! - `ec = ẑ(r)`. Because `C = I` this is *directly* a witness evaluation, so it
//!   leaves as a packed-direct claim with no lincheck term.

use crate::fold_min_len;
use crate::scratch::give_f128;
use crate::scratch::take_f128;
use crate::sumcheck_round_min_len;
use crate::zerocheck::multilinear::fold_in_place_single;
use rayon::current_num_threads;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::mem::replace;
use std::mem::take;

use super::Grinding;
use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::univariate_skip::SplitEqGhash;

/// Domain label of the standalone single-table zerocheck. The union's
/// element-region zerocheck runs the same protocol under its own label — see
/// [`prove_with_label`].
pub const LABEL: &[u8] = b"flock-element-zc-v0";

/// The prover's round messages plus the three final evaluations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    /// Per-round `(G(1), G(∞))` — Convention A, bare (no eq prefactor).
    /// Length `n_log + kappa`.
    pub rounds: Vec<(F128, F128)>,
    /// `(Âz + â_const)(r)`.
    pub ea: F128,
    /// `(B̂z + b̂_const)(r)`.
    pub eb: F128,
    /// `ẑ(r)` — the C-claim.
    pub ec: F128,
    /// Initial-equality-point then per-round PoW nonces, in transcript order.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// What a verified zerocheck leaves for Phase 2 and the opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    /// `r = (r_row, r_con)`, LSB-first (rows low). Length `n_log + kappa`.
    pub r: Vec<F128>,
    pub ea: F128,
    pub eb: F128,
    pub ec: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElementZerocheckError {
    /// Wrong number of round messages.
    BadRoundCount {
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
    /// The final consistency check `running == ea·eb + ec` failed. Any
    /// inconsistency in a round message or in the three final evaluations
    /// propagates here.
    SumcheckFinalFailed,
}

// ---------------------------------------------------------------------------
// Count-proportional row rounds
// ---------------------------------------------------------------------------

/// Engage the support-proportional row rounds while
/// `live_rows · SPARSE_GATE ≤ capacity_rows` — i.e. at or below half
/// utilization. Full utilization stays dense: the sparse path's per-column
/// bookkeeping only pays for itself once a real fraction of the rows is dead,
/// and the dense kernels are the calibrated choice at the anchor shape.
const SPARSE_GATE: usize = 2;

/// The declared row support of a region, per COLUMN — everything the
/// support-proportional row rounds need.
///
/// Rows are the LOW `nu` address bits, so a column's live rows are a PREFIX
/// `[0, live[c])` and folding maps prefix to prefix (`live' = ceil(live/2)`).
/// That is what makes the skipping a per-column loop bound rather than an
/// interval intersection.
///
/// `a_dead`/`b_dead` are the values `pa`/`pb` take on a column's DEAD rows.
/// They are not zero — on a dummy row `z = 0` so `A_0·z = 0`, leaving the
/// affine constant `a_const[y]` — but they are *constant down the column*, and
/// folding a constant preserves it, so a fully-dead entry equals `a_dead[c]` at
/// EVERY round. Substituting them analytically is what makes the sparse path
/// **bit-identical** to the dense one instead of merely value-identical: no
/// round message changes, so no pinned proof moves.
///
/// `wz` needs no such vector: the witness itself is zero on every dead word.
#[derive(Clone, Debug)]
pub struct RowSupport {
    /// Declared live rows of each region column; `0` for padding columns and
    /// inter-slot gaps (which are dead in all three tables at every row).
    pub live: Vec<usize>,
    /// `a_const[y]` / `b_const[y]` of each region column, i.e. the value the
    /// column's dead rows carry. `F128::ZERO` where `live` is 0.
    pub a_dead: Vec<F128>,
    pub b_dead: Vec<F128>,
}

impl RowSupport {
    /// Total declared rows across the region.
    fn live_rows(&self) -> usize {
        self.live.iter().sum()
    }

    /// Whether the support is sparse enough for the row rounds to pay off.
    /// `pub(crate)`: the union's witness assembly gates its live-only
    /// `pa`/`pb` derivation on the SAME predicate (see
    /// `element_r1cs::union::dead_rows_unread`), so the two cannot drift.
    pub(crate) fn worth_skipping(&self, nu: usize) -> bool {
        nu > 0 && self.live_rows() * SPARSE_GATE <= self.live.len() << nu
    }

    /// Live PAIR intervals over the post-fold index space of a round with
    /// `row_vars` remaining row variables: column `c` owns
    /// `[c·P, c·P + ceil(live[c]/2))` with `P = 2^(row_vars−1)`.
    fn pair_intervals(&self, row_vars: usize) -> Vec<(usize, usize)> {
        let p = 1usize << (row_vars - 1);
        self.live
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(c, &n)| (c * p, c * p + n.div_ceil(2)))
            .collect()
    }

    /// Halve every column's live prefix — the support after one row round.
    fn halve(&mut self) {
        for n in &mut self.live {
            *n = n.div_ceil(2);
        }
    }
}

/// Prove the zerocheck for one element table.
///
/// `pa`, `pb` are `(Az + a_const)` and `(Bz + b_const)` over the whole padded
/// domain; both are recycled into the scratch pool (round 0 reads them, the
/// first fold already writes a pooled half — see [`prove_with_support`]). `z`
/// is the committed witness, cloned into a working table so the caller keeps
/// it for Phase 2 and the opening.
///
/// All three tables are laid out `[(y or c) << n_log | j]`, i.e. rows low, which
/// makes the sumcheck's low-bit-first binding walk the row variables first and
/// the column variables last.
pub fn prove<C: Challenger>(
    pa: Vec<F128>,
    pb: Vec<F128>,
    z: &[F128],
    m_words: usize,
    ch: &mut C,
) -> (Proof, Claim) {
    prove_with_grinding(pa, pb, z, m_words, Grinding::disabled(), ch)
}

/// [`prove`] with an explicit Fiat--Shamir grinding policy.
pub fn prove_with_grinding<C: Challenger>(
    pa: Vec<F128>,
    pb: Vec<F128>,
    z: &[F128],
    m_words: usize,
    grinding: Grinding,
    ch: &mut C,
) -> (Proof, Claim) {
    prove_with_label_and_grinding(LABEL, pa, pb, z, m_words, grinding, ch)
}

/// [`prove`] under a caller-chosen domain label. The union's element-region
/// zerocheck is exactly this protocol over `M_elem − 7` region word variables
/// — the relation is block-diagonal across element slots, so the eq weight
/// batches them implicitly and the gaps (where `pa = pb = z = 0`) contribute
/// nothing.
pub fn prove_with_label<C: Challenger>(
    label: &[u8],
    pa: Vec<F128>,
    pb: Vec<F128>,
    z: &[F128],
    m_words: usize,
    ch: &mut C,
) -> (Proof, Claim) {
    prove_with_label_and_grinding(label, pa, pb, z, m_words, Grinding::disabled(), ch)
}

/// [`prove_with_label`] with an explicit Fiat--Shamir grinding policy.
pub fn prove_with_label_and_grinding<C: Challenger>(
    label: &[u8],
    pa: Vec<F128>,
    pb: Vec<F128>,
    z: &[F128],
    m_words: usize,
    grinding: Grinding,
    ch: &mut C,
) -> (Proof, Claim) {
    let out = prove_with_support_with_grinding(label, &pa, &pb, z, m_words, 0, None, grinding, ch);
    // The borrowed originals were never written; recycle them here, as the
    // pre-borrow fold used to when it swapped them out at round 1.
    give_f128(pa);
    give_f128(pb);
    out
}

/// [`prove_with_label`] with the declared row support, so the row rounds cost
/// `O(live rows · columns)` instead of `O(2^m_words)`.
///
/// `nu` is the number of LOW row variables and `support` describes their
/// per-column live prefixes ([`RowSupport`]). The output is **bit-identical**
/// to `support = None`: a dead pair contributes nothing to either round message
/// (`G(1)`'s summand is `a_const·b_const + 0 = 0` by the type's validity rule,
/// and `G(∞)`'s factor `wa[i0] + wa[i1]` vanishes in characteristic 2 because
/// both entries hold the same constant), and the one boundary pair per column
/// reads its dead half from `a_dead`/`b_dead` rather than from the buffer.
///
/// Requires the honest-witness contract the union already imposes: `z` is
/// identically zero on dummy rows, padding columns and gaps. Debug-asserted.
///
/// `pa`/`pb` are BORROWED and never written: round 0's message reads them and
/// the first fold writes a pooled half-size buffer ([`Table`]), so the caller
/// keeps the untouched originals — which is what lets the union prover return
/// a [`super::union::copy_live_region`] pair to the zero pool with only its
/// live spans declared dirty.
pub fn prove_with_support<C: Challenger>(
    label: &[u8],
    pa: &[F128],
    pb: &[F128],
    z: &[F128],
    m_words: usize,
    nu: usize,
    support: Option<&RowSupport>,
    ch: &mut C,
) -> (Proof, Claim) {
    prove_with_support_with_grinding(
        label,
        pa,
        pb,
        z,
        m_words,
        nu,
        support,
        Grinding::disabled(),
        ch,
    )
}

/// [`prove_with_support`] with an explicit element grinding policy.
#[allow(clippy::too_many_arguments)]
pub fn prove_with_support_with_grinding<C: Challenger>(
    label: &[u8],
    pa: &[F128],
    pb: &[F128],
    z: &[F128],
    m_words: usize,
    nu: usize,
    support: Option<&RowSupport>,
    grinding: Grinding,
    ch: &mut C,
) -> (Proof, Claim) {
    let n_words = 1usize << m_words;
    assert_eq!(pa.len(), n_words, "pa length");
    assert_eq!(pb.len(), n_words, "pb length");
    assert_eq!(z.len(), n_words, "z length");
    assert!(m_words >= 1, "need at least one variable");

    ch.observe_label(label);
    let mut grinding_nonces = Vec::with_capacity(grinding.zerocheck_nonce_count(m_words));
    let tau = if let Some(bits) = grinding.initial_bits(m_words) {
        let (nonce, tau) = ch.grind_pow_and_sample_f128_vec(bits, m_words);
        grinding_nonces.push(nonce);
        tau
    } else {
        ch.sample_f128_vec(m_words)
    };

    // Support-proportional row rounds, or all-dense. Decided ONCE: a mid-loop
    // switch would have to reconcile the sparse path's unwritten dead slots
    // with the dense kernel's full-array reads, and the row rounds carry
    // essentially all the work anyway (round 0 alone is half of it).
    let mut sparse = match support {
        Some(sup) if sup.worth_skipping(nu) => {
            assert_eq!(sup.live.len(), 1usize << (m_words - nu), "support columns");
            debug_assert!(
                dead_words_are_zero(z, nu, sup),
                "the witness must be zero on every dead word \
                 (dummy rows, padding columns, gaps)"
            );
            Some(sup.clone())
        }
        _ => None,
    };

    let (mut wa, mut wb) = (Table::Borrowed(pa), Table::Borrowed(pb));
    // The working copy of the witness. On the sparse path only the live
    // prefixes are ever read, so copying the dead tail (which is most of the
    // region at low utilization) would itself be O(2^m_words) — the one pass
    // that would have kept the row rounds off the count axis.
    let mut wz = match &sparse {
        Some(sup) => {
            let rows = 1usize << nu;
            let mut w = take_f128(n_words);
            for (c, &n) in sup.live.iter().enumerate() {
                w[c * rows..c * rows + n].copy_from_slice(&z[c * rows..c * rows + n]);
            }
            w
        }
        None => z.to_vec(),
    };
    let mut rounds = Vec::with_capacity(m_words);
    let mut r = Vec::with_capacity(m_words);
    for i in 0..m_words {
        let eq = SplitEqGhash::new(&tau[i + 1..]);
        let row_vars = nu.saturating_sub(i);
        let use_sparse = sparse.is_some() && row_vars > 0;
        let (g1, g_inf) = match (&sparse, use_sparse) {
            (Some(sup), true) => {
                round_message_sparse(wa.as_slice(), wb.as_slice(), &wz, &eq, row_vars, sup)
            }
            _ => round_message(wa.as_slice(), wb.as_slice(), &wz, &eq),
        };
        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho = if let Some(bits) = grinding.round_bits() {
            let (nonce, rho) = ch.grind_pow_and_sample_f128(bits);
            grinding_nonces.push(nonce);
            rho
        } else {
            ch.sample_f128()
        };
        rounds.push((g1, g_inf));
        r.push(rho);
        match (&mut sparse, use_sparse) {
            (Some(sup), true) => {
                wa.fold_sparse(rho, row_vars, &sup.live, &sup.a_dead);
                wb.fold_sparse(rho, row_vars, &sup.live, &sup.b_dead);
                fold_low_sparse_zero(&mut wz, rho, row_vars, &sup.live);
                sup.halve();
                if row_vars == 1 {
                    // Last row round: the column rounds that follow are dense,
                    // so give them a fully valid array. Only the columns with
                    // NO declared rows are unwritten, and their correct final
                    // value is the constant their dead rows carry — NOT zero.
                    // Folding a constant column preserves it, so `a_dead[c]`
                    // is what the dense path would have produced; writing zero
                    // instead loses `â_const(r)` from `ea`, which the lincheck
                    // then fails to reconcile (a count-0 slot is exactly this
                    // case). `wz` really is zero: the witness is.
                    // O(columns), negligible.
                    let done = take(&mut sparse).expect("checked");
                    for (c, &n) in done.live.iter().enumerate() {
                        if n == 0 {
                            wa.owned_mut()[c] = done.a_dead[c];
                            wb.owned_mut()[c] = done.b_dead[c];
                            wz[c] = F128::ZERO;
                        }
                    }
                }
            }
            _ => {
                wa.fold(rho);
                wb.fold(rho);
                fold_low(&mut wz, rho);
            }
        }
    }
    debug_assert_eq!(wa.as_slice().len(), 1);

    let (ea, eb, ec) = (wa.as_slice()[0], wb.as_slice()[0], wz[0]);
    // Bind all three final claims BEFORE the next challenge is drawn (which is
    // Phase 2's α). The α-batched reduction of `ea`/`eb` is only sound if α
    // comes after them — a prover that knew α could pick a product-preserving
    // (ea, eb) pair satisfying the one batched equation. `ec` rides along at the
    // same position; the opening binds it again as a claim value.
    ch.observe_f128(ea);
    ch.observe_f128(eb);
    ch.observe_f128(ec);

    // Recycle the folded ping-pong tables (`m_words ≥ 1`, so both are owned
    // by now); the borrowed originals stay with the caller.
    for t in [wa, wb] {
        if let Table::Owned(v) = t {
            give_f128(v);
        }
    }
    give_f128(wz);

    let proof = Proof {
        rounds,
        ea,
        eb,
        ec,
        grinding_nonces,
    };
    let claim = Claim { r, ea, eb, ec };
    (proof, claim)
}

/// Verify a zerocheck proof over `n_log + kappa` variables, walking the
/// challenger in lockstep with [`prove`].
pub fn verify<C: Challenger>(
    m_words: usize,
    proof: &Proof,
    ch: &mut C,
) -> Result<Claim, ElementZerocheckError> {
    verify_with_grinding(m_words, proof, Grinding::disabled(), ch)
}

/// [`verify`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_with_grinding<C: Challenger>(
    m_words: usize,
    proof: &Proof,
    grinding: Grinding,
    ch: &mut C,
) -> Result<Claim, ElementZerocheckError> {
    verify_with_label_and_grinding(LABEL, m_words, proof, grinding, ch)
}

/// [`verify`] under a caller-chosen domain label — mirror of
/// [`prove_with_label`].
pub fn verify_with_label<C: Challenger>(
    label: &[u8],
    m_words: usize,
    proof: &Proof,
    ch: &mut C,
) -> Result<Claim, ElementZerocheckError> {
    verify_with_label_and_grinding(label, m_words, proof, Grinding::disabled(), ch)
}

/// [`verify_with_label`] with an explicit element grinding policy.
pub fn verify_with_label_and_grinding<C: Challenger>(
    label: &[u8],
    m_words: usize,
    proof: &Proof,
    grinding: Grinding,
    ch: &mut C,
) -> Result<Claim, ElementZerocheckError> {
    if proof.rounds.len() != m_words {
        return Err(ElementZerocheckError::BadRoundCount {
            expected: m_words,
            got: proof.rounds.len(),
        });
    }
    if proof.grinding_nonces.len() != grinding.zerocheck_nonce_count(m_words) {
        return Err(ElementZerocheckError::BadGrindingNonceCount {
            expected: grinding.zerocheck_nonce_count(m_words),
            got: proof.grinding_nonces.len(),
        });
    }

    ch.observe_label(label);
    let mut nonce_idx = 0;
    let tau = if let Some(bits) = grinding.initial_bits(m_words) {
        let tau = ch
            .verify_pow_and_sample_f128_vec(proof.grinding_nonces[nonce_idx], bits, m_words)
            .ok_or(ElementZerocheckError::InvalidGrindingNonce { which: "initial" })?;
        nonce_idx += 1;
        tau
    } else {
        ch.sample_f128_vec(m_words)
    };

    // Convention A chain, identical in shape to `crate::zerocheck::verify`: the
    // running claim is the bare inner value `G(ρ)`; the just-bound variable's eq
    // factor is absorbed by reconstructing `G(0)` from the consistency identity.
    // A zerocheck starts at target 0.
    let mut running = F128::ZERO;
    let mut r = Vec::with_capacity(m_words);
    for (i, &(g1, g_inf)) in proof.rounds.iter().enumerate() {
        let t = tau[i];
        let one_plus_t = F128::ONE + t;
        let g0 = (running + t * g1) * one_plus_t.inv();

        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho = if let Some(bits) = grinding.round_bits() {
            let rho = ch
                .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], bits)
                .ok_or(ElementZerocheckError::InvalidGrindingNonce { which: "round" })?;
            nonce_idx += 1;
            rho
        } else {
            ch.sample_f128()
        };
        r.push(rho);

        let one_plus_rho = F128::ONE + rho;
        // G(ρ) = G(0)·(1+ρ) + G(1)·ρ + G(∞)·ρ·(1+ρ).
        running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
    }

    // The eq factors never accumulated into the running claim, so what is left
    // is the bare summand at `r`: `(Az+a_const)·(Bz+b_const) + z`.
    if running != proof.ea * proof.eb + proof.ec {
        return Err(ElementZerocheckError::SumcheckFinalFailed);
    }

    // Same transcript position as the prover — before Phase 2's α.
    ch.observe_f128(proof.ea);
    ch.observe_f128(proof.eb);
    ch.observe_f128(proof.ec);
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());

    Ok(Claim {
        r,
        ea: proof.ea,
        eb: proof.eb,
        ec: proof.ec,
    })
}

/// One eq-weighted round message `(G(1), G(∞))` for the summand
/// `wa·wb + wz`, with the current variable's eq factor left to the verifier.
///
/// `eq` carries the eq weights of the *not-yet-bound* variables split as
/// `eq = eq_lo ⊗ eq_hi` ([`SplitEqGhash`]), so only `2^n_lo + 2^n_hi` eq entries
/// are built instead of the full product. Low-bit binding: index `2x'` is
/// `(0, x')` and `2x'+1` is `(1, x')`.
///
/// `wz` is linear in the bound variable, so it contributes to `G(1)` only — the
/// `∞` (leading) coefficient of a degree-2 polynomial sees the quadratic term
/// alone.
fn round_message(wa: &[F128], wb: &[F128], wz: &[F128], eq: &SplitEqGhash) -> (F128, F128) {
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len(); // 2^n_lo x_lo values per x_hi
    let n_blocks = hi.len(); // 2^n_hi
    debug_assert_eq!(block * n_blocks, wa.len() / 2);

    // One outer block (fixed x_hi): inner sum weighted by eq_lo, scaled once by
    // eq_hi[x_hi].
    let block_fn = |x_hi: usize| -> (F128, F128) {
        let x_base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let xp = x_base + x_lo;
            let (i0, i1) = (2 * xp, 2 * xp + 1);
            let el = lo[x_lo];
            s1 += el * (wa[i1] * wb[i1] + wz[i1]);
            // Char 2: (a1 − a0)(b1 − b0) = (a0 + a1)(b0 + b1).
            s_inf += el * ((wa[i0] + wa[i1]) * (wb[i0] + wb[i1]));
        }
        let eh = hi[x_hi];
        (eh * s1, eh * s_inf)
    };

    let pairs = block * n_blocks;
    match sumcheck_round_min_len(pairs, n_blocks) {
        Some(min_len) => (0..n_blocks)
            .into_par_iter()
            .with_min_len(min_len)
            .map(block_fn)
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a1, ainf), (b1, binf)| (a1 + b1, ainf + binf),
            ),
        None => {
            let (mut g1, mut g_inf) = (F128::ZERO, F128::ZERO);
            for x_hi in 0..n_blocks {
                let (o, i) = block_fn(x_hi);
                g1 += o;
                g_inf += i;
            }
            (g1, g_inf)
        }
    }
}

/// For each LIVE pair index in `[lo, hi)`, call `f(xp, c, odd_live)`: the
/// pair's flat index, its region column, and whether the pair's ODD row is
/// still live (if not, the caller substitutes the column's dead value).
///
/// `iv` is the per-column live pair intervals, ascending and disjoint, so the
/// entry point is a `partition_point` and each interval's column is `s / pairs`.
#[inline]
fn for_live_pairs(
    iv: &[(usize, usize)],
    lo: usize,
    hi: usize,
    pairs: usize,
    live: &[usize],
    mut f: impl FnMut(usize, usize, bool),
) {
    let start = iv.partition_point(|&(_, e)| e <= lo);
    for &(s, e) in &iv[start..] {
        if s >= hi {
            break;
        }
        let c = s / pairs;
        let nl = live[c];
        for xp in s.max(lo)..e.min(hi) {
            f(xp, c, 2 * (xp - c * pairs) + 1 < nl);
        }
    }
}

/// Support-proportional sibling of [`round_message`]: the same eq-weighted
/// `(G(1), G(∞))` over the LIVE pairs only.
///
/// A pair whose two rows are both dead contributes nothing to either message —
/// see [`prove_with_support`] — so the walk covers just the per-column live
/// prefixes. The single boundary pair a column has when its live count is odd
/// reads its dead half from `a_dead`/`b_dead` (and zero for `wz`), which is why
/// this is bit-identical rather than merely value-identical.
///
/// **Block-major, like the dense kernel.** Tasks are `eq.hi` blocks and the
/// `eq_hi` factor is applied ONCE to each block's accumulated sum, not per
/// pair. Doing it per pair costs an extra F128 multiply on every element and
/// measured *slower* than the dense kernel it was meant to beat, even at 1/16
/// utilization.
fn round_message_sparse(
    wa: &[F128],
    wb: &[F128],
    wz: &[F128],
    eq: &SplitEqGhash,
    row_vars: usize,
    sup: &RowSupport,
) -> (F128, F128) {
    let rows = 1usize << row_vars; // rows per column, this round
    let pairs = rows / 2; // post-fold rows per column
    let block = eq.lo.len();
    let n_blocks = eq.hi.len();
    let iv = sup.pair_intervals(row_vars);

    let block_fn = |x_hi: usize| -> (F128, F128) {
        let base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for_live_pairs(&iv, base, base + block, pairs, &sup.live, |xp, c, odd| {
            let i0 = 2 * xp;
            let (a0, b0) = (wa[i0], wb[i0]);
            let (a1, b1, z1) = if odd {
                (wa[i0 + 1], wb[i0 + 1], wz[i0 + 1])
            } else {
                (sup.a_dead[c], sup.b_dead[c], F128::ZERO)
            };
            let el = eq.lo[xp - base];
            s1 += el * (a1 * b1 + z1);
            // Char 2: (a1 − a0)(b1 − b0) = (a0 + a1)(b0 + b1).
            s_inf += el * ((a0 + a1) * (b0 + b1));
        });
        let eh = eq.hi[x_hi];
        (eh * s1, eh * s_inf)
    };

    // Gate on the LIVE pair count, not the dense size: at low utilization the
    // real work is a few hundred pairs and rayon task spawn would dominate it.
    match sumcheck_round_min_len(live_pairs_total(&iv), n_blocks) {
        Some(min_len) => (0..n_blocks)
            .into_par_iter()
            .with_min_len(min_len)
            .map(block_fn)
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a1, ai), (b1, bi)| (a1 + b1, ai + bi),
            ),
        None => (0..n_blocks)
            .map(block_fn)
            .fold((F128::ZERO, F128::ZERO), |(a1, ai), (b1, bi)| {
                (a1 + b1, ai + bi)
            }),
    }
}

/// Total live pairs described by an interval list.
fn live_pairs_total(iv: &[(usize, usize)]) -> usize {
    iv.iter().map(|&(s, e)| e - s).sum()
}

/// Support-proportional sibling of [`fold_low`] for `pa`/`pb`: fold only each
/// column's live pair prefix, substituting `dead[c]` for a dead odd row.
///
/// Dead output slots are left UNWRITTEN. That is sound because liveness is a
/// per-column prefix that folding maps to a prefix, so nothing downstream ever
/// reads them — the message and fold of every later round are bounded by the
/// halved prefix, and the last row round fills the fully-dead columns before
/// the dense column rounds take over.
///
/// Parallel over FLAT output chunks (not columns): element blocks are narrow
/// and tall, so a column-parallel fold collapses to `2^kappa` threads.
/// The prover's view of one of the caller's `pa`/`pb` tables: BORROWED until
/// the first fold, OWNED (pooled ping-pong halves) after it. Every fold reads
/// the current view and writes a fresh pooled buffer, so the borrowed
/// original is never written — the caller keeps it whole for reuse (the union
/// prover returns its `copy_live_region` pair to the zero pool with only the
/// live spans declared dirty).
enum Table<'a> {
    Borrowed(&'a [F128]),
    Owned(Vec<F128>),
}

impl Table<'_> {
    fn as_slice(&self) -> &[F128] {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(v) => v,
        }
    }

    /// The owned buffer, for the last-row-round dead-column fixups — which
    /// always follow a fold in the same iteration, so the table is owned.
    fn owned_mut(&mut self) -> &mut Vec<F128> {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(_) => unreachable!("a fold precedes every in-place write"),
        }
    }

    /// Swap in a folded successor, recycling the previous OWNED buffer (a
    /// borrowed predecessor is the caller's to keep).
    fn replace_with(&mut self, out: Vec<F128>) {
        if let Self::Owned(old) = replace(self, Self::Owned(out)) {
            give_f128(old);
        }
    }

    /// [`fold_low_sparse`] on the view.
    fn fold_sparse(&mut self, rho: F128, row_vars: usize, live: &[usize], dead: &[F128]) {
        let out = fold_low_sparse_out(self.as_slice(), rho, row_vars, live, dead);
        self.replace_with(out);
    }

    /// [`fold_low`] on the view. The narrow serial arm folds in place when
    /// owned; a borrowed narrow table (a tiny dense round 0) is copied into a
    /// pooled buffer first — half-size regions there are below the parallel
    /// threshold, so the copy is noise.
    fn fold(&mut self, rho: F128) {
        match fold_min_len(self.as_slice().len() / 2) {
            Some(min_len) => {
                let out = fold_low_out(self.as_slice(), rho, min_len);
                self.replace_with(out);
            }
            None => match self {
                Self::Owned(v) => fold_in_place_single(v, rho),
                Self::Borrowed(s) => {
                    let mut v = take_f128(s.len());
                    v.copy_from_slice(s);
                    fold_in_place_single(&mut v, rho);
                    self.replace_with(v);
                }
            },
        }
    }
}

fn fold_low_sparse(u: &mut Vec<F128>, rho: F128, row_vars: usize, live: &[usize], dead: &[F128]) {
    let out = fold_low_sparse_out(u, rho, row_vars, live, dead);
    let old = replace(u, out);
    give_f128(old);
}

/// [`fold_low_sparse`]'s kernel: read `src`, return the pooled folded half.
fn fold_low_sparse_out(
    src: &[F128],
    rho: F128,
    row_vars: usize,
    live: &[usize],
    dead: &[F128],
) -> Vec<F128> {
    let pairs = 1usize << (row_vars - 1);
    let half = src.len() / 2;
    let iv: Vec<(usize, usize)> = live
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .map(|(c, &n)| (c * pairs, c * pairs + n.div_ceil(2)))
        .collect();
    let mut out = take_f128(half);
    {
        let total = live_pairs_total(&iv);
        // Gate on LIVE work: `par_chunks_mut` over the full output would spawn
        // a task per chunk regardless of how few pairs are live, and at low
        // utilization that spawn cost is the whole round.
        match fold_min_len(total) {
            None => {
                let out: &mut [F128] = &mut out;
                for_live_pairs(&iv, 0, half, pairs, live, |xp, c, odd| {
                    let i0 = 2 * xp;
                    let a0 = src[i0];
                    let a1 = if odd { src[i0 + 1] } else { dead[c] };
                    out[xp] = a0 + rho * (a1 + a0);
                });
            }
            Some(_) => {
                let chunk = (half / current_num_threads().max(1))
                    .next_power_of_two()
                    .max(1 << 10);
                out.par_chunks_mut(chunk).enumerate().for_each(|(k, dst)| {
                    let base = k * chunk;
                    for_live_pairs(&iv, base, base + dst.len(), pairs, live, |xp, c, odd| {
                        let i0 = 2 * xp;
                        let a0 = src[i0];
                        let a1 = if odd { src[i0 + 1] } else { dead[c] };
                        dst[xp - base] = a0 + rho * (a1 + a0);
                    });
                });
            }
        }
    }
    out
}

/// [`fold_low_sparse`] for the witness table, whose dead rows are genuinely
/// zero (so the substituted odd half is `F128::ZERO`).
fn fold_low_sparse_zero(u: &mut Vec<F128>, rho: F128, row_vars: usize, live: &[usize]) {
    let zeros = vec![F128::ZERO; live.len()];
    fold_low_sparse(u, rho, row_vars, live, &zeros);
}

/// Whether the witness is zero on every word the support declares dead — the
/// precondition the sparse row rounds rest on.
fn dead_words_are_zero(z: &[F128], nu: usize, sup: &RowSupport) -> bool {
    let rows = 1usize << nu;
    sup.live
        .iter()
        .enumerate()
        .all(|(c, &n)| z[c * rows + n..(c + 1) * rows].iter().all(|w| w.is_zero()))
}

/// Bind the low variable of one table at `rho`, halving it:
/// `u[x] ← u[2x] + rho·(u[2x+1] + u[2x])`.
///
/// Wide folds read the old table and write a **pooled** buffer, then swap and
/// recycle it — no per-round allocation, and no in-place aliasing (slot `x` is
/// also read as `2x'` for `x' = x/2`). Narrow folds fall through to the shared
/// serial kernel. Gating is the crate's [`crate::fold_min_len`], the same rule
/// the other sub-gate folds use.
fn fold_low(u: &mut Vec<F128>, rho: F128) {
    match fold_min_len(u.len() / 2) {
        Some(min_len) => {
            let out = fold_low_out(u, rho, min_len);
            let old = replace(u, out);
            give_f128(old);
        }
        None => fold_in_place_single(u, rho),
    }
}

/// [`fold_low`]'s wide kernel: read `src`, return the pooled folded half.
/// `take_f128(half)` returns a length-`half` buffer; the map writes every
/// slot, satisfying the write-before-read contract.
fn fold_low_out(src: &[F128], rho: F128, min_len: usize) -> Vec<F128> {
    let mut out = take_f128(src.len() / 2);
    out.par_iter_mut()
        .with_min_len(min_len)
        .enumerate()
        .for_each(|(x, o)| {
            let a0 = src[2 * x];
            *o = a0 + rho * (src[2 * x + 1] + a0);
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::element_r1cs::tests::{mixed_gate, mixed_witness, mult_gate, mult_witness};
    use crate::element_r1cs::{ElementTableType, broadcast_add};
    use crate::test_rng::Rng;
    use crate::zerocheck::multilinear::eq_eval;
    use crate::zerocheck::multilinear::fold_in_place_single;
    use flock_multilinear::IndexOrder;
    use flock_multilinear::evaluate;

    /// Direct MLE evaluation of `table` at `point`, binding the low variable
    /// first — the same order [`fold_low`] uses.
    fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
        evaluate(table, point, IndexOrder::LowToHigh)
    }

    /// `(pa, pb)` for a witness — the same preparation [`super::super::prove`]
    /// does.
    fn prepare(ty: &ElementTableType, z: &[F128], n_log: usize) -> (Vec<F128>, Vec<F128>) {
        let (mut pa, mut pb) = ty.apply(z, n_log);
        broadcast_add(&mut pa, ty.a_const(), n_log);
        broadcast_add(&mut pb, ty.b_const(), n_log);
        (pa, pb)
    }

    /// Brute-force the zerocheck sum `Σ_x eq(τ,x)·(pa·pb + z)(x)` over the
    /// hypercube, evaluating `eq` from its definition. Low-bit-first index
    /// convention: bit `i` of `x` is coordinate `i` of the point.
    fn brute_force_sum(pa: &[F128], pb: &[F128], z: &[F128], tau: &[F128]) -> F128 {
        let mut acc = F128::ZERO;
        for x in 0..pa.len() {
            let bits: Vec<F128> = (0..tau.len())
                .map(|i| {
                    if (x >> i) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            acc += eq_eval(tau, &bits) * (pa[x] * pb[x] + z[x]);
        }
        acc
    }

    /// The statement the zerocheck proves is TRUE for a satisfying witness: the
    /// eq-weighted sum is zero at any τ, for every shape and count including the
    /// n=0 edge. This is the differential anchor — if the relation encoding or
    /// the padding convention were wrong, this fails before any sumcheck runs.
    #[test]
    fn satisfying_witness_has_zero_eq_weighted_sum() {
        let mut rng = Rng::new(4242);
        for kappa in [2usize, 3] {
            let ty = if kappa == 2 {
                mult_gate(2)
            } else {
                mixed_gate(&mut rng)
            };
            for n_log in [2usize, 4] {
                for n in [0usize, 1, 3, 1 << n_log] {
                    if n > 1 << n_log {
                        continue;
                    }
                    let z = if kappa == 2 {
                        mult_witness(&ty, n_log, n, &mut rng)
                    } else {
                        mixed_witness(&ty, n_log, n, &mut rng)
                    };
                    assert!(ty.satisfies(&z, n_log, n), "κ={kappa} n_log={n_log} n={n}");
                    let (pa, pb) = prepare(&ty, &z, n_log);
                    let tau: Vec<F128> = (0..n_log + kappa).map(|_| rng.f128()).collect();
                    assert_eq!(
                        brute_force_sum(&pa, &pb, &z, &tau),
                        F128::ZERO,
                        "κ={kappa} n_log={n_log} n={n}"
                    );
                }
            }
        }
    }

    /// **Differential test.** For a *random* (not necessarily satisfying)
    /// instance, the prover's messages must be the honest sumcheck of the real
    /// polynomial. We check that against brute force in two independent ways:
    ///
    /// 1. the final evaluations equal the tables' MLEs at the claim point `r`,
    ///    computed by direct folding;
    /// 2. re-running the verifier's chain from the true initial target
    ///    (`brute_force_sum`) lands exactly on `ea·eb + ec`.
    ///
    /// (2) is the strong statement: every round message is pinned, because a
    /// wrong `(G(1), G(∞))` anywhere breaks the chain.
    #[test]
    fn round_messages_match_brute_force_on_random_instances() {
        let mut rng = Rng::new(31337);
        for m_words in [1usize, 2, 5, 8] {
            let n = 1usize << m_words;
            let pa: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let pb: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let z: Vec<F128> = (0..n).map(|_| rng.f128()).collect();

            // The row/column split only matters to the caller; the sumcheck
            // itself sees `m_words` variables and nothing else.
            let mut ch = FsChallenger::new(b"element-zc-diff");
            let (proof, claim) = prove(pa.clone(), pb.clone(), &z, m_words, &mut ch);

            // Re-derive τ the way the prover did.
            let mut ch2 = FsChallenger::new(b"element-zc-diff");
            ch2.observe_label(LABEL);
            let tau = ch2.sample_f128_vec(m_words);

            // (1) finals are the MLEs at r.
            assert_eq!(claim.ea, mle_eval(&pa, &claim.r), "ea m={m_words}");
            assert_eq!(claim.eb, mle_eval(&pb, &claim.r), "eb m={m_words}");
            assert_eq!(claim.ec, mle_eval(&z, &claim.r), "ec m={m_words}");

            // (2) the chain from the true target closes on ea·eb + ec.
            let mut running = brute_force_sum(&pa, &pb, &z, &tau);
            for (i, &(g1, g_inf)) in proof.rounds.iter().enumerate() {
                let t = tau[i];
                let g0 = (running + t * g1) * (F128::ONE + t).inv();
                let rho = claim.r[i];
                let one_plus_rho = F128::ONE + rho;
                running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
            }
            assert_eq!(
                running,
                claim.ea * claim.eb + claim.ec,
                "chain from brute-force target, m={m_words}"
            );
        }
    }

    /// Prove → verify roundtrip on satisfying witnesses at several shapes and
    /// counts (including non-power-of-two `n`, full utilization, and `n = 0`).
    #[test]
    fn prove_verify_roundtrip_honest() {
        let mut rng = Rng::new(909);
        for (n_log, n) in [(2usize, 0usize), (2, 3), (3, 5), (4, 16), (6, 37)] {
            let ty = mult_gate(2);
            let z = mult_witness(&ty, n_log, n, &mut rng);
            let (pa, pb) = prepare(&ty, &z, n_log);

            let mut ch_p = FsChallenger::new(b"element-zc-rt");
            let (proof, claim_p) = prove(pa, pb, &z, n_log + 2, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"element-zc-rt");
            let claim_v = verify(n_log + 2, &proof, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected at n_log={n_log} n={n}: {e:?}"));
            assert_eq!(claim_p, claim_v, "n_log={n_log} n={n}");
        }
    }

    /// A witness violating ONE constraint in ONE row must be rejected.
    #[test]
    fn unsatisfying_witness_rejected() {
        let mut rng = Rng::new(6161);
        let (n_log, n) = (4usize, 11usize);
        let ty = mult_gate(2);
        let mut z = mult_witness(&ty, n_log, n, &mut rng);
        // Break the product in row 5.
        z[2 * (1 << n_log) + 5] += F128::ONE;
        assert!(!ty.satisfies(&z, n_log, n));
        let (pa, pb) = prepare(&ty, &z, n_log);

        let mut ch_p = FsChallenger::new(b"element-zc-bad");
        let (proof, _) = prove(pa, pb, &z, n_log + 2, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"element-zc-bad");
        assert_eq!(
            verify(n_log + 2, &proof, &mut ch_v),
            Err(ElementZerocheckError::SumcheckFinalFailed)
        );
    }

    /// A non-zero entry in a dummy row is a violation too — the padding rows are
    /// inside the sum, not skipped.
    #[test]
    fn dirty_dummy_row_rejected() {
        let mut rng = Rng::new(717);
        let (n_log, n) = (4usize, 9usize);
        let ty = mult_gate(2);
        let mut z = mult_witness(&ty, n_log, n, &mut rng);
        // Row 12 is dummy; set its product column without its operands.
        z[2 * (1 << n_log) + 12] = F128::ONE;
        let (pa, pb) = prepare(&ty, &z, n_log);

        let mut ch_p = FsChallenger::new(b"element-zc-dirty");
        let (proof, _) = prove(pa, pb, &z, n_log + 2, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"element-zc-dirty");
        assert!(verify(n_log + 2, &proof, &mut ch_v).is_err());
    }

    /// Every proof component must be transcript-bound: flipping a bit anywhere
    /// makes the verifier reject.
    #[test]
    fn verify_rejects_mutations() {
        let mut rng = Rng::new(5);
        let (n_log, kappa) = (4usize, 2usize);
        let ty = mult_gate(kappa);
        let z = mult_witness(&ty, n_log, 13, &mut rng);
        let (pa, pb) = prepare(&ty, &z, n_log);
        let mut ch_p = FsChallenger::new(b"element-zc-mut");
        let (proof, _) = prove(pa, pb, &z, n_log + kappa, &mut ch_p);

        let n_rounds = proof.rounds.len();
        let mut cases: Vec<(String, Proof)> = Vec::new();
        for i in 0..n_rounds {
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
        for (name, field) in [("ea", 0usize), ("eb", 1), ("ec", 2)] {
            let mut bad = proof.clone();
            match field {
                0 => bad.ea += F128::ONE,
                1 => bad.eb += F128::ONE,
                _ => bad.ec += F128::ONE,
            }
            cases.push((name.to_string(), bad));
        }
        for (name, bad) in cases {
            let mut ch = FsChallenger::new(b"element-zc-mut");
            assert!(
                verify(n_log + kappa, &bad, &mut ch).is_err(),
                "verify accepted mutation: {name}"
            );
        }
    }

    /// AUDIT (Fiat–Shamir binding of the final claims). A *product-preserving*
    /// tamper `(ea, eb) → (ea·t, eb·t⁻¹)` leaves the zerocheck's own final check
    /// `running == ea·eb + ec` satisfied, so `verify` still returns `Ok` — the
    /// zerocheck alone is blind to it, exactly as in
    /// `crate::zerocheck::tests::audit_final_ab_claims_bound_to_transcript`.
    ///
    /// The defense is that all three finals are observed last, so the next
    /// challenge — the slot Phase 2 draws α from — must diverge from the honest
    /// run. Without that observe the α-batched reduction of `ea`/`eb` would be
    /// unsound: a prover that already knew α could pick the pair.
    #[test]
    fn audit_final_claims_bound_to_transcript() {
        let mut rng = Rng::new(0xF1A7_5A11);
        let (n_log, kappa) = (4usize, 2usize);
        let ty = mult_gate(kappa);
        let z = mult_witness(&ty, n_log, 13, &mut rng);
        let (pa, pb) = prepare(&ty, &z, n_log);
        let mut ch_p = FsChallenger::new(b"element-zc-bind");
        let (proof, _) = prove(pa, pb, &z, n_log + kappa, &mut ch_p);

        let mut ch_honest = FsChallenger::new(b"element-zc-bind");
        assert!(verify(n_log + kappa, &proof, &mut ch_honest).is_ok());
        let alpha_honest = ch_honest.sample_f128();

        let t = F128::new(3, 0);
        let mut bad = proof.clone();
        bad.ea = proof.ea * t;
        bad.eb = proof.eb * t.inv();
        let mut ch_bad = FsChallenger::new(b"element-zc-bind");
        assert!(
            verify(n_log + kappa, &bad, &mut ch_bad).is_ok(),
            "product-preserving swap is invisible to the zerocheck's own check"
        );
        assert_ne!(
            alpha_honest,
            ch_bad.sample_f128(),
            "final claims are not bound to the transcript — α would be reusable"
        );
    }

    /// Shape rejection: a truncated round list.
    #[test]
    fn verify_rejects_bad_round_count() {
        let mut rng = Rng::new(8);
        let (n_log, kappa) = (3usize, 2usize);
        let ty = mult_gate(kappa);
        let z = mult_witness(&ty, n_log, 4, &mut rng);
        let (pa, pb) = prepare(&ty, &z, n_log);
        let mut ch_p = FsChallenger::new(b"element-zc-shape");
        let (mut proof, _) = prove(pa, pb, &z, n_log + kappa, &mut ch_p);
        proof.rounds.pop();
        let mut ch = FsChallenger::new(b"element-zc-shape");
        assert!(matches!(
            verify(n_log + kappa, &proof, &mut ch),
            Err(ElementZerocheckError::BadRoundCount { .. })
        ));
    }

    /// `fold_low` must agree with the shared serial kernel at every width,
    /// including across the parallel gate — the pooled-buffer path is the one
    /// place this module writes its own kernel.
    #[test]
    fn fold_low_matches_serial_kernel() {
        let mut rng = Rng::new(2024);
        for log_n in 1..=18usize {
            let n = 1usize << log_n;
            let v: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let rho = rng.f128();
            let mut a = v.clone();
            fold_low(&mut a, rho);
            let mut b = v;
            fold_in_place_single(&mut b, rho);
            assert_eq!(a, b, "log_n={log_n}");
        }
    }
}
