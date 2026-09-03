//! Quadratic-extension folds with base-field code-switch commitments.
//!
//! A folded extension table `f = f_0 + u f_1` is never committed as an
//! extension-field word. At every code switch it becomes the base-field table
//!
//! `g(b, x) = f_b(x)`,
//!
//! with the coordinate bit `b` as the least-significant multilinear variable.
//! The current linear claim is transported by replacing a basis `B(x)` with
//! `u^b B(x)`. Consequently each recursive level spends one fold round on the
//! coordinate bit and removes only `k - 1` variables from the extension table.

use std::{
    env::{var, var_os},
    mem::replace,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use flock_multilinear::{IndexOrder, eq_table};

use crate::{
    merkle::cap_layer,
    pcs::{
        ligerito::{
            AdditiveNttF128, BasisWindowFn, Challenger, F128, F256, FOLD_LOOKAHEAD_OVERRIDE,
            FinalProof, FoldLookahead, Hash, IndexedParallelIterator, IntoParallelIterator,
            IntoParallelRefIterator, IntoParallelRefMutIterator, LigeritoProof, ParallelIterator,
            ParallelSliceMut, ProverConfig, RecursiveProof, SumcheckMessage, SumcheckMessage256,
            VerifierConfig, VirtualEqBasis, VirtualEqTerm, build_eq_table, ceil_log2,
            eval_sk_at_vks, grind_and_sample_queries, induce_sumcheck_poly_auto, ligero_commit,
            merkle_paths_for, next_s, round_msg_and_eval_blocked,
            round_msg_and_eval_eq_point_blocked, round_msg_eval_and_lookahead,
            round_msg_eval_and_lookahead_eq_point_blocked, verify_and_sample_queries,
            verify_level_opens,
        },
        ring_switch::build_eq_scaled_parallel,
    },
    scratch::{give_f256, take_f256},
};

/// Split extension values into the base-field table `g(b, x)`, with adjacent
/// `(b=0, b=1)` values for every `x`.
pub(super) fn split_coordinates(values: &[F256]) -> Vec<F128> {
    let mut split = vec![F128::ZERO; 2 * values.len()];
    split
        .par_chunks_exact_mut(2)
        .zip(values.par_iter())
        .for_each(|(out, value)| {
            out[0] = value.c0;
            out[1] = value.c1;
        });
    split
}

/// Transport an extension-valued basis across the coordinate split. For each
/// old basis value `B(x)`, the new pair is `(B(x), u B(x))`.
pub(super) fn split_basis(values: &[F256]) -> Vec<F256> {
    let mut split = vec![F256::ZERO; 2 * values.len()];
    split
        .par_chunks_exact_mut(2)
        .zip(values.par_iter())
        .for_each(|(out, &value)| {
            out[0] = value;
            out[1] = F256::U * value;
        });
    split
}

/// Inner product after the coordinate split. This is also the final residual
/// check: the proof exposes only F128 coordinate words while the basis and
/// running claim remain in F256.
pub(super) fn split_inner_product(words: &[F128], basis: &[F256]) -> F256 {
    assert_eq!(words.len(), basis.len());
    words
        .par_iter()
        .zip(basis.par_iter())
        .map(|(&word, &weight)| weight * word)
        .reduce(|| F256::ZERO, |a, b| a + b)
}

fn build_eq_table256(point: &[F256]) -> Vec<F256> {
    eq_table(point, F256::ONE, IndexOrder::LowToHigh)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RoundQuad256 {
    c: F256,
    b: F256,
    a: F256,
}

impl RoundQuad256 {
    pub(super) fn from_msg(msg: SumcheckMessage256, claim: F256) -> Self {
        Self {
            c: msg.u_0,
            b: claim + msg.u_2,
            a: msg.u_2,
        }
    }

    pub(super) fn eval(self, r: F256) -> F256 {
        self.c + r * self.b + (r * r) * self.a
    }

    pub(super) fn fold(self, rhs: Self, alpha: F128) -> Self {
        Self {
            c: self.c + rhs.c * alpha,
            b: self.b + rhs.b * alpha,
            a: self.a + rhs.a * alpha,
        }
    }
}

#[inline]
pub(super) fn observe_message<Ch: Challenger>(challenger: &mut Ch, msg: SumcheckMessage256) {
    challenger.observe_f256(msg.u_0);
    challenger.observe_f256(msg.u_2);
}

/// [`round_msg`] with a base-VALUED f-side (the just-split table at a code
/// switch): each product is F256×F128 = 2 muls instead of 3, value-identical
/// because the f words' second limbs are zero.
fn round_msg_fbase(f: &[F256], b: &[F256]) -> SumcheckMessage256 {
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_power_of_two() && f.len() >= 2);
    let half = f.len() / 2;
    let (u_0, u_2) = (0..half)
        .into_par_iter()
        .map(|j| {
            let (f0, f1) = (f[2 * j], f[2 * j + 1]);
            debug_assert!(f0.c1.is_zero() && f1.c1.is_zero(), "f must be base-valued");
            let (b0, b1) = (b[2 * j], b[2 * j + 1]);
            (b0 * f0.c0, (b0 + b1) * (f0.c0 + f1.c0))
        })
        .reduce(
            || (F256::ZERO, F256::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage256 { u_0, u_2 }
}

fn round_msg(f: &[F256], b: &[F256]) -> SumcheckMessage256 {
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_power_of_two() && f.len() >= 2);
    let half = f.len() / 2;
    let (u_0, u_2) = (0..half)
        .into_par_iter()
        .map(|j| {
            let (f0, f1) = (f[2 * j], f[2 * j + 1]);
            let (b0, b1) = (b[2 * j], b[2 * j + 1]);
            (f0 * b0, (f0 + f1) * (b0 + b1))
        })
        .reduce(
            || (F256::ZERO, F256::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage256 { u_0, u_2 }
}

fn round_msg_blocked(f: &[F256], b: &[F256], d: usize) -> SumcheckMessage256 {
    if d == 1 || f.len() == d {
        return round_msg(f, b);
    }
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_multiple_of(2 * d));
    if d >= (1 << 15) {
        // Lane-major geometry: whole-block tasks starve the cores (the
        // fold-0 message has len/(2d) = 16 blocks at the m32 leaf). The k
        // dimension is an independent XOR-sum — split it too; chunk order
        // cannot change the value.
        const KC: usize = 1 << 13;
        let kchunks = d / KC;
        let (u_0, u_2) = (0..(f.len() / (2 * d)) * kchunks)
            .into_par_iter()
            .map(|item| {
                let (j, kc) = (item / kchunks, item % kchunks);
                let lo = 2 * j * d + kc * KC;
                let hi = lo + d;
                let mut u0 = F256::ZERO;
                let mut u2 = F256::ZERO;
                for k in 0..KC {
                    let (f0, f1) = (f[lo + k], f[hi + k]);
                    let (b0, b1) = (b[lo + k], b[hi + k]);
                    u0 += f0 * b0;
                    u2 += (f0 + f1) * (b0 + b1);
                }
                (u0, u2)
            })
            .reduce(
                || (F256::ZERO, F256::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );
        return SumcheckMessage256 { u_0, u_2 };
    }
    let (u_0, u_2) = (0..f.len() / (2 * d))
        .into_par_iter()
        .map(|j| {
            let mut u0 = F256::ZERO;
            let mut u2 = F256::ZERO;
            let lo = 2 * j * d;
            let hi = lo + d;
            for k in 0..d {
                let (f0, f1) = (f[lo + k], f[hi + k]);
                let (b0, b1) = (b[lo + k], b[hi + k]);
                u0 += f0 * b0;
                u2 += (f0 + f1) * (b0 + b1);
            }
            (u0, u2)
        })
        .reduce(
            || (F256::ZERO, F256::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage256 { u_0, u_2 }
}

#[inline]
fn next_round_msg(f: &[F256], b: &[F256], d: usize) -> SumcheckMessage256 {
    if d > 1 && f.len() > d {
        round_msg_blocked(f, b, d)
    } else {
        round_msg(f, b)
    }
}

fn fold_extension(values: &[F256], r: F256, d: usize) -> Vec<F256> {
    let half = values.len() / 2;
    (0..half)
        .into_par_iter()
        .map(|o| {
            let (block, within) = (o / d, o % d);
            let lo = 2 * block * d + within;
            let hi = lo + d;
            values[lo] + r * (values[hi] + values[lo])
        })
        .collect()
}

/// Test oracle: fold a dense basis at an extension-point prefix the slow way.
#[cfg(test)]
pub(super) fn evaluate_dense_at_residual(
    basis: &[F128],
    point_prefix: &[F256],
    residual_log: usize,
) -> Vec<F256> {
    let mut values: Vec<F256> = basis.iter().copied().map(F256::from).collect();
    for &r in point_prefix {
        values = fold_extension(&values, r, 1);
    }
    assert_eq!(values.len(), 1usize << residual_log);
    values
}

fn fold_base(values: &[F128], r: F256, d: usize) -> Vec<F256> {
    let half = values.len() / 2;
    (0..half)
        .into_par_iter()
        .map(|o| {
            let (block, within) = (o / d, o % d);
            let lo = 2 * block * d + within;
            let hi = lo + d;
            F256::from(values[lo]) + r * (values[hi] + values[lo])
        })
        .collect()
}

fn fold_base_fill(
    f: &[F128],
    fill: BasisWindowFn<'_>,
    r: F256,
    d: usize,
) -> (Vec<F256>, Vec<F256>) {
    let half = f.len() / 2;
    let mut nf = vec![F256::ZERO; half];
    let mut nb = vec![F256::ZERO; half];
    let chunk = 2048usize.min(d);
    nf.par_chunks_mut(chunk)
        .zip(nb.par_chunks_mut(chunk))
        .enumerate()
        .for_each_init(
            || (vec![F128::ZERO; chunk], vec![F128::ZERO; chunk]),
            |(blo, bhi), (ci, (fo, bo))| {
                let o = ci * chunk;
                let (block, within) = (o / d, o % d);
                let lo = 2 * block * d + within;
                let hi = lo + d;
                let len = fo.len();
                fill(&mut blo[..len], lo);
                fill(&mut bhi[..len], hi);
                for k in 0..len {
                    fo[k] = F256::from(f[lo + k]) + r * (f[hi + k] + f[lo + k]);
                    bo[k] = F256::from(blo[k]) + r * (bhi[k] + blo[k]);
                }
            },
        );
    (nf, nb)
}

/// FUSED fold + next-round message for the materialized sumcheck path: one
/// parallel sweep folds BOTH sides by `r` and accumulates the next round's
/// `(u_0, u_2)` from the folded values as they are produced — replacing the
/// three full passes (fold f, fold b, `next_round_msg`) that made the
/// `initial_k` fold chain the F256 ladder's dominant cost (measured 12x the
/// F128 path at the m32 leaf; `[lig-prove-f256]`).
///
/// Message-block `j` of `next_round_msg(out, d)` covers output `[2jd, 2jd+2d)`,
/// whose inputs are the contiguous run `[4jd, 4jd+4d)` — so the sweep is
/// chunked on message-block boundaries. VALUE-IDENTICAL to the unfused
/// sequence: the per-element fold expression is verbatim, and `u_0`/`u_2` are
/// XOR-additive sums, so chunk order cannot change them. Callers must ensure
/// the folded length keeps the blocked layout (`d == 1 && half >= 2`, or
/// `d > 1 && half >= 2d`) and fall back to the unfused pair otherwise.
macro_rules! fused_fold_msg {
    ($name:ident, $fel:ty, $bel:ty, $ffold:path, $bfold:path) => {
        fn $name(
            f: &[$fel],
            b: &[$bel],
            r: F256,
            d: usize,
        ) -> (Vec<F256>, Vec<F256>, SumcheckMessage256) {
            debug_assert_eq!(f.len(), b.len());
            let half = f.len() / 2;
            debug_assert!(d.is_power_of_two() && half.is_power_of_two());
            debug_assert!(if d == 1 { half >= 2 } else { half >= 2 * d });
            let block = 2 * d;
            let zero2 = || (F256::ZERO, F256::ZERO);
            let sum2 = |(a0, a2): (F256, F256), (b0, b2): (F256, F256)| (a0 + b0, a2 + b2);
            // Pooled, uninitialized outputs: the sweep writes every slot.
            // Fresh per-fold allocations were the phase's real cost — ~2 GiB
            // of first-touch page faults per m32 prove, re-paid every prove
            // because large frees are unmapped.
            let mut nf = crate::scratch::take_f256(half);
            let mut nb = crate::scratch::take_f256(half);
            let (u_0, u_2) = if block >= (1 << 16) {
                // Lane-major geometry: a handful of huge message blocks —
                // whole-block tasks starve the cores (ONE task by the last
                // initial folds). The k dimension inside a block is
                // independent, so split it too.
                const KC: usize = 1 << 13;
                nf.par_chunks_mut(block)
                    .zip(nb.par_chunks_mut(block))
                    .enumerate()
                    .map(|(jb, (fblk, bblk))| {
                        let in0 = 2 * jb * block;
                        let (flo_h, fhi_h) = fblk.split_at_mut(d);
                        let (blo_h, bhi_h) = bblk.split_at_mut(d);
                        flo_h
                            .par_chunks_mut(KC)
                            .zip(fhi_h.par_chunks_mut(KC))
                            .zip(blo_h.par_chunks_mut(KC).zip(bhi_h.par_chunks_mut(KC)))
                            .enumerate()
                            .map(|(kc, ((flc, fhc), (blc, bhc)))| {
                                let k0 = kc * KC;
                                let mut u0 = F256::ZERO;
                                let mut u2 = F256::ZERO;
                                for k in 0..flc.len() {
                                    let i = in0 + k0 + k;
                                    let flo = $ffold(f, i, i + d, r);
                                    let fhi = $ffold(f, i + 2 * d, i + 3 * d, r);
                                    let blo = $bfold(b, i, i + d, r);
                                    let bhi = $bfold(b, i + 2 * d, i + 3 * d, r);
                                    flc[k] = flo;
                                    fhc[k] = fhi;
                                    blc[k] = blo;
                                    bhc[k] = bhi;
                                    u0 += flo * blo;
                                    u2 += (flo + fhi) * (blo + bhi);
                                }
                                (u0, u2)
                            })
                            .reduce(zero2, sum2)
                    })
                    .reduce(zero2, sum2)
            } else {
                let chunk = block.max(1 << 12).min(half);
                debug_assert!(chunk.is_multiple_of(block) || chunk == half);
                nf.par_chunks_mut(chunk)
                    .zip(nb.par_chunks_mut(chunk))
                    .enumerate()
                    .map(|(ci, (fo, bo))| {
                        let mut u0 = F256::ZERO;
                        let mut u2 = F256::ZERO;
                        let out0 = ci * chunk;
                        for (jo, (fblk, bblk)) in
                            fo.chunks_mut(block).zip(bo.chunks_mut(block)).enumerate()
                        {
                            let in0 = 2 * (out0 + jo * block);
                            for k in 0..d {
                                let flo = $ffold(f, in0 + k, in0 + d + k, r);
                                let fhi = $ffold(f, in0 + 2 * d + k, in0 + 3 * d + k, r);
                                let blo = $bfold(b, in0 + k, in0 + d + k, r);
                                let bhi = $bfold(b, in0 + 2 * d + k, in0 + 3 * d + k, r);
                                fblk[k] = flo;
                                fblk[d + k] = fhi;
                                bblk[k] = blo;
                                bblk[d + k] = bhi;
                                u0 += flo * blo;
                                u2 += (flo + fhi) * (blo + bhi);
                            }
                        }
                        (u0, u2)
                    })
                    .reduce(zero2, sum2)
            };
            (nf, nb, SumcheckMessage256 { u_0, u_2 })
        }
    };
}

/// One fold step `a[lo] + r*(a[hi] + a[lo])` per operand class. The split
/// step exploits base-VALUED (post-code-switch) F256 words: `r * x` for
/// base `x` is two F128 products — the third Karatsuba product is
/// identically zero (`p1 = r1*0`), so skipping it is value-identical.
#[inline]
fn fold_step_base(a: &[F128], lo: usize, hi: usize, r: F256) -> F256 {
    F256::from(a[lo]) + r * (a[hi] + a[lo])
}
#[inline]
fn fold_step_ext(a: &[F256], lo: usize, hi: usize, r: F256) -> F256 {
    a[lo] + r * (a[hi] + a[lo])
}
#[inline]
fn fold_step_split_base(a: &[F256], lo: usize, hi: usize, r: F256) -> F256 {
    debug_assert!(
        a[lo].c1.is_zero() && a[hi].c1.is_zero(),
        "split-base fold step requires base-valued words"
    );
    let x = a[hi].c0 + a[lo].c0;
    F256::new(a[lo].c0 + r.c0 * x, r.c1 * x)
}

fused_fold_msg!(
    fused_fold_msg_base,
    F128,
    F128,
    fold_step_base,
    fold_step_base
);
fused_fold_msg!(fused_fold_msg_ext, F256, F256, fold_step_ext, fold_step_ext);
fused_fold_msg!(
    fused_fold_msg_fbase,
    F256,
    F256,
    fold_step_split_base,
    fold_step_ext
);

/// The fused kernel applies exactly when `next_round_msg(folded, d)` keeps
/// the blocked pairing the sweep produces.
#[inline]
fn fused_fold_applies(len: usize, d: usize) -> bool {
    let half = len / 2;
    if d == 1 { half >= 2 } else { half >= 2 * d }
}

/// Evaluate a round-1 [`FoldLookahead`] (base-field quadratic coefficients
/// from the combine pass) at an EXTENSION fold challenge — the F256 ladder's
/// O(1) skip message. Exact polynomial identity (`u(r) = c₀ + r·c₁ + r²·c₂`
/// holds over any extension of the coefficient field), so the transcript is
/// bit-identical to the fold-then-message path.
fn lookahead_eval256(la: &FoldLookahead, r: F256) -> SumcheckMessage256 {
    let r2 = r * r;
    SumcheckMessage256 {
        u_0: F256::from(la.u0[0]) + r * la.u0[1] + r2 * la.u0[2],
        u_2: F256::from(la.u2[0]) + r * la.u2[1] + r2 * la.u2[2],
    }
}

/// Correct a round-1 lookahead for an L0 OOD β-glue: the coefficients are
/// LINEAR in the basis, so `b += β·eq_z` adds `β ·` (the (f, eq_z) pair's
/// own coefficients, from [`round_msg_eval_and_lookahead`]).
fn lookahead_add_scaled(la: &mut FoldLookahead, ood: &FoldLookahead, beta: F128) {
    for i in 0..3 {
        la.u0[i] += beta * ood.u0[i];
        la.u2[i] += beta * ood.u2[i];
    }
}

/// Mid-ladder alternation state: the NEXT round's message as a quadratic in
/// its not-yet-sampled F256 fold challenge — the extension-coefficient
/// counterpart of the combine pass's base-field [`FoldLookahead`]. Produced
/// INSIDE fold passes (over the freshly folded outputs, +4 muls per output
/// quad, four of the eight products shared with this round's message);
/// consumed as an O(1) skip. Exact polynomial identity → transcripts are
/// bit-identical to the fold-then-message path.
/// Output-size cap for producing mid-chain coefficients: above it, the
/// accumulation's extra products (and the wide accumulator's register
/// pressure) measurably lose to the fold pass they would save (M4 Max,
/// m32 lane-major: +10 ms at 2^23 outputs vs a ~7 ms fold); below it, the
/// skip is a clean win by both traffic and product counts. The caller
/// lookahead (free coefficients from the combine pass) is not capped.
const LA_MAX_OUTPUTS: usize = 1 << 19;

#[derive(Clone, Copy)]
pub(super) struct La256 {
    u0: [F256; 3],
    u2: [F256; 3],
}

impl La256 {
    #[inline]
    fn eval(&self, r: F256) -> SumcheckMessage256 {
        let r2 = r * r;
        SumcheckMessage256 {
            u_0: self.u0[0] + r * self.u0[1] + r2 * self.u0[2],
            u_2: self.u2[0] + r * self.u2[1] + r2 * self.u2[2],
        }
    }
}

/// Per-quad accumulation over FOLDED outputs, under the pairing the NEXT
/// fold uses: this round's message (pairs `(0,1)`, `(2,3)`) plus the next
/// round's quadratic coefficients (next fold pairs `(0,1)`→slot 0,
/// `(2,3)`→slot 1; next message pairs the slots). Char-2 collapses the
/// Karatsuba middle factors: `A0+dA0 = af[1]`, `S+dS = af[1]+af[3]`.
/// Accumulator layout: `[u0, u2, c0, c1, c2, d0, d1, d2]`.
#[inline(always)]
fn la_msg_quad(af: &[F256; 4], ab: &[F256; 4], acc: &mut [F256; 8]) {
    let m1 = af[0] * ab[0];
    let p1 = af[2] * ab[2];
    let m2 = (af[0] + af[1]) * (ab[0] + ab[1]);
    let p2 = (af[2] + af[3]) * (ab[2] + ab[3]);
    let m3 = af[1] * ab[1];
    let n1 = (af[0] + af[2]) * (ab[0] + ab[2]);
    let n3 = (af[1] + af[3]) * (ab[1] + ab[3]);
    let n2 = (af[0] + af[1] + af[2] + af[3]) * (ab[0] + ab[1] + ab[2] + ab[3]);
    acc[0] += m1 + p1;
    acc[1] += m2 + p2;
    acc[2] += m1;
    acc[3] += m1 + m2 + m3;
    acc[4] += m2;
    acc[5] += n1;
    acc[6] += n1 + n2 + n3;
    acc[7] += n2;
}

#[inline]
fn acc8_add(mut a: [F256; 8], b: [F256; 8]) -> [F256; 8] {
    for k in 0..8 {
        a[k] += b[k];
    }
    a
}

#[inline]
fn acc8_finish(acc: [F256; 8]) -> (SumcheckMessage256, La256) {
    (
        SumcheckMessage256 {
            u_0: acc[0],
            u_2: acc[1],
        },
        La256 {
            u0: [acc[2], acc[3], acc[4]],
            u2: [acc[5], acc[6], acc[7]],
        },
    )
}

#[inline]
fn fold2_step_base(a: &[F128], i: usize, r0: F256, r1: F256) -> F256 {
    let lo = F256::from(a[i]) + r0 * (a[i + 1] + a[i]);
    let hi = F256::from(a[i + 2]) + r0 * (a[i + 3] + a[i + 2]);
    lo + r1 * (hi + lo)
}

#[inline]
fn fold2_step_ext_blocked(a: &[F256], i: usize, d: usize, r0: F256, r1: F256) -> F256 {
    let lo = fold_step_ext(a, i, i + d, r0);
    let hi = fold_step_ext(a, i + 2 * d, i + 3 * d, r0);
    lo + r1 * (hi + lo)
}

/// FUSED double fold from the BASE arrays (LSB pairing): fold by `(r0, r1)`
/// straight to the quarter-size state — the half-size intermediate never
/// exists — and accumulate the next round's message plus, when `want_la`,
/// the round after's coefficients ([`La256`]) over the same in-register
/// outputs. VALUE-IDENTICAL to the unfused chain: the per-element
/// expression composes the fold steps verbatim and all sums are
/// XOR-additive.
fn fused_fold2_msg_base(
    f: &[F128],
    b: &[F128],
    r0: F256,
    r1: F256,
    want_la: bool,
) -> (Vec<F256>, Vec<F256>, SumcheckMessage256, Option<La256>) {
    debug_assert_eq!(f.len(), b.len());
    let quarter = f.len() / 4;
    debug_assert!(quarter >= 2 && quarter.is_power_of_two());
    let want_la = want_la && (4..=LA_MAX_OUTPUTS).contains(&quarter);
    let mut nf = take_f256(quarter);
    let mut nb = take_f256(quarter);
    let gran = if want_la { 4 } else { 2 };
    let chunk = (1usize << 12).clamp(gran, quarter);
    debug_assert!(chunk.is_multiple_of(gran));
    let acc = nf
        .par_chunks_mut(chunk)
        .zip(nb.par_chunks_mut(chunk))
        .enumerate()
        .map(|(ci, (fo, bo))| {
            let out0 = ci * chunk;
            let mut acc = [F256::ZERO; 8];
            for (k, (fslot, bslot)) in fo.iter_mut().zip(bo.iter_mut()).enumerate() {
                let i = 4 * (out0 + k);
                *fslot = fold2_step_base(f, i, r0, r1);
                *bslot = fold2_step_base(b, i, r0, r1);
            }
            if want_la {
                for (fq, bq) in fo.as_chunks::<4>().0.iter().zip(bo.as_chunks::<4>().0) {
                    la_msg_quad(fq, bq, &mut acc);
                }
            } else {
                for j in 0..fo.len() / 2 {
                    acc[0] += fo[2 * j] * bo[2 * j];
                    acc[1] += (fo[2 * j] + fo[2 * j + 1]) * (bo[2 * j] + bo[2 * j + 1]);
                }
            }
            acc
        })
        .reduce(|| [F256::ZERO; 8], acc8_add);
    let (msg, la) = acc8_finish(acc);
    (nf, nb, msg, want_la.then_some(la))
}

/// FUSED double fold of the EXTENSION state (block pairing `d`): the
/// mid-chain pass of the alternating schedule — absorbs the deferred skip
/// challenge `r0` and this round's `r1` in one sweep, quarter-size outputs,
/// message + optional next-round coefficients. The lane-major geometry
/// (few huge 4d-superblocks) splits the k dimension.
fn fused_fold2_msg_ext(
    f: &[F256],
    b: &[F256],
    r0: F256,
    r1: F256,
    d: usize,
    want_la: bool,
) -> (Vec<F256>, Vec<F256>, SumcheckMessage256, Option<La256>) {
    debug_assert_eq!(f.len(), b.len());
    let quarter = f.len() / 4;
    // `quarter >= d` is the fold's own requirement; the MESSAGE additionally
    // needs a blocked pair (`quarter >= 2d`) — drains (which discard it) are
    // the only callers below that.
    debug_assert!(d.is_power_of_two() && quarter.is_power_of_two() && quarter >= d);
    let want_la = want_la && quarter >= 4 * d && quarter <= LA_MAX_OUTPUTS;
    let sblock = if want_la { 4 * d } else { 2 * d };
    let mut nf = take_f256(quarter);
    let mut nb = take_f256(quarter);
    // Output o = b·d + w composes inputs {4bd+w, +d, +2d, +3d}; a superblock
    // of `sblock` outputs reads the 4·sblock inputs at 4·(superblock start).
    // The split-based big-block branch requires FULL superblocks; smaller
    // arrays (the big-d drain shapes) take the chunked branch, whose
    // n_segs-aware loop writes every output.
    if quarter == d {
        // Degenerate (drain-only) shape: one block, no message pairs —
        // parallelize the k dimension directly. The message/coefficients
        // are meaningless here (callers discard them).
        nf.par_chunks_mut(1 << 12)
            .zip(nb.par_chunks_mut(1 << 12))
            .enumerate()
            .for_each(|(ci, (fo, bo))| {
                let k0 = ci << 12;
                for (k, (fs, bs)) in fo.iter_mut().zip(bo.iter_mut()).enumerate() {
                    *fs = fold2_step_ext_blocked(f, k0 + k, d, r0, r1);
                    *bs = fold2_step_ext_blocked(b, k0 + k, d, r0, r1);
                }
            });
        return (
            nf,
            nb,
            SumcheckMessage256 {
                u_0: F256::ZERO,
                u_2: F256::ZERO,
            },
            None,
        );
    }
    let acc = if sblock >= (1 << 16) && quarter >= sblock {
        // Few huge superblocks: split each segment's k dimension too.
        const KC: usize = 1 << 13;
        nf.par_chunks_mut(sblock)
            .zip(nb.par_chunks_mut(sblock))
            .enumerate()
            .map(|(jb, (fblk, bblk))| {
                let out0 = jb * sblock;
                let in0 = 4 * out0;
                if want_la {
                    // 4 d-segments per superblock (the la quad).
                    let (f0, fr) = fblk.split_at_mut(d);
                    let (f1, fr2) = fr.split_at_mut(d);
                    let (f2, f3) = fr2.split_at_mut(d);
                    let (b0, br) = bblk.split_at_mut(d);
                    let (b1, br2) = br.split_at_mut(d);
                    let (b2, b3) = br2.split_at_mut(d);
                    f0.par_chunks_mut(KC)
                        .zip(f1.par_chunks_mut(KC))
                        .zip(f2.par_chunks_mut(KC).zip(f3.par_chunks_mut(KC)))
                        .zip(
                            b0.par_chunks_mut(KC)
                                .zip(b1.par_chunks_mut(KC))
                                .zip(b2.par_chunks_mut(KC).zip(b3.par_chunks_mut(KC))),
                        )
                        .enumerate()
                        .map(
                            |(kc, (((fc0, fc1), (fc2, fc3)), ((bc0, bc1), (bc2, bc3))))| {
                                let k0 = kc * KC;
                                let mut acc = [F256::ZERO; 8];
                                for k in 0..fc0.len() {
                                    let i = in0 + k0 + k;
                                    let af = [
                                        fold2_step_ext_blocked(f, i, d, r0, r1),
                                        fold2_step_ext_blocked(f, i + 4 * d, d, r0, r1),
                                        fold2_step_ext_blocked(f, i + 8 * d, d, r0, r1),
                                        fold2_step_ext_blocked(f, i + 12 * d, d, r0, r1),
                                    ];
                                    let ab = [
                                        fold2_step_ext_blocked(b, i, d, r0, r1),
                                        fold2_step_ext_blocked(b, i + 4 * d, d, r0, r1),
                                        fold2_step_ext_blocked(b, i + 8 * d, d, r0, r1),
                                        fold2_step_ext_blocked(b, i + 12 * d, d, r0, r1),
                                    ];
                                    fc0[k] = af[0];
                                    fc1[k] = af[1];
                                    fc2[k] = af[2];
                                    fc3[k] = af[3];
                                    bc0[k] = ab[0];
                                    bc1[k] = ab[1];
                                    bc2[k] = ab[2];
                                    bc3[k] = ab[3];
                                    la_msg_quad(&af, &ab, &mut acc);
                                }
                                acc
                            },
                        )
                        .reduce(|| [F256::ZERO; 8], acc8_add)
                } else {
                    // 2 d-segments (message pairs only — the drain shape).
                    let (f0, f1) = fblk.split_at_mut(d);
                    let (b0, b1) = bblk.split_at_mut(d);
                    f0.par_chunks_mut(KC)
                        .zip(f1.par_chunks_mut(KC))
                        .zip(b0.par_chunks_mut(KC).zip(b1.par_chunks_mut(KC)))
                        .enumerate()
                        .map(|(kc, ((fc0, fc1), (bc0, bc1)))| {
                            let k0 = kc * KC;
                            let mut acc = [F256::ZERO; 8];
                            for k in 0..fc0.len() {
                                let i = in0 + k0 + k;
                                let flo = fold2_step_ext_blocked(f, i, d, r0, r1);
                                let fhi = fold2_step_ext_blocked(f, i + 4 * d, d, r0, r1);
                                let blo = fold2_step_ext_blocked(b, i, d, r0, r1);
                                let bhi = fold2_step_ext_blocked(b, i + 4 * d, d, r0, r1);
                                fc0[k] = flo;
                                fc1[k] = fhi;
                                bc0[k] = blo;
                                bc1[k] = bhi;
                                acc[0] += flo * blo;
                                acc[1] += (flo + fhi) * (blo + bhi);
                            }
                            acc
                        })
                        .reduce(|| [F256::ZERO; 8], acc8_add)
                }
            })
            .reduce(|| [F256::ZERO; 8], acc8_add)
    } else {
        let chunk = sblock.max(1 << 12).min(quarter);
        debug_assert!(chunk.is_multiple_of(sblock) || chunk == quarter);
        let dd = d.min(quarter);
        nf.par_chunks_mut(chunk)
            .zip(nb.par_chunks_mut(chunk))
            .enumerate()
            .map(|(ci, (fo, bo))| {
                let out0 = ci * chunk;
                let mut acc = [F256::ZERO; 8];
                for (js, (fblk, bblk)) in
                    fo.chunks_mut(sblock).zip(bo.chunks_mut(sblock)).enumerate()
                {
                    let ob = out0 + js * sblock;
                    let in0 = 4 * ob;
                    let n_segs = fblk.len() / dd;
                    for k in 0..dd {
                        let mut outs = [F256::ZERO; 4];
                        let mut outsb = [F256::ZERO; 4];
                        for s in 0..n_segs {
                            outs[s] = fold2_step_ext_blocked(f, in0 + 4 * s * d + k, d, r0, r1);
                            outsb[s] = fold2_step_ext_blocked(b, in0 + 4 * s * d + k, d, r0, r1);
                            fblk[s * dd + k] = outs[s];
                            bblk[s * dd + k] = outsb[s];
                        }
                        if n_segs == 4 {
                            la_msg_quad(&outs, &outsb, &mut acc);
                        } else if n_segs >= 2 {
                            acc[0] += outs[0] * outsb[0];
                            acc[1] += (outs[0] + outs[1]) * (outsb[0] + outsb[1]);
                        }
                    }
                }
                acc
            })
            .reduce(|| [F256::ZERO; 8], acc8_add)
    };
    let (msg, la) = acc8_finish(acc);
    (nf, nb, msg, want_la.then_some(la))
}

/// FUSED single fold + message + next-round coefficients for a level's
/// FIRST round (the just-split base-valued f-side, LSB pairing) — the
/// alternation ENTRY of every recursive level: the pass that already runs
/// also produces the coefficients that make the level's second round a
/// skip.
fn fused_fold_msg_la_fbase(
    f: &[F256],
    b: &[F256],
    r: F256,
) -> (Vec<F256>, Vec<F256>, SumcheckMessage256, Option<La256>) {
    debug_assert_eq!(f.len(), b.len());
    let half = f.len() / 2;
    debug_assert!(half.is_power_of_two() && half >= 4);
    debug_assert!(half <= LA_MAX_OUTPUTS, "caller gates by size");
    let mut nf = take_f256(half);
    let mut nb = take_f256(half);
    let chunk = (1usize << 12).clamp(4, half);
    debug_assert!(chunk.is_multiple_of(4));
    let acc = nf
        .par_chunks_mut(chunk)
        .zip(nb.par_chunks_mut(chunk))
        .enumerate()
        .map(|(ci, (fo, bo))| {
            let out0 = ci * chunk;
            let mut acc = [F256::ZERO; 8];
            for (q, (fq_o, bq_o)) in fo
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(bo.as_chunks_mut::<4>().0)
                .enumerate()
            {
                let base = 2 * (out0 + 4 * q);
                let mut af = [F256::ZERO; 4];
                let mut ab = [F256::ZERO; 4];
                for t in 0..4 {
                    af[t] = fold_step_split_base(f, base + 2 * t, base + 2 * t + 1, r);
                    ab[t] = fold_step_ext(b, base + 2 * t, base + 2 * t + 1, r);
                }
                *fq_o = af;
                *bq_o = ab;
                la_msg_quad(&af, &ab, &mut acc);
            }
            acc
        })
        .reduce(|| [F256::ZERO; 8], acc8_add);
    let (msg, la) = acc8_finish(acc);
    (nf, nb, msg, Some(la))
}

/// [`fused_fold_msg_la_fbase`] for the ladder-entry BASE arrays (both sides
/// F128, LSB): the no-caller-lookahead fallback that still starts the
/// alternating schedule at round 0.
fn fused_fold_msg_la_base(
    f: &[F128],
    b: &[F128],
    r: F256,
) -> (Vec<F256>, Vec<F256>, SumcheckMessage256, Option<La256>) {
    debug_assert_eq!(f.len(), b.len());
    let half = f.len() / 2;
    debug_assert!(half.is_power_of_two() && half >= 4);
    let mut nf = take_f256(half);
    let mut nb = take_f256(half);
    let chunk = (1usize << 12).clamp(4, half);
    debug_assert!(chunk.is_multiple_of(4));
    let acc = nf
        .par_chunks_mut(chunk)
        .zip(nb.par_chunks_mut(chunk))
        .enumerate()
        .map(|(ci, (fo, bo))| {
            let out0 = ci * chunk;
            let mut acc = [F256::ZERO; 8];
            for (q, (fq_o, bq_o)) in fo
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(bo.as_chunks_mut::<4>().0)
                .enumerate()
            {
                let base = 2 * (out0 + 4 * q);
                let mut af = [F256::ZERO; 4];
                let mut ab = [F256::ZERO; 4];
                for t in 0..4 {
                    af[t] = fold_step_base(f, base + 2 * t, base + 2 * t + 1, r);
                    ab[t] = fold_step_base(b, base + 2 * t, base + 2 * t + 1, r);
                }
                *fq_o = af;
                *bq_o = ab;
                la_msg_quad(&af, &ab, &mut acc);
            }
            acc
        })
        .reduce(|| [F256::ZERO; 8], acc8_add);
    let (msg, la) = acc8_finish(acc);
    (nf, nb, msg, Some(la))
}

/// FUSED first virtual fold: one sweep folds `f` by `r`, EVALUATES the
/// (already-folded) virtual basis directly into `nb`, and accumulates the
/// next round message — the fold-0 counterpart of [`fused_fold_msg!`],
/// saving the separate materialize and message passes (a full re-read of
/// both outputs). Value-identical: `nb[o]` is the same per-term XOR-sum
/// `fill` writes, the fold expression is verbatim `fold_base`, and the
/// message pairing matches `next_round_msg(nf, nb, d)`; u_0/u_2 are
/// XOR-additive so sweep order cannot change them. Caller guards with
/// [`fused_fold_applies`].
fn fused_first_fold_virtual(
    f: &[F128],
    basis: &VirtualEqBasis256,
    r: F256,
    d: usize,
) -> (Vec<F256>, Vec<F256>, SumcheckMessage256) {
    let half = f.len() / 2;
    debug_assert_eq!(basis.len(), half);
    debug_assert!(d.is_power_of_two() && half.is_power_of_two());
    let block = 2 * d;
    let zero2 = || (F256::ZERO, F256::ZERO);
    let sum2 = |(a0, a2): (F256, F256), (b0, b2): (F256, F256)| (a0 + b0, a2 + b2);
    let mut nf = take_f256(half);
    let mut nb = take_f256(half);
    let (u_0, u_2) = if block >= (1 << 16) {
        const KC: usize = 1 << 13;
        nf.par_chunks_mut(block)
            .zip(nb.par_chunks_mut(block))
            .enumerate()
            .map(|(jb, (fblk, bblk))| {
                let out0 = jb * block;
                let in0 = 2 * out0;
                let (flo_h, fhi_h) = fblk.split_at_mut(d);
                let (blo_h, bhi_h) = bblk.split_at_mut(d);
                flo_h
                    .par_chunks_mut(KC)
                    .zip(fhi_h.par_chunks_mut(KC))
                    .zip(blo_h.par_chunks_mut(KC).zip(bhi_h.par_chunks_mut(KC)))
                    .enumerate()
                    .map(|(kc, ((flc, fhc), (blc, bhc)))| {
                        let k0 = kc * KC;
                        let mut u0 = F256::ZERO;
                        let mut u2 = F256::ZERO;
                        for k in 0..flc.len() {
                            let i = in0 + k0 + k;
                            let o = out0 + k0 + k;
                            let flo = F256::from(f[i]) + r * (f[i + d] + f[i]);
                            let fhi = F256::from(f[i + 2 * d]) + r * (f[i + 3 * d] + f[i + 2 * d]);
                            let blo = basis.value_sum_at(o);
                            let bhi = basis.value_sum_at(o + d);
                            flc[k] = flo;
                            fhc[k] = fhi;
                            blc[k] = blo;
                            bhc[k] = bhi;
                            u0 += flo * blo;
                            u2 += (flo + fhi) * (blo + bhi);
                        }
                        (u0, u2)
                    })
                    .reduce(zero2, sum2)
            })
            .reduce(zero2, sum2)
    } else {
        let chunk = block.max(1 << 12).min(half);
        debug_assert!(chunk.is_multiple_of(block) || chunk == half);
        nf.par_chunks_mut(chunk)
            .zip(nb.par_chunks_mut(chunk))
            .enumerate()
            .map(|(ci, (fo, bo))| {
                let mut u0 = F256::ZERO;
                let mut u2 = F256::ZERO;
                let out0 = ci * chunk;
                for (jo, (fblk, bblk)) in fo.chunks_mut(block).zip(bo.chunks_mut(block)).enumerate()
                {
                    let ob = out0 + jo * block;
                    let in0 = 2 * ob;
                    for k in 0..d {
                        let flo = F256::from(f[in0 + k]) + r * (f[in0 + d + k] + f[in0 + k]);
                        let fhi = F256::from(f[in0 + 2 * d + k])
                            + r * (f[in0 + 3 * d + k] + f[in0 + 2 * d + k]);
                        let blo = basis.value_sum_at(ob + k);
                        let bhi = basis.value_sum_at(ob + d + k);
                        fblk[k] = flo;
                        fblk[d + k] = fhi;
                        bblk[k] = blo;
                        bblk[d + k] = bhi;
                        u0 += flo * blo;
                        u2 += (flo + fhi) * (blo + bhi);
                    }
                }
                (u0, u2)
            })
            .reduce(zero2, sum2)
    };
    (nf, nb, SumcheckMessage256 { u_0, u_2 })
}

/// FUSED double fold with a VIRTUAL basis: fold `f` by `(r0, r1)` (block
/// pairing `d`) straight to the quarter-size state, EVALUATE the
/// twice-folded virtual basis directly into `nb`, and accumulate the next
/// round's message plus (when `want_la`) the round after's coefficients —
/// the lane-major alternation entry. `basis` must already be folded by BOTH
/// challenges; its index space is exactly the output's. Value-identical to
/// fold → materialize → fold → message: the fold expression composes the two
/// steps verbatim, `nb[o]` is the same per-term XOR-sum `fill` writes, the
/// pairings match `next_round_msg(nf, nb, d)`, and all sums are
/// XOR-additive.
fn fused_first_fold2_virtual(
    f: &[F128],
    basis: &VirtualEqBasis256,
    r0: F256,
    r1: F256,
    d: usize,
    want_la: bool,
) -> (Vec<F256>, Vec<F256>, SumcheckMessage256, Option<La256>) {
    let quarter = f.len() / 4;
    debug_assert_eq!(basis.len(), quarter);
    debug_assert!(d.is_power_of_two() && quarter.is_power_of_two() && quarter >= 2 * d);
    let want_la = want_la && quarter >= 4 * d && quarter <= LA_MAX_OUTPUTS;
    let sblock = if want_la { 4 * d } else { 2 * d };
    let mut nf = take_f256(quarter);
    let mut nb = take_f256(quarter);
    // Output o = b·d + w composes f inputs {4bd+w, +d, +2d, +3d}.
    let fold2 = |i: usize| -> F256 {
        let lo = F256::from(f[i]) + r0 * (f[i + d] + f[i]);
        let hi = F256::from(f[i + 2 * d]) + r0 * (f[i + 3 * d] + f[i + 2 * d]);
        lo + r1 * (hi + lo)
    };
    let acc = if sblock >= (1 << 16) && quarter >= sblock {
        const KC: usize = 1 << 13;
        nf.par_chunks_mut(sblock)
            .zip(nb.par_chunks_mut(sblock))
            .enumerate()
            .map(|(jb, (fblk, bblk))| {
                let out0 = jb * sblock;
                let in0 = 4 * out0;
                if want_la {
                    let (f0, fr) = fblk.split_at_mut(d);
                    let (f1, fr2) = fr.split_at_mut(d);
                    let (f2, f3) = fr2.split_at_mut(d);
                    let (b0, br) = bblk.split_at_mut(d);
                    let (b1, br2) = br.split_at_mut(d);
                    let (b2, b3) = br2.split_at_mut(d);
                    f0.par_chunks_mut(KC)
                        .zip(f1.par_chunks_mut(KC))
                        .zip(f2.par_chunks_mut(KC).zip(f3.par_chunks_mut(KC)))
                        .zip(
                            b0.par_chunks_mut(KC)
                                .zip(b1.par_chunks_mut(KC))
                                .zip(b2.par_chunks_mut(KC).zip(b3.par_chunks_mut(KC))),
                        )
                        .enumerate()
                        .map(
                            |(kc, (((fc0, fc1), (fc2, fc3)), ((bc0, bc1), (bc2, bc3))))| {
                                let k0 = kc * KC;
                                let mut acc = [F256::ZERO; 8];
                                for k in 0..fc0.len() {
                                    let i = in0 + k0 + k;
                                    let o = out0 + k0 + k;
                                    let af = [
                                        fold2(i),
                                        fold2(i + 4 * d),
                                        fold2(i + 8 * d),
                                        fold2(i + 12 * d),
                                    ];
                                    let ab = [
                                        basis.value_sum_at(o),
                                        basis.value_sum_at(o + d),
                                        basis.value_sum_at(o + 2 * d),
                                        basis.value_sum_at(o + 3 * d),
                                    ];
                                    fc0[k] = af[0];
                                    fc1[k] = af[1];
                                    fc2[k] = af[2];
                                    fc3[k] = af[3];
                                    bc0[k] = ab[0];
                                    bc1[k] = ab[1];
                                    bc2[k] = ab[2];
                                    bc3[k] = ab[3];
                                    la_msg_quad(&af, &ab, &mut acc);
                                }
                                acc
                            },
                        )
                        .reduce(|| [F256::ZERO; 8], acc8_add)
                } else {
                    let (f0, f1) = fblk.split_at_mut(d);
                    let (b0, b1) = bblk.split_at_mut(d);
                    f0.par_chunks_mut(KC)
                        .zip(f1.par_chunks_mut(KC))
                        .zip(b0.par_chunks_mut(KC).zip(b1.par_chunks_mut(KC)))
                        .enumerate()
                        .map(|(kc, ((fc0, fc1), (bc0, bc1)))| {
                            let k0 = kc * KC;
                            let mut acc = [F256::ZERO; 8];
                            for k in 0..fc0.len() {
                                let i = in0 + k0 + k;
                                let o = out0 + k0 + k;
                                let flo = fold2(i);
                                let fhi = fold2(i + 4 * d);
                                let blo = basis.value_sum_at(o);
                                let bhi = basis.value_sum_at(o + d);
                                fc0[k] = flo;
                                fc1[k] = fhi;
                                bc0[k] = blo;
                                bc1[k] = bhi;
                                acc[0] += flo * blo;
                                acc[1] += (flo + fhi) * (blo + bhi);
                            }
                            acc
                        })
                        .reduce(|| [F256::ZERO; 8], acc8_add)
                }
            })
            .reduce(|| [F256::ZERO; 8], acc8_add)
    } else {
        let chunk = sblock.max(1 << 12).min(quarter);
        debug_assert!(chunk.is_multiple_of(sblock) || chunk == quarter);
        nf.par_chunks_mut(chunk)
            .zip(nb.par_chunks_mut(chunk))
            .enumerate()
            .map(|(ci, (fo, bo))| {
                let out0 = ci * chunk;
                let mut acc = [F256::ZERO; 8];
                for (js, (fblk, bblk)) in
                    fo.chunks_mut(sblock).zip(bo.chunks_mut(sblock)).enumerate()
                {
                    let ob = out0 + js * sblock;
                    let in0 = 4 * ob;
                    let n_segs = fblk.len() / d;
                    for k in 0..d {
                        let mut af = [F256::ZERO; 4];
                        let mut ab = [F256::ZERO; 4];
                        for s in 0..n_segs {
                            af[s] = fold2(in0 + 4 * s * d + k);
                            ab[s] = basis.value_sum_at(ob + s * d + k);
                            fblk[s * d + k] = af[s];
                            bblk[s * d + k] = ab[s];
                        }
                        if n_segs == 4 {
                            la_msg_quad(&af, &ab, &mut acc);
                        } else {
                            acc[0] += af[0] * ab[0];
                            acc[1] += (af[0] + af[1]) * (ab[0] + ab[1]);
                        }
                    }
                }
                acc
            })
            .reduce(|| [F256::ZERO; 8], acc8_add)
    };
    let (msg, la) = acc8_finish(acc);
    (nf, nb, msg, want_la.then_some(la))
}

struct VirtualEqTerm256 {
    coords: Vec<F128>,
    scale: F256,
    lo: Vec<F128>,
    hi: Vec<F128>,
    n_lo: usize,
}

impl VirtualEqTerm256 {
    fn from_base(term: VirtualEqTerm) -> Self {
        Self {
            coords: term.coords,
            scale: F256::from(term.scale),
            lo: term.lo,
            hi: term.hi,
            n_lo: term.n_lo,
        }
    }

    fn rebuild(&mut self) {
        self.n_lo = self.coords.len() / 2;
        self.lo = build_eq_scaled_parallel(&self.coords[..self.n_lo], F128::ONE);
        self.hi = build_eq_scaled_parallel(&self.coords[self.n_lo..], F128::ONE);
    }

    fn fold_coord(&mut self, p: usize, r: F256) {
        self.scale *= F256::ONE + F256::from(self.coords[p]) + r;
        self.coords.remove(p);
        self.rebuild();
    }

    fn len(&self) -> usize {
        1usize << self.coords.len()
    }

    fn value_at(&self, u: usize) -> F256 {
        let mask = (1usize << self.n_lo) - 1;
        self.scale * (self.lo[u & mask] * self.hi[u >> self.n_lo])
    }

    fn add_to(&self, out: &mut [F256], g0: usize) {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot += self.value_at(g0 + i);
        }
    }
}

pub(super) struct VirtualEqBasis256 {
    terms: Vec<VirtualEqTerm256>,
}

impl VirtualEqBasis256 {
    pub(super) fn from_base(value: VirtualEqBasis) -> Self {
        Self {
            terms: value
                .terms
                .into_iter()
                .map(VirtualEqTerm256::from_base)
                .collect(),
        }
    }

    pub(super) fn fold_coord(&mut self, p: usize, r: F256) {
        for term in &mut self.terms {
            term.fold_coord(p, r);
        }
    }

    fn len(&self) -> usize {
        self.terms[0].len()
    }

    fn fill(&self, out: &mut [F256], g0: usize) {
        out.fill(F256::ZERO);
        for term in &self.terms {
            term.add_to(out, g0);
        }
    }

    fn materialize(&self) -> Vec<F256> {
        // Pooled + parallel: `fill` writes every slot of its chunk (zero
        // then add), so an uninitialized pooled buffer is fine.
        let mut out = take_f256(self.len());
        out.par_chunks_mut(1 << 12)
            .enumerate()
            .for_each(|(i, chunk)| self.fill(chunk, i << 12));
        out
    }

    /// The basis value at one index: what `fill`/`materialize` write there
    /// (the per-term XOR-sum, order-free).
    #[inline]
    fn value_sum_at(&self, u: usize) -> F256 {
        self.terms
            .iter()
            .fold(F256::ZERO, |acc, term| acc + term.value_at(u))
    }
}

enum PendingBasis {
    Extension(Vec<F256>),
}

pub(super) struct SumcheckProver256 {
    initial_f: Option<Vec<F128>>,
    initial_b: Option<Vec<F128>>,
    f: Vec<F256>,
    combined_basis: Vec<F256>,
    transcript: Vec<SumcheckMessage256>,
    pending: Option<PendingBasis>,
}

impl SumcheckProver256 {
    pub(super) fn new(f: Vec<F128>, b: Option<Vec<F128>>, first: SumcheckMessage) -> Self {
        Self {
            initial_f: Some(f),
            initial_b: b,
            f: Vec::new(),
            combined_basis: Vec::new(),
            transcript: vec![SumcheckMessage256 {
                u_0: F256::from(first.u_0),
                u_2: F256::from(first.u_2),
            }],
            pending: None,
        }
    }

    pub(super) fn first_fold_materialized(&mut self, r: F256, d: usize) -> SumcheckMessage256 {
        let f = self.initial_f.take().expect("first fold already consumed");
        let b = self.initial_b.take().expect("materialized basis missing");
        let msg = if fused_fold_applies(f.len(), d) {
            let (nf, nb, msg) = fused_fold_msg_base(&f, &b, r, d);
            self.f = nf;
            self.combined_basis = nb;
            msg
        } else {
            self.f = fold_base(&f, r, d);
            self.combined_basis = fold_base(&b, r, d);
            next_round_msg(&self.f, &self.combined_basis, d)
        };
        self.transcript.push(msg);
        msg
    }

    pub(super) fn first_fold_jit(
        &mut self,
        r: F256,
        d: usize,
        fill: BasisWindowFn<'_>,
    ) -> SumcheckMessage256 {
        let f = self.initial_f.take().expect("first fold already consumed");
        (self.f, self.combined_basis) = fold_base_fill(&f, fill, r, d);
        let msg = next_round_msg(&self.f, &self.combined_basis, d);
        self.transcript.push(msg);
        msg
    }

    /// Fold `f` by the first challenge and MATERIALIZE the (already-folded)
    /// virtual basis once — the F128 LazyRsPair's "half materializes at
    /// fold 0" design. The F256 rewrite instead re-filled the whole basis
    /// SERIALLY at every initial fold (`round_msg_virtual`), which was the
    /// m32 leaf ladder's dominant cost (~230 ms of geometric serial eq
    /// evaluation; `[lig-prove-f256]` per-fold trace). After this call the
    /// basis is ordinary materialized state and later folds ride the fused
    /// kernel.
    pub(super) fn first_fold_virtual(
        &mut self,
        r: F256,
        d: usize,
        basis: &VirtualEqBasis256,
    ) -> SumcheckMessage256 {
        let f = self.initial_f.take().expect("first fold already consumed");
        let msg = if fused_fold_applies(f.len(), d) {
            let (nf, nb, msg) = fused_first_fold_virtual(&f, basis, r, d);
            self.f = nf;
            self.combined_basis = nb;
            msg
        } else {
            self.f = fold_base(&f, r, d);
            self.combined_basis = basis.materialize();
            debug_assert_eq!(self.combined_basis.len(), self.f.len());
            next_round_msg(&self.f, &self.combined_basis, d)
        };
        self.transcript.push(msg);
        msg
    }

    /// A SKIP round: the message came from a lookahead evaluation (the
    /// caller's base-field coefficients at round 1, or a mid-ladder
    /// [`La256`]); record it — the array fold is deferred into the next
    /// fold2 pass (or a drain).
    pub(super) fn push_skip_message(&mut self, msg: SumcheckMessage256) {
        self.transcript.push(msg);
    }

    /// Whether the ladder-entry base arrays are still unconsumed (the first
    /// fold has not run) — decides which fold2 variant a deferred challenge
    /// takes.
    pub(super) fn initial_pending(&self) -> bool {
        self.initial_f.is_some()
    }

    /// The deferred fold fused with this round's: double-fold the base
    /// arrays by `(r0, r1)` in ONE pass straight to the quarter-size F256
    /// state — the half-size intermediate is never written. LSB pairing
    /// only (the caller lookahead never arrives on the lane-major
    /// materialized path).
    pub(super) fn first_fold2_materialized(
        &mut self,
        r0: F256,
        r1: F256,
        want_la: bool,
    ) -> (SumcheckMessage256, Option<La256>) {
        let f = self.initial_f.take().expect("first fold already consumed");
        let b = self.initial_b.take().expect("materialized basis missing");
        let (nf, nb, msg, la) = fused_fold2_msg_base(&f, &b, r0, r1, want_la);
        self.f = nf;
        self.combined_basis = nb;
        self.transcript.push(msg);
        (msg, la)
    }

    /// [`Self::first_fold2_materialized`] for the VIRTUAL basis (the
    /// lane-major merged opens): `basis` must already carry both fold_coords.
    pub(super) fn first_fold2_virtual(
        &mut self,
        r0: F256,
        r1: F256,
        d: usize,
        basis: &VirtualEqBasis256,
        want_la: bool,
    ) -> (SumcheckMessage256, Option<La256>) {
        let f = self.initial_f.take().expect("first fold already consumed");
        let (nf, nb, msg, la) = fused_first_fold2_virtual(&f, basis, r0, r1, d, want_la);
        self.f = nf;
        self.combined_basis = nb;
        self.transcript.push(msg);
        (msg, la)
    }

    /// Ladder-entry single fold + message + next-round coefficients (base
    /// arrays, LSB): starts the alternating schedule when no caller
    /// lookahead arrived. Falls back to the plain fused fold when the
    /// arrays are too small for coefficient quads.
    pub(super) fn first_fold_materialized_la(
        &mut self,
        r: F256,
    ) -> (SumcheckMessage256, Option<La256>) {
        if self
            .initial_f
            .as_ref()
            .is_none_or(|f| f.len() < 8 || f.len() / 2 > LA_MAX_OUTPUTS)
        {
            return (self.first_fold_materialized(r, 1), None);
        }
        let f = self.initial_f.take().expect("first fold already consumed");
        let b = self.initial_b.take().expect("materialized basis missing");
        let (nf, nb, msg, la) = fused_fold_msg_la_base(&f, &b, r);
        self.f = nf;
        self.combined_basis = nb;
        self.transcript.push(msg);
        (msg, la)
    }

    /// A recursive level's FIRST fold + the level's alternation entry: the
    /// fbase fold pass also emits the next round's coefficients. Runs AFTER
    /// every glue, so the coefficients are always fresh — no corrections.
    pub(super) fn fold_after_switch_la(&mut self, r: F256) -> (SumcheckMessage256, Option<La256>) {
        if !fused_fold_applies(self.f.len(), 1)
            || self.f.len() < 8
            || self.f.len() / 2 > LA_MAX_OUTPUTS
        {
            return (self.fold_after_switch(r), None);
        }
        let (nf, nb, msg, la) = fused_fold_msg_la_fbase(&self.f, &self.combined_basis, r);
        give_f256(replace(&mut self.f, nf));
        give_f256(replace(&mut self.combined_basis, nb));
        self.transcript.push(msg);
        (msg, la)
    }

    /// Mid-chain double fold (extension state, block pairing `d`): absorbs
    /// the deferred skip challenge and this round's in one pass.
    pub(super) fn mid_fold2(
        &mut self,
        r0: F256,
        r1: F256,
        d: usize,
        want_la: bool,
    ) -> (SumcheckMessage256, Option<La256>) {
        // The message needs a blocked pair; the degenerate shape below
        // fabricates a zero message and belongs to the drains only.
        debug_assert!(
            self.f.len() >= 8 * d,
            "mid_fold2 needs a blocked message pair; drains own the degenerate shape"
        );
        let (nf, nb, msg, la) =
            fused_fold2_msg_ext(&self.f, &self.combined_basis, r0, r1, d, want_la);
        give_f256(replace(&mut self.f, nf));
        give_f256(replace(&mut self.combined_basis, nb));
        self.transcript.push(msg);
        (msg, la)
    }

    /// Message-free double fold — the pre-switch drain when the previous
    /// round was a skip (the switch's own message replaces this round's).
    pub(super) fn fold2_drain(&mut self, r0: F256, r1: F256, d: usize) {
        let (nf, nb, _msg, _) =
            fused_fold2_msg_ext(&self.f, &self.combined_basis, r0, r1, d, false);
        give_f256(replace(&mut self.f, nf));
        give_f256(replace(&mut self.combined_basis, nb));
    }

    /// Message-free single fold — the drain before a switch (or the final
    /// residual) when this round's message came from a skip or is replaced.
    pub(super) fn drain(&mut self, r: F256, d: usize) {
        let nf = fold_extension(&self.f, r, d);
        let nb = fold_extension(&self.combined_basis, r, d);
        give_f256(replace(&mut self.f, nf));
        give_f256(replace(&mut self.combined_basis, nb));
    }

    pub(super) fn fold_materialized(&mut self, r: F256, d: usize) -> SumcheckMessage256 {
        let msg = if fused_fold_applies(self.f.len(), d) {
            let (nf, nb, msg) = fused_fold_msg_ext(&self.f, &self.combined_basis, r, d);
            // The replaced fold outputs cycle back to the shared pool; the
            // next fold (and the next prove) takes them warm instead of
            // faulting fresh pages.
            give_f256(replace(&mut self.f, nf));
            give_f256(replace(&mut self.combined_basis, nb));
            msg
        } else {
            self.f = fold_extension(&self.f, r, d);
            self.combined_basis = fold_extension(&self.combined_basis, r, d);
            next_round_msg(&self.f, &self.combined_basis, d)
        };
        self.transcript.push(msg);
        msg
    }

    pub(super) fn fold(&mut self, r: F256) -> SumcheckMessage256 {
        self.fold_materialized(r, 1)
    }

    /// The FIRST fold of a recursive level: the f-side is base-valued (the
    /// code switch just split it — `introduce_ood_with_eval` asserts the
    /// same invariant), so its products are F256×F128 = 2 muls instead of
    /// the generic 3. The b-side has been through `split_basis` and glue
    /// and stays generic. Value-identical to [`Self::fold`].
    pub(super) fn fold_after_switch(&mut self, r: F256) -> SumcheckMessage256 {
        if !fused_fold_applies(self.f.len(), 1) {
            return self.fold_materialized(r, 1);
        }
        let (nf, nb, msg) = fused_fold_msg_fbase(&self.f, &self.combined_basis, r, 1);
        give_f256(replace(&mut self.f, nf));
        give_f256(replace(&mut self.combined_basis, nb));
        self.transcript.push(msg);
        msg
    }

    /// Replace the just-produced next-round message with the message for the
    /// coordinate-split table. No transcript item is added: the code switch is
    /// a representation change at the same sumcheck boundary.
    pub(super) fn code_switch_and_replace_message(&mut self) -> SumcheckMessage256 {
        assert!(self.pending.is_none());
        let msg = self.code_switch_message();
        *self
            .transcript
            .last_mut()
            .expect("a fold message must precede a code switch") = msg;
        msg
    }

    /// The code switch when the last fold round emitted NO message (it was
    /// a skip or a drain under the alternating schedule): the switch's
    /// message is PUSHED as that round's — same transcript contents as
    /// fold-then-replace, one entry per round either way.
    pub(super) fn code_switch_and_push_message(&mut self) -> SumcheckMessage256 {
        assert!(self.pending.is_none());
        let msg = self.code_switch_message();
        self.transcript.push(msg);
        msg
    }

    fn code_switch_message(&mut self) -> SumcheckMessage256 {
        let words = split_coordinates(&self.f);
        let split_f = words.into_iter().map(F256::from).collect();
        give_f256(replace(&mut self.f, split_f));
        let split_b = split_basis(&self.combined_basis);
        give_f256(replace(&mut self.combined_basis, split_b));
        round_msg_fbase(&self.f, &self.combined_basis)
    }

    fn introduce_extension(&mut self, basis: Vec<F256>, claim: F256) -> SumcheckMessage256 {
        assert_eq!(basis.len(), self.f.len());
        let msg = round_msg(&self.f, &basis);
        debug_assert_eq!(
            basis
                .iter()
                .zip(&self.f)
                .fold(F256::ZERO, |acc, (&b, &f)| acc + b * f),
            claim
        );
        self.transcript.push(msg);
        self.pending = Some(PendingBasis::Extension(basis));
        msg
    }

    /// Introduce a base-field MLE claim on the currently split table. The
    /// answer is one F128 value because every table word is in the subfield.
    pub(super) fn introduce_ood_with_eval(
        &mut self,
        basis: Vec<F128>,
    ) -> (SumcheckMessage256, F128) {
        assert_eq!(basis.len(), self.f.len());
        // XOR-additive sum: chunk order cannot change the value.
        let answer = self
            .f
            .par_iter()
            .zip(basis.par_iter())
            .map(|(&f, &b)| {
                assert_eq!(f.c1, F128::ZERO, "OOD table must be base-valued");
                f.c0 * b
            })
            .reduce(|| F128::ZERO, |a, b| a + b);
        let ext_basis = basis.into_par_iter().map(F256::from).collect();
        let msg = self.introduce_extension(ext_basis, F256::from(answer));
        (msg, answer)
    }

    /// Introduce a claim stated on the extension table immediately before its
    /// coordinate split. The `u^b` weight transports it to the current table.
    pub(super) fn introduce_presplit_basis(
        &mut self,
        basis: Vec<F128>,
        claim: F256,
    ) -> SumcheckMessage256 {
        let basis_ext: Vec<F256> = basis.into_par_iter().map(F256::from).collect();
        self.introduce_extension(split_basis(&basis_ext), claim)
    }

    pub(super) fn glue(&mut self, beta: F128) {
        let pending = self.pending.take().expect("glue without introduce");
        match pending {
            PendingBasis::Extension(basis) => {
                self.combined_basis
                    .par_iter_mut()
                    .zip(basis.par_iter())
                    .for_each(|(dst, &src)| *dst += src * beta);
            }
        }
    }

    pub(super) fn f(&self) -> &[F256] {
        &self.f
    }

    pub(super) fn transcript(&self) -> &[SumcheckMessage256] {
        &self.transcript
    }
}

fn induced_basis(
    log_msg_cols: usize,
    log_inv_rate: usize,
    queries: &[usize],
    alpha: &[F128],
) -> Vec<F128> {
    let empty_rows = vec![Vec::new(); queries.len()];
    induce_sumcheck_poly_auto(
        log_msg_cols,
        log_inv_rate,
        &eval_sk_at_vks(log_msg_cols),
        &empty_rows,
        &[],
        queries,
        alpha,
    )
    .0
}

fn base_table(values: &[F256]) -> Vec<F128> {
    values
        .par_iter()
        .map(|value| {
            assert_eq!(value.c1, F128::ZERO, "committed words must be in F128");
            value.c0
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn recursive_prover_with_basis_impl<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    mut b_initial: Vec<F128>,
    mut target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    l0_num_lanes: usize,
    l0_lane_major: bool,
    l0_jit_basis: Option<BasisWindowFn<'_>>,
    l0_virtual_basis: Option<VirtualEqBasis>,
    mut first_msg: Option<SumcheckMessage>,
    // Round-1 coefficients from the pcs combine's fused lookahead pass —
    // the entry of the ladder's ALTERNATING SCHEDULE: round 1 becomes an
    // O(1) skip (the coefficients evaluated at the F256 challenge, exact
    // over the extension), the deferred fold fuses into the next pass, and
    // every subsequent fold pass can emit the next round's La256 (the
    // `alternation`/shape gates at the consumption site below decide).
    // Byte-identical either way: the skip is an exact polynomial identity
    // over the same messages, so `None` (or the A/B override) just takes
    // the plain fold path.
    round1_lookahead: Option<FoldLookahead>,
    challenger: &mut Ch,
) -> LigeritoProof {
    let log_n = packed_witness.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    assert!(r >= 1);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(config.fold_grinding_bits.iter().all(|&bits| bits == 0));
    assert!(config.recursive_ks.iter().all(|&k| k >= 2));

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    assert_eq!(l0_codeword.len(), block_len_0 * l0_num_lanes);
    assert_eq!(l0_tree.len(), 2 * block_len_0 - 1);
    let fold_block = if l0_lane_major {
        1usize << log_msg_cols_0
    } else {
        1
    };

    // Phase accounting behind LIG_PROVE_TRACE, mirroring the F128 impl's
    // [lig-prove] report so the two ladders stay comparable.
    let trace = var("LIG_PROVE_TRACE").is_ok();
    let t_total = Instant::now();
    let mut t_l0_ood = Duration::ZERO;
    let mut t_first = Duration::ZERO;
    let mut t_init_folds = Duration::ZERO;
    let mut t_commits = Duration::ZERO;
    let mut t_ood = Duration::ZERO;
    let mut t_grind = Duration::ZERO;
    let mut t_opens = Duration::ZERO;
    let mut t_induce = Duration::ZERO;
    let mut t_folds = Duration::ZERO;

    challenger.observe_label(b"flock-ligerito-basis-f256-split-v0");
    challenger.observe_f128(target);
    let strat = |level: usize| &config.stratified[level];
    let cap_depth = |level: usize| config.stratified[level].cap_depth();
    let initial_cap = cap_layer(l0_tree, block_len_0, cap_depth(0)).to_vec();
    challenger.observe_bytes(initial_cap.as_flattened());

    let claim_bits = |level: usize| config.claim_batch_grinding_bits[level] as u32;
    let consistency_bits = |level: usize| config.consistency_batch_grinding_bits[level] as u32;
    let ood_count = |level: usize| config.ood_samples[level];
    let l0_row = |q: usize| &l0_codeword[q * l0_num_lanes..(q + 1) * l0_num_lanes];

    let mut ood_values = Vec::new();
    let mut claim_batch_grinding_nonces = Vec::new();
    let mut consistency_batch_grinding_nonces = Vec::new();
    let mut grinding_nonces = Vec::new();

    let factored = l0_jit_basis.is_some() || l0_virtual_basis.is_some();
    let mut virtual_basis = l0_virtual_basis;
    let mut jit_ood_basis: Option<VirtualEqBasis> = None;
    // The combine pass's round-1 lookahead (message coefficients in the
    // round-0 challenge) makes round 1 an O(1) skip and fuses the deferred
    // fold-0 into fold 1's pass. Consumed on the materialized (d = 1,
    // compression-proof) path AND the virtual-basis lane-major path (the
    // merged-transport inner opens — the seeded EqPoint combine emits
    // blocked coefficients); the jit fallback (FLOCK_NO_VIRTUAL_B) keeps
    // the plain first fold. The double fold needs two initial rounds and a
    // quarter table that still supports the blocked message pairing
    // (len ≥ 8·d). FLOCK_NO_FOLD_LOOKAHEAD=1 / FOLD_LOOKAHEAD_OVERRIDE is
    // the A/B knob — value-identical either way (exact polynomial
    // identity), so proofs are byte-equal.
    let alternation = match FOLD_LOOKAHEAD_OVERRIDE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => var_os("FLOCK_NO_FOLD_LOOKAHEAD").is_none(),
    };
    let mut lookahead = round1_lookahead.filter(|_| {
        let kind_ok = virtual_basis.is_some() || (l0_jit_basis.is_none() && fold_block == 1);
        alternation && kind_ok && initial_k >= 2 && packed_witness.len() >= 8 * fold_block
    });
    let _t = Instant::now();
    for _ in 0..ood_count(0) {
        let z = challenger.sample_f128_vec(log_n);
        let mut ood_la = None;
        let (ood_msg, y, eq_z) = if factored {
            let (msg, y) = if lookahead.is_some() {
                // Same factored sweep, plus the (f, eq_z) pair's own BLOCKED
                // round-1 coefficients so the lookahead survives the β-glue.
                let (msg, y, la) =
                    round_msg_eval_and_lookahead_eq_point_blocked(&packed_witness, &z, fold_block);
                ood_la = Some(la);
                (msg, y)
            } else {
                round_msg_and_eval_eq_point_blocked(&packed_witness, &z, fold_block)
            };
            (msg, y, None)
        } else {
            // Same doubling recurrence as `build_eq_table`, parallel —
            // value-identical (seed ONE), and this table is 2^log_n words.
            let eq_z = build_eq_scaled_parallel(&z, F128::ONE);
            let (msg, y) = if lookahead.is_some() {
                // Same sweep, plus the (f, eq_z) pair's own round-1
                // coefficients so the lookahead survives the β-glue below.
                let (msg, y, la) = round_msg_eval_and_lookahead(&packed_witness, &eq_z);
                ood_la = Some(la);
                (msg, y)
            } else {
                round_msg_and_eval_blocked(&packed_witness, &eq_z, fold_block)
            };
            (msg, y, Some(eq_z))
        };
        challenger.observe_f128(y);
        ood_values.push(y);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(0));
        claim_batch_grinding_nonces.push(nonce);
        target += beta * y;
        if let Some(msg) = first_msg.as_mut() {
            msg.u_0 += beta * ood_msg.u_0;
            msg.u_2 += beta * ood_msg.u_2;
        }
        if let (Some(la), Some(ood)) = (lookahead.as_mut(), ood_la.as_ref()) {
            lookahead_add_scaled(la, ood, beta);
        }
        if let Some(vb) = virtual_basis.as_mut() {
            vb.add_term(z, beta);
        } else if factored {
            if let Some(vb) = jit_ood_basis.as_mut() {
                vb.add_term(z, beta);
            } else {
                jit_ood_basis = Some(VirtualEqBasis::new(z, beta));
            }
        } else {
            let eq_z = eq_z.expect("materialized OOD basis");
            b_initial
                .par_iter_mut()
                .zip(eq_z.par_iter())
                .for_each(|(dst, &src)| *dst += beta * src);
        }
    }

    if trace {
        t_l0_ood += _t.elapsed();
    }

    let _t = Instant::now();
    let first = match first_msg {
        Some(msg) => msg,
        None => {
            assert!(!factored, "factored L0 needs its precomputed first message");
            round_msg_and_eval_blocked(&packed_witness, &b_initial, fold_block).0
        }
    };
    let materialized = (!factored).then_some(b_initial);
    let mut sumcheck = SumcheckProver256::new(packed_witness, materialized, first);
    observe_message(challenger, sumcheck.transcript()[0]);
    if trace {
        t_first += _t.elapsed();
    }

    let mut virtual_basis = virtual_basis.map(VirtualEqBasis256::from_base);
    let mut jit = l0_jit_basis;
    let mut lane_challenges = Vec::with_capacity(initial_k);
    // The alternating schedule: a skip round evaluates a lookahead (the
    // caller's base-field coefficients at round 0, [`La256`] mid-chain) and
    // DEFERS its fold; the next fold pass absorbs both challenges and emits
    // the following round's coefficients. The chain's last round never emits
    // its own message (the code switch's replaces it), so a trailing
    // deferred fold DRAINS message-free into the switch.
    let mut pending_r0: Option<F256> = None;
    let mut la_mid: Option<La256> = None;
    let _t = Instant::now();
    let mut t_fold_j = Instant::now();
    for j in 0..initial_k {
        let challenge = challenger.sample_f256();
        lane_challenges.push(challenge);
        let last = j + 1 == initial_k;
        // La production pays only when the NEXT round exists to consume it
        // as a skip — the last round drains into the switch, so producing
        // there is pure waste (+4 F256 muls per output quad).
        let want_la = alternation && j + 2 < initial_k;
        let path;
        if last {
            // Materialize through any deferred challenge, then switch; the
            // switch's fbase message is this round's transcript entry.
            match pending_r0.take() {
                Some(r0) => {
                    path = "drain2+switch";
                    if let Some(mut vb) = virtual_basis.take() {
                        let p = fold_block.trailing_zeros() as usize;
                        vb.fold_coord(p, r0);
                        vb.fold_coord(p, challenge);
                        let _ = sumcheck.first_fold2_virtual(r0, challenge, fold_block, &vb, false);
                        // first_fold2_virtual pushes its message; the switch
                        // replaces it — same shape as the plain path.
                        let msg = sumcheck.code_switch_and_replace_message();
                        observe_message(challenger, msg);
                    } else if sumcheck.initial_pending() {
                        let _ = sumcheck.first_fold2_materialized(r0, challenge, false);
                        let msg = sumcheck.code_switch_and_replace_message();
                        observe_message(challenger, msg);
                    } else {
                        sumcheck.fold2_drain(r0, challenge, fold_block);
                        let msg = sumcheck.code_switch_and_push_message();
                        observe_message(challenger, msg);
                    }
                }
                None => {
                    path = "fold+switch";
                    // Plain (non-alternating) tail: fold with a message the
                    // switch replaces, exactly the historical shape.
                    let _ = if j == 0 {
                        if let Some(mut vb) = virtual_basis.take() {
                            vb.fold_coord(fold_block.trailing_zeros() as usize, challenge);
                            sumcheck.first_fold_virtual(challenge, fold_block, &vb)
                        } else if let Some(fill) = jit.take() {
                            match jit_ood_basis.as_ref() {
                                Some(ood) => {
                                    let combined = |out: &mut [F128], offset: usize| {
                                        fill(out, offset);
                                        ood.add_to(out, offset);
                                    };
                                    sumcheck.first_fold_jit(challenge, fold_block, &combined)
                                }
                                None => sumcheck.first_fold_jit(challenge, fold_block, fill),
                            }
                        } else {
                            sumcheck.first_fold_materialized(challenge, fold_block)
                        }
                    } else {
                        sumcheck.fold_materialized(challenge, fold_block)
                    };
                    let msg = sumcheck.code_switch_and_replace_message();
                    observe_message(challenger, msg);
                }
            }
        } else {
            let msg = if let (0, Some(la)) = (j, lookahead.as_ref()) {
                path = "lookahead-skip";
                // O(1) skip: round 1's message by polynomial identity; the
                // fold is deferred into the next double-fold pass.
                let msg = lookahead_eval256(la, challenge);
                sumcheck.push_skip_message(msg);
                pending_r0 = Some(challenge);
                msg
            } else if let Some(la) = la_mid.take() {
                path = "skip";
                let msg = la.eval(challenge);
                sumcheck.push_skip_message(msg);
                pending_r0 = Some(challenge);
                msg
            } else if let Some(r0) = pending_r0.take() {
                if let Some(mut vb) = virtual_basis.take() {
                    path = "fold2-virtual";
                    // The deferred fold and this one both bind the variable
                    // at bit log2(d): fold_coord removes it, so the second
                    // bind at the SAME position takes the next block
                    // variable — the ladder's successive block-d folds.
                    let p = fold_block.trailing_zeros() as usize;
                    vb.fold_coord(p, r0);
                    vb.fold_coord(p, challenge);
                    let (msg, la) =
                        sumcheck.first_fold2_virtual(r0, challenge, fold_block, &vb, want_la);
                    la_mid = la;
                    msg
                } else {
                    path = "fold2";
                    let (msg, la) = if sumcheck.initial_pending() {
                        sumcheck.first_fold2_materialized(r0, challenge, want_la)
                    } else {
                        sumcheck.mid_fold2(r0, challenge, fold_block, want_la)
                    };
                    la_mid = la;
                    msg
                }
            } else if j == 0 {
                // No caller lookahead: start the alternation at round 0 on
                // the materialized path; virtual/jit first folds stay plain.
                if let Some(mut vb) = virtual_basis.take() {
                    path = "virtual-once";
                    vb.fold_coord(fold_block.trailing_zeros() as usize, challenge);
                    sumcheck.first_fold_virtual(challenge, fold_block, &vb)
                } else if let Some(fill) = jit.take() {
                    path = "jit";
                    match jit_ood_basis.as_ref() {
                        Some(ood) => {
                            let combined = |out: &mut [F128], offset: usize| {
                                fill(out, offset);
                                ood.add_to(out, offset);
                            };
                            sumcheck.first_fold_jit(challenge, fold_block, &combined)
                        }
                        None => sumcheck.first_fold_jit(challenge, fold_block, fill),
                    }
                } else if want_la && fold_block == 1 {
                    path = "fold+la";
                    let (msg, la) = sumcheck.first_fold_materialized_la(challenge);
                    la_mid = la;
                    msg
                } else {
                    path = "materialized";
                    sumcheck.first_fold_materialized(challenge, fold_block)
                }
            } else {
                path = "materialized";
                sumcheck.fold_materialized(challenge, fold_block)
            };
            observe_message(challenger, msg);
        }
        if trace {
            eprintln!(
                "    init fold {j} ({path}, d {}): {:.2} ms",
                fold_block,
                t_fold_j.elapsed().as_secs_f64() * 1e3
            );
            t_fold_j = Instant::now();
        }
    }
    if trace {
        t_init_folds += _t.elapsed();
    }

    let n1 = log_n - initial_k;
    let mut current_split_dim = n1 + 1;
    let commit_split = |values: &[F256], level: usize, split_dim: usize| {
        let log_lanes = config.recursive_ks[level - 1];
        assert!(split_dim >= log_lanes);
        let log_cols = split_dim - log_lanes;
        let log_rate = config.log_inv_rates[level];
        let ntt = AdditiveNttF128::standard(log_cols + log_rate);
        ligero_commit(
            &base_table(values),
            log_cols,
            log_lanes,
            log_rate,
            &ntt,
            config.merkle_hash,
        )
    };

    let _t = Instant::now();
    let mut previous = commit_split(sumcheck.f(), 1, current_split_dim);
    let mut recursive_caps = vec![previous.cap(cap_depth(1)).to_vec()];
    challenger.observe_bytes(recursive_caps[0].as_flattened());
    if trace {
        t_commits += _t.elapsed();
    }

    let _t = Instant::now();
    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(current_split_dim);
        let (msg, y) = sumcheck.introduce_ood_with_eval(build_eq_scaled_parallel(&z, F128::ONE));
        challenger.observe_f128(y);
        ood_values.push(y);
        observe_message(challenger, msg);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(1));
        claim_batch_grinding_nonces.push(nonce);
        sumcheck.glue(beta);
    }
    if trace {
        t_ood += _t.elapsed();
    }

    let _t = Instant::now();
    let (nonce, queries_0) = grind_and_sample_queries(
        challenger,
        config.grinding_bits[0] as u32,
        block_len_0,
        config.queries[0],
        strat(0),
    );
    grinding_nonces.push(nonce);
    let (nonce, alpha_0) =
        challenger.grind_pow_and_sample_f128_vec(consistency_bits(0), ceil_log2(config.queries[0]));
    consistency_batch_grinding_nonces.push(nonce);
    if trace {
        t_grind += _t.elapsed();
    }
    let _t = Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| l0_row(q).to_vec()).collect();
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_paths_for(l0_tree, block_len_0, &queries_0, strat(0)),
    };
    if trace {
        t_opens += _t.elapsed();
    }
    let _t = Instant::now();
    let basis_0 = induced_basis(n1, log_inv_rate_0, &queries_0, &alpha_0);
    let enforced_0 = induce_enforced_sum(&opened_rows_0, &lane_challenges, &alpha_0);
    let msg = sumcheck.introduce_presplit_basis(basis_0, enforced_0);
    observe_message(challenger, msg);
    let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(1));
    claim_batch_grinding_nonces.push(nonce);
    sumcheck.glue(beta);
    if trace {
        t_induce += _t.elapsed();
    }

    let mut recursive_proofs = Vec::new();
    for i in 0..r {
        let k = config.recursive_ks[i];
        assert!(current_split_dim >= k);
        let mut level_challenges = Vec::with_capacity(k);
        let _t = Instant::now();
        // The level's alternating schedule. j = 0 is the entry: the fbase
        // fold pass (which runs AFTER every glue, so its coefficients are
        // always fresh) also emits the next round's La256 → j = 1 skips,
        // j = 2 double-folds, … A pre-switch round emits no message of its
        // own (the switch's replaces it), so a trailing deferred challenge
        // drains message-free into the switch.
        let mut level_pending: Option<F256> = None;
        let mut level_la: Option<La256> = None;
        for j in 0..k {
            let challenge = challenger.sample_f256();
            level_challenges.push(challenge);
            let switching = j + 1 == k && i + 1 != r;
            // As in the init loop: produce La only when the NEXT round
            // consumes it — the pre-switch round of a non-final level
            // drains, so its predecessor's La would be dead weight.
            let want_la = alternation && (j + 2 < k || i + 1 == r);
            if switching {
                // No message of this round's own survives (the switch's
                // replaces it) — materialize through the deferred and
                // current challenges message-free, then switch-push. The
                // level state is always F256 here, so the generic extension
                // fold covers the base-valued j = 0 edge too.
                match level_pending.take() {
                    Some(r0) => sumcheck.fold2_drain(r0, challenge, 1),
                    None => sumcheck.drain(challenge, 1),
                }
                let msg = sumcheck.code_switch_and_push_message();
                observe_message(challenger, msg);
            } else {
                let msg = if let Some(la) = level_la.take() {
                    let msg = la.eval(challenge);
                    sumcheck.push_skip_message(msg);
                    level_pending = Some(challenge);
                    msg
                } else if let Some(r0) = level_pending.take() {
                    let (msg, la) = sumcheck.mid_fold2(r0, challenge, 1, want_la);
                    level_la = la;
                    msg
                } else if j == 0 && want_la {
                    let (msg, la) = sumcheck.fold_after_switch_la(challenge);
                    level_la = la;
                    msg
                } else if j == 0 {
                    sumcheck.fold_after_switch(challenge)
                } else {
                    sumcheck.fold(challenge)
                };
                observe_message(challenger, msg);
            }
        }
        // FINAL level only: a trailing skip leaves one deferred fold —
        // materialize before the residual ships.
        if let Some(r0) = level_pending.take() {
            sumcheck.drain(r0, 1);
        }
        if trace {
            t_folds += _t.elapsed();
        }
        let extension_dim = current_split_dim - k;
        let level = i + 1;

        if i + 1 == r {
            let yr = split_coordinates(sumcheck.f());
            for &value in &yr {
                challenger.observe_f128(value);
            }
            let _t = Instant::now();
            let (nonce, queries) = grind_and_sample_queries(
                challenger,
                config.grinding_bits[level] as u32,
                previous.block_len,
                config.queries[level],
                strat(level),
            );
            grinding_nonces.push(nonce);
            let (nonce, _) = challenger.grind_pow_and_sample_f128_vec(
                consistency_bits(level),
                ceil_log2(config.queries[level]),
            );
            consistency_batch_grinding_nonces.push(nonce);
            let (nonce, _) = challenger.grind_pow_and_sample_f128(claim_bits(level));
            claim_batch_grinding_nonces.push(nonce);
            if trace {
                t_grind += _t.elapsed();
            }
            let _t = Instant::now();
            let opened_rows = queries.iter().map(|&q| previous.row(q).to_vec()).collect();
            if trace {
                t_opens += _t.elapsed();
                eprintln!(
                    "[lig-prove-f256] total = {:.2} ms",
                    t_total.elapsed().as_secs_f64() * 1e3
                );
                eprintln!(
                    "  L0 OOD (eq build + full-witness eval + fold-in): {:.2} ms",
                    t_l0_ood.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  first message + prover build:                    {:.2} ms",
                    t_first.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  initial_k folds + code switch:                   {:.2} ms",
                    t_init_folds.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  recursive commits (NTT + merkle):                {:.2} ms",
                    t_commits.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  level OODs (eq build + introduce + glue):        {:.2} ms",
                    t_ood.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  grinding (query PoW + claim/consistency PoW):    {:.2} ms",
                    t_grind.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  opens (rows + merkle paths):                     {:.2} ms",
                    t_opens.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  induce (basis + enforced sum + introduce):       {:.2} ms",
                    t_induce.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  recursive folds + code switches:                 {:.2} ms",
                    t_folds.as_secs_f64() * 1e3
                );
            }
            return LigeritoProof {
                initial_cap,
                initial_proof,
                recursive_caps,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows,
                    merkle_proof: merkle_paths_for(
                        &previous.tree,
                        previous.block_len,
                        &queries,
                        strat(level),
                    ),
                },
                sumcheck_transcript: Vec::new(),
                sumcheck_transcript_f256: sumcheck.transcript().to_vec(),
                grinding_nonces,
                ood_values,
                fold_grinding_nonces: Vec::new(),
                claim_batch_grinding_nonces,
                consistency_batch_grinding_nonces,
            };
        }

        current_split_dim = extension_dim + 1;
        let next_level = i + 2;
        let _t = Instant::now();
        let next = commit_split(sumcheck.f(), next_level, current_split_dim);
        let cap = next.cap(cap_depth(next_level)).to_vec();
        challenger.observe_bytes(cap.as_flattened());
        recursive_caps.push(cap);
        if trace {
            t_commits += _t.elapsed();
        }

        let _t = Instant::now();
        for _ in 0..ood_count(next_level) {
            let z = challenger.sample_f128_vec(current_split_dim);
            let (msg, y) =
                sumcheck.introduce_ood_with_eval(build_eq_scaled_parallel(&z, F128::ONE));
            challenger.observe_f128(y);
            ood_values.push(y);
            observe_message(challenger, msg);
            let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(next_level));
            claim_batch_grinding_nonces.push(nonce);
            sumcheck.glue(beta);
        }
        if trace {
            t_ood += _t.elapsed();
        }

        let _t = Instant::now();
        let (nonce, queries) = grind_and_sample_queries(
            challenger,
            config.grinding_bits[level] as u32,
            previous.block_len,
            config.queries[level],
            strat(level),
        );
        grinding_nonces.push(nonce);
        let (nonce, alpha) = challenger.grind_pow_and_sample_f128_vec(
            consistency_bits(level),
            ceil_log2(config.queries[level]),
        );
        consistency_batch_grinding_nonces.push(nonce);
        if trace {
            t_grind += _t.elapsed();
        }
        let _t = Instant::now();
        let opened_rows: Vec<Vec<F128>> =
            queries.iter().map(|&q| previous.row(q).to_vec()).collect();
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows.clone(),
            merkle_proof: merkle_paths_for(
                &previous.tree,
                previous.block_len,
                &queries,
                strat(level),
            ),
        });
        if trace {
            t_opens += _t.elapsed();
        }
        let _t = Instant::now();
        let basis = induced_basis(extension_dim, config.log_inv_rates[level], &queries, &alpha);
        let enforced = induce_enforced_sum(&opened_rows, &level_challenges, &alpha);
        let msg = sumcheck.introduce_presplit_basis(basis, enforced);
        observe_message(challenger, msg);
        let (nonce, beta) = challenger.grind_pow_and_sample_f128(claim_bits(next_level));
        claim_batch_grinding_nonces.push(nonce);
        sumcheck.glue(beta);
        if trace {
            t_induce += _t.elapsed();
        }
        previous = next;
    }
    unreachable!()
}

fn coordinate_fold_factor(challenge: F256) -> F256 {
    F256::ONE + challenge * (F256::ONE + F256::U)
}

fn induced_basis_at_residual(
    log_msg_cols: usize,
    queries: &[usize],
    alpha: &[F128],
    fixed: &[F256],
    residual_log: usize,
) -> Vec<F256> {
    assert_eq!(fixed.len() + residual_log, log_msg_cols);
    let sks_vks = eval_sk_at_vks(log_msg_cols);
    let inv: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();
    let weights = build_eq_table(alpha);
    let mut per_query = Vec::with_capacity(queries.len());
    for (&query, &weight) in queries.iter().zip(&weights) {
        let mut w = vec![F128::ZERO; log_msg_cols];
        if log_msg_cols > 0 {
            w[0] = F128::new(query as u64, 0);
            for k in 1..log_msg_cols {
                w[k] = next_s(w[k - 1], sks_vks[k - 1]);
            }
            for k in 0..log_msg_cols {
                w[k] *= inv[k];
            }
        }
        let prefix = fixed.iter().zip(&w).fold(F256::ONE, |acc, (&p, &wk)| {
            acc * (F256::ONE + p * (F128::ONE + wk))
        });
        per_query.push((weight, prefix, w[fixed.len()..].to_vec()));
    }
    (0..1usize << residual_log)
        .map(|y| {
            per_query
                .iter()
                .map(|&(weight, prefix, ref suffix)| {
                    let tail = suffix.iter().enumerate().fold(F256::ONE, |acc, (j, &wk)| {
                        if (y >> j) & 1 == 0 {
                            acc
                        } else {
                            acc * F256::from(wk)
                        }
                    });
                    prefix * tail * weight
                })
                .fold(F256::ZERO, |a, b| a + b)
        })
        .collect()
}

#[derive(Clone)]
struct OodResidualContext {
    point: Vec<F128>,
    beta: F128,
    /// `None` is an L0 base-table claim. `Some(s)` is a claim on the split
    /// commitment consumed by recursive level `s`.
    split_level: Option<usize>,
}

#[derive(Clone)]
struct ConsistencyResidualContext {
    log_cols: usize,
    queries: Vec<usize>,
    alpha: Vec<F128>,
    beta: F128,
    /// First recursive split level whose coordinate weight applies.
    start_level: usize,
}

fn residual_original_challenges(
    initial: &[F256],
    levels: &[Vec<F256>],
    start_level: usize,
) -> Vec<F256> {
    let initial_len = if start_level == 0 { initial.len() } else { 0 };
    let recursive_len: usize = levels[start_level..]
        .iter()
        .map(|level| level.len().saturating_sub(1))
        .sum();
    let mut out = Vec::with_capacity(initial_len + recursive_len);
    if start_level == 0 {
        out.extend_from_slice(initial);
    }
    for level in &levels[start_level..] {
        out.extend_from_slice(&level[1..]);
    }
    out
}

fn coordinate_factor_product(levels: &[Vec<F256>], start_level: usize) -> F256 {
    levels[start_level..].iter().fold(F256::ONE, |acc, level| {
        acc * coordinate_fold_factor(level[0])
    })
}

fn eq_residual(point: &[F128], fixed: &[F256], residual_log: usize, scale: F256) -> Vec<F256> {
    assert_eq!(fixed.len() + residual_log, point.len());
    let prefix = point[..fixed.len()]
        .iter()
        .zip(fixed)
        .fold(scale, |acc, (&z, &r)| acc * (F256::ONE + F256::from(z) + r));
    (0..1usize << residual_log)
        .map(|y| {
            point[fixed.len()..]
                .iter()
                .enumerate()
                .fold(prefix, |acc, (j, &z)| {
                    acc * if (y >> j) & 1 == 1 { z } else { F128::ONE + z }
                })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_verifier_with_basis_succinct<Ch, F>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    log_n: usize,
    target: F128,
    expected_initial_cap: &[Hash],
    l0_num_lanes: usize,
    eval_b_residual: F,
    challenger: &mut Ch,
) -> bool
where
    Ch: Challenger,
    F: Fn(&[F256], usize) -> Vec<F256>,
{
    let initial_k = config.initial_k;
    let rounds = config.recursive_steps;
    if rounds < 1
        || proof.initial_cap != expected_initial_cap
        || !proof.sumcheck_transcript.is_empty()
        || proof.sumcheck_transcript_f256.is_empty()
        || !proof.fold_grinding_nonces.is_empty()
        || config.fold_grinding_bits.iter().any(|&bits| bits != 0)
        || config.recursive_ks.iter().any(|&k| k < 2)
    {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-f256-split-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(proof.initial_cap.as_flattened());
    let log_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_cols_0 + config.log_inv_rates[0]);
    let strat = |level: usize| &config.stratified[level];
    let claim_bits = |level: usize| config.claim_batch_grinding_bits[level] as u32;
    let consistency_bits = |level: usize| config.consistency_batch_grinding_bits[level] as u32;
    let ood_count = |level: usize| config.ood_samples[level];

    let mut tx = 0usize;
    let mut ood_index = 0usize;
    let mut claim_nonce = 0usize;
    let mut consistency_nonce = 0usize;
    let mut query_nonce = 0usize;
    let mut claim = F256::from(target);
    let mut ood_contexts = Vec::new();
    let lane_major = l0_num_lanes < 1usize << initial_k;

    for _ in 0..ood_count(0) {
        let mut point = challenger.sample_f128_vec(log_n);
        if lane_major {
            // The reused integer-lane L0 commitment folds the high variables
            // first.  Its residual coordinate order is therefore
            // `[high variables | low variables]`, matching the rotation used
            // for the opening basis in `pcs::verify_opening_batch_*`.
            point.rotate_left(log_n - initial_k);
        }
        let Some(&y) = proof.ood_values.get(ood_index) else {
            return false;
        };
        ood_index += 1;
        challenger.observe_f128(y);
        let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
            return false;
        };
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(0)) else {
            return false;
        };
        claim_nonce += 1;
        claim += F256::from(beta * y);
        ood_contexts.push(OodResidualContext {
            point,
            beta,
            split_level: None,
        });
    }

    let Some(&first) = proof.sumcheck_transcript_f256.get(tx) else {
        return false;
    };
    tx += 1;
    observe_message(challenger, first);
    let mut quad = RoundQuad256::from_msg(first, claim);
    let mut initial_challenges = Vec::with_capacity(initial_k);
    for _ in 0..initial_k {
        let challenge = challenger.sample_f256();
        claim = quad.eval(challenge);
        initial_challenges.push(challenge);
        let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
            return false;
        };
        tx += 1;
        observe_message(challenger, msg);
        quad = RoundQuad256::from_msg(msg, claim);
    }

    let mut current_split_dim = log_n - initial_k + 1;
    let Some(cap_1) = proof.recursive_caps.first() else {
        return false;
    };
    challenger.observe_bytes(cap_1.as_flattened());
    for _ in 0..ood_count(1) {
        let point = challenger.sample_f128_vec(current_split_dim);
        let Some(&y) = proof.ood_values.get(ood_index) else {
            return false;
        };
        ood_index += 1;
        challenger.observe_f128(y);
        let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
            return false;
        };
        tx += 1;
        observe_message(challenger, msg);
        let intro = RoundQuad256::from_msg(msg, F256::from(y));
        let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
            return false;
        };
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(1)) else {
            return false;
        };
        claim_nonce += 1;
        quad = quad.fold(intro, beta);
        claim += F256::from(beta * y);
        ood_contexts.push(OodResidualContext {
            point,
            beta,
            split_level: Some(0),
        });
    }

    let Some(&nonce) = proof.grinding_nonces.get(query_nonce) else {
        return false;
    };
    let Some(queries_0) = verify_and_sample_queries(
        challenger,
        nonce,
        config.grinding_bits[0] as u32,
        block_len_0,
        config.queries[0],
        strat(0),
    ) else {
        return false;
    };
    query_nonce += 1;
    let Some(&nonce) = proof
        .consistency_batch_grinding_nonces
        .get(consistency_nonce)
    else {
        return false;
    };
    let Some(alpha_0) = challenger.verify_pow_and_sample_f128_vec(
        nonce,
        consistency_bits(0),
        ceil_log2(config.queries[0]),
    ) else {
        return false;
    };
    consistency_nonce += 1;
    if !verify_level_opens(
        &proof.initial_cap,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        l0_num_lanes,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
        strat(0),
    ) {
        return false;
    }
    let enforced_0 = induce_enforced_sum(
        &proof.initial_proof.opened_rows,
        &initial_challenges,
        &alpha_0,
    );
    let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
        return false;
    };
    tx += 1;
    observe_message(challenger, msg);
    let intro = RoundQuad256::from_msg(msg, enforced_0);
    let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
        return false;
    };
    let Some(beta_0) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(1)) else {
        return false;
    };
    claim_nonce += 1;
    quad = quad.fold(intro, beta_0);
    claim += enforced_0 * beta_0;
    let mut consistency_contexts = vec![ConsistencyResidualContext {
        log_cols: log_n - initial_k,
        queries: queries_0,
        alpha: alpha_0,
        beta: beta_0,
        start_level: 0,
    }];

    let mut level_challenges: Vec<Vec<F256>> = Vec::with_capacity(rounds);
    let mut previous_cap = cap_1.as_slice();
    let mut previous_log_lanes = config.recursive_ks[0];
    if current_split_dim < previous_log_lanes {
        return false;
    }
    let mut previous_log_cols = current_split_dim - previous_log_lanes;
    let mut previous_rate = config.log_inv_rates[1];
    let mut cap_index = 1usize;
    let mut proof_index = 0usize;

    for i in 0..rounds {
        let k = config.recursive_ks[i];
        if current_split_dim < k {
            return false;
        }
        let mut challenges = Vec::with_capacity(k);
        for _ in 0..k {
            let challenge = challenger.sample_f256();
            claim = quad.eval(challenge);
            challenges.push(challenge);
            let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
                return false;
            };
            tx += 1;
            observe_message(challenger, msg);
            quad = RoundQuad256::from_msg(msg, claim);
        }
        level_challenges.push(challenges);
        let extension_dim = current_split_dim - k;
        let level = i + 1;

        if i + 1 == rounds {
            if tx != proof.sumcheck_transcript_f256.len()
                || proof.final_proof.yr.len() != 2usize << extension_dim
            {
                return false;
            }
            for &value in &proof.final_proof.yr {
                challenger.observe_f128(value);
            }
            let Some(&nonce) = proof.grinding_nonces.get(query_nonce) else {
                return false;
            };
            let block_len = 1usize << (previous_log_cols + previous_rate);
            let Some(queries) = verify_and_sample_queries(
                challenger,
                nonce,
                config.grinding_bits[level] as u32,
                block_len,
                config.queries[level],
                strat(level),
            ) else {
                return false;
            };
            query_nonce += 1;
            let Some(&nonce) = proof
                .consistency_batch_grinding_nonces
                .get(consistency_nonce)
            else {
                return false;
            };
            let Some(alpha) = challenger.verify_pow_and_sample_f128_vec(
                nonce,
                consistency_bits(level),
                ceil_log2(config.queries[level]),
            ) else {
                return false;
            };
            consistency_nonce += 1;
            if !verify_level_opens(
                previous_cap,
                block_len,
                &queries,
                &proof.final_proof.opened_rows,
                1usize << previous_log_lanes,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
                strat(level),
            ) {
                return false;
            }
            let enforced =
                induce_enforced_sum(&proof.final_proof.opened_rows, &level_challenges[i], &alpha);
            let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
                return false;
            };
            let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(level)) else {
                return false;
            };
            claim_nonce += 1;
            claim += enforced * beta;
            consistency_contexts.push(ConsistencyResidualContext {
                log_cols: extension_dim,
                queries,
                alpha,
                beta,
                start_level: rounds,
            });

            let original_challenges =
                residual_original_challenges(&initial_challenges, &level_challenges, 0);
            if original_challenges.len() + extension_dim != log_n {
                return false;
            }
            let mut residual = eval_b_residual(&original_challenges, extension_dim);
            if residual.len() != 1usize << extension_dim {
                return false;
            }
            let initial_coordinate_scale = coordinate_factor_product(&level_challenges, 0);
            for value in &mut residual {
                *value *= initial_coordinate_scale;
            }

            for context in &ood_contexts {
                let (fixed, coordinate_scale) = match context.split_level {
                    None => (
                        original_challenges.clone(),
                        coordinate_factor_product(&level_challenges, 0),
                    ),
                    Some(split_level) => {
                        let mut fixed = level_challenges[split_level].clone();
                        for later in &level_challenges[split_level + 1..] {
                            fixed.extend_from_slice(&later[1..]);
                        }
                        (
                            fixed,
                            coordinate_factor_product(&level_challenges, split_level + 1),
                        )
                    }
                };
                let values = eq_residual(
                    &context.point,
                    &fixed,
                    extension_dim,
                    coordinate_scale * context.beta,
                );
                for (dst, value) in residual.iter_mut().zip(values) {
                    *dst += value;
                }
            }

            for context in &consistency_contexts {
                let fixed =
                    residual_original_challenges(&[], &level_challenges, context.start_level);
                let values = induced_basis_at_residual(
                    context.log_cols,
                    &context.queries,
                    &context.alpha,
                    &fixed,
                    extension_dim,
                );
                let scale = coordinate_factor_product(&level_challenges, context.start_level)
                    * context.beta;
                for (dst, value) in residual.iter_mut().zip(values) {
                    *dst += value * scale;
                }
            }

            let final_basis = split_basis(&residual);
            let residual_claim = split_inner_product(&proof.final_proof.yr, &final_basis);
            return residual_claim == claim
                && ood_index == proof.ood_values.len()
                && query_nonce == proof.grinding_nonces.len()
                && claim_nonce == proof.claim_batch_grinding_nonces.len()
                && consistency_nonce == proof.consistency_batch_grinding_nonces.len()
                && cap_index == proof.recursive_caps.len()
                && proof_index == proof.recursive_proofs.len();
        }

        current_split_dim = extension_dim + 1;
        let Some(cap) = proof.recursive_caps.get(cap_index) else {
            return false;
        };
        cap_index += 1;
        challenger.observe_bytes(cap.as_flattened());
        for _ in 0..ood_count(level + 1) {
            let point = challenger.sample_f128_vec(current_split_dim);
            let Some(&y) = proof.ood_values.get(ood_index) else {
                return false;
            };
            ood_index += 1;
            challenger.observe_f128(y);
            let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
                return false;
            };
            tx += 1;
            observe_message(challenger, msg);
            let intro = RoundQuad256::from_msg(msg, F256::from(y));
            let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
                return false;
            };
            let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(level + 1))
            else {
                return false;
            };
            claim_nonce += 1;
            quad = quad.fold(intro, beta);
            claim += F256::from(beta * y);
            ood_contexts.push(OodResidualContext {
                point,
                beta,
                split_level: Some(i + 1),
            });
        }

        let Some(&nonce) = proof.grinding_nonces.get(query_nonce) else {
            return false;
        };
        let block_len = 1usize << (previous_log_cols + previous_rate);
        let Some(queries) = verify_and_sample_queries(
            challenger,
            nonce,
            config.grinding_bits[level] as u32,
            block_len,
            config.queries[level],
            strat(level),
        ) else {
            return false;
        };
        query_nonce += 1;
        let Some(&nonce) = proof
            .consistency_batch_grinding_nonces
            .get(consistency_nonce)
        else {
            return false;
        };
        let Some(alpha) = challenger.verify_pow_and_sample_f128_vec(
            nonce,
            consistency_bits(level),
            ceil_log2(config.queries[level]),
        ) else {
            return false;
        };
        consistency_nonce += 1;
        let Some(opening) = proof.recursive_proofs.get(proof_index) else {
            return false;
        };
        proof_index += 1;
        if !verify_level_opens(
            previous_cap,
            block_len,
            &queries,
            &opening.opened_rows,
            1usize << previous_log_lanes,
            &opening.merkle_proof,
            config.merkle_hash,
            strat(level),
        ) {
            return false;
        }
        let enforced = induce_enforced_sum(&opening.opened_rows, &level_challenges[i], &alpha);
        let Some(&msg) = proof.sumcheck_transcript_f256.get(tx) else {
            return false;
        };
        tx += 1;
        observe_message(challenger, msg);
        let intro = RoundQuad256::from_msg(msg, enforced);
        let Some(&nonce) = proof.claim_batch_grinding_nonces.get(claim_nonce) else {
            return false;
        };
        let Some(beta) = challenger.verify_pow_and_sample_f128(nonce, claim_bits(level + 1)) else {
            return false;
        };
        claim_nonce += 1;
        quad = quad.fold(intro, beta);
        claim += enforced * beta;
        consistency_contexts.push(ConsistencyResidualContext {
            log_cols: extension_dim,
            queries,
            alpha,
            beta,
            start_level: i + 1,
        });

        previous_cap = cap;
        previous_log_lanes = config.recursive_ks[i + 1];
        if current_split_dim < previous_log_lanes {
            return false;
        }
        previous_log_cols = current_split_dim - previous_log_lanes;
        previous_rate = config.log_inv_rates[i + 2];
    }
    false
}

/// Evaluate queried base-field rows at extension-valued lane challenges and
/// batch the queried columns by the base-field `alpha` point.
pub(super) fn induce_enforced_sum(
    opened_rows: &[Vec<F128>],
    lane_challenges: &[F256],
    alpha: &[F128],
) -> F256 {
    let lane_weights = build_eq_table256(lane_challenges);
    let row_weights = build_eq_table(alpha);
    opened_rows
        .par_iter()
        .zip(row_weights.par_iter())
        .map(|(row, &row_weight)| {
            // The reused L0 commitment may contain only the live lanes of a
            // non-power-of-two packed batch.  The omitted logical lanes are
            // zero, so pairing the committed prefix with the corresponding
            // equality weights is exactly the padded dot product.  Recursive
            // commitments are power-of-two and therefore use every weight.
            assert!(row.len() <= lane_weights.len());
            row.iter()
                .zip(&lane_weights)
                .fold(F256::ZERO, |acc, (&word, &weight)| acc + weight * word)
                * row_weight
        })
        .reduce(|| F256::ZERO, |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use crate::{
        challenger::{Challenger, RandomChallenger},
        pcs::ligerito::extension::{
            F128, F256, VirtualEqBasis, VirtualEqBasis256, build_eq_table, build_eq_table256,
            fold_base, fold_extension, fused_first_fold_virtual, fused_fold_applies,
            fused_fold_msg_base, fused_fold_msg_ext, fused_fold_msg_fbase, induce_enforced_sum,
            next_round_msg, round_msg, round_msg_fbase, split_basis, split_coordinates,
            split_inner_product,
        },
    };

    fn random_f256(challenger: &mut RandomChallenger) -> F256 {
        F256::new(challenger.sample_f128(), challenger.sample_f128())
    }

    /// The fused fold+message sweep is value-identical to the three-pass
    /// sequence it replaces, across d=1 (the leaf), blocked lane-major d,
    /// and the boundary shapes where the fallback engages.
    #[test]
    fn fused_fold_msg_matches_unfused_three_pass() {
        let mut rng = RandomChallenger::new(0xF05E_D256);
        for &(log_n, d) in &[
            (2usize, 1usize),
            (6, 1),
            (12, 1),
            (6, 4),
            (10, 16),
            (13, 1024),
            (13, 2048),
            // Lane-major shapes: block = 2d >= 2^16 takes the nested
            // (block x k-chunk) parallel branch.
            (18, 1 << 15),
            (19, 1 << 16),
        ] {
            let n = 1usize << log_n;
            assert!(
                fused_fold_applies(n, d),
                "case must exercise the fused path"
            );
            let r = random_f256(&mut rng);
            let fb: Vec<F128> = (0..n).map(|_| rng.sample_f128()).collect();
            let bb: Vec<F128> = (0..n).map(|_| rng.sample_f128()).collect();
            let (nf, nb, msg) = fused_fold_msg_base(&fb, &bb, r, d);
            let ef = fold_base(&fb, r, d);
            let eb = fold_base(&bb, r, d);
            assert_eq!(nf, ef, "base fold f (log_n {log_n}, d {d})");
            assert_eq!(nb, eb, "base fold b (log_n {log_n}, d {d})");
            assert_eq!(
                msg,
                next_round_msg(&ef, &eb, d),
                "base msg (log_n {log_n}, d {d})"
            );
            let fx: Vec<F256> = (0..n).map(|_| random_f256(&mut rng)).collect();
            let bx: Vec<F256> = (0..n).map(|_| random_f256(&mut rng)).collect();
            let (nf, nb, msg) = fused_fold_msg_ext(&fx, &bx, r, d);
            let ef = fold_extension(&fx, r, d);
            let eb = fold_extension(&bx, r, d);
            assert_eq!(nf, ef, "ext fold f (log_n {log_n}, d {d})");
            assert_eq!(nb, eb, "ext fold b (log_n {log_n}, d {d})");
            assert_eq!(
                msg,
                next_round_msg(&ef, &eb, d),
                "ext msg (log_n {log_n}, d {d})"
            );
        }
        // Boundary shapes take the fallback, not the fused sweep.
        assert!(!fused_fold_applies(2, 2));
        assert!(!fused_fold_applies(1 << 10, 512));
        assert!(!fused_fold_applies(1 << 10, 1024));
    }

    /// The base-valued-f variants (the post-code-switch fold and round
    /// message) match their generic counterparts on split tables.
    #[test]
    fn split_base_variants_match_generic() {
        let mut rng = RandomChallenger::new(0x5B_BA5E_F01D);
        for &(log_n, d) in &[(2usize, 1usize), (12, 1), (13, 16)] {
            let n = 1usize << log_n;
            let r = random_f256(&mut rng);
            let fb: Vec<F256> = (0..n).map(|_| F256::from(rng.sample_f128())).collect();
            let bx: Vec<F256> = (0..n).map(|_| random_f256(&mut rng)).collect();
            let (nf, nb, msg) = fused_fold_msg_fbase(&fb, &bx, r, d);
            let ef = fold_extension(&fb, r, d);
            let eb = fold_extension(&bx, r, d);
            assert_eq!(nf, ef, "fbase fold f (log_n {log_n}, d {d})");
            assert_eq!(nb, eb, "fbase fold b (log_n {log_n}, d {d})");
            assert_eq!(
                msg,
                next_round_msg(&ef, &eb, d),
                "fbase msg (log_n {log_n}, d {d})"
            );
            assert_eq!(
                round_msg_fbase(&fb, &bx),
                round_msg(&fb, &bx),
                "fbase round msg (log_n {log_n})"
            );
        }
    }

    /// The fused first virtual fold matches (fold_base, materialize,
    /// next_round_msg) — flat, blocked, and nested lane-major shapes.
    #[test]
    fn fused_first_fold_virtual_matches_unfused() {
        let mut rng = RandomChallenger::new(0xF1_057F_01D);
        for &(log_n, d) in &[
            (6usize, 1usize),
            (12, 1),
            (13, 16),
            (18, 1 << 15),
            (19, 1 << 16),
        ] {
            let n = 1usize << log_n;
            assert!(
                fused_fold_applies(n, d),
                "case must exercise the fused path"
            );
            let f: Vec<F128> = (0..n).map(|_| rng.sample_f128()).collect();
            let z1: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
            let z2: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
            let mut base = VirtualEqBasis::new(z1, rng.sample_f128());
            base.add_term(z2, rng.sample_f128());
            let mut vb = VirtualEqBasis256::from_base(base);
            let r = random_f256(&mut rng);
            vb.fold_coord(d.trailing_zeros() as usize, r);
            let (nf, nb, msg) = fused_first_fold_virtual(&f, &vb, r, d);
            let ef = fold_base(&f, r, d);
            let eb = vb.materialize();
            assert_eq!(nf, ef, "virtual fold f (log_n {log_n}, d {d})");
            assert_eq!(nb, eb, "virtual basis (log_n {log_n}, d {d})");
            assert_eq!(
                msg,
                next_round_msg(&ef, &eb, d),
                "virtual msg (log_n {log_n}, d {d})"
            );
        }
    }

    #[test]
    fn coordinate_split_preserves_every_linear_claim() {
        let mut rng = RandomChallenger::new(0xC001_D1A7);
        for log_n in 0..8 {
            let n = 1usize << log_n;
            let values: Vec<F256> = (0..n).map(|_| random_f256(&mut rng)).collect();
            let basis: Vec<F256> = (0..n).map(|_| random_f256(&mut rng)).collect();
            let expected = values
                .iter()
                .zip(&basis)
                .fold(F256::ZERO, |acc, (&f, &b)| acc + f * b);
            let words = split_coordinates(&values);
            let weights = split_basis(&basis);
            assert_eq!(split_inner_product(&words, &weights), expected);
        }
    }

    #[test]
    fn virtual_eq_basis_matches_dense_extension_folds() {
        let mut rng = RandomChallenger::new(0xF256_BA51);
        for log_n in 6..11 {
            for initial_k in 1..=4 {
                let log_cols = log_n - initial_k;
                let block = 1usize << log_cols;
                let point: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
                let gamma = rng.sample_f128();
                let point_ood: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
                let beta = rng.sample_f128();
                let mut dense: Vec<F256> = build_eq_table(&point)
                    .into_iter()
                    .zip(build_eq_table(&point_ood))
                    .map(|(value, ood)| F256::from(gamma * value + beta * ood))
                    .collect();
                let mut base = VirtualEqBasis::new(point, gamma);
                base.add_term(point_ood, beta);
                let mut virtual_basis = VirtualEqBasis256::from_base(base);
                for _ in 0..initial_k {
                    let r = random_f256(&mut rng);
                    dense = fold_extension(&dense, r, block);
                    virtual_basis.fold_coord(log_cols, r);
                    assert_eq!(
                        virtual_basis.materialize(),
                        dense,
                        "log_n={log_n}, initial_k={initial_k}"
                    );
                }
            }
        }
    }

    #[test]
    fn split_ood_answer_is_a_single_base_field_value() {
        let mut rng = RandomChallenger::new(0x00D5_0127);
        let values: Vec<F256> = (0..8).map(|_| random_f256(&mut rng)).collect();
        let words = split_coordinates(&values);
        let z: Vec<F128> = (0..4).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let answer = words
            .iter()
            .zip(eq)
            .fold(F128::ZERO, |acc, (&word, weight)| acc + word * weight);
        // The committed MLE is base-valued even though its coordinate-weighted
        // claim reconstructs an extension value.
        assert_eq!(F256::from(answer).c1, F128::ZERO);
    }

    #[test]
    fn queried_consistency_uses_all_coordinate_rows() {
        let mut rng = RandomChallenger::new(0xC05E_157E);
        let lane_point: Vec<F256> = (0..3).map(|_| random_f256(&mut rng)).collect();
        let alpha: Vec<F128> = (0..2).map(|_| rng.sample_f128()).collect();
        let rows: Vec<Vec<F128>> = (0..4)
            .map(|_| (0..8).map(|_| rng.sample_f128()).collect())
            .collect();
        let lane_weights = build_eq_table256(&lane_point);
        let row_weights = build_eq_table(&alpha);
        let expected = rows
            .iter()
            .zip(row_weights)
            .fold(F256::ZERO, |outer, (row, row_weight)| {
                outer
                    + row
                        .iter()
                        .zip(&lane_weights)
                        .fold(F256::ZERO, |inner, (&word, &weight)| inner + weight * word)
                        * row_weight
            });
        assert_eq!(induce_enforced_sum(&rows, &lane_point, &alpha), expected);
    }
}
