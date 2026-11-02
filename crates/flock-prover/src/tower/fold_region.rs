use std::iter::repeat_n;

use flock_core::{
    circuit::{SigmaAssertion, builder::SlotId},
    genus95_curve_code::{EvaluationPoint, SAMPLE_ATTEMPT_BUDGET, base_evaluation_functional},
    lincheck::MatrixAssertion,
    matrix_fold::{FoldProof, JaggedClaim, JaggedRowWeight, MatrixClaim, Weight},
    zerocheck::{K_SKIP, ag_skip::R1_FUSED_ATTEMPT_BUDGET},
};
use flock_transcript::transcript_record::{TranscriptOp, TranscriptOp as Op};

use crate::{
    r1cs_hashes::fs_chain::FsChainTrace,
    schedule::Registry,
    tower::{F128, ShapeBuilder, TowerConfig, Wire, squeeze_word_wire, tower_fold_grinding},
};

/// One absorbed claim's stream ordinals on a fold tape: the four weight
/// slices and the value, in absorb order — or, for a claim the aggregate's
/// shared binding bound, where those words sit in the shared block.
#[derive(Clone)]
pub(super) struct ClaimLoc {
    pub(super) row_low_v: usize,
    pub(super) row_low_n: usize,
    pub(super) row_pt_v: usize,
    pub(super) row_pt_n: usize,
    pub(super) col_low_v: usize,
    pub(super) col_low_n: usize,
    pub(super) col_pt_v: usize,
    pub(super) col_pt_n: usize,
    pub(super) value_v: usize,
    /// A structure claim bound by its assertion's shared surface
    /// ([`locate_shared_sigma`]): both lows are the constant 1 (the
    /// `low_v`/`low_n` fields are unused) and the column point continues
    /// past its `col_pt_n` stream words with these constant bits (padding,
    /// index and plane selector).
    pub(super) consts: Option<Vec<F128>>,
}

impl ClaimLoc {
    /// The claim's row and column weights, rebuilt from located words.
    fn weights(&self, vals_rec: &[F128]) -> (Weight, Weight) {
        let row_pt = vals_rec[self.row_pt_v..self.row_pt_v + self.row_pt_n].to_vec();
        let mut col_pt = vals_rec[self.col_pt_v..self.col_pt_v + self.col_pt_n].to_vec();
        match &self.consts {
            Some(tail) => {
                col_pt.extend_from_slice(tail);
                (
                    Weight::low_eq(vec![F128::ONE], row_pt),
                    Weight::low_eq(vec![F128::ONE], col_pt),
                )
            }
            None => (
                Weight::low_eq(
                    vals_rec[self.row_low_v..self.row_low_v + self.row_low_n].to_vec(),
                    row_pt,
                ),
                Weight::low_eq(
                    vals_rec[self.col_low_v..self.col_low_v + self.col_low_n].to_vec(),
                    col_pt,
                ),
            ),
        }
    }
}

/// One fold group's ordinals: its claims, then the lambdas, col rounds,
/// bridge, mus, row rounds and output value.
pub(super) struct FoldLoc {
    pub(super) claims: Vec<ClaimLoc>,
    lam_ch0: usize,
    pub(super) col_v: usize,
    col_ch0: usize,
    pub(super) k_col: usize,
    bridge_v: usize,
    mu_ch0: usize,
    row_v: usize,
    row_ch0: usize,
    pub(super) k_row: usize,
    out_v: usize,
}

/// (public index, fold, row side, h) of one boundary-expanded low-fold eq
/// public — checker-validated against the fold's PUBLISHED ρ coordinates.
pub(super) type AlphaRec = (usize, usize, bool, usize);

/// One emitted fold group's wires: the accumulator claim (ρ_col, ρ_row,
/// value) to publish. The two endpoint identities are COPY CONSTRAINTS
/// (`connect`), not published zero-deltas — the proof itself fails on a
/// broken endpoint, and no public or checker item exists for it.
pub(super) struct FoldPub {
    /// The entry's LIVE word — the zero-claim scale (wall 3): a real
    /// entry publishes 1 (ow); an absent one is all zeros, which decodes
    /// as the zero claim. Fold outputs are always real.
    pub(super) live: Wire,
    pub(super) rho_col: Vec<Wire>,
    pub(super) rho_row: Vec<Wire>,
    pub(super) value: Wire,
}

/// The fold region's op tape for a claim-list set: per group, the
/// matrix-fold label, every claim's four weight slices + value, the
/// lambdas, col rounds, bridge, mus, row rounds, and the output value.
/// Width-driven, so mixed low widths and any claim count pin themselves.
pub(super) fn fold_region_ops(
    cfg: TowerConfig,
    fold_claims: &[Vec<MatrixClaim>],
) -> Vec<TranscriptOp> {
    fold_region_ops_bound(cfg, fold_claims, &vec![0; fold_claims.len()])
}

/// [`fold_region_ops`] where group `g`'s last `n_bound[g]` claims were bound
/// by the shared block ([`shared_binding_ops`]) and are not observed again.
pub(super) fn fold_region_ops_bound(
    cfg: TowerConfig,
    fold_claims: &[Vec<MatrixClaim>],
    n_bound: &[usize],
) -> Vec<TranscriptOp> {
    let mut want: Vec<Op> = Vec::new();
    let grinding = tower_fold_grinding(cfg);
    for (cs, &nb) in fold_claims.iter().zip(n_bound) {
        want.push(Op::Label(b"flock-matrix-fold-v0".to_vec()));
        for c in &cs[..cs.len() - nb] {
            want.extend([
                Op::ObserveSlice(c.row.low.len()),
                Op::ObserveSlice(c.row.point.len()),
                Op::ObserveSlice(c.col.low.len()),
                Op::ObserveSlice(c.col.point.len()),
                Op::ObserveScalar,
            ]);
        }
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len())); // lambdas
        for _ in 0..cs[0].col.n_vars() {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        for _ in 0..cs.len() {
            want.push(Op::ObserveScalar); // bridge
        }
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len())); // mus
        for _ in 0..cs[0].row.n_vars() {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        want.push(Op::ObserveScalar); // the output value
    }
    want
}

/// The aggregate's shared binding: its label, then per boolean assertion
/// the row low, `x_inner_rest`, the column low, `rr`, each type's `(A, B)`
/// values and the linear values, then per sigma assertion its shared
/// surface.
pub(super) fn shared_binding_ops(
    asserts: &[MatrixAssertion],
    sigmas: &[&SigmaAssertion],
) -> Vec<TranscriptOp> {
    let mut want = vec![Op::Label(b"flock-aggregate-shared-v1".to_vec())];
    for a in asserts {
        want.extend([
            Op::ObserveSlice(1 << K_SKIP),
            Op::ObserveSlice(a.x_inner_rest.len()),
            Op::ObserveSlice(a.z_partial.len()),
            Op::ObserveSlice(a.rr.len()),
        ]);
        want.extend(repeat_n(Op::ObserveScalar, 2 * a.evals.len()));
        want.extend(repeat_n(
            Op::ObserveScalar,
            a.linear_evals.iter().flatten().count(),
        ));
    }
    for sa in sigmas {
        want.push(Op::ObserveSlice(sa.shared_surface().len()));
    }
    want
}

/// Locate each sigma assertion's shared surface from value ordinal `v0`
/// (after the boolean blocks), pin it, and hand back per assertion its
/// claims' locations in [`SigmaAssertion::claims`] order: row points and
/// the GKR planes' column heads as stream words, everything else constant.
/// Returns the locations and the value cursor past the surfaces.
pub(super) fn locate_shared_sigma(
    sigmas: &[&SigmaAssertion],
    vals_rec: &[F128],
    v0: usize,
) -> (Vec<Vec<ClaimLoc>>, usize) {
    let bits = |x: usize, n: usize| -> Vec<F128> {
        (0..n)
            .map(|j| F128::new(((x >> j) & 1) as u64, 0))
            .collect()
    };
    let mut v = v0;
    let out = sigmas
        .iter()
        .map(|sa| {
            let surface = sa.shared_surface();
            assert_eq!(
                &vals_rec[v..v + surface.len()],
                &surface[..],
                "the sigma surface on the stream"
            );
            assert!(sa.element_constants.is_none(), "no element side here");
            let (nu, mu, base) = (sa.nu, sa.rho.len(), sa.base_bits);
            let rho_v = v;
            let mut at = v + mu;
            let pin_v = (!sa.boolean_pins.is_empty()).then(|| {
                at += nu;
                at - nu
            });
            let bus_v = sa.bus.as_ref().map(|_| {
                at += nu;
                at - nu
            });
            let pc_v = sa.public_columns.as_ref().map(|_| {
                at += nu;
                at - nu
            });
            let mut value_v = at;
            v = at + sa.claims().len();
            let mut claim = |row_pt_v: usize, col_pt_v: usize, col_pt_n: usize, tail: Vec<F128>| {
                let cl = ClaimLoc {
                    row_low_v: 0,
                    row_low_n: 1,
                    row_pt_v,
                    row_pt_n: nu,
                    col_low_v: 0,
                    col_low_n: 1,
                    col_pt_v,
                    col_pt_n,
                    value_v,
                    consts: Some(tail),
                };
                value_v += 1;
                cl
            };
            let mut locs = Vec::new();
            for plane in 0..3usize {
                let mut tail = vec![F128::ZERO; base - (mu - nu)];
                tail.extend(bits(plane, 3));
                locs.push(claim(rho_v, rho_v + nu, mu - nu, tail));
            }
            for &(t, _, _) in &sa.boolean_pins {
                let mut tail = bits(t, base);
                tail.extend(bits(5, 3));
                locs.push(claim(pin_v.expect("pins"), 0, 0, tail));
            }
            for (c, _) in sa.bus.iter().flat_map(|b| b.values.iter().enumerate()) {
                let mut tail = bits(c, base);
                tail.extend(bits(6, 3));
                locs.push(claim(bus_v.expect("bus"), 0, 0, tail));
            }
            for (c, _) in sa
                .public_columns
                .iter()
                .flat_map(|p| p.values.iter().enumerate())
            {
                let mut tail = bits(c, base);
                tail.extend(bits(7, 3));
                locs.push(claim(pc_v.expect("public columns"), 0, 0, tail));
            }
            assert_eq!(locs.len(), sa.claims().len(), "one location per claim");
            locs
        })
        .collect();
    (out, v)
}

/// One type's claim locations in the shared block, per assertion: its
/// `(A, B)` pair and, when it reports one, its linear-identity claim.
pub(super) type SharedLocs = Vec<Vec<(ClaimLoc, ClaimLoc, Option<ClaimLoc>)>>;

/// Locate the shared block from value ordinal `v0`, pin it against the
/// assertions, and hand back, per boolean type, each assertion's claim
/// locations inside it — the claims' slices are the block's (each type
/// reads a prefix of the points), so several claims share ordinals.
/// Returns the per-type locations and the value cursor past the block.
pub(super) fn locate_shared_binding(
    asserts: &[MatrixAssertion],
    registry: &Registry,
    vals_rec: &[F128],
    v0: usize,
) -> (SharedLocs, usize) {
    let n_types = registry.num_boolean();
    let mut per_type: SharedLocs = vec![Vec::new(); n_types];
    let mut v = v0;
    for a in asserts {
        let low_v = v;
        let x_v = low_v + (1 << K_SKIP);
        let zp_v = x_v + a.x_inner_rest.len();
        let rr_v = zp_v + a.z_partial.len();
        let ev_v = rr_v + a.rr.len();
        let lin_v = ev_v + 2 * a.evals.len();
        v = lin_v + a.linear_evals.iter().flatten().count();
        assert_eq!(
            &vals_rec[low_v..x_v],
            &a.z_skip.weights(K_SKIP)[..],
            "shared row low"
        );
        assert_eq!(
            &vals_rec[x_v..zp_v],
            &a.x_inner_rest[..],
            "shared x_inner_rest"
        );
        assert_eq!(&vals_rec[zp_v..rr_v], &a.z_partial[..], "shared z_partial");
        assert_eq!(&vals_rec[rr_v..ev_v], &a.rr[..], "shared rr");
        for (t, ty) in registry.boolean_types().iter().enumerate() {
            let inner = ty.k_log - K_SKIP;
            let loc = |value_v: usize| ClaimLoc {
                row_low_v: low_v,
                row_low_n: 1 << K_SKIP,
                row_pt_v: x_v,
                row_pt_n: inner,
                col_low_v: zp_v,
                col_low_n: a.z_partial.len(),
                col_pt_v: rr_v,
                col_pt_n: inner,
                value_v,
                consts: None,
            };
            assert_eq!(vals_rec[ev_v + 2 * t], a.evals[t].0, "shared A value");
            assert_eq!(vals_rec[ev_v + 2 * t + 1], a.evals[t].1, "shared B value");
            // The linear values follow the pairs, in type order.
            let lin = a.linear_evals[t].map(|value| {
                let at = lin_v + a.linear_evals[..t].iter().flatten().count();
                assert_eq!(vals_rec[at], value, "shared linear value");
                loc(at)
            });
            per_type[t].push((loc(ev_v + 2 * t), loc(ev_v + 2 * t + 1), lin));
        }
    }
    (per_type, v)
}

/// Locate every fold's surfaces on the value/challenge streams (counters
/// start at 0 — the bind prefix carries only byte payloads) and pin them
/// field-for-field against the gathered claims and the `FoldProof`s.
/// Returns the counters alongside, so JAGGED groups on the same tape can
/// continue the walk ([`locate_and_pin_jagged_folds`]); callers with no
/// jagged groups assert exhaustion themselves via
/// [`assert_fold_tape_exhausted`].
pub(super) fn locate_and_pin_folds(
    fold_claims: &[Vec<MatrixClaim>],
    fold_proofs: &[&FoldProof],
    vals_rec: &[F128],
    chals: &[F128],
) -> (Vec<FoldLoc>, usize, usize) {
    locate_and_pin_folds_from(fold_claims, fold_proofs, vals_rec, chals, 0, &[])
}

/// [`locate_and_pin_folds`] from value ordinal `v0`, where group `g`'s last
/// `prebound[g].len()` claims are already located (in the shared block,
/// [`locate_shared_binding`]) and absent from the group's own words.
pub(super) fn locate_and_pin_folds_from(
    fold_claims: &[Vec<MatrixClaim>],
    fold_proofs: &[&FoldProof],
    vals_rec: &[F128],
    _chals: &[F128],
    v0: usize,
    prebound: &[Vec<ClaimLoc>],
) -> (Vec<FoldLoc>, usize, usize) {
    let (mut vcur, mut ccur) = (v0, 0usize);
    let locs: Vec<FoldLoc> = fold_claims
        .iter()
        .enumerate()
        .map(|(g, cs)| {
            let pre: &[ClaimLoc] = prebound.get(g).map_or(&[], |p| &p[..]);
            let n_own = cs.len() - pre.len();
            let claims = cs[..n_own]
                .iter()
                .map(|c| {
                    let l = ClaimLoc {
                        row_low_v: vcur,
                        row_low_n: c.row.low.len(),
                        row_pt_v: vcur + c.row.low.len(),
                        row_pt_n: c.row.point.len(),
                        col_low_v: vcur + c.row.low.len() + c.row.point.len(),
                        col_low_n: c.col.low.len(),
                        col_pt_v: vcur + c.row.low.len() + c.row.point.len() + c.col.low.len(),
                        col_pt_n: c.col.point.len(),
                        value_v: vcur
                            + c.row.low.len()
                            + c.row.point.len()
                            + c.col.low.len()
                            + c.col.point.len(),
                        consts: None,
                    };
                    vcur = l.value_v + 1;
                    l
                })
                .chain(pre.iter().cloned())
                .collect::<Vec<_>>();
            let (k_col, k_row) = (cs[0].col.n_vars(), cs[0].row.n_vars());
            let lam_ch0 = ccur;
            ccur += cs.len();
            let col_v = vcur;
            let col_ch0 = ccur;
            vcur += 2 * k_col;
            ccur += k_col;
            let bridge_v = vcur;
            vcur += cs.len();
            let mu_ch0 = ccur;
            ccur += cs.len();
            let row_v = vcur;
            let row_ch0 = ccur;
            vcur += 2 * k_row;
            ccur += k_row;
            let out_v = vcur;
            vcur += 1;
            FoldLoc {
                claims,
                lam_ch0,
                col_v,
                col_ch0,
                k_col,
                bridge_v,
                mu_ch0,
                row_v,
                row_ch0,
                k_row,
                out_v,
            }
        })
        .collect();
    for ((loc, cs), fp) in locs.iter().zip(fold_claims).zip(fold_proofs) {
        for (cl, c) in loc.claims.iter().zip(cs) {
            let (row, col) = cl.weights(vals_rec);
            assert_eq!(row, c.row, "row weight on the stream");
            assert_eq!(col, c.col, "col weight on the stream");
            assert_eq!(vals_rec[cl.value_v], c.value, "claim value on the stream");
        }
        for (j, &(q1, qinf)) in fp.col_rounds.iter().enumerate() {
            assert_eq!(vals_rec[loc.col_v + 2 * j], q1, "col round q(1)");
            assert_eq!(vals_rec[loc.col_v + 2 * j + 1], qinf, "col round q(inf)");
        }
        assert_eq!(
            &vals_rec[loc.bridge_v..loc.bridge_v + loc.claims.len()],
            &fp.bridge[..],
            "the bridge on the stream"
        );
        for (j, &(q1, qinf)) in fp.row_rounds.iter().enumerate() {
            assert_eq!(vals_rec[loc.row_v + 2 * j], q1, "row round q(1)");
            assert_eq!(vals_rec[loc.row_v + 2 * j + 1], qinf, "row round q(inf)");
        }
        assert_eq!(vals_rec[loc.out_v], fp.value, "output value on the stream");
    }
    (locs, vcur, ccur)
}

/// Map every challenge ordinal to the transcript finalization and output-word
/// offset that emitted it. Grinding adds finalizations without challenges,
/// while vector squeezes emit several challenge words from one finalization.
pub(super) fn challenge_word_locs(ops: &[TranscriptOp]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut fin = 0usize;
    for op in ops {
        match op {
            Op::SqueezeScalar => out.push((fin, 0)),
            Op::SqueezeSlice(n) => out.extend((0..*n).map(|offset| (fin, offset))),
            _ => {}
        }
        if op.finalizes() {
            fin += 1;
        }
    }
    out
}

/// Replay every fold's two endpoint identities from LOCATED words alone —
/// weights rebuilt through the verifier's own `Weight::eval`, the low fold
/// included — and return the located fold outputs (what the verifier's
/// accumulator must equal, surface for surface).
pub(super) fn replay_fold_endpoints(
    locs: &[FoldLoc],
    vals_rec: &[F128],
    chals: &[F128],
) -> Vec<MatrixClaim> {
    let replay_rounds = |target: F128, base: usize, ch0: usize, n: usize| -> (F128, Vec<F128>) {
        let mut run = target;
        let mut rho = Vec::with_capacity(n);
        for j in 0..n {
            let (g1, gi) = (vals_rec[base + 2 * j], vals_rec[base + 2 * j + 1]);
            let r = chals[ch0 + j];
            let q0 = run + g1;
            run = gi * r * r + (q0 + g1 + gi) * r + q0;
            rho.push(r);
        }
        (run, rho)
    };
    locs.iter()
        .map(|loc| {
            let k = loc.claims.len();
            let lam: Vec<F128> = (0..k).map(|i| chals[loc.lam_ch0 + i]).collect();
            let target_c = loc
                .claims
                .iter()
                .zip(&lam)
                .fold(F128::ZERO, |acc, (cl, &l)| acc + l * vals_rec[cl.value_v]);
            let (run_c, rho_col) = replay_rounds(target_c, loc.col_v, loc.col_ch0, loc.k_col);
            let expect_c =
                loc.claims
                    .iter()
                    .zip(&lam)
                    .enumerate()
                    .fold(F128::ZERO, |acc, (i, (cl, &l))| {
                        let w = cl.weights(vals_rec).1;
                        acc + l * w.eval(&rho_col) * vals_rec[loc.bridge_v + i]
                    });
            assert_eq!(run_c, expect_c, "col endpoint closes from located words");

            let mus: Vec<F128> = (0..k).map(|i| chals[loc.mu_ch0 + i]).collect();
            let target_r = (0..k).zip(&mus).fold(F128::ZERO, |acc, (i, &m)| {
                acc + m * vals_rec[loc.bridge_v + i]
            });
            let (run_r, rho_row) = replay_rounds(target_r, loc.row_v, loc.row_ch0, loc.k_row);
            let w_mu = loc
                .claims
                .iter()
                .zip(&mus)
                .fold(F128::ZERO, |acc, (cl, &m)| {
                    let w = cl.weights(vals_rec).0;
                    acc + m * w.eval(&rho_row)
                });
            assert_eq!(
                run_r,
                w_mu * vals_rec[loc.out_v],
                "row endpoint closes from located words"
            );
            MatrixClaim {
                row: Weight::eq(rho_row),
                col: Weight::eq(rho_col),
                value: vals_rec[loc.out_v],
            }
        })
        .collect()
}

/// Emit the WHOLE fold region in-circuit: per group, the λ-combination of
/// the absorbed claim values, the col rounds (MergedRoundGate), the col
/// endpoint's weight evals (eq parts on the prefix slot; 64-wide lows
/// through 8 chained LeafEvalGate(8) rows with boundary-public hi-group
/// factors), then the μ side and the row endpoint — both endpoints as
/// zero-delta wires. The LAST fold's output value sits in the transcript
/// tail past the final squeeze (no chain wire) and enters as its own
/// input, bound by the row endpoint delta. Returns the per-fold publish
/// wires and the boundary-public records the checker validates.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_fold_region(
    sb: &mut ShapeBuilder,
    macs: SlotId,
    mrs: SlotId,
    pfslot: SlotId,
    pf_w: usize,
    leslot: SlotId,
    locs: &[FoldLoc],
    trace: &FsChainTrace,
    challenge_locs: &[(usize, usize)],
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vmap: &[Option<usize>],
    chals: &[F128],
    vals_rec: &[F128],
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    tail_input_last: bool,
) -> (Vec<FoldPub>, Vec<AlphaRec>) {
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    let chw = |ch: usize| -> Wire {
        let (fin, offset) = challenge_locs[ch];
        squeeze_word_wire(outs, trace, fin, offset)
    };
    // seed · Π (1 + a_j + b_j) through the prefix slot, padded (zw, zw).
    let prefix = |sb: &mut ShapeBuilder, seed: Wire, fs: &[(Wire, Wire)]| -> Wire {
        let mut s = seed;
        for chunk in fs.chunks(pf_w) {
            let mut g_in = vec![s];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(repeat_n(zw, pf_w - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(repeat_n(zw, pf_w - chunk.len()));
            g_in.push(ow);
            s = sb.gate(pfslot, &g_in)[0];
        }
        s
    };
    // One weight eval at ρ: the low factor's MLE (seeded by the single
    // absorbed low word for eq weights, or folded through 8 LeafEval rows
    // for the 64-entry lows), times the eq-point prefix product over the
    // remaining coordinates.
    let mut alpha_recs: Vec<AlphaRec> = Vec::new();
    let emit_weight = |sb: &mut ShapeBuilder,
                       vals: &mut Vec<F128>,
                       recs: &mut Vec<AlphaRec>,
                       fi: usize,
                       row_side: bool,
                       cl: &ClaimLoc,
                       rho_w: &[Wire],
                       rho_vals: &[F128]|
     -> Wire {
        let (low_v, low_n, pt_v, pt_n) = if row_side {
            (cl.row_low_v, cl.row_low_n, cl.row_pt_v, cl.row_pt_n)
        } else {
            (cl.col_low_v, cl.col_low_n, cl.col_pt_v, cl.col_pt_n)
        };
        // A shared-bound structure claim: low 1, and the column point's
        // constant tail bakes as zw/ow.
        let (low_n, tail): (usize, &[F128]) = match &cl.consts {
            Some(tail) => (1, if row_side { &[] } else { &tail[..] }),
            None => (low_n, &[]),
        };
        let s = low_n.trailing_zeros() as usize;
        let seed = if cl.consts.is_some() {
            ow
        } else if low_n == 1 {
            wv(low_v)
        } else {
            assert_eq!(low_n, 64, "the lincheck low width");
            let mut acc = zw;
            for h in 0..8 {
                let mut a = F128::ONE;
                for b in 0..3 {
                    let r = rho_vals[3 + b];
                    a *= if (h >> b) & 1 == 1 { r } else { F128::ONE + r };
                }
                vals.push(a);
                // Record the PUBLIC index (not the input ordinal): with
                // other regions sharing the builder the two need not
                // coincide.
                recs.push((sb.public_len(), fi, row_side, h));
                let a_w = sb.public_input();
                let mut g_in: Vec<Wire> = (0..8).map(|j| wv(low_v + 8 * h + j)).collect();
                g_in.extend([rho_w[0], rho_w[1], rho_w[2]]);
                g_in.push(a_w);
                g_in.push(acc);
                acc = sb.gate(leslot, &g_in)[0];
            }
            acc
        };
        let bit = |c: F128| {
            if c == F128::ZERO {
                zw
            } else {
                assert_eq!(c, F128::ONE, "a constant coordinate is a bit");
                ow
            }
        };
        let fs: Vec<(Wire, Wire)> = (0..pt_n)
            .map(|j| (wv(pt_v + j), rho_w[s + j]))
            .chain(
                tail.iter()
                    .enumerate()
                    .map(|(j, &c)| (bit(c), rho_w[s + pt_n + j])),
            )
            .collect();
        prefix(sb, seed, &fs)
    };

    let mut fold_pubs: Vec<FoldPub> = Vec::new();
    for (fi, loc) in locs.iter().enumerate() {
        let k = loc.claims.len();
        let lam_w: Vec<Wire> = (0..k).map(|i| chw(loc.lam_ch0 + i)).collect();
        let mut run_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            run_w = sb.gate(macs, &[run_w, lam_w[i], wv(cl.value_v)])[0];
        }
        let mut rho_col_w: Vec<Wire> = Vec::with_capacity(loc.k_col);
        for j in 0..loc.k_col {
            let r_w = chw(loc.col_ch0 + j);
            rho_col_w.push(r_w);
            run_w = sb.gate(
                mrs,
                &[run_w, wv(loc.col_v + 2 * j), wv(loc.col_v + 2 * j + 1), r_w],
            )[0];
        }
        let rho_col_vals: Vec<F128> = (0..loc.k_col).map(|j| chals[loc.col_ch0 + j]).collect();
        let mut exp_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let w = emit_weight(
                sb,
                vals,
                &mut alpha_recs,
                fi,
                false,
                cl,
                &rho_col_w,
                &rho_col_vals,
            );
            let t = sb.gate(macs, &[zw, w, wv(loc.bridge_v + i)])[0];
            exp_w = sb.gate(macs, &[exp_w, lam_w[i], t])[0];
        }
        // The col endpoint: running == expect, as a copy constraint.
        sb.connect(run_w, exp_w);

        let mu_w: Vec<Wire> = (0..k).map(|i| chw(loc.mu_ch0 + i)).collect();
        let mut run2_w = zw;
        for i in 0..k {
            run2_w = sb.gate(macs, &[run2_w, mu_w[i], wv(loc.bridge_v + i)])[0];
        }
        let mut rho_row_w: Vec<Wire> = Vec::with_capacity(loc.k_row);
        for j in 0..loc.k_row {
            let r_w = chw(loc.row_ch0 + j);
            rho_row_w.push(r_w);
            run2_w = sb.gate(
                mrs,
                &[
                    run2_w,
                    wv(loc.row_v + 2 * j),
                    wv(loc.row_v + 2 * j + 1),
                    r_w,
                ],
            )[0];
        }
        let rho_row_vals: Vec<F128> = (0..loc.k_row).map(|j| chals[loc.row_ch0 + j]).collect();
        let mut wmu_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let w = emit_weight(
                sb,
                vals,
                &mut alpha_recs,
                fi,
                true,
                cl,
                &rho_row_w,
                &rho_row_vals,
            );
            wmu_w = sb.gate(macs, &[wmu_w, mu_w[i], w])[0];
        }
        // The LAST fold's output value sits in the transcript tail past
        // the final squeeze — no chain wire (step 1's shape fact); it
        // enters as its own input, bound by the row endpoint delta. When
        // JAGGED groups follow on the same tape (`tail_input_last =
        // false`), their absorbs flush this word and it has a chain wire
        // like any other — the tail treatment moves to the last jagged
        // group.
        let value = if tail_input_last && fi + 1 == locs.len() {
            vals.push(vals_rec[loc.out_v]);
            sb.input()
        } else {
            wv(loc.out_v)
        };
        // The row endpoint: running == weight·value, as a copy constraint
        // (this is also what binds the LAST fold's tail-input value).
        let rhs_w = sb.gate(macs, &[zw, wmu_w, value])[0];
        sb.connect(run2_w, rhs_w);
        fold_pubs.push(FoldPub {
            live: ow,
            rho_col: rho_col_w,
            rho_row: rho_row_w,
            value,
        });
    }
    (fold_pubs, alpha_recs)
}

/// Read ONE published accumulator entry at `p`, advancing it.
///
/// Entry layout (wall 3): `[key | live | rho_col | rho_row | value]` — the
/// two KEY words only for the keyed groups (sigma, jagged), where the
/// entry names the circuit whose table it is about; the registry-keyed
/// matrix groups carry none. The LIVE word is the zero-claim scale, so a
/// block of zeros decodes as the zero claim: weights identically zero,
/// value zero, true about every table. That is what a DEAD SLOT is, and
/// why a base node and a steady node can be read at the same offsets.
pub(super) fn read_acc_entry(
    public: &[F128],
    p: &mut usize,
    keyed: bool,
    k_col: usize,
    k_row: usize,
) -> ([F128; 2], MatrixClaim) {
    let key = if keyed {
        *p += 2;
        [public[*p - 2], public[*p - 1]]
    } else {
        [F128::ZERO; 2]
    };
    let live = public[*p];
    let rho_col = public[*p + 1..*p + 1 + k_col].to_vec();
    let rho_row = public[*p + 1 + k_col..*p + 1 + k_col + k_row].to_vec();
    let value = public[*p + 1 + k_col + k_row];
    *p += 2 + k_col + k_row;
    (
        key,
        MatrixClaim {
            row: Weight::low_eq(vec![live], rho_row),
            col: Weight::low_eq(vec![live], rho_col),
            value,
        },
    )
}

/// The AG-skip surface checker: walk one AG child's published block
/// `[seed₂, nonce, point ×5, lows ×64]`. Two items stay at the checker
/// tier since phase D moved the decode in-circuit
/// ([`emit_ag_point_binding`]):
/// - the NONCE RANGE — the published nonce word (wire-connected to the
///   absorbed stream word) must be a native nonce: high half zero, low
///   half inside the schedule's scan budget. The in-circuit PowMask row
///   pins only the word's high 64 bits (and Chain100 emits no row), so
///   without this item the circuit would accept nonce words no native
///   verifier accepts;
/// - `lows == bf(point)` — the base functional at the published point.
/// Both are the leaf skip-interpolation class of obligation; the
/// genus-95 sampler itself stays out of the exit checker set. Returns
/// the number of public words consumed.
pub(super) fn check_ag_skip_publics(
    public: &[F128],
    base: usize,
    ag_r1_bits: Option<u32>,
) -> usize {
    let nonce_word = public[base + 2];
    assert_eq!(nonce_word.hi, 0, "the nonce word's high half is zero");
    let budget = match ag_r1_bits {
        Some(_) => R1_FUSED_ATTEMPT_BUDGET,
        None => SAMPLE_ATTEMPT_BUDGET,
    };
    assert!(
        nonce_word.lo < u64::from(budget),
        "the nonce is inside the schedule's scan budget"
    );
    let pt = EvaluationPoint {
        x: public[base + 3],
        y: public[base + 4],
        z1: public[base + 5],
        z2: public[base + 6],
        z3: public[base + 7],
    };
    let bf =
        base_evaluation_functional(&pt).expect("the base functional exists at a decoded point");
    for j in 0..64 {
        assert_eq!(public[base + 8 + j], bf[j], "published AG row low {j}");
    }
    72
}

/// Walk the published fold blocks from `tail0`: both endpoint deltas zero
/// per fold, the accumulator claims rebuilt from the PUBLIC SEGMENT alone,
/// and every boundary-expanded low-fold eq public validated against the
/// PUBLISHED ρ coordinates. `locs[keyed_from..]` are the KEYED groups (the
/// sigma slots ride the uniform tape's tail). Returns the rebuilt claims,
/// their keys, and the offset just past the last entry.
pub(super) fn check_fold_publics(
    public: &[F128],
    tail0: usize,
    locs: &[FoldLoc],
    alpha_recs: &[AlphaRec],
    keyed_from: usize,
) -> (Vec<MatrixClaim>, Vec<[F128; 2]>, usize) {
    let width = |i: usize, l: &FoldLoc| 2 + l.k_col + l.k_row + if i >= keyed_from { 2 } else { 0 };
    let mut p = tail0;
    let mut rebuilt: Vec<MatrixClaim> = Vec::new();
    let mut keys: Vec<[F128; 2]> = Vec::new();
    for (i, loc) in locs.iter().enumerate() {
        let (k, c) = read_acc_entry(public, &mut p, i >= keyed_from, loc.k_col, loc.k_row);
        if i >= keyed_from {
            keys.push(k);
        }
        rebuilt.push(c);
    }
    for &(idx, fi, row_side, h) in alpha_recs {
        let base: usize = tail0
            + locs[..fi]
                .iter()
                .enumerate()
                .map(|(i, l)| width(i, l))
                .sum::<usize>()
            + if fi >= keyed_from { 3 } else { 1 }; // past key + live
        let rho = if row_side {
            &public[base + locs[fi].k_col..base + locs[fi].k_col + locs[fi].k_row]
        } else {
            &public[base..base + locs[fi].k_col]
        };
        let mut e = F128::ONE;
        for b in 0..3 {
            let r = rho[3 + b];
            e *= if (h >> b) & 1 == 1 { r } else { F128::ONE + r };
        }
        assert_eq!(
            public[idx], e,
            "boundary-expanded low-fold eq public (fold {fi}, h {h})"
        );
    }
    (rebuilt, keys, p)
}

// ---------------------------------------------------------------------------
// The jagged fold groups use the aggregate challenger after the uniform folds.
// ---------------------------------------------------------------------------

/// One absorbed JAGGED claim's stream ordinals: the tagged row weight and
/// the col point + value. `terms` empty ⇔ an Eq row (the tag pins which).
pub(super) struct JClaimLoc {
    /// Eq rows: the SCALE word's ordinal (the zero-claim scale — `1` for a
    /// fresh claim, the inherited entry's live word otherwise). Combo rows:
    /// unused (0).
    pub(super) row_scale_v: usize,
    /// Eq rows: (point ordinal, len). Combo rows: unused (0, 0).
    pub(super) row_pt: (usize, usize),
    /// Combo rows: (coeff ordinal, address) per term — the address WORD
    /// sits at coeff ordinal + 1, pinned to its REGISTRY constant.
    pub(super) terms: Vec<(usize, u32)>,
    /// Masked rows: the zeroed layout columns (a constant of the child
    /// circuit); `row_pt` is the point and `row_scale_v` the size word.
    pub(super) masked: Option<Vec<u32>>,
    pub(super) col_v: usize,
    pub(super) val_v: usize,
}

/// A masked row's weight at `rho` from its located point:
/// `eq(point, ρ) + Σ_{y ∈ zeroed} eq(point, y)·eq(y, ρ)` (char 2).
fn masked_row_eval(point: &[F128], zeroed: &[u32], rho: &[F128]) -> F128 {
    let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
    let eq_y = |y: u32, x: &[F128]| {
        x.iter().enumerate().fold(F128::ONE, |e, (l, &v)| {
            e * (F128::ONE + bit((y >> l) & 1 == 1) + v)
        })
    };
    let full = point
        .iter()
        .zip(rho)
        .fold(F128::ONE, |e, (&p, &r)| e * (F128::ONE + p + r));
    zeroed
        .iter()
        .fold(full, |acc, &y| acc + eq_y(y, point) * eq_y(y, rho))
}

/// One jagged fold group's located surfaces — [`FoldLoc`]'s sibling.
pub(super) struct JaggedFoldLoc {
    /// The group's `(k_row, n_claims)` shape header word.
    pub(super) hdr_v: usize,
    pub(super) claims: Vec<JClaimLoc>,
    lam_ch0: usize,
    pub(super) col_v: usize,
    col_ch0: usize,
    pub(super) n_col: usize,
    bridge_v: usize,
    mu_ch0: usize,
    row_v: usize,
    row_ch0: usize,
    pub(super) k_row: usize,
    out_v: usize,
}

/// The jagged groups' op tape: per key, the group label + digest payload,
/// then the jagged fold's ops — the label, the shape header, the tagged
/// variable-width claim blocks, and the two sumchecks. Width-driven.
pub(super) fn jagged_fold_region_ops(
    cfg: TowerConfig,
    keys: &[([u8; 32], Vec<JaggedClaim>)],
) -> Vec<TranscriptOp> {
    let mut want: Vec<Op> = Vec::new();
    let grinding = tower_fold_grinding(cfg);
    for (_, cs) in keys {
        let n_col = cs[0].col.len();
        let k_row = cs
            .iter()
            .find_map(|c| match &c.row {
                JaggedRowWeight::Eq(_, p) | JaggedRowWeight::Masked(p, _) => Some(p.len()),
                JaggedRowWeight::Combo(_) => None,
            })
            .expect("every jagged key carries at least one Eq claim");
        want.push(Op::Label(b"flock-aggregate-jagged-v0".to_vec()));
        want.push(Op::ObserveBytes(32));
        want.push(Op::Label(b"flock-jagged-fold-v0".to_vec()));
        want.push(Op::ObserveScalar); // the (k_row, n_claims) shape header
        for c in cs {
            match &c.row {
                JaggedRowWeight::Eq(_, p) => {
                    // tag, then the zero-claim SCALE, then the point.
                    want.push(Op::ObserveScalar);
                    want.push(Op::ObserveScalar);
                    want.push(Op::ObserveSlice(p.len()));
                }
                JaggedRowWeight::Combo(t) => {
                    want.push(Op::ObserveScalar);
                    for _ in t {
                        want.extend([Op::ObserveScalar, Op::ObserveScalar]);
                    }
                }
                JaggedRowWeight::Masked(p, _) => {
                    // tag, then the mask's SIZE, then the point.
                    want.push(Op::ObserveScalar);
                    want.push(Op::ObserveScalar);
                    want.push(Op::ObserveSlice(p.len()));
                }
            }
            want.push(Op::ObserveSlice(n_col));
            want.push(Op::ObserveScalar);
        }
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len()));
        for _ in 0..n_col {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        want.extend(repeat_n(Op::ObserveScalar, cs.len()));
        if grinding.combination_bits != 0 {
            want.push(Op::Pow {
                bits: grinding.combination_bits,
            });
        }
        want.push(Op::SqueezeSlice(cs.len()));
        for _ in 0..k_row {
            want.extend([Op::ObserveScalar, Op::ObserveScalar]);
            if grinding.round_bits != 0 {
                want.push(Op::Pow {
                    bits: grinding.round_bits,
                });
            }
            want.push(Op::SqueezeScalar);
        }
        want.push(Op::ObserveScalar);
    }
    want
}

/// Payload ordinals of `ObserveBytes` operations immediately following a
/// particular label. `Pow` also contributes one payload (its nonce), so fixed
/// payload offsets are invalid as soon as grinding is enabled.
pub(super) fn labeled_bytes_payloads(ops: &[TranscriptOp], label: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut payload = 0usize;
    for (i, op) in ops.iter().enumerate() {
        if matches!(op, Op::ObserveBytes(_))
            && i > 0
            && matches!(&ops[i - 1], Op::Label(l) if l.as_slice() == label)
        {
            out.push(payload);
        }
        if op.carries_payload() {
            payload += 1;
        }
    }
    out
}

/// Locate + pin the jagged groups AFTER the uniform folds — the value and
/// challenge counters CONTINUE from the callers' (which is why
/// [`locate_and_pin_folds`] hands its counters back), and the digest
/// payloads continue after bind's two. Everything pins field-for-field:
/// the shape header, every tagged weight (Combo ADDRESS words against
/// their registry constants), col points, values, rounds, bridge, output.
#[allow(clippy::too_many_arguments)]
pub(super) fn locate_and_pin_jagged_folds(
    keys: &[([u8; 32], Vec<JaggedClaim>)],
    fps: &[&FoldProof],
    vals_rec: &[F128],
    chals: &[F128],
    payloads: &[Vec<u8>],
    digest_payloads: &[usize],
    mut vcur: usize,
    mut ccur: usize,
) -> Vec<JaggedFoldLoc> {
    assert_eq!(keys.len(), fps.len(), "one fold per jagged key");
    assert_eq!(
        keys.len(),
        digest_payloads.len(),
        "one digest payload per jagged key"
    );
    let locs: Vec<JaggedFoldLoc> = keys
        .iter()
        .zip(fps)
        .zip(digest_payloads)
        .map(|(((digest, cs), fp), &digest_payload)| {
            assert_eq!(
                payloads[digest_payload],
                digest.to_vec(),
                "the group's digest payload"
            );
            let n_col = cs[0].col.len();
            let k_row = cs
                .iter()
                .find_map(|c| match &c.row {
                    JaggedRowWeight::Eq(_, p) | JaggedRowWeight::Masked(p, _) => Some(p.len()),
                    JaggedRowWeight::Combo(_) => None,
                })
                .expect("every jagged key carries at least one Eq claim");
            assert_eq!(
                vals_rec[vcur],
                F128::new(k_row as u64, cs.len() as u64),
                "the group's shape header word"
            );
            let hdr_v = vcur;
            vcur += 1;
            let claims: Vec<JClaimLoc> = cs
                .iter()
                .map(|c| {
                    let tag_v = vcur;
                    let mut masked = None;
                    let (row_pt, terms) = match &c.row {
                        JaggedRowWeight::Eq(scale, p) => {
                            assert_eq!(vals_rec[tag_v], F128::new(0, p.len() as u64), "eq row tag");
                            assert_eq!(vals_rec[tag_v + 1], *scale, "eq row SCALE on the stream");
                            assert_eq!(
                                &vals_rec[tag_v + 2..tag_v + 2 + p.len()],
                                &p[..],
                                "eq row point on the stream"
                            );
                            vcur = tag_v + 2 + p.len();
                            ((tag_v + 2, p.len()), Vec::new())
                        }
                        JaggedRowWeight::Combo(t) => {
                            assert_eq!(
                                vals_rec[tag_v],
                                F128::new(1, t.len() as u64),
                                "combo row tag"
                            );
                            let mut terms = Vec::with_capacity(t.len());
                            for (j, &(coeff, addr)) in t.iter().enumerate() {
                                let cv = tag_v + 1 + 2 * j;
                                assert_eq!(vals_rec[cv], coeff, "combo coeff on the stream");
                                assert_eq!(
                                    vals_rec[cv + 1],
                                    F128::new(addr as u64, 0),
                                    "combo ADDRESS word == the registry constant"
                                );
                                terms.push((cv, addr));
                            }
                            vcur = tag_v + 1 + 2 * t.len();
                            ((0, 0), terms)
                        }
                        JaggedRowWeight::Masked(p, zeroed) => {
                            assert_eq!(
                                vals_rec[tag_v],
                                F128::new(2, p.len() as u64),
                                "masked row tag"
                            );
                            assert_eq!(
                                vals_rec[tag_v + 1],
                                F128::new(zeroed.len() as u64, 0),
                                "masked row SIZE on the stream"
                            );
                            assert_eq!(
                                &vals_rec[tag_v + 2..tag_v + 2 + p.len()],
                                &p[..],
                                "masked row point on the stream"
                            );
                            masked = Some(zeroed.clone());
                            vcur = tag_v + 2 + p.len();
                            ((tag_v + 2, p.len()), Vec::new())
                        }
                    };
                    let col_v = vcur;
                    assert_eq!(
                        &vals_rec[col_v..col_v + n_col],
                        &c.col[..],
                        "col point (σ) on the stream"
                    );
                    let val_v = col_v + n_col;
                    assert_eq!(vals_rec[val_v], c.value, "claim value on the stream");
                    vcur = val_v + 1;
                    JClaimLoc {
                        row_scale_v: tag_v + 1,
                        row_pt,
                        terms,
                        masked,
                        col_v,
                        val_v,
                    }
                })
                .collect();
            let lam_ch0 = ccur;
            ccur += cs.len();
            let col_v = vcur;
            let col_ch0 = ccur;
            vcur += 2 * n_col;
            ccur += n_col;
            let bridge_v = vcur;
            vcur += cs.len();
            let mu_ch0 = ccur;
            ccur += cs.len();
            let row_v = vcur;
            let row_ch0 = ccur;
            vcur += 2 * k_row;
            ccur += k_row;
            let out_v = vcur;
            vcur += 1;
            for (j, &(q1, qinf)) in fp.col_rounds.iter().enumerate() {
                assert_eq!(vals_rec[col_v + 2 * j], q1, "jagged col round q(1)");
                assert_eq!(vals_rec[col_v + 2 * j + 1], qinf, "jagged col round q(inf)");
            }
            assert_eq!(
                &vals_rec[bridge_v..bridge_v + cs.len()],
                &fp.bridge[..],
                "the jagged bridge on the stream"
            );
            for (j, &(q1, qinf)) in fp.row_rounds.iter().enumerate() {
                assert_eq!(vals_rec[row_v + 2 * j], q1, "jagged row round q(1)");
                assert_eq!(vals_rec[row_v + 2 * j + 1], qinf, "jagged row round q(inf)");
            }
            assert_eq!(
                vals_rec[out_v], fp.value,
                "jagged output value on the stream"
            );
            JaggedFoldLoc {
                hdr_v,
                claims,
                lam_ch0,
                col_v,
                col_ch0,
                n_col,
                bridge_v,
                mu_ch0,
                row_v,
                row_ch0,
                k_row,
                out_v,
            }
        })
        .collect();
    assert_eq!(vals_rec.len(), vcur, "every stream value is accounted for");
    assert_eq!(chals.len(), ccur, "every squeeze is accounted for");
    locs
}

/// Replay the jagged folds' endpoint identities from LOCATED words alone
/// and return the located entries — [`replay_fold_endpoints`]'s sibling.
pub(super) fn replay_jagged_fold_endpoints(
    locs: &[JaggedFoldLoc],
    vals_rec: &[F128],
    chals: &[F128],
) -> Vec<MatrixClaim> {
    let replay_rounds = |target: F128, base: usize, ch0: usize, n: usize| -> (F128, Vec<F128>) {
        let mut run = target;
        let mut rho = Vec::with_capacity(n);
        for j in 0..n {
            let (g1, gi) = (vals_rec[base + 2 * j], vals_rec[base + 2 * j + 1]);
            let r = chals[ch0 + j];
            let q0 = run + g1;
            run = gi * r * r + (q0 + g1 + gi) * r + q0;
            rho.push(r);
        }
        (run, rho)
    };
    let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
    locs.iter()
        .map(|loc| {
            let k = loc.claims.len();
            let lam: Vec<F128> = (0..k).map(|i| chals[loc.lam_ch0 + i]).collect();
            let target_c = loc
                .claims
                .iter()
                .zip(&lam)
                .fold(F128::ZERO, |acc, (cl, &l)| acc + l * vals_rec[cl.val_v]);
            let (run_c, rho_col) = replay_rounds(target_c, loc.col_v, loc.col_ch0, loc.n_col);
            let expect_c =
                loc.claims
                    .iter()
                    .zip(&lam)
                    .enumerate()
                    .fold(F128::ZERO, |acc, (i, (cl, &l))| {
                        let w = (0..loc.n_col).fold(F128::ONE, |w, j| {
                            w * (F128::ONE + vals_rec[cl.col_v + j] + rho_col[j])
                        });
                        acc + l * w * vals_rec[loc.bridge_v + i]
                    });
            assert_eq!(
                run_c, expect_c,
                "jagged col endpoint closes from located words"
            );

            let mus: Vec<F128> = (0..k).map(|i| chals[loc.mu_ch0 + i]).collect();
            let target_r = (0..k).zip(&mus).fold(F128::ZERO, |acc, (i, &m)| {
                acc + m * vals_rec[loc.bridge_v + i]
            });
            let (run_r, rho_row) = replay_rounds(target_r, loc.row_v, loc.row_ch0, loc.k_row);
            let w_mu = loc
                .claims
                .iter()
                .zip(&mus)
                .fold(F128::ZERO, |acc, (cl, &m)| {
                    let rw = if let Some(zeroed) = &cl.masked {
                        let point = &vals_rec[cl.row_pt.0..cl.row_pt.0 + cl.row_pt.1];
                        masked_row_eval(point, zeroed, &rho_row)
                    } else if cl.terms.is_empty() {
                        // The eq product SEEDED by the zero-claim scale.
                        (0..cl.row_pt.1).fold(vals_rec[cl.row_scale_v], |w, j| {
                            w * (F128::ONE + vals_rec[cl.row_pt.0 + j] + rho_row[j])
                        })
                    } else {
                        cl.terms.iter().fold(F128::ZERO, |a, &(cv, addr)| {
                            let e = rho_row.iter().enumerate().fold(F128::ONE, |e, (l, &r)| {
                                e * (F128::ONE + bit((addr >> l) & 1 == 1) + r)
                            });
                            a + vals_rec[cv] * e
                        })
                    };
                    acc + m * rw
                });
            assert_eq!(
                run_r,
                w_mu * vals_rec[loc.out_v],
                "jagged row endpoint closes from located words"
            );
            MatrixClaim {
                row: Weight::eq(rho_row),
                col: Weight::eq(rho_col),
                value: vals_rec[loc.out_v],
            }
        })
        .collect()
}

/// Emit the jagged fold groups in-circuit — [`emit_fold_region`]'s sibling
/// with MergedRoundGate rounds,
/// PrefixGate eq products for the weight evals (a Combo row's ADDRESS bits
/// bake as ow/zw — registry constants, count-independent — with its
/// coefficients as absorbed stream wires), both endpoints as COPY
/// CONSTRAINTS, the entries returned for publishing. The LAST group's
/// output value takes the tail-input treatment ([`emit_fold_region`] must
/// then run with `tail_input_last = false`).
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_jagged_fold_region(
    sb: &mut ShapeBuilder,
    macs: SlotId,
    mrs: SlotId,
    pfslot: SlotId,
    pf_w: usize,
    locs: &[JaggedFoldLoc],
    trace: &FsChainTrace,
    challenge_locs: &[(usize, usize)],
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vmap: &[Option<usize>],
    vals_rec: &[F128],
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
) -> Vec<FoldPub> {
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    let chw = |ch: usize| -> Wire {
        let (fin, offset) = challenge_locs[ch];
        squeeze_word_wire(outs, trace, fin, offset)
    };
    let prefix = |sb: &mut ShapeBuilder, seed: Wire, fs: &[(Wire, Wire)]| -> Wire {
        let mut s = seed;
        for chunk in fs.chunks(pf_w) {
            let mut g_in = vec![s];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(repeat_n(zw, pf_w - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(repeat_n(zw, pf_w - chunk.len()));
            g_in.push(ow);
            s = sb.gate(pfslot, &g_in)[0];
        }
        s
    };
    let mut fold_pubs: Vec<FoldPub> = Vec::new();
    for (fi, loc) in locs.iter().enumerate() {
        let k = loc.claims.len();
        let lam_w: Vec<Wire> = (0..k).map(|i| chw(loc.lam_ch0 + i)).collect();
        let mut run_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            run_w = sb.gate(macs, &[run_w, lam_w[i], wv(cl.val_v)])[0];
        }
        let mut rho_col_w: Vec<Wire> = Vec::with_capacity(loc.n_col);
        for j in 0..loc.n_col {
            let r_w = chw(loc.col_ch0 + j);
            rho_col_w.push(r_w);
            run_w = sb.gate(
                mrs,
                &[run_w, wv(loc.col_v + 2 * j), wv(loc.col_v + 2 * j + 1), r_w],
            )[0];
        }
        let mut exp_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let fs: Vec<(Wire, Wire)> = (0..loc.n_col)
                .map(|j| (wv(cl.col_v + j), rho_col_w[j]))
                .collect();
            let cw = prefix(sb, ow, &fs);
            let t = sb.gate(macs, &[zw, cw, wv(loc.bridge_v + i)])[0];
            exp_w = sb.gate(macs, &[exp_w, lam_w[i], t])[0];
        }
        sb.connect(run_w, exp_w);

        let mu_w: Vec<Wire> = (0..k).map(|i| chw(loc.mu_ch0 + i)).collect();
        let mut run2_w = zw;
        for i in 0..k {
            run2_w = sb.gate(macs, &[run2_w, mu_w[i], wv(loc.bridge_v + i)])[0];
        }
        let mut rho_row_w: Vec<Wire> = Vec::with_capacity(loc.k_row);
        for j in 0..loc.k_row {
            let r_w = chw(loc.row_ch0 + j);
            rho_row_w.push(r_w);
            run2_w = sb.gate(
                mrs,
                &[
                    run2_w,
                    wv(loc.row_v + 2 * j),
                    wv(loc.row_v + 2 * j + 1),
                    r_w,
                ],
            )[0];
        }
        let mut wmu_w = zw;
        for (i, cl) in loc.claims.iter().enumerate() {
            let rw = if let Some(zeroed) = &cl.masked {
                // eq(point, ρ), then each zeroed column's share
                // eq(point, y)·eq(y, ρ) as ONE prefix chain with y's bits
                // baked as ow/zw.
                let pt: Vec<Wire> = (0..cl.row_pt.1).map(|j| wv(cl.row_pt.0 + j)).collect();
                let fs: Vec<(Wire, Wire)> =
                    pt.iter().zip(&rho_row_w).map(|(&p, &r)| (p, r)).collect();
                let mut acc = prefix(sb, ow, &fs);
                for &y in zeroed {
                    let bit = |l: usize| if (y >> l) & 1 == 1 { ow } else { zw };
                    let fs: Vec<(Wire, Wire)> = pt
                        .iter()
                        .enumerate()
                        .map(|(l, &p)| (p, bit(l)))
                        .chain(rho_row_w.iter().enumerate().map(|(l, &r)| (r, bit(l))))
                        .collect();
                    let share = prefix(sb, ow, &fs);
                    acc = sb.gate(macs, &[acc, share, ow])[0];
                }
                acc
            } else if cl.terms.is_empty() {
                let fs: Vec<(Wire, Wire)> = (0..cl.row_pt.1)
                    .map(|j| (wv(cl.row_pt.0 + j), rho_row_w[j]))
                    .collect();
                // SEEDED by the claim's own scale wire (the zero-claim
                // form) rather than the constant 1 — free, the prefix
                // chain already takes a seed.
                prefix(sb, wv(cl.row_scale_v), &fs)
            } else {
                let mut acc = zw;
                for &(cv, addr) in &cl.terms {
                    let fs: Vec<(Wire, Wire)> = rho_row_w
                        .iter()
                        .enumerate()
                        .map(|(l, &r)| (r, if (addr >> l) & 1 == 1 { ow } else { zw }))
                        .collect();
                    let e = prefix(sb, ow, &fs);
                    acc = sb.gate(macs, &[acc, wv(cv), e])[0];
                }
                acc
            };
            wmu_w = sb.gate(macs, &[wmu_w, mu_w[i], rw])[0];
        }
        let value = if fi + 1 == locs.len() {
            vals.push(vals_rec[loc.out_v]);
            sb.input()
        } else {
            wv(loc.out_v)
        };
        let rhs_w = sb.gate(macs, &[zw, wmu_w, value])[0];
        sb.connect(run2_w, rhs_w);
        fold_pubs.push(FoldPub {
            live: ow,
            rho_col: rho_col_w,
            rho_row: rho_row_w,
            value,
        });
    }
    fold_pubs
}

/// Walk the published jagged entries from `at` — [`check_fold_publics`]'s
/// sibling (no boundary publics: jagged lows are trivially 1). The jagged
/// group is KEYED, so under the spine layout every entry leads with its
/// key; `keyed = false` is the ACC_CHAIN layout, which the lane's
/// single-key registry role leaves as it was. Returns the rebuilt entries,
/// their keys, and the offset just past the last one.
pub(super) fn check_jagged_fold_publics(
    public: &[F128],
    at: usize,
    locs: &[JaggedFoldLoc],
    keyed: bool,
) -> (Vec<MatrixClaim>, Vec<[F128; 2]>, usize) {
    let mut p = at;
    let mut out = Vec::with_capacity(locs.len());
    let mut keys = Vec::new();
    for loc in locs {
        let (k, c) = read_acc_entry(public, &mut p, keyed, loc.n_col, loc.k_row);
        if keyed {
            keys.push(k);
        }
        out.push(c);
    }
    (out, keys, p)
}

/// The checker rejects every surface its two items cover: a non-native
/// nonce word (high half set, or low half past the scan budget), a
/// tampered point coordinate, and a tampered row low. The honest block
/// passes and consumes exactly 72 words. (Seed and decode tampers are
/// IN-CIRCUIT since phase D — `emit_ag_point_binding` — and are outside
/// this checker's coverage by design.)
#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use flock_core::{
        genus95_curve_code::evaluation_point_from_nonce_pow, zerocheck::ag_skip::R1_POW_BITS,
    };
    use flock_merkle::HashKind;

    use crate::tower::{
        fold_region::{
            F128, R1_FUSED_ATTEMPT_BUDGET, base_evaluation_functional, check_ag_skip_publics,
        },
        fs_chain::ag_seed_bytes,
    };

    #[test]
    fn ag_skip_publics_checker_rejects_tampers() {
        let (s0, s1) = (F128::new(0xA6, 0x51), F128::new(0x1F, 0xB3));
        let seed = ag_seed_bytes(s0, s1);
        let (nonce, pt) = (0..R1_FUSED_ATTEMPT_BUDGET)
            .find_map(|n| {
                evaluation_point_from_nonce_pow(&seed, n, HashKind::Blake3, R1_POW_BITS)
                    .map(|p| (n, p))
            })
            .expect("a valid fused nonce exists in the budget");
        let bf = base_evaluation_functional(&pt).expect("the functional exists at a sampled point");
        let base = 3usize;
        let mut public = vec![F128::ZERO; base];
        public.extend([s0, s1, F128::new(u64::from(nonce), 0)]);
        public.extend([pt.x, pt.y, pt.z1, pt.z2, pt.z3]);
        public.extend((0..64).map(|j| bf[j]));
        assert_eq!(
            check_ag_skip_publics(&public, base, Some(R1_POW_BITS)),
            72,
            "the honest block passes"
        );

        let rejects = |mutate: &dyn Fn(&mut [F128])| -> bool {
            let mut bad = public.clone();
            mutate(&mut bad);
            catch_unwind(AssertUnwindSafe(|| {
                check_ag_skip_publics(&bad, base, Some(R1_POW_BITS))
            }))
            .is_err()
        };
        assert!(
            rejects(&|p| p[base + 2] = F128::new(u64::from(nonce), 1)),
            "a set nonce high half is rejected"
        );
        assert!(
            rejects(&|p| p[base + 2] = F128::new(u64::from(R1_FUSED_ATTEMPT_BUDGET), 0)),
            "an out-of-budget nonce is rejected"
        );
        assert!(
            rejects(&|p| p[base + 4] += F128::ONE),
            "tampered point coord"
        );
        assert!(
            rejects(&|p| p[base + 8 + 17] += F128::ONE),
            "tampered row low"
        );
    }
}
