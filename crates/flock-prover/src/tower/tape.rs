use std::mem::take;

use flock_transcript::transcript_record::{TranscriptOp, TranscriptOp as Op};

use crate::{
    r1cs_hashes::fs_chain::FsChainTrace,
    tower::{
        Wire,
        walker_common::{
            LBL_AFFINE_OPEN, LBL_ELEMENT_LC, LBL_ELEMENT_ZC, LBL_FROBENIUS, LBL_INNER_PD,
            LBL_MERGED_OPEN, LBL_MULTIPOINT, LBL_OPEN_BASIS, LBL_RING_SWITCH,
        },
    },
};

/// One packed-direct claim on the tape: its absorbed VALUE and gamma. The
/// POINT is not on the stream since merged-open v1 — it is transcript-derived
/// (gathers: the GKR's ρ_row + constant address bits; element claims: the
/// region PIOP's own challenges + the frozen prefix), and consumers rebuild
/// it from those wires and the verifier's native claims.
pub(super) struct PdRec {
    pub(super) val_v: usize,
    pub(super) fin: usize,
    pub(super) ch: usize,
    /// Word offset inside the vector squeeze shared by all batch coefficients.
    pub(super) squeeze_offset: usize,
}

#[inline]
pub(super) fn squeeze_word_wire(
    outs: &[Vec<Wire>],
    trace: &FsChainTrace,
    fin: usize,
    offset: usize,
) -> Wire {
    let (row, word) = trace.squeeze_words[fin][offset];
    outs[row][word]
}

/// One merged W-round: the (G(1), G(inf)) value index and the rho squeeze.
#[derive(Clone)]
pub(super) struct RoundRec {
    pub(super) g_v: usize,
    pub(super) fin: usize,
    pub(super) ch: usize,
}

/// The inner ligerito intake's single claim: q_eval's value index + gamma'.
pub(super) struct InnerPd {
    pub(super) q_v: usize,
    pub(super) fin: usize,
    pub(super) ch: usize,
}

/// The element PIOP on the tape: tau, the zerocheck rounds, (ea, eb, ec),
/// alpha, and the lincheck rounds.
pub(super) struct PiopRec {
    pub(super) tau_fin: usize,
    pub(super) tau_ch: usize,
    pub(super) tau_len: usize,
    pub(super) zc_rounds: Vec<RoundRec>,
    pub(super) eab_v: usize,
    pub(super) alpha_fin: usize,
    pub(super) alpha_ch: usize,
    pub(super) lc_rounds: Vec<RoundRec>,
}

/// One OOD claim on the tape: where its `z` squeezed, its `y`/intro-msg
/// values, and its beta.
pub(super) struct OodRec {
    pub(super) z_fin: usize,
    pub(super) z_ch: usize,
    pub(super) z_len: usize,
    pub(super) y_v: usize,
    pub(super) intro_v: usize,
    pub(super) beta_fin: usize,
    pub(super) beta_ch: usize,
}

/// The multipoint region of the merged open, located on the tape:
/// the group values' absorb, the batching gamma, the two-product sumcheck
/// rounds, and the anchor assist's `v` + rounds. For a pure-element inner
/// (R = 0) the RS values are absent and the sumcheck is the single
/// untwisted product — see docs/multipoint-twisted-assist.tex.
pub(super) struct MpRec {
    /// Value indices of the P group values `B_k` (stream-wireable).
    pub(super) val_vs: Vec<usize>,
    /// The batching gamma squeeze: `(fin, ch)`.
    pub(super) gamma_fin: usize,
    pub(super) gamma_ch: usize,
    /// The m two-product sumcheck rounds.
    pub(super) rounds: Vec<RoundRec>,
    /// The anchor's claimed twisted evaluation `v` (value index).
    pub(super) anchor_v: usize,
    /// The anchor's `2(m + 1)` rounds.
    pub(super) anchor_rounds: Vec<RoundRec>,
}

/// One open-phase level, located on a recorded op tape. `*_fin` are finalize
/// ordinals (indices into `FsChainTrace::squeezes`); `*_ch` index into
/// `RecordingChallenger::challenges()`.
pub(super) struct OpenLevel {
    /// Additional L0 OOD claims batched into the initial sumcheck before its
    /// first message. Nonempty only on level 0.
    pub(super) initial_ood: Vec<InitialOodRec>,
    pub(super) fold_fins: Vec<usize>,
    pub(super) fold_chs: Vec<usize>,
    /// Value index of each fold round's message `u_0` (`u_2` is `+1`).
    pub(super) fold_msg_vs: Vec<usize>,
    /// OOD claims folded before this level's queries.
    pub(super) ood: Vec<OodRec>,
    /// The level's intro message value idx (unused for the final level).
    pub(super) intro_v: usize,
    /// The intro/final beta: `(fin, ch)`.
    pub(super) beta_fin: usize,
    pub(super) beta_ch: usize,
    pub(super) q_fin: usize,
    pub(super) q_ch: usize,
    pub(super) q_count: usize,
    pub(super) a_fin: usize,
    pub(super) a_ch: usize,
    pub(super) a_count: usize,
}

/// An L0 OOD claim has no separate intro quadratic: its equality basis and
/// target are combined before the initial sumcheck message is emitted.
pub(super) struct InitialOodRec {
    pub(super) z_fin: usize,
    pub(super) z_ch: usize,
    pub(super) z_len: usize,
    pub(super) y_v: usize,
    pub(super) beta_fin: usize,
    pub(super) beta_ch: usize,
}

/// A cursor over a recorded op tape, counting finalizations (`fin`),
/// challenge words (`ch`) and observed values (`v`) as it walks.
struct Cur<'a> {
    ops: &'a [Op],
    i: usize,
    fin: usize,
    ch: usize,
    v: usize,
}
impl Cur<'_> {
    fn bump(&mut self) {
        let op = &self.ops[self.i];
        if op.finalizes() {
            self.fin += 1;
        }
        match op {
            Op::SqueezeScalar => self.ch += 1,
            Op::SqueezeSlice(n) => self.ch += n,
            Op::ObserveScalar => self.v += 1,
            Op::ObserveSlice(n) => self.v += n,
            _ => {}
        }
        self.i += 1;
    }
    fn expect_obs_scalar(&mut self) {
        assert!(
            matches!(self.ops[self.i], Op::ObserveScalar),
            "op {}: expected ObserveScalar, got {:?}",
            self.i,
            self.ops[self.i]
        );
        self.bump();
    }

    fn expect_obs_f256(&mut self) {
        assert!(
            matches!(self.ops[self.i], Op::ObserveSlice(2)),
            "op {}: expected ObserveSlice(2), got {:?}",
            self.i,
            self.ops[self.i]
        );
        self.bump();
    }

    /// PoW finalizes the transcript and absorbs a nonce but creates no
    /// field challenge or scalar message.  PIOP locators call this before
    /// every protected squeeze; the generic circuit relation constrains
    /// the skipped operation separately.
    fn skip_pows(&mut self) {
        while matches!(self.ops[self.i], Op::Pow { .. }) {
            self.bump();
        }
    }
}

/// What precedes the inner opening on a merged-open tape: the element
/// PIOP, the packed-direct coefficients, the merged W-rounds, the
/// multipoint region, and (split ladder only) the inner packed-direct
/// intake. Returned with the cursor parked on the opening's L0 cap absorb.
pub(super) struct OpenPrefix {
    pub(super) piop: Option<PiopRec>,
    pub(super) gammas: Vec<PdRec>,
    pub(super) rounds: Vec<RoundRec>,
    pub(super) mp: MpRec,
    pub(super) inner_pd: Option<InnerPd>,
    /// Value index of the opening's target (`ObserveScalar` after its label).
    #[allow(dead_code)] // the affine claim fold starts from it (next step)
    pub(super) target_v: usize,
}

/// Walk a merged-open tape up to the inner opening, whose domain label is
/// `open_label`; the cursor ends ON the opening's L0 cap absorb.
fn walk_open_prefix<'a>(
    ops: &'a [TranscriptOp],
    cap0_bytes: usize,
    open_label: &[u8],
) -> (Cur<'a>, OpenPrefix) {
    // The opening protocol begins at its domain label. Cap byte lengths are
    // not structural delimiters: under strict profiles a later recursive cap
    // can have the same length as L0, so a last-matching-length heuristic can
    // enter the tape at the wrong level.
    let label = ops
        .iter()
        .position(|o| matches!(o, Op::Label(l) if l.as_slice() == open_label))
        .expect("the inner opening's label");
    assert!(
        matches!(ops.get(label + 1), Some(Op::ObserveScalar)),
        "opening target"
    );
    let start = label + 2;
    assert!(
        matches!(ops.get(start), Some(Op::ObserveBytes(n)) if *n == cap0_bytes),
        "L0 cap"
    );
    // The merged intake runs every ring switch, absorbs every packed-direct
    // value, then protects and samples ONE coefficient vector in claim order
    // (RS first, PD second).  Consequently every PD coefficient below names
    // both a challenge ordinal and a word offset in that shared finalization.
    let mut cur = Cur {
        ops,
        i: 0,
        fin: 0,
        ch: 0,
        v: 0,
    };
    let mut gammas: Vec<PdRec> = Vec::new();
    let mut rounds: Vec<RoundRec> = Vec::new();
    let mut mp: Option<MpRec> = None;
    let mut inner_pd: Option<InnerPd> = None;
    let mut piop: Option<PiopRec> = None;
    let mut in_pd = false;
    let mut intake_rs = 0usize;
    let mut intake_pd_vals: Vec<usize> = Vec::new();
    while cur.i < start {
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == LBL_ELEMENT_ZC) {
            cur.bump();
            cur.skip_pows();
            let (tau_fin, tau_ch, tau_len) = match ops[cur.i] {
                Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
                ref o => panic!("tau, got {o:?}"),
            };
            cur.bump();
            let mut zc_rounds = Vec::with_capacity(tau_len);
            for _ in 0..tau_len {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "zc rho");
                zc_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            let eab_v = cur.v;
            cur.expect_obs_scalar(); // ea
            cur.expect_obs_scalar(); // eb
            cur.expect_obs_scalar(); // ec
            assert!(
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == LBL_ELEMENT_LC),
                "lc label"
            );
            cur.bump();
            cur.skip_pows();
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "alpha");
            let (alpha_fin, alpha_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut lc_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "lc rho");
                lc_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            piop = Some(PiopRec {
                tau_fin,
                tau_ch,
                tau_len,
                zc_rounds,
                eab_v,
                alpha_fin,
                alpha_ch,
                lc_rounds,
            });
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == LBL_MERGED_OPEN) {
            in_pd = true;
            intake_rs = 0;
            intake_pd_vals.clear();
            cur.bump();
            continue;
        }
        if in_pd {
            // Ring-switched claims front the intake on boolean-bearing
            // tapes: [label, s_hat_v slice, r_dprime slice] each, then the
            // bare gamma squeezes.
            if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == LBL_RING_SWITCH) {
                intake_rs += 1;
                cur.bump(); // label
                cur.bump(); // s_hat_v slice
                cur.skip_pows();
                cur.bump(); // r_dprime slice
                continue;
            }
            if matches!(ops[cur.i], Op::Pow { .. }) {
                // A ring-switch or packed-direct batch-coefficient witness.
                cur.bump();
                continue;
            }
            if matches!(ops[cur.i], Op::ObserveScalar) {
                intake_pd_vals.push(cur.v);
                cur.expect_obs_scalar();
                continue;
            }
            if let Op::SqueezeSlice(n) = ops[cur.i] {
                assert_eq!(
                    n,
                    intake_rs + intake_pd_vals.len(),
                    "one coefficient per merged claim"
                );
                gammas.extend(intake_pd_vals.iter().enumerate().map(|(j, &val_v)| PdRec {
                    val_v,
                    fin: cur.fin,
                    ch: cur.ch + intake_rs + j,
                    squeeze_offset: intake_rs + j,
                }));
                cur.bump();
            } else {
                panic!("merged batching vector, got {:?}", ops[cur.i]);
            }
            in_pd = false;
            // The merged W-rounds follow the intake immediately: one
            // [ObserveScalar x2, SqueezeScalar] triplet per dense variable,
            // running until the multipoint label — count-free, so boolean
            // tapes (no packed-direct claims) parse identically.
            while matches!(ops[cur.i], Op::ObserveScalar)
                && matches!(ops[cur.i + 1], Op::ObserveScalar)
            {
                let mut squeeze_i = cur.i + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
                rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == LBL_MULTIPOINT) {
            // The multipoint region: P group-value absorbs, the batching
            // gamma, m two-product rounds, then the anchor's label + v +
            // 2(m + 1) rounds. Each loop terminates on the next label /
            // squeeze, so a shape change fails loudly here.
            cur.bump();
            let mut val_vs = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                val_vs.push(cur.v);
                cur.bump();
            }
            cur.skip_pows();
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "multipoint gamma");
            let (gamma_fin, gamma_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut mp_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "multipoint round");
                mp_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            assert!(
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == LBL_FROBENIUS),
                "op {}: expected the anchor label, got {:?}",
                cur.i,
                ops[cur.i]
            );
            cur.bump();
            let anchor_v = cur.v;
            cur.expect_obs_scalar();
            let mut anchor_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                cur.skip_pows();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "anchor round");
                anchor_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            mp = Some(MpRec {
                val_vs,
                gamma_fin,
                gamma_ch,
                rounds: mp_rounds,
                anchor_v,
                anchor_rounds,
            });
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == LBL_INNER_PD) {
            cur.bump();
            let q_v = cur.v;
            cur.expect_obs_scalar(); // q_eval
            assert!(
                matches!(ops[cur.i], Op::SqueezeSlice(1)),
                "inner gamma vector"
            );
            inner_pd = Some(InnerPd {
                q_v,
                fin: cur.fin,
                ch: cur.ch,
            });
            cur.bump();
            continue;
        }
        cur.bump();
    }
    let mp = mp.expect("the multipoint region");
    // The loop bumped the target observe at `label + 1` generically.
    let target_v = cur.v - 1;
    debug_assert_eq!(cur.i, start);
    (
        cur,
        OpenPrefix {
            piop,
            gammas,
            rounds,
            mp,
            inner_pd,
            target_v,
        },
    )
}

/// Walk the recorded transcript ops and locate every open-phase squeeze the
/// circuit needs. The walk MIRRORS the succinct verifier's structure (folds,
/// cap absorbs, OOD groups, PoW, queries, alpha, beta per level), asserting
/// each op kind — a config change that moves the shape fails here, loudly,
/// not as a wrong wire.
#[allow(clippy::type_complexity)]
pub(super) fn parse_open_levels(
    ops: &[TranscriptOp],
    cap0_bytes: usize,
    r: usize,
) -> (
    usize,
    Option<PiopRec>,
    Vec<PdRec>,
    Vec<RoundRec>,
    MpRec,
    InnerPd,
    usize,
    Vec<OpenLevel>,
) {
    let (mut cur, pre) = walk_open_prefix(ops, cap0_bytes, LBL_OPEN_BASIS);
    let OpenPrefix {
        piop,
        gammas,
        rounds,
        mp,
        inner_pd,
        ..
    } = pre;
    let inner_pd = inner_pd.expect("the inner ligerito intake");
    cur.bump(); // the open-phase initial cap absorb
    let mut initial_ood = Vec::new();
    while matches!(cur.ops[cur.i], Op::SqueezeSlice(_)) {
        let z_len = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => n,
            _ => unreachable!(),
        };
        let (z_fin, z_ch) = (cur.fin, cur.ch);
        cur.bump();
        let y_v = cur.v;
        cur.expect_obs_scalar();
        cur.skip_pows();
        assert!(
            matches!(cur.ops[cur.i], Op::SqueezeScalar),
            "L0 OOD beta at op {}: context {:?}",
            cur.i,
            &cur.ops[cur.i.saturating_sub(4)..(cur.i + 4).min(cur.ops.len())]
        );
        initial_ood.push(InitialOodRec {
            z_fin,
            z_ch,
            z_len,
            y_v,
            beta_fin: cur.fin,
            beta_ch: cur.ch,
        });
        cur.bump();
    }
    let start_v = cur.v;
    cur.expect_obs_f256(); // sumcheck start msg u_0
    cur.expect_obs_f256(); // ... u_2

    let mut levels = Vec::new();
    let mut yr_v = 0usize;
    for li in 0..=r {
        // Fold batch: one double-width squeeze and two F256 message absorbs
        // per round. Fold grinding is zero in the F256 protocol.
        let mut fold_fins = Vec::new();
        let mut fold_chs = Vec::new();
        let mut fold_msg_vs = Vec::new();
        loop {
            match cur.ops[cur.i] {
                Op::Pow { .. } if matches!(cur.ops.get(cur.i + 1), Some(Op::SqueezeSlice(2))) => {
                    cur.bump()
                }
                Op::SqueezeSlice(2) => {
                    fold_fins.push(cur.fin);
                    fold_chs.push(cur.ch);
                    cur.bump();
                    fold_msg_vs.push(cur.v);
                    cur.expect_obs_f256();
                    cur.expect_obs_f256();
                }
                _ => break,
            }
        }
        let mut ood = Vec::new();
        if li < r {
            // The NEXT commitment's cap, then its OOD groups.
            assert!(
                matches!(cur.ops[cur.i], Op::ObserveBytes(_)),
                "op {}: expected the next cap absorb, got {:?}",
                cur.i,
                cur.ops[cur.i]
            );
            cur.bump();
            while !matches!(cur.ops[cur.i], Op::Pow { .. }) {
                let z_len = match cur.ops[cur.i] {
                    Op::SqueezeSlice(n) => n,
                    ref o => panic!("OOD z, got {o:?}"),
                };
                let (z_fin, z_ch) = (cur.fin, cur.ch);
                cur.bump();
                let y_v = cur.v;
                cur.expect_obs_scalar(); // base-field y
                let intro_v = cur.v;
                cur.expect_obs_f256(); // intro u_0
                cur.expect_obs_f256(); // intro u_2
                cur.skip_pows();
                assert!(matches!(cur.ops[cur.i], Op::SqueezeScalar), "OOD beta");
                ood.push(OodRec {
                    z_fin,
                    z_ch,
                    z_len,
                    y_v,
                    intro_v,
                    beta_fin: cur.fin,
                    beta_ch: cur.ch,
                });
                cur.bump();
            }
        } else {
            // Final level: the yr observes.
            yr_v = cur.v;
            while matches!(cur.ops[cur.i], Op::ObserveScalar) {
                cur.expect_obs_scalar();
            }
        }
        assert!(
            matches!(cur.ops[cur.i], Op::Pow { .. }),
            "op {}: expected query-grinding Pow, got {:?}",
            cur.i,
            cur.ops[cur.i]
        );
        cur.bump();
        let (q_fin, q_ch, q_count) = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
            ref o => panic!("op {}: expected queries squeeze, got {o:?}", cur.i),
        };
        cur.bump();
        cur.skip_pows();
        let (a_fin, a_ch, a_count) = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
            ref o => panic!("op {}: expected alpha squeeze, got {o:?}", cur.i),
        };
        cur.bump();
        let intro_v = cur.v;
        if li < r {
            cur.expect_obs_f256(); // intro u_0
            cur.expect_obs_f256(); // intro u_2
        }
        cur.skip_pows();
        assert!(matches!(cur.ops[cur.i], Op::SqueezeScalar), "beta");
        let (beta_fin, beta_ch) = (cur.fin, cur.ch);
        cur.bump();
        levels.push(OpenLevel {
            initial_ood: if li == 0 {
                take(&mut initial_ood)
            } else {
                Vec::new()
            },
            fold_fins,
            fold_chs,
            fold_msg_vs,
            ood,
            intro_v,
            beta_fin,
            beta_ch,
            q_fin,
            q_ch,
            q_count,
            a_fin,
            a_ch,
            a_count,
        });
    }
    (start_v, piop, gammas, rounds, mp, inner_pd, yr_v, levels)
}

/// One affine fold round on the tape: the 5-word grid absorb, then the
/// fold-grinding PoW fused with the 3-word `(alpha, beta, r)` squeeze.
#[allow(dead_code)] // consumed by the affine child walker (next step)
pub(super) struct AffineRoundRec {
    /// Value index of the grid's first word (`G00`; `G11` is implied).
    pub(super) grid_v: usize,
    pub(super) fin: usize,
    /// Challenge index of `alpha`; `beta` and `r` follow.
    pub(super) ch: usize,
}

/// One affine OOD claim: a fresh point squeeze (all `c` coordinates on a
/// recursive level; only the `initial_k` row coordinates on a lane-major
/// L0, whose columns are the opening point's), its `y`, and its `beta`.
#[allow(dead_code)] // consumed by the affine child walker (next step)
pub(super) struct AffineOodRec {
    pub(super) z_fin: usize,
    pub(super) z_ch: usize,
    pub(super) z_len: usize,
    pub(super) y_v: usize,
    pub(super) beta_fin: usize,
    pub(super) beta_ch: usize,
}

/// One affine level on the tape, in transcript order. Non-final levels
/// carry the overflow, the next cap, OOD claims, the queries, the
/// consistency `alpha` vector and the consistency `beta`; the final level
/// carries `yr` and the queries.
#[allow(dead_code)] // consumed by the affine child walker (next step)
pub(super) struct AffineLevel {
    pub(super) rounds: Vec<AffineRoundRec>,
    /// Non-final: value index of the `k` overflow words.
    pub(super) overflow_v: Option<usize>,
    pub(super) ood: Vec<AffineOodRec>,
    /// Final: value index of the `2^c + k` yr words.
    pub(super) yr_v: Option<usize>,
    pub(super) q_fin: usize,
    pub(super) q_ch: usize,
    pub(super) q_count: usize,
    /// Non-final: the consistency alpha squeeze `(fin, ch, count)`.
    pub(super) alpha: Option<(usize, usize, usize)>,
    /// Non-final: the consistency beta squeeze `(fin, ch)`.
    pub(super) beta: Option<(usize, usize)>,
}

/// The affine inner opening on a merged-open tape: what precedes it, the
/// L0 OOD claims, and one [`AffineLevel`] per level.
#[allow(dead_code)] // consumed by the affine child walker (next step)
pub(super) struct AffineOpenTape {
    pub(super) prefix: OpenPrefix,
    pub(super) initial_ood: Vec<AffineOodRec>,
    pub(super) levels: Vec<AffineLevel>,
}

/// Locate every affine-opening absorb and squeeze on a recorded verify
/// tape, mirroring [`flock_core::pcs::ligerito::affine::verify_affine_l0`]
/// op by op. `ks` are the levels' fold counts (`initial_k`, then the
/// recursive ks) and `yr_len` is `2^c + k` of the final level; both are
/// config authority, so the walk counts rather than infers, and a shape
/// change fails here loudly instead of as a wrong wire.
#[allow(dead_code)] // consumed by the affine child walker (next step)
pub(super) fn parse_affine_open(
    ops: &[TranscriptOp],
    cap0_bytes: usize,
    ks: &[usize],
    yr_len: usize,
) -> AffineOpenTape {
    let (mut cur, prefix) = walk_open_prefix(ops, cap0_bytes, LBL_AFFINE_OPEN);
    assert!(
        prefix.inner_pd.is_none(),
        "the affine opening has no packed-direct intake"
    );
    cur.bump(); // the L0 cap absorb
    fn expect_slice(cur: &mut Cur<'_>, n: usize, what: &str) -> usize {
        assert!(
            matches!(cur.ops[cur.i], Op::ObserveSlice(m) if m == n),
            "op {}: expected {what} ObserveSlice({n}), got {:?}",
            cur.i,
            cur.ops[cur.i]
        );
        let v = cur.v;
        cur.bump();
        v
    }
    /// A PoW fused with the squeeze after it; returns that squeeze's
    /// `(fin, ch, count)`.
    fn pow_squeeze(cur: &mut Cur<'_>, what: &str) -> (usize, usize, usize) {
        assert!(
            matches!(cur.ops[cur.i], Op::Pow { .. }),
            "op {}: expected the {what} PoW, got {:?}",
            cur.i,
            cur.ops[cur.i]
        );
        cur.bump();
        let n = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => n,
            Op::SqueezeScalar => 1,
            ref o => panic!("op {}: expected the {what} squeeze, got {o:?}", cur.i),
        };
        let at = (cur.fin, cur.ch, n);
        cur.bump();
        at
    }
    fn ood(cur: &mut Cur<'_>) -> AffineOodRec {
        let z_len = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => n,
            ref o => panic!("op {}: expected an OOD point, got {o:?}", cur.i),
        };
        let (z_fin, z_ch) = (cur.fin, cur.ch);
        cur.bump();
        let y_v = cur.v;
        cur.expect_obs_scalar();
        let (beta_fin, beta_ch, n) = pow_squeeze(cur, "OOD beta");
        assert_eq!(n, 1, "one OOD beta");
        AffineOodRec {
            z_fin,
            z_ch,
            z_len,
            y_v,
            beta_fin,
            beta_ch,
        }
    }

    let mut initial_ood = Vec::new();
    while matches!(cur.ops[cur.i], Op::SqueezeSlice(_)) {
        initial_ood.push(ood(&mut cur));
    }
    let r = ks.len() - 1;
    let mut levels = Vec::with_capacity(ks.len());
    for (li, &k) in ks.iter().enumerate() {
        let rounds = (0..k)
            .map(|_| {
                let grid_v = expect_slice(&mut cur, 5, "fold grid");
                let (fin, ch, n) = pow_squeeze(&mut cur, "fold");
                assert_eq!(n, 3, "(alpha, beta, r) per fold round");
                AffineRoundRec { grid_v, fin, ch }
            })
            .collect();
        if li == r {
            let yr_v = expect_slice(&mut cur, yr_len, "yr");
            let (q_fin, q_ch, q_count) = pow_squeeze(&mut cur, "query");
            levels.push(AffineLevel {
                rounds,
                overflow_v: None,
                ood: Vec::new(),
                yr_v: Some(yr_v),
                q_fin,
                q_ch,
                q_count,
                alpha: None,
                beta: None,
            });
            break;
        }
        let overflow_v = expect_slice(&mut cur, k, "overflow");
        assert!(
            matches!(cur.ops[cur.i], Op::ObserveBytes(_)),
            "op {}: expected the next cap absorb, got {:?}",
            cur.i,
            cur.ops[cur.i]
        );
        cur.bump();
        let mut oods = Vec::new();
        while matches!(cur.ops[cur.i], Op::SqueezeSlice(_)) {
            oods.push(ood(&mut cur));
        }
        let (q_fin, q_ch, q_count) = pow_squeeze(&mut cur, "query");
        let alpha = pow_squeeze(&mut cur, "consistency alpha");
        let (b_fin, b_ch, n) = pow_squeeze(&mut cur, "consistency beta");
        assert_eq!(n, 1, "one consistency beta");
        levels.push(AffineLevel {
            rounds,
            overflow_v: Some(overflow_v),
            ood: oods,
            yr_v: None,
            q_fin,
            q_ch,
            q_count,
            alpha: Some(alpha),
            beta: Some((b_fin, b_ch)),
        });
    }
    AffineOpenTape {
        prefix,
        initial_ood,
        levels,
    }
}

#[cfg(test)]
mod tests {
    use flock_core::{
        pcs::{
            MergedInnerOpen,
            ligerito::{LigeritoProfile, prover_config_for},
        },
        test_rng::Rng,
    };
    use flock_transcript::transcript_record::RecordingChallenger;

    use super::parse_affine_open;
    use crate::{
        challenger::FsChallenger,
        r1cs_hashes::blake3::{Blake3Setup, Compression},
    };

    /// The affine walk over a real recorded verify tape (a `fast-affine`
    /// BLAKE3 union at m = 24): every located absorb holds the proof's own
    /// words, every squeeze has the configured width, and the walk ends on
    /// the tape's last op.
    #[test]
    fn affine_open_tape_matches_the_proof() {
        const D: &[u8] = b"tower-affine-tape";
        let n_blocks = 1 << 10;
        let setup = Blake3Setup::with_profile(n_blocks, LigeritoProfile::FastAffine);
        let mut rng = Rng::new(0x7A9E_AFF1);
        let blocks: Vec<Compression> = (0..n_blocks)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, rng.next_u32() as u64, 64u32, 11u32)
            })
            .collect();
        let (proof, commitment, _) =
            setup.prove_fast(&blocks, &mut FsChallenger::with_chained_blake3(D));
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(D));
        setup
            .verify(&commitment, &proof, &mut rec)
            .expect("the recorded verify accepts");
        let MergedInnerOpen::Affine(aff) = &proof.pcs_open.inner else {
            panic!("fast-affine takes the affine inner open")
        };
        let p = &setup.pcs_params;
        let cfg = prover_config_for(p.m - 7, p.log_batch_size, p.profile).expect("affine ladder");
        let mut ks = vec![cfg.initial_k];
        ks.extend(&cfg.recursive_ks);
        let c_final = *cfg.recursive_log_msg_cols.last().unwrap();
        let yr_len = (1usize << c_final) + ks[ks.len() - 1];

        let shape = rec.shape();
        let ops = crate::tower::fs_chain::flatten_ops(shape.ops());
        let t = parse_affine_open(&ops, 32 * aff.initial_cap.len(), &ks, yr_len);
        let (vals, chals) = (rec.values(), rec.challenges());

        assert_eq!(t.levels.len(), ks.len());
        assert_eq!(t.initial_ood.len(), cfg.ood_samples[0]);
        let mut oods = t.initial_ood.iter().collect::<Vec<_>>();
        assert!(t.initial_ood.iter().all(|o| o.z_len == cfg.initial_k));
        let mut grid = 0usize;
        for (li, lvl) in t.levels.iter().enumerate() {
            assert_eq!(lvl.rounds.len(), ks[li]);
            for rd in &lvl.rounds {
                assert_eq!(vals[rd.grid_v..rd.grid_v + 5], aff.grids[grid]);
                grid += 1;
                assert!(rd.ch + 3 <= chals.len());
            }
            assert_eq!(lvl.q_count, cfg.queries[li]);
            if let Some(ov) = lvl.overflow_v {
                assert_eq!(vals[ov..ov + ks[li]], aff.overflows[li][..]);
                // The OOD point is on this level's folded message: its own c.
                let c = if li == 0 {
                    cfg.initial_log_msg_cols
                } else {
                    cfg.recursive_log_msg_cols[li - 1]
                };
                assert!(lvl.ood.iter().all(|o| o.z_len == c));
                assert_eq!(lvl.ood.len(), cfg.ood_samples[li + 1]);
                let (_, _, n) = lvl.alpha.expect("consistency alpha");
                assert_eq!(
                    n,
                    cfg.queries[li].next_power_of_two().trailing_zeros() as usize
                );
                assert!(lvl.beta.is_some());
            } else {
                let yr = lvl.yr_v.expect("final yr");
                assert_eq!(vals[yr..yr + yr_len], aff.yr[..]);
                assert!(lvl.alpha.is_none() && lvl.beta.is_none());
            }
            oods.extend(&lvl.ood);
        }
        assert_eq!(grid, aff.grids.len());
        let ys: Vec<_> = oods.iter().map(|o| vals[o.y_v]).collect();
        assert_eq!(ys, aff.ood_values);
        // The final query squeeze is the last op of the tape.
        let last = t.levels.last().unwrap();
        assert_eq!(last.q_ch + last.q_count, chals.len());
        assert!(!t.prefix.mp.rounds.is_empty() && !t.prefix.rounds.is_empty());
    }

    /// The split walk, after the prefix walk moved into [`super::walk_open_prefix`],
    /// still parses a chained-BLAKE3 `fast` union tape to one level per open
    /// level.
    #[test]
    fn split_open_tape_still_parses() {
        const D: &[u8] = b"tower-affine-tape";
        let n_blocks = 1 << 10;
        let setup = Blake3Setup::with_profile(n_blocks, LigeritoProfile::Fast);
        let mut rng = Rng::new(0x7A9E_AFF1);
        let blocks: Vec<Compression> = (0..n_blocks)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, rng.next_u32() as u64, 64u32, 11u32)
            })
            .collect();
        let (proof, commitment, _) =
            setup.prove_fast(&blocks, &mut FsChallenger::with_chained_blake3(D));
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(D));
        setup.verify(&commitment, &proof, &mut rec).unwrap();
        let lig = &proof.pcs_open.inner.split().ligerito;
        let shape = rec.shape();
        let r = lig.recursive_caps.len();
        let (.., levels) = super::parse_open_levels(
            &crate::tower::fs_chain::flatten_ops(shape.ops()),
            32 * lig.initial_cap.len(),
            r,
        );
        assert_eq!(levels.len(), r + 1);
    }
}
