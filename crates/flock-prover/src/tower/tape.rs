use std::mem::take;

use flock_transcript::transcript_record::{TranscriptOp, TranscriptOp as Op};

use crate::{r1cs_hashes::fs_chain::FsChainTrace, tower::Wire};

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

    // The opening protocol begins at its domain label. Cap byte lengths are
    // not structural delimiters: under strict profiles a later recursive cap
    // can have the same length as L0, so a last-matching-length heuristic can
    // enter the tape at the wrong level.
    let label = ops
        .iter()
        .position(
            |o| matches!(o, Op::Label(l) if l.as_slice() == b"flock-ligerito-basis-f256-split-v0"),
        )
        .expect("Ligerito opening label");
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
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-element-union-zc-v0") {
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
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-element-union-lc-v0"),
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
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-merged-open-v1") {
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
            if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0") {
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
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1") {
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
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0"),
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
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-pcs-packed-direct-v0") {
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
    let inner_pd = inner_pd.expect("the inner ligerito intake");
    let mp = mp.expect("the multipoint region");
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
