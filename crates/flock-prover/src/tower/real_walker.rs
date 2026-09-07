use std::{array::from_fn, collections::BTreeMap, env::var, iter::repeat_n, sync::OnceLock};

use flock_core::{
    circuit::{SigmaAssertion, builder::SlotId},
    element_r1cs::union::{ElementAssertion, region_slots},
    lincheck::{MatrixAssertion, SkipPoint, build_eq_table},
    matrix_fold::{JaggedAssertion, JaggedRowWeight},
    pcs::{
        jagged::{
            JaggedParams, STATE_INITIAL, STATE_SUCCESS, assist_boundaries,
            assist_sparse_transitions, frob_inv,
        },
        ring_switch::{
            build_fold_byte_table, inner_product, linearized_coefficients, moore_inverse,
            tensor_algebra_transpose,
        },
    },
    product_gkr::LiveMask,
    proof::UnionClassClaims,
    zerocheck::univariate_skip::build_eq,
};
use flock_field::QUADRATIC_NONRESIDUE;
use flock_transcript::transcript_record::{
    RecordingChallenger, Stream, StreamWord, TranscriptOp as Op, TranscriptShape,
};

use crate::{
    r1cs_hashes::fs_chain::{FsChainTrace, IV},
    tower::{
        ChildSlots, CollapsedSlots, F128, F256, FsChallenger, GkrRec, HashKind, InnerPd,
        LeafEvalGate, LeafEvalGate256, LeafOuter, Lvl, MergedChain, MixedProof, MpRec, OpenLevel,
        PdRec, PiopRec, RoundRec, SLOT_WORDS, ShapeBuilder, UnionInstance, Wire, ZskipTapeRec,
        ZskipWires, ag_seed_bytes, bytes_payload_mask, cap_payloads, cap_wires,
        check_residual_publics, circuit_structure_claim_wires, cw, decode_ag_point,
        emit_boolean_reported_check, emit_element_reported_check, emit_fs_chain, emit_mac256,
        emit_pow_checks, emit_publics_hash, emit_query_phase, emit_recombination,
        emit_residual_region, emit_spine256, flatten_ops, leaf_boolean_lcs, level_geometry,
        level_sources, observed_f256, outer_union, pack8, parse_open_levels, payload_words,
        pin_recombination, query_phase_b3_rows, replay_ligerito_spine256, squeeze_word_wire,
        strat_scheds, walker_common,
        walker_common::{
            LBL_AG_R1_NONCE, LBL_AG_R1_POINT, LBL_AG_SKIP, LBL_ELEMENT_LC, LBL_ELEMENT_ZC,
            LBL_FROBENIUS, LBL_LINCHECK, LBL_MERGED_OPEN, LBL_MULTIPOINT, LBL_PRODUCT_GKR,
            LBL_RING_SWITCH, LBL_ZEROCHECK, RS_SHAT_WORDS, TailEntry, Z_PARTIAL_WORDS,
            publish_tail, zskip_wires,
        },
    },
};

/// One recorded REAL-child verification (the leaf outer as inner), parsed:
/// the tape, each region, and the native values used by the checker.
pub(super) struct RealTape<'p> {
    pub(super) lo: &'p LeafOuter,
    // the recorded tape
    pub(super) vals_rec: Vec<F128>,
    pub(super) chals: Vec<F128>,
    /// Which byte payloads stay PUBLIC under the witness/public split.
    pub_payloads: Vec<bool>,
    /// Per level, the absorbed cap's payload index ([`cap_payloads`]).
    pub(super) cap_pays: Vec<usize>,
    // chain materials
    pub(super) trace: FsChainTrace,
    pub(super) stream: Stream,
    pub(super) bytes: Vec<u8>,
    /// The fork's four cross-link wires ([`MergedChain::cross`]).
    pub(super) cross: Vec<Option<(usize, usize)>>,
    pub(super) b3_rows: usize,
    pub(super) spread_w: usize,
    // located regions
    pub(super) gkr: GkrRec,
    piop_i: PiopRec,
    start_v_i: usize,
    gammas_i: Vec<PdRec>,
    pub(super) w_rounds: Vec<RoundRec>,
    pub(super) w_resid: Vec<RoundRec>,
    mp_i: MpRec,
    inner_pd_i: InnerPd,
    yr_v_i: usize,
    pub(super) yr_len: usize,
    pub(super) levels: Vec<OpenLevel>,
    pub(super) lvl_src: Vec<(&'p [[u8; 32]], &'p Vec<Vec<F128>>, &'p Vec<[u8; 32]>)>,
    pub(super) geo: Vec<Lvl>,
    pub(super) native_sums: Vec<F256>,
    /// The grinding ops: (fin ordinal, payload ordinal, bits).
    pub(super) pows: Vec<(usize, usize, u32)>,
    n_gather: usize,
    /// The child cell space's public-slot count — the recombination's tail.
    pub(super) n_pub_slots_c: usize,
    // the boolean PIOP's round ordinals ((ch, fin) pairs) + surfaces
    pub(super) zc_rounds_b: Vec<(usize, usize)>,
    pub(super) outer_b: (usize, usize),
    pub(super) bl_alpha: (usize, usize),
    /// The const-pin beta squeezes: (ch, fin) per pinned boolean type.
    pub(super) betas_b: Vec<(usize, usize)>,
    /// The zerocheck finals' value ordinal (v_a at, v_b at +1).
    pub(super) zc_finals_v: usize,
    /// Per pinned type, eq_prefix_sum(x_outer, n_t) — the count-derived
    /// beta term, bound through the digest-keyed circuit-structure table.
    pub(super) eps_n: Vec<F128>,
    /// (g_v, ch, fin) per boolean lc round — messages feed the in-circuit
    /// lincheck replay.
    pub(super) lc_rounds_b: Vec<(usize, usize, usize)>,
    /// The z_skip transcript surface, by flavor — see [`ZskipTapeRec`].
    pub(super) zskip: ZskipTapeRec,
    pub(super) zp_v: usize,
    /// The rs regions: (s_hat_v ordinal, r_dprime fin, r_dprime ch), plus
    /// the two rs gammas' `(fin, word offset)` and challenge ordinals — the
    /// family-H circuit. Both coefficients share one vector squeeze.
    pub(super) rs_recs: Vec<(usize, usize, usize)>,
    pub(super) rs_gam_fins: Vec<(usize, usize)>,
    // native references + replicas
    pub(super) mat_assert: MatrixAssertion,
    pub(super) el_assert: ElementAssertion,
    pub(super) sigma_native: SigmaAssertion,
    /// Which pd claim carries z_eval (order varies per tape).
    pub(super) z_ix: usize,
    pub(super) el_g0: Vec<F128>,
    pub(super) el_run_n: F128,
    pub(super) a_sum_n: F128,
    pub(super) b_sum_n: F128,
    pub(super) native_target: F128,
    pub(super) native_running: F128,
    pub(super) t_final_n: F256,
    pub(super) anc_end_n: F128,
    pub(super) mid_n: F128,
    pub(super) live_n: F128,
    pub(super) mu_i: usize,
    // anchor-expect geometry — statement constants of the real inner
    pub(super) n_log_i: usize,
    pub(super) k_cols_i: usize,
    pub(super) m_mp2: usize,
    pub(super) bounds_i: Vec<(u64, u64, u32)>,
    pub(super) x_ab_n: Vec<F128>,
    pub(super) x_c_n: Vec<F128>,
    pub(super) groups_ix: Vec<Vec<usize>>,
    /// Derived pd claim points (merged-open v1), pinned order
    /// [element c, element lc, gathers in cell-slot order].
    pub(super) pd_pts: Vec<Vec<F128>>,
    /// The deferred verify's jagged-layout export (the count win) — the
    /// independent reference for the W-value publics, tied to the native
    /// expect replica in the constructor.
    pub(super) jag: JaggedAssertion,
}

impl<'p> RealTape<'p> {
    pub(super) fn new(lo: &'p LeafOuter, domain: &'static [u8]) -> Self {
        let union_i = outer_union(&lo.shape.registry, lo.shape.counts.clone());
        let lcs = leaf_boolean_lcs(lo);
        // ONE recorded DEFERRED verify serves both needs: it is
        // transcript-identical to the plain verify for honest proofs (so
        // the tape is unchanged), it skips the sigma discharge the plain
        // pass paid, and its exported assertions ARE the method-note
        // references (verifier-exported over formulas-written-twice).
        // This is also exactly what a production node runs per child —
        // the tape cost halved when the second pass dissolved.
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
        let (mat_assert, el_assert, sigma_native, jag_assert, claims) = {
            let (claims, work, sigma) = lo
                .proof
                .verify_circuit_deferred(
                    &union_i,
                    &lo.shape.circuit,
                    &lo.public,
                    &lcs,
                    &lo.commitment,
                    &lo.pcs,
                    &mut rec,
                )
                .expect("the deferred verify accepts the leaf outer");
            assert!(
                claims.boolean.is_some(),
                "boolean claims from the real inner"
            );
            assert!(
                claims.element.is_some(),
                "element claims from the real inner"
            );
            (
                work.boolean.expect("a boolean PIOP ran"),
                work.element.expect("an element PIOP ran"),
                sigma,
                work.jagged,
                claims,
            )
        };
        let t_shape = rec.shape();
        let chals: Vec<F128> = rec.challenges().to_vec();
        let vals_rec: Vec<F128> = rec.values().to_vec();
        let ops = flatten_ops(t_shape.ops());
        let mut pub_payloads = bytes_payload_mask(&ops);
        // Prefix sums over the op tape — the locate walks below call these
        // per feature and per ROUND (437 rounds at node scale), so a
        // rescan-per-call is quadratic in practice. One pass, O(1) lookups.
        let (pre_v, pre_c, pre_f) = {
            let mut pre_v = Vec::with_capacity(ops.len() + 1);
            let mut pre_c = Vec::with_capacity(ops.len() + 1);
            let mut pre_f = Vec::with_capacity(ops.len() + 1);
            let (mut v, mut c, mut f) = (0usize, 0usize, 0usize);
            pre_v.push(0);
            pre_c.push(0);
            pre_f.push(0);
            for op in &ops {
                match op {
                    Op::SqueezeScalar => c += 1,
                    Op::SqueezeSlice(n) => c += n,
                    Op::ObserveScalar => v += 1,
                    Op::ObserveSlice(n) => v += n,
                    _ => {}
                }
                if op.finalizes() {
                    f += 1;
                }
                pre_v.push(v);
                pre_c.push(c);
                pre_f.push(f);
            }
            (pre_v, pre_c, pre_f)
        };
        let vc_at = |end: usize| -> (usize, usize) { (pre_v[end], pre_c[end]) };
        let fin_at = |end: usize| pre_f[end];

        // The region order, by label — identical to the minimal mixed inner's.
        let find = |label: &[u8]| -> Vec<usize> {
            ops.iter()
                .enumerate()
                .filter_map(|(i, op)| match op {
                    Op::Label(l) if l.as_slice() == label => Some(i),
                    _ => None,
                })
                .collect()
        };
        // The boolean zerocheck region's anchor is flavor-specific; the
        // OTHER flavor's label must be absent (one boolean zerocheck).
        let zc_l = match &lo.proof {
            MixedProof::Rs(_) => {
                assert!(
                    find(LBL_AG_SKIP).is_empty(),
                    "an RS tape carries no AG-skip region"
                );
                find(LBL_ZEROCHECK)
            }
            MixedProof::Ag(_) => {
                assert!(
                    find(LBL_ZEROCHECK).is_empty(),
                    "an AG tape carries no RS zerocheck region"
                );
                find(LBL_AG_SKIP)
            }
        };
        let lc_l = find(LBL_LINCHECK);
        let elzc_l = find(LBL_ELEMENT_ZC);
        let el_l = find(LBL_ELEMENT_LC);
        let gkr_l = find(LBL_PRODUCT_GKR);
        let mo_l = find(LBL_MERGED_OPEN);
        let rs_l = find(LBL_RING_SWITCH);
        let mp_l = find(LBL_MULTIPOINT);
        let fa_l = find(LBL_FROBENIUS);
        assert_eq!(
            (
                zc_l.len(),
                lc_l.len(),
                elzc_l.len(),
                el_l.len(),
                gkr_l.len()
            ),
            (1, 1, 1, 1, 1),
            "one region each"
        );
        assert_eq!(
            (mo_l.len(), rs_l.len(), mp_l.len(), fa_l.len()),
            (1, 2, 1, 1)
        );
        // THE FORKED ORDER. The wiring argument runs on its own chain, and
        // the flattened view splices it in at the fork point — so the GKR
        // region now PRECEDES the boolean PIOP instead of following the
        // element's. Everything downstream of the merge is unmoved.
        assert!(
            gkr_l[0] < zc_l[0],
            "the wiring fork precedes the boolean PIOP"
        );
        assert!(zc_l[0] < lc_l[0] && lc_l[0] < elzc_l[0] && elzc_l[0] < el_l[0]);
        assert!(el_l[0] < mo_l[0]);
        assert!(mo_l[0] < rs_l[0] && rs_l[1] < mp_l[0] && mp_l[0] < fa_l[0]);

        // parse_open_levels + level_geometry — the assembly's own walkers,
        // unchanged, on the real-inner tape.
        let lig = &lo.proof.pcs_open().inner.ligerito;
        let r = lig.recursive_caps.len();
        let lvl_src = level_sources(lig);
        let (start_v_i, piop_i, gammas_i, w_rounds, mp_i, inner_pd_i, yr_v_i, levels) =
            parse_open_levels(&ops, 32 * lig.initial_cap.len(), r);
        assert_eq!(levels.len(), r + 1);
        let piop_i = piop_i.expect("the real inner HAS an element PIOP");
        assert!(!piop_i.zc_rounds.is_empty() && !piop_i.lc_rounds.is_empty());
        let n_gather = lo.proof.wiring().gather.len();
        assert_eq!(
            gammas_i.len(),
            2 + n_gather,
            "pd claims = the element (c, lc) pair + the outer's gathers"
        );
        assert_eq!(w_rounds.len(), lo.pcs.m - 7, "W spans the dense domain");
        let (geo, native_sums) = level_geometry(
            &levels,
            &lvl_src,
            &chals,
            HashKind::Blake3,
            &strat_scheds(&lo.pcs),
        );
        assert!(
            geo[0].row_words <= geo[0].lanes,
            "committed width fits the fold"
        );

        // The R=2 + P schedule replays to the anchor's claimed v.
        let n_p = lo.proof.pcs_open().frobenius.group_values.len();
        assert!(n_p > 0, "the mixed inner groups its pd claims");
        assert_eq!(
            mp_i.val_vs.len(),
            2 * RS_SHAT_WORDS + n_p,
            "T0 spans the RS dual values then the P group values"
        );
        let gamma_mp = chals[mp_i.gamma_ch];
        let mut pw = F128::ONE;
        let mut t0 = F128::ZERO;
        for &vi in &mp_i.val_vs {
            t0 += pw * vals_rec[vi];
            pw *= gamma_mp;
        }
        let mut tm = t0;
        for rr in &mp_i.rounds {
            let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
            let rch = chals[rr.ch];
            let g0 = tm + g1;
            tm = g0 + (g1 + g0 + gi) * rch + gi * rch * rch;
        }
        assert_eq!(
            tm, vals_rec[mp_i.anchor_v],
            "T0 folds to the anchor's claimed v"
        );

        let gkr_rec = walk_wiring_gkr(lo, &ops, gkr_l[0], &vals_rec, &chals, &vc_at, &fin_at);
        // ROUND 2: the H(publics) region's rows — a chunk chain per 1 KiB
        // leaf of the child's public segment plus the left-fold parents.
        let h_rows = lo.public.len().div_ceil(4) + 2 * lo.public.len().div_ceil(64);

        let ChainMaterials {
            stream,
            bytes,
            trace,
            cross,
            b3_rows,
            spread_w,
            cap_pays,
            pows,
        } = parse_chain_materials(
            &t_shape,
            domain,
            rec.values(),
            rec.payloads(),
            &ops,
            &chals,
            h_rows,
            &geo,
            &lvl_src,
            &mut pub_payloads,
        );

        let (rs_recs2, rs_gam_fin2, native_target, native_running) =
            parse_rs_regions_and_two_halves(
                lo,
                &ops,
                rs_l[0],
                &gammas_i,
                &w_rounds,
                &inner_pd_i,
                &vals_rec,
                &chals,
                &vc_at,
                &fin_at,
            );

        // ---- the spine's native quad replay ----
        let t_final_n = replay_ligerito_spine256(
            &levels,
            &vals_rec,
            &chals,
            start_v_i,
            chals[inner_pd_i.ch] * vals_rec[inner_pd_i.q_v],
            &native_sums,
        );

        let (yr_len, w_resid) =
            walker_common::parse_residual_rotation(&lo.proof, &geo, &levels, &w_rounds);

        let (a_sum_n, b_sum_n, el_g0, el_run_n, z_ix) =
            element_piop_natives(&union_i, &piop_i, &el_assert, &gammas_i, &vals_rec, &chals);

        let anc_end_n = walker_common::parse_anchor_native_endpoint(&vals_rec, &chals, &mp_i);

        let (mu_i, mid_n, live_n) =
            walker_common::parse_gkr_input_check_advice(&lo.shape.circuit, &gkr_rec.r_pt);

        let AnchorBooleanRec {
            m_mp2,
            n_log_i,
            k_cols_i,
            n_pub_slots_c,
            bounds_i,
            zc_rounds_b,
            zskip,
            outer_b,
            bl_alpha,
            betas_b,
            zc_finals_v,
            lc_rounds_b,
            zp_v,
            eps_n,
            x_ab_n,
            x_c_n,
            pd_pts_n,
            groups_ix,
        } = parse_anchor_boolean_replica(
            lo,
            &union_i,
            &claims,
            &jag_assert,
            &mat_assert,
            &el_assert,
            &piop_i,
            &mp_i,
            &gkr_rec,
            &gammas_i,
            &w_rounds,
            n_gather,
            n_p,
            zc_l[0],
            lc_l[0],
            anc_end_n,
            &ops,
            &vals_rec,
            &chals,
            &vc_at,
            &fin_at,
        );

        RealTape {
            lo,
            vals_rec,
            chals,
            pub_payloads,
            cap_pays,
            trace,
            stream,
            bytes,
            cross,
            b3_rows,
            spread_w,
            gkr: gkr_rec,
            piop_i,
            start_v_i,
            gammas_i,
            w_rounds,
            w_resid,
            mp_i,
            inner_pd_i,
            yr_v_i,
            yr_len,
            levels,
            lvl_src,
            geo,
            native_sums,
            pows,
            n_gather,
            n_pub_slots_c,
            zc_rounds_b,
            outer_b,
            bl_alpha,
            betas_b,
            zc_finals_v,
            eps_n,
            lc_rounds_b,
            zskip,
            zp_v,
            rs_recs: rs_recs2,
            rs_gam_fins: rs_gam_fin2,
            mat_assert,
            el_assert,
            sigma_native,
            z_ix,
            el_g0,
            el_run_n,
            a_sum_n,
            b_sum_n,
            native_target,
            native_running,
            t_final_n,
            anc_end_n,
            mid_n,
            live_n,
            mu_i,
            n_log_i,
            k_cols_i,
            m_mp2,
            bounds_i,
            x_ab_n,
            x_c_n,
            groups_ix,
            pd_pts: pd_pts_n,
            jag: jag_assert,
        }
    }
}

/// the wiring GKR walk
///
/// Records every ordinal the transcription wires against and replays
/// the whole layer recursion natively in lockstep, input checks
/// included — the rhs consuming the DEFERRED s_sigma from the proof.
fn walk_wiring_gkr(
    lo: &LeafOuter,
    ops: &[Op],
    gkr_l0: usize,
    vals_rec: &[F128],
    chals: &[F128],
    vc_at: &impl Fn(usize) -> (usize, usize),
    fin_at: &impl Fn(usize) -> usize,
) -> GkrRec {
    walker_common::walk_wiring_gkr_core(
        ops,
        vals_rec,
        chals,
        &lo.proof.wiring().gkr,
        gkr_l0,
        lo.shape.circuit.cells().mu(),
        &lo.shape.circuit.live_mask(),
        vc_at,
        fin_at,
    )
}

/// What [`parse_chain_materials`] hands back to the constructor.
struct ChainMaterials {
    stream: Stream,
    bytes: Vec<u8>,
    trace: FsChainTrace,
    cross: Vec<Option<(usize, usize)>>,
    b3_rows: usize,
    spread_w: usize,
    cap_pays: Vec<usize>,
    pows: Vec<(usize, usize, u32)>,
}

/// the chain materials
///
/// The transcript is FORKED (the wiring runs on its own chain);
/// `merge_chain` splices the child's rows in at the fork point and
/// hands back one linear numbering plus the four cross-link wires.
#[allow(clippy::too_many_arguments)]
fn parse_chain_materials(
    t_shape: &TranscriptShape,
    domain: &[u8],
    values: &[F128],
    payloads: &[Vec<u8>],
    ops: &[Op],
    chals: &[F128],
    h_rows: usize,
    geo: &[Lvl],
    lvl_src: &[(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)],
    pub_payloads: &mut [bool],
) -> ChainMaterials {
    let MergedChain {
        stream,
        bytes,
        trace,
        cross,
        ..
    } = walker_common::merge_and_replay_chain(t_shape, domain, values, payloads, ops, chals);
    let b3_rows = trace.rows.len() + h_rows + query_phase_b3_rows(geo);
    if var("B3_CENSUS").is_ok() {
        let parents = trace.block_offsets.iter().filter(|o| o.is_none()).count();
        let blocks = trace.rows.len() - parents;
        let mut pow_by_bits = BTreeMap::<u32, usize>::new();
        for op in ops {
            if let Op::Pow { bits } = op {
                *pow_by_bits.entry(*bits).or_default() += 1;
            }
        }
        eprintln!(
            "  [b3 census] chain {} (data blocks {} | parent/fork {}; absorbed {} B, {} squeezes) | H(publics) {} | openings+caps {} = {}",
            trace.rows.len(),
            blocks,
            parents,
            bytes.len(),
            trace.squeezes.len(),
            h_rows,
            b3_rows - trace.rows.len() - h_rows,
            b3_rows
        );
        eprintln!(
            "  [pow census] {} checks by bits {:?}",
            pow_by_bits.values().sum::<usize>(),
            pow_by_bits
        );
        walker_common::census_levels_and_chain_rows(ops, t_shape, domain, geo, &trace);
    }
    let spread_w = geo.iter().map(|g| g.depth).max().unwrap().max(1);
    // Recursive caps are PROOF BODY — the in-circuit cap trees bind them
    // (chain + root connects, nothing checker-read); only the L0 cap —
    // the commitment — stays a statement public.
    let cap_pays = cap_payloads(&stream, &bytes, lvl_src);
    for &p in &cap_pays[1..] {
        pub_payloads[p] = false;
    }

    // Locate the PoW operations.
    let pows: Vec<(usize, usize, u32)> = {
        let mut out = Vec::new();
        let (mut fin, mut pay) = (0usize, 0usize);
        for op in ops {
            if let Op::Pow { bits } = op {
                out.push((fin, pay, *bits));
            }
            if op.finalizes() {
                fin += 1;
            }
            if op.carries_payload() {
                pay += 1;
            }
        }
        out
    };
    assert!(!pows.is_empty(), "the Fast profile grinds");
    ChainMaterials {
        stream,
        bytes,
        trace,
        cross,
        b3_rows,
        spread_w,
        cap_pays,
        pows,
    }
}

/// the rs×2 regions + the two-halves target, natively
#[allow(clippy::too_many_arguments)]
fn parse_rs_regions_and_two_halves(
    lo: &LeafOuter,
    ops: &[Op],
    rs_l0: usize,
    gammas_i: &[PdRec],
    w_rounds: &[RoundRec],
    inner_pd_i: &InnerPd,
    vals_rec: &[F128],
    chals: &[F128],
    vc_at: &impl Fn(usize) -> (usize, usize),
    fin_at: &impl Fn(usize) -> usize,
) -> (Vec<(usize, usize, usize)>, Vec<(usize, usize)>, F128, F128) {
    let (rs_recs2, rs_gam_ch2, rs_gam_fin2) = {
        let mut i2 = rs_l0;
        // (s_hat_v ordinal, r_dprime fin, r_dprime ch) per region.
        let mut recs: Vec<(usize, usize, usize)> = Vec::new();
        for k in 0..2 {
            assert!(
                matches!(&ops[i2], Op::Label(l) if l.as_slice() == LBL_RING_SWITCH),
                "rs region {k}"
            );
            i2 += 1;
            assert!(
                matches!(ops[i2], Op::ObserveSlice(RS_SHAT_WORDS)),
                "s_hat_v slice"
            );
            let (sv, _) = vc_at(i2);
            assert_eq!(
                &vals_rec[sv..sv + RS_SHAT_WORDS],
                &lo.proof.pcs_open().ring_switches[k].s_hat_v[..],
                "s_hat_v {k} on the stream"
            );
            i2 += 1;
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            assert!(matches!(ops[i2], Op::SqueezeSlice(7)), "r_dprime");
            recs.push((sv, fin_at(i2), vc_at(i2).1));
            i2 += 1;
        }
        // All PD values follow the RS regions. One PoW then protects one
        // vector squeeze in claim order: RS[0..2], PD[0..P].
        for pd in gammas_i {
            assert!(
                matches!(ops[i2], Op::ObserveScalar),
                "pd value before batch vector"
            );
            assert_eq!(vc_at(i2).0, pd.val_v, "pd intake order");
            i2 += 1;
        }
        while matches!(ops[i2], Op::Pow { .. }) {
            i2 += 1;
        }
        assert!(
            matches!(ops[i2], Op::SqueezeSlice(n) if n == 2 + gammas_i.len()),
            "mixed coefficient vector"
        );
        let base_ch = vc_at(i2).1;
        let fin = fin_at(i2);
        let gchs = vec![base_ch, base_ch + 1];
        let gfins = vec![(fin, 0), (fin, 1)];
        (recs, gchs, gfins)
    };
    // Native differential replay of the two-halves target and V. The
    // recursive circuit independently computes the RS, packed-direct,
    // and group parts below; these values no longer discharge soundness.
    let (native_target, native_running) = {
        let gs: Vec<F128> = rs_gam_ch2.iter().map(|&ch| chals[ch]).collect();
        let mut rs_half = F128::ZERO;
        let mut coeffs: Vec<Vec<F128>> = Vec::new();
        for (k, &(sv, _, rc)) in rs_recs2.iter().enumerate() {
            let shv = &vals_rec[sv..sv + RS_SHAT_WORDS];
            let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
            let eq = build_eq(&rdp);
            rs_half += gs[k] * inner_product(&tensor_algebra_transpose(shv), &eq);
            let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
            coeffs.push(linearized_coefficients(&build_fold_byte_table(&scaled)));
        }
        let mut target = rs_half;
        for pd in gammas_i {
            target += chals[pd.ch] * vals_rec[pd.val_v];
        }
        let mut running = target;
        for rr in w_rounds {
            let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
            let rc = chals[rr.ch];
            let g0 = running + g1;
            running = g0 + (g1 + g0 + gi) * rc + gi * rc * rc;
        }
        let fro = &lo.proof.pcs_open().frobenius;
        let mut vrs = F128::ZERO;
        for (k, cs) in coeffs.iter().enumerate() {
            for (j, &cj) in cs.iter().enumerate() {
                if cj.is_zero() {
                    continue;
                }
                let mut x = fro.values[k][j];
                for _ in 0..j {
                    x = x * x;
                }
                vrs += cj * x;
            }
        }
        let mut big_v = vrs;
        for &v in &fro.group_values {
            big_v += v;
        }
        assert_eq!(
            running,
            vals_rec[inner_pd_i.q_v] * big_v,
            "the R=2 + P merged boundary replays at real-inner scale"
        );
        (target, running)
    };
    (rs_recs2, rs_gam_fin2, native_target, native_running)
}

/// the element PIOP's natives: the GENERAL strip + g0 chain
fn element_piop_natives(
    union_i: &UnionInstance<'_>,
    piop_i: &PiopRec,
    el_assert: &ElementAssertion,
    gammas_i: &[PdRec],
    vals_rec: &[F128],
    chals: &[F128],
) -> (F128, F128, Vec<F128>, F128, usize) {
    assert_eq!(
        piop_i.zc_rounds.len(),
        piop_i.tau_len,
        "one element zc round per tau coordinate"
    );
    assert_eq!(
        el_assert.alpha, chals[piop_i.alpha_ch],
        "the located alpha is the assertion's"
    );
    let (a_sum_n, b_sum_n) = {
        let slots_el = region_slots(union_i);
        let nu_i = union_i.n_log();
        let mut a_sum = F128::ZERO;
        let mut b_sum = F128::ZERO;
        for s in &slots_el {
            let kappa = s.ty.kappa();
            let eq_con = build_eq(&el_assert.r_con[..kappa]);
            let prefix = s.layout.region_prefix(nu_i);
            let mut w = F128::ONE;
            for (j, &x) in el_assert.r_con[kappa..].iter().enumerate() {
                w *= if (prefix >> j) & 1 == 1 {
                    x
                } else {
                    F128::ONE + x
                };
            }
            let dot = |c: &[F128]| -> F128 {
                eq_con
                    .iter()
                    .zip(c)
                    .fold(F128::ZERO, |acc, (e, v)| acc + *e * *v)
            };
            a_sum += w * dot(s.ty.a_const());
            b_sum += w * dot(s.ty.b_const());
        }
        (a_sum, b_sum)
    };
    let mut el_g0: Vec<F128> = Vec::new();
    let el_run_n = {
        let mut run = F128::ZERO;
        for (k, rr) in piop_i.zc_rounds.iter().enumerate() {
            let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
            let t2 = chals[piop_i.tau_ch + k];
            let rho = chals[rr.ch];
            let g0 = (run + t2 * g1) * (F128::ONE + t2).inv();
            el_g0.push(g0);
            run = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        run
    };
    // The element c claim's position among the pd claims varies with
    // the tape — identify it by the assertion's own value.
    let z_ix = gammas_i
        .iter()
        .position(|pd| vals_rec[pd.val_v] == el_assert.z_eval)
        .expect("z_eval is one of the absorbed pd values");
    (a_sum_n, b_sum_n, el_g0, el_run_n, z_ix)
}

/// What [`parse_anchor_boolean_replica`] hands back — field-for-field the
/// anchor-expect/boolean half of [`RealTape`].
struct AnchorBooleanRec {
    m_mp2: usize,
    n_log_i: usize,
    k_cols_i: usize,
    n_pub_slots_c: usize,
    bounds_i: Vec<(u64, u64, u32)>,
    zc_rounds_b: Vec<(usize, usize)>,
    zskip: ZskipTapeRec,
    outer_b: (usize, usize),
    bl_alpha: (usize, usize),
    betas_b: Vec<(usize, usize)>,
    zc_finals_v: usize,
    lc_rounds_b: Vec<(usize, usize, usize)>,
    zp_v: usize,
    eps_n: Vec<F128>,
    x_ab_n: Vec<F128>,
    x_c_n: Vec<F128>,
    pd_pts_n: Vec<Vec<F128>>,
    groups_ix: Vec<Vec<usize>>,
}

/// the anchor-expect geometry + boolean locate + replica
#[allow(clippy::too_many_arguments)]
fn parse_anchor_boolean_replica(
    lo: &LeafOuter,
    union_i: &UnionInstance<'_>,
    claims: &UnionClassClaims,
    jag_assert: &JaggedAssertion,
    mat_assert: &MatrixAssertion,
    el_assert: &ElementAssertion,
    piop_i: &PiopRec,
    mp_i: &MpRec,
    gkr_rec: &GkrRec,
    gammas_i: &[PdRec],
    w_rounds: &[RoundRec],
    n_gather: usize,
    n_p: usize,
    zc_l0: usize,
    lc_l0: usize,
    anc_end_n: F128,
    ops: &[Op],
    vals_rec: &[F128],
    chals: &[F128],
    vc_at: &impl Fn(usize) -> (usize, usize),
    fin_at: &impl Fn(usize) -> usize,
) -> AnchorBooleanRec {
    let m_mp2 = mp_i.rounds.len();
    assert_eq!(
        mp_i.anchor_rounds.len(),
        2 * (m_mp2 + 1),
        "sigma spans the anchor layers"
    );
    assert_eq!(w_rounds.len(), m_mp2, "merged rho spans the dense domain");
    let n_log_i = union_i.n_log();
    let params_i = JaggedParams::from_heights(&union_i.jagged_heights(), n_log_i, m_mp2);
    let k_cols_i = params_i.k;
    // Recompute the recombination and f == g from located words.
    let n_pub_slots_c = pin_recombination(
        lo.shape.circuit.cells(),
        n_log_i,
        &lo.public,
        &lo.proof.wiring().gather,
        gammas_i,
        2,
        vals_rec,
        &gkr_rec.r_pt,
        gkr_rec.fgs_v,
    );
    let bounds_i = assist_boundaries(&params_i);
    let n_runs = bounds_i.len();
    let run_y0: Vec<usize> = bounds_i
        .iter()
        .scan(0usize, |y, &(_, _, len)| {
            let s = *y;
            *y += len as usize;
            Some(s)
        })
        .collect();
    let comp_ix = (0..n_runs)
        .max_by_key(|&r3| bounds_i[r3].2)
        .expect("at least one run");
    // The boolean PIOP's round ordinals, located with fins — plus the
    // MatrixAssertion surfaces the 2→1 merge connects to (z_skip's
    // squeeze, z_partial's slice).
    // Byte-payload ordinal of the op at `end` (ObserveBytes and Pow
    // share the payload counter — see [`bytes_payload_mask`]).
    let payload_at =
        |end: usize| -> usize { ops[..end].iter().filter(|o| o.carries_payload()).count() };
    let (
        zc_rounds_b,
        zskip,
        (outer_ch_b, outer_fin_b, _outer_len),
        bl_alpha,
        betas_b,
        zc_finals_v,
        lc_rounds_b,
        zp_v,
    ) = {
        let mut i2 = zc_l0 + 1;
        // A grinded zerocheck inserts one `Pow` immediately before each
        // protected squeeze.  The generic PoW locator below emits and
        // binds its BLAKE3 predicate for *every* such op; this PIOP
        // locator only needs to step past them before naming the squeeze
        // wires which feed the arithmetic replay.
        while matches!(ops[i2], Op::Pow { .. }) {
            i2 += 1;
        }
        // The flavored region head — the ChildTape walk's twin: RS has
        // two squeeze slices + 64/64 round-1 + the fused z_skip
        // squeeze; AG has ONE r_outer slice, 158/64 round-1, and r₁'s
        // 5-op seed/nonce surface (the point has no transcript word).
        let (outer, zskip) = match &lo.proof {
            MixedProof::Rs(_) => {
                assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_skip slice");
                i2 += 1;
                let outer_len = match ops[i2] {
                    Op::SqueezeSlice(n) => n,
                    ref o => panic!("r_outer slice, got {o:?}"),
                };
                let outer = (vc_at(i2).1, fin_at(i2), outer_len);
                i2 += 1;
                assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_ab");
                i2 += 1;
                assert!(matches!(ops[i2], Op::ObserveSlice(64)), "round1_c");
                i2 += 1;
                while matches!(ops[i2], Op::Pow { .. }) {
                    i2 += 1;
                }
                assert!(matches!(ops[i2], Op::SqueezeScalar), "z_skip");
                let zskip = ZskipTapeRec::Rs {
                    ch: vc_at(i2).1,
                    fin: fin_at(i2),
                };
                i2 += 1;
                (outer, zskip)
            }
            MixedProof::Ag(_) => {
                let outer_len = match ops[i2] {
                    Op::SqueezeSlice(n) => n,
                    ref o => panic!("ag r_outer slice, got {o:?}"),
                };
                let outer = (vc_at(i2).1, fin_at(i2), outer_len);
                i2 += 1;
                assert!(matches!(ops[i2], Op::ObserveSlice(158)), "ag round1_ab");
                i2 += 1;
                assert!(matches!(ops[i2], Op::ObserveSlice(64)), "ag round1_c");
                i2 += 1;
                assert!(
                    matches!(&ops[i2], Op::Label(l) if l.as_slice() == LBL_AG_R1_POINT),
                    "ag r1 seed label"
                );
                i2 += 1;
                assert!(matches!(ops[i2], Op::SqueezeScalar), "ag r1 seed s0");
                let seed_ch = vc_at(i2).1;
                let seed_fin0 = fin_at(i2);
                i2 += 1;
                assert!(matches!(ops[i2], Op::SqueezeScalar), "ag r1 seed s1");
                assert_eq!(vc_at(i2).1, seed_ch + 1, "seed words are adjacent");
                let seed_fin1 = fin_at(i2);
                i2 += 1;
                assert!(
                    matches!(&ops[i2], Op::Label(l) if l.as_slice() == LBL_AG_R1_NONCE),
                    "ag r1 nonce label"
                );
                i2 += 1;
                assert!(matches!(ops[i2], Op::ObserveBytes(4)), "ag r1 nonce bytes");
                let zskip = ZskipTapeRec::Ag {
                    seed_ch,
                    seed_fins: [seed_fin0, seed_fin1],
                    nonce_payload: payload_at(i2),
                };
                i2 += 1;
                (outer, zskip)
            }
        };
        let mut zc_r: Vec<(usize, usize)> = Vec::new();
        while matches!(ops[i2], Op::ObserveScalar) && matches!(ops[i2 + 1], Op::ObserveScalar) {
            let mut squeeze_i = i2 + 2;
            while matches!(ops[squeeze_i], Op::Pow { .. }) {
                squeeze_i += 1;
            }
            if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                break;
            }
            zc_r.push((vc_at(squeeze_i).1, fin_at(squeeze_i)));
            i2 = squeeze_i + 1;
        }
        // The zerocheck finals (v_a, v_b, ...) — the lincheck entry's
        // absorbed operands.
        let (zcf, _) = vc_at(i2);
        while matches!(ops[i2], Op::ObserveScalar) {
            i2 += 1;
        }
        assert_eq!(i2, lc_l0, "the zerocheck runs straight into the lincheck");
        i2 += 1;
        while matches!(ops[i2], Op::Pow { .. }) {
            i2 += 1;
        }
        assert!(matches!(ops[i2], Op::SqueezeScalar), "lc alpha");
        let lc_alpha = (vc_at(i2).1, fin_at(i2));
        i2 += 1;
        // The const-pin beta squeezes, one per pinned boolean type.
        let mut betas: Vec<(usize, usize)> = Vec::new();
        loop {
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            if !matches!(ops[i2], Op::SqueezeScalar) {
                break;
            }
            betas.push((vc_at(i2).1, fin_at(i2)));
            i2 += 1;
        }
        // (g_v, ch, fin) per lc round — the message ordinals feed the
        // round-0 in-circuit lincheck replay.
        let mut lc_r: Vec<(usize, usize, usize)> = Vec::new();
        while matches!(ops[i2], Op::ObserveScalar) && matches!(ops[i2 + 1], Op::ObserveScalar) {
            let mut squeeze_i = i2 + 2;
            while matches!(ops[squeeze_i], Op::Pow { .. }) {
                squeeze_i += 1;
            }
            if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                break;
            }
            lc_r.push((vc_at(i2).0, vc_at(squeeze_i).1, fin_at(squeeze_i)));
            i2 = squeeze_i + 1;
        }
        assert!(
            matches!(ops[i2], Op::ObserveSlice(Z_PARTIAL_WORDS)),
            "z_partial slice"
        );
        let (zp, _) = vc_at(i2);
        (zc_r, zskip, outer, lc_alpha, betas, zcf, lc_r, zp)
    };
    // The surface→ordinal mapping asserts (the batch-major packing the
    // minimal child pinned): x_inner_rest[0] = mlv round 0, x_outer =
    // rounds 1..1+ν, x_inner_rest[1..] = the rest; rr = the lc rounds
    // REVERSED; z_skip and z_partial are the located ops.
    {
        let inner_b = mat_assert.x_inner_rest.len();
        assert_eq!(
            zc_rounds_b.len(),
            inner_b + n_log_i,
            "zc mlv rounds = x_inner_rest + x_outer"
        );
        for (j, &x) in mat_assert.x_inner_rest.iter().enumerate() {
            let m = if j == 0 { 0 } else { n_log_i + j };
            assert_eq!(
                chals[zc_rounds_b[m].0], x,
                "x_inner_rest {j} is located zc round {m}"
            );
        }
        assert_eq!(lc_rounds_b.len(), mat_assert.rr.len(), "lc round count");
        for (j, &x) in mat_assert.rr.iter().enumerate() {
            assert_eq!(
                chals[lc_rounds_b[lc_rounds_b.len() - 1 - j].1],
                x,
                "rr {j} is the located lc round, reversed"
            );
        }
        // The z_skip pin, by flavor: RS locates the fused squeeze; AG
        // rebuilds the seed from the two located squeezes and pins the
        // assertion's point to the fused H(seed ‖ nonce) decode.
        match (&lo.proof, &zskip) {
            (MixedProof::Rs(_), ZskipTapeRec::Rs { ch, .. }) => {
                assert_eq!(chals[*ch], mat_assert.z_skip.phi8(), "z_skip located");
            }
            (MixedProof::Ag(p), ZskipTapeRec::Ag { seed_ch, .. }) => {
                let nonce = p
                    .boolean
                    .as_ref()
                    .expect("boolean side present")
                    .ag
                    .r1_nonce;
                let pt = decode_ag_point(
                    &ag_seed_bytes(chals[*seed_ch], chals[*seed_ch + 1]),
                    nonce,
                    lo.pcs.zerocheck_grinding().ag_r1_bits(),
                );
                assert_eq!(
                    mat_assert.z_skip,
                    SkipPoint::Ag(pt),
                    "z_skip is the r1 point decoded from the located seed + nonce"
                );
            }
            _ => unreachable!("the tape's zskip record matches the proof flavor"),
        }
        assert_eq!(
            &vals_rec[zp_v..zp_v + 64],
            &mat_assert.z_partial[..],
            "z_partial on the stream"
        );
        assert_eq!(
            mat_assert.alpha, chals[bl_alpha.0],
            "the located boolean lc alpha is the matrix assertion's"
        );
        // The element assertion's points: r_con = zc.r[ν..] (round
        // order), r_col = the lc bind order reversed.
        assert_eq!(
            piop_i.zc_rounds.len(),
            n_log_i + el_assert.r_con.len(),
            "element zc rounds = rows + r_con"
        );
        for (j, &x) in el_assert.r_con.iter().enumerate() {
            assert_eq!(
                chals[piop_i.zc_rounds[n_log_i + j].ch],
                x,
                "el r_con {j} is a located element zc round"
            );
        }
        assert_eq!(
            piop_i.lc_rounds.len(),
            el_assert.r_col.len(),
            "element lc round count"
        );
        for (j, &x) in el_assert.r_col.iter().enumerate() {
            assert_eq!(
                chals[piop_i.lc_rounds[piop_i.lc_rounds.len() - 1 - j].ch],
                x,
                "el r_col {j} is the located element lc round, reversed"
            );
        }
    }
    assert!(
        lc_rounds_b.len() <= 1 + k_cols_i,
        "lc rounds fit the col bits"
    );
    // The boolean lincheck ENTRY, natively: target0 = α·v_a + v_b +
    // Σ β_t·eq_prefix_sum(x_outer, n_t), with x_outer the zc mlv rows
    // (batch-major: rounds 1..1+ν) — replayed through the located lc
    // rounds it must end at the deferred MatrixAssertion's own target
    // (the method-note discipline; this pre-assert is what licenses the
    // in-circuit replay's wire map).
    let (eps_n, entry_n) = {
        let x_outer_n: Vec<F128> = (0..n_log_i).map(|j| chals[zc_rounds_b[1 + j].0]).collect();
        let pinned: Vec<usize> = mat_assert
            .betas
            .iter()
            .enumerate()
            .filter_map(|(t, b)| b.map(|_| t))
            .collect();
        assert_eq!(pinned.len(), betas_b.len(), "one squeeze per const pin");
        let mut eps = Vec::with_capacity(betas_b.len());
        let mut entry = mat_assert.alpha * vals_rec[zc_finals_v] + vals_rec[zc_finals_v + 1];
        for (k, &t) in pinned.iter().enumerate() {
            assert_eq!(
                chals[betas_b[k].0],
                mat_assert.betas[t].expect("pinned"),
                "beta {k} is the located squeeze"
            );
            let e = LiveMask::eq_prefix_sum(&x_outer_n, union_i.counts()[t]);
            entry += chals[betas_b[k].0] * e;
            eps.push(e);
        }
        let mut run = entry;
        for &(g_v, ch, _) in &lc_rounds_b {
            let (e1, einf) = (vals_rec[g_v], vals_rec[g_v + 1]);
            let r = chals[ch];
            let q0 = run + e1;
            run = einf * r * r + (q0 + e1 + einf) * r + q0;
        }
        assert_eq!(
            run, mat_assert.target,
            "the boolean lc entry replays to the assertion's target"
        );
        (eps, entry)
    };
    let _ = entry_n;
    let nat_b = claims.boolean.as_ref().expect("boolean claims");
    let x_ab_n: Vec<F128> = {
        let p = &nat_b.ab.point;
        let mut v = p.x_inner_rest.clone();
        v.extend_from_slice(&p.x_outer);
        v
    };
    let x_c_n: Vec<F128> = {
        let p = &nat_b.c.point;
        let mut v = p.x_inner_rest.clone();
        v.extend_from_slice(&p.x_outer);
        v
    };
    assert_eq!(x_ab_n.len(), 1 + n_log_i + k_cols_i, "ab point split");
    assert_eq!(x_c_n.len(), 1 + n_log_i + k_cols_i, "c point split");
    // Derived pd points (merged-open v1: they left the stream): the
    // element pair from the verifier's own claims, the gathers from
    // gate_claim_point at the GKR's row point — the same derivation the
    // verifier itself performs. Pinned against the round challenges the
    // emitter wires below.
    let pd_pts_n: Vec<Vec<F128>> = {
        let cells = lo.shape.circuit.cells();
        let el = claims.element.as_ref().expect("element claims");
        let mut v = vec![el.c_point.clone(), el.lc_point.clone()];
        for i2 in 0..n_gather {
            v.push(cells.gate_claim_point(i2, &gkr_rec.r_pt[..cells.nu()]));
        }
        v
    };
    for pt in &pd_pts_n {
        assert_eq!(pt.len(), n_log_i + k_cols_i, "pd point split");
    }
    // The element claims' coordinate wires: rows = the element zc rounds
    // [..nu], c's cols = zc rounds [nu..] then prefix bits, lc's cols =
    // the lc rounds REVERSED then prefix bits — pinned value-for-value.
    {
        let e_rounds = piop_i.zc_rounds.len();
        for j in 0..n_log_i {
            assert_eq!(pd_pts_n[0][j], chals[piop_i.zc_rounds[j].ch], "c row {j}");
            assert_eq!(pd_pts_n[1][j], chals[piop_i.zc_rounds[j].ch], "lc row {j}");
        }
        for j in 0..e_rounds - n_log_i {
            assert_eq!(
                pd_pts_n[0][n_log_i + j],
                chals[piop_i.zc_rounds[n_log_i + j].ch],
                "c col {j}"
            );
        }
        let n_lc = piop_i.lc_rounds.len();
        for j in 0..n_lc {
            assert_eq!(
                pd_pts_n[1][n_log_i + j],
                chals[piop_i.lc_rounds[n_lc - 1 - j].ch],
                "lc col {j}"
            );
        }
    }
    let mut groups_ix: Vec<Vec<usize>> = Vec::new();
    for (i2, pt) in pd_pts_n.iter().enumerate() {
        match groups_ix
            .iter_mut()
            .find(|g2| pd_pts_n[g2[0]][..n_log_i] == pt[..n_log_i])
        {
            Some(g2) => g2.push(i2),
            None => groups_ix.push(vec![i2]),
        }
    }
    assert_eq!(groups_ix.len(), n_p, "P scalar groups by shared row");

    // Native replica of the WHOLE anchor expect — validated against
    // the accepted proof before any gate exists.
    {
        let gamma_n = chals[mp_i.gamma_ch];
        let mut gpow_n = vec![F128::ONE];
        for j in 1..257 + n_p {
            gpow_n.push(gpow_n[j - 1] * gamma_n);
        }
        let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
        let point_n: Vec<F128> = mp_i.rounds.iter().map(|rr| chals[rr.ch]).collect();
        let sig_n: Vec<F128> = mp_i.anchor_rounds.iter().map(|rr| chals[rr.ch]).collect();
        let bit = |b: bool| if b { F128::ONE } else { F128::ZERO };
        let g_at_n = {
            let mut rinv = rho_mrg_n.clone();
            let mut acc = F128::ZERO;
            for (j, &gp) in gpow_n.iter().enumerate().take(128) {
                if j > 0 {
                    for x in rinv.iter_mut() {
                        *x = frob_inv(*x);
                    }
                }
                let mut prod = gp;
                for (t3, &x) in point_n.iter().enumerate() {
                    prod *= F128::ONE + rinv[t3] + x;
                }
                acc += prod;
            }
            acc
        };
        let e_at_n = rho_mrg_n
            .iter()
            .zip(&point_n)
            .fold(F128::ONE, |a, (&r3, &x)| a * (F128::ONE + r3 + x));
        let eqc_n: Vec<F128> = bounds_i
            .iter()
            .map(|&(t_c, t_next, _)| {
                let mut p = F128::ONE;
                for l in 0..=m_mp2 {
                    p *= F128::ONE + sig_n[2 * l] + bit((t_c >> l) & 1 == 1);
                    p *= F128::ONE + sig_n[2 * l + 1] + bit((t_next >> l) & 1 == 1);
                }
                p
            })
            .collect();
        let sparse_t = assist_sparse_transitions();
        let dp_native = |z_row: &[F128]| -> F128 {
            let mut gdp = [F128::ZERO; 4];
            gdp[STATE_SUCCESS] = F128::ONE;
            for layer in (0..=m_mp2).rev() {
                let za = if layer < n_log_i {
                    z_row[layer]
                } else {
                    F128::ZERO
                };
                let rb = if layer < m_mp2 {
                    point_n[layer]
                } else {
                    F128::ZERO
                };
                let eq4 = build_eq_table(&[za, rb]);
                let (rc, rd) = (sig_n[2 * layer], sig_n[2 * layer + 1]);
                let e = [
                    (F128::ONE + rc) * (F128::ONE + rd),
                    rc * (F128::ONE + rd),
                    (F128::ONE + rc) * rd,
                    rc * rd,
                ];
                let mut prev = [F128::ZERO; 4];
                for (cd, &ecd) in e.iter().enumerate() {
                    for (s2, slot2) in prev.iter_mut().enumerate() {
                        let (i0, o0) = sparse_t[cd][s2][0];
                        let (i1, o1) = sparse_t[cd][s2][1];
                        *slot2 += ecd * (eq4[i0] * gdp[o0] + eq4[i1] * gdp[o1]);
                    }
                }
                gdp = prev;
            }
            gdp[STATE_INITIAL]
        };
        let run_weights_n = |z_col: &[F128]| -> Vec<F128> {
            let mut w_at = vec![F128::ZERO; n_runs];
            let mut tot = F128::ONE;
            for (r3, &(_, _, len)) in bounds_i.iter().enumerate() {
                if r3 == comp_ix {
                    continue;
                }
                let mut w = F128::ZERO;
                for y in run_y0[r3]..run_y0[r3] + len as usize {
                    let mut s = F128::ONE;
                    for (jj, &zc2) in z_col.iter().enumerate() {
                        s *= F128::ONE + zc2 + bit((y >> jj) & 1 == 1);
                    }
                    w += s;
                }
                w_at[r3] = w;
                tot += w;
            }
            w_at[comp_ix] = tot;
            w_at
        };
        let expect_n = {
            let mut acc = F128::ZERO;
            for (si, xs) in [&x_ab_n, &x_c_n].iter().enumerate() {
                let z_row = &xs[1..1 + n_log_i];
                let run_n = run_weights_n(&xs[1 + n_log_i..]);
                let w_n = run_n
                    .iter()
                    .zip(&eqc_n)
                    .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                // The count win's tie at real-inner scale: the RS raw W
                // the region now publishes equals the deferred export's
                // claim value.
                assert_eq!(
                    jag_assert.rs[si].value, w_n,
                    "RS raw W == exported jagged claim {si}"
                );
                let coeff = if si == 0 {
                    g_at_n
                } else {
                    gpow_n[128] * g_at_n
                };
                acc += coeff * (w_n * dp_native(z_row));
            }
            for (g_ix, members) in groups_ix.iter().enumerate() {
                let mut run_n = vec![F128::ZERO; n_runs];
                for &i2 in members {
                    let pd = &gammas_i[i2];
                    let gpd = chals[pd.ch];
                    let w_at = run_weights_n(&pd_pts_n[i2][n_log_i..]);
                    for r3 in 0..n_runs {
                        run_n[r3] += gpd * w_at[r3];
                    }
                }
                let w_n = run_n
                    .iter()
                    .zip(&eqc_n)
                    .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                // The group's exported decomposition recombines to the
                // same raw group W, member for member.
                let (combo, dense) = &jag_assert.groups[g_ix];
                let mut raw = combo.as_ref().map_or(F128::ZERO, |c| c.value);
                let mut d_it = dense.iter();
                for &i2 in members {
                    let hot = pd_pts_n[i2][n_log_i..]
                        .iter()
                        .all(|&x| x == F128::ZERO || x == F128::ONE);
                    if hot {
                        continue;
                    }
                    let (g, c) = d_it.next().expect("a dense entry per non-hot member");
                    assert_eq!(*g, chals[gammas_i[i2].ch], "dense member γ_pd");
                    raw += *g * c.value;
                }
                assert!(d_it.next().is_none(), "every dense entry consumed");
                assert_eq!(
                    raw, w_n,
                    "group {g_ix} raw W == exported jagged decomposition"
                );
                let dp = dp_native(&pd_pts_n[members[0]][..n_log_i]);
                acc += gpow_n[256 + g_ix] * e_at_n * (w_n * dp);
            }
            acc
        };
        assert_eq!(
            expect_n, anc_end_n,
            "the anchor expect replays natively at real-inner scale"
        );
    }
    AnchorBooleanRec {
        m_mp2,
        n_log_i,
        k_cols_i,
        n_pub_slots_c,
        bounds_i,
        zc_rounds_b,
        zskip,
        outer_b: (outer_ch_b, outer_fin_b),
        bl_alpha,
        betas_b,
        zc_finals_v,
        lc_rounds_b,
        zp_v,
        eps_n,
        x_ab_n,
        x_c_n,
        pd_pts_n,
        groups_ix,
    }
}

/// The real tail's EXPECTED publish schedule, every width an expression
/// over the TAPE's own geometry and the proof object — the same
/// formulas the checker's positional walk consumes by — the independent
/// reference the emitted table ([`RealRegion::tail_schedule`]) is held
/// against at every build.
pub(super) fn expected_real_tail_schedule(rt: &RealTape<'_>) -> Vec<(&'static str, usize)> {
    let lc_i = rt.lo.proof.boolean_lincheck();
    let n_mat = 1
        + rt.lc_rounds_b.len()
        + 2 * lc_i.matrix_evals.len()
        + rt.mat_assert.x_inner_rest.len()
        + rt.n_log_i
        + 2 * rt.betas_b.len()
        + Z_PARTIAL_WORDS
        + 1;
    let n_ela = 1
        + rt.piop_i.zc_rounds.len()
        + rt.piop_i.lc_rounds.len()
        + 3
        + 2 * rt.el_assert.evals.len();
    let jag_vals = rt.jag.rs.len()
        + rt.jag
            .groups
            .iter()
            .map(|(combo, dense)| usize::from(combo.is_some()) + dense.len())
            .sum::<usize>();
    vec![
        ("spine t_final", 2),
        ("intake target", 1),
        ("intake running", 1),
        ("residual accs", 2 * rt.levels.len() * rt.yr_len),
        ("residual inner", 2),
        ("sigma value", 1),
        ("sigma point", rt.mu_i),
        ("element zc+lc ends", 2),
        ("anchor end", 1),
        ("matrix assertion publics", n_mat),
        ("element assertion publics", n_ela),
        ("jagged claim values", jag_vals),
    ]
}

/// What one emitted REAL child region hands back: where its public block
/// starts, the walk counts, and the assertion-emission wires the 2→1 merge
/// node CONNECTS the fold region's claim words to — all three families.
pub(super) struct RealRegion {
    pub(super) pub_base: usize,
    pub(super) n_query_pub: usize,
    pub(super) n_tail: usize,
    /// The published tail's `(name, width)` schedule, as emitted — held
    /// against [`expected_real_tail_schedule`] at every build.
    pub(super) tail_schedule: Vec<(&'static str, usize)>,
    n_mat_pub: usize,
    n_ela_pub: usize,
    /// Labeled `public_len` checkpoints through the emission — the publics
    /// census (`PUB_CENSUS=1` on the node test prints the block sizes).
    pub(super) census: Vec<(&'static str, usize, usize)>,
    /// The jagged assertion's value wires (the count win), in emission
    /// order: rs claims, then per group the combo and its dense members —
    /// the fresh-claim surfaces a merge fold connects to.
    pub(super) jag_w: Vec<Wire>,
    /// The claims' IDENTITY wires (the points-connect): σ shared, and per
    /// claim (jag_w order) the row wires — Eq: z_col coordinate wires
    /// (constant coords ride zw/ow); Combo: the γ_pd coefficient wires in
    /// term order (addresses are registry constants, bound by the fold
    /// side's shared constant publics).
    pub(super) jag_sig_w: Vec<Wire>,
    pub(super) jag_row_w: Vec<Vec<Wire>>,
    /// The z_skip surface, by flavor — see [`ZskipWires`].
    pub(super) zskip: ZskipWires,
    /// Every fresh claim in `sigma_native.claims()` as `(row, col, value)`
    /// wires, in accumulator order.
    pub(super) structure_claim_w: Vec<(Vec<Wire>, Vec<Wire>, Wire)>,
    /// element: every zc/lc round rho (round order) and the per-slot eval
    /// advice pairs (bound publics — connectable, unlike the minimal child).
    pub(super) el_zc_rho_w: Vec<Wire>,
    pub(super) el_lc_rho_w: Vec<Wire>,
    pub(super) el_eval_w: Vec<(Wire, Wire)>,
    /// boolean: the zc mlv / lc round rhos (round order), the absorbed
    /// z_partial words, and the per-type matrix_evals advice pairs.
    pub(super) b_mlv_w: Vec<Wire>,
    pub(super) b_lc_w: Vec<Wire>,
    pub(super) b_zpartial_w: Vec<Wire>,
    pub(super) mat_eval_w: Vec<(Wire, Wire)>,
    /// The residual close-out's prefix slot (and width).
    pub(super) pf: (SlotId, usize),
    /// The child's PUBLIC SEGMENT as witness wires — the app-statement
    /// plumbing (hash-chain adjacency) reads through these.
    pub(super) child_pub_w: Vec<Wire>,
    /// The child's own CIRCUIT DIGEST, absorbed by its statement binding
    /// (payload 3) — two public words. This is the KEY its sigma and
    /// jagged claims fold under, and the spine's match-gate compares it
    /// against the key an inherited entry was published with.
    pub(super) cd_w: [Wire; 2],
}

/// Decompose the polynomial-basis trace-dual table into one geometric row and
/// seven exceptional entries.  If `d_t = moore_inverse()[t]`, then
/// `d_t = g0 * ratio^t` for every `t >= 7`; the low seven corrections are the
/// only effect of the low terms in GHASH's defining polynomial.  Frobenius
/// powers preserve this form, which lets the circuit evaluate every inverse-
/// Moore row with one prefix product and seven MACs instead of wiring a
/// 128-word constant row.
pub(super) fn family_h_dual_decomposition() -> (F128, F128, [F128; 7]) {
    static DECOMP: OnceLock<(F128, F128, [F128; 7])> = OnceLock::new();
    *DECOMP.get_or_init(|| {
        let minv = moore_inverse();
        let d = &minv[..128];
        let ratio = d[8] * d[7].inv();
        let ratio_inv = ratio.inv();
        let mut g0 = d[7];
        for _ in 0..7 {
            g0 *= ratio_inv;
        }
        let mut corrections = [F128::ZERO; 7];
        let mut g = g0;
        for t in 0..128 {
            if t < 7 {
                corrections[t] = d[t] + g;
            } else {
                assert_eq!(d[t], g, "the GHASH dual basis is geometric above t=6");
            }
            g *= ratio;
        }
        (g0, ratio, corrections)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_family_h(
    sb: &mut ShapeBuilder,
    tile: SlotId,
    macs: SlotId,
    fold_macs: SlotId,
    spine: SlotId,
    spine256: SlotId,
    mac256: SlotId,
    row_capacity: usize,
    shv: &[Vec<Wire>; 2],
    values: &[Vec<Wire>; 2],
    r_dprime: &[Vec<Wire>; 2],
    gamma: [Wire; 2],
    pfslot: SlotId,
    pf_w: usize,
    zw: Wire,
    ow: Wire,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) -> (Wire, Wire) {
    assert_eq!(pf_w, 8, "the envelope family-H prefix width is eight");
    for k in 0..2 {
        assert_eq!(shv[k].len(), 128, "one full tensor-algebra row table");
        assert_eq!(values[k].len(), 128, "one Frobenius value per Moore row");
        assert_eq!(
            r_dprime[k].len(),
            7,
            "the ring-switch suffix has seven bits"
        );
    }
    let prefix = |sb: &mut ShapeBuilder, seed: Wire, a: &[Wire], b: &[Wire]| {
        assert!(a.len() <= pf_w && b.len() <= pf_w);
        let mut input = Vec::with_capacity(2 + 2 * pf_w);
        input.push(seed);
        input.extend_from_slice(a);
        input.extend(repeat_n(zw, pf_w - a.len()));
        input.extend_from_slice(b);
        input.extend(repeat_n(zw, pf_w - b.len()));
        input.push(ow);
        sb.gate(pfslot, &input)[0]
    };
    let spine_mac = |sb: &mut ShapeBuilder, acc: Wire, x: Wire, y: Wire| {
        sb.gate(spine, &[zw, zw, zw, acc, zw, zw, x, y, zw])[3]
    };
    let spine_square =
        |sb: &mut ShapeBuilder, x: Wire| sb.gate(spine, &[zw, zw, ow, zw, zw, zw, zw, zw, x])[4];

    // Equality weights are shared by the transpose dot and the seven sparse
    // corrections in every inverse-Moore coefficient.
    let mut eq_w: [Vec<Wire>; 2] = from_fn(|_| Vec::with_capacity(128));
    for k in 0..2 {
        for t in 0..128 {
            let bits: Vec<Wire> = (0..7)
                .map(|j| if (t >> j) & 1 == 1 { ow } else { zw })
                .collect();
            eq_w[k].push(prefix(sb, ow, &r_dprime[k], &bits));
        }
    }

    // The transpose is tiled so the boolean relation needs only 17 wired
    // words.  Dot-product linearity lets us accumulate each partial output
    // directly, without materializing the 128 transposed words in the element
    // layer.  Claim 0 uses the main MAC slot and claim 1 the fold MAC slot;
    // this split keeps both below 2^14 rows at the two-child fixed point.
    let mut rs_half = zw;
    for k in 0..2 {
        let dot_slot = if k == 0 { macs } else { fold_macs };
        let mut dot = zw;
        for destination_byte in 0..16 {
            let rows = &shv[k][8 * destination_byte..8 * destination_byte + 8];
            for source_byte in 0..16 {
                let selector = cw(
                    sb,
                    vals,
                    consts,
                    F128::new((source_byte | (destination_byte << 4)) as u64, 0),
                );
                let mut input = rows.to_vec();
                input.push(selector);
                let partial = sb.gate(tile, &input);
                for c in 0..8 {
                    dot = sb.gate(dot_slot, &[dot, partial[c], eq_w[k][8 * source_byte + c]])[0];
                }
            }
        }
        rs_half = sb.gate(macs, &[rs_half, gamma[k], dot])[0];
    }

    // c_j/gamma is the MLE of inverse-Moore row j at r_dprime.  The trace-
    // dual table is geometric except at indices 0..6, so one prefix product
    // plus seven correction MACs computes it.  Constants are fixed publics:
    // changing any of them changes the circuit digest.
    let (mut g0_j, mut ratio_j, mut corrections_j) = family_h_dual_decomposition();
    let minv = moore_inverse();
    // Only the eight orbit seeds are fixed publics.  Successive Frobenius
    // powers are derived by the existing spine's squaring cell, saving more
    // than one thousand public constants at a two-child node.
    let mut g0_j_w = cw(sb, vals, consts, g0_j);
    let mut corrections_j_w: [Wire; 7] = from_fn(|t| cw(sb, vals, consts, corrections_j[t]));
    let mut coeff_w: [Vec<Wire>; 2] = from_fn(|_| Vec::with_capacity(128));
    for j in 0..128 {
        let mut ratio_pows = [F128::ZERO; 7];
        ratio_pows[0] = ratio_j;
        for q in 1..7 {
            ratio_pows[q] = ratio_pows[q - 1] * ratio_pows[q - 1];
        }
        for k in 0..2 {
            let scaled_r: Vec<Wire> = (0..7)
                .map(|q| {
                    let factor = cw(sb, vals, consts, ratio_pows[q]);
                    spine_mac(sb, zw, r_dprime[k][q], factor)
                })
                .collect();
            let mut mle = prefix(sb, g0_j_w, &r_dprime[k], &scaled_r);
            for t in 0..7 {
                mle = sb.gate(macs, &[mle, corrections_j_w[t], eq_w[k][t]])[0];
            }
            coeff_w[k].push(sb.gate(macs, &[zw, gamma[k], mle])[0]);
        }

        // Pin the closed form to the native matrix on every row.  This is a
        // shape-time assertion, not witness checking, and catches a basis or
        // field-polynomial change before it can silently alter family H.
        let mut gp = g0_j;
        for t in 0..128 {
            let got = gp + if t < 7 { corrections_j[t] } else { F128::ZERO };
            assert_eq!(got, minv[j * 128 + t], "inverse-Moore row {j}, entry {t}");
            gp *= ratio_j;
        }
        g0_j *= g0_j;
        ratio_j *= ratio_j;
        for d in &mut corrections_j {
            *d *= *d;
        }
        if j + 1 < 128 {
            g0_j_w = spine_square(sb, g0_j_w);
            for d in &mut corrections_j_w {
                *d = spine_square(sb, *d);
            }
        }
    }

    // Pair the two RS claims in one F256 squaring chain.  After j extension-
    // field squarings of a+b*u, the second component is b^(2^j) and the first
    // is a^(2^j)+K_j*b^(2^j), with K_{j+1}=K_j^2+NR.  One base-field MAC
    // recovers a^(2^j). The squarings fill the residual region's narrower
    // F256 MAC relation first, then spill into the equivalent Ligerito-spine
    // multiplication only at that slot's physical row limit. This keeps both
    // existing slots in bounds without adding a table type.
    let mut vrs = zw;
    let mut k_j = F128::ZERO;
    for j in 0..128 {
        let mut pair = [values[0][j], values[1][j]];
        for _ in 0..j {
            pair = if sb.rows_in_slot(mac256) < row_capacity {
                emit_mac256(sb, mac256, [zw, zw], pair, pair)
            } else {
                emit_spine256(
                    sb,
                    spine256,
                    [zw, zw],
                    [zw, zw],
                    [ow, zw],
                    [zw, zw],
                    [zw, zw],
                    [zw, zw],
                    [zw, zw],
                    zw,
                    pair,
                )[4]
            };
        }
        let kw = cw(sb, vals, consts, k_j);
        let p0 = sb.gate(macs, &[pair[0], kw, pair[1]])[0];
        vrs = sb.gate(macs, &[vrs, coeff_w[0][j], p0])[0];
        vrs = sb.gate(macs, &[vrs, coeff_w[1][j], pair[1]])[0];
        k_j = k_j * k_j + QUADRATIC_NONRESIDUE;
    }
    (rs_half, vrs)
}

/// Emit ONE real child's complete deferred-verifier region — the swap
/// test's whole assembly (chain, PoW, query phase, W-rounds, spine,
/// residual, wiring GKR + sigma, multi-slot element PIOP with the GENERAL
/// strip, multipoint intake, anchor expect with one-hot gathers and
/// eq-table dots, and all THREE assertion emissions) — into `sb` over the
/// shared [`ChildSlots`], publishing exactly what
/// [`check_real_child_region`] walks.
pub(super) fn emit_real_child_region(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    b3_slot: SlotId,
    rt: &RealTape<'_>,
    vals: &mut Vec<F128>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
    consts: &mut Vec<(F128, Wire)>,
) -> RealRegion {
    let child_q = CollapsedSlots {
        b3: b3_slot,
        ..cs.q
    };
    let trace = &rt.trace;
    let stream = &rt.stream;
    let levels = &rt.levels[..];
    let geo = &rt.geo[..];
    let piop_i = &rt.piop_i;
    let n_log_i = rt.n_log_i;

    let leafeval: Vec<_> = geo
        .iter()
        .map(|g| {
            let lanes = g.lanes.min(8);
            match cs.le.iter().find(|(n, _)| *n == 800 + lanes) {
                Some((_, sl)) => *sl,
                None => {
                    let sl = sb.slot(LeafEvalGate256::new(lanes));
                    cs.le.push((800 + lanes, sl));
                    sl
                }
            }
        })
        .collect();
    let mut cen: Vec<(&'static str, usize, usize)> =
        vec![("start", sb.public_len(), sb.rows_in_slot(cs.macs))];
    let iv_w = pack8(&IV);
    vals.extend_from_slice(&iv_w);
    let iv2 = [
        sb.fixed_public_input(iv_w[0]),
        sb.fixed_public_input(iv_w[1]),
    ];
    let (outs, ww) = emit_fs_chain(
        sb,
        b3_slot,
        iv2,
        trace,
        stream,
        &rt.bytes,
        vals,
        consts,
        &rt.pub_payloads,
        &rt.cross,
    );

    cen.push((
        "chain payloads + shared consts",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    if var("PUB_CENSUS").is_ok() {
        let pay_pub: usize = stream
            .words
            .iter()
            .enumerate()
            .filter(|(wi, w)| {
                matches!(w, StreamWord::Bytes { payload, .. }
                    if rt.pub_payloads[*payload])
                    && ww[*wi].is_some()
            })
            .count();
        println!(
            "  [census probe] chain block: {} payload words public, {} cw consts",
            pay_pub,
            consts.len()
        );
    }
    // The PoW grinding wires: [predicate word, nonce word] per op.
    let pow_wires: Vec<[Wire; 2]> = rt
        .pows
        .iter()
        .map(|&(fin, pay, _)| {
            let sq = &trace.squeezes[fin];
            let wi = stream
                .words
                .iter()
                .position(|w| matches!(w, StreamWord::Bytes { payload, .. } if *payload == pay))
                .expect("pow nonce stream word");
            let nw = ww[wi].expect("pow nonce wired");
            [outs[sq[0]][1], nw]
        })
        .collect();
    let pow_checks: Vec<([Wire; 2], u32)> = pow_wires
        .iter()
        .zip(&rt.pows)
        .map(|(&w, &(_, _, bits))| (w, bits))
        .collect();
    emit_pow_checks(sb, b3_slot, cs.q.pow, iv2, &pow_checks, vals, consts);

    let (cd_w, pub_w) =
        emit_h_publics_region(sb, cs, child_q, iv2, rt, &ww, &mut cen, vals, consts);
    let cap_w = cap_wires(stream, &ww, &rt.cap_pays);
    let (to_publish, level_accs, query_positions) = emit_query_phase(
        sb,
        child_q,
        iv2,
        &leafeval,
        levels,
        geo,
        &rt.lvl_src,
        trace,
        &outs,
        &rt.chals,
        &cap_w,
        vals,
        consts,
        hints,
    );
    cen.push((
        "query phase decl",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    let IntakeSpineResidual {
        vmap,
        zw,
        ow,
        zassert,
        t_final,
        resid_pub,
        inner_w,
        pf: (pfslot, pf_w),
        tgt_w,
        runw,
    } = emit_intake_spine_residual(
        sb,
        cs,
        rt,
        &ww,
        &outs,
        &to_publish,
        &query_positions,
        &level_accs,
        &mut cen,
        vals,
        consts,
    );
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };

    let WiringGkrWires {
        pt_w,
        mid_w,
        live_w,
        f_w,
        g_w,
        sig_w,
    } = emit_wiring_gkr_sigma(sb, cs, rt, &outs, &wv, vals, zw, ow, zassert);

    emit_recombination_f_eq_g(sb, cs, rt, &wv, &pub_w, &pt_w, f_w, g_w, zw, ow, &mut cen);
    let ElementPiopWires {
        el_zr,
        el_alpha_w,
        va_w,
        vb_w,
        asum_w,
        bsum_w,
        el_lcw,
    } = emit_element_piop(sb, cs, rt, &outs, &wv, vals, zw, ow, zassert, &mut cen);
    let (mp_pws, mp_rho2_w, mp_sig_w, anc_w) = walker_common::emit_multipoint_intake(
        sb, cs, trace, &outs, &wv, &rt.mp_i, rt.m_mp2, zw, ow,
    );

    let (jag_w, jag_row_w, mlv_pw, lc_pw) = emit_anchor_expect(
        sb, cs, rt, &outs, &mp_pws, &mp_rho2_w, &mp_sig_w, &pt_w, anc_w, pfslot, pf_w, zw, ow,
        zassert, vals, consts, &mut cen,
    );
    let AssertionPublics {
        mat_pub,
        ela_pub,
        mat_eval_w,
        el_eval_w,
        zpartial_ws,
        eps_wires,
    } = emit_assertion_families(
        sb, cs, rt, &outs, &wv, &mlv_pw, &lc_pw, el_alpha_w, va_w, vb_w, el_lcw, pfslot, pf_w, zw,
        ow, vals, &mut cen,
    );
    let (pub_base, n_tail, tail_schedule) = emit_publishes_in_swap_order(
        sb,
        cs,
        &to_publish,
        &level_accs,
        t_final,
        tgt_w,
        runw,
        &resid_pub,
        inner_w,
        sig_w,
        &pt_w,
        el_zr,
        el_lcw,
        anc_w,
        &mat_pub,
        &ela_pub,
        &jag_w,
        &mut cen,
    );

    let n_query_pub: usize = 2 * levels.len() + levels.iter().map(|l| l.a_count).sum::<usize>();
    let el_zc_rho_w: Vec<Wire> = piop_i
        .zc_rounds
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let boolean_values: Vec<(usize, Wire)> = rt
        .sigma_native
        .boolean_pins
        .iter()
        .map(|(t, _, _)| *t)
        .zip(eps_wires)
        .collect();
    let structure_claim_w = circuit_structure_claim_wires(
        &rt.sigma_native,
        &pt_w,
        mid_w,
        live_w,
        sig_w,
        &mlv_pw[1..1 + n_log_i]
            .iter()
            .map(|&(_, w)| w)
            .collect::<Vec<_>>(),
        &boolean_values,
        Some(&el_zc_rho_w[n_log_i..]),
        Some((asum_w, bsum_w)),
        zw,
        ow,
    );
    RealRegion {
        pub_base,
        n_query_pub,
        n_tail,
        tail_schedule,
        n_mat_pub: mat_pub.len(),
        census: cen,
        jag_w,
        jag_sig_w: mp_sig_w.clone(),
        jag_row_w,
        zskip: zskip_wires(&rt.zskip, &outs, trace, &rt.stream, &ww),
        n_ela_pub: ela_pub.len(),
        structure_claim_w,
        el_zc_rho_w,
        el_lc_rho_w: piop_i
            .lc_rounds
            .iter()
            .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
            .collect(),
        el_eval_w,
        b_mlv_w: mlv_pw.iter().map(|&(_, w)| w).collect(),
        b_lc_w: rt
            .lc_rounds_b
            .iter()
            .map(|&(_, _, fin)| outs[trace.squeezes[fin][0]][0])
            .collect(),
        b_zpartial_w: zpartial_ws,
        mat_eval_w,
        pf: (pfslot, pf_w),
        child_pub_w: pub_w,
        cd_w,
    }
}

/// ROUND 2: the H(publics) region (v2 statement binding)
///
/// Payload 4 of the circuit binding is the 32-byte publics digest; the
/// child's public words themselves are witness, bound here. The returned
/// wires ARE the child's public segment — the recombination folds them.
/// Payload 3 is the child's CIRCUIT DIGEST (`bind_statement_circuit`'s
/// order: registry, counts, cap, circuit, publics) — the FOLD KEY this
/// child's claims belong under (wall 3), exported so the fold region's
/// absorbed group digest binds to the circuit actually verified here.
#[allow(clippy::too_many_arguments)]
fn emit_h_publics_region(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    child_q: CollapsedSlots,
    iv2: [Wire; 2],
    rt: &RealTape<'_>,
    ww: &[Option<Wire>],
    cen: &mut Vec<(&'static str, usize, usize)>,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) -> ([Wire; 2], Vec<Wire>) {
    let stream = &rt.stream;
    let pays = payload_words(stream);
    assert_eq!(pays[3].len(), 2, "the circuit digest payload is 32 bytes");
    let cd_w = [
        ww[pays[3][0]].expect("circuit digest word wired"),
        ww[pays[3][1]].expect("circuit digest word wired"),
    ];
    let pub_w = {
        assert_eq!(pays[4].len(), 2, "the publics digest payload is 32 bytes");
        let dw = [
            ww[pays[4][0]].expect("digest word wired"),
            ww[pays[4][1]].expect("digest word wired"),
        ];
        emit_publics_hash(sb, child_q, iv2, &rt.lo.public, dw, vals, consts)
    };
    cen.push((
        "H(publics) region",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    (cd_w, pub_w)
}

/// What [`emit_intake_spine_residual`] threads forward to the later
/// regions.
struct IntakeSpineResidual {
    vmap: Vec<Option<usize>>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
    t_final: [Wire; 2],
    resid_pub: Vec<Vec<[Wire; 2]>>,
    inner_w: [Wire; 2],
    pf: (SlotId, usize),
    tgt_w: Wire,
    runw: Wire,
}

/// intake W-rounds, spine, residual
#[allow(clippy::too_many_arguments)]
fn emit_intake_spine_residual(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    rt: &RealTape<'_>,
    ww: &[Option<Wire>],
    outs: &[Vec<Wire>],
    to_publish: &[Vec<Wire>],
    query_positions: &[Vec<Wire>],
    level_accs: &[[Wire; 2]],
    cen: &mut Vec<(&'static str, usize, usize)>,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) -> IntakeSpineResidual {
    let trace = &rt.trace;
    let stream = &rt.stream;
    let levels = &rt.levels[..];
    let geo = &rt.geo[..];
    let w_rounds = &rt.w_rounds[..];
    let mp_i = &rt.mp_i;
    let inner_pd_i = &rt.inner_pd_i;
    let gammas_i = &rt.gammas_i[..];
    let mut vmap: Vec<Option<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let StreamWord::Value(vi) = *w {
            if vmap.len() <= vi {
                vmap.resize(vi + 1, None);
            }
            vmap[vi] = Some(wi);
        }
    }
    let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
    vals.push(F128::ZERO);
    let zw = sb.public_input();
    vals.push(F128::ONE);
    let ow = sb.public_input();
    let spine256 = cs.spine256;
    // The assert-zero anchor: a dedicated zero public NO gate consumes,
    // so the zero-delta outputs connected into its class add no
    // dataflow edges (connecting them to the ubiquitous `zw` creates
    // cycles — the acyclicity check draws producer→consumer edges).
    vals.push(F128::ZERO);
    let zassert = sb.public_input();

    cen.push((
        "zero/one/anchor consts",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    // The ligerito SPINE: start gamma'·q_eval, eval/build per fold,
    // intro-folds consuming the query phase's accumulator wires.
    let t_final = walker_common::emit_ligerito_spine_walk(
        sb,
        spine256,
        outs,
        trace,
        &wv,
        levels,
        inner_pd_i,
        rt.start_v_i,
        level_accs,
        zw,
        ow,
    );

    // The RESIDUAL region via the shared emitter (lane-major rotation).
    let yr_wires: Vec<[Wire; 2]> = (0..rt.yr_len)
        .map(|y| [wv(rt.yr_v_i + 2 * y), wv(rt.yr_v_i + 2 * y + 1)])
        .collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        sb,
        &mut cs.resid,
        levels,
        geo,
        to_publish,
        query_positions,
        &rt.w_resid,
        inner_pd_i.fin,
        &yr_wires,
        trace,
        outs,
        zw,
        ow,
    );
    // THE CLOSURE, in-circuit: inner == t_r as a copy constraint.
    sb.connect(inner_w[0], t_final[0]);
    sb.connect(inner_w[1], t_final[1]);

    // The complete family-H relation.  All inputs below are already bound
    // transcript or proof wires; no target/V advice and no native checker are
    // part of the recursive statement anymore.
    let (tgt_w, runw) = walker_common::emit_family_h_boundary(
        sb,
        cs,
        trace,
        outs,
        &wv,
        &rt.rs_recs,
        &rt.rs_gam_fins,
        mp_i,
        gammas_i,
        w_rounds,
        inner_pd_i,
        pfslot,
        pf_w,
        vals,
        consts,
        zw,
        ow,
    );

    cen.push((
        "family-H + merged boundary",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));

    cen.push((
        "spine + residual advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    IntakeSpineResidual {
        vmap,
        zw,
        ow,
        zassert,
        t_final,
        resid_pub,
        inner_w,
        pf: (pfslot, pf_w),
        tgt_w,
        runw,
    }
}

/// What [`emit_wiring_gkr_sigma`] threads forward to the later regions.
struct WiringGkrWires {
    pt_w: Vec<Wire>,
    mid_w: Wire,
    live_w: Wire,
    f_w: Wire,
    g_w: Wire,
    sig_w: Wire,
}

/// the WIRING GKR in-circuit + the sigma emission
#[allow(clippy::too_many_arguments)]
fn emit_wiring_gkr_sigma(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    rt: &RealTape<'_>,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
) -> WiringGkrWires {
    let trace = &rt.trace;
    let macs = cs.macs;
    let zcr = cs.zcr;
    let gr = &rt.gkr;
    let g_alpha_w = outs[trace.squeezes[gr.alpha_fin][0]][0];
    let g_beta_w = outs[trace.squeezes[gr.beta_fin][0]][0];
    // Every former published-zero delta in this region is a COPY
    // CONSTRAINT now — the proof itself fails on a broken identity.
    let (mut cl_w, mut cr_w) = (wv(gr.top_v), wv(gr.top_v + 1));
    sb.connect(cl_w, cr_w);
    let mut pt_w: Vec<Wire> = Vec::new();
    for lr in &gr.layers {
        let lam_w = outs[trace.squeezes[lr.lam_fin][0]][0];
        let mut run_w = sb.gate(macs, &[cl_w, lam_w, cr_w])[0];
        let mut pt_next: Vec<Wire> = Vec::with_capacity(lr.rounds.len() + 1);
        for (t2, &(gv, rfin)) in lr.rounds.iter().enumerate() {
            let rho_w = outs[trace.squeezes[rfin][0]][0];
            vals.push(lr.g0s[t2]);
            let g0w = sb.input();
            let o = sb.gate(zcr, &[run_w, wv(gv), wv(gv + 1), pt_w[t2], rho_w, g0w, ow]);
            sb.connect(o[0], zassert);
            run_w = o[1];
            pt_next.push(rho_w);
        }
        let (vl0, vl1) = (wv(lr.v_v), wv(lr.v_v + 1));
        let (vr0, vr1) = (wv(lr.v_v + 2), wv(lr.v_v + 3));
        let pl2 = sb.gate(macs, &[zw, vl0, vl1])[0];
        let pr2 = sb.gate(macs, &[zw, vr0, vr1])[0];
        let gate_w = sb.gate(macs, &[pl2, lam_w, pr2])[0];
        sb.connect(gate_w, run_w);
        let ck_w = outs[trace.squeezes[lr.ck_fin][0]][0];
        let sl2 = sb.gate(macs, &[vl0, vl1, ow])[0];
        let sr2 = sb.gate(macs, &[vr0, vr1, ow])[0];
        cl_w = sb.gate(macs, &[vl0, ck_w, sl2])[0];
        cr_w = sb.gate(macs, &[vr0, ck_w, sr2])[0];
        pt_next.push(ck_w);
        pt_w = pt_next;
    }
    assert_eq!(
        pt_w.len(),
        rt.mu_i,
        "the GKR point spans the inner cell space"
    );
    // M̂(ρ) / livê(ρ), bound through the digest-keyed
    // circuit-structure claims folded by the parent.
    vals.push(rt.mid_n);
    let mid_w = sb.public_input();
    vals.push(rt.live_n);
    let live_w = sb.public_input();
    let (f_w, g_w, sig_w) = (wv(gr.fgs_v), wv(gr.fgs_v + 1), wv(gr.fgs_v + 2));
    let l1 = sb.gate(macs, &[f_w, g_alpha_w, mid_w])[0];
    let l2 = sb.gate(macs, &[l1, g_beta_w, live_w])[0];
    let l3 = sb.gate(macs, &[l2, ow, live_w])[0];
    let l4 = sb.gate(macs, &[l3, ow, ow])[0];
    sb.connect(l4, cl_w);
    let r1 = sb.gate(macs, &[g_w, g_alpha_w, sig_w])[0];
    let r2 = sb.gate(macs, &[r1, g_beta_w, live_w])[0];
    let r3 = sb.gate(macs, &[r2, ow, live_w])[0];
    let r4 = sb.gate(macs, &[r3, ow, ow])[0];
    sb.connect(r4, cr_w);
    WiringGkrWires {
        pt_w,
        mid_w,
        live_w,
        f_w,
        g_w,
        sig_w,
    }
}

/// ROUND 4: the recombination + f == g, in-circuit
#[allow(clippy::too_many_arguments)]
fn emit_recombination_f_eq_g(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    rt: &RealTape<'_>,
    wv: &impl Fn(usize) -> Wire,
    pub_w: &[Wire],
    pt_w: &[Wire],
    f_w: Wire,
    g_w: Wire,
    zw: Wire,
    ow: Wire,
    cen: &mut Vec<(&'static str, usize, usize)>,
) {
    let gammas_i = &rt.gammas_i[..];
    let n_log_i = rt.n_log_i;
    let le8 = match cs.le.iter().find(|&&(n, _)| n == 8) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(LeafEvalGate::new(8));
            cs.le.push((8, s));
            s
        }
    };
    let gather_w: Vec<Wire> = (0..rt.n_gather)
        .map(|i| wv(gammas_i[2 + i].val_v))
        .collect();
    emit_recombination(
        sb,
        cs.fold_macs,
        le8,
        pub_w,
        &gather_w,
        pt_w,
        n_log_i,
        rt.n_pub_slots_c,
        f_w,
        g_w,
        zw,
        ow,
    );

    cen.push((
        "GKR advice (g0s, mask)",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
}

/// What [`emit_element_piop`] threads forward to the later regions.
struct ElementPiopWires {
    el_zr: Wire,
    el_alpha_w: Wire,
    va_w: Wire,
    vb_w: Wire,
    asum_w: Wire,
    bsum_w: Wire,
    el_lcw: Wire,
}

/// the MULTI-SLOT element PIOP (general strip)
#[allow(clippy::too_many_arguments)]
fn emit_element_piop(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    rt: &RealTape<'_>,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
    cen: &mut Vec<(&'static str, usize, usize)>,
) -> ElementPiopWires {
    let trace = &rt.trace;
    let piop_i = &rt.piop_i;
    let macs = cs.macs;
    let zcr = cs.zcr;
    let mrslot = cs.mrs;
    let mut el_zr = zw;
    for (k, rr) in piop_i.zc_rounds.iter().enumerate() {
        let t_w = squeeze_word_wire(outs, trace, piop_i.tau_fin, k);
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        vals.push(rt.el_g0[k]);
        let g0w = sb.input();
        let o = sb.gate(
            zcr,
            &[el_zr, wv(rr.g_v), wv(rr.g_v + 1), t_w, rho_w, g0w, ow],
        );
        sb.connect(o[0], zassert);
        el_zr = o[1];
    }
    let el_alpha_w = outs[trace.squeezes[piop_i.alpha_fin][0]][0];
    let ea_w = wv(piop_i.eab_v);
    let eb_w = wv(piop_i.eab_v + 1);
    vals.push(rt.a_sum_n);
    let asum_w = sb.public_input();
    vals.push(rt.b_sum_n);
    let bsum_w = sb.public_input();
    let va_w = sb.gate(macs, &[ea_w, asum_w, ow])[0];
    let vb_w = sb.gate(macs, &[eb_w, bsum_w, ow])[0];
    let mut el_lcw = sb.gate(macs, &[va_w, el_alpha_w, vb_w])[0];
    for rr in &piop_i.lc_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        el_lcw = sb.gate(mrslot, &[el_lcw, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }

    cen.push((
        "element PIOP advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    ElementPiopWires {
        el_zr,
        el_alpha_w,
        va_w,
        vb_w,
        asum_w,
        bsum_w,
        el_lcw,
    }
}

/// the anchor EXPECT at real-inner scale
#[allow(clippy::too_many_arguments)]
fn emit_anchor_expect(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    rt: &RealTape<'_>,
    outs: &[Vec<Wire>],
    mp_pws: &[Wire],
    mp_rho2_w: &[Wire],
    mp_sig_w: &[Wire],
    pt_w: &[Wire],
    anc_w: Wire,
    pfslot: SlotId,
    pf_w: usize,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    cen: &mut Vec<(&'static str, usize, usize)>,
) -> (
    Vec<Wire>,
    Vec<Vec<Wire>>,
    Vec<(F128, Wire)>,
    Vec<(F128, Wire)>,
) {
    let trace = &rt.trace;
    let chals = &rt.chals[..];
    let w_rounds = &rt.w_rounds[..];
    let piop_i = &rt.piop_i;
    let gammas_i = &rt.gammas_i[..];
    let m_mp2 = rt.m_mp2;
    let n_log_i = rt.n_log_i;
    let k_cols_i = rt.k_cols_i;
    let macs = cs.macs;
    let spine = cs.spine;
    let t_vals_b = walker_common::baked_inner_t_vals(&rt.zskip);
    let mlv_pw: Vec<(F128, Wire)> = rt
        .zc_rounds_b
        .iter()
        .map(|&(ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let lc_pw: Vec<(F128, Wire)> = rt
        .lc_rounds_b
        .iter()
        .rev()
        .map(|&(_, ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let mut xab_pw: Vec<(F128, Wire)> = vec![lc_pw[0]];
    xab_pw.extend_from_slice(&mlv_pw[1..1 + n_log_i]);
    xab_pw.extend_from_slice(&lc_pw[1..]);
    walker_common::extend_const_coords(&mut xab_pw, &rt.x_ab_n, zw, ow);
    let (outer_ch_b, outer_fin_b) = rt.outer_b;
    let mut xc_pw: Vec<(F128, Wire)> = (0..rt.zc_rounds_b.len())
        .map(|k2| {
            if k2 < walker_common::N_BAKED_T_VALS {
                (t_vals_b[k2], cw(sb, vals, consts, t_vals_b[k2]))
            } else {
                // The exact (row, word) map — a FUSED slice squeeze (the
                // AG r_outer grind) reserves one word per row for the PoW
                // predicate, so the naive 4-per-row split misaddresses.
                let j = k2 - walker_common::N_BAKED_T_VALS;
                (
                    chals[outer_ch_b + j],
                    squeeze_word_wire(outs, trace, outer_fin_b, j),
                )
            }
        })
        .collect();
    walker_common::extend_const_coords(&mut xc_pw, &rt.x_c_n, zw, ow);
    for (i2, (&(nv, _), &xn)) in xab_pw.iter().zip(&rt.x_ab_n).enumerate() {
        assert_eq!(nv, xn, "ab point coord {i2} is the located wire");
    }
    for (i2, (&(nv, _), &xn)) in xc_pw.iter().zip(&rt.x_c_n).enumerate() {
        assert_eq!(nv, xn, "c point coord {i2} is the located wire");
    }

    let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
    let rho_mrg_w: Vec<Wire> = w_rounds
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let mut rinv_n2: Vec<F128> = rho_mrg_n.clone();
    let mut rinv_w: Vec<Wire> = rho_mrg_w.clone();
    let mut ghat = zw;
    for j in 0..RS_SHAT_WORDS {
        if j > 0 {
            let mut lvl_w = Vec::with_capacity(m_mp2);
            for t3 in 0..m_mp2 {
                let y = frob_inv(rinv_n2[t3]);
                rinv_n2[t3] = y;
                vals.push(y);
                let yw = sb.input();
                let d = sb.gate(spine, &[zw, zw, zw, rinv_w[t3], zw, zw, yw, yw, zw])[3];
                sb.connect(d, zassert);
                lvl_w.push(yw);
            }
            rinv_w = lvl_w;
        }
        let factors: Vec<(Wire, Wire)> = rinv_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        let eqj = walker_common::prefix_product(sb, pfslot, pf_w, zw, ow, &factors);
        ghat = sb.gate(spine, &[zw, zw, zw, ghat, zw, zw, mp_pws[j], eqj, zw])[3];
    }
    let e_at_w = {
        let factors: Vec<(Wire, Wire)> = rho_mrg_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        walker_common::prefix_product(sb, pfslot, pf_w, zw, ow, &factors)
    };
    // THE COUNT WIN: everything from here to `connect(anc_w, expect_w)`
    // used to be the W side of the anchor expect — per-run boundary eq
    // products with the child's jagged run boundaries baked as ow/zw
    // (`eqc_w`, THE one site where counts were circuit structure) plus the
    // eq-table dots consuming them (~7.4k rows per region, 6.8% of a
    // node's committed words). All deleted: each statement's raw W arrives
    // as a PUBLISHED CLAIM VALUE on the jagged layout table — the deferred
    // verify's own export, keyed by the child digest — checker-held here
    // and discharged at the ROOT of the accumulation tree. The claim's
    // points are wires this region already carries (σ = the anchor round
    // squeezes, z_cols = statement point wires, γ_pd = squeezes); nothing
    // count-shaped remains in the circuit.
    let mut jag_w: Vec<Wire> = Vec::new();
    // The claims' IDENTITY wires (the points-connect): per claim, in
    // jag_w order, the row wires the merge fold's absorbed words connect
    // to; σ is mp_sig_w, shared.
    let mut jag_row_w: Vec<Vec<Wire>> = Vec::new();
    let alslot = cs.alslot;
    let mut expect_w = zw;
    for (si, xs) in [&xab_pw, &xc_pw].iter().enumerate() {
        let z_row_w: Vec<Wire> = xs[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
        vals.push(rt.jag.rs[si].value);
        let w_st = sb.input();
        jag_w.push(w_st);
        jag_row_w.push(xs[1 + n_log_i..].iter().map(|&(_, w)| w).collect());
        let mut gdp = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        for layer in (0..=m_mp2).rev() {
            let za = if layer < n_log_i { z_row_w[layer] } else { zw };
            let rb = if layer < m_mp2 { mp_rho2_w[layer] } else { zw };
            let mut a_in = gdp.to_vec();
            a_in.extend_from_slice(&[za, rb, mp_sig_w[2 * layer], mp_sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            gdp = [o[0], o[1], o[2], o[3]];
        }
        let coeff = if si == 0 {
            ghat
        } else {
            sb.gate(
                spine,
                &[zw, zw, zw, zw, zw, zw, mp_pws[RS_SHAT_WORDS], ghat, zw],
            )[3]
        };
        let wd = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, w_st, gdp[0], zw])[3];
        expect_w = sb.gate(spine, &[zw, zw, zw, expect_w, zw, zw, coeff, wd, zw])[3];
    }
    for (g_ix, members) in rt.groups_ix.iter().enumerate() {
        // The γ-baked one-hot combo (all this group's gather claims) is ONE
        // published value; each dense (element) member publishes its raw eq
        // value with γ_pd applied by a MAC on the squeeze wire — the
        // exported decomposition reassembled in wires.
        let (combo, dense) = &rt.jag.groups[g_ix];
        let hots: Vec<bool> = members
            .iter()
            .map(|&i2| {
                rt.pd_pts[i2][n_log_i..n_log_i + k_cols_i]
                    .iter()
                    .all(|&x| x == F128::ZERO || x == F128::ONE)
            })
            .collect();
        let mut w_st = match combo {
            Some(c) => {
                vals.push(c.value);
                let w = sb.input();
                jag_w.push(w);
                // The combo's identity: the hot members' γ_pd squeeze
                // wires, member order == the assertion's term order.
                let gws: Vec<Wire> = members
                    .iter()
                    .zip(&hots)
                    .filter(|&(_, &h)| h)
                    .map(|(&i2, _)| {
                        let pd = &gammas_i[i2];
                        squeeze_word_wire(outs, trace, pd.fin, pd.squeeze_offset)
                    })
                    .collect();
                if let JaggedRowWeight::Combo(t) = &c.row {
                    assert_eq!(t.len(), gws.len(), "combo terms == hot members");
                }
                jag_row_w.push(gws);
                w
            }
            None => zw,
        };
        let mut d_it = dense.iter();
        for (&i2, &hot) in members.iter().zip(&hots) {
            if hot {
                assert!(i2 >= 2, "one-hot columns are gather claims");
                continue;
            }
            let (_, c) = d_it.next().expect("a dense entry per non-hot member");
            let pd = &gammas_i[i2];
            let gpd_w = squeeze_word_wire(outs, trace, pd.fin, pd.squeeze_offset);
            vals.push(c.value);
            let d_w = sb.input();
            jag_w.push(d_w);
            // The dense claim's identity: its z_col coordinate wires —
            // constant coords ride zw/ow, the rest the element PIOP's own
            // squeeze wires (the mapping the constructor pinned).
            jag_row_w.push(
                (0..k_cols_i)
                    .map(|jj| {
                        let coord = rt.pd_pts[i2][n_log_i + jj];
                        if coord == F128::ZERO {
                            zw
                        } else if coord == F128::ONE {
                            ow
                        } else if i2 == 0 {
                            outs[trace.squeezes[piop_i.zc_rounds[n_log_i + jj].fin][0]][0]
                        } else {
                            let n_lc = piop_i.lc_rounds.len();
                            outs[trace.squeezes[piop_i.lc_rounds[n_lc - 1 - jj].fin][0]][0]
                        }
                    })
                    .collect(),
            );
            w_st = sb.gate(macs, &[w_st, gpd_w, d_w])[0];
        }
        assert!(d_it.next().is_none(), "every dense entry consumed");
        let mut gdp = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        for layer in (0..=m_mp2).rev() {
            let za = if layer < n_log_i {
                if members[0] >= 2 {
                    pt_w[layer]
                } else {
                    outs[trace.squeezes[piop_i.zc_rounds[layer].fin][0]][0]
                }
            } else {
                zw
            };
            let rb = if layer < m_mp2 { mp_rho2_w[layer] } else { zw };
            let mut a_in = gdp.to_vec();
            a_in.extend_from_slice(&[za, rb, mp_sig_w[2 * layer], mp_sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            gdp = [o[0], o[1], o[2], o[3]];
        }
        let coeff = sb.gate(macs, &[zw, mp_pws[2 * RS_SHAT_WORDS + g_ix], e_at_w])[0];
        let wd = sb.gate(macs, &[zw, w_st, gdp[0]])[0];
        expect_w = sb.gate(macs, &[expect_w, coeff, wd])[0];
    }
    sb.connect(anc_w, expect_w);
    if var("ASSIST_CENSUS").is_ok() {
        eprintln!(
            "ASSIST CENSUS  runs {} of {} cols (k {}), m+1 {} — W side is {} PUBLISHED \
             claim values (the count win); the eqc_w/eq_dot machinery is gone",
            rt.bounds_i.len(),
            1usize << k_cols_i,
            k_cols_i,
            m_mp2 + 1,
            jag_w.len(),
        );
    }

    cen.push((
        "multipoint + anchor expect advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    (jag_w, jag_row_w, mlv_pw, lc_pw)
}

/// What [`emit_assertion_families`] threads forward to the publishes and
/// the region tail.
struct AssertionPublics {
    mat_pub: Vec<Wire>,
    ela_pub: Vec<Wire>,
    mat_eval_w: Vec<(Wire, Wire)>,
    el_eval_w: Vec<(Wire, Wire)>,
    zpartial_ws: Vec<Wire>,
    eps_wires: Vec<Wire>,
}

/// the assertion EMISSIONS (all three families)
#[allow(clippy::too_many_arguments)]
fn emit_assertion_families(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    rt: &RealTape<'_>,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    mlv_pw: &[(F128, Wire)],
    lc_pw: &[(F128, Wire)],
    el_alpha_w: Wire,
    va_w: Wire,
    vb_w: Wire,
    el_lcw: Wire,
    pfslot: SlotId,
    pf_w: usize,
    zw: Wire,
    ow: Wire,
    vals: &mut Vec<F128>,
    cen: &mut Vec<(&'static str, usize, usize)>,
) -> AssertionPublics {
    let trace = &rt.trace;
    let piop_i = &rt.piop_i;
    let gammas_i = &rt.gammas_i[..];
    let n_log_i = rt.n_log_i;
    let mrslot = cs.mrs;
    let spine = cs.spine;
    let bl_alpha_w = outs[trace.squeezes[rt.bl_alpha.1][0]][0];
    let mut mat_pub: Vec<Wire> = vec![bl_alpha_w];
    for &(_, _, fin) in &rt.lc_rounds_b {
        mat_pub.push(outs[trace.squeezes[fin][0]][0]);
    }
    let lc_i = rt.lo.proof.boolean_lincheck();
    let mut mat_eval_w: Vec<(Wire, Wire)> = Vec::new();
    for &(a, b) in &lc_i.matrix_evals {
        vals.push(a);
        let aw = sb.public_input();
        vals.push(b);
        let bw = sb.public_input();
        mat_pub.push(aw);
        mat_pub.push(bw);
        mat_eval_w.push((aw, bw));
    }
    // ROUND 0: the MatrixAssertion equation's remaining data, published —
    // x_inner_rest (batch-major mlv map), x_outer (mlv rounds 1..1+ν),
    // the const-pin betas + their structure-table-bound prefix values,
    // the z_partial
    // words — and the ~20-row BOOLEAN LINCHECK REPLAY, so the published
    // chain end IS the equation's bound target: entry = α·v_a + v_b +
    // Σ β_t·eps_t from absorbed finals and squeeze wires, rounds through
    // the shared MergedRoundGate slot.
    let inner_b = rt.mat_assert.x_inner_rest.len();
    let mat_x_inner_w: Vec<Wire> = (0..inner_b)
        .map(|j| {
            let m = if j == 0 { 0 } else { n_log_i + j };
            mlv_pw[m].1
        })
        .collect();
    mat_pub.extend_from_slice(&mat_x_inner_w);
    for j in 0..n_log_i {
        mat_pub.push(mlv_pw[1 + j].1);
    }
    let mat_rr_w: Vec<Wire> = lc_pw.iter().map(|&(_, w)| w).collect();
    let zpartial_ws: Vec<Wire> = (0..Z_PARTIAL_WORDS).map(|i| wv(rt.zp_v + i)).collect();
    let va_b = wv(rt.zc_finals_v);
    let vb_b = wv(rt.zc_finals_v + 1);
    let mut lcb_w = sb.gate(cs.macs, &[vb_b, bl_alpha_w, va_b])[0];
    let mut eps_wires = Vec::with_capacity(rt.betas_b.len());
    let mut beta_wires = vec![None; rt.lo.shape.registry.num_boolean()];
    for (k, &(_, bfin)) in rt.betas_b.iter().enumerate() {
        let bw = outs[trace.squeezes[bfin][0]][0];
        let type_index = rt.sigma_native.boolean_pins[k].0;
        beta_wires[type_index] = Some(bw);
        vals.push(rt.eps_n[k]);
        let ew = sb.public_input();
        eps_wires.push(ew);
        lcb_w = sb.gate(cs.macs, &[lcb_w, bw, ew])[0];
        mat_pub.push(bw);
        mat_pub.push(ew);
    }
    for &(g_v, _, fin) in &rt.lc_rounds_b {
        let rw = outs[trace.squeezes[fin][0]][0];
        lcb_w = sb.gate(mrslot, &[lcb_w, wv(g_v), wv(g_v + 1), rw])[0];
    }
    emit_boolean_reported_check(
        sb,
        spine,
        pfslot,
        pf_w,
        &rt.lo.shape.registry,
        bl_alpha_w,
        &mat_x_inner_w,
        &mat_rr_w,
        &zpartial_ws,
        &beta_wires,
        &mat_eval_w,
        lcb_w,
        zw,
        ow,
    );
    mat_pub.extend_from_slice(&zpartial_ws);
    mat_pub.push(lcb_w);
    let mut ela_pub: Vec<Wire> = vec![el_alpha_w];
    for rr in &piop_i.zc_rounds {
        ela_pub.push(outs[trace.squeezes[rr.fin][0]][0]);
    }
    for rr in &piop_i.lc_rounds {
        ela_pub.push(outs[trace.squeezes[rr.fin][0]][0]);
    }
    ela_pub.extend_from_slice(&[va_w, vb_w, wv(gammas_i[rt.z_ix].val_v)]);
    let mut el_eval_w: Vec<(Wire, Wire)> = Vec::new();
    for &(a, b) in &rt.el_assert.evals {
        vals.push(a);
        let aw = sb.public_input();
        vals.push(b);
        let bw = sb.public_input();
        ela_pub.push(aw);
        ela_pub.push(bw);
        el_eval_w.push((aw, bw));
    }
    let inner_union = UnionInstance::new(&rt.lo.shape.registry, rt.lo.shape.counts.clone());
    let el_r_con_w: Vec<Wire> = piop_i.zc_rounds[n_log_i..]
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let el_r_col_w: Vec<Wire> = piop_i
        .lc_rounds
        .iter()
        .rev()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    emit_element_reported_check(
        sb,
        spine,
        pfslot,
        pf_w,
        &inner_union,
        el_alpha_w,
        &el_r_con_w,
        &el_r_col_w,
        wv(gammas_i[rt.z_ix].val_v),
        &el_eval_w,
        el_lcw,
        zw,
        ow,
    );

    cen.push((
        "assertion eval advice",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    AssertionPublics {
        mat_pub,
        ela_pub,
        mat_eval_w,
        el_eval_w,
        zpartial_ws,
        eps_wires,
    }
}

/// the publishes, in the swap's recorded order: the query block, then
/// THE TAIL SCHEDULE — the table below IS the real walker's published
/// wire format (a DIFFERENT order from the child's, deliberately),
/// `check_real_child_region`'s independent positional walk is its
/// backstop, and the tape pin holds its `(name, width)` list. The jagged
/// block (raw W claim values in emission order — rs, then per group
/// combo + dense members, the fresh-claim surfaces a merge fold connects
/// to) closes the tail.
#[allow(clippy::too_many_arguments)]
fn emit_publishes_in_swap_order(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    to_publish: &[Vec<Wire>],
    level_accs: &[[Wire; 2]],
    t_final: [Wire; 2],
    tgt_w: Wire,
    runw: Wire,
    resid_pub: &[Vec<[Wire; 2]>],
    inner_w: [Wire; 2],
    sig_w: Wire,
    pt_w: &[Wire],
    el_zr: Wire,
    el_lcw: Wire,
    anc_w: Wire,
    mat_pub: &[Wire],
    ela_pub: &[Wire],
    jag_w: &[Wire],
    cen: &mut Vec<(&'static str, usize, usize)>,
) -> (usize, usize, Vec<(&'static str, usize)>) {
    let pub_base = sb.public_len();
    for a_wires in to_publish {
        for w in a_wires {
            sb.publish(*w);
        }
    }
    for w in level_accs {
        sb.publish(w[0]);
        sb.publish(w[1]);
    }
    cen.push((
        "TAIL: query alphas + native accs",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    let tail = [
        TailEntry::new("spine t_final", t_final.to_vec()),
        TailEntry::new("intake target", vec![tgt_w]),
        TailEntry::new("intake running", vec![runw]),
        TailEntry::new(
            "residual accs",
            resid_pub.iter().flatten().flatten().copied().collect(),
        )
        .with_census("TAIL: chain ends + residual accs"),
        TailEntry::new("residual inner", inner_w.to_vec()),
        TailEntry::new("sigma value", vec![sig_w]),
        TailEntry::new("sigma point", pt_w.to_vec()).with_census("TAIL: sigma + GKR point"),
        TailEntry::new("element zc+lc ends", vec![el_zr, el_lcw]),
        TailEntry::new("anchor end", vec![anc_w]),
        TailEntry::new("matrix assertion publics", mat_pub.to_vec()),
        TailEntry::new("element assertion publics", ela_pub.to_vec())
            .with_census("TAIL: el ends + assertion publics"),
        TailEntry::new("jagged claim values", jag_w.to_vec())
            .with_census("TAIL: jagged claim values"),
    ];
    // Family H is now internal arithmetic.  Its source words are already
    // bound where they enter the transcript/proof stream, so no duplicate
    // public re-exposure or checker-only advice remains.
    let (n_tail, tail_schedule) = publish_tail(sb, &tail, |sb, label| {
        cen.push((label, sb.public_len(), sb.rows_in_slot(cs.macs)));
    });
    (pub_base, n_tail, tail_schedule)
}

/// Walk one emitted REAL child region's public block and hold every
/// published value against the tape's native replicas — the swap test's
/// checker, extracted and base-relative. Returns the entries consumed.
pub(super) fn check_real_child_region(public: &[F128], rt: &RealTape<'_>, r: &RealRegion) -> usize {
    let chals = &rt.chals[..];
    let mut at2 = r.pub_base;
    // The openings bind to the absorbed caps by COPY CONSTRAINT (the
    // in-circuit cap tree) — no per-query publics, no checker walk.
    for (li, lvl) in rt.levels.iter().enumerate() {
        for j in 0..lvl.a_count {
            assert_eq!(public[at2 + j], chals[lvl.a_ch + j], "L{li} alpha {j}");
        }
        at2 += lvl.a_count;
    }
    for (li, want) in rt.native_sums.iter().enumerate() {
        assert_eq!(
            F256::new(public[at2 + 2 * li], public[at2 + 2 * li + 1]),
            *want,
            "L{li} enforced sum matches the native replica"
        );
    }
    let sp_base = at2 + 2 * rt.native_sums.len();
    assert_eq!(
        F256::new(public[sp_base], public[sp_base + 1]),
        rt.t_final_n,
        "the spine's t_r matches the native replay"
    );
    assert_eq!(
        public[sp_base + 2],
        rt.native_target,
        "the computed target is the native two-halves combination"
    );
    assert_eq!(
        public[sp_base + 3],
        rt.native_running,
        "the W-rounds fold the target to the native running claim"
    );
    let inner_n = check_residual_publics(
        public,
        sp_base + 4,
        &rt.levels,
        &rt.geo,
        &rt.w_resid,
        rt.inner_pd_i.ch,
        &observed_f256(&rt.vals_rec, rt.yr_v_i, rt.yr_len),
        chals,
    );
    assert_eq!(
        inner_n, rt.t_final_n,
        "inner == t_r: the real-inner statement closes"
    );
    // The GKR/element/multipoint/anchor identities are COPY CONSTRAINTS —
    // no publics, no checker items; the proof itself carries them.
    let sig_base = sp_base + 4 + 2 * rt.levels.len() * rt.yr_len + 2;
    assert_eq!(
        public[sig_base],
        rt.lo.proof.wiring().gkr.s_sigma_eval,
        "the emitted sigma value is the proof's deferred evaluation"
    );
    let sa = SigmaAssertion {
        rho: public[sig_base + 1..sig_base + 1 + rt.mu_i].to_vec(),
        nu: rt.lo.shape.circuit.cells().nu(),
        base_bits: rt.sigma_native.base_bits,
        masked_id_value: rt.mid_n,
        live_value: rt.live_n,
        value: public[sig_base],
        boolean_pins: rt.sigma_native.boolean_pins.clone(),
        element_constants: rt.sigma_native.element_constants.clone(),
    };
    assert_eq!(sa.rho, rt.sigma_native.rho, "the emitted sigma point");
    assert_eq!(sa.value, rt.sigma_native.value, "the emitted sigma value");
    assert_eq!(sa.nu, rt.sigma_native.nu, "the emitted sigma split");
    assert!(
        sa.check(&rt.lo.shape.circuit),
        "the emitted sigma assertion discharges against the real inner"
    );
    let el_base = sig_base + 1 + rt.mu_i;
    assert_eq!(
        public[el_base], rt.el_run_n,
        "the element zc chain ends at the native running claim"
    );
    assert_eq!(
        public[el_base + 1],
        rt.el_assert.target,
        "the element lc chain ends at the native assertion's target"
    );
    assert_eq!(
        public[el_base + 2],
        rt.anc_end_n,
        "the anchor rounds end at the native claim"
    );
    // The assertion emissions, held against the DEFERRED verify's own
    // assertions — a parent reads the accumulator inputs off the segment.
    let mat_base = el_base + 3;
    assert_eq!(
        public[mat_base], rt.mat_assert.alpha,
        "the emitted matrix alpha is the assertion's"
    );
    for (j, &(_, ch, _)) in rt.lc_rounds_b.iter().enumerate() {
        assert_eq!(
            public[mat_base + 1 + j],
            chals[ch],
            "matrix point coord {j} is the located round wire"
        );
    }
    let lc_i = rt.lo.proof.boolean_lincheck();
    for (j, &(a, b)) in lc_i.matrix_evals.iter().enumerate() {
        assert_eq!(
            (
                public[mat_base + 1 + rt.lc_rounds_b.len() + 2 * j],
                public[mat_base + 1 + rt.lc_rounds_b.len() + 2 * j + 1],
            ),
            (a, b),
            "matrix_evals pair {j} rides as bound advice"
        );
    }
    // ROUND 0's extension: every remaining datum of the MatrixAssertion
    // equation, published and held against the assertion itself.
    let mut mq = mat_base + 1 + rt.lc_rounds_b.len() + 2 * lc_i.matrix_evals.len();
    for (j, &x) in rt.mat_assert.x_inner_rest.iter().enumerate() {
        assert_eq!(public[mq + j], x, "x_inner_rest {j} published");
    }
    mq += rt.mat_assert.x_inner_rest.len();
    for j in 0..rt.n_log_i {
        assert_eq!(
            public[mq + j],
            chals[rt.zc_rounds_b[1 + j].0],
            "x_outer {j} published"
        );
    }
    mq += rt.n_log_i;
    for (k, &(bch, _)) in rt.betas_b.iter().enumerate() {
        assert_eq!(public[mq], chals[bch], "beta {k} published");
        assert_eq!(public[mq + 1], rt.eps_n[k], "eps {k} advice");
        mq += 2;
    }
    for (j, &z) in rt.mat_assert.z_partial.iter().enumerate() {
        assert_eq!(public[mq + j], z, "z_partial {j} published");
    }
    // The checker's offsets stay LITERAL on purpose: this walk is the
    // backstop that falsifies a wrong schedule constant, so it must not
    // share one with the emitter.
    mq += 64;
    assert_eq!(
        public[mq], rt.mat_assert.target,
        "the in-circuit boolean lc replay ends at the assertion's target"
    );
    assert_eq!(
        mq + 1,
        mat_base + r.n_mat_pub,
        "the mat block walk is complete"
    );
    let ela_base = mat_base + r.n_mat_pub;
    assert_eq!(
        public[ela_base], rt.el_assert.alpha,
        "the emitted element alpha is the assertion's"
    );
    let n_er = rt.piop_i.zc_rounds.len() + rt.piop_i.lc_rounds.len();
    for (j, rr) in rt
        .piop_i
        .zc_rounds
        .iter()
        .chain(rt.piop_i.lc_rounds.iter())
        .enumerate()
    {
        assert_eq!(
            public[ela_base + 1 + j],
            chals[rr.ch],
            "element point coord {j} is the located round wire"
        );
    }
    assert_eq!(
        public[ela_base + 1 + n_er],
        rt.vals_rec[rt.piop_i.eab_v] + rt.a_sum_n,
        "the emitted va is the strip-derived value"
    );
    assert_eq!(
        public[ela_base + 1 + n_er + 1],
        rt.vals_rec[rt.piop_i.eab_v + 1] + rt.b_sum_n,
        "the emitted vb is the strip-derived value"
    );
    assert_eq!(
        public[ela_base + 1 + n_er + 2],
        rt.el_assert.z_eval,
        "the emitted z_eval is the assertion's"
    );
    for (j, &(a, b)) in rt.el_assert.evals.iter().enumerate() {
        assert_eq!(
            (
                public[ela_base + 1 + n_er + 3 + 2 * j],
                public[ela_base + 1 + n_er + 3 + 2 * j + 1],
            ),
            (a, b),
            "per-slot eval pair {j} rides as bound advice"
        );
    }
    let mut fq = ela_base + r.n_ela_pub;
    // The jagged assertion's value surfaces (the count win), in emission
    // order — each the deferred export's own raw claim value; the full
    // claims discharge against the child's layout at the root.
    {
        let mut expect_vals: Vec<F128> = rt.jag.rs.iter().map(|c| c.value).collect();
        for (combo, dense) in &rt.jag.groups {
            if let Some(c) = combo {
                expect_vals.push(c.value);
            }
            for (_, c) in dense {
                expect_vals.push(c.value);
            }
        }
        for (j, want) in expect_vals.iter().enumerate() {
            assert_eq!(
                public[fq + j],
                *want,
                "jagged claim value {j} matches the deferred export"
            );
        }
        fq += expect_vals.len();
    }
    assert_eq!(
        fq,
        r.pub_base + r.n_query_pub + r.n_tail,
        "the jagged publics close the region's tail"
    );
    r.n_query_pub + r.n_tail
}
