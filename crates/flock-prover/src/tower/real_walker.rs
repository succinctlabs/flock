use super::*;

// **THE SWAP, step 1 — mvp9's outer becomes the inner.** The leaf-outer
// circuit proof (the first real recursion node, BLAKE3/BLAKE3 from the
// shared builder) is natively verified under a RecordingChallenger and its
// tape walked by the SAME machinery mvp10's assembly consumes:
// parse_open_levels, the region label map, level_geometry (native capped
// paths + enforced-sum replicas per level), and the R=2 + P multipoint
// schedule replayed to the anchor's claimed v — pinned before any
// assembly, the step-1 pattern every phase ran. What it establishes about
// the REAL inner's shape: the element PIOP parses at multi-slot scale, the
// packed-direct claims are the element (c, lc) pair plus every wiring
// gather, the R=2 + P>0 schedule holds, and the committed lane count is
// once more an arbitrary integer.
// ---------------------------------------------------------------------------
// The REAL child: the leaf outer's deferred verifier as a reusable region
// (the swap test's assembly, extracted so the 2→1 merge node can
// instantiate a real child-tape region per child — the emit_child_region
// precedent at leaf-outer scale)
// ---------------------------------------------------------------------------

/// One recorded REAL-child verification (the leaf outer as inner), parsed:
/// the tape pinned op-for-op, every region located, and every native
/// replica the emitter and checker consume — the swap test's step-1 walk as
/// a reusable unit. `new` runs the RECORDING verify itself, so every
/// instantiation re-asserts the whole map on that child's tape.
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
    pub(super) trace: crate::r1cs_hashes::fs_chain::FsChainTrace,
    pub(super) stream: flock_core::transcript_record::Stream,
    pub(super) bytes: Vec<u8>,
    /// The fork's four cross-link wires ([`MergedChain::cross`]).
    pub(super) cross: Vec<Option<(usize, usize)>>,
    pub(super) b3_rows: usize,
    pub(super) spread_w: usize,
    // located regions
    pub(super) gkr: GkrRec,
    pub(super) piop_i: PiopRec,
    pub(super) start_v_i: usize,
    pub(super) gammas_i: Vec<PdRec>,
    pub(super) w_rounds: Vec<RoundRec>,
    pub(super) w_resid: Vec<RoundRec>,
    pub(super) mp_i: MpRec,
    pub(super) inner_pd_i: InnerPd,
    pub(super) yr_v_i: usize,
    pub(super) yr_len: usize,
    pub(super) levels: Vec<OpenLevel>,
    pub(super) lvl_src: Vec<(&'p [[u8; 32]], &'p Vec<Vec<F128>>, &'p Vec<[u8; 32]>)>,
    pub(super) geo: Vec<Lvl>,
    pub(super) native_sums: Vec<F256>,
    /// The grinding ops: (fin ordinal, payload ordinal, bits).
    pub(super) pows: Vec<(usize, usize, u32)>,
    pub(super) n_gather: usize,
    /// The child cell space's public-slot count — the recombination's tail.
    pub(super) n_pub_slots_c: usize,
    // the boolean PIOP's round ordinals ((ch, fin) pairs) + surfaces
    pub(super) zc_rounds_b: Vec<(usize, usize)>,
    pub(super) outer_b: (usize, usize),
    #[allow(dead_code)] // The r_outer slice length — wall-4 shape data.
    pub(super) outer_len: usize,
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
    pub(super) mat_assert: flock_core::lincheck::MatrixAssertion,
    pub(super) el_assert: flock_core::element_r1cs::union::ElementAssertion,
    pub(super) sigma_native: flock_core::circuit::SigmaAssertion,
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
    #[allow(dead_code)]
    // The layout's run→column map — the eqc_w era's consumer; kept as shape data.
    pub(super) run_of: Vec<usize>,
    pub(super) x_ab_n: Vec<F128>,
    pub(super) x_c_n: Vec<F128>,
    pub(super) groups_ix: Vec<Vec<usize>>,
    /// Derived pd claim points (merged-open v1), pinned order
    /// [element c, element lc, gathers in cell-slot order].
    pub(super) pd_pts: Vec<Vec<F128>>,
    /// The deferred verify's jagged-layout export (the count win) — the
    /// independent reference for the W-value publics, tied to the native
    /// expect replica in the constructor.
    pub(super) jag: flock_core::matrix_fold::JaggedAssertion,
}

impl<'p> RealTape<'p> {
    pub(super) fn new(lo: &'p LeafOuter, domain: &'static [u8]) -> Self {
        use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};

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
                    find(b"flock-ag-skip-v1").is_empty(),
                    "an RS tape carries no AG-skip region"
                );
                find(b"flock-zerocheck-v0")
            }
            MixedProof::Ag(_) => {
                assert!(
                    find(b"flock-zerocheck-v0").is_empty(),
                    "an AG tape carries no RS zerocheck region"
                );
                find(b"flock-ag-skip-v1")
            }
        };
        let lc_l = find(b"flock-lincheck-v0");
        let elzc_l = find(b"flock-element-union-zc-v0");
        let el_l = find(b"flock-element-union-lc-v0");
        let gkr_l = find(b"flock-product-gkr-batched-v0");
        let mo_l = find(b"flock-merged-open-v1");
        let rs_l = find(b"flock-ring-switch-v0");
        let mp_l = find(b"flock-multipoint-twisted-v1");
        let fa_l = find(b"flock-frobenius-assist-v0");
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
            256 + n_p,
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

        // ---- the wiring GKR walk (the mvp10 walker, real-inner layers) ----
        // Records every ordinal the transcription wires against and replays
        // the whole layer recursion natively in lockstep, input checks
        // included — the rhs consuming the DEFERRED s_sigma from the proof.
        let gkr_rec = {
            let gkr = &lo.proof.wiring().gkr;
            let mut i = gkr_l[0] + 1;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            assert!(matches!(ops[i], Op::SqueezeScalar), "gkr alpha");
            let (_, c_alpha) = vc_at(i);
            let alpha_fin = fin_at(i);
            i += 1;
            assert!(matches!(ops[i], Op::SqueezeScalar), "gkr beta");
            let beta_fin = fin_at(i);
            i += 1;
            assert!(matches!(ops[i], Op::ObserveScalar), "top lhs");
            let (tv, _) = vc_at(i);
            assert_eq!(vals_rec[tv], gkr.top_lhs, "top_lhs on the stream");
            assert_eq!(vals_rec[tv + 1], gkr.top_rhs, "top_rhs on the stream");
            assert_eq!(gkr.top_lhs, gkr.top_rhs, "the grand products agree");
            i += 2;
            let (mut claim_l, mut claim_r) = (gkr.top_lhs, gkr.top_rhs);
            let mut r_pt: Vec<F128> = Vec::new();
            let mut lrecs: Vec<GkrLayerRec> = Vec::new();
            for (k, layer) in gkr.layers.iter().enumerate() {
                assert_eq!(layer.rounds.len(), k, "layer {k} has k rounds");
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
                assert!(matches!(ops[i], Op::SqueezeScalar), "layer {k} lambda");
                let (_, lc2) = vc_at(i);
                let lambda = chals[lc2];
                let lam_fin = fin_at(i);
                i += 1;
                let mut c_run = claim_l + lambda * claim_r;
                let mut r_prime = Vec::with_capacity(k + 1);
                let mut rrecs: Vec<(usize, usize)> = Vec::new();
                let mut g0s: Vec<F128> = Vec::new();
                for (t2, &(g1, gi)) in layer.rounds.iter().enumerate() {
                    assert!(matches!(ops[i], Op::ObserveScalar), "round obs g1");
                    let (gv, _) = vc_at(i);
                    assert_eq!(vals_rec[gv], g1, "layer {k} round {t2} g1");
                    assert_eq!(vals_rec[gv + 1], gi, "layer {k} round {t2} g_inf");
                    let mut rho_i = i + 2;
                    while matches!(ops[rho_i], Op::Pow { .. }) {
                        rho_i += 1;
                    }
                    assert!(matches!(ops[rho_i], Op::SqueezeScalar), "round rho");
                    let (_, rc2) = vc_at(rho_i);
                    let rho = chals[rc2];
                    rrecs.push((gv, fin_at(rho_i)));
                    i = rho_i + 1;
                    let r_eq = r_pt[t2];
                    let g0 = (c_run + r_eq * g1) * (F128::ONE + r_eq).inv();
                    g0s.push(g0);
                    c_run = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
                    r_prime.push(rho);
                }
                let (vv, _) = vc_at(i);
                for (j, want) in [layer.vl0, layer.vl1, layer.vr0, layer.vr1]
                    .into_iter()
                    .enumerate()
                {
                    assert!(matches!(ops[i], Op::ObserveScalar), "layer value obs");
                    assert_eq!(vals_rec[vv + j], want, "layer {k} value {j}");
                    i += 1;
                }
                assert_eq!(
                    c_run,
                    layer.vl0 * layer.vl1 + lambda * (layer.vr0 * layer.vr1),
                    "layer {k} closes"
                );
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
                assert!(matches!(ops[i], Op::SqueezeScalar), "layer {k} c_k");
                let (_, cc2) = vc_at(i);
                let c_k = chals[cc2];
                let ck_fin = fin_at(i);
                i += 1;
                claim_l = (F128::ONE + c_k) * layer.vl0 + c_k * layer.vl1;
                claim_r = (F128::ONE + c_k) * layer.vr0 + c_k * layer.vr1;
                r_prime.push(c_k);
                r_pt = r_prime;
                lrecs.push(GkrLayerRec {
                    lam_fin,
                    rounds: rrecs,
                    g0s,
                    v_v: vv,
                    ck_fin,
                });
            }
            let mu_i = lo.shape.circuit.cells().mu();
            assert_eq!(r_pt.len(), mu_i, "the GKR point spans the inner cell space");
            let alpha2 = chals[c_alpha];
            let beta2 = chals[c_alpha + 1];
            let basis = flock_core::product_gkr::s_id_basis(mu_i);
            // The LIVE-IDENTITY padding: leaves are w + α·(live⊙s_id) +
            // (β+1)·live + 1 (dead cells = 1), so the input checks carry
            // the masked closed forms.
            let mask_w = lo.shape.circuit.live_mask();
            let tail_w2 = (beta2 + F128::ONE) * mask_w.live_eval(&r_pt) + F128::ONE;
            assert_eq!(
                claim_l,
                gkr.f_eval + alpha2 * mask_w.masked_id_eval(&basis, &r_pt) + tail_w2,
                "lhs input check replays (masked)"
            );
            assert_eq!(
                claim_r,
                gkr.g_eval + alpha2 * gkr.s_sigma_eval + tail_w2,
                "rhs input check replays with the DEFERRED (masked) sigma value"
            );
            let (fv, _) = vc_at(i);
            assert!(matches!(ops[i], Op::ObserveScalar), "f_eval obs");
            assert_eq!(vals_rec[fv], gkr.f_eval, "f_eval on the stream");
            assert_eq!(vals_rec[fv + 1], gkr.g_eval, "g_eval on the stream");
            assert_eq!(vals_rec[fv + 2], gkr.s_sigma_eval, "s_sigma on the stream");
            GkrRec {
                alpha_fin,
                beta_fin,
                top_v: tv,
                layers: lrecs,
                fgs_v: fv,
                r_pt,
            }
        };
        // ROUND 2: the H(publics) region's rows — a chunk chain per 1 KiB
        // leaf of the child's public segment plus the left-fold parents.
        let h_rows = lo.public.len().div_ceil(4) + 2 * lo.public.len().div_ceil(64);

        // ---- the chain materials ----
        // The transcript is FORKED (the wiring runs on its own chain);
        // `merge_chain` splices the child's rows in at the fork point and
        // hands back one linear numbering plus the four cross-link wires.
        let MergedChain {
            stream,
            bytes,
            trace,
            cross,
            ..
        } = merge_chain(
            t_shape.ops(),
            &t_shape.stream_words_duplex(domain),
            rec.values(),
            rec.payloads(),
        );
        assert_chain_replays(&ops, &trace, &chals);
        let b3_rows = trace.rows.len() + h_rows + query_phase_b3_rows(&geo);
        if std::env::var("B3_CENSUS").is_ok() {
            let parents = trace.block_offsets.iter().filter(|o| o.is_none()).count();
            let blocks = trace.rows.len() - parents;
            let mut pow_by_bits = std::collections::BTreeMap::<u32, usize>::new();
            for op in &ops {
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
            for g in geo.iter() {
                let (leaf, path, cap) = level_query_phase_b3_rows(g);
                eprintln!(
                    "    level: q {} depth {} row_words {} -> leaf {} + path {} + cap {}",
                    g.q, g.depth, g.raw_row_words, leaf, path, cap,
                );
            }
            // CHAIN DECOMPOSITION + an independent row-count model of the
            // duplex discipline (transcript-v3), asserted against the
            // sponge trace: a squeeze row absorbs the pending partial
            // block as its MESSAGE, mutates cv, and has no header word.
            {
                let pad16 = |n: usize| n.div_ceil(16) * 16;
                let (mut hdr_w, mut pay_w, mut n_obs, mut n_sq) = (0usize, 0usize, 0usize, 0usize);
                for op in ops.iter() {
                    match op {
                        Op::Label(l) => {
                            hdr_w += 1;
                            pay_w += pad16(l.len()) / 16;
                            n_obs += 1;
                        }
                        Op::ObserveScalar => {
                            hdr_w += 1;
                            pay_w += 1;
                            n_obs += 1;
                        }
                        Op::ObserveSlice(n) => {
                            hdr_w += 1;
                            pay_w += n;
                            n_obs += 1;
                        }
                        Op::ObserveBytes(len) => {
                            hdr_w += 1;
                            pay_w += pad16(*len) / 16;
                            n_obs += 1;
                        }
                        Op::Forked { .. } | Op::Merge { .. } => {}
                        Op::Pow { .. } => {
                            pay_w += 1;
                        }
                        Op::LegacyPow { .. } => {
                            n_sq += 1;
                        }
                        Op::SqueezeScalar | Op::SqueezeSlice(_) => {
                            n_sq += 1;
                        }
                    }
                }
                let v3_rows =
                    duplex_row_count_model(t_shape.ops(), &t_shape.stream_words_duplex(domain));
                eprintln!(
                    "  [chain census] ops {} (obs {} / sq {}) | header words {} ({} B) | payload words {} | duplex rows {}",
                    ops.len(),
                    n_obs,
                    n_sq,
                    hdr_w,
                    16 * hdr_w,
                    pay_w,
                    trace.rows.len(),
                );
                assert_eq!(
                    v3_rows,
                    trace.rows.len(),
                    "the duplex row model diverged from the sponge trace"
                );
            }
        }
        let spread_w = geo.iter().map(|g| g.depth).max().unwrap().max(1);
        // Recursive caps are PROOF BODY — the in-circuit cap trees bind them
        // (chain + root connects, nothing checker-read); only the L0 cap —
        // the commitment — stays a statement public.
        let cap_pays = cap_payloads(&stream, &bytes, &lvl_src);
        for &p in &cap_pays[1..] {
            pub_payloads[p] = false;
        }

        // The PoW grinding ops, located (the mvp7 machinery).
        let pows: Vec<(usize, usize, u32)> = {
            let mut out = Vec::new();
            let (mut fin, mut pay) = (0usize, 0usize);
            for op in &ops {
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

        // ---- the rs×2 regions + the two-halves target, natively ----
        let (rs_recs2, rs_gam_ch2, rs_gam_fin2) = {
            let mut i2 = rs_l[0];
            // (s_hat_v ordinal, r_dprime fin, r_dprime ch) per region.
            let mut recs: Vec<(usize, usize, usize)> = Vec::new();
            for k in 0..2 {
                assert!(
                    matches!(&ops[i2], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0"),
                    "rs region {k}"
                );
                i2 += 1;
                assert!(matches!(ops[i2], Op::ObserveSlice(128)), "s_hat_v slice");
                let (sv, _) = vc_at(i2);
                assert_eq!(
                    &vals_rec[sv..sv + 128],
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
            for pd in &gammas_i {
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
            use flock_core::pcs::ring_switch as rsw;
            use flock_core::zerocheck::univariate_skip::build_eq;
            let gs: Vec<F128> = rs_gam_ch2.iter().map(|&ch| chals[ch]).collect();
            let mut rs_half = F128::ZERO;
            let mut coeffs: Vec<Vec<F128>> = Vec::new();
            for (k, &(sv, _, rc)) in rs_recs2.iter().enumerate() {
                let shv = &vals_rec[sv..sv + 128];
                let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
                let eq = build_eq(&rdp);
                rs_half += gs[k] * rsw::inner_product(&rsw::tensor_algebra_transpose(shv), &eq);
                let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
                coeffs.push(rsw::linearized_coefficients(&rsw::build_fold_byte_table(
                    &scaled,
                )));
            }
            let mut target = rs_half;
            for pd in &gammas_i {
                target += chals[pd.ch] * vals_rec[pd.val_v];
            }
            let mut running = target;
            for rr in &w_rounds {
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

        // ---- the spine's native quad replay ----
        let t_final_n = replay_ligerito_spine256(
            &levels,
            &vals_rec,
            &chals,
            start_v_i,
            chals[inner_pd_i.ch] * vals_rec[inner_pd_i.q_v],
            &native_sums,
        );

        // ---- the residual pairing's rotation (lane-major inners) ----
        // A pow2-lane inner (row_words == lanes — e.g. the m28-k4 slim node
        // whose 16-of-16 lanes make the commit exactly full) takes the
        // IDENTITY pairing, same as the native side's rotate gate and
        // ChildTape's conditional.
        let yr_len = lo.proof.pcs_open().inner.ligerito.final_proof.yr.len() / 2;
        let lane_major = geo[0].row_words < geo[0].lanes;
        let w_resid: Vec<RoundRec> = if lane_major {
            let k_rot = w_rounds.len() - levels[0].fold_fins.len();
            let mut v = w_rounds[k_rot..].to_vec();
            v.extend_from_slice(&w_rounds[..k_rot]);
            v
        } else {
            w_rounds.to_vec()
        };

        // ---- the element PIOP's natives: the GENERAL strip + g0 chain ----
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
            let slots_el = flock_core::element_r1cs::union::region_slots(&union_i);
            let nu_i = union_i.n_log();
            let mut a_sum = F128::ZERO;
            let mut b_sum = F128::ZERO;
            for s in &slots_el {
                let kappa = s.ty.kappa();
                let eq_con =
                    flock_core::zerocheck::univariate_skip::build_eq(&el_assert.r_con[..kappa]);
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

        // ---- the anchor's native endpoint ----
        let anc_end_n = {
            let mut t2 = vals_rec[mp_i.anchor_v];
            for rr in &mp_i.anchor_rounds {
                let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
                let r3 = chals[rr.ch];
                let g0 = t2 + g1;
                t2 = g0 + (g1 + g0 + gi) * r3 + gi * r3 * r3;
            }
            t2
        };

        // ---- the GKR input-check advice (masked M̂ and livê) ----
        let mu_i = lo.shape.circuit.cells().mu();
        let (mid_n, live_n) = {
            let basis_i = flock_core::product_gkr::s_id_basis(mu_i);
            let mask_i = lo.shape.circuit.live_mask();
            (
                mask_i.masked_id_eval(&basis_i, &gkr_rec.r_pt),
                mask_i.live_eval(&gkr_rec.r_pt),
            )
        };

        // ---- the anchor-expect geometry + boolean locate + replica ----
        let m_mp2 = mp_i.rounds.len();
        assert_eq!(
            mp_i.anchor_rounds.len(),
            2 * (m_mp2 + 1),
            "sigma spans the anchor layers"
        );
        assert_eq!(w_rounds.len(), m_mp2, "merged rho spans the dense domain");
        let n_log_i = union_i.n_log();
        let params_i = flock_core::pcs::jagged::JaggedParams::from_heights(
            &union_i.jagged_heights(),
            n_log_i,
            m_mp2,
        );
        let k_cols_i = params_i.k;
        // ROUND 4: the recombination + f == g, replayed from located words
        // (the emitter binds these; until it landed they rode only this
        // constructor's scaffolding verify).
        let n_pub_slots_c = pin_recombination(
            lo.shape.circuit.cells(),
            n_log_i,
            &lo.public,
            &lo.proof.wiring().gather,
            &gammas_i,
            2,
            &vals_rec,
            &gkr_rec.r_pt,
            gkr_rec.fgs_v,
        );
        let bounds_i = flock_core::pcs::jagged::assist_boundaries(&params_i);
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
        let run_of: Vec<usize> = {
            let mut v = Vec::with_capacity(1usize << k_cols_i);
            for (r3, &(_, _, len)) in bounds_i.iter().enumerate() {
                v.extend(std::iter::repeat_n(r3, len as usize));
            }
            assert_eq!(v.len(), 1usize << k_cols_i, "runs partition the columns");
            v
        };
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
            (outer_ch_b, outer_fin_b, outer_len),
            bl_alpha,
            betas_b,
            zc_finals_v,
            lc_rounds_b,
            zp_v,
        ) = {
            let mut i2 = zc_l[0] + 1;
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
                        matches!(&ops[i2], Op::Label(l) if l.as_slice() == b"flock-ag-skip-r1-point"),
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
                        matches!(&ops[i2], Op::Label(l) if l.as_slice() == b"flock-ag-skip-r1-nonce"),
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
            assert_eq!(i2, lc_l[0], "the zerocheck runs straight into the lincheck");
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
            assert!(matches!(ops[i2], Op::ObserveSlice(64)), "z_partial slice");
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
                        flock_core::lincheck::SkipPoint::Ag(pt),
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
                let e = flock_core::product_gkr::LiveMask::eq_prefix_sum(
                    &x_outer_n,
                    union_i.counts()[t],
                );
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
                            *x = flock_core::pcs::jagged::frob_inv(*x);
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
            let sparse_t = flock_core::pcs::jagged::assist_sparse_transitions();
            let dp_native = |z_row: &[F128]| -> F128 {
                let mut gdp = [F128::ZERO; 4];
                gdp[flock_core::pcs::jagged::STATE_SUCCESS] = F128::ONE;
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
                    let eq4 = flock_core::lincheck::build_eq_table(&[za, rb]);
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
                gdp[flock_core::pcs::jagged::STATE_INITIAL]
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
            outer_b: (outer_ch_b, outer_fin_b),
            outer_len,
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
            run_of,
            x_ab_n,
            x_c_n,
            groups_ix,
            pd_pts: pd_pts_n,
            jag: jag_assert,
        }
    }
}

/// What one emitted REAL child region hands back: where its public block
/// starts, the walk counts, and the assertion-emission wires the 2→1 merge
/// node CONNECTS the fold region's claim words to — all three families.
pub(super) struct RealRegion {
    pub(super) pub_base: usize,
    pub(super) n_query_pub: usize,
    pub(super) n_tail: usize,
    pub(super) n_mat_pub: usize,
    pub(super) n_ela_pub: usize,
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
    #[allow(dead_code)]
    pub(super) pt_w: Vec<Wire>,
    /// element: every zc/lc round rho (round order) and the per-slot eval
    /// advice pairs (bound publics — connectable, unlike the minimal child).
    #[allow(dead_code)]
    pub(super) el_zc_rho_w: Vec<Wire>,
    #[allow(dead_code)]
    pub(super) el_lc_rho_w: Vec<Wire>,
    #[allow(dead_code)]
    pub(super) el_eval_w: Vec<(Wire, Wire)>,
    /// boolean: the zc mlv / lc round rhos (round order), the absorbed
    /// z_partial words, and the per-type matrix_evals advice pairs.
    #[allow(dead_code)]
    pub(super) b_mlv_w: Vec<Wire>,
    #[allow(dead_code)]
    pub(super) b_lc_w: Vec<Wire>,
    #[allow(dead_code)]
    pub(super) b_zpartial_w: Vec<Wire>,
    #[allow(dead_code)]
    pub(super) mat_eval_w: Vec<(Wire, Wire)>,
    /// The residual close-out's prefix slot (and width).
    #[allow(dead_code)]
    pub(super) pf: (flock_core::circuit::builder::SlotId, usize),
    /// The child's PUBLIC SEGMENT as witness wires — the app-statement
    /// plumbing (hash-chain adjacency) reads through these.
    pub(super) child_pub_w: Vec<Wire>,
    /// The child's own CIRCUIT DIGEST, absorbed by its statement binding
    /// (payload 3) — two public words. This is the KEY its sigma and
    /// jagged claims fold under, and the spine's match-gate compares it
    /// against the key an inherited entry was published with.
    #[allow(dead_code)]
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
    use std::sync::OnceLock;
    static DECOMP: OnceLock<(F128, F128, [F128; 7])> = OnceLock::new();
    *DECOMP.get_or_init(|| {
        let minv = flock_core::pcs::ring_switch::moore_inverse();
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
    tile: flock_core::circuit::builder::SlotId,
    macs: flock_core::circuit::builder::SlotId,
    fold_macs: flock_core::circuit::builder::SlotId,
    spine: flock_core::circuit::builder::SlotId,
    spine256: flock_core::circuit::builder::SlotId,
    mac256: flock_core::circuit::builder::SlotId,
    row_capacity: usize,
    shv: &[Vec<Wire>; 2],
    values: &[Vec<Wire>; 2],
    r_dprime: &[Vec<Wire>; 2],
    gamma: [Wire; 2],
    pfslot: flock_core::circuit::builder::SlotId,
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
        input.extend(std::iter::repeat_n(zw, pf_w - a.len()));
        input.extend_from_slice(b);
        input.extend(std::iter::repeat_n(zw, pf_w - b.len()));
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
    let mut eq_w: [Vec<Wire>; 2] = std::array::from_fn(|_| Vec::with_capacity(128));
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
    let minv = flock_core::pcs::ring_switch::moore_inverse();
    // Only the eight orbit seeds are fixed publics.  Successive Frobenius
    // powers are derived by the existing spine's squaring cell, saving more
    // than one thousand public constants at a two-child node.
    let mut g0_j_w = cw(sb, vals, consts, g0_j);
    let mut corrections_j_w: [Wire; 7] =
        std::array::from_fn(|t| cw(sb, vals, consts, corrections_j[t]));
    let mut coeff_w: [Vec<Wire>; 2] = std::array::from_fn(|_| Vec::with_capacity(128));
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
        k_j = k_j * k_j + flock_core::field::QUADRATIC_NONRESIDUE;
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
    b3_slot: flock_core::circuit::builder::SlotId,
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
    let chals = &rt.chals[..];
    let levels = &rt.levels[..];
    let geo = &rt.geo[..];
    let w_rounds = &rt.w_rounds[..];
    let mp_i = &rt.mp_i;
    let inner_pd_i = &rt.inner_pd_i;
    let piop_i = &rt.piop_i;
    let gammas_i = &rt.gammas_i[..];
    let r = levels.len() - 1;
    let m_mp2 = rt.m_mp2;
    let n_log_i = rt.n_log_i;
    let k_cols_i = rt.k_cols_i;

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
    let iv_w = pack8(&crate::r1cs_hashes::fs_chain::IV);
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
    if std::env::var("PUB_CENSUS").is_ok() {
        let pay_pub: usize = stream
            .words
            .iter()
            .enumerate()
            .filter(|(wi, w)| {
                matches!(w, flock_core::transcript_record::StreamWord::Bytes { payload, .. }
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
                .position(|w| matches!(w, flock_core::transcript_record::StreamWord::Bytes { payload, .. } if *payload == pay))
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

    // ---- ROUND 2: the H(publics) region (v2 statement binding) ----
    // Payload 4 of the circuit binding is the 32-byte publics digest; the
    // child's public words themselves are witness, bound here. The returned
    // wires ARE the child's public segment — the recombination folds them.
    // Payload 3 is the child's CIRCUIT DIGEST (`bind_statement_circuit`'s
    // order: registry, counts, cap, circuit, publics) — the FOLD KEY this
    // child's claims belong under (wall 3), exported so the fold region's
    // absorbed group digest binds to the circuit actually verified here.
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
        chals,
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
    // ---- intake W-rounds, spine, residual ----
    let mut vmap: Vec<Option<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let flock_core::transcript_record::StreamWord::Value(vi) = *w {
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
    let mrslot = cs.mrs;
    let spine = cs.spine;
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
    let gpw = outs[trace.squeezes[inner_pd_i.fin][0]][0];
    let z2 = [zw, zw];
    let tw0 = emit_spine256(
        sb,
        spine256,
        z2,
        z2,
        z2,
        z2,
        z2,
        z2,
        [wv(inner_pd_i.q_v), zw],
        gpw,
        z2,
    );
    let mut tsp = tw0[3];
    for od in &levels[0].initial_ood {
        let bw = outs[trace.squeezes[od.beta_fin][0]][0];
        tsp = emit_spine256(
            sb,
            spine256,
            z2,
            z2,
            z2,
            tsp,
            z2,
            z2,
            [wv(od.y_v), zw],
            bw,
            z2,
        )[3];
    }
    let st = emit_spine256(
        sb,
        spine256,
        z2,
        z2,
        z2,
        z2,
        [wv(rt.start_v_i), wv(rt.start_v_i + 1)],
        [wv(rt.start_v_i + 2), wv(rt.start_v_i + 3)],
        tsp,
        ow,
        z2,
    );
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = [
                squeeze_word_wire(&outs, trace, lvl.fold_fins[j], 0),
                squeeze_word_wire(&outs, trace, lvl.fold_fins[j], 1),
            ];
            let ev = emit_spine256(sb, spine256, qc, qb, qa, z2, z2, z2, z2, zw, rw);
            tsp = ev[4];
            let bld = emit_spine256(
                sb,
                spine256,
                z2,
                z2,
                z2,
                z2,
                [wv(mv), wv(mv + 1)],
                [wv(mv + 2), wv(mv + 3)],
                tsp,
                ow,
                z2,
            );
            (qc, qb, qa) = (bld[0], bld[1], bld[2]);
        }
        if li < r {
            for od in &lvl.ood {
                let bw = outs[trace.squeezes[od.beta_fin][0]][0];
                let f = emit_spine256(
                    sb,
                    spine256,
                    qc,
                    qb,
                    qa,
                    tsp,
                    [wv(od.intro_v), wv(od.intro_v + 1)],
                    [wv(od.intro_v + 2), wv(od.intro_v + 3)],
                    [wv(od.y_v), zw],
                    bw,
                    z2,
                );
                (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
            }
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = emit_spine256(
                sb,
                spine256,
                qc,
                qb,
                qa,
                tsp,
                [wv(lvl.intro_v), wv(lvl.intro_v + 1)],
                [wv(lvl.intro_v + 2), wv(lvl.intro_v + 3)],
                level_accs[li],
                bw,
                z2,
            );
            (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
        } else {
            let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
            let f = emit_spine256(
                sb,
                spine256,
                z2,
                z2,
                z2,
                tsp,
                z2,
                z2,
                level_accs[li],
                bw,
                z2,
            );
            tsp = f[3];
        }
    }
    let t_final = tsp;

    // The RESIDUAL region via the shared emitter (lane-major rotation).
    let yr_wires: Vec<[Wire; 2]> = (0..rt.yr_len)
        .map(|y| [wv(rt.yr_v_i + 2 * y), wv(rt.yr_v_i + 2 * y + 1)])
        .collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        sb,
        &mut cs.resid,
        levels,
        geo,
        &to_publish,
        &query_positions,
        &rt.w_resid,
        inner_pd_i.fin,
        &yr_wires,
        trace,
        &outs,
        zw,
        ow,
    );
    // THE CLOSURE, in-circuit: inner == t_r as a copy constraint.
    sb.connect(inner_w[0], t_final[0]);
    sb.connect(inner_w[1], t_final[1]);

    // The complete family-H relation.  All inputs below are already bound
    // transcript or proof wires; no target/V advice and no native checker are
    // part of the recursive statement anymore.
    let shv_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        let sv = rt.rs_recs[k].0;
        (0..128).map(|i| wv(sv + i)).collect()
    });
    let value_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        mp_i.val_vs[128 * k..128 * (k + 1)]
            .iter()
            .map(|&vi| wv(vi))
            .collect()
    });
    let rdp_w: [Vec<Wire>; 2] = std::array::from_fn(|k| {
        let fin = rt.rs_recs[k].1;
        (0..7)
            .map(|j| squeeze_word_wire(&outs, trace, fin, j))
            .collect()
    });
    let gamma_w: [Wire; 2] = std::array::from_fn(|k| {
        let (fin, offset) = rt.rs_gam_fins[k];
        squeeze_word_wire(&outs, trace, fin, offset)
    });
    let (rsh_w, vrs_w) = emit_family_h(
        sb,
        cs.q.family.expect("family-H slot"),
        cs.macs,
        cs.fold_macs,
        cs.spine,
        cs.spine256,
        cs.resid
            .iter()
            .find(|&&(key, _)| key == 701)
            .expect("the child slots declare an F256 MAC slot")
            .1,
        1usize << cs.nu,
        &shv_w,
        &value_w,
        &rdp_w,
        gamma_w,
        pfslot,
        pf_w,
        zw,
        ow,
        vals,
        consts,
    );

    let mut pdh_w = zw;
    for pd in gammas_i {
        let gw = squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset);
        pdh_w = sb.gate(cs.macs, &[pdh_w, gw, wv(pd.val_v)])[0];
    }
    let tgt_w = sb.gate(cs.macs, &[rsh_w, ow, pdh_w])[0];
    let mut runw = tgt_w;
    for rr in w_rounds {
        let r_w = outs[trace.squeezes[rr.fin][0]][0];
        runw = sb.gate(mrslot, &[runw, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
    }
    let mut vgrp_w = zw;
    for &vi in &mp_i.val_vs[256..] {
        vgrp_w = sb.gate(cs.macs, &[vgrp_w, ow, wv(vi)])[0];
    }
    let v_w = sb.gate(cs.macs, &[vrs_w, ow, vgrp_w])[0];
    let rhs_v_w = sb.gate(cs.macs, &[zw, wv(inner_pd_i.q_v), v_w])[0];
    sb.connect(runw, rhs_v_w);

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
    // ---- the WIRING GKR in-circuit + the sigma emission ----
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

    // ---- ROUND 4: the recombination + f == g, in-circuit ----
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
        &pub_w,
        &gather_w,
        &pt_w,
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
    // ---- the MULTI-SLOT element PIOP (general strip) ----
    let mut el_zr = zw;
    for (k, rr) in piop_i.zc_rounds.iter().enumerate() {
        let t_w = squeeze_word_wire(&outs, trace, piop_i.tau_fin, k);
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
    // ---- the multipoint intake at R=2, P>0 ----
    let mp_gamma_w = outs[trace.squeezes[mp_i.gamma_fin][0]][0];
    let mut t0_w = zw;
    let mut pw_w = ow;
    let mut mp_pws: Vec<Wire> = vec![ow];
    for (k, &vi) in mp_i.val_vs.iter().enumerate() {
        t0_w = sb.gate(macs, &[t0_w, pw_w, wv(vi)])[0];
        if k + 1 < mp_i.val_vs.len() {
            pw_w = sb.gate(macs, &[zw, pw_w, mp_gamma_w])[0];
            mp_pws.push(pw_w);
        }
    }
    let mut tm_w = t0_w;
    let mut mp_rho2_w: Vec<Wire> = Vec::new();
    for rr in &mp_i.rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_rho2_w.push(rho_w);
        tm_w = sb.gate(mrslot, &[tm_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    sb.connect(tm_w, wv(mp_i.anchor_v));
    let mut anc_w = wv(mp_i.anchor_v);
    let mut mp_sig_w: Vec<Wire> = Vec::new();
    for rr in &mp_i.anchor_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_sig_w.push(rho_w);
        anc_w = sb.gate(mrslot, &[anc_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    assert_eq!(
        mp_sig_w.len(),
        2 * (m_mp2 + 1),
        "sigma spans the anchor layers"
    );

    // ---- the anchor EXPECT at real-inner scale ----
    let extend_const = |pw: &mut Vec<(F128, Wire)>, xn: &[F128]| {
        for &cv2 in &xn[pw.len()..] {
            let w = if cv2 == F128::ZERO {
                zw
            } else {
                assert_eq!(cv2, F128::ONE, "constant point coord is a slot-prefix bit");
                ow
            };
            pw.push((cv2, w));
        }
    };
    // The c-point's 7 baked inner constants are the zerocheck's friendly
    // challenges — the RS ghash set or the AG set, by the tape's flavor.
    let t_vals_b: Vec<F128> = match &rt.zskip {
        ZskipTapeRec::Rs { .. } => {
            use flock_core::zerocheck::univariate_skip_optimized::{
                medium_challenges_ghash, small_challenges_ghash,
            };
            let mut v: Vec<F128> = Vec::new();
            v.extend_from_slice(&small_challenges_ghash());
            v.extend_from_slice(&medium_challenges_ghash());
            v
        }
        ZskipTapeRec::Ag { .. } => flock_core::zerocheck::ag_skip::friendly_challenges().to_vec(),
    };
    assert_eq!(t_vals_b.len(), 7, "the seven baked inner constants");
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
    extend_const(&mut xab_pw, &rt.x_ab_n);
    let (outer_ch_b, outer_fin_b) = rt.outer_b;
    let mut xc_pw: Vec<(F128, Wire)> = (0..rt.zc_rounds_b.len())
        .map(|k2| {
            if k2 < 7 {
                (t_vals_b[k2], cw(sb, vals, consts, t_vals_b[k2]))
            } else {
                // The exact (row, word) map — a FUSED slice squeeze (the
                // AG r_outer grind) reserves one word per row for the PoW
                // predicate, so the naive 4-per-row split misaddresses.
                let j = k2 - 7;
                (
                    chals[outer_ch_b + j],
                    squeeze_word_wire(&outs, trace, outer_fin_b, j),
                )
            }
        })
        .collect();
    extend_const(&mut xc_pw, &rt.x_c_n);
    for (i2, (&(nv, _), &xn)) in xab_pw.iter().zip(&rt.x_ab_n).enumerate() {
        assert_eq!(nv, xn, "ab point coord {i2} is the located wire");
    }
    for (i2, (&(nv, _), &xn)) in xc_pw.iter().zip(&rt.x_c_n).enumerate() {
        assert_eq!(nv, xn, "c point coord {i2} is the located wire");
    }

    let prefix_product = |sb: &mut ShapeBuilder, factors: &[(Wire, Wire)]| -> Wire {
        let mut seed = ow;
        for chunk in factors.chunks(pf_w) {
            let mut g_in = vec![seed];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(std::iter::repeat_n(zw, pf_w - chunk.len()));
            g_in.push(ow);
            seed = sb.gate(pfslot, &g_in)[0];
        }
        seed
    };
    let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
    let rho_mrg_w: Vec<Wire> = w_rounds
        .iter()
        .map(|rr| outs[trace.squeezes[rr.fin][0]][0])
        .collect();
    let mut rinv_n2: Vec<F128> = rho_mrg_n.clone();
    let mut rinv_w: Vec<Wire> = rho_mrg_w.clone();
    let mut ghat = zw;
    for j in 0..128 {
        if j > 0 {
            let mut lvl_w = Vec::with_capacity(m_mp2);
            for t3 in 0..m_mp2 {
                let y = flock_core::pcs::jagged::frob_inv(rinv_n2[t3]);
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
        let eqj = prefix_product(sb, &factors);
        ghat = sb.gate(spine, &[zw, zw, zw, ghat, zw, zw, mp_pws[j], eqj, zw])[3];
    }
    let e_at_w = {
        let factors: Vec<(Wire, Wire)> = rho_mrg_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        prefix_product(sb, &factors)
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
            sb.gate(spine, &[zw, zw, zw, zw, zw, zw, mp_pws[128], ghat, zw])[3]
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
                        squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset)
                    })
                    .collect();
                if let flock_core::matrix_fold::JaggedRowWeight::Combo(t) = &c.row {
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
            let gpd_w = squeeze_word_wire(&outs, trace, pd.fin, pd.squeeze_offset);
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
        let coeff = sb.gate(macs, &[zw, mp_pws[256 + g_ix], e_at_w])[0];
        let wd = sb.gate(macs, &[zw, w_st, gdp[0]])[0];
        expect_w = sb.gate(macs, &[expect_w, coeff, wd])[0];
    }
    sb.connect(anc_w, expect_w);
    if std::env::var("ASSIST_CENSUS").is_ok() {
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
    // ---- the assertion EMISSIONS (all three families) ----
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
    let zpartial_ws: Vec<Wire> = (0..64).map(|i| wv(rt.zp_v + i)).collect();
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
    // ---- the publishes, in the swap's recorded order ----
    let pub_base = sb.public_len();
    for a_wires in &to_publish {
        for w in a_wires {
            sb.publish(*w);
        }
    }
    for w in &level_accs {
        sb.publish(w[0]);
        sb.publish(w[1]);
    }
    cen.push((
        "TAIL: query alphas + native accs",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    sb.publish(t_final[0]);
    sb.publish(t_final[1]);
    sb.publish(tgt_w);
    sb.publish(runw);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(w[0]);
            sb.publish(w[1]);
        }
    }
    cen.push((
        "TAIL: chain ends + residual accs",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    sb.publish(inner_w[0]);
    sb.publish(inner_w[1]);
    sb.publish(sig_w);
    for w in &pt_w {
        sb.publish(*w);
    }
    cen.push((
        "TAIL: sigma + GKR point",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    sb.publish(el_zr);
    sb.publish(el_lcw);
    sb.publish(anc_w);
    for w in &mat_pub {
        sb.publish(*w);
    }
    for w in &ela_pub {
        sb.publish(*w);
    }
    cen.push((
        "TAIL: el ends + assertion publics",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));
    // Family H is now internal arithmetic.  Its source words are already
    // bound where they enter the transcript/proof stream, so no duplicate
    // public re-exposure or checker-only advice remains.
    // ---- the JAGGED ASSERTION emission (the count win) ----
    // Raw W claim values in emission order (rs, then per group combo +
    // dense members), checker-held against the deferred export — the
    // fresh-claim surfaces a merge fold connects to.
    for w in &jag_w {
        sb.publish(*w);
    }
    cen.push((
        "TAIL: jagged claim values",
        sb.public_len(),
        sb.rows_in_slot(cs.macs),
    ));

    let n_query_pub: usize = 2 * levels.len() + levels.iter().map(|l| l.a_count).sum::<usize>();
    let n_tail = 4
        + 2 * levels.len() * rt.yr_len
        + 2
        + 1
        + rt.mu_i
        + 2
        + 1
        + mat_pub.len()
        + ela_pub.len()
        + jag_w.len();
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
        n_mat_pub: mat_pub.len(),
        census: cen,
        jag_w,
        jag_sig_w: mp_sig_w.clone(),
        jag_row_w,
        zskip: match &rt.zskip {
            ZskipTapeRec::Rs { fin, .. } => ZskipWires::Rs(outs[trace.squeezes[*fin][0]][0]),
            ZskipTapeRec::Ag {
                seed_fins,
                nonce_payload,
                ..
            } => {
                let nonce_wi = rt
                    .stream
                    .words
                    .iter()
                    .position(|w| {
                        matches!(
                            w,
                            flock_core::transcript_record::StreamWord::Bytes { payload, word: 0 }
                                if *payload == *nonce_payload
                        )
                    })
                    .expect("the r1 nonce rides one stream word");
                ZskipWires::Ag {
                    seed_w: [
                        squeeze_word_wire(&outs, trace, seed_fins[0], 0),
                        squeeze_word_wire(&outs, trace, seed_fins[1], 0),
                    ],
                    nonce_w: ww[nonce_wi].expect("the nonce payload word is wired"),
                }
            }
        },
        n_ela_pub: ela_pub.len(),
        structure_claim_w,
        pt_w,
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
    let sa = flock_core::circuit::SigmaAssertion {
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
