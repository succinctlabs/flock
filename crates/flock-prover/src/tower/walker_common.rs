//! The sub-walks the child and REAL tape walkers share, byte for byte.
//!
//! Both walkers parse and emit the SAME wire format; each fn here is one
//! such walk, extracted verbatim so the two sides cannot drift apart
//! (drift here was the §G-2 bug class). Everything protocol-specific
//! stays in the walkers — only measured-identical blocks live here.

use std::{array::from_fn, iter::repeat_n};

use flock_core::{
    circuit::{Circuit, builder::SlotId},
    product_gkr::{LiveMask, ProductGkrBatchedProof, s_id_basis},
    zerocheck::{
        ag_skip::friendly_challenges,
        univariate_skip_optimized::{medium_challenges_ghash, small_challenges_ghash},
    },
};
use flock_transcript::transcript_record::{TranscriptOp as Op, TranscriptShape};

use crate::{
    r1cs_hashes::fs_chain::FsChainTrace,
    tower::{
        ChildSlots, F128, GkrLayerRec, GkrRec, InnerPd, Lvl, MergedChain, MixedProof, MpRec,
        OpenLevel, RoundRec, ShapeBuilder, Wire, ZskipTapeRec, assert_chain_replays,
        duplex_row_count_model, emit_spine256, level_query_phase_b3_rows, merge_chain,
        squeeze_word_wire,
    },
};

/// The wiring-GKR transcript walk: alpha/beta, the top pair, the full
/// layer recursion replayed natively in lockstep (recording every ordinal
/// the assembly wires against and the per-round `g0` advice), the masked
/// input checks — the rhs consuming the DEFERRED s_sigma — and the
/// (f, g, s_sigma) triple observed last. BOTH walkers must stay on this
/// exact walk.
#[allow(clippy::too_many_arguments)]
pub(super) fn walk_wiring_gkr_core(
    ops: &[Op],
    vals_rec: &[F128],
    chals: &[F128],
    gkr: &ProductGkrBatchedProof,
    gkr_l0: usize,
    mu: usize,
    mask_w: &LiveMask,
    vc_at: &impl Fn(usize) -> (usize, usize),
    fin_at: &impl Fn(usize) -> usize,
) -> GkrRec {
    let mut i = gkr_l0 + 1;
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
    // The layer walk + native replay in lockstep.
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
    // The input checks: s_id(rho) closed-form NATIVE, s_sigma from the
    // PROOF — the deferred value the assertion carries. Masked input
    // checks under the live-identity padding: leaves are w +
    // α·(live⊙s_id) + (β+1)·live + 1 (dead cells = 1).
    assert_eq!(r_pt.len(), mu, "the GKR point spans the inner cell space");
    let alpha2 = chals[c_alpha];
    let beta2 = chals[c_alpha + 1];
    let basis = s_id_basis(mu);
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
    // The triple observed last — the assertion's value wire.
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
}

/// The chain-materials head: `merge_chain` splices the wiring fork's rows
/// in at the fork point into one linear numbering (+ the four cross-link
/// wires), and the merged sponge trace is asserted to replay the recorded
/// ops. BOTH walkers must stay on this exact splice-and-replay — the
/// chain rows ARE the wire format.
pub(super) fn merge_and_replay_chain(
    t_shape: &TranscriptShape,
    domain: &[u8],
    values: &[F128],
    payloads: &[Vec<u8>],
    ops: &[Op],
    chals: &[F128],
) -> MergedChain {
    let mc = merge_chain(
        t_shape.ops(),
        &t_shape.stream_words_duplex(domain),
        values,
        payloads,
    );
    assert_chain_replays(ops, &mc.trace, chals);
    mc
}

/// The residual pairing's rotation (lane-major inners): a pow2-lane inner
/// (row_words == lanes — e.g. the m28-k4 slim node whose 16-of-16 lanes
/// make the commit exactly full) takes the IDENTITY pairing, same as the
/// native side's rotate gate. BOTH walkers must rotate identically.
pub(super) fn parse_residual_rotation(
    proof: &MixedProof,
    geo: &[Lvl],
    levels: &[OpenLevel],
    w_rounds: &[RoundRec],
) -> (usize, Vec<RoundRec>) {
    let yr_len = proof.pcs_open().inner.ligerito.final_proof.yr.len() / 2;
    let lane_major = geo[0].row_words < geo[0].lanes;
    let w_resid: Vec<RoundRec> = if lane_major {
        let k_rot = w_rounds.len() - levels[0].fold_fins.len();
        let mut v = w_rounds[k_rot..].to_vec();
        v.extend_from_slice(&w_rounds[..k_rot]);
        v
    } else {
        w_rounds.to_vec()
    };
    (yr_len, w_resid)
}

/// The anchor's native endpoint: the anchor's claimed v folded through its
/// own rounds. BOTH walkers must replay this exact fold.
pub(super) fn parse_anchor_native_endpoint(vals_rec: &[F128], chals: &[F128], mp: &MpRec) -> F128 {
    let mut t = vals_rec[mp.anchor_v];
    for rr in &mp.anchor_rounds {
        let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
        let r = chals[rr.ch];
        let g0 = t + g1;
        t = g0 + (g1 + g0 + gi) * r + gi * r * r;
    }
    t
}

/// The GKR input-check advice (masked M̂ and livê) at the inner circuit's
/// GKR endpoint. BOTH walkers must evaluate the same masks.
pub(super) fn parse_gkr_input_check_advice(
    circuit: &Circuit,
    r_pt: &[F128],
) -> (usize, F128, F128) {
    let mu_i = circuit.cells().mu();
    let (mid_n, live_n) = {
        let basis_i = s_id_basis(mu_i);
        let mask_i = circuit.live_mask();
        (
            mask_i.masked_id_eval(&basis_i, r_pt),
            mask_i.live_eval(r_pt),
        )
    };
    (mu_i, mid_n, live_n)
}

/// The `B3_CENSUS` per-level row report plus the chain-decomposition
/// census: an independent row-count model of the duplex discipline
/// (transcript-v3), asserted against the sponge trace. Called from inside
/// each walker's census block; BOTH walkers must stay on this one model.
pub(super) fn census_levels_and_chain_rows(
    ops: &[Op],
    t_shape: &TranscriptShape,
    domain: &[u8],
    geo: &[Lvl],
    trace: &FsChainTrace,
) {
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
        let v3_rows = duplex_row_count_model(t_shape.ops(), &t_shape.stream_words_duplex(domain));
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

/// The multipoint intake at R = 2 AND P > 0: the T0 gamma-power fold over
/// the absorbed claim values, the m two-product rounds folding to the
/// anchor's claimed v (a copy constraint), then the anchor's own rounds
/// folding that v to the endpoint the expect consumes — the squeezes are
/// the sigma wires. The gamma-power wires are KEPT: the anchor expect
/// consumes mp_pws[j] (j < 128) for ĝ, mp_pws[128] for the second RS
/// statement, and mp_pws[256 + k] for the P group coefficients. BOTH
/// walkers must emit this exact intake.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_multipoint_intake(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    mp: &MpRec,
    m_mp2: usize,
    zw: Wire,
    ow: Wire,
) -> (Vec<Wire>, Vec<Wire>, Vec<Wire>, Wire) {
    let macs = cs.macs;
    let mrs = cs.mrs;
    let mp_gamma_w = outs[trace.squeezes[mp.gamma_fin][0]][0];
    let mut t0_w = zw;
    let mut pw_w = ow;
    let mut mp_pws: Vec<Wire> = vec![ow];
    for (k, &vi) in mp.val_vs.iter().enumerate() {
        t0_w = sb.gate(macs, &[t0_w, pw_w, wv(vi)])[0];
        if k + 1 < mp.val_vs.len() {
            pw_w = sb.gate(macs, &[zw, pw_w, mp_gamma_w])[0];
            mp_pws.push(pw_w);
        }
    }
    let mut tm_w = t0_w;
    let mut mp_rho2_w: Vec<Wire> = Vec::new();
    for rr in &mp.rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_rho2_w.push(rho_w);
        tm_w = sb.gate(mrs, &[tm_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    sb.connect(tm_w, wv(mp.anchor_v));
    let mut anc_w = wv(mp.anchor_v);
    let mut mp_sig_w: Vec<Wire> = Vec::new();
    for rr in &mp.anchor_rounds {
        let rho_w = outs[trace.squeezes[rr.fin][0]][0];
        mp_sig_w.push(rho_w);
        anc_w = sb.gate(mrs, &[anc_w, wv(rr.g_v), wv(rr.g_v + 1), rho_w])[0];
    }
    assert_eq!(
        mp_sig_w.len(),
        2 * (m_mp2 + 1),
        "sigma spans the anchor layers"
    );
    (mp_pws, mp_rho2_w, mp_sig_w, anc_w)
}

/// The ligerito SPINE walk: start gamma'·q_eval, the initial-OOD folds,
/// the start message, then per level the eval/build fold pairs and the
/// intro-folds consuming the query phase's accumulator wires. Returns the
/// final `t_r` pair. BOTH walkers must emit this exact spine.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_ligerito_spine_walk(
    sb: &mut ShapeBuilder,
    spine256: SlotId,
    outs: &[Vec<Wire>],
    trace: &FsChainTrace,
    wv: &impl Fn(usize) -> Wire,
    levels: &[OpenLevel],
    inner_pd: &InnerPd,
    start_v: usize,
    level_accs: &[[Wire; 2]],
    zw: Wire,
    ow: Wire,
) -> [Wire; 2] {
    let r_lvl = levels.len() - 1;
    let gpw = outs[trace.squeezes[inner_pd.fin][0]][0];
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
        [wv(inner_pd.q_v), zw],
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
        [wv(start_v), wv(start_v + 1)],
        [wv(start_v + 2), wv(start_v + 3)],
        tsp,
        ow,
        z2,
    );
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = [
                squeeze_word_wire(outs, trace, lvl.fold_fins[j], 0),
                squeeze_word_wire(outs, trace, lvl.fold_fins[j], 1),
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
        if li < r_lvl {
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
    tsp
}

/// The family-H relation's source wires, gathered from the tape: per RS
/// statement the 128 shifted-value wires, the 128 multipoint values, the
/// 7 reduced-dot words, and the batching gamma — every one an already
/// bound transcript/proof wire. BOTH walkers must gather this exact map.
#[allow(clippy::type_complexity)]
pub(super) fn family_h_source_wires(
    outs: &[Vec<Wire>],
    trace: &FsChainTrace,
    wv: &impl Fn(usize) -> Wire,
    rs_recs: &[(usize, usize, usize)],
    rs_gam_fins: &[(usize, usize)],
    val_vs: &[usize],
) -> ([Vec<Wire>; 2], [Vec<Wire>; 2], [Vec<Wire>; 2], [Wire; 2]) {
    let shv_w: [Vec<Wire>; 2] = from_fn(|k| {
        let sv = rs_recs[k].0;
        (0..128).map(|i| wv(sv + i)).collect()
    });
    let value_w: [Vec<Wire>; 2] = from_fn(|k| {
        val_vs[128 * k..128 * (k + 1)]
            .iter()
            .map(|&vi| wv(vi))
            .collect()
    });
    let rdp_w: [Vec<Wire>; 2] = from_fn(|k| {
        let fin = rs_recs[k].1;
        (0..7)
            .map(|j| squeeze_word_wire(outs, trace, fin, j))
            .collect()
    });
    let gamma_w: [Wire; 2] = from_fn(|k| {
        let (fin, offset) = rs_gam_fins[k];
        squeeze_word_wire(outs, trace, fin, offset)
    });
    (shv_w, value_w, rdp_w, gamma_w)
}

/// The anchor expect's constant point-coordinate tail: extend a (native,
/// wire) point with the frozen slot-prefix bits, each riding zw/ow. BOTH
/// walkers must freeze the same tail.
pub(super) fn extend_const_coords(pw: &mut Vec<(F128, Wire)>, xn: &[F128], zw: Wire, ow: Wire) {
    for &cv2 in &xn[pw.len()..] {
        let w = if cv2 == F128::ZERO {
            zw
        } else {
            assert_eq!(cv2, F128::ONE, "constant point coord is a slot-prefix bit");
            ow
        };
        pw.push((cv2, w));
    }
}

/// How many inner t-values [`baked_inner_t_vals`] bakes: the seven
/// friendly zerocheck challenges every inner flavor fixes. The
/// anchor-expect's c-point branches on this count — coordinates below it
/// ride baked constants, the rest are r_outer squeeze words.
pub(super) const N_BAKED_T_VALS: usize = 7;

/// The c-point's [`N_BAKED_T_VALS`] baked inner constants: the zerocheck's
/// friendly challenges — the RS ghash set or the AG set, by the tape's
/// flavor (baked constants are free in-circuit either way). BOTH walkers
/// must bake the same seven.
pub(super) fn baked_inner_t_vals(zskip: &ZskipTapeRec) -> Vec<F128> {
    let t_vals_b: Vec<F128> = match zskip {
        ZskipTapeRec::Rs { .. } => {
            let mut v: Vec<F128> = Vec::new();
            v.extend_from_slice(&small_challenges_ghash());
            v.extend_from_slice(&medium_challenges_ghash());
            v
        }
        ZskipTapeRec::Ag { .. } => friendly_challenges().to_vec(),
    };
    assert_eq!(
        t_vals_b.len(),
        N_BAKED_T_VALS,
        "the seven baked inner constants"
    );
    t_vals_b
}

/// The residual region's prefix-slot product: every chunked (1 + a + b)
/// product rides the width-`pf_w` prefix gate, seeded by `ow`. BOTH
/// walkers must chunk and pad identically.
pub(super) fn prefix_product(
    sb: &mut ShapeBuilder,
    pfslot: SlotId,
    pf_w: usize,
    zw: Wire,
    ow: Wire,
    factors: &[(Wire, Wire)],
) -> Wire {
    let mut seed = ow;
    for chunk in factors.chunks(pf_w) {
        let mut g_in = vec![seed];
        for (a, _) in chunk {
            g_in.push(*a);
        }
        g_in.extend(repeat_n(zw, pf_w - chunk.len()));
        for (_, b) in chunk {
            g_in.push(*b);
        }
        g_in.extend(repeat_n(zw, pf_w - chunk.len()));
        g_in.push(ow);
        seed = sb.gate(pfslot, &g_in)[0];
    }
    seed
}
