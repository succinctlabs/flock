use std::env::var;

use flock_core::{
    circuit::{
        SigmaAssertion,
        builder::{BuiltCircuit, SlotId},
    },
    element_r1cs::union::{ElementAssertion, region_slots},
    lincheck::{LincheckCircuit, MatrixAssertion, SkipPoint, build_eq_table},
    matrix_fold::{JaggedAssertion, JaggedRowWeight},
    pcs::{
        jagged::{
            JaggedParams, MultipointTwistedProof, STATE_INITIAL, STATE_SUCCESS, assist_boundaries,
            assist_sparse_transitions, frob_inv,
        },
        ring_switch::{
            build_fold_byte_table, inner_product, linearized_coefficients, tensor_algebra_transpose,
        },
    },
    zerocheck::univariate_skip::build_eq,
};
use flock_transcript::transcript_record::{
    RecordingChallenger, Stream, StreamWord, TranscriptOp as Op, TranscriptShape,
};
#[cfg(test)]
use {
    crate::tower::{
        AssistLayerGate, BitSpreadGate, BitSpreadTable, Blake3Gate, FamilyTransposeTileGate,
        MacGate256, MergedRoundGate, PowMaskGate, SpineGate, SpineGate256, SwapGate, ZcRoundGate,
        digest_words, gates_blake3::Rng, gates_blake3::Tree, gates_blake3::leaf_word,
        hash_to_digest, query::emit_opening,
    },
    std::any::Any,
};

use crate::{
    r1cs_hashes::{
        blake3::build_block_r1cs,
        fs_chain::{FsChainTrace, IV},
    },
    tower::{
        CollapsedSlots, ElPiopRec, EnvShape, F128, F256, FsChallenger, GkrRec, HashKind, InnerPd,
        LeafEvalGate, LeafEvalGate256, Lvl, MacGate, MergedChain, MixedInner, MixedProof, MpRec,
        OpenLevel, PdRec, RoundRec, SLOT_WORDS, ShapeBuilder, UnionInstance, Wire, ag_seed_bytes,
        assertion_mac, bytes_payload_mask, cap_payloads, cap_wires, check_residual_publics,
        circuit_structure_claim_wires, cw, declare_envelope_slots, decode_ag_point,
        emit_boolean_reported_check, emit_element_reported_check, emit_fs_chain, emit_publics_hash,
        emit_query_phase, emit_recombination, emit_residual_region, flatten_ops, level_geometry,
        level_sources, observed_f256, pack8, parse_open_levels, payload_words, pin_recombination,
        query_phase_b3_rows, replay_ligerito_spine256, squeeze_word_wire, strat_scheds,
        walker_common,
        walker_common::{
            LBL_AG_R1_NONCE, LBL_AG_R1_POINT, LBL_AG_SKIP, LBL_ELEMENT_LC, LBL_ELEMENT_ZC,
            LBL_FROBENIUS, LBL_LINCHECK, LBL_MERGED_OPEN, LBL_MULTIPOINT, LBL_PRODUCT_GKR,
            LBL_RING_SWITCH, LBL_ZEROCHECK, RS_SHAT_WORDS, TailEntry, Z_PARTIAL_WORDS,
            publish_tail, zskip_wires,
        },
    },
    verifier::{verify_ligerito_union_circuit, verify_ligerito_union_circuit_ag},
};

/// One recorded child verification, parsed: the tape pinned op-for-op, every
/// region located, and every native replica the emitter and checker consume.
/// `new` records the verification and checks the map for that tape.
pub(super) struct ChildTape<'p> {
    pub(super) inner: &'p MixedInner,
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
    // located regions. `el` is `None` for a BOOLEAN-ONLY circuit inner
    // (the hash-chain leaf) — the element PIOP region does not exist on
    // its tape, and the `el_*`/`a_sum_n`/`b_sum_n` natives below are
    // meaningful iff `el.is_some()`.
    pub(super) gkr: GkrRec,
    pub(super) el: Option<ElPiopRec>,
    pub(super) start_v: usize,
    gammas_o: Vec<PdRec>,
    pub(super) w_rounds: Vec<RoundRec>,
    pub(super) w_resid: Vec<RoundRec>,
    mp_o: MpRec,
    inner_pd2: InnerPd,
    yr_v2: usize,
    pub(super) yr_len: usize,
    pub(super) levels: Vec<OpenLevel>,
    pub(super) lvl_src: Vec<(&'p [[u8; 32]], &'p Vec<Vec<F128>>, &'p Vec<[u8; 32]>)>,
    pub(super) geo: Vec<Lvl>,
    pub(super) native_sums: Vec<F256>,
    pub(super) n_pd: usize,
    /// Packed-direct claim carrying the element lincheck's `z_eval`.
    pub(super) z_ix: Option<usize>,
    /// The child cell space's public-slot count — the recombination's tail.
    pub(super) n_pub_slots_c: usize,
    pub(super) n_p: usize,
    // the boolean PIOP's round ordinals, located with fins ((ch, fin) pairs)
    pub(super) zc_rounds_b: Vec<(usize, usize)>,
    pub(super) outer_b: (usize, usize),
    pub(super) bl_alpha: (usize, usize),
    pub(super) betas_b: Vec<(usize, usize)>,
    pub(super) zc_finals_v: usize,
    lc_msg_vs: Vec<usize>,
    pub(super) lc_rounds_b: Vec<(usize, usize)>,
    pub(super) eps_n: Vec<F128>,
    // The z_skip transcript surface, by flavor (the boolean claims' row
    // lows derive from it), and z_partial's value ordinal (the boolean
    // claims' column lows — absorbed child words, connectable).
    pub(super) zskip: ZskipTapeRec,
    pub(super) zp_v: usize,
    // published chain ordinals
    ga_c: usize,
    ga_fin: usize,
    mg_c: usize,
    mg_fin: usize,
    /// The two ring-switch regions: `(s_hat_v, r_dprime finalization,
    /// r_dprime challenge)`, plus each batching coefficient's location in
    /// their shared vector squeeze. These are the family-H source wires.
    pub(super) rs_recs: Vec<(usize, usize, usize)>,
    pub(super) rs_gam_fins: Vec<(usize, usize)>,
    // native references + replicas
    pub(super) bool_assert: MatrixAssertion,
    pub(super) el_assert: Option<ElementAssertion>,
    pub(super) sigma_native: SigmaAssertion,
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
    // anchor-expect geometry — statement constants of the inner shape
    pub(super) n_log_i: usize,
    pub(super) k_cols_i: usize,
    pub(super) m_mp2: usize,
    pub(super) bounds_i: Vec<(u64, u64, u32)>,
    pub(super) x_ab_n: Vec<F128>,
    pub(super) x_c_n: Vec<F128>,
    pub(super) groups_ix: Vec<Vec<usize>>,
    /// Derived pd claim points (merged-open v1) — see [`RealTape::pd_pts`].
    pub(super) pd_pts: Vec<Vec<F128>>,
    /// The deferred verify's jagged-layout export (the count win): the
    /// independent reference for the W-value publics the region publishes
    /// instead of rebuilding — tied member-for-member to the native expect
    /// replica in the constructor.
    pub(super) jag: JaggedAssertion,
}

/// The boolean z_skip's transcript surface, by flavor. RS: the one fused
/// PoW+squeeze — the merge assembly derives the 64 Lagrange row lows from
/// its wire in-circuit ([`emit_lagrange_lows`]). AG: the point has NO
/// transcript word — it decodes from (seed = two squeezes, nonce = a
/// 4-byte observe) through a STANDALONE hash, so the tape records the seed
/// ordinals and the nonce's payload ordinal; the consumer publishes
/// (seed, nonce, point, row lows), binds the decode IN-CIRCUIT
/// ([`emit_ag_point_binding`]), and leaves the nonce range + the
/// lows-to-functional items to the native checker
/// ([`check_ag_skip_publics`], docs/ag-recursion-plan.md).
pub(super) enum ZskipTapeRec {
    Rs {
        ch: usize,
        fin: usize,
    },
    Ag {
        /// chals ordinal of seed word s0 (s1 sits at `seed_ch + 1`).
        seed_ch: usize,
        /// The two seed squeezes' finalization ordinals.
        seed_fins: [usize; 2],
        /// The r₁ nonce's byte-payload ordinal (the ObserveBytes/Pow
        /// counter — its stream word is public under `bytes_payload_mask`).
        nonce_payload: usize,
    },
}

/// The label map: the region order the assembly builds against
/// ([`ChildTape::parse_label_map`]).
struct LabelMap {
    zc_l: Vec<usize>,
    lc_l: Vec<usize>,
    elzc_l: Vec<usize>,
    el_l: Vec<usize>,
    gkr_l: Vec<usize>,
    mo_l: Vec<usize>,
    mp_l: Vec<usize>,
}

/// Everything the chain-materials + open-phase walk hands the tape:
/// the merged chain, the located open-phase records, and the geometry
/// ([`ChildTape::parse_chain_and_open_phase`]).
struct ChainOpenParse<'p> {
    stream: Stream,
    bytes: Vec<u8>,
    trace: FsChainTrace,
    cross: Vec<Option<(usize, usize)>>,
    lvl_src: Vec<(&'p [[u8; 32]], &'p Vec<Vec<F128>>, &'p Vec<[u8; 32]>)>,
    start_v: usize,
    gammas_o: Vec<PdRec>,
    w_rounds: Vec<RoundRec>,
    mp_o: MpRec,
    inner_pd2: InnerPd,
    yr_v2: usize,
    levels: Vec<OpenLevel>,
    geo: Vec<Lvl>,
    native_sums: Vec<F256>,
    b3_rows: usize,
    spread_w: usize,
    cap_pays: Vec<usize>,
}

impl<'p> ChildTape<'p> {
    pub(super) fn new(inner: &'p MixedInner, domain: &'static [u8]) -> Self {
        let built = &inner.built;
        let proof = &inner.proof;
        let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
        let blake_r1cs = build_block_r1cs(inner.nu);
        let blake_lc = blake_r1cs.csc_lincheck_circuit();
        let lcs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
        let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
        let all_claims = match proof {
            MixedProof::Rs(p) => verify_ligerito_union_circuit(
                &union,
                &built.shape.circuit,
                &built.witness.public,
                &lcs,
                &inner.commitment,
                p,
                &inner.pcs,
                &mut rec,
            ),
            MixedProof::Ag(p) => verify_ligerito_union_circuit_ag(
                &union,
                &built.shape.circuit,
                &built.witness.public,
                &lcs,
                &inner.commitment,
                p,
                &inner.pcs,
                &mut rec,
            ),
        }
        .expect("the mixed circuit inner verifies");
        let native_claims = all_claims
            .boolean
            .clone()
            .expect("the boolean class yields the RS (ab, c) claims");
        let bool_assert = inner.work.boolean.clone().expect("boolean matrix work");
        // The element side is OPTIONAL: a boolean-only circuit inner (the
        // hash-chain leaf) has no element class, and its tape carries no
        // element PIOP region. The union is the authority.
        let has_el = union.has_element();
        let el_assert = inner.work.element.clone();
        assert_eq!(
            el_assert.is_some(),
            has_el,
            "element work travels iff the union has an element class"
        );
        let sigma_native = inner.sigma.clone();
        let t_shape = rec.shape();
        let chals: Vec<F128> = rec.challenges().to_vec();
        let vals_rec: Vec<F128> = rec.values().to_vec();
        let ops: Vec<Op> = flatten_ops(t_shape.ops()).to_vec();
        let mut pub_payloads = bytes_payload_mask(&ops);

        let LabelMap {
            zc_l,
            lc_l,
            elzc_l,
            el_l,
            gkr_l,
            mo_l,
            mp_l,
        } = Self::parse_label_map(&ops, proof, has_el);

        let vc_at = |end: usize| vc_at(&ops, end);
        let fin_at = |end: usize| fin_at(&ops, end);

        Self::check_boolean_zerocheck_slices(&ops, &vals_rec, proof, has_el, &zc_l);

        let gkr_rec = Self::parse_wiring_gkr_region(&ops, &vals_rec, &chals, proof, built, &gkr_l);

        let el_rec = Self::parse_element_piop_region(&ops, has_el, &elzc_l, &el_l);

        let (pd_recs, mp_val_v, rs_recs, rs_gam_ch, rs_gam_fins) =
            Self::parse_merged_open(&ops, &vals_rec, proof, &mo_l);
        // The pd claims are the element class's two (c, lc) — when the
        // class exists — plus one per wiring GATHER; every gather value is
        // absorbed, in proof order.
        let n_el_pd = if has_el { 2 } else { 0 };
        assert_eq!(
            pd_recs.len(),
            n_el_pd + proof.wiring().gather.len(),
            "pd claims = element (c, lc) + the wiring gathers"
        );
        let pd_vals: Vec<F128> = pd_recs.iter().map(|&pv| vals_rec[pv]).collect();
        for (k, g) in proof.wiring().gather.iter().enumerate() {
            assert!(
                pd_vals.contains(g),
                "gather value {k} rides a packed-direct claim"
            );
        }

        let fro = &proof.pcs_open().frobenius;
        let n_p = fro.group_values.len();
        assert!(n_p > 0, "a circuit inner carries scalar groups (P > 0)");
        Self::check_multipoint_schedule(&ops, &vals_rec, &chals, fro, n_p, &mp_l, mp_val_v);

        let (ga_c, ga_fin, mg_c, mg_fin, h_rows) =
            Self::parse_published_chain_ordinals(&ops, inner, &gkr_l, &mp_l);

        let ChainOpenParse {
            stream,
            bytes,
            trace,
            cross,
            lvl_src,
            start_v,
            gammas_o,
            w_rounds,
            mp_o,
            inner_pd2,
            yr_v2,
            levels,
            geo,
            native_sums,
            b3_rows,
            spread_w,
            cap_pays,
        } = Self::parse_chain_and_open_phase(
            inner, &rec, &t_shape, domain, &ops, &chals, has_el, &pd_recs, h_rows,
        );
        for &p in &cap_pays[1..] {
            pub_payloads[p] = false;
        }

        let (native_target, native_running) = Self::parse_merged_intake_natives(
            &vals_rec, &chals, &rs_recs, rs_gam_ch, &gammas_o, &w_rounds, fro, &inner_pd2,
        );

        // ---- the spine's native quad replay ----
        let t_final_n = replay_ligerito_spine256(
            &levels,
            &vals_rec,
            &chals,
            start_v,
            chals[inner_pd2.ch] * vals_rec[inner_pd2.q_v],
            &native_sums,
        );

        let anc_end_n = walker_common::parse_anchor_native_endpoint(&vals_rec, &chals, &mp_o);

        let (el_g0, el_run_n, a_sum_n, b_sum_n) =
            Self::parse_element_piop_natives(&union, &vals_rec, &chals, &el_rec, &el_assert);

        let (mu_i, mid_n, live_n) =
            walker_common::parse_gkr_input_check_advice(&built.shape.circuit, &gkr_rec.r_pt);

        // ---- the anchor-expect geometry + its FULL native replica ----
        let m_mp2 = mp_o.rounds.len();
        assert_eq!(
            mp_o.anchor_rounds.len(),
            2 * (m_mp2 + 1),
            "sigma spans the anchor layers"
        );
        assert_eq!(w_rounds.len(), m_mp2, "merged rho spans the dense domain");
        let n_log_i = union.n_log();
        // Recompute the recombination and f == g from located words.
        let n_pub_slots_c = pin_recombination(
            inner.built.shape.circuit.cells(),
            n_log_i,
            &inner.built.witness.public,
            &inner.proof.wiring().gather,
            &gammas_o,
            n_el_pd,
            &vals_rec,
            &gkr_rec.r_pt,
            gkr_rec.fgs_v,
        );
        let params_i = JaggedParams::from_heights(&union.jagged_heights(), n_log_i, m_mp2);
        let k_cols_i = params_i.k;
        let bounds_i = assist_boundaries(&params_i);
        let n_runs = bounds_i.len();
        // A run longer than one column is ALWAYS a zero-height run, and the
        // mixed inner has INTERIOR zero runs — the per-run weight is the
        // general Σ eq over the run's columns; the LONGEST run takes the
        // char-2 complement (the eq masses sum to 1).
        let run_y0: Vec<usize> = bounds_i
            .iter()
            .scan(0usize, |y, &(_, _, len)| {
                let s = *y;
                *y += len as usize;
                Some(s)
            })
            .collect();
        let comp_ix = (0..n_runs)
            .max_by_key(|&r| bounds_i[r].2)
            .expect("at least one run");
        // The boolean PIOP's round ordinals, located with fins: the RS
        // statements sit at points made of its round challenges, and the
        // MatrixAssertion's surfaces (x_inner_rest, rr, z_skip, z_partial)
        // map onto the same walk — the merge node's connects consume them.
        // Byte-payload ordinal of the op at `end` (ObserveBytes and Pow
        // share the payload counter — see [`bytes_payload_mask`]).
        let payload_at =
            |end: usize| -> usize { ops[..end].iter().filter(|o| o.carries_payload()).count() };
        let (
            zc_rounds_b,
            zskip,
            (outer_ch_b, outer_fin_b),
            bl_alpha,
            betas_b,
            zc_finals_v,
            lc_msg_vs,
            lc_rounds_b,
            zp_v,
        ) = {
            let mut i2 = zc_l[0] + 1;
            while matches!(ops[i2], Op::Pow { .. }) {
                i2 += 1;
            }
            // The flavored region head. RS: two squeeze slices (r_skip +
            // r_outer), the two 64-slices, then the fused z_skip squeeze.
            // AG: ONE squeeze slice (r_outer), the 158+64 round-1 slices,
            // then r₁'s 5-op surface — seed label, two seed squeezes,
            // nonce label, the 4-byte nonce observe (the point itself has
            // no transcript word; it decodes from seed ‖ nonce).
            let (outer, zskip) = match proof {
                MixedProof::Rs(_) => {
                    assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_skip slice");
                    i2 += 1;
                    assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "r_outer slice");
                    let outer = (vc_at(i2).1, fin_at(i2));
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
                    assert!(matches!(ops[i2], Op::SqueezeSlice(_)), "ag r_outer slice");
                    let outer = (vc_at(i2).1, fin_at(i2));
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
            let mut betas = Vec::new();
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
            let mut lc_msgs = Vec::new();
            let mut lc_r: Vec<(usize, usize)> = Vec::new();
            while matches!(ops[i2], Op::ObserveScalar) && matches!(ops[i2 + 1], Op::ObserveScalar) {
                let mut squeeze_i = i2 + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                lc_msgs.push(vc_at(i2).0);
                lc_r.push((vc_at(squeeze_i).1, fin_at(squeeze_i)));
                i2 = squeeze_i + 1;
            }
            assert!(
                matches!(ops[i2], Op::ObserveSlice(Z_PARTIAL_WORDS)),
                "z_partial slice"
            );
            let (zp, _) = vc_at(i2);
            (zc_r, zskip, outer, lc_alpha, betas, zcf, lc_msgs, lc_r, zp)
        };
        assert!(
            lc_rounds_b.len() <= 1 + k_cols_i,
            "lc rounds fit the col bits"
        );
        // The MatrixAssertion's surfaces map onto located ordinals — asserted
        // value-for-value so the merge node's connects consume VERIFIED wire
        // indices, not layout assumptions. The mlv rounds follow the
        // BATCH-MAJOR packing [k_skip | dim6 | rows | high col vars]:
        // round 0 binds x_inner_rest[0] (the dim-6 var), rounds 1..1+ν bind
        // x_outer (the rows used by the RS-point composition), and
        // the remaining rounds bind x_inner_rest[1..]. rr is the lc rounds
        // REVERSED; z_skip is the located squeeze; z_partial the located
        // slice.
        {
            let inner_b = bool_assert.x_inner_rest.len();
            assert_eq!(
                zc_rounds_b.len(),
                inner_b + n_log_i,
                "zc mlv rounds = x_inner_rest + x_outer"
            );
            for (j, &x) in bool_assert.x_inner_rest.iter().enumerate() {
                let m = if j == 0 { 0 } else { n_log_i + j };
                assert_eq!(
                    chals[zc_rounds_b[m].0], x,
                    "x_inner_rest {j} is located zc round {m}"
                );
            }
            assert_eq!(lc_rounds_b.len(), bool_assert.rr.len(), "lc round count");
            for (j, &x) in bool_assert.rr.iter().enumerate() {
                assert_eq!(
                    chals[lc_rounds_b[lc_rounds_b.len() - 1 - j].0],
                    x,
                    "rr {j} is the located lc round, reversed"
                );
            }
            // The z_skip pin, by flavor: RS locates the fused squeeze; AG
            // has no point word — rebuild the seed from the two located
            // squeezes, decode H(seed ‖ nonce) under the child's grinding
            // schedule, and pin the assertion's point to the decode.
            match (proof, &zskip) {
                (MixedProof::Rs(_), ZskipTapeRec::Rs { ch, .. }) => {
                    assert_eq!(chals[*ch], bool_assert.z_skip.phi8(), "z_skip located");
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
                        inner.pcs.zerocheck_grinding().ag_r1_bits(),
                    );
                    assert_eq!(
                        bool_assert.z_skip,
                        SkipPoint::Ag(pt),
                        "z_skip is the r1 point decoded from the located seed + nonce"
                    );
                }
                _ => unreachable!("the tape's zskip record matches the proof flavor"),
            }
            assert_eq!(
                &vals_rec[zp_v..zp_v + 64],
                &bool_assert.z_partial[..],
                "z_partial on the stream"
            );
            assert_eq!(
                chals[bl_alpha.0], bool_assert.alpha,
                "the located Boolean lincheck alpha"
            );
            let pinned: Vec<usize> = bool_assert
                .betas
                .iter()
                .enumerate()
                .filter_map(|(t, beta)| beta.map(|_| t))
                .collect();
            assert_eq!(pinned.len(), betas_b.len(), "one beta per const pin");
            for (k, &t) in pinned.iter().enumerate() {
                assert_eq!(
                    chals[betas_b[k].0],
                    bool_assert.betas[t].expect("pinned beta"),
                    "Boolean const-pin beta {k}"
                );
            }
            // The element assertion's points: r_con = zc.r[ν..] (round
            // order), r_col = the lc bind order reversed.
            if let Some(el_rec) = &el_rec {
                let el_assert = el_assert.as_ref().expect("element assertion");
                assert_eq!(
                    el_rec.zc_rounds.len(),
                    n_log_i + el_assert.r_con.len(),
                    "element zc rounds = rows + r_con"
                );
                for (j, &x) in el_assert.r_con.iter().enumerate() {
                    assert_eq!(
                        chals[el_rec.zc_rounds[n_log_i + j].2],
                        x,
                        "el r_con {j} is a located element zc round"
                    );
                }
                assert_eq!(
                    el_rec.lc_rounds.len(),
                    el_assert.r_col.len(),
                    "element lc round count"
                );
                for (j, &x) in el_assert.r_col.iter().enumerate() {
                    assert_eq!(
                        chals[el_rec.lc_rounds[el_rec.lc_rounds.len() - 1 - j].2],
                        x,
                        "el r_col {j} is the located element lc round, reversed"
                    );
                }
            }
        }
        let eps_n: Vec<F128> = bool_assert.pin_evals.iter().flatten().copied().collect();
        assert_eq!(eps_n.len(), betas_b.len(), "one prefix value per beta");
        let x_ab_n: Vec<F128> = {
            let p = &native_claims.ab.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        let x_c_n: Vec<F128> = {
            let p = &native_claims.c.point;
            let mut v = p.x_inner_rest.clone();
            v.extend_from_slice(&p.x_outer);
            v
        };
        assert_eq!(x_ab_n.len(), 1 + n_log_i + k_cols_i, "ab point split");
        assert_eq!(x_c_n.len(), 1 + n_log_i + k_cols_i, "c point split");
        // The P scalar groups, by shared row part — the same structural
        // grouping the two-product build uses (first-occurrence order).
        // Derived pd points (merged-open v1) — see RealTape's twin.
        let pd_pts_n: Vec<Vec<F128>> = {
            let cells = inner.built.shape.circuit.cells();
            let mut v: Vec<Vec<F128>> = if has_el {
                let el = all_claims.element.as_ref().expect("element claims");
                vec![el.c_point.clone(), el.lc_point.clone()]
            } else {
                Vec::new()
            };
            for i2 in 0..gammas_o.len() - n_el_pd {
                v.push(cells.gate_claim_point(i2, &gkr_rec.r_pt[..cells.nu()]));
            }
            v
        };
        for pt in &pd_pts_n {
            assert_eq!(pt.len(), n_log_i + k_cols_i, "pd point split");
        }
        if let Some(el_rec) = &el_rec {
            let e_rounds = el_rec.zc_rounds.len();
            for j in 0..n_log_i {
                assert_eq!(pd_pts_n[0][j], chals[el_rec.zc_rounds[j].2], "c row {j}");
            }
            for j in 0..e_rounds - n_log_i {
                assert_eq!(
                    pd_pts_n[0][n_log_i + j],
                    chals[el_rec.zc_rounds[n_log_i + j].2],
                    "c col {j}"
                );
            }
            let n_lc = el_rec.lc_rounds.len();
            for j in 0..n_lc {
                assert_eq!(
                    pd_pts_n[1][n_log_i + j],
                    chals[el_rec.lc_rounds[n_lc - 1 - j].2],
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

        // Native replica of the WHOLE anchor expect — validates the formula
        // against the accepted proof before any gate exists.
        {
            let gamma_n = chals[mp_o.gamma_ch];
            let mut gpow_n = vec![F128::ONE];
            for j in 1..257 + n_p {
                gpow_n.push(gpow_n[j - 1] * gamma_n);
            }
            let rho_mrg_n: Vec<F128> = w_rounds.iter().map(|rr| chals[rr.ch]).collect();
            let point_n: Vec<F128> = mp_o.rounds.iter().map(|rr| chals[rr.ch]).collect();
            let sig_n: Vec<F128> = mp_o.anchor_rounds.iter().map(|rr| chals[rr.ch]).collect();
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
                    for (t2, &x) in point_n.iter().enumerate() {
                        prod *= F128::ONE + rinv[t2] + x;
                    }
                    acc += prod;
                }
                acc
            };
            let e_at_n = rho_mrg_n
                .iter()
                .zip(&point_n)
                .fold(F128::ONE, |a, (&r, &x)| a * (F128::ONE + r + x));
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
                for (r, &(_, _, len)) in bounds_i.iter().enumerate() {
                    if r == comp_ix {
                        continue;
                    }
                    let mut w = F128::ZERO;
                    for y in run_y0[r]..run_y0[r] + len as usize {
                        let mut s = F128::ONE;
                        for (jj, &zc2) in z_col.iter().enumerate() {
                            s *= F128::ONE + zc2 + bit((y >> jj) & 1 == 1);
                        }
                        w += s;
                    }
                    w_at[r] = w;
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
                    // The count win's tie: the RAW per-statement W the
                    // region now PUBLISHES equals the deferred export's
                    // claim value — the verifier-exported reference, not a
                    // formula written twice.
                    assert_eq!(
                        inner.work.jagged.rs[si].value, w_n,
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
                        let pd = &gammas_o[i2];
                        let gpd = chals[pd.ch];
                        let w_at = run_weights_n(&pd_pts_n[i2][n_log_i..]);
                        for r in 0..n_runs {
                            run_n[r] += gpd * w_at[r];
                        }
                    }
                    let w_n = run_n
                        .iter()
                        .zip(&eqc_n)
                        .fold(F128::ZERO, |a, (&x, &e)| a + x * e);
                    // The group's exported decomposition — the γ-baked
                    // one-hot combo plus γ-outside dense members — must
                    // recombine to the same raw group W, member for member.
                    let (combo, dense) = &inner.work.jagged.groups[g_ix];
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
                        assert_eq!(*g, chals[gammas_o[i2].ch], "dense member γ_pd");
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
                "the R=2 + P anchor expect replays natively"
            );
        }

        let (yr_len, w_resid) =
            walker_common::parse_residual_rotation(proof, &geo, &levels, &w_rounds);

        let z_ix = el_assert.as_ref().map(|assertion| {
            gammas_o
                .iter()
                .position(|pd| vals_rec[pd.val_v] == assertion.z_eval)
                .expect("element z_eval is an absorbed packed-direct value")
        });

        ChildTape {
            inner,
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
            el: el_rec,
            start_v,
            gammas_o,
            w_rounds,
            w_resid,
            mp_o,
            inner_pd2,
            yr_v2,
            yr_len,
            levels,
            lvl_src,
            geo,
            native_sums,
            n_pd: pd_recs.len(),
            z_ix,
            n_pub_slots_c,
            n_p,
            zc_rounds_b,
            outer_b: (outer_ch_b, outer_fin_b),
            bl_alpha,
            betas_b,
            zc_finals_v,
            lc_msg_vs,
            lc_rounds_b,
            eps_n,
            zskip,
            zp_v,
            ga_c,
            ga_fin,
            mg_c,
            mg_fin,
            rs_recs,
            rs_gam_fins,
            bool_assert,
            el_assert,
            sigma_native,
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
            jag: inner.work.jagged.clone(),
        }
    }

    /// ---- the label map: the region order the assembly builds against ----
    fn parse_label_map(ops: &[Op], proof: &MixedProof, has_el: bool) -> LabelMap {
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
        let zc_l = match proof {
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
        assert_eq!(zc_l.len(), 1, "one boolean zerocheck");
        assert_eq!(lc_l.len(), 1, "one boolean lincheck");
        if has_el {
            assert_eq!(elzc_l.len(), 1, "one element zerocheck");
            assert_eq!(el_l.len(), 1, "one element lincheck region");
            assert!(elzc_l[0] < el_l[0], "element zc before element lc");
            assert!(lc_l[0] < el_l[0], "boolean PIOP before element PIOP");
        } else {
            assert!(
                elzc_l.is_empty() && el_l.is_empty(),
                "a boolean-only tape carries NO element region"
            );
        }
        // THE FORKED ORDER: the wiring argument's chain is spliced in at the
        // fork point, so its region precedes the boolean PIOP's.
        assert!(
            gkr_l[0] < zc_l[0],
            "the wiring fork precedes the boolean PIOP"
        );
        assert_eq!(gkr_l.len(), 1, "one batched wiring GKR");
        assert_eq!(mo_l.len(), 1, "one merged open");
        assert_eq!(
            rs_l.len(),
            2,
            "rs x 2 — one ab/c pair for the boolean class"
        );
        assert_eq!(mp_l.len(), 1, "one multipoint region");
        assert_eq!(fa_l.len(), 1, "one anchor region");
        assert!(zc_l[0] < lc_l[0], "boolean zc before boolean lc");
        assert!(gkr_l[0] < mo_l[0], "wiring GKR before the merged open");
        assert!(mo_l[0] < rs_l[0] && rs_l[1] < mp_l[0] && mp_l[0] < fa_l[0]);
        LabelMap {
            zc_l,
            lc_l,
            elzc_l,
            el_l,
            gkr_l,
            mo_l,
            mp_l,
        }
    }

    /// ---- the boolean zerocheck slices, same shape as the leaf ----
    fn check_boolean_zerocheck_slices(
        ops: &[Op],
        vals_rec: &[F128],
        proof: &MixedProof,
        has_el: bool,
        zc_l: &[usize],
    ) {
        let vc_at = |end: usize| vc_at(ops, end);
        assert_eq!(
            proof.element().is_some(),
            has_el,
            "the element proof section mirrors the union's classes"
        );
        match proof {
            MixedProof::Rs(p) => {
                let bp = p.boolean.as_ref().expect("boolean side present");
                let mut i = zc_l[0] + 1;
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
                assert!(matches!(ops[i], Op::SqueezeSlice(_)), "zc tau lo");
                i += 1;
                assert!(matches!(ops[i], Op::SqueezeSlice(_)), "zc tau hi");
                i += 1;
                assert!(matches!(ops[i], Op::ObserveSlice(64)), "round1_ab");
                let (v0, _) = vc_at(i);
                assert_eq!(
                    &vals_rec[v0..v0 + 64],
                    &bp.zerocheck.round1_ab[..],
                    "round1_ab on the stream"
                );
                i += 1;
                assert!(matches!(ops[i], Op::ObserveSlice(64)), "round1_c");
                let (v1, _) = vc_at(i);
                assert_eq!(
                    &vals_rec[v1..v1 + 64],
                    &bp.zerocheck.round1_c[..],
                    "round1_c on the stream"
                );
            }
            MixedProof::Ag(p) => {
                let bp = p.boolean.as_ref().expect("boolean side present");
                let mut i = zc_l[0] + 1;
                while matches!(ops[i], Op::Pow { .. }) {
                    i += 1;
                }
                assert!(matches!(ops[i], Op::SqueezeSlice(_)), "ag r_outer");
                i += 1;
                assert!(matches!(ops[i], Op::ObserveSlice(158)), "ag round1_ab");
                let (v0, _) = vc_at(i);
                assert_eq!(
                    &vals_rec[v0..v0 + 158],
                    &bp.ag.round1_ab[..],
                    "ag round1_ab on the stream"
                );
                i += 1;
                assert!(matches!(ops[i], Op::ObserveSlice(64)), "ag round1_c");
                let (v1, _) = vc_at(i);
                assert_eq!(
                    &vals_rec[v1..v1 + 64],
                    &bp.ag.round1_c[..],
                    "ag round1_c on the stream"
                );
            }
        }
    }

    /// ---- the wiring GKR region, walked op by op ----
    /// The transcription map: [alpha, beta squeezes | top pair observed |
    /// per layer k: lambda squeeze, k x (2 obs + squeeze) rounds — the
    /// ZcRoundGate shape VERBATIM — then (vl0, vl1, vr0, vr1) observed,
    /// the layer check, and the c_k squeeze folding the claims | the
    /// (f, g, s_sigma) triple observed last]. The walk also RECORDS the
    /// ordinals the assembly wires against, and the per-round `g0` advice.
    fn parse_wiring_gkr_region(
        ops: &[Op],
        vals_rec: &[F128],
        chals: &[F128],
        proof: &MixedProof,
        built: &BuiltCircuit,
        gkr_l: &[usize],
    ) -> GkrRec {
        let vc_at = |end: usize| vc_at(ops, end);
        let fin_at = |end: usize| fin_at(ops, end);
        walker_common::walk_wiring_gkr_core(
            ops,
            vals_rec,
            chals,
            &proof.wiring().gkr,
            gkr_l[0],
            built.shape.circuit.cells().mu(),
            &built.shape.circuit.live_mask(),
            &vc_at,
            &fin_at,
        )
    }

    /// ---- the ELEMENT PIOP region, located (mixed inners only) ----
    /// Shape, per `parse_open_levels`' element branch: [tau slice |
    /// tau_len rounds | ea, eb, ec | lc label | alpha | lc rounds].
    fn parse_element_piop_region(
        ops: &[Op],
        has_el: bool,
        elzc_l: &[usize],
        el_l: &[usize],
    ) -> Option<ElPiopRec> {
        let vc_at = |end: usize| vc_at(ops, end);
        let fin_at = |end: usize| fin_at(ops, end);
        has_el.then(|| {
            let mut i = elzc_l[0] + 1;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            let (tau_fin, tau_ch, tau_len) = match ops[i] {
                Op::SqueezeSlice(n) => (fin_at(i), vc_at(i).1, n),
                ref o => panic!("element tau, got {o:?}"),
            };
            i += 1;
            let mut zc_rounds = Vec::with_capacity(tau_len);
            for _ in 0..tau_len {
                let (gv, _) = vc_at(i);
                assert!(matches!(ops[i], Op::ObserveScalar), "el zc msg");
                assert!(matches!(ops[i + 1], Op::ObserveScalar), "el zc msg");
                let mut squeeze_i = i + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                assert!(matches!(ops[squeeze_i], Op::SqueezeScalar), "el zc rho");
                zc_rounds.push((gv, fin_at(squeeze_i), vc_at(squeeze_i).1));
                i = squeeze_i + 1;
            }
            let (eab_v, _) = vc_at(i);
            for _ in 0..3 {
                assert!(matches!(ops[i], Op::ObserveScalar), "el zc final");
                i += 1;
            }
            assert_eq!(i, el_l[0], "the lc label follows the finals");
            i += 1;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            assert!(matches!(ops[i], Op::SqueezeScalar), "el lc alpha");
            let (alpha_fin, alpha_ch) = (fin_at(i), vc_at(i).1);
            i += 1;
            let mut lc_rounds = Vec::new();
            while matches!(ops[i], Op::ObserveScalar) && matches!(ops[i + 1], Op::ObserveScalar) {
                let mut squeeze_i = i + 2;
                while matches!(ops[squeeze_i], Op::Pow { .. }) {
                    squeeze_i += 1;
                }
                if !matches!(ops[squeeze_i], Op::SqueezeScalar) {
                    break;
                }
                let (gv, _) = vc_at(i);
                lc_rounds.push((gv, fin_at(squeeze_i), vc_at(squeeze_i).1));
                i = squeeze_i + 1;
            }
            assert!(!zc_rounds.is_empty() && !lc_rounds.is_empty(), "el rounds");
            ElPiopRec {
                tau_fin,
                tau_ch,
                zc_rounds,
                eab_v,
                alpha_fin,
                alpha_ch,
                lc_rounds,
            }
        })
    }

    /// ---- the merged open: rs x 2, then PD values, then one coefficient vector ----
    #[allow(clippy::type_complexity)]
    fn parse_merged_open(
        ops: &[Op],
        vals_rec: &[F128],
        proof: &MixedProof,
        mo_l: &[usize],
    ) -> (
        Vec<usize>,
        usize,
        Vec<(usize, usize, usize)>,
        usize,
        Vec<(usize, usize)>,
    ) {
        let vc_at = |end: usize| vc_at(ops, end);
        let fin_at = |end: usize| fin_at(ops, end);
        let mut i = mo_l[0] + 1;
        let mut rs_recs: Vec<(usize, usize, usize)> = Vec::new();
        for k in 0..2 {
            assert!(
                matches!(&ops[i], Op::Label(l) if l.as_slice() == LBL_RING_SWITCH),
                "rs region {k}"
            );
            i += 1;
            assert!(
                matches!(ops[i], Op::ObserveSlice(RS_SHAT_WORDS)),
                "s_hat_v slice"
            );
            let (sv, _) = vc_at(i);
            assert_eq!(
                &vals_rec[sv..sv + RS_SHAT_WORDS],
                &proof.pcs_open().ring_switches[k].s_hat_v[..],
                "s_hat_v {k} on the stream"
            );
            i += 1;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            assert!(matches!(ops[i], Op::SqueezeSlice(7)), "r_dprime");
            rs_recs.push((sv, fin_at(i), vc_at(i).1));
            i += 1;
        }
        // Packed-direct claims contribute just their values.  Their
        // coefficients share the vector squeeze with both RS claims.
        let mut pd_recs: Vec<usize> = Vec::new(); // value index
        while matches!(ops[i], Op::ObserveScalar) {
            let (pv, _) = vc_at(i);
            i += 1;
            pd_recs.push(pv);
        }
        while matches!(ops[i], Op::Pow { .. }) {
            i += 1;
        }
        assert!(
            matches!(ops[i], Op::SqueezeSlice(n) if n == 2 + pd_recs.len()),
            "mixed coefficient vector"
        );
        let rs_gam_ch = vc_at(i).1;
        let rs_gam_fin = fin_at(i);
        i += 1;
        // W rounds until the multipoint label.
        let mut w_rounds = 0usize;
        while matches!(ops[i], Op::ObserveScalar) {
            assert!(matches!(ops[i + 1], Op::ObserveScalar), "w round pair");
            i += 2;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            assert!(matches!(ops[i], Op::SqueezeScalar), "w round squeeze");
            i += 1;
            w_rounds += 1;
        }
        assert_eq!(
            w_rounds,
            proof.pcs_open().merged_rounds.len(),
            "the W rounds fill the dense domain"
        );
        while !matches!(&ops[i], Op::Label(l) if l.as_slice() == LBL_MULTIPOINT) {
            i += 1;
        }
        i += 1;
        let (mv, _) = vc_at(i);
        (
            pd_recs,
            mv,
            rs_recs,
            rs_gam_ch,
            vec![(rs_gam_fin, 0), (rs_gam_fin, 1)],
        )
    }

    /// ---- the multipoint: the R=2 + P>0 schedule, pinned ----
    fn check_multipoint_schedule(
        ops: &[Op],
        vals_rec: &[F128],
        chals: &[F128],
        fro: &MultipointTwistedProof,
        n_p: usize,
        mp_l: &[usize],
        mp_val_v: usize,
    ) {
        let vc_at = |end: usize| vc_at(ops, end);
        let mut i = mp_l[0] + 1;
        let mut n_vals = 0usize;
        while matches!(ops[i], Op::ObserveScalar) {
            n_vals += 1;
            i += 1;
        }
        assert_eq!(
            n_vals,
            2 * RS_SHAT_WORDS + n_p,
            "2x128 RS dual values + P group values"
        );
        while matches!(ops[i], Op::Pow { .. }) {
            i += 1;
        }
        assert!(matches!(ops[i], Op::SqueezeScalar), "multipoint gamma");
        let (_, gc) = vc_at(i);
        let gamma = chals[gc];
        i += 1;
        // The located values ARE the proof's, in schedule order.
        for k in 0..n_vals {
            let want = if k < 256 {
                fro.values[k / 128][k % 128]
            } else {
                fro.group_values[k - 256]
            };
            assert_eq!(vals_rec[mp_val_v + k], want, "mp value {k}");
        }
        // T0 under the R=2 + P schedule folds through the rounds to the
        // anchor's claimed v — consecutive gamma powers across BOTH kinds.
        let mut t = F128::ZERO;
        let mut pw = F128::ONE;
        for k in 0..n_vals {
            t += pw * vals_rec[mp_val_v + k];
            pw *= gamma;
        }
        let mut rounds = 0usize;
        while matches!(ops[i], Op::ObserveScalar) && matches!(ops[i + 1], Op::ObserveScalar) {
            let (gv, _) = vc_at(i);
            i += 2;
            while matches!(ops[i], Op::Pow { .. }) {
                i += 1;
            }
            if !matches!(ops[i], Op::SqueezeScalar) {
                break;
            }
            let (_, rc) = vc_at(i);
            let (g1, gi) = (vals_rec[gv], vals_rec[gv + 1]);
            let r = chals[rc];
            let g0 = t + g1;
            t = g0 + (g1 + g0 + gi) * r + gi * r * r;
            i += 1;
            rounds += 1;
        }
        assert_eq!(rounds, fro.rounds.len(), "mp round count");
        assert!(
            matches!(&ops[i], Op::Label(l) if l.as_slice() == LBL_FROBENIUS),
            "anchor label follows the rounds"
        );
        assert_eq!(t, fro.anchor.v, "T_m == anchor.v under the R=2+P schedule");
    }

    /// ---- the published chain ordinals (GKR alpha, multipoint gamma) ----
    fn parse_published_chain_ordinals(
        ops: &[Op],
        inner: &MixedInner,
        gkr_l: &[usize],
        mp_l: &[usize],
    ) -> (usize, usize, usize, usize, usize) {
        let vc_at = |end: usize| vc_at(ops, end);
        let fin_at = |end: usize| fin_at(ops, end);
        let mut ga_i = gkr_l[0] + 1;
        while matches!(ops[ga_i], Op::Pow { .. }) {
            ga_i += 1;
        }
        assert!(matches!(ops[ga_i], Op::SqueezeScalar), "GKR fingerprint");
        let ga_fin = fin_at(ga_i);
        let (_, ga_c) = vc_at(ga_i);
        let mut mp_i = mp_l[0] + 1;
        while matches!(ops[mp_i], Op::ObserveScalar) {
            mp_i += 1;
        }
        while matches!(ops[mp_i], Op::Pow { .. }) {
            mp_i += 1;
        }
        assert!(matches!(ops[mp_i], Op::SqueezeScalar), "mp gamma op");
        let mg_fin = fin_at(mp_i);
        let (_, mg_c) = vc_at(mp_i);
        // ROUND 2: the H(publics) region's rows — a chunk chain per 1 KiB
        // leaf of the child's public segment plus the left-fold parents.
        let n_pub_i = inner.built.witness.public.len();
        let h_rows = n_pub_i.div_ceil(4) + 2 * n_pub_i.div_ceil(64);
        (ga_c, ga_fin, mg_c, mg_fin, h_rows)
    }

    /// ---- the chain materials ----
    /// The transcript is FORKED (the wiring runs on its own chain);
    /// `merge_chain` splices the child's rows in at the fork point and
    /// hands back one linear numbering plus the four cross-link wires.
    /// ---- the open-phase walk + geometry ----
    #[allow(clippy::too_many_arguments)]
    fn parse_chain_and_open_phase<'q>(
        inner: &'q MixedInner,
        rec: &RecordingChallenger<FsChallenger>,
        t_shape: &TranscriptShape,
        domain: &'static [u8],
        ops: &[Op],
        chals: &[F128],
        has_el: bool,
        pd_recs: &[usize],
        h_rows: usize,
    ) -> ChainOpenParse<'q> {
        let proof = &inner.proof;
        let MergedChain {
            stream,
            bytes,
            trace,
            cross,
            ..
        } = walker_common::merge_and_replay_chain(
            t_shape,
            domain,
            rec.values(),
            rec.payloads(),
            ops,
            chals,
        );
        let lig = &proof.pcs_open().inner.ligerito;
        assert_eq!(
            inner.commitment.cap, lig.initial_cap,
            "commitment IS the L0 cap"
        );
        let r_lvl = lig.recursive_caps.len();
        let lvl_src = level_sources(lig);
        let (start_v, piop_o, gammas_o, w_rounds, mp_o, inner_pd2, yr_v2, levels) =
            parse_open_levels(ops, 32 * lig.initial_cap.len(), r_lvl);
        assert_eq!(
            piop_o.is_some(),
            has_el,
            "the parser sees the element PIOP iff the class exists"
        );
        assert_eq!(
            gammas_o.len(),
            pd_recs.len(),
            "the parser and the region walk agree on the pd claims"
        );
        let (geo, native_sums) = level_geometry(
            &levels,
            &lvl_src,
            chals,
            HashKind::Blake3,
            &strat_scheds(&inner.pcs),
        );
        let b3_rows = trace.rows.len() + h_rows + query_phase_b3_rows(&geo);
        if var("B3_CENSUS").is_ok() {
            let parents = trace.block_offsets.iter().filter(|o| o.is_none()).count();
            let blocks = trace.rows.len() - parents;
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
            walker_common::census_levels_and_chain_rows(ops, t_shape, domain, &geo, &trace);
        }
        let spread_w = geo.iter().map(|g| g.depth).max().unwrap().max(1);
        // Recursive caps are PROOF BODY — the in-circuit cap trees bind them
        // (chain + root connects, nothing checker-read); only the L0 cap —
        // the commitment — stays a statement public.
        let cap_pays = cap_payloads(&stream, &bytes, &lvl_src);
        ChainOpenParse {
            stream,
            bytes,
            trace,
            cross,
            lvl_src,
            start_v,
            gammas_o,
            w_rounds,
            mp_o,
            inner_pd2,
            yr_v2,
            levels,
            geo,
            native_sums,
            b3_rows,
            spread_w,
            cap_pays,
        }
    }

    /// ---- the merged intake's natives (target, running, boundary) ----
    #[allow(clippy::too_many_arguments)]
    fn parse_merged_intake_natives(
        vals_rec: &[F128],
        chals: &[F128],
        rs_recs: &[(usize, usize, usize)],
        rs_gam_ch: usize,
        gammas_o: &[PdRec],
        w_rounds: &[RoundRec],
        fro: &MultipointTwistedProof,
        inner_pd2: &InnerPd,
    ) -> (F128, F128) {
        let gs: Vec<F128> = (0..2).map(|k| chals[rs_gam_ch + k]).collect();
        let mut target = F128::ZERO;
        let mut coeffs: Vec<Vec<F128>> = Vec::new();
        for (k, &(sv, _, rc)) in rs_recs.iter().enumerate() {
            let shv = &vals_rec[sv..sv + RS_SHAT_WORDS];
            let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
            let eq = build_eq(&rdp);
            target += gs[k] * inner_product(&tensor_algebra_transpose(shv), &eq);
            let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
            coeffs.push(linearized_coefficients(&build_fold_byte_table(&scaled)));
        }
        // A MIXED tape's target carries the packed-direct claims too —
        // each absorbed value against its own gamma squeeze.
        for pd in gammas_o {
            target += chals[pd.ch] * vals_rec[pd.val_v];
        }
        let mut running = target;
        for rr in w_rounds {
            let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
            let r = chals[rr.ch];
            let g0 = running + g1;
            running = g0 + (g1 + g0 + gi) * r + gi * r * r;
        }
        // The R = 2 recombination plus the P group values, against the
        // same q_eval the spine starts from.
        let mut big_v = F128::ZERO;
        for (k, cs) in coeffs.iter().enumerate() {
            for (j, &cj) in cs.iter().enumerate() {
                if cj.is_zero() {
                    continue;
                }
                let mut x = fro.values[k][j];
                for _ in 0..j {
                    x = x * x;
                }
                big_v += cj * x;
            }
        }
        for &v in &fro.group_values {
            big_v += v;
        }
        assert_eq!(
            running,
            vals_rec[inner_pd2.q_v] * big_v,
            "the R=2 + P merged boundary replays"
        );
        (target, running)
    }

    /// ---- the element PIOP's native chain + strip sums (mixed only) ----
    fn parse_element_piop_natives(
        union: &UnionInstance<'_>,
        vals_rec: &[F128],
        chals: &[F128],
        el_rec: &Option<ElPiopRec>,
        el_assert: &Option<ElementAssertion>,
    ) -> (Vec<F128>, F128, F128, F128) {
        if let Some(el_rec) = &el_rec {
            let el_assert = el_assert.as_ref().expect("element assertion");
            // This strip-sum arm HARD-CODES the single-slot zero-prefix
            // degenerate case of the REAL side's general region loop
            // (`real_walker::element_piop_natives` walks every element
            // slot with its own kappa and region prefix). Assert the
            // shape it assumes, so a multi-slot or offset element union
            // fails HERE instead of silently mis-summing — port the
            // general loop when this fires.
            let slots_el = region_slots(union);
            assert_eq!(
                slots_el.len(),
                1,
                "one element slot type (else port real_walker's general strip loop)"
            );
            assert_eq!(
                slots_el[0].layout.region_prefix(union.n_log()),
                0,
                "the element region starts the space (else port real_walker's general strip loop)"
            );
            let mut el_g0: Vec<F128> = Vec::new();
            let mut el_run_n = F128::ZERO;
            for (k, &(gv, _, ch)) in el_rec.zc_rounds.iter().enumerate() {
                let (g1, gi) = (vals_rec[gv], vals_rec[gv + 1]);
                let t = chals[el_rec.tau_ch + k];
                let rho = chals[ch];
                let g0 = (el_run_n + t * g1) * (F128::ONE + t).inv();
                el_g0.push(g0);
                el_run_n = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
            }
            assert_eq!(
                el_assert.alpha, chals[el_rec.alpha_ch],
                "the located alpha is the assertion's"
            );
            let (a_sum_n, b_sum_n) = {
                let mt = MacGate::new();
                let kappa = mt.ty.kappa();
                let eq_con = build_eq(&el_assert.r_con[..kappa]);
                // Single slot at the region start: the prefix bits are all
                // zero, so the region weight is the all-zero eq pattern.
                let w = el_assert.r_con[kappa..]
                    .iter()
                    .fold(F128::ONE, |acc, &x| acc * (F128::ONE + x));
                let dot = |c: &[F128]| -> F128 {
                    eq_con
                        .iter()
                        .zip(c)
                        .fold(F128::ZERO, |acc, (e, v)| acc + *e * *v)
                };
                (w * dot(mt.ty.a_const()), w * dot(mt.ty.b_const()))
            };
            (el_g0, el_run_n, a_sum_n, b_sum_n)
        } else {
            (Vec::new(), F128::ZERO, F128::ZERO, F128::ZERO)
        }
    }
}

/// (v, c) counters up to an op index — the walker every pin shares.
fn vc_at(ops: &[Op], end: usize) -> (usize, usize) {
    let (mut v, mut c) = (0usize, 0usize);
    for op in &ops[..end] {
        match op {
            Op::SqueezeScalar => c += 1,
            Op::SqueezeSlice(n) => c += n,
            Op::ObserveScalar => v += 1,
            Op::ObserveSlice(n) => v += n,
            _ => {}
        }
    }
    (v, c)
}

/// fin ordinal of the op at `end` = finalizing ops strictly before it.
fn fin_at(ops: &[Op], end: usize) -> usize {
    ops[..end].iter().filter(|o| o.finalizes()).count()
}

/// The gate slots a child-tape region emits into. The builder shares these
/// slots across all child regions.
/// The recursion envelope and strict Fast nodes place the two independent
/// child BLAKE workloads in identical slots; the other families still add
/// rows, not columns. The `le`/`resid` caches fill on demand during emission;
/// cache hits require same-shape children (the keyed constructor parameters
/// must match, which the merge test asserts by requiring one shared circuit).
pub(super) struct ChildSlots {
    pub(super) nu: usize,
    pub(super) q: CollapsedSlots,
    pub(super) macs: SlotId,
    pub(super) fold_macs: SlotId,
    pub(super) zcr: SlotId,
    pub(super) mrs: SlotId,
    pub(super) spine: SlotId,
    pub(super) spine256: SlotId,
    pub(super) alslot: SlotId,
    pub(super) le: Vec<(usize, SlotId)>,
    /// The residual region's keyed slot cache (`emit_residual_region`'s
    /// `leaf_slot`). Key scheme: `600` = the shared MacGate (pre-seeded,
    /// so close-out rows land on `macs` instead of a duplicate type);
    /// `701` = the shared extension-field MAC; `100 + pl` = the base-field
    /// ResidualGate at that suffix-fold count; `310 + width` and
    /// `1000 + width` = base/extension prefix gates; and `880..=882` = the
    /// three shared extension-field residual relations.
    pub(super) resid: Vec<(usize, SlotId)>,
}

impl ChildSlots {
    #[cfg(test)]
    pub(super) fn new(sb: &mut ShapeBuilder, nu2: usize, spread_w: usize) -> Self {
        Self::new_with_b3_split(sb, nu2, spread_w, false)
    }

    #[cfg(test)]
    fn new_with_b3_split(
        sb: &mut ShapeBuilder,
        nu2: usize,
        spread_w: usize,
        split_b3: bool,
    ) -> Self {
        let macs = sb.slot(MacGate::new());
        let fold_macs = sb.slot(MacGate::new());
        let mac256 = sb.slot(MacGate256::new());
        let b3 = sb.slot(Blake3Gate { nu: nu2 });
        let b3_alt = split_b3.then(|| sb.slot(Blake3Gate { nu: nu2 }));
        ChildSlots {
            nu: nu2,
            q: CollapsedSlots {
                b3,
                b3_alt,
                swap: sb.slot(SwapGate { nu: nu2 }),
                spread: sb.slot(BitSpreadGate {
                    ty: BitSpreadTable::new(spread_w),
                    nu: nu2,
                }),
                pow: sb.slot(PowMaskGate { nu: nu2 }),
                family: Some(sb.slot(FamilyTransposeTileGate { nu: nu2 })),
            },
            macs,
            fold_macs,
            zcr: sb.slot(ZcRoundGate::new()),
            mrs: sb.slot(MergedRoundGate::new()),
            spine: sb.slot(SpineGate::new()),
            spine256: sb.slot(SpineGate256::new()),
            alslot: sb.slot(AssistLayerGate::new()),
            le: Vec::new(),
            // Key 600 pre-seeds the SHARED MacGate into the residual cache:
            // emit_residual_region's close-out rows land on the same slot
            // instead of registering a duplicate type.
            resid: vec![(600, macs), (701, mac256)],
        }
    }

    /// The ENVELOPE constructor (wall 2): the same canonical declaration
    /// order [`declare_envelope_slots`] gives every envelope outer, so all
    /// their registry digests agree. Every keyed entry pre-seeds the
    /// demand caches; emission that would need a slot OUTSIDE the envelope
    /// set creates a new type and fails the digest pin loudly.
    pub(super) fn new_env(sb: &mut ShapeBuilder, nu2: usize, env: &EnvShape) -> Self {
        let mut cache: Vec<(usize, SlotId)> = Vec::new();
        let q = declare_envelope_slots(sb, nu2, &mut cache, env);
        let take = |k: usize| {
            cache
                .iter()
                .find(|&&(c, _)| c == k)
                .expect("an envelope slot")
                .1
        };
        ChildSlots {
            nu: nu2,
            q,
            macs: take(600),
            fold_macs: take(602),
            zcr: take(500),
            mrs: take(400),
            spine: take(0),
            spine256: take(700),
            alslot: take(601),
            le: vec![(8, take(8)), (808, take(808))],
            // The residual-region cache inherits every entry in its key
            // namespaces: the shared macs, base residual variants, the
            // three shared F256 residual relations, and both prefix slots.
            resid: cache
                .iter()
                .filter(|&&(k, _)| {
                    matches!(k, 600 | 701)
                        || (100..200).contains(&k)
                        || (310..400).contains(&k)
                        || (880..=882).contains(&k)
                        || (1000..1100).contains(&k)
                })
                .cloned()
                .collect(),
        }
    }

    /// The keyed cache view `pad_envelope_counts` consumes — envelope path
    /// only (`new_env`).
    pub(super) fn env_cache(&self) -> Vec<(usize, SlotId)> {
        let mut v = vec![
            (600, self.macs),
            (602, self.fold_macs),
            (500, self.zcr),
            (400, self.mrs),
            (0, self.spine),
            (700, self.spine256),
            (601, self.alslot),
        ];
        v.extend(self.le.iter().map(|&(n, s)| (n, s)));
        v.extend(self.resid.iter().filter(|&&(k, _)| k != 600).cloned());
        v
    }

    /// Every element-class slot, for the outer prover's slot inputs.
    pub(super) fn element_slot_ids(&self) -> Vec<SlotId> {
        let mut v = vec![
            self.macs,
            self.fold_macs,
            self.zcr,
            self.mrs,
            self.spine,
            self.spine256,
            self.alslot,
        ];
        v.extend(self.le.iter().map(|&(_, s)| s));
        // Key 600 is the SHARED MacGate seed (already listed as `macs`).
        v.extend(
            self.resid
                .iter()
                .filter(|&&(k, _)| k != 600)
                .map(|&(_, s)| s),
        );
        v
    }
}

/// A child region's z_skip wires, by flavor. RS: the one fused-squeeze
/// output — the merge assembly derives the 64 Lagrange row lows from it
/// IN-CIRCUIT ([`emit_lagrange_lows`]; no publish, no checker rebuild).
/// AG: the seed squeezes' outputs and the r₁ nonce's stream-word wire —
/// the merge assembly publishes them beside the point and the row lows,
/// binds the decode IN-CIRCUIT from these wires
/// ([`emit_ag_point_binding`]: hash, PoW, fiber membership), and leaves
/// the nonce range + the lows-to-functional items to the native checker
/// ([`check_ag_skip_publics`]); the in-circuit lows derivation
/// (`emit_ag_lows`) is measured-rejected — see docs/ag-recursion-plan.md.
pub(super) enum ZskipWires {
    Rs(Wire),
    Ag { seed_w: [Wire; 2], nonce_w: Wire },
}

/// What one emitted child region hands back: where its public block starts,
/// the walk counts the checker needs, and the assertion-emission wires.
/// The child tail's EXPECTED publish schedule, every width an expression
/// over the TAPE's own geometry — the independent reference the emitted
/// table ([`ChildRegion::tail_schedule`]) is held against at every build.
pub(super) fn expected_child_tail_schedule(ct: &ChildTape<'_>) -> Vec<(&'static str, usize)> {
    let jag_vals = ct.jag.rs.len()
        + ct.jag
            .groups
            .iter()
            .map(|(combo, dense)| usize::from(combo.is_some()) + dense.len())
            .sum::<usize>();
    vec![
        ("chain ordinals ga+mg", 2),
        (
            "element zc+lc ends",
            if ct.el_assert.is_some() { 2 } else { 0 },
        ),
        ("anchor end", 1),
        ("spine t_final", 2),
        ("intake target", 1),
        ("intake running", 1),
        ("residual accs", 2 * ct.levels.len() * ct.yr_len),
        ("residual inner", 2),
        ("sigma value", 1),
        ("sigma point", ct.mu_i),
        ("jagged claim values", jag_vals),
    ]
}

pub(super) struct ChildRegion {
    pub(super) pub_base: usize,
    pub(super) n_query_pub: usize,
    pub(super) n_tail: usize,
    /// The published tail's `(name, width)` schedule, as emitted — held
    /// against [`expected_child_tail_schedule`] at every build.
    pub(super) tail_schedule: Vec<(&'static str, usize)>,
    pub(super) structure_claim_w: Vec<(Vec<Wire>, Vec<Wire>, Wire)>,
    /// The jagged assertion's value wires (the count win), in emission
    /// order: rs claims, then per group the combo and its dense members —
    /// the fresh-claim surfaces a merge fold connects to.
    pub(super) jag_w: Vec<Wire>,
    /// The claims' IDENTITY wires (the points-connect): σ — the anchor
    /// round squeezes, shared by every claim of the region — and per claim
    /// (jag_w order) the row wires: Eq claims carry z_col coordinate wires
    /// (constant coords ride zw/ow), Combo claims carry the γ_pd
    /// coefficient wires in term order (addresses are registry constants,
    /// bound by the fold side's shared constant publics).
    pub(super) jag_sig_w: Vec<Wire>,
    pub(super) jag_row_w: Vec<Vec<Wire>>,
    /// The boolean MatrixAssertion's wires: the zc mlv round rhos (round
    /// order — [dim6 | x_outer | x_inner_rest]), the lc round rhos (round
    /// order — rr is these reversed), and the absorbed z_partial words.
    pub(super) b_mlv_w: Vec<Wire>,
    pub(super) b_lc_w: Vec<Wire>,
    pub(super) b_zpartial_w: Vec<Wire>,
    /// Reported matrix evaluations, constrained by the in-circuit scalar
    /// closure and connected to the aggregate fold's fresh claim values.
    pub(super) mat_eval_w: Vec<(Wire, Wire)>,
    /// The z_skip surface, by flavor — see [`ZskipWires`].
    pub(super) zskip: ZskipWires,
    /// The residual close-out's prefix slot (and width) — reusable by a
    /// caller emitting more prefix products into the same builder.
    pub(super) pf: (SlotId, usize),
    /// The child's PUBLIC SEGMENT as witness wires (the H(publics) region's
    /// inputs, in the child's declaration order). Application-statement
    /// plumbing — the hash-chain adjacency connect — reads through these.
    pub(super) child_pub_w: Vec<Wire>,
}

/// Emit ONE child's complete deferred-verifier region — chain, query phase,
/// wiring GKR, element PIOP, multipoint intake + anchor expect, W-rounds,
/// spine, residual, sigma emission — into `sb`, publishing exactly what
/// [`check_child_region`] walks. The tape supplies each ordinal and native value.
pub(super) fn emit_child_region(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    b3_slot: SlotId,
    ct: &ChildTape<'_>,
    vals: &mut Vec<F128>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
    consts: &mut Vec<(F128, Wire)>,
) -> ChildRegion {
    let child_q = CollapsedSlots {
        b3: b3_slot,
        ..cs.q
    };
    let trace = &ct.trace;
    let stream = &ct.stream;
    let chals = &ct.chals[..];
    let levels = &ct.levels[..];
    let geo = &ct.geo[..];
    // `None` for a boolean-only (chain) child: the element PIOP emission,
    // its two publics and its ChildRegion wires all vanish together.
    let el_rec = ct.el.as_ref();
    let n_el_pd = if el_rec.is_some() { 2 } else { 0 };
    let n_log_i = ct.n_log_i;
    let _n_runs = ct.bounds_i.len();

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
        &ct.bytes,
        vals,
        consts,
        &ct.pub_payloads,
        &ct.cross,
    );
    let pub_w = emit_h_publics_region(sb, child_q, iv2, ct, stream, &ww, vals, consts);
    let cap_w = cap_wires(stream, &ww, &ct.cap_pays);
    let (to_publish, level_accs, query_positions) = emit_query_phase(
        sb,
        child_q,
        iv2,
        &leafeval,
        levels,
        geo,
        &ct.lvl_src,
        trace,
        &outs,
        chals,
        &cap_w,
        vals,
        consts,
        hints,
    );
    let ga_w = outs[trace.squeezes[ct.ga_fin][0]][0];
    let mg_w = outs[trace.squeezes[ct.mg_fin][0]][0];

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
    let mrs = cs.mrs;
    let spine = cs.spine;
    // The assert-zero anchor: a dedicated zero public NO gate consumes,
    // so the zero-delta outputs connected into its class add no
    // dataflow edges (connecting them to the ubiquitous `zw` creates
    // cycles — the acyclicity check draws producer→consumer edges).
    vals.push(F128::ZERO);
    let zassert = sb.public_input();

    let (pt_w, mid_w, live_w, sig_w) = emit_wiring_gkr_and_recombination(
        sb, cs, ct, trace, &outs, &wv, &pub_w, n_el_pd, vals, zw, ow, zassert,
    );

    assert_eq!(
        ct.mp_o.val_vs.len(),
        256 + ct.n_p,
        "the R=2 + P schedule spans both claim kinds"
    );
    let (mp_pws, mp_rho2_w, mp_sig_w, anc_w) = walker_common::emit_multipoint_intake(
        sb, cs, trace, &outs, &wv, &ct.mp_o, ct.m_mp2, zw, ow,
    );

    let t_final = emit_ligerito_spine(sb, cs, ct, trace, &outs, &wv, &level_accs, zw, ow);

    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region_shared(
        sb,
        cs,
        ct,
        trace,
        &outs,
        &wv,
        &to_publish,
        &query_positions,
        t_final,
        zw,
        ow,
    );

    let (tgt_w, runw) = emit_family_h_and_intake_boundary(
        sb, cs, ct, trace, &outs, &wv, pfslot, pf_w, vals, consts, zw, ow,
    );

    let el_pub = emit_element_piop_rounds(sb, cs, ct, trace, &outs, &wv, vals, zw, ow, zassert);

    let (jag_w, jag_row_w, mlv_pw, lc_pw) = emit_anchor_expect(
        sb,
        cs,
        ct,
        trace,
        &outs,
        &mp_pws,
        &mp_rho2_w,
        &mp_sig_w,
        anc_w,
        &pt_w,
        (pfslot, pf_w),
        n_el_pd,
        vals,
        consts,
        zw,
        ow,
        zassert,
    );

    // The aggregate verifier's scalar closures are part of the recursive
    // relation too.  The reported A/B values below become fold claims; these
    // equations prevent a prover from choosing values that discharge against
    // the matrices but do not reproduce the child verifier's running target.
    let mat_eval_w: Vec<(Wire, Wire)> = ct
        .bool_assert
        .evals
        .iter()
        .map(|&(a, b)| {
            vals.push(a);
            let aw = sb.input();
            vals.push(b);
            let bw = sb.input();
            (aw, bw)
        })
        .collect();
    let mat_alpha_w = outs[trace.squeezes[ct.bl_alpha.1][0]][0];
    let mat_x_inner_w: Vec<Wire> = (0..ct.bool_assert.x_inner_rest.len())
        .map(|j| {
            let m = if j == 0 { 0 } else { n_log_i + j };
            mlv_pw[m].1
        })
        .collect();
    let mat_rr_w: Vec<Wire> = lc_pw.iter().map(|&(_, w)| w).collect();
    let zpartial_ws: Vec<Wire> = (0..Z_PARTIAL_WORDS).map(|i| wv(ct.zp_v + i)).collect();
    let mut beta_wires = vec![None; ct.inner.built.shape.registry.num_boolean()];
    let mut pin_wires = Vec::with_capacity(ct.sigma_native.boolean_pins.len());
    let mut lcb_w = assertion_mac(
        sb,
        spine,
        wv(ct.zc_finals_v + 1),
        mat_alpha_w,
        wv(ct.zc_finals_v),
        zw,
    );
    for (k, (type_index, _, value)) in ct.sigma_native.boolean_pins.iter().enumerate() {
        let beta = outs[trace.squeezes[ct.betas_b[k].1][0]][0];
        beta_wires[*type_index] = Some(beta);
        vals.push(*value);
        let eps_w = sb.input();
        assert_eq!(
            *value, ct.eps_n[k],
            "static pin claim equals lincheck prefix"
        );
        pin_wires.push((*type_index, eps_w));
        lcb_w = assertion_mac(sb, spine, lcb_w, beta, eps_w, zw);
    }
    for (&g_v, &(_, fin)) in ct.lc_msg_vs.iter().zip(&ct.lc_rounds_b) {
        let rho_w = outs[trace.squeezes[fin][0]][0];
        lcb_w = sb.gate(mrs, &[lcb_w, wv(g_v), wv(g_v + 1), rho_w])[0];
    }
    emit_boolean_reported_check(
        sb,
        spine,
        pfslot,
        pf_w,
        &ct.inner.built.shape.registry,
        mat_alpha_w,
        &mat_x_inner_w,
        &mat_rr_w,
        &zpartial_ws,
        &beta_wires,
        &mat_eval_w,
        lcb_w,
        zw,
        ow,
    );

    let el_zc_rho_w: Vec<Wire> = el_rec
        .map(|el_rec| {
            el_rec
                .zc_rounds
                .iter()
                .map(|&(_, rfin, _)| outs[trace.squeezes[rfin][0]][0])
                .collect()
        })
        .unwrap_or_default();
    if let (Some(el_assert), Some((_, el_lcw, _, _, el_alpha_w))) = (&ct.el_assert, el_pub) {
        let el_eval_w: Vec<(Wire, Wire)> = el_assert
            .evals
            .iter()
            .map(|&(a, b)| {
                vals.push(a);
                let aw = sb.input();
                vals.push(b);
                let bw = sb.input();
                (aw, bw)
            })
            .collect();
        let inner_union = UnionInstance::new(
            &ct.inner.built.shape.registry,
            ct.inner.built.shape.counts.clone(),
        );
        let el_r_col_w: Vec<Wire> = el_rec
            .expect("element transcript")
            .lc_rounds
            .iter()
            .rev()
            .map(|&(_, fin, _)| outs[trace.squeezes[fin][0]][0])
            .collect();
        emit_element_reported_check(
            sb,
            spine,
            pfslot,
            pf_w,
            &inner_union,
            el_alpha_w,
            &el_zc_rho_w[n_log_i..],
            &el_r_col_w,
            wv(ct.gammas_o[ct.z_ix.expect("element z_eval index")].val_v),
            &el_eval_w,
            el_lcw,
            zw,
            ow,
        );
    }
    let element_values = el_pub.map(|(_, _, a, b, _)| (a, b));
    let element_point = el_rec.map(|_| &el_zc_rho_w[n_log_i..]);
    let boolean_point: Vec<Wire> = mlv_pw[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
    let structure_claim_w = circuit_structure_claim_wires(
        &ct.sigma_native,
        &pt_w,
        mid_w,
        live_w,
        sig_w,
        &boolean_point,
        &pin_wires,
        element_point,
        element_values,
        zw,
        ow,
    );

    // Everything publishes HERE, after every public input is declared
    // (`built.public` lists entries in declaration order): the query
    // block (alphas + level accs), then THE TAIL SCHEDULE — the table
    // below IS the child's published wire format, `check_child_region`'s
    // independent positional walk is its backstop, and the tape pin
    // holds its `(name, width)` list.
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
    let tail = [
        // The published chain ordinals (GKR alpha, multipoint gamma).
        TailEntry::new("chain ordinals ga+mg", vec![ga_w, mg_w]),
        // The element chain ends (mixed children only).
        TailEntry::new(
            "element zc+lc ends",
            el_pub.map_or_else(Vec::new, |(el_zr, el_lcw, _, _, _)| vec![el_zr, el_lcw]),
        ),
        TailEntry::new("anchor end", vec![anc_w]),
        TailEntry::new("spine t_final", t_final.to_vec()),
        TailEntry::new("intake target", vec![tgt_w]),
        TailEntry::new("intake running", vec![runw]),
        TailEntry::new(
            "residual accs",
            resid_pub.iter().flatten().flatten().copied().collect(),
        ),
        TailEntry::new("residual inner", inner_w.to_vec()),
        // The SIGMA ASSERTION emission (route B, in-circuit): the value
        // is the deferred s_sigma stream word — the SAME wire the rhs
        // input check just consumed — and the point is the GKR's own
        // accumulated squeeze wires.
        TailEntry::new("sigma value", vec![sig_w]),
        TailEntry::new("sigma point", pt_w.clone()),
        // The JAGGED ASSERTION emission (the count win): raw W claim
        // values in emission order (rs, then per group combo + dense
        // members), checker-held against the deferred export.
        TailEntry::new("jagged claim values", jag_w.clone()),
    ];
    let (n_tail, tail_schedule) = publish_tail(sb, &tail, |_, _| {});
    let n_query_pub: usize = 2 * levels.len() + levels.iter().map(|l| l.a_count).sum::<usize>();
    ChildRegion {
        pub_base,
        n_query_pub,
        n_tail,
        tail_schedule,
        structure_claim_w,
        jag_w,
        jag_sig_w: mp_sig_w.clone(),
        jag_row_w,
        b_mlv_w: mlv_pw.iter().map(|&(_, w)| w).collect(),
        b_lc_w: ct
            .lc_rounds_b
            .iter()
            .map(|&(_, fin)| outs[trace.squeezes[fin][0]][0])
            .collect(),
        b_zpartial_w: (0..Z_PARTIAL_WORDS).map(|i| wv(ct.zp_v + i)).collect(),
        mat_eval_w,
        zskip: zskip_wires(&ct.zskip, &outs, trace, stream, &ww),
        pf: (pfslot, pf_w),
        child_pub_w: pub_w,
    }
}

/// ---- ROUND 2: the H(publics) region (v2 statement binding) ----
/// The returned wires ARE the child's public segment — the recombination
/// folds them.
#[allow(clippy::too_many_arguments)]
fn emit_h_publics_region(
    sb: &mut ShapeBuilder,
    child_q: CollapsedSlots,
    iv2: [Wire; 2],
    ct: &ChildTape<'_>,
    stream: &Stream,
    ww: &[Option<Wire>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) -> Vec<Wire> {
    let pays = payload_words(stream);
    assert_eq!(pays[4].len(), 2, "the publics digest payload is 32 bytes");
    let dw = [
        ww[pays[4][0]].expect("digest word wired"),
        ww[pays[4][1]].expect("digest word wired"),
    ];
    emit_publics_hash(
        sb,
        child_q,
        iv2,
        &ct.inner.built.witness.public,
        dw,
        vals,
        consts,
    )
}

/// ---- the WIRING GKR in-circuit ----
/// ---- ROUND 4: the recombination + f == g, in-circuit ----
#[allow(clippy::too_many_arguments)]
fn emit_wiring_gkr_and_recombination(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    ct: &ChildTape<'_>,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    pub_w: &[Wire],
    n_el_pd: usize,
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
) -> (Vec<Wire>, Wire, Wire, Wire) {
    let macs = cs.macs;
    let zcr = cs.zcr;
    let n_log_i = ct.n_log_i;
    let g = &ct.gkr;
    let alpha_w = outs[trace.squeezes[g.alpha_fin][0]][0];
    let beta_w = outs[trace.squeezes[g.beta_fin][0]][0];
    // The grand products agree: a COPY CONSTRAINT on the tops (every
    // former published-zero delta in this region is now a connect — the
    // proof itself fails on a broken identity; no public, no checker item).
    let (mut cl_w, mut cr_w) = (wv(g.top_v), wv(g.top_v + 1));
    sb.connect(cl_w, cr_w);
    let mut pt_w: Vec<Wire> = Vec::new();
    for lr in &g.layers {
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
        // The layer close: run == vl0·vl1 + lambda·(vr0·vr1).
        let (vl0, vl1) = (wv(lr.v_v), wv(lr.v_v + 1));
        let (vr0, vr1) = (wv(lr.v_v + 2), wv(lr.v_v + 3));
        let pl = sb.gate(macs, &[zw, vl0, vl1])[0];
        let pr = sb.gate(macs, &[zw, vr0, vr1])[0];
        let gate_w = sb.gate(macs, &[pl, lam_w, pr])[0];
        sb.connect(gate_w, run_w);
        // The claim fold: claim' = v0 + c·(v0 + v1).
        let ck_w = outs[trace.squeezes[lr.ck_fin][0]][0];
        let sl = sb.gate(macs, &[vl0, vl1, ow])[0];
        let sr = sb.gate(macs, &[vr0, vr1, ow])[0];
        cl_w = sb.gate(macs, &[vl0, ck_w, sl])[0];
        cr_w = sb.gate(macs, &[vr0, ck_w, sr])[0];
        pt_next.push(ck_w);
        pt_w = pt_next;
    }
    assert_eq!(
        pt_w.len(),
        ct.mu_i,
        "the GKR point spans the inner cell space"
    );
    // The input checks under the LIVE-IDENTITY padding: M̂(ρ) and livê(ρ),
    // bound through the digest-keyed circuit-structure claims folded by the
    // parent.
    vals.push(ct.mid_n);
    let mid_w = sb.public_input();
    vals.push(ct.live_n);
    let live_w = sb.public_input();
    // The two input checks, as published-zero deltas.
    let (f_w, g_w, sig_w) = (wv(g.fgs_v), wv(g.fgs_v + 1), wv(g.fgs_v + 2));
    let l1 = sb.gate(macs, &[f_w, alpha_w, mid_w])[0];
    let l2 = sb.gate(macs, &[l1, beta_w, live_w])[0];
    let l3 = sb.gate(macs, &[l2, ow, live_w])[0];
    let l4 = sb.gate(macs, &[l3, ow, ow])[0];
    sb.connect(l4, cl_w);
    let r1 = sb.gate(macs, &[g_w, alpha_w, sig_w])[0];
    let r2 = sb.gate(macs, &[r1, beta_w, live_w])[0];
    let r3 = sb.gate(macs, &[r2, ow, live_w])[0];
    let r4 = sb.gate(macs, &[r3, ow, ow])[0];
    sb.connect(r4, cr_w);

    let le8 = match cs.le.iter().find(|&&(n, _)| n == 8) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(LeafEvalGate::new(8));
            cs.le.push((8, s));
            s
        }
    };
    let gather_w: Vec<Wire> = (0..ct.n_pd - n_el_pd)
        .map(|i| wv(ct.gammas_o[n_el_pd + i].val_v))
        .collect();
    emit_recombination(
        sb,
        cs.fold_macs,
        le8,
        pub_w,
        &gather_w,
        &pt_w,
        n_log_i,
        ct.n_pub_slots_c,
        f_w,
        g_w,
        zw,
        ow,
    );
    (pt_w, mid_w, live_w, sig_w)
}

/// ---- the LIGERITO SPINE ----
#[allow(clippy::too_many_arguments)]
fn emit_ligerito_spine(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    ct: &ChildTape<'_>,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    level_accs: &[[Wire; 2]],
    zw: Wire,
    ow: Wire,
) -> [Wire; 2] {
    walker_common::emit_ligerito_spine_walk(
        sb,
        cs.spine256,
        outs,
        trace,
        wv,
        &ct.levels,
        &ct.inner_pd2,
        ct.start_v,
        level_accs,
        zw,
        ow,
    )
}

/// ---- the RESIDUAL region (shared emitter) ----
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn emit_residual_region_shared(
    sb: &mut ShapeBuilder,
    cs: &mut ChildSlots,
    ct: &ChildTape<'_>,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    to_publish: &[Vec<Wire>],
    query_positions: &[Vec<Wire>],
    t_final: [Wire; 2],
    zw: Wire,
    ow: Wire,
) -> (Vec<Vec<[Wire; 2]>>, [Wire; 2], (SlotId, usize)) {
    let levels = &ct.levels[..];
    let geo = &ct.geo[..];
    let inner_pd2 = &ct.inner_pd2;
    let yr_wires: Vec<[Wire; 2]> = (0..ct.yr_len)
        .map(|y| [wv(ct.yr_v2 + 2 * y), wv(ct.yr_v2 + 2 * y + 1)])
        .collect();
    let (resid_pub, inner_w, (pfslot, pf_w)) = emit_residual_region(
        sb,
        &mut cs.resid,
        levels,
        geo,
        to_publish,
        query_positions,
        &ct.w_resid,
        inner_pd2.fin,
        &yr_wires,
        trace,
        outs,
        zw,
        ow,
    );
    // THE CLOSURE, in-circuit: the residual side's inner and the spine's
    // t_r are the same statement scalar — a copy constraint, not a
    // checker item (both stay published as test cross-checks).
    sb.connect(inner_w[0], t_final[0]);
    sb.connect(inner_w[1], t_final[1]);
    (resid_pub, inner_w, (pfslot, pf_w))
}

/// ---- FAMILY H + the merged intake boundary ----
/// The transpose/equality dot products and inverse-Moore/Frobenius
/// recombination are all recursive-circuit arithmetic. Every source is
/// an existing transcript/proof wire; the native target is retained only
/// as a published test oracle below.
#[allow(clippy::too_many_arguments)]
fn emit_family_h_and_intake_boundary(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    ct: &ChildTape<'_>,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    pfslot: SlotId,
    pf_w: usize,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    zw: Wire,
    ow: Wire,
) -> (Wire, Wire) {
    walker_common::emit_family_h_boundary(
        sb,
        cs,
        trace,
        outs,
        wv,
        &ct.rs_recs,
        &ct.rs_gam_fins,
        &ct.mp_o,
        &ct.gammas_o,
        &ct.w_rounds,
        &ct.inner_pd2,
        pfslot,
        pf_w,
        vals,
        consts,
        zw,
        ow,
    )
}

/// ---- the ELEMENT PIOP rounds in-circuit (mixed children only) ----
/// Zerocheck rounds are ZcRoundGate rows (tau slice wires as eq weights,
/// g0 advice + zero deltas); lincheck rounds are MergedRoundGate rows.
/// The entry is DERIVED: va = ea + a_sum, vb = eb + b_sum, entry =
/// va + alpha·vb — only the two constant-strip sums are advice.
#[allow(clippy::too_many_arguments)]
fn emit_element_piop_rounds(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    ct: &ChildTape<'_>,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
) -> Option<(Wire, Wire, Wire, Wire, Wire)> {
    let macs = cs.macs;
    let zcr = cs.zcr;
    let mrs = cs.mrs;
    let el_rec = ct.el.as_ref();
    el_rec.map(|el_rec| {
        let mut el_zr = zw;
        for (k, &(gv, rfin, _)) in el_rec.zc_rounds.iter().enumerate() {
            let t_w = squeeze_word_wire(outs, trace, el_rec.tau_fin, k);
            let rho_w = outs[trace.squeezes[rfin][0]][0];
            vals.push(ct.el_g0[k]);
            let g0w = sb.input();
            let o = sb.gate(zcr, &[el_zr, wv(gv), wv(gv + 1), t_w, rho_w, g0w, ow]);
            sb.connect(o[0], zassert);
            el_zr = o[1];
        }
        let el_alpha_w = outs[trace.squeezes[el_rec.alpha_fin][0]][0];
        let ea_w = wv(el_rec.eab_v);
        let eb_w = wv(el_rec.eab_v + 1);
        vals.push(ct.a_sum_n);
        let asum_w = sb.public_input();
        vals.push(ct.b_sum_n);
        let bsum_w = sb.public_input();
        let va_w = sb.gate(macs, &[ea_w, asum_w, ow])[0];
        let vb_w = sb.gate(macs, &[eb_w, bsum_w, ow])[0];
        let mut el_lcw = sb.gate(macs, &[va_w, el_alpha_w, vb_w])[0];
        for &(gv, rfin, _) in &el_rec.lc_rounds {
            let rho_w = outs[trace.squeezes[rfin][0]][0];
            el_lcw = sb.gate(mrs, &[el_lcw, wv(gv), wv(gv + 1), rho_w])[0];
        }
        (el_zr, el_lcw, asum_w, bsum_w, el_alpha_w)
    })
}

/// ---- the anchor EXPECT in-circuit, at R = 2 AND P > 0 ----
/// expect = Σ_i γ^{128i}·ĝ(ρ″)·(w_i·DP_i) over the RS statements + Σ_k
/// γ^{256+k}·eq(ρ,ρ″)·(w_k·DP_k) over the P groups; claim == expect
/// publishes as a zero-delta. ĝ's inverse-Frobenius points ride as advice
/// bound by forward squaring deltas.
/// The c-point's 7 baked inner constants are the zerocheck's friendly
/// challenges — the RS ghash set or the AG set, by the tape's flavor
/// (baked constants are free in-circuit either way).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn emit_anchor_expect(
    sb: &mut ShapeBuilder,
    cs: &ChildSlots,
    ct: &ChildTape<'_>,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    mp_pws: &[Wire],
    mp_rho2_w: &[Wire],
    mp_sig_w: &[Wire],
    anc_w: Wire,
    pt_w: &[Wire],
    pf: (SlotId, usize),
    n_el_pd: usize,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
) -> (
    Vec<Wire>,
    Vec<Vec<Wire>>,
    Vec<(F128, Wire)>,
    Vec<(F128, Wire)>,
) {
    let (pfslot, pf_w) = pf;
    let macs = cs.macs;
    let spine = cs.spine;
    let chals = &ct.chals[..];
    let w_rounds = &ct.w_rounds[..];
    let el_rec = ct.el.as_ref();
    let m_mp2 = ct.m_mp2;
    let n_log_i = ct.n_log_i;
    let k_cols_i = ct.k_cols_i;
    let t_vals_b = walker_common::baked_inner_t_vals(&ct.zskip);

    // The statements' points as (native value, wire) pairs, pinned against
    // the native claims: ab = [LAST lc round | zc mlv rounds 1..1+ν | lc
    // rounds REVERSED tail], c = the zerocheck's r_rest verbatim.
    let mlv_pw: Vec<(F128, Wire)> = ct
        .zc_rounds_b
        .iter()
        .map(|&(ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let lc_pw: Vec<(F128, Wire)> = ct
        .lc_rounds_b
        .iter()
        .rev()
        .map(|&(ch, fin)| (chals[ch], outs[trace.squeezes[fin][0]][0]))
        .collect();
    let mut xab_pw: Vec<(F128, Wire)> = vec![lc_pw[0]];
    xab_pw.extend_from_slice(&mlv_pw[1..1 + n_log_i]);
    xab_pw.extend_from_slice(&lc_pw[1..]);
    walker_common::extend_const_coords(&mut xab_pw, &ct.x_ab_n, zw, ow);
    let (outer_ch_b, outer_fin_b) = ct.outer_b;
    let mut xc_pw: Vec<(F128, Wire)> = (0..ct.zc_rounds_b.len())
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
    walker_common::extend_const_coords(&mut xc_pw, &ct.x_c_n, zw, ow);
    for (i2, (&(nv, _), &xn)) in xab_pw.iter().zip(&ct.x_ab_n).enumerate() {
        assert_eq!(nv, xn, "ab point coord {i2} is the located wire");
    }
    for (i2, (&(nv, _), &xn)) in xc_pw.iter().zip(&ct.x_c_n).enumerate() {
        assert_eq!(nv, xn, "c point coord {i2} is the located wire");
    }

    // ĝ(ρ″): advice square-root chains for ρ^(2^-j), bound by forward
    // squaring deltas y·y + prev = 0.
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
            for t2 in 0..m_mp2 {
                let y = frob_inv(rinv_n2[t2]);
                rinv_n2[t2] = y;
                vals.push(y);
                let yw = sb.input();
                let d = sb.gate(spine, &[zw, zw, zw, rinv_w[t2], zw, zw, yw, yw, zw])[3];
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
    // e_at = eq(ρ, ρ″) for the group coefficients.
    let e_at_w = {
        let factors: Vec<(Wire, Wire)> = rho_mrg_w
            .iter()
            .copied()
            .zip(mp_rho2_w.iter().copied())
            .collect();
        walker_common::prefix_product(sb, pfslot, pf_w, zw, ow, &factors)
    };
    // THE COUNT WIN: the counts used to enter the parent's circuit HERE —
    // per-run boundary eq products with the jagged run boundaries (the
    // prefix sums of the child's per-type heights) baked as ow/zw, then
    // per-statement run-weight enumerations consuming them. All of it is
    // gone: each statement's raw W arrives as a PUBLISHED CLAIM VALUE on
    // the jagged layout table (the deferred verify's own export, keyed by
    // the child digest), checker-held here and discharged at the ROOT of
    // the accumulation tree — the eps discipline, ported. The claim's
    // points are wires this region already carries (σ = the anchor round
    // squeezes, z_cols = statement point wires, γ_pd = squeezes); nothing
    // count-shaped remains in the circuit.
    let mut jag_w: Vec<Wire> = Vec::new();
    // The claims' IDENTITY wires (the points-connect): σ shared per
    // region, and per claim — in jag_w order — the row-identity wires the
    // merge fold's absorbed words connect to.
    let mut jag_row_w: Vec<Vec<Wire>> = Vec::new();
    // Per RS statement: the published w, the DP, the coefficient.
    let alslot = cs.alslot;
    let mut expect_w = zw;
    for (si, xs) in [&xab_pw, &xc_pw].iter().enumerate() {
        let z_row_w: Vec<Wire> = xs[1..1 + n_log_i].iter().map(|&(_, w)| w).collect();
        vals.push(ct.jag.rs[si].value);
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
    // Per group: the γ-baked one-hot combo publishes as ONE value; each
    // dense (element) member publishes its raw eq value with its γ_pd
    // applied by a MAC on the squeeze wire — the exported decomposition,
    // reassembled in wires. Coefficient γ^{256+k}·e_at as before.
    for (g_ix, members) in ct.groups_ix.iter().enumerate() {
        let (combo, dense) = &ct.jag.groups[g_ix];
        let hots: Vec<bool> = members
            .iter()
            .map(|&i2| {
                ct.pd_pts[i2][n_log_i..]
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
                // wires, in member order == the assertion's term order.
                let gws: Vec<Wire> = members
                    .iter()
                    .zip(&hots)
                    .filter(|&(_, &h)| h)
                    .map(|(&i2, _)| {
                        let pd = &ct.gammas_o[i2];
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
                continue;
            }
            let (_, c) = d_it.next().expect("a dense entry per non-hot member");
            let pd = &ct.gammas_o[i2];
            let gpd_w = squeeze_word_wire(outs, trace, pd.fin, pd.squeeze_offset);
            vals.push(c.value);
            let d_w = sb.input();
            jag_w.push(d_w);
            // The dense claim's identity: its z_col coordinate wires —
            // constant coords ride zw/ow, the rest are the element PIOP's
            // own squeeze wires (the mapping the constructor pinned).
            jag_row_w.push(
                (0..k_cols_i)
                    .map(|jj| {
                        let coord = ct.pd_pts[i2][n_log_i + jj];
                        if coord == F128::ZERO {
                            zw
                        } else if coord == F128::ONE {
                            ow
                        } else {
                            let el_rec = el_rec.expect("element pd claim");
                            if i2 == 0 {
                                outs[trace.squeezes[el_rec.zc_rounds[n_log_i + jj].1][0]][0]
                            } else {
                                let n_lc = el_rec.lc_rounds.len();
                                outs[trace.squeezes[el_rec.lc_rounds[n_lc - 1 - jj].1][0]][0]
                            }
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
                if members[0] >= n_el_pd {
                    pt_w[layer]
                } else {
                    let el_rec = el_rec.expect("element pd claim");
                    outs[trace.squeezes[el_rec.zc_rounds[layer].1][0]][0]
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
    // The join: the anchor's folded claim equals the in-circuit expect.
    sb.connect(anc_w, expect_w);
    (jag_w, jag_row_w, mlv_pw, lc_pw)
}

/// Walk one emitted child region's public block and hold every published
/// value against the tape's native replicas.
/// Returns the number of public entries consumed (the region's publish
/// tail), so a multi-region caller can walk region after region.
pub(super) fn check_child_region(public: &[F128], ct: &ChildTape<'_>, r: &ChildRegion) -> usize {
    let chals = &ct.chals[..];
    // The query-phase boundary: published alphas are the recorded
    // challenges and each accumulator equals the native enforced sum.
    {
        let mut at = r.pub_base;
        // The openings bind to the absorbed caps by COPY CONSTRAINT (the
        // in-circuit cap tree) — no per-query publics, no checker walk.
        for (li, lvl) in ct.levels.iter().enumerate() {
            for j in 0..lvl.a_count {
                assert_eq!(public[at + j], chals[lvl.a_ch + j], "L{li} alpha {j}");
            }
            at += lvl.a_count;
        }
        for (li, want) in ct.native_sums.iter().enumerate() {
            assert_eq!(
                F256::new(public[at + 2 * li], public[at + 2 * li + 1]),
                *want,
                "L{li} enforced sum matches the native replica"
            );
        }
        assert_eq!(
            at + 2 * ct.native_sums.len(),
            r.pub_base + r.n_query_pub,
            "the query publics walk consumed its whole block"
        );
    }
    let base2 = r.pub_base + r.n_query_pub;
    assert_eq!(
        public[base2], chals[ct.ga_c],
        "the GKR alpha derives in-circuit"
    );
    assert_eq!(
        public[base2 + 1],
        chals[ct.mg_c],
        "the multipoint gamma derives in-circuit"
    );
    // The GKR round/close/input identities, the element zc round deltas,
    // T_m == anchor.v and claim == expect are COPY CONSTRAINTS now — no
    // publics, no checker items; the proof itself carries them.
    let el_base = base2 + 2;
    let mp_base = if let Some(el_assert) = &ct.el_assert {
        assert_eq!(
            public[el_base], ct.el_run_n,
            "the element zc chain ends at the native running claim"
        );
        // THE INDEPENDENT CLOSE: the in-circuit lincheck chain ends exactly
        // at the native ElementAssertion's target.
        assert_eq!(
            public[el_base + 1],
            el_assert.target,
            "the element lc chain ends at the native assertion's target"
        );
        el_base + 2
    } else {
        el_base
    };
    assert_eq!(
        public[mp_base], ct.anc_end_n,
        "the anchor rounds end at the native claim"
    );
    // THE LIGERITO CLOSE: the in-circuit spine reaches the native t_r.
    assert_eq!(
        F256::new(public[mp_base + 1], public[mp_base + 2]),
        ct.t_final_n,
        "the spine's final t_r matches the native replay"
    );
    // The merged intake is fully constrained; its publications retain the
    // native replay as a test oracle.
    assert_eq!(
        public[mp_base + 3],
        ct.native_target,
        "the computed RS target matches the native gamma-combination"
    );
    assert_eq!(
        public[mp_base + 4],
        ct.native_running,
        "the W-rounds fold the target to the native running claim"
    );
    // The residual region against the shared native replica — and THE
    // CLOSURE: the residual-side inner and the spine's t_r are the same
    // statement scalar, both held against published circuit outputs.
    let inner_n = check_residual_publics(
        public,
        mp_base + 5,
        &ct.levels,
        &ct.geo,
        &ct.w_resid,
        ct.inner_pd2.ch,
        &observed_f256(&ct.vals_rec, ct.yr_v2, ct.yr_len),
        chals,
    );
    assert_eq!(
        inner_n, ct.t_final_n,
        "inner == t_r: the mixed statement closes"
    );
    // The sigma assertion, as the accumulator would read it: the value and
    // the mu point coordinates, matched against the native claim.
    let sig_base = mp_base + 5 + 2 * ct.levels.len() * ct.yr_len + 2;
    assert_eq!(
        public[sig_base],
        ct.inner.proof.wiring().gkr.s_sigma_eval,
        "the emitted sigma value is the proof's deferred evaluation"
    );
    let sig_rho = &public[sig_base + 1..sig_base + 1 + ct.mu_i];
    {
        // The emitted pair IS a SigmaAssertion, rebuilt from the outer's
        // PUBLIC SEGMENT ALONE — equal to the deferred verify's own, and it
        // discharges against the inner circuit's sigma table.
        let sa = SigmaAssertion {
            rho: sig_rho.to_vec(),
            nu: ct.inner.built.shape.circuit.cells().nu(),
            base_bits: ct.sigma_native.base_bits,
            masked_id_value: ct.mid_n,
            live_value: ct.live_n,
            value: public[sig_base],
            boolean_pins: ct.sigma_native.boolean_pins.clone(),
            element_constants: ct.sigma_native.element_constants.clone(),
        };
        assert_eq!(sa.rho, ct.sigma_native.rho, "the emitted sigma point");
        assert_eq!(sa.value, ct.sigma_native.value, "the emitted sigma value");
        assert_eq!(sa.nu, ct.sigma_native.nu, "the emitted sigma split");
        assert!(
            sa.check(&ct.inner.built.shape.circuit),
            "the emitted sigma assertion discharges against the inner circuit"
        );
    }
    // The jagged assertion's value surfaces (the count win), in emission
    // order — rs claims, then per group the combo and its dense members —
    // each the deferred export's own raw claim value. The full claims
    // (points included) discharge against the child's layout, so the
    // published values are exactly what a merge fold's fresh-claim
    // surfaces connect to.
    {
        let jag_base = sig_base + 1 + ct.mu_i;
        let mut expect_vals: Vec<F128> = ct.jag.rs.iter().map(|c| c.value).collect();
        for (combo, dense) in &ct.jag.groups {
            if let Some(c) = combo {
                expect_vals.push(c.value);
            }
            for (_, c) in dense {
                expect_vals.push(c.value);
            }
        }
        for (j, want) in expect_vals.iter().enumerate() {
            assert_eq!(
                public[jag_base + j],
                *want,
                "jagged claim value {j} matches the deferred export"
            );
        }
        assert_eq!(
            jag_base + expect_vals.len(),
            r.pub_base + r.n_query_pub + r.n_tail,
            "the jagged publics close the region's tail"
        );
    }
    r.n_query_pub + r.n_tail
}

/// A Merkle leaf need not be a whole number of 64-byte blocks — and the
/// opening gate hashes the partial final block correctly.
///
/// This is the shape a MIXED CIRCUIT union produces: it commits `num_lanes`
/// ACTIVE lanes, `dense_words.div_ceil(2^log_dim)`, an arbitrary integer
/// (the top lanes are definitionally zero and never encoded — see
/// `ligerito`'s high-bit-lane commit).
///
/// BLAKE3 hashes 61 words = 976 bytes as one chunk of 16 blocks whose last
/// carries `b = 16`, and the compression's `b` is a free input to the gate,
/// so the only cost is one zero-padding wire. Pinned against
/// `merkle::hash_leaf` itself at every width, whole blocks and partial
/// alike.
#[test]
pub(super) fn partial_block_leaves_hash_correctly() {
    for words in [1usize, 3, 4, 8, 61, 64] {
        let (depth, leaf_bytes) = (2usize, 16 * words);
        let nu = 6usize;
        let mut rng = Rng(0x_B10C_0000 ^ words as u64);
        let tree = Tree::new(depth, leaf_bytes, &mut rng);
        let pos = 2usize;

        let mut sb = ShapeBuilder::new(nu);
        let slots = CollapsedSlots {
            b3: sb.slot(Blake3Gate { nu }),
            b3_alt: None,
            swap: sb.slot(SwapGate { nu }),
            spread: sb.slot(BitSpreadGate {
                ty: BitSpreadTable::new(depth),
                nu,
            }),
            pow: sb.slot(PowMaskGate { nu }),
            family: None,
        };
        let mut vals: Vec<F128> = Vec::new();
        let iv_w = pack8(&IV);
        vals.extend_from_slice(&iv_w);
        let iv = [
            sb.fixed_public_input(iv_w[0]),
            sb.fixed_public_input(iv_w[1]),
        ];
        let leaf = tree.leaf(pos);
        let leaf_w: Vec<Wire> = (0..words)
            .map(|w| {
                vals.push(leaf_word(leaf, 16 * w));
                sb.public_input()
            })
            .collect();
        vals.push(F128::new(pos as u64, 0));
        let idx_w = sb.public_input();
        let (root, _) = emit_opening(
            &mut sb, slots, iv, &leaf_w, idx_w, depth, 0, 0, None, &mut vals,
        );
        sb.publish(root[0]);
        sb.publish(root[1]);
        let shape = sb.finish().expect("the opening circuit builds");
        let hints: Vec<[u32; SLOT_WORDS]> = tree.siblings(pos);
        let hint_refs: Vec<&(dyn Any + Sync)> =
            hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
        let built = shape.run(&vals, &hint_refs);

        // The in-circuit chunk chain reproduces `hash_leaf` on a leaf that
        // is NOT block-aligned, and the fold reaches the real root.
        let n = built.public.len();
        assert_eq!(
            [built.public[n - 2], built.public[n - 1]],
            digest_words(&hash_to_digest(&tree.root)),
            "width {words}: the opening folds to the tree root"
        );
    }
}
