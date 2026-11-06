//! The affine inner opening INSIDE a child walker: the parse of a recorded
//! merged-open tape whose inner is an [`AffineProof`], and the emission of
//! the whole affine verifier against the walker's FS-chain wires.
//!
//! The walker keeps everything before the inner opening (the merged
//! sumcheck's rounds, the multipoint assist, the family-H boundary). This
//! module emits the opening's three regions:
//!
//! - the QUERY PHASE: the level openings (Merkle paths against the absorbed
//!   caps) feeding the affine
//!   consistency and final checks ([`super::affine_query`]);
//! - the SPINE: the affine claim fold from the absorbed `q_eval`
//!   ([`super::affine_fold`]);
//! - the RESIDUAL: `⟨yr_lo, W⟩` ([`super::affine_residual`]), CONNECTED to
//!   the claim fold's output.
//!
//! Nothing is published: the opening closes in-circuit.

use flock_core::{
    circuit::builder::SlotId,
    pcs::{PcsParams, ligerito::affine::AffineProof},
};
use flock_transcript::transcript_record::TranscriptOp as Op;

use crate::{
    r1cs_hashes::fs_chain::FsChainTrace,
    tower::{
        CollapsedSlots, F128, HashKind, Lvl, MacGate, PARENT, PrefixGate, ResidualWeightsGate,
        RoundRec, ShapeBuilder, Wire,
        affine_fold::{AffineFoldGate, AffineLevelWires, JoinWires, emit_affine_claim_fold},
        affine_gates::{Arith, FoldChunkGate, NovelPrefixGate, PREFIX},
        affine_query::{
            AffineFinalLevel, AffineQueryLevel, emit_affine_consistency, emit_affine_final_check,
        },
        affine_residual::{EqContext, QueryContext, emit_affine_residual},
        cw,
        envelope::slot_cached,
        geometry::{affine_level_geometry, affine_level_sources},
        hash_to_digest, pack_params,
        query::emit_opening,
        squeeze_word_wire, strat_scheds,
        tape::{AffineLevel, AffineOodRec, AffineOpenTape, OpenPrefix, parse_affine_open},
    },
};

/// A level's `(cap, opened rows, flat paths)`.
pub(super) type LevelSource<'p> = (&'p [[u8; 32]], &'p Vec<Vec<F128>>, &'p Vec<[u8; 32]>);

/// The parsed affine inner of one child.
pub(super) struct AffineInner<'p> {
    pub(super) initial_ood: Vec<AffineOodRec>,
    pub(super) levels: Vec<AffineLevel>,
    /// Value index of `q_eval`, the opening's absorbed target.
    pub(super) q_v: usize,
    pub(super) lvl_src: Vec<LevelSource<'p>>,
    pub(super) geo: Vec<Lvl>,
    /// Per level: folds `k` and own column count `c`.
    pub(super) ks: Vec<usize>,
    pub(super) cs: Vec<usize>,
    pub(super) log_n: usize,
}

impl<'p> AffineInner<'p> {
    /// Locate the affine opening on `ops` and build its geometry, with
    /// every opened row verified natively against its cap. Returns the
    /// merged-open prefix the walker consumes beside it.
    pub(super) fn parse(
        ops: &[Op],
        proof: &'p AffineProof,
        chals: &[F128],
        pcs: &PcsParams,
    ) -> (Self, OpenPrefix) {
        let vc = pcs.ligerito_verifier_config().expect("registered config");
        let mut ks = vec![vc.initial_k];
        ks.extend(&vc.recursive_ks);
        let mut cs = vec![vc.initial_log_msg_cols];
        cs.extend(&vc.recursive_log_msg_cols);
        let log_n = vc.initial_k + vc.initial_log_msg_cols;
        let yr_len = (1usize << cs[cs.len() - 1]) + ks[ks.len() - 1];
        let AffineOpenTape {
            prefix,
            initial_ood,
            levels,
        } = parse_affine_open(ops, 32 * proof.initial_cap.len(), &ks, yr_len);
        assert_eq!(levels.len(), ks.len(), "one tape level per config level");
        assert_eq!(proof.openings.len(), ks.len(), "one opening per level");
        let lvl_src = affine_level_sources(proof);
        let q_chs: Vec<usize> = levels.iter().map(|l| l.q_ch).collect();
        let geo = affine_level_geometry(
            &ks,
            &q_chs,
            &lvl_src,
            chals,
            HashKind::Blake3,
            &strat_scheds(pcs),
        );
        let q_v = prefix.target_v;
        (
            Self {
                initial_ood,
                levels,
                q_v,
                lvl_src,
                geo,
                ks,
                cs,
                log_n,
            },
            prefix,
        )
    }
}

/// What an envelope declares for the affine inner: the slot types are
/// shape-free ([`super::affine_gates`]), so the only shape parameter is the
/// subspace ladder's length — the deepest level's `c + 1` over the tower's
/// chain leaf and outer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct AffineEnvLayouts {
    /// The ladder's length `n` (`Ŵ_0..Ŵ_{n-1}`); 0 = no affine inner.
    pub(super) ladder_n: usize,
}

/// Envelope slot keys of the affine types (the keyed-cache scheme of
/// [`super::envelope::slot_cached`]). The eq prefix rides the shared
/// `PrefixGate(8)` at `310 + 8`.
pub(super) const KEY_FOLD: usize = 900;
pub(super) const KEY_LADDER: usize = 901;
pub(super) const KEY_CHUNK: usize = 902;
pub(super) const KEY_MAC: usize = 903;
pub(super) const KEY_NPREFIX: usize = 904;
const KEY_PREFIX: usize = 310 + PREFIX;

impl AffineEnvLayouts {
    /// The layouts every affine-profile shape in `shapes` needs.
    pub(super) fn of(shapes: &[&PcsParams]) -> Self {
        let ladder_n = shapes
            .iter()
            .filter(|p| p.profile.affine_fold())
            .map(|pcs| {
                let vc = pcs.ligerito_verifier_config().expect("registered config");
                let deepest = vc
                    .recursive_log_msg_cols
                    .iter()
                    .copied()
                    .chain([vc.initial_log_msg_cols])
                    .max()
                    .unwrap();
                deepest + 1
            })
            .max()
            .unwrap_or(0);
        Self { ladder_n }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ladder_n == 0
    }

    /// Every envelope key these layouts occupy (beyond the shared prefix).
    pub(super) fn keys(&self) -> Vec<usize> {
        if self.is_empty() {
            return Vec::new();
        }
        vec![KEY_FOLD, KEY_LADDER, KEY_CHUNK, KEY_MAC, KEY_NPREFIX]
    }

    /// Declare every affine type into the keyed `cache`, in canonical order.
    pub(super) fn declare(&self, sb: &mut ShapeBuilder, cache: &mut Vec<(usize, SlotId)>) {
        if self.is_empty() {
            return;
        }
        slot_cached(sb, cache, KEY_FOLD, AffineFoldGate::new);
        slot_cached(sb, cache, KEY_LADDER, || {
            ResidualWeightsGate::with_weights(self.ladder_n)
        });
        slot_cached(sb, cache, KEY_CHUNK, FoldChunkGate::new);
        slot_cached(sb, cache, KEY_MAC, MacGate::new);
        slot_cached(sb, cache, KEY_NPREFIX, NovelPrefixGate::new);
    }
}

/// `(fold, (ladder length, ladder), chunk, mac, novel prefix, eq prefix)`.
type SlotSet = (SlotId, (usize, SlotId), SlotId, SlotId, SlotId, SlotId);

/// The slots the affine inner emits into. On the envelope path they are
/// the pre-declared ones ([`AffineSlots::from_cache`]); off the envelope
/// they are created on first use.
pub(super) struct AffineSlots {
    slots: Option<SlotSet>,
    frozen: bool,
}

impl AffineSlots {
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self {
            slots: None,
            frozen: false,
        }
    }

    /// The envelope's pre-declared affine slots.
    pub(super) fn from_cache(cache: &[(usize, SlotId)], lay: &AffineEnvLayouts) -> Self {
        let take = |k: usize| {
            cache
                .iter()
                .find(|e| e.0 == k)
                .expect("an envelope affine slot")
                .1
        };
        Self {
            slots: (!lay.is_empty()).then(|| {
                (
                    take(KEY_FOLD),
                    (lay.ladder_n, take(KEY_LADDER)),
                    take(KEY_CHUNK),
                    take(KEY_MAC),
                    take(KEY_NPREFIX),
                    take(KEY_PREFIX),
                )
            }),
            frozen: true,
        }
    }

    /// The slots this holds that the walker does not list elsewhere (the
    /// shared eq prefix is the walker's on the envelope path).
    pub(super) fn owned_slots(&self) -> Vec<SlotId> {
        match (self.slots, self.frozen) {
            (Some((f, (_, l), c, m, n, _)), true) => vec![f, l, c, m, n],
            (Some((f, (_, l), c, m, n, p)), false) => vec![f, l, c, m, n, p],
            (None, _) => Vec::new(),
        }
    }

    /// `(fold, ladder, arith)` for an inner whose deepest level needs a
    /// ladder of `n`.
    fn get(
        &mut self,
        sb: &mut ShapeBuilder,
        n: usize,
        zw: Wire,
        ow: Wire,
    ) -> (SlotId, SlotId, Arith) {
        if self.slots.is_none() {
            assert!(!self.frozen, "the envelope declares no affine slots");
            self.slots = Some((
                sb.slot(AffineFoldGate::new()),
                (n, sb.slot(ResidualWeightsGate::with_weights(n))),
                sb.slot(FoldChunkGate::new()),
                sb.slot(MacGate::new()),
                sb.slot(NovelPrefixGate::new()),
                sb.slot(PrefixGate::new(PREFIX)),
            ));
        }
        let (fold, (have, ladder), chunk, mac, nprefix, prefix) = self.slots.unwrap();
        assert!(have >= n, "the declared ladder covers W_0..W_{}", n - 1);
        (
            fold,
            ladder,
            Arith {
                mac,
                chunk,
                nprefix,
                prefix,
                zw,
                ow,
            },
        )
    }
}

/// Emit one level's openings: the cap
/// layers above the shallowest stratum folded from the ABSORBED cap wires,
/// each query hashed from its leaf and climbed to its stratum terminal
/// (top-stratum queries one edge further, bound to the DERIVED layer-1
/// node so no edge runs back into the chain). Returns per query the leaf
/// word wires and the derived position wire.
#[allow(clippy::too_many_arguments)]
fn emit_level_openings(
    sb: &mut ShapeBuilder,
    q: CollapsedSlots,
    iv: [Wire; 2],
    g: &Lvl,
    (_cap, rows, paths): LevelSource<'_>,
    q_words: &[Wire],
    q_lows: &[u64],
    cap_w: &[[Wire; 2]],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    hints: &mut Vec<[u32; crate::tower::SLOT_WORDS]>,
) -> (Vec<Vec<Wire>>, Vec<Wire>) {
    let c_min = g.sched.summand_depths.last().copied().unwrap_or(g.c);
    assert!(
        g.c > 0,
        "depth-0 top summand (q = 1) unsupported in-circuit"
    );
    let n_layers = (g.c - c_min).max(1);
    let mut layers_w: Vec<Vec<[Wire; 2]>> = vec![cap_w.to_vec()];
    for _ in 0..n_layers {
        let params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
        let next: Vec<[Wire; 2]> = layers_w
            .last()
            .unwrap()
            .chunks(2)
            .map(|p| {
                let out = sb.gate(
                    q.b3,
                    &[iv[0], iv[1], p[0][0], p[0][1], p[1][0], p[1][1], params],
                );
                [out[0], out[1]]
            })
            .collect();
        layers_w.push(next);
    }
    let parent_params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
    let mut leaves = Vec::with_capacity(g.q);
    let mut positions = Vec::with_capacity(g.q);
    for k in 0..g.q {
        vals.extend_from_slice(&rows[k]);
        let leaf_w: Vec<Wire> = (0..g.raw_row_words).map(|_| sb.input()).collect();
        let (ck, stratum) = g.q_stratum(k);
        let open_depth = g.depth - ck;
        let (cv, position_w) = emit_opening(
            sb,
            q,
            iv,
            &leaf_w,
            q_words[k],
            open_depth,
            0,
            stratum << open_depth,
            Some(consts),
            vals,
        );
        let pos = g.q_pos(k, q_lows[k]);
        hints.extend(g.full_path(k, pos, paths).iter().map(hash_to_digest));
        let (bind, term) = if ck == g.c {
            let sib = cap_w[stratum ^ 1];
            let (l, r) = if stratum & 1 == 0 {
                (cv, sib)
            } else {
                (sib, cv)
            };
            let out = sb.gate(q.b3, &[iv[0], iv[1], l[0], l[1], r[0], r[1], parent_params]);
            ([out[0], out[1]], layers_w[1][stratum >> 1])
        } else {
            (cv, layers_w[g.c - ck][stratum])
        };
        sb.connect(bind[0], term[0]);
        sb.connect(bind[1], term[1]);
        leaves.push(leaf_w);
        positions.push(position_w);
    }
    (leaves, positions)
}

/// Emit the whole affine inner: openings, per-level consistency, final
/// checks, claim fold and residual, the residual total CONNECTED to the
/// claim. `rho_w` is the merged sumcheck's point in coordinate order (the
/// basis is `eq(ρ rotated left by the column count, ·)`; L0's OOD points
/// take `ρ`'s column coordinates). `wv` maps a value index to its stream
/// wire.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_affine_inner(
    sb: &mut ShapeBuilder,
    slots: &mut AffineSlots,
    q: CollapsedSlots,
    iv: [Wire; 2],
    ai: &AffineInner<'_>,
    trace: &FsChainTrace,
    outs: &[Vec<Wire>],
    wv: &impl Fn(usize) -> Wire,
    chals: &[F128],
    rho_w: &[Wire],
    cap_w: &[Vec<[Wire; 2]>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    hints: &mut Vec<[u32; crate::tower::SLOT_WORDS]>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
) {
    let sq = |fin: usize, j: usize| squeeze_word_wire(outs, trace, fin, j);
    let levels = &ai.levels;
    let fl = levels.len() - 1;
    assert_eq!(rho_w.len(), ai.log_n, "the opening point spans the table");
    let cols = ai.log_n - ai.ks[0];
    let max_c = *ai.cs.iter().max().unwrap();
    let (fold, ladder, ar) = slots.get(sb, max_c + 1, zw, ow);

    // The basis eq(ρ rotated, ·) with weight one; every OOD claim joins
    // the claim and contributes an EQ weight at its own point.
    let mut rotated = rho_w[cols..].to_vec();
    rotated.extend_from_slice(&rho_w[..cols]);
    let mut eqs = vec![EqContext {
        seed: ow,
        point: rotated,
    }];
    let ood = |o: &AffineOodRec, l0: bool, eqs: &mut Vec<EqContext>| -> JoinWires {
        let beta = sq(o.beta_fin, 0);
        let mut point: Vec<Wire> = (0..o.z_len).map(|j| sq(o.z_fin, j)).collect();
        if l0 {
            assert_eq!(
                o.z_len, ai.ks[0],
                "lane-major L0 OOD: fresh row coordinates"
            );
            point.extend_from_slice(&rho_w[..cols]);
        }
        eqs.push(EqContext { seed: beta, point });
        (beta, wv(o.y_v))
    };
    let init: Vec<JoinWires> = ai
        .initial_ood
        .iter()
        .map(|o| ood(o, true, &mut eqs))
        .collect();

    let mut lvl_w: Vec<AffineLevelWires> = Vec::with_capacity(levels.len());
    let mut all_rs: Vec<Wire> = Vec::new();
    let mut qctx: Vec<QueryContext> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let (k, c) = (ai.ks[li], ai.cs[li]);
        let g = &ai.geo[li];
        let mut rounds = Vec::with_capacity(k);
        let mut gammas = Vec::with_capacity(k);
        for rd in &lvl.rounds {
            let grid: [Wire; 5] = std::array::from_fn(|e| wv(rd.grid_v + e));
            let ch: [Wire; 3] = std::array::from_fn(|e| sq(rd.fin, e));
            rounds.push((grid, ch));
            gammas.push((ch[0], ch[1]));
            all_rs.push(ch[2]);
        }
        let q_words: Vec<Wire> = (0..lvl.q_count).map(|j| sq(lvl.q_fin, j)).collect();
        let q_lows: Vec<u64> = (0..lvl.q_count).map(|j| chals[lvl.q_ch + j].lo).collect();
        let (rows, xs) = emit_level_openings(
            sb,
            q,
            iv,
            g,
            ai.lvl_src[li],
            &q_words,
            &q_lows,
            &cap_w[li],
            vals,
            consts,
            hints,
        );
        if li == fl {
            let yr_v = lvl.yr_v.expect("the final level absorbs yr");
            let yr: Vec<Wire> = (0..(1usize << c) + k).map(|s| wv(yr_v + s)).collect();
            let fl_lvl = AffineFinalLevel {
                rows,
                xs,
                gammas,
                yr: yr.clone(),
                log_cols: c,
            };
            emit_affine_final_check(sb, &ar, ladder, &fl_lvl, zassert);
            lvl_w.push(AffineLevelWires {
                rounds,
                ood: Vec::new(),
                consistency: None,
            });
            let claim = emit_affine_claim_fold(sb, fold, ar.mac, wv(ai.q_v), &init, &lvl_w);
            assert_eq!(all_rs.len() + c, ai.log_n, "the residual spans the rest");
            let total = emit_affine_residual(sb, &ar, &all_rs, &yr[..1 << c], &eqs, &qctx);
            sb.connect(total, claim);
            return;
        }
        let ov = lvl
            .overflow_v
            .expect("a non-final level absorbs its overflow");
        let overflow: Vec<Wire> = (0..k).map(|s| wv(ov + s)).collect();
        let joins: Vec<JoinWires> = lvl.ood.iter().map(|o| ood(o, false, &mut eqs)).collect();
        let (a_fin, _, a_count) = lvl.alpha.expect("a non-final level batches its queries");
        let alpha: Vec<Wire> = (0..a_count).map(|j| sq(a_fin, j)).collect();
        let ql = AffineQueryLevel {
            rows,
            xs,
            gammas,
            overflow,
            log_cols: c,
            alpha,
        };
        let out = emit_affine_consistency(sb, &ar, ladder, &ql);
        let beta = sq(lvl.beta.expect("consistency beta").0, 0);
        qctx.push(QueryContext {
            beta,
            dim: c,
            ladders: out.ladders,
            weights: out.eq,
        });
        lvl_w.push(AffineLevelWires {
            rounds,
            ood: joins,
            consistency: Some((beta, out.enforced)),
        });
    }
    unreachable!("the final level returns");
}

/// The merged sumcheck's point `ρ` in coordinate order, as wires: each
/// W-round's squeeze, in the `w_coords` order the walker already derives.
pub(super) fn rho_wires(
    w_coords: &[RoundRec],
    outs: &[Vec<Wire>],
    trace: &FsChainTrace,
) -> Vec<Wire> {
    w_coords
        .iter()
        .map(|rr| squeeze_word_wire(outs, trace, rr.fin, 0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower::{
        config::test_config,
        envelope::{declare_envelope_slots, envelope_shape},
    };

    /// The envelope's cell budget per type (`--nocapture`): every IO word
    /// of every declared type is one cell slot of the outer's wiring space.
    #[test]
    fn envelope_io_census() {
        let cfg = test_config();
        let leaf = PcsParams {
            m: 24,
            log_inv_rate: cfg.leaf_profile().log_inv_rate(),
            log_batch_size: 6,
            profile: cfg.leaf_profile(),
            num_lanes: None,
            merkle_hash: HashKind::Blake3,
        };
        let env = envelope_shape().with_affine(&leaf, cfg.outer_profile());
        let mut sb = ShapeBuilder::new(env.nu);
        let mut cache = Vec::new();
        let q = declare_envelope_slots(&mut sb, env.nu, &mut cache, &env);
        let mut total = 0;
        for (name, s) in [
            ("b3", q.b3),
            ("swap", q.swap),
            ("spread", q.spread),
            ("pow", q.pow),
        ] {
            let w = sb.slot_inputs(s) + sb.slot_outputs(s);
            total += w;
            println!("{name:>8} {w}");
        }
        total += sb.slot_inputs(q.b3) + sb.slot_outputs(q.b3);
        let f = q.family.unwrap();
        total += sb.slot_inputs(f) + sb.slot_outputs(f);
        let mut aff = 0;
        for &(k, s) in &cache {
            let w = sb.slot_inputs(s) + sb.slot_outputs(s);
            total += w;
            if k >= 900 {
                aff += w;
            }
            println!("{k:>8} {w}");
        }
        println!("total IO words {total} (affine {aff})");
    }
}
