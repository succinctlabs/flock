use super::*;
use flock_core::lincheck::build_eq_table;
use flock_transcript::transcript_record::TranscriptOp as Op;

#[cfg(test)]
use flock_hash::blake3_compress;

/// Emit the whole QUERY PHASE — every level's Merkle openings against the
/// absorbed caps, plus the leaf-eval accumulators — as circuit rows.
///
/// This is the class-agnostic half of a deferred verifier: it reads the
/// proof's own rows and paths, wires each query's challenge word straight
/// into the opening (no masking gadget — the relation reads the low `depth`
/// columns), and folds the opened rows against the fold challenges into one
/// accumulator per level.
///
/// Per level, `2^c − 1`
/// PARENT rows fold the ABSORBED cap wires (`cap_w`, from [`cap_wires`])
/// to one root in fixed positional order — no swaps, the cap layer IS the
/// tree's depth-`c` slice — and every opening runs FULL depth (path
/// siblings from the proof, cap-internal siblings recomputed natively from
/// the cap) and CONNECTS to that root. A copy constraint binds the cap to
/// each opening.
///
/// Appends the sibling `hints` and the public `vals` in declaration order;
/// returns the per-level alpha wires to publish AFTER every input is
/// declared, and the accumulators.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_query_phase(
    sb: &mut ShapeBuilder,
    slots: CollapsedSlots,
    iv: [Wire; 2],
    leafeval: &[flock_core::circuit::builder::SlotId],
    levels: &[OpenLevel],
    geo: &[Lvl],
    lvl_src: &[(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    outs: &[Vec<Wire>],
    chals: &[F128],
    cap_w: &[Vec<[Wire; 2]>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
) -> (Vec<Vec<Wire>>, Vec<[Wire; 2]>, Vec<Vec<Wire>>) {
    let mut to_publish: Vec<Vec<Wire>> = Vec::new();
    let mut level_accs: Vec<[Wire; 2]> = Vec::new();
    let mut query_positions: Vec<Vec<Wire>> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let g = &geo[li];
        let (_cap, rows, paths) = lvl_src[li];
        // Terminals: the upper layers from the cap wires to the shallowest
        // summand (2^c − 2^c_min PARENT rows, NO root) — each query's path
        // stops at its stratum depth and connects to its schedule-constant
        // terminal wire; no hints beyond the proof's own siblings. A
        // top-summand terminal is the ABSORBED cap layer, and a direct
        // connect there puts a gate output into a class the FS chain's
        // absorb row consumes — opening → chain → squeeze → opening:
        // Cyclic. So the binding layer is layer 1: top-summand openings
        // hash ONE level further (below, in the query loop) and connect to
        // the DERIVED node, which collision resistance binds to the
        // absorbed pair. Hence at least one layer is always built.
        let c_min = g.sched.summand_depths.last().copied().unwrap_or(g.c);
        assert!(
            g.c > 0,
            "depth-0 top summand (q = 1) unsupported in-circuit"
        );
        let n_layers = (g.c - c_min).max(1);
        let mut layers_w: Vec<Vec<[Wire; 2]>> = vec![cap_w[li].clone()];
        for _ in 0..n_layers {
            let params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
            let next: Vec<[Wire; 2]> = layers_w
                .last()
                .unwrap()
                .chunks(2)
                .map(|p| {
                    let out = sb.gate(
                        slots.b3,
                        &[iv[0], iv[1], p[0][0], p[0][1], p[1][0], p[1][1], params],
                    );
                    [out[0], out[1]]
                })
                .collect();
            layers_w.push(next);
        }
        // One PARENT params wire for the top-summand extension rows (the
        // `cw` helper is shadowed by the challenge wire inside the loop).
        let parent_params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
        // alpha words: chain outputs, PUBLISHED for the checker's expansion.
        let a_wires: Vec<Wire> = (0..lvl.a_count)
            .map(|j| squeeze_word_wire(outs, trace, lvl.a_fin, j))
            .collect();
        // v: this level's fold challenges, chain outputs, wired straight in.
        let v_wires: Vec<[Wire; 2]> = lvl
            .fold_fins
            .iter()
            .map(|&f| {
                [
                    squeeze_word_wire(outs, trace, f, 0),
                    squeeze_word_wire(outs, trace, f, 1),
                ]
            })
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        // The hi-group weights of the leaf-eval split: eq over the native
        // values of the fold challenges past the 8-lane gate's three.
        let le_vars = g.lanes.min(8).trailing_zeros() as usize;
        let le_groups = g.lanes >> le_vars;
        let v_hi: Vec<F256> = lvl.fold_chs[le_vars..]
            .iter()
            .map(|&i| F256::new(chals[i], chals[i + 1]))
            .collect();
        let hw =
            flock_multilinear::eq_table(&v_hi, F256::ONE, flock_multilinear::IndexOrder::LowToHigh);
        let zero = cw(sb, vals, consts, F128::ZERO);
        let mut acc = [zero, zero];
        let mut level_positions = Vec::with_capacity(g.q);
        // Zero wire for the fold's known-zero top lanes (only declared when
        // the committed row is narrower than the fold).
        let pad_w = if g.row_words < g.lanes {
            Some([zero, zero])
        } else {
            None
        };
        for k in 0..g.q {
            vals.extend_from_slice(&rows[k]);
            let leaf_w: Vec<Wire> = (0..g.raw_row_words).map(|_| sb.input()).collect();
            let cw = squeeze_word_wire(outs, trace, lvl.q_fin, k);
            let (ck, stratum) = g.q_stratum(k);
            let open_depth = g.depth - ck;
            let (cv, position_w) = emit_opening(
                sb,
                slots,
                iv,
                &leaf_w,
                cw,
                open_depth,
                0,
                stratum << open_depth,
                Some(consts),
                vals,
            );
            level_positions.push(position_w);
            // The proof's siblings truncate at the cap; the climb to the
            // stratum terminal still runs the full `d − c_k` rows, so
            // witgen reconstitutes the cap-fold tail as extra hints.
            let pos = g.q_pos(k, chals[lvl.q_ch + k].lo);
            hints.extend(g.full_path(k, pos, paths).iter().map(hash_to_digest));
            // Output-output connects: a multi-producer class with no gate
            // consumers — witgen asserts agreement, no dataflow cycle.
            let (bind, term) = if ck == g.c {
                // Top summand: hash one level past the cap with the
                // NEIGHBOUR cap word at constant direction (the stratum's
                // parity — no swap gate), and bind the DERIVED parent to
                // layer 1. Equality of the two layer-1 producers forces
                // cv == cap[stratum] by collision resistance, with every
                // edge forward.
                let sib = cap_w[li][stratum ^ 1];
                let (l, r) = if stratum & 1 == 0 {
                    (cv, sib)
                } else {
                    (sib, cv)
                };
                let out = sb.gate(
                    slots.b3,
                    &[iv[0], iv[1], l[0], l[1], r[0], r[1], parent_params],
                );
                ([out[0], out[1]], layers_w[1][stratum >> 1])
            } else {
                (cv, layers_w[g.c - ck][stratum])
            };
            sb.connect(bind[0], term[0]);
            sb.connect(bind[1], term[1]);
            // The fold reads the full `2^folds` domain: the committed words
            // then the definitionally-zero top lanes.
            let mut fold_w: Vec<[Wire; 2]> = leaf_w.iter().map(|&w| [w, zero]).collect();
            fold_w.resize(g.lanes, pad_w.unwrap_or(fold_w[0]));
            let lanes = g.lanes.min(8);
            for h in 0..le_groups {
                let mut a_in: Vec<Wire> = fold_w[lanes * h..lanes * (h + 1)]
                    .iter()
                    .flat_map(|p| *p)
                    .collect();
                a_in.extend(v_wires[..le_vars].iter().flat_map(|p| *p));
                let weight = hw[h] * aw[k];
                vals.push(weight.c0);
                vals.push(weight.c1);
                a_in.push(sb.input());
                a_in.push(sb.input());
                a_in.extend_from_slice(&acc);
                let out = sb.gate(leafeval[li], &a_in);
                acc = [out[0], out[1]];
            }
        }
        to_publish.push(a_wires);
        level_accs.push(acc);
        query_positions.push(level_positions);
    }
    (to_publish, level_accs, query_positions)
}

/// Emit the RESIDUAL region — the third shared piece of the deferred
/// verifier, after the FS chain and the query phase. Per query, the shared
/// extension-field residual gates derive the normalized `W_k` ladder once,
/// multiply the later-level challenges into a prefix three at a time, and
/// update each eight-position accumulator chunk. Together these compute
/// `induce_sumcheck_evaluate_at_residual` (the `next_s` chain from a
/// boundary-bound q_field, a prefix over the LATER levels' fold wires,
/// suffix subset products over the `2^yr` residual positions); the
/// close-out then assembles `eval_b` from gamma' and the W-round wires
/// through ONE `pl_full`-wide prefix slot. Shorter calls pad their `(a, b)`
/// blocks with zero pairs. It folds each OOD claim and each level accumulator.
/// It then dots the absorbed `yr` words into the residual-side `inner`.
///
/// Appends public `vals` in declaration order; the caller publishes the
/// returned accumulators and `inner` after all inputs are declared. It also
/// returns the prefix slot and its width,
/// `min(pl_full, 8)`, which the anchor-expect machinery reuses for its
/// chunked products — longer factor lists seed-chain across rows. (The
/// cap keeps the schema at 19 IO words instead of 2·pl_full + 3; every
/// gate cell-slot is also a wiring gather claim, so schema words are the
/// μ AND claim-count budget.)
///
/// The accumulator gate has eight positions (`chunk_log=3`)
/// at kappa 6 regardless of the proof's residual size. The real inner's
/// yr = 32 would otherwise push its schema to kappa 7-8. A yr > 8 region
/// runs as `2^(yr_log-3)` chunks of 8:
/// - the close-out claims' HIGH-bit eq factors ride the PREFIX SLOT
///   (seed = the claim's prefix product, factors = high coords vs the
///   chunk bits) — wire-bound, no new trust;
/// - the residual rows' high subset factor `sp_hi(h)` rides the CHECKER
///   tier (`awp = aw·sp_hi`, recomputed natively from the validated
///   position by `check_residual_publics` — the alpha-expansion trust
///   class; a wrong value fails the published accumulators).
/// The smallest supported residual domain, yr = 8, takes one chunk.
/// The close-out itself (per-position eq tensors, the beta combines, the
/// yr dot) is prefix + MacGate rows since Round 3 — no dedicated types.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_residual_region(
    sb: &mut ShapeBuilder,
    leaf_slot: &mut Vec<(usize, flock_core::circuit::builder::SlotId)>,
    levels: &[OpenLevel],
    geo: &[Lvl],
    alpha_wires: &[Vec<Wire>],
    query_positions: &[Vec<Wire>],
    w_rounds: &[RoundRec],
    inner_pd_fin: usize,
    yr_wires: &[[Wire; 2]],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    outs: &[Vec<Wire>],
    zw: Wire,
    ow: Wire,
) -> (
    Vec<Vec<[Wire; 2]>>,
    [Wire; 2],
    (flock_core::circuit::builder::SlotId, usize),
) {
    let yr_len = yr_wires.len();
    assert!(yr_len.is_power_of_two());
    let yr_log = yr_len.trailing_zeros() as usize;
    let chunk_log = yr_log.min(3);
    assert_eq!(
        chunk_log, 3,
        "the shared F256 residual gates operate on eight-position chunks"
    );
    let chunk = 1usize << chunk_log;
    let n_chunks = 1usize << (yr_log - chunk_log);
    let chw = |fin: usize| -> [Wire; 2] {
        [
            squeeze_word_wire(outs, trace, fin, 0),
            squeeze_word_wire(outs, trace, fin, 1),
        ]
    };
    let base_chw = |fin: usize| -> Wire { squeeze_word_wire(outs, trace, fin, 0) };
    let mut resid_pub: Vec<Vec<[Wire; 2]>> = Vec::new();
    let weights_slot = slot_cached(sb, leaf_slot, 880, ResidualWeightsGate256::new);
    let prefix3_slot = slot_cached(sb, leaf_slot, 881, ResidualPrefix3Gate256::new);
    let acc_slot = slot_cached(sb, leaf_slot, 882, ResidualAccGate256::new);
    let scalar_macs = slot_cached(sb, leaf_slot, 600, MacGate::new);
    assert_eq!(alpha_wires.len(), levels.len());
    assert_eq!(query_positions.len(), levels.len());
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len() - 1).sum();
        // The deepest weight this level touches is `weights[pl + yr_log - 1]`
        // (the chunk-high extension below), not `pl + chunk_log`: the old
        // bound under-checked whenever the residual domain has more than one
        // 8-chunk, so the m32 walk overran the gate instead of failing here.
        let lmc = pl + yr_log;
        assert!(
            lmc <= ResidualWeightsGate256::N_WEIGHTS,
            "the residual ladder needs {lmc} normalized weights"
        );
        assert_eq!(pl % 3, 0, "one Ligerito level contributes three folds");
        let ris_w: Vec<[Wire; 2]> = levels[li + 1..]
            .iter()
            .flat_map(|l| l.fold_fins.iter().skip(1).map(|&f| chw(f)))
            .collect();
        assert_eq!(alpha_wires[li].len(), lvl.a_count);
        assert_eq!(query_positions[li].len(), geo[li].q);
        // Expand eq(alpha, k) from the transcript challenge wires. For each
        // prior weight x and next coordinate r, the low/high children are
        // x(1+r) and xr. Both are MacGate rows, so no alpha-derived advice
        // enters the residual relation.
        let mut aw = vec![ow];
        for &r in &alpha_wires[li] {
            let old = aw;
            let mut next = Vec::with_capacity(2 * old.len());
            for &x in &old {
                next.push(sb.gate(scalar_macs, &[x, x, r])[0]);
            }
            for &x in &old {
                next.push(sb.gate(scalar_macs, &[zw, x, r])[0]);
            }
            aw = next;
        }
        assert!(aw.len() >= geo[li].q, "alpha tensor covers every query");
        let mut accs: Vec<[Wire; 2]> = (0..yr_len).map(|_| [zw, zw]).collect();
        for k in 0..geo[li].q {
            let qf = query_positions[li][k];
            let w_tail = sb.gate(weights_slot, &[qf, ow]);
            let mut weights = Vec::with_capacity(ResidualWeightsGate256::N_WEIGHTS);
            weights.push(qf);
            weights.extend(w_tail);
            let mut prefix = [ow, zw];
            for at in (0..pl).step_by(3) {
                let mut g_in = vec![prefix[0], prefix[1]];
                g_in.extend(ris_w[at..at + 3].iter().flat_map(|p| *p));
                g_in.extend_from_slice(&weights[at..at + 3]);
                g_in.push(ow);
                let out = sb.gate(prefix3_slot, &g_in);
                prefix = [out[0], out[1]];
            }
            let low_weights = &weights[pl..pl + 3];
            for h in 0..n_chunks {
                // Extend the transcript-derived alpha weight by the high
                // residual subset selected by this chunk. The relevant W_j
                // values are outputs of ResidualWeightsGate256, so this
                // replaces the former free aw*sp_hi advice.
                let mut awp = aw[k];
                for j in 0..(yr_log - chunk_log) {
                    if (h >> j) & 1 == 1 {
                        awp = sb.gate(scalar_macs, &[zw, awp, weights[pl + chunk_log + j]])[0];
                    }
                }
                let mut g_in = vec![awp, prefix[0], prefix[1]];
                g_in.extend_from_slice(low_weights);
                g_in.extend(accs[h * chunk..(h + 1) * chunk].iter().flat_map(|p| *p));
                let out = sb.gate(acc_slot, &g_in);
                for (dst, src) in accs[h * chunk..(h + 1) * chunk]
                    .iter_mut()
                    .zip(out.as_chunks::<2>().0.iter())
                {
                    *dst = [src[0], src[1]];
                }
            }
        }
        resid_pub.push(accs);
    }
    // The close-out. The ligerito layer sees ONE packed-direct claim:
    // (rho, q_eval) with gamma'; rho's coords are the W-round squeezes —
    // chain wires. The OOD claims are the same shape, seed = beta, point =
    // the squeezed z.
    let total_fold_count: usize = levels.iter().map(|l| l.fold_fins.len()).sum();
    let pl_full = levels[0].fold_fins.len()
        + levels[1..]
            .iter()
            .map(|l| l.fold_fins.len() - 1)
            .sum::<usize>();
    let ris_full: Vec<[Wire; 2]> = levels[0]
        .fold_fins
        .iter()
        .map(|&f| chw(f))
        .chain(
            levels[1..]
                .iter()
                .flat_map(|l| l.fold_fins.iter().skip(1).map(|&f| chw(f))),
        )
        .collect();
    // ROUND 3: the close-out's suffix/combine/dot arithmetic rides the
    // shared 4-word MacGate (cache key 600) plus the
    // prefix slot — the SuffixGate/PartialCombineGate/FinalDotGate types
    // are DISSOLVED: 51 schema words (each a cell slot AND a gather claim)
    // bought ~30 rows of work; as mac/prefix rows the same work is ~250
    // live-prefix-cheap rows and zero types.
    let macs = match leaf_slot.iter().find(|&&(k, _)| k == 701) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(MacGate256::new());
            leaf_slot.push((701, s));
            s
        }
    };
    let pf_w = total_fold_count.min(8);
    let pfslot = match leaf_slot.iter().find(|&&(k, _)| k == 1000 + pf_w) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(PrefixGate256::new(pf_w));
            leaf_slot.push((1000 + pf_w, s));
            s
        }
    };
    // Other deferred-verifier arithmetic is base-field valued and reuses
    // the original prefix type. Return that slot to the caller.
    let base_pfslot = match leaf_slot.iter().find(|&&(k, _)| k == 310 + pf_w) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(PrefixGate::new(pf_w));
            leaf_slot.push((310 + pf_w, s));
            s
        }
    };
    // Seed-chained prefix product: any factor list, `pf_w` per row.
    let prefix_chain =
        |sb: &mut ShapeBuilder, seed: [Wire; 2], factors: &[([Wire; 2], [Wire; 2])]| -> [Wire; 2] {
            let mut s = seed;
            for chunk_f in factors.chunks(pf_w) {
                let mut g_in = vec![s[0], s[1]];
                for (a, _) in chunk_f {
                    g_in.extend_from_slice(a);
                }
                g_in.extend(std::iter::repeat_n(zw, 2 * (pf_w - chunk_f.len())));
                for (_, b) in chunk_f {
                    g_in.extend_from_slice(b);
                }
                g_in.extend(std::iter::repeat_n(zw, 2 * (pf_w - chunk_f.len())));
                g_in.push(ow);
                g_in.push(zw);
                let out = sb.gate(pfslot, &g_in);
                s = [out[0], out[1]];
            }
            s
        };
    // A split commitment adds one coordinate variable per recursive level.
    // Folding that bit at r contributes phi(r) = 1 + r(1 + u). Express it
    // as the prefix factor 1 + r + r*u so the same F256 product gate binds
    // the coordinate transport used by the native verifier.
    let coordinate_factors = |sb: &mut ShapeBuilder, start_level: usize| {
        levels[start_level.max(1)..]
            .iter()
            .map(|level| {
                let r = chw(level.fold_fins[0]);
                let ru = emit_mac256(sb, macs, [zw, zw], r, [zw, ow]);
                (r, ru)
            })
            .collect::<Vec<_>>()
    };
    let mut evb_accs: Vec<[Wire; 2]> = (0..yr_len).map(|_| [zw, zw]).collect();
    // Fold one claim (prefix product p at full-yl coord wires) into the
    // accumulators: per position, ONE prefix row computes p·eq(coords, y)
    // (high bits chunk-shared, low bits per position; eq factor =
    // 1 + coord + [bit] in char 2) and ONE MacGate row accumulates it.
    let apply_suffix =
        |sb: &mut ShapeBuilder, evb_accs: &mut [[Wire; 2]], p: [Wire; 2], coords: &[[Wire; 2]]| {
            assert_eq!(coords.len(), yr_log, "the claim tail spans yr");
            for h in 0..n_chunks {
                let ph = if n_chunks == 1 {
                    p
                } else {
                    let factors: Vec<([Wire; 2], [Wire; 2])> = coords[chunk_log..]
                        .iter()
                        .enumerate()
                        .map(|(j, &cw2)| (cw2, [if (h >> j) & 1 == 1 { ow } else { zw }, zw]))
                        .collect();
                    prefix_chain(sb, p, &factors)
                };
                for y in 0..chunk {
                    let factors: Vec<([Wire; 2], [Wire; 2])> = coords[..chunk_log]
                        .iter()
                        .enumerate()
                        .map(|(j, &cw2)| (cw2, [if (y >> j) & 1 == 1 { ow } else { zw }, zw]))
                        .collect();
                    let py = prefix_chain(sb, ph, &factors);
                    let at2 = h * chunk + y;
                    evb_accs[at2] = emit_mac256(sb, macs, evb_accs[at2], py, [ow, zw]);
                }
            }
        };
    {
        assert_eq!(
            w_rounds.len(),
            pl_full + yr_log,
            "rho spans the dense domain"
        );
        let mut factors: Vec<([Wire; 2], [Wire; 2])> = w_rounds[..pl_full]
            .iter()
            .map(|rr| [base_chw(rr.fin), zw])
            .zip(ris_full.iter().copied())
            .collect();
        factors.extend(coordinate_factors(sb, 0));
        let pw = prefix_chain(sb, [base_chw(inner_pd_fin), zw], &factors);
        let coords: Vec<[Wire; 2]> = w_rounds[pl_full..]
            .iter()
            .map(|rr| [base_chw(rr.fin), zw])
            .collect();
        apply_suffix(sb, &mut evb_accs, pw, &coords);
    }
    for od in &levels[0].initial_ood {
        let folded = od.z_len - yr_log;
        assert_eq!(folded, ris_full.len(), "L0 OOD spans every fold");
        let initial_k = levels[0].fold_fins.len();
        let z_index = |j| l0_ood_z_index(od.z_len, initial_k, geo[0].row_words, j);
        let mut factors: Vec<([Wire; 2], [Wire; 2])> = (0..folded)
            .map(|j| {
                (
                    [squeeze_word_wire(outs, trace, od.z_fin, z_index(j)), zw],
                    ris_full[j],
                )
            })
            .collect();
        factors.extend(coordinate_factors(sb, 0));
        let pw = prefix_chain(sb, [base_chw(od.beta_fin), zw], &factors);
        let coords: Vec<[Wire; 2]> = (0..yr_log)
            .map(|j| {
                [
                    squeeze_word_wire(outs, trace, od.z_fin, z_index(folded + j)),
                    zw,
                ]
            })
            .collect();
        apply_suffix(sb, &mut evb_accs, pw, &coords);
    }
    for (li, lvl) in levels.iter().enumerate() {
        for od in &lvl.ood {
            let folded = od.z_len - yr_log;
            let later: Vec<[Wire; 2]> = levels[li + 1]
                .fold_fins
                .iter()
                .map(|&f| chw(f))
                .chain(
                    levels[li + 2..]
                        .iter()
                        .flat_map(|l| l.fold_fins.iter().skip(1).map(|&f| chw(f))),
                )
                .collect();
            assert_eq!(later.len(), folded, "OOD prefix = later folds");
            let mut factors: Vec<([Wire; 2], [Wire; 2])> = (0..folded)
                .map(|j| ([squeeze_word_wire(outs, trace, od.z_fin, j), zw], later[j]))
                .collect();
            factors.extend(coordinate_factors(sb, li + 2));
            let pw = prefix_chain(sb, [base_chw(od.beta_fin), zw], &factors);
            let coords: Vec<[Wire; 2]> = (0..yr_log)
                .map(|j| {
                    let jj = folded + j;
                    [squeeze_word_wire(outs, trace, od.z_fin, jj), zw]
                })
                .collect();
            apply_suffix(sb, &mut evb_accs, pw, &coords);
        }
    }
    // beta-weighted residuals fold in per level (comb_y += beta·resid_y —
    // one MacGate row each), then the yr dot as one MAC chain.
    let mut comb = evb_accs;
    for (li, lvl) in levels.iter().enumerate() {
        let coordinate = coordinate_factors(sb, li + 1);
        let beta_w = prefix_chain(sb, [base_chw(lvl.beta_fin), zw], &coordinate);
        for y in 0..yr_len {
            comb[y] = emit_mac256(sb, macs, comb[y], beta_w, resid_pub[li][y]);
        }
    }
    let mut inner_w = [zw, zw];
    for (yw, cb) in yr_wires.iter().zip(&comb) {
        inner_w = emit_mac256(sb, macs, inner_w, *yw, *cb);
    }
    (resid_pub, inner_w, (base_pfslot, pf_w))
}

/// Check the residual region's published wires against a NATIVE replica:
/// `induce_sumcheck_evaluate_at_residual` per level, then the close-out's gamma-weighted
/// char-2 eq products and the yr dot. Walks `public` from `at` (the first
/// accumulator, `levels × 2^yr` entries, then the inner), asserting each.
/// Returns the native inner — the residual-side t_r — so the caller asserts
/// the `inner == t_r` closure in its own indexing.
#[allow(clippy::too_many_arguments)]
pub(super) fn check_residual_publics(
    public: &[F128],
    at: usize,
    levels: &[OpenLevel],
    geo: &[Lvl],
    w_rounds: &[RoundRec],
    inner_pd_ch: usize,
    yr_vals: &[F256],
    chals: &[F128],
) -> F256 {
    let yr_len = yr_vals.len();
    assert!(yr_len.is_power_of_two());
    let yr_log = yr_len.trailing_zeros() as usize;
    let mut at = at;
    let mut resid_native: Vec<Vec<F256>> = vec![vec![F256::ZERO; yr_len]; levels.len()];
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len() - 1).sum();
        let lmc = pl + yr_log;
        let sks = flock_core::pcs::ligerito::eval_sk_at_vks(lmc);
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let ris: Vec<F256> = levels[li + 1..]
            .iter()
            .flat_map(|l| {
                l.fold_chs
                    .iter()
                    .skip(1)
                    .map(|&i| F256::new(chals[i], chals[i + 1]))
            })
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        for y in 0..yr_len {
            let mut sum = F256::ZERO;
            for k in 0..geo[li].q {
                let pos = geo[li].q_pos(k, chals[lvl.q_ch + k].lo);
                let mut sk = Vec::with_capacity(lmc);
                if lmc > 0 {
                    sk.push(F128::new(pos as u64, 0));
                    for j in 1..lmc {
                        sk.push(sk[j - 1] * sk[j - 1] + sks[j - 1] * sk[j - 1]);
                    }
                }
                let mut prod = F256::ONE;
                for j in 0..pl {
                    prod *= F256::ONE + ris[j] * (F128::ONE + sk[j] * inv(sks[j]));
                }
                for j in 0..yr_log {
                    if (y >> j) & 1 == 1 {
                        prod *= sk[pl + j] * inv(sks[pl + j]);
                    }
                }
                sum += aw[k] * prod;
            }
            assert_eq!(
                F256::new(public[at], public[at + 1]),
                sum,
                "L{li} residual y={y}"
            );
            resid_native[li][y] = sum;
            at += 2;
        }
    }
    // evb + combine, natively: gamma-weighted char-2 eq products, then the
    // yr dot.
    let ris_v: Vec<F256> = levels[0]
        .fold_chs
        .iter()
        .map(|&i| F256::new(chals[i], chals[i + 1]))
        .chain(levels[1..].iter().flat_map(|l| {
            l.fold_chs
                .iter()
                .skip(1)
                .map(|&i| F256::new(chals[i], chals[i + 1]))
        }))
        .collect();
    let pl_full = ris_v.len();
    let coordinate_scale = |start_level: usize| {
        levels[start_level.max(1)..]
            .iter()
            .fold(F256::ONE, |acc, level| {
                let at = level.fold_chs[0];
                let r = F256::new(chals[at], chals[at + 1]);
                acc * (F256::ONE + r * F256::new(F128::ONE, F128::ONE))
            })
    };
    let mut inner_n = F256::ZERO;
    for y in 0..yr_len {
        let mut evb = F256::from(chals[inner_pd_ch]);
        for j in 0..pl_full {
            evb *= F256::ONE + F256::from(chals[w_rounds[j].ch]) + ris_v[j];
        }
        for j in 0..yr_log {
            evb *= if (y >> j) & 1 == 1 {
                F256::from(chals[w_rounds[pl_full + j].ch])
            } else {
                F256::from(F128::ONE + chals[w_rounds[pl_full + j].ch])
            };
        }
        evb *= coordinate_scale(0);
        let mut comb = evb;
        for od in &levels[0].initial_ood {
            let folded = od.z_len - yr_log;
            assert_eq!(folded, pl_full, "L0 OOD spans every fold");
            let initial_k = levels[0].fold_chs.len();
            let z_index = |j| l0_ood_z_index(od.z_len, initial_k, geo[0].row_words, j);
            let mut t = F256::from(chals[od.beta_ch]);
            for j in 0..folded {
                t *= F256::ONE + F256::from(chals[od.z_ch + z_index(j)]) + ris_v[j];
            }
            for j in 0..yr_log {
                t *= if (y >> j) & 1 == 1 {
                    F256::from(chals[od.z_ch + z_index(folded + j)])
                } else {
                    F256::from(F128::ONE + chals[od.z_ch + z_index(folded + j)])
                };
            }
            t *= coordinate_scale(0);
            comb += t;
        }
        for (li, lvl) in levels.iter().enumerate() {
            comb += resid_native[li][y] * chals[lvl.beta_ch] * coordinate_scale(li + 1);
            for od in &lvl.ood {
                let folded = od.z_len - yr_log;
                let later: Vec<F256> = levels[li + 1]
                    .fold_chs
                    .iter()
                    .map(|&i| F256::new(chals[i], chals[i + 1]))
                    .chain(levels[li + 2..].iter().flat_map(|l| {
                        l.fold_chs
                            .iter()
                            .skip(1)
                            .map(|&i| F256::new(chals[i], chals[i + 1]))
                    }))
                    .collect();
                let mut t = F256::from(chals[od.beta_ch]);
                for j in 0..folded {
                    t *= F256::ONE + F256::from(chals[od.z_ch + j]) + later[j];
                }
                for j in 0..yr_log {
                    t *= if (y >> j) & 1 == 1 {
                        F256::from(chals[od.z_ch + folded + j])
                    } else {
                        F256::from(F128::ONE + chals[od.z_ch + folded + j])
                    };
                }
                t *= coordinate_scale(li + 2);
                comb += t;
            }
        }
        inner_n += yr_vals[y] * comb;
    }
    assert_eq!(
        F256::new(public[at], public[at + 1]),
        inner_n,
        "the close-out inner"
    );
    inner_n
}

/// BLAKE3 serializes its output words little-endian, while "leading bits"
/// means most-significant-bit first within each serialized byte. Return the
/// circuit-word mask whose set bits are exactly that prefix.
pub(super) fn pow_leading_zero_mask(bits: u32) -> F128 {
    assert!(bits <= 128, "the fused predicate is one F128 word");
    let mut mask = 0u128;
    for k in 0..bits as usize {
        let serialized_bit = 8 * (k / 8) + (7 - k % 8);
        mask |= 1u128 << serialized_bit;
    }
    F128::new(mask as u64, (mask >> 64) as u64)
}

/// Arithmetize every grinding operation in a recorded verifier transcript.
///
/// The fused BLAKE3 row has already bound the nonce to the transcript,
/// advanced the state and produced the protected challenge. This helper adds
/// only the selected-zero relations
///
/// ```text
/// prefix_bits(predicate_word, lambda) = 0^lambda
/// nonce[64..128] = 0.
/// ```
///
/// The selected-zero equations are rows of the shared PoW-mask table, whose
/// mask and check inputs are statement constants. A zero-bit operation instead
/// enforces the canonical nonce 0.
pub(super) fn emit_pow_checks(
    sb: &mut ShapeBuilder,
    _b3: flock_core::circuit::builder::SlotId,
    pow: flock_core::circuit::builder::SlotId,
    _iv: [Wire; 2],
    pows: &[([Wire; 2], u32)],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) {
    let check_word = (!pows.is_empty()).then(|| cw(sb, vals, consts, F128::new(0, 1u64 << 63)));
    for &([predicate, nonce], bits) in pows {
        // One fused PowMask row per check: the prefix cells mask the
        // predicate and the mask word's wire-bound high half pins the
        // nonce to 64 bits — the transcript stream allocates a whole F128
        // word to the 8-byte nonce, and this is what keeps the remaining
        // eight bytes padding rather than an extra grinding knob for a
        // malicious recursive prover.
        assert!(
            bits <= 64,
            "the PowMask row's prefix cells cover the low mask half"
        );
        if bits == 0 {
            // Canonical zero nonce: the nonce rides BOTH input words — the
            // prefix cells pin its low half under the all-ones low mask,
            // the structural high-half cells pin the rest.  All 128 bits,
            // so a disabled site cannot become a grinding knob either.
            let ones = cw(sb, vals, consts, F128::new(u64::MAX, 0));
            let _ = sb.gate(
                pow,
                &[nonce, nonce, ones, check_word.expect("nonempty PoW list")],
            );
        } else {
            let mask_w = cw(sb, vals, consts, pow_leading_zero_mask(bits));
            let _ = sb.gate(
                pow,
                &[
                    predicate,
                    nonce,
                    mask_w,
                    check_word.expect("nonempty PoW list"),
                ],
            );
        }
    }
}

/// Locate fused PoW predicate and nonce wires on an arbitrary recorded tape
/// and constrain every native fused verification call in-circuit.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_recorded_pow_checks(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    spread: flock_core::circuit::builder::SlotId,
    iv: [Wire; 2],
    ops: &[flock_transcript::transcript_record::TranscriptOp],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: &flock_transcript::transcript_record::Stream,
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) {
    let (mut fin, mut pay) = (0usize, 0usize);
    let mut pows = Vec::new();
    for op in ops {
        if let Op::Pow { bits } = op {
            pows.push((fin, pay, *bits));
        }
        if op.finalizes() {
            fin += 1;
        }
        if op.carries_payload() {
            pay += 1;
        }
    }
    let checks: Vec<([Wire; 2], u32)> = pows
        .into_iter()
        .map(|(fin, pay, bits)| {
            let sq = &trace.squeezes[fin];
            let wi = stream
                .words
                .iter()
                .position(|w| matches!(w, flock_transcript::transcript_record::StreamWord::Bytes { payload, .. } if *payload == pay))
                .expect("pow nonce stream word");
            (
                [outs[sq[0]][1], ww[wi].expect("pow nonce wired")],
                bits,
            )
        })
        .collect();
    emit_pow_checks(sb, b3, spread, iv, &checks, vals, consts);
}

#[test]
pub(super) fn fused_pow_masks_match_raw_compression() {
    let cv: [u32; 8] = std::array::from_fn(|i| 0x1020_3040u32.wrapping_mul(i as u32 + 1));
    for pending_words in 0..4 {
        let pending_len = 16 * pending_words;
        for nonce in 0..64u64 {
            for bits in [1u32, 2, 7, 8, 9, 13, 16, 17, 31, 64, 127, 128] {
                let mut block = [0u8; 64];
                for (i, b) in block[..pending_len].iter_mut().enumerate() {
                    *b = (17 * i + 9) as u8;
                }
                block[pending_len..pending_len + 8].copy_from_slice(&nonce.to_le_bytes());
                let message: [u32; 16] = std::array::from_fn(|i| {
                    u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
                });
                let out = blake3_compress(
                    &cv,
                    &message,
                    flock_transcript::challenger::pow_squeeze_counter(bits, pending_len + 16),
                    64,
                    crate::r1cs_hashes::fs_chain::CHAIN_SQUEEZE,
                );
                let mut predicate = [0u8; 16];
                for (i, word) in out[4..8].iter().enumerate() {
                    predicate[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
                }
                let predicate_word = u128::from_le_bytes(predicate);
                let mask = pow_leading_zero_mask(bits);
                let mask = (mask.lo as u128) | ((mask.hi as u128) << 64);
                let circuit_accepts = predicate_word & mask == 0;
                let native_accepts =
                    (0..bits as usize).all(|k| predicate[k / 8] & (1 << (7 - k % 8)) == 0);
                assert_eq!(
                    circuit_accepts, native_accepts,
                    "pending words {pending_words}, nonce {nonce}, lambda {bits}"
                );
            }
        }
    }

    // Pin the other half of the gadget on the fused PowMask row: a
    // nonzero-bit PoW permits only a 64-bit nonce, and a zero-bit site
    // permits only the canonical nonce zero.  The prefix checks live in the
    // R1CS; the nonce-width check is the mask input word's WIRE BINDING
    // (the word must equal the statement's mask constant, whose high half
    // is zero), so the pin models both.
    let ty = PowMaskTable;
    let r1cs = ty.build_block_r1cs(0);
    let accepted = |pred: u128, nonce: u128, mask: u128| {
        let [z, _, _] = ty.build_witness(PowMaskInput { pred, nonce, mask });
        let word2 = (256..384).fold(0u128, |acc, i| acc | ((z[i] as u128) << (i - 256)));
        r1cs.satisfies(&z) && word2 == mask
    };
    // The nonce width, under a clearing predicate.
    assert!(accepted(0, 42, 0b11));
    assert!(!accepted(0, (1u128 << 100) | 42, 0b11));
    // The prefix itself.
    assert!(!accepted(0b10, 42, 0b11));
    // The canonical zero-bit shape: the nonce as both words, all-ones low mask.
    assert!(accepted(0, 0, u64::MAX as u128));
    assert!(!accepted(1, 1, u64::MAX as u128));
    assert!(!accepted(1u128 << 100, 1u128 << 100, u64::MAX as u128));
}

#[test]
pub(super) fn recursive_pow_relation_accepts_valid_and_rejects_invalid_nonce() {
    let bits = 6u32;
    let mut state_digest = [0u8; 32];
    for (i, b) in state_digest.iter_mut().enumerate() {
        *b = (29 * i + 3) as u8;
    }
    let cv: [u32; 8] = std::array::from_fn(|i| {
        u32::from_le_bytes(state_digest[4 * i..4 * i + 4].try_into().unwrap())
    });
    let fused = |nonce: u64| {
        let mut block = [0u8; 64];
        block[..8].copy_from_slice(&nonce.to_le_bytes());
        let message: [u32; 16] = std::array::from_fn(|i| {
            u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
        });
        blake3_compress(
            &cv,
            &message,
            flock_transcript::challenger::pow_squeeze_counter(bits, 16),
            64,
            crate::r1cs_hashes::fs_chain::CHAIN_SQUEEZE,
        )
    };
    let accepts = |nonce: u64| {
        let out = fused(nonce);
        let mut predicate = [0u8; 16];
        for (i, word) in out[4..8].iter().enumerate() {
            predicate[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        predicate[0] & 0b1111_1100 == 0
    };
    let good = (0..u64::MAX)
        .find(|&n| accepts(n))
        .expect("a six-bit nonce exists");
    let bad = (good + 1..u64::MAX)
        .find(|&n| !accepts(n))
        .expect("a neighboring invalid nonce exists");

    let build = |nonce: u64, circuit_bits: u32| {
        // BLAKE3 has k_log=15; nu=7 places this focused union at the
        // smallest embedded security-config size m=22.
        let nu = 7usize;
        let mut sb = ShapeBuilder::new(nu);
        let b3 = sb.slot(Blake3Gate { nu });
        let spread = sb.slot(PowMaskGate { nu });
        let mut vals = Vec::new();
        let digest_v = [
            F128::new(
                u64::from_le_bytes(state_digest[..8].try_into().unwrap()),
                u64::from_le_bytes(state_digest[8..16].try_into().unwrap()),
            ),
            F128::new(
                u64::from_le_bytes(state_digest[16..24].try_into().unwrap()),
                u64::from_le_bytes(state_digest[24..].try_into().unwrap()),
            ),
        ];
        vals.extend_from_slice(&[digest_v[0], digest_v[1], F128::new(nonce, 0)]);
        let digest_w = [sb.input(), sb.input()];
        let nonce_w = sb.input();
        let mut consts = Vec::new();
        let zero = cw(&mut sb, &mut vals, &mut consts, F128::ZERO);
        let params = cw(
            &mut sb,
            &mut vals,
            &mut consts,
            pack_params(
                flock_transcript::challenger::pow_squeeze_counter(circuit_bits, 16),
                64,
                crate::r1cs_hashes::fs_chain::CHAIN_SQUEEZE,
            ),
        );
        let h = sb.gate(
            b3,
            &[digest_w[0], digest_w[1], nonce_w, zero, zero, zero, params],
        );
        emit_pow_checks(
            &mut sb,
            b3,
            spread,
            digest_w,
            &[([h[1], nonce_w], circuit_bits)],
            &mut vals,
            &mut consts,
        );
        let shape = sb.finish().expect("the focused PoW circuit builds");
        let built = shape.run(&vals, &[]);
        (nu, b3, spread, shape, built)
    };

    let (nu, good_b3, good_spread, good_shape, good_built) = build(good, bits);
    let (_, bad_b3, bad_spread, bad_shape, bad_built) = build(bad, bits);
    assert_eq!(good_shape.circuit.digest(), bad_shape.circuit.digest());
    let (_, _, _, downgraded_shape, downgraded_built) = build(0, 0);
    assert_ne!(
        good_shape.circuit.digest(),
        downgraded_shape.circuit.digest(),
        "changing the PoW difficulty changes digest-bound counter/mask constants"
    );
    assert!(!good_shape.circuit.check_public(&downgraded_built.public));
    assert!(
        good_built
            .rows::<PowMaskGate>(good_spread)
            .iter()
            .all(|r| r.pred & r.mask == 0 && r.nonce >> 64 == 0),
        "the valid witness satisfies every fused PoW row"
    );
    assert!(
        bad_built
            .rows::<PowMaskGate>(bad_spread)
            .iter()
            .any(|r| r.pred & r.mask != 0 || r.nonce >> 64 != 0),
        "the invalid nonce reaches a failing in-circuit prefix row"
    );

    let prove = |shape: &flock_core::circuit::builder::CircuitShape,
                 built: &flock_core::circuit::builder::CircuitWitness,
                 b3_slot,
                 spread_slot| {
        let mut union = UnionInstance::new(&shape.registry, shape.counts.clone());
        union.set_dense_floor(22);
        let pcs = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: pcs_batch_for(&union, LigeritoProfile::Fast),
            profile: LigeritoProfile::Fast,
            num_lanes: union.commit_lanes(pcs_batch_for(&union, LigeritoProfile::Fast)),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs = blake3::build_block_r1cs(nu);
        let b3_lc = b3_r1cs.csc_lincheck_circuit();
        let spread_ty = PowMaskTable;
        let spread_r1cs = spread_ty.build_block_r1cs(nu);
        let spread_lc = spread_r1cs.csc_lincheck_circuit();
        let mut slots = vec![
            (
                shape.registry_slot(b3_slot),
                UnionSlotProverInput::new(
                    blake3::generate_witness_batch_major_partial(
                        built.rows::<Blake3Gate>(b3_slot),
                        nu,
                    ),
                    b3_lc,
                ),
            ),
            (
                shape.registry_slot(spread_slot),
                UnionSlotProverInput::new(
                    spread_ty
                        .generate_witness_batch_major(built.rows::<PowMaskGate>(spread_slot), nu),
                    spread_lc,
                ),
            ),
        ];
        slots.sort_by_key(|(i, _)| *i);
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs,
            slots.into_iter().map(|(_, s)| s).collect(),
            Vec::new(),
            &mut ch,
        );
        let mut lcs = vec![
            (shape.registry_slot(b3_slot), b3_lc),
            (shape.registry_slot(spread_slot), spread_lc),
        ];
        lcs.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = lcs
            .into_iter()
            .map(|(_, lc)| lc as &dyn flock_core::lincheck::LincheckCircuit)
            .collect();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &commitment,
            &proof,
            &pcs,
            &mut ch,
        )
    };

    prove(&good_shape, &good_built, good_b3, good_spread)
        .expect("a valid grinding witness proves and verifies");
    let bad_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove(&bad_shape, &bad_built, bad_b3, bad_spread)
    }));
    assert!(
        match bad_result {
            Ok(result) => result.is_err(),
            Err(_) => true,
        },
        "an invalid grinding witness must not yield an accepted recursive proof"
    );
}

/// Emit one Merkle opening as rows of the shipped BLAKE3 table plus glue,
/// wired together. Returns the two words of the root.
///
/// This is what replaces a composite row. The dataflow, per level:
///
/// ```text
///   index word ─▶ BitSpread ─bit_l─▶ Swap ─left‖right─▶ BLAKE3 ─out_lo─▶ next Swap.prev
/// ```
///
/// and before it, the chunk chain: `leaf_blocks` BLAKE3 rows whose out_lo
/// threads row to row, seeded by the IV, `CHUNK_START` on the first and
/// `CHUNK_END` on the last. Every arrow is a copy constraint on whole words.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_opening(
    sb: &mut ShapeBuilder,
    s: CollapsedSlots,
    iv: [Wire; 2],
    leaf_w: &[Wire],
    index_w: Wire,
    depth: usize,
    cap_depth: usize,
    position_prefix: usize,
    mut consts: Option<&mut Vec<(F128, Wire)>>,
    pubs: &mut Vec<F128>,
) -> ([Wire; 2], Wire) {
    // A leaf need NOT be a whole number of 64-byte blocks: a mixed circuit
    // union commits `num_lanes` ACTIVE lanes (`dense_words.div_ceil(2^log_dim)`
    // — an arbitrary integer, since the top lanes are definitionally zero and
    // never encoded), so a row can be e.g. 61 words = 976 bytes. BLAKE3 hashes
    // that as 16 blocks whose last carries b = 16 bytes with the rest of the
    // message zero — and the compression's `b` is already a free input here,
    // so the partial block costs one zero-padding wire, not a wire-format
    // change. `blocks` counts up to a chunk's 16; larger leaves would need
    // real chunk merging, which nothing here produces.
    assert!(!leaf_w.is_empty(), "a leaf has data");
    let blocks = leaf_w.len().div_ceil(4);
    assert!(blocks <= 16, "a leaf is one BLAKE3 chunk (<= 1024 bytes)");
    let mut shared = |sb: &mut ShapeBuilder, pubs: &mut Vec<F128>, v: F128| -> Wire {
        match consts.as_deref_mut() {
            Some(c) => cw(sb, pubs, c, v),
            None => {
                pubs.push(v);
                sb.public_input()
            }
        }
    };
    let zero_w = shared(sb, pubs, F128::ZERO);
    let pad_w = if leaf_w.len().is_multiple_of(4) {
        None
    } else {
        Some(zero_w)
    };

    // The index word's bits, one per level.
    // Its zero mask is empty: this row only relocates bits.  Grinding rows
    // below reuse the same table with nonzero masks to enforce predicates.
    let position_bits = depth - cap_depth;
    let position_mask = if position_bits == 128 {
        u128::MAX
    } else {
        (1u128 << position_bits) - 1
    };
    assert_eq!(
        position_prefix & position_mask as usize,
        0,
        "the fixed stratum and sampled low bits must be disjoint"
    );
    let mask_w = shared(
        sb,
        pubs,
        F128::new(position_mask as u64, (position_mask >> 64) as u64),
    );
    let prefix_w = shared(sb, pubs, F128::new(position_prefix as u64, 0));
    let spread = sb.gate(s.spread, &[index_w, zero_w, zero_w, mask_w, prefix_w]);
    let bits = &spread[..spread.len() - 1];
    let position_w = *spread.last().expect("spread emits the derived position");

    // Chunk chain: the leaf hashed as a BLAKE3 chunk.
    let mut cv = iv;
    for i in 0..blocks {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i + 1 == blocks {
            flags |= CHUNK_END;
        }
        // The final block carries only the bytes that remain.
        let words = (leaf_w.len() - 4 * i).min(4);
        let params = shared(sb, pubs, pack_params(0, 16 * words as u32, flags));
        let mw = |j: usize| -> Wire {
            if j < words {
                leaf_w[4 * i + j]
            } else {
                pad_w.expect("a short block needs the zero pad")
            }
        };
        let out = sb.gate(s.b3, &[cv[0], cv[1], mw(0), mw(1), mw(2), mw(3), params]);
        cv = [out[0], out[1]];
    }

    // Node levels: swap, then a PARENT compression over the swapped pair.
    // The sibling is the swap's hint, supplied at `run` time in this call
    // order — setup has no values. Under capping the fold stops `cap_depth`
    // levels below the root: the returned digest is the depth-`cap_depth`
    // ancestor, which the CHECKER compares against the absorbed cap layer
    // (the boundary select — the circuit never touches the cap words).
    // `cap_depth = 0` is the uncapped statement, terminal = root.
    for l in 0..(depth - cap_depth) {
        let sw = sb.gate_hinted(s.swap, &[bits[l], cv[0], cv[1]]);
        let params = shared(sb, pubs, pack_params(0, 64, PARENT));
        let out = sb.gate(s.b3, &[iv[0], iv[1], sw[0], sw[1], sw[2], sw[3], params]);
        cv = [out[0], out[1]];
    }
    (cv, position_w)
}

/// The leaf outer's artifacts (built inside [`build_fl_node_k`] and
/// [`build_node_outer_app`]) so the
/// recursion swap can consume the proof as ITS inner: the circuit shape
/// (owning registry + counts — `UnionInstance::new(&shape.registry,
/// shape.counts.clone())` reconstructs the instance), the public segment,
/// the BLAKE3/BLAKE3 circuit proof, and the boolean tables whose lincheck
/// circuits a verifier needs (in registry order via the `*_slot` indices).
pub struct LeafOuter {
    pub(super) shape: flock_core::circuit::builder::CircuitShape,
    pub(super) public: Vec<F128>,
    pub(super) proof: MixedProof,
    pub(super) commitment: flock_core::pcs::Commitment,
    pub(super) pcs: PcsParams,
    pub(super) b3_r1cs: flock_core::r1cs::BlockR1cs,
    pub(super) swap_r1cs: flock_core::r1cs::BlockR1cs,
    pub(super) spread_r1cs: flock_core::r1cs::BlockR1cs,
    pub(super) pow_r1cs: flock_core::r1cs::BlockR1cs,
    pub(super) family_r1cs: flock_core::r1cs::BlockR1cs,
    pub(super) b3_slots: Vec<usize>,
    pub(super) swap_slot: usize,
    pub(super) spread_slot: usize,
    pub(super) pow_slot: usize,
    pub(super) family_slot: usize,
}

pub(super) fn leaf_boolean_lcs(lo: &LeafOuter) -> Vec<&dyn flock_core::lincheck::LincheckCircuit> {
    let mut ordered: Vec<(usize, &dyn flock_core::lincheck::LincheckCircuit)> = vec![
        (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
        (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
        (lo.pow_slot, lo.pow_r1cs.csc_lincheck_circuit()),
        (lo.family_slot, lo.family_r1cs.csc_lincheck_circuit()),
    ];
    ordered.extend(lo.b3_slots.iter().map(|&slot| {
        (
            slot,
            lo.b3_r1cs.csc_lincheck_circuit() as &dyn flock_core::lincheck::LincheckCircuit,
        )
    }));
    ordered.sort_by_key(|(slot, _)| *slot);
    ordered.into_iter().map(|(_, circuit)| circuit).collect()
}

pub(super) fn leaf_boolean_mats(
    lo: &LeafOuter,
) -> Vec<(
    &flock_core::r1cs::SparseBinaryMatrix,
    &flock_core::r1cs::SparseBinaryMatrix,
)> {
    let mut ordered = vec![
        (lo.swap_slot, (&lo.swap_r1cs.a_0, &lo.swap_r1cs.b_0)),
        (lo.spread_slot, (&lo.spread_r1cs.a_0, &lo.spread_r1cs.b_0)),
        (lo.pow_slot, (&lo.pow_r1cs.a_0, &lo.pow_r1cs.b_0)),
        (lo.family_slot, (&lo.family_r1cs.a_0, &lo.family_r1cs.b_0)),
    ];
    ordered.extend(
        lo.b3_slots
            .iter()
            .map(|&slot| (slot, (&lo.b3_r1cs.a_0, &lo.b3_r1cs.b_0))),
    );
    ordered.sort_by_key(|(slot, _)| *slot);
    ordered.into_iter().map(|(_, matrices)| matrices).collect()
}
