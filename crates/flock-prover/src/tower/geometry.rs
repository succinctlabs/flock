use super::*;
use flock_core::lincheck::build_eq_table;
use flock_transcript::transcript_record::StreamWord;

/// The three slots a collapsed opening writes into, plus the fused PoW mask
/// slot the grinding checks ride: one 4-word [`PowMaskTable`] row carries a
/// whole check (prefix mask AND nonce width) — on the deep Merkle-index
/// slot the same check paid two 16-word rows and two bit relocations.
#[derive(Clone, Copy)]
pub(super) struct CollapsedSlots {
    pub(super) b3: flock_core::circuit::builder::SlotId,
    /// A second identical compression table for recursion shapes whose two
    /// independent child-verifier workloads each fit a smaller row domain.
    /// The Slim envelope and strict Fast nodes use distinct slots; smaller
    /// standalone and compatibility-profile circuits retain one slot.
    pub(super) b3_alt: Option<flock_core::circuit::builder::SlotId>,
    pub(super) swap: flock_core::circuit::builder::SlotId,
    pub(super) spread: flock_core::circuit::builder::SlotId,
    pub(super) pow: flock_core::circuit::builder::SlotId,
    /// Present on recursive verifier circuits.  Smaller query-only fixtures
    /// do not declare the family-H transpose relation.
    pub(super) family: Option<flock_core::circuit::builder::SlotId>,
}

/// One opened Ligerito level's geometry. Legacy levels report it from the
/// proof itself; stratified levels carry the STATEMENT's schedule and the
/// proof is validated against it (`docs/stratified-queries.tex`: the
/// allocation is config authority, never proof-derived).
pub(super) struct Lvl {
    pub(super) q: usize,
    pub(super) c: usize,
    pub(super) depth: usize,
    /// The FOLD width `2^folds` — the lane-weight domain.
    pub(super) lanes: usize,
    /// The COMMITTED width: `num_lanes` active lanes, which for a mixed
    /// union is an arbitrary integer `<= lanes` (the top lanes are
    /// definitionally zero and never encoded). Equal to `lanes` whenever
    /// the lane count happens to be a power of two.
    pub(super) row_words: usize,
    /// Number of committed F128 words. Recursive codewords are also base
    /// field rows; their extra coordinate bit is included in `folds`.
    pub(super) raw_row_words: usize,
    /// The stratified schedule this level's config mandates. Every
    /// consumer (emit, residual, checker) maps query → (stratum depth,
    /// stratum, path slice) through this.
    pub(super) sched: flock_core::pcs::stratified::LevelSchedule,
    /// The tree's layers from the cap upward, folded natively by
    /// `level_geometry`: entry `i` is the depth-`(c − i)` layer, entry 0
    /// the cap itself — [`Self::full_path`]'s sibling sources.
    cap_layers: Vec<Vec<[u8; 32]>>,
}

/// Map actual sumcheck-fold order back to the transcript point's natural
/// coordinate order. Partial L0 lane grids bind the high lane coordinates
/// first; full grids and every later level use ordinary low-to-high order.
pub(super) fn l0_ood_z_index(
    z_len: usize,
    initial_k: usize,
    committed_row_words: usize,
    fold_order: usize,
) -> usize {
    if committed_row_words == 1usize << initial_k {
        fold_order
    } else {
        let log_msg_cols = z_len - initial_k;
        if fold_order < initial_k {
            log_msg_cols + fold_order
        } else {
            fold_order - initial_k
        }
    }
}

impl Lvl {
    /// Query `k`'s (terminal depth, stratum index).
    pub(super) fn q_stratum(&self, k: usize) -> (usize, usize) {
        self.sched
            .query_strata()
            .nth(k)
            .expect("query index within schedule")
    }

    /// Query `k`'s PROOF siblings as a range into the level's flat path
    /// vec — uniformly `d − c` per query since paths truncate at the cap.
    /// The climb to a shallower summand's stratum terminal needs `c − c_k`
    /// more siblings, all folds of the cap: [`Self::full_path`] synthesizes
    /// them.
    fn path_range(&self, k: usize) -> std::ops::Range<usize> {
        let len = self.depth - self.c;
        k * len..(k + 1) * len
    }

    /// Query `k`'s FULL climb siblings — the truncated proof slice extended
    /// up to its stratum terminal at depth `c_k` with the cap-fold siblings
    /// the proof no longer carries (`self.cap_layers`, folded natively by
    /// `level_geometry`). Advice either way: the synthesized entries feed
    /// the same hint stream, and the constant-stratum terminal connect is
    /// what binds the climb.
    pub(super) fn full_path(&self, k: usize, pos: usize, paths: &[[u8; 32]]) -> Vec<[u8; 32]> {
        let (ck, _) = self.q_stratum(k);
        let mut sibs: Vec<[u8; 32]> = paths[self.path_range(k)].to_vec();
        // Path entry j is the sibling at depth d − j; the proof stops at
        // the cap (j < d − c), the tail (depths c down to c_k + 1) folds
        // out of the cap. Its indices carry SQUEEZED bits below the
        // stratum, so this is witgen data, never circuit wiring.
        for j in (self.depth - self.c)..(self.depth - ck) {
            let m = self.depth - j;
            sibs.push(self.cap_layers[self.c - m][(pos >> j) ^ 1]);
        }
        sibs
    }

    /// The position query `k` opens, from its squeezed word's low half:
    /// the low `depth − c_k` bits are sampled, the top bits ARE the
    /// stratum.
    pub(super) fn q_pos(&self, k: usize, lo: u64) -> usize {
        let (c, stratum) = self.q_stratum(k);
        let lo_bits = self.depth - c;
        (stratum << lo_bits) | ((lo as usize) & ((1usize << lo_bits) - 1))
    }
}

/// Exact BLAKE compression rows emitted by [`emit_query_phase`]. Each level
/// materializes only the cap layers down to its shallowest configured
/// stratum. A query hashes its committed row, climbs to that stratum, and
/// top-stratum queries take one additional edge so the opening binds to a
/// derived cap-layer node without creating a transcript cycle.
pub(super) fn level_query_phase_b3_rows(g: &Lvl) -> (usize, usize, usize) {
    let c_min = g.sched.summand_depths.last().copied().unwrap_or(g.c);
    let n_layers = (g.c - c_min).max(1);
    let cap_rows = (1..=n_layers).map(|j| 1usize << (g.c - j)).sum();
    let leaf_rows = g.raw_row_words.div_ceil(4) * g.q;
    let path_rows = (0..g.q)
        .map(|k| {
            let (ck, _) = g.q_stratum(k);
            (g.depth - ck) + usize::from(ck == g.c)
        })
        .sum();
    (leaf_rows, path_rows, cap_rows)
}

pub(super) fn query_phase_b3_rows(geo: &[Lvl]) -> usize {
    geo.iter()
        .map(|g| {
            let (leaf, path, cap) = level_query_phase_b3_rows(g);
            leaf + path + cap
        })
        .sum()
}

/// Place `extra` identical-relation rows on two existing slots as evenly as
/// their current loads permit. Returns `(extra_on_a, resulting_max_load)`.
pub(super) fn balance_extra_rows(a: usize, b: usize, extra: usize) -> (usize, usize) {
    let target = (a + b + extra).div_ceil(2).max(a).max(b);
    let on_a = target.saturating_sub(a).min(extra);
    (on_a, (a + on_a).max(b + extra - on_a))
}

/// The tree's layers from the cap upward, natively: entry `i` is the
/// depth-`(c − i)` layer, entry 0 the cap itself — the sibling sources for
/// [`Lvl::full_path`]'s synthesized tail. `n_layers` is clamped to at
/// least 1 so entry 0 exists even for a single-summand schedule (which
/// never indexes past it).
pub(super) fn native_cap_layers(
    cap: &[[u8; 32]],
    n_layers: usize,
    hash: HashKind,
) -> Vec<Vec<[u8; 32]>> {
    let mut layers: Vec<Vec<[u8; 32]>> = vec![cap.to_vec()];
    for _ in 1..n_layers.max(1) {
        let next: Vec<[u8; 32]> = layers
            .last()
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| core_merkle::hash_pair(&p[0], &p[1], hash))
            .collect();
        layers.push(next);
    }
    layers
}

/// The stratified schedules the inner proof's own config mandates — the
/// STATEMENT side of the query-phase geometry (None while the inner's
/// (m, profile) TOML is legacy). Derived from the same registry entry the
/// inner was proven under; never from the proof.
pub(super) fn strat_scheds(params: &PcsParams) -> Vec<flock_core::pcs::stratified::LevelSchedule> {
    params
        .ligerito_verifier_config()
        .expect("registered config")
        .stratified
}

/// The per-level `(cap, opened rows, flat sibling paths)` triples a Ligerito
/// proof reports, in level order: L0's initial cap, then each recursive cap,
/// with the FINAL level reusing the last recursive cap. Since Merkle
/// capping, this is the whole witness a query phase needs — the proof itself
/// carries it, so no prover data is ever plumbed through.
pub(super) fn level_sources(
    lig: &flock_core::pcs::ligerito::LigeritoProof,
) -> Vec<(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)> {
    let r = lig.recursive_caps.len();
    (0..=r)
        .map(|li| {
            if li == 0 {
                (
                    lig.initial_cap.as_slice(),
                    &lig.initial_proof.opened_rows,
                    &lig.initial_proof.merkle_proof,
                )
            } else if li < r {
                (
                    lig.recursive_caps[li - 1].as_slice(),
                    &lig.recursive_proofs[li - 1].opened_rows,
                    &lig.recursive_proofs[li - 1].merkle_proof,
                )
            } else {
                (
                    lig.recursive_caps[r - 1].as_slice(),
                    &lig.final_proof.opened_rows,
                    &lig.final_proof.merkle_proof,
                )
            }
        })
        .collect()
}

/// Per level: `q`, cap bits `c`, path length `d − c`, depth, lanes — plus
/// the NATIVE cross-checks that pin every piece of the plumbing before any
/// circuit exists: each opened row verifies against its cap under the
/// recorded challenge, and the recorded weights reproduce
/// `induce_sumcheck_enforced_sum`. Returns `(geo, native_sums)`; the sums
/// are what the in-circuit leaf-eval accumulators must equal.
pub(super) fn level_geometry(
    levels: &[OpenLevel],
    lvl_src: &[(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)],
    chals: &[F128],
    hash: HashKind,
    scheds: &[flock_core::pcs::stratified::LevelSchedule],
) -> (Vec<Lvl>, Vec<F256>) {
    assert_eq!(scheds.len(), levels.len(), "one schedule per open level");
    let mut geo: Vec<Lvl> = Vec::new();
    let mut native_sums: Vec<F256> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let (cap, rows, paths) = lvl_src[li];
        let q = lvl.q_count;
        assert_eq!(rows.len(), q, "L{li}: one opened row per query");
        let sched = &scheds[li];
        // The proof is VALIDATED against the statement's schedule — never
        // the other way around.
        assert_eq!(sched.queries(), q, "L{li}: schedule owes the query count");
        let c = sched.cap_depth();
        assert_eq!(cap.len(), 1 << c, "L{li}: cap is the schedule's top layer");
        assert_eq!(
            paths.len(),
            sched.total_path_siblings(),
            "L{li}: flat paths sum the per-summand walks"
        );
        let depth = sched.log_block_len;
        // The lane-fold weights are `2^folds` wide; the committed row may be
        // NARROWER (its top lanes are definitionally zero), and the dot below
        // zips — which IS the zero-fill, exactly as the native verifier does.
        let lanes = 1usize << lvl.fold_fins.len();
        let raw_row_words = rows[0].len();
        let row_words = raw_row_words;
        assert!(
            row_words >= 1 && row_words <= lanes,
            "L{li}: opened width {row_words} must fit the fold width {lanes}"
        );
        let fold_vals: Vec<F256> = lvl
            .fold_chs
            .iter()
            .map(|&i| F256::new(chals[i], chals[i + 1]))
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let eqv = flock_multilinear::eq_table(
            &fold_vals,
            F256::ONE,
            flock_multilinear::IndexOrder::LowToHigh,
        );
        let aw = build_eq_table(&alpha_vals);
        let c_min = sched.summand_depths.last().copied().unwrap_or(c);
        let lv = Lvl {
            q,
            c,
            depth,
            lanes,
            row_words,
            raw_row_words,
            sched: sched.clone(),
            cap_layers: native_cap_layers(cap, c - c_min, hash),
        };
        // Paths truncate at the cap, so every query verifies directly
        // against the absorbed layer — no terminal-layer rebuild; the
        // stratum needs no enforcement because `q_pos` derives the index
        // itself with the stratum in the top bits.
        let mut sum = F256::ZERO;
        for (k, row) in rows.iter().enumerate() {
            let pos = lv.q_pos(k, chals[lvl.q_ch + k].lo);
            let mut leaf_bytes = Vec::with_capacity(16 * lanes);
            for f in row {
                leaf_bytes.extend_from_slice(&f.lo.to_le_bytes());
                leaf_bytes.extend_from_slice(&f.hi.to_le_bytes());
            }
            let lh = core_merkle::hash_leaf(&leaf_bytes, hash);
            assert!(
                core_merkle::verify_merkle_proof_capped(
                    cap,
                    1 << depth,
                    &lh,
                    pos,
                    &paths[lv.path_range(k)],
                    hash,
                ),
                "L{li} query {k}: capped path verifies natively"
            );
            let dot = row
                .iter()
                .zip(eqv.iter())
                .map(|(&x, &e)| F256::from(x) * e)
                .fold(F256::ZERO, |a, v| a + v);
            sum += aw[k] * dot;
        }
        native_sums.push(sum);
        geo.push(lv);
    }
    (geo, native_sums)
}

pub(super) fn replay_ligerito_spine256(
    levels: &[OpenLevel],
    values: &[F128],
    challenges: &[F128],
    start_v: usize,
    initial_target: F128,
    enforced_sums: &[F256],
) -> F256 {
    let msg = |at: usize| {
        (
            F256::new(values[at], values[at + 1]),
            F256::new(values[at + 2], values[at + 3]),
        )
    };
    let quad = |at: usize, target: F256| {
        let (u0, u2) = msg(at);
        (u0, target + u2, u2)
    };
    let eval = |q: (F256, F256, F256), r: F256| q.0 + r * q.1 + r * r * q.2;

    let mut target = F256::from(initial_target);
    for od in &levels[0].initial_ood {
        target += F256::from(challenges[od.beta_ch] * values[od.y_v]);
    }
    let mut q = quad(start_v, target);
    for (li, level) in levels.iter().enumerate() {
        for (j, &mv) in level.fold_msg_vs.iter().enumerate() {
            let ch = level.fold_chs[j];
            target = eval(q, F256::new(challenges[ch], challenges[ch + 1]));
            q = quad(mv, target);
        }
        if li + 1 < levels.len() {
            for od in &level.ood {
                let y = F256::from(values[od.y_v]);
                let iq = quad(od.intro_v, y);
                let beta = challenges[od.beta_ch];
                q.0 += iq.0 * beta;
                q.1 += iq.1 * beta;
                q.2 += iq.2 * beta;
                target += y * beta;
            }
            let iq = quad(level.intro_v, enforced_sums[li]);
            let beta = challenges[level.beta_ch];
            q.0 += iq.0 * beta;
            q.1 += iq.1 * beta;
            q.2 += iq.2 * beta;
            target += enforced_sums[li] * beta;
        } else {
            target += enforced_sums[li] * challenges[level.beta_ch];
        }
    }
    target
}

pub(super) fn observed_f256(values: &[F128], start: usize, len: usize) -> Vec<F256> {
    (0..len)
        .map(|i| F256::new(values[start + 2 * i], values[start + 2 * i + 1]))
        .collect()
}

/// Stream-word indices per `observe_bytes` payload, in payload-word order.
pub(super) fn payload_words(
    stream: &flock_transcript::transcript_record::Stream,
) -> Vec<Vec<usize>> {
    let mut pay_words: Vec<Vec<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let StreamWord::Bytes { payload, word } = *w {
            if pay_words.len() <= payload {
                pay_words.resize(payload + 1, Vec::new());
            }
            assert_eq!(pay_words[payload].len(), word, "payload words in order");
            pay_words[payload].push(wi);
        }
    }
    pay_words
}

/// Locate each level's absorbed cap payload in the stream: one payload
/// index per level, in level order.
///
/// Payloads are CONTENT-matched — the flattened cap bytes must equal a
/// whole `observe_bytes` payload — searching FORWARD (levels absorb their
/// caps in transcript order: the statement's L0 cap first, then each
/// recursion round's), so a size collision with another absorbed surface
/// (the sigma V cap, a child's publics payload) cannot mislocate: a
/// different tree's 32-byte digests never reproduce this cap's bytes.
///
/// Entry 0 is the L0 cap — the COMMITMENT, a statement surface that stays
/// public. Entries 1.. are the recursive caps — PROOF BODY: since the
/// in-circuit cap trees bind them (chain + root connects, nothing
/// checker-read), their payloads demote to witness in `pub_payloads`.
pub(super) fn cap_payloads(
    stream: &flock_transcript::transcript_record::Stream,
    bytes: &[u8],
    lvl_src: &[(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)],
) -> Vec<usize> {
    let pay_words = payload_words(stream);
    let mut out = Vec::with_capacity(lvl_src.len());
    let mut from = 0usize;
    for (li, (cap, _, _)) in lvl_src.iter().enumerate() {
        let flat: Vec<u8> = cap.iter().flatten().copied().collect();
        let words = flat.len() / 16;
        let p = (from..pay_words.len())
            .find(|&p| {
                pay_words[p].len() == words
                    && pay_words[p]
                        .iter()
                        .enumerate()
                        .all(|(j, &wi)| bytes[wi * 16..wi * 16 + 16] == flat[j * 16..j * 16 + 16])
            })
            .unwrap_or_else(|| panic!("L{li}: absorbed cap payload located"));
        from = p + 1;
        out.push(p);
    }
    out
}

/// The absorbed caps' node wires: per level, `2^c` word-wire pairs in
/// cap-layer order, read off the [`cap_payloads`]-located payloads.
pub(super) fn cap_wires(
    stream: &flock_transcript::transcript_record::Stream,
    word_wire: &[Option<Wire>],
    cap_pays: &[usize],
) -> Vec<Vec<[Wire; 2]>> {
    let pay_words = payload_words(stream);
    cap_pays
        .iter()
        .map(|&p| {
            pay_words[p]
                .chunks(2)
                .map(|c| {
                    [
                        word_wire[c[0]].expect("cap word wired"),
                        word_wire[c[1]].expect("cap word wired"),
                    ]
                })
                .collect()
        })
        .collect()
}

/// ROUND 2 — the H(publics) region: re-derive the child's publics
/// commitment ([`flock_core::union::publics_digest`]) from WITNESS wires
/// and CONNECT it to the absorbed digest payload words.
///
/// Under the v2 statement binding the child's transcript absorbs 32 bytes,
/// not the segment, so the child's public words enter the PARENT as
/// witness; this region is what makes the digest binding structural: 1 KiB
/// chunk chains per leaf (the emit_opening chunk shape, pinned == the
/// native `hash_leaf`), LEFT-FOLDED through PARENT rows (== `hash_pair`) —
/// exactly the `publics_digest` chain, ending in an output-output connect
/// with no gate consumers (no cycles, no checker item). Returns the
/// public-word wires — the future consumers' handle (the wiring
/// recombination's publics-MLE evaluation is the recorded upgrade).
pub(super) fn emit_publics_hash(
    sb: &mut ShapeBuilder,
    s: CollapsedSlots,
    iv: [Wire; 2],
    child_public: &[F128],
    digest_w: [Wire; 2],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) -> Vec<Wire> {
    assert!(!child_public.is_empty(), "a circuit child has publics");
    let pw: Vec<Wire> = child_public
        .iter()
        .map(|v| {
            vals.push(*v);
            sb.input()
        })
        .collect();
    let pad_w = cw(sb, vals, consts, F128::ZERO);
    let mut cv: Option<[Wire; 2]> = None;
    for leaf in pw.chunks(64) {
        let blocks = leaf.len().div_ceil(4);
        let mut lcv = iv;
        for i in 0..blocks {
            let mut flags = 0u32;
            if i == 0 {
                flags |= CHUNK_START;
            }
            if i + 1 == blocks {
                flags |= CHUNK_END;
            }
            let words = (leaf.len() - 4 * i).min(4);
            let params = cw(sb, vals, consts, pack_params(0, 16 * words as u32, flags));
            let mw = |j: usize| if j < words { leaf[4 * i + j] } else { pad_w };
            let out = sb.gate(s.b3, &[lcv[0], lcv[1], mw(0), mw(1), mw(2), mw(3), params]);
            lcv = [out[0], out[1]];
        }
        cv = Some(match cv {
            None => lcv,
            Some(prev) => {
                let params = cw(sb, vals, consts, pack_params(0, 64, PARENT));
                let out = sb.gate(
                    s.b3,
                    &[iv[0], iv[1], prev[0], prev[1], lcv[0], lcv[1], params],
                );
                [out[0], out[1]]
            }
        });
    }
    let root = cv.expect("at least one leaf");
    sb.connect(root[0], digest_w[0]);
    sb.connect(root[1], digest_w[1]);
    pw
}
