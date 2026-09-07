use std::{
    any::Any,
    env::var,
    iter::{once, repeat_n},
    time::Instant,
};

use aggregate::{
    JaggedKeyProve, JaggedKeyVerify, SigmaKey, prove_aggregate_classes_with_grinding,
    verify_aggregate_classes_with_grinding,
};
use bincode::serialize;
use flock_core::{
    aggregate,
    aggregate::{Accumulator, TypeMatrices},
    circuit::{Circuit, WiringError},
    element_r1cs::union::ElementAssertion,
    lincheck::{LincheckCircuit, SkipPoint},
    matrix_fold::{FoldProof, JaggedAssertion, JaggedClaim},
    pcs::{LOG_PACKING, jagged::JaggedParams},
    product_gkr::ProductGkrError,
    schedule::TableClass,
    verifier::FlockVerifyError,
    zerocheck::{K_SKIP, multilinear::subspace_denominator_pair},
};
use flock_field::PHI_8_TABLE;
use flock_transcript::transcript_record::{RecordingChallenger, StreamWord, TranscriptOp as Op};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
#[cfg(test)]
use {
    crate::tower::{
        build_chain_proof, build_fl_node, gates_blake3::Rng, native_chain, pack4, test_config,
    },
    std::array::from_fn,
};

#[cfg(target_arch = "aarch64")]
use crate::prover::prove_fast_ligerito_union_circuit_ag;
use crate::{
    prover::{UnionElementSlotInput, prove_fast_ligerito_union_circuit},
    r1cs_hashes::{
        blake3::{build_block_r1cs, generate_witness_batch_major_partial_into},
        fs_chain::{IV, trace_duplex},
    },
    schedule::Registry,
    tower::{
        BitSpreadGate, BitSpreadTable, Blake3Gate, ChildSlots, DOMAIN, ENV_ACC_MAIN_WORDS, EnvTail,
        F128, FamilyTransposeTileGate, FamilyTransposeTileTable, FoldPub, FsChallenger, HashKind,
        LeafOuter, MatrixClaim, MergedChain, MixedProof, Online, PcsParams, PowMaskGate,
        PowMaskTable, RealRegion, RealTape, SLOT_WORDS, ShapeBuilder, SwapGate, SwapTable,
        TowerConfig, UnionInstance, UnionSlotProverInput, Weight, Wire, ZskipTapeRec, ZskipWires,
        assert_chain_replays, balance_extra_rows, bytes_payload_mask, challenge_word_locs,
        check_ag_skip_publics, check_fold_publics, check_jagged_fold_publics,
        check_real_child_region, emit_ag_point_binding, emit_fold_region, emit_fs_chain,
        emit_fs_chain_partitioned, emit_jagged_fold_region, emit_lagrange_lows,
        emit_real_child_region, emit_recorded_pow_checks, env_acc_chain_base, env_acc_main_base,
        env_app_base, env_pass_base, envelope_shape, flatten_ops, fold_region_ops,
        jagged_fold_region_ops, labeled_bytes_payloads, leaf_boolean_lcs, leaf_boolean_mats,
        live_element_input_from_rows, locate_and_pin_folds, locate_and_pin_jagged_folds,
        merge_chain, outer_lanes, outer_union, outer_zc_ag, pack8, pad_envelope_counts,
        payload_words, pcs_batch_for, read_acc_entry, replay_fold_endpoints,
        replay_jagged_fold_endpoints, steady_reps, tower_fold_grinding,
    },
};

/// The PRODUCTION per-proof tape cost of one child: the recorded deferred
/// verify alone — the tape (op sequence + values + challenges) and the
/// assertion references in one pass. Everything else `RealTape::new` does
/// (pins, locates, native replicas) is SHAPE-STABLE index work a real node
/// precomputes at setup; the value/hint fill from a fresh tape is index
/// copies on top of this. Union + lcs construction is included
/// (conservative — a node would cache both).
pub(super) fn record_child_verify(lo: &LeafOuter, domain: &'static [u8]) {
    let union_i = outer_union(&lo.shape.registry, lo.shape.counts.clone());
    let lcs = leaf_boolean_lcs(lo);
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
    lo.proof
        .verify_circuit_deferred(
            &union_i,
            &lo.shape.circuit,
            &lo.public,
            &lcs,
            &lo.commitment,
            &lo.pcs,
            &mut rec,
        )
        .expect("the child verifies (recorded)");
}

/// A node's PUBLISHED ACC_MAIN block, entry for entry — the surface a
/// spine parent inherits, which is not quite the accumulator the fold
/// returns: the keyed groups have a fixed number of SLOTS (one per child
/// role), and a slot this node had no fold for is present as a DEAD entry
/// (zero key, zero claim). That fixed shape is the whole point — a base
/// node and a steady node publish the same layout, so ONE parent circuit
/// reads either.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainBlock {
    pub(super) per_type: Vec<(MatrixClaim, MatrixClaim)>,
    pub(super) per_element: Vec<(MatrixClaim, MatrixClaim)>,
    /// Slot order: 0 is the FL-child slot, 1 the NODE-child slot.
    pub(super) sigma: Vec<([F128; 2], MatrixClaim)>,
    pub(super) jagged: Vec<([F128; 2], MatrixClaim)>,
    /// The PASSENGER, same two slots: (sigma-shaped, jagged-shaped).
    pub(super) passenger: Vec<([F128; 2], MatrixClaim)>,
}

/// The two keyed slots every node publishes: the fresh FL child's, then
/// the node child's.
pub(super) const N_KEY_SLOTS: usize = 2;

/// A circuit digest as the two field words a transcript absorbs it as —
/// the form the published keys and the match-gate compare in.
pub(super) fn digest_f128(d: &[u8; 32]) -> [F128; 2] {
    let w = |o: usize| {
        F128::new(
            u64::from_le_bytes(d[o..o + 8].try_into().unwrap()),
            u64::from_le_bytes(d[o + 8..o + 16].try_into().unwrap()),
        )
    };
    [w(0), w(16)]
}

/// A claim scaled by a BIT: `1` returns it unchanged, `0` returns the zero
/// claim at the same POINTS — weights identically zero, value zero. The
/// points stay because in-circuit they are the child's published words and
/// only the lows and the value pass through the gate.
pub(super) fn gate_claim(c: &MatrixClaim, live: bool) -> MatrixClaim {
    if live {
        return c.clone();
    }
    MatrixClaim {
        row: Weight::low_eq(vec![F128::ZERO], c.row.point.clone()),
        col: Weight::low_eq(vec![F128::ZERO], c.col.point.clone()),
        value: F128::ZERO,
    }
}

/// `true` when an entry's LIVE word is nonzero — a claim that is about
/// something.
pub(super) fn entry_live(c: &MatrixClaim) -> bool {
    c.row.low[0] != F128::ZERO
}

/// THE SPINE (wall 3): the node child's published block riding in as this
/// node's MAIN-fold prior. `node_child` is that child's index in `los`
/// (the steady shape: 1 — child 0 is the fresh FL).
pub struct SpineIn<'a> {
    pub(super) node_child: usize,
    pub(super) prior: &'a MainBlock,
    /// THE ADVERSARIAL LEG: re-witness the match-gate's mac rows as a
    /// cheating prover's world — the mismatched slot CLAIMS its digests
    /// match and folds the orphan live — with every row self-satisfying,
    /// then assert the proof dies on exactly the wiring product. The
    /// builder's honest asserts all still run (the publics are untouched).
    pub(super) forge: bool,
}

/// Everything [`build_node_outer_app`] hands back.
// Read only by the in-file `#[test]` benches; the lib unit sees the fields
// write-only.
#[cfg_attr(not(test), allow(dead_code))]
pub struct NodeOut {
    pub(super) lo: LeafOuter,
    /// The MAIN fold's accumulator — LIVE entries only, the thing a root
    /// discharges.
    pub(super) acc: Accumulator,
    /// The LAST online iteration (steady state under repetition).
    pub(super) online: Online,
    /// One record per online iteration (1 + steady_reps of them) — the
    /// bench's medians come from here, one setup for all of them.
    pub(super) onlines: Vec<Online>,
    pub(super) app_base: Option<usize>,
    pub(super) lane_acc: Option<Accumulator>,
    /// The published ACC_MAIN + passenger blocks — what a spine parent
    /// inherits.
    pub(super) block: MainBlock,
}

/// A LOWER-registry accumulator lane riding through an internal node
/// (task 6): the two children each carry an accumulator over a registry
/// that is NOT the fold's own (the chain registry at the first level), so
/// it cannot join the node's fold as a prior — it folds in its OWN
/// priors-only aggregate, whose prior surfaces connect WIRE-TO-WIRE to
/// the children's published accumulator claims (`claims_base` locates
/// them; a prior's surface IS what the child published).
pub struct ChainLane<'a> {
    pub(super) registry: &'a Registry,
    pub(super) mats: &'a [TypeMatrices<'a>],
    pub(super) circs: &'a [&'a dyn LincheckCircuit],
    /// The lane's sigma table owner (the chain circuit).
    pub(super) circuit: &'a Circuit,
    /// The lane's jagged table owner (the chain LAYOUT — the count win's
    /// per-digest key, inherited priors-only through internal nodes).
    pub(super) params: &'a JaggedParams,
    pub(super) priors: &'a [&'a Accumulator],
    /// The published `[rho_col | rho_row | value]` fold blocks' base in
    /// EACH child's public segment (every child shares the layout).
    pub(super) claims_base: usize,
}

/// **THE 2→1 RECURSION NODE.** Two DISTINCT real recursion nodes (seeded
/// leaf outers — one circuit, unrelated FS points) go in; ONE proof comes
/// out, carrying everything a parent needs:
///
/// - TWO REAL CHILD-TAPE REGIONS — each child's complete deferred verifier
///   (the swap assembly via [`emit_real_child_region`]) over SHARED slots,
/// - the FOLD REGION at the real registry (~35 folds via the width-driven
///   helpers), and
/// - THE CONNECTS: every fold claim's surfaces are copy-constrained to the
///   child regions' own assertion-emission wires — points to chain
///   squeezes, z_partial lows word-for-word, and (richer than the minimal
///   children) the matrix/element EVAL VALUES to the children's bound
///   advice publics. The lagrange row lows stay the boundary pattern:
///   published once per child, rebuilt by the checker from that child's
///   PUBLISHED z_skip.
///
/// The accumulator reassembles from the public segment alone, equals the
/// native verifier's, and discharges all three groups against the node
/// circuit's own matrices and sigma table. This outer IS the merge node:
/// its proof attests both children's verification AND the fold that
/// combined their claims. (It is not yet SELF-similar — normalization is
/// deliberately out of scope.)
/// Build a 2→1 RECURSION NODE over two children and return its artifacts
/// AS A [`LeafOuter`] (plus its output accumulator): the node's proof is
/// BLAKE3/BLAKE3-recursable and shaped exactly like a child input, so the
/// builder composes with ITSELF — `build_node_outer_app(&[&n0, &n1], ..)` is the
/// level-2 node consuming its own outputs. The children must share one
/// circuit digest (the foldability key); their claims land at unrelated FS
/// points. Every tape pin, connect, and checker walk of the 2→1 milestone
/// lives inside — the builder IS the test.
///
/// APPLICATION-STATEMENT plumbing: when the children carry an app block
/// (`app_stmt` = its offset in their public segments — the hash-chain span
/// (h_start, h_end), 8 words), the node connects left.h_end ==
/// right.h_start wire-to-wire and publishes the combined span as its OWN
/// app block, returning that block's offset — so the output feeds the next
/// level with the same plumbing.
pub fn build_node_outer_app(
    cfg: TowerConfig,
    los: &[&LeafOuter],
    app_stmt: Option<usize>,
    lane: Option<ChainLane<'_>>,
    spine: Option<SpineIn<'_>>,
) -> NodeOut {
    const M11_NODE_DOMAIN: &[u8] = b"flock-mvp11-two-to-one-v0";

    // ARITY IS A KNOB: the node folds `k = los.len()` children in one
    // proof. Commit and open are FLOOR-bound — they cost the same whatever
    // k is, as long as the content stays under 2^(m*-7) — so every child
    // past the first rides that toll for free, and a k-ary layer needs
    // 1/(k-1) as many nodes. What does scale with k is the per-child
    // region: mac is ~97% per-child work, which is why nu* is 16.
    let n_kids = los.len();
    assert!(n_kids >= 2, "a node folds at least two children");
    let forge_match = spine.as_ref().is_some_and(|sp| sp.forge);
    let lo0 = los[0];
    // A steady node has one fresh FL child and one prior node child.
    // These children use different circuits. They share the registry, public
    // length, and lane count. One child region
    // can therefore walk either child. Only the fold keys differ.
    // Without a spine, each child uses one circuit digest.
    if spine.is_none() {
        for lo in los {
            assert_eq!(
                lo.shape.circuit.digest(),
                lo0.shape.circuit.digest(),
                "a fresh-only node folds every child under ONE key"
            );
        }
    } else {
        assert_eq!(n_kids, 2, "the spine's steady node is 2->1");
    }
    for lo in los {
        assert_eq!(
            lo.shape.registry.digest(),
            lo0.shape.registry.digest(),
            "every child, ONE envelope registry"
        );
    }
    let registry = &lo0.shape.registry;
    let unions: Vec<UnionInstance> = los
        .iter()
        .map(|lo| outer_union(&lo.shape.registry, lo.shape.counts.clone()))
        .collect();
    let t_tapes = Instant::now();
    // The children's tapes are independent statement work — build them
    // concurrently (each is a recording verify + the region pins).
    let rts: Vec<RealTape> = { los.par_iter().map(|lo| RealTape::new(lo, DOMAIN)).collect() };
    let tape_setup_ms = t_tapes.elapsed().as_secs_f64() * 1e3;
    for i in 1..n_kids {
        assert_ne!(
            rts[0].sigma_native.rho, rts[i].sigma_native.rho,
            "distinct witnesses, distinct FS points"
        );
    }

    // The matrices + lincheck circuits, registry order (lo0's copies —
    // one circuit, one registry).
    let lcs = leaf_boolean_lcs(lo0);
    let mats = leaf_boolean_mats(lo0);
    let el_types: Vec<_> = registry
        .element_types()
        .iter()
        .map(|s| s.element_type().expect("an element slot's table"))
        .collect();
    let el_mats: Vec<_> = el_types.iter().map(|t| (t.a_0(), t.b_0())).collect();
    let n_bool = registry.num_boolean();
    let n_el = el_mats.len();

    // The native merge fold over every child's assertions.
    let bool_asserts: Vec<_> = rts.iter().map(|rt| rt.mat_assert.clone()).collect();
    let el_asserts: Vec<_> = rts
        .iter()
        .zip(&unions)
        .map(|(rt, u)| (u, rt.el_assert.clone()))
        .collect();
    let sigmas: Vec<_> = rts.iter().map(|rt| rt.sigma_native.clone()).collect();
    // THE KEYED GROUPS, per child SHAPE (wall 3): a fresh-only node has ONE
    // key (every child is the same circuit); the SPINE has one SLOT PER
    // CHILD, because its children are different circuits and claims about
    // different permutations — different layouts — cannot fold together.
    // The layout is a shape constant of the child circuit, so the key that
    // names the circuit names the table.
    let key_circuits: Vec<&Circuit> = match &spine {
        None => vec![&lo0.shape.circuit],
        Some(_) => los.iter().map(|lo| &lo.shape.circuit).collect(),
    };
    let n_keys = key_circuits.len();
    let key_digests: Vec<[u8; 32]> = key_circuits.iter().map(|c| c.digest()).collect();
    // Which children's FRESH claims ride each key: all of them under one
    // key, or child j under slot j.
    let key_kids: Vec<Vec<usize>> = match &spine {
        None => vec![(0..n_kids).collect()],
        Some(_) => (0..n_kids).map(|i| vec![i]).collect(),
    };
    let params_j: Vec<JaggedParams> = (0..n_keys)
        .map(|j| {
            let i = key_kids[j][0];
            JaggedParams::from_heights(
                &unions[i].jagged_heights(),
                unions[i].n_log(),
                los[i].commitment.params.m - LOG_PACKING,
            )
        })
        .collect();
    let jags: Vec<&JaggedAssertion> = rts.iter().map(|rt| &rt.jag).collect();
    let jagged_p: Vec<JaggedKeyProve<'_>> = (0..n_keys)
        .map(|j| {
            (
                key_digests[j],
                &params_j[j],
                key_kids[j].iter().map(|&i| jags[i]).collect(),
            )
        })
        .collect();
    let jagged_v: Vec<JaggedKeyVerify<'_>> = (0..n_keys)
        .map(|j| {
            (
                key_digests[j],
                key_kids[j].iter().map(|&i| jags[i]).collect(),
            )
        })
        .collect();
    let sigma_keys: Vec<SigmaKey<'_>> = (0..n_keys)
        .map(|j| {
            (
                key_circuits[j],
                key_kids[j].iter().map(|&i| &sigmas[i]).collect(),
            )
        })
        .collect();
    // THE PRIOR (the spine): the node child's published block, normalized
    // to this node's slots — an inherited entry whose published key names
    // the slot's circuit folds; one that does not is GATED to the zero
    // claim and its live original becomes an ORPHAN, which the passenger
    // carries rather than drops.
    let prior_acc: Option<Accumulator> = spine.as_ref().map(|sp| {
        let p = sp.prior;
        assert_eq!(p.sigma.len(), N_KEY_SLOTS, "the prior's sigma slots");
        assert_eq!(p.jagged.len(), N_KEY_SLOTS, "the prior's jagged slots");
        let want: Vec<[F128; 2]> = key_digests.iter().map(digest_f128).collect();
        let norm = |slots: &[([F128; 2], MatrixClaim)]| -> Vec<([u8; 32], MatrixClaim)> {
            slots
                .iter()
                .enumerate()
                .map(|(j, (k, c))| {
                    let hit = *k == want[j];
                    // The FL slot is the SAME shape at every level — its
                    // key is wired equal in-circuit, so a miss here is a
                    // broken spine, not a case the passenger covers.
                    assert!(j > 0 || hit, "the FL slot's inherited key must match");
                    (key_digests[j], gate_claim(c, hit))
                })
                .collect()
        };
        Accumulator {
            registry_digest: registry.digest(),
            per_type: p.per_type.clone(),
            per_element: p.per_element.clone(),
            sigma: norm(&p.sigma),
            jagged: norm(&p.jagged),
        }
    });
    let priors: Vec<&Accumulator> = prior_acc.iter().collect();
    let mut chp = FsChallenger::with_chained_blake3(M11_NODE_DOMAIN);
    let (agg, acc_p) = prove_aggregate_classes_with_grinding(
        registry,
        &mats,
        &lcs,
        &bool_asserts,
        &el_mats,
        &el_asserts,
        &sigma_keys,
        &jagged_p,
        &priors,
        tower_fold_grinding(cfg),
        &mut chp,
    )
    .expect("the node fold proves");
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(M11_NODE_DOMAIN));
    let acc_v = verify_aggregate_classes_with_grinding(
        registry,
        &bool_asserts,
        &el_asserts,
        &sigma_keys,
        &jagged_v,
        &priors,
        &agg,
        tower_fold_grinding(cfg),
        &mut rec,
    )
    .expect("the node fold verifies");
    assert_eq!(acc_p, acc_v, "prover and verifier accumulators agree");
    assert!(acc_v.discharge(&mats), "the boolean group discharges");
    assert!(
        acc_v.discharge_element(&el_mats),
        "the element group discharges"
    );
    assert!(
        acc_v.discharge_sigma(&key_circuits),
        "the sigma group discharges"
    );
    assert_eq!(acc_v.jagged.len(), n_keys, "one jagged entry per key");
    let jag_tables: Vec<([u8; 32], &JaggedParams)> = (0..n_keys)
        .map(|j| (key_digests[j], &params_j[j]))
        .collect();
    assert!(
        acc_v.discharge_jagged(&jag_tables),
        "the folded jagged entries discharge against their children's layouts"
    );

    // The fold groups in aggregate order, from the CHILDREN'S OWN
    // assertion data (the same constructors the verifier gathers with).
    let bc: Vec<_> = rts
        .iter()
        .map(|rt| rt.mat_assert.claims(registry))
        .collect();
    let ec: Vec<_> = rts
        .iter()
        .zip(&unions)
        .map(|(rt, u)| rt.el_assert.claims(u))
        .collect();
    // One group per (type, side): the PRIOR's claim first when a spine
    // rides (`gather`'s order — priors, then assertions), then one per
    // child. The fold machinery is claim-count-generic, so both arity and
    // the prior enter here only as the length of these vectors.
    let pri = prior_acc.as_ref();
    let mut fold_claims: Vec<Vec<MatrixClaim>> = Vec::new();
    for t in 0..n_bool {
        for side in 0..2 {
            let mut g: Vec<MatrixClaim> = pri
                .map(|p| {
                    if side == 0 {
                        p.per_type[t].0.clone()
                    } else {
                        p.per_type[t].1.clone()
                    }
                })
                .into_iter()
                .collect();
            g.extend((0..n_kids).map(|i| {
                if side == 0 {
                    bc[i][t].0.clone()
                } else {
                    bc[i][t].1.clone()
                }
            }));
            fold_claims.push(g);
        }
    }
    for t in 0..n_el {
        for side in 0..2 {
            let mut g: Vec<MatrixClaim> = pri
                .map(|p| {
                    if side == 0 {
                        p.per_element[t].0.clone()
                    } else {
                        p.per_element[t].1.clone()
                    }
                })
                .into_iter()
                .collect();
            g.extend((0..n_kids).map(|i| {
                if side == 0 {
                    ec[i][t].0.clone()
                } else {
                    ec[i][t].1.clone()
                }
            }));
            fold_claims.push(g);
        }
    }
    // The SIGMA slots close the uniform tape, one per key.
    let n_uni = fold_claims.len();
    for j in 0..n_keys {
        let mut g: Vec<MatrixClaim> = pri.map(|p| p.sigma[j].1.clone()).into_iter().collect();
        g.extend(key_kids[j].iter().flat_map(|&i| sigmas[i].claims()));
        fold_claims.push(g);
    }
    let mut fold_proofs: Vec<&FoldProof> = Vec::new();
    for t in 0..n_bool {
        fold_proofs.push(&agg.folds[t].0);
        fold_proofs.push(&agg.folds[t].1);
    }
    for t in 0..n_el {
        fold_proofs.push(&agg.el_folds[t].0);
        fold_proofs.push(&agg.el_folds[t].1);
    }
    fold_proofs.extend(agg.sigma_folds.iter());
    let n_folds = fold_claims.len();

    // ---- the fold tape, pinned through the width-driven helpers ----
    let t_shape = rec.shape();
    let ops = flatten_ops(t_shape.ops());
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![
        Op::Label(b"flock-aggregate-v0".to_vec()),
        Op::ObserveBytes(32),
        Op::ObserveBytes(1),
    ];
    want.extend(fold_region_ops(cfg, &fold_claims[..n_uni]));
    // The sigma group binds per key (wall 3): its label + digest precede
    // each key's fold, exactly as the jagged groups bind.
    for j in 0..n_keys {
        want.push(Op::Label(b"flock-aggregate-sigma-v1".to_vec()));
        want.push(Op::ObserveBytes(32));
        want.extend(fold_region_ops(cfg, &fold_claims[n_uni + j..n_uni + j + 1]));
    }
    // The jagged groups ride the SAME tape after the uniform folds — the
    // prior's (gated) entry first, then that key's children's claims.
    let jagged_keys: Vec<([u8; 32], Vec<JaggedClaim>)> = (0..n_keys)
        .map(|j| {
            let mut cs: Vec<JaggedClaim> = pri
                .map(|p| {
                    JaggedClaim::from_folded(&p.jagged[j].1)
                        .expect("an inherited jagged entry is scaled plain eq")
                })
                .into_iter()
                .collect();
            cs.extend(
                key_kids[j]
                    .iter()
                    .flat_map(|&i| jags[i].claims().into_iter().cloned()),
            );
            (key_digests[j], cs)
        })
        .collect();
    want.extend(jagged_fold_region_ops(cfg, &jagged_keys));
    assert_eq!(ops, want.as_slice(), "the node tape is the expected shape");
    assert_eq!(
        rec.payloads()[0],
        registry.digest(),
        "bind: registry digest"
    );
    assert_eq!(
        rec.payloads()[1],
        vec![priors.len() as u8],
        "bind: prior count"
    );
    let sigma_payloads = labeled_bytes_payloads(&ops, b"flock-aggregate-sigma-v1");
    let jagged_payloads = labeled_bytes_payloads(&ops, b"flock-aggregate-jagged-v0");
    assert_eq!(
        sigma_payloads.len(),
        n_keys,
        "one sigma digest payload per key"
    );
    assert_eq!(
        jagged_payloads.len(),
        n_keys,
        "one jagged digest payload per key"
    );
    for j in 0..n_keys {
        assert_eq!(
            rec.payloads()[sigma_payloads[j]],
            key_digests[j].to_vec(),
            "the sigma slot {j} key payload"
        );
    }
    let (locs, vcur, ccur) = locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals);
    let jfps: Vec<&FoldProof> = agg.jagged_folds.iter().collect();
    let jlocs = locate_and_pin_jagged_folds(
        &jagged_keys,
        &jfps,
        vals_rec,
        chals,
        rec.payloads(),
        &jagged_payloads,
        vcur,
        ccur,
    );
    let outs = replay_fold_endpoints(&locs, vals_rec, chals);
    for t in 0..n_bool {
        assert_eq!(outs[2 * t], acc_v.per_type[t].0, "boolean type {t} A");
        assert_eq!(outs[2 * t + 1], acc_v.per_type[t].1, "boolean type {t} B");
    }
    for t in 0..n_el {
        assert_eq!(
            outs[2 * n_bool + 2 * t],
            acc_v.per_element[t].0,
            "element type {t} A"
        );
        assert_eq!(
            outs[2 * n_bool + 2 * t + 1],
            acc_v.per_element[t].1,
            "element type {t} B"
        );
    }
    for j in 0..n_keys {
        let (d, c) = &acc_v.sigma[j];
        assert_eq!(outs[n_uni + j], *c, "sigma slot {j} accumulator");
        assert_eq!(*d, key_digests[j], "sigma slot {j} key");
    }
    let jouts = replay_jagged_fold_endpoints(&jlocs, vals_rec, chals);
    for j in 0..n_keys {
        assert_eq!(
            jouts[j], acc_v.jagged[j].1,
            "the jagged slot {j} entry from located words"
        );
    }

    // ---- the LANE (task 6): the children's LOWER-registry accumulators
    // fold PRIORS-ONLY — natively here, in-circuit below. 3 groups
    // (bool A/B + sigma) × [priorL, priorR], no fresh claims. ----
    const LANE_DOMAIN: &[u8] = b"flock-chain-lane-v0";
    let lane_native = lane.as_ref().map(|ln| {
        let el_asserts_l: [(&UnionInstance<'_>, ElementAssertion); 0] = [];
        // The jagged key rides PRIORS-ONLY through the lane, exactly like
        // the lane's other groups: the FL children's chain-keyed entries
        // fold with no fresh claims.
        let ljagged_p: Vec<JaggedKeyProve<'_>> = vec![(ln.circuit.digest(), ln.params, Vec::new())];
        let ljagged_v: Vec<JaggedKeyVerify<'_>> = vec![(ln.circuit.digest(), Vec::new())];
        let mut chp = FsChallenger::with_chained_blake3(LANE_DOMAIN);
        let (lagg, lacc_p) = prove_aggregate_classes_with_grinding(
            ln.registry,
            ln.mats,
            ln.circs,
            &[],
            &[],
            &el_asserts_l,
            &[(ln.circuit, Vec::new())],
            &ljagged_p,
            ln.priors,
            tower_fold_grinding(cfg),
            &mut chp,
        )
        .expect("the lane fold proves");
        let mut lrec = RecordingChallenger::new(FsChallenger::with_chained_blake3(LANE_DOMAIN));
        let lacc_v = verify_aggregate_classes_with_grinding(
            ln.registry,
            &[],
            &el_asserts_l,
            &[(ln.circuit, Vec::new())],
            &ljagged_v,
            ln.priors,
            &lagg,
            tower_fold_grinding(cfg),
            &mut lrec,
        )
        .expect("the lane fold verifies");
        assert_eq!(lacc_p, lacc_v, "lane prover and verifier agree");
        assert_eq!(
            lacc_v.jagged.len(),
            1,
            "the lane carries the chain jagged key"
        );
        assert!(
            lacc_v.discharge_jagged(&[(ln.circuit.digest(), ln.params)]),
            "the lane's folded jagged entry discharges against the chain layout"
        );
        let lclaims: Vec<Vec<MatrixClaim>> = vec![
            ln.priors.iter().map(|p| p.per_type[0].0.clone()).collect(),
            ln.priors.iter().map(|p| p.per_type[0].1.clone()).collect(),
            ln.priors
                .iter()
                .map(|p| p.sigma.first().expect("lane prior sigma").1.clone())
                .collect(),
        ];
        let lproofs: Vec<&FoldProof> =
            vec![&lagg.folds[0].0, &lagg.folds[0].1, &lagg.sigma_folds[0]];
        let lops: Vec<Op> = flatten_ops(lrec.shape().ops());
        let lvals: Vec<F128> = lrec.values().to_vec();
        let lchals: Vec<F128> = lrec.challenges().to_vec();
        let mut want: Vec<Op> = vec![
            Op::Label(b"flock-aggregate-v0".to_vec()),
            Op::ObserveBytes(32),
            Op::ObserveBytes(1),
        ];
        let n_uni_l = lclaims.len() - 1;
        want.extend(fold_region_ops(cfg, &lclaims[..n_uni_l]));
        want.push(Op::Label(b"flock-aggregate-sigma-v1".to_vec()));
        want.push(Op::ObserveBytes(32));
        want.extend(fold_region_ops(cfg, &lclaims[n_uni_l..]));
        // The inherited jagged claims (the priors' chain-keyed entries,
        // plain eq by construction) ride the same tape after.
        let ljagged_keys: Vec<([u8; 32], Vec<JaggedClaim>)> = vec![(
            ln.circuit.digest(),
            ln.priors
                .iter()
                .flat_map(|p| p.jagged.iter())
                .filter(|(d, _)| *d == ln.circuit.digest())
                .map(|(_, c)| {
                    JaggedClaim::from_folded(c).expect("prior jagged entries are plain eq")
                })
                .collect(),
        )];
        want.extend(jagged_fold_region_ops(cfg, &ljagged_keys));
        assert_eq!(lops, want, "the lane tape shape");
        assert_eq!(
            lrec.payloads()[0],
            ln.registry.digest(),
            "lane registry digest"
        );
        assert_eq!(
            lrec.payloads()[1],
            vec![ln.priors.len() as u8],
            "lane prior count"
        );
        let (llocs, lvcur, lccur) = locate_and_pin_folds(&lclaims, &lproofs, &lvals, &lchals);
        let ljfps: Vec<&FoldProof> = lagg.jagged_folds.iter().collect();
        let ljlocs = locate_and_pin_jagged_folds(
            &ljagged_keys,
            &ljfps,
            &lvals,
            &lchals,
            lrec.payloads(),
            &labeled_bytes_payloads(&lops, b"flock-aggregate-jagged-v0"),
            lvcur,
            lccur,
        );
        let louts = replay_fold_endpoints(&llocs, &lvals, &lchals);
        assert_eq!(louts[0], lacc_v.per_type[0].0, "lane boolean A");
        assert_eq!(louts[1], lacc_v.per_type[0].1, "lane boolean B");
        let (ld, lc2) = lacc_v.sigma.first().expect("lane sigma out");
        assert_eq!(louts[2], *lc2, "lane sigma accumulator");
        assert_eq!(
            *ld,
            ln.circuit.digest(),
            "lane sigma keys by the chain circuit"
        );
        let ljouts = replay_jagged_fold_endpoints(&ljlocs, &lvals, &lchals);
        assert_eq!(
            ljouts[0], lacc_v.jagged[0].1,
            "lane jagged entry from located words"
        );
        let lstream = lrec.shape().stream_words_duplex(LANE_DOMAIN);
        let lbytes = lstream.to_bytes(lrec.values(), lrec.payloads());
        (lacc_v, llocs, ljlocs, lstream, lbytes, lops, lchals, lvals)
    });

    // ---- ONE outer: two REAL child regions + the fold region ----

    {
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
            &t_shape.stream_words_duplex(M11_NODE_DOMAIN),
            rec.values(),
            rec.payloads(),
        );
        assert_chain_replays(&ops, &trace, chals);

        let env = envelope_shape();
        let split_b3 = n_kids == 2;
        let (fold_b3_primary_rows, b3_rows) = if split_b3 {
            let a = rts[0].b3_rows;
            let b = rts[1].b3_rows;
            let unsplit = (a + trace.rows.len()).max(b);
            let (on_a, balanced) = balance_extra_rows(a, b, trace.rows.len());
            if unsplit > (1usize << env.nu) {
                (Some(on_a), balanced)
            } else {
                (None, unsplit)
            }
        } else {
            (
                None,
                rts.iter().map(|rt| rt.b3_rows).sum::<usize>() + trace.rows.len(),
            )
        };
        if var("B3_CENSUS").is_ok() {
            let fold_pows = ops
                .iter()
                .filter(|op| matches!(op, Op::Pow { bits } if *bits != 0))
                .count();
            eprintln!(
                "  [node pow census] child checks {:?} | fold checks {} | standalone BLAKE rows 0",
                rts.iter().map(|rt| rt.pows.len()).collect::<Vec<_>>(),
                fold_pows,
            );
        }
        let nu2_content = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        // The node pins the envelope's nu* and canonical type set (wall 2).
        assert!(
            nu2_content <= env.nu,
            "node content nu {nu2_content} exceeds the envelope nu* {}",
            env.nu
        );
        let nu2 = env.nu;
        let mut sb = ShapeBuilder::new(nu2);
        // The DECLARED width is the envelope's (the max over child kinds at
        // the fixed point); a shallower child ladder rides the wide slot
        // with its high outputs unread, and one that exceeds it fails here.
        // The witness tables below build at `spread_w2`, so it must be the
        // DECLARED width.
        let spread_own2 = rts.iter().map(|rt| rt.spread_w).max().expect("a child");
        assert!(
            spread_own2 <= env.spread_w,
            "child ladder depth {spread_own2} exceeds the envelope spread width {}",
            env.spread_w
        );
        let spread_w2 = env.spread_w;
        let mut cs = ChildSlots::new_env(&mut sb, nu2, &env);
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        // The two child regions are independent gate subgraphs (each reads
        // only its own tape's inputs; the fold region joins them AFTER) —
        // declared as islands so the online phase evaluates them in
        // parallel.
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let mac_c0_start = sb.rows_in_slot(cs.macs);
        let mut mac_marks: Vec<usize> = Vec::with_capacity(n_kids);
        let regions: Vec<RealRegion> = rts
            .iter()
            .enumerate()
            .map(|(i, rt)| {
                let isl = sb.begin_island();
                let b3_slot = match (i, cs.q.b3_alt) {
                    (0, _) => cs.q.b3,
                    (1, Some(slot)) => slot,
                    (_, None) => cs.q.b3,
                    _ => panic!("split-BLAKE recursion supports exactly two children"),
                };
                let r = emit_real_child_region(
                    &mut sb,
                    &mut cs,
                    b3_slot,
                    rt,
                    &mut vals,
                    &mut hints,
                    &mut consts,
                );
                sb.end_island(isl);
                mac_marks.push(sb.rows_in_slot(cs.macs));
                r
            })
            .collect();
        let r0 = &regions[0];
        // The fold region rides the children's slots: rows, not columns.
        let (pfslot, pf_w) = r0.pf;
        let leslot = cs
            .le
            .iter()
            .find(|&&(n, _)| n == 8)
            .map(|&(_, s)| s)
            .expect("the child regions created the 8-lane leaf-eval slot");
        let iv_w = pack8(&IV);
        vals.extend_from_slice(&iv_w);
        let iv2 = [
            sb.fixed_public_input(iv_w[0]),
            sb.fixed_public_input(iv_w[1]),
        ];
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        let pub_payloads = bytes_payload_mask(&flatten_ops(t_shape.ops()));
        let (chain_outs, ww) = emit_fs_chain_partitioned(
            &mut sb,
            cs.q.b3,
            fold_b3_primary_rows.map(|n| {
                (
                    cs.q.b3_alt
                        .expect("a balanced fold chain needs the second BLAKE slot"),
                    n,
                )
            }),
            iv2,
            &trace,
            &stream,
            &bytes,
            &mut vals,
            &mut consts,
            &pub_payloads,
            &cross,
        );
        emit_recorded_pow_checks(
            &mut sb,
            cs.q.b3,
            cs.q.pow,
            iv2,
            &ops,
            &trace,
            &stream,
            &chain_outs,
            &ww,
            &mut vals,
            &mut consts,
        );
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
        let (fold_pubs, alpha_recs) = emit_fold_region(
            &mut sb,
            cs.fold_macs,
            cs.mrs,
            pfslot,
            pf_w,
            leslot,
            &locs,
            &trace,
            &challenge_word_locs(t_shape.ops()),
            &chain_outs,
            &ww,
            &vmap,
            chals,
            vals_rec,
            &mut vals,
            zw,
            ow,
            false, // the jagged group follows on the same tape
        );
        let jfold_pubs = emit_jagged_fold_region(
            &mut sb,
            cs.fold_macs,
            cs.mrs,
            pfslot,
            pf_w,
            &jlocs,
            &trace,
            &challenge_word_locs(t_shape.ops()),
            &chain_outs,
            &ww,
            &vmap,
            vals_rec,
            &mut vals,
            zw,
            ow,
        );
        // ---- THE SPINE (wall 3): the node child's published ACC_MAIN
        // block IS this node's prior, read wire-to-wire out of that
        // child's public segment at the envelope's constant offset — the
        // lane's `claims_base` machinery, now on the MAIN fold. ----
        //
        // The registry-keyed matrix entries ride straight in (lows to the
        // child's live word, piece 1's wiring). The KEYED slots go through
        // the MATCH-GATE: slot `j` folds claims about child `j`'s tables,
        // and the entry inherited at slot `j` names whatever circuit the
        // CHILD's own slot `j` was about. Slot 0 (the fresh FL slot) is the
        // same FL shape at every level, so its key is a hard CONNECT — a
        // mismatch is a broken spine, not a case. Slot 1 (the node slot)
        // genuinely mismatches exactly once, at the first steady node over
        // a base node, and there the entry is gated to the zero claim and
        // its live original rides the PASSENGER instead of being dropped.
        //
        // `g = live · match` scales the claim's lows AND its value: a
        // gated-off entry must claim ZERO, not its old value about a
        // weight that is now zero.
        // The keyed slots' entry widths — the same for a live slot and a
        // dead one, which is what makes the layout readable at a constant
        // offset. `[key(2) | live | rho_col | rho_row | value]`.
        let sig_ent = 4 + locs[n_uni].k_col + locs[n_uni].k_row;
        let jag_ent = 4 + jlocs[0].n_col + jlocs[0].k_row;
        // The spine gadget's mac rows, bracketed for the ADVERSARIAL leg:
        // 26 rows exactly — 13 per keyed slot-1 gate (two is-eq gadgets of
        // 4 rows + m, g, gv, nm, h), the FL slot emitting none (its key is
        // a hard connect).
        let mac_spine0 = sb.rows_in_slot(cs.macs);
        let spine_w = spine.as_ref().map(|sp| {
            let e = &env;
            let rk = &regions[sp.node_child];
            let cp = |i: usize| rk.child_pub_w[i];
            // The assert-zero anchor for the gadget below: producers only,
            // no consumer edges (the lagrange lows' pattern).
            vals.push(F128::ZERO);
            let za = sb.public_input();
            // eq(a, b) as a BIT: d = a + b, an advice inverse w, and
            // z = 1 + d·w with z·d == 0 — z is 1 exactly when d is 0 (to
            // claim z = 1 with d ≠ 0 a prover needs w = 0, and then
            // z·d = d ≠ 0 fails the assert).
            let is_eq =
                |sb: &mut ShapeBuilder, vals: &mut Vec<F128>, a: Wire, b: Wire, d: F128| -> Wire {
                    let d_w = sb.gate(cs.macs, &[a, b, ow])[0];
                    vals.push(d.inv());
                    let inv_w = sb.input();
                    let p_w = sb.gate(cs.macs, &[zw, d_w, inv_w])[0];
                    let z_w = sb.gate(cs.macs, &[ow, p_w, ow])[0];
                    let chk = sb.gate(cs.macs, &[zw, z_w, d_w])[0];
                    sb.connect(chk, za);
                    z_w
                };
            // Walk the child's block exactly as this node publishes its
            // own — the layouts coincide, which is the shape fact the
            // whole spine rests on.
            let mut off = env_acc_main_base(e);
            let uni_off: Vec<usize> = (0..n_uni)
                .map(|i| {
                    let o = off;
                    off += 2 + locs[i].k_col + locs[i].k_row;
                    o
                })
                .collect();
            let sig_off: Vec<usize> = (0..N_KEY_SLOTS)
                .map(|_| {
                    let o = off;
                    off += sig_ent;
                    o
                })
                .collect();
            let jag_off: Vec<usize> = (0..N_KEY_SLOTS)
                .map(|_| {
                    let o = off;
                    off += jag_ent;
                    o
                })
                .collect();
            assert!(
                off - env_acc_main_base(e) <= ENV_ACC_MAIN_WORDS,
                "the prior's ACC_MAIN block overruns its reserved width"
            );
            // One keyed slot: the published key against this node's own,
            // then the gate. Returns (g, gated value, orphan gate h).
            let slot = |sb: &mut ShapeBuilder,
                        vals: &mut Vec<F128>,
                        o: usize,
                        j: usize,
                        ent: &([F128; 2], MatrixClaim),
                        k_col: usize,
                        k_row: usize|
             -> (Wire, Wire, Wire) {
                let live_w = cp(o + 2);
                let val_w = cp(o + 3 + k_col + k_row);
                let want = digest_f128(&key_digests[j]);
                if j == 0 {
                    // The FL slot: one shape at every level, so the key is
                    // an EQUALITY, wired, not a case.
                    sb.connect(cp(o), regions[j].cd_w[0]);
                    sb.connect(cp(o + 1), regions[j].cd_w[1]);
                    assert_eq!(ent.0, want, "the FL slot's inherited key is the FL circuit");
                    return (live_w, val_w, zw);
                }
                let m0 = is_eq(sb, vals, cp(o), regions[j].cd_w[0], ent.0[0] + want[0]);
                let m1 = is_eq(sb, vals, cp(o + 1), regions[j].cd_w[1], ent.0[1] + want[1]);
                let m_w = sb.gate(cs.macs, &[zw, m0, m1])[0];
                let g_w = sb.gate(cs.macs, &[zw, live_w, m_w])[0];
                let gv_w = sb.gate(cs.macs, &[zw, g_w, val_w])[0];
                // h = live · (1 + match) — the ORPHAN gate: live exactly
                // when this entry could not fold and must ride on.
                let nm_w = sb.gate(cs.macs, &[ow, m_w, ow])[0];
                let h_w = sb.gate(cs.macs, &[zw, live_w, nm_w])[0];
                (g_w, gv_w, h_w)
            };
            let sig: Vec<(Wire, Wire, Wire)> = (0..N_KEY_SLOTS)
                .map(|j| {
                    slot(
                        &mut sb,
                        &mut vals,
                        sig_off[j],
                        j,
                        &sp.prior.sigma[j],
                        locs[n_uni].k_col,
                        locs[n_uni].k_row,
                    )
                })
                .collect();
            let jag: Vec<(Wire, Wire, Wire)> = (0..N_KEY_SLOTS)
                .map(|j| {
                    slot(
                        &mut sb,
                        &mut vals,
                        jag_off[j],
                        j,
                        &sp.prior.jagged[j],
                        jlocs[0].n_col,
                        jlocs[0].k_row,
                    )
                })
                .collect();
            (uni_off, sig_off, jag_off, sig, jag)
        });
        if spine.is_some() {
            assert_eq!(
                sb.rows_in_slot(cs.macs) - mac_spine0,
                26,
                "the spine gadget's mac-row census"
            );
        }
        // THE POINTS-CONNECT (the count win's identity bind): value, σ,
        // row identities, and the structural words — see build_fl_node's
        // block for the argument; this is the same bind at node scale.
        let mut jag_const_rec: Vec<(F128, usize)> = Vec::new();
        {
            let mut jag_consts: Vec<(F128, Wire)> = Vec::new();
            let mut cw_j = |sb: &mut ShapeBuilder,
                            vals: &mut Vec<F128>,
                            rec2: &mut Vec<(F128, usize)>,
                            v: F128|
             -> Wire {
                if let Some(&(_, w)) = jag_consts.iter().find(|&&(x, _)| x == v) {
                    return w;
                }
                vals.push(v);
                rec2.push((v, sb.public_len()));
                let w = sb.public_input();
                jag_consts.push((v, w));
                w
            };
            for (gi, loc) in jlocs.iter().enumerate() {
                let mut ci = 0usize;
                // The INHERITED claim leads the group (aggregate's gather
                // order): its scale is the gate, its value the gated one,
                // and its points are the child's published words.
                if let Some((_, _, jag_off, _, jag)) = &spine_w {
                    let cl = &loc.claims[0];
                    let o = jag_off[gi];
                    let rk = &regions[spine.as_ref().unwrap().node_child];
                    let tag = cw_j(
                        &mut sb,
                        &mut vals,
                        &mut jag_const_rec,
                        F128::new(0, cl.row_pt.1 as u64),
                    );
                    sb.connect(wv(cl.row_scale_v - 1), tag);
                    sb.connect(wv(cl.row_scale_v), jag[gi].0);
                    for j in 0..loc.n_col {
                        sb.connect(wv(cl.col_v + j), rk.child_pub_w[o + 3 + j]);
                    }
                    for j in 0..cl.row_pt.1 {
                        sb.connect(wv(cl.row_pt.0 + j), rk.child_pub_w[o + 3 + loc.n_col + j]);
                    }
                    sb.connect(wv(cl.val_v), jag[gi].1);
                    ci = 1;
                }
                for &ki in &key_kids[gi] {
                    let rk = &regions[ki];
                    for (li, &jw) in rk.jag_w.iter().enumerate() {
                        let cl = &loc.claims[ci];
                        sb.connect(wv(cl.val_v), jw);
                        for j in 0..loc.n_col {
                            sb.connect(wv(cl.col_v + j), rk.jag_sig_w[j]);
                        }
                        if cl.terms.is_empty() {
                            let tag = cw_j(
                                &mut sb,
                                &mut vals,
                                &mut jag_const_rec,
                                F128::new(0, cl.row_pt.1 as u64),
                            );
                            sb.connect(wv(cl.row_scale_v - 1), tag);
                            // A FRESH claim is live: its scale is 1.
                            sb.connect(wv(cl.row_scale_v), ow);
                            for j in 0..cl.row_pt.1 {
                                sb.connect(wv(cl.row_pt.0 + j), rk.jag_row_w[li][j]);
                            }
                        } else {
                            let tag = cw_j(
                                &mut sb,
                                &mut vals,
                                &mut jag_const_rec,
                                F128::new(1, cl.terms.len() as u64),
                            );
                            sb.connect(wv(cl.terms[0].0 - 1), tag);
                            for (tj, &(cv, addr)) in cl.terms.iter().enumerate() {
                                sb.connect(wv(cv), rk.jag_row_w[li][tj]);
                                let aw = cw_j(
                                    &mut sb,
                                    &mut vals,
                                    &mut jag_const_rec,
                                    F128::new(addr as u64, 0),
                                );
                                sb.connect(wv(cv + 1), aw);
                            }
                        }
                        ci += 1;
                    }
                }
                assert_eq!(ci, loc.claims.len(), "every jagged claim connected");
                let header_v = loc.hdr_v;
                let hw = cw_j(
                    &mut sb,
                    &mut vals,
                    &mut jag_const_rec,
                    F128::new(loc.k_row as u64, loc.claims.len() as u64),
                );
                sb.connect(wv(header_v), hw);
            }
            // THE FOLD KEY IS THE CIRCUIT VERIFIED: each group's absorbed
            // digest payload connects to the child region's own statement
            // digest, so a slot cannot fold claims about a circuit this
            // node did not verify.
            let pays_n = payload_words(&stream);
            for j in 0..n_keys {
                for p in [sigma_payloads[j], jagged_payloads[j]] {
                    assert_eq!(pays_n[p].len(), 2, "a group key payload is 32 bytes");
                    for (b, &kw) in pays_n[p].iter().enumerate() {
                        sb.connect(ww[kw].expect("key payload wired"), regions[j].cd_w[b]);
                    }
                }
            }
        }
        let mac_after_fold = sb.rows_in_slot(cs.macs);

        // ---- THE 2→1 CONNECTS: the fold's absorbed claim surfaces ARE
        // the real child regions' assertion-emission wires ----
        // Per child, per family: points to chain squeeze wires, z_partial
        // lows to absorbed child words, sigma fully (value = the child's
        // deferred s_sigma stream word), and — richer than the minimal
        // children — the matrix/element EVAL VALUES to the children's own
        // bound advice publics. Only the lagrange row lows stay the
        // boundary pattern (published below, rebuilt by the checker from
        // each child's PUBLISHED z_skip; SkipNodeGate/φ8 is the recorded
        // upgrade).
        // The lagrange-low constants, shared by both children: the 64 φ8
        // nodes and the subspace denominator inverse — statement constants
        // the checker validates below (the ONE public surface the
        // in-circuit derivation adds).
        // The RS lows machinery is emitted only when an RS child consumes
        // it; AG children take the Tier-0 published surface instead.
        let any_rs = rts
            .iter()
            .any(|tk| matches!(tk.zskip, ZskipTapeRec::Rs { .. }));
        let rs_lam: Option<(usize, Vec<Wire>, Wire, Wire)> = any_rs.then(|| {
            let lam_base = sb.public_len();
            let lam_w: Vec<Wire> = PHI_8_TABLE[..1 << K_SKIP]
                .iter()
                .map(|&v| {
                    vals.push(v);
                    sb.public_input()
                })
                .collect();
            vals.push(subspace_denominator_pair(K_SKIP).1);
            let deninv_w = sb.public_input();
            // The lows' assert-zero anchor: producers only, no consumers.
            vals.push(F128::ZERO);
            let lag_zassert = sb.public_input();
            (lam_base, lam_w, deninv_w, lag_zassert)
        });
        // Per AG child, the Tier-0 public block's base — the layout
        // [`check_ag_skip_publics`] walks.
        let mut ag_pub_bases: Vec<Option<usize>> = Vec::new();
        // THE PRIOR's uniform surfaces (the spine): claim 0 of every group.
        // The registry-keyed matrix entries ride in exactly as the lane's
        // priors do — LOWS to the child's live word, points and value
        // straight through — and the sigma slots ride the same wiring with
        // the MATCH-GATE's outputs in place of live and value.
        let cj = if let Some((uni_off, _, _, sig, _)) = &spine_w {
            let rk = &regions[spine.as_ref().unwrap().node_child];
            for (i, loc) in locs.iter().enumerate().take(n_uni) {
                let cl = &loc.claims[0];
                let o = uni_off[i];
                assert_eq!(
                    cl.row_low_n, 1,
                    "an inherited claim's lows are its live word"
                );
                sb.connect(wv(cl.row_low_v), rk.child_pub_w[o]);
                sb.connect(wv(cl.col_low_v), rk.child_pub_w[o]);
                for j in 0..cl.col_pt_n {
                    sb.connect(wv(cl.col_pt_v + j), rk.child_pub_w[o + 1 + j]);
                }
                for j in 0..cl.row_pt_n {
                    sb.connect(wv(cl.row_pt_v + j), rk.child_pub_w[o + 1 + loc.k_col + j]);
                }
                sb.connect(
                    wv(cl.value_v),
                    rk.child_pub_w[o + 1 + loc.k_col + loc.k_row],
                );
            }
            for j in 0..n_keys {
                let loc = &locs[n_uni + j];
                let cl = &loc.claims[0];
                let o = spine_w.as_ref().unwrap().1[j];
                sb.connect(wv(cl.row_low_v), sig[j].0);
                sb.connect(wv(cl.col_low_v), sig[j].0);
                for i2 in 0..cl.col_pt_n {
                    sb.connect(wv(cl.col_pt_v + i2), rk.child_pub_w[o + 3 + i2]);
                }
                for i2 in 0..cl.row_pt_n {
                    sb.connect(wv(cl.row_pt_v + i2), rk.child_pub_w[o + 3 + loc.k_col + i2]);
                }
                sb.connect(wv(cl.value_v), sig[j].1);
            }
            1
        } else {
            0
        };
        for (k, (tk, rk)) in rts.iter().zip(&regions).enumerate() {
            // Basis-generic native pre-assert: the fold's absorbed lows
            // ARE the skip functional at the child's own z_skip point.
            assert_eq!(
                &fold_claims[0][cj + k].row.low[..],
                &tk.mat_assert.z_skip.weights(K_SKIP)[..],
                "child {k}: the fold's row lows are the skip functional"
            );
            match (&tk.zskip, &rk.zskip) {
                (ZskipTapeRec::Rs { ch, .. }, ZskipWires::Rs(zskip_w)) => {
                    let (_, lam_w, deninv_w, lag_zassert) =
                        rs_lam.as_ref().expect("the RS lows machinery is emitted");
                    let lows = emit_lagrange_lows(
                        &mut sb,
                        cs.macs,
                        lam_w,
                        *deninv_w,
                        *zskip_w,
                        tk.chals[*ch],
                        &mut vals,
                        zw,
                        ow,
                        *lag_zassert,
                    );
                    for (j, &lw2) in lows.iter().enumerate() {
                        sb.connect(lw2, wv(locs[0].claims[cj + k].row_low_v + j));
                    }
                    ag_pub_bases.push(None);
                }
                (ZskipTapeRec::Ag { seed_ch, .. }, ZskipWires::Ag { seed_w, nonce_w }) => {
                    // TIER 1 (phase D): publish [seed₂, nonce, point₅,
                    // lows₆₄] — seed/nonce/lows wire-connected as before —
                    // and BIND the point in-circuit
                    // ([`emit_ag_point_binding`]). The native checker keeps
                    // the nonce range and `lows == bf(point)`.
                    let base = sb.public_len();
                    for (v, src) in [
                        (tk.chals[*seed_ch], seed_w[0]),
                        (tk.chals[*seed_ch + 1], seed_w[1]),
                    ] {
                        vals.push(v);
                        let w = sb.public_input();
                        sb.connect(w, src);
                    }
                    let nonce = match &tk.lo.proof {
                        MixedProof::Ag(p) => {
                            p.boolean
                                .as_ref()
                                .expect("boolean side present")
                                .ag
                                .r1_nonce
                        }
                        MixedProof::Rs(_) => unreachable!("an AG tape carries an AG proof"),
                    };
                    vals.push(F128::new(u64::from(nonce), 0));
                    let w = sb.public_input();
                    sb.connect(w, *nonce_w);
                    let SkipPoint::Ag(pt) = tk.mat_assert.z_skip else {
                        unreachable!("an AG tape carries an AG skip point")
                    };
                    let pt_w: [Wire; 5] = [pt.x, pt.y, pt.z1, pt.z2, pt.z3].map(|c| {
                        vals.push(c);
                        sb.public_input()
                    });
                    for (j, &lv) in fold_claims[0][cj + k].row.low.iter().enumerate() {
                        vals.push(lv);
                        let lw2 = sb.public_input();
                        sb.connect(lw2, wv(locs[0].claims[cj + k].row_low_v + j));
                    }
                    emit_ag_point_binding(
                        &mut sb,
                        cs.q.b3,
                        cs.q.pow,
                        cs.macs,
                        iv2,
                        *seed_w,
                        *nonce_w,
                        &pt_w,
                        [tk.chals[*seed_ch], tk.chals[*seed_ch + 1]],
                        nonce,
                        &pt,
                        los[k].pcs.zerocheck_grinding().ag_r1_bits(),
                        &mut vals,
                        &mut consts,
                        zw,
                        ow,
                    );
                    ag_pub_bases.push(Some(base));
                }
                _ => unreachable!("the region's zskip wires match the tape flavor"),
            }
            // Native pre-asserts (the method-note discipline).
            for t in 0..n_bool {
                let inner_t = fold_claims[2 * t][cj + k].row.point.len();
                assert_eq!(
                    &fold_claims[2 * t][cj + k].row.point[..],
                    &tk.mat_assert.x_inner_rest[..inner_t],
                    "boolean type {t} row point is x_inner_rest's head"
                );
                assert_eq!(
                    &fold_claims[2 * t][cj + k].col.point[..],
                    &tk.mat_assert.rr[..inner_t],
                    "boolean type {t} col point is rr's head"
                );
                assert_eq!(
                    &fold_claims[2 * t][cj + k].col.low[..],
                    &tk.mat_assert.z_partial[..],
                    "boolean type {t} col low is z_partial"
                );
                assert_eq!(fold_claims[2 * t][cj + k].value, tk.mat_assert.evals[t].0);
                assert_eq!(
                    fold_claims[2 * t + 1][cj + k].value,
                    tk.mat_assert.evals[t].1
                );
            }
            for t in 0..n_el {
                let kappa = fold_claims[2 * n_bool + 2 * t][cj + k].row.point.len();
                assert_eq!(
                    &fold_claims[2 * n_bool + 2 * t][cj + k].row.point[..],
                    &tk.el_assert.r_con[..kappa],
                    "element type {t} row point is r_con's head"
                );
                assert_eq!(
                    &fold_claims[2 * n_bool + 2 * t][cj + k].col.point[..],
                    &tk.el_assert.r_col[..kappa],
                    "element type {t} col point is r_col's head"
                );
                assert_eq!(
                    fold_claims[2 * n_bool + 2 * t][cj + k].value,
                    tk.el_assert.evals[t].0
                );
                assert_eq!(
                    fold_claims[2 * n_bool + 2 * t + 1][cj + k].value,
                    tk.el_assert.evals[t].1
                );
            }
            let sfi0 = if spine.is_some() { n_uni + k } else { n_uni };
            let n_structure = tk.sigma_native.claims().len();
            let sk0 = if spine.is_some() {
                cj
            } else {
                cj + n_structure * k
            };
            for (j, claim) in tk.sigma_native.claims().iter().enumerate() {
                assert_eq!(&fold_claims[sfi0][sk0 + j], claim);
            }

            // boolean A/B per type: batch-major mlv mapping for the row
            // points, lc rounds REVERSED for the col points, z_partial
            // word-for-word, values to the mat_eval advice wires.
            for t in 0..n_bool {
                for fi in [2 * t, 2 * t + 1] {
                    let cl = &locs[fi].claims[cj + k];
                    for j in 0..cl.row_pt_n {
                        let m = if j == 0 { 0 } else { tk.n_log_i + j };
                        sb.connect(wv(cl.row_pt_v + j), rk.b_mlv_w[m]);
                    }
                    let n_lc = rk.b_lc_w.len();
                    for j in 0..cl.col_pt_n {
                        sb.connect(wv(cl.col_pt_v + j), rk.b_lc_w[n_lc - 1 - j]);
                    }
                    for j in 0..cl.col_low_n {
                        sb.connect(wv(cl.col_low_v + j), rk.b_zpartial_w[j]);
                    }
                }
                sb.connect(wv(locs[2 * t].claims[cj + k].value_v), rk.mat_eval_w[t].0);
                sb.connect(
                    wv(locs[2 * t + 1].claims[cj + k].value_v),
                    rk.mat_eval_w[t].1,
                );
                // ONE lagrange-low surface per child (lagrange(z_skip) is
                // type-independent): every boolean fold's lows connect to
                // fold 0's, and fold 0's publish below.
                if t > 0 {
                    for fi in [2 * t, 2 * t + 1] {
                        for j in 0..locs[0].claims[cj + k].row_low_n {
                            sb.connect(
                                wv(locs[fi].claims[cj + k].row_low_v + j),
                                wv(locs[0].claims[cj + k].row_low_v + j),
                            );
                        }
                    }
                } else {
                    for j in 0..locs[0].claims[cj + k].row_low_n {
                        sb.connect(
                            wv(locs[1].claims[cj + k].row_low_v + j),
                            wv(locs[0].claims[cj + k].row_low_v + j),
                        );
                    }
                }
            }
            // element A/B per type: r_con = zc.r[ν..] (round order), r_col
            // = the lc rounds REVERSED, values to the per-slot eval advice.
            for t in 0..n_el {
                for fi in [2 * n_bool + 2 * t, 2 * n_bool + 2 * t + 1] {
                    let cl = &locs[fi].claims[cj + k];
                    sb.connect(wv(cl.row_low_v), ow);
                    sb.connect(wv(cl.col_low_v), ow);
                    for j in 0..cl.row_pt_n {
                        sb.connect(wv(cl.row_pt_v + j), rk.el_zc_rho_w[tk.n_log_i + j]);
                    }
                    let n_lc = rk.el_lc_rho_w.len();
                    for j in 0..cl.col_pt_n {
                        sb.connect(wv(cl.col_pt_v + j), rk.el_lc_rho_w[n_lc - 1 - j]);
                    }
                }
                sb.connect(
                    wv(locs[2 * n_bool + 2 * t].claims[cj + k].value_v),
                    rk.el_eval_w[t].0,
                );
                sb.connect(
                    wv(locs[2 * n_bool + 2 * t + 1].claims[cj + k].value_v),
                    rk.el_eval_w[t].1,
                );
            }
            // Circuit structure: every static helper evaluation rides the
            // child's key slot. A fresh-only node folds every child under
            // slot 0.
            let sfi = if spine.is_some() { n_uni + k } else { n_uni };
            let sk = if spine.is_some() {
                cj
            } else {
                cj + rk.structure_claim_w.len() * k
            };
            for (j, (row_w, col_w, value_w)) in rk.structure_claim_w.iter().enumerate() {
                let cl = &locs[sfi].claims[sk + j];
                sb.connect(wv(cl.row_low_v), ow);
                sb.connect(wv(cl.col_low_v), ow);
                assert_eq!(cl.row_pt_n, row_w.len());
                assert_eq!(cl.col_pt_n, col_w.len());
                for (j, &w) in row_w.iter().enumerate() {
                    sb.connect(wv(cl.row_pt_v + j), w);
                }
                for (j, &w) in col_w.iter().enumerate() {
                    sb.connect(wv(cl.col_pt_v + j), w);
                }
                sb.connect(wv(cl.value_v), *value_w);
            }
        }

        // Publishes: per fold, the accumulator claim [live | rho_col |
        // rho_row | value] (endpoint identities are copy constraints,
        // nothing published). This is the
        // ENVELOPE-registry surface a parent inherits, so under the
        // envelope it rides the reserved ACC_MAIN block at a constant
        // index; off-envelope it publishes inline, as before.
        // THE SPINE LAYOUT: the registry-keyed matrix entries, then the
        // sigma SLOTS, then the jagged SLOTS — `N_KEY_SLOTS` of each,
        // whatever this node's fold actually had, each leading with the
        // KEY it is about. A fresh-only node has one live slot per family
        // and publishes the other DEAD (all zeros), which decodes as the
        // zero claim, so a base node and a steady node are read at the
        // same offsets by one parent circuit.
        let key_pay = payload_words(&stream);
        let key_wires = |p: usize| -> [Wire; 2] {
            [
                ww[key_pay[p][0]].expect("key payload wired"),
                ww[key_pay[p][1]].expect("key payload wired"),
            ]
        };
        let mut acc_main_w: Vec<Wire> = Vec::new();
        let push_entry = |w: &mut Vec<Wire>,
                          key: Option<[Wire; 2]>,
                          fp: Option<&FoldPub>,
                          k_col: usize,
                          k_row: usize| {
            if let Some(k) = key {
                w.extend_from_slice(&k);
            }
            match fp {
                Some(fp) => {
                    w.push(fp.live);
                    w.extend_from_slice(&fp.rho_col);
                    w.extend_from_slice(&fp.rho_row);
                    w.push(fp.value);
                }
                None => w.extend(repeat_n(zw, 2 + k_col + k_row)),
            }
        };
        for fp in fold_pubs.iter().take(n_uni) {
            push_entry(&mut acc_main_w, None, Some(fp), 0, 0);
        }
        for j in 0..N_KEY_SLOTS {
            let live = (j < n_keys).then(|| (key_wires(sigma_payloads[j]), &fold_pubs[n_uni + j]));
            push_entry(
                &mut acc_main_w,
                Some(live.map(|(k, _)| k).unwrap_or([zw, zw])),
                live.map(|(_, fp)| fp),
                locs[n_uni].k_col,
                locs[n_uni].k_row,
            );
        }
        for j in 0..N_KEY_SLOTS {
            let live = (j < n_keys).then(|| (key_wires(jagged_payloads[j]), &jfold_pubs[j]));
            push_entry(
                &mut acc_main_w,
                Some(live.map(|(k, _)| k).unwrap_or([zw, zw])),
                live.map(|(_, fp)| fp),
                jlocs[0].n_col,
                jlocs[0].k_row,
            );
        }
        // THE PASSENGER: `out = child's passenger + h · (the orphaned
        // entry)`, word for word. `h` is live for exactly one node of a
        // spine (the first steady one over a base), and there the child's
        // own passenger is empty — so the sum is a SELECT that can never
        // silently drop a live claim: two live terms garble each other and
        // the root's discharge rejects, which is the safe direction.
        let pass_w: Vec<Wire> = match &spine_w {
            Some((_, sig_off, jag_off, sig, jag)) => {
                let e = &env;
                let rk = &regions[spine.as_ref().unwrap().node_child];
                let pb = env_pass_base(e);
                let mut out = Vec::with_capacity(sig_ent + jag_ent);
                for (base, o, h, width) in [
                    (pb, sig_off[1], sig[1].2, sig_ent),
                    (pb + sig_ent, jag_off[1], jag[1].2, jag_ent),
                ] {
                    for w in 0..width {
                        out.push(
                            sb.gate(
                                cs.fold_macs,
                                &[rk.child_pub_w[base + w], h, rk.child_pub_w[o + w]],
                            )[0],
                        );
                    }
                }
                out
            }
            _ => Vec::new(),
        };
        let fold_pub_base = env_acc_main_base(&env);
        // ---- the APPLICATION STATEMENT (hash-chain adjacency) ----
        // When the children carry an app block: left.h_end == right.h_start
        // as four copy constraints (both children's publics are witness
        // wires here), and the combined span publishes as THIS node's block.
        // The adjacency connects happen here; the PUBLISH of the combined
        // span moves to the envelope's fixed tail block (below, with the
        // padding) so its offset is level-independent. Off-envelope it
        // publishes inline, exactly as before.
        // ADJACENCY CHAINS ACROSS EVERY CONSECUTIVE PAIR: child i's h_end
        // is child i+1's h_start, so the node's own span is the first
        // child's h_start and the last child's h_end — the same statement
        // whatever the arity.
        let app_w: Option<Vec<Wire>> = app_stmt.map(|off| {
            for w in regions.windows(2) {
                for j in 0..4 {
                    sb.connect(w[0].child_pub_w[off + 4 + j], w[1].child_pub_w[off + j]);
                }
            }
            let last = &regions[n_kids - 1];
            (0..4)
                .map(|j| regions[0].child_pub_w[off + j])
                .chain((0..4).map(|j| last.child_pub_w[off + 4 + j]))
                .collect()
        });
        // The publish of the combined span rides the envelope's fixed tail
        // block (below, with the padding), never inline.
        let app_inline: Option<usize> = None;
        // ---- the LANE fold region, in-circuit: priors-only, every prior
        // surface WIRED to the child's published accumulator claim (a
        // prior's surface IS what the child published — the child_pub_w
        // words at claims_base, layout [rho_col | rho_row | value] per
        // group), lows to the constant 1. Its own chain block rides the
        // shared b3 slot; the fold rows the shared mac/mrs/prefix slots.
        let lane_pub = lane_native.as_ref().map(|ln2| {
            let (_, llocs, ljlocs, lstream, lbytes, lops, lchals, lvals) = ln2;
            let lane_ref = lane.as_ref().expect("lane native implies lane");
            // Use the protocol tracer rather than a manual finalize loop:
            // Secure fold tapes contain fused `Pow`+squeeze operations whose
            // compression counter differs from an ordinary squeeze.
            let ltrace = trace_duplex(lstream, lbytes, lops);
            assert_chain_replays(lops, &ltrace, lchals);
            let lpub_payloads = bytes_payload_mask(lops);
            let (lchain_outs, lww) = emit_fs_chain(
                &mut sb,
                cs.q.b3,
                iv2,
                &ltrace,
                lstream,
                lbytes,
                &mut vals,
                &mut consts,
                &lpub_payloads,
                &[],
            );
            emit_recorded_pow_checks(
                &mut sb,
                cs.q.b3,
                cs.q.pow,
                iv2,
                lops,
                &ltrace,
                lstream,
                &lchain_outs,
                &lww,
                &mut vals,
                &mut consts,
            );
            let mut lvmap: Vec<Option<usize>> = Vec::new();
            for (wi, w) in lstream.words.iter().enumerate() {
                if let StreamWord::Value(vi) = *w {
                    if lvmap.len() <= vi {
                        lvmap.resize(vi + 1, None);
                    }
                    lvmap[vi] = Some(wi);
                }
            }
            let lwv =
                |vi: usize| -> Wire { lww[lvmap[vi].expect("lane word")].expect("lane wired") };
            let (lfold_pubs, lalpha_recs) = emit_fold_region(
                &mut sb,
                cs.fold_macs,
                cs.mrs,
                pfslot,
                pf_w,
                leslot,
                llocs,
                &ltrace,
                &challenge_word_locs(lops),
                &lchain_outs,
                &lww,
                &lvmap,
                lchals,
                lvals,
                &mut vals,
                zw,
                ow,
                false, // the jagged group follows on the lane tape
            );
            let ljfold_pubs = emit_jagged_fold_region(
                &mut sb,
                cs.fold_macs,
                cs.mrs,
                pfslot,
                pf_w,
                ljlocs,
                &ltrace,
                &challenge_word_locs(lops),
                &lchain_outs,
                &lww,
                &lvmap,
                lvals,
                &mut vals,
                zw,
                ow,
            );
            for (k, rk) in regions.iter().enumerate() {
                let mut off = lane_ref.claims_base;
                for loc in llocs {
                    let cl = &loc.claims[k];
                    // [live | rho_col | rho_row | value]: the LOWS connect
                    // to the child's LIVE word (the zero-claim scale) —
                    // a real entry carries 1, an absent one decodes zero.
                    sb.connect(lwv(cl.row_low_v), rk.child_pub_w[off]);
                    sb.connect(lwv(cl.col_low_v), rk.child_pub_w[off]);
                    for j in 0..cl.col_pt_n {
                        sb.connect(lwv(cl.col_pt_v + j), rk.child_pub_w[off + 1 + j]);
                    }
                    for j in 0..cl.row_pt_n {
                        sb.connect(
                            lwv(cl.row_pt_v + j),
                            rk.child_pub_w[off + 1 + loc.k_col + j],
                        );
                    }
                    sb.connect(
                        lwv(cl.value_v),
                        rk.child_pub_w[off + 1 + loc.k_col + loc.k_row],
                    );
                    off += loc.k_col + loc.k_row + 2;
                }
                // The inherited JAGGED prior: child k's published entry —
                // the block right after the uniform groups in its
                // ACC_CHAIN layout — connects to the lane's absorbed claim
                // surfaces wire-to-wire, exactly like the groups above.
                for loc in ljlocs {
                    let cl = &loc.claims[k];
                    assert!(cl.terms.is_empty(), "inherited jagged claims are plain eq");
                    // The Eq-SCALE: an inherited jagged claim's scale IS the
                    // child's live word, exactly as the uniform groups' lows
                    // are — the zero-claim gate, in the wiring.
                    sb.connect(lwv(cl.row_scale_v), rk.child_pub_w[off]);
                    for j in 0..loc.n_col {
                        sb.connect(lwv(cl.col_v + j), rk.child_pub_w[off + 1 + j]);
                    }
                    for j in 0..cl.row_pt.1 {
                        sb.connect(
                            lwv(cl.row_pt.0 + j),
                            rk.child_pub_w[off + 1 + loc.n_col + j],
                        );
                    }
                    sb.connect(
                        lwv(cl.val_v),
                        rk.child_pub_w[off + 1 + loc.n_col + loc.k_row],
                    );
                    off += loc.n_col + loc.k_row + 2;
                }
            }
            // The lane's structural words (claim tags + the shape header)
            // pin to shared constant publics, like the main fold's — the
            // identities themselves are wire-bound above.
            let mut lane_const_rec: Vec<(F128, usize)> = Vec::new();
            {
                let mut jc: Vec<(F128, Wire)> = Vec::new();
                let mut cw_j = |sb: &mut ShapeBuilder,
                                vals: &mut Vec<F128>,
                                rec2: &mut Vec<(F128, usize)>,
                                v: F128|
                 -> Wire {
                    if let Some(&(_, w)) = jc.iter().find(|&&(x, _)| x == v) {
                        return w;
                    }
                    vals.push(v);
                    rec2.push((v, sb.public_len()));
                    let w = sb.public_input();
                    jc.push((v, w));
                    w
                };
                for loc in ljlocs {
                    for cl in &loc.claims {
                        let tag = cw_j(
                            &mut sb,
                            &mut vals,
                            &mut lane_const_rec,
                            F128::new(0, cl.row_pt.1 as u64),
                        );
                        sb.connect(lwv(cl.row_scale_v - 1), tag);
                    }
                    let header_v = loc.hdr_v;
                    let hw = cw_j(
                        &mut sb,
                        &mut vals,
                        &mut lane_const_rec,
                        F128::new(loc.k_row as u64, loc.claims.len() as u64),
                    );
                    sb.connect(lwv(header_v), hw);
                }
            }
            // The lane's claims are the LOWER-registry surface a parent
            // inherits: under the envelope they ride the reserved
            // ACC_CHAIN block — the same constant index at which an FL
            // child exposes its own chain fold.
            let mut lane_w: Vec<Wire> = Vec::new();
            for fp in lfold_pubs.iter().chain(&ljfold_pubs) {
                lane_w.push(fp.live);
                lane_w.extend_from_slice(&fp.rho_col);
                lane_w.extend_from_slice(&fp.rho_row);
                lane_w.push(fp.value);
            }
            let lane_words = lane_w.len();
            let lane_pub_base = env_acc_chain_base(&env);
            (
                lane_pub_base,
                lane_words,
                lalpha_recs,
                lane_w,
                lane_const_rec,
            )
        });

        if var("MAC_CENSUS").is_ok() {
            let mac_total = sb.rows_in_slot(cs.macs);
            println!("\nMAC ROW CENSUS (shared mac slot; child 0 labels, child 1 same shape):");
            for w in r0.census.windows(2) {
                if w[1].2 != w[0].2 {
                    println!("  {:42} {:6}", w[1].0, w[1].2 - w[0].2);
                }
            }
            let mut prev = mac_c0_start;
            for (i, &mk) in mac_marks.iter().enumerate() {
                println!("  {:42} {:6}", format!("= child {i} region"), mk - prev);
                prev = mk;
            }
            println!("  {:42} {:6}", "fold region", mac_after_fold - prev);
            println!(
                "  {:42} {:6}",
                "lagrange lows + tail",
                mac_total - mac_after_fold
            );
            println!("  {:42} {:6}", "TOTAL", mac_total);
        }
        if var("B3_CENSUS").is_ok() {
            eprintln!(
                "  [node emitted BLAKE rows] primary {} | secondary {} | model max {}",
                sb.rows_in_slot(cs.q.b3),
                cs.q.b3_alt.map(|slot| sb.rows_in_slot(slot)).unwrap_or(0),
                b3_rows,
            );
        }
        if var("PUB_CENSUS").is_ok() {
            println!("\nPUBLICS CENSUS (child 0; child 1 same shape):");
            for w in r0.census.windows(2) {
                println!("  {:38} {:6}", w[1].0, w[1].1 - w[0].1);
            }
            let child = r0.census.last().unwrap().1 - r0.census[0].1;
            println!("  {:38} {:6}", "= CHILD TOTAL", child);
            let tail_len: usize = locs.iter().map(|l| 2 + l.k_col + l.k_row).sum();
            println!("  {:38} {:6}", "lagrange consts", 66usize);
            println!("  {:38} {:6}", "fold region publics", tail_len);
            println!(
                "  {:38} {:6}",
                "TOTAL (2 children + shared)",
                sb.public_len()
            );
        }
        let build_ms = t_tapes.elapsed().as_secs_f64() * 1e3 - tape_setup_ms;
        let t_build2 = Instant::now();
        // publics*: the node pads to the same public-segment length the
        // leaf does (free counts: the count VECTORS deliberately differ —
        // see the assert_ne below the builders).
        let prepad_publics2 = sb.public_len();
        let app_base = {
            let _ = app_inline;
            let empty: Vec<Wire> = Vec::new();
            pad_envelope_counts(
                &mut sb,
                &cs.q,
                &cs.env_cache(),
                &env,
                zw,
                &mut hints,
                &mut vals,
                &mut consts,
                &EnvTail {
                    acc_main: &acc_main_w,
                    acc_chain: lane_pub.as_ref().map(|(_, _, _, w, _)| w).unwrap_or(&empty),
                    pass: &pass_w,
                    app: app_w.as_deref().unwrap_or(&empty),
                },
            );
            app_w.as_ref().map(|_| env_app_base(&env))
        };
        let shape2 = sb.finish().expect("the 2->1 node circuit builds");
        // The two-limb Ligerito verifier plus the split BLAKE table stays
        // below 512 cell slots, which pins the optimized mu=23 boundary.
        assert!(
            shape2.circuit.cells().slots().len() <= 512,
            "the F256 node's cell-slot budget regressed ({} slots)",
            shape2.circuit.cells().slots().len()
        );
        // ROUND-3 DATA (NODE_CENSUS=1): per-type schema words — each one a
        // cell slot AND a gather claim — plus live rows and utilization,
        // the consolidation pass's worklist.
        if var("NODE_CENSUS").is_ok() {
            let mut lab: Vec<(usize, String)> = vec![
                (shape2.registry_slot(cs.q.b3), "b3".to_string()),
                (shape2.registry_slot(cs.q.swap), "swap".to_string()),
                (shape2.registry_slot(cs.q.spread), "spread".to_string()),
                (
                    shape2.registry_slot(cs.q.family.expect("family-H slot")),
                    "family-h".to_string(),
                ),
                (shape2.registry_slot(cs.macs), "mac".to_string()),
                (shape2.registry_slot(cs.fold_macs), "fold-mac".to_string()),
                (shape2.registry_slot(cs.zcr), "zcr".to_string()),
                (shape2.registry_slot(cs.mrs), "mrs".to_string()),
                (shape2.registry_slot(cs.spine), "spine".to_string()),
                (shape2.registry_slot(cs.spine256), "spine256".to_string()),
                (shape2.registry_slot(cs.alslot), "assist".to_string()),
            ];
            if let Some(slot) = cs.q.b3_alt {
                lab.push((shape2.registry_slot(slot), "b3b".to_string()));
            }
            for &(n, s) in &cs.le {
                lab.push((shape2.registry_slot(s), format!("le{n}")));
            }
            for &(k, s) in &cs.resid {
                // Decode the cache's key scheme (see ChildSlots::resid);
                // the shared mac (600) is already labeled above.
                let name = match k {
                    600 => continue,
                    k if k >= 310 => format!("pf{}", k - 310),
                    k => format!("resid{}", k - 100),
                };
                lab.push((shape2.registry_slot(s), name));
            }
            println!("  NODE TYPE CENSUS (io = cell slots = gather claims):");
            let (mut area_b, mut area_e) = (0usize, 0usize);
            for (t, ty) in shape2.registry.types().iter().enumerate() {
                let name = lab
                    .iter()
                    .find(|(i, _)| *i == t)
                    .map(|(_, s)| s.as_str())
                    .unwrap_or("?");
                // Mirrors UnionInstance::used_cols: word-columns that carry
                // data (a boolean type's GF(2) columns bit-pack 128/word; an
                // element type's useful_bits is element_cols * 128).
                let used_cols = ty.useful_bits.div_ceil(128).min(1usize << (ty.k_log - 7));
                let area = shape2.counts[t] * used_cols;
                let native = match ty.class {
                    TableClass::Boolean => {
                        area_b += area;
                        format!("GF(2)     {:6} bit-cols", ty.useful_bits)
                    }
                    _ => {
                        area_e += area;
                        format!("GF(2^128) {:6} el-cols ", ty.useful_bits / 128)
                    }
                };
                println!(
                    "    t{t:2} {name:>8} | {native} = {used_cols:3} word-cols | io {:3} | \
                     rows {:6} ({:3}%) | area {area:9} words",
                    ty.io_schema.len(),
                    shape2.counts[t],
                    (100 * shape2.counts[t]) >> nu2,
                );
            }
            println!(
                "    class areas: GF(2) {area_b} + GF(2^128) {area_e} = dense {} words",
                area_b + area_e,
            );
        }
        let hint_refs: Vec<&(dyn Any + Sync)> =
            hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
        let build_ms = build_ms + t_build2.elapsed().as_secs_f64() * 1e3;
        // THE INDEX-FILL RUNNER (setup): compile the fill plan, then pin it
        // row-identical against the generic walk before the online loop
        // trusts it — publics, every boolean row store, and every
        // element slot's packed witness, field for field. The walk stays the
        // differential oracle; only the plan runs in the timed loop.
        let t_plan = Instant::now();
        let fill_plan = shape2.fill_plan();
        let build_ms = build_ms + t_plan.elapsed().as_secs_f64() * 1e3;
        {
            let walk = shape2.run(&vals, &hint_refs);
            let fill = shape2.run_filled(&fill_plan, &vals, &hint_refs);
            assert_eq!(walk.public, fill.public, "fill plan: public segment");
            assert_eq!(walk.witnesses, fill.witnesses, "fill plan: slot witnesses");
            assert_eq!(
                walk.rows::<Blake3Gate>(cs.q.b3),
                fill.rows::<Blake3Gate>(cs.q.b3),
                "fill plan: b3 rows"
            );
            if let Some(slot) = cs.q.b3_alt {
                assert_eq!(
                    walk.rows::<Blake3Gate>(slot),
                    fill.rows::<Blake3Gate>(slot),
                    "fill plan: second b3 rows"
                );
            }
            assert_eq!(
                walk.rows::<SwapGate>(cs.q.swap),
                fill.rows::<SwapGate>(cs.q.swap),
                "fill plan: swap rows"
            );
            assert_eq!(
                walk.rows::<BitSpreadGate>(cs.q.spread),
                fill.rows::<BitSpreadGate>(cs.q.spread),
                "fill plan: spread rows"
            );
            assert_eq!(
                walk.rows::<PowMaskGate>(cs.q.pow),
                fill.rows::<PowMaskGate>(cs.q.pow),
                "fill plan: pow rows"
            );
            let family_slot = cs.q.family.expect("family-H slot");
            assert_eq!(
                walk.rows::<FamilyTransposeTileGate>(family_slot),
                fill.rows::<FamilyTransposeTileGate>(family_slot),
                "fill plan: family-H rows"
            );
        }
        // The node proves and verifies over the circuit path. Union, PCS
        // params and the R1CS tables are per-SHAPE — offline, ahead of the
        // online loop.
        let union2 = outer_union(&shape2.registry, shape2.counts.clone());
        let pf = cfg.outer_profile();
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: pf.log_inv_rate(),
            log_batch_size: pcs_batch_for(&union2, pf),
            profile: pf,
            num_lanes: outer_lanes(&union2, pcs_batch_for(&union2, pf)),
            // BLAKE3 for BOTH Merkle and FS: the node's proof must be
            // RECURSABLE — a parent replays this transcript in-circuit,
            // and each default diverges silently (the two recorded
            // gotchas, third occurrence).
            merkle_hash: HashKind::Blake3,
        };
        let t_r1cs = Instant::now();
        let b3_r1cs2 = build_block_r1cs(nu2);
        let b3_lc2 = b3_r1cs2.csc_lincheck_circuit();
        let swap_r1cs2 = SwapTable::build_block_r1cs(nu2);
        let swap_lc2 = swap_r1cs2.csc_lincheck_circuit();
        let spread_r1cs2 = BitSpreadTable::new(spread_w2).build_block_r1cs(nu2);
        let spread_lc2 = spread_r1cs2.csc_lincheck_circuit();
        let pow_r1cs2 = PowMaskTable.build_block_r1cs(nu2);
        let pow_lc2 = pow_r1cs2.csc_lincheck_circuit();
        let family_slot = cs.q.family.expect("family-H slot");
        let family_r1cs2 = FamilyTransposeTileTable::build_block_r1cs(nu2);
        let family_lc2 = family_r1cs2.csc_lincheck_circuit();
        let build_ms = build_ms + t_r1cs.elapsed().as_secs_f64() * 1e3;
        // TOWER_STEADY=N (or the bench's STEADY_OVERRIDE) re-runs the ONLINE
        // phases (tapes + trace + asm + prove + verify) N extra times over
        // the SAME built shape: the offline setup (circuit, R1CS, PCS
        // params, warmed pools) is paid once, so iterations after the first
        // give the steady-state online cost. Every iteration's record lands
        // in `onlines` (NodeOut carries them; `online` stays the last).
        let mut steady_left = steady_reps();
        let mut onlines: Vec<Online> = Vec::with_capacity(steady_left + 1);
        let (built2, oproof, ocommit, block_pub, tapes_ms, trace_ms, asm_ms, prove_ms, verify_ms) = loop {
            // Tapes are statement work: re-run them each online iteration
            // (results discarded — identical by determinism) so the printed
            // number is the steady-state cost, not the first-touch one.
            // The ONLINE tape cost: two recorded deferred child verifies (the
            // production statement work). The pin/locate scaffolding ran once
            // above (tape_setup_ms) — its indices are shape-stable.
            let tapes_ms = {
                let t = Instant::now();
                {
                    los.par_iter()
                        .for_each(|lo| record_child_verify(lo, DOMAIN));
                }
                t.elapsed().as_secs_f64() * 1e3
            };
            let t_trace = Instant::now();
            // DEFERRED: rows and publics only — the element witnesses are never
            // packed; the assembly below feeds the prover from the rows.
            let mut built2 = shape2.run_filled_deferred(&fill_plan, &vals, &hint_refs);
            let trace_ms = t_trace.elapsed().as_secs_f64() * 1e3;

            // The two child regions' checker walks — each child's whole
            // deferred-verifier statement held against its own replicas.
            let consumed: Vec<usize> = rts
                .iter()
                .zip(&regions)
                .map(|(rt, r)| check_real_child_region(&built2.public, rt, r))
                .collect();
            for i in 0..n_kids {
                let end = regions[i].pub_base + consumed[i];
                let next = if i + 1 < n_kids {
                    regions[i + 1].pub_base
                } else {
                    fold_pub_base
                };
                assert!(
                    end <= next,
                    "child {i}'s public block overruns the next region"
                );
            }
            // The fold checker + the accumulator, reassembled from publics —
            // and THE BLOCK, which is the accumulator plus the dead slots and
            // the passenger: what a spine parent reads.
            let (rebuilt, sig_keys, mut p_at) =
                check_fold_publics(&built2.public, fold_pub_base, &locs, &alpha_recs, n_uni);
            let mut sigma_slots: Vec<([F128; 2], MatrixClaim)> = sig_keys
                .iter()
                .zip(&rebuilt[n_uni..])
                .map(|(k, c)| (*k, c.clone()))
                .collect();
            for _ in n_keys..N_KEY_SLOTS {
                sigma_slots.push(read_acc_entry(
                    &built2.public,
                    &mut p_at,
                    true,
                    locs[n_uni].k_col,
                    locs[n_uni].k_row,
                ));
            }
            let (jrebuilt, jag_keys, mut p_at) =
                check_jagged_fold_publics(&built2.public, p_at, &jlocs, true);
            let mut jagged_slots: Vec<([F128; 2], MatrixClaim)> = jag_keys
                .iter()
                .zip(&jrebuilt)
                .map(|(k, c)| (*k, c.clone()))
                .collect();
            for _ in n_keys..N_KEY_SLOTS {
                jagged_slots.push(read_acc_entry(
                    &built2.public,
                    &mut p_at,
                    true,
                    jlocs[0].n_col,
                    jlocs[0].k_row,
                ));
            }
            for j in 0..n_keys {
                assert_eq!(
                    jrebuilt[j], jouts[j],
                    "published jagged slot {j} == located native"
                );
                assert_eq!(
                    sigma_slots[j].0,
                    digest_f128(&key_digests[j]),
                    "the published sigma key names child {j}'s circuit"
                );
                assert_eq!(
                    jagged_slots[j].0,
                    digest_f128(&key_digests[j]),
                    "the published jagged key names child {j}'s layout"
                );
            }
            let tail_len = p_at - fold_pub_base;
            let passenger: Vec<([F128; 2], MatrixClaim)> = {
                let mut q = env_pass_base(&env);
                vec![
                    read_acc_entry(
                        &built2.public,
                        &mut q,
                        true,
                        locs[n_uni].k_col,
                        locs[n_uni].k_row,
                    ),
                    read_acc_entry(&built2.public, &mut q, true, jlocs[0].n_col, jlocs[0].k_row),
                ]
            };
            let acc_pub = Accumulator {
                registry_digest: registry.digest(),
                per_type: (0..n_bool)
                    .map(|t| (rebuilt[2 * t].clone(), rebuilt[2 * t + 1].clone()))
                    .collect(),
                per_element: (0..n_el)
                    .map(|t| {
                        (
                            rebuilt[2 * n_bool + 2 * t].clone(),
                            rebuilt[2 * n_bool + 2 * t + 1].clone(),
                        )
                    })
                    .collect(),
                sigma: (0..n_keys)
                    .map(|j| (key_digests[j], rebuilt[n_uni + j].clone()))
                    .collect(),
                jagged: (0..n_keys)
                    .map(|j| (key_digests[j], jrebuilt[j].clone()))
                    .collect(),
            };
            assert_eq!(
                acc_pub, acc_v,
                "the Accumulator, reassembled from the public segment alone"
            );
            assert!(
                acc_pub.discharge(&mats)
                    && acc_pub.discharge_element(&el_mats)
                    && acc_pub.discharge_sigma(&key_circuits)
                    && acc_pub.discharge_jagged(&jag_tables),
                "the public-segment accumulator discharges all four groups"
            );
            // THE PASSENGER, natively: the child's own, unless this node is
            // the one whose node slot could not fold — then the orphan itself.
            // The two are never both live in a spine, and the in-circuit form
            // is their SUM, so this select and that sum agree.
            if !passenger.is_empty() {
                let dead = |k_col: usize, k_row: usize| {
                    (
                        [F128::ZERO; 2],
                        MatrixClaim {
                            row: Weight::low_eq(vec![F128::ZERO], vec![F128::ZERO; k_row]),
                            col: Weight::low_eq(vec![F128::ZERO], vec![F128::ZERO; k_col]),
                            value: F128::ZERO,
                        },
                    )
                };
                let want: Vec<([F128; 2], MatrixClaim)> = match &spine {
                    None => vec![
                        dead(locs[n_uni].k_col, locs[n_uni].k_row),
                        dead(jlocs[0].n_col, jlocs[0].k_row),
                    ],
                    Some(sp) => {
                        let slot1 = digest_f128(&key_digests[1]);
                        [&sp.prior.sigma[1], &sp.prior.jagged[1]]
                            .iter()
                            .enumerate()
                            .map(|(t, ent)| {
                                let carried = sp.prior.passenger[t].clone();
                                if ent.0 != slot1 && entry_live(&ent.1) {
                                    assert!(
                                        !entry_live(&carried.1),
                                        "a spine orphans ONCE: the passenger was already full"
                                    );
                                    (*ent).clone()
                                } else {
                                    carried
                                }
                            })
                            .collect()
                    }
                };
                assert_eq!(
                    passenger, want,
                    "the published passenger is the child's, plus this node's orphan"
                );
            }
            let block_pub = MainBlock {
                per_type: acc_pub.per_type.clone(),
                per_element: acc_pub.per_element.clone(),
                sigma: sigma_slots,
                jagged: jagged_slots,
                passenger,
            };
            // The lagrange-low constants: the one public surface the in-circuit
            // derivation adds — validated against the verifier's own values.
            {
                if let Some((lam_base, _, _, _)) = &rs_lam {
                    for (i, &v) in PHI_8_TABLE[..1 << K_SKIP].iter().enumerate() {
                        assert_eq!(built2.public[*lam_base + i], v, "λ const {i}");
                    }
                    assert_eq!(
                        built2.public[*lam_base + (1 << K_SKIP)],
                        subspace_denominator_pair(K_SKIP).1,
                        "the subspace denominator inverse const"
                    );
                    assert_eq!(
                        built2.public[*lam_base + (1 << K_SKIP) + 1],
                        F128::ZERO,
                        "the lows' assert-zero anchor"
                    );
                }
                // The AG-skip blocks: the decode is in-circuit since
                // phase D; the checker holds the nonce range and the lows.
                for (lo_c, base) in los.iter().zip(&ag_pub_bases) {
                    if let Some(base) = base {
                        check_ag_skip_publics(
                            &built2.public,
                            *base,
                            lo_c.pcs.zerocheck_grinding().ag_r1_bits(),
                        );
                    }
                }
                for &(v, idx) in &jag_const_rec {
                    assert_eq!(built2.public[idx], v, "jagged shared constant public");
                }
                // UNDER the envelope the publish blocks live on the reserved
                // tail, so the body simply has to fit — which
                // `pad_envelope_counts` asserts — and the tail layout is
                // checked where it matters, by rebuilding both accumulators
                // at their CONSTANT bases below.
                let _ = (tail_len, prepad_publics2);
                // The LANE accumulator, reassembled from the public segment
                // alone — the parent-facing statement of the lower registry.
                if let (Some((lpb, _, lar, _, lrec)), Some((lacc_n, llocs, ljlocs, ..))) =
                    (lane_pub.as_ref(), lane_native.as_ref())
                {
                    let (lrebuilt, _, _) =
                        check_fold_publics(&built2.public, *lpb, llocs, lar, llocs.len());
                    let lu_len: usize = llocs.iter().map(|l| 2 + l.k_col + l.k_row).sum();
                    let (ljrebuilt, _, _) =
                        check_jagged_fold_publics(&built2.public, *lpb + lu_len, ljlocs, false);
                    let lane_ref = lane.as_ref().expect("lane");
                    let lacc_pub2 = Accumulator {
                        registry_digest: lane_ref.registry.digest(),
                        per_type: vec![(lrebuilt[0].clone(), lrebuilt[1].clone())],
                        per_element: Vec::new(),
                        sigma: vec![(lane_ref.circuit.digest(), lrebuilt[2].clone())],
                        jagged: vec![(lane_ref.circuit.digest(), ljrebuilt[0].clone())],
                    };
                    assert_eq!(
                        &lacc_pub2, lacc_n,
                        "the LANE accumulator, reassembled from publics alone"
                    );
                    for &(v, idx) in lrec {
                        assert_eq!(built2.public[idx], v, "lane jagged constant public");
                    }
                }
            }

            let t_asm = Instant::now();
            // Recreated per online iteration — the spread closure consumes it.
            let spread_ty2 = BitSpreadTable::new(spread_w2);
            let pow_ty2 = PowMaskTable;
            // The copy-free assembly path: the boolean drivers pack straight
            // into the union slot blocks inside the prove (live rows only under
            // elide) — no intermediate capacity-sized buffers, no memcpy. The
            // rows are hoisted to owned Vecs because the closures must be Send
            // and `built2.rows` hands out `dyn Any`-backed borrows.
            let b3_declared: Vec<_> = once(cs.q.b3).chain(cs.q.b3_alt).collect();
            let b3_rows2: Vec<_> = b3_declared
                .iter()
                .map(|&slot| (slot, built2.rows::<Blake3Gate>(slot).to_vec()))
                .collect();
            let swap_rows2 = built2.rows::<SwapGate>(cs.q.swap).to_vec();
            let spread_rows2 = built2.rows::<BitSpreadGate>(cs.q.spread).to_vec();
            let pow_rows2 = built2.rows::<PowMaskGate>(cs.q.pow).to_vec();
            let family_rows2 = built2.rows::<FamilyTransposeTileGate>(family_slot).to_vec();
            let mut bslots: Vec<(usize, UnionSlotProverInput)> = vec![
                (
                    shape2.registry_slot(cs.q.swap),
                    UnionSlotProverInput::in_place(
                        move |dst| SwapTable::generate_witness_batch_major_into(&swap_rows2, dst),
                        swap_lc2,
                    ),
                ),
                (
                    shape2.registry_slot(cs.q.spread),
                    UnionSlotProverInput::in_place(
                        move |dst| spread_ty2.generate_witness_batch_major_into(&spread_rows2, dst),
                        spread_lc2,
                    ),
                ),
                (
                    shape2.registry_slot(cs.q.pow),
                    UnionSlotProverInput::in_place(
                        move |dst| pow_ty2.generate_witness_batch_major_into(&pow_rows2, dst),
                        pow_lc2,
                    ),
                ),
                (
                    shape2.registry_slot(family_slot),
                    UnionSlotProverInput::in_place(
                        move |dst| {
                            FamilyTransposeTileTable::generate_witness_batch_major_into(
                                &family_rows2,
                                dst,
                            )
                        },
                        family_lc2,
                    ),
                ),
            ];
            bslots.extend(b3_rows2.into_iter().map(|(slot, rows)| {
                (
                    shape2.registry_slot(slot),
                    UnionSlotProverInput::in_place(
                        move |dst| generate_witness_batch_major_partial_into(&rows, nu2, dst),
                        b3_lc2,
                    ),
                )
            }));
            bslots.sort_by_key(|(i, _)| *i);
            // Element inputs straight from the slots' rows: the run was
            // DEFERRED, so the full-capacity packed intermediate never exists —
            // the prove's in_place closure scatters the live rows directly.
            let mut el_ord: Vec<(usize, Vec<Vec<F128>>)> = cs
                .element_slot_ids()
                .into_iter()
                .map(|sl| {
                    (
                        shape2.registry_slot(sl),
                        built2.take_rows_of::<Vec<F128>>(sl),
                    )
                })
                .collect();
            el_ord.sort_by_key(|(i, _)| *i);
            // THE MATCH-GATE FORGERY (the adversarial leg): re-witness the
            // spine gadget's 26 mac rows as the world a cheating prover wants —
            // the advice inverse set to 0 so both is-eq gadgets CLAIM the
            // mismatched digests are equal (z = 1), the gate then folding the
            // orphan LIVE (m = 1, g = live, gv = value) and waving the
            // passenger off (h = 0). Every forged row still satisfies the mac
            // relation (t = x·y, out = acc + t), so the element PIOP holds;
            // what cannot be reconciled are the COPY CONSTRAINTS — chk = z·d =
            // d ≠ 0 sits in the assert-zero anchor's class, and g/gv/h sit in
            // the classes of the fold tape's honest absorbed words (the native
            // fold folded the ZERO claim) and the passenger sum. The wiring
            // product is what must kill it — the same tier the chain-link
            // tamper pinned.
            if forge_match {
                let mac_ri = shape2.registry_slot(cs.macs);
                let rows = &mut el_ord
                    .iter_mut()
                    .find(|(i, _)| *i == mac_ri)
                    .expect("the mac slot's rows")
                    .1;
                let (zero, one) = (F128::ZERO, F128::ONE);
                for blk in 0..2 {
                    // sigma's node-slot gate, then jagged's — 13 rows each:
                    // [d p z chk] x2 digest words, then m, g, gv, nm, h.
                    let s = mac_spine0 + 13 * blk;
                    for w in 0..2 {
                        let b = s + 4 * w;
                        let d = rows[b][4];
                        assert_ne!(d, zero, "the forged slot genuinely mismatches");
                        rows[b + 1] = vec![zero, d, zero, zero, zero];
                        rows[b + 2] = vec![one, zero, one, zero, one];
                        rows[b + 3] = vec![zero, one, d, d, d];
                    }
                    let live = rows[s + 9][1];
                    let val = rows[s + 10][2];
                    assert_eq!(live, one, "the orphaned entry is live");
                    rows[s + 8] = vec![zero, one, one, one, one];
                    rows[s + 9] = vec![zero, live, one, live, live];
                    rows[s + 10] = vec![zero, live, val, live * val, live * val];
                    rows[s + 11] = vec![one, one, one, one, zero];
                    rows[s + 12] = vec![zero, live, zero, zero, zero];
                }
            }
            let el_inputs: Vec<UnionElementSlotInput> = el_ord
                .into_iter()
                .map(|(_, rows)| live_element_input_from_rows(rows, nu2))
                .collect();
            let mut lco: Vec<(usize, &dyn LincheckCircuit)> = vec![
                (shape2.registry_slot(cs.q.swap), swap_lc2),
                (shape2.registry_slot(cs.q.spread), spread_lc2),
                (shape2.registry_slot(cs.q.pow), pow_lc2),
                (shape2.registry_slot(family_slot), family_lc2),
            ];
            lco.extend(
                b3_declared
                    .iter()
                    .map(|&slot| (shape2.registry_slot(slot), b3_lc2 as &dyn LincheckCircuit)),
            );
            lco.sort_by_key(|(i, _)| *i);
            let lcs2: Vec<&dyn LincheckCircuit> = lco.into_iter().map(|(_, c)| c).collect();
            let asm_ms = t_asm.elapsed().as_secs_f64() * 1e3;
            let t0p = Instant::now();
            let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
            let (oproof, ocommit) = if outer_zc_ag() {
                #[cfg(target_arch = "aarch64")]
                {
                    let (p, c, _) = prove_fast_ligerito_union_circuit_ag(
                        &union2,
                        &shape2.circuit,
                        &built2.public,
                        &pcs2,
                        bslots.into_iter().map(|(_, x)| x).collect(),
                        el_inputs,
                        &mut ch2,
                    );
                    (MixedProof::Ag(p), c)
                }
                #[cfg(not(target_arch = "aarch64"))]
                unreachable!("outer_zc_ag() is false off aarch64")
            } else {
                let (p, c, _) = prove_fast_ligerito_union_circuit(
                    &union2,
                    &shape2.circuit,
                    &built2.public,
                    &pcs2,
                    bslots.into_iter().map(|(_, x)| x).collect(),
                    el_inputs,
                    &mut ch2,
                );
                (MixedProof::Rs(p), c)
            };
            let prove_ms = t0p.elapsed().as_secs_f64() * 1e3;
            let t0v = Instant::now();
            let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
            let vres = oproof.verify_circuit(
                &union2,
                &shape2.circuit,
                &built2.public,
                &lcs2,
                &ocommit,
                &pcs2,
                &mut ch2,
            );
            if forge_match {
                // The forged world's rows all satisfy their relations — only
                // the wiring product can object, and it MUST.
                assert!(
                    matches!(
                        vres,
                        Err(FlockVerifyError::Wiring(WiringError::Gkr(
                            ProductGkrError::ProductMismatch
                        )))
                    ),
                    "a forged live fold of a mismatched entry must die on the wiring product"
                );
            } else {
                vres.expect("the 2->1 node verifies");
            }
            let verify_ms = t0v.elapsed().as_secs_f64() * 1e3;
            let deferred_ms = if forge_match {
                0.0
            } else {
                let t0d = Instant::now();
                let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
                oproof
                    .verify_circuit_deferred(
                        &union2,
                        &shape2.circuit,
                        &built2.public,
                        &lcs2,
                        &ocommit,
                        &pcs2,
                        &mut ch2,
                    )
                    .expect("the 2->1 node verifies deferred");
                t0d.elapsed().as_secs_f64() * 1e3
            };
            let b3_live: Vec<usize> = once(cs.q.b3)
                .chain(cs.q.b3_alt)
                .map(|slot| shape2.counts[shape2.registry_slot(slot)])
                .collect();
            let b3_live_total: usize = b3_live.iter().sum();
            println!(
                "\nTHE 2->1 RECURSION NODE (two children + {} folds, ONE proof)\n  \
             children: dense_m {} / mu {}, one circuit, distinct FS points\n  \
             regions: 2x the complete deferred verifier (swap assembly, shared slots)\n         \
             + the fold region; CONNECTED: all points, z_partial lows, sigma fully,\n         \
             and the matrix/element EVAL VALUES to the children's bound advice —\n         \
             lagrange lows DERIVED in-circuit from each child's z_skip wire\n  \
             outer: BLAKE rows {} across {:?} | nu {} | dense_m {} | mu {} \
             (cell slots: {} gate + {} public)\n  \
             PER PROOF (online): child tapes {:.0} + witgen/trace {:.0} + witness asm {:.0} + prove {:.0} \
             = {:.0} ms | verify {:.0} ms (DEFERRED {:.0} ms) | proof {:.1} KiB\n  \
             SETUP: circuit build (per SHAPE, cacheable) {:.0} ms | tape pins+locates (shape-stable) {:.0} ms\n",
                n_folds,
                lo0.pcs.m,
                rts[0].mu_i,
                b3_live_total,
                b3_live,
                nu2,
                union2.dense_m(),
                shape2.circuit.cells().mu(),
                shape2.circuit.cells().num_gate_slots(),
                shape2.circuit.cells().num_public_slots(),
                tapes_ms,
                trace_ms,
                asm_ms,
                prove_ms,
                tapes_ms + trace_ms + asm_ms + prove_ms,
                verify_ms,
                deferred_ms,
                serialize(&oproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
                build_ms,
                tape_setup_ms,
            );
            onlines.push(Online {
                setup_ms: build_ms,
                walk_ms: trace_ms,
                tapes_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms,
                wall_ms: 0.0,
            });
            if steady_left > 0 {
                steady_left -= 1;
                continue;
            }
            break (
                built2, oproof, ocommit, block_pub, tapes_ms, trace_ms, asm_ms, prove_ms, verify_ms,
            );
        };
        let (swap_slot2, spread_slot2, pow_slot2, family_slot2) = (
            shape2.registry_slot(cs.q.swap),
            shape2.registry_slot(cs.q.spread),
            shape2.registry_slot(cs.q.pow),
            shape2.registry_slot(family_slot),
        );
        let b3_slots2 = once(cs.q.b3)
            .chain(cs.q.b3_alt)
            .map(|slot| shape2.registry_slot(slot))
            .collect();
        NodeOut {
            lo: LeafOuter {
                public: built2.public.clone(),
                shape: shape2,
                proof: oproof,
                commitment: ocommit,
                pcs: pcs2,
                b3_r1cs: b3_r1cs2,
                swap_r1cs: swap_r1cs2,
                spread_r1cs: spread_r1cs2,
                pow_r1cs: pow_r1cs2,
                family_r1cs: family_r1cs2,
                b3_slots: b3_slots2,
                swap_slot: swap_slot2,
                spread_slot: spread_slot2,
                pow_slot: pow_slot2,
                family_slot: family_slot2,
            },
            acc: acc_v,
            online: Online {
                setup_ms: build_ms,
                walk_ms: trace_ms,
                tapes_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms,
                wall_ms: 0.0,
            },
            onlines,
            app_base,
            lane_acc: lane_native.map(|(a, ..)| a),
            block: block_pub,
        }
    }
}

/// **The REAL-side tape pin** (walker plan, Phase A0 — the sibling of
/// [`chain_tape_regions_pinned`]): [`RealTape::new`] — the SAME constructor
/// the internal-node machinery instantiates per child — walks one FL
/// leaf-outer's tape. Every pin here holds a STORED field against a
/// reference computed OUTSIDE the constructor, so a parse that stores the
/// wrong value FAILS: the tape verbatim against an independent recorded
/// deferred verify of the same leaf outer, the assertion replicas against
/// that run's exports (with the constructor's own natives — strip sums,
/// mask evals, eps terms, the GKR endpoint — cross-checked against the
/// export fields the verifier computed by its own route), the structural
/// counts against the PROOF OBJECT (ladder levels, yr pairs, ring
/// switches, wiring gathers, frobenius groups), and the PoW schedule
/// against the reference tape's own op stream. The e2e node tests prove
/// the values downstream; this localizes a break to the constructor.
#[test]
#[ignore] // Heavier — builds two chain leaves + one FL node.
pub(super) fn real_tape_regions_pinned() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0007);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);
    let fl = build_fl_node(cfg, &cp0, &cp1);
    let rt = RealTape::new(&fl.lo, DOMAIN);

    // The INDEPENDENT reference run: the same deferred verify the
    // constructor records, re-run here — its recorded tape and exported
    // assertions are references the parse under test never touched.
    let union = outer_union(&fl.lo.shape.registry, fl.lo.shape.counts.clone());
    let lcs = leaf_boolean_lcs(&fl.lo);
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(DOMAIN));
    let (_claims, work, sigma) = fl
        .lo
        .proof
        .verify_circuit_deferred(
            &union,
            &fl.lo.shape.circuit,
            &fl.lo.public,
            &lcs,
            &fl.lo.commitment,
            &fl.lo.pcs,
            &mut rec,
        )
        .expect("the reference deferred verify accepts the leaf outer");

    // The stored tape IS the reference run's tape, word for word.
    assert_eq!(rt.chals, rec.challenges(), "the tape's challenges");
    assert_eq!(rt.vals_rec, rec.values(), "the tape's observed values");

    // The stored assertion replicas ARE the reference run's exports.
    assert_eq!(
        rt.sigma_native, sigma,
        "the sigma reference is the deferred verify's"
    );
    assert_eq!(
        rt.mat_assert,
        work.boolean.expect("boolean matrix work"),
        "the boolean reference is the deferred verify's"
    );
    let el = work.element.expect("element work travels on an outer");
    assert_eq!(
        rt.el_assert, el,
        "the element reference is the deferred verify's"
    );
    assert_eq!(
        rt.jag, work.jagged,
        "the jagged reference is the deferred verify's"
    );

    // The constructor's OWN natives against the exports' fields — two
    // routes to one value: the general strip loop vs the verifier's
    // affine-constant evals, the live-mask evals and the GKR endpoint vs
    // the sigma export, the count-derived eps terms vs the exported
    // boolean pin values.
    assert_eq!(
        rt.a_sum_n, el.a_const_eval,
        "the A strip sum is the exported affine-constant eval"
    );
    assert_eq!(
        rt.b_sum_n, el.b_const_eval,
        "the B strip sum is the exported affine-constant eval"
    );
    assert_eq!(
        rt.gkr.r_pt, sigma.rho,
        "the walked GKR endpoint is the exported sigma point"
    );
    assert_eq!(
        rt.mid_n, sigma.masked_id_value,
        "masked M̂ replays to the exported value"
    );
    assert_eq!(
        rt.live_n, sigma.live_value,
        "livê replays to the exported value"
    );
    assert_eq!(
        rt.eps_n.len(),
        sigma.boolean_pins.len(),
        "one count-derived eps per exported boolean pin"
    );
    assert_eq!(
        rt.betas_b.len(),
        rt.eps_n.len(),
        "one const-pin beta per eps term"
    );
    for (k, (_, _, v)) in sigma.boolean_pins.iter().enumerate() {
        assert_eq!(rt.eps_n[k], *v, "eps {k} is the exported pin value");
    }

    // Structural counts against the PROOF OBJECT, not the parse.
    let open = fl.lo.proof.pcs_open();
    let lig = &open.inner.ligerito;
    assert_eq!(
        rt.levels.len(),
        lig.recursive_caps.len() + 1,
        "one open level per recursive cap plus the base"
    );
    assert_eq!(rt.levels.len(), rt.geo.len(), "one geometry per open level");
    assert_eq!(
        rt.levels.len(),
        rt.lvl_src.len(),
        "one source triple per level"
    );
    assert_eq!(
        rt.levels.len(),
        rt.cap_pays.len(),
        "one public cap payload per level"
    );
    assert_eq!(
        rt.yr_len,
        lig.final_proof.yr.len() / 2,
        "the residual length is the proof's yr pairs"
    );
    assert_eq!(
        rt.rs_recs.len(),
        open.ring_switches.len(),
        "one rs region per proof ring switch"
    );
    assert_eq!(
        rt.rs_gam_fins.len(),
        rt.rs_recs.len(),
        "one rs gamma per rs region"
    );
    assert_eq!(
        rt.pd_pts.len(),
        2 + fl.lo.proof.wiring().gather.len(),
        "pd points = the element (c, lc) pair + the wiring gathers"
    );
    assert_eq!(
        rt.groups_ix.len(),
        open.frobenius.group_values.len(),
        "one scalar group per frobenius group value"
    );
    assert_eq!(
        rt.m_mp2,
        fl.lo.pcs.m - LOG_PACKING,
        "the merged domain spans the dense floor"
    );
    assert_eq!(
        rt.mu_i,
        fl.lo.shape.circuit.cells().mu(),
        "mu is the inner cell space's"
    );

    // The grinding schedule against the reference run's own op stream:
    // the (fin, payload, bits) triples relocated on a tape the
    // constructor never saw.
    let ops = flatten_ops(rec.shape().ops());
    let want_pows: Vec<(usize, usize, u32)> = {
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
    assert!(!want_pows.is_empty(), "the strict profile grinds");
    assert_eq!(
        rt.pows, want_pows,
        "the parsed PoW schedule is the reference tape's, op for op"
    );

    // Reading-value shape asserts: transcript order and dimensionality.
    assert!(
        rt.cap_pays.windows(2).all(|w| w[0] < w[1]),
        "cap payloads appear in transcript order"
    );
    assert!(
        rt.zc_rounds_b.windows(2).all(|w| w[0].0 < w[1].0),
        "boolean zerocheck rounds are transcript-ordered"
    );
    assert!(
        rt.lc_rounds_b.windows(2).all(|w| w[0].1 < w[1].1),
        "boolean lincheck rounds are transcript-ordered"
    );
    assert!(
        rt.pd_pts.windows(2).all(|w| w[0].len() == w[1].len()),
        "one merged-open point dimensionality"
    );

    println!(
        "\nREAL TAPE (FL leaf-outer as inner)\n           inner: dense_m {} | open levels {} | pows {} | yr {} | mu {}\n           zc rounds {} | lc rounds {} | betas {} | pd pts {} x {} | groups {}\n           b3 rows (tape model) {} | L0 lanes {} x {} words | pub slots (child) {}\n",
        union.dense_m(),
        rt.levels.len(),
        rt.pows.len(),
        rt.yr_len,
        rt.mu_i,
        rt.zc_rounds_b.len(),
        rt.lc_rounds_b.len(),
        rt.betas_b.len(),
        rt.pd_pts.len(),
        rt.pd_pts.first().map(|p| p.len()).unwrap_or(0),
        rt.groups_ix.len(),
        rt.b3_rows,
        rt.geo[0].lanes,
        rt.geo[0].row_words,
        rt.n_pub_slots_c,
    );
}

/// **Task 5: THE INTERNAL NODE carries the chain statement.** Four chain
/// segments → two first-level nodes → ONE internal node, built by
/// [`build_node_outer_app`]'s own machinery over the FL [`LeafOuter`]s
/// (RealTape walks an FL tape here for the first time): the FL-level
/// adjacency (fl0.h_end == fl1.h_start) is checked wire-to-wire at the
/// internal level through the children's witness publics, and the combined
/// span publishes as the internal node's own app block == the native
/// H^1024(h_start). Accumulators are per-level at the dev shape — the
/// cross-level threading (chain accs as PRIORS of the internal fold) is
/// task 6.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
pub(super) fn internal_node_over_two_fl_nodes() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0006);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);
    let cp2 = build_chain_proof(cfg, cp1.h_end, n_blocks);
    let cp3 = build_chain_proof(cfg, cp2.h_end, n_blocks);
    let fl0 = build_fl_node(cfg, &cp0, &cp1);
    let fl1 = build_fl_node(cfg, &cp2, &cp3);
    assert_eq!(
        fl0.lo.shape.circuit.digest(),
        fl1.lo.shape.circuit.digest(),
        "one first-level circuit digest — the FL shape is data-independent"
    );
    assert_eq!(fl0.stmt_base, fl1.stmt_base, "one statement offset");
    assert_eq!(fl1.h_start, fl0.h_end, "the FL spans are adjacent");

    let out = build_node_outer_app(cfg, &[&fl0.lo, &fl1.lo], Some(fl0.stmt_base), None, None);
    let (node, acc, app) = (out.lo, out.acc, out.app_base);
    let app = app.expect("the internal node carries the app block");
    for j in 0..4 {
        assert_eq!(
            node.public[app + j],
            pack4(cp0.h_start[4 * j..4 * j + 4].try_into().unwrap()),
            "internal statement: h_start is the whole span's start"
        );
        assert_eq!(
            node.public[app + 4 + j],
            pack4(cp3.h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "internal statement: h_end is the whole span's end"
        );
    }
    assert_eq!(
        cp3.h_end,
        native_chain(&cp0.h_start, 4 * n_blocks),
        "the internal span IS the 1024-step chain"
    );
    // Per-level accumulators at the dev shape: the internal node's own acc
    // keys sigma by the FL circuit digest; the chain-level accs live in the
    // FlNodes. (Task 6 threads them as priors.)
    let (sig_digest, _) = acc.sigma.first().expect("the node accumulated sigma");
    assert_eq!(
        *sig_digest,
        fl0.lo.shape.circuit.digest(),
        "the internal accumulator keys by the FL circuit"
    );
    // **TASK 7b's PIN, amended by the COUNT WIN: an FL node and an
    // internal node are ONE ENVELOPE.** Same registry digest (wall 2),
    // same public-segment length with the app block at the same fixed
    // offset (publics*), same PINNED lane count (lanes*) — so a parent's
    // walk cannot tell an FL child from an internal child. Under FREE
    // COUNTS the declared count vectors deliberately DIFFER: the heights
    // are data now, reaching a parent only as jagged claims, and the
    // parent's circuit never reads them.
    {
        assert_eq!(
            fl0.lo.shape.registry.digest(),
            node.shape.registry.digest(),
            "FL and internal share ONE envelope registry"
        );
        assert_ne!(
            fl0.lo.shape.counts, node.shape.counts,
            "free counts: the FL and internal declare their OWN counts"
        );
        assert_eq!(
            fl0.lo.pcs.num_lanes, node.pcs.num_lanes,
            "ONE lane count (lanes* — pinned, the layout's structural residue)"
        );
        assert_eq!(
            fl0.lo.public.len(),
            node.public.len(),
            "FL and internal share ONE public-segment length"
        );
        assert_eq!(
            app, fl0.stmt_base,
            "the app block sits at the envelope's fixed offset for BOTH child kinds"
        );
        assert_eq!(fl0.lo.pcs.m, node.pcs.m, "one dense floor m*");
    }
    println!(
        "\nINTERNAL NODE over two first-level nodes (app-statement plumbed)\n  \
         span: H^{}(h_start) | internal outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        4 * n_blocks,
        node.shape.circuit.cells().nu(),
        node.shape.circuit.cells().mu(),
        node.public.len(),
        serialize(&node.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// One node's own JAGGED LAYOUT — the table its published claims are
/// about, keyed by its circuit digest. Heights are a shape constant of
/// that circuit, which is why the key names the table.
pub(super) fn node_jagged_params(lo: &LeafOuter) -> JaggedParams {
    let u = outer_union(&lo.shape.registry, lo.shape.counts.clone());
    JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        lo.commitment.params.m - LOG_PACKING,
    )
}
