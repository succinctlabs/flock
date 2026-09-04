use std::{array::from_fn, env::var, slice::from_ref, sync::atomic::Ordering, time::Instant};

use aggregate::{
    Accumulator, JaggedKeyProve, JaggedKeyVerify, prove_aggregate_classes_with_grinding,
    verify_aggregate_classes_with_grinding,
};
use bincode::serialize;
use flock_core::{
    aggregate, element_r1cs::union::ElementAssertion, lincheck::LincheckCircuit,
    matrix_fold::MatrixClaim,
};

use crate::{
    r1cs_hashes::blake3::build_block_r1cs,
    tower::{
        ChainLane, ChainProof, DOMAIN, F128, FlNode, FsChallenger, LeafOuter, Online, RootBundle,
        RootDischargeFailure, SpanBound, SpineIn, Tower, TowerVerifyError, TowerVk, UnionInstance,
        build_chain_proof, build_fl_node, build_node_outer_app, chain_blake_r1cs,
        chain_jagged_params, env_acc_chain_base, env_acc_main_base, env_app_base, env_pass_base,
        envelope::STEADY_OVERRIDE,
        envelope_shape,
        gates_blake3::Rng,
        leaf_boolean_lcs, leaf_boolean_mats, native_chain,
        node::{N_KEY_SLOTS, digest_f128, entry_live},
        node_jagged_params,
        online::{median_total, proof_census_mixed, report_stage},
        outer_union, pack4, test_config, tower_fold_grinding,
        verify::{reassemble, verify_root},
    },
};

/// **WALL 3: THE SPINE CONVERGES.** Eight chain segments → four FLs → a
/// BASE node (two FLs, fresh-only) → node_2 (a fresh FL + the base) →
/// node_3 (a fresh FL + node_2) — and `D(node_2) == D(node_3)`: ONE steady
/// shape from level 3 on, at any depth. That is the completeness wall
/// coming down. What makes it work:
///
/// * every node's MAIN fold inherits its node child's published
///   accumulator as a PRIOR, so nothing is dropped at depth > 2 (the gap
///   this arc opened against);
/// * the keyed groups have one SLOT PER CHILD ROLE — the FL slot and the
///   node slot — so a base node (one live key) and a steady node (two)
///   publish the SAME layout, dead slots being zeros that decode as the
///   zero claim;
/// * the node slot's inherited entry is MATCH-GATED against the key it was
///   published with. It matches at every steady level but one: node_3's
///   node slot inherits node_2's, which is keyed by the BASE circuit. That
///   single orphan is gated to zero in the fold and rides the PASSENGER to
///   the root, where it discharges against the base's own tables.
///
/// The chain LANE rides all three levels unchanged, and the app statement
/// spans the whole chain: the spine grows by prepending a fresh FL, so the
/// fresh child is always the earlier segment.
///
/// Ends with the MATCH-GATE ADVERSARIAL MATRIX: (a) a forged node_3 whose
/// gadget rows claim the mismatched digests match and fold the orphan live
/// — self-satisfying rows, so it must die on exactly the wiring product;
/// (b) a dropped passenger and (c) a forged entry key, both statement
/// tampers the proofs refuse.
#[test]
#[ignore] // Heavy — eight chain proofs and eight outers.
pub(super) fn chain_spine_converges() {
    let cfg = test_config();

    let env = envelope_shape();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_5B1E);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());
    let mut cps = Vec::new();
    let mut h = h0;
    for _ in 0..8 {
        let cp = build_chain_proof(cfg, h, n_blocks);
        h = cp.h_end;
        cps.push(cp);
    }
    let fls: Vec<FlNode> = (0..4)
        .map(|i| build_fl_node(cfg, &cps[2 * i], &cps[2 * i + 1]))
        .collect();
    let app_fl = fls[0].stmt_base;
    assert_eq!(
        app_fl,
        env_app_base(&env),
        "the FL's app block is the envelope's"
    );
    for f in &fls {
        assert_eq!(f.stmt_base, app_fl, "one FL app offset");
        assert_eq!(
            f.lo.shape.circuit.digest(),
            fls[0].lo.shape.circuit.digest(),
            "one FL circuit digest"
        );
    }
    // The lane's chain-side materials, shared by every level.
    let chain_registry = &cps[0].inner.built.shape.registry;
    let blake_r1cs = build_block_r1cs(cps[0].inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let chain_circuit = &cps[0].inner.built.shape.circuit;
    let chain_jp = chain_jagged_params(&cps[0]);
    let acc_base = fls[0].fold_pub_base;
    assert_eq!(
        acc_base,
        env_acc_chain_base(&env),
        "the FL's ACC_CHAIN block"
    );

    // THE BASE: fresh-only over the LAST two FLs. The spine grows by
    // prepending, so the base covers the tail of the chain.
    let base = build_node_outer_app(
        cfg,
        &[&fls[2].lo, &fls[3].lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[2].acc, &fls[3].acc],
            claims_base: acc_base,
        }),
        None,
    );
    let app_n = base.app_base.expect("the base's app block");
    assert_eq!(app_n, app_fl, "one app offset, FL and node alike");
    let base_lane = base.lane_acc.clone().expect("the base's lane");
    assert_eq!(
        base.block.sigma.len(),
        N_KEY_SLOTS,
        "the base publishes both slots"
    );
    assert!(
        !entry_live(&base.block.sigma[1].1) && !entry_live(&base.block.jagged[1].1),
        "a fresh-only node's NODE slot is dead"
    );
    assert!(
        base.block.passenger.iter().all(|(_, c)| !entry_live(c)),
        "the base carries no passenger"
    );

    // node_2: a fresh FL + the base. Its node slot's inherited entry is
    // the base's DEAD one, and its own node-slot output is keyed by the
    // BASE circuit — the entry that will orphan one level up.
    let n2 = build_node_outer_app(
        cfg,
        &[&fls[1].lo, &base.lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[1].acc, &base_lane],
            claims_base: acc_base,
        }),
        Some(SpineIn {
            node_child: 1,
            prior: &base.block,
            forge: false,
        }),
    );
    let n2_lane = n2.lane_acc.clone().expect("node_2's lane");
    assert!(
        n2.block.passenger.iter().all(|(_, c)| !entry_live(c)),
        "node_2 orphans nothing — the base's node slot was already dead"
    );
    assert_eq!(
        n2.block.sigma[1].0,
        digest_f128(&base.lo.shape.circuit.digest()),
        "node_2's node slot is keyed by the BASE circuit"
    );
    assert_ne!(
        base.lo.shape.circuit.digest(),
        n2.lo.shape.circuit.digest(),
        "THE transitional mismatch: the base and the steady node are \
         different shapes, so node_3's node slot cannot fold what node_2 \
         published there"
    );

    // node_3: a fresh FL + node_2 — the STEADY node, and the one that
    // orphans. Its node slot names node_2's circuit, so the entry it
    // inherits (keyed by the base's) cannot fold and rides the passenger.
    let n3 = build_node_outer_app(
        cfg,
        &[&fls[0].lo, &n2.lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[0].acc, &n2_lane],
            claims_base: acc_base,
        }),
        Some(SpineIn {
            node_child: 1,
            prior: &n2.block,
            forge: false,
        }),
    );

    // ---- THE CONVERGENCE ----
    if n3.lo.shape.circuit.digest() != n2.lo.shape.circuit.digest() {
        println!("  SPINE DIGEST MISMATCH — per-slot rows (node_2 vs node_3):");
        for (t, (a, b)) in n2
            .lo
            .shape
            .counts
            .iter()
            .zip(&n3.lo.shape.counts)
            .enumerate()
        {
            if a != b {
                println!("    type {t}: n2 {a} vs n3 {b}");
            }
        }
        println!(
            "    publics {} vs {} | lanes {:?} vs {:?} | dense_m {} vs {}",
            n2.lo.public.len(),
            n3.lo.public.len(),
            n2.lo.pcs.num_lanes,
            n3.lo.pcs.num_lanes,
            n2.lo.pcs.m,
            n3.lo.pcs.m,
        );
        let (w2, w3) = (n2.lo.shape.circuit.wires(), n3.lo.shape.circuit.wires());
        println!(
            "    wire classes: {} vs {} ({} differ)",
            w2.len(),
            w3.len(),
            w2.iter().zip(w3).filter(|(a, b)| a != b).count()
        );
    }
    assert_eq!(
        n3.lo.shape.circuit.digest(),
        n2.lo.shape.circuit.digest(),
        "ONE steady spine shape: node_2 == node_3, at any depth"
    );

    // ---- THE ROOT ----
    // (1) the steady accumulator: two keyed slots, the FL's and node_2's.
    assert_eq!(n3.acc.sigma.len(), N_KEY_SLOTS, "the root's sigma slots");
    assert_eq!(
        n3.acc.sigma[0].0,
        fls[0].lo.shape.circuit.digest(),
        "FL slot key"
    );
    assert_eq!(
        n3.acc.sigma[1].0,
        n2.lo.shape.circuit.digest(),
        "node slot key"
    );
    // (2) THE PASSENGER: node_2's node-slot entries, keyed by the BASE
    // circuit — the only orphan a spine ever makes — against the base's
    // own tables.
    let pass = &n3.block.passenger;
    let base_d = base.lo.shape.circuit.digest();
    assert_eq!(
        pass[0].0,
        digest_f128(&base_d),
        "the passenger names the base"
    );
    assert_eq!(
        pass[1].0,
        digest_f128(&base_d),
        "the passenger names the base"
    );
    assert!(
        entry_live(&pass[0].1) && entry_live(&pass[1].1),
        "the orphan boarded"
    );
    let base_jp = node_jagged_params(&base.lo);
    let pass_acc = Accumulator {
        registry_digest: n3.acc.registry_digest,
        per_type: Vec::new(),
        per_element: Vec::new(),
        sigma: vec![(base_d, pass[0].1.clone())],
        jagged: vec![(base_d, pass[1].1.clone())],
    };
    assert!(
        pass_acc.discharge_sigma(&[&base.lo.shape.circuit]),
        "the passenger's sigma claim discharges against the BASE circuit's wiring"
    );
    assert!(
        pass_acc.discharge_jagged(&[(base_d, &base_jp)]),
        "the passenger's jagged claim discharges against the BASE circuit's layout"
    );
    // (3) the chain lane: eight leaves' claims in one accumulator.
    let lane3 = n3.lane_acc.clone().expect("the root's lane");
    assert!(
        lane3.discharge(&chain_mats) && lane3.discharge_sigma(&[chain_circuit]),
        "the root chain lane discharges against the chain tables"
    );
    // (4) the statement: the span is the whole chain.
    let h_end = native_chain(&h0, 8 * n_blocks);
    for j in 0..4 {
        assert_eq!(
            n3.lo.public[app_fl + j],
            pack4(h0[4 * j..4 * j + 4].try_into().unwrap()),
            "root h_start"
        );
        assert_eq!(
            n3.lo.public[app_fl + 4 + j],
            pack4(h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "root h_end == H^N(h_start)"
        );
    }
    // ---- THE MATCH-GATE ADVERSARIAL MATRIX (the owed soundness leg) ----
    // A statement-tier verify helper, the e2e tamper legs' assembly.
    let verify_with = |lo: &LeafOuter, publics: &[F128]| -> bool {
        let u = outer_union(&lo.shape.registry, lo.shape.counts.clone());
        let lcs = leaf_boolean_lcs(lo);
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        lo.proof
            .verify_circuit(
                &u,
                &lo.shape.circuit,
                publics,
                &lcs,
                &lo.commitment,
                &lo.pcs,
                &mut ch,
            )
            .is_ok()
    };
    // (a) THE FORGED LIVE FOLD — the load-bearing leg. A cheating node_3
    // re-witnesses the match-gate to claim the D_base entry MATCHES and
    // folds the orphan live (no passenger). Every forged gate row is
    // self-satisfying, so only the copy constraints can object; the
    // builder asserts the proof dies on exactly Wiring/Gkr/ProductMismatch
    // (the assert lives inside build_node_outer_app, forge: true).
    build_node_outer_app(
        cfg,
        &[&fls[0].lo, &n2.lo],
        Some(app_fl),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: chain_circuit,
            params: &chain_jp,
            priors: &[&fls[0].acc, &n2_lane],
            claims_base: acc_base,
        }),
        Some(SpineIn {
            node_child: 1,
            prior: &n2.block,
            forge: true,
        }),
    );
    // (b) THE PASSENGER DROP: zero the boarded orphan's live word — "no
    // orphan ever rode". The passenger is STATEMENT, so the honest proof
    // refuses the doctored segment.
    {
        let pass_base = env_pass_base(&env);
        assert_eq!(
            n3.lo.public[pass_base + 2],
            F128::ONE,
            "the orphan's sigma entry rides live"
        );
        let mut bad = n3.lo.public.clone();
        bad[pass_base + 2] = F128::ZERO;
        assert!(
            !verify_with(&n3.lo, &bad),
            "a dropped passenger must be rejected"
        );
    }
    // (c) THE FORGED CHILD KEY: node_2's published node-slot entry claims
    // to be keyed by the STEADY circuit instead of the base's — the lie
    // that would let node_3 fold it without a mismatch. The key words are
    // statement, so node_2's own proof refuses.
    {
        let uni_w = |c: &MatrixClaim| 2 + c.col.point.len() + c.row.point.len();
        let mut key_at = env_acc_main_base(&env);
        for (a, b) in n2.block.per_type.iter().chain(n2.block.per_element.iter()) {
            key_at += uni_w(a) + uni_w(b);
        }
        let s0 = &n2.block.sigma[0].1;
        key_at += 4 + s0.col.point.len() + s0.row.point.len(); // past the FL slot
        assert_eq!(
            n2.lo.public[key_at],
            digest_f128(&base_d)[0],
            "the offset arithmetic found the node slot's key"
        );
        let mut bad = n2.lo.public.clone();
        bad[key_at] = digest_f128(&n2.lo.shape.circuit.digest())[0];
        assert!(
            !verify_with(&n2.lo, &bad),
            "a forged entry key must be rejected"
        );
    }
    println!(
        "\nTHE SPINE CONVERGES (8 chains -> 4 FL -> base -> node_2 -> node_3)\n  \
         span H^{}(h_start) | D(node_2) == D(node_3) | 4 shapes total\n  \
         ONE steady accumulator (sigma+jagged x 2 slots) + a 2-entry passenger\n  \
         + the chain lane, all discharged at the root\n  \
         MATCH-GATE ADVERSARIAL MATRIX: forged live fold dies on the wiring\n  \
         product; a dropped passenger and a forged entry key die on the statement\n  \
         steady outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        8 * n_blocks,
        n3.lo.shape.circuit.cells().nu(),
        n3.lo.shape.circuit.cells().mu(),
        n3.lo.public.len(),
        serialize(&n3.lo.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// **Task 6: THE CHAIN TOWER, END TO END, WITH THE LANE.** Four chain
/// segments → two first-level nodes → one internal node; the chain-level
/// accumulators ride the internal node as a PRIORS-ONLY LANE (their
/// registry differs from the FL fold's, so they cannot join it), with the
/// prior surfaces connected wire-to-wire to the children's published
/// accumulator claims. The root then discharges
/// BOTH lanes — the chain lane against the chain b3 matrices + the chain
/// circuit's sigma table, the FL lane against the FL mats/element
/// types/digest — and reads the statement h_end == H^1024(h_start). Plus
/// the tamper matrix: a tampered FL statement word, a tampered lane
/// prior, and a tampered internal app word all die.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
pub(super) fn chain_tower_e2e_with_lane() {
    let cfg = test_config();

    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0007);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);
    let cp2 = build_chain_proof(cfg, cp1.h_end, n_blocks);
    let cp3 = build_chain_proof(cfg, cp2.h_end, n_blocks);
    let fl0 = build_fl_node(cfg, &cp0, &cp1);
    let fl1 = build_fl_node(cfg, &cp2, &cp3);
    assert_eq!(
        fl0.fold_pub_base, fl1.fold_pub_base,
        "one fold-block layout"
    );

    // The lane's registry materials — the CHAIN side.
    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cp0);
    let lane = ChainLane {
        registry: chain_registry,
        mats: &chain_mats,
        circs: &chain_circs,
        circuit: &cp0.inner.built.shape.circuit,
        params: &chain_jp,
        priors: &[&fl0.acc, &fl1.acc],
        claims_base: fl0.fold_pub_base,
    };
    let out = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fl0.stmt_base),
        Some(lane),
        None,
    );
    let (node, acc) = (out.lo, out.acc);
    let app = out.app_base.expect("the app block rode");
    let lane_acc = out.lane_acc.expect("the lane rode");

    // ---- THE ROOT ----
    // (1) The statement: the whole span, out of the internal node's publics.
    for j in 0..4 {
        assert_eq!(
            node.public[app + j],
            pack4(cp0.h_start[4 * j..4 * j + 4].try_into().unwrap()),
        );
        assert_eq!(
            node.public[app + 4 + j],
            pack4(cp3.h_end[4 * j..4 * j + 4].try_into().unwrap()),
        );
    }
    assert_eq!(cp3.h_end, native_chain(&cp0.h_start, 4 * n_blocks));
    // (2) The CHAIN lane discharges: boolean vs the chain b3 matrices,
    // sigma vs the chain circuit's own (masked) sigma table.
    assert!(
        lane_acc.discharge(&chain_mats),
        "chain-lane boolean discharges"
    );
    assert!(
        lane_acc.per_element.is_empty(),
        "the chain lane has no element group"
    );
    assert!(
        lane_acc.discharge_sigma(&[&cp0.inner.built.shape.circuit]),
        "chain-lane sigma discharges against the chain circuit"
    );
    // (3) The FL lane discharges: boolean vs the FL b3/swap/spread mats
    // (registry order), element vs the FL element types, sigma vs the FL
    // circuit digest's table.
    let fl_mats = leaf_boolean_mats(&fl0.lo);
    assert!(acc.discharge(&fl_mats), "FL-lane boolean discharges");
    let fl_el_mats: Vec<_> = fl0
        .lo
        .shape
        .registry
        .element_types()
        .iter()
        .map(|t| {
            let e = t.element_type().expect("element table");
            (e.a_0(), e.b_0())
        })
        .collect();
    assert!(
        acc.discharge_element(&fl_el_mats),
        "FL-lane element discharges"
    );
    assert!(
        acc.discharge_sigma(&[&fl0.lo.shape.circuit]),
        "FL-lane sigma discharges"
    );

    // ---- the tamper matrix ----
    // (a) A tampered FL STATEMENT word (its h_end): the FL proof must not
    //     verify against it — the adjacency data is statement-bound.
    {
        let union_f = outer_union(&fl0.lo.shape.registry, fl0.lo.shape.counts.clone());
        let lcs_f = leaf_boolean_lcs(&fl0.lo);
        let mut bad = fl0.lo.public.clone();
        bad[fl0.stmt_base + 4] += F128::ONE;
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        assert!(
            fl0.lo
                .proof
                .verify_circuit(
                    &union_f,
                    &fl0.lo.shape.circuit,
                    &bad,
                    &lcs_f,
                    &fl0.lo.commitment,
                    &fl0.lo.pcs,
                    &mut ch,
                )
                .is_err(),
            "a tampered FL h_end must be rejected"
        );
    }
    // (b) A tampered LANE PRIOR: the fold proof no longer matches.
    {
        let mut bad_acc = fl0.acc.clone();
        bad_acc.per_type[0].0.value += F128::ONE;
        let el_asserts_l: [(&UnionInstance<'_>, ElementAssertion); 0] = [];
        let jagged_pt: Vec<JaggedKeyProve<'_>> = vec![(
            cp0.inner.built.shape.circuit.digest(),
            &chain_jp,
            Vec::new(),
        )];
        let jagged_vt: Vec<JaggedKeyVerify<'_>> =
            vec![(cp0.inner.built.shape.circuit.digest(), Vec::new())];
        let mut chp = FsChallenger::with_chained_blake3(b"flock-chain-lane-tamper");
        let (lagg, _) = prove_aggregate_classes_with_grinding(
            chain_registry,
            &chain_mats,
            &chain_circs,
            &[],
            &[],
            &el_asserts_l,
            &[(&cp0.inner.built.shape.circuit, Vec::new())],
            &jagged_pt,
            &[&fl0.acc, &fl1.acc],
            tower_fold_grinding(cfg),
            &mut chp,
        )
        .expect("honest lane fold proves");
        let mut ch = FsChallenger::with_chained_blake3(b"flock-chain-lane-tamper");
        assert!(
            verify_aggregate_classes_with_grinding(
                chain_registry,
                &[],
                &el_asserts_l,
                &[(&cp0.inner.built.shape.circuit, Vec::new())],
                &jagged_vt,
                &[&bad_acc, &fl1.acc],
                &lagg,
                tower_fold_grinding(cfg),
                &mut ch,
            )
            .is_err(),
            "a tampered inherited claim must be rejected by the lane fold"
        );
    }
    // (c) A tampered INTERNAL app word: the internal proof's statement is
    //     bound the same way.
    {
        let union_n = outer_union(&node.shape.registry, node.shape.counts.clone());
        let lcs_n = leaf_boolean_lcs(&node);
        let mut bad = node.public.clone();
        bad[app + 7] += F128::ONE;
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        assert!(
            node.proof
                .verify_circuit(
                    &union_n,
                    &node.shape.circuit,
                    &bad,
                    &lcs_n,
                    &node.commitment,
                    &node.pcs,
                    &mut ch,
                )
                .is_err(),
            "a tampered internal h_end must be rejected"
        );
    }

    println!(
        "\nCHAIN TOWER E2E (4 chains -> 2 FL -> 1 internal, lane threaded)\n  \
         root statement: h_end == H^{}(h_start) | both lanes discharge | tampers die\n  \
         internal outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        4 * n_blocks,
        node.shape.circuit.cells().nu(),
        node.shape.circuit.cells().mu(),
        node.public.len(),
        serialize(&node.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// **Task 7a: THE M32 HEADLINE.** The chain tower at the THROUGHPUT-OPTIMAL
/// leaf size: 4 chain segments of `CHAIN_BLOCKS` (default 2^18 = 262,144)
/// compressions each — fast profile, ~16.8 MB hashed per leaf — through
/// two first-level nodes and one internal node with the lane, timed per
/// phase. The statement: h_end == H^(4·2^18)(h_start) ≈ one million
/// sequential compressions, proven and folded to one recursable proof.
/// Warm-box numbers; the cold certification wants the reboot + probe
/// ritual first (the recorded discipline).
#[test]
#[ignore] // The headline measurement — run explicitly with --nocapture.
pub(super) fn chain_tower_m32_headline() {
    let cfg = test_config();
    let n_blocks: usize = var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 18);
    let mut rng = Rng(0xC4A1_0008);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());

    // The sequential phase (the VDF delay): the chain values themselves.
    let t0 = Instant::now();
    let h_all = native_chain(&h0, 4 * n_blocks);
    let chain_ms = t0.elapsed().as_secs_f64() * 1e3;

    // The leaves, timed individually (parallelizable in deployment).
    let mut _leaf_ms: Vec<f64> = Vec::new();
    let mut mk = |start: [u32; 16]| -> ChainProof {
        let t = Instant::now();
        let cp = build_chain_proof(cfg, start, n_blocks);
        _leaf_ms.push(t.elapsed().as_secs_f64() * 1e3);
        cp
    };
    let cp0 = mk(h0);
    let cp1 = mk(cp0.h_end);
    let cp2 = mk(cp1.h_end);
    let cp3 = mk(cp2.h_end);
    assert_eq!(cp3.h_end, h_all, "the four segments ARE the chain");

    let t_fl = Instant::now();
    let fl0 = build_fl_node(cfg, &cp0, &cp1);
    let fl1 = build_fl_node(cfg, &cp2, &cp3);
    let _fl_ms = t_fl.elapsed().as_secs_f64() * 1e3 / 2.0;

    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = build_block_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cp0);
    let lane = ChainLane {
        registry: chain_registry,
        mats: &chain_mats,
        circs: &chain_circs,
        circuit: &cp0.inner.built.shape.circuit,
        params: &chain_jp,
        priors: &[&fl0.acc, &fl1.acc],
        claims_base: fl0.fold_pub_base,
    };
    let t_in = Instant::now();
    let out = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fl0.stmt_base),
        Some(lane),
        None,
    );
    let (node, acc, nt) = (out.lo, out.acc, out.online);
    let _internal_ms = t_in.elapsed().as_secs_f64() * 1e3;
    let app = out.app_base.expect("app block");
    let lane_acc = out.lane_acc.expect("lane");

    // The root.
    let t_root = Instant::now();
    for j in 0..4 {
        assert_eq!(
            node.public[app + 4 + j],
            pack4(h_all[4 * j..4 * j + 4].try_into().unwrap()),
            "root statement: h_end == H^(4·{n_blocks})(h_start)"
        );
    }
    assert!(
        lane_acc.discharge(&chain_mats)
            && lane_acc.discharge_sigma(&[&cp0.inner.built.shape.circuit])
    );
    let fl_mats = leaf_boolean_mats(&fl0.lo);
    let fl_el_mats: Vec<_> = fl0
        .lo
        .shape
        .registry
        .element_types()
        .iter()
        .map(|t| {
            let e = t.element_type().expect("element table");
            (e.a_0(), e.b_0())
        })
        .collect();
    assert!(
        acc.discharge(&fl_mats)
            && acc.discharge_element(&fl_el_mats)
            && acc.discharge_sigma(&[&fl0.lo.shape.circuit])
    );
    let root_ms = t_root.elapsed().as_secs_f64() * 1e3;

    let total_compr = 4 * n_blocks;
    // ---- SETUP vs ONLINE, per the contract on `Online` ----
    let leaves: Vec<Online> = [&cp0, &cp1, &cp2, &cp3].iter().map(|c| c.t).collect();
    let fl_t: Vec<Online> = [&fl0, &fl1].iter().map(|f| f.t).collect();
    let leaf_on = median_total(&leaves);
    let fl_on = median_total(&fl_t);
    let internal_on = nt.total();
    // A balanced tree over L leaves carries L/2 first-level nodes and
    // L/2 − 1 internal ones, so a leaf's amortised share tends to
    // leaf + FL/2 + internal/2; at four leaves the internal share is /4.
    let per_leaf_online = leaf_on + fl_on / 2.0 + internal_on / 4.0;
    println!(
        "\nCHAIN TOWER M32 HEADLINE (warm box; per-stage timing lives in tower_online_bench)\n  \
         {} compressions/leaf x 4 leaves = {} total ({:.1} MB hashed)\n  \
         sequential chain compute (the VDF delay, inherent): {:.0} ms\n  \
         ONLINE per proof (setup is per-SHAPE and excluded — see `Online`):",
        n_blocks,
        total_compr,
        (total_compr * 64) as f64 / 1e6,
        chain_ms,
    );
    report_stage("leaf", &leaves);
    report_stage("FL", &fl_t);
    report_stage("internal", from_ref(&nt));
    println!(
        "    root (both lanes + statement): {:.1} ms\n  \
         PER-LEAF ONLINE (leaf + FL/2 + internal/4): {:.0} ms -> {:.0}k compressions/sec\n  \
         internal outer: nu {} | mu {} | proof {:.1} KiB\n",
        root_ms,
        per_leaf_online,
        n_blocks as f64 / per_leaf_online,
        node.shape.circuit.cells().nu(),
        node.shape.circuit.cells().mu(),
        serialize(&node.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
    proof_census_mixed("internal node", &node.proof, &node.pcs);
    proof_census_mixed("chain leaf (m32 Fast)", &cp0.inner.proof, &cp0.inner.pcs);
    proof_census_mixed("FL node", &fl0.lo.proof, &fl0.lo.pcs);
}

/// **THE ONLINE BENCH: leaf, first-level node, internal node.** One number
/// per stage, measuring only what a prover pays PER STATEMENT — the walk,
/// the child tape sources, witness assembly, and the prove. Per-SHAPE setup
/// (circuit emit+finish, R1CS tables, PCS params, the fill plan, the tape
/// pins) is timed but reported apart and never folded into a per-proof
/// number: a shape is statement-independent, so a production prover builds
/// it once per level and reuses it for every segment.
///
/// Each stage is measured by ONE builder call whose online phases repeat
/// `BENCH_RUNS` times over FIXED inputs (STEADY_OVERRIDE — the setup is
/// paid once), taking per-phase MEDIANS — the first iteration pays
/// first-touch allocator costs that are warmup, not marginal cost.
///
/// Knobs: `BENCH_RUNS` (default 3), `CHAIN_BLOCKS` (default 256 — set
/// 262144 for the m32 production leaf), `TOWER_CONFIG=chain128` for the
/// 128-bit config. BOX DISCIPLINE: run the stability probe first and reboot if it
/// is far out of band — this box's benchmarks self-corrupt under sustained
/// load, and nothing here can tell you that happened.
#[test]
#[ignore] // Benchmark — run explicitly with --nocapture.
pub(super) fn tower_online_bench() {
    let cfg = test_config();
    let runs: usize = var("BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let n_blocks: usize = var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut rng = Rng(0xC4A1_00BE);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());

    // MEASUREMENT HYGIENE: each stage runs with only ITS OWN inputs
    // resident. An m32 chain proof and an FL node are both large, and
    // holding the whole tower alive while timing one stage inflates it
    // through allocator and pool pressure — the leaf's spread read
    // 639-1183 ms when the bench built everything up front. A production
    // prover drops a child once it has been folded, so the stages are
    // ordered to do the same.
    //
    // ONE BUILDER CALL PER STAGE (STEADY_OVERRIDE): the per-shape setup —
    // circuit emission, tape pins, R1CS, PCS params — is paid once and the
    // online phases repeat `runs` times inside the builder. The old
    // per-iteration rebuild spent ~96% of the bench's wall clock re-doing
    // byte-identical setup. The node arms therefore run as BLOCKS seconds
    // apart instead of the old minutes-apart interleave; box drift over
    // seconds is far below what the interleave guarded against.
    STEADY_OVERRIDE.store(runs, Ordering::Relaxed); // +1: iteration 0 is the shape warmup (setup tier)

    // ---- LEAF: nothing else is alive; the measured proof BECOMES cp0 ----
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let leaf = cp0.onlines.clone();
    STEADY_OVERRIDE.store(0, Ordering::Relaxed);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);

    // ---- FL: two chain children and nothing more. The measured FL is the
    // spine's FRESH child — the EARLIEST segments, since a spine PREPENDS —
    // so the measured leaf and FL become the tower's own materials. ----
    STEADY_OVERRIDE.store(runs, Ordering::Relaxed); // +1: iteration 0 is the shape warmup (setup tier)
    let fresh = build_fl_node(cfg, &cp0, &cp1);
    let fl = fresh.onlines.clone();
    STEADY_OVERRIDE.store(0, Ordering::Relaxed);

    // ---- the rest of the tower: four more segments, two more FLs. The
    // INTERNAL arm's node doubles as the spine's BASE child (identical
    // children shape — the old separate base build was pure waste), so the
    // whole tower is 6 chain proofs, 3 FLs and 2 node builds where it was
    // 10, 5 and 3. The segment pairs are scoped so they drop once folded —
    // what production carries.
    let (fl0, fl1) = {
        let cp2 = build_chain_proof(cfg, cp1.h_end, n_blocks);
        let cp3 = build_chain_proof(cfg, cp2.h_end, n_blocks);
        let cp4 = build_chain_proof(cfg, cp3.h_end, n_blocks);
        let cp5 = build_chain_proof(cfg, cp4.h_end, n_blocks);
        (
            build_fl_node(cfg, &cp2, &cp3),
            build_fl_node(cfg, &cp4, &cp5),
        )
    };
    drop(cp1);
    // The lane is what production carries: the children's chain
    // accumulators fold in a priors-only aggregate of their own.
    let chain_registry = &cp0.inner.built.shape.registry;
    let blake_r1cs = chain_blake_r1cs(cp0.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cp0);
    // ---- INTERNAL (= the spine's base) then SPINE, one call each ----
    // Both arms' online iterations run back to back inside their builder
    // call (seconds apart, not the old minutes-apart interleave), from
    // materials that are ALL resident before either starts.
    STEADY_OVERRIDE.store(runs, Ordering::Relaxed); // +1: iteration 0 is the shape warmup (setup tier)
    let base = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fresh.stmt_base),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: &cp0.inner.built.shape.circuit,
            params: &chain_jp,
            priors: &[&fl0.acc, &fl1.acc],
            claims_base: fresh.fold_pub_base,
        }),
        None,
    );
    let internal = base.onlines.clone();
    // The steady spine node: the fresh FL (the segments BEFORE the base's)
    // plus the base as the node child whose accumulator it inherits —
    // built exactly as the convergence test builds node_2.
    let spine: Vec<Online> = {
        let lane = base.lane_acc.clone().expect("the base's lane");
        build_node_outer_app(
            cfg,
            &[&fresh.lo, &base.lo],
            Some(fresh.stmt_base),
            Some(ChainLane {
                registry: chain_registry,
                mats: &chain_mats,
                circs: &chain_circs,
                circuit: &cp0.inner.built.shape.circuit,
                params: &chain_jp,
                priors: &[&fresh.acc, &lane],
                claims_base: fresh.fold_pub_base,
            }),
            Some(SpineIn {
                node_child: 1,
                prior: &base.block,
                forge: false,
            }),
        )
        .onlines
    };
    STEADY_OVERRIDE.store(usize::MAX, Ordering::Relaxed);

    // SHAPE WARMUP IS SETUP, NOT MARGINAL COST. A stage's first online
    // iteration primes the zero/scratch pools and faults in the allocator
    // arena for its buffer size classes — one-time per-shape state that a
    // production prover reaches once and keeps. Left inside the samples it
    // skews whichever arm runs FIRST (both node arms share size classes,
    // so the later arm inherits the earlier one's warmth: the internal-vs-
    // spine delta once read −7% from ordering alone). So iteration 0 is
    // reported on the setup tier and every per-proof number below is a
    // median over the STEADY iterations only.
    let steady = |runs: &[Online]| -> Vec<Online> {
        if runs.len() > 1 {
            runs[1..].to_vec()
        } else {
            runs.to_vec()
        }
    };
    let warmup = |runs: &[Online]| -> Option<f64> { (runs.len() > 1).then(|| runs[0].total()) };
    let warms = [
        ("leaf", warmup(&leaf)),
        ("FL", warmup(&fl)),
        ("internal", warmup(&internal)),
        ("spine", warmup(&spine)),
    ];
    let (leaf, fl, internal, spine) = (
        steady(&leaf),
        steady(&fl),
        steady(&internal),
        steady(&spine),
    );
    let (leaf_on, fl_on, int_on) = (
        median_total(&leaf),
        median_total(&fl),
        median_total(&internal),
    );
    // ANY binary tree over L leaves carries L/2 first-level nodes and
    // L/2 − 1 nodes above them — the count is tree-shape-indifferent — so
    // a leaf's amortised share tends to leaf + FL/2 + node/2 whichever
    // shape the tower uses. The SPINE's node is the honest one to divide
    // by: it is what every level above 2 runs.
    let node_on = if spine.is_empty() {
        int_on
    } else {
        median_total(&spine)
    };
    let per_leaf = leaf_on + fl_on / 2.0 + node_on / 2.0;
    println!(
        "\nONLINE BENCH — {n_blocks} compressions/leaf, {runs} runs/stage, profile {:?}\n  \
         per-proof ONLINE (setup is per-SHAPE, shown for reference only):",
        cfg,
    );
    for (name, w) in warms {
        if let Some(ms) = w {
            println!("    {name:9} shape warmup (setup tier, dropped from medians): {ms:.1} ms");
        }
    }
    report_stage("leaf", &leaf);
    report_stage("FL", &fl);
    report_stage("internal", &internal);
    if !spine.is_empty() {
        report_stage("spine (steady)", &spine);
        println!(
            "  the spine's node costs {:+.1}% against the fresh-only \
             internal — the tree's node COUNT is unchanged, so this delta \
             IS wall 3's whole price",
            100.0 * (node_on - int_on) / int_on,
        );
    }
    println!(
        "  AMORTISED per leaf (leaf + FL/2 + node/2): {:.0} ms \
         -> {:.0}k compressions/sec\n  \
         the leaf's walk IS the chain compute — the application's own \
         sequential work, not proving\n",
        per_leaf,
        n_blocks as f64 / per_leaf,
    );
}

/// **THE DRIVER, END TO END.** [`Tower::prove`] runs the same 8-leaf
/// steady tower `chain_spine_converges` hand-assembles — sequential
/// leaves, four FLs, the base plus two spine folds, the lane threaded,
/// convergence asserted inside the loop — and the root-side residue
/// discharges. Then the falsifiability legs: a doctored statement word, a
/// doctored lane claim, a killed passenger, a doctored sigma claim, and a
/// doctored lane jagged claim must each turn [`Tower::discharge_root`]
/// away.
#[test]
#[ignore] // Heavy — eight chain proofs and seven outers via the driver.
pub(super) fn tower_driver_e2e() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xD41_4E12);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());

    let mut tower = Tower::prove(cfg, h0, n_blocks, 8);
    assert_eq!(
        tower.statement().h_start,
        h0,
        "the statement starts where the caller did"
    );
    assert_eq!(
        tower.statement().h_end,
        native_chain(&h0, 8 * n_blocks),
        "the driver's statement IS the chain"
    );
    assert_eq!(tower.statement().n_blocks, 8 * n_blocks);
    tower.discharge_root().expect("the honest root discharges");

    // (a) a doctored statement word (the root's published h_end).
    let at = tower.app_base + 4;
    let saved = tower.root.lo.public[at];
    tower.root.lo.public[at] += F128::ONE;
    assert_eq!(
        tower.discharge_root(),
        Err(RootDischargeFailure::Statement),
        "a doctored statement word must be refused"
    );
    tower.root.lo.public[at] = saved;

    // (b) a doctored lane claim.
    let lane = tower.root.lane_acc.as_mut().expect("the lane rides");
    let saved = lane.per_type[0].0.value;
    lane.per_type[0].0.value += F128::ONE;
    assert_eq!(
        tower.discharge_root(),
        Err(RootDischargeFailure::ChainLaneBoolean),
        "a doctored lane claim must be refused"
    );
    tower
        .root
        .lane_acc
        .as_mut()
        .expect("the lane rides")
        .per_type[0]
        .0
        .value = saved;

    // (c) a killed passenger: a steady root OWES its orphan.
    let saved = tower.root.block.passenger[0].1.row.low[0];
    tower.root.block.passenger[0].1.row.low[0] = F128::ZERO;
    assert_eq!(
        tower.discharge_root(),
        Err(RootDischargeFailure::Passenger),
        "a steady root without its orphan must be refused"
    );
    tower.root.block.passenger[0].1.row.low[0] = saved;

    // (d) a doctored main-fold sigma claim.
    let saved = tower.root.acc.sigma[0].1.value;
    tower.root.acc.sigma[0].1.value += F128::ONE;
    assert_eq!(
        tower.discharge_root(),
        Err(RootDischargeFailure::Sigma),
        "a doctored sigma claim must be refused"
    );
    tower.root.acc.sigma[0].1.value = saved;

    // (e) a doctored lane JAGGED claim — the discharge leg the
    // hand-built root sections never ran; prove it can refuse.
    let lane = tower.root.lane_acc.as_mut().expect("the lane rides");
    assert!(!lane.jagged.is_empty(), "the lane carries jagged claims");
    let saved = lane.jagged[0].1.value;
    lane.jagged[0].1.value += F128::ONE;
    assert_eq!(
        tower.discharge_root(),
        Err(RootDischargeFailure::ChainLaneJagged),
        "a doctored lane jagged claim must be refused"
    );
    tower.root.lane_acc.as_mut().expect("the lane rides").jagged[0]
        .1
        .value = saved;

    tower
        .discharge_root()
        .expect("restored, the root discharges again");
}

/// **THE STANDALONE VERIFIER, END TO END.** [`TowerVk::generate`] proves
/// the six-leaf reference tower once, then [`verify_root`] checks FRESH
/// towers it never built at every depth class — k = 4 (steady, live
/// passenger, span [`SpanBound::EndpointsOnly`]), k = 3 (steady,
/// passenger-less, exact) and k = 2 (base-rooted, exact) — plus the VK's
/// own reference root; every table from the VK, every accumulator
/// REASSEMBLED from the bundle's publics and cross-checked against the
/// prover's own objects. Then the consumer-side refusals: a doctored
/// public word dies in the proof verify, a wrong statement on the
/// binding, a truncated bundle and a misfit span on the guards, an
/// unknown slot key in the decode, and a CROSS-CLASS span lie on the
/// passenger rule — while the k >= 4 in-class span degeneracy is pinned
/// as exactly what [`SpanBound::EndpointsOnly`] declares.
#[test]
#[ignore] // Heavy — four towers (6+8+6+4 leaves) end to end.
pub(super) fn tower_verify_root_e2e() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let env = envelope_shape();
    let vk = TowerVk::generate(cfg, n_blocks);

    // (0) the VK's own reference root (k = 3, steady shape) verifies
    // through the consumer path.
    let own = *vk.tower.statement();
    assert_eq!(
        verify_root(&vk, &own, &vk.tower.root_bundle()).expect("the reference root verifies"),
        SpanBound::Exact,
        "a passenger-less steady root pins its span"
    );

    // (1) a fresh k = 4 steady tower the VK never saw.
    let mut rng = Rng(0xD41_4E13);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());
    let tower = Tower::prove(cfg, h0, n_blocks, 8);
    let stmt = *tower.statement();
    assert_eq!(
        verify_root(&vk, &stmt, &tower.root_bundle()).expect("the k = 4 root verifies"),
        SpanBound::EndpointsOnly,
        "a steady root with a live passenger certifies endpoints + class"
    );

    // The reassembly cross-check: publics-decoded == the prover's own
    // objects, all three blocks.
    let (main, passenger, lane) =
        reassemble(&tower.root.lo.public, &vk).expect("the root's blocks decode");
    assert_eq!(main, tower.root.acc, "main acc, from publics alone");
    assert_eq!(
        &lane,
        tower.root.lane_acc.as_ref().expect("lane"),
        "chain lane, from publics alone"
    );
    assert_eq!(
        passenger, tower.root.block.passenger,
        "the passenger, from publics alone"
    );

    // (2) consumer-side refusals.
    // A doctored public word dies in the PROOF verify, before any
    // discharge sees it.
    let mut bad = tower.root.lo.public.clone();
    bad[tower.app_base + 4] += F128::ONE;
    let bundle = RootBundle {
        public: &bad,
        proof: &tower.root.lo.proof,
        commitment: &tower.root.lo.commitment,
    };
    assert!(
        matches!(
            verify_root(&vk, &stmt, &bundle),
            Err(TowerVerifyError::Proof(_))
        ),
        "a doctored public word must die in the proof verify"
    );
    // A wrong statement (honest proof, different claim) dies on the
    // statement binding.
    let mut wrong = stmt;
    wrong.h_end[0] ^= 1;
    assert!(
        matches!(
            verify_root(&vk, &wrong, &tower.root_bundle()),
            Err(TowerVerifyError::Discharge(RootDischargeFailure::Statement))
        ),
        "a wrong statement must die on the binding"
    );
    // A truncated public segment dies on the length guard.
    let short = &tower.root.lo.public[..tower.root.lo.public.len() - 1];
    let bundle = RootBundle {
        public: short,
        proof: &tower.root.lo.proof,
        commitment: &tower.root.lo.commitment,
    };
    assert!(
        matches!(
            verify_root(&vk, &stmt, &bundle),
            Err(TowerVerifyError::PublicsLength)
        ),
        "a truncated bundle must die on the length guard"
    );
    // A span that is not a whole number of leaf pairs dies on geometry.
    let mut misfit = stmt;
    misfit.n_blocks += 1;
    assert!(
        matches!(
            verify_root(&vk, &misfit, &tower.root_bundle()),
            Err(TowerVerifyError::Geometry)
        ),
        "a misfit span must die on geometry"
    );
    // A doctored slot KEY word dies in the decode: reassembly refuses a
    // live keyed entry naming a circuit the VK does not know.
    {
        let uni_w = |c: &MatrixClaim| 2 + c.col.point.len() + c.row.point.len();
        let mut key_at = env_acc_main_base(&env);
        for (a, b) in tower
            .root
            .acc
            .per_type
            .iter()
            .chain(tower.root.acc.per_element.iter())
        {
            key_at += uni_w(a) + uni_w(b);
        }
        let mut bad = tower.root.lo.public.clone();
        bad[key_at] += F128::ONE;
        assert!(
            matches!(reassemble(&bad, &vk), Err(TowerVerifyError::UnknownKey)),
            "an unknown slot key must be refused, not skipped"
        );
    }
    // THE IN-CLASS SPAN DEGENERACY, pinned: an inflated k >= 4 claim over
    // an honest k = 4 bundle still verifies — which is exactly why a
    // green k >= 4 result says EndpointsOnly, never Exact. The count word
    // in the app block is the recorded protocol follow-up.
    let mut inflated = stmt;
    inflated.n_blocks += 2 * 2 * n_blocks;
    assert_eq!(
        verify_root(&vk, &inflated, &tower.root_bundle())
            .expect("the in-class span lie is not detectable today"),
        SpanBound::EndpointsOnly,
        "and the result says so"
    );

    // (3) a fresh k = 3 tower — steady shape, passenger-less, exact span.
    let t3 = Tower::prove(cfg, native_chain(&h0, 128), n_blocks, 6);
    assert_eq!(
        verify_root(&vk, t3.statement(), &t3.root_bundle()).expect("the k = 3 root verifies"),
        SpanBound::Exact,
    );
    // A CROSS-CLASS span lie dies on the passenger rule: the k = 3
    // bundle claimed as k = 4 owes an orphan it does not carry.
    let mut lied = *t3.statement();
    lied.n_blocks += 2 * n_blocks;
    assert!(
        matches!(
            verify_root(&vk, &lied, &t3.root_bundle()),
            Err(TowerVerifyError::Discharge(RootDischargeFailure::Passenger))
        ),
        "a k = 3 bundle claimed steady-deep must die on the passenger"
    );

    // (4) a fresh k = 2 base-rooted tower — the other root shape.
    let t2 = Tower::prove(cfg, native_chain(&h0, 64), n_blocks, 4);
    assert_eq!(
        verify_root(&vk, t2.statement(), &t2.root_bundle()).expect("the k = 2 root verifies"),
        SpanBound::Exact,
    );
}
