//! leanXMSS leaves for the recursion tower (`docs/leanxmss-recursion-plan.md`).
//!
//! A leaf is a leanXMSS aggregate proof over a batch of signatures, proven
//! in the recursion posture ([`XmssOptions::RECURSION_LEAF`]) and
//! deferred-verified, so [`ChildTape`] can replay its verifier in-circuit.

#[cfg(test)]
use {
    crate::{
        leanxmss::{
            circuit::XmssOptions,
            reference::{Value, XmssHash, sign_keyless},
        },
        tower::{
            ChildSlots, ChildTape, SLOT_WORDS, ShapeBuilder, Wire, check_child_region,
            emit_child_region, expected_child_tail_schedule,
        },
    },
    std::{any::Any, time::Instant},
};

use flock_core::{
    aggregate::{Accumulator, Dropped},
    circuit::builder::BuiltCircuit,
    pcs::{LOG_PACKING, jagged::JaggedParams},
    union::publics_digest,
};
use flock_merkle::hash_pair;

#[cfg(target_arch = "aarch64")]
use crate::verifier::verify_ligerito_union_circuit_ag_deferred;
#[cfg(not(target_arch = "aarch64"))]
use crate::verifier::verify_ligerito_union_circuit_deferred;
use crate::{
    leanxmss::{
        circuit::{FIAT_SHAMIR_DOMAIN, Statement, XmssCircuit},
        reference::Signature,
    },
    tower::{
        F128, FlApp, FlFamily, HashKind, LeafOuter, MixedInner, MixedProof, Online, TowerConfig,
        build_first_level_core,
    },
};

/// The leanXMSS leaf's transcript domain: the leaf proves under the chained
/// BLAKE3 transcript with the leanXMSS domain.
pub(super) const XMSS_DOMAIN: &[u8] = FIAT_SHAMIR_DOMAIN;

/// One leanXMSS leaf: the proof, deferred-verified, as a tower child.
pub struct XmssLeaf {
    pub(super) inner: MixedInner,
    pub(super) n: usize,
}

/// Prove `statements` (with their signatures) under `circuit`, which must
/// be in the recursion posture, and deferred-verify the proof.
pub fn build_xmss_leaf(
    circuit: &XmssCircuit,
    statements: &[Statement],
    sigs: &[Signature],
) -> XmssLeaf {
    let options = circuit.options();
    assert!(
        options.chained_transcript && options.hashed_statement,
        "a tower leaf proves under the chained transcript with a hashed statement"
    );
    let witness = circuit.witness(statements, sigs);
    let proof = circuit.prove(&witness);
    let built = BuiltCircuit {
        shape: circuit.shape().clone(),
        witness,
    };
    let union = circuit.union();
    let pcs = XmssCircuit::pcs_params(&union);
    let lcs = circuit.lincheck_circuits();
    let mut ch = circuit.challenger();
    #[cfg(target_arch = "aarch64")]
    let (proof, verified) = {
        let v = verify_ligerito_union_circuit_ag_deferred(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &lcs,
            &proof.commitment,
            &proof.proof,
            &pcs,
            &mut ch,
        );
        (MixedProof::Ag(proof.proof), (v, proof.commitment))
    };
    #[cfg(not(target_arch = "aarch64"))]
    let (proof, verified) = {
        let v = verify_ligerito_union_circuit_deferred(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &lcs,
            &proof.commitment,
            &proof.proof,
            &pcs,
            &mut ch,
        );
        (MixedProof::Rs(proof.proof), (v, proof.commitment))
    };
    let (result, commitment) = verified;
    let (_claims, work, sigma) = result.expect("the deferred verify accepts an honest leaf");
    work.boolean
        .as_ref()
        .expect("boolean matrix work")
        .check(&union, &lcs)
        .expect("the boolean matrix work discharges");
    assert!(work.element.is_none(), "leanXMSS has no element class");
    assert!(sigma.check(&built.shape.circuit), "sigma discharges");
    drop(union);
    XmssLeaf {
        inner: MixedInner {
            nu: circuit.nu(),
            built,
            proof,
            commitment,
            pcs,
            work,
            sigma,
            full_height: true,
        },
        n: statements.len(),
    }
}

impl XmssLeaf {
    /// The public segment: the shared constants, then per signer `P`, the
    /// root and the message.
    pub fn public(&self) -> &[F128] {
        &self.inner.built.witness.public
    }
}

/// A first-level node over two leanXMSS leaves: both verified in-circuit,
/// their claims folded, and `D = hash_pair(D_0, D_1)` over the leaves'
/// publics digests published with the signature counter.
#[cfg_attr(not(test), allow(dead_code))]
pub struct XmssFlNode {
    pub(super) lo: LeafOuter,
    pub(super) acc: Accumulator,
    pub(super) stmt_base: usize,
    pub(super) fold_pub_base: usize,
    /// The node's batch digest.
    pub(super) digest: [u8; 32],
    /// Signatures under the node.
    pub(super) n: usize,
    pub(super) t: Online,
}

/// The leanXMSS leaf family under `circuit`: the leaf domain, and per
/// boolean registry type the base matrices and the lincheck circuit.
pub(super) fn xmss_family(circuit: &XmssCircuit) -> FlFamily<'_> {
    let registry = &circuit.shape().registry;
    FlFamily {
        domain: XMSS_DOMAIN,
        mats: registry
            .boolean_types()
            .iter()
            .map(|ty| (&ty.a_0, &ty.b_0))
            .collect(),
        circs: circuit.lincheck_circuits(),
        lane_words: None,
        shared: Some(Dropped::of(&circuit.union())),
    }
}

/// The first-level node over two leaves proven under `circuit`.
pub fn build_xmss_first_level_node(
    cfg: TowerConfig,
    circuit: &XmssCircuit,
    leaves: [&XmssLeaf; 2],
) -> XmssFlNode {
    const FL_DOMAIN: &[u8] = b"flock-xmss-fl-node-v0";
    let per_leaf = leaves[0].n;
    assert_eq!(leaves[1].n, per_leaf, "one batch size per first-level node");
    let inners = [&leaves[0].inner, &leaves[1].inner];
    let app = FlApp::Batches { per_leaf };
    let core = build_first_level_core(cfg, FL_DOMAIN, &inners, &xmss_family(circuit), &app);
    XmssFlNode {
        lo: core.lo,
        acc: core.acc,
        stmt_base: core.stmt_base,
        fold_pub_base: core.fold_pub_base,
        digest: hash_pair(
            &publics_digest(leaves[0].public()),
            &publics_digest(leaves[1].public()),
            HashKind::Blake3,
        ),
        n: 2 * per_leaf,
        t: core.t,
    }
}

/// `n` signers with key-free signatures, each on its own message.
#[cfg(test)]
pub(super) fn signed(n: usize, seed: u64) -> (Vec<Statement>, Vec<Signature>) {
    (0..n)
        .map(|s| {
            let pp: Value = std::array::from_fn(|j| (seed as usize + 31 * s + j) as u8);
            let msg: [u8; 32] = std::array::from_fn(|j| (7 * s + 3 * j + 1) as u8);
            let idx = (0x9E37_79B9u64.wrapping_mul(seed + s as u64 + 1) >> 7) as u32;
            let (pk, sig) = sign_keyless(XmssHash::Blake3, &pp, idx, &msg, seed + s as u64, true);
            (Statement { pk, msg }, sig)
        })
        .unzip()
}

/// Phase 2 probe: one leanXMSS leaf's tape, then its child region alone —
/// emitted, run and checked.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn xmss_child_region_emits_alone() {
    let n = 4;
    let circuit = XmssCircuit::with_options(n, None, XmssOptions::RECURSION_LEAF);
    let (statements, sigs) = signed(n, 17);
    let t = Instant::now();
    let leaf = build_xmss_leaf(&circuit, &statements, &sigs);
    eprintln!("leaf n={n} nu={} built in {:?}", circuit.nu(), t.elapsed());
    let lcs = circuit.lincheck_circuits();
    let ct = ChildTape::new(&leaf.inner, XMSS_DOMAIN, &lcs);
    let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(16);
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let mut vals: Vec<F128> = Vec::new();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let b3_slot = cs.q.b3;
    let region = emit_child_region(
        &mut sb,
        &mut cs,
        b3_slot,
        &ct,
        &mut vals,
        &mut hints,
        &mut consts,
    );
    let shape2 = sb.finish().expect("the leanXMSS child circuit builds");
    let hint_refs: Vec<&(dyn Any + Sync)> = hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
    let built2 = shape2.run(&vals, &hint_refs);
    let consumed = check_child_region(&built2.public, &ct, &region);
    assert_eq!(region.pub_base + consumed, built2.public.len());
    assert_eq!(region.tail_schedule, expected_child_tail_schedule(&ct));
}

/// Emit `ct`'s child region alone at `nu2` rows a slot and report what it
/// occupies: per gate type the rows used, and the dense words a union over
/// that content commits.
#[cfg(test)]
fn region_census(label: &str, ct: &ChildTape<'_>, nu2: usize) {
    use crate::tower::UnionInstance;
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let (mut vals, mut hints, mut consts) = (Vec::new(), Vec::new(), Vec::new());
    let b3_slot = cs.q.b3;
    let _ = emit_child_region(
        &mut sb,
        &mut cs,
        b3_slot,
        ct,
        &mut vals,
        &mut hints,
        &mut consts,
    );
    let shape = sb.finish().expect("the region builds");
    let u = UnionInstance::new(&shape.registry, shape.counts.clone());
    println!(
        "\n{label}: b3 rows (tape model) {} | pd claims {} (wiring {} + bus {}) | P {} | \
         wiring mu {} | bus layers {} | publics {} | dense words {} (content m {})",
        ct.b3_rows,
        ct.n_pd,
        ct.n_wire,
        ct.n_bus,
        ct.n_p,
        ct.mu_i,
        ct.bus.as_ref().map_or(0, |b| b.layers.len()),
        ct.inner.built.witness.public.len(),
        u.dense_words(),
        u.dense_m(),
    );
    let mut per: Vec<(usize, usize, usize, usize)> = shape
        .registry
        .types()
        .iter()
        .zip(&shape.counts)
        .enumerate()
        .map(|(i, (ty, &n_t))| {
            let cols = ty.useful_bits.div_ceil(128).min(1usize << (ty.k_log - 7));
            (cols * n_t, i, n_t, ty.k_log)
        })
        .collect();
    per.sort_by_key(|p| std::cmp::Reverse(p.0));
    for &(words, i, rows, k_log) in per.iter().take(14) {
        println!("    type {i:2} (k_log {k_log:2}) rows {rows:>7}  words {words:>8}");
    }
}

/// The size of one leanXMSS child region at `XMSS_N` signatures (default
/// 1000; the bench's signature cache is reused), beside the m32 chain
/// leaf's (`CHAIN_BLOCKS`, default 262144) — the envelope question.
#[test]
#[ignore] // Heavy: a 1000-signature leaf and an m32 chain leaf.
fn xmss_child_region_census() {
    use crate::{
        leanxmss::circuit::keyless_batch,
        tower::{DOMAIN, build_chain_proof, chain_blake_r1cs, test_config},
    };
    let n: usize = std::env::var("XMSS_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let (statements, sigs) = cached_signatures(n).unwrap_or_else(|| keyless_batch(n, 1));
    let circuit = XmssCircuit::with_options(n, None, XmssOptions::RECURSION_LEAF);
    let t = Instant::now();
    let leaf = build_xmss_leaf(&circuit, &statements, &sigs);
    println!(
        "leanXMSS leaf n={n} nu={} in {:?}",
        circuit.nu(),
        t.elapsed()
    );
    let lcs = circuit.lincheck_circuits();
    let ct = ChildTape::new(&leaf.inner, XMSS_DOMAIN, &lcs);
    region_census(&format!("leanXMSS n={n}"), &ct, 20);
    drop(ct);
    drop(leaf);

    let blocks: usize = std::env::var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 18);
    let cp = build_chain_proof(test_config(), [7u32; 16], blocks);
    let b3 = chain_blake_r1cs(cp.inner.nu);
    let ct = ChildTape::new(&cp.inner, DOMAIN, &[b3.csc_lincheck_circuit()]);
    region_census(&format!("chain {blocks} blocks"), &ct, 20);
}

/// The bench's signature cache (`target/leanxmss-blake3-keyless-v2-{n}.bin`).
#[cfg(test)]
fn cached_signatures(n: usize) -> Option<(Vec<Statement>, Vec<Signature>)> {
    use crate::leanxmss::reference::{PK_LEN, PublicKey};
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("leanxmss-blake3-keyless-v2-{n}.bin"));
    let bytes = std::fs::read(path).ok()?;
    let record = PK_LEN + 32 + (bytes.len() / n.max(1) - PK_LEN - 32);
    (bytes.len() == n * record).then(|| {
        bytes
            .chunks_exact(record)
            .map(|r| {
                let pk = PublicKey::from_bytes(r[..PK_LEN].try_into().unwrap());
                let msg = r[PK_LEN..PK_LEN + 32].try_into().unwrap();
                let sig = Signature::from_bytes(&r[PK_LEN + 32..]).unwrap();
                (Statement { pk, msg }, sig)
            })
            .unzip()
    })
}

/// Phase 3: a first-level node over two leanXMSS leaves — both verified in
/// the envelope outer, their claims folded, the batch digest published.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn xmss_first_level_node_over_two_leaves() {
    use crate::tower::test_config;
    let n: usize = std::env::var("XMSS_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    // A tall slot keeps a small batch to a few slots per gate kind, the
    // shape a production leaf has (plan_nu would spread 4 signatures over
    // 2^6-row slots: twice the gather claims of a 1000-signature leaf);
    // 2^9 still fills the step kind's first slot, so words drop.
    let nu = std::env::var("XMSS_NU")
        .ok()
        .and_then(|v| v.parse().ok())
        .or((n < 64).then_some(9));
    let circuit = XmssCircuit::with_options(n, nu, XmssOptions::RECURSION_LEAF);
    let batch = |seed: u64| {
        if n < 64 {
            signed(n, seed)
        } else {
            crate::leanxmss::circuit::keyless_batch(n, seed)
        }
    };
    let (s0, g0) = batch(17);
    let (s1, g1) = batch(18);
    let leaf0 = build_xmss_leaf(&circuit, &s0, &g0);
    let leaf1 = build_xmss_leaf(&circuit, &s1, &g1);
    let t = Instant::now();
    let fl = build_xmss_first_level_node(test_config(), &circuit, [&leaf0, &leaf1]);
    println!(
        "xmss FL over 2 x {n} signatures: built in {:?} | prove {:.0} ms | verify {:.1} ms | \
         publics {} | proof {:.1} KiB",
        t.elapsed(),
        fl.t.prove_ms,
        fl.t.verify_ms,
        fl.lo.public.len(),
        bincode::serialize(&fl.lo.proof).unwrap().len() as f64 / 1024.0,
    );
    assert_eq!(fl.n, 2 * n);
    assert_eq!(
        fl.lo.public[fl.stmt_base..fl.stmt_base + 2],
        crate::tower::node::digest_f128(&fl.digest)[..],
        "the published batch digest"
    );
}

/// The leaf layout's jagged params — the lane's jagged table owner, read
/// exactly as the opening verifier reads it (full-height union).
pub(super) fn xmss_jagged_params(leaf: &XmssLeaf) -> JaggedParams {
    let u = leaf.inner.union();
    JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        leaf.inner.commitment.params.m - LOG_PACKING,
    )
}

/// Phase 4: an internal node over two leanXMSS first-level nodes — both
/// verified in-circuit, the leanXMSS lane folded per matrix class, the
/// batch digest composed `hash_pair(D_left, D_right)` and the signature
/// counter multiplied.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn xmss_internal_node_over_two_first_level_nodes() {
    use crate::tower::{
        ChainLane, LaneFamily, build_node_outer_app, lane_block_words, node::digest_f128,
        span_count_word, test_config,
    };
    let cfg = test_config();
    let n = 4;
    let circuit = XmssCircuit::with_options(n, Some(9), XmssOptions::RECURSION_LEAF);
    let leaves: Vec<XmssLeaf> = (0..4u64)
        .map(|i| {
            let (st, sg) = signed(n, 17 + i);
            build_xmss_leaf(&circuit, &st, &sg)
        })
        .collect();
    let fl0 = build_xmss_first_level_node(cfg, &circuit, [&leaves[0], &leaves[1]]);
    let fl1 = build_xmss_first_level_node(cfg, &circuit, [&leaves[2], &leaves[3]]);
    let family = xmss_family(&circuit);
    let jp = xmss_jagged_params(&leaves[0]);
    let lane_words = lane_block_words(&fl0.acc);
    let t = Instant::now();
    let out = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fl0.stmt_base),
        Some(ChainLane {
            registry: &leaves[0].inner.built.shape.registry,
            mats: &family.mats,
            circs: &family.circs,
            circuit: &leaves[0].inner.built.shape.circuit,
            params: &jp,
            priors: &[&fl0.acc, &fl1.acc],
            claims_base: fl0.fold_pub_base,
            family: LaneFamily::Batches {
                block_words: lane_words,
                dropped: family.shared.as_ref().expect("shared"),
            },
        }),
        None,
    );
    println!(
        "xmss internal node over 2 FLs (4 x {n} signatures): built in {:?} | prove {:.0} ms | \
         publics {} | proof {:.1} KiB",
        t.elapsed(),
        out.online.prove_ms,
        out.lo.public.len(),
        bincode::serialize(&out.lo.proof).unwrap().len() as f64 / 1024.0,
    );
    let app = out.app_base.expect("the node publishes the statement");
    let d = hash_pair(&fl0.digest, &fl1.digest, HashKind::Blake3);
    assert_eq!(
        out.lo.public[app..app + 2],
        digest_f128(&d)[..],
        "batch digest"
    );
    assert_eq!(
        out.lo.public[app + 8],
        span_count_word(4 * n as u128),
        "signature counter"
    );
    let lane = out.lane_acc.as_ref().expect("the lane rides the node");
    let registry = &leaves[0].inner.built.shape.registry;
    let tables =
        flock_core::aggregate::SharedTables::new(registry, family.shared.as_ref().unwrap());
    assert!(
        lane.discharge(&tables.kept_refs()) && lane.discharge_linear(&tables.linear_refs()),
        "lane boolean"
    );
    assert!(
        lane.discharge_sigma(&[&leaves[0].inner.built.shape.circuit]),
        "lane sigma"
    );
    assert!(
        lane.discharge_jagged(&[(leaves[0].inner.built.shape.circuit.digest(), &jp)]),
        "lane jagged"
    );
}

/// Phase 5 probe: a leaf with its degree-one words dropped, through the
/// child tape.
#[test]
#[ignore]
fn xmss_dropped_words_tape_probe() {
    let n = 4;
    let options = XmssOptions {
        zero_linear_words: true,
        ..XmssOptions::RECURSION_LEAF
    };
    // 2^8-row slots: the big kinds fill whole slots, and only a full slot
    // drops its words under the full-height commit.
    let circuit = XmssCircuit::with_options(n, Some(8), options);
    let (statements, sigs) = signed(n, 17);
    let leaf = build_xmss_leaf(&circuit, &statements, &sigs);
    let union = leaf.inner.union();
    let dropping = (0..union.registry().num_boolean())
        .filter(|&t| !union.zeroed_word_cols(t).is_empty())
        .count();
    println!(
        "dropping types {dropping} of {} | checks linear rows {}",
        union.registry().num_boolean(),
        union.checks_linear_rows()
    );
    assert!(dropping > 0, "the probe drops words");
    drop(union);
    let lcs = circuit.lincheck_circuits();
    let ct = ChildTape::new(&leaf.inner, XMSS_DOMAIN, &lcs);
    let nu2 = 16;
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let (mut vals, mut hints, mut consts) = (Vec::new(), Vec::new(), Vec::new());
    let b3_slot = cs.q.b3;
    let region = emit_child_region(
        &mut sb,
        &mut cs,
        b3_slot,
        &ct,
        &mut vals,
        &mut hints,
        &mut consts,
    );
    let shape2 = sb.finish().expect("the region builds");
    let hint_refs: Vec<&(dyn Any + Sync)> = hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
    let built2 = shape2.run(&vals, &hint_refs);
    check_child_region(&built2.public, &ct, &region);
}

/// How many layout columns the C claim masks, and in how many runs.
#[test]
#[ignore]
fn xmss_masked_columns_census() {
    use flock_core::matrix_fold::JaggedRowWeight;
    for (n, nu) in [(4usize, 8usize), (4, 10), (992, 15)] {
        let options = XmssOptions {
            zero_linear_words: true,
            ..XmssOptions::RECURSION_LEAF
        };
        let circuit = XmssCircuit::with_options(n, Some(nu), options);
        let (st, sg) = if n < 64 {
            signed(n, 5)
        } else {
            crate::leanxmss::circuit::keyless_batch(n, 5)
        };
        let leaf = build_xmss_leaf(&circuit, &st, &sg);
        let union = leaf.inner.union();
        let dropping = (0..union.registry().num_boolean())
            .filter(|&t| !union.zeroed_word_cols(t).is_empty())
            .count();
        for c in &leaf.inner.work.jagged.rs {
            if let JaggedRowWeight::Masked(p, z) = &c.row {
                let runs = 1 + z.windows(2).filter(|w| w[1] != w[0] + 1).count();
                println!(
                    "n {n} nu {nu}: {} of {} types drop | layout bits {} | masked columns {} in {} runs",
                    dropping,
                    union.registry().num_boolean(),
                    p.len(),
                    z.len(),
                    runs
                );
            }
        }
    }
}
