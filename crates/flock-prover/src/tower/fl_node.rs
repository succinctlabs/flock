use std::{
    any::Any,
    iter::once,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use aggregate::{
    JaggedKeyProve, JaggedKeyVerify, prove_aggregate_classes_shared_with_grinding,
    prove_aggregate_classes_with_grinding, verify_aggregate_classes_shared_with_grinding,
    verify_aggregate_classes_with_grinding,
};
use flock_core::{
    aggregate,
    aggregate::{
        Accumulator, Dropped, ElementMatrices, SharedTables, TypeMatrices, matrix_classes,
    },
    circuit::SigmaAssertion,
    element_r1cs::union::ElementAssertion,
    lincheck::{LincheckCircuit, SkipPoint},
    matrix_fold::{FoldProof, JaggedClaim, MatrixClaim},
    pcs::{LOG_PACKING, jagged::JaggedParams},
    r1cs::BlockR1cs,
    union::publics_digest,
    zerocheck::{K_SKIP, multilinear::subspace_denominator_pair},
};
use flock_merkle::hash_pair;

use crate::tower::node::digest_f128;
use crate::{
    schedule::Registry,
    tower::{
        fold_region::{ClaimLoc, locate_shared_sigma},
        fold_region_ops_bound, locate_and_pin_folds_from, locate_shared_binding,
        shared_binding_ops,
    },
};
use flock_field::PHI_8_TABLE;
use flock_transcript::transcript_record::{RecordingChallenger, StreamWord, TranscriptOp as Op};
use std::env::var;
#[cfg(test)]
use {
    crate::tower::{
        ChainLane, LaneFamily, build_chain_proof, build_node_outer_app, gates_blake3::Rng,
        test_config,
    },
    bincode::serialize,
    std::array::from_fn,
    std::cmp::Reverse,
};

#[cfg(target_arch = "aarch64")]
use crate::prover::prove_fast_ligerito_union_circuit_ag;
use crate::{
    prover::{UnionElementSlotInput, prove_fast_ligerito_union_circuit},
    r1cs_hashes::{
        blake3::{build_block_r1cs, generate_witness_batch_major_partial_into},
        fs_chain::IV,
    },
    tower::{
        BitSpreadGate, BitSpreadTable, Blake3Gate, ChainProof, ChildSlots, ChildTape, DOMAIN,
        ENV_ACC_CHAIN_WORDS, ENV_APP_WORDS, EnvTail, F128, FamilyTransposeTileGate,
        FamilyTransposeTileTable, FsChallenger, HashKind, LeafOuter, MergedChain, MixedInner,
        MixedProof, Online, PARENT, PcsParams, PowMaskGate, PowMaskTable, SLOT_WORDS, ShapeBuilder,
        SwapGate, SwapTable, TowerConfig, UnionInstance, UnionSlotProverInput, Wire, ZskipTapeRec,
        ZskipWires, assert_chain_replays, balance_extra_rows, bytes_payload_mask,
        challenge_word_locs, check_ag_skip_publics, check_child_region, check_fold_publics,
        check_jagged_fold_publics, cw, emit_ag_point_binding, emit_child_region,
        emit_fiat_shamir_chain_partitioned, emit_fold_region, emit_jagged_fold_region,
        emit_lagrange_lows, emit_recorded_pow_checks, env_acc_chain_base, env_app_base,
        envelope_shape_with_lane, expected_child_tail_schedule, flatten_ops,
        jagged_fold_region_ops, labeled_bytes_payloads, lane_block_words,
        live_element_input_from_rows, locate_and_pin_folds, locate_and_pin_jagged_folds,
        merge_chain, native_chain, outer_lanes, outer_union, outer_zerocheck_ag, pack_params,
        pack4, pack8, pad_envelope_counts, pcs_batch_for, replay_fold_endpoints,
        replay_jagged_fold_endpoints, span_count_word, steady_reps, tower_fold_grinding,
    },
    verifier::{verify_ligerito_union_circuit_ag_deferred, verify_ligerito_union_circuit_deferred},
};

/// The first-level node as a BUILDER: [`build_first_level_node`]'s output. `lo` is
/// a real, RECURSABLE [`LeafOuter`] (BLAKE3 for both the FS chain and the
/// Merkle trees), so the internal-node machinery ([`RealTape`],
/// [`build_node_outer_app`]) consumes it exactly like a leaf outer; `acc` is
/// the folded chain accumulator the node carries up; `stmt_base` locates
/// the 8-word application-statement block (h_start, h_end) in `lo.public`.
// Several fields are read only by the in-file `#[test]` benches; the lib
// unit sees them write-only.
#[cfg_attr(not(test), allow(dead_code))]
pub struct FlNode {
    pub(super) lo: LeafOuter,
    pub(super) acc: Accumulator,
    pub(super) stmt_base: usize,
    /// The published fold blocks' base: per group `[rho_col | rho_row |
    /// value]` — the accumulator claims a PARENT's lane fold connects to
    /// wire-to-wire (a prior's surface IS this published block).
    pub(super) fold_pub_base: usize,
    pub(super) h_start: [u32; 16],
    pub(super) h_end: [u32; 16],
    /// What the FL cost, split SETUP vs ONLINE — see [`Online`]. The LAST
    /// online iteration under steady repetition; everything else in the
    /// builder is pin/check scaffolding.
    pub(super) t: Online,
    /// One record per online iteration (1 + steady_reps of them).
    pub(super) onlines: Vec<Online>,
}

/// **THE FIRST-LEVEL NODE.** k ADJACENT chain proofs (each segment starts
/// at the previous one's h_end) verified deferred in ONE outer circuit —
/// k chain-tape regions on shared slots — with their boolean + sigma
/// assertions folded k→1 in-circuit (THREE fold groups; the chain class
/// has no element side), THE ADJACENCY as a wire-to-wire copy constraint
/// per seam between the children's endpoint publics, and the combined span
/// (first h_start, last h_end) published as the node's own application
/// statement. The accumulator reassembles from the public segment and
/// discharges both groups.
/// The chain layout's jagged params — the count win's per-digest table
/// owner for the lane, rebuilt exactly as the opening verifier reads it.
pub(super) fn chain_jagged_params(cp: &ChainProof) -> JaggedParams {
    let u = UnionInstance::new(
        &cp.inner.built.shape.registry,
        cp.inner.built.shape.counts.clone(),
    );
    JaggedParams::from_heights(
        &u.jagged_heights(),
        u.n_log(),
        cp.inner.commitment.params.m - LOG_PACKING,
    )
}

/// The BLAKE3 block R1CS per nu, cached process-wide. The base matrices
/// are already shared by every `build_block_r1cs` call (see
/// `r1cs_hashes::blake3`); what this cache adds is the BLOCK, and with it
/// the lazily built CSC lincheck circuit (~178 MiB) that lives in the
/// block's `OnceLock`. Every chain leaf (its own nu), every FL and every
/// recursion node (the envelope nu*) reads one circuit through here; each
/// node's [`LeafOuter`] holds the `Arc` rather than a private copy.
pub(super) fn chain_blake_r1cs(nu: usize) -> Arc<BlockR1cs> {
    type Cache = Mutex<Vec<(usize, Arc<BlockR1cs>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, r)) = g.iter().find(|(k, _)| *k == nu) {
        return r.clone();
    }
    let r = Arc::new(build_block_r1cs(nu));
    g.push((nu, r.clone()));
    r
}

/// The FL's per-statement tape source, bare: ONE recording deferred verify
/// of a child. The pin/locate scaffolding is per-shape and lives in
/// [`ChildTape::new`]; this is what an online iteration re-pays (results
/// discarded — identical by determinism).
pub(super) fn record_child_inner_verify(
    inner: &MixedInner,
    domain: &'static [u8],
    lcs: &[&dyn LincheckCircuit],
) {
    let union = inner.union();
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
    match &inner.proof {
        MixedProof::Rs(p) => verify_ligerito_union_circuit_deferred(
            &union,
            &inner.built.shape.circuit,
            &inner.built.witness.public,
            lcs,
            &inner.commitment,
            p,
            &inner.pcs,
            &mut rec,
        ),
        MixedProof::Ag(p) => verify_ligerito_union_circuit_ag_deferred(
            &union,
            &inner.built.shape.circuit,
            &inner.built.witness.public,
            lcs,
            &inner.commitment,
            p,
            &inner.pcs,
            &mut rec,
        ),
    }
    .expect("the child verifies (recorded)");
}

/// A first-level node's leaf family: beyond the leaves' proofs, what the
/// fold and the child tapes need — the leaves' transcript domain and, per
/// boolean registry type in registry order, the base matrices and the
/// lincheck circuit.
pub(super) struct FlFamily<'a> {
    pub(super) domain: &'static [u8],
    pub(super) mats: Vec<TypeMatrices<'a>>,
    pub(super) circs: Vec<&'a dyn LincheckCircuit>,
    /// The tower's ACC_CHAIN width ([`EnvShape::acc_chain_words`]): fixed,
    /// or `None` to fit the lane this node folds ([`lane_block_words`]).
    pub(super) lane_words: Option<usize>,
    /// Shared binding (`prove_aggregate_classes_shared_with_grinding`): the
    /// fold binds each child's boolean assertion once rather than every
    /// per-type claim — the win grows with the type count — and folds per
    /// matrix class under the leaf union's dropped words. `None`: per-claim
    /// binding, per type.
    pub(super) shared: Option<Dropped>,
}

/// A first-level node's application statement, published in the envelope's
/// 10-word app block. Words 8 and 9 are the counter and its base in both
/// (see [`ENV_APP_WORDS`]).
pub(super) enum FlApp {
    /// Adjacent hash-chain segments: the span `(h_start, h_end)` in words
    /// 0..8, each child's `h_end` copy-constrained to the next `h_start`;
    /// the counter counts compressions.
    ChainSpan {
        h_start: [u32; 16],
        h_end: [u32; 16],
        per_leaf: usize,
    },
    /// Signature batches: `D = hash_pair(D_0, D_1)` over the leaves'
    /// publics digests in words 0..2 (2..8 zero); the counter counts
    /// signatures.
    Batches { per_leaf: usize },
}

/// What [`build_first_level_core`] hands back; the per-family node types
/// add their statement.
pub(super) struct FlCore {
    pub(super) lo: LeafOuter,
    pub(super) acc: Accumulator,
    pub(super) stmt_base: usize,
    pub(super) fold_pub_base: usize,
    pub(super) t: Online,
    pub(super) onlines: Vec<Online>,
}

pub fn build_first_level_node(cfg: TowerConfig, cp0: &ChainProof, cp1: &ChainProof) -> FlNode {
    build_first_level_node_k(cfg, &[cp0, cp1])
}

/// The 2-ary first-level node: two adjacent chain proofs verified deferred
/// in ONE outer, their assertions folded 2→1 per group, adjacency as one
/// four-word seam, the app statement the combined span. The `cps` slice is
/// the arity LEVER, but today it is pinned to exactly two children — the
/// split-BLAKE slot assignment (`ChildSlots::new_env` sets `b3_alt` for
/// child 1 only) has no slots for a third child.
pub fn build_first_level_node_k(cfg: TowerConfig, cps: &[&ChainProof]) -> FlNode {
    const FL_DOMAIN: &[u8] = b"flock-chain-fl-node-v0";
    let cp0 = cps[0];
    let cp_last = cps[cps.len() - 1];
    // Each child CONTINUES the chain: its h_start IS the previous h_end.
    for pair in cps.windows(2) {
        assert_eq!(pair[1].h_start, pair[0].h_end, "the segments are adjacent");
    }
    let per_leaf = cp0.n_blocks;
    assert!(
        cps.iter().all(|cp| cp.n_blocks == per_leaf),
        "one leaf size per FL"
    );
    assert_eq!(
        cp0.inner.built.shape.registry.num_boolean(),
        1,
        "one boolean type (blake3)"
    );
    let blake_r1cs = chain_blake_r1cs(cp0.inner.nu);
    let family = FlFamily {
        domain: DOMAIN,
        mats: vec![(&blake_r1cs.a_0, &blake_r1cs.b_0)],
        circs: vec![blake_r1cs.csc_lincheck_circuit()],
        lane_words: Some(ENV_ACC_CHAIN_WORDS),
        shared: None,
    };
    let inners: Vec<&MixedInner> = cps.iter().map(|cp| &cp.inner).collect();
    let core = build_first_level_core(
        cfg,
        FL_DOMAIN,
        &inners,
        &family,
        &FlApp::ChainSpan {
            h_start: cp0.h_start,
            h_end: cp_last.h_end,
            per_leaf,
        },
    );
    assert_eq!(
        cp_last.h_end,
        native_chain(
            &cp0.h_start,
            cps.iter().map(|cp| cp.inner.built.shape.counts[0]).sum(),
        ),
        "the combined span IS the concatenated chain"
    );
    FlNode {
        lo: core.lo,
        acc: core.acc,
        stmt_base: core.stmt_base,
        fold_pub_base: core.fold_pub_base,
        h_start: cp0.h_start,
        h_end: cp_last.h_end,
        t: core.t,
        onlines: core.onlines,
    }
}

/// The committed content of an outer, per type, descending: what the
/// m* floor and the lane pin have to hold (`FL_CENSUS=1`).
pub(super) fn content_census(label: &str, registry: &Registry, counts: &[usize]) {
    let u = UnionInstance::new(registry, counts.to_vec());
    let dw = u.dense_words();
    let lane = 1usize << (29 - 7 - 5);
    println!(
        "  [{label} content] dense words {dw} = {:.2} lanes of 2^{} words (pin 31)",
        dw as f64 / lane as f64,
        29 - 7 - 5,
    );
    let mut per: Vec<(usize, usize, usize, usize)> = registry
        .types()
        .iter()
        .zip(counts)
        .enumerate()
        .map(|(i, (ty, &n_t))| {
            let cols = ty.useful_bits.div_ceil(128).min(1usize << (ty.k_log - 7));
            (cols * n_t, i, cols, n_t)
        })
        .collect();
    per.sort_by_key(|p| std::cmp::Reverse(p.0));
    for &(words, i, cols, rows) in per.iter().take(24) {
        println!(
            "    type {i:2}: {words:>8} words ({cols:3} cols x {rows:6} rows) = {:.2} lanes",
            words as f64 / lane as f64
        );
    }
}

/// The app block a first-level node over `inners` publishes, natively —
/// the reference its builder holds the published words against.
pub(super) fn fl_app_words(app: &FlApp, inners: &[&MixedInner]) -> Vec<F128> {
    let k_ary = inners.len();
    let (mut words, per_leaf) = match app {
        FlApp::ChainSpan {
            h_start,
            h_end,
            per_leaf,
        } => {
            let span: Vec<F128> = (0..4)
                .map(|j| pack4(h_start[4 * j..4 * j + 4].try_into().unwrap()))
                .chain((0..4).map(|j| pack4(h_end[4 * j..4 * j + 4].try_into().unwrap())))
                .collect();
            (span, *per_leaf)
        }
        FlApp::Batches { per_leaf } => {
            let d = hash_pair(
                &publics_digest(&inners[0].built.witness.public),
                &publics_digest(&inners[1].built.witness.public),
                HashKind::Blake3,
            );
            let mut words = digest_f128(&d).to_vec();
            words.resize(8, F128::ZERO);
            (words, *per_leaf)
        }
    };
    words.push(span_count_word((per_leaf * k_ary) as u128));
    words.push(span_count_word(per_leaf as u128));
    words
}

/// **The first-level node, over any leaf family**: `inners` (exactly two,
/// one circuit) verified deferred in ONE envelope outer — a child-tape
/// region each on shared slots — their boolean, sigma and jagged
/// assertions folded 2→1 in-circuit (two fold groups per boolean type,
/// then sigma, then jagged), the fold surfaces copy-constrained to the
/// regions' wires, and `app` published as the node's statement.
pub(super) fn build_first_level_core(
    cfg: TowerConfig,
    fl_domain: &'static [u8],
    inners: &[&MixedInner],
    family: &FlFamily<'_>,
    app: &FlApp,
) -> FlCore {
    let k_ary = inners.len();
    assert_eq!(
        k_ary, 2,
        "split-BLAKE recursion supports exactly two children"
    );
    let inner0 = inners[0];
    for inner in &inners[1..] {
        assert_eq!(
            inner0.built.shape.circuit.digest(),
            inner.built.shape.circuit.digest(),
            "one leaf circuit digest, every child"
        );
    }

    let registry = &inner0.built.shape.registry;
    let n_types = registry.num_boolean();
    assert_eq!(family.mats.len(), n_types, "matrices per boolean type");
    assert_eq!(family.circs.len(), n_types, "a lincheck circuit per type");
    assert!(
        registry.element_types().is_empty(),
        "the leaf class has no element side"
    );
    let bool_asserts: Vec<_> = inners
        .iter()
        .map(|inner| inner.work.boolean.clone().expect("child boolean work"))
        .collect();
    let sigmas: Vec<_> = inners.iter().map(|inner| inner.sigma.clone()).collect();

    // ---- the native fold: boolean + sigma, NO element groups, NO priors ----
    let mats = &family.mats[..];
    let el_mats: [ElementMatrices; 0] = [];
    let el_asserts: [(&UnionInstance<'_>, ElementAssertion); 0] = [];
    let circs = &family.circs[..];
    // THE JAGGED GROUP (the count win): the children's W-claims fold under
    // the leaf digest — the layout is a shape constant of the ONE leaf
    // circuit, rebuilt here exactly as the opening verifier reads it.
    let leaf_circuit = &inner0.built.shape.circuit;
    let leaf_digest = leaf_circuit.digest();
    let leaf_union_j = inner0.union();
    let leaf_params_j = JaggedParams::from_heights(
        &leaf_union_j.jagged_heights(),
        leaf_union_j.n_log(),
        inner0.commitment.params.m - LOG_PACKING,
    );
    let jags: Vec<_> = inners.iter().map(|inner| &inner.work.jagged).collect();
    let jagged_p: Vec<JaggedKeyProve<'_>> = vec![(leaf_digest, &leaf_params_j, jags.to_vec())];
    let jagged_v: Vec<JaggedKeyVerify<'_>> = vec![(leaf_digest, jags.to_vec())];
    let mut chp = FsChallenger::with_chained_blake3(fl_domain);
    let sigma_key = [(leaf_circuit, sigmas.iter().collect())];
    let grinding = tower_fold_grinding(cfg);
    let (agg, acc_p) = match &family.shared {
        Some(dropped) => prove_aggregate_classes_shared_with_grinding(
            registry,
            dropped,
            mats,
            circs,
            &bool_asserts,
            &el_mats,
            &el_asserts,
            &sigma_key,
            &jagged_p,
            &[],
            grinding,
            &mut chp,
        ),
        None => prove_aggregate_classes_with_grinding(
            registry,
            mats,
            circs,
            &bool_asserts,
            &el_mats,
            &el_asserts,
            &sigma_key,
            &jagged_p,
            &[],
            grinding,
            &mut chp,
        ),
    }
    .expect("the first-level fold proves");
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(fl_domain));
    let acc_v = match &family.shared {
        Some(dropped) => verify_aggregate_classes_shared_with_grinding(
            registry,
            dropped,
            &bool_asserts,
            &el_asserts,
            &sigma_key,
            &jagged_v,
            &[],
            &agg,
            grinding,
            &mut rec,
        ),
        None => verify_aggregate_classes_with_grinding(
            registry,
            &bool_asserts,
            &el_asserts,
            &sigma_key,
            &jagged_v,
            &[],
            &agg,
            grinding,
            &mut rec,
        ),
    }
    .expect("the first-level fold verifies");
    assert_eq!(acc_p, acc_v, "prover and verifier accumulators agree");
    assert!(acc_v.per_element.is_empty(), "no element group accumulated");
    // The boolean fold groups: one per type, or one per matrix class under
    // shared binding — each group's claims are child-major, then
    // type-ascending within the group. Under shared binding the classes
    // that drop words under η add a linear-identity group each.
    let groups: Vec<Vec<usize>> = match &family.shared {
        Some(dropped) => {
            let classes = matrix_classes(registry, &dropped.zeroed);
            let n_classes = classes.iter().max().map_or(0, |&c| c + 1);
            (0..n_classes)
                .map(|c| (0..n_types).filter(|&t| classes[t] == c).collect())
                .collect()
        }
        None => (0..n_types).map(|t| vec![t]).collect(),
    };
    let linear: Vec<usize> = match &family.shared {
        Some(d) if d.checked => (0..groups.len())
            .filter(|&g| !d.zeroed[groups[g][0]].is_empty())
            .collect(),
        _ => Vec::new(),
    };
    // The boolean groups' discharge: the kept blocks and the linear
    // identities under shared binding, the base blocks otherwise.
    let shared_tables = family
        .shared
        .as_ref()
        .map(|d| SharedTables::new(registry, d));
    let discharges_boolean = |acc: &Accumulator| -> bool {
        match &shared_tables {
            Some(t) => acc.discharge(&t.kept_refs()) && acc.discharge_linear(&t.linear_refs()),
            None => acc.discharge(mats),
        }
    };
    assert!(discharges_boolean(&acc_v), "the boolean groups discharge");
    assert!(
        acc_v.discharge_sigma(&[leaf_circuit]),
        "the sigma group discharges against the ONE leaf circuit"
    );
    assert_eq!(acc_v.jagged.len(), 1, "one jagged key: the leaf layout");
    assert!(
        acc_v.discharge_jagged(&[(leaf_digest, &leaf_params_j)]),
        "the folded jagged entry discharges against the leaf layout"
    );

    // The folds' claim lists — no priors, so fresh only: per boolean group
    // its A and B folds, then sigma. Claim `k·|members| + j` of a group is
    // child `k`'s claim for the group's `j`-th type.
    let n_priors = 0usize;
    let bc: Vec<_> = bool_asserts.iter().map(|a| a.claims(registry)).collect();
    let mut fold_claims: Vec<Vec<MatrixClaim>> = Vec::with_capacity(2 * groups.len() + 1);
    let mut fold_proofs: Vec<&FoldProof> = Vec::with_capacity(2 * groups.len() + 1);
    for (g, members) in groups.iter().enumerate() {
        let side = |pick: fn(&(MatrixClaim, MatrixClaim)) -> &MatrixClaim| -> Vec<MatrixClaim> {
            bc.iter()
                .flat_map(|c| members.iter().map(move |&t| pick(&c[t]).clone()))
                .collect()
        };
        fold_claims.push(side(|p| &p.0));
        fold_claims.push(side(|p| &p.1));
        fold_proofs.push(&agg.folds[g].0);
        fold_proofs.push(&agg.folds[g].1);
    }
    // The linear-identity groups follow, same claim order.
    let lc: Vec<_> = bool_asserts
        .iter()
        .map(|a| a.linear_claims(registry))
        .collect();
    for (lg, &g) in linear.iter().enumerate() {
        fold_claims.push(
            lc.iter()
                .flat_map(|c| {
                    groups[g]
                        .iter()
                        .map(move |&t| c[t].clone().expect("a linear member"))
                })
                .collect(),
        );
        fold_proofs.push(&agg.linear_folds[lg]);
    }
    let n_ab = 2 * groups.len();
    let sig_g = n_ab + linear.len();
    fold_claims.push(sigmas.iter().flat_map(|s| s.claims()).collect());
    fold_proofs.push(&agg.sigma_folds[0]);
    for claims in &fold_claims[..sig_g] {
        assert_eq!(claims[0].row.low.len(), 64, "fresh lagrange low");
        assert_eq!(claims[0].col.low.len(), 64, "fresh z_partial low");
    }
    // Fold `fi`'s member types: an A/B fold's group, or a linear group's.
    let fold_members = |fi: usize| -> &[usize] {
        if fi < n_ab {
            &groups[fi / 2]
        } else {
            &groups[linear[fi - n_ab]]
        }
    };
    assert_eq!(
        fold_claims[sig_g][0].row.low.len(),
        1,
        "sigma claims are eq"
    );

    // ---- the fold tape, pinned op-for-op ----
    let t_shape = rec.shape();
    let ops = flatten_ops(t_shape.ops());
    if var("FL_CENSUS").is_ok() {
        let words = |g: &[MatrixClaim]| -> usize {
            g.iter()
                .map(|c| {
                    c.row.low.len() + c.row.point.len() + c.col.low.len() + c.col.point.len() + 1
                })
                .sum()
        };
        let bool_words: usize = fold_claims[..sig_g].iter().map(|g| words(g)).sum();
        let (mut obs, mut sq, mut pows) = (0usize, 0usize, 0usize);
        for op in ops.iter() {
            match op {
                Op::ObserveScalar => obs += 1,
                Op::ObserveSlice(n) => obs += n,
                Op::SqueezeScalar => sq += 1,
                Op::SqueezeSlice(n) => sq += n,
                Op::Pow { .. } => pows += 1,
                _ => {}
            }
        }
        println!(
            "  [fold tape] observed words {obs} (boolean claims {bool_words} in {} groups; \
             sigma claims {} = {} words) | squeezed {sq} | pow {pows}",
            sig_g,
            fold_claims[sig_g].len(),
            words(&fold_claims[sig_g]),
        );
    }
    let vals_rec = rec.values();
    let chals = rec.challenges();
    let mut want: Vec<Op> = vec![
        Op::Label(b"flock-aggregate-v0".to_vec()),
        Op::ObserveBytes(32),
        Op::ObserveBytes(1),
    ];
    let n_uni = fold_claims.len() - 1;
    let sigma_refs: Vec<&SigmaAssertion> = sigmas.iter().collect();
    let n_bound: Vec<usize> = if family.shared.is_some() {
        want.extend(shared_binding_ops(&bool_asserts, &sigma_refs));
        (0..n_uni)
            .map(|fi| k_ary * fold_members(fi).len())
            .collect()
    } else {
        vec![0; n_uni]
    };
    want.extend(fold_region_ops_bound(cfg, &fold_claims[..n_uni], &n_bound));
    // The sigma group binds per key now (wall 3): its label + digest
    // precede the fold, exactly as the jagged groups bind.
    want.push(Op::Label(b"flock-aggregate-sigma-v1".to_vec()));
    want.push(Op::ObserveBytes(32));
    // Under shared binding the fresh sigma claims were bound by their
    // assertions' surfaces.
    let n_sigma_bound = if family.shared.is_some() {
        fold_claims[n_uni].len()
    } else {
        0
    };
    want.extend(fold_region_ops_bound(
        cfg,
        &fold_claims[n_uni..],
        &[n_sigma_bound],
    ));
    // The jagged group rides the SAME tape after the uniform folds.
    let jagged_keys: Vec<([u8; 32], Vec<JaggedClaim>)> = vec![(
        leaf_digest,
        jags.iter()
            .flat_map(|a| a.claims().into_iter().cloned())
            .collect(),
    )];
    want.extend(jagged_fold_region_ops(cfg, &jagged_keys));
    assert_eq!(ops, want.as_slice(), "the first-level fold tape shape");
    assert_eq!(
        rec.payloads()[0],
        registry.digest(),
        "bind: registry digest"
    );
    assert_eq!(rec.payloads()[1], vec![0u8], "bind: prior count 0");
    let (locs, vcur, ccur) = if family.shared.is_some() {
        let (per_type, v_sigma) = locate_shared_binding(&bool_asserts, registry, vals_rec, 0);
        let (sigma_locs, v0) = locate_shared_sigma(&sigma_refs, vals_rec, v_sigma);
        let mut prebound: Vec<Vec<ClaimLoc>> = Vec::with_capacity(sig_g);
        for fi in 0..n_uni {
            let members = fold_members(fi);
            prebound.push(
                (0..k_ary)
                    .flat_map(|k| members.iter().map(move |&t| (t, k)))
                    .map(|(t, k)| {
                        let (a, b, l) = &per_type[t][k];
                        match (fi < n_ab, fi % 2) {
                            (true, 0) => a.clone(),
                            (true, _) => b.clone(),
                            (false, _) => l.clone().expect("a linear member"),
                        }
                    })
                    .collect(),
            );
        }
        prebound.push(sigma_locs.into_iter().flatten().collect());
        locate_and_pin_folds_from(&fold_claims, &fold_proofs, vals_rec, chals, v0, &prebound)
    } else {
        locate_and_pin_folds(&fold_claims, &fold_proofs, vals_rec, chals)
    };
    let jfps: Vec<&FoldProof> = agg.jagged_folds.iter().collect();
    let jlocs = locate_and_pin_jagged_folds(
        &jagged_keys,
        &jfps,
        vals_rec,
        chals,
        rec.payloads(),
        &labeled_bytes_payloads(&ops, b"flock-aggregate-jagged-v0"),
        vcur,
        ccur,
    );
    let outs = replay_fold_endpoints(&locs, vals_rec, chals);
    for g in 0..groups.len() {
        assert_eq!(outs[2 * g], acc_v.per_type[g].0, "boolean A accumulator");
        assert_eq!(
            outs[2 * g + 1],
            acc_v.per_type[g].1,
            "boolean B accumulator"
        );
    }
    for lg in 0..linear.len() {
        assert_eq!(
            outs[n_ab + lg],
            acc_v.per_linear[lg],
            "linear-identity accumulator"
        );
    }
    let (sig_digest, sig_claim) = acc_v.sigma.first().expect("sigma accumulated");
    assert_eq!(outs[sig_g], *sig_claim, "sigma accumulator");
    assert_eq!(
        *sig_digest, leaf_digest,
        "sigma keys by the leaf circuit digest"
    );
    let jouts = replay_jagged_fold_endpoints(&jlocs, vals_rec, chals);
    assert_eq!(
        jouts[0], acc_v.jagged[0].1,
        "the jagged entry from located words"
    );

    // ---- the child tapes ----
    let tapes: Vec<ChildTape> = inners
        .iter()
        .map(|inner| ChildTape::new(inner, family.domain, circs))
        .collect();
    assert!(
        tapes.iter().all(|t| t.el.is_none()),
        "boolean-only children"
    );

    // ---- the outer: k chain-tape regions + the fold region + adjacency ----
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
            &t_shape.stream_words_duplex(fl_domain),
            rec.values(),
            rec.payloads(),
        );
        assert_chain_replays(&ops, &trace, chals);

        let env = envelope_shape_with_lane(
            family
                .lane_words
                .unwrap_or_else(|| lane_block_words(&acc_v)),
        );
        let split_b3 = tapes.len() == 2;
        let (fold_b3_primary_rows, b3_rows) = if split_b3 {
            let a = tapes[0].b3_rows;
            let b = tapes[1].b3_rows;
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
                tapes.iter().map(|t| t.b3_rows).sum::<usize>() + trace.rows.len(),
            )
        };
        let nu2_content = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(7);
        // THE ENVELOPE (task 7b): a first-level node is an internal node's
        // CHILD, so its proof must carry the same geometry every other
        // envelope outer does — nu*, the canonical type set at counts*, the
        // padded public segment and the m* dense floor. Then a parent's walk
        // over an FL child is row-identical to its walk over an internal
        // child, which is what makes ONE internal circuit serve every level.
        assert!(
            nu2_content <= env.nu,
            "FL content nu {nu2_content} exceeds the envelope nu* {}",
            env.nu
        );
        let nu2 = env.nu;
        let t_build = Instant::now();
        let mut sb = ShapeBuilder::new(nu2);
        let spread_own2 = tapes.iter().map(|t| t.spread_w).max().expect("children");
        assert!(
            spread_own2 <= env.spread_w,
            "chain-child ladder depth {spread_own2} exceeds the envelope spread width {}",
            env.spread_w
        );
        let spread_w2 = env.spread_w;
        let mut cs = ChildSlots::new_env(&mut sb, nu2, &env);
        let mut vals: Vec<F128> = Vec::new();
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let mut consts: Vec<(F128, Wire)> = Vec::new();
        // The chain-child regions are independent gate subgraphs (each
        // reads only its own tape's inputs; the fold region joins them
        // AFTER), so they are declared as islands and the fill plan
        // evaluates them concurrently. A cross-island read fails plan
        // compilation — the independence is checked, not assumed.
        let regions: Vec<_> = tapes
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let isl = sb.begin_island();
                let b3_slot = match (i, cs.q.b3_alt) {
                    (0, _) => cs.q.b3,
                    (1, Some(slot)) => slot,
                    (_, None) => cs.q.b3,
                    _ => panic!("split-BLAKE recursion supports exactly two children"),
                };
                let r = emit_child_region(
                    &mut sb,
                    &mut cs,
                    b3_slot,
                    t,
                    &mut vals,
                    &mut hints,
                    &mut consts,
                );
                sb.end_island(isl);
                assert_eq!(
                    r.tail_schedule,
                    expected_child_tail_schedule(t),
                    "child {i}'s tail publishes on its declared schedule"
                );
                r
            })
            .collect();
        let b3s = cs.q.b3;
        let macs = cs.macs;
        let mrs = cs.mrs;
        let (pfslot, pf_w) = regions[0].pf;
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
        let pub_payloads = bytes_payload_mask(&ops);
        let (chain_outs, ww) = emit_fiat_shamir_chain_partitioned(
            &mut sb,
            b3s,
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
            b3s,
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
            macs,
            mrs,
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
            macs,
            mrs,
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
        // THE POINTS-CONNECT (the count win's identity bind): every
        // absorbed claim surface in the jagged fold is a child-region
        // wire — the VALUE (the wire the anchor expect consumed), σ (the
        // child's anchor round squeezes), the row identities (z_col
        // wires / γ_pd squeezes / zw-ow constants), and the structural
        // words (tags, the shape header, Combo addresses) pinned to
        // shared constant publics the checker validates. With identity
        // AND value bound, the folded entry provably says "Ĵ at the
        // identity the child's verification determined equals the value
        // its anchor expect consumed" — a cooked-identity substitution
        // has nowhere to live.
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
            let loc = &jlocs[0];
            let mut ci = 0usize;
            for rk in &regions {
                for (li, &jw) in rk.jag_w.iter().enumerate() {
                    let cl = &loc.claims[ci];
                    sb.connect(wv(cl.val_v), jw);
                    for j in 0..loc.n_col {
                        sb.connect(wv(cl.col_v + j), rk.jag_sig_w[j]);
                    }
                    if let Some(zeroed) = &cl.masked {
                        // A masked claim: its tag and mask size pin to
                        // constants (the mask itself is the circuit's).
                        let tag = cw_j(
                            &mut sb,
                            &mut vals,
                            &mut jag_const_rec,
                            F128::new(2, cl.row_pt.1 as u64),
                        );
                        sb.connect(wv(cl.row_scale_v - 1), tag);
                        let size = cw_j(
                            &mut sb,
                            &mut vals,
                            &mut jag_const_rec,
                            F128::new(zeroed.len() as u64, 0),
                        );
                        sb.connect(wv(cl.row_scale_v), size);
                        for j in 0..cl.row_pt.1 {
                            sb.connect(wv(cl.row_pt.0 + j), rk.jag_row_w[li][j]);
                        }
                    } else if cl.terms.is_empty() {
                        let tag = cw_j(
                            &mut sb,
                            &mut vals,
                            &mut jag_const_rec,
                            F128::new(0, cl.row_pt.1 as u64),
                        );
                        sb.connect(wv(cl.row_scale_v - 1), tag);
                        // A FRESH claim is live: its zero-claim scale is 1.
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
            // The group's shape header word binds too.
            let header_v = loc.hdr_v;
            let hw = cw_j(
                &mut sb,
                &mut vals,
                &mut jag_const_rec,
                F128::new(loc.k_row as u64, loc.claims.len() as u64),
            );
            sb.connect(wv(header_v), hw);
        }

        // ---- the connects: fold surfaces == child-region wires ----
        // The RS lows machinery (λ table + denominator + zero anchor) is
        // emitted only when an RS child consumes it; AG children take the
        // Tier-0 published surface instead (docs/ag-recursion-plan.md).
        let any_rs = tapes
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
            vals.push(F128::ZERO);
            let lag_zassert = sb.public_input();
            (lam_base, lam_w, deninv_w, lag_zassert)
        });
        // Per AG child, the Tier-0 public block's base — the layout
        // [`check_ag_skip_publics`] walks.
        let mut ag_pub_bases: Vec<Option<usize>> = Vec::new();
        for (k, (tk, rk)) in tapes.iter().zip(&regions).enumerate() {
            // Child k's first boolean claim: group 0's first member.
            let c0 = n_priors + k * groups[0].len();
            // Basis-generic native pre-assert: the fold's absorbed lows
            // ARE the skip functional at the child's own z_skip point.
            assert_eq!(
                &fold_claims[0][c0].row.low[..],
                &tk.bool_assert.z_skip.weights(K_SKIP)[..],
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
                        sb.connect(lw2, wv(locs[0].claims[c0].row_low_v + j));
                    }
                    ag_pub_bases.push(None);
                }
                (ZskipTapeRec::Ag { seed_ch, .. }, ZskipWires::Ag { seed_w, nonce_w }) => {
                    // TIER 1 (phase D): publish [seed₂, nonce, point₅,
                    // lows₆₄] — seed/nonce/lows wire-connected as before —
                    // and BIND the point in-circuit
                    // ([`emit_ag_point_binding`]: the two BLAKE3 decode
                    // rows, the fused-PoW row, the fiber algebra over the
                    // published coordinate wires). The native checker keeps
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
                    let nonce = match &tk.inner.proof {
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
                    let SkipPoint::Ag(pt) = tk.bool_assert.z_skip else {
                        unreachable!("an AG tape carries an AG skip point")
                    };
                    let pt_w: [Wire; 5] = [pt.x, pt.y, pt.z1, pt.z2, pt.z3].map(|c| {
                        vals.push(c);
                        sb.public_input()
                    });
                    for (j, &lv) in fold_claims[0][c0].row.low.iter().enumerate() {
                        vals.push(lv);
                        let lw2 = sb.public_input();
                        sb.connect(lw2, wv(locs[0].claims[c0].row_low_v + j));
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
                        inners[k].ag_round_one_bits(),
                        &mut vals,
                        &mut consts,
                        zw,
                        ow,
                    );
                    ag_pub_bases.push(Some(base));
                }
                _ => unreachable!("the region's zskip wires match the tape flavor"),
            }
            // Native pre-asserts, then the wire connects — all static
            // Product-GKR evaluations, boolean points + z_partial lows.
            let native_structure = tk.sigma_native.claims();
            for (j, claim) in native_structure.iter().enumerate() {
                assert_eq!(
                    &fold_claims[sig_g][n_priors + native_structure.len() * k + j],
                    claim,
                    "circuit-structure claim {j}"
                );
            }
            for (g, members) in groups.iter().enumerate() {
                for (j, &t) in members.iter().enumerate() {
                    let ci = n_priors + k * members.len() + j;
                    let fresh = &fold_claims[2 * g][ci];
                    let inner_b = fresh.row.point.len();
                    assert_eq!(
                        &fresh.row.point[..],
                        &tk.bool_assert.x_inner_rest[..inner_b],
                        "boolean row point is x_inner_rest's head"
                    );
                    assert_eq!(
                        &fresh.col.point[..],
                        &tk.bool_assert.rr[..inner_b],
                        "boolean col point is rr's head"
                    );
                    assert_eq!(
                        &fresh.col.low[..],
                        &tk.bool_assert.z_partial[..],
                        "boolean col low is z_partial"
                    );
                    assert_eq!(fresh.value, tk.bool_assert.evals[t].0);
                    assert_eq!(fold_claims[2 * g + 1][ci].value, tk.bool_assert.evals[t].1);
                }
            }

            // Circuit structure: every native claim is fully wire-bound.
            for (j, (row_w, col_w, value_w)) in rk.structure_claim_w.iter().enumerate() {
                let cl = &locs[sig_g].claims[n_priors + rk.structure_claim_w.len() * k + j];
                assert_eq!(cl.row_pt_n, row_w.len());
                // A shared-bound claim's lows and constant column tail are
                // baked; its stream words connect like any claim's.
                match &cl.consts {
                    Some(tail) => assert_eq!(cl.col_pt_n + tail.len(), col_w.len()),
                    None => {
                        sb.connect(wv(cl.row_low_v), ow);
                        sb.connect(wv(cl.col_low_v), ow);
                        assert_eq!(cl.col_pt_n, col_w.len());
                    }
                }
                for (j, &w) in row_w.iter().enumerate() {
                    sb.connect(wv(cl.row_pt_v + j), w);
                }
                for (j, &w) in col_w.iter().take(cl.col_pt_n).enumerate() {
                    sb.connect(wv(cl.col_pt_v + j), w);
                }
                sb.connect(wv(cl.value_v), *value_w);
            }
            // boolean A/B per type: batch-major x_inner_rest mapping, rr
            // reversed, z_partial word-for-word; then each value to its
            // reported evaluation; then the lagrange lows. Child k's claims
            // in fold `fi`, with the type each names.
            assert_eq!(rk.mat_eval_w.len(), n_types, "child Boolean type count");
            let child_claims: Vec<(usize, usize, usize)> = (0..sig_g)
                .flat_map(|fi| {
                    let members = fold_members(fi);
                    let base = n_priors + k * members.len();
                    members
                        .iter()
                        .enumerate()
                        .map(move |(j, &t)| (fi, base + j, t))
                })
                .collect();
            for &(fi, ci, _) in &child_claims {
                let cl = &locs[fi].claims[ci];
                {
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
            }
            for &(fi, ci, t) in &child_claims {
                let cl = &locs[fi].claims[ci];
                let value_w = if fi >= n_ab {
                    rk.lin_eval_w[t].expect("the region's linear value")
                } else if fi % 2 == 0 {
                    rk.mat_eval_w[t].0
                } else {
                    rk.mat_eval_w[t].1
                };
                sb.connect(wv(cl.value_v), value_w);
            }
            // Every claim's lagrange lows are the first's — one published
            // copy binds them all (one stream copy, under shared binding).
            let first_low = locs[0].claims[n_priors + k * groups[0].len()].row_low_v;
            for &(fi, ci, _) in &child_claims {
                let cl = &locs[fi].claims[ci];
                {
                    if cl.row_low_v != first_low {
                        for j in 0..cl.row_low_n {
                            sb.connect(wv(cl.row_low_v + j), wv(first_low + j));
                        }
                    }
                }
            }
        }

        // THE APPLICATION STATEMENT. counts* + publics*: an FL node declares
        // the same count vector and segment length every other envelope
        // outer does, and both the app block and the accumulator claims ride
        // the envelope's fixed TAIL.
        let (mut app_w, per_leaf): (Vec<Wire>, usize) = match app {
            FlApp::ChainSpan { per_leaf, .. } => {
                // THE ADJACENCY: each h_end == the next h_start, wire to
                // wire. The chain statement is 11 words: [iv0, iv1, params,
                // h_start x4 | h_end x4 published last]. The children's
                // publics are witness wires here, so adjacency is four copy
                // constraints per seam, and the node's own statement is the
                // combined span (the first child's h_start, the last
                // child's h_end).
                for rk in &regions {
                    assert_eq!(rk.child_pub_w.len(), 11, "the chain statement is 11 words");
                }
                for pair in regions.windows(2) {
                    for j in 0..4 {
                        sb.connect(pair[0].child_pub_w[11 - 4 + j], pair[1].child_pub_w[3 + j]);
                    }
                }
                let span = (0..4)
                    .map(|j| regions[0].child_pub_w[3 + j])
                    .chain((0..4).map(|j| regions[k_ary - 1].child_pub_w[11 - 4 + j]))
                    .collect();
                (span, *per_leaf)
            }
            FlApp::Batches { per_leaf } => {
                // THE BATCH DIGEST: each child region re-derived its
                // publics digest in-circuit and bound it to its transcript;
                // the node's statement hashes the two as a BLAKE3 parent
                // (`hash_pair`) — the leaf order is the statement's order.
                assert_eq!(k_ary, 2, "a batch digest pairs two children");
                let params = cw(&mut sb, &mut vals, &mut consts, pack_params(0, 64, PARENT));
                let (d0, d1) = (regions[0].pub_digest_w, regions[1].pub_digest_w);
                let out = sb.gate(
                    cs.q.b3,
                    &[iv2[0], iv2[1], d0[0], d0[1], d1[0], d1[1], params],
                );
                let mut words = vec![out[0], out[1]];
                words.resize(8, zw);
                (words, *per_leaf)
            }
        };
        // THE INHERITABLE ACCUMULATOR: per fold the deltas + the claim
        // `[rho_col | rho_row | value]`. This is the surface a PARENT's
        // lane connects to as its priors, so under the envelope it rides the
        // reserved ACC_CHAIN block (the FL folds the LEAF registry) — a
        // constant index, the same one at which an internal child exposes
        // its own lane's claims. Off-envelope it publishes inline, as before.
        let mut acc_chain_w: Vec<Wire> = Vec::new();
        for fp in fold_pubs.iter().chain(&jfold_pubs) {
            acc_chain_w.push(fp.live);
            acc_chain_w.extend_from_slice(&fp.rho_col);
            acc_chain_w.extend_from_slice(&fp.rho_row);
            acc_chain_w.push(fp.value);
        }
        let fold_pub_base = env_acc_chain_base(&env);
        // The value-binding publics stay in the BODY: nothing above reads
        // them, they only bind the claim values this outer folded.
        for k in 0..k_ary {
            for (g, members) in groups.iter().enumerate() {
                for j in 0..members.len() {
                    let ci = n_priors + k * members.len() + j;
                    sb.publish(wv(locs[2 * g].claims[ci].value_v));
                    sb.publish(wv(locs[2 * g + 1].claims[ci].value_v));
                }
            }
            for (lg, &g) in linear.iter().enumerate() {
                for j in 0..groups[g].len() {
                    let ci = n_priors + k * groups[g].len() + j;
                    sb.publish(wv(locs[n_ab + lg].claims[ci].value_v));
                }
            }
        }
        // THE COUNTER: the base C = g^{per_leaf} rides word 9 as a fixed
        // public — GROUNDED by the parent chain (every parent
        // copy-constrains it to its own, up to the root's native
        // check_public) — and word 8 = C^k via mac rows, so the count is
        // bound by proven relations, not by fixed-public metadata.
        let c_w = cw(
            &mut sb,
            &mut vals,
            &mut consts,
            span_count_word(per_leaf as u128),
        );
        let mut e_w = c_w;
        for _ in 1..k_ary {
            e_w = sb.gate(cs.macs, &[zw, e_w, c_w])[0];
        }
        app_w.push(e_w);
        app_w.push(c_w);
        let stmt_base = {
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
                    acc_chain: &acc_chain_w,
                    app: &app_w,
                    ..EnvTail::default()
                },
            );
            env_app_base(&env)
        };
        let shape2 = sb.finish().expect("the first-level node circuit builds");
        let hint_refs: Vec<&(dyn Any + Sync)> =
            hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
        // THE INDEX-FILL RUNNER (setup), the node's path: compile the plan,
        // then pin it row-identical against the generic walk before the
        // online run trusts it. run() stays the differential oracle — this
        // pin is what keeps it one, now that the FL no longer walks in the
        // timed path either.
        let fill_plan = shape2.fill_plan();
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
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
        // Per-SHAPE prover materials, hoisted above the online loop — BLAKE3
        // for BOTH the Merkle trees and the FS chain, so the node is
        // RECURSABLE (both recorded gotchas).
        if var("FL_CENSUS").is_ok() {
            content_census("FL", &shape2.registry, &shape2.counts);
        }
        let union2 = outer_union(&shape2.registry, shape2.counts.clone());
        let pf = cfg.outer_profile();
        let pcs2 = PcsParams {
            m: union2.dense_m(),
            log_inv_rate: pf.log_inv_rate(),
            log_batch_size: pcs_batch_for(&union2, pf),
            profile: pf,
            num_lanes: outer_lanes(&union2, pcs_batch_for(&union2, pf)),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs2 = chain_blake_r1cs(nu2);
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
        // ONLINE, `1 + steady_reps()` iterations over the ONE shape: tapes
        // (the recording verifies, re-run with results discarded — identical
        // by determinism), the walk (fill plan), witness assembly, prove,
        // verify. The checker asserts re-run too — they read publics only.
        let reps = 1 + steady_reps();
        let mut onlines: Vec<Online> = Vec::with_capacity(reps);
        let mut fin = None;
        for _ in 0..reps {
            let t_tapes = Instant::now();
            for inner in inners {
                record_child_inner_verify(inner, family.domain, circs);
            }
            let tapes_ms_i = t_tapes.elapsed().as_secs_f64() * 1e3;
            let t_run = Instant::now();
            // DEFERRED: rows and publics only — the element witnesses are never
            // packed, and the assembly below feeds the prover from the rows.
            let mut built2 = shape2.run_filled_deferred(&fill_plan, &vals, &hint_refs);
            let run_ms = t_run.elapsed().as_secs_f64() * 1e3;

            // Child checkers (each child's whole deferred-verifier statement
            // against its own native replicas), then the fold checker + the
            // accumulator reassembled from publics, then the app statement.
            let mut region_end = 0usize;
            for (tk, rk) in tapes.iter().zip(&regions) {
                let consumed = check_child_region(&built2.public, tk, rk);
                assert!(
                    region_end <= rk.pub_base && rk.pub_base + consumed <= fold_pub_base,
                    "the regions' public blocks are disjoint and ordered"
                );
                region_end = rk.pub_base + consumed;
            }
            // ACC_CHAIN keeps the un-keyed entry layout: the lane's registry
            // role has ONE key (the leaf circuit), so nothing to disambiguate.
            let (rebuilt, _, _) = check_fold_publics(
                &built2.public,
                fold_pub_base,
                &locs,
                &alpha_recs,
                locs.len(),
            );
            for (r, o) in rebuilt.iter().zip(&outs) {
                assert_eq!(r, o, "published fold output == located native output");
            }
            let jag_pub_at =
                fold_pub_base + locs.iter().map(|l| 2 + l.k_col + l.k_row).sum::<usize>();
            let (jrebuilt, _, _) =
                check_jagged_fold_publics(&built2.public, jag_pub_at, &jlocs, false);
            assert_eq!(
                jrebuilt[0], jouts[0],
                "published jagged entry == located native"
            );
            let acc_pub = Accumulator {
                registry_digest: registry.digest(),
                per_type: (0..groups.len())
                    .map(|t| (rebuilt[2 * t].clone(), rebuilt[2 * t + 1].clone()))
                    .collect(),
                per_element: Vec::new(),
                per_linear: rebuilt[n_ab..sig_g].to_vec(),
                sigma: vec![(leaf_digest, rebuilt[sig_g].clone())],
                jagged: vec![(leaf_digest, jrebuilt[0].clone())],
            };
            assert_eq!(
                acc_pub, acc_v,
                "the Accumulator, reassembled from the public segment alone"
            );
            assert!(
                discharges_boolean(&acc_pub)
                    && acc_pub.discharge_sigma(&[leaf_circuit])
                    && acc_pub.discharge_jagged(&[(leaf_digest, &leaf_params_j)]),
                "the public-segment accumulator discharges all three groups"
            );
            if let Some((lam_base, _, _, _)) = &rs_lam {
                for (i, &v) in PHI_8_TABLE[..1 << K_SKIP].iter().enumerate() {
                    assert_eq!(built2.public[*lam_base + i], v, "λ const {i}");
                }
            }
            // The AG-skip blocks: the decode is in-circuit since phase D;
            // the checker holds the nonce range and the row lows.
            for (inner, base) in inners.iter().zip(&ag_pub_bases) {
                if let Some(base) = base {
                    check_ag_skip_publics(&built2.public, *base, inner.ag_round_one_bits());
                }
            }
            for &(v, idx) in &jag_const_rec {
                assert_eq!(built2.public[idx], v, "jagged shared constant public");
            }
            // THE APPLICATION STATEMENT, against its native value.
            assert_eq!(
                built2.public[stmt_base..stmt_base + ENV_APP_WORDS],
                fl_app_words(app, inners)[..],
                "the node statement"
            );

            // Everything from here to the prove is WITNESS ASSEMBLY — packing
            // the walk's rows into the union's slot inputs. It is per-statement
            // (online), so it gets its own timer rather than hiding inside the
            // shape build or the prove.
            // Recreated per online iteration — the spread closure consumes it.
            let spread_ty2 = BitSpreadTable::new(spread_w2);
            let pow_ty2 = PowMaskTable;
            let t_asm = Instant::now();
            // THE COPY-FREE ASSEMBLY, the node's path: the boolean drivers pack
            // straight into the union's slot blocks inside the prove (live rows
            // only under elide) — no capacity-sized intermediates, no memcpy.
            // The rows are hoisted to owned Vecs because the closures must be
            // Send and `built2.rows` hands out `dyn Any`-backed borrows.
            let b3_declared: Vec<_> = once(cs.q.b3).chain(cs.q.b3_alt).collect();
            let b3_rows2: Vec<_> = b3_declared
                .iter()
                .map(|&s| (s, built2.rows::<Blake3Gate>(s).to_vec()))
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
            bslots.extend(b3_rows2.into_iter().map(|(s, rows)| {
                (
                    shape2.registry_slot(s),
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
                    .map(|&s| (shape2.registry_slot(s), b3_lc2 as &dyn LincheckCircuit)),
            );
            lco.sort_by_key(|(i, _)| *i);
            let lcs2: Vec<&dyn LincheckCircuit> = lco.into_iter().map(|(_, c)| c).collect();
            let asm_ms = t_asm.elapsed().as_secs_f64() * 1e3;
            let t_prove = Instant::now();
            let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
            let (oproof, ocommit) = if outer_zerocheck_ag() {
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
                unreachable!("outer_zerocheck_ag() is false off aarch64")
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
            let prove_ms = t_prove.elapsed().as_secs_f64() * 1e3;
            let t_ver = Instant::now();
            let mut ch2 = FsChallenger::with_chained_blake3(DOMAIN);
            oproof
                .verify_circuit(
                    &union2,
                    &shape2.circuit,
                    &built2.public,
                    &lcs2,
                    &ocommit,
                    &pcs2,
                    &mut ch2,
                )
                .expect("the first-level node verifies over the circuit path");
            let verify_ms2 = t_ver.elapsed().as_secs_f64() * 1e3;
            onlines.push(Online {
                setup_ms: build_ms,
                tapes_ms: tapes_ms_i,
                walk_ms: run_ms,
                witgen_ms: asm_ms,
                prove_ms,
                verify_ms: verify_ms2,
                wall_ms: 0.0,
            });
            fin = Some((built2, oproof, ocommit, acc_pub));
        }
        let (built2, oproof, ocommit, acc_pub) = fin.expect("one online iteration");
        let (swap_ri, spread_ri, pow_ri, family_ri) = (
            shape2.registry_slot(cs.q.swap),
            shape2.registry_slot(cs.q.spread),
            shape2.registry_slot(cs.q.pow),
            shape2.registry_slot(family_slot),
        );
        let b3_ris = once(cs.q.b3)
            .chain(cs.q.b3_alt)
            .map(|s| shape2.registry_slot(s))
            .collect();
        FlCore {
            lo: LeafOuter {
                shape: shape2,
                public: built2.public,
                proof: oproof,
                commitment: ocommit,
                pcs: pcs2,
                b3_r1cs: b3_r1cs2,
                swap_r1cs: swap_r1cs2,
                spread_r1cs: spread_r1cs2,
                pow_r1cs: pow_r1cs2,
                family_r1cs: family_r1cs2,
                b3_slots: b3_ris,
                swap_slot: swap_ri,
                spread_slot: spread_ri,
                pow_slot: pow_ri,
                family_slot: family_ri,
            },
            acc: acc_pub,
            stmt_base,
            fold_pub_base,
            t: *onlines.last().expect("one online iteration"),
            onlines,
        }
    }
}

/// **The first-level node's pin, through the builder** (converted-first:
/// the test IS [`build_first_level_node`]'s original body; every assert lives inside
/// the builder now, the wrapper re-checks the statement surface).
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
pub(super) fn first_level_node_two_chains_fold_and_adjacency() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0004);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());
    let cp0 = build_chain_proof(cfg, h0, n_blocks);
    let cp1 = build_chain_proof(cfg, cp0.h_end, n_blocks);
    let fl = build_first_level_node(cfg, &cp0, &cp1);
    assert_eq!(fl.h_start, cp0.h_start);
    assert_eq!(fl.h_end, cp1.h_end);
    for j in 0..4 {
        assert_eq!(
            fl.lo.public[fl.stmt_base + j],
            pack4(fl.h_start[4 * j..4 * j + 4].try_into().unwrap()),
            "the statement block reads h_start out of the public segment"
        );
        assert_eq!(
            fl.lo.public[fl.stmt_base + 4 + j],
            pack4(fl.h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "the statement block reads h_end out of the public segment"
        );
    }
    println!(
        "\nFIRST-LEVEL NODE (two adjacent chain proofs, fold + adjacency)\n  \
         chain: {} + {} compressions | node statement: h_start .. H^{}(h_start)\n  \
         outer: nu {} | mu {} | publics {} | proof {:.1} KiB\n",
        n_blocks,
        n_blocks,
        2 * n_blocks,
        fl.lo.shape.circuit.cells().nu(),
        fl.lo.shape.circuit.cells().mu(),
        fl.lo.public.len(),
        serialize(&fl.lo.proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

/// **THE ENVELOPE CONTENT PROBE** — the m\* headroom question: the FL's and
/// the internal node's UNFLOORED content (dense_words / content dense_m)
/// under free counts, against the m\*28 cap (2^21 packed words). The
/// per-type breakdown (used_cols × rows, descending) is the diet map if the
/// gap needs closing. `CHAIN_BLOCKS` sizes the leaves (the real question is
/// 262144 = m32); `TOWER_CONFIG=chain128` selects the envelope config.
#[test]
#[ignore] // Heavy at m32 — four chain proofs, two FLs, one node.
pub(super) fn envelope_content_probe() {
    let cfg = test_config();
    let n_blocks: usize = var("CHAIN_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let mut rng = Rng(0xC4A1_00CE);
    let h0: [u32; 16] = from_fn(|_| rng.next_u32());
    let mut cps = Vec::new();
    let mut h = h0;
    for _ in 0..4 {
        let cp = build_chain_proof(cfg, h, n_blocks);
        h = cp.h_end;
        cps.push(cp);
    }
    let fl0 = build_first_level_node(cfg, &cps[0], &cps[1]);
    let fl1 = build_first_level_node(cfg, &cps[2], &cps[3]);
    let chain_registry = &cps[0].inner.built.shape.registry;
    let blake_r1cs = build_block_r1cs(cps[0].inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let chain_mats = [(&blake_r1cs.a_0, &blake_r1cs.b_0)];
    let chain_circs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let chain_jp = chain_jagged_params(&cps[0]);
    let node = build_node_outer_app(
        cfg,
        &[&fl0.lo, &fl1.lo],
        Some(fl0.stmt_base),
        Some(ChainLane {
            registry: chain_registry,
            mats: &chain_mats,
            circs: &chain_circs,
            circuit: &cps[0].inner.built.shape.circuit,
            params: &chain_jp,
            priors: &[&fl0.acc, &fl1.acc],
            claims_base: fl0.fold_pub_base,
            family: LaneFamily::Chain,
        }),
        None,
    );
    println!(
        "\nENVELOPE CONTENT PROBE — {n_blocks} compressions/leaf, profile {:?}\n  \
         m28 cap = {} words | m29 cap = {} words",
        cfg.outer_profile(),
        1usize << (28 - 7),
        1usize << (29 - 7),
    );
    for (name, lo) in [("FL", &fl0.lo), ("internal", &node.lo)] {
        let u = UnionInstance::new(&lo.shape.registry, lo.shape.counts.clone());
        let dw = u.dense_words();
        println!(
            "  {name}: dense_words {dw} = {:.1}% of m28 cap | content dense_m {} | floored m {}",
            100.0 * dw as f64 / (1u64 << 21) as f64,
            u.dense_m(),
            outer_union(&lo.shape.registry, lo.shape.counts.clone()).dense_m(),
        );
        // The diet map: per-type committed words, descending.
        let mut per: Vec<(usize, usize, usize, usize)> = lo
            .shape
            .registry
            .types()
            .iter()
            .zip(&lo.shape.counts)
            .enumerate()
            .map(|(i, (ty, &n_t))| {
                let cols = ty.useful_bits.div_ceil(128).min(1usize << (ty.k_log - 7));
                (cols * n_t, i, cols, n_t)
            })
            .collect();
        per.sort_by_key(|p| Reverse(p.0));
        for &(words, i, cols, rows) in per.iter().take(8) {
            println!(
                "    type {i:2}: {words:>8} words ({cols:3} cols x {rows:6} rows) = {:.1}%",
                100.0 * words as f64 / dw as f64
            );
        }
    }
}
