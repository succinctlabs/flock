//! **THE leanXMSS AGGREGATION TOWER**: signature batches as leaves, folded
//! to one root proof (`docs/leanxmss-recursion-plan.md`).
//!
//! The pipeline is the chain tower's ([`super::driver`]) with leanXMSS
//! leaves: each batch proves as an [`XmssLeaf`], adjacent pairs fold into
//! first-level nodes ([`build_xmss_first_level_node`]), and the SPINE takes
//! over — the BASE folds the last two first-level nodes and every level
//! above prepends the next-earlier one, the leanXMSS lane
//! ([`LaneFamily::Batches`]) riding every level.
//!
//! The statement is a digest of the signature batches and their count:
//! each leaf publishes its batch's statement digest, a first-level node
//! publishes `hash_pair` of its leaves' publics digests, and each spine
//! node `hash_pair(D_fresh, D_prev)` — see [`xmss_statement`], which a
//! verifier recomputes from the batches.

use flock_core::{
    aggregate::{Accumulator, Dropped, SharedTables},
    circuit::Circuit,
    matrix_fold::MatrixClaim,
    pcs::jagged::JaggedParams,
    union::publics_digest,
};
use flock_merkle::hash_pair;

use crate::{
    leanxmss::{
        circuit::{Statement, XmssCircuit},
        reference::Signature,
    },
    schedule::Registry,
    tower::{
        ChainLane, DOMAIN, F128, FlFamily, FsChallenger, HashKind, LaneFamily, LeafOuter, NodeOut,
        RootBundle, SpineIn, TowerConfig, TowerVerifyError, XmssFlNode, XmssLeaf,
        build_node_outer_app, build_xmss_first_level_node, build_xmss_leaf,
        driver::{RootDischargeFailure, discharge_outer},
        envelope_shape_with_lane, lane_block_words,
        node::digest_f128,
        outer_union,
        query::leaf_boolean_lcs,
        span_count_word,
        verify::{AccLayout, reassemble_parts},
        xmss_leaf::{xmss_family, xmss_jagged_params},
    },
};

/// What an leanXMSS tower attests: every signature of every batch verifies,
/// against the batches' statements as `digest` binds them, and there are
/// `n_signatures` of them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct XmssStatement {
    pub digest: [u8; 32],
    pub n_signatures: usize,
}

/// The statement a tower over `batches` (in leaf order) publishes: leaf
/// `i`'s digest is its public segment's ([`publics_digest`] of
/// `circuit.public(batch)`), a first-level node's `hash_pair` of its two
/// leaves', the base `hash_pair` of the last two first-level nodes', and
/// each spine level `hash_pair(D_fl, D_prev)` prepending the next-earlier
/// first-level node.
pub fn xmss_statement(circuit: &XmssCircuit, batches: &[Vec<Statement>]) -> XmssStatement {
    assert!(
        batches.len() >= 4 && batches.len().is_multiple_of(2),
        "the tower pairs batches into first-level nodes and roots at a node"
    );
    let pair = |l: &[u8; 32], r: &[u8; 32]| hash_pair(l, r, HashKind::Blake3);
    let leaves: Vec<[u8; 32]> = batches
        .iter()
        .map(|b| publics_digest(&circuit.public(b)))
        .collect();
    let fls: Vec<[u8; 32]> = leaves.chunks(2).map(|p| pair(&p[0], &p[1])).collect();
    let k = fls.len();
    let mut digest = pair(&fls[k - 2], &fls[k - 1]);
    for fl in fls[..k - 2].iter().rev() {
        digest = pair(fl, &digest);
    }
    XmssStatement {
        digest,
        n_signatures: batches.iter().map(Vec::len).sum(),
    }
}

/// Everything a leanXMSS lane needs but its priors: the leaf tables.
struct LaneParts<'a> {
    registry: &'a Registry,
    family: &'a FlFamily<'a>,
    dropped: &'a Dropped,
    circuit: &'a Circuit,
    params: &'a JaggedParams,
    claims_base: usize,
    block_words: usize,
}

impl LaneParts<'_> {
    fn lane<'s>(&'s self, priors: &'s [&'s Accumulator]) -> ChainLane<'s> {
        ChainLane {
            registry: self.registry,
            mats: &self.family.mats,
            circs: &self.family.circs,
            circuit: self.circuit,
            params: self.params,
            priors,
            claims_base: self.claims_base,
            family: LaneFamily::Batches {
                block_words: self.block_words,
                dropped: self.dropped,
            },
        }
    }
}

/// A completed leanXMSS tower run: the root plus the material owners the
/// root-side discharge needs — leaf 0 for the leanXMSS tables, FL 0 for the
/// outer tables, the base for the passenger's.
pub struct XmssTower {
    pub(super) cfg: TowerConfig,
    pub(super) statement: XmssStatement,
    /// The app block's offset in every outer's public segment.
    pub(super) app_base: usize,
    /// Signatures per leaf — the counter base's exponent.
    pub(super) per_leaf: usize,
    pub(super) leaf: XmssLeaf,
    pub(super) fl: XmssFlNode,
    pub(super) base: Option<NodeOut>,
    pub(super) root: NodeOut,
    pub(super) expect_passenger: bool,
}

impl XmssTower {
    /// PROVE THE BATCHES: one leaf per batch (all under `circuit`, so all
    /// the same size), folded to one root. At least four batches, an even
    /// number. Every input drops once folded except leaf 0 and FL 0, the
    /// discharge's table owners.
    pub fn prove(
        cfg: TowerConfig,
        circuit: &XmssCircuit,
        batches: &[(Vec<Statement>, Vec<Signature>)],
    ) -> XmssTower {
        let n_leaves = batches.len();
        assert!(
            n_leaves >= 4 && n_leaves.is_multiple_of(2),
            "the tower pairs leaves into 2-ary FLs and roots at a node: \
             n_leaves must be even and >= 4, got {n_leaves}"
        );
        let per_leaf = circuit.signatures();
        let mut leaf0: Option<XmssLeaf> = None;
        let mut fls: Vec<XmssFlNode> = Vec::with_capacity(n_leaves / 2);
        for pair in batches.chunks(2) {
            let l0 = build_xmss_leaf(circuit, &pair[0].0, &pair[0].1);
            let l1 = build_xmss_leaf(circuit, &pair[1].0, &pair[1].1);
            fls.push(build_xmss_first_level_node(cfg, circuit, [&l0, &l1]));
            leaf0.get_or_insert(l0);
        }
        let statements: Vec<Vec<Statement>> = batches.iter().map(|b| b.0.clone()).collect();
        let statement = xmss_statement(circuit, &statements);
        let leaf = leaf0.expect("leaf 0");
        let app_base = fls[0].stmt_base;
        let claims_base = fls[0].fold_pub_base;
        let family = xmss_family(circuit);
        let jp = xmss_jagged_params(&leaf);
        let dropped = family
            .shared
            .clone()
            .expect("the leanXMSS lane binds shared");
        let parts = LaneParts {
            registry: &leaf.inner.built.shape.registry,
            family: &family,
            dropped: &dropped,
            circuit: &leaf.inner.built.shape.circuit,
            params: &jp,
            claims_base,
            block_words: lane_block_words(&fls[0].acc),
        };

        // ---- THE BASE: fresh-only over the LAST two FLs ----
        let k = fls.len();
        let base = build_node_outer_app(
            cfg,
            &[&fls[k - 2].lo, &fls[k - 1].lo],
            Some(app_base),
            Some(parts.lane(&[&fls[k - 2].acc, &fls[k - 1].acc])),
            None,
        );

        // ---- THE SPINE: prepend the next-earlier FL, level by level ----
        let (base, root) = if k == 2 {
            (None, base)
        } else {
            let mut cur: Option<NodeOut> = None;
            for i in (0..k - 2).rev() {
                let prev = cur.as_ref().unwrap_or(&base);
                let prev_lane = prev.lane_acc.clone().expect("the lane rides every level");
                let n = build_node_outer_app(
                    cfg,
                    &[&fls[i].lo, &prev.lo],
                    Some(app_base),
                    Some(parts.lane(&[&fls[i].acc, &prev_lane])),
                    Some(SpineIn {
                        node_child: 1,
                        prior: &prev.block,
                        forge: false,
                    }),
                );
                if let Some(c) = &cur {
                    assert_eq!(
                        n.lo.shape.circuit.digest(),
                        c.lo.shape.circuit.digest(),
                        "the spine converged and must stay converged"
                    );
                }
                cur = Some(n);
            }
            (
                Some(base),
                cur.expect("k > 2 built at least one spine node"),
            )
        };

        let fl = fls.into_iter().next().expect("FL 0");
        XmssTower {
            cfg,
            statement,
            app_base,
            per_leaf,
            leaf,
            fl,
            base,
            root,
            expect_passenger: k >= 4,
        }
    }

    pub fn config(&self) -> TowerConfig {
        self.cfg
    }

    pub fn statement(&self) -> &XmssStatement {
        &self.statement
    }

    /// **THE ROOT-SIDE RESIDUE**: the statement against the root's app
    /// block, the leanXMSS lane against the leaf tables (per matrix
    /// class), and the main fold and passenger as every tower discharges
    /// them ([`discharge_outer`]).
    pub fn discharge_root(&self) -> Result<(), RootDischargeFailure> {
        self.discharge_with(
            &self.root.lo.public,
            self.root
                .lane_acc
                .as_ref()
                .expect("the lane rides every level"),
            &self.root.acc,
            &self.root.block.passenger,
            &self.statement,
            self.expect_passenger,
        )
    }

    pub(super) fn discharge_with(
        &self,
        public: &[F128],
        lane: &Accumulator,
        main: &Accumulator,
        passenger: &[([F128; 2], MatrixClaim)],
        statement: &XmssStatement,
        expect_passenger: bool,
    ) -> Result<(), RootDischargeFailure> {
        use RootDischargeFailure::*;
        // (1) the statement: the digest in words 0..2, words 2..8 zero, the
        // signature counter and its base in 8 and 9.
        let app = &public[self.app_base..self.app_base + 10];
        if app[..2] != digest_f128(&statement.digest)
            || app[2..8].iter().any(|w| *w != F128::ZERO)
            || app[8] != span_count_word(statement.n_signatures as u128)
            || app[9] != span_count_word(self.per_leaf as u128)
        {
            return Err(Statement);
        }
        // (2) the leanXMSS lane vs the leaf tables (leaf 0's shape).
        let registry = &self.leaf.inner.built.shape.registry;
        let leaf_circuit = &self.leaf.inner.built.shape.circuit;
        let tables = SharedTables::new(registry, &Dropped::of(&self.leaf.inner.union()));
        if !lane.discharge(&tables.kept_refs()) || !lane.discharge_linear(&tables.linear_refs()) {
            return Err(ChainLaneBoolean);
        }
        if !lane.per_element.is_empty() {
            return Err(ChainLaneElement);
        }
        if !lane.discharge_sigma(&[leaf_circuit]) {
            return Err(ChainLaneSigma);
        }
        let jp = xmss_jagged_params(&self.leaf);
        if !lane.discharge_jagged(&[(leaf_circuit.digest(), &jp)]) {
            return Err(ChainLaneJagged);
        }
        // (3) + (4).
        discharge_outer(
            &self.fl.lo,
            &self.root.lo,
            self.base.as_ref(),
            main,
            passenger,
            expect_passenger,
        )
    }
}

/// The leanXMSS tower's VERIFICATION KEY for one leaf circuit: a reference
/// tower over six batches (the smallest run with all four shapes — leaf,
/// FL, base, steady) and the published blocks' decode plan. Shapes are
/// statement-independent, so any six valid batches give the same key.
pub struct XmssTowerVk {
    pub(super) tower: XmssTower,
    pub(super) layout: AccLayout,
}

impl XmssTowerVk {
    /// Prove the reference tower over `batches` (six, under `circuit`).
    pub fn generate(
        cfg: TowerConfig,
        circuit: &XmssCircuit,
        batches: &[(Vec<Statement>, Vec<Signature>)],
    ) -> XmssTowerVk {
        assert_eq!(batches.len(), 6, "the reference tower has six leaves");
        let tower = XmssTower::prove(cfg, circuit, batches);
        let env = envelope_shape_with_lane(lane_block_words(&tower.fl.acc));
        let layout = AccLayout::from_parts(&env, tower.fl.fold_pub_base, &tower.root);
        XmssTowerVk { tower, layout }
    }
}

/// **VERIFY A leanXMSS TOWER ROOT, STANDALONE**: the VK, the claimed
/// statement (recomputed by the consumer with [`xmss_statement`] from the
/// batches it holds), and the root bundle. The root proof verifies against
/// VK materials, the accumulators are reassembled from the bundle's
/// publics alone, and the discharge legs run against the VK's tables.
pub fn verify_xmss_root(
    vk: &XmssTowerVk,
    statement: &XmssStatement,
    bundle: &RootBundle<'_>,
) -> Result<(), TowerVerifyError> {
    // (1) geometry: the depth this statement claims, and which node shape
    // roots it (two first-level nodes: the base; more: the steady node).
    let per_pair = 2 * vk.tower.per_leaf;
    if statement.n_signatures == 0 || !statement.n_signatures.is_multiple_of(per_pair) {
        return Err(TowerVerifyError::Geometry);
    }
    let k = statement.n_signatures / per_pair;
    if k < 2 {
        return Err(TowerVerifyError::Geometry);
    }
    let base_lo = &vk
        .tower
        .base
        .as_ref()
        .expect("the VK tower keeps its base")
        .lo;
    let root_lo: &LeafOuter = if k == 2 { base_lo } else { &vk.tower.root.lo };
    if bundle.public.len() != root_lo.public.len() {
        return Err(TowerVerifyError::PublicsLength);
    }

    // (2) the root proof, against VK materials only.
    let u = outer_union(&root_lo.shape.registry, root_lo.shape.counts.clone());
    let lcs = leaf_boolean_lcs(root_lo);
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    bundle
        .proof
        .verify_circuit(
            &u,
            &root_lo.shape.circuit,
            bundle.public,
            &lcs,
            bundle.commitment,
            &root_lo.pcs,
            &mut ch,
        )
        .map_err(TowerVerifyError::Proof)?;

    // (3) the published accumulators, from the publics alone.
    let (main, passenger, lane) = reassemble_parts(
        bundle.public,
        &vk.layout,
        &vk.tower.fl.lo,
        &vk.tower.root.lo,
        base_lo,
        &vk.tower.leaf.inner.built.shape,
    )?;

    // (4) the discharge legs + the statement binding.
    vk.tower
        .discharge_with(bundle.public, &lane, &main, &passenger, statement, k >= 4)
        .map_err(TowerVerifyError::Discharge)
}

impl XmssTower {
    /// The root's transportable part: publics, proof, commitment.
    pub fn root_bundle(&self) -> RootBundle<'_> {
        RootBundle {
            public: &self.root.lo.public,
            proof: &self.root.lo.proof,
            commitment: &self.root.lo.commitment,
        }
    }
}

/// Phase 4: the leanXMSS tower end to end — eight batches (`XMSS_N`
/// signatures each, default 4 at a tall slot), four first-level nodes, the
/// base and two spine levels to a steady root; the root-side discharge
/// accepts, and refuses a wrong digest, a wrong count, and a lane claim
/// moved off its value.
#[test]
#[ignore] // Heavy — eight leaves and seven outers.
fn xmss_tower_e2e() {
    use std::time::Instant;

    use crate::tower::{test_config, xmss_leaf::signed};
    let cfg = test_config();
    let n: usize = std::env::var("XMSS_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    // A small batch takes a tall slot (one slot per gate kind, the
    // production shape); `XMSS_NU` overrides.
    let nu = std::env::var("XMSS_NU")
        .ok()
        .and_then(|v| v.parse().ok())
        .or((n < 64).then_some(9));
    let circuit =
        XmssCircuit::with_options(n, nu, crate::leanxmss::circuit::XmssOptions::RECURSION_LEAF);
    let batch = |seed: u64| {
        if n < 64 {
            signed(n, seed)
        } else {
            crate::leanxmss::circuit::keyless_batch(n, seed)
        }
    };
    let batches: Vec<(Vec<Statement>, Vec<Signature>)> =
        (0..8u64).map(|i| batch(101 + i)).collect();
    let t = Instant::now();
    let tower = XmssTower::prove(cfg, &circuit, &batches);
    println!(
        "xmss tower: 8 x {n} signatures (leaf nu {}) in {:?} | prove: FL {:.0} ms, base {:.0} ms, \
         root {:.0} ms | root proof {:.1} KiB",
        circuit.nu(),
        t.elapsed(),
        tower.fl.t.prove_ms,
        tower.base.as_ref().map_or(0.0, |b| b.online.prove_ms),
        tower.root.online.prove_ms,
        bincode::serialize(&tower.root.lo.proof).unwrap().len() as f64 / 1024.0,
    );
    assert_eq!(tower.statement().n_signatures, 8 * n);
    tower
        .discharge_root()
        .expect("the root-side residue discharges");

    // The standalone verifier: a VK from six other batches, the statement
    // recomputed from the batches alone.
    let vk_batches: Vec<(Vec<Statement>, Vec<Signature>)> =
        (0..6u64).map(|i| batch(901 + i)).collect();
    let vk = XmssTowerVk::generate(cfg, &circuit, &vk_batches);
    let statements: Vec<Vec<Statement>> = batches.iter().map(|b| b.0.clone()).collect();
    let st = xmss_statement(&circuit, &statements);
    assert_eq!(&st, tower.statement());
    verify_xmss_root(&vk, &st, &tower.root_bundle()).expect("the root verifies standalone");
    let mut swapped = statements.clone();
    swapped.swap(0, 1);
    assert!(
        matches!(
            verify_xmss_root(
                &vk,
                &xmss_statement(&circuit, &swapped),
                &tower.root_bundle()
            ),
            Err(TowerVerifyError::Discharge(RootDischargeFailure::Statement))
        ),
        "the batches in another order"
    );
    let mut edited = statements.clone();
    edited[5][0].msg[3] ^= 1;
    assert!(
        matches!(
            verify_xmss_root(
                &vk,
                &xmss_statement(&circuit, &edited),
                &tower.root_bundle()
            ),
            Err(TowerVerifyError::Discharge(RootDischargeFailure::Statement))
        ),
        "an edited message"
    );
    let mut public = tower.root.lo.public.clone();
    public[tower.app_base] += F128::ONE;
    let forged = RootBundle {
        public: &public,
        proof: &tower.root.lo.proof,
        commitment: &tower.root.lo.commitment,
    };
    assert!(
        matches!(
            verify_xmss_root(&vk, &st, &forged),
            Err(TowerVerifyError::Proof(_))
        ),
        "a public word the proof does not attest"
    );

    let lane = tower.root.lane_acc.as_ref().unwrap();
    let check = |public: &[F128], lane: &Accumulator, st: &XmssStatement| {
        tower.discharge_with(
            public,
            lane,
            &tower.root.acc,
            &tower.root.block.passenger,
            st,
            tower.expect_passenger,
        )
    };
    let mut st = *tower.statement();
    st.digest[0] ^= 1;
    assert_eq!(
        check(&tower.root.lo.public, lane, &st),
        Err(RootDischargeFailure::Statement),
        "a different digest"
    );
    let mut st = *tower.statement();
    st.n_signatures += 1;
    assert_eq!(
        check(&tower.root.lo.public, lane, &st),
        Err(RootDischargeFailure::Statement),
        "a different count"
    );
    let mut bad = lane.clone();
    bad.per_type[0].1.value += F128::ONE;
    assert_eq!(
        check(&tower.root.lo.public, &bad, tower.statement()),
        Err(RootDischargeFailure::ChainLaneBoolean),
        "a lane claim moved off its value"
    );
}
