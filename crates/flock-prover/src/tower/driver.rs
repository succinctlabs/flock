//! **THE DRIVER**: one call from a chain statement to the tower's
//! converged root — the production caller for the pipeline the e2e tests
//! used to hand-assemble.
//!
//! [`Tower::prove`] runs the whole tower over a BLAKE3 compression chain:
//! sequential LEAVES (each segment starts at the previous one's `h_end`),
//! adjacent pairs into FIRST-LEVEL nodes, then the SPINE — the BASE folds
//! the LAST two FLs and every level above prepends the next-earlier FL as
//! its fresh child, the chain LANE riding every level. From the second
//! spine fold on the circuit digest must not move (ONE steady shape —
//! asserted in the loop).
//!
//! [`Tower::discharge_root`] settles the ROOT-SIDE RESIDUE: the carried
//! accumulators against the native tables, the passenger against the
//! base's own tables, the statement against the root's publics. It does
//! NOT verify the root's own proof — that is the consuming verifier's
//! call (`verify_circuit` over the root outer, the statement-tier helper
//! the e2e tamper legs assemble). The root artifacts stay crate-internal
//! until the standalone-verifier milestone decides what a consumer sees.

use flock_core::{
    aggregate::Accumulator, circuit::Circuit, lincheck::LincheckCircuit, pcs::jagged::JaggedParams,
};

use crate::tower::{
    chain::{ChainProof, build_chain_proof},
    config::TowerConfig,
    fl_node::{FlNode, build_fl_node, chain_blake_r1cs, chain_jagged_params},
    gates_blake3::pack4,
    node::{
        ChainLane, NodeOut, SpineIn, build_node_outer_app, digest_f128, entry_live,
        node_jagged_params,
    },
    online::census_kib,
    query::leaf_boolean_mats,
};

/// What a tower run attests: `h_end == H^{n_blocks}(h_start)` over the
/// BLAKE3 compression chain.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChainStatement {
    pub h_start: [u32; 16],
    pub h_end: [u32; 16],
    /// TOTAL compressions across the span: `n_leaves * blocks_per_leaf`.
    pub n_blocks: usize,
}

/// Which root-side discharge refused — see [`Tower::discharge_root`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RootDischargeFailure {
    /// The root's app block does not carry the statement's span.
    Statement,
    /// The chain lane's boolean claims vs the chain BLAKE3 matrices.
    ChainLaneBoolean,
    /// The chain lane grew an element group it cannot have.
    ChainLaneElement,
    /// The chain lane's sigma claims vs the chain circuit's wiring.
    ChainLaneSigma,
    /// The chain lane's jagged claims vs the chain layout.
    ChainLaneJagged,
    /// The main fold's boolean claims vs the outer registry's matrices.
    Boolean,
    /// The main fold's element claims vs the outer element types.
    Element,
    /// The main fold's sigma claims vs the outer circuits' wiring.
    Sigma,
    /// The main fold's jagged claims vs the outer layouts.
    Jagged,
    /// The passenger: missing when owed, riding when not, or refusing the
    /// base's tables.
    Passenger,
}

/// A completed tower run: the ROOT (one recursable proof over the whole
/// chain) plus the material owners the root-side discharge needs — leaf 0
/// for the chain tables, FL 0 for the outer tables, the base for the
/// passenger's.
pub struct Tower {
    pub(super) cfg: TowerConfig,
    pub(super) statement: ChainStatement,
    /// The app block's offset in every outer's public segment.
    pub(super) app_base: usize,
    /// Leaf 0 — the chain lane's material owner (one shape for every
    /// leaf).
    pub(super) chain: ChainProof,
    /// FL 0 (the root's own fresh child) — the outer tables' owner (one
    /// shape for every FL).
    pub(super) fl: FlNode,
    /// The spine's BASE, kept whenever it is not the root itself: its
    /// circuit key survives at the root — as the k = 3 root's node-slot
    /// sigma key, or as the steady passenger's — so the discharge needs
    /// its tables.
    pub(super) base: Option<NodeOut>,
    pub(super) root: NodeOut,
    /// A STEADY root (two spine folds or more) owes the single orphan its
    /// transition made; a shallower root owes none.
    pub(super) expect_passenger: bool,
}

impl Tower {
    /// PROVE THE CHAIN: `n_leaves` sequential segments of
    /// `blocks_per_leaf` compressions each, folded to one root.
    ///
    /// `n_leaves` must be even and at least 4 (leaves pair into 2-ary FLs
    /// and the tower roots at a node; the k-ary FL door stays open but is
    /// not driven here). Leaves build sequentially — each segment's walk
    /// IS the chain compute — and every input drops once folded: a
    /// segment pair drops when its FL takes it (leaf 0 stays as the
    /// lane's material owner), so what stays resident is one leaf pair,
    /// the FL row (built forward, folded backward from the tail), and the
    /// current spine node.
    pub fn prove(
        cfg: TowerConfig,
        h_start: [u32; 16],
        blocks_per_leaf: usize,
        n_leaves: usize,
    ) -> Tower {
        assert!(
            n_leaves >= 4 && n_leaves.is_multiple_of(2),
            "the tower pairs leaves into 2-ary FLs and roots at a node: \
             n_leaves must be even and >= 4, got {n_leaves}"
        );
        // ---- LEAVES + FIRST LEVEL, interleaved: prove a segment pair
        // (each segment starts at the last h_end), fold it into its FL,
        // drop the pair ----
        let mut chain: Option<ChainProof> = None;
        let mut fls: Vec<FlNode> = Vec::with_capacity(n_leaves / 2);
        let mut h = h_start;
        for _ in 0..n_leaves / 2 {
            let cp0 = build_chain_proof(cfg, h, blocks_per_leaf);
            let cp1 = build_chain_proof(cfg, cp0.h_end, blocks_per_leaf);
            h = cp1.h_end;
            fls.push(build_fl_node(cfg, &cp0, &cp1));
            // Leaf 0 survives as the lane's chain-side material owner
            // (one shape for every leaf); every other leaf drops here.
            chain.get_or_insert(cp0);
        }
        let statement = ChainStatement {
            h_start,
            h_end: h,
            n_blocks: n_leaves * blocks_per_leaf,
        };
        let app_base = fls[0].stmt_base;
        let claims_base = fls[0].fold_pub_base;

        let chain = chain.expect("leaf 0");
        let registry = &chain.inner.built.shape.registry;
        let blake = chain_blake_r1cs(chain.inner.nu);
        let blake_lc = blake.csc_lincheck_circuit();
        let chain_mats = [(&blake.a_0, &blake.b_0)];
        let chain_circs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
        let chain_circuit = &chain.inner.built.shape.circuit;
        let chain_jp = chain_jagged_params(&chain);

        // ---- THE BASE: fresh-only over the LAST two FLs (the spine
        // grows by prepending, so the base covers the chain's tail) ----
        let k = fls.len();
        let base = build_node_outer_app(
            cfg,
            &[&fls[k - 2].lo, &fls[k - 1].lo],
            Some(app_base),
            Some(ChainLane {
                registry,
                mats: &chain_mats,
                circs: &chain_circs,
                circuit: chain_circuit,
                params: &chain_jp,
                priors: &[&fls[k - 2].acc, &fls[k - 1].acc],
                claims_base,
            }),
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
                    Some(ChainLane {
                        registry,
                        mats: &chain_mats,
                        circs: &chain_circs,
                        circuit: chain_circuit,
                        params: &chain_jp,
                        priors: &[&fls[i].acc, &prev_lane],
                        claims_base,
                    }),
                    Some(SpineIn {
                        node_child: 1,
                        prior: &prev.block,
                        forge: false,
                    }),
                );
                // ONE steady shape: from the second spine fold on, the
                // digest must not move (wall 3's convergence).
                if let Some(c) = &cur {
                    assert_eq!(
                        n.lo.shape.circuit.digest(),
                        c.lo.shape.circuit.digest(),
                        "the spine converged and must stay converged"
                    );
                }
                cur = Some(n); // the folded spine node drops here
            }
            (
                Some(base),
                cur.expect("k > 2 built at least one spine node"),
            )
        };

        let fl = fls.into_iter().next().expect("FL 0");
        Tower {
            cfg,
            statement,
            app_base,
            chain,
            fl,
            base,
            root,
            expect_passenger: k >= 4,
        }
    }

    pub fn config(&self) -> TowerConfig {
        self.cfg
    }

    pub fn statement(&self) -> &ChainStatement {
        &self.statement
    }

    /// The root proof's serialized size in KiB — the recursable artifact.
    pub fn root_proof_kib(&self) -> f64 {
        census_kib(&self.root.lo.proof)
    }

    /// **THE ROOT-SIDE RESIDUE.** Everything the tower defers lands here
    /// and discharges against the native tables:
    ///
    /// 1. the STATEMENT rides the root's app block (packed `h_start` then
    ///    `h_end`);
    /// 2. the CHAIN LANE — every leaf's claims — against the chain BLAKE3
    ///    matrices, the chain circuit's sigma table, and the chain layout;
    /// 3. the MAIN FOLD against the outer registry's matrices and element
    ///    types, and against the sigma/jagged tables of every circuit a
    ///    keyed slot can name — the FL's, the root's own, and the base's;
    /// 4. the PASSENGER — the spine's single orphan — against the base's
    ///    own tables, owed exactly when the root is steady.
    pub fn discharge_root(&self) -> Result<(), RootDischargeFailure> {
        use RootDischargeFailure::*;
        let root = &self.root;
        // (1) the statement.
        let words =
            |h: &[u32; 16], j: usize| pack4(h[4 * j..4 * j + 4].try_into().expect("4 words"));
        for j in 0..4 {
            if root.lo.public[self.app_base + j] != words(&self.statement.h_start, j)
                || root.lo.public[self.app_base + 4 + j] != words(&self.statement.h_end, j)
            {
                return Err(Statement);
            }
        }
        // (2) the chain lane vs the chain tables (leaf 0's shape).
        let lane = root.lane_acc.as_ref().expect("the lane rides every level");
        let blake = chain_blake_r1cs(self.chain.inner.nu);
        let chain_mats = [(&blake.a_0, &blake.b_0)];
        let chain_circuit = &self.chain.inner.built.shape.circuit;
        if !lane.discharge(&chain_mats) {
            return Err(ChainLaneBoolean);
        }
        if !lane.per_element.is_empty() {
            return Err(ChainLaneElement);
        }
        if !lane.discharge_sigma(&[chain_circuit]) {
            return Err(ChainLaneSigma);
        }
        let chain_jp = chain_jagged_params(&self.chain);
        if !lane.discharge_jagged(&[(chain_circuit.digest(), &chain_jp)]) {
            return Err(ChainLaneJagged);
        }
        // (3) the main fold vs the outer tables.
        let fl_lo = &self.fl.lo;
        if !root.acc.discharge(&leaf_boolean_mats(fl_lo)) {
            return Err(Boolean);
        }
        let el_mats: Vec<_> = fl_lo
            .shape
            .registry
            .element_types()
            .iter()
            .map(|t| {
                let e = t.element_type().expect("element table");
                (e.a_0(), e.b_0())
            })
            .collect();
        if !root.acc.discharge_element(&el_mats) {
            return Err(Element);
        }
        let fl_jp = node_jagged_params(fl_lo);
        let root_jp = node_jagged_params(&root.lo);
        let base_jp = self.base.as_ref().map(|b| node_jagged_params(&b.lo));
        let mut circuits: Vec<&Circuit> = vec![&fl_lo.shape.circuit, &root.lo.shape.circuit];
        let mut layouts: Vec<([u8; 32], &JaggedParams)> = vec![
            (fl_lo.shape.circuit.digest(), &fl_jp),
            (root.lo.shape.circuit.digest(), &root_jp),
        ];
        if let (Some(b), Some(jp)) = (&self.base, &base_jp) {
            circuits.push(&b.lo.shape.circuit);
            layouts.push((b.lo.shape.circuit.digest(), jp));
        }
        if !root.acc.discharge_sigma(&circuits) {
            return Err(Sigma);
        }
        if !root.acc.discharge_jagged(&layouts) {
            return Err(Jagged);
        }
        // (4) the passenger: (sigma-shaped, jagged-shaped), keyed by the
        // base — the only orphan a spine ever makes.
        let pass = &root.block.passenger;
        let live = pass.iter().any(|(_, c)| entry_live(c));
        if live != self.expect_passenger {
            return Err(Passenger);
        }
        if live {
            let base = self.base.as_ref().expect("a steady root keeps its base");
            let base_d = base.lo.shape.circuit.digest();
            if pass.len() != 2
                || pass[0].0 != digest_f128(&base_d)
                || pass[1].0 != digest_f128(&base_d)
                || !entry_live(&pass[0].1)
                || !entry_live(&pass[1].1)
            {
                return Err(Passenger);
            }
            let pacc = Accumulator {
                registry_digest: root.acc.registry_digest,
                per_type: Vec::new(),
                per_element: Vec::new(),
                sigma: vec![(base_d, pass[0].1.clone())],
                jagged: vec![(base_d, pass[1].1.clone())],
            };
            let jp = base_jp.as_ref().expect("computed with the base");
            if !pacc.discharge_sigma(&[&base.lo.shape.circuit])
                || !pacc.discharge_jagged(&[(base_d, jp)])
            {
                return Err(Passenger);
            }
        }
        Ok(())
    }
}
