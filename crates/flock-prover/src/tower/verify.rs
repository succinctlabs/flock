//! **THE STANDALONE ROOT VERIFIER** (productionization M2): a consumer
//! checks a tower root with a verification key derived from config alone —
//! nothing prover-supplied is trusted but the bundle's proof, publics and
//! commitment.
//!
//! [`TowerVk::generate`] is OPTION-A generation: run the builders over a
//! minimal self-generated tower (six leaves reach all four shapes — chain,
//! FL, base, steady) and keep it as the verifier's material set. Shapes
//! are statement-independent, so the dummy tower's circuits, tables and
//! layouts are byte-identical to any production tower at the same
//! `(cfg, blocks_per_leaf)`. One-time; cache per config. A serialized,
//! digest-pinned VK artifact is the wire-format milestone's job.
//!
//! [`verify_root`] then: (1) checks the bundle's geometry and public
//! length, (2) runs the native circuit verify over the root proof with VK
//! materials — the pcs params come from the VK, never the bundle — (3)
//! REASSEMBLES the published accumulators (main, chain lane, passenger)
//! from the bundle's public segment alone via [`read_acc_entry`], the
//! same decode the builders pin (`check_fold_publics`), and (4) hands
//! them to the shared discharge legs against the VK's native tables plus
//! the statement binding. Reassembly-from-publics is the soundness point:
//! the prover's in-memory accumulators are its own self-check
//! ([`Tower::discharge_root`]), not a consumer's input.

use flock_core::{
    aggregate::Accumulator, matrix_fold::MatrixClaim, pcs::Commitment, verifier::FlockVerifyError,
};

use crate::tower::{
    DOMAIN, F128, FsChallenger, TowerConfig,
    chain::MixedProof,
    driver::{ChainStatement, RootDischargeFailure, Tower},
    env_acc_main_base, env_pass_base, envelope_shape,
    fold_region::read_acc_entry,
    node::{N_KEY_SLOTS, digest_f128, entry_live},
    outer_union,
    query::{LeafOuter, leaf_boolean_lcs},
};

/// The ROOT'S TRANSPORTABLE PART: what a consumer holds beside the
/// statement. Obtained from [`Tower::root_bundle`]; the wire-format
/// milestone gives it a byte form.
pub struct RootBundle<'a> {
    pub(super) public: &'a [F128],
    pub(super) proof: &'a MixedProof,
    pub(super) commitment: &'a Commitment,
}

/// What a green [`verify_root`] CERTIFIED about the claimed span. The
/// ENDPOINTS (`h_start`, `h_end`) are always bound — they ride the app
/// block the proof pins. The COUNT is bound only where the root's shape
/// pins it: a base root folds exactly two leaf pairs, and a
/// passenger-less steady root is exactly three (a deeper spine always
/// carries its live orphan — the match-gate adversarial matrix is what
/// closes the forged-fold escape). From four pairs on, the steady shape
/// is depth-independent BY DESIGN (the spine's one-digest convergence),
/// and the app block carries no count word — so within that class the
/// count is CONSISTENT but not pinned. Binding it exactly means a count
/// word in the app block, composed child-to-parent like the hash
/// endpoints: a recorded protocol follow-up, not a verifier-side patch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SpanBound {
    /// The claimed `n_blocks` is pinned by the root's shape (two or
    /// three leaf pairs).
    Exact,
    /// The endpoints are pinned and the count is consistent (a whole
    /// number of leaf pairs, at least four) but NOT pinned within that
    /// class.
    EndpointsOnly,
}

/// Why [`verify_root`] refused.
#[derive(Debug)]
pub enum TowerVerifyError {
    /// The statement's span is not a whole number of leaf pairs at this
    /// VK's leaf size, or too shallow for a rooted tower.
    Geometry,
    /// The bundle's public segment is not the envelope length.
    PublicsLength,
    /// The root proof itself refused the native circuit verify.
    Proof(FlockVerifyError),
    /// A live keyed slot names a circuit this VK does not know.
    UnknownKey,
    /// A reassembled accumulator refused the native tables, or the
    /// statement binding failed.
    Discharge(RootDischargeFailure),
}

/// The published accumulator blocks' DECODE PLAN: base offsets and
/// per-entry `(k_col, k_row)` widths, all shape constants captured from
/// the VK's reference tower. `read_acc_entry` replays exactly the wire
/// format the builders pin (`check_fold_publics`), so this is the whole
/// layout a consumer needs.
pub(super) struct AccLayout {
    main_base: usize,
    lane_base: usize,
    pass_base: usize,
    /// Widths per UNKEYED main entry, flat: the per-type pairs then the
    /// per-element pairs, one entry per claim.
    main_unkeyed: Vec<(usize, usize)>,
    n_per_type: usize,
    sigma_w: (usize, usize),
    jagged_w: (usize, usize),
    pass_w: [(usize, usize); 2],
    /// The lane block keeps the UN-KEYED entry layout (one registry role,
    /// one implied key — the chain circuit): per-type flat, then sigma,
    /// then jagged.
    lane_unkeyed: Vec<(usize, usize)>,
    lane_sigma_w: (usize, usize),
    lane_jagged_w: (usize, usize),
}

impl AccLayout {
    fn from_tower(t: &Tower) -> AccLayout {
        let env = envelope_shape();
        let w = |c: &MatrixClaim| (c.col.point.len(), c.row.point.len());
        let root = &t.root;
        let mut main_unkeyed = Vec::new();
        for (a, b) in root.acc.per_type.iter().chain(root.acc.per_element.iter()) {
            main_unkeyed.push(w(a));
            main_unkeyed.push(w(b));
        }
        let lane = root.lane_acc.as_ref().expect("the lane rides every level");
        let mut lane_unkeyed = Vec::new();
        for (a, b) in &lane.per_type {
            lane_unkeyed.push(w(a));
            lane_unkeyed.push(w(b));
        }
        AccLayout {
            main_base: env_acc_main_base(&env),
            lane_base: t.fl.fold_pub_base,
            pass_base: env_pass_base(&env),
            main_unkeyed,
            n_per_type: root.acc.per_type.len(),
            sigma_w: w(&root.block.sigma[0].1),
            jagged_w: w(&root.block.jagged[0].1),
            pass_w: [w(&root.block.passenger[0].1), w(&root.block.passenger[1].1)],
            lane_unkeyed,
            lane_sigma_w: w(&lane.sigma[0].1),
            lane_jagged_w: w(&lane.jagged[0].1),
        }
    }
}

/// The tower's VERIFICATION KEY for one `(cfg, blocks_per_leaf)`: a
/// reference tower (the material owners — leaf 0, FL 0, base, steady
/// root) plus the accumulator blocks' decode plan.
pub struct TowerVk {
    pub(super) tower: Tower,
    pub(super) layout: AccLayout,
    pub(super) blocks_per_leaf: usize,
}

impl TowerVk {
    /// OPTION-A GENERATION: prove a minimal six-leaf tower and keep it.
    /// Six leaves is the smallest run that materializes all four shapes
    /// (chain, FL, base, steady = the k = 3 root).
    pub fn generate(cfg: TowerConfig, blocks_per_leaf: usize) -> TowerVk {
        let tower = Tower::prove(cfg, [0u32; 16], blocks_per_leaf, 6);
        let layout = AccLayout::from_tower(&tower);
        TowerVk {
            tower,
            layout,
            blocks_per_leaf,
        }
    }
}

/// Reassemble the published accumulators — main, passenger, chain lane —
/// from a root's PUBLIC SEGMENT alone, per the VK's decode plan. Live
/// keyed slots must name a circuit the VK knows (FL / steady / base);
/// dead slots decode as the zero claim and are dropped, matching the
/// prover-side accumulator's live-entries-only contract.
#[allow(clippy::type_complexity)]
pub(super) fn reassemble(
    public: &[F128],
    vk: &TowerVk,
) -> Result<(Accumulator, Vec<([F128; 2], MatrixClaim)>, Accumulator), TowerVerifyError> {
    let l = &vk.layout;
    let fl_lo = &vk.tower.fl.lo;
    let steady_lo = &vk.tower.root.lo;
    let base_lo = &vk
        .tower
        .base
        .as_ref()
        .expect("the VK tower keeps its base")
        .lo;
    let known: [[u8; 32]; 3] = [
        fl_lo.shape.circuit.digest(),
        steady_lo.shape.circuit.digest(),
        base_lo.shape.circuit.digest(),
    ];
    let name = |key: [F128; 2]| -> Result<[u8; 32], TowerVerifyError> {
        known
            .iter()
            .find(|d| digest_f128(d) == key)
            .copied()
            .ok_or(TowerVerifyError::UnknownKey)
    };

    // ---- ACC_MAIN: unkeyed pairs, then the keyed sigma + jagged slots ----
    let mut p = l.main_base;
    let mut flat: Vec<MatrixClaim> = Vec::with_capacity(l.main_unkeyed.len());
    for &(kc, kr) in &l.main_unkeyed {
        let (_, c) = read_acc_entry(public, &mut p, false, kc, kr);
        flat.push(c);
    }
    let mut pairs = flat
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[a, b]| (a.clone(), b.clone()));
    let per_type: Vec<_> = pairs.by_ref().take(l.n_per_type).collect();
    let per_element: Vec<_> = pairs.collect();
    let mut sigma: Vec<([u8; 32], MatrixClaim)> = Vec::new();
    for _ in 0..N_KEY_SLOTS {
        let (k, c) = read_acc_entry(public, &mut p, true, l.sigma_w.0, l.sigma_w.1);
        if entry_live(&c) {
            sigma.push((name(k)?, c));
        }
    }
    let mut jagged: Vec<([u8; 32], MatrixClaim)> = Vec::new();
    for _ in 0..N_KEY_SLOTS {
        let (k, c) = read_acc_entry(public, &mut p, true, l.jagged_w.0, l.jagged_w.1);
        if entry_live(&c) {
            jagged.push((name(k)?, c));
        }
    }
    let main = Accumulator {
        registry_digest: steady_lo.shape.registry.digest(),
        per_type,
        per_element,
        sigma,
        jagged,
    };

    // ---- THE PASSENGER: two keyed entries (sigma-shaped, jagged-shaped),
    // kept in block form — dead entries ride as decoded ----
    let mut p = l.pass_base;
    let passenger: Vec<([F128; 2], MatrixClaim)> = l
        .pass_w
        .iter()
        .map(|&(kc, kr)| read_acc_entry(public, &mut p, true, kc, kr))
        .collect();

    // ---- ACC_CHAIN: the lane's un-keyed layout, key implied ----
    let chain_d = vk.tower.chain.inner.built.shape.circuit.digest();
    let mut p = l.lane_base;
    let mut lflat: Vec<MatrixClaim> = Vec::with_capacity(l.lane_unkeyed.len());
    for &(kc, kr) in &l.lane_unkeyed {
        let (_, c) = read_acc_entry(public, &mut p, false, kc, kr);
        lflat.push(c);
    }
    let lane_per_type: Vec<_> = lflat
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[a, b]| (a.clone(), b.clone()))
        .collect();
    let (_, lsig) = read_acc_entry(public, &mut p, false, l.lane_sigma_w.0, l.lane_sigma_w.1);
    let (_, ljag) = read_acc_entry(public, &mut p, false, l.lane_jagged_w.0, l.lane_jagged_w.1);
    let lane = Accumulator {
        registry_digest: vk.tower.chain.inner.built.shape.registry.digest(),
        per_type: lane_per_type,
        per_element: Vec::new(),
        sigma: vec![(chain_d, lsig)],
        jagged: vec![(chain_d, ljag)],
    };

    Ok((main, passenger, lane))
}

/// **VERIFY A TOWER ROOT, STANDALONE.** Consumer inputs: the VK, the
/// claimed statement, and the root bundle. Every verifier MATERIAL —
/// circuits, r1cs tables, jagged layouts, union counts, pcs params, the
/// decode plan — comes from the VK; the bundle's publics, proof and
/// commitment are the CLAIM being checked, never the tables it is
/// checked against.
///
/// A green result is qualified by [`SpanBound`]: the endpoints are
/// always certified, the count only up to the root's depth class — read
/// its doc before treating `n_blocks` as verified.
pub fn verify_root(
    vk: &TowerVk,
    statement: &ChainStatement,
    bundle: &RootBundle<'_>,
) -> Result<SpanBound, TowerVerifyError> {
    // (1) geometry: the depth this statement claims, and which of the two
    // node shapes roots it (k = 2 is the base's; k >= 3 the steady's).
    let per_pair = 2 * vk.blocks_per_leaf;
    if statement.n_blocks == 0 || !statement.n_blocks.is_multiple_of(per_pair) {
        return Err(TowerVerifyError::Geometry);
    }
    let k = statement.n_blocks / per_pair;
    if k < 2 {
        return Err(TowerVerifyError::Geometry);
    }
    let root_lo: &LeafOuter = if k == 2 {
        &vk.tower
            .base
            .as_ref()
            .expect("the VK tower keeps its base")
            .lo
    } else {
        &vk.tower.root.lo
    };
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
    let (main, passenger, lane) = reassemble(bundle.public, vk)?;

    // (4) the shared discharge legs + the statement binding, with the
    // VK's reference tower as the table owner.
    vk.tower
        .discharge_with(bundle.public, &lane, &main, &passenger, statement, k >= 4)
        .map_err(TowerVerifyError::Discharge)?;

    // (5) what the shape actually pinned: k = 2 and 3 are exact (base
    // shape; passenger-less steady); k >= 4 certifies the endpoints and
    // the class, not the count — see [`SpanBound`].
    Ok(if k <= 3 {
        SpanBound::Exact
    } else {
        SpanBound::EndpointsOnly
    })
}
