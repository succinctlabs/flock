#[cfg(test)]
use std::env::VarError;
use std::env::var;

use flock_core::{matrix_fold::FoldGrinding, pcs::ligerito::embedded_initial_k_or_default};

use crate::{
    schedule::Registry,
    tower::{HashKind, LigeritoProfile, PcsParams, UnionInstance},
};

/// The L0 interleave for a content-sized commit: the embedded config's
/// own `initial_k` (6 everywhere except m29 Fast/Slim = 5 — the
/// recursion-node row-width choice). `prover_config_for` rejects a
/// mismatched batch, so every params site whose `m` is content-derived
/// must go through this.
pub(super) fn pcs_batch_for(union: &UnionInstance, profile: LigeritoProfile) -> usize {
    embedded_initial_k_or_default(union.dense_m(), profile)
}

/// The two production recursion towers. The LEAF (the application's chain
/// segment — the workload inner proof) proves under the rate-1/2 Fast twin
/// of the tower's security level; the OUTERS (FL / internal / spine) prove
/// under the Slim twin at the m* = 29 / nu* = 14 envelope, always ON.
///
/// Fast leaf + Slim outers is deliberate. The leaf keeps the SAME tape
/// structure as Fast (the FL/node tape walkers are level-blind), while its
/// query count follows the tower's security level: a 100-bit recursion
/// carries a 100-bit leaf (Fast100, 448q — a 128-bit leaf under a 100-bit
/// recursion balloons the FL's replayed transcript past the arity-2
/// envelope), and the 128-bit aggressive recursion carries the aggressive
/// Fast128 leaf (rate-1/2 on the rate+2 ladder; m32: Σq 675 → 527).
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TowerConfig {
    /// The 100-bit tower: Fast100 leaf, Slim100 outers.
    Chain100,
    /// The 128-bit tower on the aggressive rate ladder: Fast128 leaf,
    /// Slim128 outers.
    Chain128,
}

impl TowerConfig {
    /// The chain leaf's WORKLOAD inner-proof profile (rate 1/2).
    pub fn leaf_profile(self) -> LigeritoProfile {
        match self {
            TowerConfig::Chain100 => LigeritoProfile::Fast100,
            TowerConfig::Chain128 => LigeritoProfile::Fast,
        }
    }
    /// The recursion-path OUTER profile (rate 1/4, envelope-ON).
    pub fn outer_profile(self) -> LigeritoProfile {
        match self {
            TowerConfig::Chain100 => LigeritoProfile::Slim100,
            TowerConfig::Chain128 => LigeritoProfile::Slim,
        }
    }
}

/// Bench/test knob: which production tower the ignored tower tests and
/// benches exercise. `TOWER_CONFIG=chain100` selects [`TowerConfig::Chain100`];
/// `TOWER_CONFIG=chain128`, or unset, the 128-bit production tower. Any other
/// value is a typo and panics instead of silently selecting the default.
#[cfg(test)]
pub(super) fn test_config() -> TowerConfig {
    match var("TOWER_CONFIG").as_deref() {
        Ok("chain100") => TowerConfig::Chain100,
        Ok("chain128") | Err(VarError::NotPresent) => TowerConfig::Chain128,
        other => panic!("TOWER_CONFIG must be `chain100` or `chain128`, got {other:?}"),
    }
}

/// Phase B flip-in-place (docs/ag-recursion-plan.md): the chain leaf proves
/// under the AG-skip boolean zerocheck wherever the round-1 prover kernel
/// exists (aarch64 NEON); elsewhere it stays RS until Phase F ports the
/// kernel. Private on purpose — the endgame deletes the RS arm rather than
/// growing `TowerConfig`. `TOWER_LEAF_ZC=rs` forces the RS leaf for A/B
/// measurement on aarch64.
pub(super) fn leaf_zc_ag() -> bool {
    cfg!(target_arch = "aarch64") && !matches!(var("TOWER_LEAF_ZC").as_deref(), Ok("rs"))
}

/// Phase C flip-in-place: the envelope OUTERS (FL / internal / spine)
/// prove under the AG skip on the same terms as the leaf.
/// `TOWER_OUTER_ZC=rs` forces the RS outers for A/B measurement.
pub(super) fn outer_zc_ag() -> bool {
    cfg!(target_arch = "aarch64") && !matches!(var("TOWER_OUTER_ZC").as_deref(), Ok("rs"))
}

pub(super) fn tower_fold_grinding(cfg: TowerConfig) -> FoldGrinding {
    let profile = cfg.outer_profile();
    PcsParams {
        m: 22,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: 5,
        profile,
        num_lanes: None,
        merkle_hash: HashKind::Blake3,
    }
    .matrix_fold_grinding()
}

/// The ENVELOPE dense floor `m*` (wall 2): every recursion-path OUTER —
/// leaf and node alike — commits at this size, so a node's children look
/// ONE shape regardless of level (an L1 node's leaf children carry the
/// same query geometry as an L2 node's node children).
///
/// Ron's call 2026-08-06: m* = 29 (the fixed point closes with ~2x slack;
/// every Slim level commits m29). Re-targeting the tight m* = 28 (needs
/// the mac shave −8k words + publics arithmetization −40k+ at the fixed
/// point) is a deliberate future re-pin; `envelope_content_probe` is the
/// instrument that sizes it.
pub(super) const ENVELOPE_FLOOR_M: usize = 29;

/// A recursion-path OUTER's union instance, with the envelope floor
/// applied. Every instance over a leaf/node OUTER shape must come from
/// here — prover, verifier and tape recorder alike: the floor is
/// STATEMENT data, like the counts.
pub(super) fn outer_union<'r>(registry: &'r Registry, counts: Vec<usize>) -> UnionInstance<'r> {
    let mut u = UnionInstance::new(registry, counts);
    u.set_dense_floor(ENVELOPE_FLOOR_M);
    u
}
