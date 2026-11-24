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

/// The production recursion tower. The LEAF (the application's chain
/// segment — the workload inner proof) proves under the rate-1/2
/// FastAffine profile; the OUTERS (FL / internal / spine) prove under
/// SlimAffine at the m* = 29 / nu* = 14 envelope, always ON. Every inner
/// opening is the affine F128 fold, so the walkers carry no F256 ladder.
///
/// Fast leaf + Slim outers is deliberate: the leaf keeps the SAME tape
/// structure as the outers (the FL/node tape walkers are level-blind) at
/// the cheaper rate-1/2 commit.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TowerConfig {
    /// The 128-bit tower: FastAffine leaf, SlimAffine outers.
    Chain128,
}

impl TowerConfig {
    /// The chain leaf's WORKLOAD inner-proof profile (rate 1/2).
    pub fn leaf_profile(self) -> LigeritoProfile {
        match self {
            TowerConfig::Chain128 => LigeritoProfile::FastAffine,
        }
    }
    /// The recursion-path OUTER profile (rate 1/4, envelope-ON).
    pub fn outer_profile(self) -> LigeritoProfile {
        match self {
            TowerConfig::Chain128 => LigeritoProfile::SlimAffine,
        }
    }
}

/// Whether a tower build re-checks its own output. `Full` runs every
/// prover-side self-check: the fold discharges, the fill-plan pin against
/// the generic walk, the native checkers over the publics, the native
/// chain re-hash, and the outer's self-verify. `Skip` drops them. The
/// proof bytes are identical either way: no check feeds the proof.
/// A skipped check only means a prover bug surfaces at the consumer's
/// `verify_root` (or the root discharge) instead of at the build that
/// caused it. Tests and benches run `Full`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SelfCheck {
    Full,
    Skip,
}

impl SelfCheck {
    pub(super) fn on(self) -> bool {
        self == SelfCheck::Full
    }
}

/// The chain-leaf length the tower tests run at: 1024 blocks is the
/// smallest leaf (`(Fast, log_n 17, initial_k 6)`, m = 24) with a
/// certified affine ladder — at 256 blocks the affine fold bound has no
/// valid certificate (`scripts/affine_fold_params.py`).
#[cfg(test)]
pub(super) const DEV_BLOCKS: usize = 1024;

/// Bench/test knob: which production tower the ignored tower tests and
/// benches exercise. `TOWER_CONFIG=chain128`, or unset, selects the one
/// production tower; any other value is a typo and panics instead of
/// silently selecting the default.
#[cfg(test)]
pub(super) fn test_config() -> TowerConfig {
    match var("TOWER_CONFIG").as_deref() {
        Ok("chain128") | Err(VarError::NotPresent) => TowerConfig::Chain128,
        other => panic!("TOWER_CONFIG must be `chain128`, got {other:?}"),
    }
}

/// Phase B flip-in-place (docs/ag-recursion-plan.md): the chain leaf proves
/// under the AG-skip boolean zerocheck wherever the round-1 prover kernel
/// exists (aarch64 NEON); elsewhere it stays RS until Phase F ports the
/// kernel. Private on purpose — the endgame deletes the RS arm rather than
/// growing `TowerConfig`. `TOWER_LEAF_ZC=rs` forces the RS leaf for A/B
/// measurement on aarch64.
pub(super) fn leaf_zerocheck_ag() -> bool {
    cfg!(target_arch = "aarch64") && !matches!(var("TOWER_LEAF_ZC").as_deref(), Ok("rs"))
}

/// Phase C flip-in-place: the envelope OUTERS (FL / internal / spine)
/// prove under the AG skip on the same terms as the leaf.
/// `TOWER_OUTER_ZC=rs` forces the RS outers for A/B measurement.
pub(super) fn outer_zerocheck_ag() -> bool {
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
