//! Built-in mixed (multi-table) registry tiers — the wire-format side of
//! Phase 2 M4.
//!
//! A mixed proof's statement names its registry by a small built-in
//! [`MixedRegistryId`] rather than shipping the registry itself: **a
//! registry id pins the FULL registry** — the ordered type list (base
//! matrices, widths, useful bits, const pins) *and* the uniform row
//! capacity `nu` — so two builds agree on a registry id iff they agree on
//! the registry digest the transcript binds. The verifier rebuilds the
//! registry from the id and the proof carries only the id + the per-type
//! counts.
//!
//! Current tiers (both BLAKE3 + SHA-256; slot order is the registry's
//! canonical capacity-area-descending sort — SHA-256 (κ = 15) before
//! BLAKE3 (κ = 14) — and `M = nu + 16`):
//!
//! | id                | nu | capacity/type | M  |
//! |-------------------|----|---------------|----|
//! | `blake3+sha2@nu7` | 7  | 128           | 23 |
//! | `blake3+sha2@nu10`| 10 | 1024          | 26 |
//!
//! Adding a tier (or a type) is a wire-format change: new enum variant,
//! new id byte. The id ↔ byte mapping below is explicit and stable — it is
//! what [`crate::proof_io`] serializes.

use flock_core::challenger::Challenger;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProofJaggedLigerito};
use flock_core::r1cs::BlockR1cs;
use flock_core::schedule::{Registry, TableType};
use flock_core::union::UnionInstance;
use flock_core::verifier::{self, VerifyError};
use serde::{Deserialize, Serialize};

use crate::prover::{self, UnionSlotProverInput};
use crate::r1cs_hashes::{blake3, sha2};

/// A built-in mixed registry tier. Serialized in the wire format as the
/// stable one-byte code of [`Self::code`] (via the explicit serde impls
/// below — never as a serde variant index, so reordering variants cannot
/// silently change the format).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedRegistryId {
    /// BLAKE3 + SHA-256 at uniform capacity 2^7 = 128 invocations per type.
    Blake3Sha2Nu7,
    /// BLAKE3 + SHA-256 at uniform capacity 2^10 = 1024 invocations per type.
    Blake3Sha2Nu10,
}

impl MixedRegistryId {
    /// All tiers, smallest capacity first (the order
    /// [`Self::smallest_fitting`] searches).
    pub const ALL: [MixedRegistryId; 2] = [Self::Blake3Sha2Nu7, Self::Blake3Sha2Nu10];

    /// Stable wire code (u8). New tiers append new codes; codes are never
    /// reused.
    pub fn code(self) -> u8 {
        match self {
            Self::Blake3Sha2Nu7 => 1,
            Self::Blake3Sha2Nu10 => 2,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|id| id.code() == code)
    }

    /// Uniform log2 row capacity of the tier.
    pub fn nu(self) -> usize {
        match self {
            Self::Blake3Sha2Nu7 => 7,
            Self::Blake3Sha2Nu10 => 10,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Blake3Sha2Nu7 => "blake3+sha2@nu7",
            Self::Blake3Sha2Nu10 => "blake3+sha2@nu10",
        }
    }

    /// The smallest tier whose per-type capacity fits `max_count`, or
    /// `None` if every tier is exceeded (split the workload).
    pub fn smallest_fitting(max_count: usize) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|id| max_count <= 1usize << id.nu())
    }
}

impl Serialize for MixedRegistryId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u8(self.code())
    }
}

impl<'de> Deserialize<'de> for MixedRegistryId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let code = u8::deserialize(d)?;
        Self::from_code(code)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown mixed registry id {code}")))
    }
}

/// A materialized tier: the registry plus the per-type `BlockR1cs` views
/// the lincheck circuits are built from. Construction builds the base
/// matrices once; reuse across prove/verify calls.
pub struct MixedSetup {
    pub id: MixedRegistryId,
    pub registry: Registry,
    /// SHA-256 base block (slot 0 — wider type first).
    pub sha2_r1cs: BlockR1cs,
    /// BLAKE3 base block (slot 1).
    pub blake3_r1cs: BlockR1cs,
}

/// Per-type counts of a mixed instance, **in slot order** (SHA-256, then
/// BLAKE3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedCounts {
    pub sha2: usize,
    pub blake3: usize,
}

impl MixedSetup {
    pub fn new(id: MixedRegistryId) -> Self {
        let nu = id.nu();
        let sha2_r1cs = sha2::build_block_r1cs(nu);
        let blake3_r1cs = blake3::build_block_r1cs(nu);
        let registry = Registry::new(
            vec![
                TableType::from_block_r1cs(&sha2_r1cs),
                TableType::from_block_r1cs(&blake3_r1cs),
            ],
            nu,
        );
        // Canonical slot order: SHA-256 (κ = 15) before BLAKE3 (κ = 14).
        debug_assert_eq!(registry.types()[0].k_log, sha2::K_LOG);
        debug_assert_eq!(registry.types()[1].k_log, blake3::K_LOG);
        Self {
            id,
            registry,
            sha2_r1cs,
            blake3_r1cs,
        }
    }

    /// The union instance for the declared counts (panics if a count
    /// exceeds the tier capacity — check against `1 << id.nu()` first).
    pub fn union(&self, counts: MixedCounts) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, vec![counts.sha2, counts.blake3])
    }

    /// PCS params of the dense-stack commit for the declared counts and a
    /// profile: the committed size is the per-proof `dense_m` —
    /// count-dependent under height-`n_t` stacking, floored at the m22
    /// Ligerito config (`UnionInstance::committed_words`) — rate from the
    /// profile, `log_batch_size = 6` as the single-type setups. Both sides
    /// derive it from the public counts: the prover from its inputs, the
    /// verifier from the declared counts, so a tampered bundle
    /// `PcsParams.m` cannot redirect verification.
    pub fn pcs_params(&self, counts: MixedCounts, profile: LigeritoProfile) -> PcsParams {
        let union = self.union(counts);
        PcsParams {
            m: union.dense_m(),
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: 6,
            profile,
        }
    }

    /// Prove the mixed statement: `inputs.sha2.len()` SHA-256 and
    /// `inputs.blake3.len()` BLAKE3 compressions (the declared counts),
    /// dummy rows zeroed via the partial batch-major drivers, through the
    /// union prove entry under the `flock-mixed-v1` binding.
    pub fn prove<Ch: Challenger>(
        &self,
        sha2_inputs: &[sha2::Compression],
        blake3_inputs: &[blake3::Compression],
        profile: LigeritoProfile,
        challenger: &mut Ch,
    ) -> (R1csProofJaggedLigerito, Commitment, R1csClaim) {
        let nu = self.id.nu();
        let counts = MixedCounts {
            sha2: sha2_inputs.len(),
            blake3: blake3_inputs.len(),
        };
        let union = self.union(counts);
        let pcs_params = self.pcs_params(counts, profile);
        let slots = vec![
            UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(sha2_inputs, nu),
                self.sha2_r1cs.csc_lincheck_circuit(),
            ),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(blake3_inputs, nu),
                self.blake3_r1cs.csc_lincheck_circuit(),
            ),
        ];
        prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, challenger)
    }

    /// Verify a mixed proof against the declared counts. The Ligerito
    /// profile is recovered from the commitment's `PcsParams` (as the
    /// single-type CLI does); the remaining params are re-derived from the
    /// tier and the declared counts, so a tampered `PcsParams.m`/rate/batch
    /// in the bundle cannot redirect verification.
    pub fn verify<Ch: Challenger>(
        &self,
        counts: MixedCounts,
        commitment: &Commitment,
        proof: &R1csProofJaggedLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, VerifyError> {
        let union = self.union(counts);
        let pcs_params = self.pcs_params(counts, commitment.params.profile);
        let circuits: [&dyn LincheckCircuit; 2] = [
            self.sha2_r1cs.csc_lincheck_circuit(),
            self.blake3_r1cs.csc_lincheck_circuit(),
        ];
        verifier::verify_ligerito_jagged_union(
            &union,
            &circuits,
            commitment,
            proof,
            &pcs_params,
            challenger,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tier table sanity: codes round-trip, capacities fit as documented,
    /// and `smallest_fitting` picks the smallest adequate tier.
    #[test]
    fn tier_table_and_fitting() {
        for id in MixedRegistryId::ALL {
            assert_eq!(MixedRegistryId::from_code(id.code()), Some(id));
        }
        assert_eq!(MixedRegistryId::from_code(0), None);
        assert_eq!(MixedRegistryId::from_code(3), None);
        assert_eq!(
            MixedRegistryId::smallest_fitting(100),
            Some(MixedRegistryId::Blake3Sha2Nu7)
        );
        assert_eq!(
            MixedRegistryId::smallest_fitting(128),
            Some(MixedRegistryId::Blake3Sha2Nu7)
        );
        assert_eq!(
            MixedRegistryId::smallest_fitting(129),
            Some(MixedRegistryId::Blake3Sha2Nu10)
        );
        assert_eq!(MixedRegistryId::smallest_fitting(1025), None);
    }

    /// The id serializes as its stable one-byte code (inside bincode's u8),
    /// and unknown codes are rejected on read.
    #[test]
    fn registry_id_serde_is_stable_code() {
        for id in MixedRegistryId::ALL {
            let bytes = bincode::serialize(&id).unwrap();
            assert_eq!(bytes, vec![id.code()], "one stable byte");
            let back: MixedRegistryId = bincode::deserialize(&bytes).unwrap();
            assert_eq!(back, id);
        }
        assert!(bincode::deserialize::<MixedRegistryId>(&[0u8]).is_err());
    }

    /// The tier registry reproduces the geometry the M3/M4/M5 mixed tests
    /// pin: slot order SHA-256 then BLAKE3, M = nu + 16, and the
    /// height-`n_t` dense-stack sizes — count-proportional (367 used
    /// chunk-columns: 246 SHA-256 + 121 BLAKE3), floored at the m22
    /// Ligerito config, reaching M4's capacity-height size only at full
    /// utilization.
    #[test]
    fn tier_geometry() {
        let setup = MixedSetup::new(MixedRegistryId::Blake3Sha2Nu7);
        assert_eq!(setup.registry.m_total(), 23);
        // The CLI's flagship mix: counts (100, 37) — dense
        // 100·246 + 37·121 = 29 077 words → committed 2^15 (m = 22), HALF
        // of M4's capacity-height 2^16 (m = 23).
        let counts = MixedCounts {
            sha2: 100,
            blake3: 37,
        };
        let union = setup.union(counts);
        assert_eq!(union.dense_words(), 100 * 246 + 37 * 121);
        assert_eq!(union.dense_m(), 22);
        assert!(!union.compaction_is_identity());
        assert_eq!(
            setup.pcs_params(counts, LigeritoProfile::Fast).m,
            22,
            "nu7 tier at counts (100, 37) commits at m = 22"
        );
        // Full utilization recovers M4's capacity-height size.
        let full = MixedCounts {
            sha2: 128,
            blake3: 128,
        };
        assert_eq!(setup.union(full).dense_words(), (246 + 121) << 7);
        assert_eq!(setup.union(full).dense_m(), 23);
        assert_eq!(setup.pcs_params(full, LigeritoProfile::Fast).m, 23);
    }
}
