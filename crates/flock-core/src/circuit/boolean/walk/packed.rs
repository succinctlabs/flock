//! Packed BatchMajor output for the current identity-C union path.

use super::{ForwardTrace, WalkError, WalkPlan, checked_capacity};
use crate::field::F128;
use crate::union::SlotWitnessDest;

impl WalkPlan {
    /// Write a partial batch directly into one union slot.
    ///
    /// The current Boolean union stores `z`, `Az`, and `Bz` and derives the C
    /// claim directly from `z`, so this adapter explicitly requires identity
    /// C. General-C walks continue to use [`Self::forward_batch`] and retain
    /// `Cz`; no canonical information is discarded implicitly.
    pub fn forward_batch_identity_c_into_slot<I: AsRef<[bool]>>(
        &self,
        inputs: &[I],
        k_log: usize,
        n_blocks_log: usize,
        destination: SlotWitnessDest<'_>,
    ) -> Result<(), WalkError> {
        if !self.c_is_identity {
            return Err(WalkError::IdentityCRequired);
        }
        if k_log < 7 {
            return Err(WalkError::PackedKLog(k_log));
        }
        let block_capacity = self.validate_capacity(k_log)?;
        let batch_capacity =
            checked_capacity(n_blocks_log).ok_or(WalkError::InvalidBatchLog(n_blocks_log))?;
        if inputs.len() > batch_capacity {
            return Err(WalkError::BatchCount {
                capacity: batch_capacity,
                actual: inputs.len(),
            });
        }
        for input in inputs {
            self.validate_input(input.as_ref())?;
        }

        let chunks = block_capacity >> 7;
        let expected = chunks
            .checked_mul(batch_capacity)
            .ok_or(WalkError::BatchSizeOverflow {
                blocks: batch_capacity,
                block_capacity,
            })?;
        if destination.z.len() != expected
            || destination.a.len() != expected
            || destination.b.len() != expected
        {
            return Err(WalkError::PackedDestinationLength {
                expected,
                z: destination.z.len(),
                a: destination.a.len(),
                b: destination.b.len(),
            });
        }

        if !destination.elide_padding_writes {
            destination.z.fill(F128::ZERO);
            destination.a.fill(F128::ZERO);
            destination.b.fill(F128::ZERO);
        }

        let mut trace = ForwardTrace::default();
        for (block, input) in inputs.iter().enumerate() {
            self.forward_into(input.as_ref(), k_log, &mut trace)?;
            for chunk in 0..chunks {
                let source = chunk << 7;
                let target = chunk * batch_capacity + block;
                destination.z[target] = pack(&trace.z[source..source + 128]);
                destination.a[target] = pack(&trace.a_z[source..source + 128]);
                destination.b[target] = pack(&trace.b_z[source..source + 128]);
            }
        }
        Ok(())
    }
}

fn pack(bits: &[bool]) -> F128 {
    debug_assert_eq!(bits.len(), 128);
    let mut lo = 0u64;
    let mut hi = 0u64;
    for (bit, &value) in bits.iter().enumerate() {
        if value {
            if bit < 64 {
                lo |= 1 << bit;
            } else {
                hi |= 1 << (bit - 64);
            }
        }
    }
    F128::new(lo, hi)
}
