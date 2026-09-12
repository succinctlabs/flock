//! Per-block R1CS encoders for cryptographic hashes.
//!
//! The legacy [`blake3`] and [`sha2`] modules provide optimized encoders and
//! proving helpers. Their `*_dsl` counterparts define the same relations with
//! the typed Boolean circuit DSL for reference lowering and structural walks.
//! The legacy encoders share low-level row utilities through [`common`].

use std::sync::{Arc, Mutex};

/// A bounded cache for an owned projection keyed by its instantiation shape.
///
/// Replacing the key drops the cache's reference to the old projection;
/// callers that still hold an [`Arc`] keep only the projection they use.
pub(crate) struct ProjectionCache<T> {
    entry: Mutex<Option<(usize, Arc<T>)>>,
}

impl<T> ProjectionCache<T> {
    pub(crate) const fn new() -> Self {
        Self {
            entry: Mutex::new(None),
        }
    }

    pub(crate) fn get_or_init(&self, key: usize, build: impl FnOnce() -> T) -> Arc<T> {
        let mut entry = self
            .entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((cached_key, value)) = &*entry
            && *cached_key == key
        {
            return Arc::clone(value);
        }

        let value = Arc::new(build());
        *entry = Some((key, Arc::clone(&value)));
        value
    }
}

pub mod blake3;
pub mod blake3_dsl;
/// Shared low-level bit-packing / R1CS-row utilities (carry-save adders,
/// fused adders, lin-id slot helpers) used by the per-hash encoders.
pub mod common;
pub mod dsl;
/// The Fiat–Shamir chain: BLAKE3 over a transcript with a finalize forked at
/// every squeeze — the FS chain's witness generator, over [`blake3`]'s rows.
pub mod fs_chain;
/// The recursion tower's Merkle glue gates (`SwapTable`, `BitSpreadTable`,
/// `PowMaskTable`, `FamilyTransposeTileTable`): small R1CS tables that join
/// the tower's in-circuit Merkle openings to the BLAKE3 compressions.
pub mod merkle_glue;
/// Merkle-path layout and node-hash spec (`MerkleTreeLayout`, `HashSpec`,
/// `ChunkPathInput`, `SLOT_WORDS`) that the tower's Merkle gates and
/// [`merkle_glue`] build on.
pub mod merkle_r1cs;
pub mod sha2;
pub mod sha2_dsl;

#[cfg(test)]
mod tests {
    use super::ProjectionCache;
    use std::cell::Cell;
    use std::sync::Arc;

    #[test]
    fn projection_cache_reuses_one_shape_and_replaces_another() {
        let cache = ProjectionCache::new();
        let builds = Cell::new(0);
        let first = cache.get_or_init(3, || {
            builds.set(builds.get() + 1);
            30
        });
        let repeated = cache.get_or_init(3, || unreachable!("shape is cached"));
        assert!(Arc::ptr_eq(&first, &repeated));
        assert_eq!(builds.get(), 1);

        let second = cache.get_or_init(4, || {
            builds.set(builds.get() + 1);
            40
        });
        assert_eq!(*second, 40);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(builds.get(), 2);
    }
}
