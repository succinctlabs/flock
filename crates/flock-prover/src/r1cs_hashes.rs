//! Per-block R1CS encoders for cryptographic hashes.
//!
//! The legacy [`blake3`] and [`sha2`] modules provide optimized encoders and
//! proving helpers. Their `*_dsl` counterparts define the same relations with
//! the typed Boolean circuit DSL for reference lowering and structural walks.
//! The legacy encoders share low-level row utilities through [`common`].

pub mod blake3;
pub mod blake3_dsl;
/// Shared low-level bit-packing / R1CS-row utilities (carry-save adders,
/// fused adders, lin-id slot helpers) used by the per-hash encoders.
pub mod common;
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
