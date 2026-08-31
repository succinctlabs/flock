//! **The recursion tower**: chain leaves folded 2->1 up to a converging
//! spine — the production chain100/chain128 pipeline.
//!
//! [`TowerConfig`] names the two production configurations: the LEAF (the
//! application's BLAKE3 hash-chain segment) proves under the rate-1/2 Fast
//! twin of the tower's security level, and every OUTER (FL / internal /
//! spine) proves under the Slim twin at the m* = 29 / nu* = 14 envelope,
//! always on. The pipeline is three builders:
//!
//! - [`build_chain_proof`] — the LEAF: a Ligerito proof of one chain
//!   segment (the workload).
//! - [`build_fl_node_k`] — the FIRST-LEVEL node: adjacent chain leaves
//!   verified in-circuit (tape replay), their claims folded and proven as
//!   one envelope outer.
//! - [`build_node_outer_app`] — INTERNAL and SPINE nodes: 2->1 recursion
//!   over envelope outers, with the chain-lane accumulator riding along
//!   and the spine inheriting its base's accumulator toward the converged
//!   fixed point (`chain_spine_converges` gates the ONE-digest property).
//!
//! Bench knobs (`CHAIN_BLOCKS`, `BENCH_RUNS`, `TOWER_STEADY`, and the
//! test-only `TOWER_CONFIG=chain100`) live in the `#[test]` harness; the
//! production geometry is typed, never env-var-driven.

use crate::challenger::FsChallenger;
use crate::prover::{self, UnionSlotProverInput};
use crate::r1cs_hashes::blake3;
use crate::r1cs_hashes::merkle_r1cs::SLOT_WORDS;
#[cfg(test)]
use crate::r1cs_hashes::merkle_r1cs::{ChunkPathInput, MerkleTreeLayout, blake3_spec};
use crate::schedule::TableType;
use crate::union::UnionInstance;
#[cfg(test)]
use flock_core::circuit::builder::CircuitBuilder;
use flock_core::circuit::builder::{GateType, ShapeBuilder, SlotWitness, Wire};
use flock_core::matrix_fold::{MatrixClaim, Weight};
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::verifier;
use flock_field::{F128, F256};
use flock_merkle::{self as core_merkle, HashKind};

mod chain;
mod child_walker;
mod config;
#[cfg(test)]
mod e2e_tests;
mod envelope;
mod fl_node;
mod fold_region;
mod fs_chain;
mod gates_blake3;
mod gates_glue;
mod gates_leaf;
mod gates_spine;
mod geometry;
mod gkr;
mod node;
mod online;
mod query;
mod real_walker;
mod tape;

pub use chain::*;
use child_walker::*;
#[allow(unused_imports)]
pub use config::*;
use envelope::*;
pub use fl_node::*;
use fold_region::*;
use fs_chain::*;
use gates_blake3::*;
use gates_glue::*;
use gates_leaf::*;
use gates_spine::*;
use geometry::*;
use gkr::*;
pub use node::*;
use online::*;
pub use query::*;
use real_walker::*;
use tape::*;
