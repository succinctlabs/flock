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
//! [`Tower::prove`] drives the three end to end — the production caller —
//! and [`Tower::discharge_root`] settles the root-side residue.
//! [`verify_root`] checks a root STANDALONE: a consumer holds only the
//! statement and [`Tower::root_bundle`]'s artifacts, and every table
//! comes from a [`TowerVk`] derived from config.
//!
//! Bench knobs (`CHAIN_BLOCKS`, `BENCH_RUNS`, `TOWER_STEADY`, and the
//! test-only `TOWER_CONFIG=chain100`) live in the `#[test]` harness; the
//! production geometry is typed, never env-var-driven.

use flock_core::{
    circuit::builder::{GateType, ShapeBuilder, SlotWitness, Wire},
    matrix_fold::{MatrixClaim, Weight},
    pcs::{PcsParams, ligerito::LigeritoProfile},
};
use flock_field::{F128, F256};
use flock_merkle::HashKind;
#[cfg(test)]
use {
    crate::tower::config::test_config, crate::tower::fl_node::chain_jagged_params,
    crate::tower::node::node_jagged_params,
};

pub use crate::tower::{
    chain::{ChainProof, build_chain_proof},
    config::TowerConfig,
    driver::{ChainStatement, RootDischargeFailure, Tower},
    fl_node::{FlNode, build_fl_node, build_fl_node_k},
    node::{ChainLane, MainBlock, NodeOut, SpineIn, build_node_outer_app},
    query::LeafOuter,
    verify::{
        RootBundle, SpanBound, TowerVerifyError, TowerVk, TowerVkFingerprint, verify_root,
        verify_root_bytes,
    },
};
// The wire format (`proof_io`'s tower-root bundle) carries a `MixedProof`.
pub(crate) use crate::tower::chain::MixedProof;
use crate::{
    challenger::FsChallenger,
    prover::UnionSlotProverInput,
    r1cs_hashes::merkle_r1cs::SLOT_WORDS,
    schedule::TableType,
    tower::{
        chain::{MixedInner, native_chain},
        child_walker::{
            ChildSlots, ChildTape, ZskipTapeRec, ZskipWires, check_child_region, emit_child_region,
            expected_child_tail_schedule,
        },
        config::{leaf_zc_ag, outer_union, outer_zc_ag, pcs_batch_for, tower_fold_grinding},
        envelope::{
            ENV_ACC_MAIN_WORDS, EnvShape, EnvTail, declare_envelope_slots, env_acc_chain_base,
            env_acc_main_base, env_app_base, env_pass_base, envelope_shape, outer_lanes,
            pad_envelope_counts, slot_cached, steady_reps,
        },
        fl_node::chain_blake_r1cs,
        fold_region::{
            FoldPub, challenge_word_locs, check_ag_skip_publics, check_fold_publics,
            check_jagged_fold_publics, emit_fold_region, emit_jagged_fold_region, fold_region_ops,
            jagged_fold_region_ops, labeled_bytes_payloads, locate_and_pin_folds,
            locate_and_pin_jagged_folds, read_acc_entry, replay_fold_endpoints,
            replay_jagged_fold_endpoints,
        },
        fs_chain::{
            MergedChain, ag_seed_bytes, assert_chain_replays, bytes_payload_mask, cw,
            decode_ag_point, duplex_row_count_model, emit_fs_chain, emit_fs_chain_partitioned,
            flatten_ops, merge_chain,
        },
        gates_blake3::{
            Blake3Gate, CHUNK_END, CHUNK_START, DOMAIN, IV, PARENT, ROOT, digest_words,
            hash_to_digest, pack_params, pack4, pack8, unpack8,
        },
        gates_glue::{
            BitSpreadGate, BitSpreadTable, FamilyTransposeTileGate, FamilyTransposeTileTable,
            PowMaskGate, PowMaskTable, SwapGate, SwapTable,
        },
        gates_leaf::{LeafEvalGate, LeafEvalGate256, build_mac256},
        gates_spine::{
            AssistLayerGate, MacGate, MacGate256, MergedRoundGate, PrefixGate, PrefixGate256,
            ResidualAccGate256, ResidualPrefix3Gate256, ResidualWeightsGate256, SpineGate,
            SpineGate256, ZcRoundGate, emit_mac256, emit_spine256, live_element_input_from_rows,
        },
        geometry::{
            CollapsedSlots, Lvl, balance_extra_rows, cap_payloads, cap_wires, emit_publics_hash,
            l0_ood_z_index, level_geometry, level_query_phase_b3_rows, level_sources,
            observed_f256, payload_words, query_phase_b3_rows, replay_ligerito_spine256,
            strat_scheds,
        },
        gkr::{
            ElPiopRec, GkrLayerRec, GkrRec, assertion_mac, circuit_structure_claim_wires,
            emit_ag_point_binding, emit_boolean_reported_check, emit_element_reported_check,
            emit_lagrange_lows, emit_recombination, pin_recombination,
        },
        online::Online,
        query::{
            check_residual_publics, emit_pow_checks, emit_query_phase, emit_recorded_pow_checks,
            emit_residual_region, leaf_boolean_lcs, leaf_boolean_mats,
        },
        real_walker::{
            RealRegion, RealTape, check_real_child_region, emit_family_h, emit_real_child_region,
            expected_real_tail_schedule,
        },
        tape::{
            InnerPd, MpRec, OpenLevel, PdRec, PiopRec, RoundRec, parse_open_levels,
            squeeze_word_wire,
        },
    },
    union::UnionInstance,
};
mod chain;
mod child_walker;
mod config;
mod driver;
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
mod verify;
mod walker_common;
