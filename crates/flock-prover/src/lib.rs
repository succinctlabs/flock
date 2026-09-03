//! `flock-prover`: the Apple-silicon-optimized end-to-end Flock prover.
//!
//! Builds on [`flock_core`] (the protocol library + verifier) with the
//! top-level prove orchestration ([`prover`]), the monolithic hash R1CS
//! encoders ([`r1cs_hashes`]), and the proof wire format ([`proof_io`]).
//!
//! For convenience, the entire `flock_core` API is re-exported here, so code
//! depending on `flock-prover` can reach `field`, `pcs`, `verifier`, etc.
//! through this crate.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub use flock_core::{
    Zeroable, aggregate, all_core_pool, alloc_zeroed_vec, bits, challenger, circuit, element_r1cs,
    field, genus95_curve_code, init_perf_thread_pool, lincheck, matrix_fold, merkle, ntt, pcs,
    product_gkr, proof, r1cs, schedule, scratch, suboptimal, test_rng, transcript_record, union,
    verifier, zerocheck,
};

pub mod mixed;
pub mod proof_io;
pub mod prover;
pub mod r1cs_hashes;
pub mod tower;
