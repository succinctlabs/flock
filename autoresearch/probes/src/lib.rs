//! Autoresearch probes for the batch-major (L1′) witness-layout plan.
//! See ../witness-layout-plan.md. E0 = `layout`, E1 = `producer` +
//! `bin/e1_witness_gen`.

pub mod bit_helpers;
pub mod blake3_vwide;
pub mod blake3_witness;
pub mod direct_common;
pub mod e6;
pub mod keccak_vwide;
pub mod keccak_witness;
pub mod layout;
pub mod lincheck_fold;
pub mod producer;
pub mod sha2_vwide;
pub mod sha2_witness;
