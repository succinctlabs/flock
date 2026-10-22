// Reuse the circuit and its existing tests without a core-to-prover dependency.
include!("../../flock-core/examples/boolean_columns.rs");

#[cfg(test)]
#[path = "boolean_columns/proof.rs"]
mod proof;
