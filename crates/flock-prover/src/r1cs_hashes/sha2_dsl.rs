//! SHA-256 compression with declared columns, reusable additions, and a separate layout.

use std::sync::{Arc, OnceLock};

use flock_core::circuit::boolean::{
    BooleanCircuit, CircuitBuilder, CompiledColumns, PhysicalLayout, WalkPlan,
};
use flock_core::r1cs::BlockR1cs;

use super::{ProjectionCache, sha2};

mod columns;
mod eval;
mod layout;
pub use columns::{Sha256Cols, Sha256Schema};

const H_PORT: &str = "h_in";
const MESSAGE_PORT: &str = "message";
const H_OUT_PORT: &str = "h_out";

/// One reusable SHA-256 DSL artifact and its legacy-compatible placement.
#[derive(Debug)]
pub struct Sha256DslCircuit {
    compiled: CompiledColumns<Sha256Schema>,
    layout: PhysicalLayout,
}

impl Sha256DslCircuit {
    pub fn compiled(&self) -> &CompiledColumns<Sha256Schema> {
        &self.compiled
    }

    pub fn circuit(&self) -> &BooleanCircuit {
        self.compiled.circuit()
    }

    pub fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    /// Lower the DSL artifact to the same block-diagonal shape as the legacy
    /// SHA-256 relation.
    pub fn to_block_r1cs(&self, n_blocks_log: usize) -> BlockR1cs {
        assert!(
            n_blocks_log >= 3,
            "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
        );
        self.circuit()
            .to_block_r1cs_with_layout(sha2::K_LOG, sha2::K_SKIP, n_blocks_log, &self.layout)
            .expect("the fixed SHA-256 compatibility layout must be valid")
    }

    /// Reference evaluation of one compression, returned in legacy physical
    /// witness order and padded to `K` bits.
    pub fn evaluate_block(&self, h_in: &[u32; 8], message: &[u32; 16]) -> Vec<bool> {
        let (_, logical) = self.generate_trace(h_in, message);
        super::dsl::physical_witness(&logical, &self.layout, sha2::K)
    }

    /// Supply named inputs and compute all declared witnesses in logical order.
    pub fn generate_trace(
        &self,
        h_in: &[u32; 8],
        message: &[u32; 16],
    ) -> (Sha256Cols<bool>, Vec<bool>) {
        self.compiled
            .evaluate(|cols| {
                super::dsl::populate_words(cols.h_in, h_in);
                super::dsl::populate_words(cols.message, message);
            })
            .expect("SHA-256 has no rejecting general constraints")
    }

    /// Compile the structural forward/reverse execution plan at the legacy
    /// physical layout.
    pub fn walk_plan(&self) -> WalkPlan {
        self.circuit()
            .walk_plan_with_layout(&self.layout)
            .expect("the fixed SHA-256 compatibility layout must be valid")
    }
}

/// Build the SHA-256 DSL artifact once per process.
pub fn sha256_circuit() -> &'static Sha256DslCircuit {
    static CIRCUIT: OnceLock<Sha256DslCircuit> = OnceLock::new();
    CIRCUIT.get_or_init(build_sha256_circuit)
}

/// Cache only the compiled walk, dropping the construction artifact and its
/// normalized supports after compilation.
pub fn sha256_walk_projection() -> &'static WalkPlan {
    static WALK: OnceLock<WalkPlan> = OnceLock::new();
    WALK.get_or_init(|| build_sha256_circuit().walk_plan())
}

/// Return the cached relation for one batch shape without retaining the
/// construction artifact or structural walk.
pub fn sha256_relation_projection(n_blocks_log: usize) -> Arc<BlockR1cs> {
    assert!(
        n_blocks_log >= 3,
        "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
    );
    static RELATION: ProjectionCache<BlockR1cs> = ProjectionCache::new();
    RELATION.get_or_init(n_blocks_log, || {
        build_sha256_circuit().to_block_r1cs(n_blocks_log)
    })
}

fn build_sha256_circuit() -> Sha256DslCircuit {
    let compiled = CircuitBuilder::compile(Sha256Schema, eval::eval);
    assert_eq!(compiled.circuit().value_count(), sha2::USEFUL_BITS);
    assert_eq!(compiled.circuit().row_count(), sha2::USEFUL_BITS);
    let layout = layout::layout(compiled.circuit(), &compiled.columns());
    Sha256DslCircuit { compiled, layout }
}

#[cfg(test)]
#[path = "sha2_dsl/tests.rs"]
mod tests;
