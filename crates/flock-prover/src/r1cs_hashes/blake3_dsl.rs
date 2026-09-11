//! BLAKE3 compression with declared columns and a separate compatibility layout.

use std::sync::{Arc, OnceLock};

use flock_core::circuit::boolean::{
    BooleanCircuit, CircuitBuilder, CompiledColumns, PhysicalLayout, WalkPlan,
};
use flock_core::r1cs::BlockR1cs;

use super::{ProjectionCache, blake3};

mod columns;
mod eval;
mod layout;
pub use columns::{Blake3Cols, Blake3Schema};

const CV_PORT: &str = "cv";
const MESSAGE_PORT: &str = "message";
const COUNTER_PORT: &str = "counter";
const BLOCK_LEN_PORT: &str = "block_len";
const FLAGS_PORT: &str = "flags";
const OUT_LO_PORT: &str = "out_lo";
const OUT_HI_PORT: &str = "out_hi";

/// One reusable BLAKE3 DSL artifact and its legacy-compatible placement.
#[derive(Debug)]
pub struct Blake3DslCircuit {
    compiled: CompiledColumns<Blake3Schema>,
    layout: PhysicalLayout,
}

impl Blake3DslCircuit {
    pub fn compiled(&self) -> &CompiledColumns<Blake3Schema> {
        &self.compiled
    }

    pub fn circuit(&self) -> &BooleanCircuit {
        self.compiled.circuit()
    }

    pub fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    /// Lower to the same block-diagonal shape as the legacy relation.
    pub fn to_block_r1cs(&self, n_blocks_log: usize) -> BlockR1cs {
        assert!(
            n_blocks_log >= 3,
            "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
        );
        self.circuit()
            .to_block_r1cs_with_layout(blake3::K_LOG, blake3::K_SKIP, n_blocks_log, &self.layout)
            .expect("the fixed BLAKE3 compatibility layout must be valid")
    }

    /// Evaluate one compression in legacy physical witness order.
    pub fn evaluate_block(
        &self,
        cv: &[u32; 8],
        message: &[u32; 16],
        counter: u64,
        block_len: u32,
        flags: u32,
    ) -> Vec<bool> {
        let (_, logical) = self.generate_trace(cv, message, counter, block_len, flags);
        super::dsl::physical_witness(&logical, &self.layout, blake3::K)
    }

    /// Supply named inputs and compute all declared witnesses in logical order.
    pub fn generate_trace(
        &self,
        cv: &[u32; 8],
        message: &[u32; 16],
        counter: u64,
        block_len: u32,
        flags: u32,
    ) -> (Blake3Cols<bool>, Vec<bool>) {
        self.compiled
            .evaluate(|cols| {
                super::dsl::populate_words(cols.cv, cv);
                super::dsl::populate_words(cols.message, message);
                super::dsl::populate_words(cols.counter, &[counter as u32, (counter >> 32) as u32]);
                super::dsl::populate_words([cols.block_len], &[block_len]);
                super::dsl::populate_words([cols.flags], &[flags]);
            })
            .expect("BLAKE3 has no rejecting general constraints")
    }

    /// Compile the structural execution plan at the compatibility layout.
    pub fn walk_plan(&self) -> WalkPlan {
        self.circuit()
            .walk_plan_with_layout(&self.layout)
            .expect("the fixed BLAKE3 compatibility layout must be valid")
    }
}

/// Build the BLAKE3 DSL artifact once per process.
pub fn blake3_circuit() -> &'static Blake3DslCircuit {
    static CIRCUIT: OnceLock<Blake3DslCircuit> = OnceLock::new();
    CIRCUIT.get_or_init(build_blake3_circuit)
}

/// Cache only the compiled walk, dropping the construction artifact and its
/// normalized supports after compilation.
pub fn blake3_walk_projection() -> &'static WalkPlan {
    static WALK: OnceLock<WalkPlan> = OnceLock::new();
    WALK.get_or_init(|| build_blake3_circuit().walk_plan())
}

/// Return the cached relation for one batch shape without retaining the
/// construction artifact or structural walk.
pub fn blake3_relation_projection(n_blocks_log: usize) -> Arc<BlockR1cs> {
    assert!(
        n_blocks_log >= 3,
        "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
    );
    static RELATION: ProjectionCache<BlockR1cs> = ProjectionCache::new();
    RELATION.get_or_init(n_blocks_log, || {
        build_blake3_circuit().to_block_r1cs(n_blocks_log)
    })
}

fn build_blake3_circuit() -> Blake3DslCircuit {
    let compiled = CircuitBuilder::compile(Blake3Schema, eval::eval);
    assert_eq!(compiled.circuit().value_count(), blake3::USEFUL_BITS);
    assert_eq!(compiled.circuit().row_count(), blake3::USEFUL_BITS);
    let layout = layout::layout(compiled.circuit(), &compiled.columns());
    Blake3DslCircuit { compiled, layout }
}

#[cfg(test)]
#[path = "blake3_dsl/tests.rs"]
mod tests;
