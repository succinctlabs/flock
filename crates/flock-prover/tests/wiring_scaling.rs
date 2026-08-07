//! **How does the copy-constraint wiring scale with the cell space?**
//!
//! The question decides whether collapsing the Merkle composites into one
//! plain BLAKE3 table is worth doing. That change would cut the lincheck's
//! swept nonzeros ~5x (105.1M → 21.0M, measured in
//! `circuit_merkle::mvp5_all_levels_query_phase`) by removing four table
//! types — but it pays for it in `nu`: the composite packs 30 compressions
//! into ONE row, so splitting them out takes `nu` from 8 to 14 and `mu` from
//! 16 to 21.
//!
//! `product_gkr` runs over the whole cell space, so if its cost is linear in
//! `2^mu` the collapse trades ~71 ms of lincheck for ~54 ms of wiring and
//! nets ~10%. Two measured points (mu 15 → 0.95 ms, mu 16 → 1.74 ms) were all
//! we had, and extrapolating five doublings off two points is not a basis for
//! a rewrite. So: measure it directly.
//!
//! Isolated on purpose — `prove_wiring` is timed on its own rather than
//! inside a full prove, so nothing else moves.

use flock_core::circuit::prove_wiring;
use std::hint::black_box;
use std::time::Instant;

use flock_core::circuit::builder::{GateType, ShapeBuilder, SlotWitness};
use flock_core::element_r1cs::{ElementTableBuilder, ElementTableType};
use flock_core::field::F128;
use flock_core::schedule::IoWord;
use flock_prover::challenger::FsChallenger;
use flock_prover::schedule::TableType;
use flock_prover::union::UnionInstance;
use std::sync::Arc;

/// The smallest gate that produces real wiring: `c = a · b`, two inputs and
/// one output, so a chain of them gives one non-singleton class per gate.
struct MultGate {
    ty: Arc<ElementTableType>,
}

impl MultGate {
    fn new() -> Self {
        let mut b = ElementTableBuilder::new(2);
        b.free_wire(0).free_wire(1).mult(2, 0, 1);
        Self {
            ty: Arc::new(b.build().expect("mult block is valid")),
        }
    }
}

impl GateType for MultGate {
    type Row = ();
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::element(self.ty.clone()).with_io_schema(vec![
            IoWord::input(0),
            IoWord::input(1),
            IoWord::output(2),
        ])
    }

    fn eval(&self, _inputs: &[F128], _hint: &()) -> (Vec<F128>, ()) {
        // Values are irrelevant to the wiring cost — the grand product runs
        // over the cell space regardless of what the cells hold. All-zero is
        // a CONSISTENT assignment (every class holds 0, and 0·0 = 0), so the
        // circuit being timed is one a prover could honestly produce.
        (vec![F128::ZERO], ())
    }

    fn witness(&self, _rows: &[()], nu: usize) -> SlotWitness {
        SlotWitness::Element(vec![F128::ZERO; self.ty.width() << nu])
    }
}

/// Time `prove_wiring` across the cell-space sizes the collapse would move
/// between. Prints a table with the measured scaling factor per doubling.
#[test]
#[ignore] // Minutes and ~1 GB at the top end. `-- --ignored`.
fn wiring_cost_across_mu() {
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    println!(
        "\nWIRING COST vs CELL SPACE ({threads} threads)\n  \
         {:>4} {:>4} {:>10} {:>12} {:>10} {:>8}",
        "nu", "mu", "gates", "cells", "prove_wiring", "x/doubl"
    );

    let mut prev: Option<(usize, f64)> = None;
    for nu in 12..=19usize {
        let n_gates = (1usize << nu) - 1; // leave a row for nothing; capacity is 2^nu

        let mut sb = ShapeBuilder::new(nu);
        let g = sb.slot(MultGate::new());
        // A chain: every gate consumes the previous gate's output, so each
        // output wire is a real 2-cell class rather than a singleton.
        let seed = sb.public_input();
        let mut acc = seed;
        for _ in 0..n_gates {
            let a = sb.input();
            acc = sb.gate(g, &[a, acc])[0];
        }
        sb.publish(acc);
        let shape = sb.finish().expect("valid circuit");

        let union = UnionInstance::new(&shape.registry, shape.counts.clone());
        let mu = shape.circuit.cells().mu();
        let packed = vec![F128::ZERO; union.packed_len()];
        let public = vec![F128::ZERO; shape.circuit.num_public()];

        // Warm once — the first call pays pooled-buffer faults.
        {
            let mut ch = FsChallenger::new(b"wiring-scaling");
            black_box(prove_wiring(&shape.circuit, &packed, &public, &mut ch));
        }
        let mut ch = FsChallenger::new(b"wiring-scaling");
        let t = Instant::now();
        let out = prove_wiring(&shape.circuit, &packed, &public, &mut ch);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        black_box(out);

        let factor = match prev {
            Some((pmu, pms)) if mu == pmu + 1 => format!("{:.2}", ms / pms),
            _ => "-".to_string(),
        };
        println!(
            "  {nu:>4} {mu:>4} {n_gates:>10} {:>12} {ms:>9.2} ms {factor:>8}",
            1usize << mu
        );
        prev = Some((mu, ms));
    }
    println!();
}
