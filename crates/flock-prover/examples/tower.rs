//! The recursion tower, end to end: prove a BLAKE3 compression chain and
//! fold it to ONE recursable root proof, then discharge the root-side
//! residue.
//!
//! ```sh
//! cargo run --release --example tower -- [n_leaves] [blocks_per_leaf] [chain128|chain100]
//! ```
//!
//! Defaults: 8 leaves x 256 compressions under the 128-bit tower — the
//! smallest run that reaches the STEADY spine shape (a base plus two
//! spine folds) and boards the passenger. The production leaf is 2^18
//! compressions (`blocks_per_leaf = 262144`); expect minutes there.

use std::{env, time::Instant};

use flock_prover::tower::{Tower, TowerConfig};

fn main() {
    let mut args = env::args().skip(1);
    let n_leaves: usize = args
        .next()
        .map_or(8, |v| v.parse().expect("n_leaves: an even integer >= 4"));
    let blocks_per_leaf: usize = args
        .next()
        .map_or(256, |v| v.parse().expect("blocks_per_leaf: a positive integer"));
    let cfg = match args.next().as_deref() {
        None | Some("chain128") => TowerConfig::Chain128,
        Some("chain100") => TowerConfig::Chain100,
        Some(other) => panic!("unknown tower config {other:?}: chain128 or chain100"),
    };

    // The demo chain starts at the zero state; a deployment starts at its
    // application's own h_start.
    let h_start = [0u32; 16];

    let t0 = Instant::now();
    let tower = Tower::prove(cfg, h_start, blocks_per_leaf, n_leaves);
    let prove_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    tower
        .discharge_root()
        .expect("the root-side residue discharges");
    let discharge_s = t1.elapsed().as_secs_f64();

    let s = tower.statement();
    println!(
        "\nTOWER {:?}: {n_leaves} leaves x {blocks_per_leaf} compressions = {} total\n  \
         h_end = {:08x}{:08x}.. (== H^{}(h_start))\n  \
         ONE root proof: {:.1} KiB | prove {prove_s:.1}s | root discharge {discharge_s:.3}s",
        cfg,
        s.n_blocks,
        s.h_end[0],
        s.h_end[1],
        s.n_blocks,
        tower.root_proof_kib(),
    );
}
