//! The recursion tower, end to end: prove a BLAKE3 compression chain and
//! fold it to ONE recursable root proof, then discharge the root-side
//! residue.
//!
//! ```sh
//! cargo run --release --example tower -- [n_leaves] [blocks_per_leaf] [chain128] [verify] [checked]
//! ```
//!
//! Defaults: 8 leaves x 256 compressions under the 128-bit tower — the
//! smallest run that reaches the STEADY spine shape (a base plus two
//! spine folds) and boards the passenger. The production leaf is 2^18
//! compressions (`blocks_per_leaf = 262144`); expect minutes there.
//!
//! With `verify`, the run also generates a verification key (a one-time
//! six-leaf reference tower) and checks the root STANDALONE through
//! `verify_root` — the consumer path, nothing prover-side trusted.
//!
//! The run has two phases. SETUP (once per config and leaf size) proves a
//! six-leaf reference tower to build every kind's circuit and fill plan.
//! The ONLINE prove then pays only per-statement work, and its time is the
//! throughput number.
//!
//! The prove skips the builders' self-checks (`SelfCheck::Skip`, the
//! production mode). With `checked`, every build re-checks its own output
//! (`SelfCheck::Full`); the root is the same either way.

use std::{env, time::Instant};

use flock_prover::tower::{SelfCheck, TowerConfig, TowerProver, TowerVk, verify_root_bytes};

fn main() {
    let mut args = env::args().skip(1);
    let n_leaves: usize = args
        .next()
        .map_or(8, |v| v.parse().expect("n_leaves: an even integer >= 4"));
    let blocks_per_leaf: usize = args.next().map_or(256, |v| {
        v.parse().expect("blocks_per_leaf: a positive integer")
    });
    let cfg = match args.next().as_deref() {
        None | Some("chain128") => TowerConfig::Chain128,
        Some(other) => panic!("unknown tower config {other:?}: chain128"),
    };
    let (mut do_verify, mut checks) = (false, SelfCheck::Skip);
    for flag in args {
        match flag.as_str() {
            "verify" => do_verify = true,
            "checked" => checks = SelfCheck::Full,
            other => panic!("unknown flag {other:?}: `verify` or `checked`"),
        }
    }

    // The demo chain starts at the zero state; a deployment starts at its
    // application's own h_start.
    let h_start = [0u32; 16];

    let t_setup = Instant::now();
    let prover = TowerProver::setup(cfg, blocks_per_leaf);
    let setup_s = t_setup.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let tower = prover.prove(checks, h_start, n_leaves);
    let prove_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    tower
        .discharge_root()
        .expect("the root-side residue discharges");
    let discharge_s = t1.elapsed().as_secs_f64();

    let s = tower.statement();
    println!(
        "\nTOWER {:?} ({checks:?} self-checks): {n_leaves} leaves x {blocks_per_leaf} compressions = {} total\n  \
         h_end = {:08x}{:08x}.. (== H^{}(h_start))\n  \
         ONE root proof: {:.1} KiB | setup {setup_s:.1}s (once) | ONLINE prove {prove_s:.2}s \
         = {:.0}k compressions/s | root discharge {discharge_s:.3}s",
        cfg,
        s.n_blocks,
        s.h_end[0],
        s.h_end[1],
        s.n_blocks,
        tower.root_proof_kib(),
        s.n_blocks as f64 / prove_s / 1e3,
    );

    if do_verify {
        let t2 = Instant::now();
        let vk = TowerVk::generate(cfg, blocks_per_leaf);
        let gen_s = t2.elapsed().as_secs_f64();
        let bytes = tower.root_bundle_bytes();
        let t3 = Instant::now();
        let stmt = verify_root_bytes(&vk, &bytes).expect("the wire bundle verifies standalone");
        let verify_s = t3.elapsed().as_secs_f64();
        assert_eq!(&stmt, s, "the statement rode the wire");
        println!(
            "  VERIFIED STANDALONE (consumer path, over the wire): bundle {:.1} KiB \
             | vk generate {gen_s:.1}s (one-time) | verify {verify_s:.3}s | statement certified",
            bytes.len() as f64 / 1024.0,
        );
    }
}
