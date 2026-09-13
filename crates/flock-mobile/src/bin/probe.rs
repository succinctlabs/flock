//! Runs the mobile demo entry points and prints the report.
//!
//! ```sh
//! cargo run --release -p flock-mobile --bin flock_mobile_probe -- tower [n_leaves] [blocks_per_leaf] [global] [all_core]
//! cargo run --release -p flock-mobile --bin flock_mobile_probe -- leaf [n_blocks] [global] [all_core]
//! ```
//!
//! `global` / `all_core` size the two rayon pools (0 = the defaults: the
//! performance-core count, and every logical CPU). Peak RSS is process-wide,
//! so take each number in a fresh process. The same binary built for
//! `aarch64-apple-ios-sim` runs inside a booted simulator with
//! `xcrun simctl spawn booted <path> ...`.

use std::env;

use flock_mobile::{flock_mobile_init_threads, leaf_demo, tower_demo};

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "tower".to_string());
    let mut num = |what: &str, default: usize| -> usize {
        args.next().map_or(default, |v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{what}: a non-negative integer"))
        })
    };
    match mode.as_str() {
        "tower" => {
            let n_leaves = num("n_leaves", 4);
            let blocks_per_leaf = num("blocks_per_leaf", 256);
            let (global, all_core) = (num("global", 0), num("all_core", 0));
            let rc = flock_mobile_init_threads(global as u32, all_core as u32);
            let r = tower_demo(n_leaves, blocks_per_leaf);
            println!(
                "FLOCK MOBILE PROBE tower ({n_leaves} x {blocks_per_leaf} = {} compressions, \
                 global {} / all-core {} threads, init rc {rc})\n  \
                 prove {:.1} ms | discharge {:.1} ms | root proof {:.1} KiB\n  \
                 peak RSS {:.1} MiB | h_end[0] = {:08x}",
                r.n_blocks,
                r.threads,
                r.all_core_threads,
                r.prove_ms,
                r.discharge_ms,
                r.root_proof_kib,
                r.peak_rss_bytes as f64 / (1024.0 * 1024.0),
                r.h_end_0,
            );
        }
        "leaf" => {
            let n_blocks = num("n_blocks", 256);
            let (global, all_core) = (num("global", 0), num("all_core", 0));
            let rc = flock_mobile_init_threads(global as u32, all_core as u32);
            let r = leaf_demo(n_blocks);
            println!(
                "FLOCK MOBILE PROBE leaf ({} compressions, global {} / all-core {} threads, init rc {rc})\n  \
                 online leaf {:.1} ms = {:.0}k blake3/sec (walk {:.1} + witgen {:.1} + prove {:.1})\n  \
                 cold {:.1} ms | warm whole call {:.1} ms | peak RSS {:.1} MiB",
                r.n_blocks,
                r.threads,
                r.all_core_threads,
                r.online_wall_ms,
                r.n_blocks as f64 / (r.online_wall_ms / 1e3) / 1e3,
                r.online_walk_ms,
                r.online_witgen_ms,
                r.online_prove_ms,
                r.prove_ms,
                r.prove_warm_ms,
                r.peak_rss_bytes as f64 / (1024.0 * 1024.0),
            );
        }
        other => panic!("unknown mode {other:?}: tower or leaf"),
    }
}
