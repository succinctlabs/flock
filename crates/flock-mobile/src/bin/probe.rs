//! Runs the mobile demo entry points and prints the report.
//!
//! ```sh
//! cargo run --release -p flock-mobile --bin flock_mobile_probe -- tower [n_leaves] [blocks_per_leaf]
//! cargo run --release -p flock-mobile --bin flock_mobile_probe -- leaf [n_blocks]
//! ```
//!
//! Peak RSS is process-wide, so take each number in a fresh process. The
//! same binary built for `aarch64-apple-ios-sim` runs inside a booted
//! simulator with `xcrun simctl spawn booted <path> ...`.

use std::env;

use flock_mobile::{leaf_demo, tower_demo};

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "tower".to_string());
    match mode.as_str() {
        "tower" => {
            let n_leaves: usize = args
                .next()
                .map_or(4, |v| v.parse().expect("n_leaves: an even integer >= 4"));
            let blocks_per_leaf: usize = args.next().map_or(256, |v| {
                v.parse().expect("blocks_per_leaf: a positive integer")
            });
            let r = tower_demo(n_leaves, blocks_per_leaf);
            println!(
                "FLOCK MOBILE PROBE tower ({n_leaves} x {blocks_per_leaf} = {} compressions, {} threads)\n  \
                 prove {:.1} ms | discharge {:.1} ms | root proof {:.1} KiB\n  \
                 peak RSS {:.1} MiB | h_end[0] = {:08x}",
                r.n_blocks,
                r.threads,
                r.prove_ms,
                r.discharge_ms,
                r.root_proof_kib,
                r.peak_rss_bytes as f64 / (1024.0 * 1024.0),
                r.h_end_0,
            );
        }
        "leaf" => {
            let n_blocks: usize = args
                .next()
                .map_or(256, |v| v.parse().expect("n_blocks: a positive integer"));
            let r = leaf_demo(n_blocks);
            println!(
                "FLOCK MOBILE PROBE leaf ({} compressions, {} threads)\n  \
                 prove {:.1} ms | peak RSS {:.1} MiB",
                r.n_blocks,
                r.threads,
                r.prove_ms,
                r.peak_rss_bytes as f64 / (1024.0 * 1024.0),
            );
        }
        other => panic!("unknown mode {other:?}: tower or leaf"),
    }
}
