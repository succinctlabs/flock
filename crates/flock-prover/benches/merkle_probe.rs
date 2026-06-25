//! Merkle build probe: tight loop on `merkle::merkle_tree` over a codeword-sized
//! byte buffer. Captures the SHA-256-with-ARM-crypto-extension hash chain that
//! pcs::commit pays for the row-batch tree.
//!
//! Usage:
//!   cargo bench --bench merkle_probe --no-run
//!   RAYON_NUM_THREADS=1 samply record -- ./target/release/deps/merkle_probe-<hash> [n_runs] [m]
//!
//! Default: 50 runs at m=29 (~3 sec ST).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::merkle;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let n_runs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let m: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(29);
    // Same shape as pcs::commit at this m: 128 MB codeword bytes, 262144 leaves of 1 KB each.
    let log_msg_len = m - 7;
    let log_batch_size = 6usize;
    let log_inv_rate = 1usize;
    let k_code = log_msg_len - log_batch_size + log_inv_rate;
    let num_ntts = 1usize << log_batch_size;
    let codeword_f128 = (1usize << k_code) * num_ntts;
    let codeword_bytes = codeword_f128 * 16;
    let n_leaves = 1usize << k_code;

    println!(
        "Merkle probe: m={m}, codeword {} MB ({} F128), leaves {}",
        codeword_bytes / (1024 * 1024),
        codeword_f128,
        n_leaves
    );
    println!("n_runs={n_runs}, threads={}", rayon::current_num_threads());

    let mut rng = Rng::new(0xC0DE_FACE);
    let mut data = vec![0u8; codeword_bytes];
    for byte in data.iter_mut() {
        *byte = rng.next_u64() as u8;
    }

    // Warm-up.
    let _ = merkle::merkle_tree(&data, n_leaves);

    let t0 = Instant::now();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t = Instant::now();
        let tree = merkle::merkle_tree(&data, n_leaves);
        times.push(t.elapsed().as_secs_f64() * 1e3);
        black_box(&tree);
    }
    let total = t0.elapsed().as_secs_f64() * 1e3;
    let avg = times.iter().sum::<f64>() / n_runs as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "{n_runs} merkle_tree calls: total {total:.2} ms, avg {avg:.2} ms/call, min {min:.2} ms/call"
    );
}
