//! Native BLAKE3 hash-chain benchmark.
//!
//! Computes `H^T(x) = H(H(...H(x)...))` (T iterations) on a 32-byte input
//! using the `blake3` crate. Reports hashes/sec.
//!
//! Mirrors [`keccak_native_chain`] exactly. Two scenarios:
//!
//! 1. **Single chain (sequential)**: a hash chain has a strict data dependency
//!    (output of step i is input of step i+1), so it can't be parallelized.
//!    This measures the per-core BLAKE3 throughput — relevant for VDFs and
//!    any "BLAKE3-as-sequential-work" use case.
//!
//! 2. **Independent chains (parallel)**: many independent chains run in
//!    parallel across all cores. Reports aggregate machine throughput, which
//!    is the right comparison for batched proving (where you have many BLAKE3
//!    invocations to evaluate but no sequentiality requirement).
//!
//! Run: `cargo bench --bench blake3_native_chain`

use std::time::Instant;

use rayon::prelude::*;

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
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut i = 0;
        while i + 8 <= buf.len() {
            buf[i..i + 8].copy_from_slice(&self.next_u64().to_le_bytes());
            i += 8;
        }
        if i < buf.len() {
            let v = self.next_u64().to_le_bytes();
            let tail = buf.len() - i;
            buf[i..].copy_from_slice(&v[..tail]);
        }
    }
}

/// Compute H^T(x) where H = BLAKE3 (32-byte output) and `x` is a 32-byte input.
/// The chain has a strict data dependency — no parallelism inside.
#[inline(never)]
fn hash_chain(initial: [u8; 32], t: u64) -> [u8; 32] {
    let mut state = initial;
    for _ in 0..t {
        state = *blake3::hash(&state).as_bytes();
    }
    state
}

fn fmt_rate(hps: f64) -> String {
    if hps >= 1e9 {
        format!("{:>7.2} Ghash/s", hps / 1e9)
    } else if hps >= 1e6 {
        format!("{:>7.2} Mhash/s", hps / 1e6)
    } else if hps >= 1e3 {
        format!("{:>7.2} khash/s", hps / 1e3)
    } else {
        format!("{:>7.2}  hash/s", hps)
    }
}

fn fmt_secs(s: f64) -> String {
    if s < 1e-6 {
        format!("{:>7.1} ns", s * 1e9)
    } else if s < 1e-3 {
        format!("{:>7.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:>7.2} ms", s * 1e3)
    } else {
        format!("{:>7.2} s", s)
    }
}

fn bench_single_chain(t: u64, runs: usize) {
    let mut rng = Rng::new(0xC0FFEE_DEAD ^ t);
    let mut initial = [0u8; 32];
    rng.fill_bytes(&mut initial);

    // Warm up branch predictor + caches.
    let _ = std::hint::black_box(hash_chain(initial, 1024));

    let mut best = f64::INFINITY;
    let mut last_out = [0u8; 32];
    for run in 0..runs {
        let t0 = Instant::now();
        let out = hash_chain(std::hint::black_box(initial), t);
        let elapsed = t0.elapsed().as_secs_f64();
        let hps = t as f64 / elapsed;
        let ns_per_hash = elapsed * 1e9 / t as f64;
        last_out = out;
        best = best.min(elapsed);
        println!(
            "  run {}/{}: T={:>10}  total={}  rate={}  ({:>6.1} ns/hash)",
            run + 1,
            runs,
            t,
            fmt_secs(elapsed),
            fmt_rate(hps),
            ns_per_hash,
        );
    }
    let best_hps = t as f64 / best;
    let best_ns = best * 1e9 / t as f64;
    println!(
        "  BEST  : T={:>10}  total={}  rate={}  ({:>6.1} ns/hash)  out[0..4]={:02x}{:02x}{:02x}{:02x}",
        t,
        fmt_secs(best),
        fmt_rate(best_hps),
        best_ns,
        last_out[0],
        last_out[1],
        last_out[2],
        last_out[3],
    );
}

fn bench_parallel_chains(t_per_chain: u64, n_chains: usize, runs: usize) {
    let mut rng = Rng::new(0xBEEF_F00D ^ t_per_chain);
    let initials: Vec<[u8; 32]> = (0..n_chains)
        .map(|_| {
            let mut x = [0u8; 32];
            rng.fill_bytes(&mut x);
            x
        })
        .collect();

    // Warm up rayon thread pool.
    let _ = initials.par_iter().map(|&x| hash_chain(x, 256)).reduce(
        || [0u8; 32],
        |a, b| {
            let mut o = [0u8; 32];
            for i in 0..32 {
                o[i] = a[i] ^ b[i];
            }
            o
        },
    );

    let total_hashes = (n_chains as u64) * t_per_chain;
    let mut best = f64::INFINITY;
    let mut last_xor = [0u8; 32];
    for run in 0..runs {
        let t0 = Instant::now();
        let xor: [u8; 32] = initials
            .par_iter()
            .map(|&x| hash_chain(x, t_per_chain))
            .reduce(
                || [0u8; 32],
                |a, b| {
                    let mut o = [0u8; 32];
                    for i in 0..32 {
                        o[i] = a[i] ^ b[i];
                    }
                    o
                },
            );
        let elapsed = t0.elapsed().as_secs_f64();
        let hps = total_hashes as f64 / elapsed;
        last_xor = xor;
        best = best.min(elapsed);
        println!(
            "  run {}/{}: {} chains × T={}  total_hashes={}  wall={}  rate={}",
            run + 1,
            runs,
            n_chains,
            t_per_chain,
            total_hashes,
            fmt_secs(elapsed),
            fmt_rate(hps),
        );
    }
    let best_hps = total_hashes as f64 / best;
    println!(
        "  BEST  : {} chains × T={}  wall={}  rate={}  xor[0..4]={:02x}{:02x}{:02x}{:02x}",
        n_chains,
        t_per_chain,
        fmt_secs(best),
        fmt_rate(best_hps),
        last_xor[0],
        last_xor[1],
        last_xor[2],
        last_xor[3],
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    println!("(rayon threads available: {})", threads);
    #[cfg(target_arch = "aarch64")]
    println!("(target: aarch64 — using blake3 crate, NEON SIMD)");

    println!("\n========== Single-chain BLAKE3 (sequential) ==========");
    println!("(measures per-core throughput; hash chain cannot be parallelized)");
    for &t in &[100_000u64, 1_000_000, 10_000_000] {
        println!();
        bench_single_chain(t, 3);
    }

    println!("\n========== Independent parallel chains (aggregate) ==========");
    println!("(measures whole-machine throughput across all cores)");
    // Use one chain per available thread, with a moderate T to amortize startup.
    for &t in &[1_000_000u64, 10_000_000] {
        println!();
        bench_parallel_chains(t, threads, 3);
    }
}
