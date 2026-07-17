//! Profiler-friendly bench: amortizes setup (R1CS matrix build, witness
//! generation) across many `prove_fast` calls so a sampling profiler captures
//! the prover code path instead of setup-time BLAKE3 hashing.
//!
//! Usage:
//!   RUSTFLAGS="-C debuginfo=line-tables-only" cargo bench --bench profile_prover --no-run
//!   RAYON_NUM_THREADS=1 samply record -- ./target/release/deps/profile_prover-<hash> <N_RUNS>
//!
//! `N_RUNS` defaults to 10. At m=29 single-thread each prove_fast is ~440 ms,
//! so 10 runs = 4.4 sec, dominating the ~0.5 sec setup.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression, K_LOG, min_n_blocks_log};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
}

fn random_compression(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let n_runs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let n_blocks: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768);

    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let threads = rayon::current_num_threads();

    println!(
        "Profile-prover bench: m={m}, n_blocks={n_blocks}, n_runs={n_runs}, threads={threads}"
    );

    // ---- Setup (NOT profiled; counts the BLAKE3 matrix build) ----
    let t_setup = Instant::now();
    let setup = Blake3Setup::new(n_blocks);
    println!("Setup ms: {:.2}", t_setup.elapsed().as_secs_f64() * 1e3);

    // ---- Witness generation: precompute different block sets so each run
    // ---- hits a unique Fiat-Shamir transcript (avoids dispatch caches making
    // ---- post-warmup runs unrealistically fast).
    let t_wit = Instant::now();
    let block_sets: Vec<Vec<Compression>> = (0..=n_runs)
        .map(|run| {
            let mut rng = Rng::new(0xC0FFEE_BEEF ^ (n_blocks as u64) ^ (run as u64));
            (0..n_blocks)
                .map(|_| random_compression(&mut rng))
                .collect()
        })
        .collect();
    println!(
        "Witness blocks ms: {:.2}",
        t_wit.elapsed().as_secs_f64() * 1e3
    );

    // ---- Warm-up (1 run) — primes any lazy-init data (e.g. NTT tables) ----
    {
        let mut ch = FsChallenger::new(b"flock-profile-v0");
        let (p, _, _) = setup.prove_fast(&block_sets[0], &mut ch);
        black_box(&p);
    }

    // ---- Profiled section: N_RUNS prove_fast calls in tight loop ----
    let t_loop = Instant::now();
    let mut times = Vec::with_capacity(n_runs);
    for run in 0..n_runs {
        let blocks = &block_sets[run + 1];
        let mut ch = FsChallenger::new(b"flock-profile-v0");
        let t = Instant::now();
        let (p, _, _) = setup.prove_fast(blocks, &mut ch);
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        times.push(elapsed);
        black_box(&p);
    }
    let total = t_loop.elapsed().as_secs_f64() * 1e3;
    let avg = times.iter().sum::<f64>() / n_runs as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("\n{n_runs} prove_fast runs: total {total:.2} ms, avg {avg:.2} ms, min {min:.2} ms");
    println!(
        "Throughput (avg): {:.0} compressions/sec",
        n_blocks as f64 / (avg / 1000.0)
    );
}
