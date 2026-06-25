//! NTT butterfly probe: tight loop on `AdditiveNttF128::forward_transform_interleaved`
//! over an SoA buffer matching the PCS commit's shape. The 2-layer fused path is
//! exercised when log_d ≥ 7 (always at our production sizes).
//!
//! Usage:
//!   cargo bench --bench ntt_butterfly_probe --no-run
//!   RAYON_NUM_THREADS=1 samply record -- ./target/release/deps/ntt_butterfly_probe-<hash> [n_runs] [m]
//!
//! Default: 50 runs at m=29 (~3 sec ST). Matches BLAKE3 num_ntts=64 (K_LOG=14, log_batch_size=6).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::ntt::AdditiveNttF128;

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
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let n_runs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    // BLAKE3 m=29: log_msg_len = 22, log_batch_size = 6, k_code = 17, num_ntts = 64.
    let m: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(29);
    let log_msg_len = m - 7;
    let log_batch_size = 6usize;
    let log_inv_rate = 1usize;
    let k_code = log_msg_len - log_batch_size + log_inv_rate;
    let num_ntts = 1usize << log_batch_size;
    let buf_len = (1usize << k_code) * num_ntts;
    println!(
        "NTT butterfly probe: m={m}, k_code={k_code}, num_ntts={num_ntts}, buf_len={buf_len} F128 = {} MB",
        buf_len * 16 / (1024 * 1024)
    );
    println!("n_runs={n_runs}, threads={}", rayon::current_num_threads());

    let ntt = AdditiveNttF128::standard(k_code);

    // Fresh buffer per run — copying the seed buffer is part of the workload-shape
    // (matches pcs::commit which writes the witness into the buffer before the NTT).
    let mut rng = Rng::new(0xDEAD_BEEF);
    let seed: Vec<F128> = (0..buf_len).map(|_| rng.f128()).collect();
    let mut buffer: Vec<F128> = seed.clone();

    // Warm-up.
    ntt.forward_transform_interleaved(&mut buffer, num_ntts);
    buffer.copy_from_slice(&seed);

    let t0 = Instant::now();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t = Instant::now();
        ntt.forward_transform_interleaved(&mut buffer, num_ntts);
        times.push(t.elapsed().as_secs_f64() * 1e3);
        black_box(&buffer);
        // Reset for next run.
        buffer.copy_from_slice(&seed);
    }
    let total = t0.elapsed().as_secs_f64() * 1e3;
    let avg = times.iter().sum::<f64>() / n_runs as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("{n_runs} NTT calls: total {total:.2} ms, avg {avg:.2} ms/call, min {min:.2} ms/call");
}
