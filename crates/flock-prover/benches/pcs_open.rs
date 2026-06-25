//! PCS opening (eval) bench at the same m's as `zerocheck_phases`.
//!
//! The zerocheck protocol produces two opening points for the committed
//! witness `z` — one for the (a, b) claim and one for the (c) claim. The
//! prover then runs `pcs::open_batch` against the saved `commitment` /
//! `prover_data` to attest to those evals. This bench:
//!
//!  1. Generates random packed witnesses (no real R1CS — random F128 inputs
//!     are fine for cost, since `commit`/`open` are data-independent in time).
//!  2. Times `pcs::commit`.
//!  3. Times `pcs::open_batch` at two random points (mimicking the 2-point
//!     opening from the zerocheck claim).
//!  4. Times a single-point `pcs::open` for comparison.
//!
//! Setting `PCS_TRACE=1` will also print the internal sub-phase breakdown
//! (ring-switch / rs_eq_ind combine / AdditiveNttF128::standard / basefold).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::field::F128;
use flock_prover::pcs::{PcsParams, commit, open, open_batch, pack_witness};

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
    fn bits(&mut self, n: usize) -> Vec<bool> {
        let mut v = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mut w = self.next_u64();
            for _ in 0..64 {
                if i == n {
                    break;
                }
                v.push((w & 1) == 1);
                w >>= 1;
                i += 1;
            }
        }
        v
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
    fn f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.f128()).collect()
    }
}

fn fmt_ms(s: f64) -> String {
    format!("{:>8.2} ms", s * 1000.0)
}

fn bench_one(m: usize) {
    let params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    let n_bits = 1usize << m;
    println!(
        "\n=== m = {m} ({} bits, {} MB packed, rate-1/{}) ===",
        n_bits,
        n_bits / 8 / (1 << 20),
        1usize << params.log_inv_rate
    );

    let mut rng = Rng::new(0xC0FFEE ^ (m as u64));
    let z = rng.bits(n_bits);
    let z_packed = pack_witness(&z, m);

    // x_outer length matches what `pcs::open` expects: m - 6.
    // (Per pcs.rs docstring: "the multilinear portion of the QuirkyPoint
    //  with length m − 6".)
    let x_len = m - 6;
    let x_ab: Vec<F128> = rng.f128_vec(x_len);
    let x_c: Vec<F128> = rng.f128_vec(x_len);

    // ---- Warm-up (prime any OnceLock caches / page allocator). ----
    {
        let (cmt, pd) = commit(&z_packed, &params);
        let mut ch = FsChallenger::new(b"flock-bench-warmup");
        let _ = open_batch(&z_packed, &pd, &cmt, &[&x_ab, &x_c], &mut ch);
        black_box((cmt, pd));
    }

    let n_runs = if m >= 28 { 3 } else { 1 };

    // ---- Commit (timed N times so warm-up effects wash out) ----
    let mut best_commit = f64::INFINITY;
    let (commitment, prover_data) = {
        let mut last = None;
        for run in 0..n_runs {
            let t0 = Instant::now();
            let (cmt, pd) = commit(black_box(&z_packed), &params);
            let s = t0.elapsed().as_secs_f64();
            if n_runs > 1 {
                println!("  pcs::commit (run {})           {}", run + 1, fmt_ms(s));
            } else {
                println!("  pcs::commit                    {}", fmt_ms(s));
            }
            best_commit = best_commit.min(s);
            last = Some((cmt, pd));
        }
        last.unwrap()
    };
    if n_runs > 1 {
        println!("  pcs::commit (best)             {}", fmt_ms(best_commit));
    }

    // ---- Open: single-point (one ring_switch + one basefold) ----
    let mut best_open1 = f64::INFINITY;
    for run in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let _proof = open(
            black_box(&z_packed),
            &prover_data,
            &commitment,
            &x_ab,
            &mut ch,
        );
        let s = t0.elapsed().as_secs_f64();
        if n_runs > 1 {
            println!("  pcs::open (1 pt, run {})        {}", run + 1, fmt_ms(s));
        } else {
            println!("  pcs::open (1 pt)               {}", fmt_ms(s));
        }
        best_open1 = best_open1.min(s);
        black_box(_proof);
    }
    if n_runs > 1 {
        println!("  pcs::open (1 pt, best)         {}", fmt_ms(best_open1));
    }

    // ---- Open: batched 2-point (mirrors zerocheck → R1CS proof pattern) ----
    let mut best_open2 = f64::INFINITY;
    for run in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let _proof = open_batch(
            black_box(&z_packed),
            &prover_data,
            &commitment,
            &[&x_ab, &x_c],
            &mut ch,
        );
        let s = t0.elapsed().as_secs_f64();
        if n_runs > 1 {
            println!("  pcs::open_batch (2 pt, run {})  {}", run + 1, fmt_ms(s));
        } else {
            println!("  pcs::open_batch (2 pt)         {}", fmt_ms(s));
        }
        best_open2 = best_open2.min(s);
        black_box(_proof);
    }
    if n_runs > 1 {
        println!("  pcs::open_batch (2 pt, best)   {}", fmt_ms(best_open2));
    }

    println!(
        "  ratio open_batch(2) / open(1) : {:.2}×   ({}% over a 2× upper bound)",
        best_open2 / best_open1,
        ((best_open2 / best_open1 - 1.0) * 100.0) as i32
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");
    println!("(set PCS_TRACE=1 for inner sub-phase breakdown of open_batch)");

    let ms: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let ms: &[usize] = if ms.is_empty() {
        &[24, 26, 28, 29]
    } else {
        &ms[..]
    };
    for &m in ms {
        bench_one(m);
    }
}
