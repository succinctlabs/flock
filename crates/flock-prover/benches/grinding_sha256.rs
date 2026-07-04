//! Grinding-cost benchmark for c = 5..=16 bits of leading-zero PoW using
//! hardware-accelerated SHA-256 (`sha2` crate with `asm` feature; uses
//! the ARMv8 sha2 instructions on aarch64).
//!
//! Measures both single-threaded and multi-threaded grind cost, reporting
//! mean / median time per grind plus effective hashes/sec.
//!
//! For each `c`, runs many independent grindings (each with a fresh 32-byte
//! prefix) so the geometric noise in number of tries averages out.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use sha2::{Digest, Sha256};

#[inline(always)]
fn has_leading_zero_bits(h: &[u8; 32], n: u32) -> bool {
    let full_bytes = (n / 8) as usize;
    let extra_bits = n % 8;
    for &b in h.iter().take(full_bytes) {
        if b != 0 {
            return false;
        }
    }
    if extra_bits > 0 && (h[full_bytes] >> (8 - extra_bits)) != 0 {
        return false;
    }
    true
}

#[inline(always)]
fn hash_with_nonce(prefix: &[u8; 32], nonce: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(nonce.to_le_bytes());
    let out = hasher.finalize();
    out.into()
}

fn grind_st(prefix: &[u8; 32], c: u32) -> (u64, u64) {
    let mut nonce: u64 = 0;
    loop {
        let h = hash_with_nonce(prefix, nonce);
        if has_leading_zero_bits(&h, c) {
            return (nonce, nonce + 1); // (winning nonce, num attempts)
        }
        nonce = nonce.wrapping_add(1);
    }
}

/// MT grind: P threads each iterate `nonce = tid, tid+P, tid+2P, ...`
/// First thread to find a hit sets the shared `found` flag.
fn grind_mt(prefix: &[u8; 32], c: u32, n_threads: usize) -> (u64, u64) {
    let found = AtomicU64::new(u64::MAX);
    let total_attempts = AtomicUsize::new(0);
    let stride = n_threads as u64;

    rayon::scope(|s| {
        for tid in 0..n_threads {
            let found_ref = &found;
            let attempts_ref = &total_attempts;
            let prefix = *prefix;
            s.spawn(move |_| {
                let mut nonce = tid as u64;
                let mut local_attempts: usize = 0;
                loop {
                    // Cheap periodic check on shared found flag (every 256 tries).
                    if local_attempts & 0xff == 0 && found_ref.load(Ordering::Relaxed) != u64::MAX {
                        attempts_ref.fetch_add(local_attempts, Ordering::Relaxed);
                        return;
                    }
                    let h = hash_with_nonce(&prefix, nonce);
                    local_attempts += 1;
                    if has_leading_zero_bits(&h, c) {
                        // Use compare-and-swap so only the smallest-nonce winner records.
                        let _ = found_ref.fetch_min(nonce, Ordering::SeqCst);
                        attempts_ref.fetch_add(local_attempts, Ordering::Relaxed);
                        return;
                    }
                    nonce = nonce.wrapping_add(stride);
                }
            });
        }
    });

    (
        found.load(Ordering::SeqCst),
        total_attempts.load(Ordering::Relaxed) as u64,
    )
}

fn fmt_time(s: f64) -> String {
    if s < 1e-6 {
        format!("{:>9.1} ns", s * 1e9)
    } else if s < 1e-3 {
        format!("{:>9.2} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:>9.2} ms", s * 1e3)
    } else {
        format!("{:>9.3} s ", s)
    }
}

fn fmt_hps(h_per_sec: f64) -> String {
    if h_per_sec > 1e9 {
        format!("{:>7.2} GH/s", h_per_sec / 1e9)
    } else if h_per_sec > 1e6 {
        format!("{:>7.2} MH/s", h_per_sec / 1e6)
    } else if h_per_sec > 1e3 {
        format!("{:>7.2} kH/s", h_per_sec / 1e3)
    } else {
        format!("{:>7.0} H/s", h_per_sec)
    }
}

fn run_st(c: u32, runs: usize) -> (f64, f64, f64) {
    // (mean time, median time, mean hashes/sec)
    let mut times: Vec<f64> = Vec::with_capacity(runs);
    let mut total_attempts: u64 = 0;
    for run in 0..runs {
        let mut prefix = [0u8; 32];
        for (i, b) in prefix.iter_mut().enumerate() {
            *b = ((run.wrapping_mul(0x9E3779B97F4A7C15) >> (i * 4)) & 0xFF) as u8;
        }
        let t = Instant::now();
        let (_winning, attempts) = grind_st(&prefix, c);
        let el = t.elapsed().as_secs_f64();
        times.push(el);
        total_attempts += attempts;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = times.iter().sum();
    let mean = total / runs as f64;
    let median = times[runs / 2];
    let hps = total_attempts as f64 / total;
    (mean, median, hps)
}

fn run_mt(c: u32, runs: usize, n_threads: usize) -> (f64, f64, f64) {
    let mut times: Vec<f64> = Vec::with_capacity(runs);
    let mut total_attempts: u64 = 0;
    for run in 0..runs {
        let mut prefix = [0u8; 32];
        for (i, b) in prefix.iter_mut().enumerate() {
            *b = ((run.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xCAFE) >> (i * 4)) & 0xFF)
                as u8;
        }
        let t = Instant::now();
        let (_winning, attempts) = grind_mt(&prefix, c, n_threads);
        let el = t.elapsed().as_secs_f64();
        times.push(el);
        total_attempts += attempts;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = times.iter().sum();
    let mean = total / runs as f64;
    let median = times[runs / 2];
    let hps = total_attempts as f64 / total;
    (mean, median, hps)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let n_threads = rayon::current_num_threads();

    // Pre-warm the SHA-256 unit (first call sometimes shows JIT-style startup).
    let warm = hash_with_nonce(&[0u8; 32], 0);
    std::hint::black_box(&warm);

    println!("SHA-256 grinding cost — hardware-accelerated (sha2 crate, asm feature)");
    #[cfg(target_arch = "aarch64")]
    println!("(target: aarch64 — hardware sha256 instructions)");
    println!("MT pool: {} threads\n", n_threads);

    // Run counts scale: fewer runs for higher c (each grind takes longer).
    let runs_for = |c: u32| -> usize {
        match c {
            5..=8 => 5000,
            9..=12 => 1000,
            13..=14 => 200,
            15 => 80,
            16 => 40,
            _ => 20,
        }
    };

    println!(
        "{:>3}  {:>5}  | {:>12} {:>12} {:>12}  | {:>12} {:>12} {:>12}  | {:>5}",
        "c", "exp.", "ST mean", "ST median", "ST H/s", "MT mean", "MT median", "MT H/s", "speedup"
    );
    println!("{}", "-".repeat(110));

    for c in 5..=16u32 {
        let runs = runs_for(c);
        let expected = 1u64 << c;
        let (st_mean, st_median, st_hps) = run_st(c, runs);
        let (mt_mean, mt_median, mt_hps) = run_mt(c, runs, n_threads);
        let speedup = st_mean / mt_mean;

        println!(
            "{:>3}  {:>5}  | {} {} {}  | {} {} {}  | {:>4.1}×",
            c,
            expected,
            fmt_time(st_mean),
            fmt_time(st_median),
            fmt_hps(st_hps),
            fmt_time(mt_mean),
            fmt_time(mt_median),
            fmt_hps(mt_hps),
            speedup,
        );
    }
}
