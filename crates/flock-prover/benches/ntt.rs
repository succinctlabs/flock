//! Additive NTT benchmark — single-NTT throughput.
//!
//! Measures `AdditiveNttF128::forward_transform` on a single NTT of size 2^k.
//! This is what the PCS `commit` does: zero-pad the packed witness (length
//! 2^log_msg_len) into the codeword buffer (length 2^k_code) and forward-NTT.
//!
//! For PCS at m=29 with rate 1/2: log_msg_len = 22, k_code = 23. So the
//! interesting size is k=23 (~128 MB single buffer).
//!
//! Note: this is a SINGLE NTT (no batching across instances). The
//! `ntt_experiments::ParallelNttF128` numbers measure 32 batched NTTs and
//! aren't directly comparable.
//!
//! Run: `cargo bench --bench ntt`

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
}

fn fill_f128(n: usize, seed: u64) -> Vec<F128> {
    let mut rng = Rng::new(seed);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(F128 {
            lo: rng.next_u64(),
            hi: rng.next_u64(),
        });
    }
    v
}

/// Sparse checksum so the optimizer can't elide the work.
fn checksum(data: &[F128]) -> u64 {
    let stride = (data.len() / 64).max(1);
    let mut cs: u64 = 0;
    for x in data.iter().step_by(stride).take(64) {
        cs ^= x.lo ^ x.hi;
    }
    cs
}

fn report(label: &str, secs: f64, bytes: u64, k: usize, cs: u64) {
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let gbps = gb / secs;
    // Forward additive NTT does ~k·2^k butterflies; each butterfly = 1 F128 mul + a few XORs.
    let total_muls: u64 = (1u64 << k) * (k as u64) / 2;
    let ns_per_mul = if total_muls == 0 {
        0.0
    } else {
        secs * 1e9 / total_muls as f64
    };
    println!(
        "  {:<46}  {:>7.4} s  {:>6.2} GB/s  {:>6.2} ns/mul  [cs={:016x}]",
        label, secs, gbps, ns_per_mul, cs
    );
}

fn bench_forward(k: usize) {
    let n = 1usize << k;
    let bytes = (n as u64) * 16;
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("  build: k={k} (size 2^{k} = {n}), buffer = {:.2} GB", gb);

    let ntt = AdditiveNttF128::standard(k);
    let original = fill_f128(n, 0xFACEFEED ^ (k as u64));

    // Scalar.
    let mut data = original.clone();
    let t0 = Instant::now();
    ntt.forward_transform_scalar(&mut data);
    let secs_scalar = t0.elapsed().as_secs_f64();
    let cs_scalar = checksum(&data);
    report("forward_transform_scalar", secs_scalar, bytes, k, cs_scalar);

    // NEON single-thread.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        let mut data = original.clone();
        let t0 = Instant::now();
        ntt.forward_transform_neon(&mut data);
        let secs_neon = t0.elapsed().as_secs_f64();
        let cs_neon = checksum(&data);
        report("forward_transform_neon", secs_neon, bytes, k, cs_neon);
        assert_eq!(cs_scalar, cs_neon, "NEON differs from scalar at k={k}");

        // Parallel (NEON + rayon, per-layer).
        let mut data = original.clone();
        let t0 = Instant::now();
        ntt.forward_transform_parallel(&mut data);
        let secs_par = t0.elapsed().as_secs_f64();
        let cs_par = checksum(&data);
        report("forward_transform_parallel", secs_par, bytes, k, cs_par);
        assert_eq!(cs_scalar, cs_par, "parallel differs from scalar at k={k}");

        // Batched (cache-blocked deep layers).
        let mut data = original.clone();
        let t0 = Instant::now();
        ntt.forward_transform_batched(&mut data);
        let secs_bat = t0.elapsed().as_secs_f64();
        let cs_bat = checksum(&data);
        report("forward_transform_batched", secs_bat, bytes, k, cs_bat);
        assert_eq!(cs_scalar, cs_bat, "batched differs from scalar at k={k}");

        if secs_par > 0.0 {
            println!(
                "    speedup vs scalar: NEON {:.2}×, parallel {:.2}×, batched {:.2}×",
                secs_scalar / secs_neon,
                secs_scalar / secs_par,
                secs_scalar / secs_bat,
            );
        }
    }
}

/// Bench the PCS commit's NTT for a Boolean witness at the given m. With rate
/// 1/2: k_code = (m - 7) + 1 = m - 6. The NTT is on a 2^k_code buffer.
fn bench_pcs_commit(m: usize) {
    let log_msg_len = m - 7;
    let k_code = log_msg_len + 1;
    let n_code = 1usize << k_code;
    let n_msg = 1usize << log_msg_len;
    let bytes = (n_code as u64) * 16;
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "  PCS commit @ m={m}: log_msg_len={log_msg_len}, k_code={k_code}, buffer = {:.2} GB",
        gb
    );

    let ntt = AdditiveNttF128::standard(k_code);

    // Fill the codeword buffer with random "message" + zero-pad.
    let mut data = fill_f128(n_msg, 0xC0FFEE ^ (m as u64));
    data.resize(n_code, F128::ZERO);

    // Warm-up.
    let mut warmup = data.clone();
    ntt.forward_transform(&mut warmup);
    let _ = checksum(&warmup);

    let t0 = Instant::now();
    ntt.forward_transform(&mut data);
    let secs = t0.elapsed().as_secs_f64();
    let cs = checksum(&data);
    report(
        "commit-NTT forward (single-thread)",
        secs,
        bytes,
        k_code,
        cs,
    );
}

fn header(name: &str) {
    println!("\n===== {name} =====");
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — PMULL path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: software fallback path — NEON/PMULL disabled)");

    // Raw forward NTT benchmarks across a range of sizes.
    header("AdditiveNttF128::forward (single NTT)");
    for &k in &[16usize, 18, 20, 22, 23] {
        bench_forward(k);
    }

    // PCS commit's NTT at typical R1CS witness sizes.
    header("PCS commit NTT at increasing m");
    for &m in &[13usize, 15, 20, 24, 29] {
        bench_pcs_commit(m);
    }
}
