//! SHA-256 hash-chain proof generation benchmark.
//!
//! Run with:
//!   cargo bench --bench sha2_chain_proof
//!   (or: cargo run --release --bench sha2_chain_proof)
//!
//! Builds an honest SHA-256 chain (blocks[i+1].0 == sha256_compress(blocks[i])),
//! proves the whole chain with `Sha256HybridSetup::prove_chain`, and reports
//! both `prove_fast` (base) and `prove_chain` (full) timings + chain overhead.
//! Mirrors `benches/blake3_chain_proof.rs`. K_LOG=15 → m=29 at 16,384 blocks.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
// The hash-chain prover lives in the `sha2_chain` module, a fork of `sha2`
// with an I/O-aligned witness layout. `Sha256HybridSetup`/`Compression`/
// `sha256_compress` here are that module's own types, distinct from `sha2`'s.
use flock_prover::r1cs_hashes::sha2::{
    Compression, K_LOG, Sha256HybridSetup, min_n_blocks_log, sha256_compress,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1.0 {
        format!("{:>8.2} µs", s * 1e6)
    } else if ms < 1000.0 {
        format!("{:>8.2} ms", ms)
    } else {
        format!("{:>8.2} s ", s)
    }
}

/// Build an honest SHA-256 chain of `n` compressions.
/// Each block's H_in equals the previous block's H_out (= sha256_compress of
/// the previous). Returns `(blocks, cv_0, cv_last)`.
fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
    let mut rng = Rng::new(seed);
    let mut cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
    let cv0 = cv;
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let m: [u32; 16] = std::array::from_fn(|_| rng.nx() as u32);
        blocks.push((cv, m));
        cv = sha256_compress(&cv, &m);
    }
    (blocks, cv0, cv)
}

fn bench_one(n_blocks: usize, n_runs: usize) {
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;

    println!("\n=== {n_blocks:>5} compressions  (m = {m}, slots = {n_slots}) ===");

    let (blocks, cv_0, cv_last) = honest_chain(n_blocks, 0xC0FFEE_BEEF ^ n_blocks as u64);
    let _ = (cv_0, cv_last);

    let setup = Sha256HybridSetup::new(n_blocks);

    // Warm-up both paths.
    {
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let (p, _, _) = setup.prove_fast_basefold(&blocks, &mut ch);
        black_box(&p);
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let (proof, comm) = setup.prove_chain_basefold(&blocks, &mut ch);
        black_box(&proof);
        black_box(&comm);
    }

    // Best-of-n_runs prove_fast (base: no chain).
    let mut best_base = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let t = Instant::now();
        let (p, _, _) = setup.prove_fast_basefold(&blocks, &mut ch);
        best_base = best_base.min(t.elapsed().as_secs_f64());
        black_box(&p);
    }

    // Best-of-n_runs prove_chain (full).
    let mut best_chain = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let t = Instant::now();
        let (proof, comm) = setup.prove_chain_basefold(&blocks, &mut ch);
        best_chain = best_chain.min(t.elapsed().as_secs_f64());
        black_box(&proof);
        black_box(&comm);
    }

    let overhead = best_chain - best_base;
    println!(
        "  prove_fast  (base):  {}  ({:.0} comp/sec)",
        fmt_ms(best_base),
        n_blocks as f64 / best_base
    );
    println!(
        "  prove_chain (full):  {}  ({:.0} comp/sec)",
        fmt_ms(best_chain),
        n_blocks as f64 / best_chain
    );
    println!(
        "  chain overhead:      {}  ({:+.1}% of base)",
        fmt_ms(overhead),
        100.0 * overhead / best_base
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("SHA-256 hash-chain proof generation benchmark (prove_chain vs prove_fast).");
    println!("(honest chain, warm-up + best-of-n_runs timing)");

    // n_blocks → m: K_LOG=15, so m = 15 + ceil_log2(max(n_blocks, 8)).
    //   1     → m = 18    (lincheck floor)
    //   64    → m = 21
    //   4096  → m = 27
    //   16384 → m = 29    (headline; matches benches/sha2_proof.rs)
    //   32768 → m = 30
    // n_compressions must be a power of 2 ≥ 8 for chain protocol (the chain
    // shift sumcheck requires no padding slots; see audit fix `197b591`).
    for &(n, n_runs) in &[(8usize, 3), (64, 2), (4096, 2), (16384, 2), (32768, 2)] {
        bench_one(n, n_runs);
    }
}
