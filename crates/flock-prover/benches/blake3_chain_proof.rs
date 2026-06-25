//! BLAKE3 hash-chain proof generation benchmark.
//!
//! Run with:
//!   cargo bench --bench blake3_chain_proof
//!   (or: cargo run --release --bench blake3_chain_proof)
//!
//! Builds an honest BLAKE3 chain (blocks[i+1].cv == compress(blocks[i])[0..8]),
//! proves the whole chain with `Blake3Setup::prove_chain`, and reports timing.
//! Mirrors `benches/blake3_proof.rs` (sweep + warm-up + best-of-n_runs +
//! throughput) and the honest-chain construction used by the `blake3_chain`
//! tests. Times the PROVER path only (`prove_chain`).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
// The hash-chain prover lives entirely in the `blake3_chain` module, which is a
// fork of `blake3` with an I/O-aligned layout. `Blake3Setup`/`Compression`/
// `blake3_compress` here are that module's own types, distinct from `blake3`'s.
use flock_prover::r1cs_hashes::blake3::{
    Blake3Setup, Compression, K_LOG, blake3_compress, min_n_blocks_log,
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

/// Build an honest BLAKE3 chain of `n` compressions.
/// Replicates `blake3_chain::tests::honest_chain`: random initial cv + random
/// message per block, counter = 0, block_len = 64, flags = 0, and each block's
/// input cv equals the previous block's output chaining value (`compress(.)[0..8]`).
/// Returns `(blocks, cv_0, cv_last)`.
fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
    let mut rng = Rng::new(seed);
    let mut cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
    let cv0 = cv;
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let m: [u32; 16] = std::array::from_fn(|_| rng.nx() as u32);
        let counter = 0u64;
        let block_len = 64u32;
        let flags = 0u32;
        let block: Compression = (cv, m, counter, block_len, flags);
        blocks.push(block);
        let st = blake3_compress(&cv, &m, counter, block_len, flags);
        cv = st[0..8].try_into().unwrap();
    }
    let cv_last = cv;
    (blocks, cv0, cv_last)
}

fn bench_one(n_blocks: usize, n_runs: usize) {
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;

    println!("\n=== {n_blocks:>5} compressions  (m = {m}, slots = {n_slots}) ===");

    // Honest chain: blocks[i+1].cv == compress(blocks[i])[0..8].
    let (blocks, cv_0, cv_last) = honest_chain(n_blocks, 0xC0FFEE_BEEF ^ n_blocks as u64);
    let _ = (cv_0, cv_last); // captured per task spec; not needed for prove-only bench

    // Bench both rate profiles: fast (log_inv_rate=1, rate 1/2) and
    // slim (log_inv_rate=2, rate 1/4). Slim cuts proof size by ~half at
    // the cost of slightly slower prove (deeper Merkle paths + one extra
    // FRI fold round, partially offset by halved query count).
    for &(label, log_inv_rate) in &[("fast", 1usize), ("slim", 2usize)] {
        println!("  --- {label} (log_inv_rate={log_inv_rate}) ---");
        let setup = Blake3Setup::with_log_inv_rate(n_blocks, log_inv_rate);

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

        let mut best_base = f64::INFINITY;
        for _ in 0..n_runs {
            let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
            let t = Instant::now();
            let (p, _, _) = setup.prove_fast_basefold(&blocks, &mut ch);
            best_base = best_base.min(t.elapsed().as_secs_f64());
            black_box(&p);
        }

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
            "    prove_fast  (base):  {}  ({:.0} comp/sec)",
            fmt_ms(best_base),
            n_blocks as f64 / best_base
        );
        println!(
            "    prove_chain (full):  {}  ({:.0} comp/sec)",
            fmt_ms(best_chain),
            n_blocks as f64 / best_chain
        );
        println!(
            "    chain overhead:      {}  ({:+.1}% of base)",
            fmt_ms(overhead),
            100.0 * overhead / best_base
        );
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("BLAKE3 hash-chain proof generation benchmark (prove_chain).");
    println!("(honest chain, warm-up + best-of-n_runs timing)");

    // n_blocks → m: K_LOG=14, so m = 14 + ceil_log2(max(n_blocks, 8)).
    //   1     → m = 17    (lincheck floor)
    //   128   → m = 21
    //   8192  → m = 27
    //   32768 → m = 29    (headline; matches benches/blake3_proof.rs)
    // n_compressions must be a power of 2 ≥ 8 for chain protocol (the chain
    // shift sumcheck requires no padding slots; see audit fix `197b591`).
    for &(n, n_runs) in &[(8usize, 3), (128, 2), (8192, 2), (32768, 2), (65536, 2)] {
        bench_one(n, n_runs);
    }
}
