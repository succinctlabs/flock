//! Focused micro-benchmark for the **production** witness generator.
//!
//! This drives `generate_witness_batch_major_partial` — the BatchMajor
//! generator that `prove_fast` and `prove_fast_union_ag` actually call. The
//! bench used to call `generate_witness_with_ab_packed_and_lincheck`, which is
//! the legacy ROW-MAJOR generator reached only from the retired direct-AG
//! route (`generate_witness_ab`); it measured a path production never
//! executes. Same class of mistake the round-2 bench carried until 2026-09-01.
//!
//! Why this phase matters: witness generation runs BEFORE
//! `prove_fast_ligerito_union*`, so it is outside the `[prove_union]` trace
//! entirely — the trace's own `witgen` line reports 0.00 ms because the
//! single-slot prebuilt source is a passthrough. Measured on the real m=32
//! prove it is **~64 ms of ~850 ms (7.6%)**, the largest phase nobody had
//! attributed. Best-of-N to isolate it from the pipeline's thermal load.
//! Honors RAYON_NUM_THREADS.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::r1cs_hashes::blake3::{
    Blake3Setup, Compression, generate_witness_batch_major_partial, min_n_blocks_log,
};

use flock_core::test_rng::Rng;

fn random_compression(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    println!("(production BatchMajor generator; m=32 = 262144 blocks is the shipped size)");
    for &n_blocks in &[65536usize, 262144] {
        let n_log = min_n_blocks_log(n_blocks);
        let _setup = Blake3Setup::new(n_blocks);
        let mut rng = Rng::new(0xC0FFEE ^ n_blocks as u64);
        let blocks: Vec<Compression> = (0..n_blocks)
            .map(|_| random_compression(&mut rng))
            .collect();

        // Warm up — also primes the scratch pool, which the real prove has
        // warm by the time witgen runs.
        for _ in 0..2 {
            let r = generate_witness_batch_major_partial(&blocks, n_log);
            black_box(&r);
        }

        let n_runs = if n_blocks >= 262144 { 5 } else { 8 };
        let mut best = f64::INFINITY;
        let mut sum = 0.0;
        let mut cs = 0u64;
        for _ in 0..n_runs {
            let t = Instant::now();
            let r = generate_witness_batch_major_partial(&blocks, n_log);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            best = best.min(ms);
            sum += ms;
            cs ^= r.0[0].lo ^ r.1[0].lo ^ r.2[0].lo ^ (r.3[0] as u64);
            black_box(&r);
        }
        println!(
            "n={n_blocks:>6} (m={})  gen_witness: best {:7.2} ms   avg {:7.2} ms   checksum {cs:016x}",
            n_log + 14,
            best,
            sum / n_runs as f64
        );
    }
}
