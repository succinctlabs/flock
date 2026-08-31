//! Reproducible headline proving-throughput matrix for the README.
//!
//! Measures SHA-256 compressions and BLAKE3 compressions. Thread count is controlled through
//! `RAYON_NUM_THREADS`; `benchmarks/bench_hash_throughput.sh` runs the complete
//! single- and multi-threaded matrix and renders it as Markdown.

use std::hint::black_box;
use std::time::{Duration, Instant};

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};
use flock_prover::r1cs_hashes::sha2::Sha256HybridSetup;

use flock_core::test_rng::Rng;

fn random_sha2_input(rng: &mut Rng) -> ([u32; 8], [u32; 16]) {
    (
        std::array::from_fn(|_| rng.next_u32()),
        std::array::from_fn(|_| rng.next_u32()),
    )
}

fn random_blake3_input(rng: &mut Rng) -> Compression {
    (
        std::array::from_fn(|_| rng.next_u32()),
        std::array::from_fn(|_| rng.next_u32()),
        rng.next_u64(),
        64,
        11,
    )
}

fn best_of<T, F, O>(inputs: &[T], runs: usize, mut prove: F) -> Duration
where
    F: FnMut(&T) -> O,
{
    let mut best = Duration::MAX;
    for (run, input) in inputs[1..=runs].iter().enumerate() {
        let start = Instant::now();
        let output = prove(input);
        let elapsed = start.elapsed();
        best = best.min(elapsed);
        black_box(output);
        eprintln!(
            "    run {}/{}: {:.3} s",
            run + 1,
            runs,
            elapsed.as_secs_f64()
        );
    }
    best
}

fn report(hash: &str, batch: usize, best: Duration) {
    let seconds = best.as_secs_f64();
    let throughput = batch as f64 / seconds;
    println!(
        "RESULT\t{hash}\t{batch}\t{}\t{seconds:.6}\t{throughput:.2}",
        rayon::current_num_threads(),
    );
}

fn bench_sha2(batch: usize, runs: usize) {
    eprintln!("  SHA-256, batch {batch}");
    let setup = Sha256HybridSetup::new(batch);
    let input_sets: Vec<Vec<_>> = (0..=runs)
        .map(|run| {
            let mut rng = Rng::new(0x5A25_6000 ^ batch as u64 ^ run as u64);
            (0..batch).map(|_| random_sha2_input(&mut rng)).collect()
        })
        .collect();

    let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
    let (proof, commitment, _) = setup.prove_fast(&input_sets[0], &mut challenger);
    let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
    setup
        .verify(&commitment, &proof, &mut challenger)
        .expect("SHA-256 warm-up proof failed verification");
    black_box(proof);

    let best = best_of(&input_sets, runs, |inputs| {
        let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
        setup.prove_fast(inputs, &mut challenger)
    });
    report("sha2", batch, best);
}

fn bench_blake3(batch: usize, runs: usize) {
    eprintln!("  BLAKE3, batch {batch}");
    let setup = Blake3Setup::new(batch);
    let input_sets: Vec<Vec<_>> = (0..=runs)
        .map(|run| {
            let mut rng = Rng::new(0xB1A3_E000 ^ batch as u64 ^ run as u64);
            (0..batch).map(|_| random_blake3_input(&mut rng)).collect()
        })
        .collect();

    let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
    let (proof, commitment, _) = setup.prove_fast(&input_sets[0], &mut challenger);
    let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
    setup
        .verify(&commitment, &proof, &mut challenger)
        .expect("BLAKE3 warm-up proof failed verification");
    black_box(proof);

    let best = best_of(&input_sets, runs, |inputs| {
        let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
        setup.prove_fast(inputs, &mut challenger)
    });
    report("blake3", batch, best);
}

fn parse_log2_batches() -> Vec<u32> {
    let value = std::env::var("HASH_BENCH_LOG2S").unwrap_or_else(|_| "10 12 14 16 18".to_owned());
    let batches: Vec<u32> = value
        .split([',', ' '])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let log2 = part
                .parse::<u32>()
                .expect("HASH_BENCH_LOG2S must contain integer log2 batch sizes");
            assert!(
                log2 >= 8,
                "HASH_BENCH_LOG2S values must be at least 8 for every hash"
            );
            assert!(
                log2 < usize::BITS,
                "HASH_BENCH_LOG2S contains a batch size too large for this target"
            );
            log2
        })
        .collect();
    assert!(!batches.is_empty(), "HASH_BENCH_LOG2S must not be empty");
    batches
}

fn parse_runs() -> usize {
    let runs = std::env::var("HASH_BENCH_RUNS")
        .unwrap_or_else(|_| "3".to_owned())
        .parse::<usize>()
        .expect("HASH_BENCH_RUNS must be a positive integer");
    assert!(runs > 0, "HASH_BENCH_RUNS must be greater than zero");
    runs
}

fn enabled_x86_features() -> &'static str {
    if cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )) {
        "AVX-512 + VPCLMULQDQ"
    } else if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        "AVX2"
    } else if cfg!(all(target_arch = "x86_64", target_feature = "avx")) {
        "AVX"
    } else {
        "portable"
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let batches = parse_log2_batches();
    let runs = parse_runs();
    eprintln!(
        "Flock hash proving throughput: {} thread(s), {}, best of {runs} after one warm-up",
        rayon::current_num_threads(),
        enabled_x86_features(),
    );

    for &log2 in &batches {
        bench_sha2(1usize << log2, runs);
    }
    for &log2 in &batches {
        bench_blake3(1usize << log2, runs);
    }
}
