//! Reproducible headline proving-throughput matrix for the README.
//!
//! Measures SHA-256 compressions, BLAKE3 compressions, and Keccak-f[1600]
//! permutations with both witness layouts. Thread count is controlled through
//! `RAYON_NUM_THREADS`; `benchmarks/bench_hash_throughput.sh` runs the complete
//! single- and multi-threaded matrix and renders it as Markdown.

use std::hint::black_box;
use std::time::{Duration, Instant};

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};
use flock_prover::r1cs_hashes::keccak::{KeccakSetup, STATE_BITS, State};
use flock_prover::r1cs_hashes::sha2::Sha256HybridSetup;

const LAYOUTS: [BenchLayout; 2] = [BenchLayout::RowMajor, BenchLayout::BatchMajor];

#[derive(Clone, Copy)]
enum BenchLayout {
    RowMajor,
    BatchMajor,
}

impl BenchLayout {
    fn name(self) -> &'static str {
        match self {
            Self::RowMajor => "row-major",
            Self::BatchMajor => "batch-major",
        }
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

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

fn random_keccak_state(rng: &mut Rng) -> State {
    let mut state = [false; STATE_BITS];
    for chunk in state.chunks_mut(64) {
        let word = rng.next_u64();
        for (bit, value) in chunk.iter_mut().enumerate() {
            *value = (word >> bit) & 1 == 1;
        }
    }
    state
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

fn report(hash: &str, layout: BenchLayout, batch: usize, best: Duration) {
    let seconds = best.as_secs_f64();
    let throughput = batch as f64 / seconds;
    println!(
        "RESULT\t{hash}\t{}\t{batch}\t{}\t{seconds:.6}\t{throughput:.2}",
        layout.name(),
        rayon::current_num_threads(),
    );
}

fn bench_sha2(batch: usize, layout: BenchLayout, runs: usize) {
    eprintln!("  SHA-256, {}, batch {batch}", layout.name());
    let setup = match layout {
        BenchLayout::RowMajor => Sha256HybridSetup::new(batch),
        BenchLayout::BatchMajor => Sha256HybridSetup::new_batch_major(batch),
    };
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
    report("sha2", layout, batch, best);
}

fn bench_blake3(batch: usize, layout: BenchLayout, runs: usize) {
    eprintln!("  BLAKE3, {}, batch {batch}", layout.name());
    let setup = match layout {
        BenchLayout::RowMajor => Blake3Setup::new(batch),
        BenchLayout::BatchMajor => Blake3Setup::new_batch_major(batch),
    };
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
    report("blake3", layout, batch, best);
}

fn bench_keccak(batch: usize, layout: BenchLayout, runs: usize) {
    eprintln!("  Keccak-f[1600], {}, batch {batch}", layout.name());
    let setup = match layout {
        BenchLayout::RowMajor => KeccakSetup::new(batch),
        BenchLayout::BatchMajor => KeccakSetup::new_batch_major(batch),
    };
    let input_sets: Vec<Vec<_>> = (0..=runs)
        .map(|run| {
            let mut rng = Rng::new(0xAECC_A000 ^ batch as u64 ^ run as u64);
            (0..batch).map(|_| random_keccak_state(&mut rng)).collect()
        })
        .collect();

    let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
    let (proof, commitment, _) = setup.prove_fast(&input_sets[0], &mut challenger);
    let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
    setup
        .verify(&commitment, &proof, &mut challenger)
        .expect("Keccak warm-up proof failed verification");
    black_box(proof);

    let best = best_of(&input_sets, runs, |inputs| {
        let mut challenger = FsChallenger::new(b"flock-readme-bench-v0");
        setup.prove_fast(inputs, &mut challenger)
    });
    report("keccak", layout, batch, best);
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

    for layout in LAYOUTS {
        for &log2 in &batches {
            bench_sha2(1usize << log2, layout, runs);
        }
    }
    for layout in LAYOUTS {
        for &log2 in &batches {
            bench_blake3(1usize << log2, layout, runs);
        }
    }
    for layout in LAYOUTS {
        for &log2 in &batches {
            bench_keccak(1usize << log2, layout, runs);
        }
    }
}
