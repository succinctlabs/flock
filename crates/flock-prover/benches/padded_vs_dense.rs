//! Compares end-to-end `prove_fast` with and without the URM padding skip
//! for Keccak and SHA-2 at the same m. Toggles by overwriting
//! `r1cs.useful_bits` between `USEFUL_BITS` (padded path) and `1 << k_log`
//! (dense path — every bit is treated as useful). Both paths produce
//! byte-identical proofs on honest witnesses.
//!
//! Run single-threaded with `RAYON_NUM_THREADS=1 cargo bench --bench
//! padded_vs_dense`. Default sizes target m=30 (16384 Keccak permutations,
//! 32768 SHA-256 compressions).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::keccak::{
    K_LOG as KECCAK_K_LOG, KeccakSetup, STATE_BITS, State, USEFUL_BITS as KECCAK_USEFUL_BITS,
    min_n_keccaks_log,
};
use flock_prover::r1cs_hashes::sha2::{
    K_LOG as SHA2_K_LOG, Sha256HybridSetup, USEFUL_BITS as SHA2_USEFUL_BITS, min_n_blocks_log,
};

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
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

fn random_keccak_state(rng: &mut Rng) -> State {
    let mut s = [false; STATE_BITS];
    let mut i = 0;
    while i < STATE_BITS {
        let w = rng.next_u64();
        for b in 0..64 {
            if i + b < STATE_BITS {
                s[i + b] = (w >> b) & 1 == 1;
            }
        }
        i += 64;
    }
    s
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1.0 {
        format!("{:>9.2} µs", s * 1e6)
    } else if ms < 1000.0 {
        format!("{:>9.2} ms", ms)
    } else {
        format!("{:>9.3} s ", s)
    }
}

fn bench_keccak(n_keccaks: usize, n_runs: usize) {
    let n_log = min_n_keccaks_log(n_keccaks);
    let m = KECCAK_K_LOG + n_log;
    println!(
        "\n=== Keccak K = {n_keccaks} ({} permutations, m = {m}) ===",
        1usize << n_log
    );

    let mut setup = KeccakSetup::new(n_keccaks);
    let mut rng = Rng::new(0xABCD_1234_u64.wrapping_add(n_keccaks as u64));
    let states: Vec<State> = (0..n_keccaks)
        .map(|_| random_keccak_state(&mut rng))
        .collect();

    // Padded path (real USEFUL_BITS).
    setup.r1cs.useful_bits = KECCAK_USEFUL_BITS;
    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (_, _, _) = setup.prove_fast(&states, &mut ch);
    }
    let mut best_padded = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t = Instant::now();
        let p = setup.prove_fast(&states, &mut ch);
        let dt = t.elapsed().as_secs_f64();
        best_padded = best_padded.min(dt);
        black_box(p);
    }
    println!(
        "  padded (useful_bits={KECCAK_USEFUL_BITS}): {}",
        fmt_ms(best_padded)
    );

    // Dense path (no padding skip).
    setup.r1cs.useful_bits = 1 << KECCAK_K_LOG;
    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (_, _, _) = setup.prove_fast(&states, &mut ch);
    }
    let mut best_dense = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t = Instant::now();
        let p = setup.prove_fast(&states, &mut ch);
        let dt = t.elapsed().as_secs_f64();
        best_dense = best_dense.min(dt);
        black_box(p);
    }
    println!(
        "  dense  (useful_bits={}):  {}",
        1usize << KECCAK_K_LOG,
        fmt_ms(best_dense)
    );
    let saved = (best_dense - best_padded) / best_dense * 100.0;
    println!(
        "  ∆ end-to-end: {:+.1}%  ({})",
        -saved,
        fmt_ms(best_dense - best_padded)
    );
}

fn bench_sha2(n_compressions: usize, n_runs: usize) {
    let n_log = min_n_blocks_log(n_compressions);
    let m = SHA2_K_LOG + n_log;
    println!(
        "\n=== SHA-2 N = {n_compressions} ({} compressions, m = {m}) ===",
        1usize << n_log
    );

    let mut setup = Sha256HybridSetup::new(n_compressions);
    let mut rng = Rng::new(0xC0FFEE_5A55_u64.wrapping_add(n_compressions as u64));
    let inputs: Vec<([u32; 8], [u32; 16])> = (0..n_compressions)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.next_u32()),
                std::array::from_fn(|_| rng.next_u32()),
            )
        })
        .collect();

    // Padded path.
    setup.r1cs.useful_bits = SHA2_USEFUL_BITS;
    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (_, _, _) = setup.prove_fast(&inputs, &mut ch);
    }
    let mut best_padded = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t = Instant::now();
        let p = setup.prove_fast(&inputs, &mut ch);
        let dt = t.elapsed().as_secs_f64();
        best_padded = best_padded.min(dt);
        black_box(p);
    }
    println!(
        "  padded (useful_bits={SHA2_USEFUL_BITS}): {}",
        fmt_ms(best_padded)
    );

    // Dense path.
    setup.r1cs.useful_bits = 1 << SHA2_K_LOG;
    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (_, _, _) = setup.prove_fast(&inputs, &mut ch);
    }
    let mut best_dense = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t = Instant::now();
        let p = setup.prove_fast(&inputs, &mut ch);
        let dt = t.elapsed().as_secs_f64();
        best_dense = best_dense.min(dt);
        black_box(p);
    }
    println!(
        "  dense  (useful_bits={}):  {}",
        1usize << SHA2_K_LOG,
        fmt_ms(best_dense)
    );
    let saved = (best_dense - best_padded) / best_dense * 100.0;
    println!(
        "  ∆ end-to-end: {:+.1}%  ({})",
        -saved,
        fmt_ms(best_dense - best_padded)
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(default)".into());
    println!("URM padding-skip A/B for prove_fast (RAYON_NUM_THREADS={threads})");

    // m=30: 16384 Keccak perms, 32768 SHA-256 compressions.
    bench_keccak(16384, 4);
    bench_sha2(32768, 4);
}
