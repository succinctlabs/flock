//! Keccak-f1600 hash-chain proof generation benchmark.
//!
//! Run with:
//!   cargo bench --bench keccak_chain_proof
//!   (or: cargo run --release --bench keccak_chain_proof)
//!
//! Builds an honest Keccak-f1600 chain (x_{i+1} = keccak_f(x_i)), proves the
//! whole chain with `KeccakSetup::prove_chain`, and reports timing. Mirrors
//! `benches/blake3_chain_proof.rs`. Times the PROVER path only.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::keccak::{
    K_LOG, KeccakSetup, STATE_BITS, State, keccak_f, min_n_keccaks_log,
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

/// Build an honest Keccak chain of `n` permutations.
/// Random initial state; each subsequent input equals `keccak_f` of the prior.
/// Returns `(inputs, x_0, x_last)` where `x_last = keccak_f(inputs[n-1])`.
fn honest_chain(n: usize, seed: u64) -> (Vec<State>, State, State) {
    let mut rng = Rng::new(seed);
    let mut cur = [false; STATE_BITS];
    for b in cur.iter_mut() {
        *b = rng.nx() & 1 == 1;
    }
    let x0 = cur;
    let mut inputs = Vec::with_capacity(n);
    for _ in 0..n {
        inputs.push(cur);
        keccak_f(&mut cur);
    }
    let x_last = cur;
    (inputs, x0, x_last)
}

fn bench_one(n_keccaks: usize, n_runs: usize) {
    let n_log = min_n_keccaks_log(n_keccaks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;

    println!("\n=== {n_keccaks:>5} keccaks  (m = {m}, slots = {n_slots}) ===");

    let (inputs, _x0, _xlast) = honest_chain(n_keccaks, 0xC0FFEE_BEEF ^ n_keccaks as u64);
    let setup = KeccakSetup::new(n_keccaks);

    // Warm-up both paths.
    {
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let (p, _, _) = setup.prove_fast_basefold(&inputs, &mut ch);
        black_box(&p);
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let (proof, comm) = setup.prove_chain_basefold(&inputs, &mut ch);
        black_box(&proof);
        black_box(&comm);
    }

    // Best-of-n_runs prove_fast (base: no chain).
    let mut best_base = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let t = Instant::now();
        let (p, _, _) = setup.prove_fast_basefold(&inputs, &mut ch);
        best_base = best_base.min(t.elapsed().as_secs_f64());
        black_box(&p);
    }

    // Best-of-n_runs prove_chain (full).
    let mut best_chain = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-chain-bench-v0");
        let t = Instant::now();
        let (proof, comm) = setup.prove_chain_basefold(&inputs, &mut ch);
        best_chain = best_chain.min(t.elapsed().as_secs_f64());
        black_box(&proof);
        black_box(&comm);
    }

    let overhead = best_chain - best_base;
    println!(
        "  prove_fast  (base):  {}  ({:.0} keccak/sec)",
        fmt_ms(best_base),
        n_keccaks as f64 / best_base
    );
    println!(
        "  prove_chain (full):  {}  ({:.0} keccak/sec)",
        fmt_ms(best_chain),
        n_keccaks as f64 / best_chain
    );
    println!(
        "  chain overhead:      {}  ({:+.1}% of base)",
        fmt_ms(overhead),
        100.0 * overhead / best_base
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    println!("(rayon threads: {})", rayon::current_num_threads());
    println!("Keccak hash-chain proof generation benchmark (prove_chain).");
    println!("(honest chain, warm-up + best-of-n_runs timing)");

    // K_LOG=16 → m=28 means n_keccaks_log=12 → 4096 keccaks.
    //          → m=29 means n_keccaks_log=13 → 8192 keccaks.
    //          → m=30 means n_keccaks_log=14 → 16384 keccaks.
    for &(n, n_runs) in &[(4096usize, 2), (8192, 2), (16384, 2)] {
        bench_one(n, n_runs);
    }
}
