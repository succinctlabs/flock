//! Quick timing for the hash-chain shift sumcheck (`flock_prover::chain`).
//!
//! Run: `cargo run --release --example chain_bench --features unsound-challenger`
//! (the feature is required because this bench uses the insecure
//! `RandomChallenger` for isolation — see `src/challenger.rs`).
//!
//! Operates on pre-folded scalar In/Out vectors of length `2^n` (one F128 per
//! instance), i.e. the cost of the chain glue *given* the region fold — not the
//! PCS opening. `n = m − k_log`; for Keccak at m=29, k_log=17 → n=12 (4096
//! instances).

use std::time::Instant;

use flock_prover::chain::{prove_chain_shift, verify_chain_shift};
use flock_prover::challenger::RandomChallenger;
use flock_prover::field::F128;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn bench_n(n: usize, iters: usize) {
    let n_total = 1usize << n;
    let mut rng = Rng(0xDEAD_BEEF ^ n as u64);
    let chain: Vec<F128> = (0..=n_total).map(|_| rng.f128()).collect();
    let in_vals: Vec<F128> = chain[..n_total].to_vec();
    let out_vals: Vec<F128> = chain[1..].to_vec();
    let x0_r = chain[0];
    let xlast_r = chain[n_total];

    // Warm up + correctness sanity.
    {
        let mut chp = RandomChallenger::new(1);
        let (proof, _) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        let mut chv = RandomChallenger::new(1);
        verify_chain_shift(&proof, x0_r, xlast_r, n, &mut chv).expect("verify");
    }

    let t0 = Instant::now();
    let mut sink = F128::ZERO;
    for _ in 0..iters {
        let mut chp = RandomChallenger::new(1);
        let (proof, _) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        sink = sink + proof.g_at_point;
    }
    let prove_ns = t0.elapsed().as_nanos() / iters as u128;

    let mut chp = RandomChallenger::new(1);
    let (proof, _) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
    let t1 = Instant::now();
    for _ in 0..iters {
        let mut chv = RandomChallenger::new(1);
        let c = verify_chain_shift(&proof, x0_r, xlast_r, n, &mut chv).unwrap();
        sink = sink + c.value;
    }
    let verify_ns = t1.elapsed().as_nanos() / iters as u128;

    println!(
        "n={n:2}  N=2^{n}={n_total:>7}  rounds={:>2}  prove={:>9.3} us  verify={:>9.3} us  proof={} F128  (sink={:?})",
        n + 1,
        prove_ns as f64 / 1000.0,
        verify_ns as f64 / 1000.0,
        proof.rounds.len() * 2 + 1,
        sink,
    );
}

fn main() {
    println!("hash-chain shift sumcheck — prove/verify on pre-folded In/Out (2^n instances)");
    for &(n, iters) in &[
        (8, 2000),
        (10, 1000),
        (12, 500),
        (14, 200),
        (16, 50),
        (18, 20),
    ] {
        bench_n(n, iters);
    }
}
