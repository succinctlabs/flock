//! Multilinear rounds 3..(m−k_skip+1) bench — drives the full
//! `fold_and_compute_round_pair` chain (parallel) followed by the tail
//! `fold_in_place_pair + round_pair_naive` for small sizes.
//!
//! Inputs are the post-round-2 a_mlv, b_mlv (length `2^(m−k_skip−1)` each —
//! already folded once after round-2's binding). The C++ structure:
//!
//!   for i in 0..(n_mlv_rounds − 1):
//!     if remaining_challenges >= 9: fold_and_compute (fused)
//!     else: fold_in_place + sumcheck_round_pair (unfused)
//!   fold_in_place(last challenge)
//!
//! Target from PROTOCOL_REFERENCE.md: ~49 ms single-thread for "Multilinear
//! rounds 3..23" at m=29.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::zerocheck::multilinear::{
    fold_and_compute_round_pair_optimized, fold_in_place_pair, round_pair_naive,
};

const K_SKIP: usize = 6;

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
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
    fn f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.f128()).collect()
    }
}

/// Run the full multilinear chain on (a, b). Mirrors the C++ inner loop
/// structure exactly:
///   - while remaining ≥ 9 vars, use fused fold_and_compute
///   - else, unfused fold_in_place + round_pair_naive
///   - final fold at the last challenge
fn run_chain(
    mut a: Vec<F128>,
    mut b: Vec<F128>,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>) {
    let n_rounds = mlv_challenges.len();
    // Each iteration consumes one challenge and produces a message + folded state.
    for i in 0..(n_rounds.saturating_sub(1)) {
        let r_remaining = &mlv_challenges[i + 1..];
        if r_remaining.len() >= 9 {
            let r_fold = mlv_challenges[i];
            let (a_new, b_new, _m1, _minf) =
                fold_and_compute_round_pair_optimized(&a, &b, r_fold, r_remaining);
            a = a_new;
            b = b_new;
        } else {
            fold_in_place_pair(&mut a, &mut b, mlv_challenges[i]);
            let (_m1, _minf) = round_pair_naive(&a, &b, r_remaining);
        }
    }
    if n_rounds > 0 {
        fold_in_place_pair(&mut a, &mut b, mlv_challenges[n_rounds - 1]);
    }
    (a, b)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");

    // For each m, simulate the state entering round 3:
    //   a_mlv has length 2^(m − k_skip − 1) (already folded once)
    //   mlv_challenges from round 3 onward has length m − k_skip − 1
    for &m in &[16usize, 20, 24, 26, 28, 29] {
        let n_rest = m - K_SKIP;
        let n_after_round2 = 1usize << (n_rest - 1);
        let n_rounds_remaining = n_rest - 1;
        println!(
            "\n=== m = {m}: round-3 entry state {} F128 entries, {} rounds remaining ===",
            n_after_round2, n_rounds_remaining
        );

        let mut rng = Rng::new(0xD3CADE42 + m as u64);
        let a: Vec<F128> = (0..n_after_round2).map(|_| rng.f128()).collect();
        let b: Vec<F128> = (0..n_after_round2).map(|_| rng.f128()).collect();
        let mlv_challenges = rng.f128_vec(n_rounds_remaining);

        // Warm up.
        let _ = run_chain(a.clone(), b.clone(), &mlv_challenges);

        let n_runs = if m >= 24 { 3 } else { 1 };
        let mut best_ms = f64::INFINITY;
        let mut cs = 0u64;
        for run in 0..n_runs {
            let label = if n_runs == 1 {
                String::from("full chain rounds 3..end (parallel)")
            } else {
                format!("full chain rounds 3..end (parallel, run {})", run + 1)
            };
            let t0 = Instant::now();
            let (a_final, b_final) =
                run_chain(black_box(a.clone()), black_box(b.clone()), &mlv_challenges);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<48} {:>10.2} ms", label, elapsed);
            best_ms = best_ms.min(elapsed);
            cs ^= a_final[0].lo ^ b_final[0].lo;
        }
        if n_runs > 1 {
            println!("  {:<48} {:>10.2} ms", "  (best)", best_ms);
        }
        println!("  checksum: {cs:016x}");
    }
}
