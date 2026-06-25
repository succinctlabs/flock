//! End-to-end HyperPlonk permutation-check benchmark (no criterion / harness).
//!
//! Measures `permutation::prove` and `permutation::verify` over the boolean
//! hypercube `B_μ` (`N = 2^μ`), including the real PCS commitment to the single
//! aux polynomial `v` (μ+1 vars) and its batched opening (5 points) at the
//! sumcheck reduction point. The witness `f, g` is NOT committed (the prover
//! only commits the PIOP's own oracle `v`). The opening backend is adaptive:
//! Ligerito at μ≥7 (`v` has log_n = μ+1 ≥ 8), BaseFold below.
//!
//! Witness generation is hoisted outside the timed section. A warm-up run
//! primes the OnceLock-cached NTT/convert tables.
//!
//! Run:   `cargo bench --bench permutation`
//! On M-series, requires `.cargo/config.toml` so the `aes` feature is on.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::{Challenger, FsChallenger};
use flock_prover::field::F128;
use flock_prover::permutation::{PermutationProof, prove, verify};

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
        F128::new(self.next_u64(), self.next_u64())
    }
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        p
    }
}

/// Honest instance: random `g`, permutation `σ`, and `f(x) = g(σ⁻¹(x))` so the
/// multiset `{(f, s_id)} = {(g, s_σ)}` holds and `∏ h = 1`.
fn honest_instance(mu: usize, seed: u64) -> (Vec<F128>, Vec<F128>, Vec<usize>) {
    let n = 1usize << mu;
    let mut rng = Rng::new(seed);
    let g: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let sigma = rng.permutation(n);
    let mut sinv = vec![0usize; n];
    for (x, &sx) in sigma.iter().enumerate() {
        sinv[sx] = x;
    }
    let f: Vec<F128> = (0..n).map(|x| g[sinv[x]]).collect();
    (f, g, sigma)
}

/// Absorb the statement `(f, g, σ)` into the transcript — the PIOP caller
/// contract for `prove`/`verify`.
fn bind<C: Challenger>(ch: &mut C, f: &[F128], g: &[F128], sigma: &[usize]) {
    ch.observe_f128_slice(f);
    ch.observe_f128_slice(g);
    for &s in sigma {
        ch.observe_f128(F128::new(s as u64, 0));
    }
}

fn run_prove(f: &[F128], g: &[F128], sigma: &[usize]) -> PermutationProof {
    let mut ch = FsChallenger::new(b"flock-perm-bench-v0");
    bind(&mut ch, f, g, sigma);
    prove(f, g, sigma, &mut ch).0
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");

    for &mu in &[8usize, 10, 12, 14, 16, 18, 20] {
        let n = 1usize << mu;
        println!("\n=== μ = {mu} (N = {n} = 2^{mu}) ===");

        let (f, g, sigma) = honest_instance(mu, 0xC0FFEE ^ mu as u64);

        // Warm-up: primes NTT / convert tables and the scratch pool.
        let warm = run_prove(&f, &g, &sigma);
        {
            let mut ch = FsChallenger::new(b"flock-perm-bench-v0");
            bind(&mut ch, &f, &g, &sigma);
            verify(mu, &warm, &mut ch).expect("warm-up verify");
        }

        let n_runs = if mu >= 16 { 3 } else { 5 };

        // ---- prove ----
        let mut best_prove = f64::INFINITY;
        let mut cs = 0u64;
        let mut proof = warm;
        for _ in 0..n_runs {
            let mut ch = FsChallenger::new(b"flock-perm-bench-v0");
            bind(&mut ch, &f, &g, &sigma);
            let t0 = Instant::now();
            let (p, _claim) = prove(black_box(&f), black_box(&g), black_box(&sigma), &mut ch);
            best_prove = best_prove.min(t0.elapsed().as_secs_f64() * 1e3);
            cs ^= p.claimed_product.lo ^ p.v_0x.lo ^ p.v_1x.lo;
            proof = p;
        }

        // ---- verify ----
        let mut best_verify = f64::INFINITY;
        for _ in 0..n_runs {
            let mut ch = FsChallenger::new(b"flock-perm-bench-v0");
            bind(&mut ch, &f, &g, &sigma);
            let t0 = Instant::now();
            let claim = verify(mu, black_box(&proof), &mut ch).expect("verify");
            best_verify = best_verify.min(t0.elapsed().as_secs_f64() * 1e3);
            cs ^= claim.rho[0].lo;
        }

        let backend = match &proof.v_open {
            flock_prover::pcs::BatchOpening::Ligerito(_) => "Ligerito",
            flock_prover::pcs::BatchOpening::BaseFold(_) => "BaseFold",
        };
        let proof_bytes = bincode::serialize(&proof).expect("serialize").len();
        println!("  {:<28} {}", "PCS backend (v open)", backend);
        println!("  {:<28} {:>10.3} ms", "prove", best_prove);
        println!("  {:<28} {:>10.3} ms", "verify", best_verify);
        println!(
            "  {:<28} {:>10.2} KiB",
            "proof size",
            proof_bytes as f64 / 1024.0
        );
        println!("  checksum: {cs:016x}");
    }
}
