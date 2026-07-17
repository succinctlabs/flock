//! Round-2 URM fused fold + message bench — sweep through m=29.
//!
//! Mirrors `benches/round1.rs`. pack_bits is hoisted; fold table is built once
//! per `z` and reused across timed runs (matches the C++ harness).
//!
//! Target from PROTOCOL_REFERENCE.md: ~38 ms single-thread at m=29 (the
//! "29→23 collapse + P_2(1), P_2(∞)" step).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::{F8, F128};
use flock_prover::zerocheck::multilinear::{
    UniSkipFoldTable, uni_skip_fold_and_round_pair_optimized_packed,
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
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let len = buf.len();
        let mut i = 0;
        while i + 8 <= len {
            let v = self.next_u64();
            buf[i..i + 8].copy_from_slice(&v.to_le_bytes());
            i += 8;
        }
        if i < len {
            let v = self.next_u64().to_le_bytes();
            buf[i..].copy_from_slice(&v[..len - i]);
        }
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

fn _silence_unused() {
    let _ = F8::ZERO;
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    println!("(target: x86_64 + AVX-512/VPCLMULQDQ path active)");
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )
    )))]
    println!("(target: scalar fallback)");

    for &m in &[16usize, 20, 24, 26, 28, 29] {
        let n_bits = 1usize << m;
        let n_bytes = n_bits / 8;
        println!(
            "\n=== m = {m} ({} boolean constraints, {} MB packed) ===",
            n_bits,
            n_bytes >> 20
        );

        let mut rng = Rng::new(0xBEEF0042 + m as u64);

        // Generate packed witnesses directly.
        let mut a_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut a_packed);
        let mut b_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut b_packed);

        // URM fold challenge and mlv challenges.
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - K_SKIP);

        // Pre-build fold table (one-time per z). Time it separately.
        let t0 = Instant::now();
        let table = UniSkipFoldTable::new(K_SKIP, z);
        let table_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  fold table build (one-time)              {:>10.2} ms",
            table_ms
        );

        // Warm-up to prime caches.
        let _ = uni_skip_fold_and_round_pair_optimized_packed(
            &a_packed,
            &b_packed,
            m,
            K_SKIP,
            &table,
            &mlv_challenges,
        );

        // Three timed runs at large m.
        let n_runs = if m >= 24 { 3 } else { 1 };
        let mut best_ms = f64::INFINITY;
        let mut cs_a = 0u64;
        let mut cs_b = 0u64;
        let mut cs_msg = 0u64;
        for run in 0..n_runs {
            let label = if n_runs == 1 {
                String::from("fused fold + round-2 msg")
            } else {
                format!("fused fold + round-2 msg (run {})", run + 1)
            };
            let t0 = Instant::now();
            let (a_mlv, b_mlv, m1, minf) = uni_skip_fold_and_round_pair_optimized_packed(
                black_box(&a_packed),
                black_box(&b_packed),
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<40} {:>10.2} ms", label, elapsed);
            best_ms = best_ms.min(elapsed);
            cs_a ^= a_mlv[0].lo;
            cs_b ^= b_mlv[0].lo;
            cs_msg ^= m1.lo ^ minf.lo;
        }
        if n_runs > 1 {
            println!("  {:<40} {:>10.2} ms", "  (best)", best_ms);
        }
        println!(
            "  checksums: a_mlv[0].lo={cs_a:016x}  b_mlv[0].lo={cs_b:016x}  msg={cs_msg:016x}"
        );
    }
}
