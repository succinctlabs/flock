//! End-to-end `zerocheck::prove_packed` benchmark. **pack_bits is hoisted**
//! outside the timed section to match the C++ benchmark methodology
//! (real provers commit to packed witnesses).
//!
//! Witnesses are generated directly as packed bytes to avoid allocating
//! 3 × 512 MB of `&[bool]` at m=29.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::zerocheck::prove_packed;

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
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");

    for &m in &[16usize, 20, 24, 26, 28, 29] {
        let n_bits = 1usize << m;
        let n_bytes = n_bits / 8;
        println!(
            "\n=== m = {m} ({} boolean constraints, {} MB packed) ===",
            n_bits,
            n_bytes >> 20
        );

        let mut rng = Rng::new(0xDEAD_C0DE + m as u64);
        let n_runs = if m >= 24 { 3 } else { 1 };

        // Pre-generate n_runs + 1 distinct witness triples — first is warm-up,
        // rest are timed. Honest: c = a AND b. Distinct inputs → distinct
        // FS transcripts.
        let mut wits: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(n_runs + 1);
        for _ in 0..=n_runs {
            let mut a_packed = vec![0u8; n_bytes];
            rng.fill_bytes(&mut a_packed);
            let mut b_packed = vec![0u8; n_bytes];
            rng.fill_bytes(&mut b_packed);
            let c_packed: Vec<u8> = a_packed.iter().zip(&b_packed).map(|(x, y)| x & y).collect();
            wits.push((a_packed, b_packed, c_packed));
        }

        // Warm-up to prime the OnceLock-cached convert table.
        {
            let (a_packed, b_packed, c_packed) = &wits[0];
            let mut ch = FsChallenger::new(b"flock-bench-v0");
            let _ = prove_packed(a_packed, b_packed, c_packed, m, &mut ch);
        }

        let mut best_ms = f64::INFINITY;
        let mut cs = 0u64;
        for run in 0..n_runs {
            let label = if n_runs == 1 {
                String::from("zerocheck::prove_packed")
            } else {
                format!("zerocheck::prove_packed (run {})", run + 1)
            };
            let (a_packed, b_packed, c_packed) = &wits[run + 1];
            let mut ch = FsChallenger::new(b"flock-bench-v0");
            let t0 = Instant::now();
            let (proof, claim) = prove_packed(
                black_box(a_packed),
                black_box(b_packed),
                black_box(c_packed),
                m,
                &mut ch,
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<40} {:>10.2} ms", label, elapsed);
            best_ms = best_ms.min(elapsed);
            cs ^=
                proof.final_a_eval.lo ^ proof.final_b_eval.lo ^ proof.final_c_eval.lo ^ claim.z.lo;
        }
        if n_runs > 1 {
            println!("  {:<40} {:>10.2} ms", "  (best)", best_ms);
        }
        println!("  checksum: {cs:016x}");
    }
}
