//! Latency-stability probe: N timed `prove_fast` runs with per-run output,
//! for distribution comparison across branches (mean/median/spread/p95 +
//! significance tests run offline). Each run uses distinct inputs and a
//! fresh challenger; 3 untimed warm-up proves precede the timed runs.
//!
//!   STAB_RUNS=30 cargo bench --bench stability_probe
//!
//! Prints one `RUN <bench> <ms>` line per timed prove (machine-readable).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;

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
    fn u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let runs: usize = std::env::var("STAB_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    // STAB_WARMUP=0 exposes cold-start (first-prove) behavior.
    let warmup: usize = std::env::var("STAB_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    // ---- sha2 m=29 (default prove_fast path) ----
    {
        use flock_prover::r1cs_hashes::sha2::Sha256HybridSetup;
        let n = 16384usize;
        let setup = Sha256HybridSetup::new(n);
        let mut rng = Rng::new(0x57AB);
        let mk = |rng: &mut Rng| -> Vec<([u32; 8], [u32; 16])> {
            (0..n)
                .map(|_| {
                    (
                        std::array::from_fn(|_| rng.u32()),
                        std::array::from_fn(|_| rng.u32()),
                    )
                })
                .collect()
        };
        for _ in 0..warmup {
            let inputs = mk(&mut rng);
            let mut ch = FsChallenger::new(b"stab");
            black_box(setup.prove_fast(&inputs, &mut ch));
        }
        for _ in 0..runs {
            let inputs = mk(&mut rng);
            let mut ch = FsChallenger::new(b"stab");
            let t = Instant::now();
            let p = setup.prove_fast(&inputs, &mut ch);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            black_box(&p);
            println!("RUN sha2_m29 {ms:.3}");
        }
    }

    // ---- keccak m=30 ----
    {
        use flock_prover::r1cs_hashes::keccak::{KeccakSetup, State};
        let k = 16384usize;
        let setup = KeccakSetup::new(k);
        let mut rng = Rng::new(0x57AC);
        let mk = |rng: &mut Rng| -> Vec<State> {
            (0..k)
                .map(|_| std::array::from_fn(|_| rng.next_u64() & 1 == 1))
                .collect()
        };
        for _ in 0..warmup {
            let inputs = mk(&mut rng);
            let mut ch = FsChallenger::new(b"stab");
            black_box(setup.prove_fast(&inputs, &mut ch));
        }
        for _ in 0..runs {
            let inputs = mk(&mut rng);
            let mut ch = FsChallenger::new(b"stab");
            let t = Instant::now();
            let p = setup.prove_fast(&inputs, &mut ch);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            black_box(&p);
            println!("RUN keccak_m30 {ms:.3}");
        }
    }
}
