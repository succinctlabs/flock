//! Same-process paired A/B of the fused round-1 kernel inside the FULL
//! production AG zerocheck prove (`prove_capture_s_hat_v_c`), via the
//! [`ROUND1_UNFUSED`] toggle. Alternating unfused/fused in one process cancels
//! thermal drift (the codebase's DISABLE_FRIENDLY_HORNER methodology); the
//! paired per-round delta is the statistic, not cross-run absolute times.
//!
//!   cargo +1.95.0 bench --bench ag_round1_ab [m] [rounds]      # MT (default)
//!   RAYON_NUM_THREADS=1 cargo bench --bench ag_round1_ab       # ST
//!
//! Default m=28 (32 MB/witness, 32768 round-1 blocks), 8 paired rounds.

// The AG round-1 kernel (and the ag_skip prover entry points this bench
// drives) are aarch64-only; on other arches the bench is a no-op stub.
#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("ag_round1_ab: aarch64-only bench (NEON AG round-1 kernel)");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    aarch64_only::run()
}

#[cfg(target_arch = "aarch64")]
mod aarch64_only {
    use std::hint::black_box;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use flock_prover::challenger::FsChallenger;
    use flock_prover::zerocheck::ag_skip::{ROUND1_UNFUSED, prove_capture_s_hat_v_c};

    struct Rng(u64);
    impl Rng {
        fn n(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    pub(super) fn run() {
        let m: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(28);
        let rounds: usize = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let bytes = (1usize << m) / 8;

        // Valid witnesses: c = a ∘ b is bitwise AND over GF(2) rows.
        let mut rng = Rng(0xA0B0_C0D0);
        let mut a = vec![0u8; bytes];
        let mut b = vec![0u8; bytes];
        for x in a.iter_mut() {
            *x = rng.n() as u8;
        }
        for x in b.iter_mut() {
            *x = rng.n() as u8;
        }
        let c: Vec<u8> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();

        let nthreads = rayon::current_num_threads();
        eprintln!(
            "m={m} ({} MB/witness, {} round-1 blocks), {} paired rounds, {} threads",
            bytes >> 20,
            bytes / 1024,
            rounds,
            nthreads
        );

        let run = |unfused: bool| -> f64 {
            ROUND1_UNFUSED.store(unfused, Ordering::Relaxed);
            let mut ch = FsChallenger::new(b"ag-round1-ab");
            let t0 = Instant::now();
            let out = prove_capture_s_hat_v_c(&a, &b, &c, m, &mut ch);
            let el = t0.elapsed().as_secs_f64();
            black_box(out);
            el * 1e3
        };

        // Warm both paths (scratch pool, page faults) before timing.
        let _ = run(true);
        let _ = run(false);

        let mut deltas = Vec::new();
        let (mut us, mut fs) = (Vec::new(), Vec::new());
        for r in 0..rounds {
            // Alternate order each round to cancel any first-mover advantage.
            let (tu, tf) = if r % 2 == 0 {
                let tu = run(true);
                let tf = run(false);
                (tu, tf)
            } else {
                let tf = run(false);
                let tu = run(true);
                (tu, tf)
            };
            eprintln!(
                "  unfused {tu:7.2} ms   fused {tf:7.2} ms   paired Δ {:+.2} ms",
                tu - tf
            );
            deltas.push(tu - tf);
            us.push(tu);
            fs.push(tf);
        }
        ROUND1_UNFUSED.store(false, Ordering::Relaxed);

        let med = |mut v: Vec<f64>| -> f64 {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let (mu, mf, md) = (med(us), med(fs), med(deltas.clone()));
        let wins = deltas.iter().filter(|d| **d > 0.0).count();
        eprintln!("\nmedian: unfused {mu:.2} ms   fused {mf:.2} ms");
        eprintln!(
            "paired-delta median {md:+.2} ms ({:+.1}% of zerocheck prove); fused faster in {wins}/{rounds} rounds",
            md / mu * 100.0
        );
    }
}
