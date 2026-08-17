//! Same-process paired A/B of sumcheck LOOKAHEAD inside the full production AG
//! zerocheck prove (`prove_capture_s_hat_v_c`), via [`LOOKAHEAD_DISABLE`].
//! Alternating classic/lookahead in one process cancels thermal drift; the
//! paired per-round delta is the statistic. Proofs asserted bit-identical.
//!
//!   cargo +1.95.0 bench --bench ag_lookahead_ab [m] [rounds]      # MT (default)
//!   RAYON_NUM_THREADS=1 cargo bench --bench ag_lookahead_ab       # ST

// The AG round-1 kernel (and the ag_skip prover entry points this bench
// drives) are aarch64-only; on other arches the bench is a no-op stub.
#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("ag_lookahead_ab: aarch64-only bench (NEON AG round-1 kernel)");
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
    use flock_prover::zerocheck::ag_skip::{
        LOOKAHEAD_DISABLE, LOOKAHEAD_FRIENDLY, prove_capture_s_hat_v_c,
    };

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

        let mut rng = Rng(0x1A0C_AB00);
        let mut a = vec![0u8; bytes];
        let mut b = vec![0u8; bytes];
        for x in a.iter_mut() {
            *x = rng.n() as u8;
        }
        for x in b.iter_mut() {
            *x = rng.n() as u8;
        }
        let c: Vec<u8> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();

        eprintln!(
            "m={m} ({} MB/witness), {} paired rounds, {} threads",
            bytes >> 20,
            rounds,
            rayon::current_num_threads()
        );

        let run = |classic: bool| -> (f64, Vec<u8>) {
            LOOKAHEAD_DISABLE.store(classic, Ordering::Relaxed);
            let mut ch = FsChallenger::new(b"ag-lookahead-ab");
            let t0 = Instant::now();
            let out = prove_capture_s_hat_v_c(&a, &b, &c, m, &mut ch);
            let el = t0.elapsed().as_secs_f64();
            let ser = bincode::serialize(&out.0).unwrap();
            black_box(&out);
            (el * 1e3, ser)
        };

        // Warm both paths and assert bit-identical proofs.
        let (_, p_classic) = run(true);
        let (_, p_look) = run(false);
        assert!(p_classic == p_look, "lookahead proof != classic proof");
        eprintln!("proofs bit-identical: OK");

        let med = |mut v: Vec<f64>| -> f64 {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let mut deltas = Vec::new();
        let (mut tc, mut tl) = (Vec::new(), Vec::new());
        for r in 0..rounds {
            let (c_ms, l_ms) = if r % 2 == 0 {
                let (c_ms, _) = run(true);
                let (l_ms, _) = run(false);
                (c_ms, l_ms)
            } else {
                let (l_ms, _) = run(false);
                let (c_ms, _) = run(true);
                (c_ms, l_ms)
            };
            eprintln!(
                "  classic {c_ms:7.2} ms   lookahead {l_ms:7.2} ms   paired Δ {:+.2} ms",
                c_ms - l_ms
            );
            deltas.push(c_ms - l_ms);
            tc.push(c_ms);
            tl.push(l_ms);
        }
        LOOKAHEAD_DISABLE.store(false, Ordering::Relaxed);
        let wins = deltas.iter().filter(|d| **d > 0.0).count();
        eprintln!(
            "\nmedian: classic {:.2} ms   lookahead {:.2} ms   paired-delta {:+.2} ms ({:+.1}% of zerocheck prove); lookahead faster {wins}/{rounds}",
            med(tc.clone()),
            med(tl),
            med(deltas.clone()),
            med(deltas) / med(tc) * 100.0
        );

        // ---- Second phase: friendly-lookahead vs general-lookahead (i=1,3) ----
        let run_f = |general: bool| -> f64 {
            LOOKAHEAD_FRIENDLY.store(!general, Ordering::Relaxed);
            let mut ch = FsChallenger::new(b"ag-lookahead-ab");
            let t0 = Instant::now();
            let out = prove_capture_s_hat_v_c(&a, &b, &c, m, &mut ch);
            let el = t0.elapsed().as_secs_f64();
            black_box(out);
            el * 1e3
        };
        let mut fdeltas = Vec::new();
        let (mut tg, mut tf) = (Vec::new(), Vec::new());
        for r in 0..rounds {
            let (g_ms, f_ms) = if r % 2 == 0 {
                let g = run_f(true);
                let f = run_f(false);
                (g, f)
            } else {
                let f = run_f(false);
                let g = run_f(true);
                (g, f)
            };
            eprintln!(
                "  la-general {g_ms:7.2} ms   la-friendly {f_ms:7.2} ms   paired Δ {:+.2} ms",
                g_ms - f_ms
            );
            fdeltas.push(g_ms - f_ms);
            tg.push(g_ms);
            tf.push(f_ms);
        }
        LOOKAHEAD_FRIENDLY.store(false, Ordering::Relaxed);
        let fwins = fdeltas.iter().filter(|d| **d > 0.0).count();
        eprintln!(
            "\nmedian: la-general {:.2} ms   la-friendly {:.2} ms   paired-delta {:+.2} ms ({:+.1}%); friendly faster {fwins}/{rounds}",
            med(tg.clone()),
            med(tf),
            med(fdeltas.clone()),
            med(fdeltas) / med(tg) * 100.0
        );
    }
}
