//! Phase breakdown of the PRODUCTION AG zerocheck prove: times the production
//! kernels individually (round-1 banks-fused, skip→mlv fold) plus the full
//! `prove_capture_s_hat_v_c`, attributing the remainder to the mlv tail (+ r1
//! sample/functional/c-eval overhead). Lookahead on/off shows the tail split.
//!
//!   cargo +1.95.0 bench --bench ag_breakdown -- [m] [reps]   (default 30, 5)

// The AG round-1 kernel (and the ag_skip prover entry points this bench
// drives) are aarch64-only; on other arches the bench is a no-op stub.
#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("ag_breakdown: aarch64-only bench (NEON AG round-1 kernel)");
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
    use flock_prover::field::F128;
    use flock_prover::genus95_curve_code::round1::round1_slp_packed_banks_fused;
    use flock_prover::zerocheck::ag_skip::{
        LOOKAHEAD_DISABLE, N_INNER, fold_and_first_round, friendly_challenges,
        prove_capture_s_hat_v_c,
    };

    use flock_core::test_rng::Rng;

    pub(super) fn run() {
        let m: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        let reps: usize = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let bytes = (1usize << m) / 8;
        let mut rng = Rng(0xB0EA_D000 ^ m as u64);
        let mut a = vec![0u8; bytes];
        let mut b = vec![0u8; bytes];
        for x in a.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        for x in b.iter_mut() {
            *x = rng.next_u64() as u8;
        }
        let c: Vec<u8> = a.iter().zip(&b).map(|(&x, &y)| x & y).collect();
        let n_blocks = bytes / 1024;
        let eq: Vec<F128> = (0..n_blocks).map(|_| rng.f128()).collect();
        let w: Vec<F128> = (0..64).map(|_| rng.f128()).collect();
        let mut r_rest = friendly_challenges().to_vec();
        for _ in 0..(m - 6 - N_INNER) {
            r_rest.push(rng.f128());
        }

        eprintln!(
            "m={m} ({} MB/witness), {} reps, {} threads",
            bytes >> 20,
            reps,
            rayon::current_num_threads()
        );
        let med = |mut v: Vec<f64>| -> f64 {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let time = |f: &mut dyn FnMut()| -> f64 {
            f(); // warm
            let mut v = Vec::new();
            for _ in 0..reps {
                let t0 = Instant::now();
                f();
                v.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            med(v)
        };

        let t_r1 = time(&mut || {
            black_box(round1_slp_packed_banks_fused(&a, &b, &c, &eq));
        });
        let t_fold = time(&mut || {
            black_box(fold_and_first_round(&a, &b, &w, &r_rest));
        });
        LOOKAHEAD_DISABLE.store(false, Ordering::Relaxed);
        let t_total_la = time(&mut || {
            let mut ch = FsChallenger::new(b"ag-breakdown");
            black_box(prove_capture_s_hat_v_c(&a, &b, &c, m, &mut ch));
        });
        LOOKAHEAD_DISABLE.store(true, Ordering::Relaxed);
        let t_total_cl = time(&mut || {
            let mut ch = FsChallenger::new(b"ag-breakdown");
            black_box(prove_capture_s_hat_v_c(&a, &b, &c, m, &mut ch));
        });
        LOOKAHEAD_DISABLE.store(false, Ordering::Relaxed);

        let tail_la = t_total_la - t_r1 - t_fold;
        let tail_cl = t_total_cl - t_r1 - t_fold;
        eprintln!("\nproduction AG zerocheck breakdown (medians):");
        eprintln!(
            "  round-1 URM (banks fused)      {t_r1:7.2} ms  ({:4.1}%)",
            t_r1 / t_total_la * 100.0
        );
        eprintln!(
            "  skip->mlv fold (byte-dot u64)  {t_fold:7.2} ms  ({:4.1}%)",
            t_fold / t_total_la * 100.0
        );
        eprintln!(
            "  mlv tail + misc, LOOKAHEAD     {tail_la:7.2} ms  ({:4.1}%)",
            tail_la / t_total_la * 100.0
        );
        eprintln!("  ---------------------------------------------");
        eprintln!("  TOTAL prove (lookahead)        {t_total_la:7.2} ms");
        eprintln!("  [tail+misc classic: {tail_cl:.2} ms; total classic: {t_total_cl:.2} ms]");
    }
}
