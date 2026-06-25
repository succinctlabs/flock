//! Isolated microbench probe: original (fold + γ-mul-combine) vs γ-baked
//! (fold-with-γ-scaled-eq_r_dprime + add-combine). m=30 ST. Each path
//! produces byte-identical b_combined. Reports per-phase ms.
//!
//! Run with: RAYON_NUM_THREADS=1 cargo bench --bench gamma_bake_probe

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::pcs::ring_switch::fold_b128_elems_split;

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
}

fn fmt_ms(s: f64) -> String {
    format!("{:>8.2} ms", s * 1000.0)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    println!("gamma-bake probe ({threads} thread(s))\n");

    // m=30 setup: x_outer has length m - 6 = 24. Suffix x_outer[1..] has length
    // 23. Split into n_hi = 12 / n_lo = 11 for the dense_splits factor.
    const M: usize = 30;
    let l = 1usize << (M - 7); // 8.4M
    let n_lo = (M - 7) / 2; // 11
    let n_hi = (M - 7) - n_lo; // 12
    let b_lo = 1usize << n_lo; // 2048
    let b_hi = 1usize << n_hi; // 4096
    assert_eq!(b_lo * b_hi, l);

    println!(
        "m = {M}, L = 2^{} = {l}, n_lo = {n_lo}, n_hi = {n_hi}",
        M - 7
    );

    let mut rng = Rng::new(0xC0FFEE);

    // Build (eq_lo, eq_hi) for 2 claims.
    let eq_lo_0: Vec<F128> = (0..b_lo).map(|_| rng.f128()).collect();
    let eq_hi_0: Vec<F128> = (0..b_hi).map(|_| rng.f128()).collect();
    let eq_lo_1: Vec<F128> = (0..b_lo).map(|_| rng.f128()).collect();
    let eq_hi_1: Vec<F128> = (0..b_hi).map(|_| rng.f128()).collect();

    // Build eq_r_dprime (length 128) for 2 claims via the standard
    // tensor-product expansion of 7-coord r''.
    fn build_eq(r: &[F128]) -> Vec<F128> {
        let mut acc = vec![F128 { lo: 1, hi: 0 }];
        for &ri in r {
            let mut next = Vec::with_capacity(acc.len() * 2);
            let one = F128 { lo: 1, hi: 0 };
            for &a in &acc {
                next.push(a * (one + ri));
                next.push(a * ri);
            }
            acc = next;
        }
        acc
    }
    let r_dprime_0: Vec<F128> = (0..7).map(|_| rng.f128()).collect();
    let r_dprime_1: Vec<F128> = (0..7).map(|_| rng.f128()).collect();
    let eq_r_dprime_0 = build_eq(&r_dprime_0);
    let eq_r_dprime_1 = build_eq(&r_dprime_1);

    let g0 = rng.f128();
    let g1 = rng.f128();

    // Warm caches with a discarded run.
    {
        let _ = black_box(fold_b128_elems_split(&eq_lo_0, &eq_hi_0, &eq_r_dprime_0));
        let _ = black_box(fold_b128_elems_split(&eq_lo_1, &eq_hi_1, &eq_r_dprime_1));
    }

    // ============================================================
    // Path A: original — 2 separate folds, then γ-mul combine.
    // ============================================================
    println!("\n[PATH A] original (2 folds + γ-mul combine)");
    let mut a_fold0_total = 0.0;
    let mut a_fold1_total = 0.0;
    let mut a_combine_total = 0.0;
    let mut a_total = 0.0;
    const RUNS: usize = 5;
    let mut b_a = Vec::new();
    for run in 0..RUNS {
        let t_all = Instant::now();
        let t0 = Instant::now();
        let b0 = fold_b128_elems_split(&eq_lo_0, &eq_hi_0, &eq_r_dprime_0);
        let dt0 = t0.elapsed().as_secs_f64();
        a_fold0_total += dt0;

        let t1 = Instant::now();
        let b1 = fold_b128_elems_split(&eq_lo_1, &eq_hi_1, &eq_r_dprime_1);
        let dt1 = t1.elapsed().as_secs_f64();
        a_fold1_total += dt1;

        let tc = Instant::now();
        use rayon::prelude::*;
        let b_combined: Vec<F128> = (0..l)
            .into_par_iter()
            .map(|j| g0 * b0[j] + g1 * b1[j])
            .collect();
        let dtc = tc.elapsed().as_secs_f64();
        a_combine_total += dtc;

        let dtt = t_all.elapsed().as_secs_f64();
        a_total += dtt;

        if run == RUNS - 1 {
            b_a = b_combined;
        }
        black_box(&b0);
        black_box(&b1);
    }
    println!(
        "  fold[0]       avg {}",
        fmt_ms(a_fold0_total / RUNS as f64)
    );
    println!(
        "  fold[1]       avg {}",
        fmt_ms(a_fold1_total / RUNS as f64)
    );
    println!(
        "  γ-mul combine avg {}",
        fmt_ms(a_combine_total / RUNS as f64)
    );
    println!("  TOTAL         avg {}", fmt_ms(a_total / RUNS as f64));

    // ============================================================
    // Path B: γ-baked — scale eq_r_dprime by γ, fold, add (no γ-mul).
    // ============================================================
    println!("\n[PATH B] γ-baked (γ-scale eq_r_dprime + 2 folds + add combine)");
    let mut b_scale0_total = 0.0;
    let mut b_scale1_total = 0.0;
    let mut b_fold0_total = 0.0;
    let mut b_fold1_total = 0.0;
    let mut b_combine_total = 0.0;
    let mut b_total = 0.0;
    let mut b_b = Vec::new();
    for run in 0..RUNS {
        let t_all = Instant::now();

        let ts0 = Instant::now();
        let scaled_0: Vec<F128> = eq_r_dprime_0.iter().map(|x| g0 * *x).collect();
        let dts0 = ts0.elapsed().as_secs_f64();
        b_scale0_total += dts0;

        let t0 = Instant::now();
        let b0 = fold_b128_elems_split(&eq_lo_0, &eq_hi_0, &scaled_0);
        let dt0 = t0.elapsed().as_secs_f64();
        b_fold0_total += dt0;

        let ts1 = Instant::now();
        let scaled_1: Vec<F128> = eq_r_dprime_1.iter().map(|x| g1 * *x).collect();
        let dts1 = ts1.elapsed().as_secs_f64();
        b_scale1_total += dts1;

        let t1 = Instant::now();
        let b1 = fold_b128_elems_split(&eq_lo_1, &eq_hi_1, &scaled_1);
        let dt1 = t1.elapsed().as_secs_f64();
        b_fold1_total += dt1;

        let tc = Instant::now();
        use rayon::prelude::*;
        let b_combined: Vec<F128> = (0..l).into_par_iter().map(|j| b0[j] + b1[j]).collect();
        let dtc = tc.elapsed().as_secs_f64();
        b_combine_total += dtc;

        let dtt = t_all.elapsed().as_secs_f64();
        b_total += dtt;

        if run == RUNS - 1 {
            b_b = b_combined;
        }
        black_box(&b0);
        black_box(&b1);
    }
    println!(
        "  scale eq[0]    avg {}",
        fmt_ms(b_scale0_total / RUNS as f64)
    );
    println!(
        "  fold[0]        avg {}",
        fmt_ms(b_fold0_total / RUNS as f64)
    );
    println!(
        "  scale eq[1]    avg {}",
        fmt_ms(b_scale1_total / RUNS as f64)
    );
    println!(
        "  fold[1]        avg {}",
        fmt_ms(b_fold1_total / RUNS as f64)
    );
    println!(
        "  add combine    avg {}",
        fmt_ms(b_combine_total / RUNS as f64)
    );
    println!("  TOTAL          avg {}", fmt_ms(b_total / RUNS as f64));

    // Verify byte-identical (both paths produce the same b_combined).
    assert_eq!(b_a.len(), b_b.len());
    let mismatches: usize = b_a.iter().zip(b_b.iter()).filter(|(a, b)| a != b).count();
    println!(
        "\nbyte-identical check: {} / {} mismatches",
        mismatches,
        b_a.len()
    );
    assert_eq!(
        mismatches, 0,
        "Path A and Path B must produce byte-identical b_combined"
    );

    println!(
        "\nDelta (A - B): {:.2} ms",
        (a_total - b_total) / RUNS as f64 * 1000.0
    );
}
