//! Focused micro-benchmark for the ring-switch `fold_1b_rows` kernels — the
//! dominant work inside `pcs::open_batch` (it batches the two ring-switch
//! openings as k=2). Compares the production 8-wide and 16-wide method-of-
//! four-Russians folds (2-way and 1-way) against the tensor-split kernel
//! `fold_1b_rows_split`, which factors `eq` into two ~2^(n/2) halves so the
//! multi-MB suffix tensor is never streamed. Sweeps the split width `n_lo`.
//! Best-of-N to isolate from pipeline thermal load. Honors RAYON_NUM_THREADS.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::pcs::ring_switch::{
    build_eq_split, fold_1b_rows_1way_mfr_8wide_k4, fold_1b_rows_1way_mfr_16wide_k4,
    fold_1b_rows_2way_mfr_8wide_padded, fold_1b_rows_split, split_n_lo,
};
use flock_prover::zerocheck::PaddingSpec;
use flock_prover::zerocheck::univariate_skip::build_eq;

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
    fn next_f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn bench_one(m: usize, n_runs: usize) {
    // packed_witness has 2^(m - LOG_PACKING) = 2^(m-7) F128 elements; the suffix
    // point r has that many coords.
    let lbits = m - 7;
    let len = 1usize << lbits;
    let mut rng = Rng::new(0xF01D ^ m as u64);
    let witness: Vec<F128> = (0..len).map(|_| rng.next_f128()).collect();
    // t0 is a *real* eq tensor so the split factorization lines up with it.
    let r: Vec<F128> = (0..lbits).map(|_| rng.next_f128()).collect();
    let t0: Vec<F128> = build_eq(&r);
    let t1: Vec<F128> = (0..len).map(|_| rng.next_f128()).collect();
    let padding = PaddingSpec::dense(m);

    let bench = |label: &str, f: &dyn Fn() -> f64| {
        for _ in 0..3 {
            black_box(f());
        }
        let mut best = f64::INFINITY;
        let mut sum = 0.0;
        for _ in 0..n_runs {
            let ms = f();
            best = best.min(ms);
            sum += ms;
        }
        println!(
            "  {label:<12} best {:7.2} ms   avg {:7.2} ms",
            best,
            sum / n_runs as f64
        );
    };

    // Correctness: 16-wide 1-way must match the 8-wide reference exactly.
    let r8 = fold_1b_rows_1way_mfr_8wide_k4(&witness, &t0);
    let r16 = fold_1b_rows_1way_mfr_16wide_k4(&witness, &t0);
    assert_eq!(r8, r16, "1way 16-wide diverges from 8-wide");
    // Correctness: the tensor-split fold must be byte-identical to the 16-wide
    // materialized kernel for every split width we bench.
    for n_lo in 9..=13.min(lbits) {
        let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
        let rs = fold_1b_rows_split(&witness, &eq_lo, &eq_hi, &padding);
        assert_eq!(rs, r16, "split (n_lo={n_lo}) diverges from 16-wide");
    }

    let mb = (3 * len * 16) >> 20;
    println!(
        "m={m:>2} (len={len:>8}, {mb:>3} MB)  split_n_lo={}",
        split_n_lo(lbits)
    );
    // Old production k=2 path: one fused 2-way 8-wide fold.
    bench("2way-8w", &|| {
        let t = Instant::now();
        let r = fold_1b_rows_2way_mfr_8wide_padded(&witness, &t0, &t1, &padding);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        black_box(&r);
        ms
    });
    // Current production k=2 path: two independent 1-way 16-wide folds (streams
    // the full t tensor twice).
    bench("2x1w-16w", &|| {
        let t = Instant::now();
        let r0 = fold_1b_rows_1way_mfr_16wide_k4(&witness, &t0);
        let r1 = fold_1b_rows_1way_mfr_16wide_k4(&witness, &t1);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        black_box((&r0, &r1));
        ms
    });
    bench("1way-16w", &|| {
        let t = Instant::now();
        let r = fold_1b_rows_1way_mfr_16wide_k4(&witness, &t0);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        black_box(&r);
        ms
    });
    // Tensor-split 1-way fold across a sweep of split widths. eq_lo/eq_hi are
    // built once outside the timed region (build_eq is no longer on the hot
    // path — that is the whole point of the split). The k=2 open path runs this
    // twice; we report the 1-way time to compare against `1way-16w`.
    for n_lo in 9..=13.min(lbits) {
        let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
        let label = format!("split n_lo={n_lo}");
        bench(&label, &|| {
            let t = Instant::now();
            let r = fold_1b_rows_split(&witness, &eq_lo, &eq_hi, &padding);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            black_box(&r);
            ms
        });
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    for &m in &[29usize, 30] {
        bench_one(m, 12);
    }
}
