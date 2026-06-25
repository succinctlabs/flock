//! Microbench: MLE evaluation at a random F128 point over a 2^23 F128 buffer
//! (the packed witness shape at m=30, log_packing=7). Compares:
//!
//!   - **Naive**:    new[j] = (1 + r)·f[2j] + r·f[2j+1]      (2 muls / pair)
//!   - **Remark 1.7**: new[j] = f[2j] + r·(f[2j] + f[2j+1])  (1 mul / pair)
//!
//! Both single-thread and rayon-parallel. Run:
//! `cargo run --release --example mle_eval_bench`.

// Deliberate uninit output buffers for the parallel fold (write-before-read,
// same pattern as flock_core's internal `alloc_uninit_vec`).
#![allow(clippy::uninit_vec)]

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use rayon::prelude::*;

const D: usize = 23; // number of variables to fold

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128::new(self.nx(), self.nx())
    }
}

fn fold_one_naive_st(buf: &mut Vec<F128>, r: F128) {
    let n = buf.len();
    let half = n / 2;
    // new[j] = (1+r)·f[2j] + r·f[2j+1]
    let one_plus_r = F128::ONE + r;
    for j in 0..half {
        let f0 = buf[2 * j];
        let f1 = buf[2 * j + 1];
        buf[j] = one_plus_r * f0 + r * f1;
    }
    buf.truncate(half);
}

fn fold_one_remark17_st(buf: &mut Vec<F128>, r: F128) {
    let n = buf.len();
    let half = n / 2;
    // new[j] = f[2j] + r · (f[2j] + f[2j+1])
    for j in 0..half {
        let f0 = buf[2 * j];
        let f1 = buf[2 * j + 1];
        buf[j] = f0 + r * (f0 + f1);
    }
    buf.truncate(half);
}

fn fold_one_naive_mt(buf: &mut Vec<F128>, r: F128) {
    let n = buf.len();
    let half = n / 2;
    let one_plus_r = F128::ONE + r;
    // Need a separate output buffer to parallelize cleanly (in-place would
    // race because pair j reads buf[2j+1] which is in the "to-be-overwritten"
    // half).
    let mut out: Vec<F128> = Vec::with_capacity(half);
    unsafe {
        out.set_len(half);
    }
    out.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let f0 = buf[2 * j];
        let f1 = buf[2 * j + 1];
        *slot = one_plus_r * f0 + r * f1;
    });
    *buf = out;
}

fn fold_one_remark17_mt(buf: &mut Vec<F128>, r: F128) {
    let n = buf.len();
    let half = n / 2;
    let mut out: Vec<F128> = Vec::with_capacity(half);
    unsafe {
        out.set_len(half);
    }
    out.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let f0 = buf[2 * j];
        let f1 = buf[2 * j + 1];
        *slot = f0 + r * (f0 + f1);
    });
    *buf = out;
}

fn eval_mle<F: FnMut(&mut Vec<F128>, F128)>(
    coeffs: &[F128],
    challenges: &[F128],
    mut fold: F,
) -> F128 {
    let mut buf: Vec<F128> = coeffs.to_vec();
    for &r in challenges {
        fold(&mut buf, r);
    }
    debug_assert_eq!(buf.len(), 1);
    buf[0]
}

fn fmt_ms(s: f64) -> String {
    format!("{:>8.2} ms", s * 1000.0)
}

fn bench<F: FnMut(&mut Vec<F128>, F128)>(
    label: &str,
    coeffs: &[F128],
    challenges: &[F128],
    runs: usize,
    mut fold: F,
) -> (F128, f64) {
    // Warm-up
    let warm = eval_mle(coeffs, challenges, &mut fold);
    black_box(warm);

    let mut best = f64::INFINITY;
    let mut last = F128::ZERO;
    for _ in 0..runs {
        let t0 = Instant::now();
        let v = eval_mle(coeffs, challenges, &mut fold);
        let elapsed = t0.elapsed().as_secs_f64();
        best = best.min(elapsed);
        last = v;
        black_box(v);
    }
    println!("  [{label:>14}]  best {}", fmt_ms(best));
    (last, best)
}

fn main() {
    let n = 1usize << D;
    println!(
        "MLE eval microbench: 2^{D} = {n} F128 entries, {} variables to fold",
        D
    );
    println!("Buffer: {} MB", (n * 16) >> 20);
    println!("======================================================================");

    let mut rng = Rng::new(0xC0FFEE_BEEF);
    let coeffs: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let challenges: Vec<F128> = (0..D).map(|_| rng.f128()).collect();

    let runs = 5;
    println!("\n--- Single-thread (best of {runs}) ---");
    let (v1, _) = bench("naive ST", &coeffs, &challenges, runs, fold_one_naive_st);
    let (v2, _) = bench(
        "remark1.7 ST",
        &coeffs,
        &challenges,
        runs,
        fold_one_remark17_st,
    );
    assert_eq!(v1, v2, "naive and remark1.7 must agree (ST)");

    println!("\n--- Multi-thread (best of {runs}) ---");
    let (v3, _) = bench("naive MT", &coeffs, &challenges, runs, fold_one_naive_mt);
    let (v4, _) = bench(
        "remark1.7 MT",
        &coeffs,
        &challenges,
        runs,
        fold_one_remark17_mt,
    );
    assert_eq!(v1, v3, "naive ST and MT must agree");
    assert_eq!(v1, v4, "naive and remark1.7 MT must agree");

    println!("\n(all four results agree: {v1:?})");
}
