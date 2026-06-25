//! Microbenchmark for the lincheck verifier's matrix-multiplication cost
//! on the sha2 R1CS (K_LOG=15). Times the four stages of the
//! per-claim verifier work:
//!   - `build_quirky_eq_table` (eq_inner construction)
//!   - `sparse_row_fold` for A_0 and B_0  (the "multiply by A/B" pass — XOR-only)
//!   - `inner_product` for A and B consistency checks (the "× z_vec" pass — F128 muls)
//!   - final `build_quirky_eq_table` + `inner_product` for the derived z-claim

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::lincheck::{build_quirky_eq_table, sparse_row_fold};
use flock_prover::r1cs_hashes::sha2::{K_LOG, K_SKIP, build_matrices};

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

fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += *x * *y;
    }
    acc
}

fn fmt(s: f64) -> String {
    let us = s * 1e6;
    if us < 1.0 {
        format!("{:>8.2} ns", s * 1e9)
    } else if us < 1000.0 {
        format!("{:>8.2} µs", us)
    } else {
        format!("{:>8.2} ms", s * 1e3)
    }
}

fn time_one<F: FnMut() -> R, R>(label: &str, n_iters: usize, mut f: F) -> f64 {
    // Warm-up.
    let _ = black_box(f());
    let t = Instant::now();
    for _ in 0..n_iters {
        let r = f();
        black_box(&r);
    }
    let dt = t.elapsed().as_secs_f64() / n_iters as f64;
    println!("  {:32} {}", label, fmt(dt));
    dt
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let (a_0, b_0) = build_matrices();
    let k = 1usize << K_LOG;
    let inner_rest_len = K_LOG - K_SKIP;

    let nnz_a: usize = a_0.rows.iter().map(|r| r.len()).sum();
    let nnz_b: usize = b_0.rows.iter().map(|r| r.len()).sum();
    println!("=== lincheck verifier mult-cost — sha2 (K_LOG={K_LOG}, k={k}) ===");
    println!("  A_0 nnz = {}", nnz_a);
    println!("  B_0 nnz = {}", nnz_b);
    println!("  k_skip  = {K_SKIP}  (inner_rest_len = {inner_rest_len})");
    println!();

    let mut rng = Rng::new(0xC0FFEE_5A55);
    let z_skip = rng.f128();
    let x_inner_rest: Vec<F128> = (0..inner_rest_len).map(|_| rng.f128()).collect();
    let z_vec: Vec<F128> = (0..k).map(|_| rng.f128()).collect();
    let r_inner_skip = rng.f128();
    let r_inner_rest: Vec<F128> = (0..inner_rest_len).map(|_| rng.f128()).collect();

    // Stage 1: build eq_inner (used by both sparse_row_folds).
    let t_eq_inner = time_one("build_quirky_eq_table (eq_inner)", 200, || {
        build_quirky_eq_table(z_skip, &x_inner_rest, K_SKIP)
    });

    // Stage 2: sparse_row_fold(A_0, eq_inner) — XOR-only, O(nnz_A).
    let eq_inner = build_quirky_eq_table(z_skip, &x_inner_rest, K_SKIP);
    let t_fold_a = time_one("sparse_row_fold(A_0)            ", 200, || {
        sparse_row_fold(&a_0, &eq_inner)
    });
    let a_row = sparse_row_fold(&a_0, &eq_inner);

    // Stage 3: sparse_row_fold(B_0, eq_inner) — XOR-only, O(nnz_B).
    let t_fold_b = time_one("sparse_row_fold(B_0)            ", 200, || {
        sparse_row_fold(&b_0, &eq_inner)
    });
    let b_row = sparse_row_fold(&b_0, &eq_inner);

    // Stage 4: inner_product(a_row, z_vec) — k F128 muls.
    let t_ip_a = time_one("inner_product(a_row, z_vec)     ", 500, || {
        inner_product(&a_row, &z_vec)
    });

    // Stage 5: inner_product(b_row, z_vec) — k F128 muls.
    let t_ip_b = time_one("inner_product(b_row, z_vec)     ", 500, || {
        inner_product(&b_row, &z_vec)
    });

    // Stage 6: build_quirky_eq_table for r_inner (final z-claim derivation).
    let t_eq_r = time_one("build_quirky_eq_table (eq_r)    ", 200, || {
        build_quirky_eq_table(r_inner_skip, &r_inner_rest, K_SKIP)
    });

    // Stage 7: inner_product(eq_r_quirky, z_vec) — k F128 muls.
    let eq_r = build_quirky_eq_table(r_inner_skip, &r_inner_rest, K_SKIP);
    let t_ip_w = time_one("inner_product(eq_r, z_vec)      ", 500, || {
        inner_product(&eq_r, &z_vec)
    });

    let total = t_eq_inner + t_fold_a + t_fold_b + t_ip_a + t_ip_b + t_eq_r + t_ip_w;
    println!();
    println!("  TOTAL verifier mult work:       {}", fmt(total));
    println!();
    println!("  --- breakdown ---");
    println!(
        "  Multiply by A,B (sparse_row_fold A + B): {}  ({:.1}% — XOR-only)",
        fmt(t_fold_a + t_fold_b),
        100.0 * (t_fold_a + t_fold_b) / total
    );
    println!(
        "  Inner products (3 × k muls):             {}  ({:.1}% — F128 muls)",
        fmt(t_ip_a + t_ip_b + t_ip_w),
        100.0 * (t_ip_a + t_ip_b + t_ip_w) / total
    );
    println!(
        "  Eq-table builds (2):                     {}  ({:.1}%)",
        fmt(t_eq_inner + t_eq_r),
        100.0 * (t_eq_inner + t_eq_r) / total
    );
}
