//! DESIGN PROTOTYPE for a not-yet-implemented optimization: skipping the
//! materialization of the jagged weight table `W_rho` in the fused opening.
//!
//! These tests validate an ALGEBRAIC PROPERTY that the implementation would
//! rest on; they do not exercise production code. Kept so the derivation is
//! pinned and machine-checked rather than living in a commit message.
//!
//! Is the jagged weight's factored form CLOSED under the blocked lane fold,
//! and what is skipping materialization worth?
//!
//! W[e] = eq_row[e % h] * eq_col[e / h]   (h = uniform column height, zero
//! past the used columns). The L0 fold uses block size D = c*h, pairing block
//! 2b with 2b+1 at equal offset p, so both elements share row `p % h` and sit
//! in columns `2b*c + p/h` and `(2b+1)*c + p/h`. Hence
//!
//!   W'[j*h + r] = eq_row[r] * ( eq_col[2b*c+s]*(1+rho) + eq_col[(2b+1)*c+s]*rho )
//!               = eq_row[r] * eq_col'[j]          with j = b*c + s
//!
//! i.e. the SAME factored form with `eq_col` halved. If that holds, the L0
//! rounds never need a 2^m basis array at all.
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Materialized reference: W over the whole dense domain.
fn materialize(len: usize, h: usize, eq_row: &[F128], eq_col: &[F128]) -> Vec<F128> {
    (0..len)
        .map(|e| {
            let c = e / h;
            if c < eq_col.len() {
                eq_row[e % h] * eq_col[c]
            } else {
                F128::ZERO
            }
        })
        .collect()
}

/// Blocked fold of a materialized basis: out[b*d+p] = W[2b*d+p] + r*(W[(2b+1)*d+p] + W[2b*d+p]).
fn fold_materialized(w: &[F128], d: usize, r: F128) -> Vec<F128> {
    let half = w.len() / 2;
    (0..half)
        .map(|i| {
            let (b, p) = (i / d, i % d);
            let lo = w[2 * b * d + p];
            let hi = w[(2 * b + 1) * d + p];
            lo + r * (hi + lo)
        })
        .collect()
}

/// The claim: fold `eq_col` with stride `cols_per_block` instead.
fn fold_eq_col(eq_col: &[F128], cols_per_block: usize, r: F128) -> Vec<F128> {
    let n = eq_col.len() / 2;
    (0..n)
        .map(|j| {
            let (b, s) = (j / cols_per_block, j % cols_per_block);
            let lo = eq_col[2 * b * cols_per_block + s];
            let hi = eq_col[(2 * b + 1) * cols_per_block + s];
            lo + r * (hi + lo)
        })
        .collect()
}

/// The load-bearing claim: folding `W` is folding `eq_col`. Exhaustive over
/// small shapes, every block size, and used-column counts that leave a zero
/// tail in `eq_col` (the real instances always do).
#[test]
fn factored_form_is_closed_under_the_blocked_fold() {
    // Small instance, exhaustive: 2^k columns of height 2^n, several block
    // sizes, several used-column counts (so `eq_col` has a zero tail).
    let mut seed = 0xC0FFEEu64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        F128 {
            lo: seed,
            hi: seed.rotate_left(31),
        }
    };
    for n in [2usize, 3, 4] {
        for k in [2usize, 3] {
            let h = 1usize << n;
            let n_cols = 1usize << k;
            let len = h * n_cols;
            let eq_row = build_eq_table(&(0..n).map(|_| rnd()).collect::<Vec<_>>());
            let full_col = build_eq_table(&(0..k).map(|_| rnd()).collect::<Vec<_>>());
            for used in [n_cols, n_cols - 1, n_cols / 2 + 1] {
                // zero tail past the used columns
                let mut eq_col = full_col.clone();
                for slot in eq_col[used..].iter_mut() {
                    *slot = F128::ZERO;
                }
                let w = materialize(len, h, &eq_row, &eq_col);
                for cpb_log in 0..k {
                    let cols_per_block = 1usize << cpb_log;
                    let d = cols_per_block * h;
                    if len / d < 2 {
                        continue;
                    }
                    let r = rnd();
                    let want = fold_materialized(&w, d, r);
                    let eq_col2 = fold_eq_col(&eq_col, cols_per_block, r);
                    // Reconstruct from the FACTORED form and compare.
                    let got: Vec<F128> = (0..len / 2)
                        .map(|i| {
                            let c = i / h;
                            if c < eq_col2.len() {
                                eq_row[i % h] * eq_col2[c]
                            } else {
                                F128::ZERO
                            }
                        })
                        .collect();
                    assert_eq!(got, want, "n={n} k={k} used={used} cpb={cols_per_block}");
                }
            }
        }
    }
}

#[test]
#[ignore]
fn factored_basis_cost() {
    // m=30 commit shape: jagged m=23, h=2^16, 128 grid columns (121 used).
    let (n, k) = (16usize, 7usize);
    let h = 1usize << n;
    let len = h * (1usize << k);
    let mut seed = 0xBEEFu64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        F128 {
            lo: seed,
            hi: seed.rotate_left(31),
        }
    };
    let eq_row = build_eq_table(&(0..n).map(|_| rnd()).collect::<Vec<_>>());
    let mut eq_col = build_eq_table(&(0..k).map(|_| rnd()).collect::<Vec<_>>());
    for slot in eq_col[121..].iter_mut() {
        *slot = F128::ZERO;
    }
    let d = 1usize << 17;
    let r = rnd();

    let mut w = vec![F128::ZERO; len];
    let bench = |label: &str, f: &mut dyn FnMut() -> usize| {
        let mut b = f64::INFINITY;
        f();
        for _ in 0..4 {
            let t = Instant::now();
            let v = f();
            black_box(v);
            b = b.min(t.elapsed().as_secs_f64());
        }
        println!("  {label:<46} {:6.2} ms", b * 1e3);
        b * 1e3
    };
    let t_mat = bench("materialize W (2^23)", &mut || {
        w.par_chunks_mut(1 << 16).enumerate().for_each(|(ci, c)| {
            let g0 = ci << 16;
            for (kk, slot) in c.iter_mut().enumerate() {
                let e = g0 + kk;
                let col = e / h;
                *slot = if col < 121 {
                    eq_row[e % h] * eq_col[col]
                } else {
                    F128::ZERO
                };
            }
        });
        w.len()
    });
    let mut out = vec![F128::ZERO; len / 2];
    let t_fold = bench("fold the materialized W (2^23 -> 2^22)", &mut || {
        out.par_chunks_mut(1 << 16).enumerate().for_each(|(ci, c)| {
            let g0 = ci << 16;
            for (kk, slot) in c.iter_mut().enumerate() {
                let i = g0 + kk;
                let (b, p) = (i / d, i % d);
                let lo = w[2 * b * d + p];
                let hi = w[(2 * b + 1) * d + p];
                *slot = lo + r * (hi + lo);
            }
        });
        out.len()
    });
    let t_fac = bench("fold the FACTORED eq_col (128 entries)", &mut || {
        let cpb = d / h;
        let nn = eq_col.len() / 2;
        let v: Vec<F128> = (0..nn)
            .map(|j| {
                let (b, s) = (j / cpb, j % cpb);
                let lo = eq_col[2 * b * cpb + s];
                let hi = eq_col[(2 * b + 1) * cpb + s];
                lo + r * (hi + lo)
            })
            .collect();
        v.len()
    });
    println!(
        "\n  materialize + one fold = {:.2} ms  ->  factored = {:.4} ms\n",
        t_mat + t_fold,
        t_fac
    );
}
