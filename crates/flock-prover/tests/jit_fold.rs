//! The measurement I skipped: a COLUMN-SEGMENTED just-in-time W, producing
//! folded VALUES, for ARBITRARY (non-power-of-two) column heights.
//!
//! W[e] = eq_row[e - prefix[c]] * eq_col[c].  Within a column eq_col is
//! constant and the row index runs contiguously, so a segment of the output is
//!    W'[i] = eq_row[r0+k]*A + eq_row[r1+k]*B,   A = ec0*(1+rho), B = ec1*rho
//! with A, B hoisted per segment. Two cursors (for e and e+D) walk the column
//! structure; nothing requires h to divide D.
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const JM: usize = 23;
const LOG_D: usize = 17;
const USED: usize = 121;
const H: usize = 48_000; // DELIBERATELY not a power of two

fn bench(label: &str, f: &mut dyn FnMut() -> usize) -> f64 {
    let mut b = f64::INFINITY;
    f();
    for _ in 0..4 {
        let t = Instant::now();
        black_box(f());
        b = b.min(t.elapsed().as_secs_f64());
    }
    println!("  {label:<52} {:6.2} ms", b * 1e3);
    b * 1e3
}

#[test]
#[ignore]
fn jit_fold() {
    let len = 1usize << JM;
    let d = 1usize << LOG_D;
    let area = USED * H;
    assert!(area <= len);
    let mut seed = 0xF00Du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        F128 {
            lo: seed,
            hi: seed.rotate_left(29),
        }
    };
    let eq_row = build_eq_table(&(0..16).map(|_| rnd()).collect::<Vec<_>>());
    let eq_col: Vec<F128> = (0..USED).map(|_| rnd()).collect();
    let rho = rnd();
    let one_p = F128::ONE + rho;
    // W[e] for e < area, else 0. row = e % H, col = e / H (uniform heights).
    let wof = |e: usize| -> F128 {
        if e >= area {
            F128::ZERO
        } else {
            eq_row[e % H] * eq_col[e / H]
        }
    };

    let mut w = vec![F128::ZERO; len];
    let t_mat = bench("(1) materialize W (2^23)", &mut || {
        w.par_chunks_mut(1 << 16).enumerate().for_each(|(ci, c)| {
            let g0 = ci << 16;
            for (k, s) in c.iter_mut().enumerate() {
                *s = wof(g0 + k);
            }
        });
        w.len()
    });

    let mut out = vec![F128::ZERO; len / 2];
    let t_fold = bench("(2) fold from the materialized W", &mut || {
        out.par_chunks_mut(1 << 16).enumerate().for_each(|(ci, c)| {
            let g0 = ci << 16;
            for (k, s) in c.iter_mut().enumerate() {
                let i = g0 + k;
                let (b, p) = (i / d, i % d);
                let lo = w[2 * b * d + p];
                let hi = w[(2 * b + 1) * d + p];
                *s = lo + rho * (hi + lo);
            }
        });
        out.len()
    });

    let mut out2 = vec![F128::ZERO; len / 2];
    let t_jit = bench(
        "(3) fold with COLUMN-SEGMENTED JIT W (no materialize)",
        &mut || {
            // One task per output block; inside, walk segments where both cursors
            // stay inside a column, hoisting A and B.
            out2.par_chunks_mut(d).enumerate().for_each(|(b, ob)| {
                let (e0b, e1b) = (2 * b * d, (2 * b + 1) * d);
                let mut p = 0usize;
                while p < d {
                    let (e0, e1) = (e0b + p, e1b + p);
                    let (c0, c1) = (e0 / H, e1 / H);
                    let (r0, r1) = (e0 % H, e1 % H);
                    // segment length: until either cursor leaves its column, or the
                    // block ends
                    let n = (d - p).min(H - r0).min(H - r1);
                    let a = if c0 < USED {
                        eq_col[c0] * one_p
                    } else {
                        F128::ZERO
                    };
                    let bb = if c1 < USED {
                        eq_col[c1] * rho
                    } else {
                        F128::ZERO
                    };
                    if a == F128::ZERO && bb == F128::ZERO {
                        for s in ob[p..p + n].iter_mut() {
                            *s = F128::ZERO;
                        }
                    } else {
                        for (k, s) in ob[p..p + n].iter_mut().enumerate() {
                            *s = eq_row[r0 + k] * a + eq_row[r1 + k] * bb;
                        }
                    }
                    p += n;
                }
            });
            out2.len()
        },
    );
    assert_eq!(
        out, out2,
        "JIT fold must match the materialized fold exactly"
    );
    println!(
        "\n  materialize + fold = {:.2} ms   vs   JIT fold = {:.2} ms   ({:+.2} ms)\n",
        t_mat + t_fold,
        t_jit,
        t_jit - (t_mat + t_fold)
    );
}
