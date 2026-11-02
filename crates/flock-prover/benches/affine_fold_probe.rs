//! Probe for docs/affine-fold-f128.tex: does an F128 affine fold pay off
//! against today's F256 fold + coordinate split, before building the full
//! prototype?
//!
//! Two measurements at the m = 32 Fast ladder (m32_fast.toml):
//!   1. Recursive commits (`ligero_commit`, the real NTT + Merkle) at the
//!      split shapes (L1..L5 tables 2^20, 2^17, 2^14, 2^11, 2^8) against the
//!      native shapes an affine fold would commit (2^19, 2^15, 2^11, 2^7).
//!   2. Simple fold kernels with a dense basis, both paths written the same
//!      naive way: per round the F256 path computes the degree-2 message and
//!      folds table + basis; the affine path computes the 2x3 grid and folds
//!      table (with the multiply-by-X map T) + basis. Production F256 kernels
//!      are fused and much faster than these; the probe compares like with
//!      like, not against production.
//!
//! Also checks the multiply-by-X map against the NTT: Enc(T g)(x) = x Enc(g)(x).
//!
//! Run: `cargo bench --bench affine_fold_probe`

use std::{hint::black_box, time::Instant};

use flock_prover::{
    field::{F128, F256},
    init_perf_thread_pool,
    merkle::HashKind,
    ntt::additive_ntt_f128::AdditiveNttF128,
    pcs::ligerito::ligero_commit,
};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};

const REPS: usize = 7;

fn rand_vec(len: usize, seed: u64) -> Vec<F128> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let lo = s;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            F128::new(lo, s)
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Prefix products P_i = a_0 ... a_{i-1} of the novel-basis squaring
/// constants a_i = W_{i+1}(v_{i+1}) / W_i(v_i)^2 for the standard basis
/// v_i = x^i (so X = W^_0 because W_0(v_0) = 1).
fn carry_products(dim: usize) -> Vec<F128> {
    let v: Vec<F128> = (0..=dim).map(|i| F128::new(1u64 << i, 0)).collect();
    // w[j] = W_i(v_j) for the current i.
    let mut w = v.clone();
    let mut c = Vec::with_capacity(dim + 1);
    for i in 0..=dim {
        c.push(w[i]);
        let ci = w[i];
        for wj in w.iter_mut() {
            *wj = *wj * (*wj + ci);
        }
    }
    let mut p = vec![F128::ONE];
    for i in 0..dim {
        let a = c[i + 1] * (c[i] * c[i]).inv();
        p.push(p[i] * a);
    }
    p
}

/// (T g)[m] for a row g of length `len` (output length len + 1):
/// P_{t(m-1)} g[m-1] + P_{tz(m)} g[m + 2^{tz(m)} - 1], where t = trailing
/// ones and tz = trailing zeros; the second term exists when in range.
#[inline(always)]
fn t_at(g: &[F128], len: usize, m: usize, p: &[F128]) -> F128 {
    if m == 0 {
        return F128::ZERO;
    }
    let mut out = p[(m - 1).trailing_ones() as usize] * g[m - 1];
    let tz = m.trailing_zeros() as usize;
    let j = m + (1usize << tz) - 1;
    if j < len {
        out += p[tz] * g[j];
    }
    out
}

fn check_t_map() {
    let c = 6;
    let len = 1usize << c;
    let p = carry_products(c + 4);
    let g = rand_vec(len, 7);
    let tg: Vec<F128> = (0..=len).map(|m| t_at(&g, len, m, &p)).collect();
    let log_dom = c + 2;
    let ntt = AdditiveNttF128::standard(log_dom);
    let mut eg = g.clone();
    eg.resize(1 << log_dom, F128::ZERO);
    ntt.forward_transform(&mut eg);
    let mut etg = tg.clone();
    etg.resize(1 << log_dom, F128::ZERO);
    ntt.forward_transform(&mut etg);
    let natural = (0..1usize << log_dom).all(|i| etg[i] == F128::new(i as u64, 0) * eg[i]);
    let reversed = (0..1usize << log_dom).all(|i| {
        let x = (i as u64).reverse_bits() >> (64 - log_dom);
        etg[i] == F128::new(x, 0) * eg[i]
    });
    println!(
        "T-map check (Enc(Tg)(x) == x Enc(g)(x)): natural order {natural}, bit-reversed {reversed}"
    );
    assert!(
        natural || reversed,
        "multiply-by-X map disagrees with the NTT"
    );
}

/// F256 path, one level: round 1 from the base table, rounds 2..k on F256.
/// Rows are contiguous blocks of 2^c; each round binds the lowest row bit.
fn f256_level(f: &[F128], b: &[F128], c: usize, k: usize) -> F256 {
    let l = 1usize << c;
    let r = F256::new(F128::new(3, 5), F128::new(7, 11));
    // Round 1 (base inputs).
    let (u0, u2) = f
        .par_chunks_exact(2 * l)
        .zip(b.par_chunks_exact(2 * l))
        .map(|(fr, br)| {
            let (f0, f1) = fr.split_at(l);
            let (b0, b1) = br.split_at(l);
            let mut acc = (F128::ZERO, F128::ZERO);
            for h in 0..l {
                acc.0 += b0[h] * f0[h];
                acc.1 += (b0[h] + b1[h]) * (f0[h] + f1[h]);
            }
            acc
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1));
    let fold_base = |x: &[F128]| -> Vec<F256> {
        x.par_chunks_exact(2 * l)
            .flat_map_iter(|row| {
                let (x0, x1) = row.split_at(l);
                (0..l).map(move |h| F256::from_base(x0[h]) + r * (x0[h] + x1[h]))
            })
            .collect()
    };
    let mut ft = fold_base(f);
    let mut bt = fold_base(b);
    let mut acc = F256::from_base(u0 + u2);
    for _ in 1..k {
        let (m0, m2) = ft
            .par_chunks_exact(2 * l)
            .zip(bt.par_chunks_exact(2 * l))
            .map(|(fr, br)| {
                let (f0, f1) = fr.split_at(l);
                let (b0, b1) = br.split_at(l);
                let mut a = (F256::ZERO, F256::ZERO);
                for h in 0..l {
                    a.0 += b0[h] * f0[h];
                    a.1 += (b0[h] + b1[h]) * (f0[h] + f1[h]);
                }
                a
            })
            .reduce(|| (F256::ZERO, F256::ZERO), |a, b| (a.0 + b.0, a.1 + b.1));
        acc += m0 + m2;
        let fold = |x: &[F256]| -> Vec<F256> {
            x.par_chunks_exact(2 * l)
                .flat_map_iter(|row| {
                    let (x0, x1) = row.split_at(l);
                    (0..l).map(move |h| x0[h] + r * (x0[h] + x1[h]))
                })
                .collect()
        };
        ft = fold(&ft);
        bt = fold(&bt);
    }
    acc + ft[0] + bt[0]
}

/// Affine path, one level: rows grow by one coefficient per round; the
/// table keeps a fixed row stride of 2^c + k.
fn affine_level(f: &[F128], b: &[F128], c: usize, k: usize, p: &[F128]) -> F128 {
    let l0 = 1usize << c;
    let stride = l0 + k;
    let rows = f.len() / l0;
    let (alpha, beta, r) = (F128::new(3, 5), F128::new(7, 11), F128::new(13, 17));
    let bp: Vec<F128> = p.iter().map(|&x| beta * x).collect();
    // Re-stride the inputs (rows padded with zeros), not timed separately:
    // a real implementation would allocate the stride up front.
    let pad = |x: &[F128]| -> Vec<F128> {
        let mut out = vec![F128::ZERO; rows * stride];
        out.par_chunks_exact_mut(stride)
            .zip(x.par_chunks_exact(l0))
            .for_each(|(o, i)| o[..l0].copy_from_slice(i));
        out
    };
    let mut ft = pad(f);
    let mut bt = pad(b);
    let mut acc = F128::ZERO;
    let mut len = l0;
    for _ in 0..k {
        // Grid: <f0,W0>, <f0,W1>, <f1,W0>, <f1,W1>, <T f1,W0>, <T f1,W1>.
        let g = ft
            .par_chunks_exact(2 * stride)
            .zip(bt.par_chunks_exact(2 * stride))
            .map(|(fr, br)| {
                let (f0, f1) = fr.split_at(stride);
                let (b0, b1) = br.split_at(stride);
                let mut a = [F128::ZERO; 6];
                for h in 0..len {
                    a[0] += f0[h] * b0[h];
                    a[1] += f0[h] * b1[h];
                    a[2] += f1[h] * b0[h];
                    a[3] += f1[h] * b1[h];
                }
                // The weight is zero on the new coordinate, so m < len.
                for m in 0..len {
                    let t = t_at(f1, len, m, p);
                    a[4] += t * b0[m];
                    a[5] += t * b1[m];
                }
                a
            })
            .reduce(
                || [F128::ZERO; 6],
                |mut x, y| {
                    for i in 0..6 {
                        x[i] += y[i];
                    }
                    x
                },
            );
        acc += g.iter().fold(F128::ZERO, |s, &x| s + x);
        let half = ft.len() / 2;
        let mut nf = vec![F128::ZERO; half];
        nf.par_chunks_exact_mut(stride)
            .zip(ft.par_chunks_exact(2 * stride))
            .for_each(|(o, fr)| {
                let (f0, f1) = fr.split_at(stride);
                for m in 0..=len {
                    let mut v = f0[m] + alpha * f1[m];
                    if m > 0 {
                        v += bp[(m - 1).trailing_ones() as usize] * f1[m - 1];
                        let tz = m.trailing_zeros() as usize;
                        let j = m + (1usize << tz) - 1;
                        if j < len {
                            v += bp[tz] * f1[j];
                        }
                    }
                    o[m] = v;
                }
            });
        let mut nb = vec![F128::ZERO; half];
        nb.par_chunks_exact_mut(stride)
            .zip(bt.par_chunks_exact(2 * stride))
            .for_each(|(o, br)| {
                let (b0, b1) = br.split_at(stride);
                for m in 0..len {
                    o[m] = b0[m] + r * (b0[m] + b1[m]);
                }
            });
        ft = nf;
        bt = nb;
        len += 1;
    }
    acc + ft[0] + bt[0]
}

fn time<T>(f: impl FnOnce() -> T) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let _ = init_perf_thread_pool();
    check_t_map();

    // 1. Recursive commits: (dims, log_lanes, log_inv_rate).
    let split = [(20, 4, 3), (17, 4, 5), (14, 4, 7), (11, 4, 9), (8, 4, 11)];
    let native = [(19, 4, 3), (15, 4, 5), (11, 4, 7), (7, 4, 9)];
    let commit = |&(dims, lanes, rate): &(usize, usize, usize), poly: &[F128]| {
        let ntt = AdditiveNttF128::standard(dims - lanes + rate);
        time(|| ligero_commit(poly, dims - lanes, lanes, rate, &ntt, HashKind::Blake3))
    };
    let polys: Vec<Vec<F128>> = split
        .iter()
        .map(|s| rand_vec(1 << s.0, s.0 as u64))
        .collect();
    let (mut ts, mut tn) = (
        vec![Vec::new(); split.len()],
        vec![Vec::new(); native.len()],
    );
    for _ in 0..REPS {
        for (i, s) in split.iter().enumerate() {
            ts[i].push(commit(s, &polys[i]));
        }
        for (i, s) in native.iter().enumerate() {
            tn[i].push(commit(s, &polys[0][..1 << s.0]));
        }
    }
    println!("\nrecursive commits (median of {REPS}, ms):");
    let mut sum = (0.0, 0.0);
    for (i, s) in split.iter().enumerate() {
        let a = median(ts[i].clone());
        let b = native.get(i).map(|_| median(tn[i].clone()));
        sum.0 += a;
        sum.1 += b.unwrap_or(0.0);
        match b {
            Some(b) => println!(
                "  L{}  split 2^{:<2} {:7.2}   native 2^{:<2} {:7.2}",
                i + 1,
                s.0,
                a,
                native[i].0,
                b
            ),
            None => println!(
                "  L{}  split 2^{:<2} {:7.2}   native  (none)",
                i + 1,
                s.0,
                a
            ),
        }
    }
    println!("  total        {:7.2}            {:7.2}", sum.0, sum.1);

    // 2. Fold kernels: L0 (2^25 table, 64 rows of 2^19, k = 6) and L1
    // (split 2^20 vs native 2^19, rows of 2^16 / 2^15, k = 4).
    let p = carry_products(24);
    let cases = [("L0", 25, 19, 25, 19, 6), ("L1", 20, 16, 19, 15, 4)];
    println!("\nfold kernels, simple and dense-basis (median of {REPS}, ms):");
    for &(name, sd, sc, nd, nc, k) in &cases {
        let f = rand_vec(1 << sd, 1);
        let b = rand_vec(1 << sd, 2);
        let (mut a, mut z) = (Vec::new(), Vec::new());
        for _ in 0..REPS {
            a.push(time(|| f256_level(&f, &b, sc, k)));
            z.push(time(|| {
                affine_level(&f[..1 << nd], &b[..1 << nd], nc, k, &p)
            }));
        }
        println!(
            "  {name}  F256 on 2^{sd}: {:7.2}   affine F128 on 2^{nd}: {:7.2}",
            median(a),
            median(z)
        );
    }
}
