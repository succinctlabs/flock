//! Probe for the affine L0 on the production (lane-major, virtual-basis)
//! path: what does the statistics sweep cost when every term needs the
//! dot products against `(T^t)^s e` for `s = 0..=k0`, instead of one?
//!
//! Shape: m = 32 Fast, 46 live lanes of 2^19 columns. The F256 ladder sweeps
//! one dot product per element (the OOD term; the seeded term's statistics
//! come from the merged sumcheck). The affine ladder needs 2 (k0 + 1) - 1 =
//! 13 (seeded s = 1..6, OOD s = 0..6).
//!
//! Run: `cargo bench --bench affine_l0_stats_probe`

use std::time::Instant;

use flock_prover::{
    field::{F128, F256Unreduced},
    init_perf_thread_pool,
    pcs::ligerito::affine::{carry_products, materialize_lanes, shifted_eq_functionals},
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

const LANES: usize = 46;
const LOG_COLS: usize = 19;
const REPS: usize = 9;
const CH: usize = 1 << 12;

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

/// Production-style: reduced multiply per product, SoA weights.
fn sweep_reduced(f: &[F128], withins: &[Vec<F128>], d: usize) -> Vec<F128> {
    let n_ch = d / CH;
    let parts: Vec<Vec<F128>> = (0..LANES * n_ch)
        .into_par_iter()
        .map(|task| {
            let (e, c) = (task / n_ch, task % n_ch);
            let h0 = c * CH;
            let fs = &f[e * d + h0..e * d + h0 + CH];
            withins
                .iter()
                .map(|w| {
                    fs.iter()
                        .zip(&w[h0..h0 + CH])
                        .fold(F128::ZERO, |acc, (&x, &y)| acc + x * y)
                })
                .collect()
        })
        .collect();
    parts.into_iter().flatten().collect()
}

/// Unreduced accumulate, weights interleaved `[h][t]` (one reduce per
/// chunk and term).
fn sweep_unreduced<const T: usize>(f: &[F128], aos: &[[F128; T]], d: usize) -> Vec<F128> {
    let n_ch = d / CH;
    let parts: Vec<[F128; T]> = (0..LANES * n_ch)
        .into_par_iter()
        .map(|task| {
            let (e, c) = (task / n_ch, task % n_ch);
            let h0 = c * CH;
            let fs = &f[e * d + h0..e * d + h0 + CH];
            let mut acc = [F256Unreduced::ZERO; T];
            for (x, ws) in fs.iter().zip(&aos[h0..h0 + CH]) {
                for (a, &w) in acc.iter_mut().zip(ws) {
                    *a ^= x.mul_unreduced(w);
                }
            }
            acc.map(F256Unreduced::reduce)
        })
        .collect();
    parts.into_iter().flatten().collect()
}

/// Weights pre-split for Karatsuba: `(lo, hi, lo ^ hi)` per term.
#[derive(Clone, Copy)]
struct Split {
    lo: u64,
    hi: u64,
    mid: u64,
}

/// NEON kernel: per term, accumulate the three Karatsuba PMULLs in q
/// registers; fold `cross` and reduce once per chunk. Terms run in groups of
/// `G` so the 3G accumulators fit the register file.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[allow(unused_unsafe)]
unsafe fn chunk_neon<const G: usize>(fs: &[F128], ws: &[[Split; G]]) -> [F128; G] {
    use std::arch::aarch64::{
        vdupq_n_u64, veorq_u64, vextq_u64, vgetq_lane_u64, vmull_p64, vreinterpretq_u64_p128,
    };
    unsafe {
        let z = vdupq_n_u64(0);
        let (mut ll, mut hh, mut mm) = ([z; G], [z; G], [z; G]);
        for (x, w) in fs.iter().zip(ws) {
            let xm = x.lo ^ x.hi;
            for t in 0..G {
                ll[t] = veorq_u64(ll[t], vreinterpretq_u64_p128(vmull_p64(x.lo, w[t].lo)));
                hh[t] = veorq_u64(hh[t], vreinterpretq_u64_p128(vmull_p64(x.hi, w[t].hi)));
                mm[t] = veorq_u64(mm[t], vreinterpretq_u64_p128(vmull_p64(xm, w[t].mid)));
            }
        }
        std::array::from_fn(|t| {
            let cross = veorq_u64(veorq_u64(mm[t], ll[t]), hh[t]);
            let lo = veorq_u64(ll[t], vextq_u64::<1>(z, cross));
            let hi = veorq_u64(hh[t], vextq_u64::<1>(cross, z));
            F256Unreduced {
                r0: vgetq_lane_u64::<0>(lo),
                r1: vgetq_lane_u64::<1>(lo),
                r2: vgetq_lane_u64::<0>(hi),
                r3: vgetq_lane_u64::<1>(hi),
            }
            .reduce()
        })
    }
}

#[cfg(target_arch = "aarch64")]
fn sweep_neon<const G: usize>(f: &[F128], groups: &[Vec<[Split; G]>], d: usize) -> Vec<F128> {
    let n_ch = d / CH;
    let parts: Vec<Vec<F128>> = (0..LANES * n_ch)
        .into_par_iter()
        .map(|task| {
            let (e, c) = (task / n_ch, task % n_ch);
            let h0 = c * CH;
            let fs = &f[e * d + h0..e * d + h0 + CH];
            groups
                .iter()
                // SAFETY: aes is enabled by target-cpu=native on Apple silicon.
                .flat_map(|g| unsafe { chunk_neon::<G>(fs, &g[h0..h0 + CH]) })
                .collect()
        })
        .collect();
    parts.into_iter().flatten().collect()
}

fn split_of(w: F128) -> Split {
    Split {
        lo: w.lo,
        hi: w.hi,
        mid: w.lo ^ w.hi,
    }
}

fn time<R>(f: impl FnOnce() -> R) -> f64 {
    let t = Instant::now();
    std::hint::black_box(f());
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    init_perf_thread_pool();
    if std::env::args().any(|a| a == "materialize") {
        materialize_mode();
        return;
    }
    let d = 1usize << LOG_COLS;
    let f = rand_vec(LANES * d, 1);
    let withins: Vec<Vec<F128>> = (0..13).map(|t| rand_vec(d, 100 + t)).collect();
    let aos1: Vec<[F128; 1]> = (0..d).map(|h| [withins[0][h]]).collect();
    let aos2: Vec<[F128; 2]> = (0..d).map(|h| [withins[0][h], withins[1][h]]).collect();
    let aos13: Vec<[F128; 13]> = (0..d)
        .map(|h| std::array::from_fn(|t| withins[t][h]))
        .collect();
    // Same values either way.
    assert_eq!(
        sweep_reduced(&f, &withins[..2], d),
        sweep_unreduced(&f, &aos2, d)
    );
    let group = |ts: std::ops::Range<usize>| -> Vec<[Split; 7]> {
        (0..d)
            .map(|h| {
                std::array::from_fn(|i| {
                    split_of(
                        withins
                            .get(ts.start + i)
                            .filter(|_| ts.start + i < ts.end)
                            .map_or(F128::ZERO, |w| w[h]),
                    )
                })
            })
            .collect()
    };
    let g1: Vec<[Split; 1]> = (0..d).map(|h| [split_of(withins[0][h])]).collect();
    let g13 = vec![group(0..7), group(7..13)];
    let neon2 = sweep_neon::<1>(
        &f,
        &[
            g1.clone(),
            (0..d).map(|h| [split_of(withins[1][h])]).collect(),
        ],
        d,
    );
    let red2 = sweep_reduced(&f, &withins[..2], d);
    // Same values, different order: lane-major [term][...] vs [lane][term].
    let n_ch = d / CH;
    for e in 0..LANES {
        for t in 0..2 {
            let a = (0..n_ch).fold(F128::ZERO, |acc, c| acc + neon2[(e * n_ch + c) * 2 + t]);
            let b = (0..n_ch).fold(F128::ZERO, |acc, c| acc + red2[(e * n_ch + c) * 2 + t]);
            assert_eq!(a, b, "neon kernel disagrees");
        }
    }
    let mut rows: Vec<(&str, Vec<f64>)> = vec![
        ("reduced,   1 term ", vec![]),
        ("reduced,   2 terms", vec![]),
        ("reduced,  13 terms", vec![]),
        ("unreduced, 1 term ", vec![]),
        ("unreduced, 2 terms", vec![]),
        ("unreduced,13 terms", vec![]),
        ("neon,      1 term ", vec![]),
        ("neon,  2x7 = 13   ", vec![]),
    ];
    for rep in 0..REPS + 1 {
        let ts = [
            time(|| sweep_reduced(&f, &withins[..1], d)),
            time(|| sweep_reduced(&f, &withins[..2], d)),
            time(|| sweep_reduced(&f, &withins, d)),
            time(|| sweep_unreduced(&f, &aos1, d)),
            time(|| sweep_unreduced(&f, &aos2, d)),
            time(|| sweep_unreduced(&f, &aos13, d)),
            time(|| sweep_neon::<1>(&f, std::slice::from_ref(&g1), d)),
            time(|| sweep_neon::<7>(&f, &g13, d)),
        ];
        if rep > 0 {
            for (row, t) in rows.iter_mut().zip(ts) {
                row.1.push(t);
            }
        }
    }
    // The affine L0 path: all s = 0..=6 functionals per lane from one
    // top-down fold, for one term.
    let z = rand_vec(LOG_COLS, 77);
    let p = carry_products(LOG_COLS + 2);
    let mut fold_t = Vec::new();
    for rep in 0..REPS + 1 {
        let t = time(|| {
            (0..LANES)
                .into_par_iter()
                .map(|e| shifted_eq_functionals(&f[e * d..(e + 1) * d], &z, 6, &p))
                .collect::<Vec<_>>()
        });
        if rep > 0 {
            fold_t.push(t);
        }
    }
    rows.push(("fold, s=0..6, 1 term", fold_t));
    println!("L0 statistics sweep, {LANES} lanes x 2^{LOG_COLS} (median of {REPS}, ms):");
    for (name, v) in rows {
        let min = v.iter().copied().fold(f64::INFINITY, f64::min);
        println!("  {name}  median {:7.2}  min {:7.2}", median(v), min);
    }
}

/// The L0 folded-message pass alone, chunk sizes interleaved.
fn materialize_mode() {
    let d = 1usize << LOG_COLS;
    let f = rand_vec(LANES * d, 1);
    let p = carry_products(LOG_COLS + 4);
    let raw = rand_vec(12, 9);
    let gammas: Vec<(F128, F128)> = raw.chunks(2).map(|c| (c[0], c[1])).collect();
    type Kernel = fn(&[F128], usize, usize, &[(F128, F128)], &[F128]) -> Vec<F128>;
    let kernels: [(&str, Kernel); 3] = [
        ("cols 2^8 ", materialize_lanes::<256>),
        ("cols 2^10", materialize_lanes::<1024>),
        ("cols 2^12", materialize_lanes::<4096>),
    ];
    let mut ts = vec![Vec::new(); kernels.len()];
    for rep in 0..16 {
        for (i, (_, k)) in kernels.iter().enumerate() {
            let start = Instant::now();
            std::hint::black_box(k(&f, d, LANES, &gammas, &p));
            if rep > 0 {
                ts[i].push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    println!("L0 materialize, {LANES} lanes x 2^{LOG_COLS}, 6 rounds (15 reps, ms):");
    for ((name, _), v) in kernels.iter().zip(ts) {
        let min = v.iter().copied().fold(f64::INFINITY, f64::min);
        println!("  {name}: median {:6.2}  min {:6.2}", median(v), min);
    }
}
