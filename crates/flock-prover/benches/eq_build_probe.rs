//! eq-build PMULL-vs-mul_by_x probe.
//!
//! Measures how much time the multilinear-round eq tensor build actually takes
//! (where PMULLs by deterministic friendly challenges would happen), to size
//! the optimization opportunity.
//!
//! The structural insight: medium friendlies satisfy β_i = γ^k · (1 + β_i)
//! where k = 2^(i-1). In `build_eq`, the sibling computation
//!   t[x | (1 << i)] = t[x] * β_i
//! can be replaced by mul_by_x^k of the left child:
//!   t[x | (1 << i)] = mul_by_x^k(t[x] * (1 + β_i))
//! This saves PMULL ops at the cost of mul_by_x ops on M4 (PMULL ~3 cy
//! throughput, mul_by_x ~1 cy).
//!
//! Run with: RAYON_NUM_THREADS=1 cargo bench --bench eq_build_probe

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::{F128, mul_by_x};
use flock_prover::zerocheck::univariate_skip::{SplitEqGhash, build_eq};
use flock_prover::zerocheck::univariate_skip_optimized::{
    medium_challenges_ghash, small_challenges_ghash,
};

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

fn build_eq_with_geometric_medium(r: &[F128]) -> Vec<F128> {
    // r layout assumption matches round-2 eq build at k_skip=6:
    //   r[0..2] = 2 small friendlies (2nd and 3rd small after URM)
    //   r[2..6] = 4 medium friendlies (β_1, β_2, β_3, β_4)
    //   r[6..]  = random
    //
    // The medium bits at indices 2..6 use the geometric exploit; everything
    // else uses standard PMULL.
    let n = r.len();
    let mut t = vec![F128 { lo: 0, hi: 0 }; 1usize << n];
    t[0] = F128 { lo: 1, hi: 0 };

    for i in 0..n {
        let r_i = r[i];
        let one_minus_r = F128 { lo: 1, hi: 0 } + r_i;

        if (2..6).contains(&i) {
            // Medium bit: use mul_by_x^k exploit.
            let k = 1usize << (i - 2); // i=2→k=1, i=3→k=2, i=4→k=4, i=5→k=8
            for x in (0..(1usize << i)).rev() {
                let left = t[x] * one_minus_r;
                let mut right = left;
                for _ in 0..k {
                    right = mul_by_x(right);
                }
                t[x | (1 << i)] = right;
                t[x] = left;
            }
        } else {
            // Standard PMULL build.
            for x in (0..(1usize << i)).rev() {
                t[x | (1 << i)] = t[x] * r_i;
                t[x] *= one_minus_r;
            }
        }
    }
    t
}

fn make_r_round2(n: usize) -> Vec<F128> {
    // Mimic the round-2 eq weights: small_2, small_3, medium_1..4, then random.
    let small = small_challenges_ghash();
    let medium = medium_challenges_ghash();
    let mut r = Vec::with_capacity(n);
    r.push(small[1]);
    r.push(small[2]);
    r.extend(medium.iter().copied());
    let mut rng = Rng::new(0xCAFE_BABE);
    while r.len() < n {
        r.push(rng.f128());
    }
    r
}

fn fmt_ms(s: f64) -> String {
    format!("{:>9.4} ms", s * 1000.0)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    println!("eq-build probe: PMULL vs mul_by_x for medium friendlies\n");

    // Sweep eq sizes corresponding to round 2-7 at m=30:
    //   round 2: eq.lo built from r[7..23] (n=16)
    //   round 3: r[8..23] (n=15)
    //   ...
    for n in [10usize, 12, 14, 16, 18, 20].iter().copied() {
        let r = make_r_round2(n);
        const RUNS: usize = 50;

        // Standard build.
        let mut tot_std = 0.0;
        let mut result_std = Vec::new();
        for _ in 0..RUNS {
            let t = Instant::now();
            let eq = build_eq(&r);
            tot_std += t.elapsed().as_secs_f64();
            result_std = eq;
        }

        // Geometric-medium build.
        let mut tot_geo = 0.0;
        let mut result_geo = Vec::new();
        for _ in 0..RUNS {
            let t = Instant::now();
            let eq = build_eq_with_geometric_medium(&r);
            tot_geo += t.elapsed().as_secs_f64();
            result_geo = eq;
        }

        // Verify byte-identical.
        let mismatches = result_std
            .iter()
            .zip(&result_geo)
            .filter(|(a, b)| a != b)
            .count();
        let table_kb = (1usize << n) * 16 / 1024;

        println!(
            "n={n:>2} (table {table_kb:>6} KB):  std={}  geo={}  delta={}  mismatches={}",
            fmt_ms(tot_std / RUNS as f64),
            fmt_ms(tot_geo / RUNS as f64),
            fmt_ms((tot_std - tot_geo) / RUNS as f64),
            mismatches
        );
        black_box(&result_std);
        black_box(&result_geo);
    }

    // Also: measure SplitEqGhash::new since that's what the rounds actually call.
    println!("\nSplitEqGhash::new (round-2 shape at m=30: n=23, n_lo=16, n_hi=7):");
    let r = make_r_round2(23);
    const RUNS: usize = 30;
    let mut tot = 0.0;
    for _ in 0..RUNS {
        let t = Instant::now();
        let _eq = SplitEqGhash::new(&r);
        tot += t.elapsed().as_secs_f64();
    }
    println!("  avg: {}", fmt_ms(tot / RUNS as f64));
}
