//! Round-1 URM benchmark for the **degree-4** zerocheck.
//!
//! Compares:
//!   1. `round1_deg4_naive`             — reference, F128 throughout.
//!   2. `round1_shift_reduce_extract_z_packed_deg4` — scalar optimized.
//!
//! Optimized variant uses direct `AdditiveNttGf8` calls per row (no lookup
//! table yet) and no NEON; rough speedup over naive is from the shift_reduce
//! + convert-table tricks alone. Once we wire the new `InvNttTableSToV8Gf8`
//! lookup into this path we expect another ~5–10× over what we measure here.
//!
//! Naive only runs at m ≤ 18 — beyond that it's many seconds.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::ntt::InvNttTableSToV8Gf8;
use flock_prover::zerocheck::univariate_skip::pack_bits;
use flock_prover::zerocheck::univariate_skip_deg4::round1_deg4_naive;
use flock_prover::zerocheck::univariate_skip_deg4_optimized::{
    NttPairDeg4, medium_challenges_deg4, round1_shift_reduce_extract_z_packed_deg4,
    small_challenges_deg4,
};

const K_SKIP: usize = 6;
const N_INNER: usize = 7;

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
    fn bits(&mut self, n: usize) -> Vec<bool> {
        (0..n).map(|_| self.next_u64() & 1 == 1).collect()
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn build_protocol_r(m: usize, rng: &mut Rng) -> Vec<F128> {
    let mut r = vec![F128::ZERO; m];
    for i in 0..K_SKIP {
        r[i] = rng.f128();
    }
    let small = small_challenges_deg4();
    for i in 0..3 {
        r[K_SKIP + i] = small[i];
    }
    let med = medium_challenges_deg4();
    for i in 0..4 {
        r[K_SKIP + 3 + i] = med[i];
    }
    for i in (K_SKIP + N_INNER)..m {
        r[i] = rng.f128();
    }
    r
}

fn time_ms<R>(label: &str, n_runs: usize, f: impl Fn() -> R) -> f64 {
    // Warm up.
    let _ = f();
    let mut best = f64::INFINITY;
    let mut last_check: Option<R> = None;
    for run in 0..n_runs {
        let t0 = Instant::now();
        let r = f();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        best = best.min(dt);
        if run == 0 || n_runs > 1 {
            println!(
                "  {:<48} run {}/{}: {:>10.2} ms",
                label,
                run + 1,
                n_runs,
                dt
            );
        }
        last_check = Some(r);
    }
    black_box(last_check);
    if n_runs > 1 {
        println!("  {:<48} best:        {:>10.2} ms", label, best);
    }
    best
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — but deg-4 optimized is scalar-only for now)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");
    println!("(rayon threads available: {threads} — deg-4 driver is single-threaded internally)");

    // Build NTT pair + lookup table once.
    let ntts = NttPairDeg4::new();
    let table = InvNttTableSToV8Gf8::new(&ntts.ntt_s, &ntts.ntt_v8);

    for &m in &[16usize, 18, 20, 22, 24, 26] {
        let n_bits = 1usize << m;
        println!(
            "\n=== m = {m} ({} boolean constraints, {} KB packed/factor) ===",
            n_bits,
            n_bits / 8 / 1024
        );

        let mut rng = Rng::new(0xDEC0DE ^ (m as u64));
        let a = rng.bits(n_bits);
        let b = rng.bits(n_bits);
        let c = rng.bits(n_bits);
        let d = rng.bits(n_bits);
        let z = rng.bits(n_bits);
        let r = build_protocol_r(m, &mut rng);

        let a_p = pack_bits(&a);
        let b_p = pack_bits(&b);
        let c_p = pack_bits(&c);
        let d_p = pack_bits(&d);
        let z_p = pack_bits(&z);

        // ----- naive (small m only) -----
        let mut naive_checksum = 0u64;
        if m <= 18 {
            let (ab, zz) = (a.clone(), z.clone());
            let cc = c.clone();
            let dd = d.clone();
            let _t = time_ms("naive (round1_deg4_naive, F128)", 1, || {
                round1_deg4_naive(
                    black_box(&ab),
                    black_box(&b),
                    black_box(&cc),
                    black_box(&dd),
                    black_box(&zz),
                    m,
                    &r,
                )
            });
            let (p_abcd, p_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);
            naive_checksum = p_abcd[0].lo ^ p_z[0].lo;
        }

        // ----- optimized (scalar, single-thread) — best-of-3 at every size -----
        let n_runs = 3;
        let _t = time_ms("optimized (shift_reduce + convert, scalar)", n_runs, || {
            round1_shift_reduce_extract_z_packed_deg4(
                black_box(&a_p),
                black_box(&b_p),
                black_box(&c_p),
                black_box(&d_p),
                black_box(&z_p),
                m,
                &r,
                &ntts,
                &table,
            )
        });
        let (opt_abcd, opt_z) = round1_shift_reduce_extract_z_packed_deg4(
            &a_p, &b_p, &c_p, &d_p, &z_p, m, &r, &ntts, &table,
        );
        let opt_checksum = opt_abcd[0].lo ^ opt_z[0].lo;

        if naive_checksum != 0 {
            println!("  checksums: naive={naive_checksum:016x}  optimized={opt_checksum:016x}");
        } else {
            println!("  checksum: optimized={opt_checksum:016x}");
        }
    }
}
