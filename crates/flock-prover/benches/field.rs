//! Field-arithmetic microbenchmarks (no criterion / harness).
//!
//! Methodology:
//!   * Latency    — tight dependent chain `a = op(a, b)` repeated N times.
//!   * Throughput — 4 independent accumulators to expose ILP.
//!
//! 100M iters per measurement. A checksum is printed alongside each result so
//! the compiler can't eliminate the loop body.
//!
//! Run:   `cargo bench --bench field`
//! On M-series, requires `.cargo/config.toml` so the `aes` feature is on.

use std::hint::black_box;
use std::time::Instant;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use flock_prover::field::gf2_128::aarch64;
use flock_prover::field::gf2_128::software;
use flock_prover::field::{F8, F128, F256Unreduced};

const N: usize = 100_000_000;

// Targets (M-series single-thread, prior C++ baseline):
//   F8 mul scalar (latency)     ~6.1  ns/op
//   F128 GHASH add (throughput) ~0.06 ns/op
//   F128 GHASH mul (binius64)   ~5.7  ns/op
// These are the numbers to match.

fn header(name: &str) {
    println!("\n===== {name} =====");
}

fn report(name: &str, total_ns: f64, ops: usize, checksum: u64) {
    let ns_per = total_ns / ops as f64;
    let mops = 1000.0 / ns_per;
    println!(
        "  {:<46} {:>8.3} ns/op  {:>8.1} M/s  [cs={:016x}]",
        name, ns_per, mops, checksum
    );
}

// ---------------------------------------------------------------------------
// F8 (GF(2^8), AES poly)
// ---------------------------------------------------------------------------

fn bench_f8() {
    header("F8 (GF(2^8), AES poly)");

    // Latency: dependent chain.
    {
        let b = F8(0xAD);
        let mut a = F8(0xDE);
        let t0 = Instant::now();
        for _ in 0..N {
            a *= b;
        }
        let t = t0.elapsed().as_nanos() as f64;
        report("F8 mul (latency)", t, N, a.0 as u64);
    }

    // Throughput: 4 independent accumulators.
    {
        let b = F8(0xAD);
        let mut a0 = F8(0xDE);
        let mut a1 = F8(0xC0);
        let mut a2 = F8(0xFE);
        let mut a3 = F8(0xBA);
        let iters = N / 4;
        let t0 = Instant::now();
        for _ in 0..iters {
            a0 *= b;
            a1 *= b;
            a2 *= b;
            a3 *= b;
        }
        let t = t0.elapsed().as_nanos() as f64;
        let cs = (a0 + a1 + a2 + a3).0 as u64;
        report("F8 mul (throughput, 4× ILP)", t, iters * 4, cs);
    }

    // Inverse: rarely used; included for completeness.
    {
        let mut a = F8(0x53);
        let n_inv = 1_000_000;
        let t0 = Instant::now();
        for _ in 0..n_inv {
            a = a.inv() + F8::ONE; // +ONE to keep nonzero so inv is defined
        }
        let t = t0.elapsed().as_nanos() as f64;
        report("F8 inv (latency)", t, n_inv, a.0 as u64);
    }
}

// 16-wide NEON F8 kernel (`gf8_mul_vec16`). Reported per element so it lines up
// against the scalar `F8 mul` numbers above.
#[cfg(target_arch = "aarch64")]
fn bench_f8_vec16() {
    use core::arch::aarch64::*;
    use flock_prover::field::gf2_8::neon::gf8_mul_vec16;

    header("F8 vec16 (GF(2^8), 16-wide NEON kernel)");
    unsafe {
        let b = vld1q_u8([0xADu8; 16].as_ptr());
        // Throughput: 4 independent 16-lane accumulators (≡ 64 elements in flight).
        let mut a0 = vld1q_u8([0xDEu8; 16].as_ptr());
        let mut a1 = vld1q_u8([0xC0u8; 16].as_ptr());
        let mut a2 = vld1q_u8([0xFEu8; 16].as_ptr());
        let mut a3 = vld1q_u8([0xBAu8; 16].as_ptr());
        let iters = N / 4;
        let t0 = Instant::now();
        for _ in 0..iters {
            a0 = gf8_mul_vec16(a0, b);
            a1 = gf8_mul_vec16(a1, b);
            a2 = gf8_mul_vec16(a2, b);
            a3 = gf8_mul_vec16(a3, b);
        }
        let t = t0.elapsed().as_nanos() as f64;
        let acc = veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3));
        let mut out = [0u8; 16];
        vst1q_u8(out.as_mut_ptr(), acc);
        // 16 elements per call → per-element cost.
        report(
            "F8 vec16 mul (throughput, per element)",
            t,
            iters * 4 * 16,
            out[0] as u64,
        );
    }
}

// ---------------------------------------------------------------------------
// F128 (GHASH form)
// ---------------------------------------------------------------------------

fn bench_f128_add() {
    header("F128 add (XOR)");
    // F128 add is a pair of u64 XORs. With `b` loop-invariant, naive code lets
    // the optimizer collapse `a XOR b XOR b XOR …` to identity (`b ^ b = 0`).
    // We wrap `a` in `black_box` each iter so the compiler can't trace the
    // value across iterations — every XOR is forced to actually execute.
    // `black_box` is a compile-time barrier with no runtime cost.

    // Latency: dependent chain, black-boxed so XOR can't fold to identity.
    {
        let b = F128 {
            lo: 0xFEDCBA9876543210,
            hi: 0xA5A5A5A5A5A5A5A5,
        };
        let mut a = F128 {
            lo: 0xDEADBEEFCAFEBABE,
            hi: 0x0123456789ABCDEF,
        };
        let t0 = Instant::now();
        for _ in 0..N {
            a = black_box(a) + b;
        }
        let t = t0.elapsed().as_nanos() as f64;
        report("F128 add (latency)", t, N, a.lo);
    }

    // Throughput: 4 independent accumulators, each black-boxed.
    {
        let b = F128 {
            lo: 0xFEDCBA9876543210,
            hi: 0xA5A5A5A5A5A5A5A5,
        };
        let mut a0 = F128 {
            lo: 0x1111111111111111,
            hi: 0x2222222222222222,
        };
        let mut a1 = F128 {
            lo: 0x3333333333333333,
            hi: 0x4444444444444444,
        };
        let mut a2 = F128 {
            lo: 0x5555555555555555,
            hi: 0x6666666666666666,
        };
        let mut a3 = F128 {
            lo: 0x7777777777777777,
            hi: 0x8888888888888888,
        };
        let iters = N / 4;
        let t0 = Instant::now();
        for _ in 0..iters {
            a0 = black_box(a0) + b;
            a1 = black_box(a1) + b;
            a2 = black_box(a2) + b;
            a3 = black_box(a3) + b;
        }
        let t = t0.elapsed().as_nanos() as f64;
        let cs = (a0 + a1 + a2 + a3).lo;
        report("F128 add (throughput, 4× ILP)", t, iters * 4, cs);
    }
}

// One macro per latency / throughput pattern so we can swap the mul function
// without rewriting the surrounding boilerplate. unsafe wrappers are inlined.
macro_rules! bench_mul_latency {
    ($label:expr, $op:expr) => {{
        let b = F128 {
            lo: 0xFEDCBA9876543210,
            hi: 0xA5A5A5A5A5A5A5A5,
        };
        let mut a = F128 {
            lo: 0xDEADBEEFCAFEBABE,
            hi: 0x0123456789ABCDEF,
        };
        let t0 = Instant::now();
        for _ in 0..N {
            a = $op(a, b);
        }
        let t = t0.elapsed().as_nanos() as f64;
        report($label, t, N, a.lo);
    }};
}

macro_rules! bench_mul_throughput {
    ($label:expr, $op:expr) => {{
        let b = F128 {
            lo: 0xFEDCBA9876543210,
            hi: 0xA5A5A5A5A5A5A5A5,
        };
        let mut a0 = F128 {
            lo: 0xDEADBEEFCAFEBABE,
            hi: 0x0123456789ABCDEF,
        };
        let mut a1 = F128 {
            lo: 0x1111111111111111,
            hi: 0x2222222222222222,
        };
        let mut a2 = F128 {
            lo: 0x3333333333333333,
            hi: 0x4444444444444444,
        };
        let mut a3 = F128 {
            lo: 0x5555555555555555,
            hi: 0x6666666666666666,
        };
        let iters = N / 4;
        let t0 = Instant::now();
        for _ in 0..iters {
            a0 = $op(a0, b);
            a1 = $op(a1, b);
            a2 = $op(a2, b);
            a3 = $op(a3, b);
        }
        let t = t0.elapsed().as_nanos() as f64;
        let cs = (a0 + a1 + a2 + a3).lo;
        report($label, t, iters * 4, cs);
    }};
}

fn bench_f128_mul() {
    header("F128 mul — all PMULL variants (aarch64 + aes path)");

    // The default `Mul` impl dispatches to `aarch64::ghash_mul_binius` on
    // M-series with aes enabled, but we benchmark all four variants directly
    // so we can compare against the C++ measurements.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: aes target feature is enabled at compile time.
        bench_mul_latency!("schoolbook (latency)", |a, b| unsafe {
            aarch64::ghash_mul_schoolbook(a, b)
        });
        bench_mul_throughput!("schoolbook (throughput)", |a, b| unsafe {
            aarch64::ghash_mul_schoolbook(a, b)
        });
        bench_mul_latency!("karatsuba (latency)", |a, b| unsafe {
            aarch64::ghash_mul_karatsuba(a, b)
        });
        bench_mul_throughput!("karatsuba (throughput)", |a, b| unsafe {
            aarch64::ghash_mul_karatsuba(a, b)
        });
        bench_mul_latency!("karatsuba+barrett (latency)", |a, b| unsafe {
            aarch64::ghash_mul_karatsuba_barrett(a, b)
        });
        bench_mul_throughput!("karatsuba+barrett (throughput)", |a, b| unsafe {
            aarch64::ghash_mul_karatsuba_barrett(a, b)
        });
        bench_mul_latency!("binius (latency)  ← default Mul impl", |a, b| unsafe {
            aarch64::ghash_mul_binius(a, b)
        });
        bench_mul_throughput!("binius (throughput)", |a, b| unsafe {
            aarch64::ghash_mul_binius(a, b)
        });
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        eprintln!("  (aarch64+aes path disabled — software fallback only)");
    }

    // Software fallback (bit-loop) — for reference / cross-platform sanity.
    bench_mul_latency!("software clmul64 (latency)", software::ghash_mul);
    bench_mul_throughput!("software clmul64 (throughput)", software::ghash_mul);

    // The public Mul trait — should match `binius` exactly.
    bench_mul_latency!("F128::mul (latency)", |a: F128, b: F128| a * b);
    bench_mul_throughput!("F128::mul (throughput)", |a: F128, b: F128| a * b);
}

fn bench_f128_mul_by_x() {
    header("F128 mul_by_x (shift + conditional fold)");
    use flock_prover::field::mul_by_x;

    // Latency
    {
        let mut a = F128 {
            lo: 0xDEADBEEFCAFEBABE,
            hi: 0x0123456789ABCDEF,
        };
        let t0 = Instant::now();
        for _ in 0..N {
            a = mul_by_x(a);
        }
        let t = t0.elapsed().as_nanos() as f64;
        report("mul_by_x (latency)", t, N, a.lo);
    }
    // Throughput
    {
        let mut a0 = F128 {
            lo: 0x1111,
            hi: 0x2222,
        };
        let mut a1 = F128 {
            lo: 0x3333,
            hi: 0x4444,
        };
        let mut a2 = F128 {
            lo: 0x5555,
            hi: 0x6666,
        };
        let mut a3 = F128 {
            lo: 0x7777,
            hi: 0x8888,
        };
        let iters = N / 4;
        let t0 = Instant::now();
        for _ in 0..iters {
            a0 = mul_by_x(a0);
            a1 = mul_by_x(a1);
            a2 = mul_by_x(a2);
            a3 = mul_by_x(a3);
        }
        let t = t0.elapsed().as_nanos() as f64;
        let cs = (a0 + a1 + a2 + a3).lo;
        report("mul_by_x (throughput, 4× ILP)", t, iters * 4, cs);
    }
}

fn bench_f128_deferred() {
    header("F128 deferred-reduction sumcheck pattern");
    // The realistic sumcheck pattern: accumulate K unreduced products into a
    // single F256Unreduced, then call .reduce() once. We measure cost per
    // (mul_unreduced + xor accumulate), with the final reduce amortized.

    let b = F128 {
        lo: 0xFEDCBA9876543210,
        hi: 0xA5A5A5A5A5A5A5A5,
    };
    let mut a0 = F128 {
        lo: 0x1111111111111111,
        hi: 0x2222222222222222,
    };
    let mut a1 = F128 {
        lo: 0x3333333333333333,
        hi: 0x4444444444444444,
    };
    let mut a2 = F128 {
        lo: 0x5555555555555555,
        hi: 0x6666666666666666,
    };
    let mut a3 = F128 {
        lo: 0x7777777777777777,
        hi: 0x8888888888888888,
    };

    let iters = N / 4;
    let t0 = Instant::now();
    let mut acc0 = F256Unreduced::ZERO;
    let mut acc1 = F256Unreduced::ZERO;
    let mut acc2 = F256Unreduced::ZERO;
    let mut acc3 = F256Unreduced::ZERO;
    // Vary BOTH `lo` and `hi` of each `a` every iter. If only `lo` changes,
    // the compiler hoists PMULL(a.hi, b.*) out of the loop, halving the work.
    const STEP_LO: u64 = 1;
    const STEP_HI: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..iters {
        acc0 ^= a0.mul_unreduced(b);
        acc1 ^= a1.mul_unreduced(b);
        acc2 ^= a2.mul_unreduced(b);
        acc3 ^= a3.mul_unreduced(b);
        a0 = F128 {
            lo: a0.lo.wrapping_add(STEP_LO),
            hi: a0.hi.wrapping_add(STEP_HI),
        };
        a1 = F128 {
            lo: a1.lo.wrapping_add(STEP_LO),
            hi: a1.hi.wrapping_add(STEP_HI),
        };
        a2 = F128 {
            lo: a2.lo.wrapping_add(STEP_LO),
            hi: a2.hi.wrapping_add(STEP_HI),
        };
        a3 = F128 {
            lo: a3.lo.wrapping_add(STEP_LO),
            hi: a3.hi.wrapping_add(STEP_HI),
        };
    }
    let final_red = (acc0 ^ acc1 ^ acc2 ^ acc3).reduce();
    let t = t0.elapsed().as_nanos() as f64;
    let cs = final_red.lo ^ black_box(a0.lo) ^ a1.lo ^ a2.lo ^ a3.lo;
    report(
        "mul_unreduced + ^= acc (throughput, 4× ILP)",
        t,
        iters * 4,
        cs,
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    // Quick build-config sanity print so the reader knows which path is hot.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — PMULL path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: software fallback path — NEON/PMULL disabled)");

    println!("(N = {} iterations per measurement)", N);
    println!("Reference (M-series single-thread, prior C++ baseline):");
    println!("  F8 mul scalar (latency)        ~6.1  ns/op");
    println!("  F128 GHASH add (throughput)    ~0.06 ns/op");
    println!("  F128 GHASH mul binius64 (lat)  ~5.7  ns/op");

    bench_f8();
    #[cfg(target_arch = "aarch64")]
    bench_f8_vec16();
    bench_f128_add();
    bench_f128_mul();
    bench_f128_mul_by_x();
    bench_f128_deferred();
}
