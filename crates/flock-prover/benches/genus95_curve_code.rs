//! Microbenchmarks for the genus-95 product-code hot path (no criterion).
//!
//! The four operations the module exists to provide:
//!   * `product_code_message`            — multiply two base-code messages.
//!   * `sample_random_evaluation_point`  — random point on the cover.
//!   * `product_evaluation_functional`   — build the 222-coord eval functional.
//!   * `evaluate_product_functional`     — evaluate a message at that functional.
//!
//! Plus two field micros (`F128` mul/square) so the per-op field cost — now the
//! crate-wide GHASH `F128`, shared with the rest of flock — is visible next to
//! the code-level operations that are built on top of it.
//!
//! Each measurement runs for a fixed wall-clock window and reports throughput.
//! A checksum is printed so the optimizer can't elide the loop body.
//!
//! Run:  `cargo bench --bench genus95_curve_code`

use std::hint::black_box;
use std::time::{Duration, Instant};

use flock_prover::field::F128;
use flock_prover::genus95_curve_code::RngCore;
use flock_prover::genus95_curve_code::{
    BaseMessage, ProductMessage, Sha256Rng, base_evaluation_functional, evaluate_base_functional,
    evaluate_product_functional, product_code_message, product_evaluation_functional,
    sample_random_evaluation_point,
};

const WINDOW: Duration = Duration::from_millis(750);

fn report(name: &str, iters: u64, elapsed: Duration, checksum: u64) {
    let per_second = iters as f64 / elapsed.as_secs_f64();
    let ns_per = elapsed.as_secs_f64() * 1e9 / iters as f64;
    println!(
        "  {:<34} {:>10.2} M/s  {:>10.2} ns/op  ({} iters)  [cs={:016x}]",
        name,
        per_second / 1e6,
        ns_per,
        iters,
        checksum
    );
}

fn rand_f128(rng: &mut impl RngCore) -> F128 {
    F128::new(rng.next_u64(), rng.next_u64())
}

fn bench_product(rng: &mut impl RngCore) {
    let inputs: Vec<_> = (0..4096)
        .map(|_| (BaseMessage::random(rng), BaseMessage::random(rng)))
        .collect();

    // Amortize the per-iteration `elapsed()` timer cost over a chunk, else the
    // syscall dominates and dilutes the measured per-call cost.
    const CHUNK: u64 = 8192;
    let start = Instant::now();
    let mut iters = 0u64;
    let mut acc = 0u64;
    while start.elapsed() < WINDOW {
        for k in 0..CHUNK {
            let (left, right) = inputs[((iters + k) as usize) & (inputs.len() - 1)];
            let product = black_box(product_code_message(black_box(left), black_box(right)));
            acc ^= product.limbs[0];
        }
        iters += CHUNK;
    }
    report("product_code_message", iters, start.elapsed(), acc);
}

fn bench_sample_point(rng: &mut impl RngCore) {
    let start = Instant::now();
    let mut iters = 0u64;
    let mut acc = 0u64;
    while start.elapsed() < WINDOW {
        let point = black_box(sample_random_evaluation_point(rng).expect("sample point"));
        acc ^= point.x.lo;
        iters += 1;
    }
    report(
        "sample_random_evaluation_point",
        iters,
        start.elapsed(),
        acc,
    );
}

fn bench_functional(rng: &mut impl RngCore) {
    let points: Vec<_> = (0..16)
        .map(|_| sample_random_evaluation_point(rng).expect("sample point"))
        .collect();

    let start = Instant::now();
    let mut iters = 0u64;
    let mut acc = 0u64;
    while start.elapsed() < WINDOW {
        let point = &points[(iters as usize) & 15];
        let functional =
            black_box(product_evaluation_functional(black_box(point)).expect("functional"));
        acc ^= functional[(iters as usize) % functional.len()].lo;
        iters += 1;
    }
    report("product_evaluation_functional", iters, start.elapsed(), acc);
}

fn bench_evaluate(rng: &mut impl RngCore) {
    let point = sample_random_evaluation_point(rng).expect("sample point");
    let functional = product_evaluation_functional(&point).expect("functional");
    let messages: Vec<_> = (0..4096).map(|_| ProductMessage::random(rng)).collect();

    const CHUNK: u64 = 8192;
    let start = Instant::now();
    let mut iters = 0u64;
    let mut acc = 0u64;
    while start.elapsed() < WINDOW {
        for k in 0..CHUNK {
            let message = messages[((iters + k) as usize) & (messages.len() - 1)];
            let value = black_box(evaluate_product_functional(
                black_box(&functional),
                black_box(&message),
            ));
            acc ^= value.lo;
        }
        iters += CHUNK;
    }
    report("evaluate_product_functional", iters, start.elapsed(), acc);
}

fn bench_base_functional(rng: &mut impl RngCore) {
    let points: Vec<_> = (0..16)
        .map(|_| sample_random_evaluation_point(rng).expect("sample point"))
        .collect();

    let start = Instant::now();
    let mut iters = 0u64;
    let mut acc = 0u64;
    while start.elapsed() < WINDOW {
        let point = &points[(iters as usize) & 15];
        let functional =
            black_box(base_evaluation_functional(black_box(point)).expect("base functional"));
        acc ^= functional[(iters as usize) % functional.len()].lo;
        iters += 1;
    }
    report("base_evaluation_functional", iters, start.elapsed(), acc);
}

fn bench_base_evaluate(rng: &mut impl RngCore) {
    let point = sample_random_evaluation_point(rng).expect("sample point");
    let functional = base_evaluation_functional(&point).expect("base functional");
    let messages: Vec<_> = (0..4096).map(|_| BaseMessage::random(rng)).collect();

    const CHUNK: u64 = 8192;
    let start = Instant::now();
    let mut iters = 0u64;
    let mut acc = 0u64;
    while start.elapsed() < WINDOW {
        for k in 0..CHUNK {
            let message = messages[((iters + k) as usize) & (messages.len() - 1)];
            let value = black_box(evaluate_base_functional(
                black_box(&functional),
                black_box(&message),
            ));
            acc ^= value.lo;
        }
        iters += CHUNK;
    }
    report("evaluate_base_functional", iters, start.elapsed(), acc);
}

fn bench_field(rng: &mut impl RngCore) {
    // Throughput: 4 independent lanes expose instruction-level parallelism, so
    // the measurement reflects how many ops/sec the PMULL/CLMUL units sustain —
    // the regime the sampling and functional-build paths actually run in. This
    // is the crate-wide GHASH `F128`; cross-check `cargo bench --bench field`.
    let b = rand_f128(rng);

    // Inner chunk amortizes the `elapsed()` timer cost so it doesn't dominate.
    const CHUNK: u64 = 8192;

    let mut a = [
        rand_f128(rng),
        rand_f128(rng),
        rand_f128(rng),
        rand_f128(rng),
    ];
    let start = Instant::now();
    let mut iters = 0u64;
    while start.elapsed() < WINDOW {
        for _ in 0..CHUNK / 4 {
            for lane in &mut a {
                *lane *= b;
            }
        }
        iters += CHUNK;
    }
    report(
        "F128 mul (shared GHASH, throughput)",
        iters,
        start.elapsed(),
        a[0].lo,
    );

    let mut a = [
        rand_f128(rng),
        rand_f128(rng),
        rand_f128(rng),
        rand_f128(rng),
    ];
    let start = Instant::now();
    let mut iters = 0u64;
    while start.elapsed() < WINDOW {
        for _ in 0..CHUNK / 4 {
            for lane in &mut a {
                *lane = lane.square();
            }
        }
        iters += CHUNK;
    }
    report(
        "F128 square (dedicated 2-CLMUL, tput)",
        iters,
        start.elapsed(),
        a[0].lo,
    );
}

fn main() {
    let mut rng = Sha256Rng::seed_from_u64(0x1234_5678_9abc_def0);

    // Warm the lazily-built tables so setup cost isn't charged to the first bench.
    let point = sample_random_evaluation_point(&mut rng).expect("initial sample point");
    let functional = product_evaluation_functional(&point).expect("initial functional");
    let message = ProductMessage::random(&mut rng);
    black_box(evaluate_product_functional(&functional, &message));

    println!("genus95_curve_code microbenchmarks");
    bench_product(&mut rng);
    bench_sample_point(&mut rng);
    bench_functional(&mut rng);
    bench_evaluate(&mut rng);
    bench_base_functional(&mut rng);
    bench_base_evaluate(&mut rng);
    bench_field(&mut rng);
}
