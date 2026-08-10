//! End-to-end proof benchmark for the u64×u64 batch-multiplication R1CS in
//! `src/r1cs_hashes/mul64.rs` (K_LOG=17, 15 products per block, 94.5% fill).
//! Uses `Mul64Setup::prove_fast` — the fused (z, a, b, c, z_lincheck)
//! generator with bit-sliced (64-wide) witness evaluation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::mul64::{
    K, K_LOG, MULS_PER_BLOCK, Mul64Setup, SUB_BITS, USEFUL_BITS, circuit, min_n_blocks_log,
};

// Peak-heap tracker (wraps System) — same notion as the sha2/keccak benches.
struct PeakAlloc;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                let c = CUR.fetch_add(new - l.size(), Ordering::Relaxed) + (new - l.size());
                PEAK.fetch_max(c, Ordering::Relaxed);
            } else {
                CUR.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}
#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;
fn reset_peak() {
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak_mb() -> f64 {
    PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
}

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
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1.0 {
        format!("{:>8.2} µs", s * 1e6)
    } else if ms < 1000.0 {
        format!("{:>8.2} ms", ms)
    } else {
        format!("{:>8.2} s ", s)
    }
}

fn bench_one(n_muls: usize, n_runs: usize) {
    let n_log = min_n_blocks_log(n_muls);
    let m = K_LOG + n_log;
    let capacity = MULS_PER_BLOCK << n_log;
    let witness_bytes = (1usize << m) / 8;
    let total_useful_bits = (n_muls * SUB_BITS) as f64;
    let total_z_bits = (1u64 << m) as f64;
    let padding_pct = 100.0 * (1.0 - total_useful_bits / total_z_bits);

    println!(
        "\n=== {n_muls:>6} muls  (m = {m}, slots = {n_muls}/{capacity}, padding = \
         {padding_pct:.1}%, witness = {} MB) ===",
        witness_bytes >> 20
    );

    let setup = Mul64Setup::new(n_muls);
    let mk_inputs = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_muls)
            .map(|_| (rng.next_u64(), rng.next_u64()))
            .collect::<Vec<(u64, u64)>>()
    };
    let input_sets: Vec<Vec<(u64, u64)>> = (0..=n_runs)
        .map(|run| mk_inputs(0x9A6_C0FFEE ^ (n_muls as u64) ^ (run as u64)))
        .collect();

    // Warm-up.
    {
        let mut ch = FsChallenger::new(b"flock-mul64-bench-v0");
        let (p, _, _) = setup.prove_fast(&input_sets[0], &mut ch);
        black_box(&p);
    }

    let mut best = f64::INFINITY;
    for run in 0..n_runs {
        let inputs = &input_sets[run + 1];
        let mut ch = FsChallenger::new(b"flock-mul64-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(inputs, &mut ch);
        let elapsed = t0.elapsed().as_secs_f64();
        best = best.min(elapsed);
        black_box(&p);
        println!(
            "  [run {}/{}] prove_fast: {}",
            run + 1,
            n_runs,
            fmt_ms(elapsed)
        );
    }
    let muls_per_sec = (n_muls as f64) / best;
    println!(
        "  best prove_fast: {}   ({:.0} muls/sec, {:.2} µs/mul)",
        fmt_ms(best),
        muls_per_sec,
        1e6 * best / n_muls as f64
    );

    // Peak memory + verify + proof size.
    {
        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-mul64-bench-v0");
        let (proof, commitment, _) = setup.prove_fast(&input_sets[0], &mut ch_p);
        println!("  peak memory: {:>8.2} MB", peak_mb());

        let mut ch_v = FsChallenger::new(b"flock-mul64-bench-v0");
        let t = Instant::now();
        let _ = setup
            .verify(&commitment, &proof, &mut ch_v)
            .expect("verify failed");
        println!("  verify: {}", fmt_ms(t.elapsed().as_secs_f64()));

        let bundle = flock_prover::proof_io::R1csProofBundleLigerito { commitment, proof };
        let proof_size = bundle.to_bytes().len();
        println!(
            "  proof size: {} bytes ({:.2} KiB)",
            proof_size,
            proof_size as f64 / 1024.0
        );
        black_box(&bundle);
    }

    // Per-phase breakdown of the real Ligerito prover.
    println!("  [prove_fast breakdown]");
    let mut ch = FsChallenger::new(b"flock-mul64-bench-v0");
    let (proof, _commitment, _claim, tm) = setup.prove_fast_timed(&input_sets[0], &mut ch);
    println!(
        "    {:32} {}",
        "gen_witness_ab + lincheck",
        fmt_ms(tm.witness_s)
    );
    println!("    {:32} {}", "pcs::commit", fmt_ms(tm.commit_s));
    println!(
        "    {:32} {}",
        "zerocheck::prove_packed",
        fmt_ms(tm.zerocheck_s)
    );
    println!("    {:32} {}", "lincheck::prove", fmt_ms(tm.lincheck_s));
    println!("    {:32} {}", "pcs::open (ligerito)", fmt_ms(tm.open_s));
    black_box(&proof);
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let c = circuit();
    println!(
        "u64×u64 batch multiplication R1CS (K_LOG={K_LOG}, {MULS_PER_BLOCK} products/block, \
         {USEFUL_BITS}/{K} = {:.1}% fill).\n\
         Per product: {SUB_BITS} committed bits = 128 input + 4096 partial-product + {} adder \
         AND bits; A₀/B₀ nnz = {}/product.",
        100.0 * USEFUL_BITS as f64 / K as f64,
        c.n_adders,
        c.nnz,
    );

    // Sizes to bench; override with MUL64_NS (values ≤ 30 are log2 product
    // counts, larger values are raw counts — 15·2^k counts give 100% slot
    // fill).
    let specs: Vec<(usize, usize)> = match std::env::var("MUL64_NS") {
        Ok(s) => s
            .split([',', ' '])
            .filter(|t| !t.is_empty())
            .map(|t| {
                let v: usize = t
                    .parse()
                    .expect("MUL64_NS: space/comma-separated counts (≤ 30 ⇒ log2)");
                (if v <= 30 { 1usize << v } else { v }, 3usize)
            })
            .collect(),
        Err(_) => vec![(480usize, 2), (1920, 2), (7680, 3), (30720, 3), (122880, 3)],
    };
    for &(n, n_runs) in &specs {
        bench_one(n, n_runs);
    }
}
