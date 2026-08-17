//! BLAKE3 `prove_fast` (RS additive-NTT zerocheck) vs `prove_fast_ag` (genus-95
//! AG-code zerocheck) head-to-head — both on the standard pack + Ligerito PCS.
//! The ONLY difference is round 1 of the zerocheck; everything else (witness gen,
//! commit, lincheck, ring-switch open) is shared, so this isolates the AG-skip
//! zerocheck win at the full-proof level.
//!
//! Run twice for ST and MT:
//!   cargo bench --bench blake3_rs_vs_ag                       # MT (default)
//!   RAYON_NUM_THREADS=1 cargo bench --bench blake3_rs_vs_ag   # ST
//!
//! Defaults to m=30 (K=65536 compressions). Override with `BLAKE3_K=<n_blocks>`
//! (comma/space-separated for multiple). aarch64-only (the AG round-1 kernel is
//! NEON); on other targets only the RS side is reported.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression, K_LOG, min_n_blocks_log};

// Peak-heap tracker (mirrors blake3_lig_vs_bf.rs).
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
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
}

fn random_compression(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
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

fn fmt_kb(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

fn bench_block(n_blocks: usize, n_runs: usize, threads_label: &str) {
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;
    let witness_bytes = (1usize << m) / 8;

    println!(
        "\n=== K = {n_blocks:>6}  (m = {m}, slots = {n_slots}, witness = {} MB, {threads_label}) ===",
        witness_bytes >> 20
    );

    let setup = Blake3Setup::new(n_blocks);
    let mk_blocks = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_blocks)
            .map(|_| random_compression(&mut rng))
            .collect::<Vec<Compression>>()
    };
    let block_sets: Vec<Vec<Compression>> = (0..=n_runs)
        .map(|run| mk_blocks(0xB1A_C0FFEE ^ (n_blocks as u64) ^ (run as u64)))
        .collect();

    // Global warm-up: prime the shared scratch pool + codeword path BEFORE timing
    // either backend, so neither eats the one-time cold-codeword page-fault (which
    // would otherwise be charged to whichever runs first). See the ag-skip memory
    // note's "warm the scratch pool before A/B-ing prove time at scale" gotcha.
    {
        let mut ch = FsChallenger::new(b"flock-bench-warm");
        let (p, _, _) = setup.prove_fast(&block_sets[0], &mut ch);
        black_box(&p);
    }

    let mut rs_prove = f64::INFINITY;

    // ============ RS (additive-NTT zerocheck) + Ligerito ============
    {
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&block_sets[0], &mut ch_p);
        black_box(&p);

        for run in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-bench-v0");
            let t0 = Instant::now();
            let (p, _, _) = setup.prove_fast(&block_sets[run + 1], &mut ch_p);
            rs_prove = rs_prove.min(t0.elapsed().as_secs_f64());
            black_box(&p);
        }

        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (proof, commitment, _) = setup.prove_fast(&block_sets[0], &mut ch_p);
        let peak = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        setup
            .verify(&commitment, &proof, &mut ch_v)
            .expect("RS verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let size = bincode::serialize(&proof).expect("ser RS proof").len();
        black_box(&proof);

        println!(
            "  RS  zerocheck: prove = {}   verify = {}   proof = {}   peak = {:.1} MB",
            fmt_ms(rs_prove),
            fmt_ms(verify_t),
            fmt_kb(size),
            peak,
        );
    }

    // ============ AG (genus-95 multiplication-code zerocheck) + Ligerito ========
    #[cfg(target_arch = "aarch64")]
    {
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast_ag(&block_sets[0], &mut ch_p);
        black_box(&p);

        let mut ag_prove = f64::INFINITY;
        for run in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-bench-v0");
            let t0 = Instant::now();
            let (p, _, _) = setup.prove_fast_ag(&block_sets[run + 1], &mut ch_p);
            ag_prove = ag_prove.min(t0.elapsed().as_secs_f64());
            black_box(&p);
        }

        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (proof, commitment, _) = setup.prove_fast_ag(&block_sets[0], &mut ch_p);
        let peak = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        setup
            .verify_ag(&commitment, &proof, &mut ch_v)
            .expect("AG verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let size = bincode::serialize(&proof).expect("ser AG proof").len();
        black_box(&proof);

        println!(
            "  AG  zerocheck: prove = {}   verify = {}   proof = {}   peak = {:.1} MB",
            fmt_ms(ag_prove),
            fmt_ms(verify_t),
            fmt_kb(size),
            peak,
        );
        println!(
            "  ──> AG prove speedup vs RS: {:.3}×  ({:+.2} ms)",
            rs_prove / ag_prove,
            (rs_prove - ag_prove) * 1000.0,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!("  AG  zerocheck: (skipped — aarch64-only NEON kernel)");
}

/// Per-phase prove breakdown (witness / commit / zerocheck / lincheck / open)
/// for RS vs AG. Best-of-N by total prove time; reports that run's phases (so
/// the columns sum to the reported total). Triggered by BLAKE3_BREAKDOWN=1.
#[cfg(target_arch = "aarch64")]
fn bench_breakdown(n_blocks: usize, n_runs: usize, threads_label: &str) {
    use flock_prover::prover::ProvePhaseTimings;
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    println!("\n=== BREAKDOWN  K = {n_blocks}  (m = {m}, {threads_label}) ===");

    let setup = Blake3Setup::new(n_blocks);
    let mk = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_blocks)
            .map(|_| random_compression(&mut rng))
            .collect::<Vec<Compression>>()
    };
    let sets: Vec<Vec<Compression>> = (0..=n_runs)
        .map(|r| mk(0xB1A_C0FFEE ^ (n_blocks as u64) ^ (r as u64)))
        .collect();

    // Warm the shared scratch pool / codeword path before timing either side.
    {
        let mut ch = FsChallenger::new(b"flock-bd-warm");
        let (p, _, _, _) = setup.prove_fast_timed(&sets[0], &mut ch);
        black_box(&p);
    }

    let total =
        |t: &ProvePhaseTimings| t.witness_s + t.commit_s + t.zerocheck_s + t.lincheck_s + t.open_s;

    let mut rs_best = ProvePhaseTimings::default();
    let mut rs_min = f64::INFINITY;
    for run in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bd-v0");
        let (p, _, _, t) = setup.prove_fast_timed(&sets[run + 1], &mut ch);
        black_box(&p);
        if total(&t) < rs_min {
            rs_min = total(&t);
            rs_best = t;
        }
    }

    let mut ag_best = ProvePhaseTimings::default();
    let mut ag_min = f64::INFINITY;
    for run in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bd-v0");
        let (p, _, _, t) = setup.prove_fast_ag_timed(&sets[run + 1], &mut ch);
        black_box(&p);
        if total(&t) < ag_min {
            ag_min = total(&t);
            ag_best = t;
        }
    }

    let row = |name: &str, rs: f64, ag: f64| {
        println!(
            "  {:<10} RS {:>9}   AG {:>9}   Δ {:+8.2} ms",
            name,
            fmt_ms(rs),
            fmt_ms(ag),
            (rs - ag) * 1000.0
        );
    };
    row("witness", rs_best.witness_s, ag_best.witness_s);
    row("commit", rs_best.commit_s, ag_best.commit_s);
    row("zerocheck", rs_best.zerocheck_s, ag_best.zerocheck_s);
    row("lincheck", rs_best.lincheck_s, ag_best.lincheck_s);
    row("open", rs_best.open_s, ag_best.open_s);
    row("TOTAL", total(&rs_best), total(&ag_best));
    println!(
        "  ──> AG total speedup {:.3}×; the win is the zerocheck: {:+.2} ms",
        total(&rs_best) / total(&ag_best),
        (rs_best.zerocheck_s - ag_best.zerocheck_s) * 1000.0,
    );
}

/// Within-process A/B of the AG zerocheck: friendly-Horner tail (rounds 1..=5)
/// vs the general kernel, interleaved per run so thermal drift cancels. The two
/// paths are bit-identical; this isolates the friendly-Horner perf delta from
/// the (larger) cross-process noise floor. Triggered by AG_FRIENDLY_AB=1.
#[cfg(target_arch = "aarch64")]
fn bench_friendly_ab(n_blocks: usize, n_runs: usize, threads_label: &str) {
    use flock_prover::zerocheck::ag_skip::DISABLE_FRIENDLY_HORNER;
    use std::sync::atomic::Ordering;
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    println!(
        "\n=== FRIENDLY A/B  K = {n_blocks}  (m = {m}, {threads_label}, best of {n_runs}) ==="
    );

    let setup = Blake3Setup::new(n_blocks);
    let mk = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_blocks)
            .map(|_| random_compression(&mut rng))
            .collect::<Vec<Compression>>()
    };
    let blocks = mk(0xB1A_C0FFEE ^ (n_blocks as u64));

    // Warm scratch pool / cold codeword once (charged to neither path).
    {
        let mut ch = FsChallenger::new(b"flock-ab-warm");
        let (p, _, _, _) = setup.prove_fast_ag_timed(&blocks, &mut ch);
        black_box(&p);
    }

    let (mut fr_min, mut gen_min) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..n_runs {
        // Interleave friendly then general each run so they share thermal state.
        DISABLE_FRIENDLY_HORNER.store(false, Ordering::Relaxed);
        let mut ch = FsChallenger::new(b"flock-ab-v0");
        let (p, _, _, t) = setup.prove_fast_ag_timed(&blocks, &mut ch);
        black_box(&p);
        fr_min = fr_min.min(t.zerocheck_s);

        DISABLE_FRIENDLY_HORNER.store(true, Ordering::Relaxed);
        let mut ch = FsChallenger::new(b"flock-ab-v0");
        let (p, _, _, t) = setup.prove_fast_ag_timed(&blocks, &mut ch);
        black_box(&p);
        gen_min = gen_min.min(t.zerocheck_s);
    }
    DISABLE_FRIENDLY_HORNER.store(false, Ordering::Relaxed);

    println!(
        "  AG zerocheck: general {}   friendly {}   Δ {:+.2} ms  ({:.3}× faster)",
        fmt_ms(gen_min),
        fmt_ms(fr_min),
        (gen_min - fr_min) * 1000.0,
        gen_min / fr_min,
    );
}

/// Within-process A/B of the tail's ping-pong buffer strategy: uninit pooled
/// `take_f128` vs the old `vec![F128::ZERO; n_in/2]` (serial zero-fill, non-pooled),
/// interleaved per run. Output is bit-identical. Triggered by AG_NXT_AB=1.
#[cfg(target_arch = "aarch64")]
fn bench_nxt_ab(n_blocks: usize, n_runs: usize, threads_label: &str) {
    use flock_prover::zerocheck::ag_skip::NXT_ZEROFILL;
    use std::sync::atomic::Ordering;
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    println!(
        "\n=== NXT BUFFER A/B  K = {n_blocks}  (m = {m}, {threads_label}, best of {n_runs}) ==="
    );

    let setup = Blake3Setup::new(n_blocks);
    let mk = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_blocks)
            .map(|_| random_compression(&mut rng))
            .collect::<Vec<Compression>>()
    };
    let blocks = mk(0xB1A_C0FFEE ^ (n_blocks as u64));
    {
        let mut ch = FsChallenger::new(b"flock-nxt-warm");
        let (p, _, _, _) = setup.prove_fast_ag_timed(&blocks, &mut ch);
        black_box(&p);
    }

    let (mut pool_min, mut zero_min) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..n_runs {
        NXT_ZEROFILL.store(false, Ordering::Relaxed);
        let mut ch = FsChallenger::new(b"flock-nxt-v0");
        let (p, _, _, t) = setup.prove_fast_ag_timed(&blocks, &mut ch);
        black_box(&p);
        pool_min = pool_min.min(t.zerocheck_s);

        NXT_ZEROFILL.store(true, Ordering::Relaxed);
        let mut ch = FsChallenger::new(b"flock-nxt-v0");
        let (p, _, _, t) = setup.prove_fast_ag_timed(&blocks, &mut ch);
        black_box(&p);
        zero_min = zero_min.min(t.zerocheck_s);
    }
    NXT_ZEROFILL.store(false, Ordering::Relaxed);

    println!(
        "  AG zerocheck: vec![ZERO] {}   pooled take_f128 {}   Δ {:+.2} ms  ({:.3}× faster)",
        fmt_ms(zero_min),
        fmt_ms(pool_min),
        (zero_min - pool_min) * 1000.0,
        zero_min / pool_min,
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    let label_owned = if threads == 1 {
        "ST".to_string()
    } else {
        format!("MT, {threads} threads")
    };
    let label = label_owned.as_str();

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("BLAKE3 prove_fast RS-zerocheck vs AG-zerocheck head-to-head — {label}");

    let ks: Vec<usize> = match std::env::var("BLAKE3_K") {
        Ok(s) => s
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().expect("BLAKE3_K: integer K (n_blocks)"))
            .collect(),
        Err(_) => vec![65536],
    };
    let n_runs: usize = std::env::var("FLOCK_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let breakdown = std::env::var("BLAKE3_BREAKDOWN").is_ok();
    let friendly_ab = std::env::var("AG_FRIENDLY_AB").is_ok();
    let nxt_ab = std::env::var("AG_NXT_AB").is_ok();
    for &n in &ks {
        #[cfg(target_arch = "aarch64")]
        if nxt_ab {
            bench_nxt_ab(n, n_runs, label);
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        if friendly_ab {
            bench_friendly_ab(n, n_runs, label);
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        if breakdown {
            bench_breakdown(n, n_runs, label);
            continue;
        }
        let _ = (breakdown, friendly_ab, nxt_ab);
        bench_block(n, n_runs, label);
    }
}
