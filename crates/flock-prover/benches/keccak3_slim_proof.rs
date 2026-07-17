//! 3-wide Keccak-f proof throughput benchmark — SLIM profile (PCS rate 1/4).
//!
//! Identical to `keccak3_proof.rs` except every setup is built with
//! `with_log_inv_rate(n, SLIM_LOG_INV_RATE=2)` (rate 1/4) instead of the default
//! `new(n)` (rate 1/2). The slim profile uses a larger codeword so each query
//! closes more soundness: fewer-but-stronger queries → SMALLER proofs, at the
//! cost of ~2x prover work (lower throughput). Same KECCAK3_KS report format, so
//! benchmarks/bench_keccak.sh can drive it as a drop-in alternative.
//!
//! Times the fast prover path (`prove_fast`) for both the single-keccak encoder
//! (`keccak::KeccakSetup`, K_LOG=16) and the 3-wide encoder (`keccak3::KeccakSetup`,
//! K_LOG=17, three permutations per block) and reports keccaks/s throughput side by side.
//!
//! The 3-wide encoder packs ~97% of each block vs the single encoder's ~65%,
//! so it commits fewer bits per keccak. The win is maximal at counts of the
//! form `3·2^j` (6144, 12288, 24576, …), where the single encoder rounds its
//! outer dimension up to the next power of two but 3-wide lands exactly on a
//! power-of-two block count. At counts already a power of two (e.g. 8192) the
//! two encoders reach the same committed size and throughput is comparable.
//!
//! For single-threaded numbers, run with `RAYON_NUM_THREADS=1` — the perf
//! pool honors it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::keccak::{STATE_BITS, State};
use flock_prover::r1cs_hashes::{keccak, keccak3};

/// Slim profile: PCS log inverse rate 2 (rate 1/4) — smaller proof, slower
/// prover. The only difference from keccak3_proof.rs (which uses the default
/// rate 1/2 via `KeccakSetup::new`).
const SLIM_LOG_INV_RATE: usize = 2;

// Peak-heap tracker (wraps System), identical to keccak_proof's — records the
// high-water mark of outstanding bytes so the KECCAK3_KS report can emit a
// "peak memory:" line for bench_keccak.sh's flock3 row.
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

fn random_state(rng: &mut Rng) -> State {
    let mut s = [false; STATE_BITS];
    let mut i = 0;
    while i < STATE_BITS {
        let w = rng.next_u64();
        for b in 0..64 {
            if i + b < STATE_BITS {
                s[i + b] = (w >> b) & 1 == 1;
            }
        }
        i += 64;
    }
    s
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1000.0 {
        format!("{ms:>8.1} ms")
    } else {
        format!("{:>8.2} s ", s)
    }
}

/// keccaks/s, formatted with a K/M suffix.
fn fmt_thru(n: usize, s: f64) -> String {
    let kps = n as f64 / s;
    if kps >= 1.0e6 {
        format!("{:>7.2} M/s", kps / 1.0e6)
    } else {
        format!("{:>7.2} K/s", kps / 1.0e3)
    }
}

fn bench_pair(n_keccaks: usize, n_runs: usize) {
    let mk_states = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_keccaks)
            .map(|_| random_state(&mut rng))
            .collect::<Vec<State>>()
    };
    // One distinct input set per run (+1 warm-up), shared by both encoders.
    let state_sets: Vec<Vec<State>> = (0..=n_runs)
        .map(|run| mk_states(0xC0FFEE_BEEF ^ (n_keccaks as u64) ^ (run as u64)))
        .collect();

    let s1 = keccak::KeccakSetup::with_log_inv_rate(n_keccaks, SLIM_LOG_INV_RATE);
    let s3 = keccak3::KeccakSetup::with_log_inv_rate(n_keccaks, SLIM_LOG_INV_RATE);

    let bits1 = 1u64 << s1.m();
    let bits3 = 1u64 << s3.m();
    println!(
        "\n=== K = {n_keccaks:>6} Keccaks ===\n  single : m={:>2}  committed = {:>4} MB ({} blk-slots)\n  3-wide : m={:>2}  committed = {:>4} MB ({} blk-slots)",
        s1.m(),
        bits1 >> 23,
        s1.n_keccak_slots(),
        s3.m(),
        bits3 >> 23,
        s3.n_block_slots(),
    );

    // Warm-up both.
    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        black_box(&s1.prove_fast(&state_sets[0], &mut ch).0);
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        black_box(&s3.prove_fast(&state_sets[0], &mut ch).0);
    }

    let mut best1 = f64::INFINITY;
    let mut best3 = f64::INFINITY;
    for run in 0..n_runs {
        let inputs = &state_sets[run + 1];

        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p1, _, _) = s1.prove_fast(inputs, &mut ch);
        let e1 = t0.elapsed().as_secs_f64();
        best1 = best1.min(e1);
        black_box(&p1);

        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p3, _, _) = s3.prove_fast(inputs, &mut ch);
        let e3 = t0.elapsed().as_secs_f64();
        best3 = best3.min(e3);
        black_box(&p3);
    }

    println!(
        "  single : {}   {}",
        fmt_ms(best1),
        fmt_thru(n_keccaks, best1)
    );
    println!(
        "  3-wide : {}   {}",
        fmt_ms(best3),
        fmt_thru(n_keccaks, best3)
    );
    println!("  speedup: {:>6.2}x", best1 / best3);
}

/// Bench ONLY the 3-wide encoder (flock3) at `n_keccaks`, printing the same
/// parseable fields as `keccak_proof` (`=== K = N Keccaks ===`, `best
/// prove_fast:`, `peak memory:`, `verify:`, `proof size:`) so bench_keccak.sh's
/// `run_flock3` can scrape it. Best at counts of the form `3·2^j`, where the
/// 3-wide encoder lands on an exact power-of-two block count.
fn bench_3wide_report(n_keccaks: usize, n_runs: usize) {
    let setup = keccak3::KeccakSetup::with_log_inv_rate(n_keccaks, SLIM_LOG_INV_RATE);
    let mk_states = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_keccaks)
            .map(|_| random_state(&mut rng))
            .collect::<Vec<State>>()
    };
    let state_sets: Vec<Vec<State>> = (0..=n_runs)
        .map(|run| mk_states(0xC0FFEE_BEEF ^ (n_keccaks as u64) ^ (run as u64)))
        .collect();

    let bits = 1u64 << setup.m();
    println!(
        "\n=== K = {n_keccaks:>6} Keccaks  (3-wide, m = {}, committed = {} MB) ===",
        setup.m(),
        bits >> 23,
    );

    // Warm-up.
    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        black_box(&setup.prove_fast(&state_sets[0], &mut ch).0);
    }

    // Best-of-n_runs prove_fast (full fast prover, incl. witness gen).
    let mut best_fast = f64::INFINITY;
    for run in 0..n_runs {
        let inputs = &state_sets[run + 1];
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (proof, _, _) = setup.prove_fast(inputs, &mut ch);
        best_fast = best_fast.min(t0.elapsed().as_secs_f64());
        black_box(&proof);
    }
    println!("  best prove_fast: {}", fmt_ms(best_fast));

    // Peak memory + verify time + serialized proof size (single prove).
    {
        let states_v = &state_sets[0];
        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (proof, commitment, _) = setup.prove_fast(states_v, &mut ch_p);
        println!("  peak memory: {:>8.2} MB", peak_mb());

        let mut ch_v = FsChallenger::new(b"flock-bench-v0");
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
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");

    // KECCAK3_KS (space/comma-separated keccak counts, e.g. "6144 24576") — when
    // set, bench ONLY the 3-wide encoder (flock3) at those counts and print the
    // parseable prove_fast/peak/verify/proof-size fields consumed by
    // bench_keccak.sh. Best for counts of the form 3·2^j. Unset → the default
    // single-vs-3-wide comparison sweep below.
    if let Ok(s) = std::env::var("KECCAK3_KS") {
        println!("3-wide Keccak-f[1600] prove_fast (flock3, SLIM rate 1/4) ({threads} thread(s)).");
        let counts: Vec<usize> = s
            .split([',', ' '])
            .filter(|t| !t.is_empty())
            .map(|t| {
                t.parse()
                    .expect("KECCAK3_KS: space/comma-separated integer keccak counts")
            })
            .collect();
        for k in counts {
            bench_3wide_report(k, 3); // best-of-3 trials
        }
        return;
    }

    println!(
        "3-wide vs single Keccak-f[1600] prove_fast throughput (SLIM rate 1/4) ({threads} thread(s))."
    );

    // 3·2^j sweet spots where 3-wide halves the committed size, plus a
    // power-of-two count (8192) where the two encoders break even.
    for &(k, n_runs) in &[(6144usize, 3), (8192, 3), (12288, 2), (24576, 2)] {
        bench_pair(k, n_runs);
    }
}
