//! Keccak-f (3-wide) `prove_fast` (BaseFold) vs `prove_fast_ligerito` head-to-head.
//!
//! Run twice for ST and MT:
//!   cargo bench --bench keccak3_lig_vs_bf                              # MT (default)
//!   RAYON_NUM_THREADS=1 cargo bench --bench keccak3_lig_vs_bf          # ST
//!
//! Default K=24576 (=3·2^13 → m=30 with K_LOG=17). Override with KECCAK_K=<n>.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::keccak::{STATE_BITS, State};
use flock_prover::r1cs_hashes::keccak3::{K_LOG, KeccakSetup, min_n_blocks_log};

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

fn bench_block(n_keccaks: usize, n_runs: usize, threads_label: &str) {
    let n_blocks_log = min_n_blocks_log(n_keccaks);
    let m = K_LOG + n_blocks_log;
    let n_blocks = 1usize << n_blocks_log;
    let witness_bytes = (1usize << m) / 8;

    println!(
        "\n=== K = {n_keccaks:>6} keccaks  (m = {m}, blocks = {n_blocks}, witness = {} MB, {threads_label}) ===",
        witness_bytes >> 20
    );

    let setup = KeccakSetup::new(n_keccaks);
    let mk_inputs = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_keccaks)
            .map(|_| random_state(&mut rng))
            .collect::<Vec<State>>()
    };
    let input_sets: Vec<Vec<State>> = (0..=n_runs)
        .map(|run| mk_inputs(0xE71717_C0FFEE ^ (n_keccaks as u64) ^ (run as u64)))
        .collect();

    // BaseFold
    {
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast_basefold(&input_sets[0], &mut ch_p);
        black_box(&p);

        let mut best = f64::INFINITY;
        for run in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-bench-v0");
            let t0 = Instant::now();
            let (p, _, _) = setup.prove_fast_basefold(&input_sets[run + 1], &mut ch_p);
            let el = t0.elapsed().as_secs_f64();
            best = best.min(el);
            black_box(&p);
        }

        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (proof, commitment, _) = setup.prove_fast_basefold(&input_sets[0], &mut ch_p);
        let peak_after_prove = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let _ = setup
            .verify_basefold(&commitment, &proof, &mut ch_v)
            .expect("bf verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let bundle = flock_prover::proof_io::R1csProofBundle { commitment, proof };
        let size = bundle.to_bytes().len();
        black_box(&bundle);

        println!(
            "  BaseFold:  prove = {}   verify = {}   size = {}   peak = {:.2} MB",
            fmt_ms(best),
            fmt_ms(verify_t),
            fmt_kb(size),
            peak_after_prove,
        );
    }

    // Ligerito
    {
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&input_sets[0], &mut ch_p);
        black_box(&p);

        let mut best = f64::INFINITY;
        for run in 0..n_runs {
            let mut ch_p = FsChallenger::new(b"flock-bench-v0");
            let t0 = Instant::now();
            let (p, _, _) = setup.prove_fast(&input_sets[run + 1], &mut ch_p);
            let el = t0.elapsed().as_secs_f64();
            best = best.min(el);
            black_box(&p);
        }

        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (proof, commitment, _) = setup.prove_fast(&input_sets[0], &mut ch_p);
        let peak_after_prove = peak_mb();
        let mut ch_v = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let _ = setup
            .verify(&commitment, &proof, &mut ch_v)
            .expect("lig verify");
        let verify_t = t0.elapsed().as_secs_f64();
        let bundle = flock_prover::proof_io::R1csProofBundleLigerito { commitment, proof };
        let size = bundle.to_bytes().len();
        black_box(&bundle);

        println!(
            "  Ligerito:  prove = {}   verify = {}   size = {}   peak = {:.2} MB",
            fmt_ms(best),
            fmt_ms(verify_t),
            fmt_kb(size),
            peak_after_prove,
        );
    }
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
    println!(
        "Keccak3 (K_LOG={}) prove_fast (BaseFold) vs prove_fast_ligerito — {}",
        K_LOG, label
    );

    let ks: Vec<usize> = match std::env::var("KECCAK_K") {
        Ok(s) => s
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().expect("KECCAK_K: integer K"))
            .collect(),
        Err(_) => vec![24576], // 3 · 2^13 → m = 30
    };
    let n_runs: usize = std::env::var("FLOCK_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    for &n in &ks {
        bench_block(n, n_runs, label);
    }
}
