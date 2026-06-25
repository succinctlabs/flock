//! End-to-end proof benchmark for the **hybrid** SHA-256 encoding in
//! `src/sha2.rs` (K_LOG=15, 1 instance per block, 95.8% fill).
//! Uses `Sha256HybridSetup::prove_fast` — the fused (z, a, b, c, z_lincheck)
//! generator, mirroring `sha_packed::prove_fast`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::sha2::{K_LOG, Sha256HybridSetup, USEFUL_BITS, min_n_blocks_log};

// Peak-heap tracker (wraps System): records the high-water mark of currently
// outstanding bytes, same notion as binius64's peakmem-alloc. Negligible
// overhead (one relaxed atomic op per alloc/dealloc).
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

fn random_input(rng: &mut Rng) -> ([u32; 8], [u32; 16]) {
    let h: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (h, m)
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

fn bench_one(n_compressions: usize, n_runs: usize) {
    let n_log = min_n_blocks_log(n_compressions);
    let m = K_LOG + n_log;
    let n_blocks = 1usize << n_log;
    let witness_bytes = (1usize << m) / 8;
    let outer_fill = (n_compressions as f64) / (n_blocks as f64) * 100.0;
    let total_useful_bits = (n_compressions * USEFUL_BITS) as f64;
    let total_z_bits = (1u64 << m) as f64;
    let padding_pct = 100.0 * (1.0 - total_useful_bits / total_z_bits);

    println!(
        "\n=== {n_compressions:>5} compressions  (m = {m}, blocks = {n_compressions}/{n_blocks} = \
         {outer_fill:.0}%, padding = {padding_pct:.1}%, witness = {} MB) ===",
        witness_bytes >> 20
    );

    let setup = Sha256HybridSetup::new(n_compressions);
    let mk_inputs = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_compressions)
            .map(|_| random_input(&mut rng))
            .collect::<Vec<([u32; 8], [u32; 16])>>()
    };
    let input_sets: Vec<Vec<([u32; 8], [u32; 16])>> = (0..=n_runs)
        .map(|run| mk_inputs(0xC0FFEE_5A55 ^ (n_compressions as u64) ^ (run as u64)))
        .collect();

    // Warm-up.
    {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast(&input_sets[0], &mut ch);
        black_box(&p);
    }

    let mut best = f64::INFINITY;
    for run in 0..n_runs {
        let inputs = &input_sets[run + 1];
        let mut ch = FsChallenger::new(b"flock-bench-v0");
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
    let hashes_per_sec = (n_compressions as f64) / best;
    println!(
        "  best prove_fast: {}   ({:.0} hashes/sec)",
        fmt_ms(best),
        hashes_per_sec
    );

    // Peak memory (heap high-water mark over a single prove) + verify + proof size.
    {
        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (proof, commitment, _) = setup.prove_fast(&input_sets[0], &mut ch_p);
        let peak = peak_mb();
        println!("  peak memory: {:>8.2} MB", peak);

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

    // Per-phase breakdown of the *real* Ligerito prover (witness gen + commit +
    // zerocheck + lincheck + recursive PCS open) via prove_fast_timed, so the
    // phases decompose exactly the prover the headline number runs.
    println!("  [prove_fast breakdown]");
    let mut ch = FsChallenger::new(b"flock-bench-v0");
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
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!(
        "Hybrid SHA-256 (K_LOG=15, 1 IPB) R1CS proof_fast timings.\n\
         Boundary wins vs sha_packed expected at N = 2^k (non-3·2^k)."
    );

    // Sizes to bench. Default: 8/256 small-scale context + 4096/16384/65536. Override
    // with SHA2_LOG2S (space/comma-separated log2 compression counts, e.g. "12 14") —
    // used by benchmarks/bench_sha256.sh to sweep at the same sizes as the
    // competitors; each listed size is benched best-of-3.
    let specs: Vec<(usize, usize)> = match std::env::var("SHA2_LOG2S") {
        Ok(s) => s
            .split([',', ' '])
            .filter(|t| !t.is_empty())
            .map(|t| {
                let h: u32 = t
                    .parse()
                    .expect("SHA2_LOG2S: space/comma-separated integer log2 values");
                (1usize << h, 3usize)
            })
            .collect(),
        Err(_) => vec![(8usize, 2), (256, 2), (4096, 2), (16384, 2), (65536, 2)],
    };
    for &(n, n_runs) in &specs {
        bench_one(n, n_runs);
    }
}
