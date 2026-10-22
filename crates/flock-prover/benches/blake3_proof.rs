//! End-to-end BLAKE3 compression-function proof benchmark with per-phase
//! timing breakdown. Times the fast prover path (`Blake3Setup::prove_fast`);
//! the slow `prove` path is exercised by unit tests in `src/blake3.rs`.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    array::from_fn,
    env::var,
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use flock_core::test_rng::Rng;
use flock_prover::{
    challenger::FsChallenger,
    init_perf_thread_pool,
    merkle::HashKind,
    pcs::ligerito::LigeritoProfile,
    r1cs_hashes::blake3::{Blake3Setup, Compression, K_LOG, min_n_blocks_log},
};
// Peak-heap tracker (wraps System), as in keccak_proof/sha2_proof — lets the
// BLAKE3_LOG2S report emit a "peak memory:" line for bench_blake3.sh.
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

fn random_compression(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = from_fn(|_| rng.next_u32());
    let m: [u32; 16] = from_fn(|_| rng.next_u32());
    // counter varies per instance; block_len = 64 (full block), flags = a
    // typical CHUNK_START|CHUNK_END|ROOT for a single-block chunk.
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

fn bench_one(n_blocks: usize, n_runs: usize) {
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;
    let witness_bytes = (1usize << m) / 8;

    println!(
        "\n=== {n_blocks:>5} compressions  (m = {m}, slots = {n_slots}, witness = {} MB) ===",
        witness_bytes >> 20
    );

    // BLAKE3_PROFILE=fast|slim|secure selects the Ligerito profile (default fast).
    let mut setup = match var("BLAKE3_PROFILE").as_deref() {
        Ok("slim") => Blake3Setup::with_profile(n_blocks, LigeritoProfile::Slim),
        Ok("secure") => Blake3Setup::with_profile(n_blocks, LigeritoProfile::Secure),
        Ok("fast") | Err(_) => Blake3Setup::new(n_blocks),
        Ok(p) => panic!("BLAKE3_PROFILE must be fast, slim, or secure (got {p})"),
    };
    // FLOCK_MERKLE_HASH=sha256|blake3 selects the PCS Merkle hash (default
    // sha256). Setting it on `pcs_params` is enough: the Ligerito prover and
    // verifier configs are derived from those params, so the L0 commitment and
    // every recursive level follow.
    if let Ok(h) = var("FLOCK_MERKLE_HASH") {
        setup.pcs_params.merkle_hash =
            HashKind::parse(&h).expect("FLOCK_MERKLE_HASH must be sha256 or blake3");
    }
    let setup = setup;
    // FLOCK_FS_HASH=sha256|blake3 selects the Fiat-Shamir transcript hash
    // (default sha256), independently of the Merkle hash above.
    let fs_hash = match var("FLOCK_FS_HASH") {
        Ok(h) => HashKind::parse(&h).expect("FLOCK_FS_HASH must be sha256 or blake3"),
        Err(_) => HashKind::default(),
    };
    let fs = || FsChallenger::with_hash(b"flock-bench-v0", fs_hash);
    // Boolean zerocheck flavor. **AG is the default wherever its round-1
    // kernel exists** (aarch64 NEON); x86 falls back to RS until the AVX-512
    // AG round-1 kernel lands (docs/ag-recursion-plan.md Phase F.1). Same
    // selector convention as the tower's `leaf_zerocheck_ag()`/`outer_zerocheck_ag()`,
    // which have defaulted to AG since Phase B/C.
    //
    // Measured at m=32, paired alternating, 2026-09-01: AG 862.7 best /
    // 866.6 median vs RS 894.8 / 919.7 — **−5.8%, AG 4/4**. Both arms run
    // the DEFAULT (sparse) tail gate: an earlier reading that sparse cost AG
    // 3.2× was a measurement artifact, and re-testing it paired shows AG
    // sparse beating AG dense 4/4 (median −41.5 ms). The gate is a
    // constant (`SPARSE_TAIL_GATE`); the env override was removed.
    //
    // `prove_fast_union_ag` is the same union commit / lincheck / merged
    // opening — only zerocheck round 1 differs — so `BLAKE3_ZC=rs` still
    // isolates RS vs AG end to end on one witness.
    let zc_ag = match std::env::var("BLAKE3_ZC").as_deref() {
        Ok("ag") => {
            assert!(
                cfg!(target_arch = "aarch64"),
                "BLAKE3_ZC=ag requires aarch64 (the AG round-1 kernel is NEON)"
            );
            true
        }
        Ok("rs") => false,
        Err(_) => cfg!(target_arch = "aarch64"),
        Ok(v) => panic!("BLAKE3_ZC must be rs or ag (got {v})"),
    };
    println!(
        "  merkle hash: {}   fs hash: {}",
        setup.pcs_params.merkle_hash, fs_hash
    );
    // Generate n_runs + 2 distinct block vectors so each run hits a fresh
    // witness (and therefore a fresh Fiat-Shamir transcript). The first is
    // used for warm-up; the rest for measurements + one spare.
    let mk_blocks = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_blocks)
            .map(|_| random_compression(&mut rng))
            .collect::<Vec<Compression>>()
    };
    let block_sets: Vec<Vec<Compression>> = (0..=n_runs)
        .map(|run| mk_blocks(0xC0FFEE_BEEF ^ (n_blocks as u64) ^ (run as u64)))
        .collect();

    println!("  zerocheck: {}", if zc_ag { "ag" } else { "rs" });

    // Warm-up.
    {
        let mut ch_p = fs();
        if zc_ag {
            #[cfg(target_arch = "aarch64")]
            {
                let (p, _, _) = setup.prove_fast_union_ag(&block_sets[0], &mut ch_p);
                black_box(&p);
            }
            #[cfg(not(target_arch = "aarch64"))]
            panic!("BLAKE3_ZC=ag requires aarch64");
        } else {
            let (p, _, _) = setup.prove_fast(&block_sets[0], &mut ch_p);
            black_box(&p);
        }
    }

    // Best-of-n_runs prove_fast. Each run uses a distinct block vector so the
    // FS transcript varies across iterations.
    let mut best_fast = f64::INFINITY;
    for run in 0..n_runs {
        let blocks = &block_sets[run + 1];
        let mut ch_p = fs();
        let t0 = Instant::now();
        if zc_ag {
            #[cfg(target_arch = "aarch64")]
            {
                let (p, _, _) = setup.prove_fast_union_ag(blocks, &mut ch_p);
                black_box(&p);
            }
        } else {
            let (p, _, _) = setup.prove_fast(blocks, &mut ch_p);
            black_box(&p);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        best_fast = best_fast.min(elapsed);
        println!(
            "  [run {}/{}] prove_fast: {}",
            run + 1,
            n_runs,
            fmt_ms(elapsed)
        );
    }
    println!(
        "  best prove_fast: {}  ({:.0} compressions/sec)",
        fmt_ms(best_fast),
        n_blocks as f64 / best_fast
    );

    // Peak memory + verify time + serialized proof size (single prove), in
    // WHICHEVER flavor is selected — this used to be RS-only, which would now
    // silently drop these three lines from the default (AG) report.
    {
        let blocks_v = &block_sets[0];
        // Grind-free proofs fail PoW verification by design; skip the verify
        // so the phase TSV and summary lines below still print.
        let no_grind = std::env::var_os("FLOCK_NO_GRIND").is_some();
        let report = |peak: f64, verify: Option<f64>, size: usize| {
            println!("  peak memory: {peak:>8.2} MB");
            match verify {
                Some(v) => println!("  verify: {}", fmt_ms(v)),
                None => println!("  verify: skipped (FLOCK_NO_GRIND)"),
            }
            println!(
                "  proof size: {} bytes ({:.2} KiB)",
                size,
                size as f64 / 1024.0
            );
        };
        if zc_ag {
            #[cfg(target_arch = "aarch64")]
            {
                reset_peak();
                let mut ch_p = fs();
                let (proof, commitment, _) = setup.prove_fast_union_ag(blocks_v, &mut ch_p);
                let peak = peak_mb();
                let verify = (!no_grind).then(|| {
                    let mut ch_v = fs();
                    let t = Instant::now();
                    setup
                        .verify_union_ag(&commitment, &proof, &mut ch_v)
                        .expect("union-AG verify failed");
                    t.elapsed().as_secs_f64()
                });
                let bundle =
                    flock_prover::proof_io::R1csProofBundleLigeritoAg { commitment, proof };
                report(peak, verify, bundle.to_bytes().len());
                black_box(&bundle);
            }
        } else {
            reset_peak();
            let mut ch_p = fs();
            let (proof, commitment, _) = setup.prove_fast(blocks_v, &mut ch_p);
            let peak = peak_mb();
            let verify = (!no_grind).then(|| {
                let mut ch_v = fs();
                let t = Instant::now();
                setup
                    .verify(&commitment, &proof, &mut ch_v)
                    .expect("verify failed");
                t.elapsed().as_secs_f64()
            });
            let bundle = flock_prover::proof_io::R1csProofBundleLigerito { commitment, proof };
            report(peak, verify, bundle.to_bytes().len());
            black_box(&bundle);
        }
    }

    // Per-phase breakdown: the union prover prints one under `PCS_TRACE=1`
    // (witgen / compact / commit / zerocheck+lincheck / open).
    println!("  (per-phase breakdown: rerun with PCS_TRACE=1)");
}

fn main() {
    let _ = init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("BLAKE3 compression-function R1CS proof timings (prove_fast).");

    // Sizes to bench. Override with BLAKE3_LOG2S (space/comma-separated log2
    // compression counts, e.g. "12 14") — used by benchmarks/bench_blake3.sh
    // to sweep at the same sizes as the competitors; each listed size is benched
    // best-of-3. Default: small-scale context + the SHA-256/Keccak baseline sizes.
    // n_blocks → m: K_LOG=14, so m = 14 + ceil_log2(max(n_blocks, 8)).
    let specs: Vec<(usize, usize)> = match var("BLAKE3_LOG2S") {
        Ok(s) => s
            .split([',', ' '])
            .filter(|t| !t.is_empty())
            .map(|t| {
                let h: u32 = t
                    .parse()
                    .expect("BLAKE3_LOG2S: space/comma-separated integer log2 values");
                // `BLAKE3_RUNS` overrides the measured-run count (steady-state
                // pool behaviour needs more than the default three).
                let runs = std::env::var("BLAKE3_RUNS")
                    .ok()
                    .and_then(|r| r.parse().ok())
                    .unwrap_or(3usize);
                (1usize << h, runs)
            })
            .collect(),
        Err(_) => vec![(1usize, 3), (128, 2), (8192, 2), (32768, 2), (65536, 2)],
    };
    for &(n, n_runs) in &specs {
        bench_one(n, n_runs);
    }
}
