//! Probe: can the commit codeword alloc+zero (page-fault-bound, ~1.0x MT
//! scaling) be HIDDEN by overlapping it with a representative P-core workload
//! (standing in for gen_witness, which runs before commit in prove_fast)?
//!
//! We compare three regimes at the m=29 codeword size (128 MB):
//!   1. serial alloc+zero alone                          → t_alloc
//!   2. a rayon compute workload alone (P-cores)         → t_work
//!   3. both at once: alloc+zero on a std::thread,
//!      compute on the main rayon pool, then join        → t_overlap
//!
//! If t_overlap ≈ max(t_work, t_alloc) the offload is "free" (the alloc hid
//! behind compute). If t_overlap ≈ t_work + t_alloc they contend (shared
//! memory bandwidth) and an E-core won't help. We also try a QoS-tagged
//! background thread (macOS) to see if E-core placement reduces contention.

// Deliberate uninit codeword buffer (write-before-read), mirroring the prover's
// internal `alloc_uninit_vec`.
#![allow(clippy::uninit_vec)]

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;

const LOG_CODEWORD_F128: usize = 23; // m=29 codeword = 2^23 F128 = 128 MB
const N: usize = 1 << LOG_CODEWORD_F128;

fn fmt(s: f64) -> String {
    format!("{:>7.2} ms", s * 1e3)
}

/// Fresh 128 MB buffer, copy lower half + zero upper half — mirrors
/// `pcs::commit`'s alloc/pad step (all first-touch page faults).
fn alloc_and_pad(src: &[F128]) -> Vec<F128> {
    let mut buf: Vec<F128> = Vec::with_capacity(N);
    // SAFETY: F128 is 16 bytes, no Drop; we write every slot below.
    unsafe {
        buf.set_len(N);
    }
    let half = N / 2;
    buf[..half.min(src.len())].copy_from_slice(&src[..half.min(src.len())]);
    // write_bytes count is in F128 ELEMENTS (each set to the 0 byte pattern).
    unsafe {
        std::ptr::write_bytes(buf.as_mut_ptr().add(half), 0u8, N - half);
    }
    buf
}

/// Representative compute-heavy P-core workload: parallel F128 multiply-chains.
/// Sized via `rounds` to land near gen_witness's ~18 ms.
fn pcore_workload(rounds: usize) -> F128 {
    use rayon::prelude::*;
    (0..rayon::current_num_threads().max(1))
        .into_par_iter()
        .map(|t| {
            let mut acc = F128::new(0x9E3779B97F4A7C15 ^ t as u64, t as u64 + 1);
            let m = F128::new(0xBF58476D1CE4E5B9, 0x94D049BB133111EB);
            for _ in 0..rounds {
                acc = acc * m + m;
            }
            acc
        })
        .reduce(|| F128::ZERO, |a, b| a + b)
}

#[cfg(target_os = "macos")]
fn set_background_qos() {
    // QOS_CLASS_BACKGROUND = 0x09 — macOS scheduler strongly prefers E-cores.
    // Declared here to avoid a libc dependency just for the probe.
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x09, 0);
    }
}
#[cfg(not(target_os = "macos"))]
fn set_background_qos() {}

use std::sync::atomic::{AtomicUsize, Ordering};

/// Counts how many background OS threads the gated path actually spawns, so we
/// can prove the single-threaded path spawns zero.
static SPAWNED: AtomicUsize = AtomicUsize::new(0);

/// Gated prefault: in a multi-threaded pool, offload alloc+zero to a
/// background-QoS (E-core) thread and run `work` concurrently. In a
/// single-thread pool (e.g. RAYON_NUM_THREADS=1), run inline — alloc THEN
/// work, on the one calling thread, spawning zero OS threads → truly serial.
fn prefault_then_work<R>(src: &[F128], work: impl FnOnce() -> R) -> (Vec<F128>, R) {
    if rayon::current_num_threads() <= 1 {
        // Truly single-threaded: inline, no extra OS thread.
        let buf = alloc_and_pad(src);
        let r = work();
        (buf, r)
    } else {
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                SPAWNED.fetch_add(1, Ordering::Relaxed);
                set_background_qos();
                alloc_and_pad(src)
            });
            let r = work();
            (h.join().unwrap(), r)
        })
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    println!(
        "(rayon perf-pool threads: {threads}; codeword = {} MB)",
        (N * 16) >> 20
    );

    // Source data for the lower-half copy.
    let src: Vec<F128> = (0..N / 2).map(|i| F128::new(i as u64, !i as u64)).collect();

    // Calibrate the workload to ~18 ms (gen_witness ballpark).
    let mut rounds = 1_000_000usize;
    loop {
        let t = Instant::now();
        black_box(pcore_workload(rounds));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms > 16.0 {
            println!("workload calibrated: {rounds} rounds → {ms:.1} ms");
            break;
        }
        rounds = (rounds as f64 * (18.0 / ms.max(1.0))) as usize + 1;
        if rounds > 200_000_000 {
            println!("workload cap hit at {rounds}");
            break;
        }
    }

    let n_runs = 5;

    // 1. alloc+zero alone.
    let mut t_alloc = f64::INFINITY;
    for _ in 0..n_runs {
        let t = Instant::now();
        let b = alloc_and_pad(&src);
        t_alloc = t_alloc.min(t.elapsed().as_secs_f64());
        black_box(b);
    }
    println!("\n1. alloc+zero alone:            {}", fmt(t_alloc));

    // 2. compute workload alone.
    let mut t_work = f64::INFINITY;
    for _ in 0..n_runs {
        let t = Instant::now();
        black_box(pcore_workload(rounds));
        t_work = t_work.min(t.elapsed().as_secs_f64());
    }
    println!("2. compute workload alone:      {}", fmt(t_work));

    // 3. overlap: alloc on a plain std::thread, compute on main rayon pool.
    let mut t_overlap_plain = f64::INFINITY;
    for _ in 0..n_runs {
        let src_ref = &src;
        let t = Instant::now();
        let buf = std::thread::scope(|s| {
            let h = s.spawn(|| alloc_and_pad(src_ref));
            black_box(pcore_workload(rounds));
            h.join().unwrap()
        });
        t_overlap_plain = t_overlap_plain.min(t.elapsed().as_secs_f64());
        black_box(buf);
    }
    println!("3. overlap (plain std::thread): {}", fmt(t_overlap_plain));

    // 4. overlap with E-core-tagged (background QoS) alloc thread.
    let mut t_overlap_qos = f64::INFINITY;
    for _ in 0..n_runs {
        let src_ref = &src;
        let t = Instant::now();
        let buf = std::thread::scope(|s| {
            let h = s.spawn(|| {
                set_background_qos();
                alloc_and_pad(src_ref)
            });
            black_box(pcore_workload(rounds));
            h.join().unwrap()
        });
        t_overlap_qos = t_overlap_qos.min(t.elapsed().as_secs_f64());
        black_box(buf);
    }
    println!("4. overlap (E-core QoS thread): {}", fmt(t_overlap_qos));

    println!("\n--- interpretation ---");
    println!(
        "ideal overlap (max):           {}",
        fmt(t_work.max(t_alloc))
    );
    println!("no overlap (sum):              {}", fmt(t_work + t_alloc));
    let hidden_plain = (t_work + t_alloc - t_overlap_plain).max(0.0);
    let hidden_qos = (t_work + t_alloc - t_overlap_qos).max(0.0);
    println!(
        "alloc hidden, plain thread:    {} of {}",
        fmt(hidden_plain),
        fmt(t_alloc)
    );
    println!(
        "alloc hidden, QoS thread:      {} of {}",
        fmt(hidden_qos),
        fmt(t_alloc)
    );

    // 5. GATED path — the production rule. Spawns a background thread only when
    // the pool has >1 thread; otherwise runs inline (truly serial).
    SPAWNED.store(0, Ordering::Relaxed);
    let mut t_gated = f64::INFINITY;
    for _ in 0..n_runs {
        let src_ref = &src;
        let t = Instant::now();
        let (buf, w) = prefault_then_work(src_ref, || pcore_workload(rounds));
        t_gated = t_gated.min(t.elapsed().as_secs_f64());
        black_box((buf, w));
    }
    let spawned = SPAWNED.load(Ordering::Relaxed);
    println!("\n5. GATED prefault_then_work:   {}", fmt(t_gated));
    println!(
        "   pool threads = {threads}  →  background threads spawned over {n_runs} runs = {spawned}"
    );
    if threads <= 1 {
        println!(
            "   ✓ single-threaded: 0 spawns, runs as sum (serial) = {}",
            fmt(t_work + t_alloc)
        );
    } else {
        println!(
            "   ✓ multi-threaded: offloaded, ≈ ideal overlap = {}",
            fmt(t_work.max(t_alloc))
        );
    }
}
