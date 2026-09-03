//! Shared parallel execution resources.

use std::{env::var, sync::OnceLock, thread::available_parallelism};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Dedicated all-core (P+E) rayon pool for flat, fine-grained parallel-for
/// passes. The global pool deliberately excludes efficiency cores (perf
/// setups pin it to P-cores via `init_perf_thread_pool`) because they
/// straggle at the synchronization barriers of NTT-shaped phases. Passes
/// with many small independent work items and a single join (e.g. the PCS
/// combine's block fold: 4096 blocks of ~4 µs each) let the work-stealing
/// scheduler drain around slow cores, and measurably gain from the extra
/// E-core throughput (open_combine_probe: 18.0 → 12.8 ms, −29% at m=30 on
/// 4P+4E).
///
/// Built lazily on first use. Respects `RAYON_NUM_THREADS` (so single-thread
/// parity tests and ST bench conventions stay single-threaded). Exactly one
/// such pool may exist per process — a second copy oversubscribes the cores
/// it shares with the first, which is why this crate owns it.
pub fn all_core_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or_else(|| {
                available_parallelism()
                    .map(|value| value.get())
                    .unwrap_or(1)
            });
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(8 << 20)
            .build()
            .expect("failed to build the all-core rayon pool")
    })
}
