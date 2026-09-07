//! Shared parallel execution resources.

use std::{env::var, sync::OnceLock, thread::available_parallelism};

use rayon::{ThreadPool, ThreadPoolBuilder};

static ALL_CORE_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Size the all-core pool explicitly, before its first use. For embedders
/// (an iOS app, say) that cannot set `RAYON_NUM_THREADS` and want to measure
/// pool sizes on the device. Returns `false` if the pool was already built,
/// in which case the existing size stands.
pub fn init_all_core_pool(threads: usize) -> bool {
    assert!(threads > 0, "the all-core pool needs at least one thread");
    ALL_CORE_POOL
        .set(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .stack_size(8 << 20)
                .build()
                .expect("failed to build the all-core rayon pool"),
        )
        .is_ok()
}

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
    ALL_CORE_POOL.get_or_init(|| {
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
