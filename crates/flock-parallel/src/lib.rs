//! Shared parallel execution resources.

use std::{env::var, sync::OnceLock, thread::available_parallelism};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Returns the shared rayon pool that uses all available cores.
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
