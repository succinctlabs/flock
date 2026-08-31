//! Shared parallel execution resources.

use std::sync::OnceLock;

/// Returns the shared rayon pool that uses all available cores.
pub fn all_core_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|value| value.get())
                    .unwrap_or(1)
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(8 << 20)
            .build()
            .expect("failed to build the all-core rayon pool")
    })
}
