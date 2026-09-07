//! Shared parallel execution resources.

use std::{
    env::var,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    thread::{available_parallelism, scope},
};

use rayon::{ThreadPool, ThreadPoolBuilder, current_num_threads, scope as rayon_scope};

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

/// Tag the current thread as utility QoS (`QOS_CLASS_UTILITY = 0x11`). On
/// Apple Silicon the scheduler prefers the efficiency cores for utility
/// threads while default-QoS work holds the P-cores — helper threads that
/// should ride the E-cores (the commit's leaf pipeline, the ranked NTT top
/// tiles) want exactly this split. (Background QoS is too weak: it can be
/// starved entirely while the P pool is saturated.)
#[cfg(target_os = "macos")]
pub fn set_utility_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x11, 0);
    }
}
#[cfg(not(target_os = "macos"))]
pub fn set_utility_qos() {}

/// Run `f(0..n_chunks)` with one shared work counter drained by BOTH the
/// current rayon pool and two utility-QoS helper threads (E-cores on Apple
/// Silicon) — heterogeneous tile distribution for passes whose chunks are
/// independent. `f` must tolerate concurrent calls on distinct indices;
/// which worker runs a chunk cannot change its output.
pub fn run_hetero_chunks<F: Fn(usize) + Sync>(n_chunks: usize, f: F) {
    if n_chunks == 0 {
        return;
    }
    // A deliberately single-threaded pool (RAYON_NUM_THREADS=1, ST parity
    // runs) stays truly single-threaded: run inline, spawn nothing.
    if current_num_threads() <= 1 {
        for i in 0..n_chunks {
            f(i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let pull = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            f(i);
        }
    };
    let pull = &pull;
    scope(|s| {
        for _ in 0..2 {
            s.spawn(move || {
                set_utility_qos();
                pull();
            });
        }
        rayon_scope(|s| {
            for _ in 0..current_num_threads() {
                s.spawn(move |_| pull());
            }
        });
    });
}

/// [`run_hetero_chunks`] with per-worker state: each pull-loop worker (rayon
/// task or E-thread) builds one `S` via `init` on first use and threads it
/// through its chunks — for drains that accumulate per-worker partials the
/// caller merges afterwards. Collected states are returned in no particular
/// order.
pub fn run_hetero_chunks_stateful<S, I, F>(n_chunks: usize, init: I, f: F) -> Vec<S>
where
    S: Send,
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    if n_chunks == 0 {
        return Vec::new();
    }
    if current_num_threads() <= 1 {
        let mut s = init();
        for i in 0..n_chunks {
            f(&mut s, i);
        }
        return vec![s];
    }
    let next = AtomicUsize::new(0);
    let states: Mutex<Vec<S>> = Mutex::new(Vec::new());
    let pull = || {
        let first = next.fetch_add(1, Ordering::Relaxed);
        if first >= n_chunks {
            return;
        }
        let mut s = init();
        f(&mut s, first);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            f(&mut s, i);
        }
        states.lock().unwrap().push(s);
    };
    let pull = &pull;
    scope(|sc| {
        for _ in 0..2 {
            sc.spawn(move || {
                set_utility_qos();
                pull();
            });
        }
        rayon_scope(|sc| {
            for _ in 0..current_num_threads() {
                sc.spawn(move |_| pull());
            }
        });
    });
    states.into_inner().unwrap()
}
