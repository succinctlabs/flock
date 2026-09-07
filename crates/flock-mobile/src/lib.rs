//! C-ABI shim over the Flock prover for iOS.
//!
//! One entry point runs a small recursion tower end to end — leaf proofs,
//! folds, root discharge — and reports wall time, proof size, and the
//! process's peak resident set. Swift calls it through
//! `include/flock_mobile.h`; the `flock_mobile_probe` binary calls the same
//! function so the number can be taken on a Mac or inside the iOS simulator
//! without an app.
//!
//! Build for a device or the simulator with
//! `cargo build --release -p flock-mobile --target aarch64-apple-ios[-sim]`;
//! the static library lands in `target/<triple>/release/libflock_mobile.a`.

use std::{panic::catch_unwind, time::Instant};

use flock_prover::tower::{Tower, TowerConfig, build_chain_proof};

/// What one demo tower run measured. Every field is plain C data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FlockMobileReport {
    /// Total compressions proved: `n_leaves * blocks_per_leaf`.
    pub n_blocks: u64,
    /// `Tower::prove` wall time.
    pub prove_ms: f64,
    /// `Tower::discharge_root` wall time.
    pub discharge_ms: f64,
    /// Size of the root proof, in KiB.
    pub root_proof_kib: f64,
    /// Peak resident set size of the process so far, in bytes.
    pub peak_rss_bytes: u64,
    /// Threads in the process's all-core rayon pool.
    pub threads: u32,
    /// First word of `h_end`, so the caller can eyeball the chain state.
    pub h_end_0: u32,
}

/// Prove and discharge one 128-bit tower over a BLAKE3 compression chain
/// starting from the zero state.
///
/// `n_leaves` must be even and at least 4. Writes the measurements into
/// `*out` and returns 0 on success. Returns 1 if the proof machinery
/// panicked (the report is left untouched), and 2 if `out` is null.
///
/// # Safety
/// `out` must be null or point to a writable `FlockMobileReport`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flock_mobile_tower_demo(
    n_leaves: u32,
    blocks_per_leaf: u32,
    out: *mut FlockMobileReport,
) -> i32 {
    if out.is_null() {
        return 2;
    }
    match catch_unwind(|| tower_demo(n_leaves as usize, blocks_per_leaf as usize)) {
        Ok(report) => {
            // SAFETY: the caller promised `out` is writable and it is non-null.
            unsafe { out.write(report) };
            0
        }
        Err(_) => 1,
    }
}

/// Number of threads in the process's all-core rayon pool. Building the pool
/// is a side effect of the first call.
#[unsafe(no_mangle)]
pub extern "C" fn flock_mobile_thread_count() -> u32 {
    flock_prover::all_core_pool().current_num_threads() as u32
}

/// Peak resident set size of the calling process, in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn flock_mobile_peak_rss_bytes() -> u64 {
    peak_rss_bytes()
}

/// Prove ONE chain leaf of `n_blocks` BLAKE3 compressions from the zero
/// state, with no folding above it. This is the phone-sized workload: the
/// recursion node is a fixed m=29 dense circuit and needs ~12 GiB, while a
/// leaf's footprint scales with `n_blocks`.
///
/// Fills only `n_blocks`, `prove_ms`, `peak_rss_bytes`, and `threads` of
/// `*out`; the rest stay zero. Return codes as [`flock_mobile_tower_demo`].
///
/// # Safety
/// `out` must be null or point to a writable `FlockMobileReport`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn flock_mobile_leaf_demo(n_blocks: u32, out: *mut FlockMobileReport) -> i32 {
    if out.is_null() {
        return 2;
    }
    match catch_unwind(|| leaf_demo(n_blocks as usize)) {
        Ok(report) => {
            // SAFETY: the caller promised `out` is writable and it is non-null.
            unsafe { out.write(report) };
            0
        }
        Err(_) => 1,
    }
}

/// The leaf run behind [`flock_mobile_leaf_demo`], for Rust callers.
pub fn leaf_demo(n_blocks: usize) -> FlockMobileReport {
    let threads = flock_mobile_thread_count();
    let t0 = Instant::now();
    let proof = build_chain_proof(TowerConfig::Chain128, [0u32; 16], n_blocks);
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    drop(proof);
    FlockMobileReport {
        n_blocks: n_blocks as u64,
        prove_ms,
        peak_rss_bytes: peak_rss_bytes(),
        threads,
        ..Default::default()
    }
}

/// The demo run behind [`flock_mobile_tower_demo`], for Rust callers.
pub fn tower_demo(n_leaves: usize, blocks_per_leaf: usize) -> FlockMobileReport {
    let threads = flock_mobile_thread_count();
    let h_start = [0u32; 16];

    let t0 = Instant::now();
    let tower = Tower::prove(TowerConfig::Chain128, h_start, blocks_per_leaf, n_leaves);
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    tower
        .discharge_root()
        .expect("the root-side residue discharges");
    let discharge_ms = t1.elapsed().as_secs_f64() * 1e3;

    let s = tower.statement();
    FlockMobileReport {
        n_blocks: s.n_blocks as u64,
        prove_ms,
        discharge_ms,
        root_proof_kib: tower.root_proof_kib(),
        peak_rss_bytes: peak_rss_bytes(),
        threads,
        h_end_0: s.h_end[0],
    }
}

/// `ru_maxrss` from `getrusage`. Darwin (macOS and iOS) reports it in bytes;
/// Linux reports kilobytes, converted here so the field means bytes everywhere.
pub fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a valid, writable rusage; RUSAGE_SELF needs no other state.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return 0;
    }
    let raw = usage.ru_maxrss as u64;
    if cfg!(target_vendor = "apple") {
        raw
    } else {
        raw * 1024
    }
}
