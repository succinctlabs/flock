//! `flock-core`: the protocol library and verifier for Flock's R1CS-over-GF(2)
//! sumcheck/zerocheck PIOP with a multilinear PCS.
//!
//! This crate carries everything the verifier needs. It is portable — the NEON
//! kernels in `field`, `ntt`, `lincheck`, `zerocheck`, and `merkle` have scalar
//! fallbacks — though it is tuned for Apple silicon. The end-to-end prover, the
//! hash R1CS encoders, and the CLI live in the `flock-prover` crate built on
//! top of this one.
//!
//! Protocol flow:
//!   1. Prover commits to the witness z ∈ GF(2)^n via a multilinear PCS.
//!   2. Prover computes the row-witnesses a = A·z, b = B·z, c = C·z.
//!   3. Zerocheck PIOP reduces a·b ⊕ c = 0 to evaluation claims on (â, b̂, ĉ) at ρ.
//!   4. Lincheck PIOP reduces those to a single evaluation claim ẑ(ρ') = v.
//!   5. PCS opens ẑ at ρ'.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

use crate::field::F128;
use std::alloc::Layout;
use std::alloc::alloc_zeroed;
use std::alloc::handle_alloc_error;
use std::env::var;
use std::process::Command;
use std::str::from_utf8;
use std::sync::OnceLock;
use std::thread::available_parallelism;
pub mod aggregate;
pub mod bits;
pub mod challenger;
pub mod circuit;
pub mod element_r1cs;
pub mod field;
pub mod hash;
pub mod lincheck;
pub mod matrix_fold;
pub mod merkle;
pub mod ntt;
pub mod pcs;
pub mod permutation;
pub mod product_gkr;
pub mod proof;
pub mod r1cs;
pub mod schedule;
pub mod scratch;
pub mod transcript_record;
pub mod union;
pub mod verifier;
pub mod zerocheck;

/// Configure rayon's global thread pool to use only performance cores on
/// Apple silicon (excluding efficiency cores).
///
/// On M-series chips the 2 efficiency cores run at ~30-40% of perf-core
/// speed and become stragglers in compute-bound parallel work — the
/// work-stealing scheduler keeps assigning them tasks that hold up the perf
/// cores at synchronization barriers. Empirically, 8 threads beats 10 by
/// ~10-20% on `pcs::commit` and similar parallel-NTT workloads.
///
/// Call this **once** at program startup, before any other parallel flock
/// code runs (rayon's global pool is set on first use; if it's already
/// created, this call is a no-op).
///
/// Respects `RAYON_NUM_THREADS` — if that env var is set, this function
/// does nothing (so explicit user configuration always wins).
///
/// Returns the number of threads the pool was configured with, or `None`
/// if no change was made (either because the env var was set or because
/// rayon was already initialized).
pub fn init_perf_thread_pool() -> Option<usize> {
    if var("RAYON_NUM_THREADS").is_ok() {
        return None;
    }
    let n = perf_core_count();
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
    {
        Ok(()) => Some(n),
        Err(_) => None, // pool already built
    }
}

/// Allocate a `Vec<T>` of length `n` whose contents are NOT zero-initialized.
/// Caller MUST write every slot before reading it.
///
/// Used to skip the eager zero-init of large ping-pong buffers in hot prover
/// paths (PCS open, Round-2 fold, NTT scratch, lincheck packing). At m=29 the
/// zero-fill of a fresh 128 MB `vec![T::default(); n]` runs sequentially on
/// the main thread (~22 ms), which caps the parallel speedup of those phases.
///
/// `T: Copy` ensures `T` has no Drop impl, so the leaked uninitialized
/// elements are a no-op on drop.
///
/// # Safety contract
///
/// Reading uninitialized memory is UB per Rust's memory model regardless of
/// whether all bit patterns are valid for `T`. Caller must ensure every slot
/// is written before any read.
// `uninit_vec` flags exactly this pattern; here it is the deliberate purpose of
// the function (the safety contract above is what makes it sound).
#[allow(clippy::uninit_vec)]
pub(crate) fn alloc_uninit_vec<T: Copy>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    // SAFETY:
    // - capacity == n was just allocated, so set_len(n) is in bounds.
    // - T: Copy implies !Drop, so leaking uninit elements is a no-op.
    // - Caller upholds write-before-read.
    unsafe {
        v.set_len(n);
    }
    v
}

/// Compatibility shim — same as `alloc_uninit_vec::<F128>(n)`.
pub(crate) fn alloc_uninit_f128_vec(n: usize) -> Vec<F128> {
    alloc_uninit_vec::<F128>(n)
}

/// At/above this round width (in summed elements) a sumcheck round uses the full
/// thread pool; below it, [`sumcheck_round_min_len`] caps the fan-out.
pub(crate) const SUMCHECK_PAR_THRESHOLD: usize = 1 << 12;

/// Width-aware parallel fan-out for a sumcheck round: the number of rayon jobs
/// (≈ engaged threads) worth using for a round of `pairs` elements. Per-job
/// dispatch/join overhead is roughly constant (≈ one round's-worth of work, on
/// the order of 128 elements) while useful work is linear in `pairs`, so the
/// round time `pairs/T + c·T` is minimised at `T ≈ √(pairs/128)`. Clamped to the
/// pool size; `1` means "not worth parallelising". Empirically tuned on an
/// 8-P-core M-series; the √-shape is machine-independent, only the constant
/// shifts, so it degrades gracefully.
pub(crate) fn round_fanout(pairs: usize) -> usize {
    (pairs / 128).isqrt().clamp(1, rayon::current_num_threads())
}

/// Rayon `with_min_len` value for parallelising a sumcheck round of `pairs`
/// elements over `n_blocks` jobs, or `None` to run serial. Wide rounds keep the
/// full split (`min_len = 1`, the biggest rounds where dispatch is negligible);
/// mid rounds cap the job count to [`round_fanout`] so they don't pay full 8-way
/// overhead on too little work; rounds the fan-out deems too small run serial.
pub(crate) fn sumcheck_round_min_len(pairs: usize, n_blocks: usize) -> Option<usize> {
    if pairs >= SUMCHECK_PAR_THRESHOLD {
        Some(1)
    } else {
        match round_fanout(pairs) {
            0 | 1 => None,
            t => Some(n_blocks.div_ceil(t)),
        }
    }
}

/// At/above this fold width a fold uses the full thread pool; below it,
/// [`fold_min_len`] caps the fan-out.
///
/// Overridable via `FLOCK_FOLD_GATE` for tuning. Measured on M4 Max (10 P-core
/// pool) with `product_gkr::tests::fold_scaling_probe`: the full-split branch
/// scales well (2^17 outputs run 3.2× faster than serial), but the capped
/// fan-out branch below the gate *loses* to serial — 1.3× slower at 2^15
/// outputs, 3× slower at 2^13. Lowering the gate hands those widths the
/// full split instead of the cap.
pub(crate) const FOLD_PAR_THRESHOLD_DEFAULT: usize = 1 << 16;

pub(crate) fn fold_par_threshold() -> usize {
    static GATE: OnceLock<usize> = OnceLock::new();
    *GATE.get_or_init(|| {
        var("FLOCK_FOLD_GATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(FOLD_PAR_THRESHOLD_DEFAULT)
    })
}

/// Rayon `with_min_len` value for the fold/bind step over `half` outputs, or
/// `None` to run serial. Folds at or above [`fold_par_threshold`] take the full
/// split; narrower ones run serial by default, or cap the job count via the
/// √-rule when [`fold_sqrt_rule`] is on. The fold is a lighter, more
/// bandwidth-bound kernel than a round message (~2 muls + 1 write per output vs
/// ~10 muls/pair), so for a given width it saturates at fewer threads — hence
/// the √-rule constant of ~1024 (`T ≈ √(half/1024)`) against 128 for rounds.
pub(crate) fn fold_min_len(half: usize) -> Option<usize> {
    if half >= fold_par_threshold() {
        Some(1)
    } else if fold_sqrt_rule() {
        match (half / 1024).isqrt().clamp(1, rayon::current_num_threads()) {
            0 | 1 => None,
            t => Some(half.div_ceil(t)),
        }
    } else {
        None
    }
}

/// Whether sub-gate folds use the capped √-rule fan-out (`FLOCK_FOLD_RULE=sqrt`)
/// rather than running serial.
///
/// Serial is the default because the cap measured *worse than serial* on this
/// crate's only `fold_min_len` consumer, `product_gkr`: per
/// `fold_scaling_probe` on a 10-P-core M4 Max, the capped branch is 1.3× slower
/// than serial at 2^15 outputs and 3× slower at 2^13, and switching those
/// widths to serial cut the end-to-end fold phase from ~5.0 ms to ~4.6 ms at
/// μ=20. The √-rule was originally tuned for `logup_gkr`'s fold on an 8-P-core
/// part; that module is not in this tree, so the knob preserves it rather than
/// deleting it.
pub(crate) fn fold_sqrt_rule() -> bool {
    static RULE: OnceLock<bool> = OnceLock::new();
    *RULE.get_or_init(|| var("FLOCK_FOLD_RULE").is_ok_and(|v| v == "sqrt"))
}

/// A length-`n` all-zero vector from `alloc_zeroed` — LAZY zero pages from
/// the OS for large allocations, so untouched regions cost nothing.
/// (`vec![T::ZERO; n]` does NOT get this for custom structs: the zero-value
/// specialization only fires for built-in types, so it eagerly memsets.)
pub(crate) fn alloc_zeroed_vec<T: Copy>(n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    let layout = Layout::array::<T>(n).expect("allocation size overflows");
    // SAFETY:
    // - `alloc_zeroed` returns `n * size_of::<T>()` zeroed bytes with the
    //   layout's alignment (or null, handled below).
    // - T: Copy (no Drop) and the all-zero bit pattern must be a valid T —
    //   true for the plain-old-data field/word types this crate uses it for.
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Vec::from_raw_parts(ptr, n, n)
    }
}

/// Cached [`perf_core_count`]. The uncached version may spawn `sysctl`; this
/// memoizes it so hot paths can cheaply ask "is the current rayon pool the
/// homogeneous P-core pool?" (i.e. `current_num_threads() <= this`).
#[cfg(target_arch = "aarch64")]
pub(crate) fn perf_core_count_cached() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(perf_core_count)
}

/// Best-effort count of **physical** performance cores used to size the
/// prover's thread pool. The hot phases are CLMUL-heavy and/or
/// memory-bandwidth-bound; SMT siblings share the core's execution ports and
/// add no DRAM bandwidth, so running 2 threads per physical core only adds
/// contention (on a 32C/64T Threadripper the prove is ~16% faster at 32 threads
/// than 64). On macOS, queries `hw.perflevel0.physicalcpu` (= P-core count on
/// Apple silicon, = physical CPU count on Intel). On Linux, `available_
/// parallelism()` counts SMT siblings, so derive physical cores from `/sys`
/// topology and clamp that host-wide count to the process's affinity/cgroup
/// availability. Elsewhere, falls back to `available_parallelism()`.
fn perf_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
            && let Ok(s) = from_utf8(&out.stdout)
            && let Ok(n) = s.trim().parse::<usize>()
            && n > 0
        {
            return n;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(n) = linux_physical_cores()
            && n > 0
        {
            let available = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            return n.min(available);
        }
    }
    available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Count distinct physical cores via `/sys` topology: one entry per unique
/// `(physical_package_id, core_id)` over the online `cpuN` directories. Returns
/// `None` if the topology can't be read (caller falls back to logical count).
#[cfg(target_os = "linux")]
fn linux_physical_cores() -> Option<usize> {
    use std::collections::HashSet;
    let mut cores: HashSet<(String, String)> = HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let Some(rest) = name.strip_prefix("cpu") else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue; // skip "cpufreq", "cpuidle", etc.
        }
        let topo = path.join("topology");
        let core_id = std::fs::read_to_string(topo.join("core_id")).ok();
        let pkg = std::fs::read_to_string(topo.join("physical_package_id")).ok();
        if let (Some(c), Some(p)) = (core_id, pkg) {
            cores.insert((p.trim().to_owned(), c.trim().to_owned()));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}
