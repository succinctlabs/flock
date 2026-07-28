//! Hash + Merkle-tree throughput sanity check, run for **both** Merkle hashes
//! so the `merkle_hash` config option can be chosen on measurements rather
//! than on folklore.
//!
//! Three measurements, each repeated per hash:
//! 1. **Raw streaming hash**: one finalize over a large contiguous buffer.
//!    Reports the per-byte cost of the hash itself — SHA-256 should hit
//!    ~2.6 GB/s on M4 Max with HW acceleration, ~0.5 GB/s on the software
//!    fallback (footgun: needs `features = ["asm"]` in Cargo.toml).
//! 2. **Per-leaf hash**: digest one leaf at a time, summed over many leaves
//!    (= what `merkle_tree`'s leaf level does sequentially).
//! 3. **Merkle tree (parallel)**: full tree build over a buffer matching the
//!    PCS commit @ m=29 leaf count.
//!
//! The two hashes are not expected to win uniformly: SHA-256 has the four-way
//! hardware path and BLAKE3 does not, but BLAKE3 costs one compression per
//! 64-byte internal node against SHA-256's two, and SIMD-parallelizes within
//! leaves past 1 KiB. Leaf size is what decides it, so the tree bench sweeps
//! 16 B (one F128) through 512 B (the default `log_batch_size = 5` geometry).
//!
//! Run: `cargo bench --bench merkle`

use std::time::Instant;

use flock_prover::merkle::{HashKind, hash_leaf, merkle_tree};
use sha2::{Digest, Sha256};

const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

fn fmt_secs(s: f64) -> String {
    if s < 1e-3 {
        format!("{:>6.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:>6.2} ms", s * 1e3)
    } else {
        format!("{:>6.3} s ", s)
    }
}

fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b < 1024.0 {
        format!("{:>6.0} B", b)
    } else if b < 1024.0 * 1024.0 {
        format!("{:>5.1} KB", b / 1024.0)
    } else if b < 1024.0 * 1024.0 * 1024.0 {
        format!("{:>5.1} MB", b / 1024.0 / 1024.0)
    } else {
        format!("{:>5.2} GB", b / 1024.0 / 1024.0 / 1024.0)
    }
}

fn report(label: &str, secs: f64, bytes: u64, digests: u64) {
    let gb_per_s = bytes as f64 / (1024.0 * 1024.0 * 1024.0) / secs;
    let ns_per_byte = secs * 1e9 / bytes as f64;
    let mh_per_s = digests as f64 / secs / 1e6;
    println!(
        "  {:<50}  {}  {:>6.2} GB/s  {:>6.2} ns/B  {:>7.2} Mhash/s",
        label,
        fmt_secs(secs),
        gb_per_s,
        ns_per_byte,
        mh_per_s,
    );
}

fn header(name: &str) {
    println!("\n===== {name} =====");
}

fn alloc_pattern(bytes: usize, seed: u64) -> Vec<u8> {
    let mut buf = vec![0u8; bytes];
    // SplitMix-style fill (parallelism unnecessary for a one-shot bench setup).
    let mut state = seed;
    for chunk in buf.chunks_mut(8) {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let v = z ^ (z >> 31);
        let bytes_to_write = chunk.len().min(8);
        chunk[..bytes_to_write].copy_from_slice(&v.to_le_bytes()[..bytes_to_write]);
    }
    buf
}

/// Raw one-shot hash over the whole buffer.
fn bench_streaming(bytes: usize, kind: HashKind) {
    let data = alloc_pattern(bytes, 0xCAFE);
    let once = |d: &[u8]| -> [u8; 32] {
        match kind {
            HashKind::Sha256 => Sha256::digest(d).into(),
            HashKind::Blake3 => *blake3::hash(d).as_bytes(),
        }
    };
    // Warm-up.
    let _ = once(&data);

    let t0 = Instant::now();
    let digest = once(&data);
    let secs = t0.elapsed().as_secs_f64();
    let cs: u64 = u64::from_le_bytes(digest[..8].try_into().unwrap());
    report(
        &format!("streaming {kind} ({})", fmt_bytes(bytes as u64)),
        secs,
        bytes as u64,
        1,
    );
    // Sanity: bind to a checksum so the optimizer can't elide.
    eprintln!("  (digest cs: {:016x})", cs);
}

/// Per-leaf hash: call `hash_leaf` once per leaf, sequentially. This is what
/// the Merkle leaf level does (modulo parallelism).
fn bench_per_leaf(num_leaves: usize, leaf_size: usize, kind: HashKind) {
    let total = num_leaves * leaf_size;
    let data = alloc_pattern(total, 0xBEEF);
    // Warm-up: hash a few leaves.
    for chunk in data.chunks(leaf_size).take(8) {
        let _ = hash_leaf(chunk, kind);
    }

    let mut cs: u64 = 0;
    let t0 = Instant::now();
    for chunk in data.chunks(leaf_size) {
        let h = hash_leaf(chunk, kind);
        cs ^= u64::from_le_bytes(h[..8].try_into().unwrap());
    }
    let secs = t0.elapsed().as_secs_f64();
    report(
        &format!(
            "per-leaf hash_leaf {kind} (sequential, {} × {})",
            num_leaves,
            fmt_bytes(leaf_size as u64)
        ),
        secs,
        total as u64,
        num_leaves as u64,
    );
    eprintln!("  (leaves cs: {:016x})", cs);
}

/// How many timed tree builds to run per configuration. A single shot is far
/// too noisy to compare hashes with — run-to-run spread on an otherwise idle
/// M4 Max is ~10-15% — so take the best of several, which is the run least
/// perturbed by other work on the machine.
const TREE_RUNS: usize = 7;

/// Parallel Merkle tree (matches what `pcs::commit` calls at m=29).
/// Returns the *minimum* elapsed seconds over [`TREE_RUNS`] builds, so `main`
/// can print a head-to-head ratio that is not dominated by scheduling noise.
fn bench_merkle_tree(num_leaves: usize, leaf_size: usize, kind: HashKind) -> f64 {
    let total = num_leaves * leaf_size;
    let data = alloc_pattern(total, 0xDEAD);
    // Warm-up: one full build, so page faults and thread-pool spin-up are not
    // charged to the first measurement.
    let warm = merkle_tree(&data, num_leaves, kind);
    let cs: u64 = u64::from_le_bytes(warm.last().unwrap()[..8].try_into().unwrap());
    drop(warm);

    let mut best = f64::INFINITY;
    for _ in 0..TREE_RUNS {
        let t0 = Instant::now();
        let tree = merkle_tree(&data, num_leaves, kind);
        best = best.min(t0.elapsed().as_secs_f64());
        std::hint::black_box(&tree);
    }
    // Tree work = leaves + internal nodes; internal = num_leaves - 1 hashes.
    let total_hashes = (2 * num_leaves - 1) as u64;
    report(
        &format!(
            "merkle_tree {kind} (rayon, {} leaves × {})",
            num_leaves,
            fmt_bytes(leaf_size as u64)
        ),
        best,
        total as u64,
        total_hashes,
    );
    eprintln!("  (root cs: {:016x})", cs);
    best
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
    println!("(target: aarch64 + sha2 — four-way HW SHA-256 path active)");
    #[cfg(all(target_arch = "x86_64", target_feature = "sha"))]
    println!("(target: x86_64 + SHA-NI — four-way HW SHA-256 path active)");
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "sha2"),
        all(target_arch = "x86_64", target_feature = "sha")
    )))]
    println!("(target: software fallback path — HW SHA-256 NOT active)");

    header("Streaming hash (single digest over a large buffer)");
    for kind in KINDS {
        bench_streaming(64 * 1024, kind); // 64 KB — fits in L1
        bench_streaming(8 * 1024 * 1024, kind); // 8 MB — DRAM-ish
        bench_streaming(128 * 1024 * 1024, kind); // 128 MB — PCS m=29 codeword
    }

    header("Per-leaf hash (one call per 16-B leaf — leaf level cost)");
    for kind in KINDS {
        bench_per_leaf(8 * 1024 * 1024, 16, kind); // 8M leaves × 16 B = 128 MB
    }

    header("Merkle tree (parallel, matches PCS commit @ m=29 geometry)");
    // m=29: codeword = 2^23 F128 = 128 MB. Sweep leaf size, since that is what
    // decides which hash wins: 16 B = one F128 per leaf, 512 B / 1 KB = the
    // log_batch_size = 5 / 6 geometries `pcs::commit` actually uses, 4 KB =
    // past BLAKE3's 1 KiB chunk boundary, where its leaves stop batching.
    for (num_leaves, leaf_size) in [
        (1usize << 23, 16usize),
        (1 << 18, 512),
        (1 << 17, 1024),
        (1 << 15, 4096),
    ] {
        let mut secs = [0.0f64; KINDS.len()];
        for (i, kind) in KINDS.into_iter().enumerate() {
            secs[i] = bench_merkle_tree(num_leaves, leaf_size, kind);
        }
        let (sha, blake) = (secs[0], secs[1]);
        let (faster, ratio) = if blake < sha {
            ("blake3", sha / blake)
        } else {
            ("sha256", blake / sha)
        };
        println!(
            "  → {} leaves × {}: {faster} faster by {ratio:.2}×",
            num_leaves,
            fmt_bytes(leaf_size as u64)
        );
    }
}
