//! SHA-256 + Merkle-tree throughput sanity check.
//!
//! Three measurements:
//! 1. **Raw streaming SHA-256**: one `Sha256` finalize over a large contiguous
//!    buffer. Reports the per-byte cost of the hash itself — should hit
//!    ~2.6 GB/s on M4 Max with HW acceleration, ~0.5 GB/s on the software
//!    fallback (footgun: needs `features = ["asm"]` in Cargo.toml).
//! 2. **Per-leaf SHA-256**: digest one 16-byte leaf at a time, summed over
//!    many leaves (= what `merkle_tree`'s leaf level does sequentially).
//! 3. **Merkle tree (parallel)**: full tree build over a buffer matching the
//!    PCS commit @ m=29 leaf count.
//!
//! Run: `cargo bench --bench merkle`

use std::time::Instant;

use flock_prover::merkle::{hash_leaf, merkle_tree};
use sha2::{Digest, Sha256};

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

/// Raw SHA-256: one `digest` over the whole buffer.
fn bench_streaming_sha256(bytes: usize) {
    let data = alloc_pattern(bytes, 0xCAFE);
    // Warm-up.
    let _ = Sha256::digest(&data);

    let t0 = Instant::now();
    let digest = Sha256::digest(&data);
    let secs = t0.elapsed().as_secs_f64();
    let cs: u64 = u64::from_le_bytes(digest[..8].try_into().unwrap());
    report(
        &format!("streaming Sha256::digest ({})", fmt_bytes(bytes as u64)),
        secs,
        bytes as u64,
        1,
    );
    // Sanity: bind to a checksum so the optimizer can't elide.
    eprintln!("  (digest cs: {:016x})", cs);
}

/// Per-leaf SHA-256: call `hash_leaf` once per 16-byte leaf, sequentially.
/// This is what the Merkle leaf level does (modulo parallelism).
fn bench_per_leaf_sha256(num_leaves: usize, leaf_size: usize) {
    let total = num_leaves * leaf_size;
    let data = alloc_pattern(total, 0xBEEF);
    // Warm-up: hash a few leaves.
    for chunk in data.chunks(leaf_size).take(8) {
        let _ = hash_leaf(chunk);
    }

    let mut cs: u64 = 0;
    let t0 = Instant::now();
    for chunk in data.chunks(leaf_size) {
        let h = hash_leaf(chunk);
        cs ^= u64::from_le_bytes(h[..8].try_into().unwrap());
    }
    let secs = t0.elapsed().as_secs_f64();
    report(
        &format!(
            "per-leaf hash_leaf (sequential, {} × {})",
            num_leaves,
            fmt_bytes(leaf_size as u64)
        ),
        secs,
        total as u64,
        num_leaves as u64,
    );
    eprintln!("  (leaves cs: {:016x})", cs);
}

/// Parallel Merkle tree (matches what `pcs::commit` calls at m=29).
fn bench_merkle_tree(num_leaves: usize, leaf_size: usize) {
    let total = num_leaves * leaf_size;
    let data = alloc_pattern(total, 0xDEAD);
    // Warm-up: small tree.
    {
        let small = &data[..1024.min(total)];
        let _ = merkle_tree(small, 16);
    }

    let t0 = Instant::now();
    let tree = merkle_tree(&data, num_leaves);
    let secs = t0.elapsed().as_secs_f64();
    let root = *tree.last().unwrap();
    let cs: u64 = u64::from_le_bytes(root[..8].try_into().unwrap());
    // Tree work = leaves + internal nodes; internal = num_leaves - 1 hashes.
    let total_hashes = (2 * num_leaves - 1) as u64;
    report(
        &format!(
            "merkle_tree (rayon, {} leaves × {})",
            num_leaves,
            fmt_bytes(leaf_size as u64)
        ),
        secs,
        total as u64,
        total_hashes,
    );
    eprintln!("  (root cs: {:016x})", cs);
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

    header("Streaming SHA-256 (single digest over a large buffer)");
    bench_streaming_sha256(64 * 1024); // 64 KB — fits in L1
    bench_streaming_sha256(8 * 1024 * 1024); // 8 MB — DRAM-ish
    bench_streaming_sha256(128 * 1024 * 1024); // 128 MB — matches PCS m=29 codeword

    header("Per-leaf SHA-256 (one call per 16-B leaf — leaf level cost)");
    bench_per_leaf_sha256(8 * 1024 * 1024, 16); // 8M leaves × 16 B = 128 MB

    header("Merkle tree (parallel, matches PCS commit @ m=29 geometry)");
    // m=29: codeword = 2^23 F128 = 128 MB. Leaves are 1 F128 = 16 B each, 2^23 leaves.
    bench_merkle_tree(1 << 23, 16);
}
