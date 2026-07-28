//! Where BLAKE3's Merkle cost actually goes, and which of the crate's entry
//! points is fastest for our two node shapes.
//!
//! `benches/merkle.rs` showed BLAKE3 losing to the four-way hardware SHA-256
//! path by 1.6–2.25×. This probe isolates why, by timing the candidate
//! primitives for the two things `merkle_tree` does:
//!
//! - **Parent nodes** (two 32-byte CVs → 32 bytes), `num_leaves − 1` of them:
//!   1. `blake3::hash(l || r)` — what the naive port does: a full `Hasher`
//!      setup, one chunk, one compression.
//!   2. `hazmat::merge_subtrees_non_root` — BLAKE3's own parent-node
//!      compression. Stable public API, one compression, no chunk state.
//!   3. `platform::hash_many` with PARENT flags — the same compression, but
//!      four (NEON) at a time. `#[doc(hidden)]` "unstable, benchmarks only".
//!
//! - **Leaves** (`leaf_size` bytes → 32 bytes), `num_leaves` of them:
//!   1. `blake3::hash(leaf)` — one at a time.
//!   2. `platform::hash_many` with chunk flags — batched, but only usable when
//!      `leaf_size` is a multiple of 64 and ≤ 1024 (so: the default 512-byte
//!      `log_batch_size = 5` geometry yes, 16-byte leaves no).
//!
//! Run: `cargo bench --bench blake3_node_probe`

use std::hint::black_box;
use std::time::Instant;

use blake3::IncrementCounter;
use blake3::hazmat::{Mode, merge_subtrees_non_root};
use blake3::platform::{Platform, le_bytes_from_words_32, words_from_le_bytes_32};

/// BLAKE3 domain flags (blake3 does not re-export these; they are fixed by the
/// spec: CHUNK_START = 1, CHUNK_END = 2, PARENT = 4).
const CHUNK_START: u8 = 1;
const CHUNK_END: u8 = 2;
const PARENT: u8 = 4;

/// BLAKE3's IV, the key words for unkeyed hashing.
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

fn rep(label: &str, secs: f64, n: u64) {
    println!(
        "  {:<52}  {:>8.2} ms  {:>7.2} ns/node  {:>7.2} Mnode/s",
        label,
        secs * 1e3,
        secs * 1e9 / n as f64,
        n as f64 / secs / 1e6,
    );
}

fn pattern(bytes: usize, seed: u64) -> Vec<u8> {
    let mut buf = vec![0u8; bytes];
    let mut s = seed;
    for c in buf.chunks_mut(8) {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let v = z ^ (z >> 31);
        let n = c.len().min(8);
        c[..n].copy_from_slice(&v.to_le_bytes()[..n]);
    }
    buf
}

/// Parent nodes: `n` pairs of 32-byte children, three ways.
fn bench_parents(n: usize) {
    println!("\n===== Parent nodes ({n} × 64 B → 32 B) =====");
    let children = pattern(n * 64, 0xA11CE);
    let plat = Platform::detect();
    println!(
        "  (platform: {plat:?}, simd_degree = {})",
        plat.simd_degree()
    );

    // 1. blake3::hash over the 64-byte concatenation.
    let t = Instant::now();
    let mut acc = 0u64;
    for pair in children.chunks_exact(64) {
        let h = blake3::hash(pair);
        acc ^= u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap());
    }
    rep("blake3::hash(l || r)", t.elapsed().as_secs_f64(), n as u64);
    black_box(acc);

    // 2. hazmat parent-node compression (stable API, scalar).
    let t = Instant::now();
    let mut acc = 0u64;
    for pair in children.chunks_exact(64) {
        let l: &[u8; 32] = pair[..32].try_into().unwrap();
        let r: &[u8; 32] = pair[32..].try_into().unwrap();
        let cv = merge_subtrees_non_root(l, r, Mode::Hash);
        acc ^= u64::from_le_bytes(cv[..8].try_into().unwrap());
    }
    rep(
        "hazmat::merge_subtrees_non_root (stable)",
        t.elapsed().as_secs_f64(),
        n as u64,
    );
    black_box(acc);

    // 3. hash_many with PARENT flags — N-at-a-time, N = simd_degree.
    let deg = plat.simd_degree();
    let mut out = vec![0u8; deg * 32];
    let t = Instant::now();
    let mut acc = 0u64;
    for group in children.chunks(deg * 64) {
        let blocks: Vec<&[u8; 64]> = group
            .chunks_exact(64)
            .map(|b| <&[u8; 64]>::try_from(b).unwrap())
            .collect();
        plat.hash_many(
            &blocks,
            &IV,
            0,
            IncrementCounter::No,
            PARENT,
            0,
            0,
            &mut out[..blocks.len() * 32],
        );
        acc ^= u64::from_le_bytes(out[..8].try_into().unwrap());
    }
    rep(
        &format!("platform::hash_many PARENT ({deg}-way, unstable)"),
        t.elapsed().as_secs_f64(),
        n as u64,
    );
    black_box(acc);
}

/// Leaves of `LEAF` bytes, scalar vs batched. `LEAF` must be a multiple of 64
/// and ≤ 1024 for the batched path to apply.
fn bench_leaves<const LEAF: usize>(n: usize) {
    println!("\n===== Leaves ({n} × {LEAF} B → 32 B) =====");
    let data = pattern(n * LEAF, 0xB0B);
    let plat = Platform::detect();

    // 1. One at a time.
    let t = Instant::now();
    let mut acc = 0u64;
    for leaf in data.chunks_exact(LEAF) {
        let h = blake3::hash(leaf);
        acc ^= u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap());
    }
    rep("blake3::hash(leaf)", t.elapsed().as_secs_f64(), n as u64);
    black_box(acc);

    // 2. Batched via hash_many, one chunk per leaf (CHUNK_START | CHUNK_END).
    let deg = plat.simd_degree();
    let mut out = vec![0u8; deg * 32];
    let t = Instant::now();
    let mut acc = 0u64;
    for group in data.chunks(deg * LEAF) {
        let leaves: Vec<&[u8; LEAF]> = group
            .chunks_exact(LEAF)
            .map(|b| <&[u8; LEAF]>::try_from(b).unwrap())
            .collect();
        plat.hash_many(
            &leaves,
            &IV,
            0,
            IncrementCounter::No,
            0,
            CHUNK_START,
            CHUNK_END,
            &mut out[..leaves.len() * 32],
        );
        acc ^= u64::from_le_bytes(out[..8].try_into().unwrap());
    }
    rep(
        &format!("platform::hash_many chunk ({deg}-way, unstable)"),
        t.elapsed().as_secs_f64(),
        n as u64,
    );
    black_box(acc);
}

/// Sanity: the batched parent path must agree with the stable scalar one.
fn check_parent_agreement() {
    let plat = Platform::detect();
    let children = pattern(4 * 64, 7);
    let blocks: Vec<&[u8; 64]> = children
        .chunks_exact(64)
        .map(|b| <&[u8; 64]>::try_from(b).unwrap())
        .collect();
    let mut out = [0u8; 4 * 32];
    plat.hash_many(
        &blocks,
        &IV,
        0,
        IncrementCounter::No,
        PARENT,
        0,
        0,
        &mut out,
    );
    for (i, pair) in children.chunks_exact(64).enumerate() {
        let l: &[u8; 32] = pair[..32].try_into().unwrap();
        let r: &[u8; 32] = pair[32..].try_into().unwrap();
        let want = merge_subtrees_non_root(l, r, Mode::Hash);
        assert_eq!(
            &out[i * 32..(i + 1) * 32],
            &want[..],
            "hash_many PARENT disagrees with hazmat at lane {i}"
        );
    }
    // Round-trip the word helpers so an API change here is caught loudly.
    let w = words_from_le_bytes_32(&[0xAB; 32]);
    assert_eq!(le_bytes_from_words_32(&w), [0xAB; 32]);
    println!("(batched PARENT path agrees with hazmat::merge_subtrees_non_root)");

    // And the batched chunk path must agree with `finalize_non_root`, the
    // stable-API definition of a non-root leaf CV, for every leaf size that
    // fits in one chunk. That equality is what lets the fast path exist.
    check_leaf_agreement::<64>();
    check_leaf_agreement::<512>();
    check_leaf_agreement::<1024>();
    println!("(batched chunk path agrees with HasherExt::finalize_non_root)");
}

fn check_leaf_agreement<const LEAF: usize>() {
    use blake3::hazmat::HasherExt;
    let plat = Platform::detect();
    let data = pattern(4 * LEAF, 99);
    let leaves: Vec<&[u8; LEAF]> = data
        .chunks_exact(LEAF)
        .map(|b| <&[u8; LEAF]>::try_from(b).unwrap())
        .collect();
    let mut out = [0u8; 4 * 32];
    plat.hash_many(
        &leaves,
        &IV,
        0,
        IncrementCounter::No,
        0,
        CHUNK_START,
        CHUNK_END,
        &mut out,
    );
    for (i, leaf) in data.chunks_exact(LEAF).enumerate() {
        let want: [u8; 32] = blake3::Hasher::new().update(leaf).finalize_non_root();
        assert_eq!(
            &out[i * 32..(i + 1) * 32],
            &want[..],
            "hash_many chunk disagrees with finalize_non_root at LEAF={LEAF} lane {i}"
        );
    }
}

fn main() {
    check_parent_agreement();
    // ~2^22 nodes keeps each measurement in the tens of ms.
    bench_parents(1 << 22);
    bench_leaves::<64>(1 << 22);
    bench_leaves::<512>(1 << 19);
    bench_leaves::<1024>(1 << 18);
}
