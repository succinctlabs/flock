//! SHA-256 Merkle-path proof generation benchmark.
//!
//! Run with:
//!   cargo bench --bench sha2_merkle_proof
//!   (or: cargo run --release --bench sha2_merkle_proof)
//!
//! Builds an honest length-K SHA-256 Merkle path (block i ≥ 1 hashes
//! (z_{i-1}, sibling_i) or (sibling_i, z_{i-1}) under the public bit b_i),
//! and times three paths back-to-back:
//!   - prove_fast       : base (no chain, no Merkle column-lincheck)
//!   - prove_chain      : straight-line chain (z_i = h(M_i) consistency)
//!   - prove_merkle_path: Merkle-path column-lincheck (per-row bit selector)
//! Reports both the chain overhead and the Merkle overhead over `prove_fast`,
//! and the Merkle-vs-chain delta. K_LOG=15 → m=29 at 16,384 blocks.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::sha2::{
    Compression, K_LOG, SHA256_IV, Sha256HybridSetup, min_n_blocks_log, sha256_compress,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1.0 {
        format!("{:>8.2} µs", s * 1e6)
    } else if ms < 1000.0 {
        format!("{:>8.2} ms", ms)
    } else {
        format!("{:>8.2} s ", s)
    }
}

/// Build an honest SHA-256 Merkle path of `n` compressions.
/// - Block 0: M = (leaf, sibling_0). z_0 = sha256_compress(IV, M).
/// - Block i ≥ 1: M = (z_{i-1}, sibling_i) if b_i=0 else (sibling_i, z_{i-1}).
///   z_i = sha256_compress(IV, M).
/// All blocks use the public SHA-256 IV as H_in.
fn honest_merkle_path(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8], Vec<bool>) {
    let mut rng = Rng::new(seed);
    let leaf: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
    let mut b_bits = vec![false; n];
    for bit in b_bits.iter_mut().skip(1) {
        *bit = rng.nx() & 1 == 1;
    }
    let mut blocks = Vec::with_capacity(n);
    let mut current = leaf;
    for i in 0..n {
        let sibling: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
        let mut m = [0u32; 16];
        if !b_bits[i] {
            m[..8].copy_from_slice(&current);
            m[8..].copy_from_slice(&sibling);
        } else {
            m[..8].copy_from_slice(&sibling);
            m[8..].copy_from_slice(&current);
        }
        blocks.push((SHA256_IV, m));
        current = sha256_compress(&SHA256_IV, &m);
    }
    let root = current;
    (blocks, leaf, root, b_bits)
}

fn bench_one(n_blocks: usize, n_runs: usize) {
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;

    println!("\n=== {n_blocks:>5} compressions  (m = {m}, slots = {n_slots}) ===");

    let (blocks, leaf, root, b) = honest_merkle_path(n_blocks, 0xD15EA5E ^ n_blocks as u64);
    let setup = Sha256HybridSetup::new(n_blocks);

    // Warm-up all three paths.
    {
        let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
        let (p, _, _) = setup.prove_fast(&blocks, &mut ch);
        black_box(&p);
        let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
        let (proof, comm) = setup.prove_chain(&blocks, &mut ch);
        black_box(&proof);
        black_box(&comm);
        let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
        let (proof, comm) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch);
        black_box(&proof);
        black_box(&comm);
    }

    // Best-of-n_runs prove_fast (base).
    let mut best_base = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
        let t = Instant::now();
        let (p, _, _) = setup.prove_fast(&blocks, &mut ch);
        best_base = best_base.min(t.elapsed().as_secs_f64());
        black_box(&p);
    }

    // Best-of-n_runs prove_chain.
    let mut best_chain = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
        let t = Instant::now();
        let (proof, comm) = setup.prove_chain(&blocks, &mut ch);
        best_chain = best_chain.min(t.elapsed().as_secs_f64());
        black_box(&proof);
        black_box(&comm);
    }

    // Best-of-n_runs prove_merkle_path.
    let mut best_merkle = f64::INFINITY;
    let mut single_proof_opt = None;
    let mut single_comm_opt = None;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
        let t = Instant::now();
        let (proof, comm) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch);
        best_merkle = best_merkle.min(t.elapsed().as_secs_f64());
        single_proof_opt = Some(proof);
        single_comm_opt = Some(comm);
    }
    let single_proof = single_proof_opt.unwrap();
    let single_comm = single_comm_opt.unwrap();

    // Best-of-3 verify_merkle_path (single-path).
    let mut best_verify_single = f64::INFINITY;
    for _ in 0..3 {
        let mut chv = FsChallenger::new(b"flock-merkle-bench-v0");
        let t = Instant::now();
        setup
            .verify_merkle_path_ligerito(&single_comm, &single_proof, &leaf, &root, &b, &mut chv)
            .expect("verify failed");
        best_verify_single = best_verify_single.min(t.elapsed().as_secs_f64());
    }
    black_box(&single_proof);
    black_box(&single_comm);

    let chain_over = best_chain - best_base;
    let merkle_over = best_merkle - best_base;
    let mvc = best_merkle - best_chain;
    println!(
        "  prove_fast        :  {}  ({:.0} comp/sec)",
        fmt_ms(best_base),
        n_blocks as f64 / best_base
    );
    println!(
        "  prove_chain       :  {}  ({:.0} comp/sec)  [+{:.1}% over base]",
        fmt_ms(best_chain),
        n_blocks as f64 / best_chain,
        100.0 * chain_over / best_base
    );
    println!(
        "  prove_merkle_path :  {}  ({:.0} comp/sec)  [+{:.1}% over base]",
        fmt_ms(best_merkle),
        n_blocks as f64 / best_merkle,
        100.0 * merkle_over / best_base
    );
    println!(
        "  Δ(merkle - chain) :  {}  ({:+.1}% of base)",
        fmt_ms(mvc),
        100.0 * mvc / best_base
    );
    println!("  verify_merkle_path:  {}", fmt_ms(best_verify_single));

    // -----------------------------------------------------------------------
    // Multi-path: same N, sweep path_log to show overhead is path-count-blind.
    // -----------------------------------------------------------------------
    // The witness/commitment cost is fixed at N compressions; we only change
    // how rows are interpreted as P paths of length L = N/P. Each path_log
    // value uses an honest "all paths identical" scenario built by replication.
    let max_path_log = n_log.min(4); // cap to keep the matrix readable
    if max_path_log >= 1 {
        println!("  -- multi-path (same N, varying P) --");
        for path_log in 1..=max_path_log {
            let n_paths = 1usize << path_log;
            let l = n_blocks / n_paths;
            // Build P identical copies of an honest length-L path → shared root.
            let (one_blocks, _leaf, _root, one_b) =
                honest_merkle_path(l, 0xD15EA5E ^ (n_blocks as u64 + path_log as u64));
            let mut blocks_p: Vec<Compression> = Vec::with_capacity(n_blocks);
            let mut b_p: Vec<bool> = Vec::with_capacity(n_blocks);
            for _ in 0..n_paths {
                blocks_p.extend_from_slice(&one_blocks);
                b_p.extend_from_slice(&one_b);
            }
            let leaves_p = vec![_leaf; n_paths];
            // Warm-up.
            {
                let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
                let (proof, comm) =
                    setup.prove_merkle_paths_ligerito(path_log, &blocks_p, &b_p, &mut ch);
                black_box(&proof);
                black_box(&comm);
            }
            let mut best_mp = f64::INFINITY;
            let mut last_proof = None;
            let mut last_comm = None;
            for _ in 0..n_runs {
                let mut ch = FsChallenger::new(b"flock-merkle-bench-v0");
                let t = Instant::now();
                let (proof, comm) =
                    setup.prove_merkle_paths_ligerito(path_log, &blocks_p, &b_p, &mut ch);
                best_mp = best_mp.min(t.elapsed().as_secs_f64());
                last_proof = Some(proof);
                last_comm = Some(comm);
            }
            let proof_p = last_proof.unwrap();
            let comm_p = last_comm.unwrap();

            // Best-of-3 verify.
            let mut best_vp = f64::INFINITY;
            for _ in 0..3 {
                let mut chv = FsChallenger::new(b"flock-merkle-bench-v0");
                let t = Instant::now();
                setup
                    .verify_merkle_paths_ligerito(
                        path_log, &comm_p, &proof_p, &leaves_p, &_root, &b_p, &mut chv,
                    )
                    .expect("verify failed");
                best_vp = best_vp.min(t.elapsed().as_secs_f64());
            }
            black_box(&proof_p);
            black_box(&comm_p);

            let over = best_mp - best_base;
            println!(
                "  prove P={:>4} L={:>5}:  {}  [+{:.1}% base]   verify: {}",
                n_paths,
                l,
                fmt_ms(best_mp),
                100.0 * over / best_base,
                fmt_ms(best_vp),
            );
        }
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!(
        "SHA-256 Merkle-path proof generation benchmark \
         (prove_merkle_path vs prove_chain vs prove_fast)."
    );
    println!("(honest path, warm-up + best-of-n_runs timing)");

    // Same m grid as the chain bench. n_compressions must be a power of 2 ≥ 8
    // (Merkle path uses the same no-padding constraint as the chain).
    for &(n, n_runs) in &[(8usize, 3), (64, 2), (4096, 2), (16384, 2), (32768, 2)] {
        bench_one(n, n_runs);
    }
}
