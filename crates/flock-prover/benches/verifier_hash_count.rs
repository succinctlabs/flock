//! Count hash compression calls during VERIFICATION, exactly, by running a
//! real prove → verify roundtrip with the `hash-count` instrumentation.
//!
//!   cargo bench --bench verifier_hash_count --features hash-count
//!
//! Workload is the BLAKE3 R1CS (the witness contents don't affect verifier
//! hash counts — only m, the backend, and the rate profile do). Select runs
//! with e.g. `VHC_RUNS="lig:22:1,bf:22:1"`; entries are `<backend>:<m>:<rate>`
//! with backend ∈ {bf, lig}.
//!
//! Reported per run:
//!   - SHA-256 Merkle leaf hashes (calls + compressions; a leaf of L bytes is
//!     ceil((L+9)/64) compressions)
//!   - SHA-256 Merkle path/pair hashes (2 compressions each)
//!   - SHA-256 PoW checks (1 compression each)
//!   - BLAKE3 Fiat–Shamir absorption (bytes + squeezes, ≈ compressions)

use flock_prover::challenger::{FsChallenger, fs_count};
use flock_prover::merkle::hash_count;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression, K_LOG};

use flock_core::test_rng::Rng;

fn random_compression(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
}

fn reset_counters() {
    hash_count::reset();
    fs_count::reset();
}

fn report(label: &str, blake3_bytes: u64) {
    let (leaf_calls, leaf_compr, pair_calls) = hash_count::snapshot();
    let (squeezes, squeezed_bytes, pow) = fs_count::snapshot();
    let sha_total = leaf_compr + 2 * pair_calls + pow;

    // BLAKE3 estimate, broken out because the FS chain table has to be sized
    // off the parts, not the total — they differ in how they parallelize.
    //
    //   absorb blocks : 1 compression per 64-byte input block. Sequential
    //                   WITHIN a 1 KiB chunk (16 blocks), independent across
    //                   chunks.
    //   parent tree   : ~1 parent compression per chunk, combining chunk CVs.
    //   finalizations : 1 per squeeze, for the pending state at that point.
    //                   These are what serialize the transcript, since each
    //                   squeeze's output is re-absorbed.
    //   xof output    : 1 compression per 64 bytes of squeezed output,
    //                   counter-mode off the root CV, so mutually INDEPENDENT.
    //
    // The old form charged a flat `2 * squeezes`, which silently assumed every
    // squeeze produced ≤ 64 bytes of output. True while each was a 16-byte
    // `sample_f128`; wrong by ~60x per level once query sampling became one
    // batched `sample_f128_vec` (3888 bytes at L0). Hence the explicit split.
    let absorb_blocks = blake3_bytes.div_ceil(64);
    let parent_tree = blake3_bytes.div_ceil(1024);
    let finalizations = squeezes;
    let xof_output = squeezed_bytes.div_ceil(64);
    let blake3_est = absorb_blocks + parent_tree + finalizations + xof_output;

    println!("  [{label}]");
    println!("    SHA-256 leaf hashes : {leaf_calls:>8} calls = {leaf_compr:>8} compressions");
    println!(
        "    SHA-256 pair hashes : {pair_calls:>8} calls = {:>8} compressions",
        2 * pair_calls
    );
    println!("    SHA-256 PoW checks  : {pow:>8} calls = {pow:>8} compressions");
    println!("    SHA-256 TOTAL       : {sha_total:>8} compressions");
    println!(
        "    BLAKE3 FS transcript: {blake3_bytes:>8} bytes absorbed, {squeezes} squeezes, {squeezed_bytes} bytes squeezed"
    );
    println!(
        "      absorb blocks {absorb_blocks:>5} | parent tree {parent_tree:>4} | finalizations {finalizations:>4} | xof output {xof_output:>5}  ≈ {blake3_est} compressions"
    );
    println!(
        "    GRAND TOTAL (SHA-256 + BLAKE3 est.) ≈ {} compressions",
        sha_total + blake3_est
    );
}

fn run(m: usize, rate: usize) {
    assert!(m > K_LOG, "m must exceed K_LOG={K_LOG}");
    let n_blocks = 1usize << (m - K_LOG);
    println!("\n=== Ligerito m={m} log_inv_rate={rate} (BLAKE3 R1CS, K={n_blocks}) ===");

    let setup = Blake3Setup::with_log_inv_rate(n_blocks, rate);
    let mut rng = Rng::new(0xC0DE ^ (m as u64) << 8 ^ rate as u64);
    let blocks: Vec<Compression> = (0..n_blocks)
        .map(|_| random_compression(&mut rng))
        .collect();

    let t0 = std::time::Instant::now();
    let mut ch_p = FsChallenger::new(b"flock-hash-count");
    let (proof, commitment, _) = setup.prove_fast(&blocks, &mut ch_p);
    println!("  (prove: {:.1} s)", t0.elapsed().as_secs_f64());
    println!(
        "  PCS open proof: {} B (+ {} cap nodes in the commitment)",
        proof.pcs_open.ligerito.size_bytes(),
        commitment.cap.len(),
    );

    reset_counters();
    let mut ch_v = FsChallenger::new(b"flock-hash-count");
    let t1 = std::time::Instant::now();
    setup
        .verify(&commitment, &proof, &mut ch_v)
        .expect("lig verify");
    let dt = t1.elapsed().as_secs_f64();
    report("verify", ch_v.absorbed_bytes());
    println!("    (verify time: {:.2} ms)", dt * 1e3);
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let runs = std::env::var("VHC_RUNS").unwrap_or_else(|_| "22:1,30:1,30:2".to_string());
    for entry in runs.split(',') {
        let parts: Vec<&str> = entry.trim().split(':').collect();
        assert_eq!(parts.len(), 2, "bad VHC_RUNS entry {entry:?} (use m:rate)");
        let m: usize = parts[0].parse().expect("bad m");
        let rate: usize = parts[1].parse().expect("bad rate");
        run(m, rate);
    }
}
