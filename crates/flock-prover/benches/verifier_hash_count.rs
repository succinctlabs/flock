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

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
}

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
    let (squeezes, pow) = fs_count::snapshot();
    let sha_total = leaf_compr + 2 * pair_calls + pow;
    // BLAKE3 estimate: 1 compression per 64-byte block, 1 parent per 1 KiB
    // chunk, ~2 extra per squeeze for finalization of the pending state.
    let blake3_est = blake3_bytes.div_ceil(64) + blake3_bytes.div_ceil(1024) + 2 * squeezes;
    println!("  [{label}]");
    println!("    SHA-256 leaf hashes : {leaf_calls:>8} calls = {leaf_compr:>8} compressions");
    println!(
        "    SHA-256 pair hashes : {pair_calls:>8} calls = {:>8} compressions",
        2 * pair_calls
    );
    println!("    SHA-256 PoW checks  : {pow:>8} calls = {pow:>8} compressions");
    println!("    SHA-256 TOTAL       : {sha_total:>8} compressions");
    println!(
        "    BLAKE3 FS transcript: {blake3_bytes:>8} bytes absorbed, {squeezes} squeezes ≈ {blake3_est} compressions"
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
