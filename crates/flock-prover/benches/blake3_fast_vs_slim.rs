//! BLAKE3 Ligerito proof: fast / slim / secure profiles at m=30 (default
//! K = 2^16 compressions). Reports m/t prove throughput, verify, and proof
//! size for each profile via `Blake3Setup::with_profile`. Warm best-of-N (the
//! minimum filters background load ≈ true compute); set FLOCK_BENCH_RUNS.
//!
//!   fast:   rate 1/2, Johnson + OOD, 100-bit round-by-round
//!   slim:   rate 1/4, Johnson + OOD, 100-bit (smaller proof, slower prove)
//!   secure: rate 1/2, unique-decoding regime, 120-bit
//!
//! Run: `cargo bench --bench blake3_fast_vs_slim`  (ST: RAYON_NUM_THREADS=1)

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression, K_LOG, min_n_blocks_log};

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

fn fmt_ms(s: f64) -> String {
    format!("{:.2} ms", s * 1000.0)
}
fn fmt_kb(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.1} KB", b as f64 / 1024.0)
    }
}

fn bench_mode(
    n_blocks: usize,
    profile: flock_prover::pcs::ligerito::LigeritoProfile,
    n_runs: usize,
    label: &str,
) {
    let setup = Blake3Setup::with_profile(n_blocks, profile);
    let mut rng = Rng::new(0xB1A_3_511_3E ^ (label.len() as u64));
    let blocks: Vec<Compression> = (0..n_blocks)
        .map(|_| random_compression(&mut rng))
        .collect();

    // Warm-up: page-in the large prover buffers + prime caches so the timed
    // runs measure compute, not cold-start. Keep its proof for verify timing.
    let mut ch = FsChallenger::new(b"flock-bench-v0");
    let (proof, commitment, _) = setup.prove_fast(&blocks, &mut ch);
    black_box(&proof);

    // Best-of: the minimum over many runs is the least-contended sample, which
    // filters background load and ≈ the true compute time.
    let mut prove_t = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&blocks, &mut ch);
        prove_t = prove_t.min(t0.elapsed().as_secs_f64());
        black_box(&p);
    }

    let mut verify_t = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch_v = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let _ = setup
            .verify(&commitment, &proof, &mut ch_v)
            .expect("verify");
        verify_t = verify_t.min(t0.elapsed().as_secs_f64());
    }

    let bundle = flock_prover::proof_io::R1csProofBundleLigerito { commitment, proof };
    let size = bundle.to_bytes().len();
    black_box(&bundle);

    println!(
        "  {label:>6}:  prove = {}   verify = {}   size = {}",
        fmt_ms(prove_t),
        fmt_ms(verify_t),
        fmt_kb(size),
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    let label = if threads == 1 {
        "ST".to_string()
    } else {
        format!("MT, {threads} threads")
    };

    let n_blocks: usize = std::env::var("BLAKE3_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536);
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    let n_runs: usize = std::env::var("FLOCK_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("BLAKE3 Ligerito fast vs slim — K = {n_blocks} (m = {m}, {label})\n");

    use flock_prover::pcs::ligerito::LigeritoProfile;
    bench_mode(n_blocks, LigeritoProfile::Fast, n_runs, "fast");
    bench_mode(n_blocks, LigeritoProfile::Slim, n_runs, "slim");
    bench_mode(n_blocks, LigeritoProfile::Secure, n_runs, "secure");
}
