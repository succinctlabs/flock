//! Keccak-f[1600] (3-wide encoder) proof — fast / slim / secure profiles at a
//! fixed count, reporting the columns of paper Table 4 (`tab:keccak-fixed`):
//! m/t throughput, proof size, and verify time. Warm best-of-N (the minimum
//! filters background load ≈ true compute).
//!
//! Default K = 24576 = 3·2^13 (the 3-wide sweet spot the paper labels "≈2^14");
//! override with KECCAK3_K. Runs (best-of) FLOCK_BENCH_RUNS, default 10.
//!
//! Run: `cargo bench --bench keccak3_profiles`   (ST: RAYON_NUM_THREADS=1)

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::ligerito::LigeritoProfile;
use flock_prover::r1cs_hashes::keccak::{STATE_BITS, State};
use flock_prover::r1cs_hashes::keccak3;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn random_state(rng: &mut Rng) -> State {
    let mut s = [false; STATE_BITS];
    let mut i = 0;
    while i < STATE_BITS {
        let w = rng.next_u64();
        for b in 0..64 {
            if i + b < STATE_BITS {
                s[i + b] = (w >> b) & 1 == 1;
            }
        }
        i += 64;
    }
    s
}

fn fmt_ms(s: f64) -> String {
    format!("{:.2} ms", s * 1000.0)
}
fn fmt_kb(b: usize) -> String {
    format!("{:.1} KiB", b as f64 / 1024.0)
}

fn bench_profile(n_keccaks: usize, profile: LigeritoProfile, n_runs: usize, label: &str) {
    let setup = keccak3::KeccakSetup::with_profile(n_keccaks, profile);
    let mut rng = Rng::new(0x0EAC_2024 ^ (label.len() as u64));
    let states: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();

    // Warm-up (page-in + caches); keep its proof for verify timing.
    let mut ch = FsChallenger::new(b"flock-bench-v0");
    let (proof, commitment, _) = setup.prove_fast(&states, &mut ch);
    black_box(&proof);

    let mut prove_t = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (p, _, _) = setup.prove_fast(&states, &mut ch);
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

    let kps = n_keccaks as f64 / prove_t / 1000.0;
    println!(
        "  {label:>6}:  m = {}   throughput = {:.1}k/s   prove = {}   verify = {}   proof = {}",
        setup.m(),
        kps,
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
    let n_keccaks: usize = std::env::var("KECCAK3_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24576);
    let n_runs: usize = std::env::var("FLOCK_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("Keccak-f[1600] 3-wide — fast / slim / secure at K = {n_keccaks} ({label})\n");
    bench_profile(n_keccaks, LigeritoProfile::Fast, n_runs, "fast");
    bench_profile(n_keccaks, LigeritoProfile::Slim, n_runs, "slim");
    bench_profile(n_keccaks, LigeritoProfile::Secure, n_runs, "secure");
}
