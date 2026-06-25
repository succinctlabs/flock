//! Combined probe for the two non-PCS prover phases outside zerocheck:
//! BLAKE3 witness generation (`generate_witness_with_ab_packed_and_lincheck`)
//! and the real lincheck prover (`prove_padded_capture_z_vec` with the CSC
//! circuit), at the blake3 m=30 prove_fast shapes.
//!
//! Usage:
//!   cargo bench --bench witness_lincheck_probe --no-run
//!   ./target/release/deps/witness_lincheck_probe-<hash> [n_runs] [n_blocks]
//!   RAYON_NUM_THREADS=1 ... for ST.
//!
//! Prints best-of-N per phase, their sum, and FNV checksums over all phase
//! outputs. Inputs are seeded, so the checksums must be bit-stable across
//! any valid optimization.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::field::F128;
use flock_prover::lincheck::{QuirkyPoint, prove_padded_capture_z_vec};
use flock_prover::r1cs_hashes::blake3::{
    Blake3Setup, Compression, K_SKIP, generate_witness_with_ab_packed_and_lincheck,
    min_n_blocks_log,
};

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
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn random_compression(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x100000001b3);
        }
    }
    fn f128s(&mut self, v: &[F128]) {
        for x in v {
            self.bytes(&x.lo.to_le_bytes());
            self.bytes(&x.hi.to_le_bytes());
        }
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let n_runs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let n_blocks: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536);

    let n_log = min_n_blocks_log(n_blocks);
    let setup = Blake3Setup::new(n_blocks);
    let r1cs = &setup.r1cs;
    let m = r1cs.m;
    let circuit = r1cs.csc_lincheck_circuit();

    let mut rng = Rng::new(0xB1A3_C0DE);
    let blocks: Vec<Compression> = (0..n_blocks)
        .map(|_| random_compression(&mut rng))
        .collect();

    // Fixed claim point with the real blake3 dims.
    let x_ab = QuirkyPoint {
        z_skip: rng.f128(),
        x_inner_rest: (0..r1cs.k_log - K_SKIP).map(|_| rng.f128()).collect(),
        x_outer: (0..m - r1cs.k_log).map(|_| rng.f128()).collect(),
    };

    println!(
        "witness+lincheck probe: m={m}, n_blocks={n_blocks}, n_runs={n_runs}, threads={}",
        rayon::current_num_threads()
    );

    // ---- Phase 1: witness generation (best-of-N). ----
    let mut wit_min = f64::INFINITY;
    let mut kept = None;
    for run in 0..n_runs + 1 {
        let t = Instant::now();
        let out = generate_witness_with_ab_packed_and_lincheck(&blocks, n_log);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if run > 0 {
            // run 0 is warm-up
            wit_min = wit_min.min(el);
        }
        if kept.is_none() {
            kept = Some(out);
        } else {
            black_box(&out);
        }
    }
    let (z, a, b, z_lincheck) = kept.unwrap();
    black_box((&z, &a, &b));

    let mut h = Fnv::new();
    h.f128s(&z);
    h.f128s(&a);
    h.f128s(&b);
    h.bytes(&z_lincheck);
    println!("wit min: {wit_min:.2} ms  checksum_wit: {:016x}", h.0);

    // ---- Phase 2: lincheck prove (best-of-N, deterministic challenger). ----
    let mut lc_min = f64::INFINITY;
    let mut lc_out = None;
    for run in 0..n_runs + 1 {
        let mut ch = FsChallenger::new(b"flock-wl-probe-v0");
        let t = Instant::now();
        let out = prove_padded_capture_z_vec(
            &z_lincheck,
            m,
            r1cs.k_log,
            r1cs.k_skip,
            r1cs.useful_bits,
            circuit,
            &x_ab,
            &mut ch,
        );
        let el = t.elapsed().as_secs_f64() * 1e3;
        if run > 0 {
            lc_min = lc_min.min(el);
        }
        if lc_out.is_none() {
            lc_out = Some(out);
        } else {
            black_box(&out);
        }
    }
    let (lc_proof, lc_claim, z_vec_pre) = lc_out.unwrap();

    let mut h = Fnv::new();
    for (m1, mi) in &lc_proof.rounds {
        h.f128s(&[*m1, *mi]);
    }
    h.f128s(&lc_proof.z_partial);
    h.f128s(&[lc_claim.r_inner_skip, lc_claim.w]);
    h.f128s(&lc_claim.r_inner_rest);
    h.f128s(&z_vec_pre);
    println!("lc min: {lc_min:.2} ms  checksum_lc: {:016x}", h.0);

    println!("combined min: {:.2} ms", wit_min + lc_min);
}
