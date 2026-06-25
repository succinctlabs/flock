//! Combined probe for keccak3's witness generation and the real lincheck
//! prover (walker circuit), at the keccak3 m=30 prove_fast shapes
//! (24,576 keccaks = 8,192 3-wide blocks).
//!
//! Usage mirrors `witness_lincheck_probe`:
//!   ./target/release/deps/keccak3_witlc_probe-<hash> [n_runs] [n_keccaks]
//!   RAYON_NUM_THREADS=1 ... for ST.
//!
//! Prints best-of-N per phase, their sum, and FNV checksums over all phase
//! outputs (seeded inputs ⇒ bit-stable across valid optimizations).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::field::F128;
use flock_prover::lincheck::{QuirkyPoint, prove_padded_capture_z_vec};
use flock_prover::r1cs_hashes::keccak::State;
use flock_prover::r1cs_hashes::keccak3::{
    K_SKIP, KeccakLincheckCircuit, KeccakSetup, generate_witness_with_ab_packed_and_lincheck,
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
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn random_state(rng: &mut Rng) -> State {
    let mut s = [false; 1600];
    for chunk in s.chunks_mut(64) {
        let v = rng.next_u64();
        for (i, b) in chunk.iter_mut().enumerate() {
            *b = (v >> i) & 1 == 1;
        }
    }
    s
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
    let n_keccaks: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24576);

    let setup = KeccakSetup::new(n_keccaks);
    let r1cs = &setup.r1cs;
    let m = r1cs.m;
    let n_blocks_log = m - r1cs.k_log;

    let mut rng = Rng::new(0x3ECC_A3_C0DE);
    let states: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();

    let x_ab = QuirkyPoint {
        z_skip: rng.f128(),
        x_inner_rest: (0..r1cs.k_log - K_SKIP).map(|_| rng.f128()).collect(),
        x_outer: (0..m - r1cs.k_log).map(|_| rng.f128()).collect(),
    };

    println!(
        "keccak3 wit+lc probe: m={m}, n_keccaks={n_keccaks}, n_runs={n_runs}, threads={}",
        rayon::current_num_threads()
    );

    // ---- Phase 1: witness generation (best-of-N; run 0 = warm-up). ----
    let mut wit_min = f64::INFINITY;
    let mut kept = None;
    for run in 0..n_runs + 1 {
        let t = Instant::now();
        let out = generate_witness_with_ab_packed_and_lincheck(&states, n_blocks_log);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if run > 0 {
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
        let mut ch = FsChallenger::new(b"flock-k3-probe-v0");
        let t = Instant::now();
        let out = prove_padded_capture_z_vec(
            &z_lincheck,
            m,
            r1cs.k_log,
            r1cs.k_skip,
            r1cs.useful_bits,
            &KeccakLincheckCircuit,
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
