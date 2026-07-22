//! URM kernel probe: tight loop on `round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`
//! with everything else (witness gen, challenge sampling, surrounding zerocheck rounds) outside
//! the profiled region. Inputs are constructed once; the kernel is called N_RUNS times.
//!
//! Usage:
//!   cargo bench --bench urm_probe --no-run
//!   RAYON_NUM_THREADS=1 samply record -- ./target/release/deps/urm_probe-<hash> [n_runs] [m] [padding]
//!
//! `padding` = `dense` (default) or `blake3` (k_log=14, useful=15409 — the
//! real blake3 prove_fast call shape).
//!
//! Prints an FNV checksum of all three outputs: inputs are seeded, so the
//! checksum must be bit-stable across any valid kernel optimization.
//!
//! Default: 30 runs at m=29 (~3.2 sec ST).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::{F8, F128};
use flock_prover::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use flock_prover::zerocheck::PaddingSpec;
use flock_prover::zerocheck::univariate_skip_optimized::{
    K_SKIP, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded_with_s_hat_v,
    small_challenges_ghash,
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

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let n_runs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let m: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(29);

    let n_bytes = (1usize << m) / 8;
    let mut rng = Rng::new(0xDEAD_C0DE ^ (m as u64));

    let mut a_packed = vec![0u8; n_bytes];
    let mut b_packed = vec![0u8; n_bytes];
    for byte in a_packed.iter_mut() {
        *byte = rng.next_u64() as u8;
    }
    for byte in b_packed.iter_mut() {
        *byte = rng.next_u64() as u8;
    }
    let c_packed: Vec<u8> = a_packed.iter().zip(&b_packed).map(|(a, b)| a & b).collect();

    // Construct r with the protocol-fixed challenges at the right slots.
    let mut r: Vec<F128> = Vec::with_capacity(m);
    for _ in 0..K_SKIP {
        r.push(rng.f128());
    }
    r.extend(small_challenges_ghash().iter().copied());
    r.extend(medium_challenges_ghash().iter().copied());
    for _ in (K_SKIP + 7)..m {
        r.push(rng.f128());
    }
    assert_eq!(r.len(), m);

    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
    let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let padding_mode = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "dense".to_string());
    let padding = match padding_mode.as_str() {
        "dense" => PaddingSpec::dense(m),
        // BLAKE3 prove_fast shape: K_LOG=14, USEFUL_BITS=15,409.
        "blake3" => PaddingSpec::uniform(14, 15_409, 1usize << (m - 14)),
        other => panic!("padding arg must be 'dense' or 'blake3', got '{other}'"),
    };

    println!(
        "URM probe: m={m}, n_runs={n_runs}, threads={}, padding={padding_mode}",
        rayon::current_num_threads()
    );

    // Warm-up.
    {
        let _ = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
            &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table, &padding,
        );
    }

    let t0 = Instant::now();
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t = Instant::now();
        let (res_ab, res_c_lifted, s_hat_v_c) =
            round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table, &padding,
            );
        let elapsed = t.elapsed().as_secs_f64() * 1e3;
        times.push(elapsed);
        black_box(&res_ab);
        black_box(&res_c_lifted);
        black_box(&s_hat_v_c);
    }
    let total = t0.elapsed().as_secs_f64() * 1e3;
    let avg = times.iter().sum::<f64>() / n_runs as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("{n_runs} URM calls: total {total:.2} ms, avg {avg:.2} ms/call, min {min:.2} ms/call");

    // Output checksum (FNV-1a over all three result vectors, in order).
    // Deterministic inputs ⇒ this must never change across optimizations.
    let (res_ab, res_c_lifted, s_hat_v_c) =
        round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
            &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table, &padding,
        );
    let mut h: u64 = 0xcbf29ce484222325;
    let mut absorb = |v: &[F128]| {
        for x in v {
            for b in x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes()) {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
    };
    absorb(&res_ab);
    absorb(&res_c_lifted);
    absorb(&s_hat_v_c);
    println!("output checksum: {h:016x}");
}
