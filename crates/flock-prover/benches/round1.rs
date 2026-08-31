//! Round-1 URM bench — full sweep through m=29 (the C++ headline workload).
//!
//! At m=29 the witness is 3 × 64 MB packed bytes. We generate `*_packed`
//! directly to avoid 3 × 537 MB bool-vec allocations.
//!
//! pack_bits is measured separately (one-time cost, hoisted in the C++ bench
//! and in any real prover). The naive variant only runs at m ≤ 20 — beyond
//! that it's many seconds and uninformative.

use flock_prover::init_perf_thread_pool;
use flock_prover::zerocheck::PaddingSpec;
use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::{F8, F128};
use flock_prover::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use flock_prover::zerocheck::univariate_skip::{pack_bits, round1_extract_c_packed, round1_naive};
use flock_prover::zerocheck::univariate_skip_optimized::{
    K_SKIP, medium_challenges_ghash, round1_shift_reduce_extract_c_packed,
    round1_shift_reduce_extract_c_packed_padded_with_s_hat_v, small_challenges_ghash,
};

use flock_core::test_rng::Rng;
const N_INNER: usize = 7;

fn build_protocol_r(m: usize, outer: &[F128]) -> Vec<F128> {
    assert_eq!(outer.len(), m - K_SKIP - N_INNER);
    let mut r = vec![F128::ZERO; m];
    for (i, &small) in small_challenges_ghash().iter().enumerate() {
        r[K_SKIP + i] = small;
    }
    for (i, &med) in medium_challenges_ghash().iter().enumerate() {
        r[K_SKIP + 3 + i] = med;
    }
    for (i, &x) in outer.iter().enumerate() {
        r[K_SKIP + N_INNER + i] = x;
    }
    r
}

fn time_ms<R>(label: &str, f: impl FnOnce() -> R) -> R {
    let t0 = Instant::now();
    let r = f();
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  {:<40} {:>10.2} ms", label, elapsed);
    r
}

/// Unpack packed bytes back into a Vec<bool> for the naive (small-m) path.
fn unpack_to_bool(packed: &[u8], n_bits: usize) -> Vec<bool> {
    let mut v = Vec::with_capacity(n_bits);
    for i in 0..n_bits {
        v.push((packed[i / 8] >> (i % 8)) & 1 != 0);
    }
    v
}

fn main() {
    let _ = init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");

    // One-time table setup (K_SKIP=6).
    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
    let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);

    for &m in &[16usize, 20, 24, 26, 28, 29] {
        let n_bits = 1usize << m;
        let n_bytes = n_bits / 8;
        println!(
            "\n=== m = {m} ({} boolean constraints, {} MB packed) ===",
            n_bits,
            n_bytes >> 20
        );

        let mut rng = Rng::new(0xA110C8 + m as u64);

        // Generate packed witnesses directly. Honest witness: c = a AND b
        // ⇔ c_packed[i] = a_packed[i] & b_packed[i] (byte-wise AND).
        let setup_start = Instant::now();
        let mut a_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut a_packed);
        let mut b_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut b_packed);
        let c_packed: Vec<u8> = a_packed.iter().zip(&b_packed).map(|(x, y)| x & y).collect();
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  (witness setup, packed)                  {:>10.2} ms",
            setup_ms
        );

        // For the naive variant only — unpack to bool. Skip beyond m=20.
        let bool_inputs = if m <= 20 {
            Some((
                unpack_to_bool(&a_packed, n_bits),
                unpack_to_bool(&b_packed, n_bits),
                unpack_to_bool(&c_packed, n_bits),
            ))
        } else {
            None
        };

        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);

        // Naive (small m only).
        let mut naive_checksum = 0u64;
        if let Some((a_bits, b_bits, c_bits)) = bool_inputs.as_ref() {
            let (n_ab, n_c) = time_ms("naive (bool input)", || {
                round1_naive(
                    black_box(a_bits),
                    black_box(b_bits),
                    black_box(c_bits),
                    m,
                    K_SKIP,
                    &r,
                )
            });
            naive_checksum = n_ab[0].lo ^ n_c[0].lo;
        }

        // Pack-bits cost on its own — same as round trip from bool, since
        // we generated packed directly. Inform the comparison.
        if let Some((a_bits, _, _)) = bool_inputs.as_ref() {
            let _ = time_ms("pack_bits (one-time)", || pack_bits(black_box(a_bits)));
        }

        // Structural (Stage 1, packed input).
        let (s_ab, s_c) = time_ms("extract_c (packed, no shift_reduce)", || {
            round1_extract_c_packed(
                black_box(&a_packed),
                black_box(&b_packed),
                black_box(&c_packed),
                m,
                K_SKIP,
                &r,
                &table,
            )
        });

        // Warm up the optimized path so the OnceLock-init of the 64 KB
        // convert table is excluded from the timed run.
        let _ = round1_shift_reduce_extract_c_packed(
            &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &table,
        );

        // Optimized (Stage 2, fused NEON, packed input) — three timed runs
        // at large m so we can see the noise.
        let n_runs = if m >= 24 { 3 } else { 1 };
        let mut best_opt_ms = f64::INFINITY;
        let mut o_cs = 0u64;
        for run in 0..n_runs {
            let label = if n_runs == 1 {
                String::from("shift_reduce + fused NEON (parallel)")
            } else {
                format!("shift_reduce + fused NEON (parallel, run {})", run + 1)
            };
            let t0 = Instant::now();
            let (o_ab, o_c) = round1_shift_reduce_extract_c_packed(
                black_box(&a_packed),
                black_box(&b_packed),
                black_box(&c_packed),
                m,
                K_SKIP,
                &r,
                &table,
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<40} {:>10.2} ms", label, elapsed);
            best_opt_ms = best_opt_ms.min(elapsed);
            o_cs = o_ab[0].lo ^ o_c[0].lo;
        }
        if n_runs > 1 {
            println!("  {:<40} {:>10.2} ms", "  (best)", best_opt_ms);
        }

        // Two-bank fusion variant (also produces s_hat_v_c).
        let _ = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
            &a_packed,
            &b_packed,
            &c_packed,
            m,
            K_SKIP,
            &r,
            &table,
            &PaddingSpec::dense(m),
        );
        let mut best_fusion_ms = f64::INFINITY;
        for run in 0..n_runs {
            let label = if n_runs == 1 {
                String::from("two-bank fusion (parallel)")
            } else {
                format!("two-bank fusion (parallel, run {})", run + 1)
            };
            let t0 = Instant::now();
            let (f_ab, f_c, f_s_hat) = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                black_box(&a_packed),
                black_box(&b_packed),
                black_box(&c_packed),
                m,
                K_SKIP,
                &r,
                &table,
                &PaddingSpec::dense(m),
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<40} {:>10.2} ms", label, elapsed);
            best_fusion_ms = best_fusion_ms.min(elapsed);
            black_box((f_ab, f_c, f_s_hat));
        }
        if n_runs > 1 {
            println!("  {:<40} {:>10.2} ms", "  (best)", best_fusion_ms);
        }

        let s_cs = s_ab[0].lo ^ s_c[0].lo;
        if naive_checksum != 0 {
            println!(
                "  checksums:  naive={naive_checksum:016x}  structural={s_cs:016x}  optimized={o_cs:016x}"
            );
        } else {
            println!("  checksums:  structural={s_cs:016x}  optimized={o_cs:016x}");
        }
    }
}
