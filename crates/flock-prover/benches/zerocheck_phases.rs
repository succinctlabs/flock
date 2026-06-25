//! Per-phase breakdown of `zerocheck::prove_packed`.
//!
//! There is only **one zerocheck** (one univariate-skip round followed by one
//! multilinear sumcheck), but its prover work splits into distinct phases with
//! very different cost profiles. This bench mirrors `prove_packed` inline with
//! `Instant::now()` between each phase so all numbers come from the same
//! thermal state.
//!
//! Phases:
//!   1. Round-1 URM (univariate skip)            — `round1_shift_reduce_extract_c_packed`
//!   2. C-claim interpolation                    — `interpolate_at_z_on_lambda`
//!   3. Round-2 (fused fold + 1st mlv message)  — `uni_skip_fold_and_round_pair_optimized_packed`
//!   4. Rounds 3..(n_mlv+1) — the multilinear sumcheck tail (one sumcheck,
//!      many small rounds: fused while ≥10 vars remain, then naive).
//!   5. Final binding fold.
//!
//! Plus an end-to-end `prove_packed` run for cross-check.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::{Challenger, FsChallenger};
use flock_prover::field::{F8, F128};
use flock_prover::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use flock_prover::zerocheck::multilinear::{
    UniSkipFoldTable, fold_and_compute_round_pair_into, fold_in_place_pair,
    interpolate_at_z_on_lambda, round_pair_naive, uni_skip_fold_and_round_pair_optimized_packed,
};
use flock_prover::zerocheck::prove_packed;
use flock_prover::zerocheck::univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed, small_challenges_ghash,
};

const K_SKIP: usize = 6;
const N_INNER: usize = 7;

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
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let len = buf.len();
        let mut i = 0;
        while i + 8 <= len {
            let v = self.next_u64();
            buf[i..i + 8].copy_from_slice(&v.to_le_bytes());
            i += 8;
        }
        if i < len {
            let v = self.next_u64().to_le_bytes();
            buf[i..].copy_from_slice(&v[..len - i]);
        }
    }
}

fn time_phase<R>(label: &str, total_ms: &mut f64, f: impl FnOnce() -> R) -> R {
    let t0 = Instant::now();
    let r = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    *total_ms += ms;
    println!("    {:<50} {:>9.2} ms", label, ms);
    r
}

/// Inlined re-implementation of `prove_packed` with per-phase timing.
fn prove_with_phase_timing(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
) -> (f64, Vec<f64>) {
    let mut challenger = FsChallenger::new(b"flock-bench-v0");
    let k_skip = K_SKIP;
    let n_mlv = m - k_skip;
    challenger.observe_label(b"flock-zerocheck-v0");

    // ---- sample r with protocol-fixed inner constants ----
    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    let mut total = 0.0f64;
    let mut phases: Vec<f64> = Vec::new();

    // ---- Phase 1: Round-1 URM ----
    let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
    let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let mut p_start = total;
    let (round1_ab_opt, round1_c_opt) =
        time_phase("1. Round-1 URM (univariate skip)", &mut total, || {
            round1_shift_reduce_extract_c_packed(
                black_box(a_packed),
                black_box(b_packed),
                black_box(c_packed),
                m,
                k_skip,
                &r,
                &inv_table,
            )
        });
    phases.push(total - p_start);

    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = round1_ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = round1_c_opt.iter().map(|x| c_s * *x).collect();
    challenger.observe_f128_slice(&round1_ab);
    challenger.observe_f128_slice(&round1_c);
    let z = challenger.sample_f128();

    // ---- Phase 2: C-claim interpolation ----
    p_start = total;
    let _final_c_eval = time_phase(
        "2. C-claim interp (interpolate_at_z_on_lambda)",
        &mut total,
        || interpolate_at_z_on_lambda(&round1_c, k_skip, z),
    );
    phases.push(total - p_start);

    // ---- Phase 3: Round-2 (fused fold + 1st mlv message) ----
    p_start = total;
    let fold_table = UniSkipFoldTable::new(k_skip, z);
    let mut mlv_arg = vec![F128::ONE; n_mlv];
    mlv_arg[1..].copy_from_slice(&r[k_skip + 1..]);
    let (mut a_mlv, mut b_mlv, msg_1, msg_inf) = time_phase(
        "3. Round-2 fused fold + 1st mlv message",
        &mut total,
        || {
            uni_skip_fold_and_round_pair_optimized_packed(
                black_box(a_packed),
                black_box(b_packed),
                m,
                k_skip,
                &fold_table,
                &mlv_arg,
            )
        },
    );
    phases.push(total - p_start);

    challenger.observe_f128(msg_1);
    challenger.observe_f128(msg_inf);
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    mlv_rhos.push(challenger.sample_f128());

    // ---- Phase 4: Rounds 3..(n_mlv + 1) — multilinear sumcheck tail ----
    p_start = total;
    let mut fused_ms = 0.0f64;
    let mut naive_ms = 0.0f64;
    // Ping-pong scratch buffers — mirrors prove_packed: the fused round folds
    // into a persistent buffer rather than allocating/freeing 64 MB per round.
    // (Allocated outside the timed region, so plain zero-init is fine here —
    // the crate's `alloc_uninit_f128_vec` is crate-private.)
    let n_in = a_mlv.len();
    let (mut a_nxt, mut b_nxt) = if n_in >= 1024 {
        (vec![F128::ZERO; n_in / 2], vec![F128::ZERO; n_in / 2])
    } else {
        (Vec::new(), Vec::new())
    };
    time_phase(
        "4. Rounds 3..(n_mlv+1) — multilinear sumcheck tail",
        &mut total,
        || {
            for i in 0..(n_mlv - 1) {
                let rho_prev = mlv_rhos[i];
                let log_n_before = a_mlv.len().trailing_zeros() as usize;
                let mut r_next = vec![F128::ONE; log_n_before - 1];
                r_next[1..].copy_from_slice(&r[k_skip + i + 2..]);

                if log_n_before >= 10 {
                    let half = a_mlv.len() / 2;
                    let t = Instant::now();
                    let (_m1, _mi) = fold_and_compute_round_pair_into(
                        &a_mlv,
                        &b_mlv,
                        &mut a_nxt[..half],
                        &mut b_nxt[..half],
                        rho_prev,
                        &r_next,
                    );
                    std::mem::swap(&mut a_mlv, &mut a_nxt);
                    std::mem::swap(&mut b_mlv, &mut b_nxt);
                    a_mlv.truncate(half);
                    b_mlv.truncate(half);
                    fused_ms += t.elapsed().as_secs_f64() * 1000.0;
                } else {
                    let t = Instant::now();
                    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
                    let (_m1, _mi) = round_pair_naive(&a_mlv, &b_mlv, &r_next);
                    naive_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
                mlv_rhos.push(F128 {
                    lo: 0xDEADBEEF + i as u64,
                    hi: 0xC0FFEE + i as u64,
                });
            }
        },
    );
    phases.push(total - p_start);
    println!("        — of which fused (log_n ≥ 10): {:.2} ms", fused_ms);
    println!("        — of which naive  (log_n < 10): {:.2} ms", naive_ms);

    // ---- Phase 5: Final binding ----
    p_start = total;
    let rho_last = *mlv_rhos.last().unwrap();
    time_phase("5. Final binding fold", &mut total, || {
        fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_last);
    });
    phases.push(total - p_start);

    black_box((a_mlv, b_mlv));
    println!("    {:<50} {:>9.2} ms", "  TOTAL (sum of phases)", total);
    (total, phases)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: scalar fallback)");

    for &m in &[24usize, 26, 28, 29] {
        let n_bits = 1usize << m;
        let n_bytes = n_bits / 8;
        println!(
            "\n=== m = {m} ({} boolean constraints, {} MB packed) ===",
            n_bits,
            n_bytes >> 20
        );

        let mut rng = Rng::new(0xDEAD_C0DE + m as u64);
        let mut a_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut a_packed);
        let mut b_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut b_packed);
        let c_packed: Vec<u8> = a_packed.iter().zip(&b_packed).map(|(x, y)| x & y).collect();

        // Warm up to prime OnceLock caches and let the OS settle.
        {
            let mut ch = FsChallenger::new(b"flock-bench-v0");
            let _ = prove_packed(&a_packed, &b_packed, &c_packed, m, &mut ch);
        }

        // ---- Phase-broken-down run ----
        println!("  per-phase breakdown:");
        let (sum_ms, phases) = prove_with_phase_timing(&a_packed, &b_packed, &c_packed, m);

        // ---- End-to-end cross-check ----
        let t0 = Instant::now();
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let (proof, _claim) = prove_packed(
            black_box(&a_packed),
            black_box(&b_packed),
            black_box(&c_packed),
            m,
            &mut ch,
        );
        let e2e_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "    {:<50} {:>9.2} ms  (cross-check)",
            "  prove_packed end-to-end", e2e_ms
        );

        // Percentage table.
        let names = [
            "Round-1 URM (univariate skip)",
            "C-claim interpolation",
            "Round-2 fold + 1st mlv msg",
            "Rounds 3..(n_mlv+1) sumcheck tail",
            "Final binding",
        ];
        println!("  share of total ({:.2} ms):", sum_ms);
        for (n, p) in names.iter().zip(&phases) {
            println!(
                "    {:<50} {:>6.1}%  ({:>7.2} ms)",
                n,
                100.0 * p / sum_ms,
                p
            );
        }
        black_box(proof.final_a_eval);
    }
}
