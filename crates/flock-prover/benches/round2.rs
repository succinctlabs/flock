//! Round-2 fused fold + message bench, on the **production** dispatch.
//!
//! This drives `uni_skip_fold_and_round_pair_runs_sparse` under a run-list
//! `PaddingSpec` shaped like the real union witness, because that is the route
//! the m=32 prove actually takes. The bench used to call the DENSE entry with
//! `PaddingSpec::dense`, which measured a path production never executes — and
//! that gap produced a wrong verdict on 2026-09-01 (a two-pair unroll read
//! −3.5% here and neutral-to-noise on the real prove). Per-pair ARITHMETIC
//! changes transfer between the routes; loop-structure and dispatch changes do
//! not, so the bench must run the route that ships.
//!
//! Shape: the BLAKE3 union at m=32 has `useful_bits = 11_707` of a 2^14 block,
//! i.e. `ceil(11707/128) = 92` useful chunk-columns of 128 — 71.875% occupancy,
//! expressed here as one useful run of 92 blocks followed by 36 dead ones.
//! The dead region is left honestly zero, exactly as the prover guarantees.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F8;
use flock_prover::zerocheck::multilinear::{
    UniSkipFoldTable, uni_skip_fold_and_round_pair_runs_sparse,
};
use flock_prover::zerocheck::{PaddingRun, PaddingSpec};

const K_SKIP: usize = 6;

/// Useful chunk-columns out of 128, matching the shipped BLAKE3 union shape.
const USEFUL_COLS: usize = 92;
const TOTAL_COLS: usize = 128;

use flock_core::test_rng::Rng;

fn _silence_unused() {
    let _ = F8::ZERO;
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: non-NEON path)");
    println!(
        "(dispatch: SPARSE run-list, {USEFUL_COLS}/{TOTAL_COLS} columns useful \
         = {:.3}% occupancy — the production route)",
        100.0 * USEFUL_COLS as f64 / TOTAL_COLS as f64
    );

    // `build_b_med_counts` only skips at k_log >= K_SKIP + N_INNER = 13, and the
    // run-list needs 128 blocks, so m must leave k_log = m - 7 >= 13.
    for &m in &[20usize, 24, 26, 28, 29] {
        let n_bits = 1usize << m;
        let n_bytes = n_bits / 8;
        let col_bits = m - 7; // 128 chunk-columns of 2^(m-7) bits
        println!(
            "\n=== m = {m} ({} boolean constraints, {} MB packed) ===",
            n_bits,
            n_bytes >> 20
        );

        let mut rng = Rng::new(0xBEEF0042 + m as u64);

        // Packed witnesses; the dead column tail stays ZERO, as the prover
        // guarantees (the sparse kernel is only value-correct under that).
        let live_bytes = (USEFUL_COLS << col_bits) / 8;
        let mut a_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut a_packed[..live_bytes]);
        let mut b_packed = vec![0u8; n_bytes];
        rng.fill_bytes(&mut b_packed[..live_bytes]);

        let padding = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: 1usize << col_bits,
                n_blocks: USEFUL_COLS,
            },
            PaddingRun {
                k_log: col_bits,
                useful_bits_per_block: 0,
                n_blocks: TOTAL_COLS - USEFUL_COLS,
            },
        ]);
        assert!(
            padding.as_single_run().is_none(),
            "the sparse entry requires a run list"
        );

        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - K_SKIP);

        let t0 = Instant::now();
        let table = UniSkipFoldTable::new(K_SKIP, z);
        println!(
            "  fold table build (one-time)              {:>10.2} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );

        // Warm-up to prime caches and the scratch pool.
        let _ = uni_skip_fold_and_round_pair_runs_sparse(
            &a_packed,
            &b_packed,
            m,
            K_SKIP,
            &table,
            &mlv_challenges,
            &padding,
        );

        let n_runs = if m >= 24 { 3 } else { 1 };
        let mut best_ms = f64::INFINITY;
        let (mut cs_a, mut cs_b, mut cs_msg) = (0u64, 0u64, 0u64);
        for run in 0..n_runs {
            let label = if n_runs == 1 {
                String::from("sparse fold + round-2 msg")
            } else {
                format!("sparse fold + round-2 msg (run {})", run + 1)
            };
            let t0 = Instant::now();
            let (a_mlv, b_mlv, m1, minf, store) = uni_skip_fold_and_round_pair_runs_sparse(
                black_box(&a_packed),
                black_box(&b_packed),
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
                &padding,
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<40} {:>10.2} ms", label, elapsed);
            best_ms = best_ms.min(elapsed);
            cs_a ^= a_mlv[0].lo;
            cs_b ^= b_mlv[0].lo;
            cs_msg ^= m1.lo ^ minf.lo ^ (store.len() as u64);
        }
        if n_runs > 1 {
            println!("  {:<40} {:>10.2} ms", "  (best)", best_ms);
        }
        println!(
            "  checksums: a_mlv[0].lo={cs_a:016x}  b_mlv[0].lo={cs_b:016x}  msg={cs_msg:016x}"
        );
    }
}
