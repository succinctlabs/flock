//! NTT layout probe: does skipping zero interleaving-lanes in the Ligerito
//! commit NTT pay, and under which layout?
//!
//! Scenario: a commit interleaves `num_ntts = 16` sub-NTTs, but only `k = 12`
//! of the lanes carry real message data — the other 4 are zero padding whose
//! transform is zero. Two candidate "skip the zero lanes" layouts:
//!
//!   * Option B (low-bit interleaved): keep the SoA `codeword[pos*num_ntts+lane]`
//!     layout but narrow the lane loop from 16 → 12.
//!   * Option A (high-bit lane-major): store each lane as a contiguous block and
//!     simply don't transform the zero blocks — `k` independent single
//!     transforms.
//!
//! We measure best-of-N warm wall time for the additive NTT at an m=30-scale
//! shape and compare against the current baseline (interleaved over all 16
//! lanes).
//!
//! ## Shape (matches the real m=30 commit codeword)
//!
//! `log_dim = 19` message coords per lane; RS rate `1/2^log_inv_rate` expands
//! each lane to a codeword of `2^k_code`, `k_code = log_dim + log_inv_rate`.
//! With `log_inv_rate = 1`: per-lane domain `2^20`, 16-lane buffer `2^24 F128 =
//! 256 MB` — exactly the m=30 codeword. (The task's "2^log_dim block" refers to
//! the *message*; the NTT actually runs over the *codeword* domain `2^k_code`,
//! which is what this bench transforms.)
//!
//! ## What is (and isn't) directly comparable
//!
//! `forward_transform_interleaved` and `forward_transform` (→
//! `forward_transform_batched` on aarch64) build the SAME per-lane RS codeword
//! from the SAME twiddle domain `AdditiveNttF128::standard(k_code)` — proven by
//! the crate tests `interleaved_matches_per_lane` and `batched_matches_scalar`.
//! So the two layouts are directly comparable per lane.
//!
//! Two honest caveats, reported inline:
//!   1. On aarch64 the interleaved kernel is scalar-per-lane (portable
//!      `butterfly_row_pair` / fused-2-layer; the SIMD kernels are x86-AVX512
//!      only), whereas the single-transform batched path uses the 2-wide NEON
//!      PMULL `butterfly_neon_block`. Different kernels for the two layouts.
//!   2. The real commit skips the first `log_inv_rate` NTT layers via the
//!      replicate-fill (`forward_transform_interleaved_from_layer`). The
//!      single-transform path has NO from-layer variant, so lane-major (Option
//!      A) cannot claim that skip. We therefore report the interleaved baseline
//!      BOTH from layer 0 (apples-to-apples kernel comparison vs lane-major) and
//!      from `log_inv_rate` (what the real commit actually pays).
//!
//! Run: `cargo bench --bench ntt_layout_probe [n_runs]`

use std::env::args;
use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::ntt::AdditiveNttF128;

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

/// Best-of-N (and median) warm timing. Restores `buf` from `seed` BEFORE each
/// timed run (the NTT is in-place and destroys its input); the reset is OUTSIDE
/// the timed region. `op` receives the freshly-reset buffer.
fn best_of<O>(n_runs: usize, buf: &mut [F128], seed: &[F128], mut op: O) -> (f64, f64)
where
    O: FnMut(&mut [F128]),
{
    debug_assert_eq!(buf.len(), seed.len());
    // Warm-up (not recorded).
    buf.copy_from_slice(seed);
    op(buf);
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        buf.copy_from_slice(seed);
        let t = Instant::now();
        op(buf);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    (min, median)
}

struct Row {
    label: &'static str,
    min_ms: f64,
    median_ms: f64,
}

fn row(label: &'static str, min_ms: f64, median_ms: f64) -> Row {
    Row {
        label,
        min_ms,
        median_ms,
    }
}

fn run_shape(n_runs: usize, log_dim: usize, num_ntts: usize, k_real: usize, log_inv_rate: usize) {
    let k_code = log_dim + log_inv_rate;
    let per_lane = 1usize << k_code; // codeword elems per lane (NTT domain)
    let full_len = per_lane * num_ntts;
    let mb = |elems: usize| (elems * 16) as f64 / (1024.0 * 1024.0);

    println!(
        "\n===== shape: log_dim={log_dim}, log_inv_rate={log_inv_rate}, k_code={k_code}, \
         num_ntts={num_ntts}, real lanes k={k_real} =====",
    );
    println!(
        "  per-lane NTT domain = 2^{k_code} = {per_lane} F128 = {:.1} MB;  \
         full {num_ntts}-lane buffer = {} F128 = {:.0} MB;  {k_real}-lane buffer = {:.0} MB",
        mb(per_lane),
        full_len,
        mb(full_len),
        mb(per_lane * k_real),
    );
    println!(
        "  n_runs={n_runs}, threads={}",
        rayon::current_num_threads()
    );

    let ntt = AdditiveNttF128::standard(k_code);

    // One seed (full 16-lane size) + one working buffer. Every measurement
    // resets from `seed` so each starts from identical, cache-cold-of-transform
    // state. Lane-major reinterprets the same bytes as contiguous blocks; the
    // interleaved paths read them as SoA. Either interpretation is a valid
    // transform of `num_ntts`/`k_real` lanes of `per_lane` elements — the WORK
    // (which is what we time) is what the layout changes.
    let mut rng = Rng::new(0xDEAD_BEEF ^ ((log_inv_rate as u64) << 8) ^ num_ntts as u64);
    let seed: Vec<F128> = (0..full_len).map(|_| rng.f128()).collect();
    let mut buf: Vec<F128> = seed.clone();

    // Scalar proxies are single-threaded and stable — far fewer runs suffice.
    let scalar_runs = n_runs.div_ceil(5).max(3);

    let mut rows: Vec<Row> = Vec::new();

    // ---- PARALLEL production kernels (what commit actually uses). ----

    // (1) Baseline — interleaved full 16 lanes, from layer 0.
    let n16 = per_lane * num_ntts;
    let (m1, m1_med) = best_of(n_runs, &mut buf[..n16], &seed[..n16], |b| {
        ntt.forward_transform_interleaved(b, num_ntts);
        black_box(&b[0]);
    });
    rows.push(row("1  baseline interleaved-16 (par, layer 0)", m1, m1_med));

    // (1b) Baseline — interleaved full 16 lanes, from layer log_inv_rate
    // (= what pcs::commit actually runs, via replicate-fill).
    let (m1b, m1b_med) = best_of(n_runs, &mut buf[..n16], &seed[..n16], |b| {
        ntt.forward_transform_interleaved_from_layer(b, num_ntts, log_inv_rate);
        black_box(&b[0]);
    });
    rows.push(row(
        "1b baseline interleaved-16 (par, from_layer=r) [commit]",
        m1b,
        m1b_med,
    ));

    // (3) Option A skip — lane-major, k_real independent single transforms.
    // `forward_transform` -> `forward_transform_batched` on aarch64 (2-wide NEON,
    // cache-blocked). No from-layer variant exists: always a full transform.
    let nk = per_lane * k_real;
    let (m3, m3_med) = best_of(n_runs, &mut buf[..nk], &seed[..nk], |b| {
        for block in b.chunks_mut(per_lane) {
            ntt.forward_transform(block);
        }
        black_box(&b[0]);
    });
    rows.push(row(
        "3  Option A lane-major-12 (par, k single transforms)",
        m3,
        m3_med,
    ));

    // (4) Option A full — lane-major over all num_ntts lanes (isolates per-lane
    // single-transform overhead vs the interleaved baseline).
    let (m4, m4_med) = best_of(n_runs, &mut buf[..n16], &seed[..n16], |b| {
        for block in b.chunks_mut(per_lane) {
            ntt.forward_transform(block);
        }
        black_box(&b[0]);
    });
    rows.push(row(
        "4  Option A lane-major-16 (par, all single transforms)",
        m4,
        m4_med,
    ));

    // (3p/4p) Best-case lane-major: parallelize ACROSS lanes (one worker per
    // contiguous block, serial NEON within a lane) instead of running each
    // internally-parallel `forward_transform` sequentially. This isolates
    // whether the interleaved advantage is a layout property or just a
    // "sequential single transforms under-parallelize" artifact. aarch64 only
    // (uses the single-thread `forward_transform_neon`).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        use rayon::prelude::*;
        let (m3p, m3p_med) = best_of(n_runs, &mut buf[..nk], &seed[..nk], |b| {
            b.par_chunks_mut(per_lane)
                .for_each(|block| ntt.forward_transform_neon(block));
            black_box(&b[0]);
        });
        rows.push(row(
            "3p Option A lane-major-12 (parallel-across-lanes, serial-NEON)",
            m3p,
            m3p_med,
        ));
        let (m4p, m4p_med) = best_of(n_runs, &mut buf[..n16], &seed[..n16], |b| {
            b.par_chunks_mut(per_lane)
                .for_each(|block| ntt.forward_transform_neon(block));
            black_box(&b[0]);
        });
        rows.push(row(
            "4p Option A lane-major-16 (parallel-across-lanes, serial-NEON)",
            m4p,
            m4p_med,
        ));
    }

    // ---- Option B (interleaved narrow to k_real lanes). ----
    //
    // The optimized PARALLEL interleaved kernel asserts `num_ntts.is_power_of_two()`
    // and its cache/thread heuristics call `log2_pow2(num_ntts)`. So a 12-lane
    // interleaved transform is NOT runnable on the production kernel — Option B
    // cannot skip a non-power-of-two remainder of lanes without new kernel work.
    // We report the SCALAR interleaved path (which accepts any lane count) at
    // BOTH k_real and num_ntts lanes as a single-thread arithmetic work-ratio
    // proxy: it shows the FLOOR a generalized parallel kernel could reach, not a
    // production number. (If k_real were a power of two we would run the parallel
    // kernel instead.)
    let b_supported = k_real.is_power_of_two();
    let (m2, m2_med, m2_label): (f64, f64, &'static str) = if b_supported {
        let (mn, md) = best_of(n_runs, &mut buf[..nk], &seed[..nk], |b| {
            ntt.forward_transform_interleaved(b, k_real);
            black_box(&b[0]);
        });
        (mn, md, "2  Option B interleaved-12 (par, layer 0)")
    } else {
        // Scalar-16 reference (same kernel, so the ratio is apples-to-apples).
        let (s16, _s16m) = best_of(scalar_runs, &mut buf[..n16], &seed[..n16], |b| {
            ntt.forward_transform_interleaved_scalar(b, num_ntts);
            black_box(&b[0]);
        });
        rows.push(row(
            "   (scalar interleaved-16, ref for Option B ratio)",
            s16,
            s16,
        ));
        let (s12, s12m) = best_of(scalar_runs, &mut buf[..nk], &seed[..nk], |b| {
            ntt.forward_transform_interleaved_scalar(b, k_real);
            black_box(&b[0]);
        });
        (
            s12,
            s12m,
            "2  Option B interleaved-12 (SCALAR proxy; par kernel n/a)",
        )
    };
    // Keep the scalar-16 reference (if any) accessible for the ratio, captured
    // BEFORE the Option B row is pushed.
    let s16_ref = rows
        .iter()
        .find(|r| r.label.contains("ref for Option B"))
        .map(|r| r.min_ms);
    rows.push(row(m2_label, m2, m2_med));

    // ---- Report table. Normalize to real data (k_real real lanes). The zero
    // padding lanes carry no information, so per-real-lane charges any work done
    // on them to the k_real real lanes (exactly the cost the skip aims to remove).
    let real_codeword_elems = (k_real * per_lane) as f64;
    println!(
        "\n  {:<58} {:>9} {:>9} {:>13} {:>16}",
        "measurement", "min(ms)", "med(ms)", "ms/real-lane", "Melem/s(real)"
    );
    println!("  {}", "-".repeat(58 + 9 + 9 + 13 + 16 + 4));
    for r in &rows {
        let per_real_lane = r.min_ms / k_real as f64;
        let throughput = real_codeword_elems / (r.min_ms / 1e3) / 1e6;
        println!(
            "  {:<58} {:>9.2} {:>9.2} {:>13.3} {:>16.1}",
            r.label, r.min_ms, r.median_ms, per_real_lane, throughput,
        );
    }

    // ---- Verdict math for this shape.
    let ideal = k_real as f64 / num_ntts as f64;
    println!("\n  ideal skip ratio (k/num_ntts) = {k_real}/{num_ntts} = {ideal:.3}");
    if b_supported {
        println!(
            "  Option B skip (interleaved par):  {k_real}/{num_ntts} time = {:.3}  (ideal {ideal:.3}) -> {}",
            m2 / m1,
            if m2 < m1 { "SAVES" } else { "no gain" },
        );
    } else if let Some(s16) = s16_ref {
        println!("  Option B: NOT runnable on the parallel kernel (num_ntts must be pow2).");
        println!(
            "            scalar work-ratio (s12/s16) = {:.3}  (ideal {ideal:.3}) — arithmetic floor only",
            m2 / s16,
        );
    }
    println!(
        "  Option A skip (lane-major par):   {k_real}/{num_ntts} time = {:.3}  (ideal {ideal:.3}) -> {}",
        m3 / m4,
        if m3 < m4 { "SAVES" } else { "no gain" },
    );
    println!(
        "  lane-major vs interleaved, per-lane @{num_ntts} (m4/{num_ntts} vs m1/{num_ntts}): \
         {:.3} vs {:.3} ms -> lane-major {:.2}x {}",
        m4 / num_ntts as f64,
        m1 / num_ntts as f64,
        m4 / m1,
        if m4 < m1 { "(faster)" } else { "(slower)" },
    );

    // Best lane-major-12 across the sequential and parallel-across-lanes variants.
    let best_lm12 = rows
        .iter()
        .filter(|r| r.label.contains("lane-major-12"))
        .map(|r| r.min_ms)
        .fold(f64::INFINITY, f64::min);

    println!("\n  cross-layout decision (lower ms is better):");
    println!("    baseline interleaved-16 layer-0     : {m1:.2} ms");
    println!("    baseline interleaved-16 real-commit : {m1b:.2} ms  <- what commit pays today");
    println!("    best Option A lane-major-12 skip    : {best_lm12:.2} ms");
    let baseline_ref = m1b; // the real-commit baseline is the thing to beat
    if best_lm12 < baseline_ref {
        println!(
            "    => Option A (lane-major-12) BEATS the real-commit baseline by {:.2} ms ({:.1}%).",
            baseline_ref - best_lm12,
            100.0 * (baseline_ref - best_lm12) / baseline_ref,
        );
    } else {
        println!(
            "    => Option A (lane-major-12) does NOT beat the real-commit baseline \
             ({baseline_ref:.2} ms); best lane-major-12 is {:.2} ms ({:+.1}%).",
            best_lm12,
            100.0 * (best_lm12 - baseline_ref) / baseline_ref,
        );
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!(
        "(target: aarch64 + aes — interleaved=scalar-per-lane portable kernel; single=2-wide NEON batched)"
    );
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: non-aarch64 / software fallback path)");

    let n_runs: usize = args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);

    // m=30-representative: log_dim=19, num_ntts=16 (log_batch_size=4), k=12
    // real lanes (25% zero padding). Rate 1/2 is the pcs_commit default; rate
    // 1/4 is the "secure"/matched-codeword config.
    run_shape(n_runs, 19, 16, 12, 1);
    run_shape(n_runs, 19, 16, 12, 2);
}
