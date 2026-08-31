//! **THE VIRTUAL BASIS, isolated.** The merged transport's inner open is a
//! single EqPoint claim: `b = γ·eq(ρ, ·)`. Its L0 rounds used to source the
//! basis just-in-time for the FIRST fold only and stream a materialized
//! half-size array for the rest; `VirtualEqBasis` keeps it factored through
//! every L0 round (an eq tensor folds to an eq tensor — one scalar per bound
//! coordinate) and materializes exactly once, at the last round, at the size
//! the recursion takes over.
//!
//! This is both the ORACLE and the INSTRUMENT:
//!
//! - oracle: the two arms must produce byte-identical proofs — same b values
//!   by construction, so any divergence is a bug in the factorisation;
//! - instrument: alternating in-process arms on the SAME committed stack,
//!   which is the only way to resolve a few-ms effect on this box (the
//!   dead-lane lesson: process-level arms carry ±4-8 ms of interference).
//!
//! Default geometry is the m32 chain leaf's inner open: 2^25 packed words,
//! slim rate 1/4, `initial_k = 6`, 56 committed lanes of 64 (lane-major, so
//! L0 folds blocks). Knobs: `MICRO_RUNS` (default 7 per arm), `MICRO_M` /
//! `MICRO_K` / `MICRO_LANES` for the other shipped shapes — the envelope
//! outer is `MICRO_M=29 MICRO_K=5 MICRO_LANES=24`.
use bincode::serialize;
use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::merkle::HashKind;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{
    DirectEqInd, OpeningGrinding, PackedDirectClaim, PcsParams, VIRTUAL_B_OVERRIDE,
    commit_lane_major, open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding,
};
use flock_core::zerocheck::PaddingSpec;
use std::env::var;
use std::sync::atomic::Ordering;
use std::time::Instant;

const DOMAIN: &[u8] = b"flock-virtual-b-microbench";

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> F128 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        F128::new(self.0, self.0.rotate_left(29))
    }
}

#[test]
#[ignore] // Benchmark + byte oracle — run explicitly with --nocapture.
fn virtual_b_microbench() {
    let runs: usize = var("MICRO_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let env =
        |k: &str, d: usize| -> usize { var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) };
    let (m, k, lanes) = (
        env("MICRO_M", 32),
        env("MICRO_K", 6),
        env("MICRO_LANES", 56),
    );
    let params = PcsParams {
        m,
        log_inv_rate: 2,
        log_batch_size: k,
        profile: LigeritoProfile::Slim,
        num_lanes: Some(lanes),
        merkle_hash: HashKind::Blake3,
    };
    let cfg = params
        .ligerito_prover_config()
        .expect("m32 slim prover config");
    let log_n = params.m - 7;
    let words = 1usize << log_n;
    let d = words >> k; // lane width
    let mut rng = Rng(0x5EED_B1A5);
    // Live lanes carry data; the rest are the high-bit-lane commit's
    // definitional zeros — the same shape the leaf's own stack has.
    let mut q = vec![F128::ZERO; words];
    for w in q[..lanes * d].iter_mut() {
        *w = rng.next();
    }
    let (commitment, prover_data) = commit_lane_major(&q, &params);

    // One packed-direct EqPoint claim at a random point: exactly the merged
    // transport's inner-open configuration. The claimed value is never used
    // by the prover's work (only absorbed), so a placeholder is fine — both
    // arms absorb the same one.
    let rho: Vec<F128> = (0..log_n).map(|_| rng.next()).collect();
    let claim = || PackedDirectClaim {
        point: rho.clone(),
        value: F128::ZERO,
        eq_ind: DirectEqInd::EqPoint(rho.clone()),
    };
    let padding = PaddingSpec::dense(params.m);

    let open = |arm: u8| -> (f64, Vec<u8>) {
        VIRTUAL_B_OVERRIDE.store(arm, Ordering::Relaxed);
        let w = q.clone();
        let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
        let t = Instant::now();
        let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
            w,
            &prover_data,
            &commitment,
            &[],
            &[],
            &[claim()],
            &padding,
            &cfg,
            OpeningGrinding::disabled(),
            &mut ch,
        );
        let ms = t.elapsed().as_secs_f64() * 1e3;
        (ms, serialize(&proof).expect("serialize"))
    };

    // Warm both arms (first-touch pages, scratch pools) before timing, and
    // take the byte oracle off the warm-up pair.
    let (_, bytes_virtual) = open(1);
    let (_, bytes_jit) = open(2);
    assert_eq!(
        bytes_virtual, bytes_jit,
        "the virtual basis must be VALUE-IDENTICAL to the materialized fold"
    );

    let (mut virt, mut jit) = (Vec::new(), Vec::new());
    for i in 0..runs {
        for arm in if i % 2 == 0 { [1u8, 2] } else { [2, 1] } {
            let (ms, _) = open(arm);
            if arm == 1 { &mut virt } else { &mut jit }.push(ms);
        }
    }
    VIRTUAL_B_OVERRIDE.store(0, Ordering::Relaxed);

    let stat = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2])
    };
    let (v_min, v_med) = stat(&mut virt);
    let (j_min, j_med) = stat(&mut jit);
    println!(
        "\nVIRTUAL BASIS, inner open at m{m} / initial_k {k} / {lanes} lanes \
         ({runs} alternating pairs)\n  \
         virtual b : min {v_min:6.2} ms | median {v_med:6.2} ms\n  \
         materialized: min {j_min:6.2} ms | median {j_med:6.2} ms\n  \
         delta: {:+.2} ms on the min, {:+.2} ms on the median  (proof bytes identical)\n",
        v_min - j_min,
        v_med - j_med,
    );
}
