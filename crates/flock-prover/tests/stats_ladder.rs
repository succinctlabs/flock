//! **THE STATISTICS LADDER, isolated.** The lane-major inner open's L0
//! rounds bind the lane-block bits, and its basis is a sum of rank-1 eq
//! tensors (the seeded merged-transport point plus the L0 OOD point). The
//! ladder contracts each term against `f` ONCE into per-block statistics
//! (one sweep), derives every L0 round message from those (O(2^initial_k)
//! per round instead of O(L)), and composes the `initial_k` array folds into
//! one pass. Same field elements, so the two arms must produce byte-identical
//! proofs — this is the ORACLE (any divergence is a bug in the algebra) and
//! the INSTRUMENT (alternating in-process arms on the same committed stack).
//!
//! Default geometry is the m32 chain leaf's inner open: 2^25 packed words,
//! slim rate 1/4, `initial_k = 6`, 56 committed lanes of 64. Knobs as in
//! `virtual_b.rs` (`MICRO_RUNS`, `MICRO_M`, `MICRO_K`, `MICRO_LANES`).
use flock_core::challenger::FsChallenger;
use flock_core::field::F128;
use flock_core::merkle::HashKind;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{
    DirectEqInd, OpeningGrinding, PackedDirectClaim, PcsParams, STATS_LADDER_OVERRIDE,
    commit_lane_major, open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding,
};
use flock_core::zerocheck::PaddingSpec;
use std::sync::atomic::Ordering;

const DOMAIN: &[u8] = b"flock-stats-ladder-microbench";

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> F128 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        F128::new(self.0, self.0.rotate_left(29))
    }
}

fn run(m: usize, k: usize, lanes: usize, runs: usize) {
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
        .expect("slim prover config for this m");
    let log_n = params.m - 7;
    let words = 1usize << log_n;
    let d = words >> k;
    let mut rng = Rng(0x5747_5A7D);
    let mut q = vec![F128::ZERO; words];
    for w in q[..lanes * d].iter_mut() {
        *w = rng.next();
    }
    let (commitment, prover_data) = commit_lane_major(&q, &params);
    let rho: Vec<F128> = (0..log_n).map(|_| rng.next()).collect();
    let claim = || PackedDirectClaim {
        point: rho.clone(),
        value: F128::ZERO,
        eq_ind: DirectEqInd::EqPoint(rho.clone()),
    };
    let padding = PaddingSpec::dense(params.m);
    let open = |arm: u8| -> (f64, Vec<u8>) {
        STATS_LADDER_OVERRIDE.store(arm, Ordering::Relaxed);
        let w = q.clone();
        let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
        let t = std::time::Instant::now();
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
        (ms, bincode::serialize(&proof).expect("serialize"))
    };
    let (_, bytes_stats) = open(1);
    let (_, bytes_ladder) = open(2);
    assert_eq!(
        bytes_stats, bytes_ladder,
        "the statistics ladder must be VALUE-IDENTICAL to the incremental ladder"
    );
    let (mut st, mut la) = (Vec::new(), Vec::new());
    for i in 0..runs {
        for arm in if i % 2 == 0 { [1u8, 2] } else { [2, 1] } {
            let (ms, _) = open(arm);
            if arm == 1 { &mut st } else { &mut la }.push(ms);
        }
    }
    STATS_LADDER_OVERRIDE.store(0, Ordering::Relaxed);
    let stat = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2])
    };
    let (s_min, s_med) = stat(&mut st);
    let (l_min, l_med) = stat(&mut la);
    println!(
        "\nSTATISTICS LADDER, inner open at m{m} / initial_k {k} / {lanes} lanes \
         ({runs} alternating pairs)\n  \
         statistics : min {s_min:6.2} ms | median {s_med:6.2} ms\n  \
         incremental: min {l_min:6.2} ms | median {l_med:6.2} ms\n  \
         delta: {:+.2} ms on the min, {:+.2} ms on the median  (proof bytes identical)\n",
        s_min - l_min,
        s_med - l_med,
    );
}

/// The byte oracle at a small lane-major geometry — always on.
#[test]
fn stats_ladder_matches_incremental_small() {
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    run(
        env("SMALL_M", 26),
        env("MICRO_K", 6),
        env("MICRO_LANES", 56),
        1,
    );
}

#[test]
#[ignore] // Benchmark + byte oracle at the m32 leaf — run explicitly with --nocapture.
fn stats_ladder_microbench() {
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    run(
        env("MICRO_M", 32),
        env("MICRO_K", 6),
        env("MICRO_LANES", 56),
        env("MICRO_RUNS", 5),
    );
}
