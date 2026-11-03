//! A/B of the affine F128 prototype (`pcs::ligerito::affine`) against the
//! F256 split ladder (`recursive_prover_with_basis`) at m = 32 Fast, on the
//! plain native path: the same 2^25-word witness, dense eq basis and L0
//! commitment (2^6 LSB lanes, rate 1/2) go through both provers. Runs are
//! interleaved; each prover gets its own clones outside the timer.
//!
//! This is not the production open: that path uses a lane-major 46-lane L0,
//! a virtual basis and the fused/lookahead F256 kernels, none of which the
//! prototype has yet. The F256 side here still uses its fused kernels on the
//! materialized-basis path, so the fold columns favour it.
//!
//! Run: `cargo bench --bench affine_vs_f256`
//! Phase breakdown of one run each: `LIG_PROVE_TRACE=1 cargo bench --bench affine_vs_f256 -- trace`

use std::time::Instant;

use flock_prover::{
    challenger::FsChallenger,
    field::F128,
    init_perf_thread_pool,
    lincheck::build_eq_table,
    ntt::additive_ntt_f128::AdditiveNttF128,
    pcs::ligerito::{
        LigeritoProfile,
        affine::{affine_configs, prove_affine, verify_affine},
        ligero_commit, prover_config_for, recursive_prover_with_basis,
    },
};

const LOG_N: usize = 25;
const INITIAL_K: usize = 6;
/// Per-level fold grinding from `scripts/affine_fold_params.py` on m32_fast.
const FOLD_BITS: [usize; 6] = [4, 5, 5, 5, 5, 5];

fn rand_vec(len: usize, seed: u64) -> Vec<F128> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let lo = s;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            F128::new(lo, s)
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    init_perf_thread_pool();
    let trace_only = std::env::args().any(|a| a == "trace");
    let split = prover_config_for(LOG_N, INITIAL_K, LigeritoProfile::Fast).expect("m32 fast");
    let (affine, affine_v) = affine_configs(&split, LOG_N, &FOLD_BITS);
    println!(
        "split ladder:  ks {:?}, cols {:?}, rates {:?}",
        split.recursive_ks, split.recursive_log_msg_cols, split.log_inv_rates
    );
    println!(
        "affine ladder: ks {:?}, cols {:?}, rates {:?}, fold bits {:?}",
        affine.recursive_ks,
        affine.recursive_log_msg_cols,
        affine.log_inv_rates,
        affine.fold_grinding_bits
    );

    let witness = rand_vec(1 << LOG_N, 0xA11E);
    let z = rand_vec(LOG_N, 0x5EED);
    let basis = build_eq_table(&z);
    let target = witness
        .iter()
        .zip(&basis)
        .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);
    let log_cols = LOG_N - INITIAL_K;
    let rate = split.log_inv_rates[0];
    let ntt = AdditiveNttF128::standard(log_cols + rate);
    let l0 = ligero_commit(&witness, log_cols, INITIAL_K, rate, &ntt, split.merkle_hash);

    let run_f256 = || {
        let (w, b) = (witness.clone(), basis.clone());
        let mut ch = FsChallenger::new(b"affine-ab");
        let t = Instant::now();
        let proof = recursive_prover_with_basis(&split, w, b, target, &l0.mat, &l0.tree, &mut ch);
        (t.elapsed().as_secs_f64() * 1e3, proof.size_bytes())
    };
    let run_affine = || {
        let (w, b) = (witness.clone(), basis.clone());
        let mut ch = FsChallenger::new(b"affine-ab");
        let t = Instant::now();
        let proof = prove_affine(&affine, w, b, target, &l0.mat, &l0.tree, &mut ch);
        (t.elapsed().as_secs_f64() * 1e3, proof)
    };

    if trace_only {
        // Three rounds each; the last pair of traces is warm.
        for _ in 0..3 {
            run_f256();
            run_affine();
        }
        return;
    }

    // Correctness once: the affine proof verifies.
    let (_, proof) = run_affine();
    let mut ch = FsChallenger::new(b"affine-ab");
    let cap = l0.cap(affine.level_zero_cap_depth()).to_vec();
    let t = Instant::now();
    let ok = verify_affine(
        &affine_v,
        &proof,
        LOG_N,
        target,
        &cap,
        |rs, residual_log| {
            // eq(z) bound at rs: a prefix scalar times eq over the rest.
            let prefix = z
                .iter()
                .zip(rs)
                .fold(F128::ONE, |acc, (&zi, &ri)| acc * (F128::ONE + zi + ri));
            let mut out = build_eq_table(&z[rs.len()..]);
            assert_eq!(out.len(), 1usize << residual_log);
            out.iter_mut().for_each(|v| *v *= prefix);
            out
        },
        &mut ch,
    );
    println!(
        "affine verify: {ok} ({:.2} ms)",
        t.elapsed().as_secs_f64() * 1e3
    );
    assert!(ok, "affine proof must verify");

    for _ in 0..2 {
        run_f256();
        run_affine();
    }
    let reps = 9;
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    let (mut sa, mut sb) = (0, 0);
    for _ in 0..reps {
        let (t, s) = run_f256();
        ta.push(t);
        sa = s;
        let (t, p) = run_affine();
        tb.push(t);
        sb = p.size_bytes();
    }
    let (ma, mb) = (median(ta.clone()), median(tb.clone()));
    println!(
        "F256 split: median {ma:.2} ms  (all {ta:.1?})  proof {:.1} KiB",
        sa as f64 / 1024.0
    );
    println!(
        "affine:     median {mb:.2} ms  (all {tb:.1?})  proof {:.1} KiB",
        sb as f64 / 1024.0
    );
    println!(
        "delta: {:+.2} ms ({:+.1}%)",
        mb - ma,
        100.0 * (mb - ma) / ma
    );
}
