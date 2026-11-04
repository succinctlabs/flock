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
//! `lanes` mode is the production shape instead: a lane-major 46-lane L0
//! (`commit_lane_major`), the factored eq basis and the seeded per-lane
//! statistics, through the F256 statistics ladder
//! (`recursive_prover_with_basis_precomputed_round0_lanes`) and the affine
//! lane path (`AffineL0::Lanes`).
//!
//! `rates=1,3,5,7,9 fold=4,5,5,5,5` picks another affine ladder
//! ([`affine_ladder`]: that many levels, Johnson queries recomputed); the
//! fold grinding comes from `scripts/affine_fold_params.py` on that ladder.
//!
//! Run: `cargo bench --bench affine_vs_f256 [-- lanes] [shape=slim29] [rates=.. fold=..]`
//! Phase breakdown of one run each: `LIG_PROVE_TRACE=1 cargo bench --bench affine_vs_f256 -- trace`

use std::time::Instant;

use flock_prover::{
    challenger::FsChallenger,
    field::F128,
    init_perf_thread_pool,
    lincheck::build_eq_table,
    ntt::additive_ntt_f128::AdditiveNttF128,
    pcs::{
        commit::{PcsParams, commit_lane_major},
        ligerito::{
            LigeritoProfile, ProverConfig, VerifierConfig, VirtualEqBasis,
            affine::{
                AffineL0, affine_ladder, prove_affine, prove_affine_l0, verify_affine,
                verify_affine_l0,
            },
            ligero_commit, prover_config_for, recursive_prover_with_basis,
            recursive_prover_with_basis_precomputed_round0_lanes,
        },
    },
};

/// The bench shape: `shape=fast32` (default; the m = 32 Fast leaf) or
/// `shape=slim29` (the recursion tower's outer nodes: m = 29 Slim, 2^5
/// lanes at rate 1/4, 31 live).
struct Shape {
    log_n: usize,
    initial_k: usize,
    profile: LigeritoProfile,
    live_lanes: usize,
    rates: &'static [usize],
    fold_bits: &'static [usize],
}

/// The m = 32 Fast default ladder: L0 and three recursive levels at the
/// split ladder's rates. The split ladder's native twin has a fourth level
/// of 8 columns, where `scripts/affine_fold_params.py` finds no fold bound
/// for two of its rounds. `rates=1,2,5,7 fold=4,4,5,5` is about 3 ms faster
/// for about 21 KiB more proof. Fold grinding from the same script.
const FAST32: Shape = Shape {
    log_n: 25,
    initial_k: 6,
    profile: LigeritoProfile::Fast,
    live_lanes: 46,
    rates: &[1, 3, 5, 7],
    fold_bits: &[4, 5, 5, 5],
};

/// The tower's m = 29 Slim outer: the split ladder's rates on the three
/// native recursive levels (13, 9, 5 columns); every round certifies.
const SLIM29: Shape = Shape {
    log_n: 22,
    initial_k: 5,
    profile: LigeritoProfile::Slim,
    live_lanes: 31,
    rates: &[2, 4, 6, 8],
    fold_bits: &[4, 5, 5, 5],
};

fn shape() -> &'static Shape {
    static S: std::sync::OnceLock<&'static Shape> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        match std::env::args().find_map(|a| a.strip_prefix("shape=").map(str::to_owned)) {
            None => &FAST32,
            Some(s) if s == "fast32" => &FAST32,
            Some(s) if s == "slim29" => &SLIM29,
            Some(s) => panic!("shape must be fast32 or slim29 (got {s})"),
        }
    })
}

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

fn report(ta: Vec<f64>, tb: Vec<f64>, sa: usize, sb: usize) {
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
    let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let (na, nb) = (min(&ta), min(&tb));
    println!(
        "min:   F256 {na:.2} ms, affine {nb:.2} ms ({:+.1}%)",
        100.0 * (nb - na) / na
    );
}

fn lanes_mode(split: &ProverConfig, affine: &ProverConfig, affine_v: &VerifierConfig, trace: bool) {
    let log_cols = shape().log_n - shape().initial_k;
    let d = 1usize << log_cols;
    let mut q = rand_vec(shape().live_lanes * d, 0xA11E);
    q.resize(1 << shape().log_n, F128::ZERO);
    let point = rand_vec(shape().log_n, 0x5EED);
    let gamma = F128::new(0x1234_5678, 0x9abc);
    let low = build_eq_table(&point[..log_cols]);
    let mut seeded: Vec<F128> = (0..shape().live_lanes)
        .map(|l| {
            q[l * d..(l + 1) * d]
                .iter()
                .zip(&low)
                .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b)
        })
        .collect();
    seeded.resize(1 << shape().initial_k, F128::ZERO);
    let target = build_eq_table(&point[log_cols..])
        .iter()
        .zip(&seeded)
        .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b)
        * gamma;
    let params = PcsParams {
        m: shape().log_n + 7,
        log_inv_rate: split.log_inv_rates[0],
        log_batch_size: shape().initial_k,
        profile: shape().profile,
        num_lanes: Some(shape().live_lanes),
        merkle_hash: split.merkle_hash,
    };
    let (commitment, pd) = commit_lane_major(&q, &params);

    let run_f256 = || {
        let w = q.clone();
        let mut ch = FsChallenger::new(b"affine-ab-lanes");
        let t = Instant::now();
        let proof = recursive_prover_with_basis_precomputed_round0_lanes(
            split,
            w,
            Vec::new(),
            target,
            &pd.codeword,
            &pd.merkle_tree,
            shape().live_lanes,
            true,
            (F128::ZERO, F128::ZERO),
            None,
            None,
            Some(VirtualEqBasis::new(point.clone(), gamma)),
            Some(seeded.clone()),
            &mut ch,
        );
        (t.elapsed().as_secs_f64() * 1e3, proof.size_bytes())
    };
    let run_affine = || {
        let w = q.clone();
        let mut ch = FsChallenger::new(b"affine-ab-lanes");
        let t = Instant::now();
        let proof = prove_affine_l0(
            affine,
            AffineL0::Lanes {
                witness: w,
                point: point.clone(),
                gamma,
                seeded_stats: Some(seeded.clone()),
                num_lanes: shape().live_lanes,
            },
            target,
            &pd.codeword,
            &pd.merkle_tree,
            &mut ch,
        );
        (t.elapsed().as_secs_f64() * 1e3, proof)
    };
    if trace {
        for _ in 0..3 {
            run_f256();
            run_affine();
        }
        return;
    }
    let (_, proof) = run_affine();
    assert_eq!(
        commitment.cap.len(),
        1usize << affine.level_zero_cap_depth(),
        "split and affine share the L0 cap depth"
    );
    let mut rotated = point.clone();
    rotated.rotate_left(log_cols);
    let mut ch = FsChallenger::new(b"affine-ab-lanes");
    let t = Instant::now();
    let ok = verify_affine_l0(
        affine_v,
        &proof,
        shape().log_n,
        target,
        &commitment.cap,
        (shape().live_lanes, Some(&point[..log_cols])),
        |rs, residual_log| {
            let prefix = rotated
                .iter()
                .zip(rs)
                .fold(gamma, |acc, (&zi, &ri)| acc * (F128::ONE + zi + ri));
            let mut out = build_eq_table(&rotated[rs.len()..]);
            assert_eq!(out.len(), 1usize << residual_log);
            out.iter_mut().for_each(|v| *v *= prefix);
            out
        },
        &mut ch,
    );
    println!(
        "affine verify (lanes): {ok} ({:.2} ms)",
        t.elapsed().as_secs_f64() * 1e3
    );
    assert!(ok, "affine lanes proof must verify");
    for _ in 0..2 {
        run_f256();
        run_affine();
    }
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    let (mut sa, mut sb) = (0, 0);
    for _ in 0..9 {
        let (t, s) = run_f256();
        ta.push(t);
        sa = s;
        let (t, p) = run_affine();
        tb.push(t);
        sb = p.size_bytes();
    }
    report(ta, tb, sa, sb);
}

/// Rough in-circuit verifier cost of one opening, per level: BLAKE3 rows
/// for the Merkle openings (leaf chunk blocks plus path compressions below
/// the cap, plus the cap tree), and the final level's residual work. A
/// child-verifier estimate for the recursion tower, not a measurement.
/// `affine` levels commit F128 rows of `2^k` words; split recursive rows
/// are `2^k` words too (the F256 coordinate bit is one of the k).
fn census(label: &str, cfg: &ProverConfig, affine: bool) {
    let levels = cfg.log_inv_rates.len();
    let (mut b3, mut sib) = (0usize, 0usize);
    println!("{label}:");
    for (l, sch) in cfg.stratified.iter().enumerate() {
        let words = if l == 0 {
            shape().live_lanes
        } else {
            1usize << cfg.recursive_ks[l - 1]
        };
        let (q, d, c1) = (sch.queries(), sch.log_block_len, sch.cap_depth());
        let rows = q * (words.div_ceil(4) + d - c1) + (1usize << c1) - 1;
        b3 += rows;
        sib += q * (d - c1);
        println!("  L{l}: q={q:>3} d={d:>2} cap={c1} row={words:>2} words -> {rows:>5} b3 rows");
    }
    let c = cfg.recursive_log_msg_cols[levels - 2];
    let q_final = cfg.queries[levels - 1];
    let k_last = cfg.recursive_ks[levels - 2];
    let (yr_words, final_macs) = if affine {
        // Enc(yr)(x_q) per final query: one MAC per coefficient.
        let n = (1usize << c) + k_last;
        (n, q_final * n)
    } else {
        // yr is 2^c F256 elements; one F256 dot against the residual weights.
        (2usize << c, 1usize << c)
    };
    let rounds: usize = cfg.initial_k + cfg.recursive_ks.iter().sum::<usize>();
    println!(
        "  total {b3} b3 rows ({sib} siblings), {rounds} fold rounds, final c={c}: yr {yr_words} words, \
         ~{final_macs} final-check MAC rows ({})",
        if affine { "F128, per query" } else { "F256, once" }
    );
}

fn main() {
    init_perf_thread_pool();
    let trace_only = std::env::args().any(|a| a == "trace");
    let split = prover_config_for(shape().log_n, shape().initial_k, shape().profile).expect("split config");
    let list = |key: &str| {
        std::env::args()
            .find_map(|a| a.strip_prefix(key).map(str::to_owned))
            .map(|v| {
                v.split(',')
                    .map(|x| x.parse::<usize>().unwrap())
                    .collect::<Vec<_>>()
            })
    };
    let (affine, affine_v) = match list("rates=") {
        Some(rates) => affine_ladder(&split, shape().log_n, &rates, &list("fold=").expect("fold=..")),
        None => affine_ladder(&split, shape().log_n, shape().rates, shape().fold_bits),
    };
    println!(
        "affine ladder: ks {:?}, rates {:?}, queries {:?}, fold bits {:?}",
        affine.recursive_ks, affine.log_inv_rates, affine.queries, affine.fold_grinding_bits
    );
    if std::env::args().any(|a| a == "census") {
        census("F256 split", &split, false);
        census("affine", &affine, true);
        return;
    }
    if std::env::args().any(|a| a == "lanes") {
        lanes_mode(&split, &affine, &affine_v, trace_only);
        return;
    }
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

    let witness = rand_vec(1 << shape().log_n, 0xA11E);
    let z = rand_vec(shape().log_n, 0x5EED);
    let basis = build_eq_table(&z);
    let target = witness
        .iter()
        .zip(&basis)
        .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);
    let log_cols = shape().log_n - shape().initial_k;
    let rate = split.log_inv_rates[0];
    let ntt = AdditiveNttF128::standard(log_cols + rate);
    let l0 = ligero_commit(&witness, log_cols, shape().initial_k, rate, &ntt, split.merkle_hash);

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
        shape().log_n,
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
    report(ta, tb, sa, sb);
}
