//! E6 — end-to-end timing: L1′ prove (probe pipeline) vs production
//! `KeccakSetup::prove_fast_basefold`, plus verify times. Both pipelines
//! include witness generation.
//!
//! Usage: cargo run --release --bin e6_end_to_end -- [--m 23,26,29]
//!   [--iters 5] [--tsv out.tsv]

use flock_autoresearch_probes::e6::{keccak_setup, prove_l1_keccak, verify_l1_keccak};
use flock_autoresearch_probes::keccak_witness::random_state;
use flock_core::challenger::FsChallenger;
use flock_prover::r1cs_hashes::keccak::{K_LOG, KeccakSetup, State};
use std::io::Write;
use std::time::Instant;

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn time_median<F: FnMut()>(mut f: F, iters: usize) -> (f64, f64) {
    f();
    let mut ts: Vec<f64> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64()
        })
        .collect();
    ts.sort_by(f64::total_cmp);
    (ts[ts.len() / 2], ts[0])
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let args: Vec<String> = std::env::args().collect();
    let ms: Vec<usize> = parse_flag(&args, "--m").map_or_else(
        || vec![23, 26, 29],
        |v| v.split(',').map(|s| s.parse().unwrap()).collect(),
    );
    let iters: usize = parse_flag(&args, "--iters").map_or(5, |v| v.parse().unwrap());
    let tsv = parse_flag(&args, "--tsv");
    let threads = rayon::current_num_threads();

    for &m in &ms {
        let n_log = m - K_LOG;
        let n = 1usize << n_log;
        eprintln!("\n== keccak e2e: m={m} ({n} instances), {threads} threads ==");
        let states: Vec<State> = (0..n as u64).map(random_state).collect();

        // ---- L1' pipeline: roundtrip gate, then timing.
        let (r1cs, pcs_params) = keccak_setup(n_log);
        {
            let mut chp = FsChallenger::new(b"flock-e6-v0");
            let res = prove_l1_keccak(&r1cs, &pcs_params, &states, &mut chp);
            let mut chv = FsChallenger::new(b"flock-e6-v0");
            verify_l1_keccak(&r1cs, &res.commitment, &res.proof, &mut chv)
                .expect("L1' roundtrip");
            eprintln!("L1' prove->verify roundtrip: OK");
        }

        // ---- Production pipeline (BaseFold): roundtrip gate, then timing.
        let setup = KeccakSetup::new(n);
        {
            let mut chp = FsChallenger::new(b"flock-e6-v0");
            let (proof, commitment, _) = setup.prove_fast_basefold(&states, &mut chp);
            let mut chv = FsChallenger::new(b"flock-e6-v0");
            setup
                .verify_basefold(&commitment, &proof, &mut chv)
                .expect("production roundtrip");
            eprintln!("production prove->verify roundtrip: OK");
        }

        let mut rows: Vec<(&str, f64, f64)> = Vec::new();

        let t = time_median(
            || {
                let mut ch = FsChallenger::new(b"flock-e6-v0");
                std::hint::black_box(setup.prove_fast_basefold(&states, &mut ch));
            },
            iters,
        );
        rows.push(("prove-production", t.0, t.1));

        let t = time_median(
            || {
                let mut ch = FsChallenger::new(b"flock-e6-v0");
                std::hint::black_box(prove_l1_keccak(&r1cs, &pcs_params, &states, &mut ch));
            },
            iters,
        );
        rows.push(("prove-l1", t.0, t.1));

        // Verify timings (single proof each).
        let mut chp = FsChallenger::new(b"flock-e6-v0");
        let (p_prod, c_prod, _) = setup.prove_fast_basefold(&states, &mut chp);
        let t = time_median(
            || {
                let mut ch = FsChallenger::new(b"flock-e6-v0");
                std::hint::black_box(setup.verify_basefold(&c_prod, &p_prod, &mut ch).unwrap());
            },
            iters,
        );
        rows.push(("verify-production", t.0, t.1));

        let mut chp = FsChallenger::new(b"flock-e6-v0");
        let res = prove_l1_keccak(&r1cs, &pcs_params, &states, &mut chp);
        let t = time_median(
            || {
                let mut ch = FsChallenger::new(b"flock-e6-v0");
                std::hint::black_box(
                    verify_l1_keccak(&r1cs, &res.commitment, &res.proof, &mut ch).unwrap(),
                );
            },
            iters,
        );
        rows.push(("verify-l1", t.0, t.1));

        println!("{:<20} {:>10} {:>10}", "variant", "median ms", "min ms");
        for (name, med, min) in &rows {
            println!("{:<20} {:>10.2} {:>10.2}", name, med * 1e3, min * 1e3);
            if let Some(p) = &tsv {
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .expect("open tsv");
                writeln!(
                    f,
                    "keccak\t{m}\t{K_LOG}\t{n_log}\t{threads}\t{name}\t{:.4}\t{:.4}",
                    med * 1e3,
                    min * 1e3
                )
                .unwrap();
            }
        }
    }
}
