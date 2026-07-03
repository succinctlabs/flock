//! E6 — end-to-end timing: L1′ pipeline vs production `prove_fast*`, per
//! hash and PCS backend. Both pipelines include witness generation and use
//! scratch-recycled buffers.
//!
//! Usage: cargo run --release --bin e6_end_to_end --
//!   [--hash all|keccak|sha2|blake3] [--backend both|basefold|ligerito]
//!   [--m 23,26,29] [--iters 5] [--tsv out.tsv]

use flock_autoresearch_probes::e6::{
    L1HashSpec, prove_l1_basefold, prove_l1_ligerito, setup, verify_l1_basefold,
    verify_l1_ligerito,
};
use flock_autoresearch_probes::{blake3_vwide, blake3_witness, keccak_vwide, keccak_witness,
    sha2_vwide, sha2_witness};
use flock_core::challenger::FsChallenger;
use flock_core::lincheck::LincheckCircuit;
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

struct Report {
    tsv: Option<String>,
    hash: &'static str,
    m: usize,
    threads: usize,
}

impl Report {
    fn row(&self, name: &str, (med, min): (f64, f64)) {
        println!("{:<26} {:>10.2} {:>10.2}", name, med * 1e3, min * 1e3);
        if let Some(p) = &self.tsv {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("open tsv");
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{:.4}\t{:.4}",
                self.hash,
                self.m,
                self.threads,
                name,
                med * 1e3,
                min * 1e3
            )
            .unwrap();
        }
    }
}

/// Production prove/verify closures per backend, so the generic runner can
/// treat all hashes uniformly.
#[allow(clippy::too_many_arguments)]
fn run_hash<S: Sync>(
    hash: &'static str,
    r1cs: flock_core::r1cs::BlockR1cs,
    circuit: &dyn LincheckCircuit,
    inputs: &[S],
    direct: &(dyn Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64]) + Sync),
    prod_basefold: &dyn Fn(&[S], &mut FsChallenger),
    prod_ligerito: Option<&dyn Fn(&[S], &mut FsChallenger)>,
    backends: &[&str],
    iters: usize,
    tsv: &Option<String>,
) {
    let (r1cs, pcs_params) = setup(r1cs);
    let m = r1cs.m;
    let spec = L1HashSpec {
        r1cs: &r1cs,
        circuit,
        direct,
    };
    let rep = Report {
        tsv: tsv.clone(),
        hash,
        m,
        threads: rayon::current_num_threads(),
    };
    eprintln!(
        "\n== {hash} e2e: m={m} ({} instances), {} threads ==",
        inputs.len(),
        rep.threads
    );

    if backends.contains(&"basefold") {
        // Gate: roundtrip.
        let mut ch = FsChallenger::new(b"flock-e6-v1");
        let res = prove_l1_basefold(&spec, &pcs_params, inputs, true, &mut ch);
        let mut chv = FsChallenger::new(b"flock-e6-v1");
        verify_l1_basefold(&r1cs, circuit, &res.commitment, &res.proof, &mut chv)
            .expect("L1' basefold roundtrip");
        eprintln!("basefold roundtrip: OK");

        rep.row(
            "prove-production-bf",
            time_median(
                || {
                    let mut ch = FsChallenger::new(b"flock-e6-v1");
                    prod_basefold(inputs, &mut ch);
                },
                iters,
            ),
        );
        rep.row(
            "prove-l1-bf",
            time_median(
                || {
                    let mut ch = FsChallenger::new(b"flock-e6-v1");
                    std::hint::black_box(prove_l1_basefold(
                        &spec,
                        &pcs_params,
                        inputs,
                        true,
                        &mut ch,
                    ));
                },
                iters,
            ),
        );
        rep.row(
            "verify-l1-bf",
            time_median(
                || {
                    let mut ch = FsChallenger::new(b"flock-e6-v1");
                    std::hint::black_box(
                        verify_l1_basefold(&r1cs, circuit, &res.commitment, &res.proof, &mut ch)
                            .unwrap(),
                    );
                },
                iters,
            ),
        );
    }

    if backends.contains(&"ligerito") {
        if let Some(prod_lig) = prod_ligerito {
            let mut ch = FsChallenger::new(b"flock-e6-v1");
            let res = prove_l1_ligerito(&spec, &pcs_params, inputs, true, &mut ch);
            let mut chv = FsChallenger::new(b"flock-e6-v1");
            verify_l1_ligerito(
                &r1cs,
                circuit,
                &res.commitment,
                &res.proof,
                &pcs_params,
                &mut chv,
            )
            .expect("L1' ligerito roundtrip");
            eprintln!("ligerito roundtrip: OK");

            rep.row(
                "prove-production-lig",
                time_median(
                    || {
                        let mut ch = FsChallenger::new(b"flock-e6-v1");
                        prod_lig(inputs, &mut ch);
                    },
                    iters,
                ),
            );
            rep.row(
                "prove-l1-lig",
                time_median(
                    || {
                        let mut ch = FsChallenger::new(b"flock-e6-v1");
                        std::hint::black_box(prove_l1_ligerito(
                            &spec,
                            &pcs_params,
                            inputs,
                            true,
                            &mut ch,
                        ));
                    },
                    iters,
                ),
            );
            rep.row(
                "verify-l1-lig",
                time_median(
                    || {
                        let mut ch = FsChallenger::new(b"flock-e6-v1");
                        std::hint::black_box(
                            verify_l1_ligerito(
                                &r1cs,
                                circuit,
                                &res.commitment,
                                &res.proof,
                                &pcs_params,
                                &mut ch,
                            )
                            .unwrap(),
                        );
                    },
                    iters,
                ),
            );
        }
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let args: Vec<String> = std::env::args().collect();
    let hashes: Vec<&str> = match parse_flag(&args, "--hash").as_deref() {
        None | Some("all") => vec!["keccak", "sha2", "blake3"],
        Some("keccak") => vec!["keccak"],
        Some("sha2") => vec!["sha2"],
        Some("blake3") => vec!["blake3"],
        Some(o) => panic!("unknown hash {o}"),
    };
    let backends: Vec<&str> = match parse_flag(&args, "--backend").as_deref() {
        None | Some("both") => vec!["basefold", "ligerito"],
        Some("basefold") => vec!["basefold"],
        Some("ligerito") => vec!["ligerito"],
        Some(o) => panic!("unknown backend {o}"),
    };
    let ms: Vec<usize> = parse_flag(&args, "--m").map_or_else(
        || vec![23, 26, 29],
        |v| v.split(',').map(|s| s.parse().unwrap()).collect(),
    );
    let iters: usize = parse_flag(&args, "--iters").map_or(5, |v| v.parse().unwrap());
    let tsv = parse_flag(&args, "--tsv");

    for hash in &hashes {
        for &m in &ms {
            match *hash {
                "keccak" => {
                    use flock_prover::r1cs_hashes::keccak::{
                        K_LOG, KeccakLincheckCircuit, KeccakSetup, build_block_r1cs,
                    };
                    let n_log = m - K_LOG;
                    let n = 1usize << n_log;
                    let inputs: Vec<_> =
                        (0..n as u64).map(keccak_witness::random_state).collect();
                    let prod = KeccakSetup::new(n);
                    run_hash(
                        "keccak",
                        build_block_r1cs(n_log),
                        &KeccakLincheckCircuit,
                        &inputs,
                        &keccak_vwide::build_l1_direct,
                        &|st, ch| {
                            std::hint::black_box(prod.prove_fast_basefold(st, ch));
                        },
                        Some(&|st, ch| {
                            std::hint::black_box(prod.prove_fast(st, ch));
                        }),
                        &backends,
                        iters,
                        &tsv,
                    );
                }
                "sha2" => {
                    use flock_prover::r1cs_hashes::sha2::{
                        K_LOG, Sha256HybridSetup, build_block_r1cs,
                    };
                    let n_log = m - K_LOG;
                    let n = 1usize << n_log;
                    let inputs: Vec<_> =
                        (0..n as u64).map(sha2_witness::random_input).collect();
                    let prod = Sha256HybridSetup::new(n);
                    let r1cs = build_block_r1cs(n_log);
                    let circuit = r1cs.csc_lincheck_circuit().clone();
                    run_hash(
                        "sha2",
                        r1cs,
                        &circuit,
                        &inputs,
                        &sha2_vwide::build_l1_direct,
                        &|st, ch| {
                            std::hint::black_box(prod.prove_fast_basefold(st, ch));
                        },
                        Some(&|st, ch| {
                            std::hint::black_box(prod.prove_fast(st, ch));
                        }),
                        &backends,
                        iters,
                        &tsv,
                    );
                }
                "blake3" => {
                    use flock_prover::r1cs_hashes::blake3::{
                        Blake3Setup, K_LOG, build_block_r1cs,
                    };
                    let n_log = m - K_LOG;
                    let n = 1usize << n_log;
                    let inputs: Vec<_> =
                        (0..n as u64).map(blake3_witness::random_input).collect();
                    let prod = Blake3Setup::new(n);
                    let r1cs = build_block_r1cs(n_log);
                    let circuit = r1cs.csc_lincheck_circuit().clone();
                    run_hash(
                        "blake3",
                        r1cs,
                        &circuit,
                        &inputs,
                        &blake3_vwide::build_l1_direct,
                        &|st, ch| {
                            std::hint::black_box(prod.prove_fast_basefold(st, ch));
                        },
                        Some(&|st, ch| {
                            std::hint::black_box(prod.prove_fast(st, ch));
                        }),
                        &backends,
                        iters,
                        &tsv,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}
