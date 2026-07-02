//! E1 — witness-gen producer bench: row-major baseline vs staged L1′ scatter,
//! across all three hash encoders.
//!
//! Usage:
//!   cargo run --release --bin e1_witness_gen -- \
//!     [--hash keccak|sha2|blake3|all] [--m 23,26,29] [--iters 7] \
//!     [--groups auto|8,64,...] [--tsv results.tsv]
//!
//! Per (hash, m, thread-count) cell, reports median/min wall time for each
//! variant; `--tsv` appends machine-readable rows. Thread count comes from
//! the environment (perf pool by default, `RAYON_NUM_THREADS=1` for
//! single-core).

use flock_autoresearch_probes::layout::Layout;
use flock_autoresearch_probes::producer::{
    PerBlock, auto_group, build_compute_only, build_l1_staged_opts_nt, build_row_major,
    build_row_major_with_stripe,
};
use flock_autoresearch_probes::{blake3_witness, keccak_witness, sha2_witness};
use flock_core::field::F128;
use std::io::Write;
use std::time::Instant;

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn time_median<F: FnMut()>(mut f: F, iters: usize) -> (f64, f64) {
    f(); // warmup
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

struct Reporter {
    tsv: Option<std::fs::File>,
    hash: &'static str,
    m: usize,
    k_log: usize,
    threads: usize,
    out_gb: f64,
}

impl Reporter {
    fn row(&mut self, variant: &str, group: usize, (med, min): (f64, f64)) {
        println!(
            "{:<26} {:>10.2} {:>10.2} {:>12.1}",
            format!("{variant} (G={group})"),
            med * 1e3,
            min * 1e3,
            self.out_gb / med
        );
        if let Some(f) = &mut self.tsv {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}",
                self.hash,
                self.m,
                self.k_log,
                self.m - self.k_log,
                self.threads,
                variant,
                group,
                med * 1e3,
                min * 1e3
            )
            .unwrap();
        }
    }
}

type Driver4 = (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>);

fn run_hash<S: Sync>(
    hash: &'static str,
    k_log: usize,
    useful_bits: usize,
    m: usize,
    iters: usize,
    groups: &Option<Vec<usize>>,
    tsv: &Option<String>,
    gen_input: impl Fn(u64) -> S,
    production: impl Fn(&[S], usize) -> Driver4,
    per_block: &impl PerBlock<S>,
    // C2: direct-write V-wide producer (states, n_log, stripe, z, a, b).
    direct: Option<&dyn Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64])>,
) {
    assert!(m > k_log + 2, "need n_log >= 3 for {hash}");
    let n_log = m - k_log;
    let n = 1usize << n_log;
    let u64_per_block = (1usize << k_log) / 64;
    let total_u64 = n * u64_per_block;
    let out_gb = 3.0 * (total_u64 * 8) as f64 / 1e9;
    let threads = rayon::current_num_threads();
    let useful_chunks = useful_bits.div_ceil(128);
    let chunks_per_block = u64_per_block / 2;

    eprintln!(
        "\n== {hash}: m={m} (k_log={k_log}, n_log={n_log}, {n} instances, useful {useful_chunks}/{chunks_per_block} chunks), \
         3x{:.3} GB out, {threads} threads ==",
        out_gb / 3.0
    );

    let inputs: Vec<S> = (0..n as u64).map(gen_input).collect();

    // Correctness at small scale: staged L1' == word-transposed row-major.
    {
        let ck_n_log = n_log.min(7).max(3);
        let ck_n = 1usize << ck_n_log;
        let ck = &inputs[..ck_n];
        let ck_total = ck_n * u64_per_block;
        let l = Layout::new(k_log, ck_n_log);
        let mut row = vec![vec![0u64; ck_total]; 3];
        let mut l1 = vec![vec![0u64; ck_total]; 3];
        {
            let [z, rest @ ..] = &mut row[..] else { unreachable!() };
            let [a, b] = rest else { unreachable!() };
            build_row_major(ck, k_log, ck_n_log, 8, per_block, z, a, b);
        }
        for (g, nt) in [(8usize, false), (ck_n.min(64), true)] {
            let [z, rest @ ..] = &mut l1[..] else { unreachable!() };
            let [a, b] = rest else { unreachable!() };
            build_l1_staged_opts_nt(
                ck, k_log, ck_n_log, g, chunks_per_block, None, nt, per_block, z, a, b,
            );
            for (name, r, o) in [("z", &row[0], &l1[0]), ("a", &row[1], &l1[1]), ("b", &row[2], &l1[2])] {
                assert_eq!(
                    &l.permute_words_u64_row_to_l1(r),
                    o,
                    "{hash}: staged L1' (G={g}, nt={nt}) mismatch on {name}"
                );
            }
        }
        eprintln!("correctness check: OK");
    }

    let mut z = vec![0u64; total_u64];
    let mut a = vec![0u64; total_u64];
    let mut b = vec![0u64; total_u64];
    let mut stripe = vec![0u8; (n / 8) * u64_per_block * 64];

    let mut rep = Reporter {
        tsv: tsv.as_ref().map(|p| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("open tsv")
        }),
        hash,
        m,
        k_log,
        threads,
        out_gb,
    };
    println!("{:<26} {:>10} {:>10} {:>12}", "variant", "median ms", "min ms", "GB/s (med)");

    rep.row(
        "production-driver+stripe",
        8,
        time_median(
            || {
                let (zr, ar, br, sr) = production(&inputs, n_log);
                std::hint::black_box((&zr, &ar, &br, &sr));
                flock_core::scratch::give_f128(zr);
                flock_core::scratch::give_f128(ar);
                flock_core::scratch::give_f128(br);
            },
            iters,
        ),
    );

    rep.row(
        "row-major",
        8,
        time_median(
            || build_row_major(&inputs, k_log, n_log, 8, per_block, &mut z, &mut a, &mut b),
            iters,
        ),
    );

    // The FAIR row-major baseline: same work as production (z/a/b + fused
    // stripe) but with recycled buffers — isolates layout effects from
    // production's per-call stripe allocation.
    rep.row(
        "row-major-stripe-recycled",
        8,
        time_median(
            || {
                build_row_major_with_stripe(
                    &inputs,
                    k_log,
                    n_log,
                    8,
                    per_block,
                    Some(&mut stripe),
                    &mut z,
                    &mut a,
                    &mut b,
                )
            },
            iters,
        ),
    );

    let gs: Vec<usize> = match groups {
        Some(v) => v.clone(),
        None => vec![auto_group(n, threads)],
    };
    for &g in &gs {
        if g > n {
            continue;
        }
        rep.row(
            "L1-full",
            g,
            time_median(
                || {
                    build_l1_staged_opts_nt(
                        &inputs, k_log, n_log, g, chunks_per_block, None, false, per_block,
                        &mut z, &mut a, &mut b,
                    )
                },
                iters,
            ),
        );
        rep.row(
            "L1-useful-nt",
            g,
            time_median(
                || {
                    build_l1_staged_opts_nt(
                        &inputs, k_log, n_log, g, useful_chunks, None, true, per_block, &mut z,
                        &mut a, &mut b,
                    )
                },
                iters,
            ),
        );
        if g >= 8 {
            rep.row(
                "L1-useful-stripe-nt",
                g,
                time_median(
                    || {
                        build_l1_staged_opts_nt(
                            &inputs,
                            k_log,
                            n_log,
                            g,
                            useful_chunks,
                            Some(&mut stripe),
                            true,
                            per_block,
                            &mut z,
                            &mut a,
                            &mut b,
                        )
                    },
                    iters,
                ),
            );
        }
        rep.row(
            "compute-only",
            g,
            time_median(|| build_compute_only(&inputs, k_log, n_log, g, per_block), iters),
        );
    }

    // C2 direct-write producer: requires pre-zeroed dest (padding words are
    // never written) — zero once outside the timed region, matching the
    // recycled-buffer steady state.
    if let Some(direct) = direct {
        z.fill(0);
        a.fill(0);
        b.fill(0);
        stripe.fill(0);
        rep.row(
            "L1-direct-stripe",
            8,
            time_median(
                || direct(&inputs, n_log, Some(&mut stripe), &mut z, &mut a, &mut b),
                iters,
            ),
        );
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let args: Vec<String> = std::env::args().collect();
    let hashes: Vec<String> = parse_flag(&args, "--hash")
        .map_or_else(|| vec!["all".into()], |v| vec![v]);
    let hashes: Vec<&str> = if hashes[0] == "all" {
        vec!["keccak", "sha2", "blake3"]
    } else {
        hashes.iter().map(|s| s.as_str()).collect::<Vec<_>>()
    };
    let ms: Vec<usize> = parse_flag(&args, "--m").map_or_else(
        || vec![23, 26, 29],
        |v| v.split(',').map(|s| s.parse().unwrap()).collect(),
    );
    let iters: usize = parse_flag(&args, "--iters").map_or(7, |v| v.parse().unwrap());
    let groups: Option<Vec<usize>> = parse_flag(&args, "--groups").and_then(|v| {
        if v == "auto" {
            None
        } else {
            Some(v.split(',').map(|s| s.parse().unwrap()).collect())
        }
    });
    let tsv = parse_flag(&args, "--tsv");

    for hash in &hashes {
        for &m in &ms {
            match *hash {
                "keccak" => run_hash(
                    "keccak",
                    flock_prover::r1cs_hashes::keccak::K_LOG,
                    flock_prover::r1cs_hashes::keccak::USEFUL_BITS,
                    m,
                    iters,
                    &groups,
                    &tsv,
                    keccak_witness::random_state,
                    flock_prover::r1cs_hashes::keccak::generate_witness_with_ab_packed_and_lincheck,
                    &keccak_witness::build_block_witness,
                    Some(&flock_autoresearch_probes::keccak_vwide::build_l1_direct),
                ),
                "sha2" => run_hash(
                    "sha2",
                    flock_prover::r1cs_hashes::sha2::K_LOG,
                    flock_prover::r1cs_hashes::sha2::USEFUL_BITS,
                    m,
                    iters,
                    &groups,
                    &tsv,
                    sha2_witness::random_input,
                    flock_prover::r1cs_hashes::sha2::generate_witness_with_ab_packed_and_lincheck,
                    &sha2_witness::build_block_witness,
                    Some(&flock_autoresearch_probes::sha2_vwide::build_l1_direct),
                ),
                "blake3" => run_hash(
                    "blake3",
                    flock_prover::r1cs_hashes::blake3::K_LOG,
                    flock_prover::r1cs_hashes::blake3::USEFUL_BITS,
                    m,
                    iters,
                    &groups,
                    &tsv,
                    blake3_witness::random_input,
                    flock_prover::r1cs_hashes::blake3::generate_witness_with_ab_packed_and_lincheck,
                    &blake3_witness::build_block_witness,
                    Some(&flock_autoresearch_probes::blake3_vwide::build_l1_direct),
                ),
                other => panic!("unknown hash {other}"),
            }
        }
    }
}
