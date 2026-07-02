//! E3 — lincheck partial-fold bench: production stripe path vs fused-from-L1′.
//!
//! Production pays: (a) the byte-stripe build during witness gen, then
//! (b) the fold over the stripe. The fused L1′ fold pays neither — it reads
//! the committed witness directly. Reported here:
//!   - fold-only comparison: stripe-fast / stripe-NEON-oblock / fused-L1′;
//!   - the totals that matter: witgen+stripe+stripe-fold vs
//!     witgen(no stripe)+fused-fold.
//!
//! Usage: cargo run --release --bin e3_lincheck_fold -- [--hash all]
//!   [--m 23,26,29] [--iters 5] [--tsv out.tsv]

use flock_autoresearch_probes::lincheck_fold::partial_fold_l1;
use flock_autoresearch_probes::{blake3_vwide, blake3_witness, keccak_vwide, keccak_witness,
    sha2_vwide, sha2_witness};
use flock_core::field::F128;
use flock_core::lincheck::{build_eq_table, partial_fold_packed_z_fast_padded};
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

struct Rng(u64);
impl Rng {
    fn f128(&mut self) -> F128 {
        let mut next = || {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        F128 { lo: next(), hi: next() }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_hash<S: Sync>(
    hash: &'static str,
    k_log: usize,
    useful_bits: usize,
    m: usize,
    iters: usize,
    tsv: &Option<String>,
    gen_input: impl Fn(u64) -> S,
    direct: &dyn Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64]),
) {
    let n_log = m - k_log;
    let n = 1usize << n_log;
    let total_u64 = (1usize << m) / 64;
    let u64_per_block = (1usize << k_log) / 64;
    let threads = rayon::current_num_threads();

    eprintln!("\n== {hash}: m={m} (n_log={n_log}), {threads} threads ==");

    let inputs: Vec<S> = (0..n as u64).map(gen_input).collect();
    let (mut z, mut a, mut b) =
        (vec![0u64; total_u64], vec![0u64; total_u64], vec![0u64; total_u64]);
    let mut stripe = vec![0u8; (n / 8) * u64_per_block * 64];
    direct(&inputs, n_log, Some(&mut stripe), &mut z, &mut a, &mut b);

    let mut rng = Rng(0xE3 ^ (m as u64) << 8);
    let point: Vec<F128> = (0..n_log).map(|_| rng.f128()).collect();
    let eq_outer = build_eq_table(&point);

    // Correctness gate before timing.
    {
        let fused = partial_fold_l1(&z, m, k_log, useful_bits, &eq_outer);
        let refr = partial_fold_packed_z_fast_padded(&stripe, m, k_log, useful_bits, &eq_outer);
        assert_eq!(fused, refr, "{hash}: fused != stripe fold");
        eprintln!("correctness gate: OK");
    }

    let mut rows: Vec<(&str, f64, f64)> = Vec::new();

    // Fold-only comparisons.
    let t = time_median(
        || {
            std::hint::black_box(partial_fold_packed_z_fast_padded(
                &stripe, m, k_log, useful_bits, &eq_outer,
            ));
        },
        iters,
    );
    rows.push(("fold-stripe-fast", t.0, t.1));

    #[cfg(target_arch = "aarch64")]
    {
        use flock_core::lincheck::partial_fold_packed_z_neon_oblock_padded;
        let t = time_median(
            || {
                std::hint::black_box(partial_fold_packed_z_neon_oblock_padded(
                    &stripe, m, k_log, useful_bits, &eq_outer,
                ));
            },
            iters,
        );
        rows.push(("fold-stripe-neon-oblock", t.0, t.1));
    }

    let t = time_median(
        || {
            std::hint::black_box(partial_fold_l1(&z, m, k_log, useful_bits, &eq_outer));
        },
        iters,
    );
    rows.push(("fold-fused-l1", t.0, t.1));

    // Totals: witness gen with/without the stripe (the stripe's production
    // cost lives in witness gen; dropping it is E3's payoff).
    z.fill(0);
    a.fill(0);
    b.fill(0);
    let t = time_median(
        || direct(&inputs, n_log, Some(&mut stripe), &mut z, &mut a, &mut b),
        iters,
    );
    rows.push(("witgen-with-stripe", t.0, t.1));
    let t = time_median(|| direct(&inputs, n_log, None, &mut z, &mut a, &mut b), iters);
    rows.push(("witgen-no-stripe", t.0, t.1));

    println!("{:<26} {:>10} {:>10}", "variant", "median ms", "min ms");
    for (name, med, min) in &rows {
        println!("{:<26} {:>10.2} {:>10.2}", name, med * 1e3, min * 1e3);
        if let Some(p) = tsv {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("open tsv");
            writeln!(
                f,
                "{hash}\t{m}\t{k_log}\t{n_log}\t{threads}\t{name}\t{:.4}\t{:.4}",
                med * 1e3,
                min * 1e3
            )
            .unwrap();
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
    let ms: Vec<usize> = parse_flag(&args, "--m").map_or_else(
        || vec![23, 26, 29],
        |v| v.split(',').map(|s| s.parse().unwrap()).collect(),
    );
    let iters: usize = parse_flag(&args, "--iters").map_or(5, |v| v.parse().unwrap());
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
                    &tsv,
                    keccak_witness::random_state,
                    &keccak_vwide::build_l1_direct,
                ),
                "sha2" => run_hash(
                    "sha2",
                    flock_prover::r1cs_hashes::sha2::K_LOG,
                    flock_prover::r1cs_hashes::sha2::USEFUL_BITS,
                    m,
                    iters,
                    &tsv,
                    sha2_witness::random_input,
                    &sha2_vwide::build_l1_direct,
                ),
                "blake3" => run_hash(
                    "blake3",
                    flock_prover::r1cs_hashes::blake3::K_LOG,
                    flock_prover::r1cs_hashes::blake3::USEFUL_BITS,
                    m,
                    iters,
                    &tsv,
                    blake3_witness::random_input,
                    &blake3_vwide::build_l1_direct,
                ),
                _ => unreachable!(),
            }
        }
    }
}
