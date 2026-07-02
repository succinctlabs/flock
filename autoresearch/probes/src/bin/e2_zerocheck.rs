//! E2 — zerocheck parity control: prove_packed_padded on row-major vs
//! L1′-permuted a/b/c buffers (real witness data, all three hashes).
//!
//! The claim under test: the zerocheck kernels are address-bit generic, so
//! the L1′ layout costs nothing. Additionally, under L1′ the per-block
//! padding becomes a contiguous suffix expressible with the EXISTING
//! `PaddingSpec` as one giant block (`k_log = m`) with a useful prefix — no
//! kernel changes needed.
//!
//! Verification gates (per config, before timing):
//! - prove→verify roundtrip must pass;
//! - padded output must be byte-identical to the dense output on the same
//!   buffers (the padded prover's contract).
//!
//! Usage: like e1_witness_gen:
//!   cargo run --release --bin e2_zerocheck -- [--hash all] [--m 23,26,29]
//!     [--iters 5] [--tsv out.tsv]

use flock_autoresearch_probes::producer::{PerBlock, build_row_major};
use flock_autoresearch_probes::{blake3_vwide, blake3_witness, keccak_vwide, keccak_witness,
    sha2_vwide, sha2_witness};
use flock_core::challenger::FsChallenger;
use flock_core::zerocheck::{self, PaddingSpec};
use std::io::Write;
use std::time::Instant;

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn as_bytes(v: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) }
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

#[allow(clippy::too_many_arguments)]
fn run_hash<S: Sync>(
    hash: &'static str,
    k_log: usize,
    useful_bits: usize,
    m: usize,
    iters: usize,
    tsv: &Option<String>,
    gen_input: impl Fn(u64) -> S,
    per_block: &impl PerBlock<S>,
    direct: &dyn Fn(&[S], usize, Option<&mut [u8]>, &mut [u64], &mut [u64], &mut [u64]),
) {
    let n_log = m - k_log;
    let n = 1usize << n_log;
    let total_u64 = (1usize << m) / 64;
    let threads = rayon::current_num_threads();
    let useful_chunks = useful_bits.div_ceil(128);
    let chunks_per_block = (1usize << k_log) / 128;

    eprintln!(
        "\n== {hash}: m={m} (n_log={n_log}, useful {useful_chunks}/{chunks_per_block} chunks), {threads} threads =="
    );

    let inputs: Vec<S> = (0..n as u64).map(gen_input).collect();

    // Row-major buffers (production layout), c aliases z.
    let (mut z_r, mut a_r, mut b_r) =
        (vec![0u64; total_u64], vec![0u64; total_u64], vec![0u64; total_u64]);
    build_row_major(&inputs, k_log, n_log, 8, per_block, &mut z_r, &mut a_r, &mut b_r);

    // L1' buffers via the direct producer.
    let (mut z_l, mut a_l, mut b_l) =
        (vec![0u64; total_u64], vec![0u64; total_u64], vec![0u64; total_u64]);
    direct(&inputs, n_log, None, &mut z_l, &mut a_l, &mut b_l);

    // Padding specs: production per-block for row-major; one giant block with
    // a useful *prefix* for L1' (padding chunk-columns form the MSB suffix).
    let pad_row = PaddingSpec { k_log, useful_bits_per_block: useful_bits };
    let pad_l1 = PaddingSpec {
        k_log: m,
        useful_bits_per_block: useful_chunks << (7 + n_log),
    };
    let dense = PaddingSpec::dense(m);

    let configs: [(&str, &[u64], &[u64], &[u64], PaddingSpec); 4] = [
        ("rowmajor-dense", &a_r, &b_r, &z_r, dense),
        ("rowmajor-padded", &a_r, &b_r, &z_r, pad_row),
        ("l1-dense", &a_l, &b_l, &z_l, dense),
        ("l1-suffix-skip", &a_l, &b_l, &z_l, pad_l1),
    ];

    // ---- Verification gates ----
    let mut dense_proofs = std::collections::HashMap::new();
    for (name, a, b, c, pad) in &configs {
        let mut ch = FsChallenger::new(b"flock-e2-v0");
        let (proof, _claim) = zerocheck::prove_packed_padded(
            as_bytes(a), as_bytes(b), as_bytes(c), m, pad, &mut ch,
        );
        let mut chv = FsChallenger::new(b"flock-e2-v0");
        zerocheck::verify(m, &proof, &mut chv)
            .unwrap_or_else(|e| panic!("{hash}/{name}: verify rejected: {e:?}"));
        // Padded must be byte-identical to dense on the same buffers.
        let key = a.as_ptr() as usize;
        match dense_proofs.entry(key) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(proof);
            }
            std::collections::hash_map::Entry::Occupied(o) => {
                assert_eq!(o.get(), &proof, "{hash}/{name}: padded != dense proof");
            }
        }
    }
    eprintln!("verification gates (roundtrip + padded==dense): OK");

    // ---- Timing ----
    println!("{:<20} {:>10} {:>10}", "config", "median ms", "min ms");
    for (name, a, b, c, pad) in &configs {
        let (med, min) = time_median(
            || {
                let mut ch = FsChallenger::new(b"flock-e2-v0");
                let r = zerocheck::prove_packed_padded(
                    as_bytes(a), as_bytes(b), as_bytes(c), m, pad, &mut ch,
                );
                std::hint::black_box(&r);
            },
            iters,
        );
        println!("{:<20} {:>10.2} {:>10.2}", name, med * 1e3, min * 1e3);
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
    // Buffers reclaimed; keep clippy quiet about the mut producers.
    let _ = (&mut z_r, &mut z_l);
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let args: Vec<String> = std::env::args().collect();
    let hashes: Vec<&str> = match parse_flag(&args, "--hash").as_deref() {
        None | Some("all") => vec!["keccak", "sha2", "blake3"],
        Some(h) => vec![match h {
            "keccak" => "keccak",
            "sha2" => "sha2",
            "blake3" => "blake3",
            other => panic!("unknown hash {other}"),
        }],
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
                    &keccak_witness::build_block_witness,
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
                    &sha2_witness::build_block_witness,
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
                    &blake3_witness::build_block_witness,
                    &blake3_vwide::build_l1_direct,
                ),
                _ => unreachable!(),
            }
        }
    }
}
