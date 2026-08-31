//! Dump an `induce_sumcheck_poly` oracle from the **real Ligerito**
//! (`src/pcs/ligerito.rs::induce_sumcheck_poly`) so the CUDA port
//! (`cuda-ghash/test_induce.cu`) is validated bit-for-bit against the code the
//! recursive prover runs.
//!
//! The induced
//! basis builder. Given the opened query rows, it produces a length-`2^log_n`
//! basis poly (novel-basis tensor per query, α-weighted and summed) plus the
//! `enforced_sum` scalar. We feed it real `sks_vks = eval_sk_at_vks(log_n)` and
//! deterministic random inputs, call the real function, and dump inputs+outputs.
//!
//! Output (LE) to argv[1] (default induce_vectors.bin), magic "INDC":
//!   magic u32, log_msg_cols u32, v_len u32, num_interleaved u32,
//!   n_queries u32, alpha_len u32, sks_len u32
//!   v_challenges[v_len], alpha[alpha_len], sks_vks[sks_len]   : {lo,hi} each
//!   queries[n_queries]                                        : u64 each
//!   opened_rows[n_queries * num_interleaved]                 : {lo,hi} each
//!   n u32 (=2^log_msg_cols)
//!   basis_poly[n]                                            : {lo,hi} each
//!   enforced_sum                                             : {lo,hi}
//!
//! Run:
//!   cargo run --release --bin dump_induce_vectors -- cuda-ghash/induce_vectors.bin 10 2 8

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;
use flock_prover::pcs::ligerito::{eval_sk_at_vks, induce_sumcheck_poly};

use flock_core::test_rng::Rng;

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "induce_vectors.bin".to_string());
    let log_msg_cols: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let v_len: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let n_queries: usize = env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(8);

    let num_interleaved = 1usize << v_len;
    // build_eq_table(alpha) must have >= n_queries entries.
    let alpha_len = (usize::BITS - (n_queries - 1).leading_zeros()) as usize; // ceil(log2(n_queries))
    let n = 1usize << log_msg_cols;

    let mut rng = Rng::new(0xC0FFEE);
    let v_challenges: Vec<F128> = (0..v_len).map(|_| rng.f128()).collect();
    let alpha: Vec<F128> = (0..alpha_len).map(|_| rng.f128()).collect();
    // sks_vks from the REAL helper (length log_n + 1).
    let sks_vks = eval_sk_at_vks(log_msg_cols);

    // Distinct query positions in [0, 2^log_msg_cols).
    let mut queries: Vec<usize> = Vec::with_capacity(n_queries);
    {
        let mut seen = std::collections::HashSet::new();
        while queries.len() < n_queries {
            let q = (rng.next_u64() as usize) % n.max(1);
            if seen.insert(q) {
                queries.push(q);
            }
        }
    }
    let opened_rows: Vec<Vec<F128>> = (0..n_queries)
        .map(|_| (0..num_interleaved).map(|_| rng.f128()).collect())
        .collect();

    // The real induced-basis builder.
    let (basis_poly, enforced_sum) = induce_sumcheck_poly(
        log_msg_cols,
        &sks_vks,
        &opened_rows,
        &v_challenges,
        &queries,
        &alpha,
    );
    assert_eq!(basis_poly.len(), n);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x494E_4443u32.to_le_bytes())?; // "INDC"
    for v in [
        log_msg_cols,
        v_len,
        num_interleaved,
        n_queries,
        alpha_len,
        sks_vks.len(),
    ] {
        w.write_all(&(v as u32).to_le_bytes())?;
    }
    for &x in &v_challenges {
        write_f128(&mut w, x)?;
    }
    for &x in &alpha {
        write_f128(&mut w, x)?;
    }
    for &x in &sks_vks {
        write_f128(&mut w, x)?;
    }
    for &q in &queries {
        w.write_all(&(q as u64).to_le_bytes())?;
    }
    for row in &opened_rows {
        for &x in row {
            write_f128(&mut w, x)?;
        }
    }
    w.write_all(&(n as u32).to_le_bytes())?;
    for &x in &basis_poly {
        write_f128(&mut w, x)?;
    }
    write_f128(&mut w, enforced_sum)?;
    w.flush()?;
    eprintln!(
        "wrote induce oracle to {path}: log_msg_cols={log_msg_cols} v_len={v_len} \
         num_interleaved={num_interleaved} n_queries={n_queries} alpha_len={alpha_len} n={n}"
    );
    Ok(())
}
