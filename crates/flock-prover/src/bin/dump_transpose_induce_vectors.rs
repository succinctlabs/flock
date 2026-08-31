//! Dump an induce-via-NTT oracle from the real `induce_sumcheck_poly_via_ntt`
//! (the upstream sparse-NTT fast path) so the GPU transpose-NTT induce
//! (`cuda-ghash/test_transpose_induce.cu`) is validated byte-for-bit.
//!
//! The induced basis = transpose(forward additive NTT)(scattered query weights),
//! truncated to the message coords. The GPU does scatter + full transpose-NTT;
//! both equal this oracle.
//!
//! Output (LE) "TRNI": magic, log_msg_cols, log_inv_rate, n_queries, alpha_len,
//!   queries[n_queries] (u64),  alpha[alpha_len] {lo,hi},
//!   n (=2^log_msg_cols),  basis[n] {lo,hi}
//!
//! Run: cargo run --release --bin dump_transpose_induce_vectors -- out.bin 16 1 218

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;
use flock_prover::pcs::ligerito::induce_sumcheck_poly_via_ntt;

use flock_core::test_rng::Rng;

fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (n - 1).ilog2() as usize + 1
    }
}
fn wf(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let path = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "transpose_induce_vectors.bin".to_string());
    let log_msg_cols = arg(2, 16);
    let log_inv_rate = arg(3, 1);
    let n_queries = arg(4, 218);

    let log_block = log_msg_cols + log_inv_rate;
    let block_len = 1usize << log_block;
    let n = 1usize << log_msg_cols;
    let alpha_len = ceil_log2(n_queries);

    let mut rng = Rng::new(0xC0FFEE);
    let alpha: Vec<F128> = (0..alpha_len).map(|_| rng.f128()).collect();
    // distinct query positions in the FULL codeword domain [0, block_len).
    let mut queries: Vec<usize> = Vec::with_capacity(n_queries);
    {
        let mut seen = std::collections::HashSet::new();
        while queries.len() < n_queries {
            let q = (rng.next_u64() as usize) % block_len;
            if seen.insert(q) {
                queries.push(q);
            }
        }
    }
    // enforced_sum needs opened_rows + v_challenges; basis does not depend on them.
    // Use empty v_challenges (num_interleaved=1) + 1-col rows so the call is valid.
    let v_challenges: Vec<F128> = Vec::new();
    let opened_rows: Vec<Vec<F128>> = (0..n_queries).map(|_| vec![rng.f128()]).collect();

    let (basis, _enforced) = induce_sumcheck_poly_via_ntt(
        log_msg_cols,
        log_inv_rate,
        &opened_rows,
        &v_challenges,
        &queries,
        &alpha,
    );
    assert_eq!(basis.len(), n);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x54524E49u32.to_le_bytes())?; // "TRNI"
    for v in [log_msg_cols, log_inv_rate, n_queries, alpha_len] {
        w.write_all(&(v as u32).to_le_bytes())?;
    }
    for &q in &queries {
        w.write_all(&(q as u64).to_le_bytes())?;
    }
    for &x in &alpha {
        wf(&mut w, x)?;
    }
    w.write_all(&(n as u32).to_le_bytes())?;
    for &x in &basis {
        wf(&mut w, x)?;
    }
    w.flush()?;
    eprintln!(
        "wrote transpose-induce oracle to {path}: log_msg_cols={log_msg_cols} \
               log_inv_rate={log_inv_rate} n_queries={n_queries} block_len={block_len} n={n}"
    );
    Ok(())
}
