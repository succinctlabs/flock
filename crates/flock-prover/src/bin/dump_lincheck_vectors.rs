//! Dump a **lincheck** prover oracle from the *real* `flock` field + protocol so
//! the CUDA port (`cuda-ghash/test_lincheck.cu`) can be checked bit-for-bit.
//!
//! Lincheck (`src/lincheck.rs`) reduces the zerocheck's two MLE claims
//! `â(x)=v_a`, `b̂(x)=v_b` to a single `z`-claim. Its prover hot path is:
//!   1. `α = challenger.sample_f128()`
//!   2. `eq_inner = build_quirky_eq_table(z_skip, x_inner_rest, k_skip)`  (len k)
//!   3. `comb_vec = circuit.fold_alpha_batched(α, eq_inner)`              (len k)
//!      — the α-batched CSC column marginal of the two base matrices.
//!   4. `z_vec = partial_fold_packed_z(z_packed, …, build_eq_table(x_outer))` (len k)
//!   5. `inner_rest_len = k_log − k_skip` rounds of TOP-BIT product-sumcheck on
//!      `(comb_vec, z_vec)`: per round msg `(e1, einf)`, observe, sample `r`,
//!      fold both at `r`.
//!   6. `z_partial = z_vec` (len 2^k_skip) observed; `r_inner_skip` sampled;
//!      `w = Σ lagrange_weights_naive(k_skip, r_inner_skip)·z_partial`.
//!
//! We run the REAL `lincheck::prove_padded_capture_z_vec` to get the golden
//! proof/claim (and the exact pre-sumcheck `z_vec`), then recompute the
//! deterministic intermediates (`eq_inner`, `comb_vec`) with the same public
//! helpers, and replay a second identical challenger to recover `α`. The CSC
//! arrays are flattened here (mirrors `lincheck.rs::csc_from_rows`) so the CUDA
//! `lincheck_csc_fold` kernel reproduces `comb_vec` exactly.
//!
//! Output: little-endian binary to argv[1] (default lincheck_vectors.bin):
//!   magic       u32 = 0x4C4E434B ("LNCK")
//!   m,k_log,k_skip,useful_bits : u32 each
//!   domain_len  u32, domain bytes
//!   z_packed    (2^m / 8) bytes
//!   a_nnz u32, a_col_ptr[k+1] u32, a_rows[a_nnz] u32
//!   b_nnz u32, b_col_ptr[k+1] u32, b_rows[b_nnz] u32
//!   z_skip {lo,hi}; x_inner_rest[k_log-k_skip]{lo,hi}; x_outer[m-k_log]{lo,hi}
//!   alpha {lo,hi}
//!   comb_vec[2^k_log] {lo,hi}        — golden CSC-fold output
//!   z_vec_pre[2^k_log] {lo,hi}       — golden partial-fold output
//!   for round in 0..(k_log-k_skip):  e1{lo,hi}, einf{lo,hi}, r{lo,hi}
//!   z_partial[2^k_skip] {lo,hi}
//!   r_inner_skip {lo,hi}
//!   w {lo,hi}
//!
//! Run:
//!   cargo run --release --bin dump_lincheck_vectors -- cuda-ghash/lincheck_vectors.bin 10 4 2 16

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::challenger::{Challenger, FsChallenger};
use flock_prover::field::F128;
use flock_prover::lincheck::{
    self, CscCircuit, LincheckCircuit, QuirkyPoint, SkipPoint, build_eq_table,
    build_quirky_eq_table,
};
use flock_prover::r1cs::SparseBinaryMatrix;

const DOMAIN: &[u8] = b"flock-lincheck-test";

use flock_core::test_rng::Rng;

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}
fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Build a random k×k sparse boolean matrix: each row gets `nnz_per_row`
/// distinct random columns (so the CSC walker has real work to fold).
fn random_matrix(k: usize, nnz_per_row: usize, rng: &mut Rng) -> SparseBinaryMatrix {
    let rows: Vec<Vec<usize>> = (0..k)
        .map(|_| {
            let mut cols = Vec::with_capacity(nnz_per_row);
            for _ in 0..nnz_per_row {
                let c = (rng.next_u64() as usize) % k;
                if !cols.contains(&c) {
                    cols.push(c);
                }
            }
            cols.sort_unstable();
            cols
        })
        .collect();
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows,
    }
}

/// Flatten a sparse matrix to CSC arrays — verbatim mirror of
/// `lincheck.rs::csc_from_rows`, so the CUDA kernel folds the same columns.
fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "lincheck_vectors.bin".to_string());
    let m: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let k_log: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let k_skip: usize = env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    let k = 1usize << k_log;
    let useful_bits: usize = env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(k);

    assert!(m >= k_log, "need m >= k_log");
    assert!(k_skip <= k_log, "need k_skip <= k_log");
    let n_log = m - k_log;
    assert!(
        n_log >= 3,
        "need n_log >= 3 (n_outer >= 8 for byte stripes)"
    );
    assert!(useful_bits <= k, "useful_bits <= k");
    let inner_rest_len = k_log - k_skip;
    let n_total = 1usize << m;

    let mut rng = Rng::new(0x11C4EC0);

    // --- Witness: random bool vector, but zero out padding rows so the proof is
    //     padding-honest (rows [useful_bits, k) of every block are 0).
    let mut z_logical: Vec<bool> = (0..n_total).map(|_| (rng.next_u64() & 1) == 1).collect();
    if useful_bits < k {
        for i_outer in 0..(n_total / k) {
            for i_inner in useful_bits..k {
                z_logical[i_inner + i_outer * k] = false;
            }
        }
    }
    let z_packed = lincheck::pack_z_lincheck(&z_logical, m, k_log);

    // --- Base matrices + CSC flatten.
    let a_0 = random_matrix(k, 3, &mut rng);
    let b_0 = random_matrix(k, 4, &mut rng);
    let (a_col_ptr, a_rows) = csc_from_rows(&a_0);
    let (b_col_ptr, b_rows) = csc_from_rows(&b_0);
    let circuit = CscCircuit::from_matrices(&a_0, &b_0);

    // --- Quirky claim point.
    let x_ab = QuirkyPoint {
        z_skip: SkipPoint::Phi8(rng.f128()),
        x_inner_rest: (0..inner_rest_len).map(|_| rng.f128()).collect(),
        x_outer: (0..n_log).map(|_| rng.f128()).collect(),
    };

    // --- Run the REAL prover (captures the exact pre-sumcheck z_vec).
    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed,
        m,
        k_log,
        k_skip,
        useful_bits,
        &circuit,
        &x_ab,
        &mut ch,
    );
    assert_eq!(z_vec_pre.len(), k);
    assert_eq!(proof.rounds.len(), inner_rest_len);
    assert_eq!(proof.z_partial.len(), 1usize << k_skip);

    // --- Recover the deterministic intermediates with the same public helpers.
    //     α is the challenger's first sample after the domain-label observe; a
    //     fresh identical challenger replays exactly that prefix.
    let mut ch2 = FsChallenger::new(DOMAIN);
    ch2.observe_label(b"flock-lincheck-v0");
    let alpha = ch2.sample_f128();
    let eq_inner = build_quirky_eq_table(x_ab.z_skip.phi8(), &x_ab.x_inner_rest, k_skip);
    let comb_vec = circuit.fold_alpha_batched(alpha, &eq_inner);
    assert_eq!(comb_vec.len(), k);
    // sanity: the public build_eq_table(x_outer) the prover folds against.
    debug_assert_eq!(build_eq_table(&x_ab.x_outer).len(), 1usize << n_log);

    // r_rounds: prove() reverses r_rounds into claim.r_inner_rest, so undo it.
    let r_rounds: Vec<F128> = claim.r_inner_rest.iter().rev().copied().collect();

    // --- Write.
    let mut w = BufWriter::new(File::create(&path)?);
    write_u32(&mut w, 0x4C4E_434B)?; // "LNCK"
    write_u32(&mut w, m as u32)?;
    write_u32(&mut w, k_log as u32)?;
    write_u32(&mut w, k_skip as u32)?;
    write_u32(&mut w, useful_bits as u32)?;
    write_u32(&mut w, DOMAIN.len() as u32)?;
    w.write_all(DOMAIN)?;
    w.write_all(&z_packed)?;

    write_u32(&mut w, a_rows.len() as u32)?;
    for &v in &a_col_ptr {
        write_u32(&mut w, v)?;
    }
    for &v in &a_rows {
        write_u32(&mut w, v)?;
    }
    write_u32(&mut w, b_rows.len() as u32)?;
    for &v in &b_col_ptr {
        write_u32(&mut w, v)?;
    }
    for &v in &b_rows {
        write_u32(&mut w, v)?;
    }

    write_f128(&mut w, x_ab.z_skip.phi8())?;
    for &x in &x_ab.x_inner_rest {
        write_f128(&mut w, x)?;
    }
    for &x in &x_ab.x_outer {
        write_f128(&mut w, x)?;
    }

    write_f128(&mut w, alpha)?;
    for &x in &comb_vec {
        write_f128(&mut w, x)?;
    }
    for &x in &z_vec_pre {
        write_f128(&mut w, x)?;
    }
    for (round, ((e1, einf), r)) in proof.rounds.iter().zip(r_rounds.iter()).enumerate() {
        write_f128(&mut w, *e1)?;
        write_f128(&mut w, *einf)?;
        write_f128(&mut w, *r)?;
        let _ = round;
    }
    for &x in &proof.z_partial {
        write_f128(&mut w, x)?;
    }
    write_f128(&mut w, claim.r_inner_skip.phi8())?;
    write_f128(&mut w, claim.w)?;
    w.flush()?;

    eprintln!(
        "wrote lincheck oracle to {path}: m={m} k_log={k_log} k_skip={k_skip} \
         useful_bits={useful_bits} n_log={n_log} rounds={inner_rest_len} \
         nnz_a={} nnz_b={}",
        a_rows.len(),
        b_rows.len()
    );
    Ok(())
}
