//! Dump the REAL BLAKE3 R1CS lincheck base matrices (A_0, B_0) as CSC arrays for
//! the CUDA `bench_ligerito` lincheck phase, replacing its synthetic fixed-8-nnz
//! stand-in. GF(2), implicit ones — only column pointers + row indices (no values),
//! exactly what `lincheck_csc_fold` consumes. Const-pin (col 512) needs no special
//! handling: `CscCircuit::fold_alpha_batched` folds it generically as a dense column.
//!
//! CSC layout mirrors `lincheck.rs::csc_from_rows`: for column `c`, its row indices
//! are `rows_flat[col_ptr[c] .. col_ptr[c+1]]` (the rows with a 1 in column c).
//!
//! File format (all little-endian):
//!   magic        u32 = 0x424C334D ("BL3M")
//!   n_cols       u32   (= K = 16384)
//!   useful_bits  u32   (= 15409)
//!   a_nnz        u32, a_col_ptr[n_cols+1] u32, a_rows[a_nnz] u32
//!   b_nnz        u32, b_col_ptr[n_cols+1] u32, b_rows[b_nnz] u32
//!
//! Run: cargo run --release --quiet --bin dump_blake3_lincheck_matrices -- <out.bin>

use std::{
    env,
    fs::File,
    io::{BufWriter, Result, Write},
};

use env::args;
use flock_prover::{
    r1cs::SparseBinaryMatrix,
    r1cs_hashes::blake3::{K, USEFUL_BITS, build_matrices},
};

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

fn write_u32(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32s(w: &mut impl Write, v: &[u32]) -> Result<()> {
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let out = args()
        .nth(1)
        .unwrap_or_else(|| "blake3_lincheck_matrices.bin".to_string());

    let (a_0, b_0) = build_matrices();
    assert_eq!(a_0.num_cols, K);
    assert_eq!(b_0.num_cols, K);
    let (a_col_ptr, a_rows) = csc_from_rows(&a_0);
    let (b_col_ptr, b_rows) = csc_from_rows(&b_0);

    eprintln!(
        "BLAKE3 lincheck matrices: n_cols={K} useful_bits={USEFUL_BITS} \
         nnz_a={} nnz_b={} (total {})",
        a_rows.len(),
        b_rows.len(),
        a_rows.len() + b_rows.len()
    );

    let f = File::create(&out)?;
    let mut w = BufWriter::new(f);
    write_u32(&mut w, 0x424C_334D)?; // "BL3M"
    write_u32(&mut w, K as u32)?;
    write_u32(&mut w, USEFUL_BITS as u32)?;
    write_u32(&mut w, a_rows.len() as u32)?;
    write_u32s(&mut w, &a_col_ptr)?;
    write_u32s(&mut w, &a_rows)?;
    write_u32(&mut w, b_rows.len() as u32)?;
    write_u32s(&mut w, &b_col_ptr)?;
    write_u32s(&mut w, &b_rows)?;
    w.flush()?;
    eprintln!("wrote {out}");
    Ok(())
}
