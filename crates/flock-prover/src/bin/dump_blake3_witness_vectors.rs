//! Dump a BLAKE3 R1CS **witness-generation** oracle from the real `flock`
//! generator so the CUDA port (`cuda-ghash/test_blake3_witness.cu`) can be
//! checked bit-for-bit.
//!
//! Witness gen (`blake3::generate_witness_with_ab_packed_and_lincheck`, the S4
//! GPU target) takes `n_blocks` BLAKE3 `Compression` inputs and produces:
//!   - `z`, `a`, `b` : F128-packed witness + (a = A·z, b = B·z) products,
//!     each `n_total · 128` F128 where `n_total = 2^n_blocks_log` (padding
//!     blocks are honest zeros),
//!   - `z_lincheck`  : the stripe-packed witness for lincheck,
//!     `(n_total / 8) · K` bytes (K = 2^14).
//!
//! We run the real generator on random Compressions and dump the inputs +
//! golden outputs. The CUDA test reads the same Compression inputs, runs its
//! per-block trace + stripe transpose, and asserts every output bit-for-bit.
//!
//! Output: little-endian binary to argv[1] (default blake3_witness_vectors.bin):
//!   magic        u32 = 0x42335754 ("B3WT")
//!   n_blocks_log u32
//!   n_blocks     u32
//!   k_log        u32 (= 14)
//!   per block (n_blocks):  cv[8] u32, m[16] u32, counter u64, block_len u32, flags u32
//!   z[n_total·128]          {lo,hi} u64 each
//!   a[n_total·128]          {lo,hi} u64 each
//!   b[n_total·128]          {lo,hi} u64 each
//!   z_lincheck[(n_total/8)·K]  bytes
//!
//! Run:
//!   cargo run --release --bin dump_blake3_witness_vectors -- cuda-ghash/blake3_witness_vectors.bin 24 5

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;
use flock_prover::r1cs_hashes::blake3::{
    Compression, K_LOG, generate_witness_with_ab_packed_and_lincheck, min_n_blocks_log,
};

use flock_core::test_rng::Rng;

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}
fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "blake3_witness_vectors.bin".to_string());
    let n_blocks: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let n_blocks_log: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| min_n_blocks_log(n_blocks));

    let n_total = 1usize << n_blocks_log;
    assert!(
        n_blocks <= n_total,
        "n_blocks {n_blocks} > 2^{n_blocks_log}"
    );
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "need n_total >= 8 and divisible by 8"
    );
    let k = 1usize << K_LOG;

    let mut rng = Rng::new(0xB1A3E3 ^ ((n_blocks as u64) << 8));
    let blocks: Vec<Compression> = (0..n_blocks)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            (cv, m, rng.next_u64(), rng.next_u32(), rng.next_u32())
        })
        .collect();

    let (z, a, b, z_lincheck) = generate_witness_with_ab_packed_and_lincheck(&blocks, n_blocks_log);
    assert_eq!(z.len(), n_total * (k / 128));
    assert_eq!(z_lincheck.len(), (n_total / 8) * k);

    let mut w = BufWriter::new(File::create(&path)?);
    write_u32(&mut w, 0x4233_5754)?; // "B3WT"
    write_u32(&mut w, n_blocks_log as u32)?;
    write_u32(&mut w, n_blocks as u32)?;
    write_u32(&mut w, K_LOG as u32)?;
    for (cv, m, counter, block_len, flags) in &blocks {
        for &x in cv {
            write_u32(&mut w, x)?;
        }
        for &x in m {
            write_u32(&mut w, x)?;
        }
        write_u64(&mut w, *counter)?;
        write_u32(&mut w, *block_len)?;
        write_u32(&mut w, *flags)?;
    }
    for &x in &z {
        write_f128(&mut w, x)?;
    }
    for &x in &a {
        write_f128(&mut w, x)?;
    }
    for &x in &b {
        write_f128(&mut w, x)?;
    }
    w.write_all(&z_lincheck)?;
    w.flush()?;

    eprintln!(
        "wrote blake3 witness oracle to {path}: n_blocks={n_blocks} n_blocks_log={n_blocks_log} \
         n_total={n_total} m={} k_log={K_LOG} z_f128={} z_lincheck_bytes={}",
        K_LOG + n_blocks_log,
        z.len(),
        z_lincheck.len()
    );
    Ok(())
}
