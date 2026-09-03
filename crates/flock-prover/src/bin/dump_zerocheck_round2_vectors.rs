//! Dump a zerocheck **round-2** oracle (fold-at-z + first multilinear message)
//! from the real `uni_skip_fold_and_round_pair_optimized_packed_padded` so the
//! CUDA port (`cuda-ghash/test_zerocheck_round2.cu`) can be checked bit-for-bit.
//!
//! Round-2 folds the packed witness a/b at the URM challenge z (over the skip
//! domain) into a_mlv/b_mlv (F128, length 2^(m-6)), then computes the first
//! multilinear sumcheck message:
//!   a_mlv[row] = Σ_{j=0..8} foldtable[j*256 + a_packed[row*8 + j]]
//!   (msg_1, msg_inf) = eq-weighted deg-2 message over (a_mlv, b_mlv),
//!                      eq = build_eq(mlv_challenges[1..]), msg_1 = mlv[0]·g_one
//! foldtable = UniSkipFoldTable::new(6, z).data  (8×256 F128).
//!
//! Output (LE) to argv[1] (default zerocheck_round2_vectors.bin):
//!   magic u32 = 0x5A523202 ("ZR2"); m u32
//!   z {lo,hi}; mlv_challenges[m-6] {lo,hi}
//!   foldtable[8*256] {lo,hi}
//!   a_packed[2^(m-3)]; b_packed[2^(m-3)]   (bytes)
//!   a_mlv[2^(m-6)] {lo,hi}; b_mlv[2^(m-6)] {lo,hi}
//!   msg_1 {lo,hi}; msg_inf {lo,hi}
//!
//! Run: cargo run --release --bin dump_zerocheck_round2_vectors -- cuda-ghash/zerocheck_round2_vectors.bin 15

use std::{
    env,
    fs::File,
    io::{BufWriter, Result, Write},
};

use env::args;
use flock_core::test_rng::Rng;
use flock_prover::{
    field::F128,
    zerocheck::{
        PaddingSpec,
        multilinear::{UniSkipFoldTable, uni_skip_fold_and_round_pair_optimized_packed_padded},
        univariate_skip::pack_bits,
    },
};
const K_SKIP: usize = 6;

fn wf(w: &mut impl Write, x: F128) -> Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "zerocheck_round2_vectors.bin".to_string());
    let m: usize = args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    assert!(m > K_SKIP);
    let n_total = 1usize << m;
    let n_out = 1usize << (m - K_SKIP);

    let mut rng = Rng::new(0x2C7A12 ^ (m as u64));
    let a_bits: Vec<bool> = (0..n_total).map(|_| rng.bit()).collect();
    let b_bits: Vec<bool> = (0..n_total).map(|_| rng.bit()).collect();
    let a_packed = pack_bits(&a_bits);
    let b_packed = pack_bits(&b_bits);

    let z = rng.f128();
    // mlv_challenges[0] = ONE (Convention A), rest random.
    let mut mlv = vec![F128::ONE; m - K_SKIP];
    for v in mlv.iter_mut().skip(1) {
        *v = rng.f128();
    }

    let table = UniSkipFoldTable::new(K_SKIP, z);
    assert_eq!(table.n_chunks, 8);
    assert_eq!(table.data.len(), 8 * 256);

    let padding = PaddingSpec::dense(m);
    let (a_mlv, b_mlv, msg_1, msg_inf) = uni_skip_fold_and_round_pair_optimized_packed_padded(
        &a_packed, &b_packed, m, K_SKIP, &table, &mlv, &padding,
    );
    assert_eq!(a_mlv.len(), n_out);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x5A52_3202u32.to_le_bytes())?;
    w.write_all(&(m as u32).to_le_bytes())?;
    wf(&mut w, z)?;
    for &x in &mlv {
        wf(&mut w, x)?;
    }
    for &x in &table.data {
        wf(&mut w, x)?;
    }
    w.write_all(&a_packed)?;
    w.write_all(&b_packed)?;
    for &x in &a_mlv {
        wf(&mut w, x)?;
    }
    for &x in &b_mlv {
        wf(&mut w, x)?;
    }
    wf(&mut w, msg_1)?;
    wf(&mut w, msg_inf)?;
    w.flush()?;
    eprintln!("wrote zerocheck round-2 oracle to {path}: m={m} n_out={n_out}");
    Ok(())
}
