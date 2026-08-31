//! Dump a zerocheck **multilinear sumcheck tail** oracle (rounds 2..n of
//! `zerocheck::prove_packed`) so the CUDA port (`cuda-ghash/test_zerocheck_tail.cu`)
//! can be checked bit-for-bit.
//!
//! The tail operates on `a_mlv`, `b_mlv` (F128, length 2^L). Each round, over the
//! current arrays with ADJACENT pairing (a[2x], a[2x+1]):
//!   eq      = build_eq(r[1..])                       (length = half)
//!   g_one   = Σ_x eq[x]·a[2x+1]·b[2x+1]
//!   g_inf   = Σ_x eq[x]·(a[2x]+a[2x+1])·(b[2x]+b[2x+1])
//!   message = (r[0]·g_one, g_inf)                    (r[0]=ONE in zerocheck)
//! then fold by ρ:  a[x] = a[2x] + ρ·(a[2x+1]+a[2x])  (and b). Computed via the
//! real `round_pair_naive` + `fold_in_place_pair`.
//!
//! Output (LE) to argv[1] (default zerocheck_tail_vectors.bin):
//!   magic u32 = 0x5A54414C ("ZTAL"); L u32
//!   a_mlv[2^L] {lo,hi}; b_mlv[2^L] {lo,hi}
//!   for round in 0..L (current log = L-round):
//!     r_rest[log-1] {lo,hi}   (the r[1..]; r[0]=ONE implicit)
//!     msg_1 {lo,hi}; msg_inf {lo,hi}; rho {lo,hi}
//!   final_a {lo,hi}; final_b {lo,hi}
//!
//! Run: cargo run --release --bin dump_zerocheck_tail_vectors -- cuda-ghash/zerocheck_tail_vectors.bin 16

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;
use flock_prover::zerocheck::multilinear::{fold_in_place_pair, round_pair_naive};

use flock_core::test_rng::Rng;

fn wf(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "zerocheck_tail_vectors.bin".to_string());
    let l: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let n = 1usize << l;

    let mut rng = Rng::new(0x2C7A11 ^ (l as u64));
    let mut a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let mut b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x5A54_414Cu32.to_le_bytes())?;
    w.write_all(&(l as u32).to_le_bytes())?;
    for &x in &a {
        wf(&mut w, x)?;
    }
    for &x in &b {
        wf(&mut w, x)?;
    }

    for _round in 0..l {
        let log_cur = a.len().trailing_zeros() as usize;
        // r-vector: r[0] = ONE (zerocheck Convention A), r[1..] random.
        let mut r = vec![F128::ONE; log_cur];
        for v in r.iter_mut().skip(1) {
            *v = rng.f128();
        }
        let (msg_1, msg_inf) = round_pair_naive(&a, &b, &r);
        let rho = rng.f128();
        for &x in &r[1..] {
            wf(&mut w, x)?;
        }
        wf(&mut w, msg_1)?;
        wf(&mut w, msg_inf)?;
        wf(&mut w, rho)?;
        fold_in_place_pair(&mut a, &mut b, rho);
    }
    assert_eq!(a.len(), 1);
    wf(&mut w, a[0])?;
    wf(&mut w, b[0])?;
    w.flush()?;
    eprintln!("wrote zerocheck tail oracle to {path}: L={l} n={n} rounds={l}");
    Ok(())
}
