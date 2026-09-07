//! Dump a FULL zerocheck `prove_packed` transcript (round-1 + round-2 + tail +
//! final binding + c-interp) so the CUDA orchestrator
//! (`cuda-ghash/test_zerocheck_full.cu`) can replay it end-to-end bit-for-bit.
//!
//! Output (LE) to argv[1] (default zerocheck_full_vectors.bin):
//!   magic u32 = 0x5A434656 ("ZCFV"); m u32; domain_len u32, domain bytes
//!   M (64*64 F8 bytes, col-major); f8mul (256*256 bytes)
//!   a_packed; b_packed; c_packed   (2^m/8 bytes each)
//!   round1_ab[64] {lo,hi}; round1_c[64] {lo,hi}
//!   n_mlv u32; multilinear_rounds[n_mlv] {msg_1,msg_inf} each {lo,hi}
//!   final_a {lo,hi}; final_b {lo,hi}; final_c {lo,hi}
//!
//! Run: cargo run --release --bin dump_zerocheck_full_vectors -- cuda-ghash/zerocheck_full_vectors.bin 15

use std::{
    env,
    fs::File,
    io::{BufWriter, Result, Write},
};

use env::args;
use flock_core::test_rng::Rng;
use flock_prover::{
    challenger::FsChallenger,
    field::{F8, F128},
    ntt::AdditiveNttGf8,
    zerocheck::{prove_packed, univariate_skip::pack_bits},
};
const DOMAIN: &[u8] = b"flock-zerocheck-full-test";
const K_SKIP: usize = 6;

fn wf(w: &mut impl Write, x: F128) -> Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}
fn wu(w: &mut impl Write, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "zerocheck_full_vectors.bin".to_string());
    let m: usize = args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    assert!(m >= K_SKIP + 7);
    let n_total = 1usize << m;

    let mut rng = Rng::new(0x2CF0 ^ (m as u64));
    let a: Vec<bool> = (0..n_total).map(|_| rng.bit()).collect();
    let b: Vec<bool> = (0..n_total).map(|_| rng.bit()).collect();
    let c: Vec<bool> = (0..n_total).map(|_| rng.bit()).collect();
    let a_packed = pack_bits(&a);
    let b_packed = pack_bits(&b);
    let c_packed = pack_bits(&c);

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, _claim) = prove_packed(&a_packed, &b_packed, &c_packed, m, &mut ch);

    // Extension matrix M and F8 mul table (for the round-1 kernel).
    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
    let mut mcol = vec![0u8; 64 * 64];
    for s in 0..64 {
        let mut col = vec![F8::ZERO; 64];
        col[s] = F8(1);
        ntt_s.inverse(&mut col);
        ntt_l.forward(&mut col);
        for i in 0..64 {
            mcol[s * 64 + i] = col[i].0;
        }
    }
    let mut f8mul = vec![0u8; 256 * 256];
    for x in 0..256usize {
        for y in 0..256usize {
            f8mul[x * 256 + y] = (F8(x as u8) * F8(y as u8)).0;
        }
    }

    let mut w = BufWriter::new(File::create(&path)?);
    wu(&mut w, 0x5A43_4656)?; // "ZCFV"
    wu(&mut w, m as u32)?;
    wu(&mut w, DOMAIN.len() as u32)?;
    w.write_all(DOMAIN)?;
    w.write_all(&mcol)?;
    w.write_all(&f8mul)?;
    w.write_all(&a_packed)?;
    w.write_all(&b_packed)?;
    w.write_all(&c_packed)?;
    for &x in &proof.round1_ab {
        wf(&mut w, x)?;
    }
    for &x in &proof.round1_c {
        wf(&mut w, x)?;
    }
    wu(&mut w, proof.multilinear_rounds.len() as u32)?;
    for &(m1, mi) in &proof.multilinear_rounds {
        wf(&mut w, m1)?;
        wf(&mut w, mi)?;
    }
    wf(&mut w, proof.final_a_eval)?;
    wf(&mut w, proof.final_b_eval)?;
    wf(&mut w, proof.final_c_eval)?;
    w.flush()?;
    eprintln!(
        "wrote full zerocheck transcript to {path}: m={m} n_mlv={}",
        proof.multilinear_rounds.len()
    );
    Ok(())
}
