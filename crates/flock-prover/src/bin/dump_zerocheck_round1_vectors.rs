//! Dump a zerocheck round-one oracle from the Rust prover.
//!
//! The canonical CUDA round-one test uses this output.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::{F8, F128};
use flock_prover::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use flock_prover::zerocheck::PaddingSpec;
use flock_prover::zerocheck::univariate_skip::{pack_bits, round1_naive};
use flock_prover::zerocheck::univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded,
    small_challenges_ghash,
};

const K_SKIP: usize = 6;
const N_INNER: usize = 7;

use flock_core::test_rng::Rng;

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}
fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "zc_cpustyle_vectors.bin".to_string());
    let m: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let k_log: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(m);
    let useful_bits: usize = env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1usize << k_log);

    assert!(m >= K_SKIP + N_INNER);
    assert!(k_log <= m && useful_bits <= (1usize << k_log));
    let n_total = 1usize << m;
    let mut rng = Rng::new(0x2ECEC0 ^ ((m as u64) << 8));

    let mut a = vec![false; n_total];
    let mut b = vec![false; n_total];
    let mut c = vec![false; n_total];
    let block = 1usize << k_log;
    for i in 0..n_total {
        if (i & (block - 1)) < useful_bits {
            a[i] = rng.bit();
            b[i] = rng.bit();
            c[i] = rng.bit();
        }
    }
    let a_packed = pack_bits(&a);
    let b_packed = pack_bits(&b);
    let c_packed = pack_bits(&c);

    let mut r = vec![F128::ZERO; m];
    for value in r.iter_mut().take(K_SKIP) {
        *value = rng.f128();
    }
    for (i, value) in small_challenges_ghash().iter().enumerate() {
        r[K_SKIP + i] = *value;
    }
    for (i, value) in medium_challenges_ghash().iter().enumerate() {
        r[K_SKIP + 3 + i] = *value;
    }
    for value in r.iter_mut().take(m).skip(K_SKIP + N_INNER) {
        *value = rng.f128();
    }

    let padding = PaddingSpec::uniform(k_log, useful_bits, 1usize << (m - k_log));
    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
    let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let (ab_opt, c_opt) = round1_shift_reduce_extract_c_packed_padded(
        &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table, &padding,
    );
    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = c_opt.iter().map(|x| c_s * *x).collect();
    let (ab_naive, c_naive) = round1_naive(&a, &b, &c, m, K_SKIP, &r);
    assert_eq!(ab_naive, round1_ab);
    assert_eq!(c_naive, round1_c);

    let mut mcol = vec![0u8; 64 * 64];
    for s in 0..64 {
        let mut column = vec![F8::ZERO; 64];
        column[s] = F8(1);
        ntt_s.inverse(&mut column);
        ntt_l.forward(&mut column);
        for i in 0..64 {
            mcol[s * 64 + i] = column[i].0;
        }
    }
    let mut f8mul = vec![0u8; 256 * 256];
    for x in 0..256 {
        for y in 0..256 {
            f8mul[x * 256 + y] = (F8(x as u8) * F8(y as u8)).0;
        }
    }

    let mut w = BufWriter::new(File::create(&path)?);
    write_u32(&mut w, 0x5A43_5231)?;
    write_u32(&mut w, m as u32)?;
    write_u32(&mut w, K_SKIP as u32)?;
    write_u32(&mut w, k_log as u32)?;
    write_u32(&mut w, useful_bits as u32)?;
    for &x in &r {
        write_f128(&mut w, x)?;
    }
    w.write_all(&mcol)?;
    w.write_all(&f8mul)?;
    w.write_all(&a_packed)?;
    w.write_all(&b_packed)?;
    w.write_all(&c_packed)?;
    for &x in &round1_ab {
        write_f128(&mut w, x)?;
    }
    for &x in &round1_c {
        write_f128(&mut w, x)?;
    }
    w.flush()?;
    eprintln!("wrote canonical zerocheck round-one oracle to {path}: m={m}");
    Ok(())
}
