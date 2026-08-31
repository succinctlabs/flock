//! Dump GF(2^256) arithmetic vectors from the real `flock_field::F256`
//! (the quadratic extension the F256 fold ladder runs in), so the CUDA/host
//! port (`cuda-ghash/f256.cuh` / `test_f256_host.cpp`) is validated
//! bit-for-bit against the field the prover uses.
//!
//! Output (LE) to argv[1] (default f256_vectors.bin), magic "F256":
//!   magic u32
//!   n_mul u32,  per case: a (c0.lo c0.hi c1.lo c1.hi), b (4×u64), a·b (4×u64)
//!   n_base u32, per case: a (4×u64), b (lo hi), a·b (4×u64)      [F256×F128]
//!   n_xinv u32, per case: z (lo hi), x⁻¹·z (lo hi)
//!   n_ub u32,   per case: b (4×u64), u·b (4×u64)                 [split_basis pair]
//!
//! Run:  cargo run --release --bin dump_f256_vectors -- cuda-ghash/f256_vectors.bin

use env::args;
use std::env;
use std::fs::File;
use std::io::Result;
use std::io::{BufWriter, Write};
use std::iter::once;

use flock_prover::field::{F128, F256, mul_by_x_inv};

use flock_core::test_rng::Rng;

fn w128(w: &mut impl Write, x: F128) -> Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn w256(w: &mut impl Write, x: F256) -> Result<()> {
    w128(w, x.c0)?;
    w128(w, x.c1)
}

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "f256_vectors.bin".to_string());
    let mut rng = Rng::new(0xF256_F256);
    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x3635_3246u32.to_le_bytes())?; // "F256"

    // Full products — random pairs plus the identities/edges.
    let mut muls: Vec<(F256, F256)> = vec![
        (F256::ONE, F256::U),
        (F256::U, F256::U),
        (F256::ZERO, rng.f256()),
        (F256::from(rng.f128()), F256::from(rng.f128())),
    ];
    for _ in 0..64 {
        muls.push((rng.f256(), rng.f256()));
    }
    w.write_all(&(muls.len() as u32).to_le_bytes())?;
    for &(a, b) in &muls {
        w256(&mut w, a)?;
        w256(&mut w, b)?;
        w256(&mut w, a * b)?;
    }

    // Base products (F256 × F128) — the post-code-switch fold's 2-mul step.
    let bases: Vec<(F256, F128)> = (0..32).map(|_| (rng.f256(), rng.f128())).collect();
    w.write_all(&(bases.len() as u32).to_le_bytes())?;
    for &(a, b) in &bases {
        w256(&mut w, a)?;
        w128(&mut w, b)?;
        w256(&mut w, a * b)?;
    }

    // x⁻¹ shift-and-fold.
    let xs: Vec<F128> = once(F128::ONE).chain((0..32).map(|_| rng.f128())).collect();
    w.write_all(&(xs.len() as u32).to_le_bytes())?;
    for &z in &xs {
        w128(&mut w, z)?;
        w128(&mut w, mul_by_x_inv(z))?;
    }

    // u·B — the split_basis odd slot (extension.rs::split_basis).
    let ubs: Vec<F256> = (0..32).map(|_| rng.f256()).collect();
    w.write_all(&(ubs.len() as u32).to_le_bytes())?;
    for &b in &ubs {
        w256(&mut w, b)?;
        w256(&mut w, F256::U * b)?;
    }

    w.flush()?;
    eprintln!(
        "wrote F256 oracle to {path}: {} muls, {} base muls, {} x-inv, {} u·B (from real flock_field::F256)",
        muls.len(),
        bases.len(),
        xs.len(),
        ubs.len()
    );
    Ok(())
}
