//! Dump an a·b sumcheck oracle sourced from the **real Ligerito**
//! `SumcheckProver` (`src/pcs/ligerito.rs`), not a hand-replicated formula, so
//! the CUDA sumcheck kernel (`cuda-ghash/test_sumcheck_ab.cu`) is validated
//! against the exact code the prover runs (`recursive_prover_with_basis` →
//! `SumcheckProver::fold` → `fold_and_msg_lsb`).
//!
//! This is the step-3 re-validation for the Ligerito re-aim
//! Ligerito's per-round sumcheck folds
//! `(f, combined_basis)` with `nf[j] = f[2j]·(1+r) + f[2j+1]·r` (≡ the CUDA
//! kernel's `f[2j] + r·(f[2j]+f[2j+1])`) and sends `{u_0, u_2}` — the same math
//! `dump_sumcheck_vectors` hand-replicates, but this oracle comes from the
//! Ligerito `SumcheckProver` itself.
//!
//! Emits the SAME `SMC1` format as `dump_sumcheck_vectors`, so `test_sumcheck_ab`
//! consumes it unchanged. Per round k it writes the message over the CURRENT
//! (f, basis) then the challenge r_k that folds to the next round — matching the
//! `SumcheckProver` message/fold ordering (msg over current array, then fold).
//!
//! Run:
//!   cargo run --release --bin dump_ligerito_sumcheck_vectors -- cuda-ghash/sumcheck_vectors.bin 12

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;
use flock_prover::pcs::ligerito::SumcheckProver;

use flock_core::test_rng::Rng;

fn write_f128(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "sumcheck_vectors.bin".to_string());
    let log_len: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let init_len = 1usize << log_len;

    let mut rng = Rng::new(0xC0FFEE);
    let f: Vec<F128> = (0..init_len).map(|_| rng.f128()).collect();
    let basis: Vec<F128> = (0..init_len).map(|_| rng.f128()).collect();

    // The real Ligerito sumcheck prover. h1 (initial claim) does not affect the
    // fold/message values; pass the honest sum for realism.
    let h1 = f
        .iter()
        .zip(basis.iter())
        .fold(F128::ZERO, |acc, (&x, &y)| acc + x * y);
    let (mut sc, mut msg) = SumcheckProver::new(f.clone(), basis.clone(), h1);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x534D_4331u32.to_le_bytes())?; // "SMC1" (same as dump_sumcheck_vectors)
    w.write_all(&(log_len as u32).to_le_bytes())?;
    w.write_all(&(init_len as u32).to_le_bytes())?;
    for &x in &f {
        write_f128(&mut w, x)?;
    }
    for &x in &basis {
        write_f128(&mut w, x)?;
    }

    // Per round: `msg` is over the CURRENT (f, basis); write it + the challenge,
    // then fold (which returns the next round's message). The SumcheckProver
    // folds `combined_basis` internally but doesn't expose it, so we fold a copy
    // `bcur` with the SAME r to recover final_b for the CUDA test's final check.
    let mut bcur = basis.clone();
    for _k in 0..log_len {
        let r = rng.f128();
        write_f128(&mut w, r)?;
        write_f128(&mut w, msg.u_0)?;
        write_f128(&mut w, msg.u_2)?;

        let half = bcur.len() / 2;
        let one_plus_r = F128::ONE + r;
        let mut nb = Vec::with_capacity(half);
        for j in 0..half {
            nb.push(bcur[2 * j] * one_plus_r + bcur[2 * j + 1] * r);
        }
        bcur = nb;

        msg = sc.fold(r); // same r as the bcur fold above
    }
    write_f128(&mut w, sc.f()[0])?; // final_f (length 1)
    write_f128(&mut w, bcur[0])?; // final_b
    w.flush()?;
    eprintln!(
        "wrote ligerito sumcheck oracle to {path}: log_len={log_len} init_len={init_len} \
         (sourced from real SumcheckProver)"
    );
    Ok(())
}
