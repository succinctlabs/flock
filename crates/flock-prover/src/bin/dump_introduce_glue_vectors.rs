//! Dump an `introduce_new` + `glue` oracle from the **real Ligerito**
//! `SumcheckProver` (`src/pcs/ligerito.rs`) so the CUDA port
//! (`cuda-ghash/test_introduce_glue.cu`) is validated against the code the
//! recursive prover runs.
//!
//! During a Ligerito open, when a level's
//! induced basis (step 4) enters the running sumcheck, the prover calls
//!   introduce_new_with_eval(b_new) -> (msg{u_0,u_2}, h_new = Σ_x f·b_new)
//! (via `round_msg_and_eval_lsb`), then `glue(β)`:
//!   combined_basis[j] += β·b_new[j];   t_r += β·h_new.
//! The message + h_new come from the real `SumcheckProver`; `glue` is a trivial
//! AXPY we replicate to recover the glued basis (the prover doesn't expose it).
//!
//! Output (LE) to argv[1] (default introduce_glue_vectors.bin), magic "INGL":
//!   magic u32, log_len u32, len u32 (=2^log_len)
//!   f[len], b1[len], b_new[len]   : {lo,hi} each
//!   beta, u_0, u_2, h_new         : {lo,hi} each
//!   glued_cb[len]                 : {lo,hi} each   (= b1 + β·b_new)
//!
//! Run:
//!   cargo run --release --bin dump_introduce_glue_vectors -- cuda-ghash/introduce_glue_vectors.bin 12

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
        .unwrap_or_else(|| "introduce_glue_vectors.bin".to_string());
    let log_len: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let len = 1usize << log_len;

    let mut rng = Rng::new(0xC0FFEE);
    let f: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
    let b1: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
    let b_new: Vec<F128> = (0..len).map(|_| rng.f128()).collect();

    let h1 = f
        .iter()
        .zip(b1.iter())
        .fold(F128::ZERO, |a, (&x, &y)| a + x * y);
    let (mut sc, _msg0) = SumcheckProver::new(f.clone(), b1.clone(), h1);

    // Real introduce: message {u_0,u_2} + h_new = Σ f·b_new.
    let (msg, h_new) = sc.introduce_new_with_eval(b_new.clone());

    let beta = rng.f128();
    // glue(β): combined_basis = b1 + β·b_new (no folds happened, so cb == b1).
    let glued_cb: Vec<F128> = b1
        .iter()
        .zip(b_new.iter())
        .map(|(&c, &v)| c + beta * v)
        .collect();

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x494E_474Cu32.to_le_bytes())?; // "INGL"
    w.write_all(&(log_len as u32).to_le_bytes())?;
    w.write_all(&(len as u32).to_le_bytes())?;
    for v in [&f, &b1, &b_new] {
        for &x in v {
            write_f128(&mut w, x)?;
        }
    }
    write_f128(&mut w, beta)?;
    write_f128(&mut w, msg.u_0)?;
    write_f128(&mut w, msg.u_2)?;
    write_f128(&mut w, h_new)?;
    for &x in &glued_cb {
        write_f128(&mut w, x)?;
    }
    w.flush()?;
    eprintln!("wrote introduce+glue oracle to {path}: log_len={log_len} len={len}");
    Ok(())
}
