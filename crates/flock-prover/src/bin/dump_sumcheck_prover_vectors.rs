//! Dump a full `SumcheckProver` run (the heart of Ligerito's recursive open)
//! from the **real** `ligerito::SumcheckProver`, so the composed CUDA driver
//! (`cuda-ghash/test_sumcheck_prover.cu`) is validated end-to-end on one
//! resident sumcheck: `new` → (`fold` | `introduce_new`+`glue`)* → final.
//!
//! This
//! composes the step-3 (fold+message) and step-5 (glue) kernels into the real
//! state machine, driven by a scripted op sequence that exercises folds and
//! α-batched basis introductions at multiple dims (mirroring the recursive
//! ladder's intro/glue events). Challenges/β and the introduced bases come from
//! the script; the device must reproduce the full message transcript + final f.
//! (`introduce_new`'s h_new doesn't affect messages — only the verifier's
//! running claim — so an arbitrary value is fine here.)
//!
//! Output (LE) to argv[1] (default sumcheck_prover_vectors.bin), magic "SCPV":
//!   magic u32, log_len u32, len u32 (=2^log_len)
//!   f[len], b1[len]                : {lo,hi} each
//!   msg0 (u_0,u_2)                 : {lo,hi}
//!   n_ops u32
//!   per op: op_type u32  (0=fold, 1=intro+glue)
//!     fold:  r, msg(u_0,u_2)                              : {lo,hi}
//!     intro: cur_len u32, b_new[cur_len], beta, msg       : {lo,hi}
//!   final_f                        : {lo,hi}
//!
//! Run:
//!   cargo run --release --bin dump_sumcheck_prover_vectors -- cuda-ghash/sumcheck_prover_vectors.bin 14

use env::args;
use std::env;
use std::fs::File;
use std::io::Result;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;
use flock_prover::pcs::ligerito::SumcheckProver;

use flock_core::test_rng::Rng;

fn wf(w: &mut impl Write, x: F128) -> Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

enum Op {
    Fold,
    IntroGlue,
}

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "sumcheck_prover_vectors.bin".to_string());
    let log_len: usize = args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(14);
    let len = 1usize << log_len;

    let mut rng = Rng::new(0xC0FFEE);
    let f: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
    let b1: Vec<F128> = (0..len).map(|_| rng.f128()).collect();

    // Build the op script: intro+glue events at a couple of dims interleaved
    // with folds, ending folded to length 1. Track current dim so introduced
    // bases match the current f length.
    let mut ops: Vec<Op> = Vec::new();
    {
        let mut cur = log_len; // current log-length
        let mut fold_idx = 0usize;
        while cur > 0 {
            // Introduce after the 1st and 3rd folds (if room) — exercises intro
            // at dims log_len-1 and log_len-3.
            if (fold_idx == 1 || fold_idx == 3) && cur >= 1 {
                ops.push(Op::IntroGlue);
            }
            ops.push(Op::Fold);
            cur -= 1;
            fold_idx += 1;
        }
    }

    let h1 = f
        .iter()
        .zip(b1.iter())
        .fold(F128::ZERO, |a, (&x, &y)| a + x * y);
    let (mut sc, msg0) = SumcheckProver::new(f.clone(), b1.clone(), h1);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x5343_5056u32.to_le_bytes())?; // "SCPV"
    w.write_all(&(log_len as u32).to_le_bytes())?;
    w.write_all(&(len as u32).to_le_bytes())?;
    for &x in &f {
        wf(&mut w, x)?;
    }
    for &x in &b1 {
        wf(&mut w, x)?;
    }
    wf(&mut w, msg0.u_0)?;
    wf(&mut w, msg0.u_2)?;
    w.write_all(&(ops.len() as u32).to_le_bytes())?;

    let mut cur_len = len;
    for op in &ops {
        match op {
            Op::Fold => {
                let r = rng.f128();
                let msg = sc.fold(r);
                w.write_all(&0u32.to_le_bytes())?;
                wf(&mut w, r)?;
                wf(&mut w, msg.u_0)?;
                wf(&mut w, msg.u_2)?;
                cur_len /= 2;
            }
            Op::IntroGlue => {
                let b_new: Vec<F128> = (0..cur_len).map(|_| rng.f128()).collect();
                let h_new = rng.f128(); // arbitrary: does not affect messages
                let msg = sc.introduce_new(b_new.clone(), h_new);
                let beta = rng.f128();
                sc.glue(beta);
                w.write_all(&1u32.to_le_bytes())?;
                w.write_all(&(cur_len as u32).to_le_bytes())?;
                for &x in &b_new {
                    wf(&mut w, x)?;
                }
                wf(&mut w, beta)?;
                wf(&mut w, msg.u_0)?;
                wf(&mut w, msg.u_2)?;
                // intro+glue does not change the length.
            }
        }
    }
    wf(&mut w, sc.f()[0])?; // final_f
    w.flush()?;
    eprintln!(
        "wrote sumcheck-prover oracle to {path}: log_len={log_len} len={len} ops={} \
         (from real SumcheckProver)",
        ops.len()
    );
    Ok(())
}
