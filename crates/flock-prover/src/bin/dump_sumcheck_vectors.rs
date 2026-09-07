//! Dump an a·b multilinear sumcheck oracle (the PCS-open sumcheck:
//! per-round message `(u_0, u_2)` + folded a,b) from the *real* `flock` field,
//! so the CUDA port (`cuda-ghash/test_sumcheck_ab.cu`) can be checked
//! bit-for-bit against it.
//!
//! GPU `pcs::open` (Ligerito) sumcheck vectors:
//! the degree-2 sumcheck of `S = Σ_x a(x)·b(x)` the Ligerito prover runs (the
//! `fold_and_msg_lsb` message/fold convention in `src/pcs/ligerito.rs`). Each
//! round, over the CURRENT a,b (adjacent pairing `(a[2j],a[2j+1])`, matching
//! the CPU prover), the message is the {0, ∞} pair:
//!   u_0 = Σ_j a[2j]·b[2j]                        (= u(0))
//!   u_2 = Σ_j (a[2j]+a[2j+1])·(b[2j]+b[2j+1])    (= u(∞), leading coeff)
//! then fold by the round challenge r: a'[j] = a[2j] + r·(a[2j]+a[2j+1]) (and b).
//! The verifier recovers the middle coeff from the running claim,
//! so the prover only sends (u_0, u_2) — see [`RoundMessage`].
//!
//! Output: little-endian binary to argv[1] (default sumcheck_vectors.bin):
//!   magic     u32 = 0x534D4331 ("SMC1")
//!   log_len   u32   (L; a,b have length 2^L)
//!   init_len  u32   (= 2^L)
//!   init_len * { lo, hi } : u64 each   — a_init
//!   init_len * { lo, hi } : u64 each   — b_init
//!   for round k in 0..L:
//!     r_k  { lo, hi }   — fold challenge
//!     u_0  { lo, hi }   — round message u(0)
//!     u_2  { lo, hi }   — round message u(∞)
//!   final_a  { lo, hi }   — a folded to length 1
//!   final_b  { lo, hi }   — b folded to length 1
//!
//! Run:
//!   cargo run --release --bin dump_sumcheck_vectors -- cuda-ghash/sumcheck_vectors.bin 12
//!   cargo run --release --bin dump_sumcheck_vectors -- out.bin 22

use std::{
    env,
    fs::File,
    io::{BufWriter, Result, Write},
};

use env::args;
use flock_core::test_rng::Rng;
use flock_prover::field::F128;

fn write_f128(w: &mut impl Write, x: F128) -> Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "sumcheck_vectors.bin".to_string());
    let log_len: usize = args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let init_len = 1usize << log_len;

    let mut rng = Rng::new(0xC0FFEE);
    let mut a: Vec<F128> = (0..init_len).map(|_| rng.f128()).collect();
    let mut b: Vec<F128> = (0..init_len).map(|_| rng.f128()).collect();

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x534D_4331u32.to_le_bytes())?; // "SMC1"
    w.write_all(&(log_len as u32).to_le_bytes())?;
    w.write_all(&(init_len as u32).to_le_bytes())?;
    for &x in &a {
        write_f128(&mut w, x)?;
    }
    for &x in &b {
        write_f128(&mut w, x)?;
    }

    // L rounds: message over current a,b (adjacent pairing), then fold by r.
    for _k in 0..log_len {
        let half = a.len() / 2;
        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        for j in 0..half {
            let a0 = a[2 * j];
            let a1 = a[2 * j + 1];
            let b0 = b[2 * j];
            let b1 = b[2 * j + 1];
            u_0 += a0 * b0;
            u_2 += (a0 + a1) * (b0 + b1);
        }
        // Challenge generated deterministically (the CUDA test replays it).
        let r = rng.f128();
        write_f128(&mut w, r)?;
        write_f128(&mut w, u_0)?;
        write_f128(&mut w, u_2)?;

        let mut a_next = Vec::with_capacity(half);
        let mut b_next = Vec::with_capacity(half);
        for j in 0..half {
            a_next.push(a[2 * j] + r * (a[2 * j] + a[2 * j + 1]));
            b_next.push(b[2 * j] + r * (b[2 * j] + b[2 * j + 1]));
        }
        a = a_next;
        b = b_next;
    }
    assert_eq!(a.len(), 1);
    write_f128(&mut w, a[0])?;
    write_f128(&mut w, b[0])?;
    w.flush()?;
    eprintln!(
        "wrote sumcheck oracle to {path}: log_len={log_len} init_len={init_len} rounds={log_len}"
    );
    Ok(())
}
