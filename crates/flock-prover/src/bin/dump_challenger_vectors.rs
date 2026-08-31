//! Dump a Fiat-Shamir transcript script + results from the **real** flock
//! `FsChallenger` (`src/challenger.rs`), so the host C++ port
//! (`cuda-ghash/challenger.hpp` / `test_challenger.cpp`) is validated
//! byte-for-byte against the code the prover's transcript uses.
//!
//! The host-side Ligerito challenger uses the generated vectors to check
//! challenger is what derives every challenge/β/query/grinding nonce in the
//! recursive ladder. We run a representative op sequence (observe / sample /
//! slice / vec / grind / label) on the real challenger and dump the ops + the
//! sampled outputs; the C++ replays and must reproduce them.
//!
//! Output (LE) to argv[1] (default challenger_vectors.bin), magic "CHLG":
//!   magic u32, domain_len u32, domain bytes, n_ops u32
//!   per op: op_type u8 then
//!     1 observe_f128 : F128{lo,hi}
//!     2 observe_bytes: u32 len, bytes
//!     3 sample_f128  : F128 result
//!     4 observe_label: u32 len, bytes
//!     5 grind        : u32 bits, u64 nonce
//!     6 observe_slice: u32 n, n*F128
//!     7 sample_vec   : u32 n, n*F128 results
//!
//! Run:  cargo run --release --bin dump_challenger_vectors -- cuda-ghash/challenger_vectors.bin

use env::args;
use flock_prover::field::F256;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::Result;
use std::io::{BufWriter, Write};

use flock_prover::challenger::{Challenger, FsChallenger};
use flock_prover::field::F128;

use flock_core::test_rng::Rng;
use flock_prover::pcs::stratified::LevelSchedule;

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "challenger_vectors.bin".to_string());
    let domain = b"flock-ligerito-test-v0";

    let mut ch = FsChallenger::new(domain);
    let mut rng = Rng::new(0xC0FFEE);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x4348_4C47u32.to_le_bytes())?; // "CHLG"
    w.write_all(&(domain.len() as u32).to_le_bytes())?;
    w.write_all(domain)?;

    // Buffer the ops so we can write n_ops first.
    let mut body: Vec<u8> = Vec::new();
    let mut n_ops = 0u32;
    macro_rules! emit { ($($b:expr),*) => {{ $( body.extend_from_slice($b); )* n_ops += 1; }} }

    // 1. observe a few scalars
    for _ in 0..3 {
        let v = rng.f128();
        ch.observe_f128(v);
        let mut t = vec![1u8];
        t.extend_from_slice(&v.lo.to_le_bytes());
        t.extend_from_slice(&v.hi.to_le_bytes());
        emit!(&t);
    }
    // 2. observe_bytes (mimics a Merkle root)
    {
        let mut root = [0u8; 32];
        for (i, b) in root.iter_mut().enumerate() {
            *b = (rng.next_u64() as u8).wrapping_add(i as u8);
        }
        ch.observe_bytes(&root);
        let mut t = vec![2u8];
        t.extend_from_slice(&(root.len() as u32).to_le_bytes());
        t.extend_from_slice(&root);
        emit!(&t);
    }
    // 3. sample a scalar challenge
    {
        let r = ch.sample_f128();
        let mut t = vec![3u8];
        t.extend_from_slice(&r.lo.to_le_bytes());
        t.extend_from_slice(&r.hi.to_le_bytes());
        emit!(&t);
    }
    // 4. observe a slice (e.g. OOD values)
    {
        let vals: Vec<F128> = (0..4).map(|_| rng.f128()).collect();
        ch.observe_f128_slice(&vals);
        let mut t = vec![6u8];
        t.extend_from_slice(&(vals.len() as u32).to_le_bytes());
        for v in &vals {
            t.extend_from_slice(&v.lo.to_le_bytes());
            t.extend_from_slice(&v.hi.to_le_bytes());
        }
        emit!(&t);
    }
    // 5. sample a vector of challenges (e.g. lane-fold rs)
    {
        let n = 5usize;
        let rs = ch.sample_f128_vec(n);
        let mut t = vec![7u8];
        t.extend_from_slice(&(n as u32).to_le_bytes());
        for v in &rs {
            t.extend_from_slice(&v.lo.to_le_bytes());
            t.extend_from_slice(&v.hi.to_le_bytes());
        }
        emit!(&t);
    }
    // 6. grind (small bits so it's fast; the smallest-nonce result is deterministic)
    for &bits in &[8u32, 12u32] {
        let nonce = ch.grind_pow(bits);
        let mut t = vec![5u8];
        t.extend_from_slice(&bits.to_le_bytes());
        t.extend_from_slice(&nonce.to_le_bytes());
        emit!(&t);
    }
    // 6b. sample_distinct_queries (replicates ligerito::sample_distinct_queries;
    // the only nontrivial op is sample_f128, which is validated byte-exact).
    {
        let block_len = 1000usize;
        let count = 20usize;
        let mut seen = HashSet::new();
        let mut qs: Vec<usize> = Vec::new();
        while qs.len() < count {
            let v = ch.sample_f128();
            let q = (v.lo as usize) % block_len;
            if seen.insert(q) {
                qs.push(q);
            }
        }
        qs.sort_unstable();
        let mut t = vec![8u8];
        t.extend_from_slice(&(block_len as u64).to_le_bytes());
        t.extend_from_slice(&(count as u32).to_le_bytes());
        for &q in &qs {
            t.extend_from_slice(&(q as u64).to_le_bytes());
        }
        emit!(&t);
    }

    // 6c. sample_f256 (one two-word vec squeeze — the F256 ladder's fold challenges)
    for _ in 0..2 {
        let r = ch.sample_f256();
        let c = r.coordinates();
        let mut t = vec![9u8];
        for w in c {
            t.extend_from_slice(&w.lo.to_le_bytes());
            t.extend_from_slice(&w.hi.to_le_bytes());
        }
        emit!(&t);
    }
    // 6d. observe_f256 (two-coordinate slice observe — ladder message absorbs)
    {
        let v = F256::new(rng.f128(), rng.f128());
        ch.observe_f256(v);
        let c = v.coordinates();
        let mut t = vec![10u8];
        for w in c {
            t.extend_from_slice(&w.lo.to_le_bytes());
            t.extend_from_slice(&w.hi.to_le_bytes());
        }
        emit!(&t);
    }
    // 6e. fused grind + scalar squeeze (claim-batch β sites; includes bits=0,
    // which still absorbs the canonical 0 nonce)
    for &bits in &[0u32, 6u32] {
        let (nonce, r) = ch.grind_pow_and_sample_f128(bits);
        let mut t = vec![11u8];
        t.extend_from_slice(&bits.to_le_bytes());
        t.extend_from_slice(&nonce.to_le_bytes());
        t.extend_from_slice(&r.lo.to_le_bytes());
        t.extend_from_slice(&r.hi.to_le_bytes());
        emit!(&t);
    }
    // 6f. fused grind + vector squeeze (consistency-α / γ sites)
    {
        let (bits, n) = (3u32, 7usize);
        let (nonce, rs) = ch.grind_pow_and_sample_f128_vec(bits, n);
        let mut t = vec![12u8];
        t.extend_from_slice(&bits.to_le_bytes());
        t.extend_from_slice(&(n as u32).to_le_bytes());
        t.extend_from_slice(&nonce.to_le_bytes());
        for v in &rs {
            t.extend_from_slice(&v.lo.to_le_bytes());
            t.extend_from_slice(&v.hi.to_le_bytes());
        }
        emit!(&t);
    }
    // 6g. stratified query phase (grind_and_sample_queries): PoW + ONE
    // `count`-word squeeze + the schedule mapping. The schedule comes from
    // the real LevelSchedule::decompose; the word→index mapping mirrors
    // ligerito.rs::queries_from_words (private — spec inlined here).
    {
        let (bits, log_block_len, count) = (2u32, 10usize, 279usize);
        let sched = LevelSchedule::decompose(count, log_block_len);
        assert_eq!(sched.queries(), count);
        let (nonce, words) = ch.grind_pow_and_sample_f128_vec(bits, count);
        let queries: Vec<usize> = sched
            .query_strata()
            .zip(&words)
            .map(|((c, stratum), v)| {
                let lo_bits = log_block_len - c;
                let mask = (1usize << lo_bits) - 1;
                (stratum << lo_bits) | ((v.lo as usize) & mask)
            })
            .collect();
        let mut t = vec![13u8];
        t.extend_from_slice(&bits.to_le_bytes());
        t.extend_from_slice(&(log_block_len as u32).to_le_bytes());
        t.extend_from_slice(&(count as u32).to_le_bytes());
        t.extend_from_slice(&nonce.to_le_bytes());
        for &q in &queries {
            t.extend_from_slice(&(q as u64).to_le_bytes());
        }
        emit!(&t);
    }

    // 7. observe a label, then sample again (binds to everything above)
    {
        let label = b"level-1";
        ch.observe_label(label);
        let mut t = vec![4u8];
        t.extend_from_slice(&(label.len() as u32).to_le_bytes());
        t.extend_from_slice(label);
        emit!(&t);
    }
    {
        let r = ch.sample_f128();
        let mut t = vec![3u8];
        t.extend_from_slice(&r.lo.to_le_bytes());
        t.extend_from_slice(&r.hi.to_le_bytes());
        emit!(&t);
    }

    w.write_all(&n_ops.to_le_bytes())?;
    w.write_all(&body)?;
    w.flush()?;
    eprintln!("wrote challenger oracle to {path}: {n_ops} ops (from real FsChallenger)");
    Ok(())
}
