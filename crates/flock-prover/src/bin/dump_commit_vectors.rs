//! Dump a full PCS-commit oracle (input + golden codeword + Merkle root) from
//! the *real* `flock` implementation, so the CUDA port (`cuda-ghash/`) can be
//! checked bit-for-bit against it — the same workflow as `dump_ghash_vectors`,
//! but for the whole commit pipeline (interleaved additive NTT + SHA-256
//! Merkle tree) rather than a single field multiply.
//!
//! The GPU side reads this file, runs its own NTT + Merkle kernels on the same
//! `z_packed`, and asserts:
//!   - device codeword == `codeword` (validates the additive-NTT kernel), and
//!   - device root     == `root`     (validates the SHA-256 Merkle kernel).
//!
//! Output: little-endian binary to argv[1] (default commit_vectors.bin). The
//! file is self-describing (carries every derived param), so the CUDA loader
//! never has to recompute `PcsParams`:
//!
//!   magic           u32 = 0x434D5431 ("CMT1")
//!   m               u32
//!   log_inv_rate    u32
//!   log_batch_size  u32
//!   k_code          u32   (= log_dim, the per-lane NTT size in log2)
//!   num_ntts        u32   (= 2^log_batch_size, interleaved lanes)
//!   n_positions     u32   (codeword positions per lane)
//!   n_leaves        u32   (Merkle leaves = n_positions)
//!   leaf_size_bytes u32   (= num_ntts * 16)
//!   msg_len         u32   (= 2^log_msg_len, length of z_packed)
//!   msg_len   * { lo, hi } : u64 each   — z_packed (the message)
//!   cw_len          u32   (= n_positions * num_ntts, length of codeword)
//!   cw_len    * { lo, hi } : u64 each   — golden codeword (post-NTT, SoA)
//!   root            [u8; 32]            — golden Merkle root
//!
//! Run (defaults m=20, rate 1/2, batch 32 lanes):
//!   cargo run --release --bin dump_commit_vectors -- cuda-ghash/commit_vectors.bin
//!   cargo run --release --bin dump_commit_vectors -- out.bin 24 1 5

use env::args;
use flock_prover::merkle::cap_layer;
use std::env;
use std::fs::File;
use std::io::Result;
use std::io::{BufWriter, Write};

use flock_hash::HashKind;
use flock_prover::pcs::{PcsParams, commit, pack_witness};

use flock_core::test_rng::Rng;

/// Site-specific draws kept verbatim from this file's former local `Rng`.
trait RngExt {
    fn bits_packed(&mut self, n: usize) -> Vec<bool>;
}
impl RngExt for Rng {
    fn bits_packed(&mut self, n: usize) -> Vec<bool> {
        let mut out = Vec::with_capacity(n);
        let mut word = 0u64;
        for i in 0..n {
            if i % 64 == 0 {
                word = self.next_u64();
            }
            out.push((word >> (i % 64)) & 1 == 1);
        }
        out
    }
}

fn main() -> Result<()> {
    let path = args()
        .nth(1)
        .unwrap_or_else(|| "commit_vectors.bin".to_string());
    let m: usize = args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let log_inv_rate: usize = args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let log_batch_size: usize = args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let params = PcsParams {
        m,
        log_inv_rate,
        log_batch_size,
        // Power-of-two commit: the CUDA kernels implement the classic dense
        // path, not the integer-lane commit.
        num_lanes: None,
        profile: Default::default(),
        // Pin SHA-256: the CUDA Merkle kernels (cuda-ghash/merkle.cuh) implement it.
        merkle_hash: HashKind::Sha256,
    };

    // Reproducible Boolean witness → packed F128 message, then the real commit.
    let mut rng = Rng::new(0xC0FFEE);
    let z = rng.bits_packed(1usize << m);
    let z_packed = pack_witness(&z, m);
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());

    let (commitment, prover_data) = commit(&z_packed, &params);
    let codeword = &prover_data.codeword;
    // The commitment now carries a CAP LAYER instead of a single root; the
    // tree root is the depth-0 cap, which is what the CUDA Merkle kernel
    // reproduces.
    let root = cap_layer(&prover_data.merkle_tree, commitment.params.n_leaves(), 0)[0];

    let mut w = BufWriter::new(File::create(&path)?);
    let u32le = |w: &mut BufWriter<File>, v: usize| -> Result<()> {
        w.write_all(&(v as u32).to_le_bytes())
    };

    w.write_all(&0x434D_5431u32.to_le_bytes())?; // "CMT1"
    u32le(&mut w, m)?;
    u32le(&mut w, log_inv_rate)?;
    u32le(&mut w, log_batch_size)?;
    u32le(&mut w, params.k_code())?;
    u32le(&mut w, params.num_ntts())?;
    u32le(&mut w, params.n_positions())?;
    u32le(&mut w, params.n_leaves())?;
    u32le(&mut w, params.leaf_size_bytes())?;

    u32le(&mut w, z_packed.len())?;
    for v in &z_packed {
        w.write_all(&v.lo.to_le_bytes())?;
        w.write_all(&v.hi.to_le_bytes())?;
    }

    u32le(&mut w, codeword.len())?;
    for v in codeword {
        w.write_all(&v.lo.to_le_bytes())?;
        w.write_all(&v.hi.to_le_bytes())?;
    }

    w.write_all(&root)?;
    w.flush()?;

    eprintln!(
        "wrote commit oracle to {path}: m={m} rate=1/2^{log_inv_rate} lanes={} \
         | msg={} F128, codeword={} F128 ({:.1} MB), root={}",
        params.num_ntts(),
        z_packed.len(),
        codeword.len(),
        (codeword.len() * 16) as f64 / (1usize << 20) as f64,
        hex32(&root),
    );
    Ok(())
}

fn hex32(b: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}
