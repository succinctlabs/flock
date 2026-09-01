//! Dump a complete run of the REAL F256 Ligerito ladder
//! (`pcs::ligerito::recursive_prover_with_basis`, the
//! "flock-ligerito-basis-f256-split-v0" transcript — the only ladder the
//! branch verifier accepts) on a random witness/basis at a REGISTERED config,
//! so the CUDA port (`cuda-ghash/ligerito_f256.cuh` via
//! `cuda-ghash/test_ligerito_f256.cu`) can reproduce every proof field
//! byte-for-byte: caps, opens, capped paths, yr, the F256 sumcheck
//! transcript, OOD values, and all three PoW nonce families.
//!
//! The GPU test rebuilds the L0 commit itself (NTT + Merkle) and checks its
//! cap against the dumped one, so the commit path is validated too.
//!
//! Format ("LF25", all LE) — see the inline writes; the CUDA test reads in
//! lockstep.
//!
//! Run:  cargo run --release --bin dump_ligerito_f256_vectors -- \
//!         cuda-ghash/ligerito_f256_vectors.bin [m=22]

use std::{
    env,
    fs::File,
    io::{BufWriter, Result, Write},
};

use env::args;
use flock_core::test_rng::Rng;
use flock_prover::{
    challenger::FsChallenger,
    field::F128,
    ntt::AdditiveNttF128,
    pcs::{
        LOG_PACKING,
        ligerito::{
            LigeritoProfile, ligero_commit, prover_config_for, recursive_prover_with_basis,
        },
    },
};

fn wf(w: &mut impl Write, x: F128) -> Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn wu32(w: &mut impl Write, v: usize) -> Result<()> {
    w.write_all(&(v as u32).to_le_bytes())
}

fn wu32s(w: &mut impl Write, v: &[usize]) -> Result<()> {
    for &x in v {
        wu32(w, x)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = args().collect();
    let path = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "ligerito_f256_vectors.bin".to_string());
    let m: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(22);
    let log_n = m - LOG_PACKING;

    let cfg = prover_config_for(log_n, 6, LigeritoProfile::Fast)
        .unwrap_or_else(|e| panic!("no fast config for m={m}: {e}"));
    let initial_k = cfg.initial_k;
    let r = cfg.recursive_steps;
    let len = 1usize << log_n;
    let domain = b"flock-lig-f256-test-v0";

    let mut rng = Rng::new(0x1F25_6);
    let f: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
    let b: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
    let target = rng.f128();

    // L0 commit exactly as prove_ligerito's pcs::commit shapes it.
    let log_msg_cols_0 = log_n - initial_k;
    let ntt = AdditiveNttF128::standard(log_msg_cols_0 + cfg.log_inv_rates[0]);
    let wtns = ligero_commit(
        &f,
        log_msg_cols_0,
        initial_k,
        cfg.log_inv_rates[0],
        &ntt,
        cfg.merkle_hash,
    );

    let mut ch = FsChallenger::new(domain);
    let proof = recursive_prover_with_basis(
        &cfg,
        f.clone(),
        b.clone(),
        target,
        &wtns.mat,
        &wtns.tree,
        &mut ch,
    );

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x3532_464Cu32.to_le_bytes())?; // "LF25"
    wu32(&mut w, domain.len())?;
    w.write_all(domain)?;
    wu32(&mut w, m)?;
    wu32(&mut w, log_n)?;
    wu32(&mut w, initial_k)?;
    wu32(&mut w, r)?;
    wu32s(&mut w, &cfg.log_inv_rates)?;
    wu32s(&mut w, &cfg.recursive_ks)?;
    wu32s(&mut w, &cfg.queries)?;
    wu32s(&mut w, &cfg.grinding_bits)?;
    wu32s(&mut w, &cfg.claim_batch_grinding_bits)?;
    wu32s(&mut w, &cfg.consistency_batch_grinding_bits)?;
    wu32s(&mut w, &cfg.ood_samples)?;
    w.write_all(&(wtns.block_len as u64).to_le_bytes())?;
    wu32(&mut w, wtns.num_interleaved)?;
    for &x in &f {
        wf(&mut w, x)?;
    }
    for &x in &b {
        wf(&mut w, x)?;
    }
    wf(&mut w, target)?;

    // ---- expected proof ----
    wu32(&mut w, proof.initial_cap.len())?;
    for h in &proof.initial_cap {
        w.write_all(h)?;
    }
    wu32(&mut w, proof.recursive_caps.len())?;
    for cap in &proof.recursive_caps {
        wu32(&mut w, cap.len())?;
        for h in cap {
            w.write_all(h)?;
        }
    }
    let write_open =
        |w: &mut BufWriter<File>, rows: &[Vec<F128>], path_hashes: &[[u8; 32]]| -> Result<()> {
            wu32(w, rows.len())?;
            wu32(w, rows.first().map_or(0, |r| r.len()))?;
            for row in rows {
                for &x in row {
                    wf(w, x)?;
                }
            }
            wu32(w, path_hashes.len())?;
            for h in path_hashes {
                w.write_all(h)?;
            }
            Ok(())
        };
    wu32(&mut w, 2 + proof.recursive_proofs.len())?;
    write_open(
        &mut w,
        &proof.initial_proof.opened_rows,
        &proof.initial_proof.merkle_proof,
    )?;
    for p in &proof.recursive_proofs {
        write_open(&mut w, &p.opened_rows, &p.merkle_proof)?;
    }
    write_open(
        &mut w,
        &proof.final_proof.opened_rows,
        &proof.final_proof.merkle_proof,
    )?;
    wu32(&mut w, proof.final_proof.yr.len())?;
    for &x in &proof.final_proof.yr {
        wf(&mut w, x)?;
    }
    assert!(
        proof.sumcheck_transcript.is_empty(),
        "legacy transcript must be empty"
    );
    wu32(&mut w, proof.sumcheck_transcript_f256.len())?;
    for msg in &proof.sumcheck_transcript_f256 {
        wf(&mut w, msg.u_0.c0)?;
        wf(&mut w, msg.u_0.c1)?;
        wf(&mut w, msg.u_2.c0)?;
        wf(&mut w, msg.u_2.c1)?;
    }
    wu32(&mut w, proof.ood_values.len())?;
    for &x in &proof.ood_values {
        wf(&mut w, x)?;
    }
    assert!(
        proof.fold_grinding_nonces.is_empty(),
        "F256 ladder never fold-grinds"
    );
    let wnonces = |w: &mut BufWriter<File>, v: &[u64]| -> Result<()> {
        wu32(w, v.len())?;
        for &n in v {
            w.write_all(&n.to_le_bytes())?;
        }
        Ok(())
    };
    wnonces(&mut w, &proof.grinding_nonces)?;
    wnonces(&mut w, &proof.claim_batch_grinding_nonces)?;
    wnonces(&mut w, &proof.consistency_batch_grinding_nonces)?;
    w.flush()?;
    eprintln!(
        "wrote F256-ladder oracle to {path}: m={m} r={r} transcript={} msgs, \
         {} oods, caps {}+{}, opens {}, yr {} (from the real F256 driver)",
        proof.sumcheck_transcript_f256.len(),
        proof.ood_values.len(),
        proof.initial_cap.len(),
        proof.recursive_caps.len(),
        2 + proof.recursive_proofs.len(),
        proof.final_proof.yr.len(),
    );
    Ok(())
}
