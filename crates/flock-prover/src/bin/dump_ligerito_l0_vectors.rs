//! Dump the full Ligerito **L0 phase** from the real `FsChallenger` +
//! `SumcheckProver` + `ligero_commit` + `induce_sumcheck_poly`, exactly as
//! `recursive_prover_with_basis_impl` runs it (`src/pcs/ligerito.rs`), so the
//! CUDA orchestrator (`cuda-ghash/test_ligerito_l0.cu`) can reproduce it
//! byte-for-byte. Now with a REAL L0 commit so query rows are actually opened.
//!
//! L0 phase: build L0 commit (wtns_0) → observe label/target/root → SumcheckProver
//! new → initial_k folds → commit f¹ → OOD intro/glue loop → query grind + sample
//! queries + α → open rows + multi-proof → induce basis₀ → introduce/glue.
//!
//! Format ("L0SC") — see the inline writes; the orchestrator reads in lockstep.
//!
//! Args: path log_n initial_k fold_bits log_ni1 log_inv_rate_1 ood_count
//!       log_inv_rate_0 num_queries_0 query_grind_bits
//! Run:  cargo run --release --bin dump_ligerito_l0_vectors -- \
//!         cuda-ghash/ligerito_l0_vectors.bin 14 4 0 3 1 2 1 40 0

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_hash::HashKind;
use flock_prover::challenger::{Challenger, FsChallenger};
use flock_prover::field::F128;
use flock_prover::lincheck::build_eq_table;
use flock_prover::ntt::AdditiveNttF128;
use flock_prover::pcs::ligerito::{
    LigeroWitness, SumcheckProver, eval_sk_at_vks, induce_sumcheck_poly, ligero_commit,
};

// The multi-proof left the live protocol (cap layers replaced it); the CUDA
// oracle pair keeps a frozen copy. Same story for the single-root absorb:
// `LigeroWitness::root()` was retired with the cap-layer switch, but this
// replay pins the transcript shape the CUDA kernels implement.
#[path = "dump_common/merkle_octopus.rs"]
mod merkle_octopus;
use merkle_octopus::merkle_multi_proof;

fn wtns_root(w: &LigeroWitness) -> flock_prover::merkle::Hash {
    w.tree[w.tree.len() - 1]
}

const PROVER_LABEL: &[u8] = b"flock-ligerito-basis-v0";

use flock_core::test_rng::Rng;

fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (n - 1).ilog2() as usize + 1
    }
}

// Replicates ligerito::sample_distinct_queries (private). Only nontrivial op is
// sample_f128 (validated byte-exact).
fn sample_distinct_queries<Ch: Challenger>(
    ch: &mut Ch,
    block_len: usize,
    count: usize,
) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let v = ch.sample_f128();
        let q = (v.lo as usize) % block_len;
        if seen.insert(q) {
            out.push(q);
        }
    }
    out.sort_unstable();
    out
}

fn wf(w: &mut impl Write, x: F128) -> std::io::Result<()> {
    w.write_all(&x.lo.to_le_bytes())?;
    w.write_all(&x.hi.to_le_bytes())
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let path = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| "ligerito_l0_vectors.bin".to_string());
    let log_n = arg(2, 14);
    let initial_k = arg(3, 4);
    let fold_bits = arg(4, 0) as u32;
    let _ = arg(5, 3); // (was log_ni1; now derived from k_rec below)
    let log_inv_rate_1 = arg(6, 1);
    let ood_count = arg(7, 2);
    let log_inv_rate_0 = arg(8, 1);
    let num_queries_0 = arg(9, 40);
    let query_grind_bits = arg(10, 0) as u32;
    // Recursive levels (general r), uniform per-level config.
    let r = arg(11, 2);
    let k_rec = arg(12, 3); // recursive_ks[i] = k_rec for all i
    let rate_rec = arg(13, 1); // log_inv_rates for recursive commits
    let ood_rec = arg(14, 1);
    let nq_rec = arg(15, 24);
    let grind_rec = arg(16, 0) as u32;
    let foldgrind_rec = arg(17, 0) as u32;
    // log_num_interleaved_1 must equal recursive_ks[0] (= k_rec).
    let log_ni1 = k_rec;

    let len = 1usize << log_n;
    let domain = b"flock-bench-v0";
    let n1 = log_n - initial_k;

    let mut rng = Rng::new(0xC0FFEE);
    let f: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
    let b1: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
    let target = rng.f128();

    // ---- L0 commit (the upstream witness commit) ----
    let log_msg_cols_0 = log_n - initial_k;
    let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate_0);
    let wtns_0 = ligero_commit(
        &f,
        log_msg_cols_0,
        initial_k,
        log_inv_rate_0,
        &ntt_0,
        HashKind::Sha256,
    );
    let l0_block_len = wtns_0.block_len;

    let mut ch = FsChallenger::new(domain);
    ch.observe_label(PROVER_LABEL);
    ch.observe_f128(target);
    ch.observe_bytes(&wtns_root(&wtns_0));

    let (mut sc, start_msg) = SumcheckProver::new(f.clone(), b1.clone(), target);
    ch.observe_f128(start_msg.u_0);
    ch.observe_f128(start_msg.u_2);

    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x4C30_5343u32.to_le_bytes())?; // "L0SC"
    w.write_all(&(domain.len() as u32).to_le_bytes())?;
    w.write_all(domain)?;
    w.write_all(&(log_n as u32).to_le_bytes())?;
    w.write_all(&(len as u32).to_le_bytes())?;
    for &x in &f {
        wf(&mut w, x)?;
    }
    for &x in &b1 {
        wf(&mut w, x)?;
    }
    wf(&mut w, target)?;
    w.write_all(&(log_inv_rate_0 as u32).to_le_bytes())?;
    w.write_all(&wtns_root(&wtns_0))?;
    w.write_all(&(initial_k as u32).to_le_bytes())?;
    w.write_all(&fold_bits.to_le_bytes())?;
    wf(&mut w, start_msg.u_0)?;
    wf(&mut w, start_msg.u_2)?;

    // ---- initial_k lane folds ----
    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for _ in 0..initial_k {
        if fold_bits > 0 {
            let nonce = ch.grind_pow(fold_bits);
            w.write_all(&nonce.to_le_bytes())?;
        }
        let r = ch.sample_f128();
        let msg = sc.fold(r);
        ch.observe_f128(msg.u_0);
        ch.observe_f128(msg.u_2);
        wf(&mut w, r)?;
        wf(&mut w, msg.u_0)?;
        wf(&mut w, msg.u_2)?;
        r_lane_fold.push(r);
    }
    wf(&mut w, sc.f()[0])?;

    // ---- commit f¹ ----
    let log_msg_cols_1 = n1 - log_ni1;
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    let f1 = sc.f().to_vec();
    let wtns_1 = ligero_commit(
        &f1,
        log_msg_cols_1,
        log_ni1,
        log_inv_rate_1,
        &ntt_1,
        HashKind::Sha256,
    );
    ch.observe_bytes(&wtns_root(&wtns_1));
    w.write_all(&(log_ni1 as u32).to_le_bytes())?;
    w.write_all(&(log_inv_rate_1 as u32).to_le_bytes())?;
    w.write_all(&wtns_root(&wtns_1))?;

    // ---- OOD intro/glue ----
    w.write_all(&(ood_count as u32).to_le_bytes())?;
    for _ in 0..ood_count {
        let z = ch.sample_f128_vec(n1);
        let eq_z = build_eq_table(&z);
        let (intro, y) = sc.introduce_new_with_eval(eq_z);
        ch.observe_f128(y);
        ch.observe_f128(intro.u_0);
        ch.observe_f128(intro.u_2);
        let beta = ch.sample_f128();
        sc.glue(beta);
        for &zi in &z {
            wf(&mut w, zi)?;
        }
        wf(&mut w, y)?;
        wf(&mut w, intro.u_0)?;
        wf(&mut w, intro.u_2)?;
        wf(&mut w, beta)?;
    }

    // ---- query phase + open L0 ----
    let nonce_0 = ch.grind_pow(query_grind_bits);
    let queries_0 = sample_distinct_queries(&mut ch, l0_block_len, num_queries_0);
    let alpha_len = ceil_log2(num_queries_0);
    let alpha_0 = ch.sample_f128_vec(alpha_len);
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| wtns_0.row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_multi_proof(&wtns_0.tree, l0_block_len, &queries_0);

    w.write_all(&query_grind_bits.to_le_bytes())?;
    w.write_all(&(num_queries_0 as u32).to_le_bytes())?;
    w.write_all(&nonce_0.to_le_bytes())?;
    w.write_all(&(l0_block_len as u64).to_le_bytes())?;
    for &q in &queries_0 {
        w.write_all(&(q as u64).to_le_bytes())?;
    }
    w.write_all(&(alpha_len as u32).to_le_bytes())?;
    for &x in &alpha_0 {
        wf(&mut w, x)?;
    }
    w.write_all(&(merkle_proof_0.len() as u32).to_le_bytes())?;
    for h in &merkle_proof_0 {
        w.write_all(h)?;
    }

    // ---- induce basis₀, introduce + glue ----
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let (basis_0, enforced_sum_0) = induce_sumcheck_poly(
        n1,
        &sks_vks_n1,
        &opened_rows_0,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    let intro_msg_0 = sc.introduce_new(basis_0.clone(), enforced_sum_0);
    ch.observe_f128(intro_msg_0.u_0);
    ch.observe_f128(intro_msg_0.u_2);
    let beta_0 = ch.sample_f128();
    sc.glue(beta_0);

    // basis_0 has length 2^n1 — dump for direct induce validation.
    for &x in &basis_0 {
        wf(&mut w, x)?;
    }
    wf(&mut w, enforced_sum_0)?;
    wf(&mut w, intro_msg_0.u_0)?;
    wf(&mut w, intro_msg_0.u_2)?;
    wf(&mut w, beta_0)?;
    wf(&mut w, sc.f()[0])?; // post-introduce folded head (light check)

    // ==== Recursive levels (general r): ligerito.rs:2855-2972 ====
    // Level i queries wtns_prev (init wtns_1) and, if not last, commits wtns_next.
    w.write_all(&(r as u32).to_le_bytes())?;
    w.write_all(&(k_rec as u32).to_le_bytes())?;
    w.write_all(&(rate_rec as u32).to_le_bytes())?;
    w.write_all(&(ood_rec as u32).to_le_bytes())?;
    w.write_all(&foldgrind_rec.to_le_bytes())?;
    w.write_all(&grind_rec.to_le_bytes())?;

    let mut prev_mat = wtns_1.mat.clone();
    let mut prev_tree = wtns_1.tree.clone();
    let mut prev_block_len = wtns_1.block_len;
    let mut prev_ni = wtns_1.num_interleaved;

    for i in 0..r {
        let mut level_rs = Vec::with_capacity(k_rec);
        for _ in 0..k_rec {
            if foldgrind_rec > 0 {
                let n = ch.grind_pow(foldgrind_rec);
                w.write_all(&n.to_le_bytes())?;
            }
            let ri = ch.sample_f128();
            let msg = sc.fold(ri);
            ch.observe_f128(msg.u_0);
            ch.observe_f128(msg.u_2);
            wf(&mut w, ri)?;
            wf(&mut w, msg.u_0)?;
            wf(&mut w, msg.u_2)?;
            level_rs.push(ri);
        }
        if i == r - 1 {
            let yr = sc.f().to_vec();
            for v in &yr {
                ch.observe_f128(*v);
            }
            let nonce_last = ch.grind_pow(grind_rec);
            let queries_last = sample_distinct_queries(&mut ch, prev_block_len, nq_rec);
            let mp = merkle_multi_proof(&prev_tree, prev_block_len, &queries_last);
            w.write_all(&(yr.len() as u32).to_le_bytes())?;
            for v in &yr {
                wf(&mut w, *v)?;
            }
            w.write_all(&nonce_last.to_le_bytes())?;
            w.write_all(&(prev_block_len as u64).to_le_bytes())?;
            w.write_all(&(nq_rec as u32).to_le_bytes())?;
            for &q in &queries_last {
                w.write_all(&(q as u64).to_le_bytes())?;
            }
            w.write_all(&(mp.len() as u32).to_le_bytes())?;
            for h in &mp {
                w.write_all(h)?;
            }
        } else {
            let n_next = sc.f().len().trailing_zeros() as usize;
            let log_msg_cols_next = n_next - k_rec;
            let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + rate_rec);
            let f_evals = sc.f().to_vec();
            let wtns_next = ligero_commit(
                &f_evals,
                log_msg_cols_next,
                k_rec,
                rate_rec,
                &ntt_next,
                HashKind::Sha256,
            );
            ch.observe_bytes(&wtns_root(&wtns_next));
            w.write_all(&wtns_root(&wtns_next))?;
            for _ in 0..ood_rec {
                let z = ch.sample_f128_vec(n_next);
                let eq_z = build_eq_table(&z);
                let (intro, y) = sc.introduce_new_with_eval(eq_z);
                ch.observe_f128(y);
                ch.observe_f128(intro.u_0);
                ch.observe_f128(intro.u_2);
                let beta = ch.sample_f128();
                sc.glue(beta);
                for &zi in &z {
                    wf(&mut w, zi)?;
                }
                wf(&mut w, y)?;
                wf(&mut w, intro.u_0)?;
                wf(&mut w, intro.u_2)?;
                wf(&mut w, beta)?;
            }
            let nonce_i = ch.grind_pow(grind_rec);
            let queries_i = sample_distinct_queries(&mut ch, prev_block_len, nq_rec);
            let alpha_len_i = ceil_log2(nq_rec);
            let alpha_i = ch.sample_f128_vec(alpha_len_i);
            let opened_rows_i: Vec<Vec<F128>> = queries_i
                .iter()
                .map(|&q| prev_mat[q * prev_ni..(q + 1) * prev_ni].to_vec())
                .collect();
            let mp = merkle_multi_proof(&prev_tree, prev_block_len, &queries_i);
            let sks_i = eval_sk_at_vks(n_next);
            let (basis_i, esum_i) = induce_sumcheck_poly(
                n_next,
                &sks_i,
                &opened_rows_i,
                &level_rs,
                &queries_i,
                &alpha_i,
            );
            let intro_i = sc.introduce_new(basis_i.clone(), esum_i);
            ch.observe_f128(intro_i.u_0);
            ch.observe_f128(intro_i.u_2);
            let beta_i = ch.sample_f128();
            sc.glue(beta_i);

            w.write_all(&nonce_i.to_le_bytes())?;
            w.write_all(&(prev_block_len as u64).to_le_bytes())?;
            w.write_all(&(nq_rec as u32).to_le_bytes())?;
            for &q in &queries_i {
                w.write_all(&(q as u64).to_le_bytes())?;
            }
            w.write_all(&(alpha_len_i as u32).to_le_bytes())?;
            for &x in &alpha_i {
                wf(&mut w, x)?;
            }
            w.write_all(&(mp.len() as u32).to_le_bytes())?;
            for h in &mp {
                w.write_all(h)?;
            }
            w.write_all(&(basis_i.len() as u32).to_le_bytes())?;
            for &x in &basis_i {
                wf(&mut w, x)?;
            }
            wf(&mut w, esum_i)?;
            wf(&mut w, intro_i.u_0)?;
            wf(&mut w, intro_i.u_2)?;
            wf(&mut w, beta_i)?;

            prev_mat = wtns_next.mat.clone();
            prev_tree = wtns_next.tree.clone();
            prev_block_len = wtns_next.block_len;
            prev_ni = wtns_next.num_interleaved;
        }
    }
    w.flush()?;
    eprintln!(
        "wrote ligerito-L0 oracle to {path}: log_n={log_n} initial_k={initial_k} \
         fold_bits={fold_bits} ood={ood_count} rate0=1/{} queries={num_queries_0} \
         grind={query_grind_bits} (real prover L0 incl. induce)",
        1 << log_inv_rate_0
    );
    Ok(())
}
