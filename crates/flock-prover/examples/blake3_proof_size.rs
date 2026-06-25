//! Print BLAKE3 proof sizes at m=29 for both `prove_fast` (2 PCS claims) and
//! `prove_chain` (3 PCS claims). Approximate serialized size: counts the F128 /
//! Hash / RoundMessage / position bytes; Vec lengths excluded.
//!
//! Run: `cargo run --release --example blake3_proof_size`

use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::BatchOpeningProof;
use flock_prover::proof::R1csProof;
use flock_prover::r1cs_hashes::blake3::{
    Blake3Setup, Compression, K_LOG, blake3_compress, min_n_blocks_log,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
    let mut rng = Rng::new(seed);
    let mut cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
    let cv0 = cv;
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let m: [u32; 16] = std::array::from_fn(|_| rng.nx() as u32);
        let counter = 0u64;
        let block_len = 64u32;
        let flags = 0u32;
        blocks.push((cv, m, counter, block_len, flags));
        let st = blake3_compress(&cv, &m, counter, block_len, flags);
        cv = st[0..8].try_into().unwrap();
    }
    (blocks, cv0, cv)
}

/// Size of one batched PCS opening (shared between prove_fast and prove_chain).
/// F128 = 16 B, Hash = 32 B, RoundMessage = 32 B, query position = 8 B.
fn pcs_open_size(po: &BatchOpeningProof) -> (usize, usize, usize) {
    let rs_bytes: usize = po.ring_switches.iter().map(|r| r.s_hat_v.len() * 16).sum();
    let bf = &po.basefold;
    let mut bf_bytes = bf.round_messages.len() * 32
        + 32 // post_row_batch_commit
        + bf.round_commitments.len() * 32
        + 2 * 16 // final_a, final_b
        + bf.final_codeword.len() * 16
        // Shared multi-proofs (one per tree, not per query).
        + bf.initial_multi_proof.len() * 32
        + bf.post_row_batch_multi_proof.len() * 32
        + bf.epoch_multi_proofs.iter().map(|p| p.len() * 32).sum::<usize>();
    for q in &bf.queries {
        bf_bytes += 8
            + q.initial_leaf.len() * 16
            + q.post_row_batch_leaf.len() * 16
            + q.epoch_leaves.iter().map(|l| l.len() * 16).sum::<usize>();
    }
    let total = rs_bytes + bf_bytes;
    (rs_bytes, bf_bytes, total)
}

/// Detailed BaseFold breakdown so we can see what's eating the bytes.
fn dump_basefold_breakdown(po: &BatchOpeningProof) {
    let bf = &po.basefold;
    let n_q = bf.queries.len();
    let rm_b = bf.round_messages.len() * 32;
    let rc_b = bf.round_commitments.len() * 32;
    let post_b = 32;
    let final_ab_b = 2 * 16;
    let final_cw_b = bf.final_codeword.len() * 16;
    let init_mp_b = bf.initial_multi_proof.len() * 32;
    let postrb_mp_b = bf.post_row_batch_multi_proof.len() * 32;
    let epoch_mp_b: usize = bf.epoch_multi_proofs.iter().map(|p| p.len() * 32).sum();
    let epoch_mp_total: usize = bf.epoch_multi_proofs.iter().map(|p| p.len()).sum();
    let nonquery =
        rm_b + rc_b + post_b + final_ab_b + final_cw_b + init_mp_b + postrb_mp_b + epoch_mp_b;

    // Per-query breakdown (leaves only — paths are now shared across queries).
    let q = &bf.queries[0];
    let q_pos = 8;
    let q_init_leaf = q.initial_leaf.len() * 16;
    let q_postrb_leaf = q.post_row_batch_leaf.len() * 16;
    let q_epoch_leaves: usize = q.epoch_leaves.iter().map(|l| l.len() * 16).sum();
    let q_one = q_pos + q_init_leaf + q_postrb_leaf + q_epoch_leaves;

    println!("\n  ── BaseFold breakdown ─────────────────");
    println!(
        "  one-shot pieces:                {} ({} non-query bytes)",
        fmt_kb(nonquery),
        nonquery
    );
    println!(
        "    round_messages [{}]:           {}",
        bf.round_messages.len(),
        fmt_kb(rm_b)
    );
    println!(
        "    round_commitments [{}]:        {}",
        bf.round_commitments.len(),
        fmt_kb(rc_b)
    );
    println!("    post_row_batch_commit:          {}", fmt_kb(post_b));
    println!("    final_a + final_b:              {}", fmt_kb(final_ab_b));
    println!(
        "    final_codeword [{}]:              {}",
        bf.final_codeword.len(),
        fmt_kb(final_cw_b)
    );
    println!(
        "    initial_multi_proof [{}]:       {}",
        bf.initial_multi_proof.len(),
        fmt_kb(init_mp_b)
    );
    println!(
        "    post_row_batch_multi_proof [{}]: {}",
        bf.post_row_batch_multi_proof.len(),
        fmt_kb(postrb_mp_b)
    );
    println!(
        "    epoch_multi_proofs [{} epochs, {} total]: {}",
        bf.epoch_multi_proofs.len(),
        epoch_mp_total,
        fmt_kb(epoch_mp_b)
    );
    println!("  per-query (×{n_q}):               {}", fmt_kb(q_one));
    println!("    position:                       {}", fmt_kb(q_pos));
    println!(
        "    initial_leaf [{}]:              {}",
        q.initial_leaf.len(),
        fmt_kb(q_init_leaf)
    );
    println!(
        "    post_row_batch_leaf [{}]:        {}",
        q.post_row_batch_leaf.len(),
        fmt_kb(q_postrb_leaf)
    );
    println!(
        "    epoch_leaves [{} epochs, {} total]: {}",
        q.epoch_leaves.len(),
        q.epoch_leaves.iter().map(|l| l.len()).sum::<usize>(),
        fmt_kb(q_epoch_leaves)
    );
    println!("  queries total:                  {}", fmt_kb(n_q * q_one));
    println!(
        "  basefold total:                 {}",
        fmt_kb(nonquery + n_q * q_one)
    );
}

fn r1cs_proof_size(p: &R1csProof) -> (usize, usize, (usize, usize, usize)) {
    let zc = &p.zerocheck;
    let zc_bytes =
        (zc.round1_ab.len() + zc.round1_c.len()) * 16 + zc.multilinear_rounds.len() * 32 + 3 * 16;
    let lc_bytes = p.lincheck.rounds.len() * 32 + p.lincheck.z_partial.len() * 16;
    let pcs = pcs_open_size(&p.pcs_open);
    (zc_bytes, lc_bytes, pcs)
}

fn fmt_kb(b: usize) -> String {
    if b < 10_000 {
        format!("{:>6} B", b)
    } else {
        format!("{:>6.1} KB", b as f64 / 1024.0)
    }
}

fn main() {
    // m=29 headline for BLAKE3: K_LOG=14, n_blocks=32768 → m=14+15=29.
    let n_blocks: usize = 32768;
    let n_log = min_n_blocks_log(n_blocks);
    let m = K_LOG + n_log;
    assert_eq!(m, 29);

    println!("BLAKE3 proof sizes at m=29 ({n_blocks} compressions)");
    println!("===============================================================");

    let setup = Blake3Setup::new(n_blocks);
    let (blocks, cv_0, cv_last) = honest_chain(n_blocks, 0xC0FFEE_BEEF);

    // ---- prove_fast_basefold (base, 2 PCS claims: [ab, c]). This example
    // breaks down the BaseFold proof layout; the Ligerito backend (the
    // `prove_fast` default) is covered by the *_lig_vs_bf benches.
    let mut ch = FsChallenger::new(b"blake3-size");
    let (proof_fast, _comm, _claim) = setup.prove_fast_basefold(&blocks, &mut ch);
    let (zc_f, lc_f, (rs_f, bf_f, pcs_f)) = r1cs_proof_size(&proof_fast);
    let total_fast = zc_f + lc_f + pcs_f;
    println!("\n[prove_fast_basefold] (2 PCS claims: ab, c)");
    println!("  zerocheck:        {}", fmt_kb(zc_f));
    println!("  lincheck:         {}", fmt_kb(lc_f));
    println!("  pcs_open total:   {}", fmt_kb(pcs_f));
    println!("    ring_switches:  {}", fmt_kb(rs_f));
    println!("    basefold+queries: {}", fmt_kb(bf_f));
    println!("  TOTAL:            {}", fmt_kb(total_fast));
    dump_basefold_breakdown(&proof_fast.pcs_open);

    // ---- prove_chain_basefold (full, 3 PCS claims: [ab, c, chain] + shift sumcheck).
    let mut ch = FsChallenger::new(b"blake3-size");
    let (chain_proof, comm) = setup.prove_chain_basefold(&blocks, &mut ch);
    // ChainProof fields: zerocheck, lincheck, shift, pcs_open.
    let zc_c = (chain_proof.zerocheck.round1_ab.len() + chain_proof.zerocheck.round1_c.len()) * 16
        + chain_proof.zerocheck.multilinear_rounds.len() * 32
        + 3 * 16;
    let lc_c = chain_proof.lincheck.rounds.len() * 32 + chain_proof.lincheck.z_partial.len() * 16;
    let shift_c = chain_proof.shift.rounds.len() * 32 + 16;
    let (rs_c, bf_c, pcs_c) = pcs_open_size(&chain_proof.pcs_open);
    let total_chain = zc_c + lc_c + shift_c + pcs_c;

    println!("\n[prove_chain] (3 PCS claims: ab, c, chain + shift sumcheck)");
    println!("  zerocheck:        {}", fmt_kb(zc_c));
    println!("  lincheck:         {}", fmt_kb(lc_c));
    println!("  shift_sumcheck:   {}", fmt_kb(shift_c));
    println!("  pcs_open total:   {}", fmt_kb(pcs_c));
    println!(
        "    ring_switches:  {}  ({} entries: ab,c,chain)",
        fmt_kb(rs_c),
        chain_proof.pcs_open.ring_switches.len()
    );
    println!("    basefold+queries: {}", fmt_kb(bf_c));
    println!("  TOTAL:            {}", fmt_kb(total_chain));

    // ---- Delta + verify (sanity).
    let delta = total_chain.saturating_sub(total_fast);
    println!(
        "\nChain proof overhead (size): +{} (+{:.1}%)",
        fmt_kb(delta),
        100.0 * delta as f64 / total_fast as f64
    );

    let mut chv = FsChallenger::new(b"blake3-size");
    setup
        .verify_chain_basefold(&comm, &chain_proof, &cv_0, &cv_last, &mut chv)
        .expect("chain proof must verify");
    println!("\n(verify_chain_basefold succeeded ✓)");
}
