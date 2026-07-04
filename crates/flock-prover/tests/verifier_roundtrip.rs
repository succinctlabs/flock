//! End-to-end prove → verify roundtrips and tamper-rejection tests.
//!
//! These live in `flock-prover` (not `flock-core`) because they exercise the
//! prove path; the verifier they call lives in `flock-core`. Moved here from
//! `flock_core::verifier`'s in-crate test module when the crates were split.

use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::{self, PcsParams};
use flock_prover::proof_io::R1csProofBundle;
use flock_prover::prover::{prove, prove_ligerito};
use flock_prover::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};
use flock_prover::verifier::{self, VerifyError, verify};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn bits(&mut self, n: usize) -> Vec<bool> {
        (0..n).map(|_| self.next_u64() & 1 == 1).collect()
    }
}

fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k).map(|i| vec![i]).collect(),
    }
}

fn default_pcs_params(m: usize) -> PcsParams {
    PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 5,
        profile: Default::default(),
    }
}

/// Build an identity-`C` R1CS with identity `A_0`/`B_0` at the given shape.
fn identity_r1cs(m: usize, k_log: usize, k_skip: usize, useful_bits: usize) -> BlockR1cs {
    BlockR1cs {
        m,
        k_log,
        k_skip,
        useful_bits,
        a_0: identity(1 << k_log),
        b_0: identity(1 << k_log),
        c_0: identity(1 << k_log),
        layout: WitnessLayout::RowMajor,
        const_pin: None,
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    }
}

/// End-to-end R1CS roundtrip using the **Ligerito** PCS backend.
/// Requires a larger m than the BaseFold roundtrip — Ligerito's
/// per-level query counts demand block_len ≥ ~243 at L0, so m ≥ 19 or so.
#[test]
#[ignore] // Heavier — run with `cargo test r1cs_prove_verify_roundtrip_ligerito -- --ignored --nocapture`
fn r1cs_prove_verify_roundtrip_ligerito() {
    let m = 22;
    let k_log = 6;
    let k_skip = 6;
    let r1cs = identity_r1cs(m, k_log, k_skip, 1 << k_log);
    let mut rng = Rng::new(20_240_609);
    let z = rng.bits(r1cs.n());
    assert!(r1cs.satisfies(&z));

    // log_batch_size = 6 so Ligerito's initial_k = 6 reuses the L0 commit.
    let pcs_params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    let mut ch_p = FsChallenger::new(b"flock-lig-r1cs-v0");
    let z_packed = pcs::pack_witness(&z, r1cs.m);
    let (proof, commitment, claim_p) = prove_ligerito(&r1cs, z_packed, &pcs_params, &mut ch_p);

    let mut ch_v = FsChallenger::new(b"flock-lig-r1cs-v0");
    let lc_circuit = r1cs.sparse_lincheck_circuit();
    let claim_v = verifier::verify_ligerito(
        &r1cs,
        &commitment,
        &proof,
        &lc_circuit,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("ligerito verify rejected honest proof: {e:?}"));
    assert_eq!(claim_p, claim_v);
}

/// Break down where the bytes go in a full R1cs proof bundle. Uses identity
/// R1CS at a moderate m with the production-style PCS config
/// (log_batch_size = 6 like keccak/blake3). Prints to stderr — run with
/// `cargo test full_proof_size_breakdown -- --nocapture --ignored`.
#[test]
#[ignore]
fn full_proof_size_breakdown() {
    for &m in &[18usize, 20, 22] {
        let k_log = 6;
        let k_skip = 6;
        let r1cs = identity_r1cs(m, k_log, k_skip, 1 << k_log);
        let mut rng = Rng::new(m as u64);
        let z = rng.bits(r1cs.n());
        assert!(r1cs.satisfies(&z));

        // Production-style: log_batch_size = 6 (matches keccak/blake3 setups).
        let pcs_params = PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: Default::default(),
        };
        let mut ch_p = FsChallenger::new(b"break-down");
        let z_packed = pcs::pack_witness(&z, r1cs.m);
        let (proof, commitment, _) = prove(&r1cs, &z_packed, &pcs_params, &mut ch_p);

        let kb = |b: usize| {
            if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
            } else if b >= 1024 {
                format!("{:.1} KB", b as f64 / 1024.0)
            } else {
                format!("{} B", b)
            }
        };

        // Bincode-serialize each piece independently to get its share.
        let zerocheck_b = bincode::serialize(&proof.zerocheck).unwrap().len();
        let lincheck_b = bincode::serialize(&proof.lincheck).unwrap().len();
        let basefold = &proof.pcs_open.basefold;
        let basefold_b = bincode::serialize(basefold).unwrap().len();
        let ring_switches_b = bincode::serialize(&proof.pcs_open.ring_switches)
            .unwrap()
            .len();
        let commitment_b = bincode::serialize(&commitment).unwrap().len();

        // BaseFold sub-breakdown.
        let round_msgs_b = bincode::serialize(&basefold.round_messages).unwrap().len();
        let queries_b = bincode::serialize(&basefold.queries).unwrap().len();
        let init_mp_b = basefold.initial_multi_proof.len() * 32;
        let post_rb_mp_b = basefold.post_row_batch_multi_proof.len() * 32;
        let epoch_mp_b: usize = basefold
            .epoch_multi_proofs
            .iter()
            .map(|m| m.len() * 32)
            .sum();
        let final_codeword_b = basefold.final_codeword.len() * 16;
        let basefold_other = basefold_b
            - round_msgs_b
            - queries_b
            - init_mp_b
            - post_rb_mp_b
            - epoch_mp_b
            - final_codeword_b;

        // Per-query leaf breakdown
        let n_queries = basefold.queries.len();
        let (init_leaves_b, post_rb_leaves_b, epoch_leaves_b, positions_b) = {
            let mut a = 0;
            let mut b = 0;
            let mut c = 0;
            let mut d = 0;
            for q in &basefold.queries {
                a += q.initial_leaf.len() * 16;
                b += q.post_row_batch_leaf.len() * 16;
                c += q.epoch_leaves.iter().map(|el| el.len() * 16).sum::<usize>();
                d += 8; // usize approx
            }
            (a, b, c, d)
        };

        let bundle = R1csProofBundle {
            commitment: commitment.clone(),
            proof,
        };
        let total_b = bundle.to_bytes().len();

        eprintln!("=========================================================================");
        eprintln!("m = {m}   (PCS: log_inv_rate=1, log_batch_size=6, n_queries={n_queries})");
        eprintln!("  Commitment           : {}", kb(commitment_b));
        eprintln!("  Zerocheck proof      : {}", kb(zerocheck_b));
        eprintln!("  Lincheck proof       : {}", kb(lincheck_b));
        eprintln!("  Ring-switch proofs   : {}", kb(ring_switches_b));
        eprintln!("  BaseFold proof       : {}", kb(basefold_b));
        eprintln!("    round_messages     : {}", kb(round_msgs_b));
        eprintln!(
            "    queries (LEAVES)   : {}    [largest single line]",
            kb(queries_b)
        );
        eprintln!("      initial_leaves   : {}", kb(init_leaves_b));
        eprintln!("      post_rb_leaves   : {}", kb(post_rb_leaves_b));
        eprintln!("      epoch_leaves     : {}", kb(epoch_leaves_b));
        eprintln!(
            "      positions+tags   : {}",
            kb(positions_b + queries_b - init_leaves_b - post_rb_leaves_b - epoch_leaves_b)
        );
        eprintln!("    initial_multi_proof: {}", kb(init_mp_b));
        eprintln!("    post_rb_multi_proof: {}", kb(post_rb_mp_b));
        eprintln!("    epoch_multi_proofs : {}", kb(epoch_mp_b));
        eprintln!("    final_codeword     : {}", kb(final_codeword_b));
        eprintln!("    bookkeeping        : {}", kb(basefold_other));
        eprintln!("  TOTAL bundle         : {}", kb(total_b));
    }
}

/// End-to-end honest prove-verify roundtrip including PCS.
#[test]
fn r1cs_prove_verify_roundtrip_honest() {
    for &m in &[13usize, 14, 15] {
        let k_log = 6;
        let k_skip = 6;
        let r1cs = identity_r1cs(m, k_log, k_skip, 1 << k_log);
        let mut rng = Rng::new(20_240_525 + m as u64);
        let z = rng.bits(r1cs.n());
        assert!(r1cs.satisfies(&z));

        let pcs_params = default_pcs_params(m);
        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let z_packed = pcs::pack_witness(&z, r1cs.m);
        let (proof, commitment, claim_p) = prove(&r1cs, &z_packed, &pcs_params, &mut ch_p);

        let mut ch_v = FsChallenger::new(b"flock-test-v0");
        let lc_circuit = r1cs.sparse_lincheck_circuit();
        let claim_v = verify(&r1cs, &commitment, &proof, &lc_circuit, &mut ch_v)
            .unwrap_or_else(|e| panic!("verify rejected honest proof at m={m}: {e:?}"));

        assert_eq!(claim_p, claim_v, "claim mismatch at m={m}");
    }
}

#[test]
fn r1cs_verify_rejects_mutated_lincheck() {
    let m = 14;
    let r1cs = identity_r1cs(m, 6, 6, 64);
    let mut rng = Rng::new(99);
    let z = rng.bits(r1cs.n());
    let pcs_params = default_pcs_params(m);

    let mut ch_p = FsChallenger::new(b"flock-test-v0");
    let z_packed = pcs::pack_witness(&z, r1cs.m);
    let (mut proof, commitment, _) = prove(&r1cs, &z_packed, &pcs_params, &mut ch_p);
    proof.lincheck.z_partial[0].lo ^= 1;

    let mut ch_v = FsChallenger::new(b"flock-test-v0");
    let lc_circuit = r1cs.sparse_lincheck_circuit();
    let res = verify(&r1cs, &commitment, &proof, &lc_circuit, &mut ch_v);
    assert!(matches!(res, Err(VerifyError::Lincheck(_))));
}

#[test]
fn r1cs_verify_rejects_mutated_pcs() {
    let m = 14;
    let r1cs = identity_r1cs(m, 6, 6, 64);
    let mut rng = Rng::new(99);
    let z = rng.bits(r1cs.n());
    let pcs_params = default_pcs_params(m);

    let mut ch_p = FsChallenger::new(b"flock-test-v0");
    let z_packed = pcs::pack_witness(&z, r1cs.m);
    let (mut proof, commitment, _) = prove(&r1cs, &z_packed, &pcs_params, &mut ch_p);
    // Mutate the BaseFold final_a in the batched opening — must trip
    // the final sumcheck consistency check.
    proof.pcs_open.basefold.final_a.lo ^= 1;

    let mut ch_v = FsChallenger::new(b"flock-test-v0");
    let lc_circuit = r1cs.sparse_lincheck_circuit();
    let res = verify(&r1cs, &commitment, &proof, &lc_circuit, &mut ch_v);
    assert!(matches!(res, Err(VerifyError::PcsAb(_))));
}
