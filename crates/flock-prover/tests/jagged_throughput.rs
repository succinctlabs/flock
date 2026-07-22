//! Throughput comparison between the direct and jagged opening paths
//! (Phase 1 gate measurement of the multi-table design): BLAKE3 at m = 30
//! (2^16 compressions), timing witness generation, both prove paths, and
//! both verifiers. Run explicitly:
//!
//! `cargo test --release -p flock-prover --test jagged_throughput -- --ignored --nocapture`

use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::prover;
use flock_prover::r1cs_hashes::blake3;
use flock_prover::verifier;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
}

const DOMAIN: &[u8] = b"flock-jagged-throughput-v0";
const ITERS: usize = 2;

#[test]
#[ignore] // Heavy (m = 30) — run explicitly with --ignored --nocapture
fn blake3_m30_direct_vs_jagged_throughput() {
    let n_blocks = 1usize << 16; // k_log = 14 => m = 30
    let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
    let mut rng = Rng::new(0x7B96_00B3);
    let inputs: Vec<blake3::Compression> = (0..n_blocks)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect();
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();
    println!(
        "BLAKE3 m={} ({} blocks), {} timed iterations per path (best-of reported)",
        setup.m(),
        n_blocks,
        ITERS
    );

    // Witness generation, timed once (identical for both paths).
    let t = Instant::now();
    let (z, a, b, stripe) = blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log());
    let witgen_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("witness gen: {witgen_ms:.0} ms");
    drop((z, a, b, stripe));

    let mut best_direct = f64::INFINITY;
    let mut best_jagged = f64::INFINITY;
    let mut direct_out = None;
    let mut jagged_out = None;
    for _ in 0..ITERS {
        let (z, a, b, stripe) =
            blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log());
        let mut ch = FsChallenger::new(DOMAIN);
        let t = Instant::now();
        let out = prover::prove_fast_ligerito_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            stripe,
            lc_circuit,
            None,
            &mut ch,
        );
        best_direct = best_direct.min(t.elapsed().as_secs_f64() * 1e3);
        direct_out = Some(out);

        let (z, a, b, stripe) =
            blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log());
        let mut ch = FsChallenger::new(DOMAIN);
        let t = Instant::now();
        let out = prover::prove_fast_ligerito_jagged_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            stripe,
            lc_circuit,
            None,
            &mut ch,
        );
        best_jagged = best_jagged.min(t.elapsed().as_secs_f64() * 1e3);
        jagged_out = Some(out);
    }
    let (proof_d, comm_d, _) = direct_out.unwrap();
    let (proof_j, comm_j, _) = jagged_out.unwrap();
    assert_eq!(comm_d.root, comm_j.root, "commitment root diverged");

    let hps = |prove_ms: f64| n_blocks as f64 / ((witgen_ms + prove_ms) / 1e3);
    println!(
        "prove direct: {best_direct:.0} ms  ({:.0} hashes/s e2e)",
        hps(best_direct)
    );
    println!(
        "prove jagged: {best_jagged:.0} ms  ({:.0} hashes/s e2e)",
        hps(best_jagged)
    );
    println!(
        "jagged prove overhead: {:+.1}% (prove-only), {:+.1}% (e2e incl. witness gen)",
        (best_jagged / best_direct - 1.0) * 100.0,
        ((witgen_ms + best_jagged) / (witgen_ms + best_direct) - 1.0) * 100.0
    );

    let t = Instant::now();
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito(
        &setup.r1cs,
        &comm_d,
        &proof_d,
        lc_circuit,
        &setup.pcs_params,
        &mut ch,
    )
    .expect("direct verify failed");
    let vd_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_jagged(
        &setup.r1cs,
        &comm_j,
        &proof_j,
        lc_circuit,
        &setup.pcs_params,
        &mut ch,
    )
    .expect("jagged verify failed");
    let vj_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("verify direct: {vd_ms:.1} ms, verify jagged: {vj_ms:.1} ms");

    let db = bincode::serialize(&proof_d).unwrap().len();
    let jb = bincode::serialize(&proof_j).unwrap().len();
    println!("proof size: direct {db} B, jagged {jb} B ({:+} B)", jb as i64 - db as i64);
}
