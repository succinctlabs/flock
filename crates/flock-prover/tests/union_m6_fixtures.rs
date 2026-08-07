//! Byte-identity anchors for the MERGED transport — the shipped wire-v6
//! protocol. An optimization must produce byte-identical proofs; only a
//! deliberate protocol change may move these digests, and it must re-pin
//! them with a history entry below.
//!
//! The fixtures are SHA-256 digests over deterministic seeded witnesses.
//! Everything downstream of the seeds is deterministic — witness drivers
//! are pure, the challenger is Fiat-Shamir, and all parallel reductions are
//! XOR/add in GF(2^128) (associative + commutative, so the rayon split
//! cannot change a value) — so the digests are stable across runs and
//! thread counts.
//!
//! Covers the mixed union path at full, partial, and zero-count utilization
//! (nu = 10, digested at the `proof_io` WIRE encoding) AND single-slot
//! full-utilization anchors (BLAKE3, SHA-256) for identity compaction and
//! the power-of-two-lane commit.
//!
//! Run with `cargo test --release -p flock-prover --test union_m6_fixtures
//! -- --ignored`. To regenerate digests after an INTENTIONAL transcript
//! change, run with `M6_FIXTURES_PRINT=1 ... --nocapture` and update the
//! constants.
//!
//! Re-pin history (of the file's earlier, jagged-transport fixtures):
//! integer-lane union commit; BLAKE3 I/O-region word alignment. The jagged
//! fixtures themselves were removed with the jagged transport (2026-08-02)
//! after a final green run; these merged pins were minted just before that
//! removal, against the same witness streams.
//! Re-pinned on `recursion_circuit` 2026-08-02 at the jagged-removal merge:
//! this branch has replacement sampling (`4e46b0a`, one batched squeeze per
//! Ligerito level), which `multitable` predates — the multitable-minted
//! digests can never match here. Same statements, same witness streams.
//! Re-pinned 2026-08-02: Merkle capping (proof_io v7): cap layers absorbed instead of roots (ObserveBytes 32 -> 32*2^c per commit absorb); octopus multi-proofs replaced by flat per-query capped paths.

use ::sha2 as sha2_hash;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::proof::{R1csClaim, R1csProofMergedLigerito};
use flock_prover::challenger::FsChallenger;
use flock_prover::mixed::MixedRegistryId;
use flock_prover::pcs::{Commitment, PcsParams};
use flock_prover::proof_io::MixedProofBundleLigerito;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::{blake3, sha2};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
use sha2_hash::Digest as _;
use std::array::from_fn;
use std::env::var_os;

const DOMAIN: &[u8] = b"flock-m6-fixture-v0";

/// SplitMix64 PRNG, deterministic.
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

fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<blake3::Compression> {
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = from_fn(|_| rng.next_u32());
            let m: [u32; 16] = from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

fn random_sha2_inputs(rng: &mut Rng, n: usize) -> Vec<sha2::Compression> {
    (0..n)
        .map(|_| (from_fn(|_| rng.next_u32()), from_fn(|_| rng.next_u32())))
        .collect()
}

// Re-pinned 2026-08-02: multipoint-twisted assist (proof_io v8) — the
// per-statement assist became 128K dual values + one product sumcheck +
// one untwisted anchor; transcript + wire moved by design.
// Re-pinned 2026-08-02: two-product multipoint grouping (proof_io v9) —
// packed-direct claims collapse into merged-column scalar groups (one
// dual value each); the multipoint label bumped to v1, so even the
// boolean-only fixtures here (no packed-direct claims) move.
fn check(label: &str, expected: &str, got: String) {
    if var_os("M6_FIXTURES_PRINT").is_some() {
        println!("(\"{label}\", \"{got}\"),");
        return;
    }
    assert_eq!(
        got, expected,
        "M6 byte-identity broken for fixture `{label}`: the prover's output \
         bytes diverged from the pre-M6 base commit"
    );
}

/// SHA-256 over the merged proof bundle: bincode(proof) ‖ commitment root ‖
/// the two claim values.
fn merged_bundle_digest(
    proof: &R1csProofMergedLigerito,
    commitment: &Commitment,
    claim: &R1csClaim,
) -> String {
    let mut h = sha2_hash::Sha256::new();
    h.update(bincode::serialize(proof).expect("proof serializes"));
    h.update(commitment.cap.as_flattened());
    for v in [claim.ab.value, claim.c.value] {
        h.update(v.lo.to_le_bytes());
        h.update(v.hi.to_le_bytes());
    }
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHIPPED mixed transcript, pinned at the WIRE encoding: the digest is
/// over `MixedProofBundleLigerito::to_bytes()` — magic, version, flavor,
/// registry id, counts vector, commitment, and the merged proof — plus the
/// claim values. The registry here (BLAKE3+SHA-256 at ν = 10) IS the
/// `Blake3Sha2Nu10` tier, so this pins exactly what `proof_io` puts on disk
/// for a v6 mixed proof. Same count ladder and witness streams as the
/// removed jagged fixture, whose digests pinned the same statements.
#[test]
#[ignore] // Heavier — run with `cargo test --release ... -- --ignored`.
fn m6_merged_union_proof_bytes_pinned() {
    const FIXTURES: [(&str, [usize; 2], &str); 4] = [
        (
            "merged-nu10-1024-1024",
            [1024, 1024],
            "f4effb7060f7f8f5fb7a914bc0ea5393dafd1af8ad5ae524141c4759ce386487",
        ),
        (
            "merged-nu10-50-37",
            [50, 37],
            "682b7ea380d97e0186dd668d124d625cba281bb0ecefb55b403d20839e348ff8",
        ),
        (
            "merged-nu10-8-8",
            [8, 8],
            "2390d125d4a002b66650759fd47bca7a09dacf111287cd37385cbfaabec8e538",
        ),
        (
            "merged-nu10-0-64",
            [0, 64],
            "e0d3d4d094431e8feb1450814e51f15173101526bcf3e564e3eb3b7582212c8f",
        ),
    ];

    let nu = 10usize;
    let sha2_r1cs = sha2::build_block_r1cs(nu);
    let blake3_r1cs = blake3::build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            TableType::from_block_r1cs(&blake3_r1cs),
            TableType::from_block_r1cs(&sha2_r1cs),
        ],
        nu,
    );
    let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_circuit = blake3_r1cs.csc_lincheck_circuit();

    for (label, counts, expected) in FIXTURES {
        let [n_sha2, n_blake3] = counts;
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            // The shipped union configuration (integer-lane commit).
            num_lanes: union.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        // Same per-fixture seeds as the jagged fixture: same witnesses,
        // same statements — only the transport differs.
        let mut rng = Rng::new(0x4D36_0000 ^ ((n_sha2 as u64) << 16) ^ n_blake3 as u64);
        let sha2_inputs = random_sha2_inputs(&mut rng, n_sha2);
        let blake3_inputs = random_blake3_inputs(&mut rng, n_blake3);

        let slots = vec![
            UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(&sha2_inputs, nu),
                s2_circuit,
            ),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(&blake3_inputs, nu),
                b3_circuit,
            ),
        ];
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prover::prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch);
        let bundle = MixedProofBundleLigerito {
            registry_id: MixedRegistryId::Blake3Sha2Nu10,
            counts: counts.iter().map(|&n| n as u64).collect(),
            commitment: commitment.clone(),
            proof: proof.clone(),
        };
        let mut h = sha2_hash::Sha256::new();
        h.update(bundle.to_bytes());
        for v in [claim.ab.value, claim.c.value] {
            h.update(v.lo.to_le_bytes());
            h.update(v.hi.to_le_bytes());
        }
        let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        check(label, expected, got);
    }
}

/// Single-slot MERGED anchors at full utilization: identity compaction
/// (q IS the padded buffer — no compaction copy), the power-of-two-lane
/// commit path (`num_lanes: None`, which the integer-lane mixed config
/// never exercises), and the full-utilization `generate_witness_batch_major`
/// drivers. These replace the single-table direct-jagged anchors when that
/// path is removed.
#[test]
#[ignore] // Heavier — run with `cargo test --release ... -- --ignored`.
fn m6_single_slot_merged_anchor_proof_bytes_pinned() {
    // BLAKE3, 256 blocks (m = 22).
    {
        const EXPECTED: &str = "c8d1fd3dc58d497a9cd69df9e06335a7b585a91d48b78f2d58ff4a694ae3f96b";
        let n_blocks = 256usize;
        let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
        let mut rng = Rng::new(0x4D36_B3B3);
        let inputs = random_blake3_inputs(&mut rng, n_blocks);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let registry = Registry::new(
            vec![TableType::from_block_r1cs(&setup.r1cs)],
            setup.r1cs.n_log(),
        );
        let union = UnionInstance::new(&registry, vec![n_blocks]);
        assert!(union.compaction_is_identity());
        let slot = UnionSlotProverInput::new(
            blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
            circuit,
        );
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prover::prove_fast_ligerito_union(&union, &setup.pcs_params, vec![slot], &mut ch);
        check(
            "merged-anchor-blake3-m22",
            EXPECTED,
            merged_bundle_digest(&proof, &commitment, &claim),
        );
    }

    // SHA-256, 128 blocks (m = 22).
    {
        const EXPECTED: &str = "96204108483f72c006bede7f3706c63222b439e6f7e3b2e50440b400e29ed5bb";
        let n_blocks = 128usize;
        let setup = sha2::Sha256HybridSetup::new_batch_major(n_blocks);
        let mut rng = Rng::new(0x4D36_5252);
        let inputs = random_sha2_inputs(&mut rng, n_blocks);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let registry = Registry::new(
            vec![TableType::from_block_r1cs(&setup.r1cs)],
            setup.r1cs.n_log(),
        );
        let union = UnionInstance::new(&registry, vec![n_blocks]);
        assert!(union.compaction_is_identity());
        let slot = UnionSlotProverInput::new(
            sha2::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
            circuit,
        );
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prover::prove_fast_ligerito_union(&union, &setup.pcs_params, vec![slot], &mut ch);
        check(
            "merged-anchor-sha2-m22",
            EXPECTED,
            merged_bundle_digest(&proof, &commitment, &claim),
        );
    }
}
