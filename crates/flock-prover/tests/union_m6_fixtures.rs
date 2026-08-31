//! Byte-identity anchors for the MERGED transport — currently proof-IO v22.
//! An optimization must produce byte-identical proofs; only a
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
//! Re-pinned 2026-08-05: stratified queries (all TOMLs stratified = true):
//! caps at the top set bit of each level's query count, per-summand path
//! lengths (docs/stratified-queries.tex).
//! Re-pinned 2026-08-12, two 2026-08-11 protocol changes at once: path
//! truncation (`ec51b71` — per-query paths stop at the cap depth, the
//! census's redundant siblings leave the wire) and the assist transcript
//! fork (`4787509` — merged opens prove the assist beside the inner open
//! on a forked chain, unconditional). Roundtrip suites green; digests
//! stable across two print runs.
//! Re-pinned 2026-08-13 for the deliberate v18 Ligerito protocol changes:
//! two-point OOD binding, the Flock paper's Appendix C.3 algebraic grinding,
//! and strict-128 Johnson query schedules. All six fixtures moved; roundtrip
//! suites are green and the digests were identical across two print runs.
//! Re-pinned 2026-08-13 for the deliberate v19 Ligerito protocol change:
//! F256 mutual correlated agreement and the base-field split handoff. All six
//! fixtures moved; roundtrip suites are green and the digests were stable
//! across repeated deterministic generation.
//! Re-pinned 2026-08-13 for proof-IO v20. Strict Fast and Slim proofs now
//! include every non-Ligerito grinding nonce. All six deterministic digests
//! were stable across two print runs.
//! Re-pinned 2026-08-14 (fourth): the legacy cleanup — proof-IO v21 (the
//! R1cs flavor's payload became the merged union proof) moves every
//! wire-digested fixture, and the standalone setups now prove over the
//! single-slot union commit with INTEGER LANES (dense stack), so both
//! anchors move too; the anchor tests now pin `setup.prove_fast` itself.
//! Stable across two print runs.
//! Re-pinned 2026-08-14 (third): SHA-256 lin-id drops, measured — W never
//! materialized, E_NEW/A_NEW only every other round (EA_PERIOD 2):
//! USEFUL_BITS 29,054 -> 25,470 (227 -> 199 chunk-columns) at 47.4M
//! template nnz (the accepted blake3-envelope density; full zk.golf-style
//! inlining measured 184M and was rejected). sha2-touching fixtures move;
//! the BLAKE3 anchor is byte-identical. Stable across two print runs.
//! Re-pinned 2026-08-14 (second): SHA-256 R1CS "Option F" — the zk.golf
//! sha256 record's systematic techniques (K folded as constant adds, fused
//! 4-op T1 tree and 3-op a_new tree, T1 slots dissolved, schedule steps
//! 93 -> 92): USEFUL_BITS 31,401 -> 29,054 (246 -> 227 chunk-columns).
//! Every sha2-touching fixture moves, incl. the sha2 anchor; the BLAKE3
//! anchor is byte-identical — the change is sha2-local. Digests stable
//! across two print runs.
//! Re-pinned 2026-08-14: BLAKE3 R1CS "Option F" — the zk.golf record's
//! fused 3-operand adders (61 rows vs 62) and round-1 constant-c adders
//! (30/29 rows) shrink the table 10,416 → 10,298 AND rows, USEFUL_BITS
//! 11,825 → 11,707 (93 → 92 chunk-columns), so every blake3-committed
//! fixture's bytes move. The pure SHA-256 anchor is untouched — the change
//! is blake3-local. Digests stable across two print runs.
//! Re-pinned 2026-08-19: the consistency-batch grinding off-by-one fix —
//! each ladder level now grinds its OWN `consistency_batch_grinding_bits`
//! entry (level ℓ was ground under bits[ℓ+1]; bits[0] was never applied),
//! restoring the exact schedule `validate()` certifies. Both m22 anchors
//! and the nu10 full-utilization fixture move (their levels' bits differ);
//! the three small nu10 shapes hold (adjacent bits equal). Roundtrip
//! suites green; digests stable across two print runs.
//! Re-pinned 2026-08-27: the profile consolidation (proof-IO v22, bloat
//! ledger §C). The grind-free `Fast`/`Slim` were deleted and `Fast` now
//! carries the former Fast128 schedule: aggressive +2/level ladder and
//! 16-bit query PoW at every level. All six digests move (every level's
//! rate, query count, cap and PoW change). Full workspace suite green;
//! digests stable across two print runs.

use ::sha2 as sha2_hash;
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

const DOMAIN: &[u8] = b"flock-m6-fixture-v0";

use flock_core::test_rng::Rng;

fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<blake3::Compression> {
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

fn random_sha2_inputs(rng: &mut Rng, n: usize) -> Vec<sha2::Compression> {
    (0..n)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.next_u32()),
                std::array::from_fn(|_| rng.next_u32()),
            )
        })
        .collect()
}

// Re-pinned 2026-08-02: multipoint-twisted assist (proof_io v8) — the
// per-statement assist became 128K dual values + one product sumcheck +
// one untwisted anchor; transcript + wire moved by design.
// Re-pinned 2026-08-04: merged-open v1 — the packed-direct intake absorbs
// VALUE-ONLY (claim points are transcript-derived and verifier-recomputed,
// never prover messages; ~92 KB of self-absorbed points deleted from the
// recursion replay). Label flock-merged-open-v0 -> v1.
// Re-pinned 2026-08-05: BLAKE3 R1CS "Option E" lin-id drop — the b3 table
// narrowed 121 -> 93 word-cols (b_new/d_new slots dissolved into the
// cascade), so every blake3-committed fixture's bytes move. The pure
// SHA-256 anchor is untouched — the drop is blake3-local.
// Re-pinned 2026-08-02: two-product multipoint grouping (proof_io v9) —
// packed-direct claims collapse into merged-column scalar groups (one
// dual value each); the multipoint label bumped to v1, so even the
// boolean-only fixtures here (no packed-direct claims) move.
// Re-pinned 2026-08-13 after circuit digests began binding fixed-public
// declarations and the retained registry. The statement transcript changes
// by design; two deterministic print runs agreed for all six fixtures.
fn check(label: &str, expected: &str, got: String) {
    if std::env::var_os("M6_FIXTURES_PRINT").is_some() {
        println!("(\"{label}\", \"{got}\"),");
        return;
    }
    assert_eq!(
        got, expected,
        "M6 byte-identity broken for fixture `{label}`: the prover's output \
         bytes diverged from the pinned proof-IO protocol"
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
/// for the current v21 mixed proof. It retains the removed jagged fixture's
/// statements and witness streams; the Ligerito query ladder intentionally
/// changed with v18.
#[test]
// Default-run (~2 s for both anchors): these pins are what makes the
// "fixture anchors byte-stable" claim enforceable in CI.
// Re-pinned 2026-08-31: main's profile consolidation (proof-IO v22) merged
// on top of the BLAKE3 default — both sides' pins were stale. Two
// deterministic print runs agreed.
fn m6_merged_union_proof_bytes_pinned() {
    const FIXTURES: [(&str, [usize; 2], &str); 4] = [
        (
            "merged-nu10-1024-1024",
            [1024, 1024],
            "c3d7b17119826833b218d05b2684f1b2a9c1a99c91507861a6bdf61834408ff9",
        ),
        (
            "merged-nu10-50-37",
            [50, 37],
            "0f15486ead29dbb3c27222522fa68dc4c24d240a688e2391dc0f4477ee4e0e8a",
        ),
        (
            "merged-nu10-8-8",
            [8, 8],
            "16773ba2aece640a77de8712a2c0004bd0f0acfee548a1ff6ec56c436047cb96",
        ),
        (
            "merged-nu10-0-64",
            [0, 64],
            "e0e6e9cdfc8dc7834b81bd3ad758630cc1a6e622835f3ec7c25ac81e66e1cb74",
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
            profile: flock_core::pcs::ligerito::LigeritoProfile::Fast,
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
// Default-run (~2 s for both anchors): see m6_merged_union_proof_bytes_pinned.
fn m6_single_slot_merged_anchor_proof_bytes_pinned() {
    // BLAKE3, 256 blocks (m = 22).
    {
        const EXPECTED: &str = "ae19c381659e84562c829383f6b01137741b4e2ed29890daa555c332199d2f72";
        let n_blocks = 256usize;
        // The setup API IS the shipped single-slot union path since the
        // 2026-08-14 consolidation — the anchor pins it directly.
        let setup = blake3::Blake3Setup::new(n_blocks);
        let mut rng = Rng::new(0x4D36_B3B3);
        let inputs = random_blake3_inputs(&mut rng, n_blocks);
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) = setup.prove_fast(&inputs, &mut ch);
        check(
            "merged-anchor-blake3-m22",
            EXPECTED,
            merged_bundle_digest(&proof, &commitment, &claim),
        );
    }

    // SHA-256, 128 blocks (m = 22).
    {
        const EXPECTED: &str = "3bddd36713d2b607a662ff96ad2ef4ace93fd91f0014e7f7b2ab1f44c668338c";
        let n_blocks = 128usize;
        let setup = sha2::Sha256HybridSetup::new(n_blocks);
        let mut rng = Rng::new(0x4D36_5252);
        let inputs = random_sha2_inputs(&mut rng, n_blocks);
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) = setup.prove_fast(&inputs, &mut ch);
        check(
            "merged-anchor-sha2-m22",
            EXPECTED,
            merged_bundle_digest(&proof, &commitment, &claim),
        );
    }
}
