//! M3: the first end-to-end MIXED proofs — BLAKE3 and SHA-256 in one
//! statement, proved through the union entries under the real multi-table
//! binding (`flock-mixed-v1`: registry digest + counts vector + commitment
//! root, design doc §"Statement, transcript, wire format").
//!
//! Registry shape: the real SHA-256 (κ = 15) and BLAKE3 (κ = 14) base
//! blocks at uniform capacity 2^ν. Slot order is the registry order —
//! capacity area descending, so SHA-256 before BLAKE3 — and M = ν + 16
//! (areas 2^{ν+15} + 2^{ν+14} round up to 2^{ν+16}; the top quarter of the
//! address space is the gap). ν = 6 puts M = 22, the smallest embedded
//! Ligerito config, keeping the tests tractable. Full utilization only —
//! the batch-major drivers fill every row; partial counts are M4.
//!
//! Covers: the mixed prove → verify roundtrip (asserting the commitment
//! root equals a direct commit of the independently assembled union
//! buffer), the statement tamper matrix (wrong counts vector, tampered
//! registry digest, swapped slot order), one PIOP and one opening tamper
//! through the existing error paths, a single-type roundtrip under the new
//! binding, and an informational mixed-vs-singles throughput smoke. The
//! byte-identity differentials against the direct jagged path live in
//! `tests/union_roundtrip.rs` on the harness binding.

use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{PcsParams, VerifyErrorJagged};
use flock_core::proof::R1csProofJaggedLigerito;
use flock_core::r1cs::BlockR1cs;
use flock_core::union::SlotWitness;
use flock_core::verifier::VerifyError;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::{blake3, sha2};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
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

const DOMAIN: &[u8] = b"flock-mixed-e2e-v0";

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

/// The M3 mixed registry: the real SHA-256 (κ = 15) and BLAKE3 (κ = 14)
/// base blocks (via `TableType::from_block_r1cs` on the modules' block
/// R1CS) at uniform capacity 2^ν, fed in width-ASCENDING order to exercise
/// the registry's canonical sort. Slot order — capacity area descending,
/// under uniform capacity simply κ descending — is SHA-256 then BLAKE3,
/// and M = ν + 16.
fn mixed_registry(nu: usize) -> (Registry, BlockR1cs, BlockR1cs) {
    let sha2_r1cs = sha2::build_block_r1cs(nu);
    let blake3_r1cs = blake3::build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            TableType::from_block_r1cs(&blake3_r1cs),
            TableType::from_block_r1cs(&sha2_r1cs),
        ],
        nu,
    );
    assert_eq!(
        registry.types()[0].k_log,
        sha2::K_LOG,
        "slot order: SHA-256 (wider) first"
    );
    assert_eq!(registry.types()[1].k_log, blake3::K_LOG);
    assert_eq!(registry.m_total(), nu + 16);
    (registry, sha2_r1cs, blake3_r1cs)
}

/// PCS params over the committed DENSE stack: `m = dense_m` — the
/// compacted stack's variable count, not the union's `M`. Count-dependent
/// under height-`n_t` stacking (M5), floored at the m22 Ligerito config;
/// at ν = 6 the padded size 2^15 IS the floor, so every count vector in
/// these tests commits `m = 22` (the count-proportional shrink needs
/// ν ≥ 7 — see `mixed_area_saving_roundtrip`). Rate, batch size, and
/// profile match the single-type setups.
fn union_pcs_params(union: &UnionInstance<'_>) -> PcsParams {
    PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
    }
}

/// THE milestone test: a mixed BLAKE3 + SHA-256 statement proved and
/// verified end-to-end under the `flock-mixed-v1` binding, plus the
/// statement/proof tamper matrix against the same proof.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn mixed_blake3_sha256_roundtrip_and_tamper() {
    let nu = 6usize;
    let n_per_type = 1usize << nu; // full utilization (partial counts are M4)
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    assert_eq!(
        registry.m_total(),
        22,
        "ν = 6 must land on the m = 22 embedded Ligerito config"
    );
    let union = UnionInstance::new(&registry, vec![n_per_type, n_per_type]);
    let pcs_params = union_pcs_params(&union);
    // The dense-stack shape on this registry at FULL utilization (heights =
    // capacity, M4's grid exactly): 367 of 512 chunk-columns used (SHA-256
    // 246/256, BLAKE3 121/128; the top quarter is the gap), a genuinely
    // non-identity compaction (BLAKE3's columns stack at 246, not 256),
    // rounding back to the padded word count at this ratio.
    assert!(!union.compaction_is_identity());
    assert_eq!(union.dense_words(), (246 + 121) << nu);
    assert_eq!(union.committed_words(), 1 << (union.m_total() - 7));

    let mut rng = Rng::new(0x03_31_2B_B3);
    let blake3_inputs = random_blake3_inputs(&mut rng, n_per_type);
    let sha2_inputs = random_sha2_inputs(&mut rng, n_per_type);
    let sha2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let blake3_circuit = blake3_r1cs.csc_lincheck_circuit();

    // ---- Prove: per-slot inputs in slot order (SHA-256 first).
    let slots = vec![
        UnionSlotProverInput::new(
            sha2::generate_witness_batch_major(&sha2_inputs, nu),
            sha2_circuit,
        ),
        UnionSlotProverInput::new(
            blake3::generate_witness_batch_major(&blake3_inputs, nu),
            blake3_circuit,
        ),
    ];
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) =
        prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, &mut ch_p);

    // ---- The commitment is a commitment to the COMPACTED union buffer
    // (the M4 dense stack): regenerate the witnesses, assemble them
    // independently, compact, commit directly, and compare roots. Also pin
    // that the compaction genuinely moved data: q differs from the padded
    // buffer's prefix (BLAKE3's slot stacks 10 columns lower).
    let (z_s, a_s, b_s, _) = sha2::generate_witness_batch_major(&sha2_inputs, nu);
    let (z_b, a_b, b_b, _) = blake3::generate_witness_batch_major(&blake3_inputs, nu);
    let (z_union, _, _) = union.assemble_witness(vec![
        SlotWitness {
            z_packed: z_s,
            a_packed: a_s,
            b_packed: b_s,
        },
        SlotWitness {
            z_packed: z_b,
            a_packed: a_b,
            b_packed: b_b,
        },
    ]);
    let q = union.compact_witness(&z_union);
    assert_eq!(q.len(), union.committed_words());
    assert_ne!(
        q[..],
        z_union[..q.len()],
        "compaction must move the second slot's columns"
    );
    let (comm_direct, _prover_data) = flock_core::pcs::commit(&q, &pcs_params);
    assert_eq!(
        commitment.root, comm_direct.root,
        "commitment root must equal a direct commit of the compacted union stack"
    );

    // ---- Verify (circuits in slot order).
    let circuits: [&dyn LincheckCircuit; 2] = [sha2_circuit, blake3_circuit];
    let verify = |union: &UnionInstance<'_>, proof: &R1csProofJaggedLigerito| {
        let mut ch_v = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_jagged_union(
            union,
            &circuits,
            &commitment,
            proof,
            &pcs_params,
            &mut ch_v,
        )
    };
    let claim_v = verify(&union, &proof)
        .unwrap_or_else(|e| panic!("mixed verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_v, claim);

    // ---- Tamper: wrong counts vector. The binding absorbs the counts
    // before any challenge, so a verifier declaring different counts walks
    // a diverged transcript from the first squeeze (downstream, the
    // lincheck's count-derived const-pin target would also mismatch, and —
    // since M5's height-n_t stacking — so would the count-derived jagged
    // heights/col_prefix_sums of the opening) — reject.
    {
        let union_bad = UnionInstance::new(&registry, vec![n_per_type, n_per_type - 1]);
        assert!(
            verify(&union_bad, &proof).is_err(),
            "wrong counts vector must reject"
        );
    }

    // ---- Tamper: registry digest. `useful_bits + 1` rounds to the same
    // chunk-column count, so the heights, per-type combs, and pin targets
    // are all unchanged — the ONLY verifier-side divergence is the registry
    // digest inside the binding, isolating it as load-bearing.
    {
        let mut blake3_ty = TableType::from_block_r1cs(&blake3_r1cs);
        blake3_ty.useful_bits += 1;
        let registry_bad =
            Registry::new(vec![TableType::from_block_r1cs(&sha2_r1cs), blake3_ty], nu);
        assert_ne!(
            registry.digest(),
            registry_bad.digest(),
            "tamper must move the registry digest"
        );
        let union_bad = UnionInstance::new(&registry_bad, vec![n_per_type, n_per_type]);
        assert_eq!(
            union.jagged_heights(),
            union_bad.jagged_heights(),
            "tamper must be invisible to the heights — digest-only"
        );
        assert!(
            verify(&union_bad, &proof).is_err(),
            "tampered registry must reject"
        );
    }

    // ---- Tamper: PIOP (a lincheck round message) — rejects through the
    // existing union-lincheck error path.
    {
        let mut bad = proof.clone();
        bad.lincheck.rounds[0].0.lo ^= 1;
        match verify(&union, &bad) {
            Err(VerifyError::Lincheck(flock_core::lincheck::VerifyError::ConsistencyFailed {
                ..
            })) => {}
            other => panic!(
                "tampered lincheck round: expected Lincheck(ConsistencyFailed), got {other:?}"
            ),
        }
    }

    // ---- Tamper: opening (the virtual-open `f_eval`) — rejects through
    // the existing jagged opening error path.
    {
        let mut bad = proof.clone();
        bad.pcs_open.f_eval.lo ^= 1;
        match verify(&union, &bad) {
            Err(VerifyError::PcsJagged(VerifyErrorJagged::VirtualOpen)) => {}
            other => panic!("tampered f_eval: expected PcsJagged(VirtualOpen), got {other:?}"),
        }
    }
}

/// M4/M5 — partial utilization (dynamic counts): mixed roundtrips at
/// several count vectors, including non-powers-of-two and a zero count for
/// one type, through the partial batch-major drivers (dummy rows
/// identically zero — pin included). Under height-`n_t` stacking every
/// partial vector here exercises genuinely truncated jagged columns
/// (heights 50/37/0 < 64) and per-proof col_prefix_sums; the committed
/// LENGTH stays 2^15 words throughout only because ν = 6's padded size
/// equals the m22 config floor. Verifies acceptance at every utilization,
/// rejects wrong-count tampering against the partial proof, and prints
/// verify wall times across utilizations (informational — the verifier's
/// control flow is registry-static, so times should be flat).
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn mixed_partial_counts_roundtrip_and_tamper() {
    use std::time::Instant;

    let nu = 6usize; // capacity 64 per type; M = 22 (m22 Ligerito config)
    let capacity = 1usize << nu;
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let sha2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let blake3_circuit = blake3_r1cs.csc_lincheck_circuit();
    let circuits: [&dyn LincheckCircuit; 2] = [sha2_circuit, blake3_circuit];
    let mut rng = Rng::new(0x04_31_2B_B3);

    // Counts in slot order (SHA-256, BLAKE3): full, non-power-of-two
    // partials, and a zero count for one type.
    let count_vectors: [[usize; 2]; 4] = [[64, 64], [50, 37], [0, 64], [37, 0]];
    let mut verify_ms = Vec::new();
    for counts in count_vectors {
        let [n_sha2, n_blake3] = counts;
        assert!(n_sha2 <= capacity && n_blake3 <= capacity);
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = union_pcs_params(&union);
        // ν = 6: the padded size 2^15 IS the m22 config floor, so every
        // count vector commits m = 22 — but with genuinely truncated
        // count-height columns inside the stack.
        assert_eq!(pcs_params.m, 22);
        let blake3_inputs = random_blake3_inputs(&mut rng, n_blake3);
        let sha2_inputs = random_sha2_inputs(&mut rng, n_sha2);

        let slots = vec![
            UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(&sha2_inputs, nu),
                sha2_circuit,
            ),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(&blake3_inputs, nu),
                blake3_circuit,
            ),
        ];
        let mut ch_p = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, &mut ch_p);

        let verify = |union: &UnionInstance<'_>| {
            let mut ch_v = FsChallenger::new(DOMAIN);
            verifier::verify_ligerito_jagged_union(
                union,
                &circuits,
                &commitment,
                &proof,
                &pcs_params,
                &mut ch_v,
            )
        };
        let t = Instant::now();
        let claim_v = verify(&union).unwrap_or_else(|e| {
            panic!("partial-count verifier rejected honest proof (counts {counts:?}): {e:?}")
        });
        verify_ms.push((counts, t.elapsed().as_secs_f64() * 1e3));
        assert_eq!(claim_v, claim);

        // Wrong-count tampering: a verifier declaring one more (or one
        // fewer at zero) invocation walks a diverged transcript from the
        // first squeeze (counts bind before any challenge) and, downstream,
        // a wrong const-pin target — reject.
        let bad_counts = if n_sha2 < capacity {
            vec![n_sha2 + 1, n_blake3]
        } else {
            vec![n_sha2 - 1, n_blake3]
        };
        let union_bad = UnionInstance::new(&registry, bad_counts.clone());
        assert!(
            verify(&union_bad).is_err(),
            "wrong counts {bad_counts:?} vs {counts:?} must reject"
        );
    }

    println!("mixed partial-count verify times (registry-static control flow):");
    for (counts, ms) in &verify_ms {
        println!("  counts (sha2, blake3) = {counts:?}: {ms:.1} ms");
    }
}

/// M5 — THE area gate, end to end: at ν = 7 (M = 23, padded 2^16 words) a
/// partial-utilization mix at counts (32, 32) commits the height-`n_t`
/// dense stack of 32·(246 + 121) = 11 744 words → 2^15 committed words
/// (the m22 config floor) — HALF of M4's capacity-height 2^16 — and the
/// proof roundtrips through the smaller commitment. Wrong-count tampering
/// still rejects (transcript binding first; the count-derived
/// heights/col_prefix_sums would diverge downstream too). The committed
/// size is asserted from the returned commitment, not just the sizing
/// arithmetic.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn mixed_area_saving_roundtrip() {
    let nu = 7usize; // capacity 128 per type; padded 2^16 words
    let counts = [32usize, 32usize]; // (sha2, blake3), quarter utilization
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    assert_eq!(registry.m_total(), 23);
    let union = UnionInstance::new(&registry, counts.to_vec());
    // The halving: dense 11 744 words → committed 2^15 (config floor;
    // next_pow2 alone would give 2^14) vs M4's capacity-height 2^16.
    assert_eq!(union.dense_words(), 11_744);
    assert_eq!(union.committed_words(), 1 << 15);
    assert_eq!(union.dense_m(), 22);
    assert_eq!(
        2 * union.committed_words(),
        1 << (union.m_total() - 7),
        "counts (32, 32) must commit HALF of the capacity-height size"
    );
    let pcs_params = union_pcs_params(&union);

    let mut rng = Rng::new(0x05_31_2B_B3);
    let sha2_inputs = random_sha2_inputs(&mut rng, counts[0]);
    let blake3_inputs = random_blake3_inputs(&mut rng, counts[1]);
    let sha2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let blake3_circuit = blake3_r1cs.csc_lincheck_circuit();

    let slots = vec![
        UnionSlotProverInput::new(
            sha2::generate_witness_batch_major_partial(&sha2_inputs, nu),
            sha2_circuit,
        ),
        UnionSlotProverInput::new(
            blake3::generate_witness_batch_major_partial(&blake3_inputs, nu),
            blake3_circuit,
        ),
    ];
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) =
        prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, &mut ch_p);
    assert_eq!(
        commitment.params.m, 22,
        "the produced commitment must be to the 2^15-word dense stack"
    );

    let circuits: [&dyn LincheckCircuit; 2] = [sha2_circuit, blake3_circuit];
    let verify = |union: &UnionInstance<'_>| {
        let mut ch_v = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_jagged_union(
            union,
            &circuits,
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
    };
    let claim_v = verify(&union)
        .unwrap_or_else(|e| panic!("area-saving verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_v, claim);

    // Wrong declared counts against the smaller commitment: reject.
    let union_bad = UnionInstance::new(&registry, vec![counts[0] + 1, counts[1]]);
    assert!(
        verify(&union_bad).is_err(),
        "wrong counts must reject against the count-sized commitment"
    );
}

/// Mis-ordered per-slot inputs (BLAKE3 before SHA-256) can never produce a
/// proof: slots must arrive in registry order — capacity area descending —
/// and the witness assembly asserts every slot buffer's length against its
/// slot before anything transcript-visible happens.
#[test]
#[should_panic(expected = "slot z_packed length mismatch")]
fn mixed_prove_rejects_swapped_slot_order() {
    let nu = 6usize;
    let n_per_type = 1usize << nu;
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let union = UnionInstance::new(&registry, vec![n_per_type, n_per_type]);
    let pcs_params = union_pcs_params(&union);
    let mut rng = Rng::new(0x03_31_5A_9D);
    let blake3_inputs = random_blake3_inputs(&mut rng, n_per_type);
    let sha2_inputs = random_sha2_inputs(&mut rng, n_per_type);

    // WRONG order: BLAKE3 (κ = 14) first, SHA-256 (κ = 15) second.
    let slots = vec![
        UnionSlotProverInput::new(
            blake3::generate_witness_batch_major(&blake3_inputs, nu),
            blake3_r1cs.csc_lincheck_circuit(),
        ),
        UnionSlotProverInput::new(
            sha2::generate_witness_batch_major(&sha2_inputs, nu),
            sha2_r1cs.csc_lincheck_circuit(),
        ),
    ];
    let mut ch_p = FsChallenger::new(DOMAIN);
    let _ = prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, &mut ch_p);
}

/// A single-type instance through the NEW binding roundtrips. The proof is
/// (correctly) NOT byte-identical to
/// `prove_fast_ligerito_jagged_from_witness` on the same statement +
/// witness: the `flock-mixed-v1` binding absorbs the registry digest + the
/// counts vector where the direct path absorbs the `BlockR1cs` statement
/// digest — domain-separated on purpose — so no byte-identity is (or ever
/// will be) asserted here. The byte-identity regression anchor is
/// `tests/union_roundtrip.rs`, which pins the harness binding.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn blake3_single_type_roundtrip_under_mixed_binding() {
    let n_blocks = 256usize;
    let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
    let mut rng = Rng::new(0x03_31_00_B3);
    let inputs = random_blake3_inputs(&mut rng, n_blocks);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();

    let registry = Registry::new(
        vec![TableType::from_block_r1cs(&setup.r1cs)],
        setup.r1cs.n_log(),
    );
    let union = UnionInstance::new(&registry, vec![n_blocks]);
    let slot = UnionSlotProverInput::new(
        blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
        lc_circuit,
    );
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) =
        prover::prove_fast_ligerito_jagged_union(&union, &setup.pcs_params, vec![slot], &mut ch_p);

    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_v = verifier::verify_ligerito_jagged_union(
        &union,
        &[lc_circuit],
        &commitment,
        &proof,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("single-type mixed-binding verifier rejected honest proof: {e:?}"));
    assert_eq!(claim_v, claim);
}

/// Informational mixed-vs-singles prove-time smoke — prints prove wall
/// times (including witness generation, matching `jagged_throughput`'s
/// accounting) for the mixed instance and for the two single-type jagged
/// instances at the same per-type sizes. No timing assertions. Run with
/// `cargo test --release -p flock-prover --test union_mixed -- --ignored
/// --nocapture mixed_throughput_smoke`.
#[test]
#[ignore] // Heavy + informational — run explicitly with --ignored --nocapture
fn mixed_throughput_smoke() {
    use std::time::Instant;

    // ν = 10: 1024 invocations per type; mixed M = 26, singles at m = 24
    // (BLAKE3) / 25 (SHA-256).
    let nu = 10usize;
    let n_per_type = 1usize << nu;
    let mut rng = Rng::new(0x03_31_77_77);
    let blake3_inputs = random_blake3_inputs(&mut rng, n_per_type);
    let sha2_inputs = random_sha2_inputs(&mut rng, n_per_type);

    // Single-type baselines through the existing jagged path. One untimed
    // warm-up prove per path (hot scratch pool), then one timed run.
    let b3_setup = blake3::Blake3Setup::new_batch_major(n_per_type);
    assert_eq!(b3_setup.n_blocks_log(), nu);
    let b3_circuit = b3_setup.r1cs.csc_lincheck_circuit();
    let mut b3_ms = 0.0;
    for timed in [false, true] {
        let mut ch = FsChallenger::new(DOMAIN);
        let t = Instant::now();
        let (z, a, b, stripe) = blake3::generate_witness_batch_major(&blake3_inputs, nu);
        let _ = prover::prove_fast_ligerito_jagged_from_witness(
            &b3_setup.r1cs,
            &b3_setup.pcs_params,
            z,
            a,
            b,
            stripe,
            b3_circuit,
            None,
            &mut ch,
        );
        if timed {
            b3_ms = t.elapsed().as_secs_f64() * 1e3;
        }
    }

    let s2_setup = sha2::Sha256HybridSetup::new_batch_major(n_per_type);
    assert_eq!(s2_setup.n_blocks_log(), nu);
    let s2_circuit = s2_setup.r1cs.csc_lincheck_circuit();
    let mut s2_ms = 0.0;
    for timed in [false, true] {
        let mut ch = FsChallenger::new(DOMAIN);
        let t = Instant::now();
        let (z, a, b, stripe) = sha2::generate_witness_batch_major(&sha2_inputs, nu);
        let _ = prover::prove_fast_ligerito_jagged_from_witness(
            &s2_setup.r1cs,
            &s2_setup.pcs_params,
            z,
            a,
            b,
            stripe,
            s2_circuit,
            None,
            &mut ch,
        );
        if timed {
            s2_ms = t.elapsed().as_secs_f64() * 1e3;
        }
    }

    // The mixed instance at the same per-type sizes.
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let union = UnionInstance::new(&registry, vec![n_per_type, n_per_type]);
    let pcs_params = union_pcs_params(&union);
    flock_core::scratch::prewarm_prover(registry.m_total());
    let s2_mix_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_mix_circuit = blake3_r1cs.csc_lincheck_circuit();
    let mut mixed_ms = 0.0;
    for timed in [false, true] {
        let mut ch = FsChallenger::new(DOMAIN);
        let t = Instant::now();
        let slots = vec![
            UnionSlotProverInput::new(
                sha2::generate_witness_batch_major(&sha2_inputs, nu),
                s2_mix_circuit,
            ),
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major(&blake3_inputs, nu),
                b3_mix_circuit,
            ),
        ];
        let _ = prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, &mut ch);
        if timed {
            mixed_ms = t.elapsed().as_secs_f64() * 1e3;
        }
    }

    let singles = b3_ms + s2_ms;
    println!("mixed throughput smoke, {n_per_type} invocations per type (prove incl. witgen):");
    println!("  blake3-only jagged (m = {}): {b3_ms:.0} ms", b3_setup.m());
    println!("  sha2-only jagged   (m = {}): {s2_ms:.0} ms", s2_setup.m());
    println!(
        "  mixed union        (M = {}): {mixed_ms:.0} ms",
        registry.m_total()
    );
    println!(
        "  singles sum {singles:.0} ms; mixed / sum = {:.2}",
        mixed_ms / singles
    );
}

/// Low-utilization prove-time smoke: counts (8, 8) at ν = 10 against full
/// utilization (1024, 1024). Under height-`n_t` stacking the low-count
/// instance commits 2^15 words (config floor; dense 2 936) versus 2^19 at
/// full — a 16x smaller commit — and since M6 the PIOP/opening passes are
/// support-proportional too (zerocheck tail, lincheck row folds,
/// virtual-open f-side, round-0 prime, witness scatter), so the low-count
/// prove is dominated by the count-independent floor: the per-type comb
/// builds (O(nnz), registry-static) and the m22-config-floor opening
/// machinery. One LOOSE timing assertion (partial must beat full by a
/// noise-proof margin); precise numbers are printed, not asserted. Run with
/// `cargo test --release -p flock-prover --test union_mixed -- --ignored
/// --nocapture mixed_low_utilization_smoke`.
#[test]
#[ignore] // Heavy + informational — run explicitly with --ignored --nocapture
fn mixed_low_utilization_smoke() {
    use std::time::Instant;

    let nu = 10usize;
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_circuit = blake3_r1cs.csc_lincheck_circuit();
    let mut rng = Rng::new(0x05_31_77_77);
    flock_core::scratch::prewarm_prover(registry.m_total());

    let mut results = Vec::new();
    for counts in [[8usize, 8usize], [1024, 1024]] {
        let [n_sha2, n_blake3] = counts;
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = union_pcs_params(&union);
        let sha2_inputs = random_sha2_inputs(&mut rng, n_sha2);
        let blake3_inputs = random_blake3_inputs(&mut rng, n_blake3);
        // One untimed warm-up (hot scratch pool), then one timed run.
        let mut prove_ms = 0.0;
        for timed in [false, true] {
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
            let t = Instant::now();
            let (proof, commitment, claim) =
                prover::prove_fast_ligerito_jagged_union(&union, &pcs_params, slots, &mut ch);
            if timed {
                prove_ms = t.elapsed().as_secs_f64() * 1e3;
                // Roundtrip while we're here.
                let circuits: [&dyn LincheckCircuit; 2] = [s2_circuit, b3_circuit];
                let mut ch_v = FsChallenger::new(DOMAIN);
                let claim_v = verifier::verify_ligerito_jagged_union(
                    &union,
                    &circuits,
                    &commitment,
                    &proof,
                    &pcs_params,
                    &mut ch_v,
                )
                .unwrap_or_else(|e| {
                    panic!("low-utilization verifier rejected honest proof {counts:?}: {e:?}")
                });
                assert_eq!(claim_v, claim);
            }
        }
        results.push((counts, union.committed_words(), prove_ms));
    }

    println!(
        "low-utilization smoke @ nu = {nu} (M = {}), prove incl. witgen:",
        registry.m_total()
    );
    for (counts, committed, ms) in &results {
        println!(
            "  counts (sha2, blake3) = {counts:?}: committed 2^{} words, {ms:.0} ms",
            committed.trailing_zeros()
        );
    }

    // The M6 gate, asserted LOOSELY (wall times of single runs are noisy;
    // the precise ratios live in the printed output and the milestone
    // notes): a 128x-fewer-invocations instance must prove well under the
    // full-utilization time. Pre-M6 the ratio hovered around 0.7 (only
    // commit/opening scaled); post-M6 it sits around 0.5.
    let (_, _, low_ms) = results[0];
    let (_, _, full_ms) = results[1];
    assert!(
        low_ms < 0.8 * full_ms,
        "low-utilization prove ({low_ms:.0} ms) must beat full utilization \
         ({full_ms:.0} ms) by a clear margin — support-proportional passes \
         may have regressed"
    );
}
